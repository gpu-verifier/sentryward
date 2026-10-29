#include "sentryward/algorithm/memory/symbolic_address_model.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

#include "sentryward/algorithm/memory/address_solver.h"

namespace sentryward::algorithm {
namespace address_solver {

std::string ToString(Int value) {
  if (value == 0) return "0";
  const bool negative = value < 0;
  std::string digits;
  while (value != 0) {
    const auto digit = static_cast<int>(value % 10);
    digits.push_back(static_cast<char>('0' + (negative ? -digit : digit)));
    value /= 10;
  }
  if (negative) digits.push_back('-');
  return {digits.rbegin(), digits.rend()};
}

}  // namespace address_solver

namespace {

using Json = nlohmann::json;
using address_solver::Int;
using address_solver::ToString;
using AtomId = std::size_t;
using NameId = std::uint32_t;

constexpr unsigned kMaxDepth = 64;

std::optional<Int> Checked(Int a, Int b, bool multiply) {
  Int result;
  if (multiply ? __builtin_mul_overflow(a, b, &result) : __builtin_add_overflow(a, b, &result))
    return std::nullopt;
  return result;
}

// Offset normal form (dynamic.pdf eq. 1): constant + sum(c_k * atom_k),
// terms sorted by atom with nonzero coefficients.
struct Affine final {
  Int constant = 0;
  std::vector<std::pair<AtomId, Int>> terms;

  [[nodiscard]] std::string key() const {
    std::string result = ToString(constant);
    for (const auto& [atom, coefficient] : terms)
      result += "+" + ToString(coefficient) + "*a" + std::to_string(atom);
    return result;
  }
};

// a + scale * b; nullopt on overflow.
std::optional<Affine> Combine(const Affine& a, const Affine& b, Int scale) {
  Affine result;
  const auto scaled = Checked(b.constant, scale, true);
  if (!scaled) return std::nullopt;
  const auto constant = Checked(a.constant, *scaled, false);
  if (!constant) return std::nullopt;
  result.constant = *constant;
  std::map<AtomId, Int> terms(a.terms.begin(), a.terms.end());
  for (const auto& [atom, coefficient] : b.terms) {
    const auto product = Checked(coefficient, scale, true);
    if (!product) return std::nullopt;
    const auto sum = Checked(terms[atom], *product, false);
    if (!sum) return std::nullopt;
    terms[atom] = *sum;
  }
  for (const auto& [atom, coefficient] : terms)
    if (coefficient != 0) result.terms.emplace_back(atom, coefficient);
  return result;
}

Int Gcd(Int a, Int b) {
  if (a < 0) a = -a;
  if (b < 0) b = -b;
  while (b != 0) {
    const Int r = a % b;
    a = b;
    b = r;
  }
  return a;
}

std::optional<Int> Integer(const Json& value) {
  if (value.is_number_unsigned()) return Int(value.get<std::uint64_t>());
  if (value.is_number_integer()) return Int(value.get<std::int64_t>());
  return std::nullopt;
}

bool KnownSpace(std::string_view space) {
  return !space.empty() && space != "generic" && space != "unknown";
}

}  // namespace

struct SymbolicAddressModel::Impl {
  // An atom is one interned provenance (dynamic.pdf Definitions 2.1, 2.3):
  // a parameter symbol, a ground gather application, an opaque producer
  // value, or a relaxed nonlinear product.
  struct Atom final {
    std::string key;
    std::string gather;
    std::vector<Int> args;
  };

  enum class Base {
    kAllocation,  // allocation-relative offset in a named allocation
    kAbsolute,    // absolute address in the binding's space, allocation unknown
    kSymbolic,    // symbolic allocation term, e.g. a tensor map's runtime target
    kUnknown,
  };

  struct Shape final {
    NameId space = 0;
    Base base = Base::kUnknown;
    NameId identity = 0;  // allocation name or symbolic allocation term
    NameId alias_scope = 0;
    std::optional<Affine> offset;  // nullopt: region with unknown placement
    std::optional<std::uint64_t> region_bytes; // conservative TMA bounding interval
    std::uint64_t region_element = 0;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> region_axes; // stride, extent
    bool absent = false;           // guard is false: empty footprint
  };

  struct TermBound final {
    Affine term;
    std::optional<Int> lower, upper;
    std::vector<Int> multiples;
  };

  struct GatherPolicy final {
    bool injective = false;
    std::optional<Int> lower, upper;
  };

  struct PairKey final {
    std::uint32_t first, second;
    std::optional<std::uint64_t> first_bytes, second_bytes;
    friend bool operator==(const PairKey&, const PairKey&) = default;
  };

  struct PairHash final {
    std::size_t operator()(const PairKey& key) const noexcept {
      std::size_t result = 0;
      const auto mix = [&](std::uint64_t value) {
        result ^= std::hash<std::uint64_t>{}(value) + 0x9e3779b97f4a7c15ULL + (result << 6) + (result >> 2);
      };
      mix(key.first);
      mix(key.second);
      mix(key.first_bytes.value_or(~0ULL));
      mix(key.second_bytes.value_or(~0ULL));
      return result;
    }
  };

  ConstantAddressModel constants;
  std::optional<MemoryAnalysisError> error;
  SymbolicSolverOptions options;
  std::unique_ptr<address_solver::Solver> solver;
  std::size_t capacity;
  SymbolicAddressStats stats;

  std::vector<std::string> names{""};
  std::unordered_map<std::string, NameId> name_ids{{"", 0}};
  std::vector<Atom> atoms;
  std::unordered_map<std::string, AtomId> atom_ids;
  std::vector<Shape> shapes;
  std::unordered_map<std::string, std::uint32_t> shape_ids;
  std::unordered_map<std::uint32_t, std::uint32_t> binding_shapes;

  std::vector<TermBound> term_bounds;
  std::map<std::string, GatherPolicy> gather_policies;
  // The input object of each tensor-property lookup symbol.
  std::map<std::string, std::string> gather_objects;
  // Lookup symbols an address invariant names element by element (term bounds, joint injectivity).
  std::set<std::string> pinned_gathers;
  std::vector<std::vector<std::string>> joint_injective;
  std::set<std::pair<NameId, NameId>> distinct_allocations;

  std::list<std::pair<PairKey, OverlapResult>> lru;
  std::unordered_map<PairKey, decltype(lru)::iterator, PairHash> cache;

  Impl(const std::vector<ConstantAddressBinding>& bindings, std::string_view invariants,
       SymbolicSolverOptions solver_options, std::size_t cache_capacity, std::string_view tensor_properties)
      : constants(bindings, cache_capacity), options(solver_options),
        solver(address_solver::MakeSolver()), capacity(cache_capacity) {
    error = constants.validate();
    try {
      if (options.timeout_ms == 0) throw std::invalid_argument("SMT timeout_ms must be positive");
      ParseInvariants(invariants);
      for (const auto& bound : term_bounds)
        for (const auto& [atom, ignored] : bound.term.terms) {
          (void)ignored;
          if (!atoms[atom].gather.empty()) pinned_gathers.insert(atoms[atom].gather);
        }
      for (const auto& group : joint_injective) pinned_gathers.insert(group.begin(), group.end());
      // Trusted element properties bound every lookup of their object's elements.
      for (const auto& property : Json::parse(tensor_properties)) {
        const auto object = property.at("object").get<std::string>();
        const auto symbol = TensorPropertySymbol(object, property.at("element_type").get<std::string>());
        auto& policy = gather_policies[symbol];
        gather_objects[symbol] = object;
        policy.injective = policy.injective || property.value("injective", false);
        Tighten(policy.lower, policy.upper, property);
      }
      for (const auto& binding : bindings) {
        const auto shape = InternShape(ParseBinding(binding));
        binding_shapes.emplace(binding.address.value(), shape);
      }
      CheckInvariantsConsistent();
    } catch (const std::exception& failure) {
      if (!error) error = MemoryAnalysisError{
          std::string("invalid symbolic-address model: ") + failure.what(), std::nullopt, std::nullopt};
    }
  }

  NameId Name(std::string_view text) {
    const auto found = name_ids.find(std::string(text));
    if (found != name_ids.end()) return found->second;
    const auto id = static_cast<NameId>(names.size());
    names.emplace_back(text);
    name_ids.emplace(names.back(), id);
    return id;
  }

  AtomId Intern(std::string key, std::string gather = {}, std::vector<Int> args = {}) {
    const auto [found, inserted] = atom_ids.emplace(key, atoms.size());
    if (inserted) atoms.push_back({std::move(key), std::move(gather), std::move(args)});
    return found->second;
  }

  static Affine Single(AtomId atom) { return {0, {{atom, 1}}}; }

  static std::string GatherKey(const std::string& symbol, const std::vector<Int>& args) {
    std::string key = "gather:" + symbol + "(";
    for (std::size_t i = 0; i < args.size(); ++i) key += (i ? "," : "") + ToString(args[i]);
    return key + ")";
  }

  // Scalar offset grammar. Unsupported operators (engine region descriptors,
  // views, unknown fields) yield nullopt and never become correlated atoms.
  std::optional<Affine> Scalar(const Json& value, unsigned depth = 0) {
    if (depth > kMaxDepth) return std::nullopt;
    if (const auto number = Integer(value)) return Affine{*number, {}};
    if (!value.is_object() || !value.contains("operator") || !value.at("operator").is_string())
      return std::nullopt;
    const auto op = value.at("operator").get<std::string>();
    const auto only = [&](std::initializer_list<std::string_view> allowed) {
      for (const auto& [field, ignored] : value.items()) {
        (void)ignored;
        if (field != "operator" && std::find(allowed.begin(), allowed.end(), field) == allowed.end())
          return false;
      }
      return std::all_of(allowed.begin(), allowed.end(),
                         [&](auto field) { return value.contains(std::string(field)); });
    };
    if (op == "param") {
      if (!only({"symbol"}) || !value.at("symbol").is_string()) return std::nullopt;
      const auto symbol = value.at("symbol").get<std::string>();
      if (symbol.empty()) return std::nullopt;
      return Single(Intern("param:" + symbol));
    }
    if (op == "gather") {
      if (!only({"symbol", "args"}) || !value.at("symbol").is_string() || !value.at("args").is_array())
        return std::nullopt;
      const auto symbol = value.at("symbol").get<std::string>();
      if (symbol.empty()) return std::nullopt;
      std::vector<Int> args;
      for (const auto& arg : value.at("args")) {
        const auto number = Integer(arg);
        if (!number) return std::nullopt;  // gather arguments must be ground
        args.push_back(*number);
      }
      auto key = GatherKey(symbol, args);
      return Single(Intern(std::move(key), symbol, std::move(args)));
    }
    if (op == "opaque") {
      if (!only({"provenance"}) || value.at("provenance").is_null()) return std::nullopt;
      return Single(Intern("opaque:" + value.at("provenance").dump()));
    }
    if (op == "value_dag") {
      if (!only({"nodes", "root"}) || !value.at("nodes").is_array() ||
          !value.at("root").is_number_unsigned()) return std::nullopt;
      const auto& nodes = value.at("nodes");
      const auto root = value.at("root").get<std::size_t>();
      if (nodes.empty() || nodes.size() > 4096 || root >= nodes.size()) return std::nullopt;
      std::vector<AtomId> interned;
      interned.reserve(nodes.size());
      for (const auto& node : nodes) {
        if (!node.is_object() || !node.contains("op") || !node["op"].is_string() ||
            !node.contains("type") || !node["type"].is_string() ||
            !node.contains("attributes") || !node["attributes"].is_string() ||
            !node.contains("operands") || !node["operands"].is_array()) return std::nullopt;
        if (node["op"] == "Load" && !node.contains("occurrence")) return std::nullopt;
        auto key = node;
        for (auto& operand : key["operands"]) {
          if (!operand.is_number_unsigned()) return std::nullopt;
          const auto id = operand.get<std::size_t>();
          if (id >= interned.size()) return std::nullopt;
          operand = interned[id];
        }
        // Intern exact structure, types and read occurrences; no arithmetic
        // rewrite drops integer wraparound or conversion semantics.
        interned.push_back(Intern("value:" + key.dump()));
      }
      return Single(interned[root]);
    }
    // Existing producer scalars: the value of one PTX operand for one lane's
    // instruction occurrence. The whole node is its provenance.
    if (op == "ptx_memory_operand" || op == "ptx_scalar") {
      if (!only({"operands"})) return std::nullopt;
      return Single(Intern("opaque:" + value.dump()));
    }
    if (op != "add" && op != "sub" && op != "mul" && op != "neg") return std::nullopt;
    if (!only({"operands"}) || !value.at("operands").is_array()) return std::nullopt;
    const auto& operands = value.at("operands");
    if (operands.empty() || (op == "neg" && operands.size() != 1) ||
        (op == "sub" && operands.size() != 2))
      return std::nullopt;
    std::vector<Affine> parts;
    for (const auto& operand : operands) {
      auto part = Scalar(operand, depth + 1);
      if (!part) return std::nullopt;
      parts.push_back(std::move(*part));
    }
    if (op == "neg") return Combine({}, parts[0], -1);
    if (op == "sub") return Combine(parts[0], parts[1], -1);
    Affine result = parts[0];
    for (std::size_t i = 1; i < parts.size(); ++i) {
      std::optional<Affine> next;
      if (op == "add") next = Combine(result, parts[i], 1);
      else if (parts[i].terms.empty()) next = Combine({}, result, parts[i].constant);
      else if (result.terms.empty()) next = Combine({}, parts[i], result.constant);
      else {
        // Both factors symbolic: an opaque nonlinear atom interned by its
        // normalized factors, relaxed to an unconstrained integer (Lemma 4.2).
        auto left = result.key(), right = parts[i].key();
        if (right < left) std::swap(left, right);
        next = Single(Intern("nonlinear:(" + left + ")*(" + right + ")"));
      }
      if (!next) return std::nullopt;
      result = std::move(*next);
    }
    return result;
  }

  static std::string TensorMapTarget(const Json& expression) {
    const auto descriptor = expression.value("descriptor", Json::object());
    const auto location = descriptor.value("allocation_location", Json());
    if (!location.is_array() || location.size() != 2 || !location[0].is_string() ||
        !Integer(location[1]))
      return {};
    return "tensor_map_target(" + location[0].get<std::string>() + "+" +
           ToString(*Integer(location[1])) + ")";
  }

  Shape ParseBinding(const ConstantAddressBinding& binding) {
    Shape shape;
    shape.space = Name(binding.address_space);
    shape.alias_scope = Name(binding.alias_scope);
    const bool resolved = binding.allocation != "unresolved";
    if (resolved) {
      shape.base = Base::kAllocation;
      shape.identity = Name(binding.allocation);
    }
    if (binding.expression.empty()) {
      if (resolved) shape.offset = Affine{Int(binding.offset), {}};
      return shape;
    }
    const auto expression = Json::parse(binding.expression);
    if (!expression.is_object()) return shape;
    if (expression.contains("guard") && expression.at("guard") == Json(false)) shape.absent = true;
    if (expression.value("operator", std::string{}) == "tma_region") {
      // A TMA box reads its tensor map's runtime global target. Two symbolic
      // targets are never assumed equal or distinct without an invariant.
      if (!resolved) {
        if (auto target = TensorMapTarget(expression); !target.empty()) {
          shape.base = Base::kSymbolic;
          shape.identity = Name(target);
        }
      }
      const auto& d = expression.at("descriptor");
      // Cover the whole tile, including holes between rows. Ignoring OOB masks
      // only enlarges the region: disjointness is sound, overlap is not a witness.
      if ((d.contains("base_byte_offset") && d.at("base_byte_offset").is_null()) ||
          (d.contains("packed_element_bits") && !d.at("packed_element_bits").is_null())) return shape;
      const auto& origins = expression.at("coordinates");
      if (!d.contains("box_shape") || !d.contains("coordinate_strides_bytes") ||
          !d.contains("element_bytes") || !origins.is_array()) return shape;
      const auto& box = d.at("box_shape"); const auto& strides = d.at("coordinate_strides_bytes");
      if (box.size()!=origins.size() || strides.size()!=origins.size()) return shape;
      auto base = Integer(d.value("base_byte_offset",Json(0))); auto width = Integer(d.at("element_bytes"));
      if (!base || !width || *base<0 || *width<=0) return shape;
      auto placed = Checked(*base, Int(binding.offset), false);
      if (!placed) return shape;
      Affine offset{*placed,{}};
      std::vector<std::pair<std::uint64_t,std::uint64_t>> axes;
      for (std::size_t i=0;i<origins.size();++i) {
        auto stride=Integer(strides[i]), size=Integer(box[i]); auto origin=Scalar(origins[i]);
        if (!stride || !size || !origin || *stride<0 || *size<=0) return shape;
        auto next=Combine(offset,*origin,*stride); auto span=Checked(*size-1,*stride,true);
        if (!next || !span) return shape;
        width=Checked(*width,*span,false); if (!width) return shape;
        if (*stride>Int(UINT64_MAX) || *size>Int(UINT64_MAX)) return shape;
        axes.emplace_back(static_cast<std::uint64_t>(*stride),static_cast<std::uint64_t>(*size));
        offset=std::move(*next);
      }
      if (*width>Int(UINT64_MAX)) return shape;
      shape.offset=std::move(offset); shape.region_bytes=static_cast<std::uint64_t>(*width);
      shape.region_axes=std::move(axes); shape.region_element=d.at("element_bytes").get<std::uint64_t>();
      return shape;
    }
    const auto address = expression.find("address");
    if (address == expression.end() || !address->is_object()) return shape;
    const auto base = address->find("base");
    const bool has_base = base != address->end() && base->is_object();
    if (has_base) {
      if (base->value("identity", Json()) != Json(binding.allocation) ||
          base->value("space", Json()) != Json(binding.address_space))
        throw std::invalid_argument("symbolic base disagrees with the binding of AddressId " +
                                    std::to_string(binding.address.value()));
    }
    const auto view = address->find("view");
    const bool plain = view == address->end() || view->is_null();
    const auto offset = address->find("offset");
    if (!plain || offset == address->end()) return shape;
    if (has_base && resolved) {
      shape.offset = Scalar(*offset);
    } else if (!has_base && !resolved) {
      // No base and no allocation: the offset is the absolute address.
      if (auto absolute = Scalar(*offset)) {
        shape.base = Base::kAbsolute;
        shape.offset = std::move(absolute);
      }
    }
    return shape;
  }

  std::uint32_t InternShape(Shape shape) {
    std::string key = std::to_string(shape.space) + "|" + std::to_string(static_cast<int>(shape.base)) +
        "|" + std::to_string(shape.identity) + "|" + std::to_string(shape.alias_scope) + "|" +
        (shape.absent ? "absent" : "present") + "|" +
        (shape.region_bytes ? std::to_string(*shape.region_bytes) : "exact") + "|" + (shape.offset ? shape.offset->key() : "region");
    for (auto [stride,extent]:shape.region_axes)
      key += ":axis:"+std::to_string(stride)+":"+std::to_string(extent);
    key += ":element:"+std::to_string(shape.region_element);
    const auto [found, inserted] = shape_ids.emplace(std::move(key), static_cast<std::uint32_t>(shapes.size()));
    if (inserted) shapes.push_back(std::move(shape));
    return found->second;
  }

  // ---- Address-invariant environment Gamma (dynamic.pdf Section 5) ----

  static Int Required(const Json& value, const char* field) {
    const auto number = Integer(value);
    if (!number) throw std::invalid_argument(std::string(field) + " must be an integer");
    return *number;
  }

  static std::vector<std::string> Names(const Json& value, const char* field) {
    if (!value.is_array()) throw std::invalid_argument(std::string(field) + " must be an array");
    std::vector<std::string> result;
    for (const auto& item : value) {
      if (!item.is_string() || item.get<std::string>().empty())
        throw std::invalid_argument(std::string(field) + " entries must be nonempty strings");
      result.push_back(item.get<std::string>());
    }
    if (result.size() < 2 || std::set(result.begin(), result.end()).size() != result.size())
      throw std::invalid_argument(std::string(field) + " requires at least two distinct names");
    return result;
  }

  static void Keys(const Json& value, std::initializer_list<std::string_view> allowed) {
    for (const auto& [field, ignored] : value.items()) {
      (void)ignored;
      if (std::find(allowed.begin(), allowed.end(), field) == allowed.end())
        throw std::invalid_argument("unsupported address-invariant field: " + field);
    }
  }

  static void Tighten(std::optional<Int>& lower, std::optional<Int>& upper, const Json& value) {
    if (value.contains("lower_bound")) {
      const auto bound = Required(value.at("lower_bound"), "lower_bound");
      if (!lower || bound > *lower) lower = bound;
    }
    if (value.contains("upper_bound")) {
      const auto bound = Required(value.at("upper_bound"), "upper_bound");
      if (!upper || bound < *upper) upper = bound;
    }
    if (lower && upper && *lower > *upper) throw std::invalid_argument("empty invariant range");
  }

  void ParseInvariants(std::string_view text) {
    const auto values = Json::parse(text);
    if (!values.is_array()) throw std::invalid_argument("address invariants must be an array");
    for (const auto& value : values) {
      if (!value.is_object()) throw std::invalid_argument("address invariant must be an object");
      if (value.contains("term")) {
        Keys(value, {"term", "lower_bound", "upper_bound", "multiple_of"});
        auto term = Scalar(value.at("term"));
        if (!term || term->terms.empty())
          throw std::invalid_argument("invariant term must be a symbolic scalar expression");
        TermBound bound{std::move(*term), {}, {}, {}};
        Tighten(bound.lower, bound.upper, value);
        if (value.contains("multiple_of")) {
          const auto modulus = Required(value.at("multiple_of"), "multiple_of");
          if (modulus <= 0) throw std::invalid_argument("multiple_of must be positive");
          bound.multiples.push_back(modulus);
        }
        if (!bound.lower && !bound.upper && bound.multiples.empty())
          throw std::invalid_argument("term invariant requires a bound or multiple_of");
        term_bounds.push_back(std::move(bound));
      } else if (value.contains("joint_injective")) {
        Keys(value, {"joint_injective"});
        joint_injective.push_back(Names(value.at("joint_injective"), "joint_injective"));
      } else if (value.contains("distinct_allocations")) {
        Keys(value, {"distinct_allocations"});
        const auto list = Names(value.at("distinct_allocations"), "distinct_allocations");
        for (std::size_t i = 0; i < list.size(); ++i)
          for (std::size_t j = i + 1; j < list.size(); ++j) {
            const auto a = Name(list[i]), b = Name(list[j]);
            distinct_allocations.emplace(std::min(a, b), std::max(a, b));
          }
      } else {
        throw std::invalid_argument("address invariant needs term, joint_injective or distinct_allocations "
                                    "(element bounds of input objects are tensor_properties)");
      }
    }
  }

  // Ground instantiation for a selected atom set: closes over term bounds that
  // share atoms and over joint-injectivity siblings, then emits bounds,
  // alignment and the finite injectivity lemmas of Proposition 4.1.
  address_solver::Query Instantiate(std::set<AtomId> selected, std::map<AtomId, std::size_t>& variables) {
    address_solver::Query query;
    std::vector<bool> used(term_bounds.size());
    for (bool changed = true; changed;) {
      changed = false;
      for (std::size_t i = 0; i < term_bounds.size(); ++i) {
        if (used[i]) continue;
        const auto& terms = term_bounds[i].term.terms;
        if (!std::any_of(terms.begin(), terms.end(), [&](const auto& t) { return selected.contains(t.first); }))
          continue;
        used[i] = true;
        changed = true;
        for (const auto& [atom, ignored] : terms) { (void)ignored; selected.insert(atom); }
      }
      for (const auto& group : joint_injective) {
        std::set<std::vector<Int>> tuples;
        for (const auto atom : selected)
          if (std::find(group.begin(), group.end(), atoms[atom].gather) != group.end())
            tuples.insert(atoms[atom].args);
        for (const auto& symbol : group)
          for (const auto& args : tuples)
            changed = selected.insert(Intern(GatherKey(symbol, args), symbol, args)).second || changed;
      }
    }
    variables.clear();
    for (const auto atom : selected) {
      variables.emplace(atom, query.variables.size());
      query.variables.push_back(atoms[atom].key.size() <= 96 ? atoms[atom].key
                                                             : atoms[atom].key.substr(0, 93) + "...");
    }
    const auto linear = [&](const Affine& form) {
      address_solver::Linear result{form.constant, {}};
      for (const auto& [atom, coefficient] : form.terms) result.terms.emplace_back(variables.at(atom), coefficient);
      return result;
    };
    for (std::size_t i = 0; i < term_bounds.size(); ++i) {
      if (!used[i]) continue;
      const auto& bound = term_bounds[i];
      if (bound.lower || bound.upper) query.bounds.push_back({linear(bound.term), bound.lower, bound.upper});
      for (const auto modulus : bound.multiples) query.multiples.emplace_back(linear(bound.term), modulus);
    }
    std::map<std::string, std::vector<AtomId>> applications;
    for (const auto atom : selected)
      if (!atoms[atom].gather.empty()) applications[atoms[atom].gather].push_back(atom);
    for (const auto& [symbol, members] : applications) {
      const auto policy = gather_policies.find(symbol);
      if (policy == gather_policies.end()) continue;
      for (const auto atom : members) {
        if (policy->second.lower || policy->second.upper)
          query.bounds.push_back({linear(Single(atom)), policy->second.lower, policy->second.upper});
      }
      if (!policy->second.injective) continue;
      for (std::size_t i = 0; i < members.size(); ++i)
        for (std::size_t j = i + 1; j < members.size(); ++j)
          if (atoms[members[i]].args.size() == atoms[members[j]].args.size())
            query.distinct.push_back({{variables.at(members[i]), variables.at(members[j])}});
    }
    for (const auto& group : joint_injective) {
      std::set<std::vector<Int>> tuples;
      for (const auto& symbol : group)
        if (const auto found = applications.find(symbol); found != applications.end())
          for (const auto atom : found->second) tuples.insert(atoms[atom].args);
      for (auto a = tuples.begin(); a != tuples.end(); ++a)
        for (auto b = std::next(a); b != tuples.end(); ++b) {
          if (a->size() != b->size()) continue;
          std::vector<std::pair<std::size_t, std::size_t>> clause;
          for (const auto& symbol : group)
            clause.emplace_back(variables.at(atom_ids.at(GatherKey(symbol, *a))),
                                variables.at(atom_ids.at(GatherKey(symbol, *b))));
          query.distinct.push_back(std::move(clause));
        }
    }
    return query;
  }

  void CheckInvariantsConsistent() {
    if (!solver || (term_bounds.empty() && joint_injective.empty())) return;
    std::set<AtomId> selected;
    for (const auto& bound : term_bounds)
      for (const auto& [atom, ignored] : bound.term.terms) { (void)ignored; selected.insert(atom); }
    // Unknown is not proof of consistency; each pair rechecks its own ground
    // environment before an unsat overlap is accepted.
    std::map<AtomId, std::size_t> variables;
    if (solver->consistent(Instantiate(std::move(selected), variables), options.timeout_ms) ==
        address_solver::Verdict::kUnsat)
      throw std::invalid_argument("address invariants are unsatisfiable");
  }

  // ---- The decision procedure (dynamic.pdf Section 4.4, layers 0 and 2) ----

  std::string Describe(const Shape& shape) const {
    switch (shape.base) {
      case Base::kAllocation: return names[shape.space] + " allocation " + names[shape.identity];
      case Base::kAbsolute: return "absolute " + names[shape.space] + " address";
      case Base::kSymbolic: return names[shape.space] + " " + names[shape.identity];
      case Base::kUnknown: break;
    }
    return "unresolved " + names[shape.space] + " allocation";
  }

  OverlapResult Decide(const Shape& a, std::optional<std::uint64_t> wa,
                       const Shape& b, std::optional<std::uint64_t> wb) {
    if (a.region_bytes) wa=a.region_bytes;
    if (b.region_bytes) wb=b.region_bytes;
    // Layer 0: allocation separation.
    if (a.absent || b.absent) { ++stats.layer0_disjoint; return OverlapResult::Disjoint(); }
    if (KnownSpace(names[a.space]) && KnownSpace(names[b.space]) && a.space != b.space) {
      ++stats.layer0_disjoint;
      return OverlapResult::Disjoint();
    }
    if (a.base == Base::kAllocation && b.base == Base::kAllocation && a.space == b.space &&
        KnownSpace(names[a.space]) && a.identity != b.identity &&
        !(a.alias_scope != 0 && a.alias_scope == b.alias_scope)) {
      ++stats.layer0_disjoint;
      return OverlapResult::Disjoint();
    }
    // Layer 2, ground allocation facts: Gamma |= alloc(a) != alloc(b).
    const auto named = [](const Shape& s) { return s.base == Base::kAllocation || s.base == Base::kSymbolic; };
    if (named(a) && named(b) && a.identity != b.identity &&
        distinct_allocations.contains({std::min(a.identity, b.identity), std::max(a.identity, b.identity)})) {
      ++stats.ground_disjoint;
      return OverlapResult::Disjoint();
    }
    const bool same_instance =
        (a.base == Base::kAllocation && b.base == Base::kAllocation && a.identity == b.identity &&
         a.space == b.space) ||
        (a.base == Base::kAbsolute && b.base == Base::kAbsolute && a.space == b.space) ||
        (a.base == Base::kSymbolic && b.base == Base::kSymbolic &&
         a.identity == b.identity && a.space == b.space);
    if (!same_instance)
      return OverlapResult::Unknown("allocation identity is unresolved between " + Describe(a) +
                                    " and " + Describe(b), "allocation_identity_unresolved");
    if (!wa || !wb || !a.offset || !b.offset)
      return OverlapResult::Unknown("symbolic footprint in " + Describe(a) +
                                    " is a region or has unknown extent", "memory_extent_unresolved");
    // Ovl(a,b) <=> -w(a) < Off(a) - Off(b) < w(b).
    const auto difference = Combine(*a.offset, *b.offset, -1);
    if (!difference) return OverlapResult::Unknown("symbolic offset difference overflows", "address_arithmetic_overflow");
    const Int lower = -Int(*wa), upper = Int(*wb);
    if (difference->terms.empty()) {
      const auto d = difference->constant;
      if (d <= lower || d >= upper) { ++stats.ground_disjoint; return OverlapResult::Disjoint(); }
      if (!a.region_bytes && !b.region_bytes) {
        ++stats.ground_overlap;
        return OverlapResult::Overlap(Evidence(a, *wa, b, *wb, d));
      }
      // Overlapping bounding intervals do not imply overlapping strided tiles.
      // Let SMT select one bounded element coordinate from each tile.
    }
    const Int goal_lower = a.region_bytes ? -Int(a.region_element) : lower;
    const Int goal_upper = b.region_bytes ? Int(b.region_element) : upper;
    // Layer 1: congruence. The goal is the constant plus a multiple of g, the
    // gcd of its coefficients (atoms and TMA axis strides), for every integer
    // valuation, so it needs no invariant: no such value strictly inside the
    // goal interval means disjoint.
    Int g = 0;
    for (const auto& [atom, coefficient] : difference->terms) { (void)atom; g = Gcd(g, coefficient); }
    for (const auto* shape : {&a, &b})
      for (const auto& [stride, extent] : shape->region_axes)
        if (extent > 1 && stride != 0) g = Gcd(g, Int(stride));
    if (g > 1) {
      Int residue = (difference->constant - (goal_lower + 1)) % g;
      if (residue < 0) residue += g;
      if (goal_lower + 1 + residue >= goal_upper) { ++stats.congruence_disjoint; return OverlapResult::Disjoint(); }
    }
    // Layer 1.5: S x (x - y) for two different elements x, y of one injective input object is a
    // nonzero multiple of S, so with every other term (the constant and the axes) in [lo, hi] the
    // goal is unreachable once lo + |S| >= goal_upper and hi - |S| <= goal_lower.
    if (difference->terms.size() == 2 && difference->terms[0].second == -difference->terms[1].second) {
      const auto &x = atoms[difference->terms[0].first], &y = atoms[difference->terms[1].first];
      const auto policy = gather_policies.find(x.gather);
      if (!x.gather.empty() && x.gather == y.gather && x.args != y.args && policy != gather_policies.end() &&
          policy->second.injective) {
        const Int s = difference->terms[0].second < 0 ? -difference->terms[0].second : difference->terms[0].second;
        Int lo = difference->constant, hi = difference->constant;
        for (const auto& [stride, extent] : a.region_axes)
          if (extent > 1) hi += Int(stride) * Int(extent - 1);
        for (const auto& [stride, extent] : b.region_axes)
          if (extent > 1) lo -= Int(stride) * Int(extent - 1);
        if (lo + s >= goal_upper && hi - s <= goal_lower) { ++stats.injective_disjoint; return OverlapResult::Disjoint(); }
      }
    }
    // Layer 2: SMT over Gamma AND Ovl.
    if (!solver)
      return OverlapResult::Unknown("symbolic offsets in " + Describe(a) +
                                    " need the SMT layer, which this build lacks", "address_solver_unavailable");
    std::set<AtomId> selected;
    for (const auto& [atom, ignored] : difference->terms) { (void)ignored; selected.insert(atom); }
    std::map<AtomId, std::size_t> variables;
    auto query = Instantiate(std::move(selected), variables);
    query.goal = {difference->constant, {}};
    for (const auto& [atom, coefficient] : difference->terms)
      query.goal.terms.emplace_back(variables.at(atom), coefficient);
    const auto add_axes = [&](const Shape& shape, Int sign, const char* side) {
      for (std::size_t i=0;i<shape.region_axes.size();++i) {
        auto [stride,extent]=shape.region_axes[i];
        if (extent<=1 || stride==0) continue;
        const auto variable=query.variables.size();
        query.variables.push_back(std::string("tma_")+side+"_axis"+std::to_string(i));
        query.bounds.push_back({{0,{{variable,1}}},Int(0),Int(extent-1)});
        query.goal.terms.emplace_back(variable,sign*Int(stride));
      }
    };
    add_axes(a,1,"first"); add_axes(b,-1,"second");
    query.goal_lower = goal_lower;
    query.goal_upper = goal_upper;
    ++stats.solver_queries;
    const auto answer = solver->solve(query, options.timeout_ms);
    using address_solver::Verdict;
    if (answer.environment == Verdict::kUnsat) {
      ++stats.solver_unknown;
      return OverlapResult::Unknown("address invariants are unsatisfiable for the atoms of this pair", "inconsistent_address_constraints");
    }
    if (answer.environment == Verdict::kSat && answer.goal == Verdict::kUnsat) {
      ++stats.solver_unsat;
      return OverlapResult::Disjoint();
    }
    if (answer.environment == Verdict::kSat && answer.goal == Verdict::kSat) {
      ++stats.solver_sat;
      return OverlapResult::Unknown("symbolic offsets in " + Describe(a) +
                                    " may overlap under the address invariants; " + answer.detail, "dynamic_address_overlap_unresolved");
    }
    ++stats.solver_unknown;
    return OverlapResult::Unknown(answer.detail.empty() ? "SMT result unknown" : answer.detail, "address_solver_inconclusive");
  }

  OverlapEvidence Evidence(const Shape& a, std::uint64_t wa, const Shape& b, std::uint64_t wb, Int d) const {
    OverlapEvidence evidence{"symbolic byte ranges intersect for every valuation (offset difference " +
                                 ToString(d) + ")", std::nullopt};
    if (a.base != Base::kAllocation || !a.offset->terms.empty() || !b.offset->terms.empty()) return evidence;
    const Int start = std::max(a.offset->constant, b.offset->constant);
    const Int end = std::min(a.offset->constant + Int(wa), b.offset->constant + Int(wb));
    const Int max = Int(std::numeric_limits<std::uint64_t>::max());
    if (start >= 0 && start <= max && end - start <= max)
      evidence.constant_region = ByteOverlap{names[a.space], names[a.identity],
                                             static_cast<std::uint64_t>(start),
                                             static_cast<std::uint64_t>(end - start)};
    return evidence;
  }

  // A family's footprint: one allocation, equal symbolic suffix, offsets in progression.
  // `cross_unit`: also the suffix's tensor-property lookups (FamilyFootprint::gathers) and strided
  // regions (TMA boxes) whose suffix is such lookups.
  std::optional<FamilyFootprint> Footprint(const MemoryAccessFamily& family, bool cross_unit) {
    if(!family.first.bytes || !std::holds_alternative<AddressId>(family.first.address))return {};
    const auto shape=[&](const MemoryAccess& access)->const Shape* {
      const auto id=std::get<AddressId>(access.address).value();
      auto it=binding_shapes.find(id);
      return it==binding_shapes.end()?nullptr:&shapes[it->second];
    };
    const auto* first=shape(family.first);
    if(!first || (first->region_bytes && !cross_unit) || !first->offset || !KnownSpace(names[first->space]) ||
        first->base!=Base::kAllocation)return {};
    auto suffix=*first->offset;suffix.constant=0;
    FamilyFootprint result{names[first->space],
        names[first->identity],
        names[first->alias_scope],suffix.key(),first->offset->constant,0,
        first->region_bytes?*first->region_bytes:*family.first.bytes,true,first->absent,{}};
    if(first->region_bytes){result.region_axes=first->region_axes;result.region_element=first->region_element;}
    for(std::size_t k=1;k<family.count;++k) {
      const auto* current=shape(family.at(family.first_member+k));
      if(!current || current->region_bytes!=first->region_bytes || current->region_axes!=first->region_axes ||
          current->region_element!=first->region_element || !current->offset || current->space!=first->space ||
          current->base!=first->base || current->identity!=first->identity || current->alias_scope!=first->alias_scope ||
          current->absent!=first->absent || current->offset->terms!=first->offset->terms)return {};
      Int difference;
      if(__builtin_sub_overflow(current->offset->constant,first->offset->constant,&difference))return {};
      if(k==1)result.step=difference;
      auto expected=Checked(result.step,Int(k),true);
      if(!expected || difference!=*expected)return {};
    }
    auto bias=Checked(result.step,Int(family.first_member),true);
    if(!bias)return {};
    Int offset;
    if(__builtin_sub_overflow(result.offset,*bias,&offset))return {};
    result.offset=offset;
    if(cross_unit)for(const auto& [atom,coefficient]:suffix.terms) {
      const auto& lookup=atoms[atom];
      const auto object=gather_objects.find(lookup.gather);
      if(object==gather_objects.end() || lookup.args.size()!=1) { result.gathers.clear(); break; }
      result.gathers.push_back({lookup.gather,object->second,lookup.args.front(),coefficient,
                                !pinned_gathers.contains(lookup.gather)});
    }
    // A region is a cross-unit footprint only as a lookup part (CheckCrossUnit's lookup pass decides it).
    if(first->region_bytes && result.gathers.empty())return {};
    return result;
  }

  // A family footprint as one shape: its lowest member's offset plus the lookups, and the
  // progression as one region axis. Nullopt when it is not affine or has a non-lookup suffix.
  std::optional<Shape> FamilyShape(const MemoryAccessFamily& family, const FamilyFootprint& footprint) {
    if (!footprint.affine || family.count == 0 || footprint.bytes == 0 ||
        (!footprint.symbolic.empty() && footprint.gathers.empty()))
      return std::nullopt;
    Shape shape;
    shape.space = Name(footprint.space);
    shape.base = Base::kAllocation;
    shape.identity = Name(footprint.allocation);
    shape.alias_scope = Name(footprint.alias);
    shape.absent = footprint.empty;
    const Int step = footprint.step;
    const auto lowest = Checked(step, Int(step < 0 ? family.first_member + family.count - 1 : family.first_member), true);
    const auto constant = lowest ? Checked(footprint.offset, *lowest, false) : std::nullopt;
    if (!constant) return std::nullopt;
    std::map<AtomId, Int> terms;
    for (const auto& gather : footprint.gathers) {
      const auto atom = Intern(GatherKey(gather.symbol, {gather.arg}), gather.symbol, {gather.arg});
      const auto sum = Checked(terms[atom], gather.coefficient, false);
      if (!sum) return std::nullopt;
      terms[atom] = *sum;
    }
    Affine offset{*constant, {}};
    for (const auto& [atom, coefficient] : terms)
      if (coefficient != 0) offset.terms.emplace_back(atom, coefficient);
    shape.offset = std::move(offset);
    // A region member: its own axes and element; the progression adds one axis.
    shape.region_axes = footprint.region_axes;
    shape.region_element = footprint.region_axes.empty() ? footprint.bytes : footprint.region_element;
    Int width = Int(footprint.bytes);
    if (family.count > 1 && step != 0) {
      const Int stride = step < 0 ? -step : step;
      const auto span = Checked(stride, Int(family.count - 1), true);
      const auto total = span ? Checked(*span, width, false) : std::nullopt;
      if (!total || *total > Int(UINT64_MAX)) return std::nullopt;
      width = *total;
      shape.region_axes.push_back({static_cast<std::uint64_t>(stride), static_cast<std::uint64_t>(family.count)});
    }
    if (!shape.region_axes.empty()) shape.region_bytes = static_cast<std::uint64_t>(width);
    return shape;
  }

  OverlapResult Refine(const MemoryAccess& first, const MemoryAccess& second, OverlapResult base) {
    const auto* a = std::get_if<AddressId>(&first.address);
    const auto* b = std::get_if<AddressId>(&second.address);
    if (!a || !b) return base;
    const auto sa = binding_shapes.find(a->value()), sb = binding_shapes.find(b->value());
    if (sa == binding_shapes.end() || sb == binding_shapes.end()) return base;
    ++stats.refined;
    PairKey key{sa->second, sb->second, first.bytes, second.bytes};
    if (std::tie(key.second, key.second_bytes) < std::tie(key.first, key.first_bytes))
      key = {key.second, key.first, key.second_bytes, key.first_bytes};
    if (const auto cached = cache.find(key); cached != cache.end()) {
      ++stats.cache_hits;
      lru.splice(lru.begin(), lru, cached->second);
      return cached->second->second;
    }
    auto result = Decide(shapes[key.first], key.first_bytes, shapes[key.second], key.second_bytes);
    if (capacity == 0) return result;
    if (lru.size() == capacity) {
      cache.erase(lru.back().first);
      lru.pop_back();
    }
    lru.emplace_front(key, result);
    cache.emplace(key, lru.begin());
    return result;
  }
};

SymbolicAddressModel::SymbolicAddressModel(const std::vector<ConstantAddressBinding>& bindings,
                                           std::string_view address_invariants,
                                           SymbolicSolverOptions solver,
                                           std::size_t cache_capacity,
                                           std::string_view tensor_properties)
    : impl_(std::make_unique<Impl>(bindings, address_invariants, solver, cache_capacity, tensor_properties)) {}
SymbolicAddressModel::~SymbolicAddressModel() = default;
SymbolicAddressModel::SymbolicAddressModel(SymbolicAddressModel&&) noexcept = default;
SymbolicAddressModel& SymbolicAddressModel::operator=(SymbolicAddressModel&&) noexcept = default;

std::optional<MemoryAnalysisError> SymbolicAddressModel::validate() const { return impl_->error; }

OverlapResult SymbolicAddressModel::mayOverlap(const MemoryAccess& first, const MemoryAccess& second) {
  if (impl_->error) return OverlapResult::Unknown("invalid symbolic-address model: " + impl_->error->message, "invalid_address_model");
  auto base = impl_->constants.mayOverlap(first, second);
  if (base.status != OverlapStatus::kUnknown) return base;
  return impl_->Refine(first, second, std::move(base));
}

std::unique_ptr<MemoryStorageFilter> SymbolicAddressModel::storageFilter(
    MemoryAccessView accesses) {
  return impl_->error ? nullptr : impl_->constants.storageFilter(accesses);
}

std::unique_ptr<MemoryConflictCandidates> SymbolicAddressModel::conflictCandidates(
    MemoryAccessView accesses) {
  return impl_->error ? nullptr : impl_->constants.conflictCandidates(accesses);
}

std::unique_ptr<MemoryConflictCandidates> SymbolicAddressModel::conflictOrder(
    MemoryAccessView accesses, std::span<const std::size_t> ranks) {
  return impl_->error ? nullptr : impl_->constants.conflictOrder(accesses, ranks);
}

std::optional<FamilyFootprint> SymbolicAddressModel::familyFootprint(const MemoryAccessFamily& family) {
  if(auto constant=impl_->constants.familyFootprint(family))return constant;
  return impl_->Footprint(family,false);
}
std::optional<FamilyFootprint> SymbolicAddressModel::crossUnitFootprint(const MemoryAccessFamily& family) {
  if(auto constant=impl_->constants.familyFootprint(family))return constant;
  return impl_->Footprint(family,true);
}

OverlapResult SymbolicAddressModel::translatedFamilyOverlap(const MemoryAccessFamily& a, const FamilyFootprint& x,
                                                            const MemoryAccessFamily& b, const FamilyFootprint& y) {
  if(impl_->error)return OverlapResult::Unknown("invalid symbolic-address model: "+impl_->error->message,"invalid_address_model");
  auto sa=impl_->FamilyShape(a,x),sb=impl_->FamilyShape(b,y);
  if(!sa || !sb)return OverlapResult::Unknown("family footprint is not an affine lookup form","translated_family_unsupported");
  return impl_->Decide(*sa,x.bytes,*sb,y.bytes);
}
const ConstantAddressModel& SymbolicAddressModel::constantModel() const noexcept { return impl_->constants; }
std::unique_ptr<MemoryConflictCandidates> SymbolicAddressModel::conflictCandidatesWithExemptions(
    MemoryAccessView accesses, std::span<const std::size_t> ranks,
    std::span<const AccessId> exempt_writes) {
  return impl_->error ? nullptr
      : impl_->constants.conflictCandidatesWithExemptions(accesses, ranks, exempt_writes);
}

SymbolicAddressStats SymbolicAddressModel::stats() const noexcept { return impl_->stats; }
bool SymbolicAddressModel::SolverAvailable() noexcept {
  return std::string_view(address_solver::BackendName()) != "unavailable";
}
const char* SymbolicAddressModel::SolverName() noexcept { return address_solver::BackendName(); }

}  // namespace sentryward::algorithm
