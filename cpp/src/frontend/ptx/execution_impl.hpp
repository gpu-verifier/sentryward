#pragma once
#include "sentryward/frontend/ptx/execution.h"
#include "template_execution.h"
#include "sentryward/frontend/ptx/shared_execution.h"
#include "sentryward/frontend/ptx/layouts.h"
#include "sentryward/frontend/ptx/resources.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <coroutine>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <regex>
#include <array>
#include <set>
#include <tuple>
#include <stdexcept>
#include <unordered_map>

namespace sentryward::frontend::ptx {
namespace {
// Last-resort re-executions after ordinary execution, including the shared-memory
// cluster retry, stopped on undetermined control (one kernel per process; per
// thread, since class mode executes units on look-ahead workers):
//   exact_control_pass: regions with no memory or synchronization effect;
//   data_branch_pass:   additionally Representative control flow representative outcomes.
thread_local bool exact_control_pass = false, data_branch_pass = false;
// Data predicates (table atoms): whether a run took one of their forward branches as an opaque region,
// and the re-run that takes none (SpecializeFunction's safety net when the run then fails on control).
thread_local bool data_opaque_used = false, data_opaque_disabled = false;
// Shared-memory routing (ExecutionError::shared_routed) is off while a routed run that failed is
// repeated without it (SpecializeFunctionUnrouted).
thread_local bool shared_routing_disabled = false;
// The line of the branch a routed shared-model run was started for (SharedRoutingAttempt).
thread_local std::optional<std::uint32_t> routed_branch_line;
bool Assumed(const FrontendConfig &c, const char *key) {
  return c.raw.value("semantic_assumptions", nlohmann::json::object()).value(key, false);
}
bool DataBranchesAssumed(const FrontendConfig &c) { return Assumed(c, "representative_branches") || Assumed(c, "bounded_computation_loops"); }
} // namespace
class TensorPropertyDecisions {
public:
  void Insert(const std::string &object, std::uint32_t instruction) {
    const std::lock_guard lock(mutex_);
    sites_[object].insert(instruction);
  }
  std::map<std::string, std::set<std::uint32_t>> Snapshot() const {
    const std::lock_guard lock(mutex_);
    return sites_;
  }

private:
  mutable std::mutex mutex_;
  std::map<std::string, std::set<std::uint32_t>> sites_;
};
std::map<std::string, std::set<std::uint32_t>> TensorPropertyDecisionSnapshot(const ExecutionProgram &program) {
  const auto &decisions = program.tensor_property_decisions;
  return decisions ? decisions->Snapshot() : std::map<std::string, std::set<std::uint32_t>>{};
}
std::vector<LocalMemoryDeclaration> LocalMemoryDeclarations(const Module& module, const Function& function) {
  static const std::regex pattern(R"(\.(?:b|u|s|f)(8|16|32|64)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\[\s*([0-9]+)\s*\]\s*;?\s*$)");
  std::vector<LocalMemoryDeclaration> result;
  for (const auto& directive : function.directives) {
    if (directive.name != ".local") continue;
    const auto text = std::string(module.view(directive.source));
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) continue;
    const auto width = std::stoull(match[1]) / 8;
    const auto count = std::stoull(match[3]);
    if (!count || count > UINT64_MAX / width) continue;
    std::string resource = "local:";
    for (const auto id : directive.scope) resource += std::to_string(id) + '/';
    resource += match[2].str();
    result.push_back({match[2], std::move(resource), directive.scope, count * width});
  }
  return result;
}
bool LocalStoreInstruction(const Instruction &i) {
  const auto &s = i.semantics;
  return i.opcode == "st" && s.category == "memory" && s.memory_space == "local" && s.memory_mode == "write" &&
         !s.atomic_add && !s.asynchronous && i.memory_width && !i.address_lanes;
}
LocalRegionStores::LocalRegionStores(const Module &module, const Function &function) {
  std::vector<const Instruction *> stores;
  std::unordered_map<std::string, std::vector<std::uint32_t>> definitions;
  for (const auto &i : function.instructions) {
    for (const auto &name : i.destinations) definitions[name].push_back(i.id);
    if (LocalStoreInstruction(i)) stores.push_back(&i);
  }
  if (stores.empty()) return;
  const auto declarations = LocalMemoryDeclarations(module, function);
  const auto operand = [&](const Instruction &i, std::size_t n) { return Trim(module.view(i.operands.at(n))); };
  const auto wide = [](const Instruction &i) {
    return std::any_of(i.modifiers.begin(), i.modifiers.end(), [](const auto &m) { return m == "b64" || m == "u64" || m == "s64"; });
  };
  // The chain walk's depth: it ends a cyclic chain (a single-definition register that reads
  // itself) and bounds the recursion; a deeper chain gets no shape (the store is not lifted).
  constexpr unsigned kChainDepth = 32;
  for (const auto *store : stores) {
    unsigned bits = 0, components = 1;
    for (const auto &m : store->modifiers) {
      if (m == "v2" || m == "v4" || m == "v8") components = std::stoul(m.substr(1));
      else if (m.size() > 1 && std::string_view("busf").find(m[0]) != std::string_view::npos &&
               std::all_of(m.begin() + 1, m.end(), [](unsigned char c) { return std::isdigit(c); }))
        bits = std::stoul(m.substr(1));
    }
    if (!bits || bits % 8 || store->operands.size() != 2 || *store->memory_width != components * bits / 8) continue;
    const auto address = operand(*store, 0);
    if (address.size() < 2 || address.front() != '[' || address.back() != ']') continue;
    const auto split = SplitAddress(std::string_view(address).substr(1, address.size() - 2));
    if (!split) continue;
    // The array a register addresses through the chain, memoized per name (one definition each,
    // so every name is walked once). An add of two registers continues through the one that
    // reaches an array; the other becomes an index term.
    std::map<std::string, std::optional<LocalRegionStore>> walked;
    std::function<std::optional<LocalRegionStore>(const std::string &, unsigned)> walk =
        [&](const std::string &name, unsigned depth) -> std::optional<LocalRegionStore> {
      const LocalMemoryDeclaration *array = nullptr;
      for (const auto &d : declarations)
        if (d.name == name && d.scope.size() <= store->scope.size() &&
            std::equal(d.scope.begin(), d.scope.end(), store->scope.begin()) && (!array || d.scope.size() > array->scope.size()))
          array = &d;
      if (array) {
        LocalRegionStore result;
        result.declaration = *array;
        return result;
      }
      if (depth > kChainDepth || !name.starts_with('%')) return {};
      if (const auto found = walked.find(name); found != walked.end()) return found->second;
      std::optional<LocalRegionStore> result;
      const auto found = definitions.find(name);
      const Instruction *definition = found != definitions.end() && found->second.size() == 1
          ? &function.instructions[found->second.front()] : nullptr;
      if (definition && !definition->predicate && definition->scope == store->scope && wide(*definition)) {
        const auto &d = *definition;
        if (d.opcode == "mov" && d.operands.size() == 2) result = walk(operand(d, 1), depth + 1);
        if (d.opcode == "add" && d.operands.size() == 3 &&
            std::find(d.modifiers.begin(), d.modifiers.end(), "cc") == d.modifiers.end()) {
          const auto a = operand(d, 1), b = operand(d, 2);
          if (const auto n = ParseInteger(b)) { if ((result = walk(a, depth + 1))) result->offset += *n; }
          else if (const auto n = ParseInteger(a)) { if ((result = walk(b, depth + 1))) result->offset += *n; }
          else {
            auto left = walk(a, depth + 1), right = walk(b, depth + 1);
            const auto &term = left ? b : a;
            if (!left != !right && term.starts_with('%')) {
              result = left ? std::move(left) : std::move(right);
              result->index_terms.push_back(term);
            }
          }
        }
        if (result) result->chain.push_back(d.id);
      }
      walked.emplace(name, result);
      return result;
    };
    auto shape = walk(split->first, 0);
    if (!shape) continue;
    shape->offset += split->second;
    shape->element_bytes = bits / 8;
    shape->components = components;
    if (shape->declaration.bytes % shape->element_bytes || shape->Slots() > kLiftedLocalSlots ||
        components > shape->Slots())
      continue;
    shapes_.emplace(store->id, std::move(*shape));
  }
}
namespace {
using Json = nlohmann::json;
using sentryward::algorithm::ProfileScope;
constexpr std::uint32_t kMissing = UINT32_MAX;
std::string_view TrimView(std::string_view x) {
  while (!x.empty() && std::isspace(static_cast<unsigned char>(x.front())))
    x.remove_prefix(1);
  while (!x.empty() && std::isspace(static_cast<unsigned char>(x.back())))
    x.remove_suffix(1);
  return x;
}
bool In(std::string_view x, std::initializer_list<std::string_view> xs) {
  return std::find(xs.begin(), xs.end(), x) != xs.end();
}
// An instruction an opaque compute region may skip: register computation or a private read.
// Warp-level operations depend on which lanes reach them, so a region never holds one.
bool OpaqueRegionInstruction(const Instruction &i) {
  const auto &s = i.semantics;
  const bool private_read = s.category == "memory" && s.memory_mode == "read" && s.memory_space &&
                            In(*s.memory_space, {"param", "local", "const"});
  return i.control == "none" && (private_read || (s.category == "compute" && !WarpLevelOpcode(i.opcode)));
}
bool Unknown(const Scalar &s) {
  return std::holds_alternative<std::monostate>(s);
}
std::optional<bool> Truth(const Scalar &s) {
  auto n = ScalarInteger(s);
  if (!n)
    return {};
  return *n != 0;
}
Scalar MakeDataPredicate(bool value);
bool DataPredicate(const Scalar &v);
Scalar NegatePredicate(const Scalar &s) {
  auto truth = Truth(s);
  if (truth && DataPredicate(s)) return MakeDataPredicate(!*truth);
  return truth ? Scalar(!*truth) : Scalar{};
}
Scalar PredicateBinary(std::string_view op, const Scalar &a, const Scalar &b) {
  // A data predicate operand: the plain result, a data predicate unless the other operand alone
  // decides it.
  if (DataPredicate(a) || DataPredicate(b)) {
    const auto alone = PredicateBinary(op, DataPredicate(a) ? Scalar{} : a, DataPredicate(b) ? Scalar{} : b);
    if (!Unknown(alone)) return alone;
    const auto exact = Truth(PredicateBinary(op, PlainValue(a), PlainValue(b)));
    return exact ? MakeDataPredicate(*exact) : Scalar{};
  }
  auto x = Truth(a), y = Truth(b);
  if (op == "and") {
    if ((x && !*x) || (y && !*y))
      return false;
    if (x && y)
      return true;
  }
  if (op == "or") {
    if ((x && *x) || (y && *y))
      return true;
    if (x && y)
      return false;
  }
  if (op == "xor" && x && y)
    return *x != *y;
  return {};
}
std::optional<unsigned> Width(const Instruction &i) {
  for (const auto &m : i.modifiers)
    if (m.size() > 1 && In(m.substr(0, 1), {"b", "u", "s"}) &&
        std::all_of(m.begin() + 1, m.end(),
                    [](unsigned char c) { return std::isdigit(c); }))
      return std::stoul(m.substr(1));
  return {};
}
bool Floating(const Instruction &i) {
  for (const auto &m : i.modifiers)
    if (In(m, {"f16", "f16x2", "f32", "f64", "bf16", "bf16x2"}))
      return true;
  return false;
}
// Whether some modifier is a signed type: 's' followed by a digit.
bool SignedModifier(const Instruction &i) {
  return std::any_of(i.modifiers.begin(), i.modifiers.end(), [](const auto &modifier) {
    return modifier.size() > 1 && modifier[0] == 's' && std::isdigit(modifier[1]);
  });
}
std::int64_t Int64(const Integer &n, std::string_view what) {
  if (n < std::numeric_limits<std::int64_t>::min() ||
      n > std::numeric_limits<std::int64_t>::max())
    throw std::runtime_error(std::string(what) +
                             " outside signed 64-bit address range");
  return n.convert_to<std::int64_t>();
}
// Register integers are only meaningful modulo 2^width (see Compare), so
// operations that are not congruence-preserving first reduce operands to the
// instruction's width and signedness. Shift amounts are always u32.
Integer Reduce(const Integer &n, unsigned width, bool sign) {
  Integer modulus = Integer(1) << width, r = n & (modulus - 1);
  if (sign && (r & (modulus >> 1)) != 0) r -= modulus;
  return r;
}
// Interval arithmetic on BoundedInteger operands (tensor_properties); exact like Integer.
bool FitsType(const BoundedInteger &v, unsigned width, bool sign) {
  const Integer low = sign ? -(Integer(1) << (width - 1)) : Integer(0);
  const Integer high = (Integer(1) << (width - unsigned(sign))) - 1;
  return low <= v->lower && v->upper <= high;
}
std::vector<std::string> BoundedSources(std::initializer_list<const Scalar *> operands) {
  std::vector<std::string> sources;
  for (const auto *operand : operands) {
    if (const auto *b = std::get_if<BoundedInteger>(operand))
      sources.insert(sources.end(), (*b)->sources.begin(), (*b)->sources.end());
    if (const auto *p = std::get_if<BoundedAllocationPointer>(operand))
      sources.insert(sources.end(), (*p)->sources.begin(), (*p)->sources.end());
  }
  std::sort(sources.begin(), sources.end());
  sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
  return sources;
}
Scalar MakeBounded(Integer lower, Integer upper, std::string identity, std::vector<std::string> sources,
                   std::optional<AffineForm> affine = std::nullopt, std::optional<BoundedTable> table = std::nullopt) {
  if (upper < lower) std::swap(lower, upper);
  return BoundedInteger{std::make_shared<const BoundedValue>(BoundedValue{std::move(lower), std::move(upper),
      std::move(identity), std::move(sources), std::move(affine), std::move(table)})};
}
Scalar MakeBoundedPointer(std::string resource, std::optional<std::uint64_t> extent, AffineForm offset, Integer lower,
                          Integer upper, std::vector<std::string> sources, std::string offset_identity = {},
                          std::optional<BoundedTable> table = std::nullopt,
                          std::optional<AffineForm> abstract = std::nullopt) {
  return BoundedAllocationPointer{std::make_shared<const BoundedPointerValue>(BoundedPointerValue{
      std::move(resource), extent, std::move(offset), std::move(lower), std::move(upper), std::move(sources),
      std::move(offset_identity), std::move(table), std::move(abstract)})};
}
// Tables cover keys with at most this many values (O(entries) per tabulated operation).
constexpr std::uint64_t kBoundedTableLimit = 256;
// The exact values of a bounded value over a small-domain key: its own table, or, for a
// non-affine value with at most kBoundedTableLimit values, the identity table of itself.
// Affine values keep their affine forms instead (no table is built for them).
std::optional<BoundedTable> TableOf(const BoundedInteger &v) {
  if (v->table) return v->table;
  if (v->affine || v->upper - v->lower >= Integer(kBoundedTableLimit)) return std::nullopt;
  BoundedTable table{v->identity, v->lower, {}};
  for (Integer k = v->lower; k <= v->upper; ++k) table.values.push_back(k);
  return table;
}
// Whether two tables cover the same key: its name, first value and length.
bool SameKey(const BoundedTable &a, const BoundedTable &b) {
  return a.key == b.key && a.key_lower == b.key_lower && a.values.size() == b.values.size();
}
Scalar TableValue(BoundedTable table, std::string identity, std::vector<std::string> sources) {
  const auto [low, high] = std::minmax_element(table.values.begin(), table.values.end());
  return MakeBounded(*low, *high, std::move(identity), std::move(sources), std::nullopt, std::move(table));
}
std::string BoundedName(const Scalar &value) {
  if (const auto *b = std::get_if<BoundedInteger>(&value)) return (*b)->identity;
  return ScalarInteger(value)->convert_to<std::string>();
}
// The exact affine form of an integer or a bounded value, if it has one.
std::optional<AffineForm> AffineOf(const Scalar &value) {
  if (const auto *b = std::get_if<BoundedInteger>(&value)) return (*b)->affine;
  if (const auto n = ScalarInteger(value)) return AffineForm{*n, {}};
  return std::nullopt;
}
AffineForm AffineSum(const AffineForm &a, const AffineForm &b, const Integer &scale_b = Integer(1)) {
  AffineForm result{a.constant + scale_b * b.constant, {}};
  std::map<ElementAtom, Integer> terms;
  for (const auto &[atom, coefficient] : a.terms) terms[atom] += coefficient;
  for (const auto &[atom, coefficient] : b.terms) terms[atom] += scale_b * coefficient;
  for (auto &[atom, coefficient] : terms) if (coefficient != 0) result.terms.emplace_back(atom, coefficient);
  return result;
}
AffineForm AffineScale(const AffineForm &a, const Integer &scale) {
  AffineForm result{a.constant * scale, {}};
  if (scale != 0) for (const auto &[atom, coefficient] : a.terms) result.terms.emplace_back(atom, coefficient * scale);
  return result;
}
// a / divisor when divisor > 0 divides the constant and every coefficient: every value of
// a is then a multiple of divisor, so floor and truncating division are both exact.
std::optional<AffineForm> AffineQuotient(const std::optional<AffineForm> &a, const Integer &divisor) {
  if (!a || a->constant % divisor != 0) return std::nullopt;
  AffineForm result{a->constant / divisor, {}};
  for (const auto &[atom, coefficient] : a->terms) {
    if (coefficient % divisor != 0) return std::nullopt;
    result.terms.emplace_back(atom, coefficient / divisor);
  }
  return result;
}
// The exact form of an integer or a bounded value in element atoms and element functions.
std::optional<AffineForm> AbstractOf(const Scalar &value) {
  if (const auto *b = std::get_if<BoundedInteger>(&value)) return (*b)->affine ? (*b)->affine : (*b)->abstract;
  if (const auto n = ScalarInteger(value)) return AffineForm{*n, {}};
  return std::nullopt;
}
// A canonical text of a form (an exact identity of its value).
std::string FormText(const AffineForm &form) {
  std::string text = "[" + form.constant.str();
  for (const auto &[atom, coefficient] : form.terms)
    text += "+" + coefficient.str() + "*" + atom.object + "@" + atom.byte_offset.str() + ":" + atom.element_type +
            (atom.function.empty() ? "" : "|" + atom.function);
  return text + "]";
}
// ---- table atoms (BoundedValue::exact; class mode with fixed shape-array inputs) ------------
// A table atom is a value read from (or computed from reads of) a fixed read-only table: its exact
// value is known, and its affine form in element atoms is kept for the slot forms that class
// admission compares up to δ. Every operation computes the exact result as on plain values; the
// form is computed by the bounded rules on the atoms as trusted elements over the whole table
// (bounded twins without the exact value), and kept only when the result has an affine form or
// an element-function form. So a table atom never decides anything the plain value would not.
const BoundedValue *PinnedInteger(const Scalar &v) {
  const auto *b = std::get_if<BoundedInteger>(&v);
  return b && b->value && b->value->exact ? b->value.get() : nullptr;
}
const BoundedPointerValue *PinnedPointer(const Scalar &v) {
  const auto *p = std::get_if<BoundedAllocationPointer>(&v);
  return p && p->value && p->value->exact ? p->value.get() : nullptr;
}
bool Pinned(const Scalar &v) { return PinnedInteger(v) || PinnedPointer(v); }
// A bounded value or pointer that is not a table atom (a trusted tensor_properties element).
bool UnpinnedBounded(const Scalar &v) {
  return (std::holds_alternative<BoundedInteger>(v) && !PinnedInteger(v)) ||
         (std::holds_alternative<BoundedAllocationPointer>(v) && !PinnedPointer(v));
}
// The bounded twin of a table atom: the same value without its exact value.
Scalar Unpinned(const Scalar &v) {
  if (const auto *b = PinnedInteger(v)) {
    auto copy = *b;
    copy.exact.reset();
    return BoundedInteger{std::make_shared<const BoundedValue>(std::move(copy))};
  }
  if (const auto *p = PinnedPointer(v)) {
    auto copy = *p;
    copy.exact.reset();
    return BoundedAllocationPointer{std::make_shared<const BoundedPointerValue>(std::move(copy))};
  }
  return v;
}
// `symbolic` (computed from bounded twins) pinned to the exact result `concrete`: a table atom when
// `symbolic` is a bounded integer whose range holds the exact value, else the plain result. With an
// affine or element-function form the identity is the canonical form text; without one it is table
// data (its comparisons give data predicates), named by the twin's identity, or by a fresh name when
// that one grows long (an exact value is equal to itself only by name; equal values need not share it).
thread_local std::uint64_t table_data_names = 0;
Scalar PinInteger(const Scalar &symbolic, const Scalar &concrete) {
  const auto *n = std::get_if<Integer>(&concrete);
  const auto *b = std::get_if<BoundedInteger>(&symbolic);
  if (!n || !b || !b->value || *n < (*b)->lower || *n > (*b)->upper) return concrete;
  auto value = *b->value;
  if (value.affine || value.abstract) value.identity = "atom" + FormText(value.affine ? *value.affine : *value.abstract);
  else if (value.identity.size() > 256 || value.identity == "data-predicate")
    value.identity = "data#" + std::to_string(++table_data_names);
  value.sources.clear();
  value.table.reset();
  value.exact = *n;
  return BoundedInteger{std::make_shared<const BoundedValue>(std::move(value))};
}
Scalar PinPointer(const Scalar &symbolic, const Scalar &concrete) {
  const auto *p = std::get_if<AllocationPointer>(&concrete);
  const auto *b = std::get_if<BoundedAllocationPointer>(&symbolic);
  if (!p || !b || !b->value || (*b)->resource != p->resource || (*b)->extent_bytes != p->extent_bytes ||
      ((*b)->offset_identity.empty() ? (*b)->offset.terms.empty() : !(*b)->abstract) ||
      p->offset < (*b)->lower || p->offset > (*b)->upper)
    return concrete;
  auto value = *b->value;
  value.sources.clear();
  value.table.reset();
  value.exact = p->offset;
  return BoundedAllocationPointer{std::make_shared<const BoundedPointerValue>(std::move(value))};
}
// A data predicate: the exact outcome of a comparison of table atoms that their ranges do not
// decide (BoundedValue::exact 0 or 1, no form). Every consumer reads it as that bool; a forward
// branch on it is taken as an opaque compute region where one is (the representation the trusted
// element bounds would give: a lane's choice there depends on the data, not on the unit's
// structure), and by the exact outcome elsewhere.
Scalar MakeDataPredicate(bool value) {
  BoundedValue predicate{Integer(0), Integer(1), "data-predicate", {}, std::nullopt};
  predicate.exact = Integer(value ? 1 : 0);
  return BoundedInteger{std::make_shared<const BoundedValue>(std::move(predicate))};
}
bool DataPredicate(const Scalar &v) {
  const auto *b = PinnedInteger(v);
  return b && b->identity == "data-predicate";
}
// The form of `result`, a fresh non-affine bounded result of `op` on (a, b): the operands'
// linear combination for add, sub, and mul and shl by a constant; otherwise, when one operand
// is a constant and the other k + c x atom, a new element function of that atom (the result
// is the exact value of op on exact operands, so a function of the element), bounded by the
// result's range. Any other result keeps no form.
Scalar WithAbstract(std::string_view op, const Scalar &a, const Scalar &b, Scalar result) {
  const auto *r = std::get_if<BoundedInteger>(&result);
  if (!r || (*r)->affine || (*r)->abstract) return result;
  for (const auto *operand : {&a, &b}) // an operand returned unchanged keeps its own forms
    if (const auto *o = std::get_if<BoundedInteger>(operand); o && o->value == r->value) return result;
  const auto fa = AbstractOf(a), fb = AbstractOf(b);
  // Constants are plain operands only (a table atom is a bounded element here).
  const auto x = std::holds_alternative<BoundedInteger>(a) ? std::nullopt : ScalarInteger(a);
  const auto y = std::holds_alternative<BoundedInteger>(b) ? std::nullopt : ScalarInteger(b);
  std::optional<AffineForm> form;
  if (op == "add" || op == "sub") {
    if (fa && fb) form = AffineSum(*fa, *fb, Integer(op == "add" ? 1 : -1));
  } else if (op == "mul") {
    if (fa && y) form = AffineScale(*fa, *y);
    else if (fb && x) form = AffineScale(*fb, *x);
  } else if (op == "shl") {
    if (fa && y && *y >= 0 && *y < 64) form = AffineScale(*fa, Integer(1) << y->convert_to<unsigned>());
  } else if (In(op, {"shr", "div", "rem", "and", "or", "xor", "min", "max"})) {
    const bool left = std::holds_alternative<BoundedInteger>(a);
    const auto &operand = left ? fa : fb;
    const auto &constant = left ? y : x;
    if (operand && constant && operand->terms.size() == 1) {
      auto atom = operand->terms.front().first;
      const auto inner = "lin(" + operand->constant.str() + "," + operand->terms.front().second.str() + "," +
                         (atom.function.empty() ? std::string("x") : atom.function) + ")";
      atom.function = std::string(op) + "(" + (left ? inner + "," + constant->str() : constant->str() + "," + inner) + ")";
      atom.lower = (*r)->lower;
      atom.upper = (*r)->upper;
      form = AffineForm{0, {{std::move(atom), Integer(1)}}};
    }
  }
  if (!form) return result;
  auto value = *r->value;
  value.abstract = std::move(form);
  return BoundedInteger{std::make_shared<const BoundedValue>(std::move(value))};
}
Scalar IntegerBinary(std::string_view op, const Scalar &a, const Scalar &b, const Instruction *i);
std::optional<Scalar> BoundedBinaryValue(std::string_view op, const Scalar &a, const Scalar &b,
                                         const Instruction *i);
// std::nullopt when neither operand is bounded; unknown when the result has no bound here.
std::optional<Scalar> BoundedBinary(std::string_view op, const Scalar &a, const Scalar &b,
                                    const Instruction *i) {
  // Table atoms (and plain operands): the plain result, pinned to the twins' form.
  if ((PinnedInteger(a) || PinnedInteger(b)) && !UnpinnedBounded(a) && !UnpinnedBounded(b)) {
    const auto concrete = IntegerBinary(op, PlainValue(a), PlainValue(b), i);
    if (!std::holds_alternative<Integer>(concrete)) return concrete;
    return PinInteger(BoundedBinary(op, Unpinned(a), Unpinned(b), i).value_or(Scalar{}), concrete);
  }
  auto result = BoundedBinaryValue(op, a, b, i);
  if (result) return WithAbstract(op, a, b, std::move(*result));
  return result;
}
// BoundedBinary without the forms that use element functions.
std::optional<Scalar> BoundedBinaryValue(std::string_view op, const Scalar &a, const Scalar &b,
                                         const Instruction *i) {
  const auto *p = std::get_if<BoundedInteger>(&a), *q = std::get_if<BoundedInteger>(&b);
  if (!p && !q) return std::nullopt;
  const auto x = ScalarInteger(a), y = ScalarInteger(b);
  if ((!p && !x) || (!q && !y)) return Scalar{};
  const Integer al = p ? (*p)->lower : *x, ah = p ? (*p)->upper : *x, bl = q ? (*q)->lower : *y, bh = q ? (*q)->upper : *y;
  const auto identity = std::string(op) + "(" + BoundedName(a) + "," + BoundedName(b) + ")";
  const auto sources = BoundedSources({&a, &b});
  // Tables over one key (or a table and a constant): concrete execution's operator per
  // entry, exact and at least as tight as the intervals. Any entry it leaves unknown
  // (or two keys) falls back to the interval rules below.
  {
    const auto ta = p ? TableOf(*p) : std::nullopt, tb = q ? TableOf(*q) : std::nullopt;
    if ((!p || ta) && (!q || tb) && (ta || tb) && (!ta || !tb || SameKey(*ta, *tb))) {
      const auto &key = ta ? *ta : *tb;
      BoundedTable table{key.key, key.key_lower, {}};
      table.values.reserve(key.values.size());
      for (std::size_t k = 0; k < key.values.size(); ++k) {
        const auto entry = ScalarInteger(IntegerBinary(op, Scalar(ta ? ta->values[k] : *x),
                                                       Scalar(tb ? tb->values[k] : *y), i));
        if (!entry) { table.values.clear(); break; }
        table.values.push_back(*entry);
      }
      if (!table.values.empty()) return TableValue(std::move(table), identity, sources);
    }
  }
  const auto fa = AffineOf(a), fb = AffineOf(b);
  if (op == "add") return MakeBounded(al + bl, ah + bh, identity, sources,
      fa && fb ? std::optional(AffineSum(*fa, *fb)) : std::nullopt);
  if (op == "sub") return MakeBounded(al - bh, ah - bl, identity, sources,
      fa && fb ? std::optional(AffineSum(*fa, *fb, Integer(-1))) : std::nullopt);
  if (op == "mul") {
    const std::array<Integer, 4> corners{al * bl, al * bh, ah * bl, ah * bh};
    std::optional<AffineForm> affine;
    if (!p && fb) affine = AffineScale(*fb, *x);
    if (!q && fa) affine = AffineScale(*fa, *y);
    return MakeBounded(*std::min_element(corners.begin(), corners.end()),
                       *std::max_element(corners.begin(), corners.end()), identity, sources, affine);
  }
  if (op == "shl" && !q && *y >= 0 && *y < 64) {
    const Integer scale = Integer(1) << y->convert_to<unsigned>();
    return MakeBounded(al * scale, ah * scale, identity, sources,
                       fa ? std::optional(AffineScale(*fa, scale)) : std::nullopt);
  }
  // shr, div and rem by a constant of a value that fits the instruction's type:
  // concrete execution's own operators applied to the bounds. Both shifts (logical
  // on a nonnegative value, arithmetic) and division by c > 0 are monotone; a
  // nonnegative remainder stays in [0, c-1], exactly [lo%c, hi%c] within one quotient.
  // An affine value that 2^k (shr, k < width) or c (div) divides exactly keeps its quotient
  // form: (x << 32) >> 30 is 4x when x << 32 fits the type, i.e. when the bounds prove that
  // sign-extending the low 32 bits of x leaves x unchanged.
  if (In(op, {"shr", "div", "rem"}) && !q && i) {
    const auto width = Width(*i);
    if (!width) return Scalar{};
    const bool sign = Has(*i, "s" + std::to_string(*width));
    if (!FitsType(*p, *width, sign)) return Scalar{};
    if (op == "shr") {
      const Integer n = Reduce(*y, 32, false);
      if (n > 1000000) return Scalar{};
      const auto k = n.convert_to<unsigned>();
      return MakeBounded(Integer(al >> k), Integer(ah >> k), identity, sources,
                         k < *width ? AffineQuotient(fa, Integer(1) << k) : std::nullopt);
    }
    const Integer c = Reduce(*y, *width, sign);
    if (c != *y || c <= 0) return Scalar{};
    if (op == "div") return MakeBounded(Integer(al / c), Integer(ah / c), identity, sources, AffineQuotient(fa, c));
    if (al < 0) return Scalar{};
    if (al / c == ah / c) return MakeBounded(Integer(al % c), Integer(ah % c), identity, sources);
    return MakeBounded(Integer(0), std::min<Integer>(ah, c - 1), identity, sources);
  }
  // A nonnegative value masked by a nonnegative constant stays within [0, min(value, mask)];
  // a low-bit mask that covers the whole range leaves the value unchanged.
  if (op == "and" && !q && *y >= 0 && al >= 0) {
    if (((*y + 1) & *y) == 0 && ah <= *y) return a;
    return MakeBounded(Integer(0), std::min(ah, *y), identity, sources);
  }
  // A negative power-of-two mask -2^k rounds down to a multiple of 2^k: monotone.
  if (op == "and" && !q && *y < 0 && ((-*y) & (-*y - 1)) == 0)
    return MakeBounded(Integer(al & *y), Integer(ah & *y), identity, sources);
  // Two's complement over the exact values: x & y clears bits of x, so it is at most a
  // nonnegative x, at most x when y < 0, and nonnegative when either operand is; two
  // negative operands share their sign bits above 2^m >= -min(x, y).
  if (op == "and") {
    const bool a_nonnegative = al >= 0, b_nonnegative = bl >= 0;
    Integer lower = 0, upper;
    if (!a_nonnegative && !b_nonnegative) {
      const Integer magnitude = -std::min(al, bl);
      unsigned m = 0;
      while ((Integer(1) << m) < magnitude) ++m;
      lower = -(Integer(1) << m);
    }
    if (a_nonnegative && b_nonnegative) upper = std::min(ah, bh);
    else if (a_nonnegative) upper = ah;
    else if (b_nonnegative) upper = bh;
    else if (ah < 0 && bh < 0) upper = std::min(ah, bh);
    else upper = std::max(ah, bh);
    return MakeBounded(lower, upper, identity, sources);
  }
  // Nonnegative operands below 2^n: x | y lies in [max(x, y), x + y] and below 2^n; so does
  // x ^ y, from 0.
  if (In(op, {"or", "xor"}) && al >= 0 && bl >= 0) {
    unsigned n = 0;
    while ((Integer(1) << n) <= std::max(ah, bh)) ++n;
    const Integer limit = (Integer(1) << n) - 1;
    return op == "or" ? MakeBounded(std::max(al, bl), std::min(Integer(ah + bh), limit), identity, sources)
                      : MakeBounded(Integer(0), limit, identity, sources);
  }
  if (In(op, {"min", "max"}) && i) {
    const auto width = Width(*i);
    if (!width) return Scalar{};
    const bool sign = Has(*i, "s" + std::to_string(*width));
    for (const auto *v : {p, q}) if (v && !FitsType(*v, *width, sign)) return Scalar{};
    if (!p && Reduce(*x, *width, sign) != *x) return Scalar{};
    if (!q && Reduce(*y, *width, sign) != *y) return Scalar{};
    // An operation the bounds decide returns that operand unchanged.
    if (op == "min" ? ah <= bl : al >= bh) return a;
    if (op == "min" ? bh <= al : bl >= ah) return b;
    return op == "min" ? MakeBounded(std::min(al, bl), std::min(ah, bh), identity, sources)
                       : MakeBounded(std::max(al, bl), std::max(ah, bh), identity, sources);
  }
  return Scalar{};
}
Scalar IntegerBinary(std::string_view op, const Scalar &a, const Scalar &b,
                     const Instruction *i = nullptr) {
  if (auto bounded = BoundedBinary(op, a, b, i)) return *bounded;
  auto x = ScalarInteger(a), y = ScalarInteger(b);
  if (op == "mul" && (Unknown(a) || x) && (Unknown(b) || y) &&
      ((x && *x == 0) || (y && *y == 0))) return Integer(0);
  if (!x || !y)
    return {};
  if (auto width = i ? Width(*i) : std::nullopt;
      width && In(op, {"shr", "min", "max", "div", "rem"})) {
    const bool sign = Has(*i, "s" + std::to_string(*width));
    *x = Reduce(*x, *width, sign);
    *y = op == "shr" ? Reduce(*y, 32, false) : Reduce(*y, *width, sign);
    if (op == "div" || op == "rem") {
      if (*y == 0) return {};
      // boost::multiprecision truncates toward zero, as PTX div/rem do.
      return op == "div" ? Integer(*x / *y) : Integer(*x % *y);
    }
  }
  if (op == "add")
    return Integer(*x + *y);
  if (op == "sub")
    return Integer(*x - *y);
  if (op == "mul")
    return Integer(*x * *y);
  if (op == "and")
    return Integer(*x & *y);
  if (op == "or")
    return Integer(*x | *y);
  if (op == "xor")
    return Integer(*x ^ *y);
  if (op == "min")
    return Integer(std::min(*x, *y));
  if (op == "max")
    return Integer(std::max(*x, *y));
  if (op == "shl" || op == "shr") {
    if (*y < 0)
      return {};
    if (*y > 1000000)
      throw std::runtime_error(
          "PTX shift exceeds finite execution resource limit");
    auto n = y->convert_to<unsigned>();
    if (op == "shl")
      return Integer(*x << n);
    return Integer(*x >> n);
  }
  if (op == "div" || op == "rem") {
    if (*y == 0)
      return {};
    Integer q = *x / *y;
    Integer r = *x % *y;
    if (r != 0 && ((*x < 0) != (*y < 0))) {
      --q;
      r += *y;
    }
    return op == "div" ? q : r;
  }
  return {};
}
std::string AllocationIdentity(const AllocationPointer &p, const FrontendConfig &c) {
  for (const auto &r : c.resources)
    if (r.value("id", "") == p.resource && r.contains("allocation"))
      return r["allocation"].value("object", p.resource);
  return p.resource;
}
// A pointer plus a bounded affine offset stays inside its resource (tensor_properties).
// The exact offset range must lie in (-2^63, 2^63): it then determines the hardware address
// (bounded values are exact and congruent to their registers modulo 2^64). As for an
// AllocationPointer, an offset may leave the resource on the way (a pointer decremented
// before a loop); a consumer that places the access (a footprint, an address space)
// requires the range inside the resource.
std::optional<Scalar> BoundedPointerArithmetic(std::string_view op, const Scalar &a, const Scalar &b) {
  const bool pointer_a = std::holds_alternative<BoundedAllocationPointer>(a) ||
      (std::holds_alternative<AllocationPointer>(a) && std::holds_alternative<BoundedInteger>(b));
  const bool pointer_b = std::holds_alternative<BoundedAllocationPointer>(b) ||
      (std::holds_alternative<AllocationPointer>(b) && std::holds_alternative<BoundedInteger>(a));
  if (!pointer_a && !pointer_b) return std::nullopt;
  if ((op != "add" && op != "sub") || (op == "sub" && !pointer_a)) return Scalar{};
  const auto &pointer = pointer_a ? a : b, &offset = pointer_a ? b : a;
  std::string resource;
  std::optional<std::uint64_t> extent;
  AffineForm base;
  Integer lower, upper;
  const BoundedPointerValue *tabled = nullptr; // a base with a non-affine offset
  if (const auto *p = std::get_if<AllocationPointer>(&pointer)) {
    resource = p->resource; extent = p->extent_bytes; base = {p->offset, {}}; lower = upper = p->offset;
  } else if (const auto *q = std::get_if<BoundedAllocationPointer>(&pointer)) {
    resource = (*q)->resource; extent = (*q)->extent_bytes; base = (*q)->offset;
    lower = (*q)->lower; upper = (*q)->upper;
    if (!(*q)->offset_identity.empty()) tabled = q->value.get();
  } else return Scalar{};
  const auto form = AffineOf(offset);
  const Integer sign = op == "sub" ? -1 : 1;
  if (!form || tabled) {
    // A non-affine offset: its exact byte offsets over one small-domain key (a table
    // base plus a constant or a same-key table; a concrete base plus a table), and its form
    // in element atoms and element functions when both parts have one.
    const auto *bounded = std::get_if<BoundedInteger>(&offset);
    const auto constant = ScalarInteger(offset);
    const auto offsets_of = bounded ? TableOf(*bounded) : std::nullopt;
    const auto base_form = tabled ? tabled->abstract : std::optional<AffineForm>(base);
    const auto offset_form = AbstractOf(offset);
    std::optional<AffineForm> abstract;
    if (base_form && offset_form) abstract = AffineSum(*base_form, *offset_form, sign);
    std::optional<BoundedTable> offsets;
    if (tabled) {
      if (tabled->table) {
        offsets = *tabled->table;
        if (constant) {
          for (auto &value : offsets->values) value += sign * *constant;
        } else if (offsets_of && SameKey(*offsets_of, *offsets)) {
          for (std::size_t k = 0; k < offsets->values.size(); ++k) offsets->values[k] += sign * offsets_of->values[k];
        } else offsets.reset();
      }
    } else if (offsets_of && base.terms.empty()) {
      offsets = *offsets_of;
      for (auto &value : offsets->values) value = base.constant + sign * value;
    }
    Integer lo, hi;
    if (offsets) {
      const auto [low, high] = std::minmax_element(offsets->values.begin(), offsets->values.end());
      lo = *low;
      hi = *high;
    } else if (abstract && (bounded || constant)) {
      // The exact range: the base's range plus (minus) the offset's.
      Integer low = bounded ? (*bounded)->lower : *constant, high = bounded ? (*bounded)->upper : *constant;
      if (sign < 0) { std::swap(low, high); low = -low; high = -high; }
      lo = lower + low;
      hi = upper + high;
    } else return Scalar{};
    if (lo <= -(Integer(1) << 63) || hi >= (Integer(1) << 63)) return Scalar{};
    const auto identity = (tabled ? tabled->offset_identity : base.terms.empty() ? base.constant.str() : FormText(base)) +
        (op == "sub" ? "-" : "+") + BoundedName(offset);
    return MakeBoundedPointer(std::move(resource), extent, AffineForm{0, {}}, lo, hi, BoundedSources({&a, &b}),
                              identity, std::move(offsets), std::move(abstract));
  }
  Integer low, high;
  if (const auto *bounded = std::get_if<BoundedInteger>(&offset)) { low = (*bounded)->lower; high = (*bounded)->upper; }
  else low = high = *ScalarInteger(offset);
  if (sign < 0) { std::swap(low, high); low = -low; high = -high; }
  lower += low; upper += high;
  if (lower <= -(Integer(1) << 63) || upper >= (Integer(1) << 63)) return Scalar{};
  return MakeBoundedPointer(std::move(resource), extent, AffineSum(base, *form, sign), std::move(lower),
                            std::move(upper), BoundedSources({&a, &b}));
}
// The 8-byte pointer-table slot read through a bounded pointer whose offset plus `shift` is slot
// aligned for every element value: exact when the slots of its range hold pointers into one object
// at offsets affine in the slot index (offset = first entry + stride x index); otherwise unknown.
Scalar BoundedTableEntry(const std::map<std::uint64_t, Scalar> &table, const BoundedPointerValue &pointer,
                         const Integer &shift) {
  const auto &form = pointer.offset;
  const Integer first = pointer.lower + shift, last = pointer.upper + shift;
  if (!pointer.offset_identity.empty() || (form.constant + shift) % 8 != 0 || first < 0 || first % 8 != 0 ||
      last - first >= Integer(8 * kBoundedTableLimit))
    return {};
  for (const auto &[atom, coefficient] : form.terms)
    if (coefficient % 8 != 0) return {};
  std::optional<AllocationPointer> base;
  Integer stride = 0;
  for (Integer slot = first; slot <= last; slot += 8) {
    const auto found = table.find(slot.convert_to<std::uint64_t>());
    const auto *entry = found == table.end() ? nullptr : std::get_if<AllocationPointer>(&found->second);
    if (!entry || (base && (entry->resource != base->resource || entry->extent_bytes != base->extent_bytes)))
      return {};
    if (!base) base = *entry;
    else if (slot == first + 8) stride = entry->offset - base->offset;
    else if (entry->offset - base->offset != stride * ((slot - first) / 8)) return {};
  }
  if (!base) return {};
  if (stride == 0) return *base;
  AffineForm offset{base->offset + stride * ((form.constant + shift - first) / 8), {}};
  for (const auto &[atom, coefficient] : form.terms) offset.terms.emplace_back(atom, coefficient / 8 * stride);
  const Integer end = base->offset + stride * ((last - first) / 8);
  const auto [low, high] = std::minmax(base->offset, end);
  if (low < 0 || high >= (Integer(1) << 63)) return {};
  return MakeBoundedPointer(base->resource, base->extent_bytes, std::move(offset), low, high, pointer.sources);
}
// Whether loads of `resource` read element atoms (a fixed shape-array table, a tensor_properties
// object or a pointer table): their slots stay at plain offsets, so that the δ of the object is the
// shift of its element reads, which the atoms' byte offsets carry.
bool AtomSource(const FrontendConfig &c, const std::string &resource) {
  for (const char *key : {"global_memory_inputs", "tensor_property_inputs", "pointer_array_inputs"})
    if (const auto found = c.fixed_input.find(key); found != c.fixed_input.end() && found->is_array())
      for (const auto &entry : *found)
        if (entry.value("resource", "") == resource) return true;
  return false;
}
std::optional<Scalar> PointerArithmetic(std::string_view op, const Scalar &a,
                                        const Scalar &b, const FrontendConfig &c) {
  // Table atoms: the plain result; a pointer pinned to the twins' bounded pointer when it has one
  // and does not point into an atom source.
  if ((Pinned(a) || Pinned(b)) && !UnpinnedBounded(a) && !UnpinnedBounded(b)) {
    auto concrete = PointerArithmetic(op, PlainValue(a), PlainValue(b), c);
    if (!concrete || !std::holds_alternative<AllocationPointer>(*concrete) ||
        AtomSource(c, std::get<AllocationPointer>(*concrete).resource))
      return concrete;
    return PinPointer(BoundedPointerArithmetic(op, Unpinned(a), Unpinned(b)).value_or(Scalar{}), *concrete);
  }
  if (auto bounded = BoundedPointerArithmetic(op, a, b)) return bounded;
  bool ap = a.index() >= 3 && !std::holds_alternative<DescriptorFragment>(a) &&
      !std::holds_alternative<BoundedInteger>(a);
  bool bp = b.index() >= 3 && !std::holds_alternative<DescriptorFragment>(b) &&
      !std::holds_alternative<BoundedInteger>(b);
  if (!ap && !bp)
    return {};
  auto x = ScalarInteger(a), y = ScalarInteger(b);
  if (op=="shr" && y && *y==32 && std::holds_alternative<AllocationPointer>(a))
    return Scalar(*PointerFragment(a,32,32));
  if (const auto *bits=std::get_if<PointerBits>(&a); bits && op=="shl" && y && *y>=0 && *y+bits->destination_bit+bits->width<=64) {
    auto shifted=*bits; shifted.destination_bit+=y->convert_to<unsigned>(); return Scalar(shifted);
  }
  if (op=="or") {
    const auto *left=std::get_if<PointerBits>(&a), *right=std::get_if<PointerBits>(&b);
    if (left && right) {
      if (left->destination_bit>right->destination_bit) std::swap(left,right);
      if (*left->pointer==*right->pointer && left->source_bit==0 && left->destination_bit==0 && left->width==32 &&
          right->source_bit==32 && right->destination_bit==32 && right->width==32) {
        auto whole=*left; whole.width=64; return RestorePointer(whole);
      }
    }
  }
  auto advance = [](const Scalar &p, const Integer &n) -> Scalar {
    if (auto v = std::get_if<ParameterPointer>(&p))
      return ParameterPointer{v->region, Integer(v->offset + n)};
    if (auto v = std::get_if<AllocationPointer>(&p))
      return AllocationPointer{v->resource, Integer(v->offset + n),
                               v->extent_bytes};
    if (auto v = std::get_if<ClusterSharedPointer>(&p))
      return ClusterSharedPointer{Integer(v->local_address + n),
                                  v->target_rank};
    return {};
  };
  if (op == "add") {
    if (ap && y)
      return advance(a, *y);
    if (bp && x)
      return advance(b, *x);
  }
  if (op == "sub") {
    if (ap && y)
      return advance(a, Integer(-*y));
    if (auto left = std::get_if<ParameterPointer>(&a))
      if (auto right = std::get_if<ParameterPointer>(&b);
          right && left->region == right->region)
        return Scalar(Integer(left->offset - right->offset));
    if (auto left = std::get_if<AllocationPointer>(&a))
      if (auto right = std::get_if<AllocationPointer>(&b);
          right && AllocationIdentity(*left,c) == AllocationIdentity(*right,c))
        return Scalar(Integer(left->offset - right->offset));
    if (auto left = std::get_if<ClusterSharedPointer>(&a))
      if (auto right = std::get_if<ClusterSharedPointer>(&b);
          right && left->target_rank == right->target_rank)
        return Scalar(Integer(left->local_address - right->local_address));
  }
  return Scalar{};
}
Scalar IntegerProduct(const Instruction &i, const Scalar &a, const Scalar &b) {
  if ((PinnedInteger(a) || PinnedInteger(b)) && !UnpinnedBounded(a) && !UnpinnedBounded(b)) {
    const auto concrete = IntegerProduct(i, PlainValue(a), PlainValue(b));
    if (!std::holds_alternative<Integer>(concrete)) return concrete;
    return PinInteger(IntegerProduct(i, Unpinned(a), Unpinned(b)), concrete);
  }
  if (std::holds_alternative<BoundedInteger>(a) || std::holds_alternative<BoundedInteger>(b)) {
    // mul.lo is congruent to the exact product; mul.wide is exact for in-range operands, and
    // mul.hi is its high half, a monotone shift of it.
    const auto width = Width(i);
    if (!width) return {};
    const bool sign = Has(i, "s" + std::to_string(*width));
    for (const auto *v : {&a, &b}) {
      if (const auto *bounded = std::get_if<BoundedInteger>(v); bounded && !FitsType(*bounded, *width, sign)) return {};
      if (const auto n = ScalarInteger(*v); n && Reduce(*n, *width, sign) != *n) return {};
    }
    auto product = BoundedBinary("mul", a, b, &i).value_or(Scalar{});
    if (!Has(i, "hi")) return product;
    const auto *exact = std::get_if<BoundedInteger>(&product);
    if (!exact) return {};
    const auto identity = "hi(" + (*exact)->identity + ")";
    if (auto table = (*exact)->table) {
      for (auto &value : table->values) value = Integer(value >> *width);
      return TableValue(std::move(*table), identity, (*exact)->sources);
    }
    return MakeBounded(Integer((*exact)->lower >> *width), Integer((*exact)->upper >> *width), identity,
                       (*exact)->sources);
  }
  auto x = ScalarInteger(a), y = ScalarInteger(b);
  if ((Unknown(a) || x) && (Unknown(b) || y) &&
      ((x && *x == 0) || (y && *y == 0))) return Integer(0);
  if (!x || !y)
    return {};
  auto width = Width(i);
  if (!width)
    return Integer(*x * *y);
  Integer modulus = Integer(1) << *width, mask = modulus - 1;
  bool sign = Has(i, "s" + std::to_string(*width));
  Integer lhs = *x & mask, rhs = *y & mask;
  if (sign) {
    if ((lhs & (modulus >> 1)) != 0)
      lhs -= modulus;
    if ((rhs & (modulus >> 1)) != 0)
      rhs -= modulus;
  }
  Integer product = lhs * rhs;
  if (Has(i, "wide"))
    return product;
  if (Has(i, "hi"))
    return Integer(product >> *width);
  Integer result = product & mask;
  if (sign && (result & (modulus >> 1)) != 0)
    result -= modulus;
  return result;
}
Scalar CompareInteger(std::string_view relation, const Integer &a,
                      const Integer &b) {
  if (relation == "eq")
    return a == b;
  if (relation == "ne")
    return a != b;
  if (relation == "lt")
    return a < b;
  if (relation == "le")
    return a <= b;
  if (relation == "gt")
    return a > b;
  if (relation == "ge")
    return a >= b;
  return {};
}
Scalar ExactIntegerToFloat(const Scalar &value, const Instruction &instruction) {
  std::vector<std::string> types;
  for (const auto &m : instruction.modifiers)
    if (In(m,{"s8","u8","s16","u16","s32","u32","s64","u64","f16","f32","f64"})) types.push_back(m);
    else if (!In(m,{"rn","rz","rm","rp","ftz"})) return {};
  auto number=ScalarInteger(value);
  if (!number || types.size()!=2 || types[0][0]!='f' || !In(types[1].substr(0,1),{"s","u"})) return {};
  const auto width=std::stoul(types[1].substr(1));
  Integer n=*number&((Integer(1)<<width)-1);
  if (types[1][0]=='s' && ((n>>(width-1))&1)!=0) n-=Integer(1)<<width;
  if (n==0) return Integer(0);
  const bool negative=n<0; if (negative) n=-n;
  const unsigned fraction=types[0]=="f16"?10:types[0]=="f32"?23:52;
  const unsigned exponent_bits=types[0]=="f16"?5:types[0]=="f32"?8:11;
  const unsigned high=boost::multiprecision::msb(n), bias=(1U<<(exponent_bits-1))-1;
  if (high>bias) return {};
  Integer significand;
  if (high>fraction) {
    const auto shift=high-fraction;
    if ((n&((Integer(1)<<shift)-1))!=0) return {};
    significand=n>>shift;
  } else significand=n<<(fraction-high);
  return Integer((Integer(negative)<<(fraction+exponent_bits)) |
      (Integer(high+bias)<<fraction) | (significand&((Integer(1)<<fraction)-1)));
}
// Whether one of the six integer relations holds between a value and itself.
bool ReflexiveRelation(std::string_view relation) {
  return relation == "eq" || relation == "le" || relation == "ge";
}
// The integer relation between values in [al, ah] and [bl, bh], where the ranges
// decide it: eq and ne for disjoint ranges, and the four order relations.
std::optional<bool> IntervalOrder(std::string_view relation, const Integer &al, const Integer &ah,
                                  const Integer &bl, const Integer &bh) {
  if (relation == "eq" || relation == "ne") {
    if (ah < bl || bh < al) return relation == "ne";
    return std::nullopt;
  }
  if (relation == "lt") { if (ah < bl) return true; if (al >= bh) return false; }
  if (relation == "le") { if (ah <= bl) return true; if (al > bh) return false; }
  if (relation == "gt") { if (al > bh) return true; if (ah <= bl) return false; }
  if (relation == "ge") { if (al >= bh) return true; if (ah < bl) return false; }
  return std::nullopt;
}
// An integer comparison decided by the operands' bounds; values that may wrap at
// the comparison width stay unknown.
Scalar BoundedCompare(const Instruction &i, std::string_view relation, const Scalar &a, const Scalar &b) {
  const auto *p = std::get_if<BoundedInteger>(&a), *q = std::get_if<BoundedInteger>(&b);
  const auto x = ScalarInteger(a), y = ScalarInteger(b);
  const auto width = Width(i);
  if ((!p && !x) || (!q && !y) || !width || Floating(i)) return {};
  const bool sign = Has(i, "s" + std::to_string(*width));
  // eq and ne compare the values modulo 2^width (exact values congruent to the registers), so
  // they need no range check; the order relations need values that fit the type.
  const bool equality = relation == "eq" || relation == "ne";
  if (!equality)
    for (const auto *v : {p, q}) if (v && !FitsType(*v, *width, sign)) return {};
  const Integer al = p ? (*p)->lower : Reduce(*x, *width, sign), ah = p ? (*p)->upper : al;
  const Integer bl = q ? (*q)->lower : Reduce(*y, *width, sign), bh = q ? (*q)->upper : bl;
  const auto congruent = [&](const Integer &u, const Integer &v) { return Reduce(u - v, *width, false) == 0; };
  if (p && q && *p == *q && In(relation, {"eq", "ne", "le", "ge", "lt", "gt"}))
    return ReflexiveRelation(relation);
  // Tables over one key (or a table and a constant): the relation per entry, decided
  // when every entry agrees.
  if (In(relation, {"eq", "ne", "le", "ge", "lt", "gt"})) {
    const auto ta = p ? TableOf(*p) : std::nullopt, tb = q ? TableOf(*q) : std::nullopt;
    if ((!p || ta) && (!q || tb) && (ta || tb) && (!ta || !tb || SameKey(*ta, *tb))) {
      const auto size = (ta ? *ta : *tb).values.size();
      std::optional<bool> decided;
      bool agree = true;
      for (std::size_t k = 0; agree && k < size; ++k) {
        const Integer u = ta ? ta->values[k] : al, v = tb ? tb->values[k] : bl;
        const bool holds = relation == "eq" ? congruent(u, v) : relation == "ne" ? !congruent(u, v) : relation == "lt" ? u < v :
                           relation == "le" ? u <= v : relation == "gt" ? u > v : u >= v;
        if (!decided) decided = holds;
        else if (*decided != holds) agree = false;
      }
      if (agree && decided) return *decided;
    }
  }
  if (equality) {
    // The difference lies in [al - bh, ah - bl]: unequal when no multiple of 2^width lies
    // there, equal when it is a single multiple.
    const Integer modulus = Integer(1) << *width, low = al - bh, high = ah - bl;
    Integer rest = low % modulus;
    if (rest < 0) rest += modulus;
    const Integer first = rest == 0 ? low : low + (modulus - rest);
    if (first > high) return relation == "ne";
    if (low == high) return relation == "eq";
    return {};
  }
  if (const auto order = IntervalOrder(relation, al, ah, bl, bh)) return *order;
  return {};
}
Scalar Compare(const Instruction &i, std::string relation, const Scalar &a,
               const Scalar &b, const FrontendConfig &c) {
  // Table atoms: the trusted-bounds comparison of their bounded twins when it decides (then it is
  // the exact outcome), else the exact outcome as a data predicate.
  if (Pinned(a) || Pinned(b)) {
    if (!UnpinnedBounded(a) && !UnpinnedBounded(b) && !DataPredicate(a) && !DataPredicate(b)) {
      auto bounded = Compare(i, relation, Unpinned(a), Unpinned(b), c);
      if (!Unknown(bounded)) return bounded;
    }
    const auto exact = Compare(i, std::move(relation), PlainValue(a), PlainValue(b), c);
    const auto truth = Truth(exact);
    return truth ? MakeDataPredicate(*truth) : exact;
  }
  auto x = ScalarInteger(a), y = ScalarInteger(b);
  if (Unknown(a) || Unknown(b))
    return {};
  if (std::holds_alternative<BoundedInteger>(a) || std::holds_alternative<BoundedInteger>(b))
    return BoundedCompare(i, relation, a, b);
  if (std::holds_alternative<BoundedAllocationPointer>(a) || std::holds_alternative<BoundedAllocationPointer>(b)) {
    // A bounded pointer is never null; pointers into different allocations differ; within one
    // resource, the offset ranges decide the comparison or leave it unknown.
    const auto range = [](const Scalar &v) -> std::optional<std::tuple<std::string, Integer, Integer>> {
      if (const auto *p = std::get_if<AllocationPointer>(&v)) return std::tuple{p->resource, p->offset, p->offset};
      if (const auto *q = std::get_if<BoundedAllocationPointer>(&v)) return std::tuple{(*q)->resource, (*q)->lower, (*q)->upper};
      return std::nullopt;
    };
    const auto left = range(a), right = range(b);
    if ((!left && x && *x == 0) || (!right && y && *y == 0))
      return In(relation, {"eq", "ne"}) ? Scalar(relation == "ne") : Scalar{};
    if (!left || !right || Floating(i)) return {};
    const auto &[left_resource, al, ah] = *left;
    const auto &[right_resource, bl, bh] = *right;
    if (left_resource != right_resource) {
      const AllocationPointer lp{left_resource, 0, std::nullopt}, rp{right_resource, 0, std::nullopt};
      if (AllocationIdentity(lp, c) != AllocationIdentity(rp, c) && In(relation, {"eq", "ne"})) return relation == "ne";
      return {};
    }
    if (a == b && In(relation, {"eq", "ne", "le", "ge", "lt", "gt"}))
      return ReflexiveRelation(relation);
    if (const auto order = IntervalOrder(relation, al, ah, bl, bh)) return *order;
    return {};
  }
  if (Has(i, "f32")) {
    if (!x || !y)
      return {};
    auto bits = [](const Integer &n, bool ftz) {
      auto b = (n & 0xffffffff).convert_to<std::uint32_t>();
      if (ftz && (b & 0x7f800000) == 0 && (b & 0x7fffff))
        b &= 0x80000000;
      return b;
    };
    float lhs = std::bit_cast<float>(bits(*x, Has(i, "ftz"))),
          rhs = std::bit_cast<float>(bits(*y, Has(i, "ftz")));
    bool nan = std::isnan(lhs) || std::isnan(rhs);
    if (relation == "num")
      return !nan;
    if (relation == "nan")
      return nan;
    if (In(relation, {"equ", "neu", "ltu", "leu", "gtu", "geu"})) {
      if (nan)
        return true;
      relation.pop_back();
    } else if (nan)
      return false;
    if (relation == "eq")
      return lhs == rhs;
    if (relation == "ne")
      return lhs != rhs;
    if (relation == "lt")
      return lhs < rhs;
    if (relation == "le")
      return lhs <= rhs;
    if (relation == "gt")
      return lhs > rhs;
    if (relation == "ge")
      return lhs >= rhs;
    return {};
  }
  if (a.index() >= 3 || b.index() >= 3) {
    if (std::holds_alternative<DescriptorFragment>(a) || std::holds_alternative<DescriptorFragment>(b)) {
      if (a==b && In(relation,{"eq","ne"})) return relation=="eq";
      return {};
    }
    if (std::holds_alternative<PointerBits>(a) || std::holds_alternative<PointerBits>(b)) {
      if (a==b && In(relation,{"eq","ne"})) return relation=="eq";
      return {};
    }
    if (a.index() == b.index()) {
      if (auto l = std::get_if<AllocationPointer>(&a)) {
        const auto &r = std::get<AllocationPointer>(b);
        if (AllocationIdentity(*l,c) == AllocationIdentity(r,c))
          return CompareInteger(relation,l->offset,r.offset);
        if (relation == "eq" || relation == "ne") return relation == "ne";
        return {};
      }
      if (relation == "eq")
        return a == b;
      if (relation == "ne")
        return a != b;
      if (auto l = std::get_if<ParameterPointer>(&a)) {
        const auto &r = std::get<ParameterPointer>(b);
        if (l->region == r.region)
          return CompareInteger(relation, l->offset, r.offset);
      }
      if (auto l = std::get_if<AllocationPointer>(&a)) {
        const auto &r = std::get<AllocationPointer>(b);
        if (l->resource == r.resource)
          return CompareInteger(relation, l->offset, r.offset);
      }
    } else if ((std::holds_alternative<ParameterPointer>(a) ||
                std::holds_alternative<AllocationPointer>(a)) &&
               y && *y == 0 && In(relation, {"eq", "ne"}))
      return relation == "ne";
    else if ((std::holds_alternative<ParameterPointer>(b) ||
              std::holds_alternative<AllocationPointer>(b)) &&
             x && *x == 0 && In(relation, {"eq", "ne"}))
      return relation == "ne";
    return {};
  }
  if (!x || !y)
    return {};
  if (auto width = Width(i)) {
    Integer modulus = Integer(1) << *width, mask = modulus - 1;
    *x &= mask;
    *y &= mask;
    if (Has(i, "s" + std::to_string(*width))) {
      if ((*x & (modulus >> 1)) != 0)
        *x -= modulus;
      if ((*y & (modulus >> 1)) != 0)
        *y -= modulus;
    }
  }
  return CompareInteger(relation, *x, *y);
}
struct Operand {
  std::string text;
  std::uint32_t reg = kMissing;
  Scalar constant;
  std::vector<std::uint32_t> registers;
  std::optional<std::pair<std::string, Integer>> address;
};
struct Decoded {
  const Instruction *instruction = nullptr;
  std::vector<Operand> operands;
  std::vector<std::uint32_t> uses, destinations;
  std::vector<std::uint32_t> destination_slots;
  std::uint32_t predicate = kMissing;
  std::optional<std::pair<std::string, std::uint32_t>> wait_loop;
  bool replay_wait_result = false;
};
// A thread's .local memory as its own exact stores left it: (allocation, byte offset) ->
// one stored element. Slots of one allocation never overlap.
using LocalAddress = std::pair<std::string, std::int64_t>;
template <class V> struct LocalSlot {
  unsigned bytes{};
  V value{};
};
template <class V> using LocalSlots = std::map<LocalAddress, LocalSlot<V>>;
// The most distinct slots one template-mode load selects between.
constexpr std::size_t kLocalSelectTargets = 64;
// Forgets every slot that overlaps `bytes` bytes at `at`.
template <class V>
void ForgetLocal(LocalSlots<V> &slots, const LocalAddress &at, std::uint64_t bytes) {
  auto it = slots.lower_bound(at);
  if (it != slots.begin())
    if (const auto previous = std::prev(it);
        previous->first.first == at.first && previous->first.second + std::int64_t(previous->second.bytes) > at.second)
      it = previous;
  while (it != slots.end() && it->first.first == at.first && it->first.second < at.second + std::int64_t(bytes))
    it = slots.erase(it);
}
// The target of an ld/st through a .local or generic address: `exact` when the address is a
// whole access inside a .local allocation; `may_alias` when the access may reach .local memory.
struct LocalTarget {
  std::optional<LocalAddress> exact;
  bool may_alias = false;
};
bool LocalSpace(const Instruction &i) {
  return i.semantics.memory_space && In(*i.semantics.memory_space, {"local", "generic"});
}
// Whether executing an instruction may write .local memory: a write or atomic in the .local or
// generic space, or an unclassified memory or asynchronous form. Calls are handled by the executors.
bool MayWriteLocal(const Instruction &i) {
  const auto &s = i.semantics;
  if (!s.memory_space) return s.operation == "unsupported" && (s.memory_effect || s.asynchronous);
  return In(*s.memory_space, {"local", "generic"}) && s.memory_mode && *s.memory_mode != "read";
}
const Operand *AddressOperand(const Decoded &d) {
  for (const auto &o : d.operands)
    if (o.address) return &o;
  return nullptr;
}
LocalTarget LocalTargetOf(const Instruction &i, const Scalar &pointer, const Integer &displacement,
                          std::uint64_t bytes) {
  if (const auto *p = std::get_if<AllocationPointer>(&pointer); p && p->resource.starts_with("local:")) {
    const Integer offset = p->offset + displacement;
    if (offset >= 0 && p->extent_bytes && offset + bytes <= *p->extent_bytes)
      return {LocalAddress{p->resource, offset.convert_to<std::int64_t>()}, true};
    return {{}, true};
  }
  if (*i.semantics.memory_space == "local") return {{}, true};
  const auto *bounded = std::get_if<BoundedAllocationPointer>(&pointer);
  const bool elsewhere = std::holds_alternative<AllocationPointer>(pointer) ||
      std::holds_alternative<ParameterPointer>(pointer) || std::holds_alternative<ClusterSharedPointer>(pointer) ||
      (bounded && !(*bounded)->resource.starts_with("local:"));
  return {{}, !elsewhere};
}
// The elements of a vector operand ({a, b}), or the operand itself.
std::vector<std::string_view> VectorElements(std::string_view text) {
  text = TrimView(text);
  if (text.starts_with('{') && text.ends_with('}')) text = text.substr(1, text.size() - 2);
  std::vector<std::string_view> parts;
  for (;;) {
    const auto comma = text.find(',');
    parts.push_back(TrimView(text.substr(0, comma)));
    if (comma == text.npos) break;
    text.remove_prefix(comma + 1);
  }
  return parts;
}
// What a load of `bytes` reads back from a slot of the same width holding `stored`: an integer
// truncates to the slot and extends as the load's type does; a pointer survives only a whole
// 8-byte round trip, and a bounded value only when its range fits the loaded type.
Scalar LocalLoaded(const Scalar &stored, unsigned bytes, bool sign) {
  if (const auto n = ScalarInteger(stored)) return Reduce(*n, bytes * 8, sign);
  if (bytes == 8 && (std::holds_alternative<AllocationPointer>(stored) ||
                     std::holds_alternative<ParameterPointer>(stored) ||
                     std::holds_alternative<BoundedAllocationPointer>(stored)))
    return stored;
  if (const auto *b = std::get_if<BoundedInteger>(&stored); b && FitsType(*b, bytes * 8, sign)) return stored;
  return {};
}
struct State {
  std::vector<Scalar> registers;
  std::vector<std::optional<DynamicKey>> origins;
  std::vector<std::uint64_t> occurrences, branch_visits;
  std::map<std::string, std::uint64_t> backedges;
  LocalSlots<Scalar> local;
};
struct Deferred {
  std::uint32_t lane, instruction;
  std::uint64_t occurrence;
  bool shared = false;
};
class LaneTask {
public:
  struct promise_type {
    std::optional<LaneTrace> result;
    std::optional<Deferred> dependency;
    std::exception_ptr error;
    LaneTask get_return_object() {
      return LaneTask(std::coroutine_handle<promise_type>::from_promise(*this));
    }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_always yield_value(Deferred value) noexcept {
      dependency = value;
      return {};
    }
    void return_value(LaneTrace trace) { result = std::move(trace); }
    void unhandled_exception() noexcept { error = std::current_exception(); }
  };
  explicit LaneTask(std::coroutine_handle<promise_type> handle) : handle_(handle) {}
  LaneTask(LaneTask &&other) noexcept : handle_(std::exchange(other.handle_, {})) {}
  LaneTask(const LaneTask &) = delete;
  ~LaneTask() { if (handle_) handle_.destroy(); }
  bool Done() const { return handle_.done(); }
  const auto &Dependency() const { return handle_.promise().dependency; }
  void Resume() {
    handle_.promise().dependency.reset();
    handle_.resume();
    if (handle_.promise().error) std::rethrow_exception(handle_.promise().error);
  }
  LaneTrace TakeResult() { return std::move(*handle_.promise().result); }
private:
  std::coroutine_handle<promise_type> handle_;
};
struct ExecutorResources;
struct WarpContext {
  std::map<std::tuple<std::uint32_t, std::uint64_t, std::uint32_t>, Scalar>
      shuffle;
  std::map<std::tuple<std::uint32_t, std::uint64_t, std::uint32_t>, std::uint32_t>
      vote_masks;
  SharedExecution *shared = nullptr;
  ExecutorResources *resources = nullptr;
  // activemask (only in functions that contain it): the conditional control decisions (pc, next pc)
  // of the warp's common path, recorded by the first lane to make each; a lane that makes another
  // decision, merges a region or finishes has left it. `queries` are the activemask instances.
  struct LaneControl {
    std::uint64_t decisions = 0;
    bool diverged = false;
  };
  std::vector<std::pair<std::uint32_t, std::uint32_t>> decisions;
  std::map<std::uint32_t, LaneControl> control;
  std::set<std::pair<std::uint32_t, std::uint64_t>> queries;
  void Diverge(std::uint32_t lane) {
    auto &own = control[lane];
    if (own.diverged) return;
    own.diverged = true;
    for (const auto &[instruction, occurrence] : queries)
      shuffle.try_emplace(std::tuple{instruction, occurrence, lane}, Scalar{});
  }
  void Decide(std::uint32_t lane, std::uint32_t from, std::uint32_t to) {
    auto &own = control[lane];
    if (own.diverged) return;
    if (own.decisions == decisions.size()) decisions.emplace_back(from, to);
    else if (decisions[own.decisions] != std::pair{from, to}) return Diverge(lane);
    ++own.decisions;
  }
};
// A producer Resources with the ExecutorFacts and UnitFrame it reads and names
// through, in one allocation (facts and frame first: they outlive the
// Resources). Its holders keep one pointer, and the template task's
// coroutine frame does not grow.
struct ExecutorResources {
  ExecutorFacts facts;
  UnitFrame frame;
  Resources resources;
  ExecutorResources(const ExecutionProgram &in_progress, Coordinates coordinates, AddressTable &addresses,
                    BarrierTable &barriers)
      : facts(in_progress), frame(coordinates, in_progress.config.cluster_dim),
        resources(facts, frame, addresses, barriers, SlotConfig{Consumer::Executor, {}}) {}
};
class TemplateTask {
public:
  struct promise_type {
    std::optional<ExecutionProgram> result;
    std::exception_ptr error;
    TemplateTask get_return_object() {return TemplateTask(std::coroutine_handle<promise_type>::from_promise(*this));}
    std::suspend_always initial_suspend() noexcept {return {};}
    std::suspend_always final_suspend() noexcept {return {};}
    std::suspend_always yield_value(bool) noexcept {return {};}
    void return_value(std::optional<ExecutionProgram> value) {result=std::move(value);}
    void unhandled_exception() noexcept {error=std::current_exception();}
  };
  explicit TemplateTask(std::coroutine_handle<promise_type> handle):handle_(handle){}
  TemplateTask(TemplateTask&& other) noexcept:handle_(std::exchange(other.handle_,{})){}
  TemplateTask(const TemplateTask&)=delete;
  ~TemplateTask(){if(handle_)handle_.destroy();}
  bool Done()const{return handle_.done();}
  void Resume(){handle_.resume();if(handle_.promise().error)std::rethrow_exception(handle_.promise().error);}
  std::optional<ExecutionProgram> TakeResult(){return std::move(handle_.promise().result);}
private:
  std::coroutine_handle<promise_type> handle_;
};
struct ProbeValue {
  DynamicKey key;
  Scalar alternative;
};
struct WaitIdentity {
  std::string kind;
  Scalar address, target;
  bool operator==(const WaitIdentity &) const = default;
};
struct WaitProbe {
  std::size_t trace_index;
  WaitIdentity identity;
};
// Whether a vote.sync mask is an integer naming every lane of the warp.
bool FullWarpMask(const Scalar &value) {
  const auto plain = PlainValue(value);
  const auto *mask = std::get_if<Integer>(&plain);
  return mask && (*mask & Integer(0xffffffffu)) == Integer(0xffffffffu);
}
class Executor : public std::enable_shared_from_this<Executor> {
public:
  Executor(const Module &m, const Function &f, const FrontendConfig &c,
           std::function<void()> poll, SpecializationPurpose purpose)
      : m_(m), f_(f), c_(c), poll_(std::move(poll)), purpose_(purpose) {
    if (purpose_ == SpecializationPurpose::BranchScopeCapture ||
        purpose_ == SpecializationPurpose::VerificationWithBranchScope) {
      const auto choices = c_.raw.value("user_branch_decisions", Json::array());
      if (!choices.is_array() || choices.size() > 4096)
        throw std::length_error("branch-scope user-choice limit");
      for (std::size_t index = 0; index < choices.size(); ++index) {
        const auto& choice = choices[index];
        const auto line = choice.at("line").get<std::uint32_t>();
        const auto taken = choice.contains("resolved_taken")
            ? choice.at("resolved_taken").get<bool>() : choice.at("taken").get<bool>();
        if (!capture_decisions_.emplace(line, std::pair{index, taken}).second)
          throw std::invalid_argument("ambiguous branch-scope user choice");
      }
    }
    Prepare();
  }
  ExecutionProgram Run();
  std::optional<ExecutionProgram> RunTemplates();
  TemplateTask RunTemplateTask(SharedExecution*);
  static std::optional<std::vector<ExecutionProgram>> RunTemplateCluster(const Module &,const Function &,
      const std::vector<FrontendConfig>&,const std::function<void()>&,SpecializationPurpose);
  static std::vector<ExecutionProgram> RunCluster(const Module &, const Function &,
      const std::vector<FrontendConfig> &, const std::function<void()> &, SpecializationPurpose);

private:
  const Module &m_;
  const Function &f_;
  const FrontendConfig c_;
  std::function<void()> poll_;
  SpecializationPurpose purpose_;
  std::map<std::uint32_t, std::pair<std::size_t, bool>> capture_decisions_;
  std::optional<BranchScopeDecision> CaptureDecision(std::uint32_t line,
      bool taken, std::uint64_t visit) const {
    const auto found = capture_decisions_.find(line);
    if (found == capture_decisions_.end() || found->second.second != taken) return {};
    return BranchScopeDecision{taken, visit, found->second.first};
  }
  std::set<std::uint32_t> label_entries_;
  // Two-arm merge, shared by the template and per-lane executors: a forward one-armed diamond
  // whose straight-line fall-through arm holds only listed compute opcodes, private reads and
  // tcgen05 ld/st/wait (no synchronization) and whose written registers are referenced nowhere
  // outside the arm, so the arm's registers are dead after the join (consumers of the lane
  // traces never see an arm value after the join). `uniform` is the static proof of a
  // warp-uniform predicate: it is computed in the branch's basic block from a full-warp
  // vote.sync and constants only, so every lane of the warp reaches the branch with the same
  // value. Lowering checks that no tcgen05 ordering is pending at the branch.
  struct MergedArm {
    std::vector<std::uint32_t> writes;
    bool aligned = false, uniform = false;
  };
  mutable std::map<std::uint32_t, std::optional<MergedArm>> merged_arms_;
  mutable std::vector<std::pair<std::uint32_t, std::uint32_t>> reference_ranges_;
  const MergedArm *MergedArmAt(std::uint32_t pc) const;
  // Whether `reg`, named by `at` inside a lexical scope nested in `outer` (an inline-asm
  // block), is a temporary declared there, which `outer` cannot name: an opaque region record
  // of `outer` omits it. `shadows`: `outer` names another register of that name.
  mutable std::unique_ptr<RegisterTable> scoped_registers_;
  bool ScopeLocal(const Instruction &at, std::uint32_t reg, const std::vector<std::uint32_t> &outer,
                  bool *shadows = nullptr) const {
    if (at.scope.size() <= outer.size()) return false;
    if (!scoped_registers_) scoped_registers_ = std::make_unique<RegisterTable>(m_, f_);
    const auto visible = scoped_registers_->Resolve(outer, names_[reg]);
    if (scoped_registers_->Resolve(at.scope, names_[reg]) == visible) return false;
    if (shadows) *shadows = scoped_registers_->DeclaredWidth(visible).has_value();
    return true;
  }
  // The join of the opaque compute region of the forward branch at `pc`: its instructions up
  // to the target are OpaqueRegionInstructions. With `arms` (the exact last-resort pass),
  // forward branches inside extend it to their furthest target (if-else, nested ifs), and a
  // nested lexical scope must close inside it and declare only ScopeLocal temporaries.
  std::optional<std::uint32_t> OpaqueRegionJoin(std::uint32_t pc, bool arms) const;
  // Whether a data predicate's forward branch at `pc` is taken as its opaque region (with arms):
  // the region's written registers, and every register computed from them by pure computation
  // anywhere in the function (flow-insensitive), reach only pure computation, the stored value of a
  // store, or the predicate of a forward branch that is itself admitted (recursively). So no
  // address, no guard and no other control decision reads a value the region leaves unknown. O(uses)
  // per admitted region, memoized.
  bool DataOpaqueAdmitted(std::uint32_t pc) const;
  mutable std::map<std::uint32_t, bool> data_opaque_;
  mutable std::vector<std::vector<std::uint32_t>> register_uses_; // reg -> instruction indexes that read it
  // The registers the region [pc+1, join) reads and writes, by name, without ScopeLocal ones.
  void OpaqueRegionNames(std::uint32_t pc, std::uint32_t join, std::vector<std::string> &reads,
                         std::vector<std::string> &writes) const;
  // A st.local an opaque compute region may hold: its LocalRegionStore shape, once the byte
  // offset is proved an aligned access inside the array for every value it can take; else
  // null, with the reason in `refusal`. Cached per store. `region_store_refusals_`: the
  // regions (by branch pc) that only such a store kept from being skipped.
  const LocalRegionStore *ProvenLocalStore(std::uint32_t pc, std::string &refusal) const;
  std::string LocalStoreRefusal(std::uint32_t pc, const LocalRegionStore &store) const;
  // The region's st.local stores, each proved (ProvenLocalStore): every slot of their arrays
  // may change while the region is skipped. O(region instructions).
  template <class V> void ForgetRegionLocal(std::uint32_t pc, std::uint32_t join, LocalSlots<V> &slots) const {
    for (auto index = pc + 1; index < join; ++index) {
      if (!LocalStoreInstruction(*decoded_[index].instruction)) continue;
      std::string refusal;
      if (const auto *store = ProvenLocalStore(index, refusal))
        ForgetLocal(slots, {store->declaration.resource, 0}, store->declaration.bytes);
    }
  }
  mutable std::unique_ptr<LocalRegionStores> local_stores_;
  mutable std::map<std::uint32_t, std::string> local_store_proofs_, region_store_refusals_;
  // Static definitions of each register (decoded_ destinations), built on first use.
  mutable std::vector<std::vector<std::uint32_t>> register_definitions_;
  // Whether `definition` precedes `use` on every path from the entry (block dominance, then
  // order inside a block); false for irreducible control.
  bool StrictlyDominates(std::uint32_t definition, std::uint32_t use) const;
  bool StaticWarpUniform(std::uint32_t use, std::uint32_t reg, unsigned depth) const;
  mutable ThreadRecordPool record_pool_;
  std::vector<std::string> names_;
  std::unordered_map<std::string, std::uint32_t> ids_;
  std::vector<Decoded> decoded_;
  std::map<std::string, std::uint32_t> labels_;
  std::map<std::tuple<std::string, std::uint64_t, std::uint32_t>,
           const PointerParameter *>
      pointers_;
  std::map<std::string, ParameterRegion> constant_inputs_;
  // Class mode (fixed_input.table_atoms): a whole aligned element of a constant input reads as a
  // table atom (BoundedValue::exact). Per resource: its element width in bytes, its allocation object.
  bool table_atoms_ = false;
  struct ConstantElements {
    unsigned bytes = 0;      // element width
    std::string object;      // allocation object
    std::uint64_t row = 0;   // elements per row of the table (its records)
  };
  std::map<std::string, ConstantElements> constant_elements_;
  // Per (resource, signedness, column): the least and greatest element of that column of a record
  // table's rows (every element of any other table: TableElement), read at the element width.
  mutable std::map<std::tuple<std::string, bool, std::uint64_t>, std::pair<Integer, Integer>> constant_ranges_;
  // The table atom of the element of constant input `resource` at `offset` read as `value`, or the
  // plain value when the read is not one whole aligned element. O(table) once per (resource, sign).
  Scalar TableElement(const std::string &resource, const ParameterRegion &region, std::uint64_t offset,
                      unsigned width, bool sign, const Integer &value) const {
    const auto element = constant_elements_.find(resource);
    if (!table_atoms_ || element == constant_elements_.end() || element->second.bytes != width || offset % width)
      return value;
    // A record table lists records of a few typed fields (a scheduler's work items, per-group problem
    // shapes): its rows are short, at most kRecordFields elements, or at least no longer than the
    // table has rows. An element's range is then its column's, and the units of a class, which read
    // different records at the same fields, get equal bounds. Any other table (one row, or a few long
    // rows of like elements, such as a CSR's row pointers or payloads per head) is a list: a unit's
    // elements lie at any column, so only the range over the whole table is the same for every unit,
    // and the table keeps its bounds when the same bytes are written with another row length.
    // Both ranges are true bounds of every element they are given to: the choice changes no result,
    // only which forms class admission finds equal. kRecordFields is a tuning constant, not a
    // semantic one. (Record tables in the examples have 2 to 8 fields per row.)
    constexpr std::uint64_t kRecordFields = 16;
    const std::uint64_t fields = element->second.row;
    const std::uint64_t rows = fields ? region.size_bytes / width / fields : 0;
    const std::uint64_t row = fields > 1 && rows > 1 && (fields <= kRecordFields || fields <= rows) ? fields : 0;
    const std::uint64_t column = row ? offset / width % row : 0;
    auto [range, added] = constant_ranges_.try_emplace({resource, sign, column});
    if (added) {
      // The table's elements are its whole-element fields (Prepare), read as `sign`ed. O(elements).
      bool first = true;
      const Integer modulus = Integer(1) << (8 * width);
      for (const auto &[at, field] : region.offset_values) {
        if (field.first != width || at % width || (row && at / width % row != column)) continue;
        Integer read = field.second & (modulus - 1);
        if (sign && read >= (modulus >> 1)) read -= modulus;
        if (first || read < range->second.first) range->second.first = read;
        if (first || read > range->second.second) range->second.second = read;
        first = false;
      }
      if (first) range->second = {value, value};
    }
    const auto type = std::string(sign ? "s" : "u") + std::to_string(width * 8);
    ElementAtom atom{element->second.object, type, Integer(offset)};
    BoundedValue pinned{range->second.first, range->second.second, {}, {},
                        AffineForm{0, {{std::move(atom), Integer(1)}}}};
    pinned.identity = "atom" + FormText(*pinned.affine);
    pinned.exact = value;
    if (value < pinned.lower || value > pinned.upper) return value;
    return BoundedInteger{std::make_shared<const BoundedValue>(std::move(pinned))};
  }
  // Whether the function declares .local memory: only then can a .local slot be filled.
  bool local_memory_ = false;
  // Whether the function contains activemask: only then do lanes record their control decisions.
  bool convergence_ = false;
  // Pointer tables (input pointer_arrays): table resource -> slot byte offset -> entry
  // (an AllocationPointer, or Integer 0 for a null entry).
  std::map<std::string, std::map<std::uint64_t, Scalar>> pointer_tables_;
  // tensor_properties: object resource -> element bound; loads of whole elements are
  // BoundedInteger. property_decisions_ records the instructions those bounds decided, for the
  // current run (Run starts a new one).
  struct TensorProperty {
    std::string object; unsigned bits; bool sign; Integer lower, upper;
    // The element type: s or u, then the width in bits.
    std::string Type() const { return std::string(sign ? "s" : "u") + std::to_string(bits); }
  };
  std::map<std::string, TensorProperty> tensor_properties_;
  std::shared_ptr<TensorPropertyDecisions> property_decisions_ = std::make_shared<TensorPropertyDecisions>();
  // An element whose bounds coincide is that value: appends it to `values` and records
  // `instruction` as decided by the bound.
  bool ExactElement(const TensorProperty &property, std::uint32_t instruction, std::vector<Scalar> &values) const {
    if (property.lower != property.upper) return false;
    values.push_back(property.lower);
    property_decisions_->Insert(property.object, instruction);
    return true;
  }
  // Whether every value within the bounds is unchanged by a load of its low-order `width`
  // bytes, read as a `width`-byte integer (signed when `sign`).
  static bool NarrowElementFits(const TensorProperty &property, std::uint32_t width, bool sign) {
    const Integer range = Integer(1) << (8 * width - (sign ? 1 : 0));
    return property.lower >= (sign ? Integer(-range) : Integer(0)) && property.upper < range;
  }
  void NoteBoundedDecision(std::uint32_t instruction, std::initializer_list<const Scalar *> operands) const {
    for (const auto &object : BoundedSources(operands)) property_decisions_->Insert(object, instruction);
  }
  // branch_decisions: a user choice decides only the visits whose predicate is not statically
  // known; the PTX lines where a known predicate disagreed with the choice and was used instead.
  mutable std::set<std::uint32_t> static_choice_lines_;
  std::uint32_t Intern(std::string name) {
    auto [it, inserted] = ids_.emplace(name, names_.size());
    if (inserted)
      names_.push_back(std::move(name));
    return it->second;
  }
  void Prepare();
  struct ShuffleSelection { std::uint32_t lane; bool valid; };
  std::optional<ShuffleSelection> SelectShuffle(const Decoded&, const State&,
                                               std::uint32_t warp, std::uint32_t lane) const;
  std::optional<std::pair<std::string, std::uint32_t>>
  WaitLoop(const Instruction &) const;
  enum class ControlLoop { None, Computation, Polling, Unknown };
  struct ControlBlock {
    std::uint32_t begin{}, end{};
    std::vector<std::uint32_t> successors, predecessors, loops;
  };
  struct NaturalLoop {
    std::uint32_t header{};
    std::set<std::uint32_t> blocks, latches;
    ControlLoop kind = ControlLoop::Unknown;
  };
  struct ControlChoice {
    ControlLoop kind = ControlLoop::Unknown;
    std::uint32_t loop = kMissing;
    bool taken_stays = false, post_test = false;
  };
  mutable bool control_ready_ = false, control_irreducible_ = false;
  mutable std::vector<ControlBlock> control_blocks_;
  mutable std::vector<std::uint32_t> control_block_for_pc_, control_dom_, control_postdom_;
  mutable std::vector<NaturalLoop> control_loops_;
  mutable std::map<std::uint32_t, ControlChoice> control_choices_;
  void PrepareControl() const;
  ControlChoice ClassifyControl(std::uint32_t pc) const;

  std::optional<std::pair<std::uint32_t, Integer>>
  Assertion(const Instruction &, std::uint32_t) const;
  State Initial(const WarpContract &, std::uint32_t) const;
  Scalar Value(const Operand &o, const State &s) const {
    return o.reg == kMissing ? o.constant
           : Unknown(s.registers[o.reg]) && !s.origins[o.reg] &&
                   !Unknown(o.constant)
               ? o.constant
               : s.registers[o.reg];
  }
  Scalar ValueText(std::string_view text, const State &s) const;
  void Set(const Decoded &, State &, std::vector<Scalar>) const;
  void Invalidate(const Decoded &d, State &s) const {
    for (auto id : d.destinations)
      s.registers[id] = Scalar{};
  }
  void Origins(const Decoded &, State &, std::uint64_t) const;
  ExecutedInstruction Record(const Decoded &, const State &, std::uint64_t,
                             std::optional<bool>) const;
  LocalTarget Local(const Decoded &d, const State &s) const {
    const auto *o = AddressOperand(d);
    if (!o || !d.instruction->memory_width) return {{}, true};
    return LocalTargetOf(*d.instruction, ValueText(o->address->first, s), o->address->second,
                         *d.instruction->memory_width);
  }
  void Data(const Decoded &, State &, std::uint32_t, std::uint32_t,
            std::uint64_t, const Json &, WarpContext &,
            bool probe = false) const;
  LaneTask Lane(const WarpContract &, std::uint32_t, WarpContext &) const;
  bool SharedData(const Decoded &, State &, std::uint32_t, std::uint32_t,
      std::uint64_t, const Json &, WarpContext &, const LaneTrace &, bool wait = false) const;
  [[noreturn]] void UnresolvedControl(const Instruction &, const State &,
                                     const LaneTrace &, std::uint32_t) const;
  // Shared-memory routing: whether the plain run (no shared-memory model) must stop at the
  // undecided branch at `pc` instead of skipping it (opaque region or merged arm), because its
  // predicate may be computed from a shared-memory load (SharedFedPredicate); SharedRouted throws
  // the routing error (ExecutionError::shared_routed).
  bool SharedRouted(std::uint32_t pc, bool runtime) const {
    return !runtime && !shared_routing_disabled && SharedFedPredicate(pc);
  }
  [[noreturn]] void RouteToSharedModel(const Instruction &) const;
  // A run with the shared-memory model reached the branch it was routed for and cannot decide it
  // either: stop it (SharedRoutingAttempt), so the caller runs the CTA unrouted.
  static void RoutedBranchUndecided(const Instruction &i, bool runtime) {
    if (runtime && routed_branch_line && *routed_branch_line == i.line)
      throw ExecutionError("shared_routing_undecided", i.line,
                           "the shared-memory model does not decide the routed branch either");
  }
  // Whether the guard predicate of the instruction at `pc` may be computed by integer computation
  // from the result of an ld.shared (state spaces shared, shared::cta, shared::cluster):
  // flow-insensitive over the static register definitions (every definition of a register; the
  // registers an instruction reads, including its address registers and guard predicate, reach all
  // its destinations), not through floating-point instructions (a floating-point type or ftz
  // modifier, including a floating-point ld.shared). This is the static form of UnresolvedControl's
  // dynamic chain, restricted to the values the shared-memory model forwards. O(register uses) once
  // per function.
  bool SharedFedPredicate(std::uint32_t pc) const;
  mutable std::vector<char> shared_fed_; // per register; empty until first use
  const std::vector<std::vector<std::uint32_t>> &RegisterUses() const;
  std::optional<WaitIdentity> Identity(const Decoded &, const State &) const;
  std::optional<std::pair<std::uint32_t, std::set<std::uint32_t>>>
  Fallback(const Decoded &, const State &, std::uint32_t, const WaitIdentity &,
           std::uint32_t, std::uint32_t, WarpContext &) const;
  [[noreturn]] void Error(const Instruction &i, std::uint32_t warp,
                          std::uint32_t lane, std::string message, std::string code = "ptx_execution_not_supported") const {
    throw ExecutionError(std::move(code), i.line, "warp " + std::to_string(warp) + " lane " +
                             std::to_string(lane) + ", PTX line " +
                             std::to_string(i.line) + " (" + i.opcode +
                             "): " + message);
  }
};
bool IsWait(const Instruction &i) {
  return i.semantics.category == "mbarrier" &&
         In(i.semantics.operation, {"try_wait_parity", "try_wait_state",
                                    "test_wait_parity", "test_wait_state"});
}
// A write of shared memory whose stored bytes the shared-value model (SharedConstants) does not
// track: an atomic or reduction on shared memory, stmatrix (one row of `bytes` per address lane),
// the TMEM address tcgen05.alloc writes, a wmma store (a strided tile: every byte), and a generic
// store, atomic or reduction (which may address shared memory). Shared-control execution
// (SharedData) forgets the bytes it may write.
struct UntrackedSharedWrite {
  unsigned operand{};    // the address operand
  std::uint64_t bytes{}; // bytes written at the address (0: unknown)
  bool generic = false;  // a generic address: only one that is a shared address is forgotten
};
std::optional<UntrackedSharedWrite> UntrackedShared(const Instruction &i) {
  const auto &s = i.semantics;
  const std::uint64_t width = i.memory_width.value_or(0);
  if (i.opcode == "stmatrix") return UntrackedSharedWrite{0, width};
  if (i.opcode == "tcgen05" && s.operation == "alloc") return UntrackedSharedWrite{0, 4};
  if (i.opcode == "wmma" && s.operation == "wmma_write" && s.memory_space &&
      In(*s.memory_space, {"shared", "shared.cta", "shared.cluster", "generic"}))
    return UntrackedSharedWrite{0, 0, *s.memory_space == "generic"};
  if (!In(i.opcode, {"st", "atom", "red"}) || !s.memory_space) return std::nullopt;
  const unsigned operand = i.opcode == "atom" ? 1 : 0;
  // st.shared and st.async store tracked bytes (SharedData); atom and red on shared memory do not.
  if (i.opcode != "st" && In(*s.memory_space, {"shared", "shared.cta", "shared.cluster"}))
    return UntrackedSharedWrite{operand, width};
  if (*s.memory_space == "generic") return UntrackedSharedWrite{operand, width, true};
  return std::nullopt;
}
} // namespace
} // namespace sentryward::frontend::ptx

#include "execution_prepare.hpp"
#include "execution_control.hpp"
#include "execution_registers.hpp"
#include "execution_data.hpp"
#include "execution_shared.hpp"
#include "execution_lane.hpp"
#include "execution_regions.hpp"
#include "execution_templates.hpp"
#include "execution_scheduler.hpp"
