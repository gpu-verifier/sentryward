#pragma once
#include "core.hpp"

namespace rg {
// A caller premise quantified over the integer domains of the listed Arena
// variables. This is not a license to choose a branch or interpret note text.
struct ScopedPredicateAssumption {
  Id predicate = invalid;
  std::vector<Id> bound_indices;
  std::string reason;
};

class PreparedScopedPredicates {
 public:
  struct Limits {
    size_t max_match_work = 4096;
    size_t max_search_states = 256;
    size_t max_template_nodes = 4096;
    size_t max_template_edges = 16384;
    size_t max_preparation_work = 32768;
    size_t max_assumptions = 64;
  };
  Limits limits;
  mutable uint64_t queries = 0, cache_hits = 0, match_work = 0,
                   exhausted_queries = 0;
  uint64_t preparation_work = 0;
  mutable std::unordered_set<Id> applied_predicates;

  void add(Arena &a, const ScopedPredicateAssumption &premise) {
    if (owner_ && owner_ != &a)
      throw Unsupported("scoped predicate belongs to another Arena");
    if (records_.size() >= limits.max_assumptions ||
        premise.predicate >= a.nodes.size() || premise.reason.empty() ||
        premise.bound_indices.empty() || premise.bound_indices.size() > 8)
      throw Unsupported("scoped predicate requires a bounded scope and provenance");
    static const std::map<std::string, std::string> inverse{
        {"eq", "ne"}, {"ne", "eq"}, {"lt", "ge"},
        {"ge", "lt"}, {"le", "gt"}, {"gt", "le"}};
    const auto &predicate = a.nodes[premise.predicate];
    if (predicate.op != "compare" || predicate.args.size() != 2 ||
        !predicate.data.empty() || !inverse.contains(predicate.tag))
      throw Unsupported("scoped predicate requires a Boolean comparison");
    Record record;
    record.premise = premise;
    for (Id variable : premise.bound_indices) {
      if (variable >= a.nodes.size() ||
          !record.positions.emplace(variable, record.domains.size()).second)
        throw Unsupported("scoped predicate has an invalid or duplicate binder");
      const auto &node = a.nodes[variable];
      if (node.op != "var" || !node.bounds || node.data.size() != 2 ||
          node.bounds->first != node.data[0] || node.bounds->second != node.data[1] ||
          node.data[0].denominator() != 1 || node.data[1].denominator() != 1 ||
          node.data[0] > node.data[1])
        throw Unsupported("scoped predicate binder must have a closed integer domain");
      record.domains.push_back(*node.bounds);
    }
    std::vector<Id> todo{premise.predicate};
    std::unordered_set<Id> seen, used;
    size_t edges = 0, preparation = 0;
    auto charge = [&](size_t count) {
      if (count > limits.max_preparation_work - preparation)
        throw Unsupported("scoped predicate preparation work cap");
      preparation += count;
    };
    while (!todo.empty()) {
      charge(1);
      const Id id = todo.back(); todo.pop_back();
      if (!seen.insert(id).second) continue;
      if (seen.size() > limits.max_template_nodes)
        throw Unsupported("scoped predicate template size cap");
      const auto &node = a.nodes[id];
      if (node.args.size() > limits.max_template_edges - edges)
        throw Unsupported("scoped predicate template edge cap");
      edges += node.args.size();
      // Charge before a.terms copies an operand list and before queue growth.
      charge(node.args.size());
      charge(node.args.size());
      charge(node.data.size());
      charge(node.op.size());
      charge(node.tag.size());
      if (node.op == "fold")
        throw Unsupported("scoped predicate cannot contain a Fold");
      if (node.op == "var") {
        if (!record.positions.contains(id))
          throw Unsupported("scoped predicate has an unscoped binder");
        used.insert(id);
      }
      // Precompute the only algebraic capture supported: c*i+d. Other
      // operators retain their entire exact head, coefficients and children.
      if (node.op == "var" || node.op == "add" || node.op == "mul") {
        Id variable = invalid;
        Q coefficient(0), offset(0);
        bool valid = true;
        for (const auto &[term, scale] : a.terms(id)) {
          if (term == a.one) offset += scale;
          else if (record.positions.contains(term) &&
                   (variable == invalid || variable == term)) {
            variable = term; coefficient += scale;
          } else { valid = false; break; }
        }
        if (valid && variable != invalid && coefficient != 0 &&
            coefficient.denominator() == 1 && offset.denominator() == 1)
          record.affine.emplace(id, Affine{record.positions.at(variable), coefficient, offset});
      }
      todo.insert(todo.end(), node.args.begin(), node.args.end());
      a.check();
    }
    if (used.size() != record.positions.size())
      throw Unsupported("scoped predicate declares an unused binder");
    const auto &node = a.nodes[premise.predicate];
    const Id opposite = a.compare(inverse.at(node.tag), node.args[0], node.args[1]);
    for (const auto &prior : records_)
      if (prior.premise.predicate == opposite && prior.positions == record.positions)
        throw Unsupported("inconsistent scoped predicate assumptions");
    owner_ = &a;
    record.rendered_predicate = render(a, premise.predicate);
    preparation_work += preparation;
    records_.push_back(std::move(record));
    cache_.clear();
  }

  // Only template nodes and bounded matching alternatives are visited. No
  // index domain or input array is enumerated. Failed queries are cached too.
  Id apply(Arena &a, Id candidate) const {
    if (records_.empty()) return candidate;
    if (owner_ != &a || candidate >= a.nodes.size())
      throw Unsupported("scoped predicate comparison Arena mismatch");
    if (a.nodes[candidate].op != "compare") return candidate;
    if (const auto found = cache_.find(candidate); found != cache_.end()) {
      ++cache_hits; return found->second;
    }
    ++queries;
    Work work{a, limits};
    Id result = candidate;
    for (auto &record : records_) {
      std::vector<Id> captures(record.domains.size(), invalid);
      if (match(record, record.premise.predicate, candidate, captures, work, 0)) {
        ++record.matches;
        applied_predicates.insert(candidate);
        result = a.one;
        break;
      }
      if (work.exhausted) break;
    }
    match_work += work.used;
    exhausted_queries += work.exhausted;
    cache_.emplace(candidate, result);
    return result;
  }

  size_t configured_count() const { return records_.size(); }
  size_t applied_count() const {
    return std::count_if(records_.begin(), records_.end(),
                         [](const Record &r) { return r.matches != 0; });
  }
  J report() const {
    J result{{"configured_count", configured_count()}, {"applied_count", applied_count()},
             {"applied_predicates", applied_predicates.size()}, {"queries", queries},
             {"cache_hits", cache_hits}, {"match_work", match_work},
             {"preparation_work", preparation_work},
             {"exhausted_queries", exhausted_queries},
             {"configured", J::array()}, {"applied", J::array()}};
    for (size_t i = 0; i < records_.size(); ++i) {
      const auto &record = records_[i];
      J scope = J::array();
      for (size_t j = 0; j < record.domains.size(); ++j)
        scope.push_back({{"index", record.premise.bound_indices[j]},
                         {"lower", str(record.domains[j].first)},
                         {"upper", str(record.domains[j].second)}});
      J item{{"assumption", i}, {"predicate", record.premise.predicate},
             {"predicate_expression", record.rendered_predicate},
             {"integer_scope", std::move(scope)}, {"reason", record.premise.reason},
             {"matched_predicates", record.matches}};
      result["configured"].push_back(item);
      if (record.matches) result["applied"].push_back(std::move(item));
    }
    return result;
  }

 private:
  struct Affine { size_t slot; Q coefficient, offset; };
  struct Record {
    ScopedPredicateAssumption premise;
    std::unordered_map<Id, size_t> positions;
    std::vector<std::pair<Q, Q>> domains;
    std::unordered_map<Id, Affine> affine;
    std::string rendered_predicate;
    mutable size_t matches = 0;
  };
  struct Work {
    Arena &a;
    const Limits &limits;
    size_t used = 0, states = 0;
    bool exhausted = false;
    std::unordered_map<Id, bool> integer_cache{};
    bool charge(size_t count = 1) {
      a.check();
      if (count > limits.max_match_work - used) { exhausted = true; return false; }
      used += count; return true;
    }
    bool branch() {
      if (!charge() || states == limits.max_search_states) { exhausted = true; return false; }
      ++states; return true;
    }
    bool integer(Id id, unsigned depth = 0) {
      if (!charge() || depth >= 128) { exhausted = true; return false; }
      if (const auto found = integer_cache.find(id); found != integer_cache.end()) return found->second;
      const auto &node = a.nodes[id];
      bool valid = false;
      if (node.op == "const") valid = node.data[0].denominator() == 1;
      else if (node.op == "var")
        valid = node.bounds && node.bounds->first.denominator() == 1 &&
                node.bounds->second.denominator() == 1 && node.bounds->first <= node.bounds->second;
      else if (node.op == "add" || node.op == "mul") {
        valid = charge(node.data.size()) &&
                std::all_of(node.data.begin(), node.data.end(), [](const Q &x) { return x.denominator() == 1; });
        for (Id child : node.args) {
          if (!valid || !integer(child, depth + 1)) { valid = false; break; }
        }
      }
      integer_cache.emplace(id, valid); return valid;
    }
  };
  Arena *owner_ = nullptr;
  std::vector<Record> records_;
  mutable std::unordered_map<Id, Id> cache_;

  // A receipt is reviewable without recursively expanding a shared DAG into
  // an unbounded string. Exact IDs remain available when this view truncates.
  static std::string render(const Arena &a, Id root) {
    constexpr size_t max_characters = 8192;
    size_t remaining_nodes = 256;
    std::string result;
    auto append = [&](const std::string &text) {
      result.append(text, 0, std::min(text.size(), max_characters - result.size()));
    };
    std::function<void(Id, unsigned)> visit = [&](Id id, unsigned depth) {
      if (result.size() == max_characters) return;
      const auto &node = a.nodes[id];
      if (!depth || !remaining_nodes) { append("#" + std::to_string(id)); return; }
      --remaining_nodes;
      if (node.op == "const") { append(str(node.data[0])); return; }
      append(node.op); append("["); append(node.tag);
      for (const auto &value : node.data) {
        if (result.size() == max_characters) break;
        append(","); append(str(value));
      }
      append("](");
      for (size_t i = 0; i < node.args.size(); ++i) {
        if (result.size() == max_characters) break;
        if (!remaining_nodes) { append("..."); break; }
        if (i) append(",");
        visit(node.args[i], depth - 1);
      }
      append(")");
    };
    visit(root, 8);
    return result;
  }

  static bool capture(const Record &record, const Affine &affine, Id candidate,
                      std::vector<Id> &captures, Work &work) {
    const auto &node = work.a.nodes[candidate];
    if (!node.bounds || !node.facts.finite || !work.integer(candidate)) return false;
    const Q first = (node.bounds->first - affine.offset) / affine.coefficient;
    const Q last = (node.bounds->second - affine.offset) / affine.coefficient;
    const auto &domain = record.domains[affine.slot];
    if (std::min(first, last) < domain.first || std::max(first, last) > domain.second) return false;
    if (!work.charge(node.args.size() + 4)) return false;
    const Id value = affine.coefficient == 1 && affine.offset == 0 ? candidate :
        work.a.mul({work.a.constant(Q(1) / affine.coefficient),
                    work.a.sub(candidate, work.a.constant(affine.offset))});
    if (!work.integer(value)) return false;
    auto &prior = captures[affine.slot];
    if (prior != invalid) return prior == value;
    prior = value; return true;
  }

  // compare() chooses an affine sign using Arena atom IDs. That choice is
  // intentionally not invariant under scoped binder substitution. Inspect
  // one opposite orientation without rebuilding any numerical ancestors.
  // These are the same stored Add/Mul shells, not a second normalizer.
  static Id negate_comparison_shell(Id id, Work &work) {
    const auto &node = work.a.nodes[id];
    if (!node.facts.finite) return invalid;
    constexpr size_t max_terms = 32;
    if ((node.op == "add" || node.op == "mul") &&
        node.args.size() > max_terms) return invalid;
    const size_t copied = (node.op == "add" || node.op == "mul")
                              ? node.args.size() + node.data.size() : 1;
    if (!work.charge(copied + 1)) return invalid;
    if (const auto value = work.a.value(id)) return work.a.constant(-*value);
    Bounds bounds;
    if (node.bounds) bounds = std::pair(Q(-node.bounds->second), Q(-node.bounds->first));
    Facts facts{true, bounds && bounds->first > 0, bounds && bounds->first >= 0};
    if (node.op == "add") {
      if (node.data.size() != node.args.size()) return invalid;
      auto coefficients = node.data;
      for (auto &coefficient : coefficients) coefficient = -coefficient;
      return work.a.intern("add", node.args, std::move(coefficients), "", facts, bounds);
    }
    if (node.op == "mul") {
      if (node.data.size() != 1) return invalid;
      const Q coefficient = -node.data[0];
      if (coefficient == 1 && node.args.size() == 1) return node.args[0];
      return work.a.intern("mul", node.args, {coefficient}, "", facts, bounds);
    }
    return work.a.intern("mul", {id}, {Q(-1)}, "", facts, bounds);
  }

  static bool match_comparison(const Record &record, const Node &pattern,
                               const Node &candidate, std::vector<Id> &captures,
                               Work &work, unsigned depth) {
    if (pattern.args.size() != 2 || candidate.args.size() != 2 ||
        !pattern.data.empty() || !candidate.data.empty()) return false;
    if (!work.charge(captures.size() + 4)) return false;
    const auto prior = captures;
    if (pattern.tag == candidate.tag &&
        match(record, pattern.args[0], candidate.args[0], captures, work, depth + 1) &&
        match(record, pattern.args[1], candidate.args[1], captures, work, depth + 1))
      return true;
    captures = prior;
    if (work.exhausted) return false;
    std::string reversed = candidate.tag;
    if (reversed == "lt") reversed = "gt";
    else if (reversed == "le") reversed = "ge";
    else if (reversed == "gt") reversed = "lt";
    else if (reversed == "ge") reversed = "le";
    else if (reversed != "eq" && reversed != "ne") return false;
    if (pattern.tag != reversed || !work.branch()) return false;
    const Id left = negate_comparison_shell(candidate.args[0], work);
    const Id right = negate_comparison_shell(candidate.args[1], work);
    if (left != invalid && right != invalid && !work.exhausted &&
        match(record, pattern.args[0], left, captures, work, depth + 1) &&
        match(record, pattern.args[1], right, captures, work, depth + 1))
      return true;
    captures = prior;
    return false;
  }

  static bool match(const Record &record, Id pattern, Id candidate,
                    std::vector<Id> &captures, Work &work, unsigned depth) {
    if (!work.charge() || depth >= 128) { work.exhausted = true; return false; }
    if (const auto affine = record.affine.find(pattern); affine != record.affine.end())
      return capture(record, affine->second, candidate, captures, work);
    const auto &p = work.a.nodes[pattern], &c = work.a.nodes[candidate];
    if (p.op == "compare" && c.op == "compare")
      return match_comparison(record, p, c, captures, work, depth);
    if (p.op != c.op || p.tag != c.tag || p.args.size() != c.args.size()) return false;
    if (!work.charge(p.args.size() + p.data.size())) return false;
    if (p.op == "add" || p.op == "mul") {
      if ((p.op == "mul" && p.data != c.data) ||
          p.data.size() != c.data.size()) return false;
      std::vector<bool> used(c.args.size(), false);
      return match_ac(record, p, c, 0, used, captures, work, depth + 1);
    }
    if (p.data != c.data) return false;
    for (size_t i = 0; i < p.args.size(); ++i)
      if (!match(record, p.args[i], c.args[i], captures, work, depth + 1)) return false;
    return true;
  }
  static bool match_ac(const Record &record, const Node &pattern, const Node &candidate,
                       size_t position, std::vector<bool> &used,
                       std::vector<Id> &captures, Work &work, unsigned depth) {
    if (!work.charge() || depth >= 128) { work.exhausted = true; return false; }
    if (position == pattern.args.size()) return true;
    for (size_t i = 0; i < candidate.args.size(); ++i) {
      if (!work.charge()) return false;
      if (used[i] || (pattern.op == "add" && pattern.data[position] != candidate.data[i])) continue;
      if (!work.branch()) return false;
      auto prior = captures;
      used[i] = true;
      if (match(record, pattern.args[position], candidate.args[i], captures, work, depth + 1) &&
          match_ac(record, pattern, candidate, position + 1, used, captures, work, depth + 1)) return true;
      used[i] = false;
      captures = std::move(prior);
      if (work.exhausted) return false;
    }
    return false;
  }
};
} // namespace rg
