#pragma once
#include "checker.hpp"

namespace rg {

// A proof of a positive exponential scale on complete reference reductions.
// Quotients require both mass and moment; logarithmic outputs require mass
// and exact cancellation with the output pivot. The finite pivot need not be
// the maximum (threshold-delayed rescaling is permitted).
struct OnlineSoftmaxProof {
  Id replacement = invalid;
  Id numerator_spec = invalid, denominator_spec = invalid;
  Id common_scale = invalid;
  std::vector<Id> numerator_parameters, denominator_parameters;
  uint64_t visited_nodes = 0, contribution_pieces = 0;
  bool logarithmic = false, base2_pivot = false;
  Id pivot = invalid;
};

struct OnlineSoftmaxBudget {
  uint64_t nodes = 100000, pieces = 30000;
  size_t maximum_specs = 8, maximum_candidates = 8;
};

class OnlineSoftmaxPattern {
  Checker &checker;
  Arena &a;
  OnlineSoftmaxBudget budget;
  uint64_t visits = 0, made_pieces = 0;
  bool exhausted = false;
  // Only maxima certified in this attempt may justify a weight bound.
  std::unordered_set<Id> certified_maxima;

  bool tick() {
    a.check();
    if (++visits > budget.nodes) exhausted = true;
    return !exhausted;
  }
  bool piece_budget() {
    if (++made_pieces > budget.pieces) exhausted = true;
    return !exhausted;
  }
  Id leaf(Id index) {
    return piece_budget() ? checker.leaf(index) : invalid;
  }
  Id concat(Id left, Id right) {
    if (left == invalid || right == invalid || !piece_budget()) return invalid;
    return checker.concat(left, right);
  }
  Id bind(Id piece, Id variable, const Z &count) {
    if (piece == invalid || !piece_budget()) return invalid;
    return checker.bind_piece(piece, variable, count);
  }
  Id whole_domain(Id sid) {
    const auto &spec = checker.specs[sid];
    Id variable = a.var("online-softmax-domain:" + std::to_string(sid),
                        0, Q(spec.count - 1));
    return bind(leaf(variable), variable, spec.count);
  }
  std::optional<std::vector<Box>> bounded_boxes(Id piece) {
    struct Work { Id piece; bool leave; };
    std::vector<Work> todo{{piece, false}};
    std::vector<std::pair<Id, Z>> domains;
    std::vector<Box> result;
    while (!todo.empty()) {
      if (!tick()) return {};
      auto work = todo.back(); todo.pop_back();
      if (work.piece == invalid || work.piece >= checker.pieces.size()) return {};
      const auto &node = checker.pieces[work.piece];
      if (work.leave) {
        domains.pop_back();
      } else if (node.kind == PieceNode::Concat) {
        todo.push_back({node.right, false});
        todo.push_back({node.left, false});
      } else if (node.kind == PieceNode::Bind) {
        domains.emplace_back(node.var, node.count);
        todo.push_back({work.piece, true});
        todo.push_back({node.left, false});
      } else {
        auto box = checker.box(node.index, domains);
        if (!box) return {};
        result.push_back(std::move(*box));
      }
    }
    return result;
  }

  struct Call {
    Id sid;
    std::vector<Id> parameters;
  };
  std::optional<Call> semantic_call(Id root) const {
    const auto &node = a.nodes[root];
    if (node.op != "call" || node.data.size() != 1 ||
        node.data[0].denominator() != 1 || node.data[0] < 0 ||
        node.data[0] >= Q(checker.specs.size())) return {};
    Id sid = node.data[0].numerator().convert_to<Id>();
    if (node.args.size() + 1 != checker.specs[sid].slots) return {};
    return Call{sid, node.args};
  }
  struct Quotient { Id numerator, denominator; };
  std::optional<Quotient> quotient(Id root) {
    const auto &node = a.nodes[root];
    if (node.op != "mul" || node.data.size() != 1) return {};
    Id denominator = invalid;
    std::vector<Id> numerator{a.constant(node.data[0])};
    for (Id child : node.args) {
      const auto &part = a.nodes[child];
      if (part.op == "recip") {
        if (denominator != invalid || part.args.size() != 1) return {};
        denominator = part.args[0];
      } else numerator.push_back(child);
    }
    if (denominator == invalid || numerator.size() == 1) return {};
    return Quotient{a.mul(std::move(numerator)), denominator};
  }

  // A maximum is summarized separately from the sum invariant.  The key omits
  // the reduced coordinate, but keeps every row/input parameter.  It therefore
  // never identifies different rows or different score functions.
  struct MaximumKey {
    Id sid;
    std::vector<Id> parameters;
    bool operator<(const MaximumKey &other) const {
      return std::tie(sid, parameters) < std::tie(other.sid, other.parameters);
    }
  };
  using MaximumCandidates = std::map<MaximumKey, Id>;
  struct MaximumEntry {
    Id rewritten = invalid;
    bool neutral = false, reducible = false;
    MaximumCandidates candidates;
  };

  MaximumCandidates join_maximum(const MaximumCandidates &left,
                                 const MaximumCandidates &right) {
    MaximumCandidates result;
    for (const auto &[key, piece] : left) {
      auto found = right.find(key);
      if (found != right.end()) result.emplace(key, concat(piece, found->second));
    }
    return result;
  }

  // Only a proved contiguous domain is assigned an opaque maximum node.  A
  // duplicate, gap, unknown coordinate, or nonrectangular domain declines this
  // optional rewrite.  No key instances are enumerated.
  std::optional<Id> canonical_maximum(const MaximumCandidates &candidates) {
    if (candidates.size() != 1 || exhausted) return {};
    const auto &[key, piece] = *candidates.begin();
    if (piece == invalid) return {};
    auto boxes = bounded_boxes(piece);
    if (!boxes || boxes->empty()) return {};
    std::optional<Z> first, last;
    for (const Box &box : *boxes) {
      if (!tick() || box.mult <= 0) return {};
      Z end = box.start;
      for (const auto &[stride, count] : box.axes) {
        if (stride <= 0 || count <= 0) return {};
        end += stride * (count - 1);
      }
      first = first ? std::min(*first, box.start) : box.start;
      last = last ? std::max(*last, end) : end;
    }
    Z count = *last - *first + 1;
    for (Box &box : *boxes) box.start -= *first;
    if (!cover_boxes(*boxes, count, a)) return {};
    ++a.stats["online_softmax_maximum_domains"];
    // This symbol is local proof vocabulary, not an unchecked input axiom.
    // Every construction above has proved max(score(key)) over exactly this
    // interval, and the score template is identified by sid.
    Id result = a.intern("online_softmax_maximum", key.parameters,
                         {Q(key.sid), Q(*first), Q(count)}, "", {true, false, false});
    certified_maxima.insert(result);
    return result;
  }

  // A range certificate for one stored weight template, not an assumption
  // about all online pivots. For a proved maximum M over [first, first+count),
  // matching score(k) and checking k's entire integer range proves score(k)<=M.
  // Thus b+c*score(k)-c*M <= b for c>0. Delayed Select pivots do not satisfy
  // this certificate and are deliberately left unchanged.
  Id bounded_weight_clamp(Id root) {
    const auto &clamp = a.nodes[root];
    if (clamp.op != "clamp" || clamp.args.size() != 1 ||
        clamp.data.size() != 2 || clamp.data[0] > 0) return root;
    Id weight = clamp.args[0];
    const auto &exponential = a.nodes[weight];
    if (exponential.op != "exp2" || exponential.args.size() != 1 ||
        !exponential.facts.finite || !exponential.facts.pos) return root;
    const auto &exponent = a.nodes[exponential.args[0]];
    if (exponent.op != "add" || exponent.args.size() != exponent.data.size() ||
        exponent.args.size() > 3) return root;
    Q offset = 0, score_scale = 0, maximum_scale = 0;
    Id score = invalid, maximum = invalid;
    for (size_t i = 0; i < exponent.args.size(); ++i) {
      if (!tick()) return root;
      Id term = exponent.args[i];
      Q coefficient = exponent.data[i];
      if (term == a.one) offset += coefficient;
      else if (coefficient > 0 && a.nodes[term].op == "call" && score == invalid) {
        score = term;
        score_scale = coefficient;
      } else if (coefficient < 0 && certified_maxima.contains(term) && maximum == invalid) {
        maximum = term;
        maximum_scale = coefficient;
      } else return root;
    }
    if (score == invalid || maximum == invalid || score_scale != -maximum_scale ||
        offset > 8 || !a.nodes[score].facts.finite) return root;
    // Round only the *proved upper bound* upward, with a bounded exact integer
    // power. Negative offsets safely use the weaker upper bound 2^0.
    Z upper_exponent = -floorq(-offset);
    if (upper_exponent < 0) upper_exponent = 0;
    Q upper = Q(pow2(upper_exponent.convert_to<unsigned>()));
    if (clamp.data[1] < upper) return root;
    const auto &domain = a.nodes[maximum];
    Id sid = domain.data[0].numerator().convert_to<Id>();
    Q first = domain.data[1], last = first + domain.data[2] - 1;
    for (const auto &match : checker.match_term(sid, score)) {
      if (!tick()) return root;
      if (match.coef != a.one || match.params != domain.args) continue;
      const auto &index = a.nodes[match.index];
      if (!index.bounds || index.bounds->first < first || index.bounds->second > last)
        continue;
      // Bounds alone would also admit fractional keys. Match the existing
      // integer-coordinate vocabulary: integral affine combinations of bounded
      // index variables. This visits stored terms, never key instances.
      bool integral = true;
      for (const auto &[term, coefficient] : a.terms(match.index)) {
        if (!tick()) return root;
        const auto &part = a.nodes[term];
        if (coefficient.denominator() != 1 ||
            (term != a.one && (part.op != "var" || !part.bounds ||
              part.bounds->first.denominator() != 1 ||
              part.bounds->second.denominator() != 1))) {
          integral = false;
          break;
        }
      }
      if (!integral) continue;
      ++a.stats["online_softmax_weight_range_proofs"];
      ++a.stats["online_softmax_weight_clamps_removed"];
      return weight;
    }
    return root;
  }

  std::optional<Id> canonicalize_pivots(Id root) {
    std::vector<Id> maximum_specs;
    for (Id sid : checker.active) {
      const auto &spec = checker.specs[sid];
      if (spec.maximum && spec.facts.finite && spec.count > 0 &&
          checker.patterns.nodes[spec.pattern].op == "call")
        maximum_specs.push_back(sid);
    }
    if (maximum_specs.size() > budget.maximum_specs) return {};
    std::unordered_map<Id, MaximumEntry> memo;
    std::unordered_map<Id, Id> finished;
    auto finish = [&](Id old) {
      auto known = finished.find(old);
      if (known != finished.end()) return known->second;
      auto &entry = memo.at(old);
      Id result = entry.rewritten;
      if (entry.reducible)
        if (auto canonical = canonical_maximum(entry.candidates)) result = *canonical;
      finished.emplace(old, result);
      return result;
    };
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    while (!todo.empty()) {
      if (!tick()) return {};
      auto [old, ready] = todo.back(); todo.pop_back();
      if (memo.contains(old)) continue;
      const auto &raw = a.nodes[old];
      if (!ready) {
        todo.emplace_back(old, true);
        for (Id child : raw.args)
          if (!memo.contains(child)) todo.emplace_back(child, false);
        continue;
      }
      bool is_maximum = raw.op == "max";
      bool is_fold = raw.op == "fold" && raw.tag == "Max";
      std::vector<Id> args;
      for (Id child : raw.args)
        args.push_back(is_maximum || is_fold ? memo.at(child).rewritten : finish(child));
      MaximumEntry entry;
      entry.rewritten = checker.reapply(old, args);
      if (raw.op == "clamp") entry.rewritten = bounded_weight_clamp(entry.rewritten);
      entry.neutral = a.neginf(entry.rewritten);
      entry.reducible = is_maximum || is_fold;
      if (is_maximum) {
        bool first = true, invalid_part = false;
        for (Id child : raw.args) {
          const auto &part = memo.at(child);
          if (part.neutral) continue;
          if (part.candidates.empty()) { invalid_part = true; break; }
          if (first) { entry.candidates = part.candidates; first = false; }
          else entry.candidates = join_maximum(entry.candidates, part.candidates);
        }
        if (invalid_part) entry.candidates.clear();
      } else if (is_fold && raw.args.size() == 2 && raw.data.size() == 2) {
        Id variable = raw.data[0].numerator().convert_to<Id>();
        Z count = raw.data[1].numerator();
        const auto &initial = memo.at(raw.args[0]);
        if (count == 0) entry = initial;
        else if (count > 0) {
          MaximumCandidates bound;
          for (const auto &[key, piece] : memo.at(raw.args[1]).candidates)
            if (std::none_of(key.parameters.begin(), key.parameters.end(),
                             [&](Id parameter) { return a.depends(parameter, variable); }))
              bound.emplace(key, bind(piece, variable, count));
          entry.candidates = initial.neutral ? std::move(bound)
              : join_maximum(initial.candidates, bound);
        }
      } else if (auto call = semantic_call(entry.rewritten);
                 call && checker.specs[call->sid].maximum &&
                 checker.specs[call->sid].facts.finite &&
                 checker.specs[call->sid].count > 0 &&
                 std::find(maximum_specs.begin(), maximum_specs.end(), call->sid) != maximum_specs.end()) {
        entry.reducible = true;
        entry.candidates.emplace(MaximumKey{call->sid, call->parameters}, whole_domain(call->sid));
      } else if (a.nodes[entry.rewritten].op == "call" &&
                 a.nodes[entry.rewritten].facts.finite) {
        for (Id sid : maximum_specs) {
          for (const auto &match : checker.match_term(sid, entry.rewritten)) {
            if (match.coef != a.one) continue;
            MaximumKey key{sid, match.params};
            // Alternative captures are not multiple contributions.  Ambiguity
            // for the same key is declined instead of concatenating it.
            auto found = entry.candidates.find(key);
            if (found != entry.candidates.end()) {
              entry.candidates.clear();
              break;
            }
            entry.candidates.emplace(std::move(key), leaf(match.index));
          }
          if (entry.candidates.size() > budget.maximum_candidates) return {};
        }
      }
      memo.emplace(old, std::move(entry));
    }
    if (exhausted) return {};
    return finish(root);
  }

  struct SumEntry {
    bool zero = false;
    Id coefficient = invalid;
    // Rational weights do not change the symbolic common scale. Keep them
    // attached to persistent contribution domains until exact coverage closes.
    std::map<Q, Id> pieces;
  };
  using MaybeSum = std::optional<SumEntry>;
  MaybeSum scaled(const MaybeSum &part, Id factor) {
    if (!part || !a.nodes[factor].facts.finite) return {};
    if (part->zero || factor == a.zero) return SumEntry{true, a.zero, {}};
    Id coefficient = checker.canonical_coefficient(a.mul({part->coefficient, factor}));
    Q scalar(1);
    if (auto value = a.value(coefficient)) { scalar = *value; coefficient = a.one; }
    else if (a.nodes[coefficient].op == "mul") {
      scalar = a.nodes[coefficient].data[0];
      coefficient = a.mul(a.nodes[coefficient].args);
    }
    SumEntry result{false, coefficient, {}};
    for (const auto &[weight, piece] : part->pieces) result.pieces.emplace(weight * scalar, piece);
    return result;
  }
  MaybeSum contribution(Id coefficient, Id piece) {
    return scaled(SumEntry{false, a.one, {{Q(1), piece}}}, coefficient);
  }
  MaybeSum joined(const MaybeSum &left, const MaybeSum &right) {
    if (!left || !right) return {};
    if (left->zero) return right;
    if (right->zero) return left;
    if (!checker.equal_coefficients(left->coefficient, right->coefficient)) return {};
    SumEntry result = *left;
    for (const auto &[weight, piece] : right->pieces) {
      auto [it, added] = result.pieces.emplace(weight, piece);
      if (!added) it->second = concat(it->second, piece);
    }
    if (result.pieces.size() > budget.maximum_candidates) return {};
    return result;
  }

  // The forest stores the multiset until final closure.  Each DAG node is
  // visited once per reference component; binding and concatenation are O(1).
  MaybeSum reduction(Id root, const Call &expected) {
    std::unordered_map<Id, MaybeSum> memo;
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    while (!todo.empty()) {
      if (!tick()) return {};
      auto [old, ready] = todo.back(); todo.pop_back();
      if (memo.contains(old)) continue;
      const auto &node = a.nodes[old];
      if (!ready) {
        todo.emplace_back(old, true);
        for (Id child : node.args)
          if (!memo.contains(child)) todo.emplace_back(child, false);
        continue;
      }
      MaybeSum result;
      if (old == a.zero) result = SumEntry{true, a.zero, {}};
      else if (auto call = semantic_call(old);
               call && call->sid == expected.sid && call->parameters == expected.parameters)
        result = contribution(a.one, whole_domain(expected.sid));
      else if (node.facts.finite) {
        std::optional<std::pair<Id, Id>> unique;
        bool ambiguous = false;
        const auto &pattern = checker.patterns.nodes[checker.specs[expected.sid].pattern];
        if (pattern.op == node.op || (checker.exponential_mode &&
            (node.op == "mul" || node.op == "select" || node.op == "exp" || node.op == "exp2"))) {
          for (const auto &match : checker.match_term(expected.sid, old, expected.parameters)) {
            if (match.params != expected.parameters || !a.nodes[match.coef].facts.finite) continue;
            auto candidate = std::pair(match.index, match.coef);
            if (unique && *unique != candidate) { ambiguous = true; break; }
            unique = candidate;
          }
        }
        if (unique && !ambiguous)
          result = contribution(unique->second, leaf(unique->first));
        else if (!ambiguous && (node.op == "add" || node.op == "add_raw")) {
          result = SumEntry{true, a.zero, {}};
          for (size_t i = 0; i < node.args.size() && result; ++i)
            result = joined(result, scaled(memo.at(node.args[i]), a.constant(node.data[i])));
        } else if (!ambiguous && node.op == "mul") {
          size_t participating = 0, selected = 0;
          for (size_t i = 0; i < node.args.size(); ++i)
            if (memo.at(node.args[i])) { ++participating; selected = i; }
          if (participating == 1) {
            std::vector<Id> factors{a.constant(node.data[0])};
            for (size_t i = 0; i < node.args.size(); ++i)
              if (i != selected) factors.push_back(node.args[i]);
            result = scaled(memo.at(node.args[selected]), a.mul(std::move(factors)));
          }
        } else if (!ambiguous && node.op == "fold" && node.tag == "Sum" &&
                   node.args.size() == 2 && node.data.size() == 2) {
          Id variable = node.data[0].numerator().convert_to<Id>();
          Z count = node.data[1].numerator();
          if (count == 0) result = memo.at(node.args[0]);
          else if (count > 0) {
            MaybeSum body = memo.at(node.args[1]);
            if (body && !body->zero) {
              if (a.depends(body->coefficient, variable) ||
                  std::any_of(expected.parameters.begin(), expected.parameters.end(),
                              [&](Id parameter) { return a.depends(parameter, variable); })) body.reset();
              else for (auto &[weight, piece] : body->pieces) piece = bind(piece, variable, count);
            }
            result = joined(memo.at(node.args[0]), body);
          }
        }
      }
      memo.emplace(old, std::move(result));
    }
    return exhausted ? MaybeSum{} : memo.at(root);
  }

  Id guarded_mass(Id root) {
    const auto &node = a.nodes[root];
    if (node.op != "select" || node.args.size() != 3) return root;
    const auto arms = node.args;
    if (a.nodes[arms[0]].op != "compare") return root;
    // The tested mass is the selected value, not necessarily an operand of
    // the canonical predicate (x+1 != 0 can normalize to x != -1).
    if (arms[2] == a.one && arms[0] == a.compare("ne", arms[1], a.zero)) return arms[1];
    if (arms[1] == a.one && arms[0] == a.compare("eq", arms[2], a.zero)) return arms[2];
    return root;
  }
  bool complete(SumEntry &sum, Id sid) {
    if (sum.zero || sum.pieces.empty() || exhausted) return false;
    const Z count = checker.specs[sid].count;
    if (count <= 0) return false;
    Z denominator = 1;
    for (const auto &[weight, piece] : sum.pieces)
      denominator = denominator / boost::multiprecision::gcd(denominator, weight.denominator()) * weight.denominator();
    std::vector<Box> combined;
    Z mass = 0;
    for (const auto &[weight, piece] : sum.pieces) {
      auto boxes = bounded_boxes(piece);
      if (!boxes) return false;
      Z multiplier = weight.numerator() * (denominator / weight.denominator());
      for (auto &box : *boxes) {
        box.mult *= multiplier;
        Z size = box.mult; for (const auto &[stride, n] : box.axes) size *= n;
        mass += size; combined.push_back(std::move(box));
      }
    }
    // Exact generating-polynomial identity proves every coordinate has the
    // same total weight. Total cardinality alone would not reject gaps.
    Q uniform(mass, denominator * count);
    if (uniform == 0) return false;
    Z extra = uniform.denominator() / boost::multiprecision::gcd(denominator, uniform.denominator());
    for (auto &box : combined) box.mult *= extra;
    denominator *= extra;
    Z multiplicity = uniform.numerator() * (denominator / uniform.denominator());
    if (!cover_boxes(combined, count, a, multiplicity)) return false;
    sum.coefficient = a.mul({sum.coefficient, a.constant(uniform)});
    ++a.stats["weighted_reduction_coverage_proofs"];
    return true;
  }

  struct LogProjection { Id mass, pivot; bool base2; };
  std::optional<LogProjection> logarithmic_projection(Id root) {
    const auto &node = a.nodes[root];
    if (node.op == "ideal_lse" && node.tag == "ptx_lg2_f32" &&
        node.args.size() == 2 && node.data.empty())
      return LogProjection{node.args[0], node.args[1], true};
    // A small additive projection keeps the ordinary natural-log identity
    // available without any compiler constant interpretation.
    if (node.op != "add" || node.args.size() > 8 ||
        node.args.size() != node.data.size()) return {};
    Id mass = invalid;
    std::vector<Id> pivot_terms;
    // Copy before constructing any arena nodes.
    auto args = node.args; auto coefficients = node.data;
    for (size_t i = 0; i < args.size(); ++i) {
      if (!tick()) return {};
      const auto &term = a.nodes[args[i]];
      if (term.op == "log") {
        if (mass != invalid || coefficients[i] != 1 || term.args.size() != 1) return {};
        mass = term.args[0];
      } else pivot_terms.push_back(a.mul({a.constant(coefficients[i]), args[i]}));
    }
    if (mass == invalid) return {};
    return LogProjection{mass, a.add(std::move(pivot_terms)), false};
  }
  std::optional<OnlineSoftmaxProof> prove_logarithmic(Id reference, Id optimized) {
    const auto &left = a.nodes[reference];
    if (left.op != "log" || left.args.size() != 1) return {};
    auto denominator = semantic_call(left.args[0]);
    if (!denominator || !checker.active.contains(denominator->sid)) return {};
    const auto &ds = checker.specs[denominator->sid];
    // A masked exponential is only nonnegative pointwise. Strict positivity
    // belongs to this actual complete reduction, proved by its scoped witness,
    // not to every term or every instantiation of the reference Spec.
    const auto &mass_facts = a.nodes[left.args[0]].facts;
    if (ds.maximum || ds.count <= 0 || !ds.facts.finite || !ds.facts.nn ||
        !mass_facts.finite || !mass_facts.pos) return {};
    auto right = logarithmic_projection(optimized);
    if (!right) return {};
    ++a.stats["online_softmax_pattern_attempts"];
    ++a.stats["online_softmax_lse_attempts"];
    auto decline = [&](const char *reason) -> std::optional<OnlineSoftmaxProof> {
      last_failure = exhausted ? "budget_exhausted" : reason;
      ++a.stats["online_softmax_pattern_declined"];
      ++a.stats["online_softmax_decline_" + last_failure];
      return {};
    };
    auto rewritten = canonicalize_pivots(optimized);
    if (!rewritten) return decline("pivot_domains");
    right = logarithmic_projection(*rewritten);
    if (!right) return decline("lse_projection");
    if (!a.nodes[right->pivot].facts.finite) return decline("lse_pivot_not_finite");
    auto mass = reduction(guarded_mass(right->mass), *denominator);
    if (!mass || mass->zero) return decline("mass_contributions");
    if (!complete(*mass, denominator->sid)) return decline("mass_coverage");
    if (!a.nodes[mass->coefficient].facts.finite || !a.nodes[mass->coefficient].facts.pos)
      return decline("scale_not_positive_finite");
    // Complete nonempty coverage of a strictly positive reference mass and a
    // finite positive coefficient proves mass > 0, including the supported
    // mass==0 guard. The exp2 form uses the scoped ideal_lse tail's exact
    // semantic ln(mass)+ln(2)*pivot, not a rational approximation to ln(2).
    Id inverse_scale = right->base2 ? a.exp2(right->pivot) : a.exp(right->pivot);
    if (checker.exponential_mode ? !checker.scalar.equal(a.mul({mass->coefficient,inverse_scale}),a.one) : a.mul({mass->coefficient, inverse_scale}) != a.one)
      return decline("lse_scale_not_cancelled");
    ++a.stats["online_softmax_pattern_proofs"];
    ++a.stats["online_softmax_lse_proofs"];
    OnlineSoftmaxProof proof{reference, invalid, denominator->sid,
                             mass->coefficient, {}, denominator->parameters,
                             visits, made_pieces};
    proof.logarithmic = true; proof.base2_pivot = right->base2; proof.pivot = right->pivot;
    return proof;
  }

public:
  std::string last_failure;
  explicit OnlineSoftmaxPattern(Checker &owner, OnlineSoftmaxBudget limits = {})
      : checker(owner), a(owner.a), budget(limits) {}

  // Call after normalize(reference, true) and normalize(optimized), while the
  // corresponding reference specs remain active.  This never normalizes a new
  // reference, changes active specs, or installs assumptions.  Failed attempts
  // leave both caller roots unchanged.
  std::optional<OnlineSoftmaxProof> prove(Id reference, Id optimized) {
    visits = made_pieces = 0; exhausted = false; last_failure.clear();
    certified_maxima.clear();
    if (a.nodes[reference].op == "log") return prove_logarithmic(reference, optimized);
    auto left = quotient(reference), right = quotient(optimized);
    if (!left || !right) return {};
    auto numerator = semantic_call(left->numerator), denominator = semantic_call(left->denominator);
    if (!numerator || !denominator || !checker.active.contains(numerator->sid) ||
        !checker.active.contains(denominator->sid)) return {};
    const auto &ns = checker.specs[numerator->sid], &ds = checker.specs[denominator->sid];
    const auto &mass_facts = a.nodes[left->denominator].facts;
    if (ns.maximum || ds.maximum || ns.count <= 0 || ns.count != ds.count ||
        !ns.facts.finite || !ds.facts.finite || !ds.facts.nn ||
        !mass_facts.finite || !mass_facts.pos) return {};
    ++a.stats["online_softmax_pattern_attempts"];
    auto decline = [&](const char *reason) -> std::optional<OnlineSoftmaxProof> {
      last_failure = exhausted ? "budget_exhausted" : reason;
      ++a.stats["online_softmax_pattern_declined"];
      ++a.stats["online_softmax_decline_" + last_failure];
      return {};
    };
    auto rewritten = canonicalize_pivots(optimized);
    if (!rewritten) return decline("pivot_domains");
    right = quotient(*rewritten);
    if (!right) return decline("output_projection");
    auto mass = reduction(guarded_mass(right->denominator), *denominator);
    if (!mass || mass->zero) return decline("mass_contributions");
    auto moment = reduction(right->numerator, *numerator);
    if (!moment || moment->zero) return decline("moment_contributions");
    if (!complete(*mass, denominator->sid)) return decline("mass_coverage");
    if (!complete(*moment, numerator->sid)) return decline("moment_coverage");
    if (!checker.equal_coefficients(mass->coefficient,moment->coefficient)) return decline("different_scales");
    if (!a.nodes[mass->coefficient].facts.finite || !a.nodes[mass->coefficient].facts.pos)
      return decline("scale_not_positive_finite");
    // Positivity of this complete, nonempty exponential mass proves the guard
    // (if present) selects the mass.  The same finite nonzero factor cancels.
    if(checker.exponential_mode && !checker.scalar.equal(reference,a.divide(
       a.mul({moment->coefficient,checker.call(numerator->sid,numerator->parameters)}),
       a.mul({mass->coefficient,checker.call(denominator->sid,denominator->parameters)}))))return decline("ep_output_projection");
    ++a.stats["online_softmax_pattern_proofs"];
    return OnlineSoftmaxProof{reference, numerator->sid, denominator->sid,
                             mass->coefficient, numerator->parameters, denominator->parameters,
                             visits, made_pieces};
  }
};
} // namespace rg
