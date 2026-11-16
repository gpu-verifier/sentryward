#pragma once
#include "checker.hpp"
#include <functional>
#include <set>
#include "whole_expression_templates.hpp"
#include <memory>
#include <algorithm>
#include <iterator>
#include <tuple>
#include <unordered_set>

namespace rg {
struct LinearReductionBudget {
  uint64_t nodes = 100000, pieces = 30000;
  size_t groups = 32, targets = 4;
  size_t factor_choices = 128, polynomial_terms = 512, polynomial_degree = 24;
  size_t distributed_terms = 128;
  uint64_t polynomial_work = 100000;
  // Members of a reference target that may be expanded explicitly when no
  // group closes it: a short explicit sum, not a reduction to enumerate.
  size_t instantiated_members = 8;
};
struct LinearReductionProof { Id replacement = invalid; J receipt; };

// A single exact decomposition, not a collection of competing explanations:
// value = residual + sum(group.coefficient * target(group.coverage)).
// Every group also retains its actual value as a persistent witness. A partial
// match (including the output's g_j) therefore remains part of the expression.
class LinearReductionPattern {
  Checker &checker;
  Arena &a;
  LinearReductionBudget budget;
  struct Miss { const char *reason; };
  uint64_t visits = 0, piece_count = 0, scalar_work = 0, choices = 0;
  size_t distributed_terms = 0;
  // Retry mode after a distribution-budget miss: a weighted Sum whose weight
  // depends on no binder below the decomposed root is decomposed once without
  // the weight, and its form is scaled (exact distributivity, O(groups)).
  bool factor_weights = false;
  // Receipt record of the current attempt: instantiated targets, and the
  // nested closures and nested gap completions it used.
  J instantiated_receipt = J::array();
  size_t attempt_nested_closures = 0, attempt_nested_gaps = 0;
  // Restores a mode flag on every exit, including a Timeout.
  struct FlagScope {
    bool &flag; bool saved;
    FlagScope(bool &f, bool value) : flag(f), saved(f) { flag = value; }
    ~FlagScope() { flag = saved; }
  };
  // Sets Checker::closure_owner for one term match and restores it on every exit (a Miss included).
  // A closure query's pattern (owned_closure), or a nested sub-proof charged to one: closure queries
  // its term matches reach are charged to it.
  bool closure_query = false;
  struct OwnerScope {
    LinearReductionPattern *&slot; LinearReductionPattern *saved;
    OwnerScope(LinearReductionPattern *&s, LinearReductionPattern *owner) : slot(s), saved(s) { slot = owner; }
    ~OwnerScope() { slot = saved; }
  };
  size_t factored_sums = 0;
  size_t closed_groups = 0, partial_groups = 0;
  struct Target { Id sid; std::vector<Id> parameters; };
  std::vector<Target> targets;
  struct Key {
    size_t target; Id coefficient;
    bool operator<(const Key &o) const {
      return std::tie(target, coefficient) < std::tie(o.target, o.coefficient);
    }
  };
  // The witness denotes the weighted value, including the coefficient.
  // guard: a predicate such that the reference summand is zero at every index of the group
  // whenever the guard is 0 (the instantiated reference mask, when all members share it), or one.
  struct Group { Id piece, witness; Z multiplicity; Id guard = invalid; };
  struct Form { Id residual = invalid; std::map<Key, Group> groups; };
  struct Witness {
    enum Kind { Original, Add, Scale, Bind } kind;
    Id left = invalid, right = invalid, value = invalid, variable = invalid;
    Z count = 0;
  };
  std::vector<Witness> witnesses;
  std::unordered_map<Id, Id> original_witnesses;
  std::map<std::pair<Id, size_t>, bool> coverage_cache;
  std::set<size_t> completed_targets;
  bool shared_targets_skipped = false;

  void tick() { a.check(); if (++visits > budget.nodes) throw Miss{"node_budget"}; }
  LinearReductionBudget remaining_budget() const;
  void charge(const LinearReductionPattern &child);
  void scalar_tick() {
    a.check(); if (++scalar_work > budget.polynomial_work) throw Miss{"polynomial_budget"};
  }
  Id witness(Witness n) {
    tick(); witnesses.push_back(std::move(n)); return Id(witnesses.size() - 1);
  }
  Id original(Id n);
  Id sum_witness(Id x, Id y) {
    if (x == invalid) return y;
    if (y == invalid) return x;
    return witness({Witness::Add, x, y});
  }
  Id scale_witness(Id x, Id coefficient) {
    if (x == invalid || coefficient == a.zero) return invalid;
    if (coefficient == a.one) return x;
    return witness({Witness::Scale, x, invalid, coefficient});
  }
  Id bind_witness(Id x, Id variable, const Z &count) {
    if (x == invalid || count == 0) return invalid;
    return witness({Witness::Bind, x, invalid, invalid, variable, count});
  }
  void piece_tick() {
    tick(); if (++piece_count > budget.pieces) throw Miss{"piece_budget"};
  }
  Id leaf(Id index) { piece_tick(); return checker.leaf(index); }
  Id concat(Id x, Id y) { piece_tick(); return checker.concat(x, y); }
  Id bind_piece(Id p, Id v, const Z &count) {
    piece_tick(); return checker.bind_piece(p, v, count);
  }
  void insert(Form &form, const Key &key, const Group &group);
  void append(Form &to, const Form &from, Id coefficient);
  Form bound(const Form &body, Id variable, const Z &count);

  std::vector<Id> capture(const Target &target, Id axis_value) const;
  Id reference_guard(const Target &target, Id index);

public:
  // ---- Exact case analysis over Select conditions -------------------------------------------------
  // A case fixes the truth of some (rebuilt) condition nodes. resolve() rebuilds a value under a
  // case: a Select whose rebuilt condition is a constant, a fixed condition, or the complement of a
  // fixed comparison becomes the chosen arm (only that arm is rebuilt); any other Select is kept and
  // its rebuilt condition is reported as open. Folds and Calls are not entered: a condition under a
  // binder is not one value. Each rebuilt node is the Arena's own constructor applied to rebuilt
  // operands (Checker::reapply), so resolve(x) equals x on every input whose conditions have the
  // fixed truths. Work is bounded by case_rebuild_limit nodes per case.
  static constexpr size_t case_variable_limit = 6, case_rebuild_limit = 20000;
  struct CaseMiss {};
  // A case of a chain lemma (finite_cases) that no input satisfies: a value it fixes finite resolves to
  // -inf, or a fixed condition resolves to the constant of the opposite truth. All of its cases hold.
  struct EmptyCase {};
  using CaseFix = std::vector<std::pair<Id, bool>>;
  int case_truth(Id c, const CaseFix &fixed);
  Id log2_of(Id x);
  bool positive_multiple(Id x, Id y);
  bool alias_conditions_only(const CaseFix &fixed);
  Id resolve(Id x, const CaseFix &fixed, std::unordered_map<Id, Id> &memo, std::vector<Id> &open,
             size_t &work);
  // ---- Finiteness in a case (the chain lemmas of merge_by_finiteness_chain only) ---------------------
  // Set only while a chain lemma is decided (FlagScope); every other case analysis is unchanged.
  bool finite_cases = false;
  // The nodes the current case fixes finite; recomputed from the fixed truths by all_cases.
  std::unordered_set<Id> case_finite;
  void collect_case_finite(const CaseFix &fixed);
  Id finite_view(Id v);
  Id nearest_open(const std::vector<Id> &roots, const std::vector<Id> &open);
  // Whether `holds` accepts every case of the conditions met while resolving `roots`: the roots are
  // resolved under the current case; when `holds` rejects and an open condition remains, the first
  // one met (operands before their Select, so one with no open Select inside) is fixed both ways.
  // At most case_variable_limit splits, so at most 2^case_variable_limit cases. Exact: every input
  // falls into one case, and in that case each root equals its resolved value.
  // Roots listed in `conditions` are themselves split candidates (before the Selects' conditions)
  // while their truth is open.
  // In the chain lemmas (finite_cases) the values the case fixes finite are their finite views
  // (collect_case_finite, resolve), and after the open roots of `conditions` the open condition nearest
  // to a resolved root is split first (nearest_open); exact for any choice of the split condition.
  template <class Holds>
  bool all_cases(const std::vector<Id> &roots, Holds &&holds, CaseFix &fixed, size_t &cases,
                 const std::vector<size_t> &conditions = {}) {
    if (++cases > (size_t(1) << (case_variable_limit + 1))) throw CaseMiss{};
    if (finite_cases) collect_case_finite(fixed);
    std::unordered_map<Id, Id> memo; std::vector<Id> open; size_t work = 0;
    std::vector<Id> resolved;
    try {
      for (Id r : roots) resolved.push_back(resolve(r, fixed, memo, open, work));
      if (finite_cases)
        for (const auto &[c, truth] : fixed) {
          std::vector<Id> ignored;
          if (const auto v = a.value(resolve(c, fixed, memo, ignored, work)); v && (*v != 0) != truth) throw EmptyCase{};
        }
      if (holds(resolved, fixed)) return true;
    } catch (const EmptyCase &) {
      ++a.stats["linear_reduction_chain_empty_cases"];
      return true;
    }
    size_t open_roots = 0;
    for (auto it = conditions.rbegin(); it != conditions.rend(); ++it)
      if (case_truth(resolved[*it], fixed) < 0 && std::find(open.begin(), open.end(), resolved[*it]) == open.end()) {
        open.insert(open.begin(), resolved[*it]); ++open_roots;
      }
    if (open.empty() || fixed.size() >= case_variable_limit) return false;
    const Id variable = finite_cases && open_roots == 0 ? nearest_open(resolved, open) : open.front();
    for (const bool truth : {true, false}) {
      fixed.emplace_back(variable, truth);
      const bool ok = all_cases(roots, holds, fixed, cases, conditions);
      fixed.pop_back();
      if (!ok) return false;
    }
    return true;
  }
  bool raw_exponents_at_most_zero(const std::vector<Id> &exponents, const CaseFix &fixed);
  // x <= 1 on every input, by R's own Select tree: select(c, A, B) is A where c holds and B where it
  // fails, so x <= 1 everywhere when A <= 1 under the path's truths plus (c, true) and B <= 1 under
  // them plus (c, false); a non-Select leaf is decided by at_most_one under the path's truths (each
  // of them holds on every input that reaches the leaf). The conditions inside a leaf (inside an
  // exponent, say) are not split. Exact. At most vote_arm_depth levels, so at most 2^vote_arm_depth
  // leaves.
  static constexpr size_t vote_arm_depth = 6;
  bool at_most_one_by_arms(Id x, CaseFix &path, size_t &leaves);
  bool maximum_dominates(Id upper, Id lower);
  bool at_most_one(Id x, const CaseFix &fixed);

  bool finite_in_case(Id x, const CaseFix &fixed);
  bool case_equal(Id x, Id y, Id guard);
  Id case_finite_rebuild(Id v, const CaseFix &fixed, bool &ok);
  std::pair<Id, Id> without_common_factors(Id x, Id y);
  // Groups of one target whose coefficients are different nodes. Group g joins the target's largest
  // group b when, in every case of the Select conditions met in c_g, c_b and g's guard, the guard
  // resolves to 0 (every reference summand of g is 0 there) or c_g = c_b by the exact EP equality.
  // Then sum_{t in g} c_g s(t) = sum_{t in g} c_b s(t) on every input, so g's members join b with
  // b's coefficient. At most (groups - 1) attempts per target, each at most 2^case_variable_limit
  // cases of a rebuild bounded by case_rebuild_limit nodes plus one EP equality.
  static constexpr size_t merge_form_limit = size_t(1) << 16;
  size_t local_form_merges = 0;  // groups merged by equal local forms in the attempt in progress
  void merge_case_equal_groups(Form &form);
  // Support of a target whose groups all joined one group: predicates P_1..P_m such that the target's
  // Sum T is 0 wherever every P_i is 0. Each group g of the decomposition contributes its own guard
  // (before any merge), or, if it joined through a chain walk, the walk's last predicate F_last (its
  // guard gives F_0 and the steps give F_last). Exact: the groups' members partition T's domain (the
  // joined group's coverage certificate) and every reference summand of g is 0 where g's guard is 0,
  // so T != 0 makes some guard nonzero and hence some P_i hold.
  std::map<size_t, std::vector<Id>> support_premises;

  // ---- Merge through a chain of finiteness invariants -------------------------------------------
  // Where the direct merge above declines for a group g of a target with base b, g may still join b
  // through the coefficients of the target's other groups: a walk c_0 = c_g, c_1 = s(c_0), ...,
  // c_r = c_b, where s(c) is the coefficient d of another group of the target nearest to c (fewest
  // factors in the multiset symmetric difference of the two products) among those strictly nearer to
  // c_b than c (ties: nearer to c_b, then the order of the keys), and predicates F_0, ..., F_{r-1}
  // such that, each by an exact case analysis (chain_lemma):
  //   link:  c_i = c_{i+1} wherever F_i holds;
  //   entry: F_0 holds wherever g's guard is nonzero;
  //   step:  F_{i+1} holds wherever F_i holds.
  // On an input where g's guard is nonzero, entry and the steps give every F_i, and the links give
  // c_g = c_1 = ... = c_b; where the guard is 0, every reference summand of g is 0. So, as for the
  // direct merge, sum_{t in g} c_g s(t) = sum_{t in g} c_b s(t) on every input and g's members join b.
  // F_i is the first finiteness predicate [v != -inf] of the link's two values (finiteness_predicates)
  // under which the link holds. Bounds: a walk has at most (groups - 1) links (each strictly nearer to
  // c_b); per link at most chain_predicate_limit predicates, each one or two case analyses; per group
  // one entry and at most (groups - 2) steps; every case analysis within case_variable_limit and
  // case_rebuild_limit. Proved lemmas are kept in Checker::proved_chain_lemmas (they are properties of
  // the nodes alone), declines per pattern.
  static constexpr size_t chain_predicate_limit = 4, chain_predicate_depth = 8;
  std::set<std::tuple<int, Id, Id, Id>> declined_chain_lemmas;
  // kind 0: u = v wherever w != 0 (case_equal with guard w); kind 1: v holds (is nonzero) wherever u
  // does, by the case analysis of v and u in which u is split first: in every case u resolves to 0 or
  // v to a nonzero truth. Both with finite_cases set.
  // The chain mode for one chain lemma or chain-mode output check: sets finite_cases and, on every exit
  // (normal return or any exception), restores it and clears case_finite (and support_finite when asked),
  // so that no later analysis can run in a stale mode or with stale facts.
  struct ChainScope {
    LinearReductionPattern &pattern; bool saved, support;
    ChainScope(LinearReductionPattern &p, bool clear_support) : pattern(p), saved(p.finite_cases), support(clear_support) {
      pattern.finite_cases = true;
    }
    ~ChainScope() {
      pattern.finite_cases = saved; pattern.case_finite.clear();
      if (support) pattern.support_finite.clear();
    }
  };
  bool chain_lemma(int kind, Id u, Id v, Id w);
  std::vector<Id> finiteness_predicates(Id x, Id y);
  std::vector<bool> finiteness_chain_joins(const std::vector<std::pair<Id, Id>> &groups, Id base,
                                           const std::vector<bool> &walk = {}, std::vector<Id> *terminal = nullptr);
  void merge_by_finiteness_chain(Form &form, const std::vector<Key> &keys, const Key &base, std::map<Id, Id> &premise);

  // ---- Output equality in the chain mode ----------------------------------------------------------
  // (X, v): wherever the proved Sum X of an alias (X = C * T) is nonzero, v is finite: T != 0 makes some
  // premise P_i of T's support hold (support_premises), and every P_i gives [v != -inf] (chain_lemma).
  std::vector<std::pair<Id, Id>> support_finite;
  std::vector<Id> extended_values(const std::vector<Id> &roots);
  void prepare_support_finite(const std::vector<Id> &roots);
  bool chain_case_equal(Id x, Id y);

  // ---- Reduction atoms under non-linear operators ------------------------------------------------
  // A Sum X of the optimized value that is an operand of a reciprocal, logarithm or comparison may
  // itself be a complete reduction: X = C * call(T) for a reference Call T (found anywhere in the
  // reference, outside Fold bodies). This is decided by an independent decomposition of X with
  // the single target T (vote aliases, masked sums and case-split merges included): exactly one
  // complete group with coefficient C and no residual. Each such X is then replaced by C * call(T)
  // in the optimized value (an identity). Memoized per (X, T); at most atom_limit atoms and
  // budget.targets targets, each decomposition charged to this pattern's budget.
  static constexpr size_t atom_limit = 8;
  // A reference Call T and an optimized Sum X proved equal to C * T with C finite and positive:
  // comparisons of T (or of a positive multiple of T) with 0 have the truth of X's comparison.
  // Each entry: Call -> (X, C) with X = C * Call.
  std::unordered_map<Id, std::pair<Id, Id>> atom_aliases;
  // Light case mode: the second exact attempt of an output check after the ordinary one declined
  // (light_attempt). It decides comparisons of aliased Calls, their positive multiples and their proved
  // Sums by each other's truth, decides [x <= 0] and [x > 0] of a finite nonnegative x by [x == 0], keeps
  // a finite unbounded value Max as an atom, uses a proved Sum unrebuilt where only alias conditions are
  // fixed, and admits reduction atoms with a finite residual (residual_atoms). The ordinary attempt
  // runs exactly as before; the light one has the same budgets.
  bool light_cases = false, residual_atoms = false;
  // (value, residual_atoms mode, target spec, target parameters) -> atom or invalid.
  std::map<std::tuple<Id, bool, Id, std::vector<Id>>, Id> atom_cache;
  // The support (support_premises) of each residual-free atom's single target, under atom_cache's key;
  // per Call of an alias in atom_supports.
  std::map<std::tuple<Id, bool, Id, std::vector<Id>>, std::vector<Id>> atom_support_cache;
  std::unordered_map<Id, std::vector<Id>> atom_supports;
  Id reduction_atom(Id x, const Target &target);
  bool light_attempt(Id reference, Id optimized);
  bool shares_reference_call(Id reference, Id optimized);
  // with_products: also products with a Sum factor (see below). atom_products records whether the
  // last scan met one, so a caller repeats a declined proof with them only where they exist.
  bool atom_products = false;
  Id substitute_reduction_atoms(Id reference, Id optimized, bool with_products = false);

  // ---- Any-of votes read through 0/1 values -----------------------------------------------------
  // The scalar fold gives vote.sync.any as Fold(Max, initial = own, body = the k-th lane's
  // predicate) over the lane index, possibly as a chain of pieces through the initial (a single lane
  // m as BitOr(acc, p_m), imported as 1 - (1 - acc)(1 - p_m)); the issuing lane's own predicate is
  // the innermost initial. The readers below collect the comparisons [x < 1] that a vote's value
  // forces false, without reading any Fold body. Every value they accept is a 0/1 value (finite,
  // bounds within [0, 1], integer known bits: zero_one), so each of its forms below is exact:
  //   zero_forces(v): v = 0 makes every collected comparison false.
  //     - a comparison c: c = 0 is c false; [x < 1] adds x (no other comparison adds anything);
  //       [w != 0], [w > 0] and [w >= 1] of a 0/1 value w are w itself (c = 0 is w = 0);
  //     - 1 - u: v = 0 is u = 1 (one_forces(u));
  //     - a Max fold max(s, e(0), ..., e(n-1)): v >= s >= 0 for every n >= 0, so v = 0 gives s = 0
  //       (zero_forces(s)); the body e is not read.
  //   one_forces(u): u = 1 makes every collected comparison false.
  //     - a product 1 * f_1 * ... * f_k of 0/1 factors: it is 1 only when every f_j is 1
  //       (one_forces(f_j));
  //     - 1 - v: u = 1 is v = 0 (zero_forces(v));
  //     - [w == 0], [w <= 0] and [w < 1] of a 0/1 value w: u = 1 is w = 0 (zero_forces(w)).
  // At most vote_read_limit nodes per reading; beyond it the reading declines.
  static constexpr size_t vote_read_limit = 4096;
  bool zero_one(Id x) {
    const auto &n = a.nodes[x];
    return n.facts.finite && n.bounds && n.bounds->first >= 0 && n.bounds->second <= 1 &&
           a.known_bits(x).integer;
  }
  Id complement_of(Id x) const;
  Id boolean_comparison_operand(Id x, bool &negated);
  bool zero_forces(Id v, std::vector<Id> &members, size_t &work);
  bool one_forces(Id u, std::vector<Id> &members, size_t &work);
  bool zero_forces_value(Id v, std::vector<Id> &members, size_t &work);
  bool one_forces_value(Id u, std::vector<Id> &members, size_t &work);
  Id vote_rescale_arm(Id node);
  std::optional<std::pair<size_t, std::vector<Id>>> chain_vote(Id p);
  Id vote_rescale_reading(const std::vector<Id> &args, size_t quiet, const std::vector<Id> &members,
                          bool zero_one_reading);

private:
  bool target_call(Id node, const Target &target) const {
    const auto &n = a.nodes[node];
    return n.op == "call" && n.data == std::vector<Q>{Q(target.sid)} &&
           n.args == target.parameters;
  }
  bool collect_targets(Id reference, Id optimized, bool reuse_existing);
  struct Candidate {
    size_t target, specificity; Id index, coefficient;
  };
  std::optional<Candidate> match(Id value);

  // Nested reductions. A root-level target whose body is a product with one
  // nested reduction call, body_t(r) = G(r) * C(args(r)), may have a nested
  // definition C that depends on the target's own reduction binder (e.g. a
  // triangular mask s < r): pattern() slots that binder as an argument of C,
  // so instantiating body_t at a concrete r instantiates C's arguments at r.
  // A term matches body_t(r) when its factors are G(r) times the rest S, and
  // S closes as exactly one complete group of C at those arguments (masked
  // gaps included).
  bool nested_mode = false;
  // Bounds the presence bitmap of a nested-gap completion (one byte per index).
  static constexpr size_t max_nested_count = 65536;
  struct NestedBody { Id call = invalid; std::vector<Id> rest; };
  std::optional<NestedBody> nested_body(const Spec &spec) const;
  std::optional<Target> nested_target(Id call_pattern, const std::vector<Id> &cap);
  // Memoized per (value, target): a closure is an exact property of these
  // immutable nodes and specs. The memo lives as long as this pattern.
  std::map<std::tuple<Id, Id, std::vector<Id>>, bool> closure_cache;
  bool closes_as(Id value, const Target &target);
  std::optional<Candidate> nested_match(Id value);
  bool complete_with_nested_gaps(const Key &key, const Group &group);
  // The free variables of the decomposed root are output coordinates; every
  // binder crossed by bound() below it is absent from that set. A weight over
  // root-free variables only therefore keeps each scaled coefficient
  // independent of every crossed binder. Cost O(|free(weight)| log |free(root)|).
  bool outside_binders(Id weight, Id root) const {
    const auto &free = a.nodes[root].free;
    return std::all_of(a.nodes[weight].free.begin(), a.nodes[weight].free.end(),
                       [&](Id v) { return std::binary_search(free.begin(), free.end(), v); });
  }
  Form decompose(Id root);
  bool zero_on_interval(const Target &target, const Z &lo, const Z &hi);
  Id instantiated_target(const Target &target);
  Id expand_targets(Id root, const std::map<size_t, Id> &members);
  bool complete_with_zero_gaps(std::vector<Box> boxes, const Target &target);
  bool complete(const Key &key, const Group &group);
  bool complete_with_zero_edge(std::vector<Box> boxes, const Target &target, const Z &covered);

  // Bounded ring normalization only after reduction closure. Calls, reads and
  // remaining Fold nodes are indivisible atoms; their bodies are not expanded.
  using Monomial = std::vector<Id>;
  using Polynomial = std::map<Monomial, Q>;
  std::unordered_map<Id, Polynomial> scalar_cache, witness_cache;
  void add_polynomial(Polynomial &to, const Polynomial &from, Q scale = Q(1));
  Polynomial multiply(const Polynomial &x, const Polynomial &y);
  const Polynomial &polynomial(Id root) {
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    while (!todo.empty()) {
      scalar_tick(); auto [id, ready] = todo.back(); todo.pop_back();
      if (scalar_cache.contains(id)) continue;
      const auto &n = a.nodes[id];
      if (!n.facts.finite) throw Miss{"nonfinite_scalar"};
      if (!ready && (n.op == "add" || n.op == "add_raw" || n.op == "mul")) {
        todo.emplace_back(id, true);
        for (Id child : n.args) if (!scalar_cache.contains(child)) todo.emplace_back(child, false);
        continue;
      }
      Polynomial result;
      if (auto constant = a.value(id)) { if (*constant != 0) result[{}] = *constant; }
      else if (n.op == "add" || n.op == "add_raw") {
        for (size_t j = 0; j < n.args.size(); ++j) add_polynomial(result, scalar_cache.at(n.args[j]), n.data[j]);
      } else if (n.op == "mul") {
        result[{}] = n.data[0];
        for (Id child : n.args) result = multiply(result, scalar_cache.at(child));
      } else result[{id}] = Q(1);
      scalar_cache.emplace(id, std::move(result));
    }
    return scalar_cache.at(root);
  }
  // Bind witnesses only arise in partial/unrecognized residues. Keep the
  // original Sum semantics as an opaque scalar rather than distributing it.
  std::unordered_map<Id, Id> materialized;
  Id materialize(Id root);
  const Polynomial &witness_polynomial(Id root) {
    if (root == invalid) return polynomial(a.zero);
    if (auto it = witness_cache.find(root); it != witness_cache.end()) return it->second;
    // Aggregate path weights over the shared forest, then add each leaf once.
    // This avoids copying a growing polynomial at every Add-prefix node, while
    // preserving multiplicity when a witness is reachable through two edges.
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    std::unordered_set<Id> seen;
    std::vector<Id> postorder;
    while (!todo.empty()) {
      scalar_tick(); auto [id, ready] = todo.back(); todo.pop_back();
      if (seen.contains(id)) continue;
      const Witness &w = witnesses[id];
      if (!ready && (w.kind == Witness::Add || w.kind == Witness::Scale)) {
        todo.emplace_back(id, true);
        if (!seen.contains(w.left)) todo.emplace_back(w.left, false);
        if (w.right != invalid && !seen.contains(w.right)) todo.emplace_back(w.right, false);
        continue;
      }
      seen.insert(id); postorder.push_back(id);
    }
    std::unordered_map<Id, Polynomial> weights;
    weights[root][{}] = Q(1);
    Polynomial result;
    for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
      scalar_tick(); Id id = *it;
      Polynomial weight = std::move(weights.at(id)); weights.erase(id);
      const Witness &w = witnesses[id];
      if (w.kind == Witness::Add) {
        add_polynomial(weights[w.left], weight); add_polynomial(weights[w.right], weight);
      } else if (w.kind == Witness::Scale) {
        add_polynomial(weights[w.left], multiply(weight, polynomial(w.value)));
      } else {
        const Polynomial &leaf = polynomial(w.kind == Witness::Original ? w.value : materialize(id));
        add_polynomial(result, multiply(weight, leaf));
      }
    }
    return witness_cache.emplace(root, std::move(result)).first->second;
  }

public:
  std::string last_failure;
  // The last row-parametric attempt's families and work (also on a decline).
  J parametric_report = J::object();
  explicit LinearReductionPattern(Checker &owner, LinearReductionBudget limits = {})
      : checker(owner), a(owner.a), budget(limits) {}
  // Makes this pattern a closure query's (owned_closure; public for its unit test).
  void mark_closure_query() { closure_query = true; }
  static std::optional<Id> owned_closure(Checker &checker, Id value, Id sid, const std::vector<Id> &parameters);
  std::optional<Id> close_target(Id value, Id sid, const std::vector<Id> &parameters);
private:
  std::optional<LinearReductionProof> prove_attempt(Id reference, Id optimized,
                                                   bool reuse_existing);
  std::optional<LinearReductionProof> attempt(Id reference, Id optimized, bool reuse_existing);
  std::optional<LinearReductionProof> direct(Id reference, Id optimized);
  // Each arm pair is one bounded sub-proof with this pattern's own budgets:
  // one direct() proof, i.e. at most two attempt() calls of at most two
  // budgeted prove_attempt() passes each. This bounds their number per root
  // pair (a select chain over a state-pool slot table has one pair per slot),
  // so select_arms does at most max_select_arm_pairs * 4 budgeted passes.
  static constexpr size_t max_select_arm_pairs = 1024;
  std::optional<LinearReductionProof> select_arms(Id reference, Id optimized);
  bool nested_targets(Id reference);
  // Fused reductions. When every monomial of the reference holds at most one
  // target call, R = r0 + Sum_t alpha_t call_t, and the targets in that form
  // share one count N with the alpha_t free of their reduction coordinate,
  // Sum_t alpha_t call_t = Sum_{i<N} F(i) with F(i) = Sum_t alpha_t body_t(i).
  // One term of an explicit optimized sum then matches F(i) as a whole, with
  // one scalar proof, where term-wise matching would first have to distribute
  // that term's factors into every constituent body.
  struct FusedForm { std::vector<size_t> members; std::vector<Id> alphas; Z count; };
  std::optional<FusedForm> fused_form(Id reference);
  Id fused_body(const FusedForm &form, const Z &index);
  // A sum that several terms of the optimized sums share (a common factor
  // such as an inner prediction) is closed first when its own decomposition
  // completes a target group: value = residual + Sum coefficient * call is
  // the same exact identity the final certificate uses. Only add nodes that
  // occur as a product factor below the root, referenced at least twice, are
  // candidates; the scan visits stored add/mul shells once.
  size_t inner_closures = 0;
  Id closed_rewrite = invalid;
  // Closure targets also include eligible reference calls below nonlinear
  // nodes (e.g. a norm inside rsqrt): an inner closure needs the call itself,
  // not a linear position of it in the reference. Bounded scan of the stored
  // reference DAG; at most max_closure_targets targets in all.
  static constexpr size_t max_closure_targets = 8;
  void add_closure_targets(Id reference);
  static bool unary_numeric(const std::string &op) {
    return op == "recip" || op == "reciprocalsqrt" || op == "sqrt" || op == "exp" || op == "exp2" ||
           op == "log" || op == "log2";
  }
  std::optional<Z> coordinate_at(Id id, Id u, const Z &value, unsigned depth = 0);
  // A group of plain leaves (an explicit sum, no binders) whose coordinates
  // depend on one free integer variable u with at most max_case_values
  // values is complete when, for EVERY value of u, the leaves' coordinates
  // are exactly [0, count) once each. Cost O(values * leaves).
  static constexpr size_t max_case_values = 64;
  bool complete_by_cases(const Key &key, const Group &group);
  Id close_shared_sums(Id root);
  std::optional<LinearReductionProof> fused_attempt(Id reference, Id optimized);
  // ---- Row-parametric closures ---------------------------------------------
  // A reference reduction R whose body holds a nested Call C(args(r)) with
  // arguments that depend on R's own coordinate r (a norm over rows,
  // Sum_r C(r)^2, of a per-row reduction C) needs one closure of C per row r
  // on the optimized side, where every row is an explicit sum. A Family is
  // such a nested Call with r replaced by a formal integer variable A over
  // R's domain [0, count): target(A) = (sid, parameters(A)); nested Calls of
  // a family's own body whose arguments depend on A are families too.
  //
  // A family is proved once: two candidate rows propose the syntax of a row
  // L(A) (WholeExpressionTemplates; syntax only), and decomposing L(A) with
  // the single target(A) proves L(A) = C(A) := residual(A) + c(A) *
  // target(A) for every integer A in its domain (complete coverage of the
  // target's own domain; every step is an exact identity over free
  // variables within their bounds). A candidate row V is then replaced by
  // C(j) only when V is exactly L(j), node for node after full
  // instantiation, at a coordinate j that is an integer inside A's domain
  // for every value of its own free variables. Anything else stays explicit.
  struct Family {
    Target target;
    Id axis = invalid;          // the formal variable A
    size_t position = 0;        // a parameter that is A itself
    size_t depth = 0;
    // (row, coordinate) pairs by the row's arity: only rows of one shape
    // can propose one template.
    std::map<size_t, std::vector<std::pair<Id, Id>>> samples;
    std::unique_ptr<WholeExpressionTemplates> rows;
    Id closed = invalid;        // C(A)
    bool failed = false;
    size_t attempts = 0, instances = 0, declines = 0;
  };
  static constexpr size_t max_families = 8, max_family_depth = 3, max_family_attempts = 4;
  static constexpr size_t max_discovery_terms = 4;
  // Structural work of the parametric route (the DAG scan and rebuild, and
  // template construction and instantiation) is bounded separately from the
  // node budget, which keeps bounding every decomposition, factor choice
  // and body match. One unit per visited or rebuilt node.
  static constexpr uint64_t max_structure_nodes = uint64_t(1) << 20;
  uint64_t structure = 0;
  std::vector<Family> families;
  std::map<std::pair<size_t, Id>, std::optional<Id>> row_coordinates;
  void structure_tick(uint64_t units = 1) {
    a.check();
    structure += units;
    if (structure > max_structure_nodes) throw Miss{"structure_budget"};
  }
  bool family_spec(Id sid, size_t arguments);
  void collect_families();
  bool row_value(Id value, Id axis);
  std::optional<Id> row_coordinate(size_t index, Id value);
  void prove_family(Family &f, std::vector<std::pair<Id, Id>> &samples);
  std::optional<Id> close_row(Id value, bool &proved_new);
  std::optional<Id> close_ordinary(Id value);
  // Candidates as in close_shared_sums: sums that are product factors used
  // at least twice or arguments of unary numeric nodes. Computed once on the
  // optimized DAG; a rewrite never changes which stored node is a candidate.
  struct ParametricScan { std::unordered_map<Id, size_t> uses; std::unordered_set<Id> unary_arguments; };
  void parametric_scan(Id root, ParametricScan &scan);
  Id parametric_pass(Id root, const ParametricScan &scan, bool &proved_new);
  // Two Sum Calls of one count are equal when their bodies, instantiated at
  // one fresh coordinate k over that count, are equal by the exact scalar
  // engine: then every summand agrees and so does the sum. Nested Calls of
  // the second body that equal (by this rule, at most max_call_equality_depth
  // levels) a nested Call of the first body are identified first. This
  // relates, e.g., the reference's output-row reduction and the same
  // reduction registered again inside a norm over all rows (patterns that
  // differ in operand order or slot numbering). Memoized per attempt.
  static constexpr unsigned max_call_equality_depth = 3;
  std::map<std::pair<Id, Id>, bool> call_equalities;
  std::optional<std::pair<Id, Id>> call_body(Id call, Id &axis, unsigned depth);
  Id construct_body(Id root, const std::vector<Id> &cap);
  std::vector<Id> call_atoms(Id root);
  Id replace_atoms(Id root, const std::unordered_map<Id, Id> &replacement);
  std::unordered_map<Id, Id> equal_call_pairs(const std::vector<Id> &expected, const std::vector<Id> &actual,
                                              unsigned depth);
  bool equal_calls(Id x, Id y, unsigned depth);
  std::optional<LinearReductionProof> parametric_attempt(Id reference, Id optimized);
public:
  std::optional<LinearReductionProof> prove(Id reference, Id optimized);
};
} // namespace rg

#include "linear_reduction_cases.hpp"
#include "linear_reduction_case_bounds.hpp"
#include "linear_reduction_finiteness.hpp"
#include "linear_reduction_atoms.hpp"
#include "linear_reduction_targets.hpp"
#include "linear_reduction_forms.hpp"
#include "linear_reduction_proof.hpp"
#include "linear_reduction_fusion.hpp"
#include "linear_reduction_parametric.hpp"
