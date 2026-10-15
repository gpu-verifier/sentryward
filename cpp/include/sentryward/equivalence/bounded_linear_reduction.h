#pragma once

#include "sentryward/equivalence/polynomial_certificate.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sentryward::equivalence::bounded_linear_reduction {

using Atom = polynomial_certificate::Atom;
using NodeId = std::size_t;
using Coefficient = polynomial_certificate::CanonicalPolynomial;
using WorkCharge = polynomial_certificate::WorkCharge;

enum class NodeKind { Literal, Atom, Identity, Add, Subtract, Negate, Multiply, Divide };
struct Node {
  NodeKind kind = NodeKind::Atom;
  std::array<NodeId,2> children{};
  std::size_t child_count = 0;
  Atom atom = 0;
  std::string_view literal;
  std::string_view type;
  // Default-deny admission. Set only after checking finite ideal semantics,
  // attributes, provenance, and the selected acyclic head's source domain.
  bool admitted = false;
};
struct Statistics {
  std::size_t nodes = 0;
  // Peak live scalar coefficients, including the constant-only node cache,
  // query frontier and final atom accumulator. NOT all expanded prefixes.
  std::size_t cached_terms = 0;
  std::size_t work = 0;
  std::size_t edges = 0;
  std::size_t queries = 0;
};
struct RankedDagStatistics {
  std::size_t nodes=0,edges=0,work=0;
};
// Frozen epoch owners map every stored node ID to its normalized equivalence
// class (owner[owner[id]]==owner[id]).
// A head is admissible ONLY when every child-class minimum stored-node rank is
// strictly below its result-class rank. Any mixture of returned heads remains
// acyclic, including when proven aliases make the full class graph cyclic.
// This is a structural factory, not arithmetic admission: callers still check
// types, attributes, provenance, finiteness, and availability of child heads.
// Child spans must refer to immutable epoch storage until the call returns.
std::optional<std::vector<bool>> rankedDagHeads(
    std::span<const NodeId> owners,
    const std::function<std::span<const NodeId>(NodeId)>& children,
    std::size_t max_nodes,std::size_t max_work,RankedDagStatistics* = nullptr);

struct Term {
  Atom atom = 0;
  Coefficient weight;
  bool operator==(const Term&) const = default;
};

class Builder;
class DagSession;
class LinearForm {
 public:
  LinearForm() = default; // Exact zero, not Unknown.
  const Coefficient& constant() const { return constant_; }
  const std::vector<Term>& terms() const { return terms_; }
  bool operator==(const LinearForm&) const = default;
 private:
  friend class Builder;
  friend class DagSession;
  Coefficient constant_;
  std::vector<Term> terms_; // Sorted distinct atom IDs; zero weights omitted.
};
using Result = std::optional<LinearForm>;

struct Limits {
  std::size_t max_atoms = 512;
  // Compact retained DagSession headers, not expanded forms/Fold members.
  // Default4096; an explicit physical-epoch policy may request up to65536.
  // This also incidentally bounds a single query's reached stored catalog.
  std::size_t max_nodes = 4096;
  std::size_t max_cached_terms = 32768;
  std::size_t max_intervals = 1024; // Combined left + right pieces.
  std::size_t max_work = 1000000;
  polynomial_certificate::Limits coefficients;
};
enum class Operation { Add, Subtract, Negate, Multiply, Divide };

struct WeightedInterval {
  std::uint64_t begin = 0;
  std::uint64_t end = 0; // Half-open, not begin + unchecked count.
  Coefficient weight;
};

// All source values MUST have a certified finite ideal-real interpretation.
// This is an exact linear certificate, not floating-point reassociation.
// Multiply admits a constant side only; Divide admits a nonzero constant
// denominator only. No arbitrary denominator cancellation or Fold expansion.
class Builder {
 public:
  // One hook observes BOTH local and exact-coefficient primitive charges.
  // DagSession already delegates its charges through this same Builder.
  explicit Builder(Limits = {}, WorkCharge = {});
  Result literal(std::string_view);
  Result atom(Atom);
  Result apply(Operation, std::span<const LinearForm* const>);
  std::optional<Coefficient> coefficient(std::string_view);
  // true is a certificate, false is a certificate mismatch (not a kernel
  // inequivalence result), empty is Unknown. Fold intervals stay symbolic.
  std::optional<bool> intervalsEqual(std::span<const WeightedInterval>,
                                     std::span<const WeightedInterval>);
  std::optional<std::vector<WeightedInterval>> canonicalIntervals(
      std::span<const WeightedInterval>);
  std::size_t work() const;
  bool exhausted() const;
 private:
  friend class DagSession;
  friend Result summarizeDag(NodeId, const std::function<std::optional<Node>(NodeId)>&,
                             Limits, Statistics*);
  bool charge(std::size_t);
  bool admitted(const LinearForm&);
  bool constantCoefficient(const Coefficient&);
  std::optional<Coefficient> combineCoefficient(const Coefficient&, const Coefficient&, bool);
  std::optional<Coefficient> productCoefficient(const Coefficient&, const Coefficient&);
  std::optional<Coefficient> reciprocalCoefficient(const Coefficient&);
  Result combine(const LinearForm&, const LinearForm&, bool);
  Result negate(const LinearForm&);
  Result scale(const LinearForm&, const Coefficient&);
  Limits limits_;
  std::size_t local_work_ = 0;
  bool exhausted_ = false;
  // One callable identity, including for a stateful value-captured functor.
  std::shared_ptr<WorkCharge> work_charge_;
  polynomial_certificate::Builder coefficients_;
};

// A frozen selected-head catalog shared by queries. Resolve each stored ID
// once; retain only its header and an optional EXACT constant coefficient.
// Each query validates its entire reached DAG, then accumulates reverse-
// topological scalar adjoints. No growing LinearForm is copied/cached for an
// internal prefix. Duplicate DAG edges contribute their full multiplicity.
//
// All borrowed headers/string_views and the Builder MUST outlive the session;
// the resolver's selected graph MUST remain immutable. Results own their
// coefficients and remain valid after subsequent queries. A limit yields
// persistent Unknown; invalid/cyclic heads remain Unknown, not opaque aliases.
// Fold/Select/nonlinear heads may be admitted opaque atoms by the resolver;
// Identity supports a validated outside-binder zero-count Fold seed. No binder
// or logical Fold member is opened by this API. Per query work is O(reached
// stored V+E+atom ordering), with separately bounded exact coefficient work.
class DagSession {
 public:
  enum class Cap { None, Nodes, CachedTerms, Work, Atoms, Coefficients };
  using Resolver = std::function<std::optional<Node>(NodeId)>;
  DagSession(Builder&, Resolver, Limits = {});
  DagSession(const DagSession&) = delete;
  DagSession& operator=(const DagSession&) = delete;
  Result summarize(NodeId);
  Statistics statistics() const;
  bool exhausted() const { return exhausted_ || builder_.exhausted(); }
  Cap cap() const { return cap_; }
 private:
  enum class State { Active, Ready, Unknown };
  struct Entry {
    Node node;
    State state = State::Active;
    std::optional<Coefficient> constant;
    std::optional<Coefficient> scale;
    NodeId linear_child = 0;
  };
  bool charge(std::size_t);
  bool coefficientFailure();
  bool live(std::size_t);
  bool catalog(NodeId);
  Builder& builder_;
  Resolver resolve_;
  Limits limits_;
  std::unordered_map<NodeId,Entry> entries_;
  std::size_t constant_terms_ = 0, peak_terms_ = 0, edges_ = 0, queries_ = 0;
  std::size_t initial_work_ = 0;
  bool exhausted_ = false;
  Cap cap_ = Cap::None;
};

// One-query compatibility wrapper around the SAME DagSession implementation.
Result summarizeDag(NodeId root, const std::function<std::optional<Node>(NodeId)>& resolve,
                    Limits = {}, Statistics* = nullptr);

} // namespace sentryward::equivalence::bounded_linear_reduction
