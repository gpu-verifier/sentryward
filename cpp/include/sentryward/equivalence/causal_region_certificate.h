#pragma once

#include "sentryward/equivalence/bounded_linear_reduction.h"
#include "sentryward/equivalence/finite_prefix_certificate.h"

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace sentryward::equivalence::causal_region_certificate {
namespace prefix=finite_prefix_certificate;
using ExprId=prefix::ExprId;
using ScopeId=prefix::ScopeId;
using Interval=prefix::Interval;
struct Use {ExprId expression=0;ScopeId scope=0;bool operator==(const Use&) const=default;};
enum class Kind {Opaque,Zero,Add,Subtract,Negate,Multiply,NaturalExp,Select,Sum};
struct Node {
  Kind kind=Kind::Opaque;
  std::array<ExprId,3> children{};
  unsigned arity=0;
  std::string_view type,attributes,provenance;
  bool finite_ideal=false,exact_dependencies=false;
  std::array<std::uint32_t,8> free_indices{};
  unsigned free_count=0;
  std::uint32_t binder=0;
  std::uint64_t count=0;
};
using Resolve=std::function<std::optional<Node>(Use)>;
struct ScopeResult {std::optional<ScopeId> scope;std::size_t work=0;};
// Re-read the actual Sum header, validate binder==parent lexical rank and
// count/type/policy, and mint ONLY its original BODY scope. Seed stays in parent.
// The existing adapter debits its own SAME aggregate allowance internally,
// before work, and reports that delta. Builder records the local delta only.
using BindBody=std::function<ScopeResult(Use,std::size_t)>;
struct FoldUse {Use head;ScopeId body_scope=0;std::uint32_t binder=0;std::uint64_t count=0;};
struct Weight {Use factor,target_potential,source_potential;};
struct ProjectionRequest {
  Use body; // Actual or owner-bound joined family, never a widened graph body.
  std::uint32_t time_axis=0,state_axis=0;
  std::uint64_t state_extent=0;
  bool operator==(const ProjectionRequest&) const=default;
};
enum class ProductKind {InjectionProjection,StateProjection,EntryProjection,InjectionIdentity,EntryIdentity};
struct OutputTarget {
  // Private owned records from the SAME Engine epoch. time_role is anchored
  // to ACTUAL source dispatch cases/private StateIds, never P(T) equality.
  std::uint64_t coordinate_token=0,time_role_token=0;
  ScopeId original_scope=0;
  Interval image;
  bool operator==(const OutputTarget&) const=default;
};
struct OutputTargetRequest {
  Use region_root,output_coordinate,source_dispatch,source_coordinate;
  prefix::Builder::CompletedPrefixFamilyId completed;
  prefix::Builder::CompletedPrefixView completed_prefix;
  ProjectionRequest projection;
  std::size_t max_work=0;
};
struct OutputTargetResult {
  prefix::RelationStatus status=prefix::RelationStatus::Unknown;
  OutputTarget target;
  std::size_t work=0;
};
// Scan/cache ACTUAL stored source dispatch once: exact cases/global endpoint
// coverage, attached private StateIds in completed.frame, full C projection/
// state Sum seed/body/coverage, and actual source-root connection. Then prove
// region's reachable exact-u64 coordinate is the SAME affine global coordinate
// in original scopes/output mapping, with no-wrap image subset. A constant P
// or equal image LENGTH cannot anchor the time role. No supplied proof flags.
// This callback owns its aggregate debit-before-work like the body factories.
using ProveOutputTarget=std::function<OutputTargetResult(const OutputTargetRequest&)>;
struct StateTarget {
  std::uint64_t coordinate_token=0,state_role_token=0;
  ScopeId original_scope=0;Interval image;
  bool operator==(const StateTarget&) const=default;
};
enum class StateTargetMode {Identity,PairedRestriction};
struct StateTargetRequest {
  Use region_root,state_coordinate,source_state,source_coordinate;
  prefix::Builder::StateId terminal;
  prefix::Builder::StateView terminal_state;
  prefix::Builder::FrameView frame;
  std::size_t max_work=0;
  StateTargetMode mode=StateTargetMode::Identity;
};
struct StateTargetResult {
  prefix::RelationStatus status=prefix::RelationStatus::Unknown;
  StateTarget target;std::size_t work=0;
  // PairedRestriction only: full source coordinate's RESTRICTED image, with
  // its original scope retained. target describes the narrow region side.
  std::optional<StateTarget> source_target={};
};
// Attach the ACTUAL source root to the private terminal StateId by the full
// typed/scoped relation, then prove the actual RHS coordinate is the SAME exact
// state role/image in ORIGINAL scopes. No identity matrix or C=1 is invented.
// PairedRestriction additionally validates BOTH original uses/coordinates:
// derive the narrow image from actual typed cells and owned Catalog domains,
// attach the full source use to terminal, and prove a unit-offset assignment
// with exact counts/no-wrap/captures through the SAME full scoped mapper. The
// two targets must retain their respective original scopes and share the
// adapter-owned assignment/role tokens and image. The local coordinate need
// not contain the global offset as a graph node. This is geometry/attachment,
// NOT a proof of the region's recurrence function or of a full-state union.
// Shared debit-before-work convention is identical to ProveOutputTarget.
using ProveStateTarget=std::function<StateTargetResult(const StateTargetRequest&)>;
class Builder;
struct OutputDomainId {
  bool operator==(const OutputDomainId&) const=default;
 private:
  std::uint64_t owner=0;std::size_t index=0;friend class Builder;
};
struct ProductRequest {
  ProductKind kind=ProductKind::InjectionProjection;
  prefix::Builder::FrameView frame;
  // Exactly one tagged target. Variable domain NEVER fabricates a StateView.
  std::optional<prefix::Builder::StateView> endpoint;
  std::optional<prefix::Builder::CompletedPrefixFamilyId> completed;
  std::optional<OutputTarget> output;
  std::optional<StateTarget> state_output;
  std::optional<OutputDomainId> output_domain;
  std::optional<ProjectionRequest> projection;
  std::optional<FoldUse> time_fold;
  std::optional<FoldUse> state_fold;
  std::optional<Weight> weight;
  // Every occurrence is retained, with ORIGINAL source scope. The bridge
  // validates all uses, crossed-binder independence, captures and complete
  // P/Q/C/state body relations using the SAME scoped mapper. No new AC walker,
  // graph enode, representative scope widening or caller-admitted flag.
  std::array<Use,8> factors{};
  unsigned factor_count=0;
  std::size_t max_work=0;
};
struct ProductResult {
  prefix::RelationStatus status=prefix::RelationStatus::Unknown;
  Interval time,state;
  // Private owner/epoch-bound coordinate proof record in the owning adapter.
  // Only that adapter can resolve it; NOT an ExprId, scope or supplied mask fact.
  std::uint64_t geometry_token=0;
  std::size_t work=0;
};
using RelateProduct=std::function<ProductResult(const ProductRequest&)>;
struct MaskRequest {
  Use predicate;
  bool selected_when_true=true;
  std::uint64_t geometry_token=0;
  std::optional<std::uint64_t> endpoint;
  std::optional<OutputTarget> output;
  std::optional<StateTarget> state_output;
  std::optional<OutputDomainId> output_domain;
  std::size_t max_work=0;
};
// Validate actual exact unsigned predicate as U<=T (or its selected-arm
// complement), in the namespace/context of the private coordinate token.
// Unsupported type/modifier/provenance, wrong/strict/shifted inequality fails.
// BindBody, RelateProduct and ProveMask all follow the same single-debit
// convention: internally charge the owning shared ledger, return charged work.
using ProveMask=std::function<prefix::RelationResult(const MaskRequest&)>;
struct Limits {
  // Default retained-header capacity remains8192. Opt-in physical-epoch
  // callers may configure up to65536; this never admits a node or proof.
  std::size_t max_nodes=8192,max_projections=256,max_certificates=4096;
  // Default remains1M; explicitly configured epoch work is capped32M.
  // Pieces/projections/certificates retain their existing hard maxima.
  std::size_t max_pieces=4096,max_work=1000000;
};
struct Statistics {
  std::size_t nodes=0,forest_queries=0,products=0,masks=0,pieces=0,work=0;
  std::size_t syntax_candidates=0;
  std::size_t stored_slots=0,queries=0,certificates=0;
};
// Untrusted syntax only: all uses retain ORIGINAL scopes/occurrences. A group
// is one stored outer time Fold, or all outside-time projected terms whose
// combined state coverage must be proved once. Identity retains one outside
// term only. Neither grouping nor Weight admits a frame, domain or equality.
struct Candidate {
  std::size_t group=0;
  std::optional<FoldUse> time_fold;
  std::optional<FoldUse> state_fold;
  std::array<Use,8> factors{};
  unsigned factor_count=0;
  std::optional<Weight> weight;
  std::optional<Use> predicate;
  bool selected_when_true=true;
};
enum class CandidateMode {Contraction,Identity};
class Builder {
 public:
  struct ProjectionId {
    bool operator==(const ProjectionId&) const=default;
   private:std::uint64_t owner=0;std::size_t index=0;friend class Builder;
  };
  struct Certificate {
    bool operator==(const Certificate&) const=default;
   private:std::uint64_t owner=0;std::size_t index=0;friend class Builder;
  };
  struct StateRestrictionId {
    bool operator==(const StateRestrictionId&) const=default;
   private:std::uint64_t owner=0;std::size_t index=0;friend class Builder;
  };
  struct Request {Use root;prefix::Builder::StateId endpoint;ProjectionId projection;};
  struct ParameterizedRequest {
    Use root,output_coordinate,source_dispatch,source_coordinate;
    prefix::Builder::CompletedPrefixFamilyId completed;
    ProjectionId projection;
  };
  struct IdentityRequest {
    Use root,state_coordinate,source_state,source_coordinate;
    prefix::Builder::StateId terminal;
    std::optional<StateRestrictionId> restriction={};
  };
  struct StateRestrictionRequest {
    Use region_root,state_coordinate,source_state,source_coordinate;
    prefix::Builder::StateId terminal;
    bool operator==(const StateRestrictionRequest&) const=default;
  };
  // Borrowed core, callbacks, headers and shared ledger MUST
  // outlive this immutable epoch. charge_shared is the EXISTING aggregate
  // allowance: all header, scope, mapper, mask and interval work charges it.
  Builder(prefix::Builder&,Resolve,BindBody,
          RelateProduct,ProveMask,std::function<bool(std::size_t)> charge_shared,Limits={},ProveOutputTarget={},ProveStateTarget={});
  ~Builder();
  Builder(const Builder&)=delete;
  Builder& operator=(const Builder&)=delete;
  std::optional<ProjectionId> projection(const ProjectionRequest&);
  // One cached immutable view from the SAME parser later used by region().
  // This can discover actual P/Q tile hints before core.begin(), but cannot
  // admit a caller-proved body, widened scope, seed or output equality. Span
  // lives until Builder destruction; its records never mutate after publish.
  std::optional<std::span<const Candidate>> candidates(Use actual_root,CandidateMode=CandidateMode::Contraction);
  // Exact component equality, NOT whole-output dispatch/residual equality.
  // contraction: actual Sum_s(C(T,s)*actual certified state).
  // region: actual fully covered causal time/state Sum + actual entry correction.
  // Fixed component API; do NOT call it per logical output/selector point.
  std::optional<Certificate> contraction(const Request&);
  std::optional<Certificate> region(const Request&);
  // One parameterized region for an ACTUAL output domain. Source dispatch
  // scanner is cached by its owning adapter, not called afresh per endpoint.
  std::optional<Certificate> parameterizedRegion(const ParameterizedRequest&);
  // Same API requires root==source_dispatch and coordinate==source_coordinate;
  // ProveOutputTarget must prove the complete actual dispatch contraction.
  std::optional<Certificate> parameterizedDispatch(const ParameterizedRequest&);
  // One paired component domain, minted ONLY by the checked pair factory.
  // Both graph scopes remain unchanged. Source identity then denotes the
  // terminal function restricted to this image, not the entire source root.
  std::optional<StateRestrictionId> stateRestriction(const StateRestrictionRequest&);
  // True vector function S_terminal(s), separately tagged from contractions.
  // Source method requires root==source_state and coordinate==source_coordinate.
  // RHS has only the ACTUAL time Sum, not a synthetic state Sum/C matrix.
  // Optional restriction must be the SAME owned pair on both sides. Original
  // query uses/scopes remain distinct; full old time and entry correction are
  // still checked by identityRegion. Default unrestricted API is unchanged.
  std::optional<Certificate> identityState(const IdentityRequest&);
  std::optional<Certificate> identityRegion(const IdentityRequest&);
  std::optional<bool> equal(Certificate,Certificate) const;
  Statistics statistics() const;
  bool exhausted() const;
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace sentryward::equivalence::causal_region_certificate
