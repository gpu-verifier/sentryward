#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <list>
#include <optional>
#include <span>
#include <string_view>
#include <tuple>
#include <vector>
#include <unordered_map>

namespace sentryward::equivalence::finite_prefix_certificate {
using ExprId=std::uint64_t;
using ScopeId=std::uint32_t;
enum class Kind { Opaque, Zero, One, Add, Subtract, Negate, Multiply, NaturalExp };
struct Node {
  Kind kind=Kind::Opaque;
  std::array<ExprId,2> children{};
  unsigned arity=0;
  std::string_view type,attributes,provenance;
  // The adapter reads actual immutable selected-head admission/property facts.
  // Unknown/exact/bit-observing/unsupported-policy failures default to deny.
  bool finite_ideal=false,exact_dependencies=false;
  std::array<std::uint32_t,8> free_indices{};
  unsigned free_count=0;
};
struct Interval {std::uint64_t begin=0,end=0;bool operator==(const Interval&) const=default;};
// One relation request carries at most the time and state axes. Scope IDs
// identify immutable lexical/domain contexts, not merely free-index sets.
struct AxisMap {
  std::uint32_t reference_axis=0;
  std::optional<std::uint32_t> actual_axis;
  // actual(reference_axis) = actual_axis + offset, or the fixed point offset.
  std::uint64_t offset=0,extent=0;
};
struct RelationRequest {
  ExprId reference=0,actual=0;
  ScopeId reference_scope=0,actual_scope=0;
  std::array<AxisMap,2> axes{};
  unsigned axis_count=0;
  // The callback must stop before this work allowance; it may not reset a
  // nested body-walk ledger or hide work performed after a local cap.
  std::size_t max_work=0;
};
enum class RelationStatus { Unknown, Proved };
struct RelationResult {RelationStatus status=RelationStatus::Unknown;std::size_t work=0;};
using Resolve=std::function<std::optional<Node>(ExprId,ScopeId)>;
// This callback MUST be the full typed/finite/scoped/affine proof factory.
// It validates complete leaves, declared domains/offsets, injectivity/capture,
// and its selected witness's cycles. A shape/hash/name match is insufficient.
using Relate=std::function<RelationResult(const RelationRequest&)>;
using WorkCharge=std::function<bool(std::size_t)>;
struct Limits {
  // max_states bounds retained exact membership runs, NOT logical endpoints.
  // Memo limits are resident bounds; eviction never resets max_work.
  // Default4096 is unchanged. An explicitly configured physical-header
  // resource policy may retain up to65536 runs; every run is still minted by
  // the same complete update proof, fields/charges and checked AP membership.
  std::size_t max_frames=256,max_states=4096,max_cached_nodes=8192,max_relations=4096;
  // Default remains1M. An explicitly configured physical-epoch resource
  // policy may supply up to16M; this does not alter any proof/cap guard above.
  std::size_t max_work=1000000;
};
struct Statistics {
  std::size_t frames=0,states=0,resolves=0,relations=0,updates=0,work=0;
  std::size_t membership_runs=0,peak_cached_nodes=0,peak_cached_relations=0;
};
class Builder {
 public:
  struct FrameId {
    bool operator==(const FrameId&) const=default;
   private:std::uint64_t owner=0;std::size_t id=0;friend class Builder;
  };
  struct StateId {
    bool operator==(const StateId&) const=default;
    // Indexing hint only; equality still compares the complete private key.
    // Never mint a certificate from a fingerprint or use it as an ExprId.
    std::size_t hash() const noexcept {
      auto h=std::hash<std::uint64_t>{}(owner);
      const auto mix=[&](std::size_t v){h^=v+0x9e3779b9u+(h<<6)+(h>>2);};
      mix(std::hash<std::size_t>{}(run));mix(std::hash<std::uint64_t>{}(ordinal));return h;
    }
   private:std::uint64_t owner=0;std::size_t run=0;std::uint64_t ordinal=0;friend class Builder;
  };
  struct Signature {
    bool operator==(const Signature&) const=default;
   private:std::uint64_t owner=0;std::size_t frame=0;std::uint64_t endpoint=0;friend class Builder;
  };
  struct CompletedPrefixFamilyId {
    bool operator==(const CompletedPrefixFamilyId&) const=default;
    std::size_t hash() const noexcept {
      auto h=std::hash<std::uint64_t>{}(owner);
      h^=terminal.hash()+0x9e3779b9u+(h<<6)+(h>>2);return h;
    }
   private:std::uint64_t owner=0;StateId terminal;friend class Builder;
  };
  struct FrameRequest {
    ExprId potential_template=0,injection_template=0;
    ScopeId template_scope=0,state_scope=0;
    Interval time;
    std::uint32_t template_time_axis=0;
    std::optional<std::uint32_t> template_state_axis,actual_state_axis,output_time_axis;
    std::uint64_t state_extent=0;
    ExprId entry_state=0,entry_potential=0;
    // A preceding frame is optional for arbitrary actual initial state. When
    // supplied its actual state body is proved equal, never reset or discarded.
    std::optional<StateId> carry;
  };
  struct UpdateRequest {
    ExprId state=0,potential=0,injection=0;
    std::uint64_t endpoint=0;
  };
  struct FrameView {
    ExprId potential_template,injection_template,entry_state,entry_potential;
    ScopeId template_scope,state_scope;
    Interval time;
    std::uint32_t template_time_axis;
    std::optional<std::uint32_t> template_state_axis,actual_state_axis,output_time_axis;
    std::uint64_t state_extent;
    std::optional<StateId> carry;
  };
  struct StateView {FrameId frame;ExprId expression,potential;std::uint64_t endpoint;};
  struct CompletedPrefixView {FrameId frame;FrameView definition;StateId terminal;};
  // When a shared allowance is supplied, Relate debits that SAME allowance
  // before its internal work and returns the charged delta. Builder accounts
  // this delta locally only; its own operations debit shared work directly.
  // The hook is synchronous/non-reentrant and must not throw. Empty preserves
  // the original local-only budget behavior.
  Builder(Resolve,Relate,Limits={},WorkCharge={});
  Builder(const Builder&)=delete;
  Builder& operator=(const Builder&)=delete;
  // Registers actual finite templates/entry. No public family/admission token
  // can be fabricated; advance() invokes Relate before minting any StateId.
  std::optional<FrameId> begin(const FrameRequest&);
  std::optional<StateId> advance(FrameId,std::optional<StateId>,const UpdateRequest&);
  std::optional<Signature> signature(StateId);
  std::optional<bool> equal(const Signature&,const Signature&);
  // The final certified endpoint's predecessor chain covers the actual frame
  // consecutively by advance(). No endpoint/member enumeration is needed.
  // This is a prefix-family theorem, not a causal-region/output certificate.
  std::optional<CompletedPrefixFamilyId> complete(StateId);
  // Views are minted from valid private handles. They do not admit a region,
  // body relation or projection, and cannot be submitted as a fabricated ID.
  std::optional<FrameView> view(FrameId);
  std::optional<StateView> view(StateId);
  std::optional<CompletedPrefixView> view(CompletedPrefixFamilyId);
  Statistics statistics() const;
  bool chargesSharedWork() const noexcept {return static_cast<bool>(charge_shared_);}
  bool exhausted() const;
 private:
  struct Frame {FrameRequest request;std::optional<std::size_t> tail_run;};
  struct State {std::size_t frame;ExprId expression,potential;std::uint64_t endpoint;};
  struct MembershipRun {
    std::size_t frame;
    ExprId first_expression,first_potential,expression_stride=0,potential_stride=0;
    std::uint64_t first_endpoint,count=1;
  };
  bool charge(std::size_t=1,bool shared=true);
  std::optional<Node> read(ExprId,ScopeId);
  std::optional<State> state(StateId) const;
  bool dependencies(const Node&,std::optional<std::uint32_t>) const;
  bool related(const RelationRequest&);
  bool weight(ExprId,ExprId,ExprId,ScopeId);
  bool scaled(ExprId,ExprId,ExprId,ExprId,ScopeId);
  bool update(ExprId,ExprId,ExprId,ExprId,ExprId,ScopeId,bool);
  std::optional<StateId> append(std::size_t,const UpdateRequest&);
  Resolve resolve_;Relate relate_;Limits limits_;WorkCharge charge_shared_;Statistics statistics_;
  bool exhausted_=false;std::uint64_t owner_=0;
  struct NodeKey {ExprId expr;ScopeId scope;bool operator==(const NodeKey&) const=default;};
  struct NodeHash {std::size_t operator()(const NodeKey&) const;};
  struct CachedNode {std::optional<Node> node;std::list<NodeKey>::iterator age;};
  std::unordered_map<NodeKey,CachedNode,NodeHash> nodes_;
  std::list<NodeKey> node_age_;
  using RelationKey=std::tuple<ExprId,ExprId,ScopeId,ScopeId,
      std::uint32_t,std::optional<std::uint32_t>,std::uint64_t,std::uint64_t,
      std::uint32_t,std::optional<std::uint32_t>,std::uint64_t,std::uint64_t,unsigned>;
  struct RelationHash {std::size_t operator()(const RelationKey&) const;};
  struct CachedRelation {bool proved;std::list<RelationKey>::iterator age;};
  std::unordered_map<RelationKey,CachedRelation,RelationHash> relations_;
  std::list<RelationKey> relation_age_;
  std::vector<Frame> frames_;std::vector<MembershipRun> runs_;
};
} // namespace sentryward::equivalence::finite_prefix_certificate
