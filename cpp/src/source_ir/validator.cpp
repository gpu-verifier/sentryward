#include "sentryward/algorithm/source_ir/validator.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace sentryward::algorithm {
namespace {

template <typename... Callables>
struct Overloaded : Callables... {
  using Callables::operator()...;
};

template <typename... Callables>
Overloaded(Callables...) -> Overloaded<Callables...>;

struct StateDefinition final {
  BarrierLifetimeId barrier;
  SourceInstructions::Handle instruction;
};

struct StateUse final {
  BarrierLifetimeId barrier;
  StateId state;
  SourceInstructions::Handle instruction;
};

struct TxContribution final {
  std::uint64_t bytes;
  SourceInstructions::Handle instruction;
};

enum class CollectiveKind {
  kThreads,
  kWarp,
};

struct CollectiveCall final {
  SourceInstructions::Handle instruction;
  CollectiveKind kind;
  std::optional<std::uint32_t> lane_mask;
};

void AddDiagnostic(std::vector<Diagnostic>& diagnostics,
                   DiagnosticClass classification, DiagnosticCode code,
                   DiagnosticMessage message,
                   const SourceInstruction* instruction = nullptr) {
  diagnostics.push_back(Diagnostic{
      classification,
      code,
      std::move(message.text),
      instruction == nullptr ? std::nullopt
                             : std::optional<InstId>(instruction->id),
      instruction == nullptr
          ? std::nullopt
          : std::optional<SourceLocation>(instruction->location),
      std::move(message.ids),
  });
}

template <typename Id>
DiagnosticMessage IdValue(Id id) {
  return MessageIdText(id);
}

void AddDiagnostic(std::vector<Diagnostic>& out, DiagnosticClass classification,
    DiagnosticCode code, DiagnosticMessage message, SourceInstructions::Handle ref) {
  const auto instruction=*ref;
  AddDiagnostic(out,classification,code,std::move(message),&instruction);
}

bool IsValidEdgeKind(SourceEdgeKind kind) {
  return kind == SourceEdgeKind::kProgramOrder ||
         kind == SourceEdgeKind::kCompleteBeforeIssue;
}

// Dense per-actor memory chains admit streaming structural validation.
bool IsDenseMemoryChains(const SourceProgram& program,
                        const std::map<ActorId,const ActorInfo*>& actors) {
  if(!program.collectives.empty() || !program.dependencies.empty())return false;
  std::set<ActorId> seen;
  std::optional<ActorId> previous;
  auto edge=program.edges.begin();
  for(std::size_t i=0;i<program.instructions.size();++i) {
    const auto instruction=program.instructions[i];
    if(instruction.id.value()!=i || !actors.contains(instruction.actor) ||
       (!std::holds_alternative<ImmediateLoad>(instruction.op) &&
        !std::holds_alternative<ImmediateStore>(instruction.op)))return false;
    if(previous==instruction.actor) {
      if(edge==program.edges.end() ||
         *edge!=SourceEdge{InstId(i-1),InstId(i),SourceEdgeKind::kProgramOrder})return false;
      ++edge;
    } else if(!seen.insert(instruction.actor).second)return false;
    previous=instruction.actor;
  }
  return edge==program.edges.end();
}

// Rank r is the r-th smallest unique instruction ID; the first record declaring it owns it.
struct InstructionIndex final {
  std::vector<InstId> ids;
  std::vector<ActorId> actors;
  std::vector<std::size_t> records;  // empty when every rank is its own record
  bool dense=false;
  std::size_t record(std::size_t rank) const{return records.empty()?rank:records[rank];}
  std::optional<std::size_t> find(InstId id) const {
    if(dense)return id.value()<ids.size()?std::optional<std::size_t>(id.value()):std::nullopt;
    const auto it=std::lower_bound(ids.begin(),ids.end(),id);
    if(it==ids.end() || *it!=id)return std::nullopt;
    return static_cast<std::size_t>(it-ids.begin());
  }
};

// Declared edge kinds per (from rank, to rank) pair. Capacity is fixed: each edge adds at most
// one key, so the load stays at most 1/2 and probing terminates.
class EdgeKindSet final {
 public:
  explicit EdgeKindSet(std::size_t edges)
      : keys_(capacity(edges)),kinds_(keys_.size()),shift_(64-std::countr_zero(keys_.size())) {}
  // Returns the pair's kind mask before adding kind.
  std::uint8_t insert(std::uint64_t key,unsigned kind) {
    if(kind>=8)throw std::logic_error("source edge kind outside the kind mask");
    const auto bit=static_cast<std::uint8_t>(1u<<kind);
    for(auto i=static_cast<std::size_t>((key*0x9e3779b97f4a7c15ULL)>>shift_);;i=(i+1)&(keys_.size()-1)) {
      if(!kinds_[i]){keys_[i]=key;kinds_[i]=bit;return 0;}
      if(keys_[i]==key){const auto before=kinds_[i];kinds_[i]|=bit;return before;}
    }
  }
 private:
  static std::size_t capacity(std::size_t edges) {
    if(edges>(std::size_t{1}<<60))throw std::length_error("source edge table size");
    return std::bit_ceil(std::max<std::size_t>(16,2*edges));
  }
  std::vector<std::uint64_t> keys_;
  std::vector<std::uint8_t> kinds_;
  unsigned shift_;
};

using RankPairs=std::vector<std::pair<std::uint32_t,std::uint32_t>>;
struct Adjacency final { std::vector<std::size_t> offsets; std::vector<std::uint32_t> targets; };

// Successor lists of ranks [0, ranks) in pair order (stable counting sort by source).
Adjacency MakeAdjacency(std::size_t ranks,RankPairs pairs) {
  Adjacency result{std::vector<std::size_t>(ranks+1),std::vector<std::uint32_t>(pairs.size())};
  for(const auto& pair:pairs)++result.offsets[pair.first+1];
  std::partial_sum(result.offsets.begin(),result.offsets.end(),result.offsets.begin());
  std::vector<std::size_t> next(result.offsets.begin(),result.offsets.end()-1);
  for(const auto& pair:pairs)result.targets[next[pair.first]++]=pair.second;
  return result;
}

}  // namespace

std::vector<Diagnostic> SourceValidator::Validate(
    const SourceProgram& program) const {
  std::vector<Diagnostic> diagnostics;

  std::map<ActorId, const ActorInfo*> actors;
  std::map<std::tuple<CtaId, WarpId, std::uint32_t>, ActorId> coordinates;
  for (const ActorInfo& actor : program.actors) {
    if (!actors.emplace(actor.id, &actor).second) {
      AddDiagnostic(
          diagnostics, DiagnosticClass::kInvalidInput,
          DiagnosticCode::kDuplicateActorId,
          "actor " + IdValue(actor.id) + " is declared more than once");
    }
    if (actor.lane >= 32) {
      AddDiagnostic(
          diagnostics, DiagnosticClass::kInvalidInput,
          DiagnosticCode::kInvalidActorLane,
          "actor " + IdValue(actor.id) + " has a lane index outside [0, 32)");
    }
    const auto coordinate = std::tuple(actor.cta, actor.warp, actor.lane);
    if (const auto [found, inserted] =
            coordinates.emplace(coordinate, actor.id);
        !inserted && found->second != actor.id) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kDuplicateActorCoordinates,
                    "actors " + IdValue(found->second) + " and " +
                        IdValue(actor.id) +
                        " have identical CTA, warp, and lane coordinates");
    }
  }

  if(diagnostics.empty() && IsDenseMemoryChains(program,actors))return diagnostics;

  // Rank the instruction IDs once; ID-keyed lookups become rank-indexed arrays.
  InstructionIndex ranks;
  {
    const auto n=program.instructions.size();
    std::vector<InstId> record_ids(n);
    std::vector<ActorId> record_actors(n);
    bool ascending=true;
    for (std::size_t i=0;i<n;++i) {
      record_ids[i]=program.instructions.id(i);record_actors[i]=program.instructions.actor(i);
      ascending=ascending && (!i || record_ids[i-1]<record_ids[i]);
    }
    std::vector<bool> duplicate;  // per record; strictly ascending IDs have none
    if (!ascending) {
      // Sorting by (ID, record) puts the record std::map::emplace kept first.
      std::vector<std::pair<InstId,std::size_t>> order(n);
      for(std::size_t i=0;i<n;++i)order[i]={record_ids[i],i};
      std::sort(order.begin(),order.end());
      duplicate.assign(n,false);
      ranks.ids.reserve(n);ranks.actors.reserve(n);ranks.records.reserve(n);
      for (std::size_t k=0;k<n;++k) {
        if(k && order[k].first==order[k-1].first){duplicate[order[k].second]=true;continue;}
        ranks.ids.push_back(order[k].first);ranks.actors.push_back(record_actors[order[k].second]);
        ranks.records.push_back(order[k].second);
      }
    }
    std::optional<ActorId> checked_actor;
    bool known_actor=false;
    for (std::size_t i=0;i<n;++i) {
      if (!duplicate.empty() && duplicate[i]) {
        const auto instruction_view = program.instructions.view(i);
        AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kDuplicateInstructionId,
                      "instruction " + IdValue(record_ids[i]) +
                          " is declared more than once",
                      &*instruction_view);
      }
      if(checked_actor!=record_actors[i]){checked_actor=record_actors[i];known_actor=actors.contains(record_actors[i]);}
      if (!known_actor) {
        const auto instruction_view = program.instructions.view(i);
        AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kUnknownInstructionActor,
                      "instruction " + IdValue(record_ids[i]) +
                          " references unknown actor " +
                          IdValue(record_actors[i]),
                      &*instruction_view);
      }
    }
    if(ascending){ranks.ids=std::move(record_ids);ranks.actors=std::move(record_actors);}
    // Unique ascending IDs are 0..m-1 exactly when the largest is m-1.
    ranks.dense=ranks.ids.empty() || std::size_t(ranks.ids.back().value())+1==ranks.ids.size();
  }
  const auto handle=[&](std::size_t rank){return program.instructions.ref(ranks.record(rank));};
  // Every caller's ID is a declared one, as the former ID-keyed map's at() required.
  const auto handleOf=[&](InstId id) {
    const auto rank=ranks.find(id);
    if(!rank)throw std::out_of_range("instruction ID without a rank");
    return handle(*rank);
  };

  std::map<AsyncOpId, const SourceInstruction*> async_operations;
  std::map<BarrierLifetimeId, SourceInstructions::Handle> barrier_initializers;
  std::map<BarrierLifetimeId, SourceInstructions::Handle> barrier_uses;
  std::set<BarrierLifetimeId> named_barriers;
  std::map<StateId, StateDefinition> state_definitions;
  std::vector<StateUse> state_uses;
  std::vector<TxContribution> tx_contributions;
  std::map<SyncId, std::vector<CollectiveCall>> collective_calls;
  // Collective ID per record, sized at the first collective call; empty without one.
  std::vector<std::optional<SyncId>> record_syncs;

  const auto recordSync = [&](std::size_t record, SyncId sync) {
    if (record_syncs.empty()) record_syncs.resize(program.instructions.size());
    record_syncs[record] = sync;
  };
  const auto recordBarrierUse = [&](BarrierLifetimeId barrier,
                                    const SourceInstruction& instruction) {
    barrier_uses.try_emplace(barrier, handleOf(instruction.id));
  };
  const auto recordStateDefinition = [&](BarrierLifetimeId barrier,
                                         StateId state,
                                         const SourceInstruction& instruction) {
    if (!state_definitions
             .emplace(state, StateDefinition{barrier, handleOf(instruction.id)})
             .second) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kDuplicateStateDefinition,
                    "state " + IdValue(state) + " is defined more than once",
                    &instruction);
    }
  };
  const auto recordAsync = [&](AsyncOpId id,
                               const SourceInstruction& instruction) {
    if (!async_operations.emplace(id, &instruction).second) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kDuplicateAsyncOpId,
                    "asynchronous operation " + IdValue(id) +
                        " is declared more than once",
                    &instruction);
    }
  };
  const auto recordCompletion = [&](const MBarrierCompletion& completion,
                                    const SourceInstruction& instruction) {
    recordBarrierUse(completion.barrier, instruction);
    tx_contributions.push_back(TxContribution{completion.bytes, handleOf(instruction.id)});
  };
  const auto validateParity = [&](std::uint8_t parity,
                                  const SourceInstruction& instruction) {
    if (parity > 1) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kInvalidParity,
                    "instruction " + IdValue(instruction.id) +
                        " has parity outside {0, 1}",
                    &instruction);
    }
  };

  const auto validateRef = [&](const MilestoneRef& ref,
                               const SourceInstruction* user) {
    const auto rank = ranks.find(ref.instruction);
    bool valid = rank.has_value();
    if (valid) {
      const auto found = handle(*rank);
      switch (ref.milestone) {
        case Milestone::kIssue:
        case Milestone::kComplete:
          valid = ref.effect_index == 0;
          break;
        case Milestone::kReadsComplete:
          valid = found.holdsOperation<AsyncOp>() && ref.effect_index == 0;
          break;
        case Milestone::kEffect:
          valid = found.holdsOperation<AsyncOp>() &&
                  ref.effect_index < found.effectCount();
          break;
        default:
          valid = false;
      }
    }
    if (!valid) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kInvalidMilestoneReference,
                    "milestone reference has an unknown instruction, kind, or "
                    "unavailable endpoint",
                    user);
    }
  };

  for (const EndpointDependency& dependency : program.dependencies) {
    validateRef(dependency.from, nullptr);
    validateRef(dependency.to, nullptr);
  }

  for (std::size_t i=0;i<program.instructions.size();++i) {
    const auto instruction_view=program.instructions.view(i);
    const auto& instruction=*instruction_view;
    if(std::holds_alternative<ImmediateLoad>(instruction.op) ||
       std::holds_alternative<ImmediateStore>(instruction.op))continue;
    std::visit(
        Overloaded{
            [](const ImmediateLoad&) {},
            [](const ImmediateStore&) {},
            [&](const AsyncLoad& operation) {
              recordAsync(operation.async_id, instruction);
              if (operation.completion.has_value()) {
                recordCompletion(*operation.completion, instruction);
              }
            },
            [&](const AsyncStore& operation) {
              recordAsync(operation.async_id, instruction);
              if (operation.completion.has_value()) {
                recordCompletion(*operation.completion, instruction);
              }
            },
            [&](const AsyncOp& operation) {
              recordAsync(operation.async_id, instruction);
              if (operation.effects.size() >
                  std::numeric_limits<std::uint32_t>::max()) {
                AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                              DiagnosticCode::kInvalidMemoryEffect,
                              "too many asynchronous effects", &instruction);
              }
              for (const MilestoneRef& ref : operation.completion_requires)
                validateRef(ref, &instruction);
              for (const AsyncEffect& effect : operation.effects) {
                if (effect.accesses.size() >
                    std::numeric_limits<std::uint32_t>::max()) {
                  AddDiagnostic(
                      diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kInvalidMemoryEffect,
                      "too many memory accesses in an asynchronous effect",
                      &instruction);
                }
                for (const MilestoneRef& ref : effect.predecessors)
                  validateRef(ref, &instruction);
                for (const MemoryEffect& access : effect.accesses) {
                  bool valid = access.mode == AccessMode::kRead ||
                               access.mode == AccessMode::kWrite;
                  if (access.atomic_scope) {
                    valid = valid && access.mode == AccessMode::kWrite &&
                        std::holds_alternative<AddressId>(access.endpoint) &&
                        (access.bytes == 2 || access.bytes == 4 || access.bytes == 8) &&
                        (*access.atomic_scope == AtomicScope::kCta ||
                         *access.atomic_scope == AtomicScope::kGpu ||
                         *access.atomic_scope == AtomicScope::kSys);
                  }
                  if (const auto* reg =
                          std::get_if<RegisterAddress>(&access.endpoint)) {
                    valid = valid && actors.contains(reg->actor) &&
                            !reg->name.empty();
                  }
                  if (!valid) {
                    AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                                  DiagnosticCode::kInvalidMemoryEffect,
                                  "memory effect has an invalid access mode or "
                                  "register endpoint",
                                  &instruction);
                  }
                }
                if (effect.completion) {
                  std::visit(
                      Overloaded{
                          [&](const ArriveEffect& arrival) {
                            recordBarrierUse(arrival.barrier, instruction);
                            if (arrival.arrivals == 0) {
                              AddDiagnostic(
                                  diagnostics, DiagnosticClass::kInvalidInput,
                                  DiagnosticCode::kInvalidDeferredArrival,
                                  "deferred arrival count must be positive",
                                  &instruction);
                            }
                          },
                          [&](const CompleteTxEffect& completion) {
                            recordBarrierUse(completion.barrier, instruction);
                            tx_contributions.push_back(
                                {completion.bytes, handleOf(instruction.id)});
                          }},
                      *effect.completion);
                }
              }
            },
            [&](const Await& operation) {
              for (const MilestoneRef& ref : operation.requirements)
                validateRef(ref, &instruction);
            },
            [&](const NamedBarrierArrive& operation) {
              named_barriers.insert(operation.barrier);
              if (!operation.expected_arrivals)
                AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kZeroInitialArrivalCount, "named barrier count is zero", &instruction);
            },
            [&](const NamedBarrierSync& operation) {
              named_barriers.insert(operation.barrier);
              if (!operation.expected_arrivals)
                AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kZeroInitialArrivalCount, "named barrier count is zero", &instruction);
            },
            [&](const MBarrierInit& operation) {
              if (operation.expected_arrivals == 0) {
                AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                              DiagnosticCode::kZeroInitialArrivalCount,
                              "barrier " + IdValue(operation.barrier) +
                                  " has zero expected arrivals",
                              &instruction);
              }
              if (!barrier_initializers.emplace(operation.barrier, handleOf(instruction.id))
                       .second) {
                AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                              DiagnosticCode::kDuplicateBarrierInitialization,
                              "barrier lifetime " + IdValue(operation.barrier) +
                                  " is initialized more than once",
                              &instruction);
              }
            },
            [&](const MBarrierArrive& operation) {
              recordBarrierUse(operation.barrier, instruction);
              if (operation.result.has_value()) {
                recordStateDefinition(operation.barrier, *operation.result,
                                      instruction);
              }
            },
            [&](const MBarrierExpectTx& operation) {
              recordBarrierUse(operation.barrier, instruction);
              tx_contributions.push_back(
                  TxContribution{operation.bytes, handleOf(instruction.id)});
            },
            [&](const MBarrierArriveExpectTx& operation) {
              recordBarrierUse(operation.barrier, instruction);
              tx_contributions.push_back(
                  TxContribution{operation.bytes, handleOf(instruction.id)});
              if (operation.result.has_value()) {
                recordStateDefinition(operation.barrier, *operation.result,
                                      instruction);
              }
            },
            [&](const MBarrierCompleteTx& operation) {
              recordBarrierUse(operation.barrier, instruction);
              tx_contributions.push_back(
                  TxContribution{operation.bytes, handleOf(instruction.id)});
            },
            [&](const MBarrierAddArrivals& operation) {
              recordBarrierUse(operation.barrier, instruction);
            },
            [&](const MBarrierWaitState& operation) {
              recordBarrierUse(operation.barrier, instruction);
              state_uses.push_back(
                  StateUse{operation.barrier, operation.state, handleOf(instruction.id)});
            },
            [&](const MBarrierWaitParity& operation) {
              recordBarrierUse(operation.barrier, instruction);
              validateParity(operation.parity, instruction);
            },
            [&](const MBarrierTryWaitState& operation) {
              recordBarrierUse(operation.barrier, instruction);
              state_uses.push_back(
                  StateUse{operation.barrier, operation.state, handleOf(instruction.id)});
              AddDiagnostic(diagnostics, DiagnosticClass::kUnsupported,
                            DiagnosticCode::kRawTryWaitUnsupported,
                            "raw state try_wait is outside the v1 model",
                            &instruction);
            },
            [&](const MBarrierTryWaitParity& operation) {
              recordBarrierUse(operation.barrier, instruction);
              validateParity(operation.parity, instruction);
              AddDiagnostic(diagnostics, DiagnosticClass::kUnsupported,
                            DiagnosticCode::kRawTryWaitUnsupported,
                            "raw parity try_wait is outside the v1 model",
                            &instruction);
            },
            [&](const SyncThreads& operation) {
              collective_calls[operation.sync_id].push_back(CollectiveCall{
                  handleOf(instruction.id), CollectiveKind::kThreads, std::nullopt});
              recordSync(i, operation.sync_id);
            },
            [&](const SyncWarp& operation) {
              collective_calls[operation.sync_id].push_back(CollectiveCall{
                  handleOf(instruction.id), CollectiveKind::kWarp, operation.lane_mask});
              recordSync(i, operation.sync_id);
            },
        },
        instruction.op);
  }

  for (const auto barrier : named_barriers) {
    if (const auto init = barrier_initializers.find(barrier); init != barrier_initializers.end())
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
          DiagnosticCode::kCollectiveKindMismatch, "named barrier must not have an mbarrier initializer", init->second);
  }
  for (const auto& [barrier, use] : barrier_uses) {
    if (named_barriers.contains(barrier))
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
          DiagnosticCode::kCollectiveKindMismatch, "named and mbarrier operations share one identity", use);
    if (!barrier_initializers.contains(barrier)) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kMissingBarrierInitialization,
                    "barrier lifetime " + IdValue(barrier) +
                        " is used without an initializer",
                    use);
    }
  }

  for (const StateUse& use : state_uses) {
    const auto definition = state_definitions.find(use.state);
    if (definition == state_definitions.end()) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kMissingStateDefinition,
                    "state " + IdValue(use.state) + " has no definition",
                    use.instruction);
      continue;
    }
  }

  constexpr std::uint64_t kMaxSignedTransaction =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  for (const TxContribution& contribution : tx_contributions) {
    if (contribution.bytes > kMaxSignedTransaction) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kTransactionValueOutOfRange,
                    "transaction byte count exceeds int64_t",
                    contribution.instruction);
    }
  }

  // Edges in declaration order, as rank pairs; a pair enters the graph with its first kind.
  const auto m=ranks.ids.size();
  RankPairs graph_pairs,program_order_pairs;
  std::vector<std::size_t> indegrees(m),order_indegrees(m);
  {
    EdgeKindSet declared(program.edges.size());
    for (const SourceEdge& edge : program.edges) {
      if (!IsValidEdgeKind(edge.kind)) {
        AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kInvalidEdgeKind,
                      "source edge has an unknown dependency kind");
        continue;
      }

      const auto from = ranks.find(edge.from);
      const auto to = ranks.find(edge.to);
      if (!from || !to) {
        AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kUnknownEdgeEndpoint,
                      "source edge references an unknown instruction");
        continue;
      }

      const auto kind=static_cast<unsigned>(edge.kind);
      const auto before=declared.insert((std::uint64_t(*from)<<32)|*to,kind);
      if ((before>>kind)&1u) {
        AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kDuplicateEdge,
                      "source edge is declared more than once", handle(*to));
        continue;
      }

      if (edge.kind == SourceEdgeKind::kProgramOrder &&
          ranks.actors[*from] != ranks.actors[*to]) {
        AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kProgramOrderCrossesActors,
                      "program-order edge connects different actors", handle(*to));
      } else if (edge.kind == SourceEdgeKind::kProgramOrder) {
        program_order_pairs.emplace_back(std::uint32_t(*from),std::uint32_t(*to));
        ++order_indegrees[*to];
      }

      if (!before) {
        graph_pairs.emplace_back(std::uint32_t(*from),std::uint32_t(*to));
        ++indegrees[*to];
      }
    }
  }

  {
    // The processed set (instructions without a cyclic ancestor) does not depend on the order.
    const auto successors=MakeAdjacency(m,std::move(graph_pairs));
    std::vector<std::uint32_t> ready;
    for(std::size_t rank=0;rank<m;++rank)if(!indegrees[rank])ready.push_back(std::uint32_t(rank));
    std::size_t visited = 0;
    while (!ready.empty()) {
      const auto current=ready.back();ready.pop_back();++visited;
      for(auto k=successors.offsets[current];k<successors.offsets[current+1];++k)
        if(!--indegrees[successors.targets[k]])ready.push_back(successors.targets[k]);
    }
    if (visited != m) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kCyclicSourceGraph,
                    "the union of source dependency edges contains a cycle");
    }
    indegrees=std::vector<std::size_t>{};
  }

  // Consecutive collective calls of each actor in its program order.
  std::vector<std::pair<SyncId, SyncId>> collective_order;
  {
    // Accepted program-order pairs join instructions of one actor, so global program-order
    // indegrees restricted to an actor are that actor's. Actors without a root pop nothing.
    const auto successors=MakeAdjacency(m,std::move(program_order_pairs));
    std::map<ActorId, std::vector<std::uint32_t>> actor_roots;
    std::vector<std::uint32_t>* roots=nullptr;
    for (std::size_t rank=0;rank<m;++rank) {
      if(order_indegrees[rank])continue;
      if(!roots || ranks.actors[rank]!=ranks.actors[roots->back()])roots=&actor_roots[ranks.actors[rank]];
      roots->push_back(std::uint32_t(rank));
    }
    // Min-heap of ranks (rank order is ID order); its two smallest are ready[0] and
    // min(ready[1], ready[2]).
    std::vector<std::uint32_t> ready;
    for (auto& [actor, actor_program] : actor_roots) {
      ready=std::move(actor_program);
      std::make_heap(ready.begin(),ready.end(),std::greater<>{});
      bool reported_incomparability = false;
      std::optional<SyncId> previous;
      while (!ready.empty()) {
        if (ready.size() > 1 && !reported_incomparability) {
          const auto first = ready[0];
          const auto second = ready.size()==2 ? ready[1] : std::min(ready[1],ready[2]);
          AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                        DiagnosticCode::kIncompleteThreadProgramOrder,
                        "instructions " + IdValue(ranks.ids[first]) + " and " +
                            IdValue(ranks.ids[second]) + " of actor " + IdValue(actor) +
                            " are unordered in the source graph",
                        handle(second));
          reported_incomparability = true;
        }

        std::pop_heap(ready.begin(),ready.end(),std::greater<>{});
        const auto current=ready.back();ready.pop_back();
        if (!record_syncs.empty())
          if (const auto sync=record_syncs[ranks.record(current)]) {
            if(previous)collective_order.emplace_back(*previous,*sync);
            previous=sync;
          }
        for (auto k=successors.offsets[current];k<successors.offsets[current+1];++k)
          if (!--order_indegrees[successors.targets[k]]) {
            ready.push_back(successors.targets[k]);std::push_heap(ready.begin(),ready.end(),std::greater<>{});
          }
      }
    }
    order_indegrees=std::vector<std::size_t>{};
  }

  std::map<SyncId, const CollectiveSpec*> collective_specs;
  for (const CollectiveSpec& spec : program.collectives) {
    if (!collective_specs.emplace(spec.sync_id, &spec).second) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kDuplicateCollectiveSpec,
                    "collective " + IdValue(spec.sync_id) +
                        " has more than one specification");
    }

    if (spec.expected_participants.empty()) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kCollectiveParticipantMismatch,
                    "collective " + IdValue(spec.sync_id) +
                        " has no declared participants");
    }

    std::set<ActorId> participants;
    for (ActorId participant : spec.expected_participants) {
      if (!participants.insert(participant).second) {
        AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kDuplicateCollectiveParticipant,
                      "collective " + IdValue(spec.sync_id) +
                          " repeats participant " + IdValue(participant));
      }
      if (!actors.contains(participant)) {
        AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kUnknownCollectiveParticipant,
                      "collective " + IdValue(spec.sync_id) +
                          " references unknown participant " +
                          IdValue(participant));
      }
    }
  }

  for (const auto& [sync_id, calls] : collective_calls) {
    if (!collective_specs.contains(sync_id)) {
      AddDiagnostic(
          diagnostics, DiagnosticClass::kInvalidInput,
          DiagnosticCode::kUnknownCollective,
          "collective call references unknown sync id " + IdValue(sync_id),
          calls.front().instruction);
    }
  }

  for (const auto& [sync_id, spec] : collective_specs) {
    const auto found_calls = collective_calls.find(sync_id);
    const std::vector<CollectiveCall> empty_calls;
    const std::vector<CollectiveCall>& calls =
        found_calls == collective_calls.end() ? empty_calls
                                              : found_calls->second;
    const CollectiveKind expected_kind = spec->lane_mask.has_value()
                                             ? CollectiveKind::kWarp
                                             : CollectiveKind::kThreads;

    std::set<ActorId> declared(spec->expected_participants.begin(),
                               spec->expected_participants.end());
    std::set<ActorId> called;
    for (const CollectiveCall& call : calls) {
      if (!called.insert(call.instruction->actor).second) {
        AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kDuplicateCollectiveCall,
                      "actor " + IdValue(call.instruction->actor) +
                          " calls collective " + IdValue(sync_id) +
                          " more than once",
                      call.instruction);
      }
      if (call.kind != expected_kind) {
        AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kCollectiveKindMismatch,
                      "collective calls disagree with its specification kind",
                      call.instruction);
      }
      if (expected_kind == CollectiveKind::kWarp &&
          call.lane_mask != spec->lane_mask) {
        AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kCollectiveMaskMismatch,
                      "syncwarp call disagrees with its declared lane mask",
                      call.instruction);
      }
    }
    if (called != declared) {
      AddDiagnostic(
          diagnostics, DiagnosticClass::kInvalidInput,
          DiagnosticCode::kCollectiveParticipantMismatch,
          "collective callers do not equal its declared participants");
    }

    std::optional<CtaId> collective_cta;
    std::optional<WarpId> collective_warp;
    const auto inspectActor = [&](ActorId actor_id,
                                  const SourceInstruction* instruction) {
      const auto actor = actors.find(actor_id);
      if (actor == actors.end()) {
        return;
      }
      if (!collective_cta.has_value()) {
        collective_cta = actor->second->cta;
      } else if (expected_kind == CollectiveKind::kWarp &&
                 *collective_cta != actor->second->cta) {
        AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kCollectiveCtaMismatch,
                      "collective participants belong to different CTAs",
                      instruction);
      }
      if (expected_kind == CollectiveKind::kWarp) {
        if (!collective_warp.has_value()) {
          collective_warp = actor->second->warp;
        } else if (*collective_warp != actor->second->warp) {
          AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                        DiagnosticCode::kCollectiveWarpMismatch,
                        "syncwarp participants belong to different warps",
                        instruction);
        }
        if (actor->second->lane < 32 && spec->lane_mask.has_value() &&
            ((*spec->lane_mask & (std::uint32_t{1} << actor->second->lane)) ==
             0)) {
          AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                        DiagnosticCode::kCollectiveMaskMismatch,
                        "syncwarp participant is absent from its lane mask",
                        instruction);
        }
      }
    };

    for (ActorId participant : declared) {
      inspectActor(participant, nullptr);
    }
    for (const CollectiveCall& call : calls) {
      const auto instruction=*call.instruction;
      inspectActor(instruction.actor, &instruction);
    }
  }

  // Consecutive calls suffice: all other per-actor constraints are transitive.
  std::map<SyncId, std::size_t> collective_pending;
  std::map<SyncId, std::vector<SyncId>> collective_successors;
  std::set<std::pair<SyncId, SyncId>> collective_edges;
  for (const auto& [sync, calls] : collective_calls) {
    static_cast<void>(calls);
    collective_pending.emplace(sync, 0);
  }
  for (const auto& [previous, current] : collective_order) {
    if (collective_edges.emplace(previous, current).second) {
      collective_successors[previous].push_back(current);
      ++collective_pending.at(current);
    }
  }
  std::vector<SyncId> collective_ready;
  for (const auto& [sync, pending] : collective_pending) {
    if (pending == 0) collective_ready.push_back(sync);
  }
  for (std::size_t index = 0; index < collective_ready.size(); ++index) {
    for (SyncId successor : collective_successors[collective_ready[index]]) {
      if (--collective_pending.at(successor) == 0)
        collective_ready.push_back(successor);
    }
  }
  if (collective_ready.size() != collective_pending.size()) {
    AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                  DiagnosticCode::kIncompatibleCollectiveOrder,
                  "collective calls contain incompatible per-actor orders");
  }

  return diagnostics;
}

}  // namespace sentryward::algorithm
