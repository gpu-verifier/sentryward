#include "sentryward/algorithm/source_ir/validator.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
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
                   std::string message,
                   const SourceInstruction* instruction = nullptr) {
  diagnostics.push_back(Diagnostic{
      classification,
      code,
      std::move(message),
      instruction == nullptr ? std::nullopt
                             : std::optional<InstId>(instruction->id),
      instruction == nullptr
          ? std::nullopt
          : std::optional<SourceLocation>(instruction->location),
  });
}

template <typename Id>
std::string IdValue(Id id) {
  return std::to_string(id.value());
}

void AddDiagnostic(std::vector<Diagnostic>& out, DiagnosticClass classification,
    DiagnosticCode code, std::string message, SourceInstructions::Handle ref) {
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

  std::map<InstId, SourceInstructions::Handle> instructions;
  for (std::size_t i=0;i<program.instructions.size();++i) {
    const auto id = program.instructions.id(i);
    const auto actor = program.instructions.actor(i);
    if (!instructions.emplace(id, program.instructions.ref(i)).second) {
      const auto instruction_view = program.instructions.view(i);
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kDuplicateInstructionId,
                    "instruction " + IdValue(id) +
                        " is declared more than once",
                    &*instruction_view);
    }
    if (!actors.contains(actor)) {
      const auto instruction_view = program.instructions.view(i);
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kUnknownInstructionActor,
                    "instruction " + IdValue(id) +
                        " references unknown actor " +
                        IdValue(actor),
                    &*instruction_view);
    }
  }

  std::map<AsyncOpId, const SourceInstruction*> async_operations;
  std::map<BarrierLifetimeId, SourceInstructions::Handle> barrier_initializers;
  std::map<BarrierLifetimeId, SourceInstructions::Handle> barrier_uses;
  std::set<BarrierLifetimeId> named_barriers;
  std::map<StateId, StateDefinition> state_definitions;
  std::vector<StateUse> state_uses;
  std::vector<TxContribution> tx_contributions;
  std::map<SyncId, std::vector<CollectiveCall>> collective_calls;

  const auto recordBarrierUse = [&](BarrierLifetimeId barrier,
                                    const SourceInstruction& instruction) {
    barrier_uses.try_emplace(barrier, instructions.at(instruction.id));
  };
  const auto recordStateDefinition = [&](BarrierLifetimeId barrier,
                                         StateId state,
                                         const SourceInstruction& instruction) {
    if (!state_definitions
             .emplace(state, StateDefinition{barrier, instructions.at(instruction.id)})
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
    tx_contributions.push_back(TxContribution{completion.bytes, instructions.at(instruction.id)});
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
    const auto found = instructions.find(ref.instruction);
    bool valid = found != instructions.end();
    if (valid) {
      switch (ref.milestone) {
        case Milestone::kIssue:
        case Milestone::kComplete:
          valid = ref.effect_index == 0;
          break;
        case Milestone::kReadsComplete:
          valid = found->second.holdsOperation<AsyncOp>() && ref.effect_index == 0;
          break;
        case Milestone::kEffect:
          valid = found->second.holdsOperation<AsyncOp>() &&
                  ref.effect_index < found->second.effectCount();
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
                                {completion.bytes, instructions.at(instruction.id)});
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
              if (!barrier_initializers.emplace(operation.barrier, instructions.at(instruction.id))
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
                  TxContribution{operation.bytes, instructions.at(instruction.id)});
            },
            [&](const MBarrierArriveExpectTx& operation) {
              recordBarrierUse(operation.barrier, instruction);
              tx_contributions.push_back(
                  TxContribution{operation.bytes, instructions.at(instruction.id)});
              if (operation.result.has_value()) {
                recordStateDefinition(operation.barrier, *operation.result,
                                      instruction);
              }
            },
            [&](const MBarrierCompleteTx& operation) {
              recordBarrierUse(operation.barrier, instruction);
              tx_contributions.push_back(
                  TxContribution{operation.bytes, instructions.at(instruction.id)});
            },
            [&](const MBarrierAddArrivals& operation) {
              recordBarrierUse(operation.barrier, instruction);
            },
            [&](const MBarrierWaitState& operation) {
              recordBarrierUse(operation.barrier, instruction);
              state_uses.push_back(
                  StateUse{operation.barrier, operation.state, instructions.at(instruction.id)});
            },
            [&](const MBarrierWaitParity& operation) {
              recordBarrierUse(operation.barrier, instruction);
              validateParity(operation.parity, instruction);
            },
            [&](const MBarrierTryWaitState& operation) {
              recordBarrierUse(operation.barrier, instruction);
              state_uses.push_back(
                  StateUse{operation.barrier, operation.state, instructions.at(instruction.id)});
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
                  instructions.at(instruction.id), CollectiveKind::kThreads, std::nullopt});
            },
            [&](const SyncWarp& operation) {
              collective_calls[operation.sync_id].push_back(CollectiveCall{
                  instructions.at(instruction.id), CollectiveKind::kWarp, operation.lane_mask});
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

  std::set<std::tuple<InstId, InstId, int>> exact_edges;
  std::set<std::pair<InstId, InstId>> graph_edges;
  std::map<InstId, std::vector<InstId>> successors;
  std::map<InstId, std::vector<InstId>> program_order_successors;
  std::map<InstId, std::size_t> indegrees;
  for (const auto& [id, instruction] : instructions) {
    static_cast<void>(instruction);
    indegrees.emplace(id, 0);
    program_order_successors.emplace(id, std::vector<InstId>{});
  }

  for (const SourceEdge& edge : program.edges) {
    if (!IsValidEdgeKind(edge.kind)) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kInvalidEdgeKind,
                    "source edge has an unknown dependency kind");
      continue;
    }

    const auto from = instructions.find(edge.from);
    const auto to = instructions.find(edge.to);
    if (from == instructions.end() || to == instructions.end()) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kUnknownEdgeEndpoint,
                    "source edge references an unknown instruction");
      continue;
    }

    const auto exact =
        std::tuple(edge.from, edge.to, static_cast<int>(edge.kind));
    if (!exact_edges.insert(exact).second) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kDuplicateEdge,
                    "source edge is declared more than once", to->second);
      continue;
    }

    if (edge.kind == SourceEdgeKind::kProgramOrder &&
        from->second.actor() != to->second.actor()) {
      AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                    DiagnosticCode::kProgramOrderCrossesActors,
                    "program-order edge connects different actors", to->second);
    } else if (edge.kind == SourceEdgeKind::kProgramOrder) {
      program_order_successors[edge.from].push_back(edge.to);
    }

    if (graph_edges.emplace(edge.from, edge.to).second) {
      successors[edge.from].push_back(edge.to);
      ++indegrees.at(edge.to);
    }
  }

  std::set<InstId> ready;
  for (const auto& [id, indegree] : indegrees) {
    if (indegree == 0) {
      ready.insert(id);
    }
  }
  std::size_t visited = 0;
  while (!ready.empty()) {
    const InstId current = *ready.begin();
    ready.erase(ready.begin());
    ++visited;
    for (InstId successor : successors[current]) {
      std::size_t& indegree = indegrees.at(successor);
      --indegree;
      if (indegree == 0) {
        ready.insert(successor);
      }
    }
  }
  if (visited != instructions.size()) {
    AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                  DiagnosticCode::kCyclicSourceGraph,
                  "the union of source dependency edges contains a cycle");
  }

  std::map<ActorId, std::vector<InstId>> actor_instructions;
  std::map<ActorId, std::vector<InstId>> actor_order;
  for (const auto& [id, instruction] : instructions) {
    actor_instructions[instruction.actor()].push_back(id);
  }
  for (const auto& [actor, actor_program] : actor_instructions) {
    std::map<InstId, std::size_t> actor_indegrees;
    for (InstId instruction : actor_program) {
      actor_indegrees.emplace(instruction, 0);
    }
    for (InstId instruction : actor_program) {
      for (InstId successor : program_order_successors[instruction]) {
        ++actor_indegrees.at(successor);
      }
    }

    std::set<InstId> actor_ready;
    for (const auto& [instruction, indegree] : actor_indegrees) {
      if (indegree == 0) {
        actor_ready.insert(instruction);
      }
    }
    bool reported_incomparability = false;
    while (!actor_ready.empty()) {
      if (actor_ready.size() > 1 && !reported_incomparability) {
        const InstId first = *actor_ready.begin();
        const InstId second = *std::next(actor_ready.begin());
        AddDiagnostic(diagnostics, DiagnosticClass::kInvalidInput,
                      DiagnosticCode::kIncompleteThreadProgramOrder,
                      "instructions " + IdValue(first) + " and " +
                          IdValue(second) + " of actor " + IdValue(actor) +
                          " are unordered in the source graph",
                      instructions.at(second));
        reported_incomparability = true;
      }

      const InstId current = *actor_ready.begin();
      actor_ready.erase(actor_ready.begin());
      actor_order[actor].push_back(current);
      for (InstId successor : program_order_successors[current]) {
        std::size_t& indegree = actor_indegrees.at(successor);
        --indegree;
        if (indegree == 0) {
          actor_ready.insert(successor);
        }
      }
    }
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
  for (const auto& [actor, order] : actor_order) {
    static_cast<void>(actor);
    std::optional<SyncId> previous;
    for (InstId id : order) {
      const auto handle = instructions.at(id);
      if (!handle.holdsOperation<SyncThreads>() && !handle.holdsOperation<SyncWarp>()) continue;
      const auto instruction_view=instructions.at(id).operator->();
      const auto& instruction=*instruction_view;
      const SourceOp& op = instruction.op;
      std::optional<SyncId> current;
      if (const auto* threads = std::get_if<SyncThreads>(&op))
        current = threads->sync_id;
      else if (const auto* warp = std::get_if<SyncWarp>(&op))
        current = warp->sync_id;
      if (!current) continue;
      if (previous && collective_edges.emplace(*previous, *current).second) {
        collective_successors[*previous].push_back(*current);
        ++collective_pending.at(*current);
      }
      previous = current;
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
