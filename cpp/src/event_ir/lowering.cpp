#include "sentryward/algorithm/event_ir/lowering.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "sentryward/algorithm/source_ir/operation.h"
#include "sentryward/algorithm/source_ir/validator.h"

namespace sentryward::algorithm {

namespace {

template <typename... Visitors>
struct Overloaded : Visitors... {
  using Visitors::operator()...;
};

template <typename... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

struct LoweredInstruction final {
  Milestones milestones;
  bool blocking;
};

struct CollectiveCall final {
  InstId instruction;
  ActorId actor;
  CollectiveKind kind;
  Milestones milestones;
};

Diagnostic unsupportedTryWait(const SourceInstruction& instruction) {
  return Diagnostic{DiagnosticClass::kUnsupported,
                    DiagnosticCode::kRawTryWaitUnsupported,
                    "raw try_wait must be canonicalized before lowering",
                    instruction.id, instruction.location, {}};
}

EventId addNormalEvent(EventDagBuilder& graph,
                       const SourceInstruction& instruction,
                       EventPayload payload) {
  return graph.addEvent(EventSchedulingClass::kNormal, instruction.id,
                        instruction.actor, std::move(payload));
}

void addRequiredEdge(
    EventDagBuilder& graph, EventId from, EventId to, EventEdgeReason reason,
    std::optional<SourceEdgeProvenance> source_edge = std::nullopt) {
  if (!graph.addEdge(from, to, reason, std::move(source_edge))) {
    throw std::logic_error("lowering edge references a missing event");
  }
}

std::optional<CompleteTxEffect> lowerCompletion(
    const std::optional<MBarrierCompletion>& completion) {
  if (!completion.has_value()) {
    return std::nullopt;
  }
  return CompleteTxEffect{completion->barrier, completion->bytes};
}

// The key is the whole record, so equivalent edges are equal values; Lowerer::Run sorts
// edge values on that basis. The binding stops compiling if SourceEdge gains a field.
bool lessSourceEdge(const SourceEdge& left, const SourceEdge& right) {
  const auto& [from, to, kind] = left;
  if (from != right.from) {
    return from < right.from;
  }
  if (to != right.to) {
    return to < right.to;
  }
  return static_cast<int>(kind) < static_cast<int>(right.kind);
}

bool lessCollectiveSpec(const CollectiveSpec* left,
                        const CollectiveSpec* right) {
  return left->sync_id < right->sync_id;
}

bool lessDiagnostic(const Diagnostic& left, const Diagnostic& right) {
  if (left.classification != right.classification) {
    return static_cast<int>(left.classification) <
           static_cast<int>(right.classification);
  }
  if (left.code != right.code) {
    return static_cast<int>(left.code) < static_cast<int>(right.code);
  }
  if (left.instruction.has_value() != right.instruction.has_value()) {
    return left.instruction.has_value();
  }
  return left.instruction.has_value() &&
         left.instruction->value() < right.instruction->value();
}

}  // namespace

Result<LoweredProgram, Diagnostic> Lowerer::Run(
    const SourceProgram& source, bool synchronization_only,
    CollectiveOrdering ordering) const {
  const std::vector<Diagnostic> diagnostics =
      SourceValidator().Validate(source);
  if (!diagnostics.empty()) {
    return Result<LoweredProgram, Diagnostic>::Failure(*std::min_element(
        diagnostics.begin(), diagnostics.end(), lessDiagnostic));
  }

  std::vector<std::size_t> reordered_instructions;
  bool sorted=true;
  std::optional<InstId> previous_id;
  for(std::size_t i=0;i<source.instructions.size();++i) {
    const auto id=source.instructions.id(i);
    if(previous_id && id<*previous_id){sorted=false;break;}
    previous_id=id;
  }
  if(!sorted) {
    reordered_instructions.reserve(source.instructions.size());
    for(std::size_t i=0;i<source.instructions.size();++i)reordered_instructions.push_back(i);
    std::sort(reordered_instructions.begin(),reordered_instructions.end(),[&](auto a,auto b) {
      return source.instructions.id(a)<source.instructions.id(b);
    });
  }
  auto instructions=std::views::iota(std::size_t{0},source.instructions.size()) |
      std::views::transform([&](std::size_t i){return reordered_instructions.empty()?i:reordered_instructions[i];});

  EventDagBuilder graph;
  // Project at construction; event IDs and all dependency edges are unchanged.
  const auto emit = [&](EventDagBuilder& builder, const SourceInstruction& instruction,
                        EventPayload payload) {
    if (synchronization_only) {
      payload = std::visit(Overloaded{
          [](const ImmediateMemoryIssueEvent&) -> EventPayload { return OperationIssueEvent{}; },
          [](const ImmediateMemoryExecEvent&) -> EventPayload { return OperationCompleteEvent{}; },
          [](const AsyncMemoryIssueEvent& op) -> EventPayload {
            return GenericAsyncIssueEvent{op.async_id, {}};
          },
          [](const AsyncMemoryExecEvent& op) -> EventPayload {
            std::optional<CompletionAction> completion;
            if (op.completion) completion = *op.completion;
            return GenericAsyncEffectEvent{op.async_id, 0, {}, completion, true, false, 2, true};
          },
          [](GenericAsyncEffectEvent op) -> EventPayload {
            op.detached_access_count = op.accesses.size();
            op.accesses = {}; op.detached_accesses = true; return op;
          },
          [](auto op) -> EventPayload { return op; }
      }, std::move(payload));
    }
    return addNormalEvent(builder, instruction, std::move(payload));
  };
  graph.reserve(std::min<std::size_t>(256,source.instructions.storedBindingCount()),
                std::min<std::size_t>(256,source.edges.storedBindingCount()));
  OriginMap origins;
  StateDefMap state_defs;
  BarrierInitializationTable barrier_initializations;
  CollectiveTable collectives;
  std::set<InstId> nonblocking_instructions;
  std::map<SyncId, std::vector<CollectiveCall>> collective_calls;

  std::optional<ActorId> previous_actor;
  std::optional<std::pair<std::uint64_t,std::uintptr_t>> previous_cohort;
  std::size_t event_begin=0;
  const auto finish_thread=[&] {
    graph.endEdgeGroup();
    if(previous_cohort)graph.compressThreadTail(event_begin,previous_cohort->first,previous_cohort->second);
  };
  // Later passes visit only what they act on: the thread transitions (by step), the
  // requirement holders (step, position) and the state waits (position).
  struct ThreadTransition final {
    std::size_t step;
    std::optional<std::pair<std::uint64_t,std::uintptr_t>> cohort;
  };
  std::vector<ThreadTransition> transitions;
  std::vector<std::pair<std::size_t,std::size_t>> requirement_holders;
  std::vector<std::size_t> wait_positions;
  std::size_t next_step=0;
  for (auto position : instructions) {
    const auto step=next_step++;
    const auto instruction_view=source.instructions.view(position); const auto& snapshot=*instruction_view;
    const auto* instruction=&snapshot;
    auto cohort=source.instructions.threadCohort(position);
    // The requirement pass replays these transitions; it has no transition rule of its own.
    if(previous_actor!=instruction->actor || previous_cohort!=cohort) {
      finish_thread();
      event_begin=graph.beginThread();
      previous_actor=instruction->actor;previous_cohort=cohort;
      if(cohort)graph.beginEdgeGroup(cohort->first,cohort->second);
      transitions.push_back({step,cohort});
    }
    std::optional<Diagnostic> lowering_error;
    const LoweredInstruction lowered = std::visit(
        Overloaded{
            [&](const ImmediateLoad& op) {
              const EventId issue = emit(
                  graph, *instruction,
                  ImmediateMemoryIssueEvent{MemoryDirection::kLoad, op.address,
                                            op.bytes});
              const EventId done = emit(
                  graph, *instruction,
                  ImmediateMemoryExecEvent{MemoryDirection::kLoad, op.address,
                                           op.bytes});
              addRequiredEdge(graph, issue, done,
                              EventEdgeReason::kInternalIssueToDone);
              return LoweredInstruction{{issue, done}, true};
            },
            [&](const ImmediateStore& op) {
              const EventId issue = emit(
                  graph, *instruction,
                  ImmediateMemoryIssueEvent{MemoryDirection::kStore, op.address,
                                            op.bytes});
              const EventId done = emit(
                  graph, *instruction,
                  ImmediateMemoryExecEvent{MemoryDirection::kStore, op.address,
                                           op.bytes});
              addRequiredEdge(graph, issue, done,
                              EventEdgeReason::kInternalIssueToDone);
              return LoweredInstruction{{issue, done}, true};
            },
            [&](const AsyncLoad& op) {
              const EventId issue = emit(
                  graph, *instruction,
                  AsyncMemoryIssueEvent{MemoryDirection::kLoad, op.async_id,
                                        op.source, op.destination, op.bytes});
              const EventId done = emit(
                  graph, *instruction,
                  AsyncMemoryExecEvent{MemoryDirection::kLoad, op.async_id,
                                       op.source, op.destination, op.bytes,
                                       lowerCompletion(op.completion)});
              addRequiredEdge(graph, issue, done,
                              EventEdgeReason::kInternalIssueToDone);
              return LoweredInstruction{{issue, done}, false};
            },
            [&](const AsyncStore& op) {
              const EventId issue = emit(
                  graph, *instruction,
                  AsyncMemoryIssueEvent{MemoryDirection::kStore, op.async_id,
                                        op.source, op.destination, op.bytes});
              const EventId done = emit(
                  graph, *instruction,
                  AsyncMemoryExecEvent{MemoryDirection::kStore, op.async_id,
                                       op.source, op.destination, op.bytes,
                                       lowerCompletion(op.completion)});
              addRequiredEdge(graph, issue, done,
                              EventEdgeReason::kInternalIssueToDone);
              return LoweredInstruction{{issue, done}, false};
            },
            [&](const AsyncOp& op) {
              std::vector<BarrierLifetimeId> barriers;
              for (const AsyncEffect& effect : op.effects) {
                if (effect.completion) {
                  barriers.push_back(std::visit(
                      [](const auto& action) { return action.barrier; },
                      *effect.completion));
                }
              }
              std::sort(barriers.begin(), barriers.end());
              barriers.erase(std::unique(barriers.begin(), barriers.end()),
                             barriers.end());
              const EventId issue = emit(
                  graph, *instruction,
                  GenericAsyncIssueEvent{op.async_id, std::move(barriers)});
              const EventId reads =
                  emit(graph, *instruction,
                                 GenericAsyncReadsCompleteEvent{op.async_id});
              const EventId done = emit(
                  graph, *instruction, GenericAsyncCompleteEvent{op.async_id});
              Milestones milestones{issue, done, reads, {}};
              milestones.effects.reserve(op.effects.size());
              addRequiredEdge(graph, issue, reads,
                              EventEdgeReason::kAsyncAggregate);
              addRequiredEdge(graph, reads, done,
                              EventEdgeReason::kAsyncAggregate);
              for (std::size_t index = 0; index < op.effects.size(); ++index) {
                const AsyncEffect& effect = op.effects[index];
                const EventId execution = emit(
                    graph, *instruction,
                    GenericAsyncEffectEvent{
                        op.async_id, static_cast<std::uint32_t>(index),
                        effect.accesses, effect.completion});
                milestones.effects.push_back(execution);
              }
              const auto has_read=[&](std::size_t i) {
                const auto& accesses=op.effects[i].accesses;
                return std::any_of(accesses.begin(),accesses.end(),[](const MemoryEffect& access) {
                  return access.mode==AccessMode::kRead;
                });
              };
              for(std::size_t begin=0;begin<op.effects.size();) {
                const bool read=has_read(begin);
                auto end=begin+1;
                while(end<op.effects.size() && has_read(end)==read)++end;
                const auto first=milestones.effects[begin];
                std::vector<EventEdge> columns{
                    {issue,first,EventEdgeReason::kInternalIssueToDone,{}},
                    {first,done,EventEdgeReason::kAsyncAggregate,{}}};
                std::vector<EventEdgeCodec::Delta> deltas{{0,1,0,0},{1,0,0,0}};
                if(read) {
                  columns.push_back({first,reads,EventEdgeReason::kAsyncAggregate,{}});
                  deltas.push_back({1,0,0,0});
                }
                if(!graph.addEdgeColumns(columns,deltas,end-begin))
                  throw std::logic_error("invalid async effect edge family");
                begin=end;
              }
              return LoweredInstruction{std::move(milestones), false};
            },
            [&](const Await&) {
              const EventId issue =
                  emit(graph, *instruction, AwaitIssueEvent{});
              const EventId done =
                  emit(graph, *instruction, AwaitDoneEvent{});
              addRequiredEdge(graph, issue, done,
                              EventEdgeReason::kInternalIssueToDone);
              return LoweredInstruction{{issue, done}, true};
            },
            [&](const NamedBarrierArrive& op) {
              const auto arrive = emit(graph, *instruction,
                  NamedBarrierArriveEvent{op.barrier, op.expected_arrivals, false});
              return LoweredInstruction{{arrive, arrive}, true};
            },
            [&](const NamedBarrierSync& op) {
              const auto arrive = emit(graph, *instruction,
                  NamedBarrierArriveEvent{op.barrier, op.expected_arrivals, true});
              const auto done = emit(graph, *instruction,
                  NamedBarrierReturnEvent{op.barrier});
              addRequiredEdge(graph, arrive, done, EventEdgeReason::kInternalIssueToDone);
              return LoweredInstruction{{arrive, done}, true};
            },
            [&](const MBarrierInit& op) {
              const EventId event = emit(
                  graph, *instruction,
                  MBarrierInitEvent{op.barrier, op.expected_arrivals});
              if (!barrier_initializations.record(
                      op.barrier,
                      BarrierInitialization{event, op.expected_arrivals})) {
                throw std::logic_error(
                    "validated source has duplicate barrier initialization");
              }
              return LoweredInstruction{{event, event}, true};
            },
            [&](const MBarrierArrive& op) {
              const EventId event = emit(
                  graph, *instruction,
                  MBarrierUpdateEvent{op.barrier, op.arrivals, 0, op.result});
              if (op.result.has_value() &&
                  !state_defs.define(*op.result, event)) {
                throw std::logic_error(
                    "validated source has duplicate state definition");
              }
              return LoweredInstruction{{event, event}, true};
            },
            [&](const MBarrierExpectTx& op) {
              const EventId event = emit(
                  graph, *instruction,
                  MBarrierUpdateEvent{op.barrier, 0, op.bytes, std::nullopt});
              return LoweredInstruction{{event, event}, true};
            },
            [&](const MBarrierArriveExpectTx& op) {
              const EventId event =
                  emit(graph, *instruction,
                                 MBarrierUpdateEvent{op.barrier, op.arrivals,
                                                     op.bytes, op.result});
              if (op.result.has_value() &&
                  !state_defs.define(*op.result, event)) {
                throw std::logic_error(
                    "validated source has duplicate state definition");
              }
              return LoweredInstruction{{event, event}, true};
            },
            [&](const MBarrierCompleteTx& op) {
              const EventId event =
                  emit(graph, *instruction,
                                 MBarrierCompleteTxEvent{op.barrier, op.bytes});
              return LoweredInstruction{{event, event}, true};
            },
            [&](const MBarrierAddArrivals& op) {
              const EventId event = emit(
                  graph, *instruction,
                  MBarrierAddArrivalsEvent{op.barrier, op.arrivals});
              return LoweredInstruction{{event, event}, true};
            },
            [&](const MBarrierWaitState& op) {
              const EventId issue =
                  emit(graph, *instruction,
                                 WaitStateIssueEvent{op.barrier, op.state});
              const EventId done =
                  emit(graph, *instruction,
                                 WaitStateDoneEvent{op.barrier, op.state});
              addRequiredEdge(graph, issue, done,
                              EventEdgeReason::kInternalIssueToDone);
              return LoweredInstruction{{issue, done}, true};
            },
            [&](const MBarrierWaitParity& op) {
              const EventId issue =
                  emit(graph, *instruction,
                                 WaitParityIssueEvent{op.barrier, op.parity});
              const EventId done =
                  emit(graph, *instruction,
                                 WaitParityDoneEvent{op.barrier, op.parity});
              addRequiredEdge(graph, issue, done,
                              EventEdgeReason::kInternalIssueToDone);
              return LoweredInstruction{{issue, done}, true};
            },
            [&](const MBarrierTryWaitState&) {
              lowering_error = unsupportedTryWait(*instruction);
              return LoweredInstruction{{EventId(), EventId()}, true};
            },
            [&](const MBarrierTryWaitParity&) {
              lowering_error = unsupportedTryWait(*instruction);
              return LoweredInstruction{{EventId(), EventId()}, true};
            },
            [&](const SyncThreads& op) {
              const EventId issue =
                  emit(graph, *instruction,
                                 CollectiveIssueEvent{
                                     op.sync_id, CollectiveKind::kSyncThreads});
              const EventId done =
                  emit(graph, *instruction,
                                 CollectiveDoneEvent{
                                     op.sync_id, CollectiveKind::kSyncThreads});
              collective_calls[op.sync_id].push_back(
                  CollectiveCall{instruction->id,
                                 instruction->actor,
                                 CollectiveKind::kSyncThreads,
                                 {issue, done}});
              return LoweredInstruction{{issue, done}, true};
            },
            [&](const SyncWarp& op) {
              const auto kind = op.memory_ordering ? CollectiveKind::kSyncWarp : CollectiveKind::kWarpData;
              const EventId issue = emit(
                  graph, *instruction,
                  CollectiveIssueEvent{op.sync_id, kind});
              const EventId done = emit(
                  graph, *instruction,
                  CollectiveDoneEvent{op.sync_id, kind});
              if (!op.memory_ordering)
                addRequiredEdge(graph, issue, done, EventEdgeReason::kInternalIssueToDone);
              collective_calls[op.sync_id].push_back(
                  CollectiveCall{instruction->id,
                                 instruction->actor,
                                 kind,
                                 {issue, done}});
              return LoweredInstruction{{issue, done}, true};
            }},
        instruction->op);

    if (lowering_error.has_value()) {
      return Result<LoweredProgram, Diagnostic>::Failure(
          std::move(*lowering_error));
    }
    if (!origins.record(instruction->id, lowered.milestones)) {
      throw std::logic_error("validated source produced duplicate origin map");
    }
    // IDs ascend in this pass, so every insertion appends.
    if (!lowered.blocking && lowered.milestones.issue != lowered.milestones.done)
      nonblocking_instructions.emplace_hint(nonblocking_instructions.end(), instruction->id);
    if (std::holds_alternative<AsyncOp>(instruction->op) || std::holds_alternative<Await>(instruction->op))
      requirement_holders.emplace_back(step, position);
    else if (std::holds_alternative<MBarrierWaitState>(instruction->op))
      wait_positions.push_back(position);
  }
  finish_thread();

  const auto resolve = [&](const MilestoneRef& ref) -> EventId {
    // Scalar lookup first; on a miss, the path below raises the direct lookup's exceptions.
    if (const auto event = origins.resolve(ref)) return *event;
    const Milestones milestones = *origins.findMilestones(ref.instruction);
    switch (ref.milestone) {
      case Milestone::kIssue:
        return milestones.issue;
      case Milestone::kComplete:
        return milestones.done;
      case Milestone::kReadsComplete:
        return milestones.reads_complete.value();
      case Milestone::kEffect:
        return milestones.effects.at(ref.effect_index);
    }
    throw std::logic_error("validated source has an invalid milestone");
  };
  // The lowering loop's thread transitions, replayed in step order around the requirement
  // holders: the graph sees the edge groups in the lowering loop's sequence.
  std::size_t transition=0;
  const auto replay_transitions=[&](std::size_t until) {
    for(;transition<transitions.size() && transitions[transition].step<=until;++transition) {
      graph.endEdgeGroup();
      if(const auto& cohort=transitions[transition].cohort)graph.beginEdgeGroup(cohort->first,cohort->second);
    }
  };
  for (const auto& [step, position] : requirement_holders) {
    replay_transitions(step);
    const auto instruction_view=source.instructions.view(position); const auto& snapshot=*instruction_view;
    const auto* instruction=&snapshot;
    const Milestones milestones = *origins.findMilestones(instruction->id);
    const auto addRequirements = [&](const std::vector<MilestoneRef>& refs,
                                     EventId target, EventEdgeReason reason) {
      for (const MilestoneRef& ref : refs) {
        addRequiredEdge(graph, resolve(ref), target, reason,
                        SourceEdgeProvenance{ref.instruction, instruction->id});
      }
    };
    if (const auto* async = std::get_if<AsyncOp>(&instruction->op)) {
      addRequirements(async->completion_requires, milestones.done,
                      EventEdgeReason::kAsyncAggregate);
      for (std::size_t index = 0; index < async->effects.size(); ++index) {
        addRequirements(async->effects[index].predecessors,
                        milestones.effects[index],
                        EventEdgeReason::kAsyncEffectDependency);
      }
    } else if (const auto* wait = std::get_if<Await>(&instruction->op)) {
      addRequirements(wait->requirements, milestones.done,
                      EventEdgeReason::kAwaitRequirement);
    }
  }
  replay_transitions(std::numeric_limits<std::size_t>::max());
  graph.endEdgeGroup();
  // Release the lowering loop's tables now (assigning {} would keep their capacity).
  decltype(transitions)().swap(transitions);decltype(requirement_holders)().swap(requirement_holders);
  for (const EndpointDependency& dependency : source.dependencies) {
    addRequiredEdge(graph, resolve(dependency.from), resolve(dependency.to),
                    EventEdgeReason::kEndpointDependency,
                    SourceEdgeProvenance{dependency.from.instruction,
                                         dependency.to.instruction});
  }

  const auto lower_source_edge=[&](const SourceEdge& edge) {
    // Endpoint events only, looked up source, then target, then the kind check.
    const auto source_milestone=(edge.kind==SourceEdgeKind::kProgramOrder &&
        nonblocking_instructions.contains(edge.from)) ? Milestone::kIssue : Milestone::kComplete;
    const EventId source_event = *origins.resolve({edge.from, source_milestone});
    const EventId target_event = *origins.resolve({edge.to, Milestone::kIssue});
    EventEdgeReason reason;
    switch (edge.kind) {
      case SourceEdgeKind::kProgramOrder:
        reason = EventEdgeReason::kSourceProgramOrder;
        break;
      case SourceEdgeKind::kCompleteBeforeIssue:
        reason = EventEdgeReason::kSourceCompletionDependency;
        break;
      default:
        throw std::logic_error("invalid source edge kind during lowering");
    }
    addRequiredEdge(graph, source_event, target_event, reason,
                    SourceEdgeProvenance{edge.from, edge.to});
  };
  {
    // Equivalent edges are equal values, so the sorted sequence is unique: one family pass
    // fills the logical edges and the sort compares values, not two materialized records.
    // The block frees the copy before the graph is frozen.
    std::vector<SourceEdge> ordered_edges(source.edges.size());
    std::size_t covered=0;
    source.edges.forEachAffineColumn([&](const SourceEdge& value,const SourceEdgeCodec::Delta& delta,
        std::size_t first,std::size_t stride,std::size_t count) {
      auto edge=value;SourceEdgeCodec::Binding base{};SourceEdgeCodec::split(edge,base);
      ordered_edges[first]=value;
      for(std::size_t member=1;member<count;++member) {
        const auto binding=SourceEdgeCodec::translate(base,delta,member);
        if(!binding)throw std::logic_error("invalid compressed record binding");
        SourceEdgeCodec::bind(edge,*binding);ordered_edges[first+stride*member]=edge;
      }
      covered+=count;
    });
    if(covered!=ordered_edges.size())throw std::logic_error("source edge families do not cover every edge");
    if(!std::is_sorted(ordered_edges.begin(),ordered_edges.end(),lessSourceEdge))
      std::sort(ordered_edges.begin(),ordered_edges.end(),lessSourceEdge);
    for(const auto& edge:ordered_edges)lower_source_edge(edge);
  }

  for (const auto position : wait_positions) {
    const auto instruction_view=source.instructions.view(position); const auto& snapshot=*instruction_view;
    const auto* instruction=&snapshot;
    const auto* wait = std::get_if<MBarrierWaitState>(&instruction->op);
    if (wait == nullptr) {
      continue;
    }
    const EventId definition = *state_defs.definition(wait->state);
    const EventId issue = origins.milestones(instruction->id)->issue;
    addRequiredEdge(graph, definition, issue, EventEdgeReason::kStateDefUse);
  }
  decltype(wait_positions)().swap(wait_positions);

  std::vector<const CollectiveSpec*> collective_specs;
  collective_specs.reserve(source.collectives.size());
  for (const CollectiveSpec& spec : source.collectives) {
    collective_specs.push_back(&spec);
  }
  std::sort(collective_specs.begin(), collective_specs.end(),
            lessCollectiveSpec);

  for (const CollectiveSpec* spec : collective_specs) {
    const auto found_calls = collective_calls.find(spec->sync_id);
    if (found_calls == collective_calls.end() || found_calls->second.empty()) {
      return Result<LoweredProgram, Diagnostic>::Failure(Diagnostic{
          DiagnosticClass::kInvalidInput,
          DiagnosticCode::kCollectiveParticipantMismatch,
          "collective has no participant calls", std::nullopt, std::nullopt, {}});
    }
    std::vector<CollectiveCall>& calls = found_calls->second;
    std::sort(calls.begin(), calls.end(),
              [](const CollectiveCall& left, const CollectiveCall& right) {
                if (left.actor != right.actor) {
                  return left.actor < right.actor;
                }
                return left.instruction < right.instruction;
              });
    const CollectiveKind kind = calls.front().kind;
    if (std::any_of(calls.begin(), calls.end(), [&](const auto& call) { return call.kind != kind; }))
      return Result<LoweredProgram, Diagnostic>::Failure(Diagnostic{
          DiagnosticClass::kInvalidInput, DiagnosticCode::kCollectiveParticipantMismatch,
          "collective participants disagree on memory-ordering semantics", std::nullopt, std::nullopt, {}});
    const EventId completion = graph.addEvent(
        EventSchedulingClass::kUrgentDerived, std::nullopt, std::nullopt,
        CollectiveCompleteEvent{spec->sync_id, kind});

    std::vector<ActorId> expected = spec->expected_participants;
    std::sort(expected.begin(), expected.end());
    std::vector<CollectiveParticipant> participants;
    participants.reserve(calls.size());
    for (const CollectiveCall& call : calls) {
      addRequiredEdge(graph, call.milestones.issue, completion,
                      EventEdgeReason::kCollectiveArrival);
      // A warp data collective still waits in CollectiveRuntime. Its return
      // is execution-gated, not a memory-order release. Adding this edge for
      // shfl/elect/vote/match/redux would publish unrelated preceding stores,
      // so only the execution-order relation carries it.
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-shfl-sync
      if (kind != CollectiveKind::kWarpData ||
          ordering == CollectiveOrdering::kExecution)
        addRequiredEdge(graph, completion, call.milestones.done,
                        EventEdgeReason::kCollectiveRelease);
      participants.push_back(
          CollectiveParticipant{call.actor, call.instruction, call.milestones});
    }

    if (!collectives.record(LoweredCollective{
            spec->sync_id, kind, completion, std::move(expected),
            spec->lane_mask, std::move(participants)})) {
      throw std::logic_error("validated source has duplicate collective spec");
    }
  }

  EventDag lowered_graph = std::move(graph).freeze();
  origins.reserveEvents(lowered_graph.eventCount());
  if (!lowered_graph.isAcyclic()) {
    return Result<LoweredProgram, Diagnostic>::Failure(Diagnostic{
        DiagnosticClass::kInvalidInput, DiagnosticCode::kCyclicSourceGraph,
        "source lowering produced a cyclic event graph", std::nullopt,
        std::nullopt, {}});
  }

  return Result<LoweredProgram, Diagnostic>::Success(LoweredProgram(
      std::move(lowered_graph), std::move(origins), std::move(state_defs),
      std::move(barrier_initializations), std::move(collectives), source.actors));
}

}  // namespace sentryward::algorithm
