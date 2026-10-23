#include "sentryward/algorithm/memory/access.h"
#include "sentryward/algorithm/memory/family_access.h"
#include "sentryward/algorithm/event_ir/event_families.h"

#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <unordered_map>

namespace sentryward::algorithm {
namespace {

using Extraction = Result<MemoryAccessRecords, MemoryAnalysisError>;

template <typename Operation>
bool matchesAsync(const Operation& op, const AsyncMemoryExecEvent& event) {
  if (op.async_id != event.async_id || op.source != event.source ||
      op.destination != event.destination || op.bytes != event.bytes ||
      op.completion.has_value() != event.completion.has_value()) {
    return false;
  }
  return !op.completion ||
         (op.completion->barrier == event.completion->barrier &&
          op.completion->bytes == event.completion->bytes);
}

bool matchesSource(const SourceOp& op, const ImmediateMemoryExecEvent& event) {
  if (event.direction == MemoryDirection::kLoad) {
    const auto* load = std::get_if<ImmediateLoad>(&op);
    return load && load->address == event.address && load->bytes == event.bytes;
  }
  if (event.direction == MemoryDirection::kStore) {
    const auto* store = std::get_if<ImmediateStore>(&op);
    return store && store->address == event.address &&
           store->bytes == event.bytes;
  }
  return false;
}

bool matchesSource(const SourceOp& op, const AsyncMemoryExecEvent& event) {
  if (event.direction == MemoryDirection::kLoad) {
    const auto* load = std::get_if<AsyncLoad>(&op);
    return load && matchesAsync(*load, event);
  }
  if (event.direction == MemoryDirection::kStore) {
    const auto* store = std::get_if<AsyncStore>(&op);
    return store && matchesAsync(*store, event);
  }
  return false;
}

bool matchesSource(const SourceOp& op, const GenericAsyncEffectEvent& event) {
  const auto* async = std::get_if<AsyncOp>(&op);
  if (!async || async->async_id != event.async_id ||
      event.effect_index >= async->effects.size())
    return false;
  const AsyncEffect& effect = async->effects[event.effect_index];
  return effect.accesses == event.accesses &&
         effect.completion == event.completion;
}

}  // namespace

namespace {
using VisitResult = Result<std::size_t, MemoryAnalysisError>;
using Visitor = std::function<std::optional<MemoryAnalysisError>(const MemoryAccess&)>;
VisitResult VisitEvents(const EventRecords& events,const SourceProgram& source,const Visitor& visitor) {
  const auto failure=[](std::string message,std::optional<EventId> event=std::nullopt) {
    return VisitResult::Failure({std::move(message),event,std::nullopt});
  };
  bool dense=true;
  for(std::size_t i=0;i<source.instructions.size();++i)
    if(source.instructions.id(i).value()!=i) {dense=false;break;}
  std::unordered_map<std::uint32_t,std::size_t> indices;
  if(!dense) {
    indices.reserve(source.instructions.size());
    for(std::size_t i=0;i<source.instructions.size();++i)
      if(!indices.emplace(source.instructions.id(i).value(),i).second) return failure("duplicate source instruction ID");
  }
  std::vector<std::size_t> offsets(source.instructions.size()+1,0);
  for(std::size_t i=0;i<source.instructions.size();++i) offsets[i+1]=offsets[i]+source.instructions.effectCount(i);
  std::vector<std::uint64_t> seen((offsets.back()+63)/64,0);
  std::size_t seen_count=0,access_count=0;
  std::optional<std::size_t> cached_position;
  SourceInstructions::Iterator::Arrow cached_instruction;
  for (std::size_t event_index=0;event_index<events.size();++event_index) {
    const auto event_view=events.view(event_index);
    const Event& event=*event_view;
    const auto* immediate =
        std::get_if<ImmediateMemoryExecEvent>(&event.payload);
    const auto* async = std::get_if<AsyncMemoryExecEvent>(&event.payload);
    const auto* generic = std::get_if<GenericAsyncEffectEvent>(&event.payload);
    const bool order_complete = std::holds_alternative<OperationCompleteEvent>(event.payload);
    if (!immediate && !async && !generic && !order_complete) {
      continue;
    }
    if (!event.origin || !event.actor) {
      return failure("memory execution lacks source provenance", event.id);
    }
    std::size_t position=event.origin->value();
    if(!dense) {
      const auto found=indices.find(event.origin->value());
      position=found==indices.end()?source.instructions.size():found->second;
    }
    if (position >= source.instructions.size()) {
      return failure("memory execution refers to a missing instruction",
                     event.id);
    }
    if (cached_position != position) {
      cached_instruction = source.instructions.view(position);
      cached_position = position;
    }
    const SourceInstruction& instruction=*cached_instruction;
    std::optional<ImmediateMemoryExecEvent> bound_immediate;
    std::optional<AsyncMemoryExecEvent> bound_async;
    std::optional<GenericAsyncEffectEvent> bound_generic;
    if (order_complete) {
      if (const auto* op = std::get_if<ImmediateLoad>(&instruction.op))
        bound_immediate = ImmediateMemoryExecEvent{MemoryDirection::kLoad, op->address, op->bytes};
      else if (const auto* op = std::get_if<ImmediateStore>(&instruction.op))
        bound_immediate = ImmediateMemoryExecEvent{MemoryDirection::kStore, op->address, op->bytes};
      else return failure("operation completion does not match a compute access", event.id);
      immediate = &*bound_immediate;
    }
    if (generic && generic->detached_accesses) {
      if (!generic->accesses.empty()) return failure("synchronization endpoint contains memory accesses", event.id);
      if (const auto* op = std::get_if<AsyncOp>(&instruction.op)) {
        if (op->async_id != generic->async_id || generic->effect_index >= op->effects.size() ||
            op->effects[generic->effect_index].completion != generic->completion ||
            generic->detached_transfer || generic->detached_access_count != op->effects[generic->effect_index].accesses.size())
          return failure("compute effect and synchronization endpoint disagree", event.id);
        bound_generic = *generic;
        bound_generic->accesses = op->effects[generic->effect_index].accesses;
        bound_generic->detached_accesses = false;
        generic = &*bound_generic;
      } else {
        const auto bind = [&](const auto& op, MemoryDirection direction) {
          std::optional<CompleteTxEffect> completion;
          std::optional<CompletionAction> action;
          if (op.completion) {
            completion = CompleteTxEffect{op.completion->barrier, op.completion->bytes};
            action = *completion;
          }
          if (op.async_id != generic->async_id || generic->effect_index != 0 || action != generic->completion ||
              !generic->detached_transfer || generic->detached_access_count != 2)
            return false;
          bound_async = AsyncMemoryExecEvent{direction, op.async_id, op.source, op.destination, op.bytes, completion};
          return true;
        };
        bool matched = false;
        if (const auto* op = std::get_if<AsyncLoad>(&instruction.op)) matched = bind(*op, MemoryDirection::kLoad);
        if (const auto* op = std::get_if<AsyncStore>(&instruction.op)) matched = bind(*op, MemoryDirection::kStore);
        if (!matched) return failure("compute transfer and synchronization endpoint disagree", event.id);
        async = &*bound_async;
        generic = nullptr;
      }
    }
    if (instruction.actor != *event.actor ||
        (immediate && !matchesSource(instruction.op, *immediate)) ||
        (async && !matchesSource(instruction.op, *async)) ||
        (generic && !matchesSource(instruction.op, *generic))) {
      return failure("source and memory execution metadata disagree", event.id);
    }
    const std::uint32_t effect_index = generic ? generic->effect_index : 0;
    if(effect_index>=offsets[position+1]-offsets[position]) return failure("source effect index is out of range",event.id);
    const auto bit=offsets[position]+effect_index;
    const auto mask=std::uint64_t{1}<<(bit%64);
    if (seen[bit/64]&mask) {
      return failure("source effect has duplicate execution events", event.id);
    }
    seen[bit/64]|=mask;
    ++seen_count;
    const std::size_t operands =
        generic ? generic->accesses.size() : (async ? 2 : 1);
    const auto max_id = std::numeric_limits<std::uint32_t>::max();
    if (operands > 0 &&
        (access_count > max_id || operands - 1 > max_id - access_count)) {
      return failure("memory access ID range exhausted", event.id);
    }
    std::optional<MemoryAnalysisError> visitor_error;
    const auto append = [&](AccessOperand operand, AccessMode mode,
                            MemoryEndpoint address, std::optional<std::uint64_t> bytes,
                            std::uint32_t access_index = 0,
                            std::optional<AtomicScope> atomic_scope = std::nullopt) {
      if(visitor_error) return;
      visitor_error=visitor(MemoryAccess{
          AccessId(static_cast<std::uint32_t>(access_count++)), event.id,
          instruction.id, instruction.actor, operand, mode, address, bytes,
          instruction.location, effect_index, access_index, atomic_scope});
    };
    if (immediate) {
      append(AccessOperand::kImmediate,
             immediate->direction == MemoryDirection::kLoad
                 ? AccessMode::kRead
                 : AccessMode::kWrite,
             immediate->address, immediate->bytes);
    } else if (async) {
      append(AccessOperand::kSource, AccessMode::kRead, async->source,
             async->bytes);
      append(AccessOperand::kDestination, AccessMode::kWrite,
             async->destination, async->bytes);
    } else {
      for (std::size_t index = 0; index < generic->accesses.size(); ++index) {
        const MemoryEffect& effect = generic->accesses[index];
        if (effect.mode != AccessMode::kRead &&
            effect.mode != AccessMode::kWrite) {
          return failure("generic memory execution has invalid access mode",
                         event.id);
        }
        append(AccessOperand::kEffect, effect.mode, effect.endpoint,
               effect.bytes, static_cast<std::uint32_t>(index), effect.atomic_scope);
      }
    }
    if(visitor_error) return VisitResult::Failure(std::move(*visitor_error));
  }
  if(seen_count!=offsets.back()) return failure("event graph omits source memory/effect executions");
  return VisitResult::Success(access_count);
}
Extraction ExtractFromEvents(const EventRecords& events,const SourceProgram& source) {
  MemoryAccessRecords accesses;
  std::map<ActorId, CtaId> ctas;
  for (const auto& actor : source.actors) ctas.emplace(actor.id, actor.cta);
  std::optional<ActorId> previous;
  std::size_t begin = 0;
  // Threads may be lowered by cohort, not by ID. A family's actor progression
  // must not decrease, so a thread joins its predecessor only in ascending order.
  std::uintptr_t ascending = 1;
  const auto finish = [&] {
    if (!previous) return;
    const auto cta = ctas.find(*previous);
    if (cta != ctas.end()) (void)accesses.compressThreadTail(begin, cta->second.value(), ascending, true);
  };
  auto result=VisitEvents(events,source,[&](const MemoryAccess& access) -> std::optional<MemoryAnalysisError> {
    if (previous != access.actor) {
      finish();
      if (previous && access.actor.value() < previous->value()) ++ascending;
      begin = accesses.beginThread(); previous = access.actor;
    }
    accesses.push_back(access);return std::nullopt;
  });
  finish();
  if(!result) return Extraction::Failure(std::move(result).error());
  return Extraction::Success(std::move(accesses));
}
}  // namespace

std::vector<MemoryAccessFamily> BindMemoryAccessFamilies(
    const EventRecords& events, const MemoryAccessRecords& accesses) {
  EventFamilies layout(events);
  std::vector<MemoryAccessFamily> result;
  accesses.forEachAffineColumn([&](const MemoryAccess& first, const auto& delta,
                                  auto, auto, std::size_t count) {
    auto prototype = first;
    MemoryAccessCodec::Binding base;
    MemoryAccessCodec::split(prototype, base);
    for (std::size_t offset = 0; offset < count;) {
      auto translated = MemoryAccessCodec::translate(base, delta, offset);
      if (!translated) throw std::logic_error("invalid compute access binding");
      auto access = prototype;
      MemoryAccessCodec::bind(access, *translated);
      const auto [family, member] = layout.locate(access.event);
      const auto& event_family = layout.families()[family];
      const auto length = delta[1] == event_family.stride
          ? std::min(count-offset, event_family.count-member) : std::size_t{1};
      result.push_back({std::move(access), delta, length, family, member, event_family.count});
      offset += length;
    }
  });
  return result;
}

Result<std::vector<MemoryAccessFamily>,MemoryAnalysisError> ExtractMemoryAccessFamilies(
    const HappensBeforeGraph& graph,const SourceProgram& source) {
  using Output=Result<std::vector<MemoryAccessFamily>,MemoryAnalysisError>;
  EventFamilies layout(graph.events());
  std::vector<MemoryAccessFamily> families;
  std::map<std::tuple<std::size_t,unsigned,std::uint32_t>,std::size_t> tails;
  // Validate every source/event binding while retaining only affine family runs.
  const auto checked=VisitEvents(graph.events(),source,[&](const MemoryAccess& access)->std::optional<MemoryAnalysisError> {
    const auto [event_family,member]=layout.locate(access.event);
    const auto key=std::tuple{event_family,static_cast<unsigned>(access.operand),access.access_index};
    const auto found=tails.find(key);
    if(found!=tails.end()) {
      auto& family=families[found->second];
      if(member==family.first_member+family.count) {
        auto a=family.first,b=access;
        MemoryAccessCodec::Binding x,y;MemoryAccessCodec::split(a,x);MemoryAccessCodec::split(b,y);
        if(a==b) {
          if(family.count==1) {
            family.delta=*MemoryAccessCodec::difference(x,y);++family.count;return {};
          }
          if(auto predicted=MemoryAccessCodec::translate(x,family.delta,family.count);predicted && *predicted==y) {
            ++family.count;return {};
          }
        }
      }
    }
    tails[key]=families.size();
    families.push_back({access,{},1,event_family,member,layout.families()[event_family].count});
    return {};
  });
  if(!checked)return Output::Failure(checked.error());
  return Output::Success(std::move(families));
}

Extraction ExtractMemoryAccesses(const HappensBeforeGraph& graph,
                                 const SourceProgram& source) {
  return ExtractFromEvents(graph.events(), source);
}

Extraction ExtractMemoryAccesses(const EventDag& graph,
                                 const SourceProgram& source) {
  return ExtractFromEvents(graph.events(), source);
}

Extraction ExtractMemoryAccesses(const EventRecords& events, const SourceProgram& source) {
  return ExtractFromEvents(events, source);
}

Result<std::size_t,MemoryAnalysisError> VisitMemoryAccesses(const EventDag& graph,
    const SourceProgram& source,const Visitor& visitor) {
  return VisitEvents(graph.events(),source,visitor);
}

Result<std::size_t,MemoryAnalysisError> VisitMemoryAccesses(const EventRecords& events,
    const SourceProgram& source,const Visitor& visitor) {
  return VisitEvents(events,source,visitor);
}

}  // namespace sentryward::algorithm
