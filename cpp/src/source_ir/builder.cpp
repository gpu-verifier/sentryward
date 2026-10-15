#include "sentryward/algorithm/source_ir/builder.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sentryward::algorithm {
InstId SourceProgramBuilder::append(ActorId actor, SourceLocation location, SourceOp op,
                                   std::optional<PtxOrigin> origin, bool program_order) {
  if (program.instructions.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("instruction ID space exhausted");
  const InstId id(static_cast<std::uint32_t>(program.instructions.size()));
  program.instructions.push_back({id, actor, std::move(location), std::move(op), std::move(origin)});
  if (program_order) {
    first_.try_emplace(actor, id);
    if (auto previous=last_.find(actor); previous!=last_.end())
      program.edges.pushBackAffine({previous->second,id,SourceEdgeKind::kProgramOrder});
    last_.insert_or_assign(actor,id);
  }
  return id;
}
void SourceProgramBuilder::appendGeneratedThread(ActorId actor,std::size_t count,
    std::function<SourceInstruction(std::size_t)> generate,
    std::optional<std::pair<std::uint64_t,std::uintptr_t>> cohort,
    SourceInstructions::AffineGenerator columns,const void* family) {
  if(!count)return;
  const auto begin=program.instructions.size();
  if(begin>std::uint64_t(UINT32_MAX)+1 || count>std::uint64_t(UINT32_MAX)+1-begin)
    throw std::overflow_error("instruction ID space exhausted");
  if(!generate)throw std::invalid_argument("missing source template generator");
  program.instructions.appendGenerated(count,[begin,actor,family,generate=std::move(generate)](std::size_t i) {
    auto value=generate(i);
    if(!family){value.id=InstId(begin+i);value.actor=actor;}
    return value;
  },cohort,std::move(columns),family);
  const InstId first(begin),last(begin+count-1);
  first_.try_emplace(actor,first);
  if(auto previous=last_.find(actor);previous!=last_.end())
    program.edges.pushBackAffine({previous->second,first,SourceEdgeKind::kProgramOrder});
  if(count>1)program.edges.appendAffine(
      SourceEdge{InstId(begin),InstId(begin+1),SourceEdgeKind::kProgramOrder},
      SourceEdgeCodec::Delta{1,1},count-1);
  last_.insert_or_assign(actor,last);
}
SourceInstruction& SourceProgramBuilder::editInstruction(InstId id) {
  auto& result=program.instructions.edit(id.value());
  if(result.id!=id) throw std::logic_error("native builder requires dense instruction IDs");
  return result;
}
SourceInstruction SourceProgramBuilder::instruction(InstId id) const {
  return program.instructions.at(id.value());
}
void SourceProgramBuilder::initializeNamedBarrier(BarrierLifetimeId barrier,
    std::uint32_t expected,std::span<const InstId> uses) {
  if(uses.empty()) throw std::invalid_argument("named barrier has no uses");
  const auto actor=instruction(uses.front()).actor;
  const auto id=append(actor,instruction(uses.front()).location,MBarrierInit{barrier,expected},std::nullopt,false);
  initializers_[actor].push_back(id);
  for(auto use:uses) program.dependencies.push_back({{id,Milestone::kComplete},{use,Milestone::kIssue}});
}
void SourceProgramBuilder::alignWarp(std::span<const std::pair<InstId,InstId>> calls,
                                    bool join_completions, std::size_t anchor) {
  if(calls.size()!=32 || anchor>=calls.size()) throw std::invalid_argument("aligned warp needs 32 lane calls");
  std::vector<MilestoneRef> requirements;
  requirements.reserve(32);
  for(const auto& [first,tail]:calls) {
    (void)tail;
    requirements.push_back({first,Milestone::kIssue});
  }
  const auto [first,tail]=calls[anchor];
  if(first!=tail) {
    const auto actor=instruction(first).actor;
    const auto gate=append(actor,instruction(first).location,Await{std::move(requirements)},std::nullopt,false);
    program.edges.pushBackAffine({first,gate,SourceEdgeKind::kProgramOrder});
    program.edges.pushBackAffine({gate,tail,SourceEdgeKind::kProgramOrder});
    replaced_edges_.emplace(first,tail);
    requirements={{gate,Milestone::kComplete}};
  }
  const auto effect_requirements=requirements;
  if(join_completions) for(const auto& call:calls) requirements.push_back({call.first,Milestone::kComplete});
  const auto append_unique=[](auto& to,const auto& from) {
    for(const auto& ref:from) if(std::find(to.begin(),to.end(),ref)==to.end()) to.push_back(ref);
  };
  for(const auto& [head,end]:calls) {
    auto* wait=std::get_if<Await>(&editInstruction(end).op);
    if(!wait) throw std::invalid_argument("aligned lane tail is not an await");
    append_unique(wait->requirements,requirements);
    if(auto* op=std::get_if<AsyncOp>(&editInstruction(head).op))
      for(auto& effect:op->effects) append_unique(effect.predecessors,effect_requirements);
  }
}
void SourceProgramBuilder::finishEdges() {
  for(const auto& [actor,ids]:initializers_) {
    for(std::size_t i=1;i<ids.size();++i) program.edges.pushBackAffine({ids[i-1],ids[i],SourceEdgeKind::kProgramOrder});
    if(!ids.empty()) program.edges.pushBackAffine({ids.back(),first_.at(actor),SourceEdgeKind::kProgramOrder});
  }
  if(!replaced_edges_.empty()) {
    SourceEdges retained;
    for(const auto& edge:program.edges)
      if(!replaced_edges_.contains({edge.from,edge.to}))retained.pushBackAffine(edge);
    program.edges=std::move(retained);
  }
  initializers_.clear(); replaced_edges_.clear();
}
}  // namespace sentryward::algorithm
