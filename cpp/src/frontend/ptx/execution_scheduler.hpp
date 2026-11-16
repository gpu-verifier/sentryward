#pragma once
#include "execution_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
ExecutionProgram Executor::Run() {
  ProfileScope lanes("frontend.execute.lanes");
  // Only this run's decisions are the program's: a template attempt that fell back
  // (RunTemplates) on this executor may have recorded others.
  property_decisions_ = std::make_shared<TensorPropertyDecisions>();
  static_choice_lines_.clear();
  ExecutionProgram result;
  result.module = &m_;
  result.function = &f_;
  result.config = c_;
  result.register_names = names_;
  result.static_execution_counts.resize(decoded_.size());
  result.lane_traces.reserve(c_.warps.size() * c_.warp_size);
  for (const auto &warp : c_.warps) {
    WarpContext context;
    std::vector<LaneTask> tasks;
    tasks.reserve(c_.warp_size);
    for (std::uint32_t lane = 0; lane < c_.warp_size; ++lane) {
      const auto tid = warp.warp_id * c_.warp_size + lane;
      if (c_.analysis_thread_range && (tid < c_.analysis_thread_range->first ||
          tid >= c_.analysis_thread_range->second)) continue;
      tasks.push_back(Lane(warp, lane, context));
    }
    std::size_t remaining = tasks.size();
    while (remaining) {
      bool progress = false;
      for (auto &task : tasks) {
        if (task.Done()) continue;
        const auto &dependency = task.Dependency();
        if (dependency && !context.shuffle.contains(
                {dependency->instruction, dependency->occurrence, dependency->lane}))
          continue;
        ProfileScope resume("frontend.execute.lane_resume");
        task.Resume();
        ++resume.edges;
        progress = true;
        if (task.Done()) --remaining;
      }
      if (poll_) poll_();
      if (!progress)
        throw std::runtime_error("shuffle source has no matching dynamic occurrence");
    }
    ProfileScope collect("frontend.execute.trace_collection");
    for (auto &task : tasks) {
      auto trace = task.TakeResult();
      for (const auto &record : trace.instructions)
        ++result.static_execution_counts[record.instruction_id()];
      collect.nodes += trace.instructions.size();
      result.lane_traces.push_back(std::move(trace));
    }
    lanes.nodes += collect.nodes;
    lanes.edges += tasks.size();
  }
  result.record_compression = record_pool_.Statistics();
  result.tensor_property_decisions = property_decisions_;
  result.static_branch_choice_lines = static_choice_lines_;
  return result;
}
std::optional<std::vector<ExecutionProgram>> Executor::RunTemplateCluster(const Module& module,const Function& function,
    const std::vector<FrontendConfig>& configs,const std::function<void()>& poll,SpecializationPurpose purpose) {
  SharedExecution runtime;
  std::vector<std::shared_ptr<Executor>> executors;
  std::vector<TemplateTask> tasks;
  std::vector<ExecutionProgram> results(configs.size());
  for(std::size_t rank=0;rank<configs.size();++rank) {
    if(configs[rank].fixed_input.value("cluster_ctarank",std::uint32_t(rank))!=rank)
      throw ExecutionError("shared_control_cluster",0,"cooperative configurations must be in CTA rank order");
    executors.push_back(std::make_shared<Executor>(module,function,configs[rank],poll,purpose));
    tasks.push_back(executors.back()->RunTemplateTask(&runtime));
  }
  std::size_t remaining=tasks.size();
  while(remaining) {
    const auto before=std::pair{runtime.revision,remaining};
    for(std::size_t rank=0;rank<tasks.size();++rank)if(!tasks[rank].Done()) {
      tasks[rank].Resume();
      if(tasks[rank].Done()) {
        auto result=tasks[rank].TakeResult();
        if(!result)return {};
        results[rank]=std::move(*result);--remaining;
      }
    }
    runtime.Progress();if(poll)poll();
    if(remaining && before==std::pair{runtime.revision,remaining})return {};
  }
  for(auto &result:results)runtime.Export(result);
  return results;
}

std::vector<ExecutionProgram> Executor::RunCluster(const Module &module, const Function &function,
    const std::vector<FrontendConfig> &configs, const std::function<void()> &poll,
    SpecializationPurpose purpose) {
  if(auto compact=RunTemplateCluster(module,function,configs,poll,purpose))return std::move(*compact);
  SharedExecution runtime;
  std::vector<ExecutionProgram> results(configs.size());
  std::vector<std::unique_ptr<Executor>> executors;
  std::vector<AddressTable> addresses(configs.size());
  std::vector<BarrierTable> barriers(configs.size());
  std::vector<std::unique_ptr<ExecutorResources>> adapters;
  std::map<std::pair<std::size_t,std::uint32_t>,WarpContext> contexts;
  struct Task { std::size_t rank; WarpContext *context; LaneTask task; };
  std::vector<Task> tasks;
  for (std::size_t rank=0;rank<configs.size();++rank) {
    const auto &config=configs[rank];
    if (config.fixed_input.value("cluster_ctarank",std::uint32_t(rank))!=rank)
      throw ExecutionError("shared_control_cluster",0,"cooperative configurations must be in CTA rank order");
    executors.push_back(std::make_unique<Executor>(module,function,config,poll,purpose));
    auto &executor=*executors.back(); auto &result=results[rank];
    result.module=&module; result.function=&function; result.config=config;
    result.register_names=executor.names_; result.static_execution_counts.resize(function.instructions.size());
    const auto coordinates=config.fixed_input.value("cta_coordinate",Coordinates{0,0,0});
    adapters.push_back(std::make_unique<ExecutorResources>(result,coordinates,addresses[rank],barriers[rank]));
    for (const auto &warp : config.warps) {
      auto &context=contexts[{rank,warp.warp_id}]; context.shared=&runtime; context.resources=adapters.back().get();
      for (std::uint32_t lane=0;lane<config.warp_size;++lane)
        tasks.push_back({rank,&context,executor.Lane(warp,lane,context)});
    }
  }
  std::size_t remaining=tasks.size();
  auto progress=[&] {
    std::size_t shuffles=0; for (const auto &[key,context] : contexts) { (void)key; shuffles+=context.shuffle.size(); }
    return std::tuple{runtime.revision,remaining,shuffles};
  };
  while (remaining) {
    const auto before=progress();
    for (auto &entry : tasks) {
      auto &task=entry.task; if (task.Done()) continue;
      const auto &dep=task.Dependency();
      if (dep && !dep->shared && !entry.context->shuffle.contains({dep->instruction,dep->occurrence,dep->lane})) continue;
      task.Resume(); if (task.Done()) --remaining;
    }
    runtime.Progress(); if (poll) poll();
    if (remaining && before==progress()) {
      std::uint32_t line=0;
      for (const auto &entry : tasks) if (!entry.task.Done() && entry.task.Dependency()) {
        line=function.instructions[entry.task.Dependency()->instruction].line; break;
      }
      throw ExecutionError("shared_control_blocked",line,"shared-value propagation has no enabled publication or collective");
    }
  }
  for (auto &entry : tasks) {
    auto trace=entry.task.TakeResult(); auto &result=results[entry.rank];
    for (const auto &e : trace.instructions) ++result.static_execution_counts[e.instruction_id()];
    result.lane_traces.push_back(std::move(trace));
  }
  for (std::size_t rank = 0; rank < results.size(); ++rank) {
    auto &result = results[rank];
    runtime.Export(result);
    result.record_compression = executors[rank]->record_pool_.Statistics();
    result.tensor_property_decisions = executors[rank]->property_decisions_;
    result.static_branch_choice_lines = executors[rank]->static_choice_lines_;
    for (auto it=runtime.tensor_maps.begin();it!=runtime.tensor_maps.end();++it)
      result.config.tma_descriptors["executed."+it.key()]=it.value();
  }
  return results;
}
} // namespace
} // namespace sentryward::frontend::ptx
