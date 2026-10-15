#include "sentryward/algorithm/source_ir/initialization_recovery.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "sentryward/algorithm/source_ir/validator.h"

namespace sentryward::algorithm {
namespace {

struct OrderedInstruction final {
  std::size_t source_index;
  std::size_t indegree = 0;
  std::vector<std::size_t> successors;
};

template <typename Consumer>
void ForEachBarrierUse(const SourceOp& operation, Consumer&& consume) {
  std::visit(
      [&](const auto& op) {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, MBarrierInit>) {
          return;
        } else if constexpr (requires { op.barrier; }) {
          consume(op.barrier);
        } else if constexpr (std::is_same_v<T, AsyncLoad> ||
                             std::is_same_v<T, AsyncStore>) {
          if (op.completion) consume(op.completion->barrier);
        } else if constexpr (std::is_same_v<T, AsyncOp>) {
          for (const auto& effect : op.effects) {
            if (effect.completion) {
              std::visit([&](const auto& action) { consume(action.barrier); },
                         *effect.completion);
            }
          }
        }
      },
      operation);
}

}  // namespace

InitializationRecovery RecoverDuplicateInitializers(SourceProgram& program) {
  InitializationRecovery result{{}, SourceValidator{}.Validate(program)};
  if (result.diagnostics.empty() ||
      std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                  [](const Diagnostic& diagnostic) {
                    return diagnostic.code !=
                           DiagnosticCode::kDuplicateBarrierInitialization;
                  })) {
    return result;
  }

  std::map<BarrierLifetimeId, std::vector<std::size_t>> initializers;
  std::unordered_map<std::uint32_t, BarrierLifetimeId> initialized_barrier;
  for (std::size_t i = 0; i < program.instructions.size(); ++i) {
    const auto instruction = program.instructions[i];
    if (const auto* init =
            std::get_if<MBarrierInit>(&instruction.op)) {
      initializers[init->barrier].push_back(i);
      initialized_barrier.emplace(program.instructions[i].id.value(), init->barrier);
    }
  }
  std::erase_if(initializers,
                [](const auto& entry) { return entry.second.size() < 2; });

  std::map<BarrierLifetimeId, std::string> reasons;
  for (const auto& instruction : program.instructions) {
    const MBarrierCompletion* completion = nullptr;
    if (const auto* load = std::get_if<AsyncLoad>(&instruction.op);
        load && load->completion) {
      completion = &*load->completion;
    } else if (const auto* store = std::get_if<AsyncStore>(&instruction.op);
               store && store->completion) {
      completion = &*store->completion;
    }
    if (completion && initializers.contains(completion->barrier)) {
      reasons.try_emplace(completion->barrier,
          "legacy async completion lacks initialization-before-issue checking");
    }
  }
  std::set<ActorId> selected_actors;
  for (const auto& [barrier, indices] : initializers) {
    const ActorId actor = program.instructions[indices.front()].actor;
    if (std::any_of(indices.begin(), indices.end(), [&](std::size_t index) {
          return program.instructions[index].actor != actor;
        })) {
      reasons.emplace(barrier, "initializers belong to different actors");
    }
    selected_actors.insert(actor);
  }
  const auto refuse = [&]() {
    for (auto& diagnostic : result.diagnostics) {
      // Validator diagnostics name a concrete duplicate initialization.
      if (diagnostic.instruction) {
        const auto barrier = initialized_barrier.at(diagnostic.instruction->value());
        const auto reason = reasons.find(barrier);
        diagnostic.message += "; initialization recovery refused: ";
        diagnostic.message += reason != reasons.end()
                                  ? reason->second
                                  : "another duplicate barrier is unsafe (all-or-nothing recovery)";
      }
    }
    return result;
  };
  if (!reasons.empty()) return refuse();

  // Index affected actors only; validated PO, not IDs, determines their order.
  std::vector<OrderedInstruction> nodes;
  std::unordered_map<std::uint32_t, std::size_t> node_by_id;
  for (std::size_t i = 0; i < program.instructions.size(); ++i) {
    const auto& instruction = program.instructions[i];
    if (selected_actors.contains(instruction.actor)) {
      node_by_id.emplace(instruction.id.value(), nodes.size());
      nodes.push_back({i, 0, {}});
    }
  }
  for (const auto& edge : program.edges) {
    if (edge.kind != SourceEdgeKind::kProgramOrder) continue;
    const auto source = node_by_id.find(edge.from.value());
    if (source == node_by_id.end()) continue;
    const std::size_t target = node_by_id.at(edge.to.value());
    nodes[source->second].successors.push_back(target);
    ++nodes[target].indegree;
  }
  std::vector<std::size_t> order;
  order.reserve(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].indegree == 0) order.push_back(i);
  }
  std::vector<std::size_t> rank(nodes.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    const std::size_t current = order[i];
    rank[current] = i;
    for (const std::size_t next : nodes[current].successors) {
      if (--nodes[next].indegree == 0) order.push_back(next);
    }
  }
  const auto position = [&](std::size_t source_index) {
    return rank[node_by_id.at(program.instructions[source_index].id.value())];
  };
  std::map<BarrierLifetimeId, std::size_t> retained;
  for (auto& [barrier, indices] : initializers) {
    std::sort(indices.begin(), indices.end(), [&](std::size_t left, std::size_t right) {
      return position(left) < position(right);
    });
    retained.emplace(barrier, indices.back());
  }
  for (const auto& node : nodes) {
    const auto& instruction = program.instructions[node.source_index];
    ForEachBarrierUse(instruction.op, [&](BarrierLifetimeId barrier) {
      const auto last = retained.find(barrier);
      if (last != retained.end() &&
          instruction.actor == program.instructions[last->second].actor &&
          position(node.source_index) < position(last->second)) {
        reasons.try_emplace(barrier, "non-initialization use at instruction " +
                                        std::to_string(instruction.id.value()) +
                                        " precedes the retained initializer on its actor");
      }
    });
  }
  if (!reasons.empty()) return refuse();

  // Prepare all allocations before mutation, so eligibility is all-or-nothing.
  std::vector<std::size_t> replaced;
  std::unordered_set<std::uint32_t> replaced_ids;
  for (const auto& [barrier, indices] : initializers) {
    const auto& last = program.instructions[indices.back()];
    const auto& last_init = std::get<MBarrierInit>(last.op);
    for (const std::size_t index : indices) {
      if (index == indices.back()) continue;
      const auto& old = program.instructions[index];
      result.warnings.push_back(
          {barrier, old.id, last.id, old.location, last.location,
           std::get<MBarrierInit>(old.op).expected_arrivals,
           last_init.expected_arrivals});
      replaced.push_back(index);
      replaced_ids.insert(old.id.value());
    }
  }
  const auto canonicalize = [&](MilestoneRef& ref) {
    if (ref.milestone == Milestone::kIssue &&
        replaced_ids.contains(ref.instruction.value())) {
      ref.milestone = Milestone::kComplete;
    }
  };
  for (const auto index : replaced) program.instructions.edit(index).op = Await{};
  for (std::size_t i=0;i<program.instructions.size();++i) {
    auto instruction=program.instructions[i];
    if (auto* op = std::get_if<AsyncOp>(&instruction.op)) {
      for (auto& ref : op->completion_requires) canonicalize(ref);
      for (auto& effect : op->effects) {
        for (auto& ref : effect.predecessors) canonicalize(ref);
      }
    } else if (auto* wait = std::get_if<Await>(&instruction.op)) {
      for (auto& ref : wait->requirements) canonicalize(ref);
    }
    if (instruction != program.instructions[i]) program.instructions.edit(i) = std::move(instruction);
  }
  for (auto& edge : program.dependencies) {
    canonicalize(edge.from);
    canonicalize(edge.to);
  }
  result.diagnostics.clear();
  return result;
}

}  // namespace sentryward::algorithm
