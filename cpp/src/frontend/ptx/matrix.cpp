#include "sentryward/frontend/ptx/matrix.h"

#include "sentryward/frontend/ptx/layouts.h"

namespace sentryward::frontend::ptx {
namespace {
// Resource provenance lookup is lane-specific. Restore it for the caller even
// when a supplying lane has an unresolved row address or other semantic error.
class RestoreLane {
public:
  RestoreLane(Resources &resources, const LaneTrace &lane)
      : resources_(resources), lane_(lane) {}
  ~RestoreLane() { resources_.SetLane(lane_); }
  RestoreLane(const RestoreLane &) = delete;
  RestoreLane &operator=(const RestoreLane &) = delete;

private:
  Resources &resources_;
  const LaneTrace &lane_;
};
[[noreturn]] void Participation(std::string message) {
  throw SemanticError("matrix_memory_participation", std::move(message));
}
} // namespace

MatrixResolver::MatrixResolver(const ExecutionProgram &program)
    : program_(program) {
  if (std::none_of(program.function->instructions.begin(), program.function->instructions.end(),
      [](const auto &i) { return i.opcode == "ldmatrix" || i.opcode == "stmatrix"; })) return;
  for (const auto &lane : program.lane_traces) {
    if (lane.lane_id >= 32)
      Participation("matrix warp lane is outside 0..31");
    for (const auto &execution : lane.instructions) {
      const auto &instruction =
          program.function->instructions.at(execution.instruction_id());
      if (instruction.opcode != "ldmatrix" && instruction.opcode != "stmatrix")
        continue;
      auto &occurrence = occurrences_[{lane.warp_id, execution.Key()}];
      if (occurrence.executions[lane.lane_id])
        Participation("duplicate matrix occurrence in one physical lane");
      occurrence.lanes[lane.lane_id] = &lane;
      occurrence.executions[lane.lane_id] = &execution;
    }
  }
}
std::vector<Fragment>
MatrixResolver::Resolve(const LaneTrace &current,
                        const ExecutedInstruction &execution,
                        Resources &resources, const nlohmann::json &contract) {
  (void)contract; // Contracts cannot override the instruction's lane mapping.
  if (&resources.program != &program_)
    Participation("matrix resources belong to another execution program");
  const auto &instruction =
      program_.function->instructions.at(execution.instruction_id());
  auto shape_it = shapes_.find(execution.instruction_id());
  if (shape_it == shapes_.end()) {
    try {
      shape_it =
          shapes_
              .emplace(execution.instruction_id(),
                       DecodeMatrixShape(*program_.module, instruction,
                                         instruction.opcode == "stmatrix"))
              .first;
    } catch (const LayoutError &error) {
      throw SemanticError("matrix_memory_variant", error.what());
    }
  }
  const auto &shape = shape_it->second;
  if (current.lane_id >= 32)
    Participation("matrix data lane is outside 0..31");
  auto found = occurrences_.find({current.warp_id, execution.Key()});
  if (found == occurrences_.end())
    Participation("matrix instruction has no recorded warp occurrence");
  auto &occurrence = found->second;
  if (occurrence.executions[current.lane_id] != &execution ||
      occurrence.lanes[current.lane_id] != &current)
    Participation("matrix atom is not its physical lane's recorded occurrence");
  if (occurrence.rows.empty()) {
    for (const auto *peer : occurrence.executions) {
      if (!peer)
        Participation("aligned matrix memory requires the same occurrence in "
                      "all 32 lanes");
      if (peer->instruction_id() != execution.instruction_id() ||
          peer->predicate_value() != true)
        Participation("matrix peers disagree on instruction or participation");
    }
    // Commit only a complete row cache. A failed resolution leaves this
    // occurrence uncached, avoiding partial rows on a later diagnostic retry.
    std::vector<std::uint64_t> rows;
    rows.reserve(8 * shape.count);
    {
      RestoreLane restore(resources, current);
      for (unsigned lane = 0; lane < 8 * shape.count; ++lane) {
        resources.SetLane(*occurrence.lanes[lane]);
        const auto resolved =
            resources.Shared(*occurrence.executions[lane], shape.address);
        if (!resolved)
          throw SemanticError("matrix_memory_row_address",
                              "matrix supplying lane " + std::to_string(lane) +
                                  " has no concrete shared row address");
        const auto &[owner, offset] = *resolved;
        if (owner != resources.coordinates)
          throw SemanticError("matrix_memory_row_owner",
                              "matrix row address must be CTA-local");
        if (offset % 16 || offset > (1ULL << 32) - 16)
          throw SemanticError(
              "matrix_memory_row_alignment",
              "matrix row requires a 16-byte-aligned in-range shared address");
        rows.push_back(offset);
      }
    }
    occurrence.rows = std::move(rows);
  }
  const auto intervals = MatrixLaneFootprint(shape.count, shape.transpose,
                                             current.lane_id, occurrence.rows);
  std::vector<Fragment> result;
  result.reserve(intervals.size());
  const auto allocation = "cta:" + CoordinateText(resources.coordinates) +
                          ":shared:physical.shared.cta.0";
  for (const auto &interval : intervals)
    result.emplace_back(
        resources.addresses.Intern("shared", allocation, interval.offset),
        interval.size);
  return result;
}
} // namespace sentryward::frontend::ptx
