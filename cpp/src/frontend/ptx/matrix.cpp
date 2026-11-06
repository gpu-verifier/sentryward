#include "sentryward/frontend/ptx/matrix.h"

#include "sentryward/frontend/ptx/layouts.h"

#include <algorithm>

namespace sentryward::frontend::ptx {
namespace {
// Resource provenance lookup is lane-specific. Restore it for the caller even
// when a supplying lane has an unresolved row address or other semantic error.
class RestoreLane {
public:
  RestoreLane(Resources &resources, LaneRef lane)
      : resources_(resources), lane_(lane) {}
  ~RestoreLane() { resources_.SetLane(lane_); }
  RestoreLane(const RestoreLane &) = delete;
  RestoreLane &operator=(const RestoreLane &) = delete;

private:
  Resources &resources_;
  LaneRef lane_;
};
[[noreturn]] void Participation(std::string message) {
  throw SemanticError("matrix_memory_participation", std::move(message));
}
} // namespace

MatrixResolver::MatrixResolver(const CtaFacts &facts)
    : facts_(facts) {
  const auto &function = facts.function();
  if (std::none_of(function.instructions.begin(), function.instructions.end(),
      [](const auto &i) { return i.opcode == "ldmatrix" || i.opcode == "stmatrix"; })) return;
  for (std::size_t index = 0; index < facts.LaneCount(); ++index) {
    const auto lane = facts.Lane(index);
    const auto lane_id = facts.LaneId(lane);
    if (lane_id >= 32)
      Participation("matrix warp lane is outside 0..31");
    const auto warp = facts.Warp(lane);
    for (const auto execution : facts.ExpandLane(lane, ExpansionSite::Matrix)) {
      const auto &instruction =
          function.instructions.at(execution.instruction_id());
      if (instruction.opcode != "ldmatrix" && instruction.opcode != "stmatrix")
        continue;
      auto &occurrence = occurrences_[{warp, execution.Key()}];
      if (occurrence.lanes[lane_id])
        Participation("duplicate matrix occurrence in one physical lane");
      occurrence.lanes[lane_id] = lane;
      occurrence.positions[lane_id] = execution.ref()->pos;
    }
  }
}
std::vector<Fragment>
MatrixResolver::Resolve(LaneRef current, std::size_t position,
                        const RecordView &execution,
                        Resources &resources, const nlohmann::json &contract) {
  (void)contract; // Contracts cannot override the instruction's lane mapping.
  if (&resources.facts != &facts_)
    Participation("matrix resources belong to another execution program");
  const auto &instruction =
      facts_.function().instructions.at(execution.instruction_id());
  std::unique_lock lock(mutex_);
  auto shape_it = shapes_.find(execution.instruction_id());
  if (shape_it == shapes_.end()) {
    try {
      shape_it =
          shapes_
              .emplace(execution.instruction_id(),
                       DecodeMatrixShape(facts_.module(), instruction,
                                         instruction.opcode == "stmatrix"))
              .first;
    } catch (const LayoutError &error) {
      throw SemanticError("matrix_memory_variant", error.what());
    }
  }
  const auto &shape = shape_it->second; // map nodes are stable and never changed
  lock.unlock();
  const auto current_lane_id = facts_.LaneId(current);
  if (current_lane_id >= 32)
    Participation("matrix data lane is outside 0..31");
  auto found = occurrences_.find({facts_.Warp(current), execution.Key()});
  if (found == occurrences_.end())
    Participation("matrix instruction has no recorded warp occurrence");
  auto &occurrence = found->second;
  // The recorded occurrence is this physical lane's own expanded record: the
  // same element exactly when the view is that Expanded position.
  const auto &recorded = occurrence.lanes[current_lane_id];
  const bool recorded_here =
      recorded && execution.kind() == RecordKind::Expanded &&
      execution.ref() == RecordRef{*recorded, occurrence.positions[current_lane_id]} && *recorded == current;
  if (!recorded_here)
    Participation("matrix atom is not its physical lane's recorded occurrence");
  // Rows an earlier lane resolved (immutable once set); otherwise this lane
  // resolves them. Concurrent lanes first wait for the earlier lanes of the warp.
  lock.lock();
  changed_.wait(lock, [&] { return occurrence.resolved_by < position || !EarlierPending(occurrence, position); });
  const bool cached = occurrence.resolved_by < position;
  lock.unlock();
  if (!cached) {
    // PTX ISA ldmatrix (9.7.16.5.15) / stmatrix (9.7.16.5.16): ".aligned ... all threads in the
    // warp must execute the same" instruction. A thread of the warp that has no such occurrence
    // (it exited, diverged or had a false guard: a false-guard lane records no occurrence)
    // violates that: undefined behavior. A lane the launch does not have keeps the participation
    // refusal, as does a recorded peer whose guard is not known to be true.
    const auto misses = [&](std::size_t lane) {
      if (!facts_.FindLane(facts_.Warp(current), static_cast<std::uint32_t>(lane)))
        Participation("aligned matrix memory requires the same occurrence in all 32 lanes");
      throw SemanticError("aligned_lanes_missing",
                          instruction.opcode + " .aligned is not executed by lane " + std::to_string(lane) +
                              " of its warp (undefined behavior)",
                          {{"missing_lane", lane}, {"warp", facts_.Warp(current)},
                           {"isa_section", instruction.opcode == "ldmatrix" ? "9.7.16.5.15 ldmatrix" : "9.7.16.5.16 stmatrix"}});
    };
    for (std::size_t lane = 0; lane < occurrence.lanes.size(); ++lane) {
      if (!occurrence.lanes[lane]) misses(lane);
      // The constructor expands every lane (ExpansionSite::Matrix): the
      // recorded element.
      const auto peer = facts_.Meta({*occurrence.lanes[lane], occurrence.positions[lane]});
      if (peer.instruction_id() != execution.instruction_id())
        Participation("matrix peers disagree on instruction or participation");
      if (peer.predicate_value() != true)
        Participation("aligned matrix memory requires every peer's guard to be true");
    }
    // Commit only a complete row cache. A failed resolution leaves this
    // occurrence uncached, avoiding partial rows on a later diagnostic retry.
    std::vector<std::uint64_t> rows;
    rows.reserve(8 * shape.count);
    {
      RestoreLane restore(resources, current);
      for (unsigned lane = 0; lane < 8 * shape.count; ++lane) {
        resources.SetLane(*occurrence.lanes[lane]);
        const auto row_record = facts_.Meta({*occurrence.lanes[lane], occurrence.positions[lane]});
        const auto resolved = resources.Shared(row_record, shape.address);
        if (!resolved)
          throw SemanticError("matrix_memory_row_address",
                              "matrix supplying lane " + std::to_string(lane) +
                                  " has no concrete shared row address");
        const auto &[owner, offset] = *resolved;
        if (owner != resources.frame.self())
          throw SemanticError("matrix_memory_row_owner",
                              "matrix row address must be CTA-local");
        if (offset % 16 || offset > (1ULL << 32) - 16)
          throw SemanticError(
              "matrix_memory_row_alignment",
              "matrix row requires a 16-byte-aligned in-range shared address");
        rows.push_back(offset);
      }
    }
    {
      const std::lock_guard guard(mutex_);
      if (occurrence.resolved_by == SIZE_MAX) occurrence.rows = std::move(rows);
      occurrence.resolved_by = std::min(occurrence.resolved_by, position);
    }
    changed_.notify_all();
  }
  const auto intervals = MatrixLaneFootprint(shape.count, shape.transpose,
                                             current_lane_id, occurrence.rows);
  std::vector<Fragment> result;
  result.reserve(intervals.size());
  // ConstantSteps {Shared(self), offset} with the name formatted once (no
  // range check).
  const auto shared = resources.interner().Prepare(AllocationKey::Shared(resources.frame.self()));
  for (const auto &interval : intervals)
    result.emplace_back(resources.interner().Apply(shared, interval.offset),
                        interval.size);
  return result;
}
bool MatrixResolver::EarlierPending(const Occurrence &occurrence, std::size_t position) const {
  if (finished_.empty()) return false; // lanes in order: every earlier lane has ended
  for (const auto &lane : occurrence.lanes) {
    if (!lane) continue;
    const auto other = position_of_.at(lane->index);
    if (other < position && !finished_[other]) return true;
  }
  return false;
}
void MatrixResolver::BeginConcurrent(std::span<const std::size_t> order) {
  const std::lock_guard guard(mutex_);
  position_of_.assign(facts_.LaneCount(), SIZE_MAX);
  for (std::size_t position = 0; position < order.size(); ++position) position_of_.at(order[position]) = position;
  finished_.assign(order.size(), 0);
}
void MatrixResolver::Finished(std::size_t position) {
  {
    const std::lock_guard guard(mutex_);
    finished_.at(position) = 1;
  }
  changed_.notify_all();
}
void MatrixResolver::EndConcurrent() {
  const std::lock_guard guard(mutex_);
  position_of_.clear();
  finished_.clear();
}
} // namespace sentryward::frontend::ptx
