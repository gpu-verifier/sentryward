#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include "sentryward/frontend/ptx/layouts.h"
#include "sentryward/frontend/ptx/resources.h"
#include "sentryward/frontend/ptx/unit_facts.h"

namespace sentryward::frontend::ptx {
// One instance per specialized CTA. The facts, their program and its lane
// storage must outlive the resolver and remain unchanged while cached
// occurrences exist. For ldmatrix/stmatrix kernels the constructor expands
// every lane (ExpansionSite::Matrix).
//
// The row cache is shared by the lanes of the CTA, also when they are lowered
// on worker threads (`resources` is the caller's). `position` is the calling
// lane's position in the CTA's lowering order: rows answer a call only if a
// lane at an earlier position resolved them, which is when lowering the lanes
// one by one in that order finds them cached; the caller then notes no reads.
// Between BeginConcurrent and EndConcurrent, lanes run concurrently and each
// lane's end is reported (Finished): a call first waits until every earlier
// lane of the occurrence's warp has resolved it or ended, so that, as in
// order, the earliest lane that reaches an occurrence resolves its rows. Wait
// edges point to earlier positions, so they cannot form a cycle.
class MatrixResolver {
public:
  explicit MatrixResolver(const CtaFacts &facts);
  MatrixResolver(const MatrixResolver &) = delete;
  MatrixResolver &operator=(const MatrixResolver &) = delete;
  [[nodiscard]] std::vector<Fragment>
  Resolve(LaneRef current, std::size_t position, const RecordView &execution,
          Resources &resources, const nlohmann::json &contract);
  // `order`: the lowering order (lane indices). O(lanes).
  void BeginConcurrent(std::span<const std::size_t> order);
  void Finished(std::size_t position);
  void EndConcurrent();

private:
  struct Key {
    std::uint32_t warp;
    DynamicKey dynamic;
    auto operator<=>(const Key &) const = default;
  };
  // Per physical lane id: the recording lane and the position of its record
  // (an expanded record; positions[l] is meaningful iff lanes[l] is engaged).
  // `rows` (immutable once set) were resolved by lanes at `resolved_by` (the
  // earliest such position).
  struct Occurrence {
    std::array<std::optional<LaneRef>, 32> lanes{};
    std::array<std::size_t, 32> positions{};
    std::vector<std::uint64_t> rows;
    std::size_t resolved_by = SIZE_MAX;
  };
  // Concurrent lanes: whether a lane of `occurrence` before `position` may still resolve it.
  [[nodiscard]] bool EarlierPending(const Occurrence &occurrence, std::size_t position) const;
  const CtaFacts &facts_;
  std::map<Key, Occurrence> occurrences_; // keys fixed by the constructor
  std::map<std::uint32_t, MatrixShape> shapes_;
  // Concurrent lanes: lane index -> position, and per position whether the lane ended.
  std::vector<std::size_t> position_of_;
  std::vector<std::uint8_t> finished_;
  std::mutex mutex_; // shapes_, the rows and resolved_by of each occurrence, finished_
  std::condition_variable changed_;
};
} // namespace sentryward::frontend::ptx
