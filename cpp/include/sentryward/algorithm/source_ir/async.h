#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "sentryward/algorithm/base/shared_array.h"
#include "sentryward/algorithm/source_ir/memory_effect.h"

namespace sentryward::algorithm {

enum class Milestone { kIssue, kReadsComplete, kComplete, kEffect };

struct MilestoneRef final {
  InstId instruction;
  Milestone milestone;
  // Only kEffect accepts a nonzero index.
  std::uint32_t effect_index = 0;

  friend bool operator==(const MilestoneRef&, const MilestoneRef&) = default;
};

struct ArriveEffect final {
  BarrierLifetimeId barrier;
  std::uint32_t arrivals;

  friend bool operator==(const ArriveEffect&, const ArriveEffect&) = default;
};

struct CompleteTxEffect final {
  BarrierLifetimeId barrier;
  std::uint64_t bytes;

  friend bool operator==(const CompleteTxEffect&,
                         const CompleteTxEffect&) = default;
};

using CompletionAction = std::variant<ArriveEffect, CompleteTxEffect>;

struct AsyncEffect final {
  // These accesses and the optional barrier action execute atomically.
  SharedArray<MemoryEffect> accesses;
  std::optional<CompletionAction> completion = std::nullopt;
  std::vector<MilestoneRef> predecessors = {};

  friend bool operator==(const AsyncEffect&, const AsyncEffect&) = default;
};

struct AsyncOp final {
  AsyncOpId async_id;
  // Effects share issue but are otherwise independently schedulable.
  // ReadsComplete joins every effect containing a read. Complete joins all
  // effects.
  std::vector<AsyncEffect> effects;
  // Gates the final join only, not any effect or its barrier action.
  std::vector<MilestoneRef> completion_requires = {};

  friend bool operator==(const AsyncOp&, const AsyncOp&) = default;
};

// The state spaces a bidirectional generic-async proxy fence (fence.proxy.async) orders.
enum ProxyFenceSpace : std::uint8_t { kProxyFenceShared = 1, kProxyFenceGlobal = 2 };

struct Await final {
  std::vector<MilestoneRef> requirements;
  // Nonzero: the await is a fence.proxy.async over these ProxyFenceSpace bits. It orders, in its
  // thread, prior and subsequent accesses of different proxies (memory_race_checker.h).
  std::uint8_t proxy_fence = 0;

  friend bool operator==(const Await&, const Await&) = default;
};

struct EndpointDependency final {
  MilestoneRef from;
  MilestoneRef to;

  friend bool operator==(const EndpointDependency&,
                         const EndpointDependency&) = default;
};

}  // namespace sentryward::algorithm
