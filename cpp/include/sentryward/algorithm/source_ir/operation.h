#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "sentryward/algorithm/base/strong_id.h"
#include "sentryward/algorithm/source_ir/async.h"

namespace sentryward::algorithm {

struct ImmediateLoad final {
  AddressId address;
  std::uint32_t bytes;

  friend bool operator==(const ImmediateLoad&, const ImmediateLoad&) = default;
};

struct ImmediateStore final {
  AddressId address;
  std::uint32_t bytes;

  friend bool operator==(const ImmediateStore&,
                         const ImmediateStore&) = default;
};

struct MBarrierCompletion final {
  BarrierLifetimeId barrier;
  std::uint64_t bytes;

  friend bool operator==(const MBarrierCompletion&,
                         const MBarrierCompletion&) = default;
};

struct AsyncLoad final {
  AsyncOpId async_id;
  AddressId source;
  AddressId destination;
  std::uint64_t bytes;
  std::optional<MBarrierCompletion> completion;

  friend bool operator==(const AsyncLoad&, const AsyncLoad&) = default;
};

struct AsyncStore final {
  AsyncOpId async_id;
  AddressId source;
  AddressId destination;
  std::uint64_t bytes;
  std::optional<MBarrierCompletion> completion;

  friend bool operator==(const AsyncStore&, const AsyncStore&) = default;
};

struct MBarrierInit final {
  BarrierLifetimeId barrier;
  std::uint32_t expected_arrivals;

  friend bool operator==(const MBarrierInit&, const MBarrierInit&) = default;
};

struct MBarrierArrive final {
  BarrierLifetimeId barrier;
  std::uint32_t arrivals;
  std::optional<StateId> result;

  friend bool operator==(const MBarrierArrive&,
                         const MBarrierArrive&) = default;
};

struct MBarrierExpectTx final {
  BarrierLifetimeId barrier;
  std::uint64_t bytes;

  friend bool operator==(const MBarrierExpectTx&,
                         const MBarrierExpectTx&) = default;
};

struct MBarrierArriveExpectTx final {
  BarrierLifetimeId barrier;
  std::uint32_t arrivals;
  std::uint64_t bytes;
  std::optional<StateId> result;

  friend bool operator==(const MBarrierArriveExpectTx&,
                         const MBarrierArriveExpectTx&) = default;
};

struct MBarrierCompleteTx final {
  BarrierLifetimeId barrier;
  std::uint64_t bytes;

  friend bool operator==(const MBarrierCompleteTx&,
                         const MBarrierCompleteTx&) = default;
};

struct MBarrierAddArrivals final {
  BarrierLifetimeId barrier;
  std::uint32_t arrivals;

  friend bool operator==(const MBarrierAddArrivals&,
                         const MBarrierAddArrivals&) = default;
};

struct MBarrierWaitState final {
  BarrierLifetimeId barrier;
  StateId state;

  friend bool operator==(const MBarrierWaitState&,
                         const MBarrierWaitState&) = default;
};

struct MBarrierWaitParity final {
  BarrierLifetimeId barrier;
  std::uint8_t parity;

  friend bool operator==(const MBarrierWaitParity&,
                         const MBarrierWaitParity&) = default;
};

struct MBarrierTryWaitState final {
  BarrierLifetimeId barrier;
  StateId state;
  std::optional<std::uint32_t> time_hint_ns;

  friend bool operator==(const MBarrierTryWaitState&,
                         const MBarrierTryWaitState&) = default;
};

struct MBarrierTryWaitParity final {
  BarrierLifetimeId barrier;
  std::uint8_t parity;
  std::optional<std::uint32_t> time_hint_ns;

  friend bool operator==(const MBarrierTryWaitParity&,
                         const MBarrierTryWaitParity&) = default;
};

struct NamedBarrierArrive final {
  BarrierLifetimeId barrier;
  std::uint32_t expected_arrivals;
  friend bool operator==(const NamedBarrierArrive&, const NamedBarrierArrive&) = default;
};

struct NamedBarrierSync final {
  BarrierLifetimeId barrier;
  std::uint32_t expected_arrivals;
  friend bool operator==(const NamedBarrierSync&, const NamedBarrierSync&) = default;
};

struct SyncThreads final {
  SyncId sync_id;

  friend bool operator==(const SyncThreads&, const SyncThreads&) = default;
};

struct SyncWarp final {
  SyncId sync_id;
  std::uint32_t lane_mask;
  // Data collectives rendezvous, but do not publish prior memory accesses.
  bool memory_ordering = true;

  friend bool operator==(const SyncWarp&, const SyncWarp&) = default;
};

using SourceOp =
    std::variant<ImmediateLoad, ImmediateStore, AsyncLoad, AsyncStore,
                 MBarrierInit, MBarrierArrive, MBarrierExpectTx,
                 MBarrierArriveExpectTx, MBarrierCompleteTx, MBarrierWaitState,
                 MBarrierWaitParity, MBarrierTryWaitState,
                 MBarrierTryWaitParity, SyncThreads, SyncWarp, AsyncOp, Await,
                 MBarrierAddArrivals, NamedBarrierArrive, NamedBarrierSync>;

}  // namespace sentryward::algorithm
