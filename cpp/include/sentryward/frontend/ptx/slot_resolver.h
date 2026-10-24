#pragma once
// Pure slot resolution. Memory and Tma resolve one access but append each
// table write as an intern step instead of applying it, and catch any
// exception at their entry. The outcome -- the steps in write order and the
// terminal footprint or the original exception -- is committed by
// UnitInterner::Commit at the Resources::Memory/Tma call site, so the tables,
// results and exceptions are those of applying each write immediately.
//
// Exactness: E1 the bodies read back only bindings of their own
// steps, and read them from the step (E2: a stored binding equals its step);
// E3 every bridge or extractor call precedes every step, so deferring the
// writes only lets pure memo work run past a would-be alias_scope_conflict.
// The memos (memory resources, the extractor) are per instance.
#include "sentryward/frontend/ptx/resources.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sentryward::frontend::ptx {
class AddressValueExtractor;

class SlotResolver {
public:
  // `core`: the Resources whose values, owners, lane caches and records the
  // bodies read; it owns this resolver.
  SlotResolver(Resources &core, SlotConfig config);
  ~SlotResolver();
  SlotResolver(const SlotResolver &) = delete;
  SlotResolver &operator=(const SlotResolver &) = delete;
  // Each runs inside a FactSource::SlotScope and writes no table.
  [[nodiscard]] SlotOutcome<ResolvedFootprint> Memory(const RecordView &, const MemoryRequest &);
  [[nodiscard]] SlotOutcome<ResolvedTma> Tma(const RecordView &, const TmaRequest &);

private:
  struct MemoryResourceInfo {
    std::string space, object;
    std::uint64_t base = 0;
    std::optional<std::uint64_t> extent;
  };
  SlotTerminal<ResolvedFootprint> MemoryBody(const RecordView &, const MemoryRequest &, std::vector<SlotStep> &prefix,
                                             FallbackUse &fallback);
  ResolvedTma TmaBody(const RecordView &, const TmaRequest &, std::vector<SlotStep> &steps,
                      FallbackUse &fallback);
  // The binding step of `bytes` at `offset` in `resource` (or in the pointer
  // resource whose concrete allocation it names): checks captured extents and
  // adds the concrete allocation's base; memoized per resource.
  ConstantStep MemoryResource(std::string resource, std::uint64_t offset, std::uint64_t bytes);
  // The address-value extractor's expression for `operand` on the active lane
  // (the extractor is created on first use). Each Definition it requests is
  // read with the requested lane active; the active lane is then restored.
  std::optional<nlohmann::json> ExtractAddress(const RecordView &, std::string_view operand);
  // The tensor map of `operand`; `descriptor` receives its key (or
  // "explicit:" + the contract's descriptor).
  const nlohmann::json &TensorMap(const RecordView &, std::string_view, const nlohmann::json &contract,
                                  const nlohmann::json *dynamic_tensor_maps, std::string &descriptor) const;
  Resources &core_;
  SlotConfig config_;
  std::unique_ptr<AddressValueExtractor> address_values_;
  std::unordered_map<std::string, MemoryResourceInfo> memory_resources_;
};
} // namespace sentryward::frontend::ptx
