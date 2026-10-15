#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/base/result.h"
#include "sentryward/algorithm/base/source_location.h"
#include "sentryward/algorithm/source_ir/program.h"

namespace sentryward::algorithm {

using AccessId = StrongId<struct AccessTag>;

enum class AccessOperand { kImmediate, kSource, kDestination, kEffect };

struct MemoryAccess final {
  AccessId id;
  EventId event;
  InstId instruction;
  ActorId actor;
  AccessOperand operand;
  AccessMode mode;
  MemoryEndpoint address;
  std::optional<std::uint64_t> bytes;
  SourceLocation location;
  std::uint32_t effect_index = 0;
  std::uint32_t access_index = 0;
  std::optional<AtomicScope> atomic_scope = std::nullopt;

  friend bool operator==(const MemoryAccess&, const MemoryAccess&) = default;
};

struct MemoryAccessCodec final {
  using Binding = std::array<std::uint32_t, 6>;
  using Delta = std::array<std::int64_t, 6>;
  static bool split(MemoryAccess& access, Binding& binding) {
    binding = {access.id.value(), access.event.value(), access.instruction.value(), access.actor.value(), 0, 0};
    access.id = AccessId{}; access.event = EventId{}; access.instruction = InstId{}; access.actor = ActorId{};
    std::visit([&](auto& address) {
      if constexpr (requires { address.actor; }) { binding[5] = address.actor.value(); address.actor = ActorId{}; }
      else { binding[4] = address.value(); address = AddressId{}; }
    }, access.address);
    return true;
  }
  static void bind(MemoryAccess& access, const Binding& binding) {
    access.id = AccessId(binding[0]); access.event = EventId(binding[1]);
    access.instruction = InstId(binding[2]); access.actor = ActorId(binding[3]);
    std::visit([&](auto& address) {
      if constexpr (requires { address.actor; }) address.actor = ActorId(binding[5]);
      else address = AddressId(binding[4]);
    }, access.address);
  }
  static std::optional<Delta> difference(const Binding& a, const Binding& b) {
    Delta delta{};
    for (std::size_t i = 0; i < a.size(); ++i) delta[i] = std::int64_t(b[i]) - a[i];
    return delta;
  }
  static std::optional<Binding> translate(Binding binding, const Delta& delta, std::size_t index) {
    for (std::size_t i = 0; i < binding.size(); ++i) {
      if (delta[i] >= 0) {
        if (delta[i] && index > (UINT32_MAX - binding[i]) / std::uint64_t(delta[i])) return {};
      } else if (index > binding[i] / std::uint64_t(-delta[i])) return {};
      binding[i] = static_cast<std::uint32_t>(binding[i] + delta[i] * std::int64_t(index));
    }
    return binding;
  }
  static std::size_t hash(const MemoryAccess& access) {
    auto hash = std::hash<std::string_view>{}(access.location.file.view());
    HashCombine(hash, access.location.line); HashCombine(hash, access.location.column);
    HashCombine(hash, static_cast<unsigned>(access.mode)); HashCombine(hash, static_cast<unsigned>(access.operand));
    HashCombine(hash, access.bytes.value_or(0)); HashCombine(hash, access.address.index());
    HashCombine(hash, access.effect_index); HashCombine(hash, access.access_index);
    return hash;
  }
};
using MemoryAccessRecords = TemplateRecords<MemoryAccess, MemoryAccessCodec>;

// Models consume either compact records or an existing concrete access table.
class MemoryAccessView final {
 public:
  MemoryAccessView() = default;
  MemoryAccessView(std::span<const MemoryAccess> values) : values_(values) {}
  MemoryAccessView(const std::vector<MemoryAccess>& values) : values_(values) {}
  template<std::size_t N> MemoryAccessView(const std::array<MemoryAccess, N>& values) : values_(values) {}
  template<std::size_t N> MemoryAccessView(const MemoryAccess (&values)[N]) : values_(values) {}
  MemoryAccessView(const MemoryAccessRecords& records) : records_(&records) {}
  std::size_t size() const { return records_ ? records_->size() : values_.size(); }
  bool empty() const { return size() == 0; }
  MemoryAccess operator[](std::size_t index) const { return records_ ? records_->at(index) : values_[index]; }
  MemoryAccess at(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("memory access view");
    return (*this)[index];
  }
  class Iterator {
   public:
    using value_type = MemoryAccess;
    using reference = MemoryAccess;
    using pointer = void;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    Iterator() = default;
    Iterator(const MemoryAccessView* owner, std::size_t index) : owner_(owner), index_(index) {}
    MemoryAccess operator*() const { return (*owner_)[index_]; }
    Iterator& operator++() { ++index_; return *this; }
    Iterator operator++(int) { auto old = *this; ++*this; return old; }
    friend bool operator==(const Iterator&, const Iterator&) = default;
   private:
    const MemoryAccessView* owner_ = nullptr;
    std::size_t index_ = 0;
  };
  Iterator begin() const { return {this, 0}; }
  Iterator end() const { return {this, size()}; }
 private:
  const MemoryAccessRecords* records_ = nullptr;
  std::span<const MemoryAccess> values_;
};

struct MemoryAnalysisError final {
  std::string message;
  std::optional<EventId> event;
  std::optional<MemoryEndpoint> address;

  friend bool operator==(const MemoryAnalysisError&,
                         const MemoryAnalysisError&) = default;
};

[[nodiscard]] Result<MemoryAccessRecords, MemoryAnalysisError>
ExtractMemoryAccesses(const HappensBeforeGraph& graph,
                      const SourceProgram& source);

[[nodiscard]] Result<MemoryAccessRecords, MemoryAnalysisError>
ExtractMemoryAccesses(const EventDag& graph, const SourceProgram& source);

[[nodiscard]] Result<MemoryAccessRecords, MemoryAnalysisError>
ExtractMemoryAccesses(const EventRecords& events, const SourceProgram& source);

// Validate and consume one access at a time without materializing an access
// array. The callback must not retain references to its temporary argument.
[[nodiscard]] Result<std::size_t, MemoryAnalysisError> VisitMemoryAccesses(
    const EventDag& graph, const SourceProgram& source,
    const std::function<std::optional<MemoryAnalysisError>(const MemoryAccess&)>& visitor);

[[nodiscard]] Result<std::size_t, MemoryAnalysisError> VisitMemoryAccesses(
    const EventRecords& events, const SourceProgram& source,
    const std::function<std::optional<MemoryAnalysisError>(const MemoryAccess&)>& visitor);

}  // namespace sentryward::algorithm
