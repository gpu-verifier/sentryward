#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sentryward/algorithm/memory/model.h"

namespace sentryward::algorithm {

struct ConstantAddressBinding final {
  AddressId address;
  std::string address_space;
  std::string allocation;
  std::uint64_t offset;
  std::string alias_scope = {};
  // Serialized memory expression; an empty string denotes a constant binding.
  std::string expression = {};

  friend bool operator==(const ConstantAddressBinding&,
                         const ConstantAddressBinding&) = default;
};

struct ConstantAddressCacheStats final {
  std::size_t queries = 0;
  std::size_t hits = 0;
  std::size_t misses = 0;
  std::size_t evictions = 0;
  std::size_t entries = 0;

  friend bool operator==(const ConstantAddressCacheStats&,
                         const ConstantAddressCacheStats&) = default;
};

class ConstantAddressModel final : public MemoryModel {
 public:
  explicit ConstantAddressModel(const std::vector<ConstantAddressBinding>& bindings,
                                std::size_t cache_capacity = 4096);

  ConstantAddressModel(const ConstantAddressModel&) = delete;
  ConstantAddressModel& operator=(const ConstantAddressModel&) = delete;
  ConstantAddressModel(ConstantAddressModel&&) = default;
  ConstantAddressModel& operator=(ConstantAddressModel&&) = default;

  [[nodiscard]] std::optional<MemoryAnalysisError> validate() const override;
  [[nodiscard]] OverlapResult mayOverlap(const MemoryAccess& first,
                                         const MemoryAccess& second) override;
  [[nodiscard]] ConstantAddressCacheStats cacheStats() const noexcept;
  [[nodiscard]] std::optional<FamilyFootprint> familyFootprint(const MemoryAccessFamily&) override;
  [[nodiscard]] std::unique_ptr<MemoryStorageFilter> storageFilter(
      MemoryAccessView accesses) override;
  [[nodiscard]] std::unique_ptr<MemoryConflictCandidates> conflictCandidates(
      MemoryAccessView accesses) override;
  [[nodiscard]] std::unique_ptr<MemoryConflictCandidates> conflictOrder(
      MemoryAccessView accesses, std::span<const std::size_t> ranks) override;
  [[nodiscard]] std::unique_ptr<MemoryConflictCandidates> conflictCandidatesWithExemptions(
      MemoryAccessView accesses, std::span<const std::size_t> ranks,
      std::span<const AccessId> exempt_writes) override;

 private:
  [[nodiscard]] std::unique_ptr<MemoryConflictCandidates> makeCandidates(
      MemoryAccessView accesses, std::span<const std::size_t> ranks,
      bool order_only,
      std::optional<std::span<const AccessId>> exempt_writes = std::nullopt);
  // Optional diagnostics inspect owned capacities without changing model state.
  friend class MemoryCensus;
  enum class FootprintKind { kEmpty, kResolved, kMissing, kRegister };
  using NameId = std::size_t;

  struct NameHash final {
    using is_transparent = void;
    [[nodiscard]] std::size_t operator()(std::string_view name) const noexcept;
  };

  struct PreparedBinding final {
    NameId address_space;
    NameId allocation;
    std::uint64_t offset;
    NameId alias_scope;
    bool abstract;
  };

  struct FootprintKey final {
    FootprintKind kind;
    std::uint32_t unresolved_address;
    NameId address_space;
    NameId allocation;
    std::uint64_t offset;
    std::uint64_t bytes;
    NameId alias_scope = 0;

    friend auto operator<=>(const FootprintKey&, const FootprintKey&) = default;
  };

  using PairKey = std::pair<FootprintKey, FootprintKey>;
  using CacheEntry = std::pair<PairKey, OverlapResult>;

  struct PairHash final {
    [[nodiscard]] std::size_t operator()(const PairKey& pair) const noexcept;
  };

  [[nodiscard]] NameId intern(std::string_view name);
  [[nodiscard]] FootprintKey footprint(const MemoryAccess& access);
  [[nodiscard]] OverlapResult compare(const PairKey& pair) const;

  std::vector<std::string> names_{""};
  std::unordered_map<std::string, NameId, NameHash, std::equal_to<>> name_ids_{{"", 0}};
  std::unordered_map<std::uint32_t, PreparedBinding> bindings_;
  std::optional<MemoryAnalysisError> error_;
  std::size_t cache_capacity_;
  ConstantAddressCacheStats stats_;
  std::list<CacheEntry> cache_lru_;
  std::unordered_map<PairKey, std::list<CacheEntry>::iterator, PairHash> cache_index_;
};

}  // namespace sentryward::algorithm
