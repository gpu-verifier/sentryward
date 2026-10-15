#include "sentryward/algorithm/memory/constant_address_model.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <limits>
#include <tuple>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sentryward::algorithm {
namespace {
using StorageKey = std::pair<std::size_t, std::size_t>;
constexpr auto kEmptyStorage = std::numeric_limits<std::size_t>::max();

class ConstantStorageFilter final : public MemoryStorageFilter {
 public:
  explicit ConstantStorageFilter(std::vector<StorageKey> keys)
      : keys_(std::move(keys)) {
    const auto row_bytes = ((keys_.size() + 63) / 64) * sizeof(std::uint64_t);
    capacity_ = std::max<std::size_t>(1, (64U * 1024U * 1024U) /
        std::max<std::size_t>(1, row_bytes + sizeof(Row)));
    for (std::size_t i = 0; i < keys_.size(); ++i) {
      const auto key = keys_[i];
      if (key.first == kEmptyStorage) continue;
      groups_[key].push_back(i);
      spaces_[key.first].push_back(i);
    }
  }

  void retainCompatible(AccessId access, DynamicBitset& candidates) override {
    const auto key = keys_.at(access.value());
    if (key.first == kEmptyStorage) { candidates.reset(); return; }
    if (key.first == 0) return;
    auto found = cache_.find(key);
    if (found == cache_.end()) {
      if (rows_.size() == capacity_) {
        cache_.erase(rows_.back().first);
        rows_.pop_back();
      }
      DynamicBitset mask(keys_.size());
      const auto add = [&](const auto& index, auto bucket) {
        if (const auto it = index.find(bucket); it != index.end())
          for (auto i : it->second) mask.set(i);
      };
      add(spaces_, std::size_t{0});
      if (key.second == 0) add(spaces_, key.first);
      else {
        add(groups_, StorageKey{key.first, 0});
        add(groups_, key);
      }
      rows_.emplace_front(key, std::move(mask));
      found = cache_.emplace(key, rows_.begin()).first;
    } else {
      rows_.splice(rows_.begin(), rows_, found->second);
    }
    candidates &= found->second->second;
  }

 private:
  using Row = std::pair<StorageKey, DynamicBitset>;
  std::vector<StorageKey> keys_;
  std::map<StorageKey, std::vector<std::size_t>> groups_;
  std::map<std::size_t, std::vector<std::size_t>> spaces_;
  std::list<Row> rows_;
  std::map<StorageKey, std::list<Row>::iterator> cache_;
  std::size_t capacity_;
};
}  // namespace

std::unique_ptr<MemoryStorageFilter> ConstantAddressModel::storageFilter(
    MemoryAccessView accesses) {
  if (error_) return nullptr;
  std::vector<StorageKey> keys;
  keys.reserve(accesses.size());
  std::map<std::tuple<unsigned, std::size_t, std::size_t>, std::size_t> groups;
  const auto group = [&](unsigned kind, std::size_t a, std::size_t b = 0) {
    return groups.emplace(std::tuple{kind, a, b}, groups.size() + 1).first->second;
  };
  const auto register_space = intern("register");
  for (const auto& access : accesses) {
    if (access.bytes == 0) { keys.emplace_back(kEmptyStorage, 0); continue; }
    if (const auto* reg = std::get_if<RegisterAddress>(&access.address)) {
      keys.emplace_back(register_space, group(2, reg->actor.value(), intern(reg->name)));
      continue;
    }
    const auto fp = footprint(access);
    const auto allocation = fp.allocation == 0 ? 0 :
        fp.alias_scope != 0 ? group(1, fp.alias_scope) : group(0, fp.allocation);
    keys.emplace_back(fp.address_space, allocation);
  }
  return std::make_unique<ConstantStorageFilter>(std::move(keys));
}

ConstantAddressModel::ConstantAddressModel(
    const std::vector<ConstantAddressBinding>& bindings, std::size_t cache_capacity)
    : cache_capacity_(cache_capacity) {
  std::map<AddressId, const ConstantAddressBinding*> ordered_bindings;
  for (const ConstantAddressBinding& binding : bindings) {
    if ((binding.address_space.empty() || binding.allocation.empty()) &&
        !error_.has_value()) {
      error_ = MemoryAnalysisError{
          "constant-address space and allocation names must be nonempty",
          std::nullopt, binding.address};
    }
    const auto [entry, inserted] = ordered_bindings.emplace(binding.address, &binding);
    if (!inserted && *entry->second != binding && !error_.has_value()) {
      error_ = MemoryAnalysisError{
          "conflicting constant-address bindings for AddressId " +
              std::to_string(binding.address.value()),
          std::nullopt, binding.address};
    }
  }
  bindings_.reserve(ordered_bindings.size());
  std::map<std::pair<std::string_view, std::string_view>, std::string_view> scopes;
  for (const auto& [id, source] : ordered_bindings) {
    const ConstantAddressBinding& binding = *source;
    const auto [entry, inserted] = scopes.emplace(
        std::pair<std::string_view, std::string_view>{
            binding.address_space, binding.allocation}, binding.alias_scope);
    if (!inserted && entry->second != binding.alias_scope && !error_) {
      error_ = MemoryAnalysisError{"inconsistent alias scope for one allocation", std::nullopt, id};
    }
    bindings_.emplace(id.value(), PreparedBinding{
        intern(binding.address_space), intern(binding.allocation),
        binding.offset, intern(binding.alias_scope), !binding.expression.empty()});
  }
  cache_index_.reserve(std::min(cache_capacity_, std::size_t{4096}));
}

std::size_t ConstantAddressModel::NameHash::operator()(
    std::string_view name) const noexcept {
  return std::hash<std::string_view>{}(name);
}

std::size_t ConstantAddressModel::PairHash::operator()(
    const PairKey& pair) const noexcept {
  std::size_t result = 0;
  const auto mix = [&](std::uint64_t value) {
    result ^= std::hash<std::uint64_t>{}(value) + 0x9e3779b97f4a7c15ULL +
              (result << 6) + (result >> 2);
  };
  for (const FootprintKey* key : {&pair.first, &pair.second}) {
    mix(static_cast<std::uint64_t>(key->kind));
    mix(key->unresolved_address);
    mix(key->address_space);
    mix(key->allocation);
    mix(key->offset);
    mix(key->bytes);
    mix(key->alias_scope);
  }
  return result;
}

ConstantAddressModel::NameId ConstantAddressModel::intern(std::string_view name) {
  const auto found = name_ids_.find(name);
  if (found != name_ids_.end()) return found->second;
  const NameId id = names_.size();
  names_.emplace_back(name);
  name_ids_.emplace(names_.back(), id);
  return id;
}

std::optional<MemoryAnalysisError> ConstantAddressModel::validate() const {
  return error_;
}

ConstantAddressCacheStats ConstantAddressModel::cacheStats() const noexcept {
  ConstantAddressCacheStats result = stats_;
  result.entries = cache_lru_.size();
  return result;
}

ConstantAddressModel::FootprintKey ConstantAddressModel::footprint(
    const MemoryAccess& access) {
  if (access.bytes == 0) {
    return {FootprintKind::kEmpty, 0, 0, 0, 0, 0};
  }
  if (const auto* reg = std::get_if<RegisterAddress>(&access.address)) {
    return {FootprintKind::kRegister,
            reg->actor.value(),
            0,
            intern(reg->name),
            0,
            access.bytes.value_or(1)};
  }
  const AddressId address = std::get<AddressId>(access.address);
  const auto binding = bindings_.find(address.value());
  if (binding == bindings_.end()) {
    return {FootprintKind::kMissing, address.value(), 0, 0, 0, access.bytes.value_or(0)};
  }
  const PreparedBinding& resolved = binding->second;
  if (resolved.abstract || !access.bytes || names_[resolved.allocation] == "unresolved" ||
      names_[resolved.address_space] == "generic") {
    const auto space = names_[resolved.address_space] == "generic" ? 0 : resolved.address_space;
    const auto allocation = names_[resolved.allocation] == "unresolved" ? 0 : resolved.allocation;
    return {FootprintKind::kMissing, address.value(), space, allocation, resolved.offset,
            access.bytes.value_or(0), resolved.alias_scope};
  }
  return {FootprintKind::kResolved, 0,                resolved.address_space,
          resolved.allocation,      resolved.offset, *access.bytes, resolved.alias_scope};
}

OverlapResult ConstantAddressModel::compare(const PairKey& pair) const {
  if (error_.has_value()) {
    return OverlapResult::Unknown("invalid constant-address model: " +
                                  error_->message);
  }
  const auto& [first, second] = pair;
  if (first.kind == FootprintKind::kEmpty ||
      second.kind == FootprintKind::kEmpty) {
    return OverlapResult::Disjoint();
  }
  if (first.kind == FootprintKind::kRegister ||
      second.kind == FootprintKind::kRegister) {
    if (first.kind != second.kind ||
        first.unresolved_address != second.unresolved_address ||
        first.allocation != second.allocation) {
      return OverlapResult::Disjoint();
    }
    return OverlapResult::Overlap({"same actor-owned register", std::nullopt});
  }
  if (first.address_space != 0 && second.address_space != 0) {
    if (first.address_space != second.address_space) return OverlapResult::Disjoint();
    if (first.allocation != 0 && second.allocation != 0 &&
        first.allocation != second.allocation &&
        !(first.alias_scope != 0 && first.alias_scope == second.alias_scope))
      return OverlapResult::Disjoint();
  }
  if (first.kind == FootprintKind::kMissing ||
      second.kind == FootprintKind::kMissing) {
    std::string reason;
    for (const FootprintKey* key : {&first, &second}) {
      if (key->kind != FootprintKind::kMissing) {
        continue;
      }
      if (!reason.empty()) {
        reason += "; ";
      }
      reason += "constant address or extent unavailable for AddressId " +
                std::to_string(key->unresolved_address);
    }
    return OverlapResult::Unknown(std::move(reason));
  }
  if (first.address_space != second.address_space) {
    return OverlapResult::Disjoint();
  }
  if (first.allocation != second.allocation) {
    if (first.alias_scope != 0 && first.alias_scope == second.alias_scope) {
      return OverlapResult::Unknown("allocation-relative addresses may share storage in " + names_[first.alias_scope]);
    }
    return OverlapResult::Disjoint();
  }

  // Subtract offsets, never form an unsigned interval endpoint.
  const FootprintKey& earlier = first.offset <= second.offset ? first : second;
  const FootprintKey& later = first.offset <= second.offset ? second : first;
  const std::uint64_t distance = later.offset - earlier.offset;
  if (distance >= earlier.bytes) {
    return OverlapResult::Disjoint();
  }
  const std::uint64_t bytes = std::min(earlier.bytes - distance, later.bytes);
  return OverlapResult::Overlap(
      {"constant byte ranges intersect",
       ByteOverlap{names_[first.address_space], names_[first.allocation], later.offset,
                   bytes}});
}

std::optional<FamilyFootprint> ConstantAddressModel::familyFootprint(const MemoryAccessFamily& family) {
  using I=address_solver::Int;
  const auto first=footprint(family.first);
  if(first.kind==FootprintKind::kEmpty) { FamilyFootprint result;result.empty=true;return result; }
  if(first.kind==FootprintKind::kRegister) {
    return FamilyFootprint{"register",names_[first.allocation],"","",
        I(first.unresolved_address)-I(family.delta[5])*I(family.first_member),I(family.delta[5]),1,true,false};
  }
  if(first.kind!=FootprintKind::kResolved)return std::nullopt;
  FamilyFootprint result{names_[first.address_space],names_[first.allocation],names_[first.alias_scope],"",I(first.offset),0,first.bytes,true,false};
  // Address IDs refer to a flat external table; verify every binding, not samples.
  for(std::size_t k=1;k<family.count;++k) {
    const auto current=footprint(family.at(family.first_member+k));
    if(current.kind!=first.kind || current.address_space!=first.address_space || current.allocation!=first.allocation ||
        current.alias_scope!=first.alias_scope || current.bytes!=first.bytes)return std::nullopt;
    if(k==1)result.step=I(current.offset)-I(first.offset);
    if(I(current.offset)!=I(first.offset)+result.step*I(k))return std::nullopt;
  }
  result.offset-=result.step*I(family.first_member);
  return result;
}

OverlapResult ConstantAddressModel::mayOverlap(const MemoryAccess& first,
                                               const MemoryAccess& second) {
  ++stats_.queries;
  PairKey pair{footprint(first), footprint(second)};
  if (pair.second < pair.first) {
    std::swap(pair.first, pair.second);
  }
  const auto cached = cache_index_.find(pair);
  if (cached != cache_index_.end()) {
    ++stats_.hits;
    cache_lru_.splice(cache_lru_.begin(), cache_lru_, cached->second);
    return cached->second->second;
  }
  ++stats_.misses;
  OverlapResult result = compare(pair);
  if (cache_capacity_ == 0) {
    return result;
  }
  if (cache_lru_.size() == cache_capacity_) {
    static_assert(std::is_nothrow_copy_assignable_v<PairKey>);
    static_assert(std::is_nothrow_move_assignable_v<OverlapResult>);
    OverlapResult cached_result = result;
    const auto victim = std::prev(cache_lru_.end());
    auto node = cache_index_.extract(victim->first);
    if (node.empty() || node.mapped() != victim)
      throw std::logic_error("constant-address cache eviction index mismatch");
    node.key() = pair;
    victim->first = pair;
    victim->second = std::move(cached_result);
    cache_lru_.splice(cache_lru_.begin(), cache_lru_, victim);
    const auto inserted = cache_index_.insert(std::move(node));
    if (!inserted.inserted)
      throw std::logic_error("constant-address cache replacement key already exists");
    ++stats_.evictions;
    return result;
  }
  cache_lru_.emplace_front(std::move(pair), result);
  cache_index_.emplace(cache_lru_.front().first, cache_lru_.begin());
  return result;
}

}  // namespace sentryward::algorithm
