#include "sentryward/algorithm/memory/constant_address_model.h"
#include "sentryward/algorithm/memory/conflict_order.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>

namespace sentryward::algorithm {
namespace {
using Region = ConstantConflictRegion;

struct CoarseAccess {
  EventId event;
  bool write, missing, memory, empty;
  std::size_t space, allocation, alias;
};

class MixedCandidates final : public MemoryConflictCandidates {
 public:
  MixedCandidates(std::vector<CoarseAccess> accesses, std::vector<bool> special,
                  std::unique_ptr<MemoryStorageFilter> filter,
                  std::unique_ptr<MemoryConflictCandidates> exact)
      : accesses_(std::move(accesses)), special_(std::move(special)),
        filter_(std::move(filter)), exact_(std::move(exact)),
        all_(accesses_.size()), writes_(accesses_.size()), candidates_(accesses_.size()) {
    for (std::size_t i = 0; i < accesses_.size(); ++i) {
      if (!accesses_[i].memory || accesses_[i].empty) continue;
      all_.set(i);
      if (accesses_[i].write) writes_.set(i);
      if (special_[i]) points_.push_back(i);
    }
  }
  std::optional<MemoryConflictPair> next() override {
    while (position_ < points_.size()) {
      const auto i = points_[position_];
      const auto& first = accesses_[i];
      if (!prepared_) {
        candidates_ = first.write ? all_ : writes_;
        filter_->retainCompatible(AccessId(i), candidates_);
        cursor_ = candidates_.find_first();
        prepared_ = true;
      }
      while (cursor_ != DynamicBitset::npos) {
        const auto j = cursor_;
        cursor_ = candidates_.find_next(cursor_);
        const auto& second = accesses_[j];
        if (first.event == second.event || (special_[j] && j <= i)) continue;
        // Known same-allocation ranges are covered by the interval stream.
        if (!first.missing && !second.missing &&
            (first.allocation == second.allocation || first.space != second.space ||
             first.alias == 0 || first.alias != second.alias)) continue;
        return MemoryConflictPair{AccessId(i), AccessId(j)};
      }
      ++position_;
      prepared_ = false;
    }
    return exact_->next();
  }
 private:
  std::vector<CoarseAccess> accesses_;
  std::vector<bool> special_;
  std::unique_ptr<MemoryStorageFilter> filter_;
  std::unique_ptr<MemoryConflictCandidates> exact_;
  DynamicBitset all_, writes_, candidates_;
  std::vector<std::size_t> points_;
  std::size_t position_ = 0, cursor_ = DynamicBitset::npos;
  bool prepared_ = false;
};

class IntervalCandidates final : public MemoryConflictCandidates {
 public:
  explicit IntervalCandidates(std::vector<Region> regions) : rows_(std::move(regions)) {
    std::sort(rows_.begin(), rows_.end(), [](const auto& a, const auto& b) {
      return std::tie(a.group, a.offset, a.access) < std::tie(b.group, b.offset, b.access);
    });
  }
  std::optional<MemoryConflictPair> next() override {
    while (position_ < rows_.size()) {
      const auto& row = rows_[position_];
      if (!prepared_) {
        if (position_ == 0 || row.group != rows_[position_ - 1].group) {
          active_.clear(); writes_.clear(); ends_ = {};
        }
        while (!ends_.empty()) {
          const auto [overflow, end, index] = ends_.top();
          if (overflow || end > row.offset) break;
          ends_.pop(); active_.erase(index); writes_.erase(index);
        }
        candidates_ = row.write ? &active_ : &writes_;
        cursor_ = candidates_->begin();
        prepared_ = true;
      }
      while (cursor_ != candidates_->end()) {
        const auto& earlier = rows_[*cursor_++];
        if (earlier.event != row.event) return MemoryConflictPair{earlier.access, row.access};
      }
      active_.insert(position_);
      if (row.write) writes_.insert(position_);
      const auto end = row.offset + row.bytes;
      ends_.emplace(end < row.offset, end, position_);
      ++position_;
      prepared_ = false;
    }
    return std::nullopt;
  }

 private:
  std::vector<Region> rows_;
  std::set<std::size_t> active_, writes_;
  const std::set<std::size_t>* candidates_ = nullptr;
  std::set<std::size_t>::const_iterator cursor_;
  using End = std::tuple<bool, std::uint64_t, std::size_t>;
  std::priority_queue<End, std::vector<End>, std::greater<End>> ends_;
  std::size_t position_ = 0;
  bool prepared_ = false;
};

class CombinedCandidates final : public MemoryConflictCandidates {
 public:
  CombinedCandidates(std::unique_ptr<MemoryConflictCandidates> first,
                     std::unique_ptr<MemoryConflictCandidates> second)
      : first_(std::move(first)), second_(std::move(second)) {}
  std::optional<MemoryConflictPair> next() override {
    if (first_) {
      if (auto pair = first_->next()) return pair;
      first_.reset();
    }
    return second_->next();
  }
 private:
  std::unique_ptr<MemoryConflictCandidates> first_, second_;
};
}  // namespace

std::unique_ptr<MemoryConflictCandidates> ConstantAddressModel::conflictCandidates(
    MemoryAccessView accesses) {
  return makeCandidates(accesses, {}, false);
}

std::unique_ptr<MemoryConflictCandidates> ConstantAddressModel::conflictOrder(
    MemoryAccessView accesses, std::span<const std::size_t> ranks) {
  return makeCandidates(accesses, ranks, true);
}

std::unique_ptr<MemoryConflictCandidates> ConstantAddressModel::conflictCandidatesWithExemptions(
    MemoryAccessView accesses, std::span<const std::size_t> ranks,
    std::span<const AccessId> exempt_writes) {
  return makeCandidates(accesses, ranks, false, exempt_writes);
}

std::unique_ptr<MemoryConflictCandidates> ConstantAddressModel::makeCandidates(
    MemoryAccessView accesses, std::span<const std::size_t> ranks,
    bool order_only, std::optional<std::span<const AccessId>> exempt_writes) {
  if (error_) throw std::invalid_argument(error_->message);
  using Group = std::tuple<FootprintKind, std::uint32_t, NameId, NameId>;
  std::map<Group, std::size_t> groups;
  std::map<std::pair<NameId, NameId>, NameId> aliases;
  std::set<std::pair<NameId, NameId>> ambiguous_aliases;
  std::vector<CoarseAccess> coarse;
  coarse.reserve(accesses.size());
  bool incomplete = false;
  std::vector<Region> rows;
  rows.reserve(accesses.size());
  const bool needs_exact = order_only || exempt_writes.has_value();
  for (std::size_t i = 0; i < accesses.size(); ++i) {
    const auto& access = accesses[i];
    if (access.id.value() != i || (access.mode != AccessMode::kRead && access.mode != AccessMode::kWrite))
      throw std::invalid_argument("invalid access table for constant conflict candidates");
    if (needs_exact && access.event.value() >= ranks.size())
      throw std::invalid_argument("missing topological rank for memory access");
    const auto fp = footprint(access);
    coarse.push_back({access.event, access.mode == AccessMode::kWrite,
                      fp.kind == FootprintKind::kMissing, fp.kind != FootprintKind::kRegister,
                      fp.kind == FootprintKind::kEmpty, fp.address_space, fp.allocation, fp.alias_scope});
    if (fp.kind == FootprintKind::kEmpty) continue;
    if (fp.kind == FootprintKind::kMissing) {
      if (needs_exact) return nullptr;
      incomplete = true;
      continue;
    }
    if (fp.kind == FootprintKind::kResolved && fp.alias_scope != 0) {
      auto [it, inserted] = aliases.emplace(std::make_pair(fp.address_space, fp.alias_scope), fp.allocation);
      if (!inserted && it->second != fp.allocation) {
        if (needs_exact) return nullptr;
        ambiguous_aliases.emplace(fp.address_space, fp.alias_scope);
      }
    }
    const Group key{fp.kind, fp.unresolved_address, fp.address_space, fp.allocation};
    const auto group = groups.try_emplace(key, groups.size()).first->second;
    rows.push_back({access.id, access.event, access.mode == AccessMode::kWrite, group,
                   fp.kind == FootprintKind::kRegister ? 0 : fp.offset,
                   fp.kind == FootprintKind::kRegister ? 1 : fp.bytes});
  }
  if (order_only) return MakeConflictOrder(std::move(rows), ranks);
  if (exempt_writes) {
    std::vector<bool> exempt_accesses(accesses.size()), full_groups(groups.size());
    for (const auto id : *exempt_writes) {
      if (id.value() >= accesses.size() ||
          accesses[id.value()].mode != AccessMode::kWrite)
        throw std::invalid_argument("exemption must reference a valid memory write");
      exempt_accesses[id.value()] = true;
    }
    for (const auto& row : rows)
      if (exempt_accesses[row.access.value()]) full_groups[row.group] = true;
    std::vector<Region> full, reduced;
    for (auto& row : rows)
      (full_groups[row.group] ? full : reduced).push_back(std::move(row));
    return std::make_unique<CombinedCandidates>(
        MakeConflictOrder(std::move(reduced), ranks),
        std::make_unique<IntervalCandidates>(std::move(full)));
  }
  auto exact = std::make_unique<IntervalCandidates>(std::move(rows));
  if (!incomplete && ambiguous_aliases.empty()) return exact;
  std::vector<bool> special;
  special.reserve(coarse.size());
  for (const auto& item : coarse)
    special.push_back(item.missing || ambiguous_aliases.contains({item.space, item.alias}));
  return std::make_unique<MixedCandidates>(std::move(coarse), std::move(special),
                                         storageFilter(accesses), std::move(exact));
}

}  // namespace sentryward::algorithm
