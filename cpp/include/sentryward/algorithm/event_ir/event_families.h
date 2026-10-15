#pragma once

#include <map>
#include <memory>
#include <vector>
#include "sentryward/algorithm/base/affine_index.h"
#include "sentryward/algorithm/event_ir/event_records.h"

namespace sentryward::algorithm {

class EventFamilies final {
 public:
  struct Family { EventId first; std::uint32_t stride; std::size_t count; };
  explicit EventFamilies(const EventRecords& events) : count_(events.size()) {
    events.forEachAffineColumnWithBlock([&](const Event& event,const auto&,std::size_t begin,
        std::size_t stride,std::size_t count,std::size_t block,std::size_t width,std::size_t members) {
      if(bands_.empty() || bands_.rbegin()->first!=block)
        bands_.emplace(block,Band{width,members,{}});
      auto& band=bands_.rbegin()->second;
      const auto column=(begin-block)%width;
      if(column==band.columns.size())band.columns.push_back(families_.size());
      else if(column+1!=band.columns.size())throw std::logic_error("unordered family column");
      families_.push_back({event.id, static_cast<std::uint32_t>(stride), count});
    });
    std::size_t end=families_.size();
    for(auto it=bands_.rbegin();it!=bands_.rend();++it) {
      auto& band=it->second;
      if(band.columns.size()!=band.width)throw std::logic_error("incomplete family block");
      const auto first=band.columns.front();band.columns.push_back(end);end=first;
    }
  }
  [[nodiscard]] std::pair<std::size_t, std::size_t> locate(EventId id) const {
    if (id.value() >= count_) throw std::out_of_range("event family ID");
    const auto it = std::prev(bands_.upper_bound(id.value()));
    const auto offset = id.value() - it->first;
    const auto& band=it->second;
    const auto column=offset%band.width;
    auto first=band.columns[column],last=band.columns[column+1];
    if(last-first>1) {
      auto found=std::upper_bound(families_.begin()+first,families_.begin()+last,id,
        [](EventId value,const Family& family){return value<family.first;});
      if(found==families_.begin()+first)throw std::logic_error("missing family member");
      first=std::prev(found)-families_.begin();
    }
    const auto& family=families_[first];
    const auto member=(id.value()-family.first.value())/family.stride;
    if(member>=family.count)throw std::logic_error("missing family member");
    return {first,member};
  }
  [[nodiscard]] const std::vector<Family>& families() const { return families_; }
  // Join an event progression with the storage's actual block/column layout.
  // A column-stride mismatch is resolved by residue patterns within the block,
  // never by scanning unrelated families or enumerating individual members.
  template<class Visit> void intersect(AffineDomain query,Visit visit) const {
    if(!query.count)return;
    if(query.last()>=count_)throw std::out_of_range("event progression domain");
    if(query.count==1) {
      auto [family,member]=locate(EventId(query.first));
      visit(family,AffineIntersection{0,member,0,0,1});return;
    }
    for(std::size_t start=0;start<query.count;) {
      const auto first=query.first+query.step*start;
      const auto band_it=std::prev(bands_.upper_bound(first));const auto& band=band_it->second;
      const auto end=band_it->first+band.width*band.count;
      const auto count=query.step?std::min<std::uint64_t>(query.count-start,1+(end-1-first)/query.step):query.count;
      const auto period=query.step?band.width/std::gcd<std::uint64_t>(band.width,query.step):1;
      for(std::size_t residue=0;residue<std::min<std::uint64_t>(period,count);++residue) {
        const auto members=1+(count-1-residue)/period;
        for(std::size_t done=0;done<members;) {
          const auto left=start+residue+period*done;
          auto [family,member]=locate(EventId(query.first+query.step*left));
          const auto& f=families_[family];
          const auto step=query.step*period/f.stride;
          const auto n=step?std::min<std::uint64_t>(members-done,1+(f.count-1-member)/step):members-done;
          visit(family,AffineIntersection{left,member,period,step,n});done+=n;
        }
      }
      start+=count;
    }
  }
  [[nodiscard]] std::size_t eventCount() const { return count_; }
  // HB construction only appends derived completion nodes to the frozen base.
  void appendSingletons(std::size_t count) {
    if(count<count_)throw std::logic_error("event family prefix was removed");
    if(count==count_)return;
    Band band{count-count_,1,{}};
    band.columns.reserve(band.width+1);
    for(auto id=count_;id<count;++id) {
      band.columns.push_back(families_.size());
      families_.push_back({EventId(static_cast<std::uint32_t>(id)),1,1});
    }
    band.columns.push_back(families_.size());
    bands_.emplace(count_,std::move(band));count_=count;
  }
 private:
  struct Band { std::size_t width, count; std::vector<std::size_t> columns; };
  std::size_t count_;
  std::map<std::size_t, Band> bands_;
  std::vector<Family> families_;
};

// Exact event-to-trace-rank map. Each family stores piecewise affine ranks.
class FamilyTraceIndex final {
 public:
  struct Run { std::size_t end, rank; std::int64_t step; };
  struct Prefix { std::size_t rank, count; std::int64_t step; };
  void reset(std::shared_ptr<const EventFamilies> layout) { layout_ = std::move(layout); }
  [[nodiscard]] bool enabled() const { return bool(layout_); }
  [[nodiscard]] std::size_t size() const { return size_; }
  [[nodiscard]] std::size_t runCount() const {
    std::size_t count = 0; for (const auto& [family, runs] : rows_) count += runs.size();
    return count;
  }
  void insert(EventId id, std::size_t rank, std::size_t count = 1) {
    const auto [family, member] = layout_->locate(id);
    auto& rows = rows_[family];
    auto next = rows.lower_bound(member);
    if ((next != rows.end() && next->first < member + count) ||
        (next != rows.begin() && std::prev(next)->second.end > member))
      throw std::logic_error("repeated event in reference trace");
    auto current = rows.emplace_hint(next, member, Run{member + count, rank, 1});
    if (current != rows.begin()) {
      auto previous = std::prev(current);
      const auto step = std::int64_t(rank) - std::int64_t(previous->second.rank);
      const auto length = previous->second.end - previous->first;
      const auto inferred = length == 1 ? step : previous->second.step;
      if (previous->second.end == member &&
          step == inferred * std::int64_t(length) && (count == 1 || inferred == 1)) {
        previous->second.step = inferred;
        previous->second.end = member + count; rows.erase(current);
      }
    }
    size_ += count;
  }
  [[nodiscard]] std::optional<std::size_t> get(EventId id) const {
    if (id.value() >= layout_->eventCount()) return {};
    const auto [family, member] = layout_->locate(id);
    const auto group = rows_.find(family);
    if (group == rows_.end()) return {};
    auto it = group->second.upper_bound(member);
    if (it == group->second.begin()) return {};
    --it;
    if (it->second.end <= member) return {};
    return it->second.rank + it->second.step * std::int64_t(member - it->first);
  }
  [[nodiscard]] Prefix prefix(EventId id, std::int64_t event_step, std::size_t count) const {
    const auto [family, member] = layout_->locate(id);
    const auto& rows = rows_.at(family);
    const auto next = rows.upper_bound(member);
    if (next == rows.begin()) throw std::logic_error("missing trace event");
    const auto& [first, run] = *std::prev(next);
    if (member >= run.end) throw std::logic_error("missing trace event");
    const auto rank = run.rank + run.step * std::int64_t(member - first);
    if (event_step == 0) return {rank, count, 0};
    if (event_step != layout_->families()[family].stride) return {rank, 1, 0};
    return {rank, std::min(count, run.end - member), run.step};
  }
  void erase(EventId id) {
    const auto [family, member] = layout_->locate(id);
    auto& rows = rows_.at(family);
    auto it = std::prev(rows.upper_bound(member));
    const auto run = it->second;
    if (member < it->first || member >= run.end) throw std::logic_error("missing trace event");
    if (member + 1 < run.end)
      rows.emplace(member + 1, Run{run.end, run.rank + run.step * std::int64_t(member + 1 - it->first), run.step});
    if (member == it->first) rows.erase(it); else it->second.end = member;
    --size_;
  }
 private:
  std::shared_ptr<const EventFamilies> layout_;
  std::map<std::size_t, std::map<std::size_t, Run>> rows_;
  std::size_t size_ = 0;
};

}  // namespace sentryward::algorithm
