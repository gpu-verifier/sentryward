#pragma once
#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include "sentryward/algorithm/base/template_records.h"

namespace sentryward::value_analysis {
struct MemoryInstance final {
  std::size_t node = 0;
  std::size_t instance = 0;
  friend auto operator<=>(const MemoryInstance&, const MemoryInstance&) = default;
};

// Allocation identity includes address space, owner, and lifetime.
struct ConstantRegion final {
  std::string allocation;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  friend auto operator<=>(const ConstantRegion&, const ConstantRegion&) = default;
};

struct OrderedMemoryAction final {
  MemoryInstance operation;
  std::optional<ConstantRegion> region;
  std::string unsupported_reason;
  // Byte offset in the logical value when a native access has several fragments.
  std::optional<std::uint64_t> value_byte_offset = std::nullopt;
  std::optional<std::size_t> duplicate_write_group = std::nullopt;
  // Object identity and payload size only: offset is relative to the original
  // Load address operand, NOT a constant allocation-relative memory region.
  std::optional<ConstantRegion> symbolic_region = std::nullopt;
  friend bool operator==(const OrderedMemoryAction&,const OrderedMemoryAction&)=default;
};


// Consecutive logical actions retain their order and individual instance IDs.
struct MemoryActionCodec final {
  using Fields=std::array<std::uint64_t,3>;
  using Binding=Fields;
  using Delta=std::array<std::int64_t,3>;
  static Fields fields(const OrderedMemoryAction& action) {
    return {action.operation.instance,action.region?action.region->offset:0,action.value_byte_offset.value_or(0)};
  }
  static bool compatible(const OrderedMemoryAction& a,const OrderedMemoryAction& b) {
    return a.operation.node==b.operation.node && a.region.has_value()==b.region.has_value() &&
      (!a.region || (a.region->allocation==b.region->allocation && a.region->bytes==b.region->bytes)) &&
      a.value_byte_offset.has_value()==b.value_byte_offset.has_value() &&
      a.unsupported_reason==b.unsupported_reason && a.duplicate_write_group==b.duplicate_write_group &&
      a.symbolic_region==b.symbolic_region;
  }
  static bool split(OrderedMemoryAction& action,Binding& binding) {
    binding=fields(action);
    action.operation.instance=0;
    if(action.region)action.region->offset=0;
    if(action.value_byte_offset)action.value_byte_offset=0;
    return true;
  }
  static void bind(OrderedMemoryAction& action,const Binding& binding) {
    action.operation.instance=binding[0];
    if(action.region)action.region->offset=binding[1];
    if(action.value_byte_offset)action.value_byte_offset=binding[2];
  }
  static std::size_t hash(const OrderedMemoryAction& action) {
    auto h=std::hash<std::size_t>{}(action.operation.node);
    const auto add=[&](auto x){algorithm::HashCombine(h,std::hash<decltype(x)>{}(x));};
    add(action.region.has_value());
    if(action.region){add(action.region->allocation);add(action.region->bytes);}
    add(action.value_byte_offset.has_value());add(action.unsupported_reason);
    add(action.duplicate_write_group.has_value());
    if(action.duplicate_write_group)add(*action.duplicate_write_group);
    add(action.symbolic_region.has_value());
    if(action.symbolic_region){add(action.symbolic_region->allocation);add(action.symbolic_region->offset);add(action.symbolic_region->bytes);}
    return h;
  }
  static std::optional<Delta> difference(const Fields& a,const Fields& b) {
    Delta delta{};
    for(std::size_t i=0;i<a.size();++i) {
      auto magnitude=a[i]>b[i]?a[i]-b[i]:b[i]-a[i];
      if(magnitude>std::uint64_t(INT64_MAX))return {};
      delta[i]=a[i]>b[i]?-static_cast<std::int64_t>(magnitude):static_cast<std::int64_t>(magnitude);
    }
    return delta;
  }
  static std::optional<Fields> translate(Fields values,const Delta& delta,std::size_t index) {
    for(std::size_t i=0;i<values.size();++i) {
      auto magnitude=static_cast<std::uint64_t>(delta[i]<0?-delta[i]:delta[i]);
      auto available=delta[i]<0?values[i]:UINT64_MAX-values[i];
      if(magnitude && index>available/magnitude)return {};
      if(delta[i]<0)values[i]-=magnitude*index;else values[i]+=magnitude*index;
    }
    return values;
  }
};

class MemoryActions final {
  algorithm::TemplateRecords<OrderedMemoryAction,MemoryActionCodec> records_;
  bool explicit_thread_=false;
 public:
  MemoryActions()=default;
  MemoryActions(const MemoryActions&)=default;
  MemoryActions& operator=(const MemoryActions&)=default;
  MemoryActions(MemoryActions&& other) noexcept
      :records_(std::move(other.records_)),explicit_thread_(std::exchange(other.explicit_thread_,false)){}
  MemoryActions& operator=(MemoryActions&& other) noexcept {
    if(this!=&other){records_=std::move(other.records_);explicit_thread_=std::exchange(other.explicit_thread_,false);}
    return *this;
  }
  std::size_t size() const noexcept{return records_.size();}
  bool empty() const noexcept{return records_.empty();}
  std::size_t storedRunCount() const noexcept{return records_.bindingTemplateCount();}
  std::size_t storedBindingCount() const noexcept{return records_.storedBindingCount();}
  std::size_t beginThread(){explicit_thread_=true;return records_.beginThread();}
  void endThread(std::size_t begin,std::uint64_t scope,std::uintptr_t history) {
    records_.compressThreadTail(begin,scope,history);explicit_thread_=false;
  }
  template<class Visitor> void forEachAffineColumn(Visitor&& visit) const {
    records_.forEachAffineColumn(std::forward<Visitor>(visit));
  }
  using CachedColumn=algorithm::TemplateRecords<OrderedMemoryAction,MemoryActionCodec>::CachedColumn;
  template<class Visitor> void forEachCheckpointColumn(Visitor&& visit) const {
    records_.forEachAffineColumnWithBlock(std::forward<Visitor>(visit));
  }
  void appendCachedBlock(std::size_t width,std::size_t members,const std::vector<CachedColumn>& columns) {
    records_.appendCachedBlock(width,members,columns);
  }
  void push_back(OrderedMemoryAction action) {
    if(explicit_thread_){records_.push_back(std::move(action));return;}
    const auto begin=records_.beginThread();records_.push_back(std::move(action));
    records_.compressThreadTail(begin,0,0);
  }
  void appendAffine(OrderedMemoryAction action,const MemoryActionCodec::Delta& delta,std::size_t count) {
    records_.appendAffine(std::move(action),delta,count);
  }
  OrderedMemoryAction at(std::size_t index) const {
    return records_.at(index);
  }
  OrderedMemoryAction operator[](std::size_t index) const{return at(index);}
};

// Borrowed, replayable view; decoding does not allocate an expanded action list.
class MemoryActionRange final : public std::ranges::transform_view<
    std::ranges::iota_view<std::size_t,std::size_t>,std::function<OrderedMemoryAction(std::size_t)>> {
  using Base=std::ranges::transform_view<std::ranges::iota_view<std::size_t,std::size_t>,
      std::function<OrderedMemoryAction(std::size_t)>>;
  const MemoryActions* compact_=nullptr;
 public:
  MemoryActionRange():MemoryActionRange(std::span<const OrderedMemoryAction>{}){}
  MemoryActionRange(std::span<const OrderedMemoryAction> actions)
      :Base(std::views::iota(std::size_t{0},actions.size()),[actions](auto i){return actions[i];}){}
  MemoryActionRange(const std::vector<OrderedMemoryAction>& actions):MemoryActionRange(std::span(actions)){}
  template<std::size_t N> MemoryActionRange(const std::array<OrderedMemoryAction,N>& actions):MemoryActionRange(std::span<const OrderedMemoryAction>(actions)){}
  MemoryActionRange(std::initializer_list<OrderedMemoryAction> actions):MemoryActionRange(std::span(actions.begin(),actions.size())){}
  MemoryActionRange(const MemoryActions& actions)
      :Base(std::views::iota(std::size_t{0},actions.size()),[&actions](auto i){return actions.at(i);}),compact_(&actions){}
  const MemoryActions* compact() const noexcept{return compact_;}
};
} // namespace sentryward::value_analysis
