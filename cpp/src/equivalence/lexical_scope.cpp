#include "sentryward/equivalence/lexical_scope.h"

#include <bit>
#include <deque>
#include <functional>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sentryward::equivalence::lexical_scope {
struct Catalog::Impl {
  Limits limits;
  Catalog::Charge charge_fn;
  Statistics stats;
  std::deque<std::vector<Count>> scopes{{}}; // Slot zero is Unknown.
  std::map<std::vector<Count>,Id> keys;
  struct Binding {
    Id parent;std::uint32_t binder;std::uint64_t count;
    bool operator==(const Binding&) const = default;
  };
  struct BindingHash {
    std::size_t operator()(const Binding& key) const noexcept {
      auto hash=std::hash<Id>{}(key.parent);
      const auto mix=[&](std::size_t value) {
        hash^=value+std::size_t{0x9e3779b9}+(hash<<6)+(hash>>2);
      };
      mix(std::hash<std::uint32_t>{}(key.binder));
      mix(std::hash<std::uint64_t>{}(key.count));return hash;
    }
  };
  std::unordered_map<Binding,Id,BindingHash> bindings;

  explicit Impl(Limits l,Catalog::Charge callback):limits(l),charge_fn(std::move(callback)) {
    if(!l.max_scopes || l.max_scopes>1024 || !l.max_bindings ||
       l.max_bindings>16384 || !l.max_metadata_slots ||
       l.max_metadata_slots>131072 || !l.max_work || l.max_work>1048576 ||
       !l.max_levels || l.max_levels>64)
      throw std::invalid_argument("lexical_scope_invalid_limits");
  }
  bool charge(std::size_t amount) {
    if(amount>limits.max_work-stats.work || (charge_fn && !charge_fn(amount))) {
      stats.exhausted=true;return false;
    }
    stats.work+=amount;return true;
  }
  std::optional<Id> intern(std::vector<Count> key,std::size_t reserved_slots=0) {
    // Full-key comparisons/copies are charged by lexical depth and map height.
    if(!charge(4*(key.size()+1)*(1+std::bit_width(keys.size()+1))))return {};
    if(auto found=keys.find(key);found!=keys.end())return found->second;
    const auto slots=2*(key.size()+1)+4;
    if(stats.scopes>=limits.max_scopes ||
       reserved_slots>limits.max_metadata_slots-stats.metadata_slots ||
       slots>limits.max_metadata_slots-stats.metadata_slots-reserved_slots) {
      stats.exhausted=true;return {};
    }
    const auto id=scopes.size();
    scopes.push_back(key);keys.emplace(std::move(key),id);
    ++stats.scopes;stats.metadata_slots+=slots;return id;
  }
};
Catalog::Catalog(Limits limits,Charge callback)
    :impl_(std::make_unique<Impl>(limits,std::move(callback))) {}
Catalog::~Catalog()=default;
Catalog::Catalog(Catalog&&) noexcept=default;
Catalog& Catalog::operator=(Catalog&&) noexcept=default;

std::optional<Id> Catalog::root(std::uint32_t rank,
                              std::span<const std::uint64_t> prefix) {
  if(!impl_ || prefix.size()>rank)return {};
  if(rank>impl_->limits.max_levels) {impl_->stats.exhausted=true;return {};}
  if(!impl_->charge(rank+prefix.size()+1))return {};
  for(auto count:prefix)if(!count)return {};
  std::vector<Count> key(rank);
  for(std::size_t i=0;i<prefix.size();++i)key[i]=prefix[i];
  return impl_->intern(std::move(key));
}
std::optional<Id> Catalog::bindBody(Id parent,std::uint32_t binder,
                                   std::uint64_t count) {
  if(!impl_ || parent==unknown || parent>=impl_->scopes.size())return {};
  const auto& scope=impl_->scopes[parent];
  if(binder!=scope.size())return {}; // Absolute lexical levels, no rebinding.
  const Impl::Binding key{parent,binder,count};
  // Cached valid bindings survive subsequent optional cap exhaustion.
  if(auto found=impl_->bindings.find(key);found!=impl_->bindings.end())return found->second;
  if(scope.size()>=impl_->limits.max_levels ||
     impl_->stats.bindings>=impl_->limits.max_bindings ||
     4>impl_->limits.max_metadata_slots-impl_->stats.metadata_slots) {
    impl_->stats.exhausted=true;return {};
  }
  if(!impl_->charge(scope.size()+1))return {};
  auto next=scope;next.emplace_back(count);
  auto id=impl_->intern(std::move(next),4);if(!id)return {};
  impl_->bindings.emplace(key,*id);++impl_->stats.bindings;
  impl_->stats.metadata_slots+=4;return id;
}
std::optional<Id> Catalog::prefix(Id actual,std::uint32_t rank) {
  if(!impl_ || actual==unknown || actual>=impl_->scopes.size())return {};
  const auto& scope=impl_->scopes[actual];
  if(rank>scope.size())return {};
  if(rank==scope.size())return actual;
  // rank<size makes these keys disjoint from bindBody (binder==size), so
  // prefixes and body bindings share ONE metadata/work/key-count budget.
  const Impl::Binding key{actual,rank,0};
  if(auto found=impl_->bindings.find(key);found!=impl_->bindings.end())return found->second;
  if(impl_->stats.bindings>=impl_->limits.max_bindings ||
      4>impl_->limits.max_metadata_slots-impl_->stats.metadata_slots) {
    impl_->stats.exhausted=true;return {};
  }
  if(!impl_->charge(rank+1))return {};
  std::vector<Count> next(scope.begin(),scope.begin()+rank);
  auto id=impl_->intern(std::move(next),4);if(!id)return {};
  impl_->bindings.emplace(key,*id);++impl_->stats.bindings;
  impl_->stats.metadata_slots+=4;return id;
}
std::optional<std::uint32_t> Catalog::rank(Id id) const noexcept {
  if(!impl_ || id==unknown || id>=impl_->scopes.size())return {};
  return static_cast<std::uint32_t>(impl_->scopes[id].size());
}
Count Catalog::count(Id id,std::uint32_t level) const noexcept {
  if(!impl_ || id==unknown || id>=impl_->scopes.size() ||
     level>=impl_->scopes[id].size())return {};
  return impl_->scopes[id][level];
}
std::optional<std::span<const Count>> Catalog::counts(Id id) const noexcept {
  if(!impl_ || id==unknown || id>=impl_->scopes.size())return {};
  return std::span<const Count>{impl_->scopes[id]};
}
Statistics Catalog::statistics() const noexcept {
  return impl_?impl_->stats:Statistics{};
}
} // namespace sentryward::equivalence::lexical_scope
