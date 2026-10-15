#include "sentryward/algorithm/analysis/family_hb_reachability.h"
#include "sentryward/algorithm/analysis/alternate_topological_order.h"
#include "sentryward/algorithm/analysis/chain_frontier.h"
#include <array>
#include <bit>
#include <deque>
#include <unordered_map>

namespace sentryward::algorithm {
namespace {
std::shared_ptr<EventRelations> BuildTopology(const HappensBeforeGraph& graph) {
  ProfileScope profile("oracle.family.build_topology");
  // Only the small certified delta is copied; immutable base relations stay shared.
  return std::make_shared<EventRelations>(*graph.relationStorage());
}
class FamilyIndex final : public FamilyHbReachability {
 public:
  FamilyIndex(const HappensBeforeGraph& graph,HbReachabilityOptions options,bool reverse)
      : graph_(graph),version_(graph.version()),options_(options),reverse_(reverse),
        topology_owner_(BuildTopology(graph)),topology_(*topology_owner_),chains_(graph,topology_) {
    ProfileScope profile("oracle.family.alternate_order");
    alternate_.emplace(graph.events(),std::shared_ptr<const EventRelations>(topology_owner_));
    alternate_edge_count_=graph.edgeCount();profile.nodes=alternate_->runCount();
  }
  const HappensBeforeGraph& graph() const noexcept override { return graph_; }
  std::string_view name() const noexcept override { return "event-family-relations"; }
  HbReachabilityStats stats() const noexcept override { return stats_; }
  void RefreshAppendOnly() override {
    if(graph_.version()==version_)return;
    ProfileScope profile("oracle.family.append");
    if(alternate_ && alternate_->size()!=graph_.eventCount())alternate_.reset();
    if(alternate_)for(std::size_t i=alternate_edge_count_;i<graph_.edgeCount();++i) {
      const auto edge=graph_.edges()[i];
      if(alternate_->rank(edge.from)>=alternate_->rank(edge.to)) { alternate_.reset();break; }
    }
    alternate_edge_count_=graph_.edgeCount();
    topology_.append(graph_.eventCount(),graph_.edges());chains_.append();version_=graph_.version();
  }
  bool reaches(EventId from,EventId to) override {
    unchanged();validate(from);validate(to);++stats_.queries;
    if(auto answer=known(from,to))return *answer;
    const ChainQueryIndex::Request request{to,1,true};
    const auto answer=query(std::span(&from,1),std::span(&request,1))[0]!=0;
    remember(from,to,answer);return answer;
  }
  std::vector<std::uint8_t> reachesBatch(std::span<const HbQuery> queries) override {
    unchanged();++stats_.batches;stats_.queries+=queries.size();
    std::vector<std::uint8_t> answers(queries.size());
    std::vector<std::size_t> pending;
    std::unordered_map<std::uint32_t,std::size_t> sources,targets;
    for(std::size_t i=0;i<queries.size();++i) {
      const auto& q=queries[i];validate(q.from);validate(q.to);
      if(auto answer=known(q.from,q.to)) { answers[i]=*answer;continue; }
      pending.push_back(i);
      ++sources[q.from.value()];++targets[q.to.value()];
    }
    if(pending.empty())return answers;
    const bool prefer_transpose=targets.size()<sources.size();
    std::array<std::vector<std::size_t>,2> directions;
    for(auto index:pending) {
      const auto& q=queries[index];
      const auto from_uses=sources.at(q.from.value()),to_uses=targets.at(q.to.value());
      const bool transpose=to_uses==from_uses?prefer_transpose:to_uses>from_uses;
      directions[transpose].push_back(index);
    }
    const auto propagate=[&](bool transpose) {
      auto& pending_direction=directions[transpose];
      if(pending_direction.empty())return;
      std::unordered_map<std::uint32_t,std::size_t> ranks;
      const auto seed_of=[&](std::size_t index) { return transpose?queries[index].to:queries[index].from; };
      for(auto index:pending_direction)ranks.try_emplace(seed_of(index).value(),0);
      // Rank each distinct seed once; sorting must not decode ranks per comparison.
      for(auto& [id,rank]:ranks) {
        const EventId seed(id);
        rank=alternate_?alternate_->rank(seed):
            graph_.traceOrder()?graph_.traceOrder()->rank(seed):id;
      }
      std::sort(pending_direction.begin(),pending_direction.end(),[&](auto a,auto b) {
        const auto x=seed_of(a),y=seed_of(b);
        return std::tuple(ranks.at(x.value()),x,a)<std::tuple(ranks.at(y.value()),y,b);
      });
      const auto limit=std::min(std::size_t(64),options_.max_batch_sources);
      for(std::size_t begin=0;begin<pending_direction.size();) {
        auto end=begin;
        std::vector<EventId> seeds;
        std::vector<ChainQueryIndex::Request> requests;
        seeds.reserve(limit);
        while(end<pending_direction.size() && seeds.size()<limit) {
          const auto seed=seed_of(pending_direction[end]);
          const auto label=std::uint64_t(1)<<seeds.size();
          seeds.push_back(seed);
          do {
            const auto& q=queries[pending_direction[end++]];
            requests.push_back({transpose?q.from:q.to,label,true});
          } while(end<pending_direction.size() && seed_of(pending_direction[end])==seed);
        }
        ProfileScope batch_profile("oracle.family.pair_batch");
        batch_profile.nodes=seeds.size();batch_profile.edges=requests.size();
        const auto result=query(seeds,requests,transpose);
        for(std::size_t i=0;i<result.size();++i) {
          const auto index=pending_direction[begin+i];answers[index]=bool(result[i]);remember(queries[index].from,queries[index].to,answers[index]);
        }
        begin=end;
      }
    };
    propagate(false);propagate(true);
    return answers;
  }
  std::vector<std::uint8_t> allReachMany(std::span<const EventId> sources,std::span<const EventId> targets) override {
    unchanged();++stats_.batches;std::vector<std::uint8_t> result(targets.size());
    for(auto source:sources)if(!graph_.contains(source))return result;
    for(std::size_t i=0;i<targets.size();++i)result[i]=graph_.contains(targets[i]);
    const auto seeds=unique(sources);
    // Transpose the complete relation when targets are the smaller side.
    // One bit per target; intersect source answers to implement universal
    // quantification, rather than materializing sources x targets queries.
    if(targets.size()>1 && targets.size()<seeds.size() &&
        std::all_of(result.begin(),result.end(),[](auto valid){return valid!=0;})) {
      ProfileScope profile("oracle.family.all_relation_transpose");
      for(std::size_t begin=0;begin<targets.size();begin+=64) {
        const auto count=std::min(std::size_t(64),targets.size()-begin);
        const auto full=count==64?UINT64_MAX:(std::uint64_t(1)<<count)-1;
        std::vector<ChainQueryIndex::Request> requests;
        requests.reserve(seeds.size());
        for(auto source:seeds)requests.push_back({source,full,true});
        const auto answers=query(targets.subspan(begin,count),requests,true);
        auto common=full;
        for(auto answer:answers)common&=answer;
        for(std::size_t i=0;i<count;++i)result[begin+i]=result[begin+i] && ((common>>i)&1);
        profile.nodes+=count;profile.edges+=requests.size();
      }
      return result;
    }
    if(targets.size()==1 && result[0]) {
      std::vector<ChainQueryIndex::Request> requests;
      for(auto source:seeds) {
        if(auto answer=known(source,targets[0])) { if(!*answer)result[0]=false; }
        else requests.push_back({source,1,true});
      }
      if(!result[0])return result;
      const auto answers=query(targets,requests,true);
      for(std::size_t i=0;i<answers.size();++i)remember(requests[i].target,targets[0],answers[i]!=0);
      result[0]=std::all_of(answers.begin(),answers.end(),[](auto value){return value==1;});return result;
    }
    for(std::size_t begin=0;begin<seeds.size();begin+=64) {
      const auto count=std::min(std::size_t(64),seeds.size()-begin);
      const auto full=count==64?UINT64_MAX:(std::uint64_t(1)<<count)-1;
      std::vector<ChainQueryIndex::Request> requests;std::vector<std::size_t> indexes;
      for(std::size_t i=0;i<targets.size();++i)if(result[i]) { requests.push_back({targets[i],full,true});indexes.push_back(i); }
      const auto answers=query(std::span(seeds).subspan(begin,count),requests);
      for(std::size_t i=0;i<answers.size();++i)result[indexes[i]]=answers[i]==full;
    }
    return result;
  }
  std::vector<std::optional<EventId>> findAncestors(std::span<const EventId> sources,std::span<const EventId> targets) override {
    unchanged();++stats_.batches;std::vector<std::optional<EventId>> result(targets.size());const auto seeds=unique(sources);
    if(targets.size()==1 && graph_.contains(targets[0])) {
      std::vector<ChainQueryIndex::Request> requests;
      for(auto source:seeds)requests.push_back({source,1,true});
      const auto answers=query(targets,requests,true,true);
      for(std::size_t i=0;i<answers.size();++i)if(answers[i]) { result[0]=seeds[i];break; }
      return result;
    }
    for(std::size_t begin=0;begin<seeds.size();begin+=64) {
      const auto count=std::min(std::size_t(64),seeds.size()-begin);
      const auto full=count==64?UINT64_MAX:(std::uint64_t(1)<<count)-1;
      std::vector<ChainQueryIndex::Request> requests;std::vector<std::size_t> indexes;
      for(std::size_t i=0;i<targets.size();++i)if(!result[i] && graph_.contains(targets[i])) { requests.push_back({targets[i],full,false});indexes.push_back(i); }
      const auto answers=query(std::span(seeds).subspan(begin,count),requests);
      for(std::size_t i=0;i<answers.size();++i)if(answers[i])result[indexes[i]]=seeds[begin+std::countr_zero(answers[i])];
    }
    return result;
  }
 private:
  void unchanged() const { if(graph_.version()!=version_)throw std::logic_error("graph changed during family HB query"); }
  void validate(EventId event) const { if(!graph_.contains(event))throw std::invalid_argument("HB query references an unknown event"); }
  std::vector<EventId> unique(std::span<const EventId> values) const {
    std::vector<EventId> result;for(auto value:values)if(graph_.contains(value))result.push_back(value);
    std::sort(result.begin(),result.end());result.erase(std::unique(result.begin(),result.end()),result.end());return result;
  }
  static std::uint64_t key(EventId from,EventId to) { return (std::uint64_t(from.value())<<32)|to.value(); }
  std::optional<bool> known(EventId from,EventId to) {
    if(auto answer=chains_.sameChain(reverse_?to:from,reverse_?from:to)) { ++stats_.filtered_queries;return answer; }
    const auto a=reverse_?to:from,b=reverse_?from:to;
    if((graph_.traceOrder() && graph_.traceOrder()->rank(a)>=graph_.traceOrder()->rank(b)) ||
        (alternate_ && alternate_->rank(a)>=alternate_->rank(b))) { ++stats_.filtered_queries;return false; }
    if(auto it=cache_.find(key(from,to));it!=cache_.end() && (it->second.answer || it->second.version==version_)) { ++stats_.cache_hits;return it->second.answer; }
    return {};
  }
  void remember(EventId from,EventId to,bool answer) {
    const auto limit=std::min(options_.cached_pairs,options_.index_budget_bytes/64);if(!limit)return;
    const auto id=key(from,to);
    if(auto it=cache_.find(id);it!=cache_.end()) { it->second={answer,version_};return; }
    while(cache_.size()>=limit) { cache_.erase(cache_fifo_.front());cache_fifo_.pop_front(); }
    cache_.emplace(id,CacheEntry{answer,version_});cache_fifo_.push_back(id);
    stats_.peak_index_bytes=std::max(stats_.peak_index_bytes,cache_.size()*64);
  }
  std::vector<std::uint64_t> query(std::span<const EventId> seeds,std::span<const ChainQueryIndex::Request> requests,
      bool transpose=false,bool stop_on_any=false) {
    if(requests.empty())return {};
    std::size_t lo1=SIZE_MAX,hi1=0,lo2=SIZE_MAX,hi2=0;
    const auto endpoint=[&](EventId event) {
      if(const auto* order=graph_.traceOrder()) { const auto r=order->rank(event);lo1=std::min(lo1,r);hi1=std::max(hi1,r); }
      if(alternate_) { const auto r=alternate_->rank(event);lo2=std::min(lo2,r);hi2=std::max(hi2,r); }
    };
    for(auto seed:seeds)endpoint(seed);for(const auto& request:requests)endpoint(request.target);
    if(rank_bounds_.size()<topology_.layout()->families().size())rank_bounds_.resize(topology_.layout()->families().size(),{SIZE_MAX,0});
    return chains_.query(seeds,requests,[&](std::size_t family) {
      if(const auto* order=graph_.traceOrder()) {
        auto& bounds=rank_bounds_[family];
        if(bounds.first==SIZE_MAX) { const auto& group=topology_.layout()->families()[family];bounds=order->rankBounds(group.first,group.stride,group.count); }
        if(bounds.second<lo1 || bounds.first>hi1)return false;
      }
      if(alternate_) { const auto bounds=alternate_->familyBounds(family);if(bounds.second<lo2 || bounds.first>hi2)return false; }
      return true;
    },stats_.edge_visits,transpose!=reverse_,stop_on_any);
  }
  const HappensBeforeGraph& graph_;std::size_t version_;HbReachabilityOptions options_;bool reverse_;
  std::shared_ptr<EventRelations> topology_owner_;EventRelations& topology_;ChainQueryIndex chains_;
  std::optional<AlternateTopologicalOrder> alternate_;std::size_t alternate_edge_count_=0;
  struct CacheEntry { bool answer;std::size_t version; };
  std::unordered_map<std::uint64_t,CacheEntry> cache_;std::deque<std::uint64_t> cache_fifo_;
  std::vector<std::pair<std::size_t,std::size_t>> rank_bounds_;HbReachabilityStats stats_;
};
}
std::unique_ptr<FamilyHbReachability> TryFamilyHbReachability(const HappensBeforeGraph& graph,const HbReachabilityOptions& options,bool reverse) {
  if(options.max_batch_sources==0)throw std::invalid_argument("max_batch_sources must be positive");
  if(options.index_budget_bytes<1024*1024)return nullptr;
  if(graph.events().storedBindingCount()*2>=graph.eventCount())return nullptr;
  const EventFamilies families(graph.events());if(families.families().size()*2>=graph.eventCount())return nullptr;
  return std::make_unique<FamilyIndex>(graph,options,reverse);
}
}  // namespace sentryward::algorithm
