#include "sentryward/algorithm/analysis/compact_hb_reachability.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sentryward::algorithm {
namespace {
struct BudgetExceeded {};
class CompactIndex final : public HbReachability {
  struct Node { std::uint32_t left=0,right=0,value=0; };
 public:
  CompactIndex(const HappensBeforeGraph& graph,const HbReachabilityOptions& options)
      : graph_(graph),version_(graph.version()),count_(graph.eventCount()) {
    if(count_>std::numeric_limits<std::uint32_t>::max() ||
        count_>options.index_budget_bytes/32)throw BudgetExceeded{};
    base_bytes_=32*count_;
    max_nodes_=std::min<std::size_t>((options.index_budget_bytes-base_bytes_)/sizeof(Node),
                                   std::numeric_limits<std::uint32_t>::max());
    if(!max_nodes_)throw BudgetExceeded{};
    work_limit_=64*count_+4*max_nodes_;
    auto order=graph.topologicalOrder();
    if(!order)throw std::invalid_argument("HB index requires an acyclic graph");
    chains_.resize(count_);positions_.resize(count_);roots_.resize(count_);
    std::vector<std::uint32_t> tails;tails.reserve(count_);
    for(auto event:*order) {
      std::optional<std::uint32_t> previous;
      for(auto pred:graph.predecessors(event)) {
        tick();++stats_.edge_visits;
        if(tails[chains_[pred.value()]]==pred.value() &&
            (!previous || positions_[pred.value()]>positions_[*previous]))
          previous=pred.value();
      }
      const auto id=event.value();
      if(previous) {
        chains_[id]=chains_[*previous];positions_[id]=positions_[*previous]+1;
        tails[chains_[id]]=id;
      } else {
        chains_[id]=tails.size();positions_[id]=1;tails.push_back(id);
      }
    }
    chain_count_=tails.size();
    allocate({});
    for(auto event:*order) {
      auto root=std::uint32_t{0};
      for(auto pred:graph.predecessors(event)) {
        tick();++stats_.edge_visits;
        const auto id=pred.value();
        if(chains_[id]==chains_[event.value()]) {
          root=merge(root,roots_[id],0,chain_count_);
          continue;
        }
        // If pred is already represented, all its ancestors are represented too.
        if(get(root,chains_[id])>=positions_[id])continue;
        root=merge(root,roots_[id],0,chain_count_);
        root=set(root,0,chain_count_,chains_[id],positions_[id]);
      }
      roots_[event.value()]=root;
    }
  }
  const HappensBeforeGraph& graph() const noexcept override {return graph_;}
  std::string_view name() const noexcept override {return "persistent-chain";}
  HbReachabilityStats stats() const noexcept override {return stats_;}
  std::vector<std::uint8_t> reachesBatch(std::span<const HbQuery> queries) override {
    unchanged();
    return HbReachability::reachesBatch(queries);
  }
  bool reaches(EventId from,EventId to) override {
    unchanged();
    if(!graph_.contains(from) || !graph_.contains(to))
      throw std::invalid_argument("HB query references an unknown event");
    ++stats_.queries;
    if(chains_[from.value()]==chains_[to.value()])
      return positions_[from.value()]<=positions_[to.value()];
    return get(roots_[to.value()],chains_[from.value()])>=positions_[from.value()];
  }
 private:
  void unchanged() const {
    if(graph_.version()!=version_ || graph_.eventCount()!=count_)
      throw std::logic_error("graph changed during HB index use");
  }
  void tick() {if(++work_>work_limit_)throw BudgetExceeded{};}
  std::uint32_t allocate(Node node) {
    if(nodes_.size()==max_nodes_)throw BudgetExceeded{};
    if(nodes_.size()==nodes_.capacity()) {
      const auto old_capacity=nodes_.capacity();
      const auto capacity=std::min(max_nodes_-old_capacity,
                                  std::max<std::size_t>(16,old_capacity*2));
      if(capacity<=old_capacity)throw BudgetExceeded{};
      stats_.peak_index_bytes=std::max(stats_.peak_index_bytes,
          base_bytes_+(old_capacity+capacity)*sizeof(Node));
      nodes_.reserve(capacity);
    }
    const auto id=static_cast<std::uint32_t>(nodes_.size());nodes_.push_back(node);return id;
  }
  std::uint32_t get(std::uint32_t root,std::uint32_t chain) const {
    std::size_t lo=0,hi=chain_count_;
    while(root && hi-lo>1) {
      const auto mid=lo+(hi-lo)/2;
      if(chain<mid){root=nodes_[root].left;hi=mid;}
      else {root=nodes_[root].right;lo=mid;}
    }
    return root?nodes_[root].value:0;
  }
  std::uint32_t set(std::uint32_t root,std::size_t lo,std::size_t hi,
                    std::uint32_t chain,std::uint32_t value) {
    tick();auto node=nodes_[root];
    if(hi-lo==1) {
      if(node.value>=value)return root;
      node.value=value;return allocate(node);
    }
    const auto mid=lo+(hi-lo)/2;
    if(chain<mid)node.left=set(node.left,lo,mid,chain,value);
    else node.right=set(node.right,mid,hi,chain,value);
    return allocate(node);
  }
  std::uint32_t merge(std::uint32_t a,std::uint32_t b,std::size_t lo,std::size_t hi) {
    tick();
    if(a==b || !b)return a;
    if(!a)return b;
    const auto x=nodes_[a],y=nodes_[b];
    if(hi-lo==1)return x.value>=y.value?a:b;
    const auto mid=lo+(hi-lo)/2;
    const auto left=merge(x.left,y.left,lo,mid),right=merge(x.right,y.right,mid,hi);
    if(left==x.left && right==x.right)return a;
    if(left==y.left && right==y.right)return b;
    return allocate({left,right,0});
  }
  const HappensBeforeGraph& graph_;
  std::size_t version_,count_,chain_count_=0,base_bytes_=0,max_nodes_=0,work_=0,work_limit_=0;
  std::vector<std::uint32_t> chains_,positions_,roots_;
  std::vector<Node> nodes_;
  HbReachabilityStats stats_;
};
}
std::unique_ptr<HbReachability> TryCompactHbReachability(
    const HappensBeforeGraph& graph,const HbReachabilityOptions& options) {
  try {return std::make_unique<CompactIndex>(graph,options);}
  catch(const BudgetExceeded&) {return {};}
}
}
