#pragma once
#include <algorithm>
#include <deque>
#include <map>
#include <numeric>
#include <unordered_map>
#include "sentryward/algorithm/analysis/family_coordinates.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"

namespace sentryward::algorithm {
struct FrontierRectangle final { MemberSet members; std::uint64_t labels; };
struct FrontierAdvance final {
  std::size_t begin,end;
  MemberSet members;
  std::uint64_t labels;
};

// Rectangles partition (member, source-label) pairs; each has its own earliest position.
class ChainFrontier final {
 public:
  struct Cell { std::size_t position; MemberSet members; std::uint64_t labels; };
  std::vector<FrontierAdvance> add(std::size_t position,std::size_t end,
      MemberSet members,std::uint64_t labels) {
    if(members.empty() || !labels)return {};
    if(cells_.empty()) {
      cells_.push_back({position,members,labels});
      return {{position,end,std::move(members),labels}};
    }
    if(cells_.size()==1 && cells_[0].labels==labels && cells_[0].members==members) {
      auto& cell=cells_[0];
      if(position>=cell.position)return {};
      const auto previous=cell.position;cell.position=position;
      return {{position,previous,std::move(members),labels}};
    }
    std::vector<FrontierRectangle> fresh{{std::move(members),labels}};
    for(const auto& cell:cells_)if(cell.position<=position)
      fresh=subtract(fresh,cell.members,cell.labels);
    if(fresh.empty())return {};
    auto uncovered=fresh;
    std::vector<Cell> retained;
    std::vector<FrontierAdvance> result;
    for(auto& cell:cells_) {
      if(cell.position<=position) { retained.push_back(std::move(cell));continue; }
      std::vector<FrontierRectangle> rest{{cell.members,cell.labels}};
      for(const auto& item:fresh) {
        const auto mask=cell.members.intersect(item.members);
        const auto shared=cell.labels&item.labels;
        if(!mask.empty() && shared) {
          result.push_back({position,cell.position,mask,shared});
          uncovered=subtract(uncovered,mask,shared);
          rest=subtract(rest,mask,shared);
        }
      }
      for(auto& item:rest)retained.push_back({cell.position,std::move(item.members),item.labels});
    }
    for(const auto& item:uncovered)result.push_back({position,end,item.members,item.labels});
    for(auto& item:fresh)retained.push_back({position,std::move(item.members),item.labels});
    cells_.clear();
    for(auto& cell:retained) {
      auto found=std::find_if(cells_.begin(),cells_.end(),[&](const Cell& old) {
        return old.position==cell.position && (old.labels==cell.labels || old.members==cell.members);
      });
      if(found==cells_.end())cells_.push_back(std::move(cell));
      else if(found->labels==cell.labels)found->members.unite(cell.members);
      else found->labels|=cell.labels;
    }
    return result;
  }
  std::size_t fragments() const { return cells_.size(); }
 private:
  static std::vector<FrontierRectangle> subtract(const std::vector<FrontierRectangle>& items,
      const MemberSet& mask,std::uint64_t labels) {
    std::vector<FrontierRectangle> result;
    for(const auto& item:items) {
      if(!(labels&item.labels)) { result.push_back(item);continue; }
      auto intersection=item.members.intersect(mask);
      auto outside=item.members.subtract(mask);
      if(!outside.empty())result.push_back({std::move(outside),item.labels});
      if(!intersection.empty() && (item.labels&~labels))
        result.push_back({std::move(intersection),item.labels&~labels});
    }
    return result;
  }
  std::vector<Cell> cells_;
};

class ChainQueryIndex final {
 public:
  struct Request { EventId target; std::uint64_t labels; bool all; };
  using Filter = std::function<bool(std::size_t)>;
  ChainQueryIndex(const HappensBeforeGraph& graph,const EventRelations& topology)
      : topology_(topology),coordinates_(graph,topology,true) {
    ProfileScope profile("oracle.chain.build");
    HbIndexCensus("chains.begin",this,topology_.relations().size());
    const auto n=topology_.layout()->families().size();
    {
    std::vector<std::size_t> next(n,SIZE_MAX),previous(n,SIZE_MAX);
    HbIndexCensus("chains.next_previous",this,n,(next.capacity()+previous.capacity())*sizeof(std::size_t));
    for(std::size_t i=0;i<topology_.relations().size();++i) {
      const auto& edge=topology_.relations()[i];
      if(edge.source==edge.target || next[edge.source]!=SIZE_MAX || previous[edge.target]!=SIZE_MAX ||
          !identity(edge) || edge.first_source || edge.first_target ||
          edge.count!=coordinates_[edge.source].count || edge.count!=coordinates_[edge.target].count)continue;
      next[edge.source]=edge.target;previous[edge.target]=edge.source;
    }
    family_chain_.resize(n,SIZE_MAX);position_.resize(n);
    for(std::size_t root=0;root<n;++root)if(previous[root]==SIZE_MAX) {
      const auto chain=chains_.size();chains_.push_back({});
      for(auto f=root;f!=SIZE_MAX;f=next[f]) {
        if(family_chain_[f]!=SIZE_MAX)throw std::logic_error("cyclic dependency chain");
        family_chain_[f]=chain;position_[f]=chains_.back().size();chains_.back().push_back(f);
      }
    }
    for(auto chain:family_chain_)if(chain==SIZE_MAX)throw std::logic_error("cyclic dependency chain");
    }
    HbIndexVector("chains.family_chain",this,family_chain_);
    HbIndexVector("chains.positions",this,position_);
    HbIndexRows("chains.members",this,chains_);
    exits_.resize(chains_.size());reverse_exits_.resize(chains_.size());appendExits();
    HbIndexCensus("chains.end",this,chains_.size());
    profile.nodes=chains_.size();profile.edges=relation_count_;
  }
  void append() {
    coordinates_.append(topology_);
    while(family_chain_.size()<topology_.layout()->families().size()) {
      const auto f=family_chain_.size();family_chain_.push_back(chains_.size());position_.push_back(0);
      chains_.push_back({f});exits_.push_back({});reverse_exits_.push_back({});
    }
    appendExits();
  }
  std::optional<bool> sameChain(EventId from,EventId to) const {
    const auto [a,i]=topology_.layout()->locate(from);const auto [b,j]=topology_.layout()->locate(to);
    if(family_chain_[a]!=family_chain_[b] || coordinates_[a].coordinate(i)!=coordinates_[b].coordinate(j))return {};
    return position_[a]<=position_[b];
  }
  std::vector<std::uint64_t> query(std::span<const EventId> seeds,std::span<const Request> requests,
      const Filter& filter,std::size_t& edge_visits,bool transpose=false,bool stop_on_any=false) const {
    if(seeds.size()>64)throw std::invalid_argument("chain source-label chunk exceeds 64");
    ProfileScope profile("oracle.chain.query");
    std::vector<std::uint64_t> answers(requests.size());
    const auto position=[&](std::size_t f) { return transpose?chains_[family_chain_[f]].size()-1-position_[f]:position_[f]; };
    struct Target { std::size_t position,member,index; };
    std::unordered_map<std::size_t,std::vector<Target>> targets;
    std::size_t remaining=0;
    std::vector<bool> done(requests.size());
    for(std::size_t i=0;i<requests.size();++i) {
      if(!requests[i].labels) { done[i]=true;continue; }
      auto [f,m]=topology_.layout()->locate(requests[i].target);
      targets[family_chain_[f]].push_back({position(f),coordinates_[f].coordinate(m),i});++remaining;
    }
    for(auto& [chain,items]:targets)std::sort(items.begin(),items.end(),[](auto a,auto b){return a.position<b.position;});
    std::unordered_map<std::size_t,ChainFrontier> frontiers;
    std::unordered_map<std::size_t,std::vector<FrontierAdvance>> pending;
    std::deque<std::size_t> queue;
    std::size_t irregular=0,peak_fragments=0;
    const auto add=[&](std::size_t f,MemberSet mask,std::uint64_t labels) {
      if(mask.empty() || !labels)return;
      if(filter && !filter(f)) { ++profile.pruned;return; }
      const auto chain=family_chain_[f];
      auto& frontier=frontiers[chain];
      auto advances=frontier.add(position(f),chains_[chain].size(),std::move(mask),labels);
      peak_fragments=std::max(peak_fragments,frontier.fragments());
      for(auto& delta:advances) {
        if(auto found=targets.find(chain);found!=targets.end()) {
          auto it=std::lower_bound(found->second.begin(),found->second.end(),delta.begin,
              [](auto item,auto p){return item.position<p;});
          for(;it!=found->second.end() && it->position<delta.end;++it) {
            const auto index=it->index;
            if(done[index] || !delta.members.contains(it->member))continue;
            answers[index]|=delta.labels&requests[index].labels;
            if(requests[index].all?answers[index]==requests[index].labels:answers[index]!=0) {
              done[index]=true;
              if(stop_on_any) { remaining=0;return; }
              --remaining;
            }
          }
        }
        auto [it,inserted]=pending.try_emplace(chain);
        if(inserted)queue.push_back(chain);
        auto& work=it->second;
        auto same=std::find_if(work.begin(),work.end(),[&](const auto& old) {
          return old.begin==delta.begin && old.end==delta.end &&
              (old.labels==delta.labels || old.members==delta.members);
        });
        if(same==work.end())work.push_back(std::move(delta));
        else if(same->labels==delta.labels)same->members.unite(delta.members);
        else same->labels|=delta.labels;
      }
    };
    for(std::size_t i=0;i<seeds.size();++i) {
      auto [f,m]=topology_.layout()->locate(seeds[i]);
      MemberSet seed(coordinates_[f].active.extent());seed.set(coordinates_[f].coordinate(m));
      add(f,std::move(seed),std::uint64_t(1)<<i);
    }
    while(!queue.empty() && remaining) {
      const auto chain=queue.front();queue.pop_front();
      auto work=std::move(pending.at(chain));pending.erase(chain);
      for(const auto& delta:work) {
        ++profile.nodes;
        const auto& exits=transpose?reverse_exits_[chain]:exits_[chain];
        auto exit=std::lower_bound(exits.begin(),exits.end(),delta.begin,
            [](const auto& e,auto p){return e.first<p;});
        for(;exit!=exits.end() && exit->first<delta.end;++exit) {
          auto edge=topology_.relations()[exit->second];
          if(transpose) {
            std::swap(edge.source,edge.target);std::swap(edge.first_source,edge.first_target);
            std::swap(edge.source_step,edge.target_step);
          }
          if(filter && !filter(edge.target)) { ++profile.pruned;continue; }
          ++edge_visits;++profile.edges;
          add(edge.target,TransferMembers(edge,coordinates_,delta.members,irregular),delta.labels);
          if(!remaining)break;
        }
        if(!remaining)break;
      }
    }
    ProfileScope irregular_profile("oracle.chain.irregular_members");irregular_profile.nodes=irregular;
    ProfileScope fragments_profile("oracle.chain.frontier_fragments");fragments_profile.nodes=peak_fragments;
    return answers;
  }
  std::size_t chainCount() const { return chains_.size(); }
  std::size_t chainOf(std::size_t family) const { return family_chain_.at(family); }
  std::size_t chainPosition(std::size_t family,bool reverse=false) const {
    return reverse?chains_[family_chain_.at(family)].size()-1-position_.at(family):position_.at(family);
  }
  const std::vector<std::pair<std::size_t,std::size_t>>& chainExits(std::size_t chain,bool reverse=false) const {
    return reverse?reverse_exits_.at(chain):exits_.at(chain);
  }
 private:
  bool identity(const EventRelations::Relation& edge) const {
    return (edge.count==1 || (edge.source_step && edge.target_step)) &&
        edge.first_source==edge.first_target && coordinates_[edge.source].same(coordinates_[edge.target]);
  }
  void appendExits() {
    std::vector<std::size_t> changed,reverse_changed;
    const bool initial=relation_count_==0;
    for(auto i=relation_count_;i<topology_.relations().size();++i) {
      const auto& edge=topology_.relations()[i];const auto chain=family_chain_[edge.source];
      if(chain==family_chain_[edge.target] && position_[edge.source]<position_[edge.target] && identity(edge))continue;
      exits_[chain].emplace_back(position_[edge.source],i);if(!initial)changed.push_back(chain);
      const auto target=family_chain_[edge.target];
      reverse_exits_[target].emplace_back(chains_[target].size()-1-position_[edge.target],i);if(!initial)reverse_changed.push_back(target);
      if(HbIndexCensusEnabled() && (i+1)%1000000==0)HbIndexCensus("exits.progress",this,i+1,changed.size());
    }
    if(relation_count_==0) {
    HbIndexRows("exits.forward",this,exits_);
    HbIndexRows("exits.reverse",this,reverse_exits_);
    HbIndexVector("exits.changed",this,changed);
    HbIndexVector("exits.reverse_changed",this,reverse_changed);
    }
    if(initial) {
      for(auto& row:exits_)std::sort(row.begin(),row.end());
      for(auto& row:reverse_exits_)std::sort(row.begin(),row.end());
      relation_count_=topology_.relations().size();return;
    }
    std::sort(changed.begin(),changed.end());changed.erase(std::unique(changed.begin(),changed.end()),changed.end());
    for(auto chain:changed)std::sort(exits_[chain].begin(),exits_[chain].end());
    std::sort(reverse_changed.begin(),reverse_changed.end());reverse_changed.erase(std::unique(reverse_changed.begin(),reverse_changed.end()),reverse_changed.end());
    for(auto chain:reverse_changed)std::sort(reverse_exits_[chain].begin(),reverse_exits_[chain].end());
    relation_count_=topology_.relations().size();
  }
  const EventRelations& topology_;
  FamilyCoordinates coordinates_;
  std::vector<std::size_t> family_chain_,position_;
  std::vector<std::vector<std::size_t>> chains_;
  std::vector<std::vector<std::pair<std::size_t,std::size_t>>> exits_;
  std::vector<std::vector<std::pair<std::size_t,std::size_t>>> reverse_exits_;
  std::size_t relation_count_=0;
};
}  // namespace sentryward::algorithm
