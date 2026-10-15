#pragma once
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include "sentryward/algorithm/analysis/member_set.h"
#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/event_ir/event_relations.h"

namespace sentryward::algorithm {
struct FamilyCoordinateMap final {
  std::uint64_t domain;
  std::size_t count, first;
  std::int64_t step;
  MemberSet active;
  std::vector<std::size_t> irregular;
  std::size_t coordinate(std::size_t member) const {
    return irregular.empty()?std::size_t(std::int64_t(first)+step*std::int64_t(member)):irregular.at(member);
  }
  std::optional<std::size_t> member(std::size_t coordinate) const {
    if(!active.contains(coordinate))return {};
    if(!irregular.empty()) {
      const auto it=std::find(irregular.begin(),irregular.end(),coordinate);
      return std::size_t(it-irregular.begin());
    }
    if(!step)return std::size_t(0);
    return std::size_t((std::int64_t(coordinate)-std::int64_t(first))/step);
  }
  MemberSet mask(std::size_t begin,std::size_t length) const {
    if(!begin && length==count)return active;
    MemberSet result(active.extent());
    if(irregular.empty() && length) {
      const auto a=coordinate(begin),b=coordinate(begin+length-1);
      result.addRange(std::min(a,b),std::max(a,b)+1);
      result=result.intersect(active);
    }
    else for(std::size_t i=begin;i<begin+length;++i)result.set(coordinate(i));
    return result;
  }
  bool same(const FamilyCoordinateMap& other) const {
    return domain==other.domain && count==other.count && first==other.first &&
      step==other.step && irregular==other.irregular;
  }
};

class FamilyCoordinates final {
 public:
  FamilyCoordinates(const HappensBeforeGraph& graph,const EventRelations& topology,
      bool relation_coordinates=false) {
    HbIndexCensus("coordinates.begin",this,topology.layout()->families().size());
    if(relation_coordinates) {
      // Query coordinates name family members, not executing actors. A full
      // identity edge is an exact proof that these two member indices agree,
      // including actorless async effects. Actor metadata is not needed.
      const auto& families=topology.layout()->families();
      std::vector<std::size_t> parents(families.size());
      for(std::size_t i=0;i<parents.size();++i)parents[i]=i;
      const auto root=[&](std::size_t i) {
        while(parents[i]!=i){parents[i]=parents[parents[i]];i=parents[i];}return i;
      };
      for(const auto& edge:topology.relations()) {
        if(edge.first_source || edge.first_target ||
           edge.count!=families[edge.source].count || edge.count!=families[edge.target].count ||
           (edge.count>1 && (!edge.source_step || !edge.target_step)))continue;
        const auto a=root(edge.source),b=root(edge.target);parents[std::max(a,b)]=std::min(a,b);
      }
      std::unordered_map<std::size_t,std::shared_ptr<const FamilyCoordinateMap>> components;
      for(std::size_t i=0;i<families.size();++i) {
        const auto component=root(i);auto [it,added]=components.try_emplace(component);
        if(added) {
          const auto count=families[i].count;MemberSet mask(count);mask.addRange(0,count);
          it->second=std::make_shared<const FamilyCoordinateMap>(FamilyCoordinateMap{
              (std::uint64_t(1)<<63)+component,count,0,1,std::move(mask),{}});
        }
        maps_.push_back(it->second);
      }
      HbIndexVector("coordinates.maps",this,maps_);
      HbIndexCensus("coordinates.components",this,components.size(),sizeof(FamilyCoordinateMap));
      private_domains_=components.size();return;
    }
    std::unordered_map<std::uint32_t,std::pair<std::uint64_t,std::size_t>> actors;
    std::unordered_map<std::uint64_t,std::size_t> extents;
    for(const auto& actor:graph.actors()) {
      const auto coordinate=std::size_t(actor.warp.value())*32+actor.lane;
      actors.emplace(actor.id.value(),std::pair{actor.cta.value(),coordinate});
      extents[actor.cta.value()]=std::max(extents[actor.cta.value()],coordinate+1);
    }
    // Synthetic event-only graphs have no launch metadata; ActorId is their domain coordinate.
    std::size_t synthetic_extent=0;
    if(actors.empty())graph.events().forEachAffineColumn([&](const Event& event,const auto& delta,auto,auto,std::size_t count) {
      if(event.actor)synthetic_extent=std::max(synthetic_extent,
          std::size_t(std::max(std::int64_t(event.actor->value()),
              std::int64_t(event.actor->value())+delta[2]*std::int64_t(count-1)))+1);
    });
    std::map<std::tuple<std::uint32_t,std::int64_t,std::size_t>,std::shared_ptr<const FamilyCoordinateMap>> shared;
    graph.events().forEachAffineColumn([&](const Event& event,const auto& delta,auto,auto,std::size_t count) {
      const auto family=maps_.size();
      if(!event.actor || (count>1 && !delta[2])) { appendPrivate(family,count);return; }
      const auto key=std::tuple{event.actor->value(),delta[2],count};
      if(auto it=shared.find(key);it!=shared.end()) { maps_.push_back(it->second);return; }
      std::vector<std::size_t> positions;
      std::uint64_t domain=UINT64_MAX;
      std::size_t extent=synthetic_extent;
      bool compatible=true;
      for(std::size_t i=0;i<count;++i) {
        const auto actor=std::uint32_t(std::int64_t(event.actor->value())+delta[2]*std::int64_t(i));
        if(actors.empty())positions.push_back(actor);
        else {
          const auto found=actors.find(actor);
          if(found==actors.end() || (i && domain!=found->second.first)) { compatible=false;break; }
          domain=found->second.first;extent=extents.at(domain);positions.push_back(found->second.second);
        }
      }
      if(!compatible) { appendPrivate(family,count);return; }
      auto value=std::make_shared<FamilyCoordinateMap>(FamilyCoordinateMap{domain,count,positions.front(),0,MemberSet(extent),{}});
      value->step=count>1?std::int64_t(positions[1])-std::int64_t(positions[0]):0;
      for(std::size_t i=0;i<count;++i) {
        if(value->active.contains(positions[i])) { compatible=false;break; }
        value->active.set(positions[i]);
        if(std::int64_t(positions[i])!=std::int64_t(positions[0])+value->step*std::int64_t(i))value->irregular=positions;
      }
      if(!compatible) { appendPrivate(family,count);return; }
      maps_.push_back(value);shared.emplace(key,std::move(value));
    });
    if(maps_.size()!=topology.layout()->families().size())throw std::logic_error("coordinate family mismatch");
  }
  const FamilyCoordinateMap& operator[](std::size_t family) const { return *maps_.at(family); }
  void append(const EventRelations& topology) {
    while(maps_.size()<topology.layout()->families().size())appendPrivate(maps_.size(),1);
  }
  std::size_t privateDomains() const { return private_domains_; }
 private:
  void appendPrivate(std::size_t family,std::size_t count) {
    MemberSet mask(count);mask.addRange(0,count);
    maps_.push_back(std::make_shared<FamilyCoordinateMap>(FamilyCoordinateMap{
        (std::uint64_t(1)<<63)+family,count,0,1,std::move(mask),{}}));
    ++private_domains_;
  }
  std::vector<std::shared_ptr<const FamilyCoordinateMap>> maps_;
  std::size_t private_domains_=0;
};

// Relation transfer uses CTA-local coordinates, including cross-domain correspondence.
inline MemberSet TransferMembers(const EventRelations::Relation& edge,
    const FamilyCoordinates& coordinates,const MemberSet& members,std::size_t& irregular_visits) {
  const auto& source=coordinates[edge.source];const auto& target=coordinates[edge.target];
  const auto selected=members.intersect(source.mask(edge.first_source,edge.source_step?edge.count:1));
  MemberSet result(target.active.extent());
  if(selected.empty())return result;
  if(!edge.target_step) { result.set(target.coordinate(edge.first_target));return result; } // Gather.
  if(!edge.source_step)return target.mask(edge.first_target,edge.count); // Broadcast.
  if(source.irregular.empty() && target.irregular.empty() && source.step==target.step) {
    const auto offset=std::int64_t(target.coordinate(edge.first_target))-std::int64_t(source.coordinate(edge.first_source));
    return selected.shifted(offset,target.active.extent()).intersect(target.mask(edge.first_target,edge.count));
  }
  // Exact irregular mapping; never replace it by an interval hull.
  selected.forEach([&](std::size_t coordinate) {
    ++irregular_visits;
    const auto index=source.member(coordinate);
    result.set(target.coordinate(edge.first_target+*index-edge.first_source));
  });
  return result;
}
}  // namespace sentryward::algorithm
