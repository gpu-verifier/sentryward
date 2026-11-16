#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

// Necessary complete capture facts only. A conservative dependency union
// may reject a valid piece; it never permits a factor to cross a binder.
inline auto Engine::JoinedProducts::captures(
    std::span<const Use> factors,Registry::Id scope,
    std::span<const std::uint32_t> axes) -> std::optional<std::vector<JoinedFamilies::Capture>> {
  std::vector<JoinedFamilies::Capture> result;
  for(auto use:factors) {
    if(!products.admitted(use) || !registry.prefix(use.scope,scope))return {};
    const auto body=registry.heads.owner(use.body);
    const auto& free=registry.heads.engine.free_indices_[body];
    if(!registry.heads.charge(free.size()+1))return {};
    for(auto level:free) {
      if(std::find(axes.begin(),axes.end(),level)!=axes.end())continue;
      if(level>=registry.scopes[use.scope].domains.size())return {};
      result.emplace_back(level,registry.scopes[use.scope].domains[level]);
    }
  }
  if(!registry.heads.charge(result.size()*(1+std::bit_width(result.size()+1))))return {};
  std::sort(result.begin(),result.end());result.erase(std::unique(result.begin(),result.end()),result.end());return result;
}

inline auto Engine::JoinedProducts::join(std::span<const Piece> input,JoinedFamilies::Box domain) -> std::optional<Id> {
  if(input.empty() || input.size()>=1024 || families.size()>=256 ||
      !registry.heads.charge(input.size()*(28+std::bit_width(input.size()+1))))return {};
  std::vector<Piece> checked;std::vector<JoinedFamilies::Piece> geometric;
  checked.reserve(input.size());geometric.reserve(input.size());
  std::map<PieceKey,std::size_t> positions;
  for(const auto& piece:input) {
    if(!piece.count || piece.count>piece.factors.size())return {};
    const auto original=std::span(piece.factors.data(),piece.count);
    auto factors=products.factors(original);if(!factors || factors->empty())return {};
    auto scope=products.common(*factors);if(!scope)return {};
    std::array<Registry::Image,2> images;
    for(unsigned axis=0;axis<2;++axis) {
      auto image=registry.view(piece.images[axis]);
      if(!image || image->scope!=*scope || (axis && image->level==images[axis-1].level))return {};
      images[axis]=*image;
    }
    const std::array<std::uint32_t,2> axes{images[0].level,images[1].level};
    auto all_captures=captures(*factors,*scope,axes);if(!all_captures)return {};
    // Geometry's body anchor must carry exactly ALL product captures. Its
    // existing family proof is used only for intervals/scopes, never as an
    // asserted equality of the omitted product factors.
    std::optional<Use> anchor;std::string_view type;
    for(auto use:*factors) {
      const auto body=registry.heads.owner(use.body);
      if(type.empty())type=registry.heads.snapshot[body].type;
      else if(type!=registry.heads.snapshot[body].type)return {};
      if(use.scope==*scope) {
        auto own=captures(std::span(&use,1),*scope,axes);
        if(!own)return {};if(*own==*all_captures && !anchor)anchor=use;
      }
    }
    if(!anchor)return {};
    JoinedFamilies::Piece gp{anchor->body,*scope,piece.images};
    Piece cp;cp.count=static_cast<unsigned>(factors->size());cp.images=piece.images;
    std::copy(factors->begin(),factors->end(),cp.factors.begin());
    cp.reference_product=products.referenceProduct(std::span(cp.factors.data(),cp.count));
    if(!cp.reference_product)return {};
    const PieceKey key{registry.heads.owner(gp.body),gp.scope,
        images[0].level,images[0].coordinate,images[1].level,images[1].coordinate,
        JoinedFamilies::Box{{{images[0].begin,images[0].end},{images[1].begin,images[1].end}}}};
    if(!registry.heads.charge(12+6*std::bit_width(positions.size()+1)) ||
        !positions.emplace(key,checked.size()).second)return {};
    checked.push_back(cp);geometric.push_back(gp);
  }
  auto joined=geometry.join(geometric,2,domain);if(!joined)return {};
  const auto* layout=geometry.view(*joined);if(!layout || layout->pieces.size()!=checked.size())return {};
  auto family=std::make_unique<Family>(Family{*joined,{}});family->pieces.reserve(checked.size());
  for(const auto& piece:layout->pieces) {
    auto t=registry.view(piece.images[0]),s=registry.view(piece.images[1]);if(!t || !s)return {};
    const PieceKey key{registry.heads.owner(piece.body),piece.scope,t->level,t->coordinate,s->level,s->coordinate,
        JoinedFamilies::Box{{{t->begin,t->end},{s->begin,s->end}}}};
    if(!registry.heads.charge(12+6*std::bit_width(positions.size()+1)))return {};
    auto pos=positions.find(key);if(pos==positions.end())return {};family->pieces.push_back(checked[pos->second]);
  }
  Id result;result.owner=this;result.id=families.size();families.push_back(std::move(family));return result;
}

// Prove a complete state-vector sample at a retained exact-u64 scalar or
// private actual-dispatch singleton. Both use the SAME original-image
// assignment and complete typed/scoped factor relation below.
inline auto Engine::JoinedProducts::sample(Id id,std::span<const Use> actual,const Registry::ScalarPoint& global_point,std::uint32_t state_axis) -> bool {
  const auto* family=view(id);if(!family || actual.empty())return false;
  const auto* layout=geometry.view(family->geometry);if(!layout)return false;
  auto actual_scope=products.common(actual);if(!actual_scope)return false;
  auto global=registry.globalValue(global_point);
  auto extent=registry.count(*actual_scope,state_axis);
  if(!global || !extent ||
      layout->domain[1]!=std::pair<std::uint64_t,std::uint64_t>{0,*extent})return false;
  const auto point=*global;
  if(point<layout->domain[0].first || point>=layout->domain[0].second)return false;
  if(!registry.heads.charge(1+std::bit_width(layout->boxes.size()+1)))return false;
  const auto first=std::lower_bound(layout->boxes.begin(),layout->boxes.end(),point,
      [](const auto& box,std::uint64_t p){return box[0].second<=p;});
  if(first==layout->boxes.end() || point<(*first)[0].first)return false;
  const auto band=(*first)[0];
  for(std::size_t i=first-layout->boxes.begin();i<layout->boxes.size() && layout->boxes[i][0]==band;++i) {
    if(!registry.heads.charge())return false;
    const auto& piece=family->pieces[i];const auto& original=layout->pieces[i];
    if(!piece.reference_product)return false;
    auto context=registry.sampleContext(piece.images[0],piece.images[1],*actual_scope,global_point,state_axis);
    const auto* assignment=context?registry.context(*context):nullptr;
    if(!assignment || assignment->reference_scope!=original.scope ||
        !products.equal(*piece.reference_product,actual,*context))return false;
  }
  return true;
}

} // namespace
} // namespace sentryward::equivalence
