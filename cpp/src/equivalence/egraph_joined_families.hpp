#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::JoinedFamilies::intervalLimits(std::size_t work) -> bounded_linear_reduction::Limits {
  // Opt-in uses the SAME frozen physical-epoch allowance; each primitive
  // still pre-debits heads.charge. This is not a fresh per-family budget.
  bounded_linear_reduction::Limits limits;
  limits.max_work=std::min(work,epoch_work_allowance::maximum_work);
  limits.coefficients.max_work=limits.max_work;return limits;
}

// Canonical interval equality rejects multiplicity, gaps, overlaps and
// duplicates; no logical coordinate is inspected. Builder work is charged
// through primitive hooks to the SAME epoch allowance before any work.
inline auto Engine::JoinedFamilies::cover(std::span<const std::pair<std::uint64_t,std::uint64_t>> pieces,
    std::pair<std::uint64_t,std::uint64_t> domain) -> bool {
  if(pieces.size()>=1024 || !registry.heads.charge(pieces.size()+1))return false;
  auto one=intervals.coefficient("1");if(!sync() || !one)return false;
  std::vector<bounded_linear_reduction::WeightedInterval> values;values.reserve(pieces.size());
  for(auto [begin,end]:pieces) {
    if(begin>=end || !registry.heads.charge(2))return false;
    values.push_back({begin,end,*one});
  }
  const std::array<bounded_linear_reduction::WeightedInterval,1> target{{{domain.first,domain.second,*one}}};
  auto result=intervals.intervalsEqual(values,target);return sync() && result && *result;
}

inline auto Engine::JoinedFamilies::join(std::span<const Piece> pieces,unsigned rank,Box domain) -> std::optional<Id> {
  if(rank==0 || rank>2 || pieces.empty() || pieces.size()>=1024 ||
      !registry.heads.charge(pieces.size()*(20+std::bit_width(pieces.size()+1))))return {};
  for(unsigned axis=0;axis<rank;++axis)if(domain[axis].first>=domain[axis].second)return {};
  std::vector<Box> boxes;boxes.reserve(pieces.size());
  std::vector<PieceKey> piece_keys;piece_keys.reserve(pieces.size());
  std::optional<std::vector<Capture>> captures;std::string_view type;
  for(const auto& piece:pieces) {
    auto body=registry.heads.owner(piece.body);
    if(body>=registry.heads.snapshot.size())return {};
    const auto& k=registry.heads.snapshot[body];
    if(piece.scope>=registry.scopes.size() || !realType(k.type) || !k.provenance.empty() ||
        !registry.heads.engine.finite_[body] || !registry.heads.charge())return {};
    if(type.empty())type=k.type;else if(type!=k.type)return {};
    Box box{};std::array<std::uint32_t,2> levels{};std::array<Class,2> coordinates{};
    for(unsigned axis=0;axis<rank;++axis) {
      auto image=registry.view(piece.images[axis]);
      if(!image || image->scope!=piece.scope || (axis && image->level==levels[axis-1]) ||
          image->begin<domain[axis].first || image->end>domain[axis].second)return {};
      levels[axis]=image->level;coordinates[axis]=image->coordinate;box[axis]={image->begin,image->end};
    }
    const auto& free=registry.heads.engine.free_indices_[body];
    if(free.size()>8 || !registry.heads.charge(free.size()+1))return {};
    std::vector<Capture> actual;
    for(auto level:free) {
      bool reduced=false;for(unsigned axis=0;axis<rank;++axis)reduced|=levels[axis]==level;
      if(reduced)continue;
      if(level>=registry.scopes[piece.scope].domains.size())return {};
      actual.emplace_back(level,registry.scopes[piece.scope].domains[level]);
    }
    if(captures && *captures!=actual)return {};captures=std::move(actual);
    boxes.push_back(box);piece_keys.emplace_back(body,piece.scope,levels[0],coordinates[0],levels[1],coordinates[1],box);
  }
  // Exact complete keys, not shape/name equality. Immutable owner/epoch,
  // scopes/occurrences and actual coordinate IDs are retained in the key.
  if(!registry.heads.charge(pieces.size()*(6+6*std::bit_width(keys.size()+1))))return {};
  Key key{rank,domain,piece_keys};
  if(auto it=keys.find(key);it!=keys.end())return it->second;
  if(families.size()>=256)return {};
  std::vector<std::size_t> order(pieces.size());std::iota(order.begin(),order.end(),0);
  std::sort(order.begin(),order.end(),[&](auto a,auto b){return boxes[a]<boxes[b];});
  std::vector<std::pair<std::uint64_t,std::uint64_t>> bands;
  if(rank==1) {
    bands.reserve(order.size());for(auto i:order)bands.push_back(boxes[i][0]);
  } else {
    for(std::size_t i=0;i<order.size();) {
      const auto band=boxes[order[i]][0];bands.push_back(band);
      std::vector<std::pair<std::uint64_t,std::uint64_t>> inner;
      do {inner.push_back(boxes[order[i]][1]);++i;}
        while(i<order.size() && boxes[order[i]][0]==band);
      if(!cover(inner,domain[1]))return {};
    }
  }
  if(!cover(bands,domain[0]))return {};
  if(!registry.heads.charge(pieces.size()*10+captures->size()*2))return {};
  auto family=std::make_unique<Family>(Family{rank,domain,type,{}, {},std::move(*captures)});
  family->pieces.reserve(pieces.size());family->boxes.reserve(pieces.size());
  for(auto i:order){family->pieces.push_back(pieces[i]);family->boxes.push_back(boxes[i]);}
  Id id;id.owner=this;id.id=families.size();families.push_back(std::move(family));keys.emplace(std::move(key),id);return id;
}

inline auto Engine::JoinedFamilies::uniform(Id id,Class reference,Registry::Id scope,std::span<const std::uint32_t> axes,
    std::span<const Registry::ImageId> reference_images) -> bool {
  const auto* family=view(id);if(!family || axes.size()!=family->rank ||
      (!reference_images.empty() && reference_images.size()!=axes.size()))return false;
  for(const auto& piece:family->pieces) {
    if(!registry.heads.charge())return false;
    Registry::Context context{scope,piece.scope};context.axis_count=family->rank;
    for(unsigned axis=0;axis<family->rank;++axis) {
      auto image=registry.view(piece.images[axis]);if(!image)return false;
      context.axes[axis]={axes[axis],image->level,image->begin,image->end-image->begin,image->coordinate,false};
      if(image->derived)context.axes[axis].coordinate_image=piece.images[axis];
      if(!reference_images.empty()) {
        auto ref=registry.view(reference_images[axis]);
        if(!ref || ref->scope!=scope || ref->level!=axes[axis])return false;
        auto relative=registry.relativeImage(piece.images[axis],reference_images[axis]);
        auto r=relative?registry.view(*relative):std::nullopt;if(!r)return false;
        context.axes[axis].offset=r->begin;context.axes[axis].actual_coordinate=r->coordinate;
        context.axes[axis].coordinate_image=*relative;
      }
    }
    if(!registry.orderContextAxes(context) ||
        !registry.heads.engine.affineBodyMapWithHeads(reference,piece.body,UINT32_MAX,
        registry.heads.snapshot,false,registry.heads.remaining,registry.heads,UINT32_MAX,&registry,&context))return false;
  }
  return true;
}

} // namespace
} // namespace sentryward::equivalence
