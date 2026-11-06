#include "production.hpp"
#include "sentryward/scalar_fold/output_family.h"
#include <array>
#include <numeric>
namespace rg {
namespace {
J edgeJson(const Edge& e) {
  J j={{"node",e.node}};
  if(!e.conversion.empty())j["conversion"]=e.conversion;
  if(!e.converted_type.empty())j["converted_type"]=e.converted_type;
  if(!e.pointer_provenance.empty())j["pointer_provenance"]=e.pointer_provenance;
  return j;
}
// Appends one step to an edge's conversion chain.
void appendConversion(Edge& e,const std::string& step,const std::string& type) {
  if(!e.conversion.empty())e.conversion+="; ";
  e.conversion+=step;e.converted_type=type;
}
Edge edge(const sentryward::scalar_fold::Ref& e) {
  return Edge(e.node,e.conversion,e.converted_type,e.pointer_provenance);
}
uint64_t checked(Z n) {
  if(n<0 || n>UINT64_MAX)throw Unsupported("output extent overflow");
  return n.convert_to<uint64_t>();
}
uint64_t product(const std::vector<uint64_t>& xs) {
  Z n=1;for(auto x:xs){if(!x)throw Unsupported("empty output axis");n*=x;}return checked(n);
}
struct Builder {
  Document doc;
  std::unordered_map<std::string,Id> intern;
  std::map<std::string,std::unordered_map<Id,Edge>> contexts;
  Builder(){doc.meta={{"format","spverify.scalar-fold.v2"},{"numerical_semantics","ideal_real"},{"outputs",J::array()},{"output_families",J::array()}};}
  Edge add(RawNode n) {
    // Key includes immediate operands only, never recursively hashes a DAG.
    J key={n.kind,n.name,n.type,n.attributes,n.level,n.count,n.reducer,J::array()};
    for(const auto& e:n.operands)key[7].push_back(edgeJson(e));
    auto text=key.dump();auto it=intern.find(text);
    if(it!=intern.end()){doc.nodes[it->second].instruction|=n.instruction;return Edge(it->second);}
    if(doc.nodes.size()>=invalid)throw Unsupported("raw node ID overflow");
    Id id=doc.nodes.size();doc.nodes.push_back(std::move(n));intern.emplace(std::move(text),id);return Edge(id);
  }
  Edge constant(uint64_t n,std::string type="u64"){RawNode r;r.kind="Constant";r.name=std::to_string(n);r.type=std::move(type);return add(std::move(r));}
  Edge index(unsigned i){RawNode r;r.kind="Index";r.type="u64";r.level=i;return add(std::move(r));}
  Edge op(std::string name,Edge a,Edge b,std::string type="u64"){RawNode r;r.kind="Operation";r.name=std::move(name);r.type=std::move(type);r.operands={a,b};return add(std::move(r));}
  Edge source(const Document& src,Edge root,unsigned rank) {
    std::vector<Edge> identity;for(unsigned i=0;i<rank;++i)identity.push_back(index(i));
    return clone(src,root,identity,rank,rank);
  }
  Edge clone(const Document& src,Edge root,const std::vector<Edge>& coordinates,unsigned old_rank,unsigned new_rank) {
    J key={old_rank,new_rank,J::array()};for(const auto& e:coordinates)key[2].push_back(edgeJson(e));
    auto& memo=contexts[key.dump()]; // Each Builder has exactly one source Document.
    auto use=[&](const Edge& e){auto r=memo.at(e.node);if(!e.conversion.empty())r.conversion=e.conversion;
      if(!e.converted_type.empty())r.converted_type=e.converted_type;
      if(!e.pointer_provenance.empty())r.pointer_provenance=e.pointer_provenance;return r;};
    std::vector<std::pair<Id,bool>> stack{{root.node,false}};
    while(!stack.empty()) {
      auto [id,ready]=stack.back();stack.pop_back();if(memo.contains(id))continue;
      const auto& n=src.nodes.at(id);
      if(!ready){stack.emplace_back(id,true);for(auto e:n.operands){if(e.node>=id)throw Unsupported("non-topological expression");if(!memo.contains(e.node))stack.emplace_back(e.node,false);}continue;}
      if(n.kind=="Index")memo.emplace(id,n.level<old_rank?coordinates.at(n.level):index(n.level-old_rank+new_rank));
      else {auto copy=n;for(auto& e:copy.operands)e=use(e);
        if(n.kind=="Fold"){if(n.level<old_rank)throw Unsupported("captured Fold binder");copy.level=n.level-old_rank+new_rank;}
        memo.emplace(id,add(std::move(copy)));}
    }
    return use(root);
  }
};
struct Axis {uint64_t stride,size;unsigned index;};
struct Range {uint64_t first,last,step;};
// Exact finite AP intersection. Work depends on stored component pairs, not members.
bool intersects(Range a,Range b) {
  if(std::max(a.first,b.first)>std::min(a.last,b.last))return false;
  Z g=std::gcd(a.step,b.step),delta=Z(b.first)-a.first;
  if(delta%g!=0)return false;
  Z modulus=Z(b.step)/g,aa=Z(a.step)/g;
  Z x=0,y=1,r=modulus,s=aa;
  while(s!=0){Z q=r/s;Z nr=r-q*s;r=s;s=nr;Z nx=x-q*y;x=y;y=nx;}
  Z i=modulus==1?Z(0):Z((delta/g*x)%modulus);if(i<0)i+=modulus;
  Z first=Z(a.first)+Z(a.step)*i,period=Z(a.step)*modulus,lo=std::max(a.first,b.first);
  if(first<lo)first+=((lo-first+period-1)/period)*period;
  return first<=std::min(a.last,b.last);
}
// Pairwise disjointness of n items, swept along one axis on which item i spans
// [lo(i), hi(i)]: only an active item p whose span meets the current item b's
// is tested with overlap(p,b), not all pairs.
template<class Lo,class Hi,class Overlap> void require_disjoint(std::size_t n,Lo lo,Hi hi,Overlap overlap) {
  std::vector<std::size_t> order(n);std::iota(order.begin(),order.end(),std::size_t(0));
  std::sort(order.begin(),order.end(),[&](std::size_t x,std::size_t y){return lo(x)<lo(y);});
  std::vector<std::size_t> active;
  for(auto b:order) {
    std::erase_if(active,[&](std::size_t p){return hi(p)<lo(b);});
    for(auto p:active)if(overlap(p,b))throw Unsupported("overlapping outputs");
    active.push_back(b);
  }
}
// Pairwise disjointness of boxes (one Range per axis), swept along axis `sweep`: only boxes whose
// ranges on that axis meet are compared on every axis. Precondition: all rows have the same length,
// and with two or more boxes that length exceeds `sweep` (a single box is never indexed).
void requireDisjointBoxes(const std::vector<std::vector<Range>>& boxes,std::size_t sweep) {
  require_disjoint(boxes.size(),[&](std::size_t x){return boxes[x][sweep].first;},
      [&](std::size_t x){return boxes[x][sweep].last;},
      [&](std::size_t p,std::size_t q){
        for(std::size_t i=0;i<boxes[p].size();++i)if(!intersects(boxes[p][i],boxes[q][i]))return false;
        return true;});
}
// A family dimension's digit on one (tensor or grid) axis: the axis coordinate gains
// slope x ((j_dim / divisor) % radix) over `count` positions; radix 0: a single term (no Remainder).
struct Term {unsigned dim;uint64_t divisor,radix,slope,count;};
// Why a family box does not bind to the axes (bindDimension, tileAxis); each caller words the diagnostic
// in a table with one message per value, checked against box_error_count (Extent stays last).
enum class BoxError {Stride,ExcludedAxis,FusedStride,FusedCrossing,FusedAlignment,FusedExtent,Holes,Extent};
constexpr std::size_t box_error_count=std::size_t(BoxError::Extent)+1;
// Binds family dimension `dim` (extent n > 1, byte stride `stride`) of a box at `base` to `axes`
// (ascending strides; terms are appended by axis index). The largest axis stride dividing `stride`
// takes one term when the dimension fits that axis from `base`; otherwise, at unit slope, the
// dimension is fused over consecutive axes as an exact mixed-radix chain: each digit min(rest, axis
// size), aligned (a digit spanning a whole axis starts at 0) and dividing the rest. No term may use
// the axis `excluded`. O(axes).
std::optional<BoxError> bindDimension(const std::vector<Axis>& axes,const std::vector<uint64_t>& base,
    std::optional<unsigned> excluded,unsigned dim,uint64_t n,uint64_t stride,std::vector<std::vector<Term>>& terms) {
  const auto blocked=[&](const Axis& x){return excluded && x.index==*excluded;};
  auto reverse=std::find_if(axes.rbegin(),axes.rend(),[&](Axis x){return stride>0 && stride%x.stride==0;});
  if(reverse==axes.rend())return BoxError::Stride;
  auto it=std::prev(reverse.base());
  if(blocked(*it))return BoxError::ExcludedAxis;
  const auto slope=stride/it->stride;
  if(Z(slope)*(n-1)<Z(it->size)-Z(base[it->index])){terms[it->index].push_back({dim,1,0,slope,n});return std::nullopt;}
  if(slope!=1)return BoxError::FusedStride;
  uint64_t remaining=n,divisor=1,physical=stride;
  while(remaining>1) {
    if(it==axes.end() || it->stride!=physical || blocked(*it))return BoxError::FusedCrossing;
    const auto digit=std::min(remaining,it->size);
    if(remaining>=it->size && base[it->index])return BoxError::FusedAlignment;
    if(remaining%digit)return BoxError::FusedExtent;
    terms[it->index].push_back({dim,divisor,digit,1,digit});remaining/=digit;
    divisor=checked(Z(divisor)*digit);physical=checked(Z(physical)*digit);++it;
  }
  return std::nullopt;
}
// Sorts one axis's terms by slope and checks that they tile base + step x [0, count) exactly (each
// slope the product of the smaller terms' counts) inside an axis of `size`. O(terms log terms).
std::optional<BoxError> tileAxis(std::vector<Term>& terms,uint64_t base,uint64_t size,uint64_t& step,uint64_t& count) {
  std::sort(terms.begin(),terms.end(),[](const Term& x,const Term& y){return x.slope<y.slope;});
  step=terms.empty()?1:terms.front().slope;count=1;
  for(const auto& term:terms) {
    if(Z(term.slope)!=Z(step)*count)return BoxError::Holes;
    count=checked(Z(count)*term.count);
  }
  if(Z(base)+Z(step)*(count-1)>=Z(size))return BoxError::Extent;
  return std::nullopt;
}
// The most segments splitPiece cuts one dimension into, per axis. A run along a dense mixed-radix
// chain of axes is a union of at most two maximal aligned segments per axis (one rising to an
// aligned start, one falling to the run's end); a run that needs more is left to the refusal.
constexpr std::size_t kSegmentsPerAxis=2;
// A part of a piece: the family coordinates first + [0, extent), relative to the piece's own, whose
// first element is at byte `offset`.
struct Part {uint64_t offset;std::vector<uint64_t> first,extent;};
// The parts of a piece (first element at byte `offset`; dimensions `extent` at byte `strides`) that
// is no box on `axes` (ascending strides) because a dimension crosses an axis boundary. Each
// dimension is cut, along its own run from the piece's first element, into segments: from each
// start, the longest run that stays inside the axis it moves along (the largest stride dividing its
// stride, as bindDimension), or, at unit slope from a start aligned to the whole rows of the dense
// axes below some axis, those rows times what remains of that axis. The parts are the products of
// the segments; the caller places each as a box and refuses unless every one is. Empty when no
// dimension is cut, a segment would start outside the axes, or a dimension needs more than
// kSegmentsPerAxis segments per axis. O(dims x axes^2), and O(product of segment counts) parts.
std::vector<Part> splitPiece(const std::vector<Axis>& axes,uint64_t origin,uint64_t offset,
    const std::vector<uint64_t>& extent,const std::vector<uint64_t>& strides) {
  std::vector<std::vector<std::pair<uint64_t,uint64_t>>> segments(extent.size());
  bool cut=false;
  for(std::size_t d=0;d<extent.size();++d) {
    const auto n=extent[d],stride=strides[d];
    auto& list=segments[d];
    const auto home=std::find_if(axes.rbegin(),axes.rend(),[&](const Axis& x){return stride>0 && stride%x.stride==0;});
    if(n<=1 || home==axes.rend()){list.push_back({0,n});continue;}
    const auto a=std::size_t(std::prev(home.base())-axes.begin());
    const auto slope=stride/axes[a].stride;
    for(uint64_t start=0;start<n;) {
      if(list.size()>=kSegmentsPerAxis*axes.size())return {};
      // The axis coordinates of the segment's first element.
      std::vector<uint64_t> digit(axes.size());
      auto rem=checked(Z(offset)+Z(start)*stride-origin);
      for(std::size_t i=axes.size();i-->0;){digit[i]=rem/axes[i].stride;rem%=axes[i].stride;if(digit[i]>=axes[i].size)return {};}
      const auto rest=n-start;
      auto length=std::min<uint64_t>(rest,(axes[a].size-1-digit[a])/slope+1);
      if(slope==1) {
        Z row=1;
        for(std::size_t i=a;i<axes.size() && row<=rest;++i) {
          if(i>a && Z(axes[i].stride)!=Z(axes[i-1].stride)*axes[i-1].size)break;
          length=std::max(length,checked(Z(row*std::min(Z(axes[i].size-digit[i]),Z(Z(rest)/row)))));
          if(digit[i])break;
          row*=axes[i].size;
        }
      }
      list.push_back({start,length});start+=length;
    }
    cut|=list.size()>1;
  }
  if(!cut)return {};
  std::vector<Part> parts{{offset,std::vector<uint64_t>(extent.size()),extent}};
  for(std::size_t d=0;d<extent.size();++d) {
    if(segments[d].size()==1)continue;
    std::vector<Part> next;
    for(const auto& part:parts)for(const auto& [start,length]:segments[d]) {
      auto p=part;p.offset=checked(Z(part.offset)+Z(start)*strides[d]);p.first[d]=start;p.extent[d]=length;
      next.push_back(std::move(p));
    }
    parts=std::move(next);
  }
  return parts;
}
// An axis coordinate: base + the digits `terms` of the family coordinates index(first + dim);
// `remainder(t)` says whether digit t takes its Remainder.
template<class Remainder> Edge axisCoordinate(Builder& x,uint64_t base,const std::vector<Term>& terms,unsigned first,
    Remainder remainder) {
  Edge c=x.constant(base);
  for(const auto& t:terms) {
    Edge j=x.index(first+t.dim);
    if(t.divisor!=1)j=x.op("Divide",j,x.constant(t.divisor));
    if(remainder(t))j=x.op("Remainder",j,x.constant(t.radix));
    if(t.slope!=1)j=x.op("Multiply",j,x.constant(t.slope));
    c=x.op("Add",c,j);
  }
  return c;
}
// Lane `slot` (width bytes) of a packed value of `packed` bytes and type `type`: reinterpreted as
// unsigned bits, shifted right by slot x width bytes and truncated to the lane. False (value
// unchanged) when the type is not b or u bits of the packed width.
bool extractLane(Builder& b,Edge& value,const std::string& type,uint64_t packed,uint64_t width,uint64_t slot) {
  const auto bits=std::to_string(packed*8),wide="u"+bits,narrow="u"+std::to_string(width*8);
  if(type!="b"+bits && type!=wide)return false;
  if(type!=wide)appendConversion(value,"Bitcast("+type+" -> "+wide+")",wide);
  if(slot)value=b.op("ShiftRight",value,b.constant(slot*width*8,wide),wide);
  appendConversion(value,"Convert("+wide+" -> "+narrow+")",narrow);
  return true;
}
// One compared output piece, output-piece-<n>: the reference family (left body over left_coords)
// and the kernel family (right body over right_coords), each a dense `domain` of `width`-byte
// elements, and their dense_tensor mapping.
void emitPiece(Builder& a,Builder& b,J& maps,uint64_t width,const std::vector<uint64_t>& domain,
    const Edge& left,const std::vector<Edge>& left_coords,const Edge& right,const std::vector<Edge>& right_coords) {
  std::vector<uint64_t> strides(domain.size());auto s=width;
  for(size_t i=domain.size();i-->0;){strides[i]=s;s=checked(Z(s)*domain[i]);}
  const auto name="output-piece-"+std::to_string(maps.size());
  const auto bindings=[](const std::vector<Edge>& xs){
    J result=J::array();for(const auto& x:xs)result.push_back(edgeJson(x));return result;};
  J common={{"allocation",name},{"offset",0},{"element_bytes",width},{"shape",domain},{"strides_bytes",strides}};
  common["body"]=edgeJson(left);common["coordinate_bindings"]=bindings(left_coords);
  a.doc.meta["output_families"].push_back(common);
  common["body"]=edgeJson(right);common["coordinate_bindings"]=bindings(right_coords);
  b.doc.meta["output_families"].push_back(common);
  maps.push_back({{"kind","dense_tensor"},{"left_allocation",name},{"right_allocation",name},
    {"left_offset",0},{"right_offset",0},{"element_bytes",width},{"shape",domain},
    {"left_strides_bytes",strides},{"right_strides_bytes",strides}});
}
bool placementRoot(const Document& doc,const J& family) {
  const auto& n=doc.nodes.at(Edge(family.at("body")).node);
  return n.kind=="Operation" && n.name==sentryward::scalar_fold::kPlacementOperation;
}
// The axes the slot of a symbolic-position tensor runs over (pullPartition), as positions
// [first, first + count) of `axes` (ascending strides) when that is more than the slot axis alone:
// the slot axis (the largest stride dividing the placement coefficient) with the fewest following
// dense axes (each stride the previous stride x size) whose product extent holds every write's slot
// range [base + factor x lower, base + factor x upper]. A write's base is its mixed-radix coordinate
// on those axes, from its byte offset as pullPartition decomposes it. Because the run is dense, a
// fused slot coordinate s places a write at s x the run's first stride, and its digits are the run's
// tensor coordinates. None when one axis holds every range or no run does, or a placement is not one
// pullPartition places; pullPartition then binds or refuses the slot axis alone. pullPartition checks
// every write against the fused extent again. O(writes x axes^2).
std::optional<std::pair<std::size_t,std::size_t>> slotRun(const Document& right,const std::vector<Axis>& axes,
    const std::vector<J>& families,uint64_t width,uint64_t origin) {
  std::optional<std::size_t> first;std::size_t count=1;
  for(const J& f:families) {
    if(!placementRoot(right,f))return std::nullopt;
    const auto terms=sentryward::scalar_fold::parsePlacementAttributes(right.nodes.at(Edge(f.at("body")).node).attributes);
    if(!terms || terms->size()!=1 || terms->front().coefficient<=0 || terms->front().lower<0)return std::nullopt;
    const auto coefficient=uint64_t(terms->front().coefficient);
    std::optional<std::size_t> axis;
    for(std::size_t i=axes.size();i-->0 && !axis;)if(coefficient%axes[i].stride==0)axis=i;
    if(!axis || (first && *first!=*axis))return std::nullopt;
    first=axis;
    const auto packed=input_unsigned(f.at("element_bytes"));
    if(!packed || packed%width)return std::nullopt;
    const Z reach=Z(coefficient/axes[*axis].stride)*terms->front().upper;
    for(uint64_t lane=0;lane<packed/width;++lane) {
      const Z offset=Z(input_unsigned(f.at("offset")))+Z(lane)*width-origin;
      if(offset<0)return std::nullopt;
      // Whether the write's range fits the run [first, first + n): its fused coordinate is what remains
      // of its offset below the axes above the run (pullPartition's decomposition), in first strides.
      const auto fits=[&](std::size_t n) {
        Z rem=offset,extent=1;
        for(std::size_t i=axes.size();i-->*first+n;)rem%=axes[i].stride;
        for(std::size_t i=*first;i<*first+n;++i)extent*=axes[i].size;
        return rem/axes[*first].stride+reach<extent;
      };
      while(!fits(count)) {
        const auto next=*first+count;
        if(next==axes.size() || Z(axes[next].stride)!=Z(axes[next-1].stride)*axes[next-1].size)return std::nullopt;
        ++count;
      }
    }
  }
  if(!first || count==1)return std::nullopt;
  return std::pair{*first,count};
}
// Pull mode (index-function memory) for a tensor whose kernel
// families lie at symbolic positions: each family is a dense box at a constant part, moved
// along one tensor axis (the slot axis) by factor x atom (a SymbolicPlacement annotation).
// Families with equal boxes form one comparison piece over (slot axis, box):
//   body(s, j) = Select(s == base + factor x atom_n, F_n(j), ... Select(s == ..atom_1.., F_1(j),
//                InitialCell(o(s, j))))
// with the last family outermost (lifted members in unit order: the order of a reference's
// sequential row loop). The arms of one piece are pairwise disjoint for every valid input:
// memory elimination emits each final cell once per segment and its census proves segments
// disjoint or never both written at meeting ranges; lifted members of class mode are disjoint
// by the cross-unit race check. The completion is the initial contents, exact where no family
// writes. The boxes must tile the non-slot axes exactly (as for constant families).
// Cost: O(families x rank) conditions and one clone per family body (template-sized).
void pullPartition(const Document& left,const Document& right,const J& m,Builder& a,Builder& b,J& maps,
    const J& reference,const std::vector<J>& families) {
  namespace sf=sentryward::scalar_fold;
  const auto refuse=[](const std::string& code,const std::string& message){throw Unsupported(message,code,J::object());};
  auto shape=m.at("shape").get<std::vector<uint64_t>>();
  auto steps=m.at("right_strides_bytes").get<std::vector<uint64_t>>();
  const auto width=input_unsigned(m.at("element_bytes")),origin=input_unsigned(m.at("right_offset"));
  const unsigned rank=unsigned(shape.size());
  std::vector<Axis> axes;
  for(unsigned i=0;i<rank;++i)if(shape[i]>1)axes.push_back({steps[i],shape[i],i});
  std::sort(axes.begin(),axes.end(),[](Axis x,Axis y){return x.stride<y.stride;});
  for(std::size_t i=1;i<axes.size();++i)
    if(axes[i].stride<checked(Z(axes[i-1].stride)*axes[i-1].size))refuse("symbolic_output_placement_unsupported","overlapping output layout");
  // A slot over several dense axes (slotRun) is one axis of their product extent here; the
  // coordinates below split it into the run's tensor coordinates again.
  std::vector<Axis> slot_axes;
  if(const auto run=slotRun(right,axes,families,width,origin)) {
    const auto begin=axes.begin()+std::ptrdiff_t(run->first);
    slot_axes.assign(begin,begin+std::ptrdiff_t(run->second));
    uint64_t extent=1;for(const auto& axis:slot_axes)extent=checked(Z(extent)*axis.size);
    *begin={begin->stride,extent,begin->index};
    axes.erase(begin+1,begin+std::ptrdiff_t(run->second));
  }
  const Edge target(reference.at("body"));
  const auto output_type=target.converted_type.empty()?left.nodes.at(target.node).type:target.converted_type;
  static constexpr auto box_messages=std::to_array<const char*>({"family stride is no multiple of a tensor stride",
      "a family dimension moves along the slot axis","unsupported fused-axis stride",
      "fused family axis crosses padding or the slot axis","fused family axis is not aligned",
      "fused family axis is not aligned","family axes overlap or leave holes","family extent outside the tensor"});
  static_assert(box_messages.size()==box_error_count);
  const auto bound=[&](std::optional<BoxError> e){
    if(e)refuse("symbolic_output_placement_unsupported",box_messages[std::size_t(*e)]);
  };
  struct Arm {Edge body,atom;uint64_t packed,slot,slot_base;};
  struct Piece {
    std::vector<uint64_t> base,fs;std::vector<std::vector<Term>> terms;std::vector<Arm> arms;
    // Per tensor axis: the covered progression base + step x [0, count).
    std::vector<uint64_t> step,count;
  };
  std::map<std::tuple<uint64_t,std::vector<uint64_t>,std::vector<uint64_t>>,Piece> pieces;
  std::optional<Axis> slot;uint64_t factor=0;
  for(const J& f:families) {
    const auto packed=input_unsigned(f.at("element_bytes"));
    if(!packed || packed%width)refuse("symbolic_output_placement_unsupported","invalid packed family");
    auto fs=f.at("shape").get<std::vector<uint64_t>>(),st=f.at("strides_bytes").get<std::vector<uint64_t>>();
    if(fs.size()!=st.size())throw Unsupported("invalid output axes");
    if(!placementRoot(right,f))refuse("symbolic_output_placement_unsupported","a constant family in a symbolic-position tensor");
    const Edge root(f.at("body"));
    const auto& n=right.nodes.at(root.node);
    const auto terms=sf::parsePlacementAttributes(n.attributes);
    if(!root.conversion.empty() || !terms || terms->size()+1!=n.operands.size())
      refuse("symbolic_output_placement_unsupported","malformed symbolic placement");
    if(terms->size()!=1)refuse("symbolic_output_placement_unsupported","a placement with more than one atom");
    const auto& term=terms->front();
    if(term.coefficient<=0 || term.lower<0)refuse("symbolic_output_placement_unsupported","negative placement coefficient or atom");
    // The slot axis: the largest tensor stride dividing the coefficient; one per tensor.
    std::optional<Axis> axis;uint64_t q=0;
    for(auto it=axes.rbegin();it!=axes.rend() && !axis;++it)
      if(uint64_t(term.coefficient)%it->stride==0){axis=*it;q=uint64_t(term.coefficient)/it->stride;}
    if(!axis)refuse("symbolic_output_placement_unsupported","placement coefficient is no multiple of a tensor stride");
    if(slot && (slot->index!=axis->index || factor!=q))refuse("symbolic_output_placement_unsupported","placements move along different axes");
    slot=axis;factor=q;
    for(uint64_t lane=0;lane<packed/width;++lane) {
      std::vector<uint64_t> base(rank);
      auto rem=checked(Z(input_unsigned(f.at("offset")))+Z(lane)*width-origin);
      for(auto it=axes.rbegin();it!=axes.rend();++it){base[it->index]=rem/it->stride;rem%=it->stride;}
      if(rem)refuse("symbolic_output_placement_unsupported","output in padding");
      if(base[slot->index]+Z(factor)*Z(term.upper)>=Z(slot->size))
        refuse("symbolic_write_outside_output","a symbolic write's slot range ["+std::to_string(term.lower)+", "+
            std::to_string(term.upper)+"] leaves the mapped output tensor");
      auto& piece=pieces[{checked(Z(input_unsigned(f.at("offset")))+Z(lane)*width-Z(base[slot->index])*slot->stride),fs,st}];
      if(piece.arms.empty()) {
        // Each family dimension binds as for constant families (bindDimension, tileAxis); the
        // slot axis binds none.
        piece.base=base;piece.base[slot->index]=0;piece.fs=fs;piece.terms.assign(rank,{});
        piece.step.assign(rank,1);piece.count.assign(rank,1);
        for(unsigned d=0;d<fs.size();++d)
          if(fs[d]>1)bound(bindDimension(axes,base,slot->index,d,fs[d],st[d],piece.terms));
        for(const auto& axis:axes)
          bound(tileAxis(piece.terms[axis.index],base[axis.index],axis.size,
                         piece.step[axis.index],piece.count[axis.index]));
      }
      piece.arms.push_back({n.operands[0],n.operands[1],packed,lane,base[slot->index]});
    }
  }
  // The boxes tile the non-slot axes: pairwise disjoint, with the right total.
  std::vector<std::vector<Range>> ranges;Z total=0,expected=1;
  for(const auto& axis:axes)if(axis.index!=slot->index)expected*=axis.size;
  for(const auto& [key,piece]:pieces) {
    auto& row=ranges.emplace_back();Z count=1;
    for(const auto& axis:axes) {
      if(axis.index==slot->index){row.push_back({0,0,1});continue;}
      const auto extent=piece.count[axis.index];
      row.push_back({piece.base[axis.index],checked(Z(piece.base[axis.index])+Z(piece.step[axis.index])*(extent-1)),
          piece.step[axis.index]});count*=extent;
    }
    total+=count;
  }
  // Swept along the widest non-slot axis, as in partition: the slot axis is {0} in every row.
  std::size_t sweep=0;
  for(std::size_t i=0;i<axes.size();++i)
    if(axes[i].index!=slot->index && (axes[sweep].index==slot->index || axes[i].size>axes[sweep].size))sweep=i;
  requireDisjointBoxes(ranges,sweep);
  if(total!=expected)throw Unsupported("incomplete output coverage");
  const auto convert=[](Edge e,const std::string& from,const std::string& to) {
    if(from!=to)appendConversion(e,"Convert("+from+" -> "+to+")",to);
    return e;
  };
  const auto type_of=[&](const Edge& e){return e.converted_type.empty()?b.doc.nodes.at(e.node).type:e.converted_type;};
  for(const auto& [key,piece]:pieces) {
    // Piece coordinates: index 0 is the slot axis; 1 + d is family dimension d.
    const unsigned piece_rank=unsigned(piece.fs.size())+1;
    std::vector<uint64_t> domain{slot->size};domain.insert(domain.end(),piece.fs.begin(),piece.fs.end());
    // A digit takes its Remainder unless its quotient stays below the radix (single terms and the
    // outermost digit of a fused dimension).
    const auto remainder=[&](const Term& t){return t.radix && t.radix<piece.fs[t.dim]/t.divisor;};
    const auto tensor_coordinates=[&](Builder& x) {
      std::vector<Edge> coords(rank,x.constant(0));
      for(unsigned t=0;t<rank;++t) {
        if(shape[t]<=1)continue;
        const auto fused=std::find_if(slot_axes.begin(),slot_axes.end(),[&](const Axis& s){return s.index==t;});
        if(fused!=slot_axes.end()) {
          // A digit of the fused slot coordinate; the last digit is the quotient (s < extent).
          Edge digit=x.index(0);
          if(fused->stride!=slot_axes.front().stride)digit=x.op("Divide",digit,x.constant(fused->stride/slot_axes.front().stride));
          if(std::next(fused)!=slot_axes.end())digit=x.op("Remainder",digit,x.constant(fused->size));
          coords[t]=digit;continue;
        }
        if(t==slot->index){coords[t]=x.index(0);continue;}
        coords[t]=axisCoordinate(x,piece.base[t],piece.terms[t],1,remainder);
      }
      return coords;
    };
    const auto left_coords=tensor_coordinates(a),right_coords=tensor_coordinates(b);
    Edge offset=b.constant(origin);
    for(unsigned t=0;t<rank;++t)offset=b.op("Add",offset,b.op("Multiply",right_coords[t],b.constant(steps[t])));
    RawNode cell;cell.kind="InputValue";cell.name=m.at("right_allocation").get<std::string>();cell.type=output_type;
    cell.operands={offset};
    Edge body=b.add(std::move(cell));
    std::vector<Edge> family_coords;for(unsigned d=0;d<piece.fs.size();++d)family_coords.push_back(b.index(1+d));
    for(const auto& arm:piece.arms) {
      Edge value=b.clone(right,arm.body,family_coords,unsigned(piece.fs.size()),piece_rank);
      if(arm.packed!=width && !extractLane(b,value,type_of(value),arm.packed,width,arm.slot))
        refuse("symbolic_output_placement_unsupported","packed output requires unsigned bits");
      if(const auto ty=type_of(value);ty!=output_type) {
        // A trailing Bitcast(output_type -> ty) of the use is undone exactly (a bit identity).
        const auto inverse="Bitcast("+output_type+" -> "+ty+")";
        if(value.conversion.ends_with(inverse)) {
          value.conversion.resize(value.conversion.size()-inverse.size());
          while(!value.conversion.empty() && (value.conversion.back()==' ' || value.conversion.back()==';'))value.conversion.pop_back();
          const auto arrow=value.conversion.rfind("-> ");
          value.converted_type=arrow==std::string::npos?std::string():value.conversion.substr(arrow+3,value.conversion.size()-arrow-4);
          if(type_of(value)!=output_type)refuse("symbolic_output_placement_unsupported","an output arm's type differs from the tensor's");
        } else appendConversion(value,"Bitcast("+ty+" -> "+output_type+")",output_type);
      }
      Edge atom=b.clone(right,arm.atom,{},0,piece_rank);
      const auto atom_type=type_of(atom);
      if(atom_type.empty() || (atom_type[0]!='s' && atom_type[0]!='u'))refuse("symbolic_output_placement_unsupported","non-integer placement atom");
      Edge position=convert(atom,atom_type,"s64");
      if(factor!=1)position=b.op("Multiply",position,b.constant(factor,"s64"),"s64");
      if(arm.slot_base)position=b.op("Add",b.constant(arm.slot_base,"s64"),position,"s64");
      RawNode select;select.kind="Operation";select.name="Select";select.type=output_type;
      select.operands={b.op("Equal",convert(b.index(0),"u64","s64"),position,"pred"),value,body};
      body=b.add(std::move(select));
    }
    std::vector<Edge> identity;for(unsigned i=0;i<piece_rank;++i)identity.push_back(b.index(i));
    const Edge reference_body=a.source(left,target,rank);
    emitPiece(a,b,maps,width,domain,reference_body,left_coords,body,identity);
  }
}
void partition(const Document& left,const Document& right,const J& m,Builder& a,Builder& b,J& maps) {
  auto shape=m.at("shape").get<std::vector<uint64_t>>();
  auto steps=m.at("right_strides_bytes").get<std::vector<uint64_t>>();
  const auto width=input_unsigned(m.at("element_bytes")),origin=input_unsigned(m.at("right_offset"));
  if(!width || shape.empty() || steps.size()!=shape.size())throw Unsupported("invalid output mapping");
  const J* lf=nullptr;std::vector<J> families;
  for(const auto& f:left.meta.at("output_families"))if(f.at("allocation")==m.at("left_allocation")){if(lf)throw Unsupported("reference requires one logical family per tensor");lf=&f;}
  if(!lf || lf->at("shape")!=m.at("shape") || lf->at("strides_bytes")!=m.at("left_strides_bytes") ||
     lf->at("offset")!=m.at("left_offset") || lf->at("element_bytes")!=width)throw Unsupported("reference output does not match mapping");
  for(const auto& f:right.meta.at("output_families"))if(f.at("allocation")==m.at("right_allocation"))families.push_back(f);
  if(std::any_of(families.begin(),families.end(),[&](const J& f){return placementRoot(right,f);})) {
    pullPartition(left,right,m,a,b,maps,*lf,families);return;
  }
  std::sort(families.begin(),families.end(),[](const J& x,const J& y){return x.at("offset")<y.at("offset");});
  std::vector<Axis> axes;uint64_t extent=width;
  for(unsigned i=0;i<shape.size();++i){if(!shape[i])throw Unsupported("empty output shape");if(shape[i]>1)axes.push_back({steps[i],shape[i],i});}
  std::sort(axes.begin(),axes.end(),[](Axis x,Axis y){return x.stride<y.stride;});
  for(auto ax:axes){if(ax.stride<extent || ax.stride%width)throw Unsupported("overlapping output layout");extent=checked(Z(extent)+Z(ax.stride)*(ax.size-1));}
  auto emit=[&](const J& f,const std::vector<Edge>& coords,const std::vector<Edge>& right_coords,std::vector<uint64_t> domain,unsigned slot=0) {
    // Import each source DAG once, retaining its binder levels. Coordinate
    // substitution belongs to the importer's existing memoized environment,
    // not to per-output raw-DAG cloning. Bindings are evaluated simultaneously
    // in the destination output scope before entering the source scope.
    auto av=a.source(left,Edge(lf->at("body")),shape.size());
    auto bv=b.source(right,Edge(f.at("body")),f.at("shape").size());
    uint64_t packed=f.at("element_bytes");
    auto ty=bv.converted_type.empty()?b.doc.nodes.at(bv.node).type:bv.converted_type;
    const Edge reference_body(lf->at("body"));
    auto output_type=reference_body.converted_type.empty()?left.nodes.at(reference_body.node).type:reference_body.converted_type;
    auto convert=[&](const std::string& step,const std::string& target){appendConversion(bv,step,target);};
    if(packed!=width) {
      if(!extractLane(b,bv,ty,packed,width,slot))throw Unsupported("packed output requires unsigned bits");
      if(output_type!=bv.converted_type)convert("Bitcast("+bv.converted_type+" -> "+output_type+")",output_type);
    } else if(const auto bits=std::to_string(width*8);ty=="b"+bits && (output_type=="s"+bits || output_type=="u"+bits)) {
      // Raw stored bytes (e.g. a compared slice of a wider store) are compared
      // as the reference's integer element, as the packed slots above. The
      // importer's output transport reinterprets raw bits as floating values.
      convert("Bitcast("+ty+" -> "+output_type+")",output_type);
    }
    emitPiece(a,b,maps,width,domain,av,coords,bv,right_coords);
  };
  // Scalar strided families are APs too. Routing them to the rectangle
  // certificate falsely requires every individual family to be dense.
  bool rectangular=false;for(const auto& f:families)if(f.at("shape").size()!=1)rectangular=true;
  if(rectangular) {
    // A packed family (element_bytes = P * width) is P families of width-byte
    // elements at byte offsets slot * width; emit extracts each slot exactly.
    // A piece refers to its family in families, which outlives pieces.
    // A piece covers the family coordinates first + [0, extent) (the whole family unless split below);
    // offset is the byte offset of its first element.
    struct Piece {const J* family;uint64_t offset,slot;std::vector<uint64_t> first,extent;};
    std::vector<Piece> pieces;
    for(const auto& f:families) {
      const auto packed=input_unsigned(f.at("element_bytes"));
      if(!packed || packed%width)throw Unsupported("invalid rectangular output");
      const auto fs=f.at("shape").get<std::vector<uint64_t>>();
      for(uint64_t slot=0;slot<packed/width;++slot)
        pieces.push_back({&f,checked(Z(input_unsigned(f.at("offset")))+Z(slot)*width),slot,std::vector<uint64_t>(fs.size()),fs});
    }
    // Refine physical tensor axes by the declared family strides. A block
    // [3,30] at strides [64,2] is rectangular in (row, column/32, column%32),
    // even though it is not an AP in the original column axis. Only exact
    // divisor chains are admitted; no elements or speculative layouts.
    const auto tensor_axes=axes;
    std::vector<uint64_t> grid_shape;
    std::vector<Axis> grid_axes;
    for(const auto& axis:tensor_axes) {
      std::vector<uint64_t> cuts{1,axis.size};
      for(const auto& piece:pieces) {
        const J& f=*piece.family;
        auto fs=f.at("shape").get<std::vector<uint64_t>>(),st=f.at("strides_bytes").get<std::vector<uint64_t>>();
        if(fs.size()!=st.size())throw Unsupported("invalid output axes");
        for(unsigned i=0;i<fs.size();++i)if(fs[i]>1 && st[i]%axis.stride==0) {
          auto c=st[i]/axis.stride;
          if(c>1 && c<axis.size && axis.size%c==0)cuts.push_back(c);
        }
      }
      std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
      bool chain=true;for(size_t i=1;i<cuts.size();++i)chain&=cuts[i]%cuts[i-1]==0;
      if(!chain)cuts={1,axis.size};
      for(size_t i=1;i<cuts.size();++i) {
        auto size=cuts[i]/cuts[i-1];
        grid_axes.push_back({checked(Z(axis.stride)*cuts[i-1]),size,unsigned(grid_shape.size())});
        grid_shape.push_back(size);
      }
    }
    axes=std::move(grid_axes);
    static constexpr auto box_messages=std::to_array<const char*>({"unsupported rectangular stride",
        "unsupported rectangular stride","unsupported fused-axis stride","fused axis crosses padding",
        "fused axis is not aligned","fused axis partial extent","output axes overlap or leave holes",
        "output extent outside tensor"});
    static_assert(box_messages.size()==box_error_count);
    const auto bound=[](std::optional<BoxError> e){if(e)throw Unsupported(box_messages[std::size_t(*e)]);};
    struct Box {std::vector<uint64_t> base,count,step;std::vector<std::vector<Term>> bindings;};
    std::vector<Box> boxes;std::vector<std::vector<Range>> ranges;Z total=0;
    // The box of a piece on the grid axes, or why it is none.
    const auto place=[&](const Piece& piece,Box& box)->std::optional<BoxError> {
      const J& f=*piece.family;
      auto fs=piece.extent,st=f.at("strides_bytes").get<std::vector<uint64_t>>();
      if(fs.empty() || fs.size()!=st.size())throw Unsupported("invalid rectangular output");
      box=Box{std::vector<uint64_t>(grid_shape.size()),std::vector<uint64_t>(grid_shape.size(),1),std::vector<uint64_t>(grid_shape.size(),1),std::vector<std::vector<Term>>(grid_shape.size())};
      auto rem=checked(Z(piece.offset)-origin);
      for(auto it=axes.rbegin();it!=axes.rend();++it){box.base[it->index]=rem/it->stride;rem%=it->stride;if(box.base[it->index]>=it->size)throw Unsupported("output outside tensor");}
      if(rem)throw Unsupported("output in padding");
      for(unsigned dim=0;dim<fs.size();++dim)
        if(fs[dim]>1)if(auto e=bindDimension(axes,box.base,std::nullopt,dim,fs[dim],st[dim],box.bindings))return e;
      for(unsigned axis=0;axis<grid_shape.size();++axis) {
        // Each tensor axis can be an AP, not necessarily a dense interval.
        // Mixed-radix family axes must still tile that AP exactly. Tiled on a copy: the
        // coordinates below add the digits in binding order.
        auto terms=box.bindings[axis];
        if(auto e=tileAxis(terms,box.base[axis],grid_shape[axis],box.step[axis],box.count[axis]))return e;
      }
      return std::nullopt;
    };
    std::vector<Piece> placed;
    const auto record=[&](const Piece& piece,Box box) {
      total+=product(box.count);
      auto& row=ranges.emplace_back();
      for(unsigned i=0;i<grid_shape.size();++i)row.push_back({box.base[i],checked(Z(box.base[i])+Z(box.step[i])*(box.count[i]-1)),box.step[i]});
      boxes.push_back(std::move(box));placed.push_back(piece);
    };
    for(const auto& piece:pieces) {
      Box box;
      const auto error=place(piece,box);
      if(!error){record(piece,std::move(box));continue;}
      // A piece is no box when a dimension crosses an axis boundary, e.g. a run of words that wraps
      // from the end of one row into the next. Its parts (splitPiece) are placed instead, all or none.
      const auto parts=splitPiece(axes,origin,piece.offset,piece.extent,piece.family->at("strides_bytes").get<std::vector<uint64_t>>());
      if(parts.empty())bound(error);
      for(const auto& cut:parts) {
        auto part=piece;part.offset=cut.offset;
        for(std::size_t d=0;d<part.first.size();++d){part.first[d]+=cut.first[d];part.extent[d]=cut.extent[d];}
        // A part that is no box, or lies outside the tensor, leaves the piece's own refusal.
        bool boxed=false;
        try{boxed=!place(part,box);}catch(const Unsupported&){}
        if(!boxed)bound(error);
        record(part,std::move(box));
      }
    }
    pieces=std::move(placed);
    // Pairwise disjointness, swept along the widest grid axis: only boxes whose
    // ranges on that axis meet are compared on every axis (not all pairs).
    // Without grid axes the tensor is one element, which every box covers.
    if(grid_shape.empty() && boxes.size()>1)throw Unsupported("overlapping outputs");
    unsigned sweep=0;
    for(unsigned i=1;i<grid_shape.size();++i)if(grid_shape[i]>grid_shape[sweep])sweep=i;
    requireDisjointBoxes(ranges,sweep);
    if(total!=product(shape))throw Unsupported("incomplete output coverage");
    for(size_t i=0;i<pieces.size();++i) {
      auto& box=boxes[i];std::vector<Edge> coords(shape.size(),a.constant(0)),right_coords;
      std::vector<Edge> grid_coords;
      for(unsigned axis=0;axis<grid_shape.size();++axis)
        grid_coords.push_back(axisCoordinate(a,box.base[axis],box.bindings[axis],0,
                                             [](const Term& t){return t.radix!=0;}));
      for(const auto& axis:axes) {
        auto parent=std::find_if(tensor_axes.rbegin(),tensor_axes.rend(),[&](Axis t){return axis.stride>=t.stride;});
        if(parent==tensor_axes.rend() || axis.stride%parent->stride)throw Unsupported("invalid output axis refinement");
        auto x=grid_coords[axis.index];auto factor=axis.stride/parent->stride;
        if(factor!=1)x=a.op("Multiply",x,a.constant(factor));
        coords[parent->index]=a.op("Add",coords[parent->index],x);
      }
      const J& f=*pieces[i].family;
      for(unsigned j=0;j<f.at("shape").size();++j)
        right_coords.push_back(pieces[i].first[j]?b.op("Add",b.constant(pieces[i].first[j]),b.index(j)):b.index(j));
      emit(f,coords,right_coords,pieces[i].extent,pieces[i].slot);
    }
    return;
  }
  // Every element is placed on a tensor element below (none in padding), the
  // elements are disjoint, and there are as many as tensor elements: a bijection.
  std::vector<Range> ranges;Z total=0;
  for(const auto& f:families) {
    auto packed=input_unsigned(f.at("element_bytes")),count=input_unsigned(f.at("shape")[0]),step=input_unsigned(f.at("strides_bytes")[0]);
    auto begin=checked(Z(input_unsigned(f.at("offset")))-origin);
    if(!packed || packed%width || !count || step<packed || step%width || begin%width || Z(begin)+Z(count-1)*step+packed>extent)throw Unsupported("invalid packed output range");
    total+=Z(count)*packed;
    for(uint64_t slot=0;slot<packed/width;++slot)ranges.push_back({begin/width+slot,checked(Z(begin/width+slot)+Z(count-1)*(step/width)),step/width});
  }
  if(total!=Z(product(shape))*width)throw Unsupported("incomplete output coverage");
  require_disjoint(ranges.size(),[&](std::size_t i){return ranges[i].first;},[&](std::size_t i){return ranges[i].last;},
    [&](std::size_t p,std::size_t b){return intersects(ranges[p],ranges[b]);});
  for(const auto& f:families) {
    uint64_t packed=f.at("element_bytes"),step=f.at("strides_bytes")[0],size=f.at("shape")[0];
    for(uint64_t slot=0;slot<packed/width;++slot)for(uint64_t start=0;start<size;) {
      auto rem=checked(Z(input_unsigned(f.at("offset")))-origin+Z(start)*step+Z(slot)*width);
      std::vector<uint64_t> base(shape.size()),residue(shape.size()),slopes(shape.size());
      for(auto it=axes.rbegin();it!=axes.rend();++it){base[it->index]=rem/it->stride;rem%=it->stride;residue[it->index]=rem;if(base[it->index]>=it->size)throw Unsupported("output outside tensor");}
      if(rem)throw Unsupported("output in padding");
      auto count=size-start,delta=step;
      for(auto it=axes.rbegin();it!=axes.rend();++it){if(delta%it->stride==0){auto slope=delta/it->stride;if(slope)count=std::min(count,(it->size-1-base[it->index])/slope+1);slopes[it->index]=slope;delta=0;}else count=std::min(count,(it->stride-1-residue[it->index])/delta+1);}
      // A step no axis absorbs moves inside one innermost element window, past
      // the element: the next element is placed (or refused) on its own.
      if(delta)count=std::min<uint64_t>(count,1);
      if(!count)throw Unsupported("empty output partition");
      std::vector<Edge> coords;for(unsigned j=0;j<shape.size();++j)coords.push_back(a.op("Add",a.constant(base[j]),a.op("Multiply",a.index(0),a.constant(slopes[j]))));
      emit(f,coords,{b.op("Add",b.constant(start),b.index(0))},{count},slot);start+=count;
    }
  }
}
}
Document fromArtifact(const sentryward::scalar_fold::Artifact& a) {
  Document d;d.nodes.reserve(a.graph.size());
  for(const auto& n:a.graph.nodes()) {
    RawNode r;r.kind=sentryward::scalar_fold::kindName(n.kind);r.name=n.name;r.type=n.type;r.attributes=n.attributes;r.level=n.index_level;
    r.instruction=a.graph.instructionModes(d.nodes.size());
    for(const auto& e:n.operands)r.operands.push_back(edge(e));
    if(n.fold){r.level=n.fold->binder_level;r.count=n.fold->count;r.reducer=sentryward::scalar_fold::reducerName(n.fold->reducer);}
    d.nodes.push_back(std::move(r));
  }
  d.meta={{"format","spverify.scalar-fold.v2"},{"numerical_semantics","ideal_real"},{"outputs",J::array()},{"output_families",J::array()}};
  for(const auto& f:a.output_families)d.meta["output_families"].push_back({{"allocation",f.allocation},{"offset",f.offset},{"element_bytes",f.element_bytes},{"shape",f.shape},{"strides_bytes",f.strides_bytes},{"body",edgeJson(edge(f.body))}});
  if(!a.outputs.empty())throw Unsupported("point outputs require a family adapter");
  return d;
}
namespace {
J outputTensors(const J& mapping) {
  const auto& outputs=mapping.at("outputs");
  return outputs.value("kind","")=="dense_tensor"?J::array({outputs}):outputs.at("tensors");
}
}
std::map<std::string,sentryward::scalar_fold::ComparedOutput> comparedOutputs(const J& mapping) {
  std::map<std::string,sentryward::scalar_fold::ComparedOutput> result;
  try {
    for(const auto& m:outputTensors(mapping)) {
      sentryward::scalar_fold::ComparedOutput layout{input_unsigned(m.at("right_offset")),input_unsigned(m.at("element_bytes")),
          m.at("shape").get<std::vector<uint64_t>>(),m.at("right_strides_bytes").get<std::vector<uint64_t>>()};
      if(!result.emplace(m.at("right_allocation").get<std::string>(),std::move(layout)).second)return {};
    }
  } catch(const std::exception&) {return {};}
  return result;
}
void alignOutputs(Document& left,Document& right,J& mapping) {
  if(left.meta.value("numerical_semantics","")!="ideal_real" || right.meta.value("numerical_semantics","")!="ideal_real")throw Unsupported("requires explicit ideal_real semantics");
  if(!left.meta.value("outputs",J::array()).empty() || !right.meta.value("outputs",J::array()).empty())throw Unsupported("point output mapping unsupported");
  auto tensors=outputTensors(mapping);
  if(tensors.empty())throw Unsupported("empty output mapping");
  std::set<std::string> la,ra;for(const auto& m:tensors){if(!la.insert(m.at("left_allocation")).second || !ra.insert(m.at("right_allocation")).second)throw Unsupported("duplicate mapped allocation");}
  for(const auto& f:left.meta.at("output_families"))if(!la.contains(f.at("allocation")))throw Unsupported("unmapped reference output");
  for(const auto& f:right.meta.at("output_families"))if(!ra.contains(f.at("allocation")))throw Unsupported("unmapped optimized output");
  // A placement annotation is only a family root (output_family.h); anywhere else (a
  // consolidated or nested use) its position would be lost.
  {
    std::set<Id> roots;
    for(const auto& f:right.meta.at("output_families"))if(placementRoot(right,f))roots.insert(Edge(f.at("body")).node);
    for(Id id=0;id<right.nodes.size();++id) {
      const auto& n=right.nodes[id];
      if(n.kind=="Operation" && n.name==sentryward::scalar_fold::kPlacementOperation && !roots.contains(id))
        throw Unsupported("symbolic placement outside an output family root","symbolic_output_placement_unsupported",J::object());
      for(const auto& e:n.operands)if(roots.contains(e.node))
        throw Unsupported("symbolic placement used as a value","symbolic_output_placement_unsupported",J::object());
    }
    for(Id id:roots) {
      // Atoms are exact input elements: no output coordinate or binder below them.
      std::vector<Id> work(right.nodes[id].operands.size()>1?right.nodes[id].operands.size()-1:0);
      for(std::size_t k=1;k<right.nodes[id].operands.size();++k)work[k-1]=right.nodes[id].operands[k].node;
      std::set<Id> seen;
      while(!work.empty()) {
        const auto x=work.back();work.pop_back();if(!seen.insert(x).second)continue;
        const auto& n=right.nodes.at(x);
        if(n.kind=="Index" || n.kind=="Fold")
          throw Unsupported("a placement atom depends on an output coordinate","symbolic_output_placement_unsupported",J::object());
        for(const auto& e:n.operands)work.push_back(e.node);
      }
    }
  }
  Builder a,b;J maps=J::array();for(const auto& m:tensors)partition(left,right,m,a,b,maps);
  left=std::move(a.doc);right=std::move(b.doc);mapping["outputs"]={{"kind","dense_tensors"},{"tensors",std::move(maps)}};
}
ComparisonResult compareArtifact(Document reference,const sentryward::scalar_fold::Artifact& optimized,J mapping,const ComparisonOptions& options,const ContextBuilder& build_context) {
  try {auto right=fromArtifact(optimized);alignOutputs(reference,right,mapping);return compare(std::move(reference),std::move(right),std::move(mapping),options,{}, {},build_context);}
  catch(const Unsupported& e){return ComparisonResult::fromReport({{"status","translation_or_supported_fragment_blocked"},{"blocked_stage","object_adapter"},{"reason",e.code},{"diagnostics",J::array({{{"code",e.code},{"stage","equivalence_import"},{"message",e.what()},{"details",e.details}}})}});}
  catch(const std::bad_alloc& e){return ComparisonResult::fromReport({{"status","allocation_failed"},{"blocked_stage","object_adapter"},{"reason","memory_allocation_failed"},{"message",e.what()}});}
  catch(const std::exception& e){return ComparisonResult::fromReport({{"status","internal_error"},{"blocked_stage","object_adapter"},{"reason","unexpected_exception"},{"message",e.what()}});}
}
}
