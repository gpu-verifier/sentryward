#include "production.hpp"
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
    auto text=key.dump();auto it=intern.find(text);if(it!=intern.end())return Edge(it->second);
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
    if(packed!=width) {
      auto bits=packed*8,small=width*8;auto ty=bv.converted_type.empty()?b.doc.nodes.at(bv.node).type:bv.converted_type;
      if(ty!="b"+std::to_string(bits) && ty!="u"+std::to_string(bits))throw Unsupported("packed output requires unsigned bits");
      auto convert=[&](std::string step,std::string target){if(!bv.conversion.empty())bv.conversion+="; ";bv.conversion+=step;bv.converted_type=std::move(target);};
      if(ty!="u"+std::to_string(bits))convert("Bitcast("+ty+" -> u"+std::to_string(bits)+")","u"+std::to_string(bits));
      if(slot)bv=b.op("ShiftRight",bv,b.constant(slot*small,"u"+std::to_string(bits)),"u"+std::to_string(bits));
      Edge target(lf->at("body"));auto output_type=target.converted_type.empty()?left.nodes.at(target.node).type:target.converted_type;
      convert("Convert(u"+std::to_string(bits)+" -> u"+std::to_string(small)+")","u"+std::to_string(small));
      if(output_type!=bv.converted_type)convert("Bitcast("+bv.converted_type+" -> "+output_type+")",output_type);
    }
    std::vector<uint64_t> strides(domain.size());auto s=width;for(size_t i=domain.size();i-->0;){strides[i]=s;s=checked(Z(s)*domain[i]);}
    auto name="output-piece-"+std::to_string(maps.size());
    J common={{"allocation",name},{"offset",0},{"element_bytes",width},{"shape",domain},{"strides_bytes",strides}};
    auto bindings=[](const std::vector<Edge>& xs){J result=J::array();for(const auto& x:xs)result.push_back(edgeJson(x));return result;};
    common["body"]=edgeJson(av);common["coordinate_bindings"]=bindings(coords);a.doc.meta["output_families"].push_back(common);
    common["body"]=edgeJson(bv);common["coordinate_bindings"]=bindings(right_coords);b.doc.meta["output_families"].push_back(common);
    maps.push_back({{"kind","dense_tensor"},{"left_allocation",name},{"right_allocation",name},{"left_offset",0},{"right_offset",0},
      {"element_bytes",width},{"shape",domain},{"left_strides_bytes",strides},{"right_strides_bytes",strides}});
  };
  // Scalar strided families are APs too. Routing them to the rectangle
  // certificate falsely requires every individual family to be dense.
  bool rectangular=false;for(const auto& f:families)if(f.at("shape").size()!=1)rectangular=true;
  if(rectangular) {
    // A packed family (element_bytes = P * width) is P families of width-byte
    // elements at byte offsets slot * width; emit extracts each slot exactly.
    std::vector<std::pair<J,uint64_t>> pieces;
    for(const auto& f:families) {
      const auto packed=input_unsigned(f.at("element_bytes"));
      if(!packed || packed%width)throw Unsupported("invalid rectangular output");
      for(uint64_t slot=0;slot<packed/width;++slot) {
        auto piece=f;piece["offset"]=checked(Z(input_unsigned(f.at("offset")))+Z(slot)*width);
        pieces.emplace_back(std::move(piece),slot);
      }
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
      for(const auto& [f,slot]:pieces) {
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
    struct Term {unsigned dim;uint64_t divisor,radix,slope,count;};
    struct Box {std::vector<uint64_t> base,count,step;std::vector<std::vector<Term>> bindings;};
    std::vector<Box> boxes;Z total=0;
    for(const auto& [f,slot]:pieces) {
      auto fs=f.at("shape").get<std::vector<uint64_t>>(),st=f.at("strides_bytes").get<std::vector<uint64_t>>();
      if(fs.empty() || fs.size()!=st.size())throw Unsupported("invalid rectangular output");
      Box box{std::vector<uint64_t>(grid_shape.size()),std::vector<uint64_t>(grid_shape.size(),1),std::vector<uint64_t>(grid_shape.size(),1),std::vector<std::vector<Term>>(grid_shape.size())};
      auto rem=checked(Z(input_unsigned(f.at("offset")))-origin);
      for(auto it=axes.rbegin();it!=axes.rend();++it){box.base[it->index]=rem/it->stride;rem%=it->stride;if(box.base[it->index]>=it->size)throw Unsupported("output outside tensor");}
      if(rem)throw Unsupported("output in padding");
      for(unsigned dim=0;dim<fs.size();++dim) {
        if(fs[dim]<=1)continue;
        auto reverse=std::find_if(axes.rbegin(),axes.rend(),[&](Axis x){return st[dim]>0 && st[dim]%x.stride==0;});
        auto it=reverse==axes.rend()?axes.end():std::prev(reverse.base());
        if(it==axes.end())throw Unsupported("unsupported rectangular stride");
        auto slope=st[dim]/it->stride;
        if(Z(slope)*(fs[dim]-1)<it->size-box.base[it->index]){box.bindings[it->index].push_back({dim,1,0,slope,fs[dim]});continue;}
        if(slope!=1)throw Unsupported("unsupported fused-axis stride");
        auto remaining=fs[dim],divisor=uint64_t(1),physical=st[dim];
        while(remaining>1) {
          if(it==axes.end() || it->stride!=physical)throw Unsupported("fused axis crosses padding");
          auto n=std::min(remaining,it->size);
          if(remaining>=it->size && box.base[it->index])throw Unsupported("fused axis is not aligned");
          if(remaining%n)throw Unsupported("fused axis partial extent");
          box.bindings[it->index].push_back({dim,divisor,n,1,n});remaining/=n;divisor=checked(Z(divisor)*n);physical=checked(Z(physical)*n);++it;
        }
      }
      for(unsigned axis=0;axis<grid_shape.size();++axis) {
        auto terms=box.bindings[axis];std::sort(terms.begin(),terms.end(),[](Term x,Term y){return x.slope<y.slope;});
        // Each tensor axis can be an AP, not necessarily a dense interval.
        // Mixed-radix family axes must still tile that AP exactly.
        auto step=terms.empty()?uint64_t(1):terms.front().slope;
        auto count=uint64_t(1);for(auto t:terms){if(Z(t.slope)!=Z(step)*count)throw Unsupported("output axes overlap or leave holes");count=checked(Z(count)*t.count);}
        if(Z(box.base[axis])+Z(step)*(count-1)>=grid_shape[axis])throw Unsupported("output extent outside tensor");
        box.count[axis]=count;box.step[axis]=step;
      }
      total+=product(box.count);boxes.push_back(std::move(box));
    }
    // Pairwise disjointness, swept along the widest grid axis: only boxes whose
    // ranges on that axis meet are compared on every axis (not all pairs).
    std::vector<std::vector<Range>> ranges(boxes.size());
    for(std::size_t b=0;b<boxes.size();++b)for(unsigned i=0;i<grid_shape.size();++i) {
      const auto& box=boxes[b];
      ranges[b].push_back({box.base[i],checked(Z(box.base[i])+Z(box.step[i])*(box.count[i]-1)),box.step[i]});
    }
    unsigned sweep=0;
    for(unsigned i=1;i<grid_shape.size();++i)if(grid_shape[i]>grid_shape[sweep])sweep=i;
    std::vector<std::size_t> order(boxes.size());std::iota(order.begin(),order.end(),std::size_t(0));
    std::sort(order.begin(),order.end(),[&](std::size_t x,std::size_t y){return ranges[x][sweep].first<ranges[y][sweep].first;});
    std::vector<std::size_t> active;
    for(auto b:order) {
      std::erase_if(active,[&](std::size_t p){return ranges[p][sweep].last<ranges[b][sweep].first;});
      for(auto p:active){bool overlap=true;for(unsigned i=0;i<grid_shape.size() && overlap;++i)
        overlap=intersects(ranges[b][i],ranges[p][i]);
        if(overlap)throw Unsupported("overlapping outputs");}
      active.push_back(b);
    }
    if(total!=product(shape))throw Unsupported("incomplete output coverage");
    for(size_t i=0;i<pieces.size();++i) {
      auto& box=boxes[i];std::vector<Edge> coords(shape.size(),a.constant(0)),right_coords;
      std::vector<Edge> grid_coords;
      for(unsigned axis=0;axis<grid_shape.size();++axis){auto c=a.constant(box.base[axis]);for(auto t:box.bindings[axis]){auto x=a.index(t.dim);if(t.divisor!=1)x=a.op("Divide",x,a.constant(t.divisor));if(t.radix)x=a.op("Remainder",x,a.constant(t.radix));if(t.slope!=1)x=a.op("Multiply",x,a.constant(t.slope));c=a.op("Add",c,x);}grid_coords.push_back(c);}
      for(const auto& axis:axes) {
        auto parent=std::find_if(tensor_axes.rbegin(),tensor_axes.rend(),[&](Axis t){return axis.stride>=t.stride;});
        if(parent==tensor_axes.rend() || axis.stride%parent->stride)throw Unsupported("invalid output axis refinement");
        auto x=grid_coords[axis.index];auto factor=axis.stride/parent->stride;
        if(factor!=1)x=a.op("Multiply",x,a.constant(factor));
        coords[parent->index]=a.op("Add",coords[parent->index],x);
      }
      const auto& [f,slot]=pieces[i];
      for(unsigned j=0;j<f.at("shape").size();++j)right_coords.push_back(b.index(j));
      emit(f,coords,right_coords,f.at("shape").get<std::vector<uint64_t>>(),slot);
    }
    return;
  }
  std::vector<Range> ranges;Z total=0;bool strided=false;
  for(const auto& f:families) {
    auto packed=input_unsigned(f.at("element_bytes")),count=input_unsigned(f.at("shape")[0]),step=input_unsigned(f.at("strides_bytes")[0]);
    auto begin=checked(Z(input_unsigned(f.at("offset")))-origin);
    if(!packed || packed%width || !count || step<packed || step%width || begin%width || Z(begin)+Z(count-1)*step+packed>extent)throw Unsupported("invalid packed output range");
    strided|=count>1 && step!=packed;total+=Z(count)*packed;
    for(uint64_t slot=0;slot<packed/width;++slot)ranges.push_back({begin/width+slot,checked(Z(begin/width+slot)+Z(count-1)*(step/width)),step/width});
  }
  if(total!=Z(product(shape))*width || (strided && extent!=total))throw Unsupported("incomplete output coverage");
  std::sort(ranges.begin(),ranges.end(),[](Range x,Range y){return x.first<y.first;});std::vector<Range> active;
  for(auto r:ranges){std::erase_if(active,[&](Range x){return x.last<r.first;});for(auto x:active)if(intersects(x,r))throw Unsupported("overlapping outputs");active.push_back(r);}
  for(const auto& f:families) {
    uint64_t packed=f.at("element_bytes"),step=f.at("strides_bytes")[0],size=f.at("shape")[0];
    for(uint64_t slot=0;slot<packed/width;++slot)for(uint64_t start=0;start<size;) {
      auto rem=checked(Z(input_unsigned(f.at("offset")))-origin+Z(start)*step+Z(slot)*width);
      std::vector<uint64_t> base(shape.size()),residue(shape.size()),slopes(shape.size());
      for(auto it=axes.rbegin();it!=axes.rend();++it){base[it->index]=rem/it->stride;rem%=it->stride;residue[it->index]=rem;if(base[it->index]>=it->size)throw Unsupported("output outside tensor");}
      if(rem)throw Unsupported("output in padding");
      auto count=size-start,delta=step;
      for(auto it=axes.rbegin();it!=axes.rend();++it){if(delta%it->stride==0){auto slope=delta/it->stride;if(slope)count=std::min(count,(it->size-1-base[it->index])/slope+1);slopes[it->index]=slope;delta=0;}else count=std::min(count,(it->stride-1-residue[it->index])/delta+1);}
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
    for(const auto& e:n.operands)r.operands.push_back(edge(e));
    if(n.fold){r.level=n.fold->binder_level;r.count=n.fold->count;r.reducer=sentryward::scalar_fold::reducerName(n.fold->reducer);}
    d.nodes.push_back(std::move(r));
  }
  d.meta={{"format","spverify.scalar-fold.v2"},{"numerical_semantics","ideal_real"},{"outputs",J::array()},{"output_families",J::array()}};
  for(const auto& f:a.output_families)d.meta["output_families"].push_back({{"allocation",f.allocation},{"offset",f.offset},{"element_bytes",f.element_bytes},{"shape",f.shape},{"strides_bytes",f.strides_bytes},{"body",edgeJson(edge(f.body))}});
  if(!a.outputs.empty())throw Unsupported("point outputs require a family adapter");
  return d;
}
void alignOutputs(Document& left,Document& right,J& mapping) {
  if(left.meta.value("numerical_semantics","")!="ideal_real" || right.meta.value("numerical_semantics","")!="ideal_real")throw Unsupported("requires explicit ideal_real semantics");
  if(!left.meta.value("outputs",J::array()).empty() || !right.meta.value("outputs",J::array()).empty())throw Unsupported("point output mapping unsupported");
  auto tensors=mapping.at("outputs").value("kind","")=="dense_tensor"?J::array({mapping.at("outputs")}):mapping.at("outputs").at("tensors");
  if(tensors.empty())throw Unsupported("empty output mapping");
  std::set<std::string> la,ra;for(const auto& m:tensors){if(!la.insert(m.at("left_allocation")).second || !ra.insert(m.at("right_allocation")).second)throw Unsupported("duplicate mapped allocation");}
  for(const auto& f:left.meta.at("output_families"))if(!la.contains(f.at("allocation")))throw Unsupported("unmapped reference output");
  for(const auto& f:right.meta.at("output_families"))if(!ra.contains(f.at("allocation")))throw Unsupported("unmapped optimized output");
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
