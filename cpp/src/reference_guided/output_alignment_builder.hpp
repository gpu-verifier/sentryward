#pragma once
#include "output_adapter_utils.hpp"

namespace rg::output_alignment_detail {
struct OutputAlignmentBuilder {
  Document doc;
  std::unordered_map<std::string,Id> intern;
  std::map<std::string,std::unordered_map<Id,Edge>> contexts;
  OutputAlignmentBuilder(){doc.meta={{"format","spverify.scalar-fold.v2"},{"numerical_semantics","ideal_real"},{"outputs",J::array()},{"output_families",J::array()}};}
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
    auto& memo=contexts[key.dump()]; // Each OutputAlignmentBuilder has exactly one source Document.
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
} // namespace rg::output_alignment_detail
