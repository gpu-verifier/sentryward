#pragma once

#include "sentryward/scalar_fold/graph.h"
#include <algorithm>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <unordered_map>

namespace sentryward::equivalence {

// Exact ordering of the existing serialized structural keys, without building
// their expanded ancestor strings. Immutable shared node bodies are skipped by
// identity. Only each node's own header/footer is stored. No hashing is used as
// an equality proof, and conversion/provenance belong to each compared use.
// Valid until graph collection/remapping; appending immutable nodes is allowed.
class StructuralOrder {
 public:
  explicit StructuralOrder(const scalar_fold::Graph& graph,
                           std::function<void()> charge = {})
      : graph_(graph), charge_(std::move(charge)) {}
  bool less(const scalar_fold::Ref& a, const scalar_fold::Ref& b) {
    (void)graph_.node(a);(void)graph_.node(b);
    std::vector<Part> left{Part::reference(a)}, right{Part::reference(b)};
    while (!left.empty() && !right.empty()) {
      if (charge_) charge_();
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
      ++segments_;
#endif
      auto& x=left.back();auto& y=right.back();
      if ((x.kind==Kind::Body && y.kind==Kind::Body && x.node==y.node) ||
          (x.kind==Kind::Reference && y.kind==Kind::Reference && *x.ref==*y.ref)) {
        left.pop_back();right.pop_back();continue;
      }
      if(x.kind!=Kind::Text){expand(left);continue;}
      if(y.kind!=Kind::Text){expand(right);continue;}
      const auto count=std::min(x.text.size(),y.text.size());
      const auto order=x.text.substr(0,count).compare(y.text.substr(0,count));
      if(order)return order<0;
      x.text.remove_prefix(count);y.text.remove_prefix(count);
      if(x.text.empty())left.pop_back();
      if(y.text.empty())right.pop_back();
    }
    return left.empty() && !right.empty();
  }
  std::size_t visitedSegments() const {
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
    return segments_;
#else
    return 0;
#endif
  }
  std::size_t storedHeaders() const {return headers_.size();}
 private:
  enum class Kind {Text, Body, Reference};
  struct Part {
    Kind kind;std::string_view text;scalar_fold::Id node=0;
    const scalar_fold::Ref* ref=nullptr;
    static Part literal(std::string_view text){return {Kind::Text,text};}
    static Part body(scalar_fold::Id node){return {Kind::Body,{},node};}
    static Part reference(const scalar_fold::Ref& ref){return {Kind::Reference,{},0,&ref};}
  };
  struct Header {std::string prefix,suffix;};
  const scalar_fold::Graph& graph_;
  std::function<void()> charge_;
  std::unordered_map<scalar_fold::Id,Header> headers_;
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
  std::size_t segments_=0;
#endif
  const Header& header(scalar_fold::Id id) {
    auto it=headers_.find(id);if(it!=headers_.end())return it->second;
    const auto& n=graph_.node(id);std::ostringstream prefix,suffix;
    prefix<<int(n.kind)<<':'<<n.type<<':'<<n.name<<':'<<n.attributes<<':'<<n.index_level;
    if(n.kind==scalar_fold::Kind::Constant && n.type=="u64") {
      prefix.str("");prefix<<"C"<<std::setw(20)<<std::setfill('0')<<std::stoull(n.name);
    }
    if(n.fold)suffix<<':'<<int(n.fold->reducer)<<':'<<n.fold->count<<':'<<n.fold->binder_level;
    return headers_.emplace(id,Header{prefix.str(),suffix.str()}).first->second;
  }
  void expand(std::vector<Part>& stack) {
    const auto item=stack.back();stack.pop_back();
    if(item.kind==Kind::Reference) {
      const auto& ref=*item.ref;
      // Reversed push order is the original "/conversion/type/provenance".
      if(!ref.pointer_provenance.empty())stack.push_back(Part::literal(ref.pointer_provenance));
      stack.push_back(Part::literal("/"));
      if(!ref.converted_type.empty())stack.push_back(Part::literal(ref.converted_type));
      stack.push_back(Part::literal("/"));
      if(!ref.conversion.empty())stack.push_back(Part::literal(ref.conversion));
      stack.push_back(Part::literal("/"));stack.push_back(Part::body(ref.node));
    } else {
      const auto& n=graph_.node(item.node);const auto& text=header(item.node);
      if(!text.suffix.empty())stack.push_back(Part::literal(text.suffix));
      for(auto it=n.operands.rbegin();it!=n.operands.rend();++it) {
        stack.push_back(Part::literal("]"));stack.push_back(Part::reference(*it));
        stack.push_back(Part::literal("["));
      }
      stack.push_back(Part::literal(text.prefix));
    }
  }
};

}  // namespace sentryward::equivalence
