#pragma once
#include "sentryward/value_analysis/indexed_source.h"
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sentryward::value_analysis {
// Pass-local structural identity of immutable use DAGs. A proof allocation is
// not a coordinate scope. Child IDs keep the key local (no ancestor strings),
// and each stored use/edge is visited once. The caller supplies the existing
// complete annotated Value identity, including occurrence and result index.
class IndexedSourceIdentity final {
 public:
  using Ptr=std::shared_ptr<const IndexedSourceUse>;
  explicit IndexedSourceIdentity(std::function<std::string(const value_ir::Value&)> value_key)
      :value_key_(std::move(value_key)){}
  std::size_t use(const Ptr& root) {
    if(!root)return 0;
    if(auto i=memo_.find(root.get());i!=memo_.end())return i->second.id;
    struct Frame {Ptr value;std::size_t child=0;};
    std::vector<Frame> work{{root}};
    while(!work.empty()) {
      auto& frame=work.back();
      if(frame.child<frame.value->children.size()) {
        const auto child=frame.value->children[frame.child++];
        if(child && !memo_.contains(child.get()))work.push_back({child});
        continue;
      }
      const auto& node=*frame.value;
      std::string key;
      const auto field=[&](const std::string& text){key+=std::to_string(text.size())+":"+text;};
      const auto number=[&](auto n){field(std::to_string(n));};
      field(value_key_(node.prototype));number(static_cast<unsigned>(node.kind));
      number(bool(node.binding));
      if(node.binding) {
        const auto& b=*node.binding;
        field(value_key_(b.leaf));number(b.base);number(b.stride);number(b.fixed);
        number(b.producer_uses_bindings);number(b.period);number(b.outer_stride);
        number(bool(b.producer));if(b.producer)field(value_key_(*b.producer));
        number(bool(b.numerical_field));if(b.numerical_field)field(b.numerical_field->toString());
        number(bool(b.initial));
        if(b.initial) {
          const auto& initial=*b.initial;
          number(initial.id);field(initial.region.allocation);
          number(initial.region.offset);number(initial.region.bytes);field(initial.type.toString());
        }
        number(b.quotient_terms.size());
        for(const auto& term:b.quotient_terms){number(term.divisor);number(term.coefficient);}
      }
      number(node.children.size());
      for(const auto& child:node.children)number(child?memo_.at(child.get()).id:0);
      auto [entry,inserted]=interned_.try_emplace(std::move(key),interned_.size()+1);
      (void)inserted;
      // Retain ownership: address reuse must not alias a subsequently allocated
      // proof. The retained DAG is existing storage, not a copied subtree.
      memo_.emplace(frame.value.get(),Entry{frame.value,entry->second});
      work.pop_back();
    }
    return memo_.at(root.get()).id;
  }
  std::size_t storedUses()const{return memo_.size();}
  std::string context(const Ptr& root,std::string_view index,
                      std::span<const std::uint64_t> domains) {
    std::string key=std::to_string(use(root))+":"+std::to_string(index.size())+":"+std::string(index)+"#";
    for(auto extent:domains)key+=std::to_string(extent)+",";
    return key;
  }
 private:
  struct Entry {Ptr owner;std::size_t id;};
  std::function<std::string(const value_ir::Value&)> value_key_;
  std::unordered_map<const IndexedSourceUse*,Entry> memo_;
  std::unordered_map<std::string,std::size_t> interned_;
};
}
