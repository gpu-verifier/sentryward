#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::freeze(std::span<const Class> roots) -> FrozenIndex {
  FrozenIndex result;result.finite.resize(nodes_.size());
  for(auto root:roots)result.imported_roots.push_back(find(root));
  for(Class id=0;id<nodes_.size();++id)result.finite[find(id)]=finite_[find(id)];
  for(Class id=0;id<nodes_.size();++id) {
    auto key=nodes_[id];for(auto& child:key.children)child=find(child);
    canonicalizeCommutative(key);
    if(idealOperation(key))key.attributes="ideal_arithmetic";
    result.memo.emplace(std::move(key),find(id));
  }
  return result;
}

inline auto Engine::import(const s::Graph& graph, std::span<const s::Ref> roots,
                          std::vector<unsigned char>* storage_roots,
                          std::span<const s::Ref> witnesses,std::vector<Class>* witness_outputs,
                          bool captured_coordinate) -> std::vector<Class> {
  graph.validateRoots(roots,captured_coordinate?1:options_.outer_binders);
  const auto normalizeUse = [&](s::Ref ref) {
    for(;;) {
      poll();
      if(!ref.conversion.empty()) {
        if(auto inactive=normalizeInactiveFiniteSaturation(graph,ref))ref=std::move(*inactive);
        if(auto normalized=normalizeConversions(graph.node(ref).type,graph.type(ref),ref.conversion)) {
          ref.conversion=*normalized;
          if(ref.conversion.empty())ref.converted_type.clear();
        }
      }
      auto reduced=packedLowUse(graph,ref);
      if(!reduced)reduced=packedHighUse(graph,ref);
      if(!reduced)return ref;
      ref=std::move(*reduced);
    }
  };
  std::vector<bool> live(graph.size(), false);
  std::vector<s::Id> pending, order;
  for (const auto& root : roots) { poll(); pending.push_back(normalizeUse(root).node); }
  while (!pending.empty()) {
    poll();
    auto id = pending.back(); pending.pop_back();
    if (live[id]) continue;
    live[id] = true; order.push_back(id);
    for (const auto& operand : graph.node(id).operands) pending.push_back(normalizeUse(operand).node);
  }
  std::sort(order.begin(), order.end());
  const bool certify_storage=storage_roots && options_.storage_output_bytes &&
      options_.storage_output_bytes<=8;
  std::vector<unsigned char> storage_nodes(certify_storage?graph.size():0);
  const auto storageUse=[&](const s::Ref& ref) {
    if(!certify_storage || !storage_nodes.at(ref.node) || !ref.pointer_provenance.empty())return false;
    const auto expected=8*options_.storage_output_bytes;
    auto current=graph.node(ref).type;
    if(bitWidth(current)!=expected)return false;
    // Inspect the original edge, before ideal/numeric canonicalization.
    // Only exact, same-width bitcasts may carry a bit-copy certificate.
    for(std::size_t position=0;position<ref.conversion.size();) {
      poll();const auto end=ref.conversion.find("; ",position);
      const auto step=ref.conversion.substr(position,end==std::string::npos?end:end-position);
      const auto arrow=step.find(" -> ");
      if(!step.starts_with("Bitcast(") || arrow==std::string::npos || step.back()!=')')return false;
      const auto from=step.substr(8,arrow-8),to=step.substr(arrow+4,step.size()-arrow-5);
      if(from!=current || bitWidth(from)!=expected || bitWidth(to)!=expected)return false;
      current=to;
      if(end==std::string::npos)break;
      position=end+2;
    }
    return current==graph.type(ref);
  };
  struct FoldPrefix { std::uint64_t count; Class initial, imported_body; };
  std::unordered_map<s::Id, FoldPrefix> prefixes;
  std::unordered_map<s::Id, Class> mapping;
  std::unordered_map<s::Id, std::optional<ConstantValue>> constants;
  const auto knownUse = [&](const s::Ref& ref)->std::optional<ConstantValue> {
    if(!ref.pointer_provenance.empty())return {};
    const auto found=constants.find(ref.node);if(found==constants.end())return {};
    const auto& value=found->second;if(!value)return {};
    return constantValue(value->type,value->literal,ref.conversion,ref.converted_type);
  };
  const auto use = [&](s::Ref ref) {
    ref=normalizeUse(std::move(ref));
    if(auto value=knownUse(ref)) {
      Key constant;constant.kind=static_cast<int>(s::Kind::Constant);
      constant.type=value->type;constant.name=value->literal;return intern(std::move(constant));
    }
    Class base = mapping.at(ref.node);
    if(ref.pointer_provenance.empty() && graph.node(ref).kind==s::Kind::InitialCell) {
      if(auto cast=leadingBitcast(graph.node(ref).type,ref.conversion)) {
        auto read=nodes_[base];read.type=cast->target;
        base=intern(std::move(read));ref.conversion=cast->remainder;
      }
    }
    if (ref.conversion.empty() && ref.pointer_provenance.empty()) return base;
    if(ref.pointer_provenance.empty() && ref.conversion.find("; ")!=std::string::npos) {
      auto text=ref.conversion;
      while(!text.empty()) {
        const auto end=text.find("; ");const auto step=text.substr(0,end);
        const auto arrow=step.find(" -> ");
        if(arrow==std::string::npos || step.back()!=')')throw std::invalid_argument("invalid_conversion_chain");
        Key k;k.kind=6;k.name=step;k.type=step.substr(arrow+4,step.size()-arrow-5);k.children={base};
        base=intern(std::move(k));
        if(end==std::string::npos)break;text.erase(0,end+2);
      }
      return base;
    }
    Key k;
    k.kind = 6;
    k.type = graph.type(ref);
    k.name = ref.conversion;
    k.provenance = ref.pointer_provenance;
    k.children = {base};
    return intern(std::move(k));
  };
  std::unordered_map<s::Id,IndexAffine> index_affines;
  for (auto id : order) {
    poll();
    const auto& node = graph.node(id);
    if(certify_storage) {
      bool copy=node.kind==s::Kind::InitialCell && !node.name.empty() &&
          node.attributes.empty() && !node.fold && node.operands.size()==1 &&
          node.operands[0].pointer_provenance.empty() && graph.type(node.operands[0])=="u64" &&
          bitWidth(node.type)==8*options_.storage_output_bytes;
      if(node.kind==s::Kind::Operation && node.name=="Select" && !node.fold && node.operands.size()==3 &&
          (node.attributes.empty() || node.attributes=="ideal_arithmetic") &&
          graph.type(node.operands[0])=="pred" && node.operands[0].conversion.empty() &&
          node.operands[0].pointer_provenance.empty() &&
          graph.type(node.operands[1])==node.type && graph.type(node.operands[2])==node.type)
        copy=storageUse(node.operands[1]) && storageUse(node.operands[2]);
      storage_nodes[id]=copy;
    }
    Key k{static_cast<int>(node.kind), node.type, node.name, node.attributes, {},
          node.fold, node.index_level, {}};
    if(captured_coordinate && node.kind==s::Kind::Index) {
      if(node.index_level)throw std::invalid_argument("captured_coordinate_rank");
      k.index_level=branch_fact_binder;
    }
    for (const auto& operand : node.operands) k.children.push_back(use(operand));
    std::optional<ConstantValue> evaluated;
    if(node.kind==s::Kind::Constant)evaluated=constantValue(node.type,node.name);
    else if(node.kind==s::Kind::Operation) {
      std::vector<ConstantValue> args;bool known=true;
      for(const auto& operand:node.operands) {
        auto value=knownUse(operand);if(!value){known=false;break;}
        args.push_back(std::move(*value));
      }
      if(known)evaluated=constantOperation(node.name,node.type,node.attributes,args);
    }
    constants.emplace(id,evaluated);
    if(evaluated) {
      k=Key{};k.kind=static_cast<int>(s::Kind::Constant);
      k.type=evaluated->type;k.name=evaluated->literal;
    }
    if(node.kind==s::Kind::Constant && !evaluated && k.type!="real" && realType(k.type) && k.attributes.empty() &&
        !floatingInfinity(k.type,k.name))
      k.attributes="encoding="+k.type;
    std::optional<IndexAffine> form;
    if(node.type=="u64" && node.attributes.empty()) {
      if(node.kind==s::Kind::Index)form=IndexAffine{0,{{k.index_level,1}}};
      else if(evaluated && evaluated->type=="u64") {
        std::uint64_t value=0;
        const auto& text=evaluated->literal;
        auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
        if(parsed.ec==std::errc{} && parsed.ptr==text.data()+text.size())form=IndexAffine{value,{}};
      } else if(node.kind==s::Kind::Operation && node.operands.size()==2) {
        const auto& x=node.operands[0];const auto& y=node.operands[1];
        if(x.conversion.empty() && y.conversion.empty() && x.pointer_provenance.empty() &&
            y.pointer_provenance.empty() && index_affines.contains(x.node) && index_affines.contains(y.node))
          form=combineAffine(node.name,index_affines.at(x.node),index_affines.at(y.node));
      }
    }
    if(form) {
      index_affines.emplace(id,*form);
      const auto constant=[&](std::uint64_t value) {
        Key key;key.kind=static_cast<int>(s::Kind::Constant);key.type="u64";
        key.name=std::to_string(value);return intern(std::move(key));
      };
      std::optional<Class> canonical;
      if(form->constant || form->coefficients.empty())canonical=constant(form->constant);
      for(auto [level,coefficient]:form->coefficients) {
        Key index;index.kind=static_cast<int>(s::Kind::Index);index.type="u64";index.index_level=level;
        auto term=intern(std::move(index));
        if(coefficient!=1) {
          Key scale;scale.kind=static_cast<int>(s::Kind::Operation);scale.type="u64";
          scale.name="Multiply";scale.children={term,constant(coefficient)};term=intern(std::move(scale));
        }
        if(canonical) {
          Key sum;sum.kind=static_cast<int>(s::Kind::Operation);sum.type="u64";
          sum.name="Add";sum.children={*canonical,term};canonical=intern(std::move(sum));
        } else canonical=term;
      }
      mapping.emplace(id,*canonical);continue;
    }
    // Type validation is independent of whether typed arithmetic permits
    // dropping rounding/FTZ. Numeric projection must not repair invalid IR.
    const auto legacy_primitive=node.operands.size()==1?
        legacyIdealPrimitive(k,graph.type(node.operands[0])):std::nullopt;
    if (idealOperation(k,true) || legacy_primitive)
      for (const auto& operand : node.operands)
        if (graph.type(operand) != k.type)
          throw std::invalid_argument("arithmetic operand/result types differ");
    if(comparison(k) && graph.type(node.operands[0])!=graph.type(node.operands[1]))
      throw std::invalid_argument("comparison operand types differ");
    if(idealSelect(k) && (graph.type(node.operands[0])!="pred" ||
        graph.type(node.operands[1])!=k.type || graph.type(node.operands[2])!=k.type))
      throw std::invalid_argument("Select predicate/branch types differ");
    if(node.kind==s::Kind::Fold && realType(node.type)) {
      const auto& initial=node.operands[0];
      auto prefix=prefixes.find(initial.node);
      if(initial.conversion.empty() && initial.pointer_provenance.empty() &&
          prefix!=prefixes.end()) {
        const auto& previous=graph.node(initial);
        if(previous.type==node.type && previous.fold->reducer==node.fold->reducer &&
            previous.fold->binder_level==node.fold->binder_level &&
            node.fold->count<=std::numeric_limits<std::uint64_t>::max()-prefix->second.count &&
            shiftedProjectedBody(prefix->second.imported_body,k.children[1],
                                 node.fold->binder_level,prefix->second.count,node.fold->count)) {
          k.fold->count+=prefix->second.count;
          k.children={prefix->second.initial,prefix->second.imported_body};
          prefixes.emplace(id,FoldPrefix{k.fold->count,k.children[0],k.children[1]});
        }
      }
      if(!prefixes.contains(id))
        prefixes.emplace(id,FoldPrefix{node.fold->count,k.children[0],k.children[1]});
    }
    const auto imported=intern(std::move(k));
    mapping.emplace(id, imported);
  }
  std::vector<Class> result;
  result.reserve(roots.size());
  for (const auto& root : roots) { poll(); result.push_back(use(root)); }
  if(storage_roots) {
    storage_roots->clear();storage_roots->reserve(roots.size());
    for(const auto& root:roots) {poll();storage_roots->push_back(storageUse(root));}
  }
  if(witness_outputs) {
    witness_outputs->clear();witness_outputs->reserve(witnesses.size());
    for(const auto& ref:witnesses) {
      poll();if(ref.node>=live.size() || !live[ref.node])throw std::invalid_argument("scoped_witness_not_reachable");
      witness_outputs->push_back(use(ref));
    }
  }
  return result;
}

} // namespace
} // namespace sentryward::equivalence
