#include "sentryward/equivalence/output_family.h"
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>

namespace sentryward::equivalence {
namespace {
std::uint64_t validate(const scalar_fold::OutputFamily& family) {
  if(family.allocation.empty() || !family.element_bytes || family.shape.empty() ||
      family.shape.size()!=family.strides_bytes.size() || family.shape.size()>UINT32_MAX)
    throw std::invalid_argument("invalid_output_family_domain");
  std::vector<std::pair<std::uint64_t,std::uint64_t>> axes;
  for(std::size_t i=0;i<family.shape.size();++i) {
    if(!family.shape[i])throw std::invalid_argument("empty_output_family_axis");
    if(family.shape[i]>1)axes.emplace_back(family.strides_bytes[i],family.shape[i]);
  }
  std::sort(axes.begin(),axes.end());
  auto extent=family.element_bytes;
  for(const auto& [stride,count]:axes) {
    if(stride!=extent)throw std::invalid_argument("output_family_not_dense_permutation");
    if(count>UINT64_MAX/extent)throw std::invalid_argument("output_family_domain_overflow");
    extent*=count;
  }
  if(extent>UINT64_MAX-family.offset)throw std::invalid_argument("output_family_domain_overflow");
  return extent;
}
std::uint64_t validatePair(const scalar_fold::OutputFamily& a,
                           const scalar_fold::OutputFamily& b,
                           const TensorOutputMapping& mapping) {
  if(a.allocation!=mapping.left_allocation || b.allocation!=mapping.right_allocation ||
      a.offset!=mapping.left_offset || b.offset!=mapping.right_offset ||
      a.element_bytes!=mapping.element_bytes || b.element_bytes!=mapping.element_bytes ||
      a.shape!=mapping.shape || b.shape!=mapping.shape ||
      a.strides_bytes!=mapping.left_strides_bytes || b.strides_bytes!=mapping.right_strides_bytes)
    throw std::invalid_argument("output_family_mapping_mismatch");
  const auto bytes=validate(a);
  if(validate(b)!=bytes)throw std::invalid_argument("output_family_domain_mismatch");
  return bytes;
}
struct Reshaped {
  scalar_fold::Graph graph;
  scalar_fold::OutputFamily family;
};
Reshaped reshape(const scalar_fold::Graph& source,const scalar_fold::OutputFamily& family,
                 const std::vector<std::uint64_t>& shape,const std::vector<std::uint64_t>& strides) {
  namespace s=scalar_fold;
  const auto bytes=validate(family);
  source.validateRoots(std::span<const s::Ref>(&family.body,1),family.shape.size());
  Reshaped result;result.family=family;
  result.family.shape=shape;result.family.strides_bytes=strides;
  if(validate(result.family)!=bytes)throw std::invalid_argument("output_family_domain_mismatch");
  auto& graph=result.graph;
  auto coordinate=graph.constant("u64","0");
  for(std::size_t i=0;i<shape.size();++i) {
    if(shape[i]==1)continue;
    auto term=graph.index(i);
    if(strides[i]!=1)
      term=graph.operation("Multiply","u64",{term,graph.constant("u64",std::to_string(strides[i]))});
    coordinate=graph.operation("Add","u64",{coordinate,term});
  }
  std::vector<s::Ref> coordinates;
  for(std::size_t i=0;i<family.shape.size();++i) {
    if(family.shape[i]==1){coordinates.push_back(graph.constant("u64","0"));continue;}
    std::optional<std::size_t> direct;
    for(std::size_t j=0;j<shape.size();++j)
      if(family.shape[i]==shape[j] && family.strides_bytes[i]==strides[j])direct=j;
    if(direct){coordinates.push_back(graph.index(*direct));continue;}
    auto index=coordinate;
    if(family.strides_bytes[i]!=1)index=graph.operation("Divide","u64",{index,graph.constant("u64",std::to_string(family.strides_bytes[i]))});
    if(family.shape.size()>1)index=graph.operation("Remainder","u64",{index,graph.constant("u64",std::to_string(family.shape[i]))});
    coordinates.push_back(index);
  }
  const auto shifted=[&](std::uint32_t level) {
    const auto result=static_cast<std::int64_t>(level)+static_cast<std::int64_t>(shape.size())-static_cast<std::int64_t>(family.shape.size());
    if(result<0 || result>UINT32_MAX)throw std::invalid_argument("output_family_binder_overflow");
    return static_cast<std::uint32_t>(result);
  };
  std::vector<bool> live(source.size());std::vector<s::Id> pending{family.body.node};
  while(!pending.empty()) {
    auto id=pending.back();pending.pop_back();
    if(live.at(id))continue;live[id]=true;
    for(const auto& operand:source.node(id).operands)pending.push_back(operand.node);
  }
  std::vector<s::Ref> values(source.size());
  const auto use=[&](const s::Ref& old) {
    auto ref=values.at(old.node);ref.conversion=old.conversion;
    ref.converted_type=old.converted_type;ref.pointer_provenance=old.pointer_provenance;
    return ref;
  };
  for(s::Id id=0;id<source.size();++id)if(live[id]) {
    const auto& n=source.node(id);std::vector<s::Ref> args;
    for(const auto& operand:n.operands)args.push_back(use(operand));
    switch(n.kind) {
      case s::Kind::Constant:values[id]=graph.constant(n.type,n.name);break;
      case s::Kind::Parameter:values[id]=graph.parameter(n.type,n.name);break;
      case s::Kind::InitialCell:values[id]=graph.initialCell(n.name,args.at(0),n.type);break;
      case s::Kind::Index:values[id]=n.index_level<family.shape.size()?coordinates[n.index_level]:graph.index(shifted(n.index_level));break;
      case s::Kind::Operation:values[id]=graph.operation(n.name,n.type,std::move(args),n.attributes);break;
      case s::Kind::Fold:
        values[id]=graph.fold(n.fold->reducer,args.at(0),args.at(1),shifted(n.fold->binder_level),n.fold->count,n.type);break;
    }
  }
  result.family.body=use(family.body);
  graph.validateRoots(std::span<const s::Ref>(&result.family.body,1),shape.size());
  return result;
}
}
FamilyResult compareOutputFamilySets(const scalar_fold::Graph& left,const scalar_fold::Graph& right,
                                    std::span<const FamilyPair> pairs,Options options) {
  FamilyResult result;
  const auto started=std::chrono::steady_clock::now();
  try {
    if(!options.max_nodes || options.timeout.count()<=0)
      throw std::invalid_argument("invalid_budget");
    if(pairs.empty())throw std::invalid_argument("empty_output_family_cohort");
    std::set<std::string_view> left_seen,right_seen;
    std::vector<scalar_fold::Ref> a,b;a.reserve(pairs.size());b.reserve(pairs.size());
    std::uint64_t outputs=0,covered_bytes=0;
    const scalar_fold::OutputFamily* domain=nullptr;
    for(const auto& pair:pairs) {
      if(std::chrono::steady_clock::now()-started>=options.timeout) {
        result.proof.status=Status::NotProven;result.proof.reason="time_budget";return result;
      }
      if(!pair.left || !pair.right || !pair.mapping)
        throw std::invalid_argument("invalid_output_family_pair");
      const auto bytes=validatePair(*pair.left,*pair.right,*pair.mapping);
      if(!left_seen.insert(pair.left->allocation).second || !right_seen.insert(pair.right->allocation).second)
        throw std::invalid_argument("duplicate_output_family_allocation");
      if(domain && (pair.left->shape!=domain->shape || pair.left->element_bytes!=domain->element_bytes))
        throw std::invalid_argument("output_family_cohort_domain_mismatch");
      domain=pair.left;
      const auto count=bytes/pair.left->element_bytes;
      if(count>UINT64_MAX-outputs || bytes>UINT64_MAX-covered_bytes)
        throw std::invalid_argument("output_family_coverage_overflow");
      outputs+=count;covered_bytes+=bytes;
      a.push_back(pair.left->body);b.push_back(pair.right->body);
    }
    options.outer_binders=static_cast<std::uint32_t>(domain->shape.size());
    options.outer_index_extents=domain->shape;
    options.storage_output_bytes=domain->element_bytes;
    const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-started);
    if(elapsed>=options.timeout) {
      result.proof.status=Status::NotProven;result.proof.reason="time_budget";return result;
    }
    options.timeout-=elapsed;
    result.proof=compareExpressionSets(left,a,right,b,options);
    if(result.proof.status==Status::Equivalent) {
      result.outputs=outputs;result.covered_bytes=covered_bytes;
    }
  } catch(const std::invalid_argument& error) {
    result.proof.status=Status::Unsupported;result.proof.reason=error.what();
  }
  return result;
}
FamilyResult compareOutputFamilies(const scalar_fold::Graph& left,const scalar_fold::OutputFamily& a,
                                  const scalar_fold::Graph& right,const scalar_fold::OutputFamily& b,
                                  const TensorOutputMapping& mapping,Options options) {
  FamilyResult result;
  try {
    const bool reshape_left=a.shape!=mapping.shape || a.strides_bytes!=mapping.left_strides_bytes;
    const bool reshape_right=b.shape!=mapping.shape || b.strides_bytes!=mapping.right_strides_bytes;
    if(reshape_left || reshape_right) {
      if(reshape_left) {
        auto normalized=reshape(left,a,mapping.shape,mapping.left_strides_bytes);
        return compareOutputFamilies(normalized.graph,normalized.family,right,b,mapping,options);
      }
      auto normalized=reshape(right,b,mapping.shape,mapping.right_strides_bytes);
      return compareOutputFamilies(left,a,normalized.graph,normalized.family,mapping,options);
    }
    const FamilyPair pair{&a,&b,&mapping};
    return compareOutputFamilySets(left,right,std::span(&pair,1),options);
  }catch(const std::invalid_argument& error) {
    result.proof.status=Status::Unsupported;result.proof.reason=error.what();
  }
  return result;
}
}
