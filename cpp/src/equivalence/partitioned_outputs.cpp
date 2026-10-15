#include "sentryward/equivalence/partitioned_outputs.h"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace sentryward::equivalence {
namespace {
namespace s=scalar_fold;
struct Limit:std::runtime_error { using std::runtime_error::runtime_error; };
struct Piece { std::uint64_t begin,end; s::Ref body; bool scalar; };
std::uint64_t extent(const TensorOutputMapping& m) {
  if(m.shape.empty() || !m.element_bytes || m.shape.size()!=m.left_strides_bytes.size() ||
      m.left_strides_bytes!=m.right_strides_bytes)
    throw std::invalid_argument("partitioned_output_layout_not_aligned");
  std::vector<std::pair<std::uint64_t,std::uint64_t>> axes;
  for(std::size_t i=0;i<m.shape.size();++i) {
    if(!m.shape[i])throw std::invalid_argument("empty_output_axis");
    if(m.shape[i]>1)axes.emplace_back(m.left_strides_bytes[i],m.shape[i]);
  }
  std::sort(axes.begin(),axes.end());auto bytes=m.element_bytes;
  for(auto [stride,count]:axes) {
    if(stride!=bytes || count>UINT64_MAX/bytes)
      throw std::invalid_argument("invalid_partitioned_output_layout");
    bytes*=count;
  }
  return bytes;
}
std::vector<Piece> pieces(const s::Artifact& a,const std::string& allocation,
                         std::uint64_t start,std::uint64_t bytes,std::uint64_t width) {
  if(bytes>UINT64_MAX-start)throw std::invalid_argument("output_extent_overflow");
  std::vector<Piece> result;
  const auto append=[&](std::uint64_t offset,std::uint64_t length,s::Ref body,bool scalar) {
    if(offset<start || length>bytes || offset-start>bytes-length ||
        (offset-start)%width || length%width || !length)
      throw std::invalid_argument("output_piece_outside_mapping");
    a.graph.validateRoots(std::span<const s::Ref>(&body,1),scalar?0:1);
    result.push_back({offset-start,offset-start+length,body,scalar});
  };
  for(const auto& output:a.outputs)if(output.allocation==allocation) {
    if(output.bytes!=width)throw std::invalid_argument("output_scalar_width_mismatch");
    append(output.offset,output.bytes,output.value,true);
  }
  for(const auto& family:a.output_families)if(family.allocation==allocation) {
    if(family.element_bytes!=width || family.shape.size()!=1 || family.strides_bytes!=std::vector<std::uint64_t>{width} ||
        !family.shape[0] || family.shape[0]>UINT64_MAX/width)
      throw std::invalid_argument("partitioned_output_requires_flat_contiguous_family");
    append(family.offset,family.shape[0]*width,family.body,false);
  }
  std::sort(result.begin(),result.end(),[](const auto& x,const auto& y){return x.begin<y.begin;});
  std::uint64_t cursor=0;
  for(const auto& piece:result) {
    if(piece.begin!=cursor)throw std::invalid_argument("output_coverage_gap_or_overlap");
    cursor=piece.end;
  }
  if(cursor!=bytes)throw std::invalid_argument("incomplete_output_coverage");
  return result;
}
class Rebinder {
  const s::Graph& source;
  std::map<std::tuple<std::uint64_t,bool,bool>,std::unordered_map<s::Id,s::Ref>> contexts;
  const Options& options;
  std::chrono::steady_clock::time_point deadline;
 public:
  s::Graph graph;
  Rebinder(const s::Graph& source,const Options& options,std::chrono::steady_clock::time_point deadline)
      :source(source),options(options),deadline(deadline){}
  s::Ref bind(s::Ref root,std::uint64_t skip,bool scalar,bool singleton) {
    auto& mapped=contexts[{skip,scalar,singleton}];
    const auto use=[&](const s::Ref& old) {
      auto value=mapped.at(old.node);value.conversion=old.conversion;
      value.converted_type=old.converted_type;value.pointer_provenance=old.pointer_provenance;return value;
    };
    std::vector<std::pair<s::Id,bool>> pending{{root.node,false}};
    while(!pending.empty()) {
      if(graph.size()>options.max_nodes)throw Limit("node_budget");
      if(std::chrono::steady_clock::now()>=deadline)throw Limit("time_budget");
      auto [id,ready]=pending.back();pending.pop_back();
      if(mapped.contains(id))continue;
      const auto& n=source.node(id);
      if(!ready) {
        pending.emplace_back(id,true);
        for(const auto& arg:n.operands)if(!mapped.contains(arg.node))pending.emplace_back(arg.node,false);
        continue;
      }
      std::vector<s::Ref> args;for(const auto& arg:n.operands)args.push_back(use(arg));
      s::Ref value;
      const auto level=[&](std::uint32_t old) {
        if(scalar && old==UINT32_MAX)throw std::invalid_argument("output_binder_overflow");
        return old+std::uint32_t(scalar);
      };
      switch(n.kind) {
        case s::Kind::Constant:value=graph.constant(n.type,n.name);break;
        case s::Kind::Parameter:value=graph.parameter(n.type,n.name);break;
        case s::Kind::InitialCell:value=graph.initialCell(n.name,args.at(0),n.type);break;
        case s::Kind::Index:
          value=graph.index(level(n.index_level));
          if(!scalar && n.index_level==0) {
            if(singleton)value=graph.constant("u64",std::to_string(skip));
            else if(skip)value=graph.operation("Add","u64",{value,graph.constant("u64",std::to_string(skip))});
          }
          break;
        case s::Kind::Operation:value=graph.operation(n.name,n.type,std::move(args),n.attributes);break;
        case s::Kind::Fold:
          value=graph.fold(n.fold->reducer,args.at(0),args.at(1),level(n.fold->binder_level),n.fold->count,n.type);break;
      }
      mapped.emplace(id,value);
    }
    return use(root);
  }
};
}
FamilyResult comparePartitionedOutputs(const s::Artifact& left,const s::Artifact& right,
                                      std::span<const TensorOutputMapping> mappings,Options options) {
  FamilyResult result;
  try {
    const auto deadline=std::chrono::steady_clock::now()+options.timeout;
    Rebinder a(left.graph,options,deadline),b(right.graph,options,deadline);
    std::vector<s::Ref> roots_a,roots_b;
    std::uint64_t shared_extent=1;
    std::set<std::string> seen_a,seen_b;
    for(const auto& mapping:mappings) {
      if(!seen_a.insert(mapping.left_allocation).second || !seen_b.insert(mapping.right_allocation).second)
        throw std::invalid_argument("duplicate_output_allocation_mapping");
      const auto bytes=extent(mapping);
      auto x=pieces(left,mapping.left_allocation,mapping.left_offset,bytes,mapping.element_bytes);
      auto y=pieces(right,mapping.right_allocation,mapping.right_offset,bytes,mapping.element_bytes);
      std::size_t i=0,j=0;std::uint64_t position=0;
      while(position<bytes) {
        const auto end=std::min(x[i].end,y[j].end);
        // One engine shares the lexical index across roots. Its safe domain
        // is the UNION (maximum extent), not the intersection of piece sizes.
        shared_extent=std::max(shared_extent,(end-position)/mapping.element_bytes);
        const bool singleton=end-position==mapping.element_bytes;
        roots_a.push_back(a.bind(x[i].body,(position-x[i].begin)/mapping.element_bytes,x[i].scalar,singleton));
        roots_b.push_back(b.bind(y[j].body,(position-y[j].begin)/mapping.element_bytes,y[j].scalar,singleton));
        position=end;if(position==x[i].end)++i;if(position==y[j].end)++j;
      }
      if(bytes>UINT64_MAX-result.covered_bytes || bytes/mapping.element_bytes>UINT64_MAX-result.outputs)
        throw std::invalid_argument("output_coverage_overflow");
      result.covered_bytes+=bytes;result.outputs+=bytes/mapping.element_bytes;
    }
    const auto covered=[](const s::Artifact& artifact,const std::set<std::string>& seen) {
      for(const auto& output:artifact.outputs)if(!seen.contains(output.allocation))return false;
      for(const auto& family:artifact.output_families)if(!seen.contains(family.allocation))return false;
      return true;
    };
    if(!covered(left,seen_a) || !covered(right,seen_b))throw std::invalid_argument("unmapped_output_allocation");
    if(std::chrono::steady_clock::now()>=deadline)throw Limit("time_budget");
    options.timeout=std::chrono::duration_cast<std::chrono::milliseconds>(deadline-std::chrono::steady_clock::now());
    options.outer_binders=1;
    options.outer_index_extents={shared_extent};
    result.proof=compareExpressionSets(a.graph,roots_a,b.graph,roots_b,options);
  } catch(const Limit& error) {result.proof.status=Status::NotProven;result.proof.reason=error.what();}
    catch(const std::invalid_argument& error) {result.proof.status=Status::Unsupported;result.proof.reason=error.what();}
  return result;
}
}
