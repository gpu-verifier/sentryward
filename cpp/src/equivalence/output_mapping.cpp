#include "sentryward/equivalence/output_mapping.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>

namespace sentryward::equivalence {
namespace {
struct Range { std::uint64_t start,end; std::size_t mapping; };
using Ranges=std::map<std::string,std::vector<Range>>;
using Key=std::tuple<std::size_t,std::uint64_t,std::uint64_t>;
struct Mapped { Key key; scalar_fold::Ref value; };
std::uint64_t end(std::uint64_t start,std::uint64_t bytes) {
  if(!bytes || bytes>UINT64_MAX-start) throw std::invalid_argument("invalid_output_interval");
  return start+bytes;
}
Ranges ranges(std::span<const OutputRangeMapping> maps,bool left) {
  Ranges result;
  for(std::size_t i=0;i<maps.size();++i) {
    const auto& m=maps[i];
    const auto& allocation=left?m.left_allocation:m.right_allocation;
    auto start=left?m.left_offset:m.right_offset;
    if(allocation.empty()) throw std::invalid_argument("empty_output_allocation");
    result[allocation].push_back({start,end(start,m.bytes),i});
  }
  for(auto& [allocation,rs]:result) {
    (void)allocation;
    std::sort(rs.begin(),rs.end(),[](const auto& a,const auto& b){return a.start<b.start;});
    for(std::size_t i=1;i<rs.size();++i)
      if(rs[i].start<rs[i-1].end) throw std::invalid_argument("overlapping_output_mapping");
  }
  return result;
}
std::vector<Mapped> mapOutputs(std::span<const scalar_fold::Output> outputs,const Ranges& ranges) {
  std::vector<Mapped> result;
  result.reserve(outputs.size());
  for(const auto& o:outputs) {
    auto stop=end(o.offset,o.bytes);
    auto object=ranges.find(o.allocation);
    if(object==ranges.end()) throw std::invalid_argument("unmapped_output_allocation");
    const auto& rs=object->second;
    auto it=std::upper_bound(rs.begin(),rs.end(),o.offset,
      [](auto offset,const auto& range){return offset<range.start;});
    if(it==rs.begin()) throw std::invalid_argument("unmapped_output_region");
    --it;
    if(stop>it->end) throw std::invalid_argument("unmapped_output_region");
    result.push_back({{it->mapping,o.offset-it->start,o.bytes},o.value});
  }
  std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.key<b.key;});
  for(std::size_t i=1;i<result.size();++i) {
    auto [p,offset,bytes]=result[i-1].key;
    auto [q,next,unused]=result[i].key;
    (void)unused;
    if(p==q && next<offset+bytes) throw std::invalid_argument("overlapping_recorded_outputs");
  }
  return result;
}
} // namespace
OutputPairing pairOutputs(std::span<const scalar_fold::Output> left,
                          std::span<const scalar_fold::Output> right,
                          std::span<const OutputRangeMapping> mapping) {
  OutputPairing result;
  try {
    if(left.empty() || right.empty()) throw std::invalid_argument("empty_output_set");
    auto a=mapOutputs(left,ranges(mapping,true));
    auto b=mapOutputs(right,ranges(mapping,false));
    if(a.size()!=b.size()) throw std::invalid_argument("output_partition_mismatch");
    for(std::size_t i=0;i<a.size();++i) {
      if(a[i].key!=b[i].key) throw std::invalid_argument("output_partition_mismatch");
      auto bytes=std::get<2>(a[i].key);
      if(bytes>UINT64_MAX-result.covered_bytes) throw std::invalid_argument("output_size_overflow");
      result.covered_bytes+=bytes;
      result.left.push_back(a[i].value);
      result.right.push_back(b[i].value);
    }
    std::uint64_t required_bytes=0;
    for(const auto& region:mapping) {
      if(region.bytes>UINT64_MAX-required_bytes)throw std::invalid_argument("output_size_overflow");
      required_bytes+=region.bytes;
    }
    if(result.covered_bytes!=required_bytes)
      throw std::invalid_argument("incomplete_declared_output_domain");
    result.complete=true;
  } catch(const std::invalid_argument& e) {
    result.reason=e.what();result.left.clear();result.right.clear();result.covered_bytes=0;
  }
  return result;
}
OutputPairing pairTensorOutputs(std::span<const scalar_fold::Output> left,
                                std::span<const scalar_fold::Output> right,
                                const TensorOutputMapping& mapping) {
  try {
    if(mapping.shape.empty() || !mapping.element_bytes || mapping.left_allocation.empty() ||
       mapping.right_allocation.empty())throw std::invalid_argument("invalid_tensor_output_mapping");
    const auto validate=[&](const std::vector<std::uint64_t>& strides) {
      if(strides.size()!=mapping.shape.size())throw std::invalid_argument("output_layout_rank_mismatch");
      std::vector<std::pair<std::uint64_t,std::uint64_t>> axes;
      for(std::size_t i=0;i<strides.size();++i) {
        if(!mapping.shape[i])throw std::invalid_argument("empty_output_axis");
        if(mapping.shape[i]>1)axes.emplace_back(strides[i],mapping.shape[i]);
      }
      std::sort(axes.begin(),axes.end());
      auto extent=mapping.element_bytes;
      for(auto [stride,count]:axes) {
        if(stride!=extent)throw std::invalid_argument("output_layout_not_dense_permutation");
        if(count>UINT64_MAX/extent)throw std::invalid_argument("output_size_overflow");
        extent*=count;
      }
      return extent;
    };
    auto extent=validate(mapping.left_strides_bytes);
    if(validate(mapping.right_strides_bytes)!=extent)throw std::invalid_argument("output_layout_size_mismatch");
    end(mapping.left_offset,extent);end(mapping.right_offset,extent);
    const auto translate=[&](std::span<const scalar_fold::Output> outputs,bool is_left) {
      std::vector<scalar_fold::Output> translated;translated.reserve(outputs.size());
      const auto& object=is_left?mapping.left_allocation:mapping.right_allocation;
      const auto& strides=is_left?mapping.left_strides_bytes:mapping.right_strides_bytes;
      auto base=is_left?mapping.left_offset:mapping.right_offset;
      for(const auto& output:outputs) {
        if(output.allocation!=object)throw std::invalid_argument("unmapped_output_allocation");
        if(output.bytes!=mapping.element_bytes || output.offset<base ||
           output.offset-base>=extent || (output.offset-base)%mapping.element_bytes)
          throw std::invalid_argument("output_scalar_partition_mismatch");
        auto offset=output.offset-base;
        std::uint64_t linear=0;
        for(std::size_t i=0;i<mapping.shape.size();++i) {
          auto coordinate=mapping.shape[i]==1?0:(offset/strides[i])%mapping.shape[i];
          linear=linear*mapping.shape[i]+coordinate;
        }
        translated.push_back({"mapped",linear*mapping.element_bytes,output.bytes,output.value});
      }
      return translated;
    };
    auto a=translate(left,true),b=translate(right,false);
    OutputRangeMapping identity{"mapped","mapped",0,0,extent};
    return pairOutputs(a,b,std::span(&identity,1));
  } catch(const std::invalid_argument& e) {
    OutputPairing result;result.reason=e.what();return result;
  }
}

OutputPairing pairTensorOutputs(std::span<const scalar_fold::Output> left,
                                std::span<const scalar_fold::Output> right,
                                std::span<const TensorOutputMapping> mapping) {
  OutputPairing result;
  try {
    if(mapping.empty())throw std::invalid_argument("empty_output_mapping");
    const auto group=[&](std::span<const scalar_fold::Output> outputs,bool is_left) {
      std::map<std::string,std::size_t> ids;
      for(std::size_t i=0;i<mapping.size();++i) {
        const auto& name=is_left?mapping[i].left_allocation:mapping[i].right_allocation;
        if(name.empty())throw std::invalid_argument("empty_output_allocation");
        if(!ids.emplace(name,i).second)throw std::invalid_argument("duplicate_output_allocation_mapping");
      }
      std::vector<std::vector<scalar_fold::Output>> grouped(mapping.size());
      for(const auto& output:outputs) {
        const auto found=ids.find(output.allocation);
        if(found==ids.end())throw std::invalid_argument("unmapped_output_allocation");
        grouped[found->second].push_back(output);
      }
      return grouped;
    };
    auto a=group(left,true),b=group(right,false);
    for(std::size_t i=0;i<mapping.size();++i) {
      auto part=pairTensorOutputs(a[i],b[i],mapping[i]);
      if(!part.complete)throw std::invalid_argument(part.reason);
      if(part.covered_bytes>UINT64_MAX-result.covered_bytes)
        throw std::invalid_argument("output_size_overflow");
      result.covered_bytes+=part.covered_bytes;
      result.left.insert(result.left.end(),part.left.begin(),part.left.end());
      result.right.insert(result.right.end(),part.right.begin(),part.right.end());
    }
    result.complete=true;
  } catch(const std::invalid_argument& error) {
    result.reason=error.what();result.left.clear();result.right.clear();result.covered_bytes=0;
  }
  return result;
}
} // namespace sentryward::equivalence
