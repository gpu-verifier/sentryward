#include "sentryward/equivalence/epoch_work_allowance.h"

#include <algorithm>

namespace sentryward::equivalence::epoch_work_allowance {
namespace {
std::size_t add(std::size_t a,std::size_t b,bool& saturated) noexcept {
  if(a>=maximum_work || b>maximum_work-a) {saturated=true;return maximum_work;}
  return a+b;
}
std::size_t multiply(std::size_t a,std::size_t b,bool& saturated) noexcept {
  if(a>maximum_work/b) {saturated=true;return maximum_work;}
  return a*b;
}
}
std::size_t retainedHeaderCapacity(const Size& size) noexcept {
  return std::clamp(size.headers,default_retained_headers,maximum_retained_headers);
}
void PhysicalEpoch::addHeader(std::size_t stored_edges,
    std::span<const std::size_t> metadata_lengths) noexcept {
  size_.headers=add(size_.headers,1,size_.saturated);
  size_.edges=add(size_.edges,stored_edges,size_.saturated);
  std::size_t bytes=0;
  for(std::size_t i=0;i<std::min<std::size_t>(metadata_lengths.size(),8);++i) {
    const auto part=std::min(metadata_lengths[i],metadata_bytes_per_header-bytes);
    bytes+=part;
  }
  size_.bounded_metadata_bytes=add(size_.bounded_metadata_bytes,bytes,size_.saturated);
}
Allowance PhysicalEpoch::allowance(std::size_t floor) const noexcept {
  Allowance out;out.physical=size_;
  bool saturated=size_.saturated;
  const auto headers=multiply(size_.headers,header_work,saturated);
  const auto edges=multiply(size_.edges,edge_work,saturated);
  const auto metadata=multiply(size_.bounded_metadata_bytes,metadata_byte_work,saturated);
  floor=std::clamp(floor,default_work,maximum_work);
  out.max_work=std::max(floor,add(add(headers,edges,saturated),metadata,saturated));
  out.physical.saturated=saturated;return out;
}
} // namespace sentryward::equivalence::epoch_work_allowance
