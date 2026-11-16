#include "sentryward/frontend/ptx/shared_execution.h"
#include <algorithm>

namespace sentryward::frontend::ptx {
void SharedConstants::Write(const SharedValueSite &source, std::uint64_t address,
                            const std::vector<SharedByte> &values) {
  if (values.size()>UINT64_MAX-address) throw std::overflow_error("shared write extent");
  for (std::size_t k=0;k<values.size();++k) {
    if (values[k]) bytes_[address+k]={values[k],source};
    else bytes_.erase(address+k);
  }
}
void SharedConstants::Invalidate(std::uint64_t address,std::uint64_t width) {
  if (!width) return;
  auto begin=bytes_.lower_bound(address);
  auto end=width>UINT64_MAX-address?bytes_.end():bytes_.lower_bound(address+width);
  bytes_.erase(begin,end);
}
namespace {
// Whether every value of `v` is unchanged by keeping its low-order `bytes` bytes and reading
// them back as a `bytes`-byte integer, signed when `sign` (FitsType in execution.cpp).
bool FitsLoad(const BoundedInteger &v,unsigned bytes,bool sign) {
  const Integer low=sign?-(Integer(1)<<(8*bytes-1)):Integer(0);
  const Integer high=(Integer(1)<<(8*bytes-unsigned(sign)))-1;
  return low<=v->lower && v->upper<=high;
}
// The same bounded value: one identity (the same value in every execution), range, exact
// value and named sources.
bool SameBounded(const BoundedInteger &a,const BoundedInteger &b) {
  return a.value==b.value || (a==b && a->sources==b->sources);
}
} // namespace
SharedConstantRead SharedConstants::Read(std::uint64_t address,unsigned width,bool sign) const {
  if (!width || width>UINT64_MAX-address) return {};
  Integer value=0;
  std::optional<PointerBits> pointer;
  std::optional<DescriptorFragment> descriptor;
  const BoundedFragment *bounded=nullptr;
  SharedConstantRead result;
  for (unsigned k=0;k<width;++k) {
    const auto it=bytes_.find(address+k);
    if (it==bytes_.end()) return {};
    const auto &byte=*it->second.value;
    if (const auto *fragment=std::get_if<DescriptorFragment>(&byte)) {
      if (k==0) descriptor=*fragment;
      if (!descriptor || pointer || bounded || fragment->descriptor!=descriptor->descriptor ||
          fragment->offset!=descriptor->offset+k) return {};
    } else if (const auto *bits=std::get_if<PointerBits>(&byte)) {
      if (k==0) pointer=*bits;
      if (!pointer || descriptor || bounded || *bits->pointer!=*pointer->pointer ||
          bits->source_bit!=pointer->source_bit+8*k) return {};
    } else if (const auto *part=std::get_if<BoundedFragment>(&byte)) {
      // Only the whole value as one store wrote it: byte k at address + k, same width.
      if (k==0) bounded=part;
      if (!bounded || pointer || descriptor || !SameBounded(part->value,bounded->value) ||
          part->offset!=k || part->bytes!=width) return {};
    } else {
      if (pointer || descriptor || bounded) return {};
      value |= Integer(std::get<std::uint8_t>(byte)) << (8*k);
    }
    if (std::find(result.sources.begin(),result.sources.end(),it->second.source)==result.sources.end())
      result.sources.push_back(it->second.source);
  }
  if (pointer) { pointer->width=width*8; result.value=RestorePointer(*pointer); }
  else if (descriptor) { descriptor->bytes=width; result.value=*descriptor; }
  else if (bounded) {
    if (!bounded->value.value || !FitsLoad(bounded->value,width,sign)) return {};
    result.value=bounded->value;
  }
  else result.value=std::move(value);
  if (std::holds_alternative<std::monostate>(result.value)) result.sources.clear();
  return result;
}
void SharedExecution::Write(const SharedWrite &write) {
  Invalidate(write.address,write.bytes.size());
  constants.Write(write.source,write.address,write.bytes);
  if (std::any_of(write.bytes.begin(),write.bytes.end(),[](const auto &byte){return byte.has_value();}))
    writes_[write.source.cta].insert(write.source);
  if (std::any_of(write.bytes.begin(),write.bytes.end(),
                  [](const auto &byte){return byte && std::holds_alternative<BoundedFragment>(*byte);}))
    bounded_writes_[write.source.cta].insert(write.source);
}
void SharedExecution::Invalidate(std::uint64_t address,std::uint64_t width) {
  constants.Invalidate(address,width);
  if(!width)return;
  auto first=tensor_fields.lower_bound(address>=127?address-127:0);
  while(first!=tensor_fields.end() && (first->first<=address || first->first-address<width)) {
    tensor_templates.erase(first->first);
    first=tensor_fields.erase(first);
  }
}
void SharedExecution::Clear() {
  constants.Clear();tensor_fields.clear();tensor_templates.clear();
}
Scalar SharedExecution::Read(const SharedValueSite &site,std::uint64_t address,unsigned width,bool sign) {
  auto result=constants.Read(address,width,sign);
  for (const auto &source:result.sources) dependencies_[site.cta].insert({source,site});
  if (std::holds_alternative<BoundedInteger>(result.value))
    for (const auto &source:result.sources) bounded_dependencies_[site.cta].insert({source,site});
  return std::move(result.value);
}
void SharedExecution::Export(ExecutionProgram &program) const {
  const auto cta=program.config.fixed_input.value("cta_coordinate",Coordinates{0,0,0});
  if (const auto found=writes_.find(cta);found!=writes_.end())
    program.shared_constant_writes.assign(found->second.begin(),found->second.end());
  if (const auto found=dependencies_.find(cta);found!=dependencies_.end())
    program.shared_value_dependencies.assign(found->second.begin(),found->second.end());
  if (const auto found=bounded_writes_.find(cta);found!=bounded_writes_.end())
    program.shared_bounded_writes=found->second.size();
  if (const auto found=bounded_dependencies_.find(cta);found!=bounded_dependencies_.end())
    program.shared_bounded_reads=found->second.size();
}
void SharedPhase::Update(std::int64_t arrivals,std::int64_t amount) {
  pending-=arrivals; tx+=amount;
  if (pending==0 && tx==0) { ++generation; pending=expected; }
}
void SharedExecution::Init(std::uint64_t address,std::int64_t count) {
  if (count<=0) throw ExecutionError("shared_control_invalid_sync",0,"barrier count must be positive");
  SharedPhase phase; phase.expected=phase.pending=count; barriers[address]=phase; ++revision;
}
std::int64_t SharedExecution::Update(std::uint64_t address,std::int64_t arrivals,
                                    std::int64_t tx,SharedActor actor) {
  (void)actor;
  const auto it=barriers.find(address); if (it==barriers.end()) throw SharedBlocked{};
  const auto generation=it->second.generation;
  it->second.Update(arrivals,tx); ++revision; return generation;
}
void SharedExecution::Wait(std::uint64_t address,std::int64_t target,SharedActor actor,bool parity) {
  (void)actor;
  const auto it=barriers.find(address); if (it==barriers.end()) throw SharedBlocked{};
  const auto &barrier=it->second;
  if (parity) {
    if (target!=0 && target!=1) throw ExecutionError("shared_control_invalid_sync",0,"parity must be zero or one");
    if (barrier.generation%2==target) throw SharedBlocked{};
  } else {
    if (target<barrier.generation-1 || target>barrier.generation)
      throw ExecutionError("shared_control_invalid_sync",0,"state wait outside current/previous generation");
    if (target==barrier.generation) throw SharedBlocked{};
  }
}
void SharedExecution::Collective(std::string key,std::int64_t count,SharedActor actor,
                                 DynamicKey occurrence,bool blocking) {
  auto [it,inserted]=collectives.try_emplace(key);
  auto &barrier=it->second;
  if (inserted) barrier.expected=barrier.pending=count;
  const auto ticket_key=std::tuple{key,actor,occurrence};
  const bool first=!tickets.contains(ticket_key);
  if (first && barrier.expected!=count && barrier.pending==barrier.expected && barrier.tx==0)
    barrier.expected=barrier.pending=count;
  if ((first && barrier.expected!=count) || count<=0)
    throw ExecutionError("shared_control_invalid_sync",0,"collective participant count changes");
  const auto [ticket,fresh]=tickets.try_emplace(ticket_key,barrier.generation);
  if (fresh) { barrier.Update(1,0); ++revision; }
  if (blocking && ticket->second>=barrier.generation) throw SharedBlocked{};
}
void SharedExecution::Defer(std::uint64_t address,std::int64_t arrivals,std::int64_t tx,
                           std::vector<SharedWrite> writes) {
  completions_.push_back({address,arrivals,tx,std::move(writes)}); ++revision;
}
void SharedExecution::Progress() {
  std::vector<Completion> remaining;
  for (auto &c:completions_)
    if (const auto it=barriers.find(c.address);it!=barriers.end()) {
      for (const auto &write:c.writes) Write(write);
      it->second.Update(c.arrivals,c.tx); ++revision;
    } else remaining.push_back(std::move(c));
  completions_=std::move(remaining);
}
std::vector<SharedByte> SharedBytes(const Scalar &value,unsigned width) {
  std::vector<SharedByte> result(width);
  // A table atom stores its exact value (ScalarInteger), as before.
  if (auto number=ScalarInteger(value))
    for (unsigned k=0;k<width;++k) result[k]=((*number>>(8*k))&255).convert_to<std::uint8_t>();
  else if (auto fragment=std::get_if<DescriptorFragment>(&value)) {
    if (width<=fragment->bytes)
      for (unsigned k=0;k<width;++k) result[k]=DescriptorFragment{fragment->descriptor,fragment->offset+k,1};
  } else if (const auto *bounded=std::get_if<BoundedInteger>(&value)) {
    if (bounded->value)
      for (unsigned k=0;k<width;++k) result[k]=BoundedFragment{*bounded,k,width};
  } else for (unsigned k=0;k<width;++k)
    if (auto bits=PointerFragment(value,8*k,8)) result[k]=*bits;
  return result;
}
} // namespace sentryward::frontend::ptx
