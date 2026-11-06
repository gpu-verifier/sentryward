#include "sentryward/frontend/ptx/serialization.h"
#include "sentryward/frontend/ptx/selector_patterns.h"
#include "sentryward/algorithm/base/affine_index.h"
#include "sentryward/algorithm/semantics/affine_template_schedule.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/memory/symbolic_address_model.h"
#include "sentryward/value_ir/operations.h"
#include <nlohmann/json.hpp>
#include <deque>
#include <map>
#include <set>
#include <unordered_map>

namespace sentryward::frontend::ptx {
namespace {
namespace a=algorithm; namespace v=value_ir; namespace va=value_analysis;
using Domain=a::AffineDomain;
using Match=a::AffineIntersection;
struct SerializationError:std::runtime_error {
  std::string code;
  SerializationError(std::string c,std::string text):std::runtime_error(std::move(text)),code(std::move(c)){}
};
void Require(bool condition,const char* code,const char* message) {
  if(!condition)throw SerializationError(code,message);
}
template<class T,class Codec> T Advance(T value,const typename Codec::Delta& delta,std::size_t i) {
  typename Codec::Binding binding;Codec::split(value,binding);
  auto next=Codec::translate(binding,delta,i);
  Require(bool(next),"invalid_template_binding","Affine binding exceeds its domain");
  Codec::bind(value,*next);return value;
}
template<class Delta> Delta Scale(Delta delta,std::uint64_t scale) {
  for(auto& x:delta) {
    const auto product=__int128(x)*scale;
    Require(product>=INT64_MIN && product<=INT64_MAX,"invalid_template_binding","Affine binding stride overflows");
    x=static_cast<std::int64_t>(product);
  }return delta;
}
template<class T,class Codec> struct Column {
  T first;typename Codec::Delta delta;std::size_t count;
  T at(std::size_t i) const {return Advance<T,Codec>(first,delta,i);}
};
using AccessColumn=Column<a::MemoryAccess,a::MemoryAccessCodec>;
using BindingColumn=Column<SemanticMemoryBinding,SemanticMemoryBindingCodec>;
using AccessKey=std::tuple<a::AccessOperand,std::uint32_t,std::uint32_t>;
AccessKey Key(const SourceMemoryKey& key){return {key.operand,key.effect_index,key.access_index};}
AccessKey Key(const a::MemoryAccess& key){return {key.operand,key.effect_index,key.access_index};}
std::string Allocation(const a::ConstantAddressBinding& x) {
  return va::AllocationIdentity(x.address_space,x.allocation);
}
struct ActionColumn {
  va::OrderedMemoryAction first;va::MemoryActionCodec::Delta delta;
  Domain events;std::size_t count;
};
struct EndpointColumn {Domain instances,events;};

class ActorIndex {
  struct Run {a::ActorInfo first;std::array<std::int64_t,3> delta{};std::size_t count=1;};
  std::vector<Run> runs_;a::AffineIndex index_;
  static auto fields(const a::ActorInfo& x){return std::array<std::uint32_t,3>{x.cta.value(),x.warp.value(),x.lane};}
 public:
  explicit ActorIndex(const std::vector<a::ActorInfo>& actors) {
    for(const auto& actor:actors) {
      if(!runs_.empty()) {
        auto& run=runs_.back();
        if(actor.id.value()==std::uint64_t(run.first.id.value())+run.count) {
          const auto a=fields(run.first),b=fields(actor);bool matches=true;
          for(unsigned j=0;j<3;++j) {
            const auto difference=std::int64_t(b[j])-a[j];
            if(run.count==1)run.delta[j]=difference;
            matches&=difference==run.delta[j]*std::int64_t(run.count);
          }
          if(matches){++run.count;continue;}
        }
      }
      runs_.push_back({actor});
    }
    for(std::size_t i=0;i<runs_.size();++i)index_.add({runs_[i].first.id.value(),1,runs_[i].count},i);
    index_.seal();
  }
  void validate(Domain actors,const SemanticOccurrence& occurrence,const std::array<std::int64_t,3>& delta) const {
    const std::array<std::uint32_t,3> expected{occurrence.cta,occurrence.warp,occurrence.lane};
    std::size_t covered=0;
    index_.intersect(actors,[&](auto id,auto m) {
      if(!actors.step){m.count=actors.count;m.left_step=1;m.right_step=0;}
      const auto& run=runs_[id];const auto actual=fields(run.first);
      for(unsigned j=0;j<3;++j)
        Require(__int128(actual[j])+__int128(run.delta[j])*m.right==__int128(expected[j])+__int128(delta[j])*m.left &&
            (m.count==1 || __int128(run.delta[j])*m.right_step==__int128(delta[j])*m.left_step),
            "changed_checked_occurrence","Checked thread coordinates differ from their value binding");
      covered+=m.count;
    });
    Require(covered==actors.count,"changed_checked_occurrence","Checked actor identity is missing");
  }
};

// The constant part and terms of a tensor_properties footprint (slot_resolver.cpp
// AffineOffsetJson): {"operator":"add","operands":[constant, {"operator":"mul","operands":
// [{"operator":"gather","symbol":TensorPropertySymbol(object,type),"args":[element byte
// offset]}, coefficient]}...]}, the symbol read back by ParseTensorPropertySymbol. Any other
// expression has no position, except a TMA tile region (ParseTmaRegion). O(terms).
struct AbstractAffinePosition {std::int64_t constant=0;std::vector<va::SymbolicAddressTerm> base;};
std::optional<AbstractAffinePosition> ParseAffineOffset(const nlohmann::json& offset);
std::optional<AbstractAffinePosition> ParseTmaRegion(const nlohmann::json& region);
std::optional<AbstractAffinePosition> ParseAbstractAffine(const a::ConstantAddressBinding& address) {
  if(address.expression.empty() || address.offset || !address.alias_scope.empty())return {};
  const auto json=nlohmann::json::parse(address.expression,nullptr,false);
  if(json.is_object() && json.value("operator",std::string())=="tma_region" &&
     !address.allocation.empty() && address.allocation!="unresolved")return ParseTmaRegion(json);
  if(!json.is_object() || !json.contains("address") || !json["address"].is_object())return {};
  const auto& where=json["address"];
  const auto base=where.find("base");const auto offset=where.find("offset");
  if(base==where.end() || offset==where.end() || !base->is_object() ||
     base->value("identity",std::string())!=address.allocation)return {};
  return ParseAffineOffset(*offset);
}
// The footprint form of an offset (above).
std::optional<AbstractAffinePosition> ParseAffineOffset(const nlohmann::json& offset) {
  if(!offset.is_object() || offset.value("operator",std::string())!="add" || !offset.contains("operands"))return {};
  const auto integer=[](const nlohmann::json& x)->std::optional<std::int64_t> {
    if(x.is_number_integer() && (x.is_number_unsigned()?x.get<std::uint64_t>()<=std::uint64_t(INT64_MAX):true))
      return x.get<std::int64_t>();
    return {};
  };
  const auto& operands=offset["operands"];
  if(!operands.is_array() || operands.size()<2)return {};
  AbstractAffinePosition result;
  const auto constant=integer(operands[0]);if(!constant)return {};
  result.constant=*constant;
  for(std::size_t i=1;i<operands.size();++i) {
    const auto& term=operands[i];
    if(!term.is_object() || term.value("operator",std::string())!="mul" || !term.contains("operands") ||
       !term["operands"].is_array() || term["operands"].size()!=2)return {};
    const auto& gather=term["operands"][0];const auto coefficient=integer(term["operands"][1]);
    if(!coefficient || !*coefficient || !gather.is_object() || gather.value("operator",std::string())!="gather" ||
       !gather.contains("args") || !gather["args"].is_array() || gather["args"].size()!=1)return {};
    const auto element=integer(gather["args"][0]);
    auto symbol=a::ParseTensorPropertySymbol(gather.value("symbol",std::string()));
    if(!element || !symbol)return {};
    result.base.push_back({std::move(symbol->first),std::move(symbol->second),*element,*coefficient});
  }
  return result;
}
// The first byte of a TMA tile region (slot_resolver.cpp TmaBody: {"operator":"tma_region",
// "descriptor":D,"coordinates":[...]}, a tile::gather4 row's region {column, row}) in its
// allocation: D's base_byte_offset + sum of coordinate x coordinate_strides_bytes, each
// coordinate an integer or an affine form as above. Equal atoms merge; the terms are sorted by
// atom. No position without such coordinates, integer strides and base, for a packed map, when
// the constant part is negative (a fragment adds its nonnegative place to it), or when a value
// leaves int64. O(rank + terms log terms).
std::optional<AbstractAffinePosition> ParseTmaRegion(const nlohmann::json& region) {
  // References into the parsed expression, and the coordinates first: most regions have an
  // opaque coordinate, which ends the parse before the descriptor is read. O(rank).
  const auto coordinates=region.find("coordinates");
  if(coordinates==region.end() || !coordinates->is_array())return {};
  for(const auto& coordinate:*coordinates)
    if(!coordinate.is_number_integer() && (!coordinate.is_object() ||
       coordinate.value("operator",std::string())!="add"))return {};
  const auto descriptor=region.find("descriptor");
  if(descriptor==region.end() || !descriptor->is_object())return {};
  const auto strides=descriptor->find("coordinate_strides_bytes");
  const auto base=descriptor->find("base_byte_offset");
  const auto packed=descriptor->find("packed_element_bits");
  if(strides==descriptor->end() || !strides->is_array() || strides->size()!=coordinates->size() ||
     (base!=descriptor->end() && !base->is_number_integer()) ||
     (packed!=descriptor->end() && !packed->is_null()))return {};
  __int128 constant=base==descriptor->end()?0:base->get<std::int64_t>();
  AbstractAffinePosition result;
  for(std::size_t d=0;d<coordinates->size();++d) {
    if(!(*strides)[d].is_number_integer())return {};
    const __int128 stride=(*strides)[d].get<std::int64_t>();
    const auto& coordinate=(*coordinates)[d];
    if(coordinate.is_number_integer()){constant+=__int128(coordinate.get<std::int64_t>())*stride;continue;}
    const auto form=ParseAffineOffset(coordinate);
    if(!form)return {};
    constant+=__int128(form->constant)*stride;
    for(auto term:form->base) {
      const auto scaled=__int128(term.coefficient)*stride;
      if(scaled<INT64_MIN || scaled>INT64_MAX)return {};
      term.coefficient=static_cast<std::int64_t>(scaled);
      result.base.push_back(std::move(term));
    }
  }
  if(constant<0 || constant>INT64_MAX)return {};
  result.constant=static_cast<std::int64_t>(constant);
  // Sorted by atom, equal atoms merged (their coefficients add; zero sums drop).
  const auto atom=[](const va::SymbolicAddressTerm& t){return std::tie(t.object,t.element_type,t.element_offset);};
  std::sort(result.base.begin(),result.base.end(),[&](const auto& x,const auto& y){return atom(x)<atom(y);});
  std::vector<va::SymbolicAddressTerm> merged;
  for(const auto& term:result.base) {
    if(!merged.empty() && atom(merged.back())==atom(term)) {
      const auto sum=__int128(merged.back().coefficient)+term.coefficient;
      if(sum<INT64_MIN || sum>INT64_MAX)return {};
      merged.back().coefficient=static_cast<std::int64_t>(sum);
    } else merged.push_back(term);
  }
  std::erase_if(merged,[](const auto& t){return !t.coefficient;});
  if(merged.empty())return {};
  result.base=std::move(merged);
  return result;
}

// An explicit address table is already part of the input. Index its affine
// metadata once; do not decode it again for each logical memory binding.
struct AddressRun {
  a::ConstantAddressBinding first;std::int64_t offset_step=0;std::size_t count=1;
  std::optional<AbstractAffinePosition> position=std::nullopt;  // one per run: equal expressions
};
// A piece of a symbolic binding column: `count` bindings from `begin` with one base and the
// constant part `constant + step * i` at the piece's i-th binding.
struct SymbolicPiece {std::size_t begin,count;std::int64_t constant,step;const std::vector<va::SymbolicAddressTerm>* base;};
class AddressIndex {
  std::vector<AddressRun> runs_;a::AffineIndex index_;
  struct SignedMatch {
    __int128 left,left_step,right,right_step;
    std::size_t count;
  };
  template<class Consumer> void intersect(const BindingColumn& binding,Consumer consume) const {
    if(!binding.count)return;
    const auto step=binding.delta[8];
    const auto first=__int128(binding.first.address.value());
    const auto last=first+__int128(step)*(binding.count-1);
    Require(last>=0 && last<=UINT64_MAX,"invalid_template_binding","Address identity progression overflows");
    const bool reverse=step<0;
    const auto magnitude=static_cast<std::uint64_t>(reverse?-__int128(step):__int128(step));
    index_.intersect({static_cast<std::uint64_t>(reverse?last:first),magnitude,step?binding.count:1},
        [&](auto id,auto m) {
      SignedMatch match{m.left,m.left_step,m.right,m.right_step,m.count};
      if(!step)match={0,1,m.right,0,binding.count};
      else if(reverse) {
        match.left=__int128(binding.count)-1-m.left;
        match.left_step=-__int128(m.left_step);
      }
      consume(id,match);
    });
  }
 public:
  explicit AddressIndex(const std::vector<a::ConstantAddressBinding>& addresses) {
    for(const auto& value:addresses) {
      if(!runs_.empty()) {
        auto& run=runs_.back();const auto& x=run.first;
        if(value.address.value()==std::uint64_t(x.address.value())+run.count &&
            value.address_space==x.address_space && value.allocation==x.allocation &&
            value.alias_scope==x.alias_scope && value.expression==x.expression) {
          const auto difference=__int128(value.offset)-x.offset;
          if(run.count==1 && difference>=INT64_MIN && difference<=INT64_MAX)run.offset_step=static_cast<std::int64_t>(difference);
          if(difference==__int128(run.offset_step)*run.count){++run.count;continue;}
        }
      }
      runs_.push_back({value});
    }
    for(auto& run:runs_)if(!run.offset_step)run.position=ParseAbstractAffine(run.first);
    for(std::size_t i=0;i<runs_.size();++i)index_.add({runs_[i].first.address.value(),1,runs_[i].count},i);
    index_.seal();
  }
  // The column split into pieces of affine symbolic positions, in binding order; empty when
  // some binding's address has no abstract affine position, the column is not symbolic, or
  // its matches interleave. O(address runs met) per column.
  std::vector<SymbolicPiece> symbolicPieces(const BindingColumn& binding) const {
    std::vector<SymbolicPiece> pieces;
    if(binding.first.region || binding.delta[10] || binding.first.value_byte_offset)return pieces;
    std::vector<SymbolicPiece> matches;bool valid=true;std::size_t covered=0;
    intersect(binding,[&](auto id,auto m) {
      const auto& run=runs_[id];
      if(!run.position || (m.count>1 && m.left_step!=1) || m.left<0){valid=false;return;}
      const auto constant=__int128(run.position->constant)+binding.first.component_offset;
      if(constant<0 || constant>INT64_MAX){valid=false;return;}
      matches.push_back({static_cast<std::size_t>(m.left),m.count,static_cast<std::int64_t>(constant),0,&run.position->base});
      covered+=m.count;
    });
    if(!valid || covered!=binding.count)return {};
    std::sort(matches.begin(),matches.end(),[](const auto& x,const auto& y){return x.begin<y.begin;});
    for(const auto& match:matches) {
      if(!pieces.empty()) {
        auto& piece=pieces.back();
        if(piece.begin+piece.count!=match.begin){pieces.clear();return pieces;}
        const auto delta=__int128(match.constant)-piece.constant;
        if(*piece.base==*match.base && (match.count==1 || delta==0)) {
          if(piece.count==1 && delta>=INT64_MIN && delta<=INT64_MAX && (match.count==1 || !delta))
            piece.step=static_cast<std::int64_t>(delta);
          if(delta==__int128(piece.step)*piece.count && (match.count==1 || !piece.step)){piece.count+=match.count;continue;}
        }
      } else if(match.begin){return pieces;}
      pieces.push_back(match);
    }
    return pieces;
  }
  void validate(const BindingColumn& binding) const {
    if(!binding.first.region)return;
    const auto& b=binding.first;
    std::size_t covered=0;
    intersect(binding,[&](auto id,auto m) {
      const auto& run=runs_[id];const auto& address=run.first;
      Require(address.expression.empty() && (address.alias_scope.empty() || address.address_space=="tensor") &&
          Allocation(address)==b.region->allocation,"changed_checked_memory_region","Checked allocation differs from its value binding");
      const auto expected=__int128(address.offset)+__int128(run.offset_step)*m.right+
          b.component_offset+__int128(binding.delta[10])*m.left;
      const auto actual=__int128(b.region->offset)+__int128(binding.delta[9])*m.left;
      Require(expected==actual && (m.count==1 ||
          __int128(run.offset_step)*m.right_step+__int128(binding.delta[10])*m.left_step==__int128(binding.delta[9])*m.left_step),
          "changed_checked_memory_region","Checked byte range differs from its value binding");
      covered+=m.count;
    });
    Require(covered==binding.count,"changed_checked_memory_region","Checked address is missing");
  }
  std::optional<va::ConstantRegion> symbolic(const BindingColumn& binding) const {
    if(binding.first.region || binding.delta[10])return {};
    std::optional<va::ConstantRegion> result;std::size_t covered=0;bool valid=true;
    intersect(binding,[&](auto id,auto m) {
      const auto& address=runs_[id].first;
      if(address.expression.empty() || !address.alias_scope.empty() ||
          address.address_space.empty() || address.allocation.empty()) {valid=false;return;}
      const va::ConstantRegion region{Allocation(address),binding.first.component_offset,binding.first.bytes};
      if(result && *result!=region)valid=false;
      else result=region;
      covered+=m.count;
    });
    return valid && covered==binding.count?result:std::nullopt;
  }
};

class Serializer {
  const SemanticProgram& program_;const a::SourceArtifact& checked_;
  const a::MemoryRaceResult& race_;const a::MemoryAccessRecords& accesses_;
  CheckedMemoryActions& result_;
  const a::HappensBeforeGraph& certified_;
  a::EventRecords events_;a::HappensBeforeEdges edges_;
  std::vector<AccessColumn> access_columns_;
  std::map<AccessKey,a::AffineIndex> access_index_;
  std::vector<ActionColumn> actions_;
  std::map<std::size_t,std::vector<EndpointColumn>> reads_,writes_;
  std::unordered_map<std::size_t,std::uint32_t> value_events_;
  std::unordered_map<std::size_t,std::uint32_t> load_inputs_;
  std::optional<SelectorPatterns> selectors_;
  std::map<std::uint32_t,std::size_t> duplicate_groups_;
  // Groups from this index on are same-value write obligations.
  std::size_t same_value_groups_begin_=SIZE_MAX;
  // The checked access of `length` actions of a same-value obligation write, from the action at
  // `begin` of `action`'s column: one run, resolved only to name an unproven write. O(1).
  void sameValueWrites(const va::OrderedMemoryAction& action,
      const va::MemoryActionCodec::Delta& delta,std::size_t begin,std::size_t length,std::uint32_t access) {
    if(!length)return;
    const auto first=__int128(action.operation.instance)+__int128(delta[0])*begin;
    const auto last=first+__int128(delta[0])*(length-1);
    Require(first>=0 && last>=0 && first<=SIZE_MAX && last<=SIZE_MAX,"invalid_template_binding",
        "Affine binding exceeds its domain");
    result_.same_value_writes.push_back({{action.operation.node,static_cast<std::size_t>(first)},delta[0],length,
        a::AccessId(access)});
  }
  struct Partition {Domain offsets;std::uint64_t bytes;};
  std::map<std::uint32_t,std::vector<Partition>> duplicate_parts_;
  struct MembershipRun {std::size_t begin;Domain threads;};
  std::unordered_map<const void*,std::vector<MembershipRun>> membership_;
  const std::vector<MembershipRun>& membership(const v::ThreadMembers& members) {
    const auto key=members.storageIdentity();auto [it,inserted]=membership_.try_emplace(key);
    if(!inserted)return it->second;
    // Explicit launch/member metadata is indexed once per shared domain, not
    // once per operation or per dependency. Subsequent joins use progressions.
    for(std::size_t i=0;i<members.size();) {
      std::size_t end=i+1;std::uint64_t step=0;
      if(end<members.size() && members[end]>members[i]) {
        step=members[end]-members[i];++end;
        while(end<members.size() && members[end]==members[i]+step*(end-i))++end;
      }
      it->second.push_back({i,{members[i],step,end-i}});i=end;
    }
    return it->second;
  }
  std::uint32_t addFamily(std::size_t count) {
    Require(count && count<=UINT32_MAX-events_.size(),"serialization_size_limit","Event identities exceed their finite domain");
    const auto first=std::uint32_t(events_.size());
    events_.appendAffine(a::Event{a::EventId(first),a::EventSchedulingClass::kNormal,{}, {},a::OperationCompleteEvent{}},
        a::EventRecordCodec::Delta{1,0,0,0,0,0,0},count);return first;
  }
  void edge(Domain from,Domain to) {
    Require(from.count==to.count,"invalid_template_dependency","Dependency domains differ");
    if(!from.count)return;
    edges_.appendAffine(a::HappensBeforeEdge{a::EventId(std::uint32_t(from.first)),a::EventId(std::uint32_t(to.first)),
        a::HappensBeforeEdgeReason::kEndpointDependency,{},{}},
        a::HappensBeforeEdgeCodec::Delta{std::int64_t(from.step),std::int64_t(to.step),0,0},from.count);
  }
  void orderingSkeleton() {
    a::ProfileScope profile("value_analysis.serialization.ordering_endpoints");
    a::EventFamilies families(certified_.events());
    std::map<std::size_t,std::uint32_t> write_ends;
    accesses_.forEachAffineColumn([&](const auto& first,const auto& delta,auto,auto,std::size_t count) {
      if(first.mode!=a::AccessMode::kWrite || !std::holds_alternative<a::AddressId>(first.address))return;
      Require(delta[1]>=0,"invalid_template_binding","Checked event progression decreases");
      families.intersect({first.event.value(),std::uint64_t(delta[1]),count},[&](auto family,auto) {
        if(!write_ends.contains(family))write_ends.emplace(family,addFamily(families.families()[family].count));
      });
    });
    // An atomic ordering endpoint may contain both copy reads and writes.
    // Refine it locally into entry -> numerical work -> exit; all original
    // outgoing edges leave the exit. This does not mutate certified race HB.
    certified_.edges().forEachAffineColumn([&](const auto& stored_first,const auto& stored_delta,auto,auto,std::size_t count) {
      if(!count)return;
      auto first=stored_first;auto delta=stored_delta;
      // Normalize only the enumeration direction. The dependency endpoints
      // are unchanged; the target may still progress in reverse member order.
      if(delta[0]<0) {
        first=Advance<a::HappensBeforeEdge,a::HappensBeforeEdgeCodec>(first,delta,count-1);
        for(auto& x:delta) {
          Require(x!=INT64_MIN,"invalid_template_dependency","Ordering stride overflows");x=-x;
        }
      }
      std::size_t covered=0;
      families.intersect({first.from.value(),std::uint64_t(delta[0]),count},[&](auto family,auto m) {
        if(!delta[0]){m.count=count;m.left_step=1;m.right_step=0;}
        auto e=Advance<a::HappensBeforeEdge,a::HappensBeforeEdgeCodec>(first,delta,m.left);
        auto d=Scale(delta,m.left_step);
        if(auto it=write_ends.find(family);it!=write_ends.end()) {
          e.from=a::EventId(it->second+m.right);d[0]=m.right_step;
        }
        edges_.appendAffine(e,d,m.count);covered+=m.count;
      });
      Require(covered==count,"invalid_template_dependency","Ordering endpoint is missing");
    });
    for(auto [family,exit]:write_ends) {
      const auto& f=families.families()[family];edge({f.first.value(),f.stride,f.count},{exit,1,f.count});
    }
    accesses_.forEachAffineColumn([&](const auto& first,const auto& delta,auto,auto,std::size_t count) {
      if(!std::holds_alternative<a::AddressId>(first.address))return;
      Require(delta[2]>=0 && delta[1]>=0,"invalid_template_binding","Checked instruction/event identity progression decreases");
      std::size_t covered=0;
      families.intersect({first.event.value(),std::uint64_t(delta[1]),count},[&](auto family,auto m) {
        if(!delta[1]){m.count=count;m.left_step=1;m.right_step=0;}
        auto access=Advance<a::MemoryAccess,a::MemoryAccessCodec>(first,delta,m.left);
        auto step=Scale(delta,m.left_step);
        if(first.mode==a::AccessMode::kWrite) {
          access.event=a::EventId(write_ends.at(family)+m.right);step[1]=m.right_step;
        }
        const auto id=access_columns_.size();access_columns_.push_back({access,step,m.count});
        access_index_[Key(access)].add({access.instruction.value(),std::uint64_t(step[2]),m.count},id);
        covered+=m.count;
      });
      Require(covered==count,"invalid_template_binding","Checked access endpoint is missing");
    });
  }
  void bindMemory() {
    a::ProfileScope profile("value_analysis.serialization.bind_templates");
    for(auto& [key,index]:access_index_){(void)key;index.seal();}
    AddressIndex address_index(checked_.addresses);
    // Source metadata is independent of memory effects; use the exact source
    // column partition to validate occurrence identities in bulk.
    struct SourceColumn {
      a::ActorId actor;std::optional<a::PtxOrigin> origin;
      std::int64_t actor_step,occurrence_step;
    };
    std::vector<SourceColumn> source_columns;a::AffineIndex source_index;
    checked_.program.instructions.forEachAffineColumn([&](const auto& first,const auto& delta,auto,auto,std::size_t count) {
      const auto id=source_columns.size();
      source_columns.push_back({first.actor,first.ptx_origin,delta[1],delta[3]});
      Require(delta[0]>=0,"invalid_template_binding","Source instruction identity decreases");
      source_index.add({first.id.value(),std::uint64_t(delta[0]),count},id);
    });source_index.seal();
    ActorIndex actors(checked_.program.actors);
    std::vector<std::vector<Domain>> uncovered(access_columns_.size());
    for(std::size_t i=0;i<uncovered.size();++i)uncovered[i].push_back({0,1,access_columns_[i].count});
    program_.memory_bindings.forEachAffineColumn([&](const auto& first,const auto& delta,auto,auto,std::size_t count) {
      // A changing node identity denotes distinct expression templates, not
      // members of one template. Preserve these explicit template boundaries.
      const auto consume=[&](BindingColumn b) {
        ++result_.statistics.binding_patterns;
        Require(b.delta[0]>=0 && b.delta[7]>=0,"invalid_template_binding","Memory binding progression decreases");
        const auto found=access_index_.find(Key(b.first.source));
        Require(found!=access_index_.end(),"missing_checked_memory_mapping","Mapped effect is absent from the checked accesses");
        address_index.validate(b);
        const auto symbolic_region=address_index.symbolic(b);
        const auto symbolic_pieces=symbolic_region?address_index.symbolicPieces(b):std::vector<SymbolicPiece>{};
        std::size_t source_covered=0;
        source_index.intersect({b.first.source.instruction.value(),std::uint64_t(b.delta[0]),b.count},[&](auto id,auto match) {
          if(!b.delta[0]){match.count=b.count;match.left_step=1;match.right_step=0;}
          const auto& column=source_columns[id];
          const auto binding=b.at(match.left);
          Require(column.origin.has_value() && column.actor_step>=0,
              "changed_checked_occurrence","Checked occurrence identity is missing");
          const auto& origin=*column.origin;
          Require(origin.instruction_index==binding.occurrence.instruction &&
              __int128(origin.occurrence)+__int128(column.occurrence_step)*match.right==binding.occurrence.occurrence &&
              (match.count==1 || (!b.delta[4] && __int128(column.occurrence_step)*match.right_step==__int128(b.delta[5])*match.left_step)),
              "changed_checked_occurrence","Checked occurrence differs from its value binding");
          actors.validate({column.actor.value()+std::uint64_t(column.actor_step)*match.right,
              std::uint64_t(column.actor_step)*match.right_step,match.count},binding.occurrence,
              {b.delta[1]*std::int64_t(match.left_step),b.delta[2]*std::int64_t(match.left_step),b.delta[3]*std::int64_t(match.left_step)});
          source_covered+=match.count;
        });
        Require(source_covered==b.count,"changed_checked_occurrence","Mapped instruction is absent");
        std::size_t covered=0;
        found->second.intersect({b.first.source.instruction.value(),std::uint64_t(b.delta[0]),b.count},[&](auto id,auto match) {
          if(!b.delta[0]){match.count=b.count;match.left_step=1;match.right_step=0;}
          const auto& column=access_columns_[id];
          const auto access=column.at(match.right);const auto binding=b.at(match.left);
          Require(column.delta[3]>=0,"invalid_template_binding","Checked actor progression decreases");
          actors.validate({access.actor.value(),std::uint64_t(column.delta[3])*match.right_step,match.count},binding.occurrence,
              {b.delta[1]*std::int64_t(match.left_step),b.delta[2]*std::int64_t(match.left_step),b.delta[3]*std::int64_t(match.left_step)});
          const auto* address=std::get_if<a::AddressId>(&access.address);
          Require(address && binding.mode==access.mode && binding.address==*address && access.bytes==binding.bytes &&
              (match.count==1 || __int128(b.delta[8])*match.left_step==__int128(column.delta[4])*match.right_step),
              "changed_checked_memory_access","Checked access differs from its value binding");
          const Domain endpoints{access.event.value(),std::uint64_t(column.delta[1])*match.right_step,match.count};
          const Domain instances{binding.value.instance,std::uint64_t(b.delta[7])*match.left_step,match.count};
          auto& target=binding.mode==a::AccessMode::kRead?reads_:writes_;
          target[binding.value.node].push_back({instances,endpoints});
          va::OrderedMemoryAction action{binding.value,binding.region,binding.unsupported_reason,binding.value_byte_offset,{}};
          action.symbolic_region=symbolic_region;
          if(action.symbolic_region && !binding.value_byte_offset) {
            const auto& node=program_.values.node(binding.value.node);
            if(dynamic_cast<const v::LoadOp*>(&node.operation())) {
              const auto bits=program_.values.type(node.result(0,binding.value.instance)).bitSize();
              Require(bits && bits%8==0 && binding.component_offset<=binding.bytes &&
                  bits/8<=binding.bytes-binding.component_offset,
                  "invalid_symbolic_memory_payload","Symbolic Load component exceeds its source access");
              action.symbolic_region->bytes=bits/8;
              // Value lowering has already applied the component displacement
              // to this Load's address operand (e.g. each cp.async byte).
              action.symbolic_region->offset=0;
            }
          }
          // Duplicate-write obligations are explicit checker output. They are
          // not inferred or waived by choosing a serialization order.
          const va::MemoryActionCodec::Delta action_delta{b.delta[7]*std::int64_t(match.left_step),
              b.delta[9]*std::int64_t(match.left_step),b.delta[11]*std::int64_t(match.left_step)};
          const Domain access_ids{access.id.value(),std::uint64_t(column.delta[0])*match.right_step,match.count};
          std::size_t next=0;
          // The payload bytes of a positioned action (as for symbolic_region: a Load's result,
          // a Store's value); a fragmented native access has no position.
          std::uint64_t payload=0;
          // A TMA region fragment lies at its place in the region (MemoryAttributes::region_offset).
          std::uint64_t region_offset=0;
          if(!symbolic_pieces.empty() && !binding.value_byte_offset) {
            const auto& node=program_.values.node(binding.value.node);
            const auto* load=dynamic_cast<const v::LoadOp*>(&node.operation());
            const auto* store=dynamic_cast<const v::StoreOp*>(&node.operation());
            const auto value=load?node.result(0,binding.value.instance):
                store?node.instanceOperands(binding.value.instance).at(1):v::Value{};
            if(program_.values.contains(value)) {
              const auto bits=program_.values.type(value).bitSize();
              if(bits && bits%8==0)payload=bits/8;
            }
            if(load || store)region_offset=(load?load->attributes():store->attributes()).region_offset.value_or(0);
          }
          const auto emit=[&](va::OrderedMemoryAction part,const va::MemoryActionCodec::Delta& delta,
              std::size_t begin,std::size_t length) {
            actions_.push_back({std::move(part),delta,{endpoints.first+endpoints.step*begin,endpoints.step,length},length});
          };
          // A symbolic column is cut where its binding pieces change base or constant progression.
          // Binding index of match member j: match.left + j*match.left_step (left_step >= 1).
          const auto append=[&](std::size_t begin,std::size_t length,std::optional<std::size_t> group) {
            if(!length)return;
            auto part=Advance<va::OrderedMemoryAction,va::MemoryActionCodec>(action,action_delta,begin);
            part.duplicate_write_group=group;
            if(!payload){emit(std::move(part),action_delta,begin,length);return;}
            const auto step=std::max<std::size_t>(match.left_step,1);
            for(std::size_t j=begin;j<begin+length;) {
              const auto index=std::size_t(match.left)+j*step;
              auto piece=std::upper_bound(symbolic_pieces.begin(),symbolic_pieces.end(),index,
                  [](std::size_t x,const SymbolicPiece& p){return x<p.begin;});
              Require(piece!=symbolic_pieces.begin(),"invalid_template_binding","Symbolic binding piece is missing");
              --piece;
              const auto last=std::min(begin+length,j+(piece->begin+piece->count-index+step-1)/step);
              auto member=Advance<va::OrderedMemoryAction,va::MemoryActionCodec>(action,action_delta,j);
              member.duplicate_write_group=group;
              const auto constant=__int128(piece->constant)+__int128(piece->step)*(index-piece->begin)+region_offset;
              const auto stride=__int128(piece->step)*step;
              Require(constant>=0 && constant<=INT64_MAX && stride>=INT64_MIN && stride<=INT64_MAX,
                  "invalid_template_binding","Symbolic position progression overflows");
              member.symbolic_position=va::SymbolicPosition{{symbolic_region->allocation,
                  static_cast<std::uint64_t>(constant),payload},*piece->base};
              auto delta=action_delta;delta[1]=static_cast<std::int64_t>(stride);
              emit(std::move(member),delta,j,last-j);j=last;
            }
          };
          // The duplicate groups on the access-id progression, in order. A group between members
          // jumps to the next member at or after it: O(log groups) per member or skipped gap, not
          // one step per group in the progression's hull.
          for(auto it=duplicate_groups_.lower_bound(std::uint32_t(access_ids.first));
              it!=duplicate_groups_.end() && it->first<=access_ids.last();++it) {
            const auto m=a::IntersectAffine(access_ids,{it->first,0,1});
            if(!m.count) {
              if(!access_ids.step)break;
              const auto member=access_ids.first+((it->first-access_ids.first)/access_ids.step+1)*access_ids.step;
              if(member>access_ids.last() || member>UINT32_MAX)break;
              const auto jump=duplicate_groups_.lower_bound(std::uint32_t(member));
              if(jump==duplicate_groups_.end())break;
              it=std::prev(jump);continue;
            }
            append(next,m.left-next,{});
            const auto length=access_ids.step?1:match.count;
            append(m.left,length,it->second);next=m.left+length;
            if(it->second>=same_value_groups_begin_) [[unlikely]]
              sameValueWrites(action,action_delta,m.left,length,it->first);
            Require(binding.region.has_value(),"dynamic_address_unsupported","Duplicate-store byte regions are not constant");
            Require(b.delta[10]>=0,"invalid_template_binding","Duplicate-store component progression decreases");
            duplicate_parts_[it->first].push_back({{binding.component_offset+std::uint64_t(b.delta[10])*match.left_step*m.left,
                std::uint64_t(b.delta[10])*match.left_step,length},binding.region->bytes});
          }
          append(next,match.count-next,{});
          // Several components may bind the same checked access (including a
          // dynamic-address load). Coverage is over access identities, not the
          // sum of component sizes; repeated components must not count twice.
          std::vector<Domain> remaining;
          for(auto domain:uncovered[id]) {
            const auto pieces=a::SubtractAffine(domain,{match.right,match.right_step,match.count});
            remaining.insert(remaining.end(),pieces.begin(),pieces.end());
          }
          uncovered[id]=std::move(remaining);
          covered+=match.count;
        });
        Require(covered==b.count,"missing_checked_memory_mapping","Memory binding coverage is incomplete");
      };
      if(delta[6] && count>1) {
        for(std::size_t i=0;i<count;++i)consume({Advance<SemanticMemoryBinding,SemanticMemoryBindingCodec>(first,delta,i),{},1});
      } else consume({first,delta,count});
    });
    for(const auto& remaining:uncovered)
      Require(remaining.empty(),"missing_checked_memory_mapping","Checked memory accesses are not completely covered by value bindings");
    std::map<std::size_t,std::vector<std::tuple<std::uint64_t,std::uint64_t,std::uint64_t>>> partitions;
    for(auto [access,group]:duplicate_groups_) {
      auto& pieces=duplicate_parts_[access];
      std::sort(pieces.begin(),pieces.end(),[](auto x,auto y){return x.offsets.first<y.offsets.first;});
      // Equal byte-sized partitions are compared as progression domains. No
      // thread/event instances are materialized to check an assumption warning.
      std::vector<std::tuple<std::uint64_t,std::uint64_t,std::uint64_t>> normalized;
      for(auto p:pieces) {
        Require(p.offsets.count==1 || p.offsets.step==p.bytes,"duplicate_write_value_equality_unproved",
            "Duplicate-store fragments do not form a regular byte partition");
        if(!normalized.empty() && std::get<1>(normalized.back())==p.bytes &&
            std::get<0>(normalized.back())+p.bytes*std::get<2>(normalized.back())==p.offsets.first)
          std::get<2>(normalized.back())+=p.offsets.count;
        else normalized.emplace_back(p.offsets.first,p.bytes,p.offsets.count);
      }
      auto [it,inserted]=partitions.try_emplace(group,normalized);
      Require(!normalized.empty() && (inserted || it->second==normalized),"duplicate_write_value_equality_unproved",
          "Duplicate-store value fragments have different byte partitions");
    }
  }
  void operand(const v::Node& consumer,const v::Instance& pattern,std::size_t begin,std::size_t count,bool rebind,v::Value value) {
    for(auto index:value.indices())operand(consumer,pattern,begin,count,rebind,index);
    if(value.threadSelector())operand(consumer,pattern,begin,count,rebind,*value.threadSelector());
    if(value.conversion()) {
      auto source=value.conversion()->source;
      if(value.threadSelector())source=program_.values.fromThread(source,*value.threadSelector());
      operand(consumer,pattern,begin,count,rebind,source);return;
    }
    value=program_.values.resolve(value);
    const auto& producer=program_.values.definingNode(value);
    const auto target_family=load_inputs_.contains(consumer.id())?load_inputs_.at(consumer.id()):value_events_.at(consumer.id());
    const auto target=target_family+begin;
    const auto source=value_events_.at(producer.id());
    if(value.threadSelector()) {
      const auto selector=program_.values.resolve(*value.threadSelector());
      const auto& selector_node=program_.values.definingNode(selector);
      if(const auto* constant=dynamic_cast<const v::ConstantOp*>(&selector_node.operation());
          constant && !selector.conversion() && selector.indices().empty() && !selector.threadSelector()) {
        const auto thread=std::stoull(constant->literal());
        const auto member=producer.memberIndex(thread);
        Require(member.has_value(),"missing_value_member_mapping","Selected value thread is absent");
        edge({source+*member,0,count},{target,1,count});return;
      }
      // Evaluate the selector as finite index patterns, not once per lane and
      // not an artificial all-producer -> all-consumer barrier.
      Require(pattern.origin.cta && pattern.origin.thread,"missing_value_member_mapping","Thread selector has no execution coordinates");
      if(!selectors_)selectors_.emplace(program_.values);
      std::size_t covered=0;
      for(const auto& r:membership(consumer.instanceThreads())) {
        auto domain=a::IntersectAffine({begin,1,count},{r.begin,1,r.threads.count});if(!domain.count)continue;
        Domain threads{r.threads.first+r.threads.step*domain.right,r.threads.step,domain.count};
        std::vector<SelectorPatterns::Piece> pieces;
        try {pieces=selectors_->evaluate(selector,*pattern.origin.cta,*pattern.origin.thread,threads,rebind);}
        catch(const std::invalid_argument& error){throw SerializationError("unsupported_selector_pattern",error.what());}
        for(const auto& piece:pieces) {
          Require(piece.first>=0 && piece.first<=UINT64_MAX && piece.step>=0 && piece.step<=UINT64_MAX,
              "unsupported_selector_pattern","Selected thread progression is not increasing");
          const auto consumer_members=a::IntersectAffine(piece.threads,r.threads);
          Require(consumer_members.count==piece.threads.count,"missing_value_member_mapping","Selector domain is not a member subset");
          for(const auto& l:membership(producer.instanceThreads())) {
            auto m=a::IntersectAffine({std::uint64_t(piece.first),std::uint64_t(piece.step),piece.threads.count},l.threads);
            if(!m.count)continue;
            if(!piece.step){m.count=piece.threads.count;m.left_step=1;m.right_step=0;}
            edge({source+l.begin+m.right,m.right_step,m.count},
                {target_family+r.begin+consumer_members.right+consumer_members.right_step*m.left,
                 consumer_members.right_step*m.left_step,m.count});covered+=m.count;
          }
        }
      }
      Require(covered==count,"missing_value_member_mapping","Selected value member coverage is incomplete");return;
    }
    const auto& from=producer.instanceThreads();const auto& to=consumer.instanceThreads();
    if(!rebind || !pattern.origin.thread || producer.origin().cta!=pattern.origin.cta ||
        value.instanceIndex()>=from.size() || from[value.instanceIndex()]!=*pattern.origin.thread) {
      edge({source+value.instanceIndex(),0,count},{target,1,count});return;
    }
    std::size_t covered=0;
    const auto& left=membership(from);const auto& right=membership(to);
    for(const auto& r:right) {
      auto domain=a::IntersectAffine({begin,1,count},{r.begin,1,r.threads.count});
      if(!domain.count)continue;
      Domain threads{r.threads.first+r.threads.step*domain.right,r.threads.step,domain.count};
      for(const auto& l:left) {
        auto m=a::IntersectAffine(threads,l.threads);if(!m.count)continue;
        edge({source+l.begin+m.right,m.right_step,m.count},
            {target_family+r.begin+domain.right+m.left,m.left_step,m.count});
        covered+=m.count;
      }
    }
    Require(covered==count,"missing_value_member_mapping","Value operand member mapping is incomplete");
  }
  void numericalDependencies() {
    a::ProfileScope profile("value_analysis.serialization.value_dependencies");
    // Traverse every shared operand pattern once. Scheduling uses the existing
    // DAG, not repeated backward searches from each output or MMA binding.
    std::set<std::size_t> needed;std::vector<std::size_t> pending;
    for(const auto& [node,items]:reads_){(void)items;pending.push_back(node);}
    for(const auto& [node,items]:writes_){(void)items;pending.push_back(node);}
    const auto enqueue=[&](auto&& self,v::Value value)->void {
      pending.push_back(program_.values.resolve(value).nodeId());
      if(value.conversion())self(self,value.conversion()->source);
      for(auto index:value.indices())self(self,index);
      if(value.threadSelector())self(self,*value.threadSelector());
    };
    while(!pending.empty()) {
      const auto id=pending.back();pending.pop_back();if(!needed.insert(id).second)continue;
      const auto& node=program_.values.node(id);
      value_events_.emplace(id,addFamily(node.instanceCount()));
      if(reads_.contains(id))load_inputs_.emplace(id,addFamily(node.instanceCount()));
      node.forEachInstancePattern([&](const auto& pattern,auto,auto,bool) {
        for(auto value:pattern.operands)enqueue(enqueue,value);
      });
    }
    result_.statistics.value_families=needed.size();
    for(auto id:needed) {
      const auto& node=program_.values.node(id);
      node.forEachInstancePattern([&](const auto& pattern,auto begin,auto count,bool rebind) {
        for(auto value:pattern.operands)operand(node,pattern,begin,count,rebind,value);
      });
    }
    for(const auto& [id,columns]:reads_)for(const auto& column:columns) {
      edge({load_inputs_.at(id)+column.instances.first,column.instances.step,column.instances.count},column.events);
      edge(column.events,{value_events_.at(id)+column.instances.first,column.instances.step,column.instances.count});
    }
    for(const auto& [id,columns]:writes_)for(const auto& column:columns)
      edge({value_events_.at(id)+column.instances.first,column.instances.step,column.instances.count},column.events);
  }
  void schedule() {
    a::ProfileScope profile("value_analysis.serialization.schedule");
    a::AffineTemplateSchedule ready(events_,edges_);
    result_.statistics.event_families=ready.familyCount();result_.statistics.relations=ready.relationCount();
    std::vector<std::vector<std::size_t>> family_actions(ready.familyCount());
    for(std::size_t i=0;i<actions_.size();++i) {
      const auto& action=actions_[i];const auto [family,member]=ready.layout()->locate(a::EventId(action.events.first));
      const auto [last_family,last_member]=ready.layout()->locate(a::EventId(action.events.last()));
      (void)member;(void)last_member;
      Require(family==last_family,"action_family_split_required","Action binding crosses ordering family boundaries");
      family_actions[family].push_back(i);
    }
    while(true) {
      auto batch=ready.next(true);if(!batch)batch=ready.next(false);if(!batch)break;
      ++result_.statistics.scheduled_batches;
      result_.statistics.singleton_batches+=batch->count==1;
      result_.statistics.largest_batch_members=std::max(result_.statistics.largest_batch_members,std::size_t(batch->count));
      for(auto id:family_actions[batch->family]) {
        const auto& column=actions_[id];
        auto m=a::IntersectAffine(column.events,{batch->first.value(),batch->stride,batch->count});
        if(!m.count)continue;
        if(!column.events.step){m.count=column.count;m.left_step=1;}
        auto action=Advance<va::OrderedMemoryAction,va::MemoryActionCodec>(column.first,column.delta,m.left);
        result_.actions.appendAffine(std::move(action),Scale(column.delta,m.left_step),m.count);
        ++result_.statistics.action_patterns;
      }
      ready.finish(*batch);
    }
    result_.statistics.domain_refinements=ready.domainRefinements();
    result_.statistics.logical_events=ready.executedCount();result_.statistics.logical_actions=result_.actions.size();
    Require(ready.executedCount()==events_.size(),"incompatible_numerical_order",
        "Value dependencies and the certified execution order cannot be jointly serialized");
    Require(result_.actions.size()==program_.memory_bindings.size(),"unexecuted_mapped_memory","Serialization omitted memory bindings");
  }
 public:
  Serializer(const SemanticProgram& program,const a::SourceArtifact& checked,const a::PassCertificate& certificate,
      const a::MemoryRaceResult& race,const a::MemoryAccessRecords& accesses,CheckedMemoryActions& result)
      :program_(program),checked_(checked),race_(race),accesses_(accesses),result_(result),
       certified_(certificate.happensBeforeGraph),events_(certificate.happensBeforeGraph.events()){}
  void run() {
    for(std::size_t group=0;group<race_.duplicate_write_groups.size();++group) {
      const auto& members=race_.duplicate_write_groups[group];
      Require(members.size()>=2,"invalid_duplicate_write_group","Duplicate-write group has fewer than two accesses");
      std::optional<a::MemoryAccess> first;
      for(auto id:members) {
        Require(id.value()<accesses_.size(),"invalid_duplicate_write_group","Duplicate access is missing");
        const auto access=accesses_.at(id.value());
        Require(access.id==id && access.mode==a::AccessMode::kWrite && (!first || (first->address==access.address && first->bytes==access.bytes)) &&
            duplicate_groups_.emplace(id.value(),group).second,"invalid_duplicate_write_group","Duplicate-write footprints differ");
        first=access;
      }
    }
    // Same-value write obligations: the race check established one footprint per group from
    // the address overlap, not from address identity (each thread binds its own address).
    const auto begin=race_.duplicate_write_groups.size();
    for(std::size_t k=0;k<race_.same_value_write_groups.size();++k) {
      const auto& members=race_.same_value_write_groups[k];
      Require(members.size()>=2,"invalid_duplicate_write_group","Same-value write group has fewer than two accesses");
      std::optional<std::uint64_t> bytes;
      for(auto id:members) {
        Require(id.value()<accesses_.size(),"invalid_duplicate_write_group","Same-value write access is missing");
        const auto access=accesses_.at(id.value());
        Require(access.id==id && access.mode==a::AccessMode::kWrite && access.bytes && (!bytes || *bytes==*access.bytes) &&
            duplicate_groups_.emplace(id.value(),begin+k).second,"invalid_duplicate_write_group","Same-value write footprints differ");
        bytes=access.bytes;
      }
    }
    if(!race_.same_value_write_groups.empty())
      same_value_groups_begin_=result_.contract.same_value_groups_begin=begin;
    orderingSkeleton();bindMemory();numericalDependencies();schedule();
    std::map<std::string,va::AllocationInfo> allocations;
    for(const auto& address:checked_.addresses) {
      if(address.allocation.empty() || address.address_space.empty())continue;
      const auto id=Allocation(address);const bool global=address.address_space=="global";
      allocations.try_emplace(id,va::AllocationInfo{id,global || address.address_space=="const",global});
    }
    for(auto& [key,allocation]:allocations){(void)key;result_.contract.allocations.push_back(std::move(allocation));}
    // The trusted properties the race check used decide symbolic-segment aliasing too.
    const auto properties=nlohmann::json::parse(checked_.tensor_properties_json,nullptr,false);
    Require(properties.is_array(),"invalid_tensor_properties","Tensor properties are not an array");
    for(const auto& record:properties) {
      Require(record.is_object() && record.contains("object") && record.contains("element_type"),
          "invalid_tensor_properties","Tensor property record has no object or element type");
      const auto bound=[&](const char* key)->std::optional<std::int64_t> {
        const auto found=record.find(key);
        if(found==record.end() || !found->is_number_integer())return {};
        if(found->is_number_unsigned() && found->get<std::uint64_t>()>std::uint64_t(INT64_MAX))return {};
        return found->get<std::int64_t>();
      };
      va::TensorPropertyFact fact{record["object"].get<std::string>(),
          record["element_type"].get<std::string>(),bound("lower_bound"),bound("upper_bound"),
          record.value("injective",false)};
      if(const auto bits=record.find("field_bits");bits!=record.end()) {
        Require(bits->is_number_unsigned() && bits->get<unsigned>()>=1 && bits->get<unsigned>()<=8 &&
            record.contains("field_values") && record["field_values"].is_array(),
            "invalid_tensor_properties","Tensor property field values are malformed");
        fact.field_bits=bits->get<unsigned>();
        for(const auto& value:record["field_values"]) {
          Require(value.is_number_unsigned() && value.get<std::uint64_t>()<(std::uint64_t{1}<<fact.field_bits),
              "invalid_tensor_properties","Tensor property field value does not fit its width");
          fact.field_values.set(value.get<std::size_t>());
        }
      }
      result_.contract.tensor_properties.push_back(std::move(fact));
    }
  }
};
}
CheckedMemoryActions SerializeCheckedMemory(const SemanticProgram& program,const a::SourceArtifact& checked,
    const a::PassCertificate& certificate,const a::MemoryRaceResult& race,const a::MemoryAccessRecords& accesses,
    bool lifetimes_checked) {
  CheckedMemoryActions result;result.contract={true,race.status==a::MemoryRaceStatus::kRaceFree && race.search_complete,lifetimes_checked,{}};
  try {
    Require(result.contract.race_free && lifetimes_checked,"unchecked_memory_order","Memory elimination requires synchronization, race, and lifecycle checks");
    Require(!(race.warning_count || !race.duplicate_tma_store_warnings.empty()) ||
        (race.duplicate_write_groups_complete && !race.duplicate_write_groups.empty()),
        "unordered_writes_accepted_by_assumption","Accepted unordered writes have no complete value-equality obligations");
    if(!program.valuesComplete()) {
      const auto& d=program.value_diagnostics.front();throw SerializationError(d.code,d.message);
    }
    if(!program.mappingComplete()) {
      result.diagnostics.push_back(program.diagnostics.front());return result;
    }
    Require(program.source_complete && checked.verification_limitations.empty() && checked.source_violations.empty(),
        "incomplete_source_semantics","Source translation or lifecycle semantics is incomplete");
    Serializer(program,checked,certificate,race,accesses,result).run();
  } catch(const SerializationError& error) {
    result.diagnostics.push_back({error.code,error.what(),{},{},{}});
  }
  return result;
}
}
