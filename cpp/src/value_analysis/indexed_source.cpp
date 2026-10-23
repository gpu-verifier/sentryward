#include "sentryward/algorithm/diagnostics/development_trace.h"
#include "sentryward/value_analysis/indexed_source.h"
#include "sentryward/value_analysis/scalar_specialization.h"
#include "sentryward/value_analysis/ideal_scalar_semantics.h"
#include "sentryward/value_analysis/packed_storage.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/ptx_operation.h"
#include "sentryward/algorithm/diagnostics/compression_audit.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include <algorithm>
#include <bit>
#include <set>
#include <sstream>
#include <typeinfo>
#include <unordered_map>
#include <type_traits>
#include <array>
#include <numeric>

namespace sentryward::value_analysis {
namespace {
namespace v = value_ir;
std::string attrs(const v::Operation& op) {
  std::ostringstream out; op.printAttributes(out); return out.str();
}
std::optional<std::int64_t> literal(const std::string& text) {
  try {
    std::size_t used=0; auto x=std::stoll(text,&used,0);
    if(used==text.size())return x;
  } catch(const std::exception&) {}
  return {};
}
// Proof tables do not depend on key order. Hash the complete structural tuple;
// buckets still compare typed handles/certificates, never hash equality alone.
struct TupleHash {
  template<class... Ts> std::size_t operator()(const std::tuple<Ts...>& key)const {
    std::size_t h=0;
    std::apply([&](const auto&... parts) {
      ((h^=std::hash<std::decay_t<decltype(parts)>>{}(parts)+
             std::size_t(0x9e3779b97f4a7c15ULL)+(h<<6)+(h>>2)),...);
    },key);
    return h;
  }
};
class Matcher {
  const v::Graph& g;
  const MemoryEliminationResult& memory;
  ScalarSpecialization specialization;
  PackedStorageDecoder packed;
  std::size_t work=0;
  using Use=IndexedSourceUse;
  using Ptr=std::shared_ptr<const Use>;
  using Kind=Use::Kind;
  std::size_t member_count=0;
  std::size_t coordinate_xor=0;
  static bool admittedPure(const v::Operation& operation) {
    return dynamic_cast<const v::UnaryOp*>(&operation) || dynamic_cast<const v::BinaryOp*>(&operation) ||
        dynamic_cast<const v::FmaOp*>(&operation) || dynamic_cast<const v::CompareOp*>(&operation) ||
        dynamic_cast<const v::SelectOp*>(&operation) || dynamic_cast<const v::IdealExp2Op*>(&operation) ||
        dynamic_cast<const v::PtxScalarOp*>(&operation);
  }
  struct CoordinateTree {
    struct Run {std::size_t first,count;std::int64_t stride;};
    std::size_t count=0,base=1;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> nodes;
    std::vector<Run> runs;
    std::pair<std::uint64_t,std::uint64_t> range(std::size_t first,std::size_t end)const {
      std::pair<std::uint64_t,std::uint64_t> result{UINT64_MAX,0};
      const auto add=[&](std::size_t i){result.first=std::min(result.first,nodes[i].first);
        result.second=std::max(result.second,nodes[i].second);};
      for(first+=base,end+=base;first<end;first/=2,end/=2) {
        if(first&1)add(first++);if(end&1)add(--end);
      }
      return result;
    }
    // The hull predicate is monotone under prefix extension. Find its first
    // violation in the segment tree; accepted subtrees are never expanded.
    std::size_t prefix(std::size_t first,std::size_t end,std::uint64_t mask,
        std::size_t& visits)const {
      auto hull=std::pair<std::uint64_t,std::uint64_t>{UINT64_MAX,0};
      const auto walk=[&](auto&& self,std::size_t node,std::size_t lo,std::size_t hi)->std::size_t {
        ++visits;
        if(hi<=first || lo>=end)return end;
        const auto trial=std::pair{std::min(hull.first,nodes[node].first),
            std::max(hull.second,nodes[node].second)};
        const auto bits=std::bit_width(trial.first^trial.second);
        const auto varying=bits==64?UINT64_MAX:((std::uint64_t{1}<<bits)-1);
        if(first<=lo && hi<=end && !(mask&varying)){hull=trial;return end;}
        if(hi-lo==1)return lo;
        const auto mid=lo+(hi-lo)/2;
        auto stop=self(self,node*2,lo,mid);
        return stop<end?stop:self(self,node*2+1,mid,hi);
      };
      return walk(walk,1,0,base);
    }
  };
  struct SelectorBounds {std::uint64_t lo,hi;std::size_t end=SIZE_MAX;const CoordinateTree* coordinates=nullptr;};
  struct FixedSelector {v::Value value;std::size_t end=SIZE_MAX;};
  struct SelectorFact {v::Value value;std::optional<SelectorBounds> bounds;};
  using SelectorKey=std::tuple<const void*,std::uint64_t,std::size_t,std::size_t,std::size_t,std::size_t,std::size_t,std::size_t>;
  std::map<SelectorKey,std::vector<SelectorFact>> selector_facts;
  struct FixedFact {v::Value value;std::optional<FixedSelector> proof;};
  std::map<SelectorKey,std::vector<FixedFact>> fixed_selector_facts;
  std::map<std::tuple<const void*,std::uint64_t,unsigned>,std::optional<CoordinateTree>> coordinate_bounds;
  std::size_t selector_scope_begin=0,selector_scope_end=SIZE_MAX;
  std::size_t selector_truncated=0;
  std::size_t selector_member_visits=0,selector_new_states=0,selector_hits=0,selector_collision_checks=0;
  void selectorWork(std::size_t amount=1) {
    if(amount>2000000-work)throw IndexedSourceLimit("indexed source selector-domain proof budget exceeded");
    work+=amount;
  }
  const CoordinateTree* coordinateMetadata(v::Value value,unsigned axis) {
    const auto& node=g.node(value.nodeId());
    if(!node.hasUniformThreadTemplate() || axis>=3)return nullptr;
    const auto cta=node.instanceOrigin(value.instanceIndex()).cta;if(!cta)return nullptr;
    // Nonzero members use Graph's shared thread bindings, but member zero is
    // the per-node stored prototype. Validate that exception on EVERY lookup,
    // including a query of a different member after another node primed cache.
    const auto prototype=node.instanceOrigin(0);
    if(node.instanceThreads().empty() || !prototype.thread ||
       *prototype.thread!=node.instanceThreads()[0] || prototype.cta!=cta ||
       !prototype.thread_coordinates || !prototype.coordinate_value ||
       *prototype.coordinate_value!=prototype.thread_coordinates->at(axis))return nullptr;
    const auto key=std::tuple{node.instanceThreads().storageIdentity(),*cta,axis};
    auto found=coordinate_bounds.find(key);
    if(found==coordinate_bounds.end()) {
      std::optional<CoordinateTree> tree{std::in_place};tree->count=node.instanceCount();
      while(tree->base<tree->count) {
        if(tree->base>SIZE_MAX/2)return nullptr;tree->base*=2;
      }
      if(tree->base>SIZE_MAX/2)return nullptr;
      selectorWork(tree->base*2);tree->nodes.assign(tree->base*2,{UINT64_MAX,0});
      for(std::size_t i=0;i<tree->count;++i) {
        selectorWork();++selector_member_visits;const auto origin=node.instanceOrigin(i);
        if(origin.cta!=cta || !origin.thread_coordinates){tree.reset();break;}
        const auto x=origin.thread_coordinates->at(axis);
        if(!origin.coordinate_value || *origin.coordinate_value!=x){tree.reset();break;}
        tree->nodes[tree->base+i]={x,x};
      }
      if(tree) {
        for(auto i=tree->base;i-->1;)
          tree->nodes[i]={std::min(tree->nodes[2*i].first,tree->nodes[2*i+1].first),
              std::max(tree->nodes[2*i].second,tree->nodes[2*i+1].second)};
        // Exact affine runs of the STORED coordinate metadata. Each member
        // belongs to one run; discontinuities are never sampled away.
        for(std::size_t first=0;first<tree->count;) {
          auto end=first+1;std::int64_t stride=0;
          if(end<tree->count) {
            const auto delta=__int128(tree->nodes[tree->base+end].first)-tree->nodes[tree->base+first].first;
            if(delta>=INT64_MIN && delta<=INT64_MAX) {
              stride=static_cast<std::int64_t>(delta);++end;
              while(end<tree->count) {
                selectorWork();
                if(__int128(tree->nodes[tree->base+end].first)-tree->nodes[tree->base+end-1].first!=stride)break;
                ++end;
              }
            }
          }
          selectorWork();tree->runs.push_back({first,end-first,stride});first=end;
        }
      }
      found=coordinate_bounds.emplace(key,std::move(tree)).first;
    }
    if(!found->second)return nullptr;
    const auto& tree=*found->second;
    if(tree.count!=node.instanceCount() ||
       tree.nodes[tree.base].first!=*prototype.coordinate_value)return nullptr;
    return &tree;
  }
  // This proof is scoped to the original full annotated use and the actual
  // immutable rebinding domain. Resolving one representative is not a proof.
  std::optional<FixedSelector> fixedInDomain(v::Value value,v::Value context,unsigned depth=0) {
    if(depth>=48){++selector_truncated;return {};}
    if(g.threadBindingRelation(value,context)==v::Graph::ThreadBindingRelation::Fixed)return FixedSelector{value};
    const auto& context_node=g.node(context.nodeId());
    const auto origin=context_node.instanceOrigin(context.instanceIndex());
    if(!context_node.hasUniformThreadTemplate() || !origin.cta)return {};
    const SelectorKey key{context_node.instanceThreads().storageIdentity(),*origin.cta,context.instanceIndex(),
        value.nodeId(),value.instanceIndex(),value.resultIndex(),selector_scope_begin,selector_scope_end};
    if(auto found=fixed_selector_facts.find(key);found!=fixed_selector_facts.end())
      for(const auto& entry:found->second){selectorWork();++selector_collision_checks;
        if(entry.value==value){++selector_hits;return entry.proof;}}
    selectorWork();++selector_new_states;const auto before=selector_truncated;
    auto proof=fixedInDomainImpl(value,context,depth);
    if(proof || before==selector_truncated)fixed_selector_facts[key].push_back({value,proof});
    return proof;
  }
  std::optional<FixedSelector> fixedInDomainImpl(v::Value value,v::Value context,unsigned depth) {
    if(value.threadSelector()) {
      const auto bounds=selectorBounds(*value.threadSelector(),context,depth+1);
      if(!bounds || bounds->lo!=bounds->hi)return {};
      // selected() preserves conversions/components/provenance. Its exact
      // representative selector must agree with the domain-wide proof.
      const auto selected_value=specialization.selected(value);
      if(selected_value.threadSelector() ||
         g.threadBindingRelation(selected_value,context)!=v::Graph::ThreadBindingRelation::Fixed)return {};
      const auto actual=specialization.known(specialization.selected(*value.threadSelector()));
      if(!actual)return {};
      const auto integer=literal(actual->literal);
      if(!integer || *integer<0 || static_cast<std::uint64_t>(*integer)!=bounds->lo)return {};
      return FixedSelector{selected_value,bounds->end};
    }
    // Conversion is an interpretation edge, not permission to forget a
    // varying selector embedded in its source.
    if(value.conversion() && value.indices().empty() && !value.pointer()) {
      const auto& conversion=*value.conversion();
      auto source=fixedInDomain(conversion.source,context,depth+1);if(!source)return {};
      auto result=conversion.spec->kind==v::TypeConversionKind::Bitcast
          ?g.bitcast(source->value,conversion.spec->target)
          :g.convert(source->value,conversion.spec->target,conversion.spec->attributes);
      return FixedSelector{result,source->end};
    }
    if(value.conversion() || value.pointer() || !value.indices().empty())return {};
    const auto& node=g.node(value.nodeId());const auto& op=node.operation();
    if(!node.hasUniformThreadTemplate() ||
       g.threadBindingRelation(value,context)!=v::Graph::ThreadBindingRelation::SameDomain)return {};
    if(dynamic_cast<const v::ConstantOp*>(&op) || dynamic_cast<const v::InputParameterOp*>(&op))
      return FixedSelector{value};
    if(const auto* coordinate=dynamic_cast<const v::CoordinateOp*>(&op)) {
      if(coordinate->kind()==v::CoordinateKind::BlockIndex ||
         coordinate->kind()==v::CoordinateKind::ClusterBlockIndex ||
         coordinate->kind()==v::CoordinateKind::ClusterRank)return FixedSelector{value};
      return {};
    }
    if(!admittedPure(op))return {};
    const auto inputs=g.instanceOperands(value);selectorWork(inputs.size());
    auto end=selector_scope_end;
    for(const auto& input:inputs) {
      const auto child=fixedInDomain(input,context,depth+1);if(!child)return {};
      end=std::min(end,child->end);
    }
    // Deterministic pure expression over proved invariant uses. Retain its
    // original handle; never rebuild its ancestors or erase type annotations.
    return FixedSelector{value,end};
  }
  std::optional<SelectorBounds> selectorBounds(v::Value value,v::Value context,unsigned depth=0) {
    if(depth>=48){++selector_truncated;return {};}
    const auto& context_node=g.node(context.nodeId());
    const auto context_origin=context_node.instanceOrigin(context.instanceIndex());
    if(!context_origin.cta || !context_node.hasUniformThreadTemplate())return {};
    const auto domain=context_node.instanceThreads().storageIdentity();
    // Fixed/SameDomain is relative to the concrete context binding; retain
    // it until a separate normalized-domain proof can justify sharing.
    const SelectorKey key{domain,*context_origin.cta,context.instanceIndex(),
        value.nodeId(),value.instanceIndex(),value.resultIndex(),selector_scope_begin,selector_scope_end};
    if(auto found=selector_facts.find(key);found!=selector_facts.end())
      for(const auto& entry:found->second){selectorWork();++selector_collision_checks;
        if(entry.value==value){++selector_hits;return entry.bounds;}}
    selectorWork();++selector_new_states;const auto truncated_before=selector_truncated;
    auto result=[&]()->std::optional<SelectorBounds> {
      const auto type=g.type(value);
      if(!type.isScalar() || type.element().isParameterized() || !type.element().isBitLike() ||
         !type.element().width() || type.element().width()>64)return {};
      const auto type_name=type.element().toString();
      if(type_name.empty() || (type_name[0]!='u' && type_name[0]!='b'))return {};
      const auto width=type.element().width();
      const auto width_mask=width==64?UINT64_MAX:((std::uint64_t{1}<<width)-1);
      if(auto fixed=fixedInDomain(value,context,depth+1)) {
        const auto constant=specialization.known(specialization.selected(fixed->value));
        if(!constant)return {};
        const auto number=literal(constant->literal);
        if(!number || *number<0 || static_cast<std::uint64_t>(*number)>width_mask)return {};
        return SelectorBounds{static_cast<std::uint64_t>(*number),static_cast<std::uint64_t>(*number),fixed->end};
      }
      if(value.threadSelector() || value.conversion() || value.pointer() || !value.indices().empty())return {};
      const auto& node=g.node(value.nodeId());
      if(const auto* coordinate=dynamic_cast<const v::CoordinateOp*>(&node.operation())) {
        if(!node.hasUniformThreadTemplate() || coordinate->kind()!=v::CoordinateKind::ThreadIndex ||
           g.threadBindingRelation(value,context)!=v::Graph::ThreadBindingRelation::SameDomain)return {};
        const auto axis=static_cast<unsigned>(coordinate->axis());
        const auto* metadata=coordinateMetadata(value,axis);if(!metadata)return {};
        const auto& tree=*metadata;
        const auto end=std::min(selector_scope_end,tree.count);
        if(selector_scope_begin>=end)return {};
        selectorWork(2*std::bit_width(tree.base));
        const auto bounds=tree.range(selector_scope_begin,end);
        if(bounds.second>width_mask)return {};
        return SelectorBounds{bounds.first,bounds.second,end,&tree};
      }
      const auto* binary=dynamic_cast<const v::BinaryOp*>(&node.operation());
      if(!binary || !node.hasUniformThreadTemplate() ||
         g.threadBindingRelation(value,context)!=v::Graph::ThreadBindingRelation::SameDomain)return {};
      const auto inputs=g.instanceOperands(value);
      if(inputs.size()!=2)return {};
      auto left=selectorBounds(inputs[0],context,depth+1),right=selectorBounds(inputs[1],context,depth+1);
      if(!left || !right)return {};
      if(left->lo==left->hi && right->lo==right->hi) {
        std::ostringstream attributes;binary->printAttributes(attributes);
        const std::vector<equivalence::ConstantValue> constants{
            {g.type(inputs[0]).element().toString(),std::to_string(left->lo)},
            {g.type(inputs[1]).element().toString(),std::to_string(right->lo)}};
        const auto exact=equivalence::constantOperation(std::string(binary->name()),type_name,attributes.str(),constants);
        if(!exact)return {};const auto number=literal(exact->literal);
        if(!number || *number<0 || static_cast<std::uint64_t>(*number)>width_mask)return {};
        return SelectorBounds{static_cast<std::uint64_t>(*number),static_cast<std::uint64_t>(*number),std::min(left->end,right->end)};
      }
      if(binary->opcode()!=v::BinaryOpcode::BitAnd)return {};
      if(left->lo==left->hi)std::swap(left,right);
      if(right->lo!=right->hi)return {};
      const auto variable_bits=std::bit_width(left->lo^left->hi);
      const auto varying_mask=variable_bits==64?UINT64_MAX:((std::uint64_t{1}<<variable_bits)-1);
      auto end=std::min(left->end,right->end);
      if(right->lo&varying_mask) {
        if(!left->coordinates || selector_scope_end==SIZE_MAX)return {};
        std::size_t visits=0;
        end=left->coordinates->prefix(selector_scope_begin,end,right->lo,visits);
        selectorWork(visits);
        if(end<=selector_scope_begin)return {};
      }
      const auto constant=left->lo&right->lo;
      // When the full hull failed, its minimum need not be in the accepted
      // prefix. Query the proven prefix's own hull before choosing its value.
      if(end<left->end && left->coordinates) {
        selectorWork(2*std::bit_width(left->coordinates->base));
        const auto prefix=left->coordinates->range(selector_scope_begin,end);
        return SelectorBounds{prefix.first&right->lo,prefix.first&right->lo,end};
      }
      return SelectorBounds{constant,constant,end};
    }();
    if(result || truncated_before==selector_truncated)selector_facts[key].push_back({value,result});
    return result;
  }
  std::map<const void*,bool> dense_member_domains;
  bool coordinate_metadata_indexed=false;
  std::map<std::uint64_t,bool> linear_x_ctas;
  bool denseDomain(const v::Value& value) {
    if(!coordinate_metadata_indexed) {
      g.forEachThreadBinding([&](const v::SourceOrigin& origin) {
        if(++work>2000000)throw IndexedSourceLimit("indexed source coordinate metadata budget exceeded");
        if(!origin.cta)return;
        auto [it,inserted]=linear_x_ctas.try_emplace(*origin.cta,true);
        if(!origin.thread || !origin.thread_coordinates || origin.thread_coordinates->at(0)!=*origin.thread)
          it->second=false;
      });
      coordinate_metadata_indexed=true;
    }
    const auto cta=g.node(value.nodeId()).origin().cta;
    if(!cta || !linear_x_ctas.contains(*cta) || !linear_x_ctas.at(*cta))return false;
    const auto& members=g.node(value.nodeId()).instanceThreads();
    auto [it,inserted]=dense_member_domains.try_emplace(members.storageIdentity(),true);
    if(inserted)for(std::size_t i=0;i<members.size();++i) {
      if(++work>2000000)throw IndexedSourceLimit("indexed source member-domain proof budget exceeded");
      if(members[i]!=i){it->second=false;break;}
    }
    return it->second;
  }
  std::optional<std::size_t> fixedXorSelector(const v::Value& value) {
    const auto fail=[&](const char* reason)->std::optional<std::size_t> {
      SENTRYWARD_AUDIT(audit->event("SOURCE_SELECTOR_REJECT",reason,value.instanceIndex(),member_count,value.nodeId()));
      return {};
    };
    if(!value.threadSelector() || !value.indices().empty() || value.conversion() || value.pointer())return {};
    const auto selector=*value.threadSelector();
    const auto* op=dynamic_cast<const v::BinaryOp*>(&g.node(selector.nodeId()).operation());
    if(!op || op->opcode()!=v::BinaryOpcode::BitXor)return fail("not_xor");
    const auto operands=g.instanceOperands(selector);
    if(operands.size()!=2)return {};
    for(unsigned side=0;side<2;++side) {
      const auto coordinate=operands[side],constant=operands[1-side];
      const auto* c=dynamic_cast<const v::CoordinateOp*>(&g.node(coordinate.nodeId()).operation());
      const auto* k=dynamic_cast<const v::ConstantOp*>(&g.node(constant.nodeId()).operation());
      if(!c || !k || c->kind()!=v::CoordinateKind::ThreadIndex || c->axis()!=v::Axis::X)continue;
      const auto known=specialization.known(constant);
      const auto mask=known?literal(known->literal):std::optional<std::int64_t>{};
      const auto plain=value.withoutSelection();
      if(!mask || *mask<0 || static_cast<std::uint64_t>(*mask)>UINT32_MAX || !denseDomain(plain) ||
          !std::has_single_bit(member_count) || static_cast<std::size_t>(*mask)>=member_count ||
          g.node(plain.nodeId()).instanceCount()!=member_count ||
          g.threadBindingRelation(coordinate,plain)!=v::Graph::ThreadBindingRelation::SameDomain)return fail("coordinate_domain");
      return static_cast<std::size_t>(*mask);
    }
    return fail("coordinate_or_constant");
  }
  // Operation objects are immutable and owned by g for this matcher's lifetime.
  // Rendering attributes costs O(attribute bytes) once, not once per member pair.
  std::unordered_map<const v::Operation*,std::string> operation_attributes;
  const std::string& attributes(const v::Operation& op) {
    auto [it,inserted]=operation_attributes.try_emplace(&op);
    if(inserted)it->second=attrs(op);
    return it->second;
  }
  struct Equation { __int128 row=0,col=0,value=0; };
  // A two-coordinate affine certificate has at most two basis equations.
  // Copying/refining it in the member loop must not allocate a vector.
  struct GridFit { std::array<Equation,2> basis{};unsigned size=0; };
  std::optional<std::uint64_t> grid_columns;
  // runSlices supplies every stored member, just like matchGrid. Its exact
  // selectors may be resolved per supplied record, but never extrapolated as
  // a template member relation without an independent range certificate.
  bool finite_slice_domain=false;
  std::size_t grid_origin=0;
  std::size_t grid_max_validated=0;
  GridFit grid_prefix;
  std::unordered_map<const Use*,GridFit> grid_fits;
  std::unordered_map<const Use*,GridFit> bit_test_fits;
  // At most two independent coordinate constraints. No per-leaf sample list,
  // no solver, and no expanded matrix is retained by the resulting source.
  static bool constrain(GridFit& fit,Equation e) {
    if(!e.row && !e.col)return !e.value;
    if(!fit.size){fit.basis[fit.size++]=e;return true;}
    const auto& a=fit.basis[0];
    const auto det=a.row*e.col-a.col*e.row;
    if(!det)return a.row*e.value==e.row*a.value && a.col*e.value==e.col*a.value;
    if(fit.size==1){fit.basis[fit.size++]=e;return true;}
    const auto& b=fit.basis[1];
    const auto d=a.row*b.col-a.col*b.row;
    return e.row*(a.value*b.col-a.col*b.value)+e.col*(a.row*b.value-a.value*b.row)==e.value*d;
  }
  static std::optional<std::pair<std::int64_t,std::int64_t>> coefficients(const GridFit& fit) {
    __int128 row=0,col=0;
    if(fit.size==1) {
      const auto& a=fit.basis[0];
      if(a.row && a.value%a.row==0)row=a.value/a.row;
      else if(a.col && a.value%a.col==0)col=a.value/a.col;
      else return {};
    } else if(fit.size==2) {
      const auto& a=fit.basis[0];const auto& b=fit.basis[1];
      const auto det=a.row*b.col-a.col*b.row;
      row=a.value*b.col-a.col*b.value;col=a.row*b.value-a.value*b.row;
      if(row%det || col%det)return {};
      row/=det;col/=det;
    }
    if(row<INT64_MIN || row>INT64_MAX || col<INT64_MIN || col>INT64_MAX)return {};
    return std::pair{static_cast<std::int64_t>(row),static_cast<std::int64_t>(col)};
  }
  Equation gridPoint(std::size_t index,__int128 value=0) const {
    return {__int128(index/ *grid_columns)-__int128(grid_origin/ *grid_columns),
            __int128(index% *grid_columns)-__int128(grid_origin% *grid_columns),value};
  }
  // Immutable interpretation facts have a direct table for unannotated values.
  // The common case needs one hash lookup, not a bucket allocation followed by
  // a separately allocated record. References survive unordered-map rehashes.
  struct Slice {
    ViewSource source;
    std::uint64_t offset;
    // This proof belongs to the direct read, not an arbitrary ancestor path.
    // Wrappers may copy it, but bulk reuse checks the input read identity below.
    const ReadSlicePattern* pattern = nullptr;
    std::size_t read_instance = 0;
    std::uint64_t read_offset = 0;
    const InitialReadFamily* initial_pattern = nullptr;
    std::optional<v::Value> initial_reader = {};
  };
  std::map<MemoryInstance,InitialReadFamily> initial_ranges;
  using InitialDomainKey=std::tuple<std::size_t,std::size_t,std::size_t,std::uint64_t,std::size_t>;
  std::map<InitialDomainKey,IndexedValueBinding> initial_domain_bindings;
  // Substitute i XOR mask in a bounded dyadic index expression. This is
  // algebra on retained coefficients, not a traversal of domain members.
  std::optional<IndexedValueBinding> xorBinding(IndexedValueBinding binding,std::size_t mask) {
    if(!mask)return binding;
    if(binding.period)return {};
    std::map<std::uint64_t,__int128> coefficients;
    __int128 base=binding.base;
    const auto add=[&](std::uint64_t divisor,__int128 coefficient) {
      const auto shifted=mask/divisor;
      base+=coefficient*shifted;
      coefficients[divisor]+=coefficient;
      for(std::uint64_t bit=1;bit && bit<=shifted;bit*=2)if(shifted&bit) {
        if(++work>2000000)throw IndexedSourceLimit("indexed source coordinate composition budget exceeded");
        if(divisor>UINT64_MAX/(2*bit))return false;
        coefficients[divisor*bit]-=2*bit*coefficient;
        coefficients[divisor*bit*2]+=4*bit*coefficient;
      }
      return true;
    };
    if(!add(1,binding.stride))return {};
    for(const auto& q:binding.quotient_terms)
      if(!std::has_single_bit(q.divisor) || !add(q.divisor,q.coefficient))return {};
    if(base<INT64_MIN || base>INT64_MAX)return {};
    binding.base=static_cast<std::int64_t>(base);binding.stride=0;binding.quotient_terms.clear();
    for(const auto& [divisor,coefficient]:coefficients)if(coefficient) {
      if(coefficient<INT64_MIN || coefficient>INT64_MAX)return {};
      if(divisor==1)binding.stride=static_cast<std::int64_t>(coefficient);
      else binding.quotient_terms.push_back({divisor,static_cast<std::int64_t>(coefficient)});
    }
    return binding;
  }
  // Prove a proposed mixed-radix binding against already certified initial
  // read ranges. Formula discovery is not read-from evidence. Each retained
  // range is checked in full; no thread/member or operand ancestor is visited.
  bool coversInitialBinding(const IndexedValueBinding& binding,const Slice& slice,
      const v::Value& target,std::size_t index,std::size_t end) {
    const auto reject=[&](const char* reason,const InitialReadFamily* range=nullptr,
        std::size_t lo=0,std::size_t hi=0) {
      SENTRYWARD_AUDIT({
        (void)audit;
        static std::set<std::tuple<std::size_t,std::string>> shown;
        if(shown.size()<32 && shown.emplace(target.nodeId(),reason).second) {
          SENTRYWARD_DEBUG_PRINT(indexed_source_0);
          for(const auto& q:binding.quotient_terms)SENTRYWARD_DEBUG_PRINT(indexed_source_1);
          if(range)SENTRYWARD_DEBUG_PRINT(indexed_source_2);
          SENTRYWARD_DEBUG_PRINT(indexed_source_3);
        }
      });
      return false;
    };
    if(grid_columns || binding.period || !slice.initial_reader ||
        slice.read_instance!=target.instanceIndex() || target.instanceIndex()<index ||
        !binding.initial || end<=index ||
        g.threadBindingRelation(*slice.initial_reader,target)!=v::Graph::ThreadBindingRelation::SameDomain)
      return reject("domain_precondition");
    const auto origin=target.instanceIndex()-index;
    if(end>SIZE_MAX-origin)return reject("domain_overflow");
    auto cursor=origin;const auto stop=origin+end;
    auto it=initial_ranges.upper_bound({target.nodeId(),cursor});
    if(it==initial_ranges.begin())return reject("no_initial_range");--it;
    while(cursor<stop) {
      if(it==initial_ranges.end())return reject("range_exhausted");
      const auto& range=it->second;
      if(range.first.node!=target.nodeId() || cursor<range.first.instance ||
          cursor-range.first.instance>=range.count ||
          range.initial.region.allocation!=binding.initial->region.allocation ||
          range.initial.type!=binding.initial->type ||
          range.initial.region.bytes!=binding.initial->region.bytes)return reject("coverage_or_source",&range);
      const auto count=std::min(stop-cursor,range.count-(cursor-range.first.instance));
      const auto lo=cursor-origin,hi=lo+count-1;
      if(work>2000000-(binding.quotient_terms.size()+1))
        throw IndexedSourceLimit("indexed source range proof budget exceeded");
      work+=binding.quotient_terms.size()+1;
      // A retained affine range may only contain one cell of each quotient.
      // This conservative test never treats matching endpoints as a proof.
      for(const auto& term:binding.quotient_terms)
        if(!term.divisor || lo/term.divisor!=hi/term.divisor)return reject("quotient_boundary",&range,lo,hi);
      const auto value=indexedBindingValue(binding,lo);
      const auto expected=static_cast<__int128>(range.initial.region.offset)+slice.read_offset+
          static_cast<__int128>(range.byte_stride)*(cursor-range.first.instance);
      if(!value || *value!=expected)return reject("address_formula",&range,lo,hi);
      if(count>1 && binding.stride!=range.byte_stride)return reject("range_stride",&range,lo,hi);
      cursor+=count;++it;
    }
    return true;
  }
  void indexInitialRanges() {
    // Index already materialized read-from records once. Existing compact
    // families stay compact; no family members or expression ancestors are
    // enumerated. Every additional range is checked against all its records.
    initial_ranges=memory.initial_read_families;
    std::optional<InitialReadFamily> pending;
    const auto flush=[&] {
      if(pending)initial_ranges.emplace(pending->first,*pending);
      pending.reset();
    };
    for(const auto& [key,replacement]:memory.reads) {
      const auto initial=initialReplacement(replacement);
      if(!initial){flush();continue;}
      bool extend=false;
      if(pending && pending->first.node==key.node &&
         key.instance>=pending->first.instance &&
         key.instance-pending->first.instance==pending->count &&
         pending->initial.region.allocation==initial->region.allocation &&
         pending->initial.region.bytes==initial->region.bytes && pending->initial.type==initial->type) {
        const auto delta=static_cast<__int128>(initial->region.offset)-pending->initial.region.offset;
        if(pending->count==1 && delta>=INT64_MIN && delta<=INT64_MAX) {
          pending->byte_stride=static_cast<std::int64_t>(delta);extend=true;
        } else extend=delta==static_cast<__int128>(pending->byte_stride)*pending->count;
      }
      if(extend)++pending->count;
      else {flush();pending=InitialReadFamily{key,1,0,*initial};}
    }
    flush();
    // Serialization may split an otherwise identical affine read at action
    // block boundaries. Join those stored ranges directly: a two-member
    // block is not a two-member limit on the source-coordinate proof.
    // This visits range records, never their logical members.
    for(auto it=initial_ranges.begin();it!=initial_ranges.end();) {
      auto next=std::next(it);
      if(next==initial_ranges.end())break;
      auto& a=it->second;const auto& b=next->second;
      bool join=a.count && b.count && a.first.node==b.first.node &&
          b.first.instance>=a.first.instance && b.first.instance-a.first.instance==a.count &&
          b.count<=SIZE_MAX-a.count && a.initial.region.allocation==b.initial.region.allocation &&
          a.initial.region.bytes==b.initial.region.bytes && a.initial.type==b.initial.type;
      const auto delta=static_cast<__int128>(b.initial.region.offset)-a.initial.region.offset;
      auto stride=a.byte_stride;
      if(join && a.count==1) {
        join=delta>=INT64_MIN && delta<=INT64_MAX;
        if(join)stride=static_cast<std::int64_t>(delta);
      }
      join=join && delta==static_cast<__int128>(stride)*a.count &&
          (b.count==1 || b.byte_stride==stride);
      if(join) {a.byte_stride=stride;a.count+=b.count;initial_ranges.erase(next);}
      else ++it;
    }
  }
  struct SliceFact {std::uint64_t offset,width;std::optional<Slice> result;};
  using SliceFacts=std::unordered_map<std::tuple<std::uint64_t,std::uint64_t>,std::optional<Slice>,TupleHash>;
  struct Interpretation {
    v::Value input,canonical;
    std::optional<v::Value> selected;
    std::optional<v::Value> selected_structural;
    Interpretation* selected_info=nullptr;
    Interpretation* selected_structural_info=nullptr;
    const v::ValueType* type=nullptr;
    const v::Node* node=nullptr;
    // 0 unknown, 1 ineligible, 2 eligible. Depends only on this retained full
    // typed use and immutable graph metadata, never on a query coordinate.
    unsigned char template_eligibility=0;
    bool promotion_checked=false;
    std::optional<Bf16PromotionField> promotion;
    bool read_checked=false;
    const v::LoadOp* read=nullptr;
    std::unique_ptr<SliceFact> first_slice;
    std::unique_ptr<SliceFacts> other_slices;
  };
  using InterpretationKey=std::tuple<std::size_t,std::size_t,std::size_t,
      const v::TypeConversion*,const v::Value*,const v::PointerProperties*>;
  std::unordered_map<InterpretationKey,std::vector<std::unique_ptr<Interpretation>>,TupleHash> interpretations;
  std::unordered_map<std::tuple<std::size_t,std::size_t,std::size_t>,Interpretation,TupleHash> plain_interpretations;
  std::size_t interpretation_count=0,interpretation_hits=0,interpretation_misses=0;
  std::size_t template_eligibility_evaluations=0,template_eligibility_hits=0,worklist_fact_reuses=0;
  std::size_t interpretation_discards=0;
  std::size_t stored_requests=0,request_endpoint_lookups=0,prototype_fact_reuses=0;
  std::unordered_map<Ptr,Ptr> one_hot_seeds;
  void beginDomain() {
    uses.clear();
    one_hot_seeds.clear();
    bit_test_fits.clear();
    // Interpretation/selection/type facts depend only on the immutable input,
    // not on this domain's coordinate fit. Reuse them across source domains.
    // Evict only BETWEEN domains: a recursive match borrows stable references.
    // Storage is bounded by this soft limit plus the largest single domain;
    // proof certificates, affine fits and exact-use interning remain local.
    constexpr std::size_t retained_fact_limit=1u<<20;
    if(interpretation_count>retained_fact_limit) {
      interpretation_discards+=interpretation_count;
      SENTRYWARD_AUDIT(audit->add(AuditCounter::cache_clears);
        audit->add(AuditCounter::cache_entries_discarded,interpretation_count));
      interpretations.clear();plain_interpretations.clear();interpretation_count=0;
    }
  }
  Interpretation& interpretation(const v::Value& value) {
    if(value.indices().empty() && !value.threadSelector() && !value.conversion() && !value.pointer()) {
      auto [it,inserted]=plain_interpretations.try_emplace({value.nodeId(),value.instanceIndex(),value.resultIndex()});
      if(inserted){it->second.input=value;it->second.canonical=g.canonicalInterpretation(value);
        ++interpretation_count;++interpretation_misses;
        SENTRYWARD_AUDIT(audit->add(AuditCounter::source_fact_misses));}
      else {++interpretation_hits;SENTRYWARD_AUDIT(audit->add(AuditCounter::source_fact_hits));}
      return it->second;
    }
    auto& bucket=interpretations[{value.nodeId(),value.instanceIndex(),value.resultIndex(),
        value.conversion().get(),value.threadSelector(),value.pointer().get()}];
    for(const auto& item:bucket)if(item->input==value){++interpretation_hits;
      SENTRYWARD_AUDIT(audit->add(AuditCounter::source_fact_hits));return *item;}
    // Full typed handles distinguish component selections and retain their
    // annotation owners. Heap entries remain stable across table/vector growth.
    auto item=std::make_unique<Interpretation>();
    item->input=value;item->canonical=g.canonicalInterpretation(value);
    auto& result=*item;bucket.push_back(std::move(item));
    ++interpretation_count;++interpretation_misses;
    SENTRYWARD_AUDIT(audit->add(AuditCounter::source_fact_misses));return result;
  }
  const v::Value& canonical(const v::Value& value) {return interpretation(value).canonical;}
  bool preserve_data_select=false;
  const v::Value& selected(Interpretation& item) {
    auto& cached=preserve_data_select?item.selected_structural:item.selected;
    if(!cached)cached=specialization.selected(item.canonical,0,preserve_data_select);
    return *cached;
  }
  const v::Value& selected(const v::Value& value) {return selected(interpretation(value));}
  Interpretation& selectedInfo(Interpretation& item) {
    auto& cached=preserve_data_select?item.selected_structural_info:item.selected_info;
    if(!cached) {
      const auto& value=selected(item);
      cached=value==item.canonical?&item:&interpretation(value);
    }
    return *cached;
  }
  const v::ValueType& valueType(Interpretation& item) {
    if(!item.type)item.type=&g.type(item.canonical);
    return *item.type;
  }
  const v::ValueType& valueType(const v::Value& value) {return valueType(interpretation(value));}
  const v::Node& valueNode(Interpretation& item) {
    if(!item.node)item.node=&g.node(item.canonical.nodeId());
    return *item.node;
  }
  const v::Node& valueNode(const v::Value& value) {return valueNode(interpretation(value));}
  const std::optional<Bf16PromotionField>& promotion(Interpretation& item) {
    if(!item.promotion_checked) {
      item.promotion=bf16PromotionField(g,item.canonical,[&](v::Value x)->std::optional<std::int64_t> {
        if(auto c=specialization.known(x))return literal(c->literal);
        return {};
      });
      item.promotion_checked=true;
    }
    return item.promotion;
  }
  // Exact nodes are interned; persistent refinement shares unchanged children.
  // A diamond costs its distinct use edges, not its tree expansion.
  using ValueKey=std::tuple<std::size_t,std::size_t,std::size_t>;
  std::unordered_map<ValueKey,std::vector<Ptr>,TupleHash> exact_nodes;
  std::unordered_map<std::size_t,std::vector<Ptr>> uses;
  struct Certificate {
    Ptr result;
    std::size_t begin,end;
    // A spatial proof is conditional on instance = base + member index.
    // It cannot enter the ordinary exact-Value memo as an unconditional range.
    bool spatial = false;
    // The spatial coordinate can be the outer axis of an interleaved finite
    // source sequence. A certificate then covers this lattice, not every
    // index in [begin,end): instance advances once per index_stride positions.
    std::size_t index_stride = 1;
    // All operands stay fixed (or are themselves proved broadcasts) when this
    // target's thread changes. Unlike a spatial proof, this certificate needs
    // no member-coordinate equation. Do not turn fixed singleton DAGs into
    // coordinate-dependent proofs: that destroys their exact-value reuse.
    bool thread_invariant = false;
  };
  struct Memo { Ptr input; v::Value target; Certificate certificate; };
  // Almost every structural key has one typed target. Store its certificate
  // inline rather than allocating/freeing a one-element vector per visited use.
  // Full-handle collisions remain exact and allocate only when encountered.
  struct MemoBucket {Memo first;std::unique_ptr<std::vector<Memo>> collisions;};
  using MatchKey=std::tuple<const Use*,std::size_t,std::size_t,std::size_t,std::size_t>;
  std::unordered_map<MatchKey,MemoBucket,TupleHash> memo;
  struct TemplateMemo {
    Ptr input;
    v::Value target;
    __int128 instance_base;
    Certificate certificate;
  };
  using TemplateKey=std::tuple<const Use*,std::size_t,std::size_t,std::size_t>;
  std::unordered_map<TemplateKey,TemplateMemo,TupleHash> template_memo;
  struct Request {
    Ptr old;v::Value input;std::size_t index;std::size_t coordinate_xor=0;
    // Interpretation entries are stable across hash-table growth. Eviction is
    // only at beginDomain(), never while this worklist is active. Retaining the
    // exact request handle also retains every annotation owner.
    Interpretation* facts=nullptr;
    Interpretation* prototype_facts=nullptr;
  };
  std::optional<Request> grid_failure_witness;
  unsigned grid_failure_witnesses=0;
  unsigned mapped_admission_witnesses=0;
  unsigned mapped_root_witnesses=0;
  struct MatchResult {Request request;std::optional<Certificate> value;bool done=false,charged=false;};
  using JobKey=std::tuple<const Use*,std::size_t,std::size_t,std::size_t,std::size_t,std::size_t>;
  struct MatchDriver {
    std::unordered_map<JobKey,std::vector<std::size_t>,TupleHash> keys;
    std::vector<MatchResult> results;
    std::optional<Request> needed;
    std::size_t active=0;
  };
  MatchDriver* match_driver=nullptr;
  using OperandKey=std::tuple<std::size_t,std::size_t>;
  std::unordered_map<OperandKey,std::vector<v::Value>,TupleHash> operand_cache;
  std::span<const v::Value> operands(Interpretation& info) {
    const auto& resolved=info.canonical;
    const auto& node=valueNode(info);
    // The stored prototype needs no allocation or SourceOrigin construction.
    if(!resolved.instanceIndex())return node.operands();
    const OperandKey key{node.id(),resolved.instanceIndex()};
    if(const auto found=operand_cache.find(key);found!=operand_cache.end())return found->second;
    // Stored records and collective members are borrowed from the graph;
    // only a rebound template member list is retained here.
    auto list=g.instanceOperands(resolved);
    if(list.borrowed())return list.span();
    return operand_cache.emplace(key,std::move(list).toVector()).first->second;
  }
  void charge(unsigned depth) {
    if(++work>2000000 || depth>128) {
      SENTRYWARD_AUDIT({
        audit->event("SOURCE_MATCH_LIMIT",depth>128?"template_depth":"use_target_work",work,depth);
        const auto end=std::min<unsigned>(depth,128);
        for(unsigned d=end>8?end-8:0;d<=end;++d) {
          const auto& [a,b,index]=match_stack[d];
          const auto& x=g.node(a.nodeId());const auto& y=g.node(b.nodeId());
          SENTRYWARD_DEBUG_PRINT(indexed_source_4);
        }
      });
      throw IndexedSourceLimit("indexed source matching exceeds its work or depth budget");
    }
  }
  Ptr exact(const v::Value& input) {
    const auto& value=canonical(input);
    auto& bucket=exact_nodes[{value.nodeId(),value.instanceIndex(),value.resultIndex()}];
    for(const auto& entry:bucket)if(entry->prototype==value)return entry;
    auto node=std::make_shared<const Use>(Use{value,Kind::Exact,{},{}});
    bucket.push_back(node);return node;
  }
  Ptr make(v::Value value,Kind kind,std::vector<Ptr> children={},
                  std::optional<IndexedValueBinding> binding={}) {
    // Intern immutable refinements when they are constructed. A shared input
    // diamond must not become a tree of newly allocated, identical use nodes.
    // Grid fits carry additional mutable constraints, so retain their existing
    // separate identity; this table is local to one affine domain.
    const auto payload=[&]() -> std::shared_ptr<const IndexedValueBinding> {
      return binding?std::make_shared<const IndexedValueBinding>(std::move(*binding)):nullptr;
    };
    if(grid_columns)return std::make_shared<const Use>(Use{value,kind,payload(),std::move(children)});
    auto hash=TupleHash{}(std::tuple{value.nodeId(),value.instanceIndex(),value.resultIndex(),std::size_t(kind)});
    const auto mix=[&](auto x){hash^=std::hash<decltype(x)>{}(x)+0x9e3779b9+(hash<<6)+(hash>>2);};
    for(const auto& child:children)mix(child.get());
    if(binding){mix(binding->base);mix(binding->stride);mix(binding->period);mix(binding->outer_stride);mix(binding->power_of_two);
      for(const auto& t:binding->quotient_terms){mix(t.divisor);mix(t.coefficient);}}
    const auto sameBinding=[](const auto& a,const auto& b) {
      if(bool(a)!=bool(b))return false;
      if(!a)return true;
      if(a->leaf!=b->leaf || a->base!=b->base || a->stride!=b->stride ||
         a->fixed!=b->fixed || a->producer!=b->producer ||
         a->producer_uses_bindings!=b->producer_uses_bindings || a->period!=b->period ||
         a->outer_stride!=b->outer_stride || a->numerical_field!=b->numerical_field ||
         a->quotient_terms!=b->quotient_terms || a->power_of_two!=b->power_of_two ||
         bool(a->initial)!=bool(b->initial))return false;
      if(!a->initial)return true;
      const auto& x=*a->initial;const auto& y=*b->initial;
      return x.id==y.id && x.region.allocation==y.region.allocation &&
          x.region.offset==y.region.offset && x.region.bytes==y.region.bytes && x.type==y.type;
    };
    auto& bucket=uses[hash];
    for(const auto& old:bucket)
      if(old->prototype==value && old->kind==kind && old->children==children && sameBinding(old->binding,binding))
        return old;
    auto node=std::make_shared<const Use>(Use{value,kind,payload(),std::move(children)});
    bucket.push_back(node);return node;
  }
  std::size_t sampleEnd(const IndexedValueBinding& b,std::size_t index) const {
    if(b.fixed || (!b.stride && !b.outer_stride && b.quotient_terms.empty()))return member_count;
    if(b.stride)return index+1;
    auto span=member_count-index;
    if(b.period && b.outer_stride)span=std::min<std::size_t>(span,b.period-index%b.period);
    for(const auto& t:b.quotient_terms)span=std::min<std::size_t>(span,t.divisor-index%t.divisor);
    return index+span;
  }
  Certificate bindingCertificate(Ptr node,std::size_t index,bool full_fit=true) const {
    if(!full_fit)return {std::move(node),index,index+1};
    const auto& b=*node->binding;
    // One exact target recurs at the same column when the row coefficient is
    // zero. This is an equality of the index formula, not a sampled memory
    // proof. The ordinary memo also requires the full target handle to match.
    if(b.period && (b.stride || b.outer_stride || !b.quotient_terms.empty())) {
      auto period=b.period;
      for(const auto& t:b.quotient_terms)period=std::max(period,t.divisor);
      bool divides=period%b.period==0;
      for(const auto& t:b.quotient_terms)divides=divides && t.divisor && period%t.divisor==0;
      // A common multiple shifts each quotient by a constant. Zero total
      // drift proves this exact congruence class, without enumerating members.
      if(divides && indexedBindingValue(b,period)==std::optional<std::int64_t>(b.base))
        return {std::move(node),index,member_count,false,static_cast<std::size_t>(period)};
    }
    const auto end=sampleEnd(b,index);
    return {std::move(node),index,end};
  }
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
  unsigned attribute_witnesses=0;
  unsigned slice_witnesses=0;
  std::array<unsigned,3> range_witnesses{};
  unsigned binding_witnesses=0;
  std::array<std::tuple<v::Value,v::Value,std::size_t>,129> match_stack{};
#endif
  using SliceKey=std::tuple<std::size_t,std::size_t,std::uint64_t,std::uint64_t>;
  std::unordered_map<SliceKey,std::optional<Slice>,TupleHash> slices;
  std::set<SliceKey> active_slices;
  // The pass's shared extents when its owner lends them; otherwise our own.
  std::unique_ptr<AssembledPieceEnds> own_piece_ends;
  AssembledPieceEnds& piece_ends;
  // Assemblies whose unrepresentable extent this matcher already reported.
  std::set<const AssembledRead*> reported_extents;
  bool rejected([[maybe_unused]] const char* reason,[[maybe_unused]] v::Value a,
                [[maybe_unused]] v::Value b,[[maybe_unused]] std::size_t index) {
    SENTRYWARD_AUDIT(audit->event("SOURCE_PATTERN_REJECTED",reason,index,b.nodeId(),a.nodeId()));
    SENTRYWARD_AUDIT({
      (void)audit;
      if(attribute_witnesses<8 && (std::string_view(reason)=="operation_attributes_mismatch" ||
          std::string_view(reason)=="operation_mismatch")) {
        ++attribute_witnesses;
        const auto& x=valueNode(a).operation();const auto& y=valueNode(b).operation();
        SENTRYWARD_DEBUG_PRINT(indexed_source_5);
      }
    });
    return false;
  }
  static std::optional<Slice> copySlice(const Slice* source) {
    return source?std::optional<Slice>(*source):std::nullopt;
  }
  const Slice* slice(const v::Value& input,std::uint64_t offset,std::uint64_t width,unsigned depth=0) {
    SENTRYWARD_AUDIT(audit->add(AuditCounter::source_slice_queries));
    if(depth>128)return nullptr;
    auto& facts=interpretation(input);
    const auto same=[&](const auto& entry){return entry->offset==offset && entry->width==width;};
    if(facts.first_slice && same(facts.first_slice))
      return facts.first_slice->result?&*facts.first_slice->result:nullptr;
    const auto key=std::tuple{offset,width};
    if(facts.other_slices)if(auto it=facts.other_slices->find(key);it!=facts.other_slices->end())
      return it->second?&*it->second:nullptr;
    auto source=sliceImpl(input,offset,width,depth);
    const Slice* result=nullptr;
    if(!facts.first_slice) {
      facts.first_slice=std::make_unique<SliceFact>(SliceFact{offset,width,std::move(source)});
      if(facts.first_slice->result)result=&*facts.first_slice->result;
    } else {
      // Matrix components can request many different offsets on one producer.
      // Do not linearly scan a per-value list and introduce quadratic work.
      if(!facts.other_slices)facts.other_slices=std::make_unique<SliceFacts>();
      auto [it,inserted]=facts.other_slices->try_emplace(key,std::move(source));
      if(it->second)result=&*it->second;
    }
    // Return a stable borrowed source. Copying its Value/type/allocation on
    // every cache hit would restore an ancestor-sized cost in member matching.
    return result;
  }
  std::optional<Slice> sliceImpl(const v::Value& input,std::uint64_t offset,std::uint64_t width,unsigned depth) {
    if(depth>128)return {};
    const auto& value=selected(input);
    if(value.threadSelector())return {};
    if(!value.indices().empty()) {
      if(value.conversion())return {};
      const auto base=value.withoutSelection();const auto& type=valueType(base);
      if(!type.hasStaticShape() || type.element().isParameterized() || type.element().width()%8 ||
          type.shape().size()!=value.indices().size())return {};
      unsigned __int128 index=0;
      for(std::size_t i=0;i<value.indices().size();++i) {
        const auto coordinate=specialization.known(value.indices()[i]);
        const auto k=coordinate?literal(coordinate->literal):std::nullopt;
        if(!k || *k<0 || std::uint64_t(*k)>=type.shape()[i])return {};
        index=index*type.shape()[i]+*k;
        if(index>UINT64_MAX)return {};
      }
      const auto at=index*(type.element().width()/8)+offset;
      if(at>UINT64_MAX)return {};
      return copySlice(slice(base,static_cast<std::uint64_t>(at),width,depth+1));
    }
    const auto& type=valueType(value);
    if(!type.hasStaticShape() || type.element().isParameterized() || type.bitSize()%8 ||
       offset>type.bitSize()/8 || width>type.bitSize()/8-offset)return {};
    // A register may pack nonadjacent memory elements (e.g. ldmatrix.trans).
    // Follow a proved disjoint bit field, not the packed register's byte index.
    // The decoder rejects arithmetic carries, truncated fields and holes.
    if(type.isScalar() && width<type.bitSize()/8) {
      if(auto fields=packed.borrowComposed(value))for(const auto& field:*fields)
        if(field.value!=value && !(field.start%8) && !(field.width%8) &&
           offset>=field.start/8 && offset-field.start/8<=field.width/8 &&
           width<=field.width/8-(offset-field.start/8))
          return copySlice(slice(field.value,offset-field.start/8,width,depth+1));
    }
    if(value.conversion()) {
      const auto& c=*value.conversion();
      if(c.spec->kind==v::TypeConversionKind::Bitcast &&
         valueType(c.source).bitSize()==type.bitSize())return copySlice(slice(c.source,offset,width,depth+1));
      return Slice{value,offset};
    }
    const auto& definition=valueNode(value);
    const auto* load=dynamic_cast<const v::LoadOp*>(&definition.operation());
    if(!load)return Slice{value,offset};
    if(load->attributes().guarded)return {};
    if(memory.symbolic_initial_reads.contains({value.nodeId(),value.instanceIndex()}))return Slice{value,offset};
    const auto read=memory.canonicalRead({value.nodeId(),value.instanceIndex()});
    const SliceKey key{read.node,read.instance,offset,width};
    if(auto it=slices.find(key);it!=slices.end())return it->second;
    if(!active_slices.insert(key).second)return {};
    const auto from=[&](const ViewSource& source,std::uint64_t at)->std::optional<Slice> {
      if(const auto* v=std::get_if<v::Value>(&source))return copySlice(slice(*v,at,width,depth+1));
      const auto& initial=std::get<InitialMemorySymbol>(source);
      if(at>initial.region.bytes || width>initial.region.bytes-at)return {};
      return Slice{initial,at};
    };
    std::optional<Slice> found;
    if(auto initial=memory.initialRead(read))found=from(*initial,offset);
    else if(const auto* raw=memory.readReplacement(read)) {
      if(const auto* source=std::get_if<v::Value>(raw))found=from(*source,offset);
      else if(const auto* initial=std::get_if<InitialMemorySymbol>(raw))found=from(*initial,offset);
      else if(const auto* part=std::get_if<ReinterpretedRead>(raw)) {
        if(offset<=UINT64_MAX-part->byte_offset)found=from(part->source,offset+part->byte_offset);
      } else {
        const auto& assembly=std::get<AssembledRead>(*raw);
        const auto& extents=piece_ends.extents(assembly);
        // Our first use of an unrepresentable assembly reports its first bad
        // piece: the same bitSize() error, or the extent limit. Later uses
        // search the representable prefix, as after a partial measurement.
        if(extents.stop!=AssembledPieceEnds::Stop::None && reported_extents.insert(&assembly).second) {
          if(extents.stop==AssembledPieceEnds::Stop::Dynamic)(void)assembly.pieces[extents.ends.size()].type.bitSize();
          throw IndexedSourceLimit("assembled source extent overflow", "index_representation_overflow");
        }
        const auto& ends=extents.ends;
        // Different scalar slices must not each rescan the complete producer.
        const auto it=std::upper_bound(ends.begin(),ends.end(),offset);
        if(it!=ends.end() && width<=*it-offset) {
          const auto index=static_cast<std::size_t>(it-ends.begin());
          const auto begin=index?ends[index-1]:0;
          const auto& part=assembly.pieces[index];
          if(offset-begin<=UINT64_MAX-part.byte_offset)found=from(part.source,part.byte_offset+offset-begin);
        }
      }
    }
    // Retain the read-from range proof only if following the source did not
    // change its producer occurrence or byte interpretation.
    const MemoryInstance original_read{value.nodeId(),value.instanceIndex()};
    if(found)if(const auto* initial=std::get_if<InitialMemorySymbol>(&found->source)) {
      auto it=initial_ranges.upper_bound(original_read);
      if(it!=initial_ranges.begin()) {
        const auto& range=std::prev(it)->second;
        if(original_read.node==range.first.node && original_read.instance>=range.first.instance &&
           original_read.instance-range.first.instance<range.count &&
           initial->region.allocation==range.initial.region.allocation) {
          const auto expected=static_cast<__int128>(range.initial.region.offset)+
              static_cast<__int128>(range.byte_stride)*(original_read.instance-range.first.instance)+offset;
          if(expected==static_cast<__int128>(initial->region.offset)+found->offset) {
            found->initial_pattern=&range;found->initial_reader=value;
            found->read_instance=original_read.instance;found->read_offset=offset;
          }
        }
      }
    }
    if(found)if(const auto* pattern=memory.readSlicePattern(original_read,offset,width)) {
      const auto* source=std::get_if<v::Value>(&found->source);
      const auto expected=static_cast<__int128>(pattern->source_offset)+
          static_cast<__int128>(pattern->byte_stride)*(original_read.instance-pattern->first.instance)+
          (offset-pattern->result_offset);
      const auto mapped_instance=static_cast<__int128>(pattern->source.instanceIndex())+
          static_cast<__int128>(pattern->source_instance_stride)*
              (original_read.instance-pattern->first.instance);
      if(source && pattern->sameSourceTemplate(*source) && mapped_instance==source->instanceIndex() &&
         expected>=0 && expected<=UINT64_MAX &&
         found->offset==static_cast<std::uint64_t>(expected)) {
        found->pattern=pattern;found->read_instance=original_read.instance;found->read_offset=offset;
      }
    }
    active_slices.erase(key);slices.emplace(key,found);return found;
  }
  static std::optional<InitialMemorySymbol> initialReplacement(const ReadReplacement& read) {
    if(auto* x=std::get_if<InitialMemorySymbol>(&read))return *x;
    if(auto* x=std::get_if<ReinterpretedRead>(&read))
      if(auto* src=std::get_if<InitialMemorySymbol>(&x->source)) {
        if(!x->type.hasStaticShape() || x->type.element().isParameterized() || x->type.bitSize()%8)return {};
        if(x->byte_offset>src->region.bytes || x->type.bitSize()/8>src->region.bytes-x->byte_offset ||
           src->region.offset>UINT64_MAX-x->byte_offset)return {};
        auto out=*src;out.region.offset+=x->byte_offset;
        out.region.bytes=x->type.bitSize()/8;out.type=x->type;return out;
      }
    return {};
  }
  std::optional<Certificate> bind(const Ptr& old,std::int64_t a,std::int64_t b,
      std::size_t index,const InitialMemorySymbol* initial=nullptr,
      const v::Value* producer=nullptr,std::optional<v::ElementType> field={},bool power_of_two=false) {
    // Equality at this concrete target needs no index substitution. Retain
    // the exact source instead of manufacturing a zero-stride binding and
    // rebuilding every pure ancestor. This is NOT a spatial range proof;
    // later, different target Values still require their own match.
    const bool exact_literal=old->kind==Kind::Exact || old->kind==Kind::OneHotLiteral;
    if(exact_literal && a==b && !field && !producer)
      return Certificate{old,index,member_count};
    const auto sameProducer=[&](const IndexedValueBinding& binding) {
      return bool(binding.producer)==bool(producer) && (!producer || *binding.producer==*producer);
    };
    const auto fresh=[&] {
      IndexedValueBinding binding{old->prototype,a,0,
          initial?std::optional<InitialMemorySymbol>(*initial):std::nullopt,false,
          producer?std::optional<v::Value>(*producer):std::nullopt};
      binding.numerical_field=field;binding.power_of_two=power_of_two;return binding;
    };
    if(grid_columns) {
      auto fit=grid_prefix;
      auto binding=fresh();
      if(old->kind==Kind::Binding) {
        const auto& previous=*old->binding;
        if(bool(previous.initial)!=bool(initial) || !sameProducer(previous) || previous.numerical_field!=field ||
           previous.power_of_two!=power_of_two ||
           (initial && previous.initial->region.allocation!=initial->region.allocation))return {};
        fit=grid_fits.at(old.get());
        binding=previous;
      } else if(!exact_literal)return {};
      const auto quotientDelta=[&]() {
        __int128 delta=0;
        for(const auto& term:binding.quotient_terms)
          delta+=__int128(term.coefficient)*(__int128(index/term.divisor)-__int128(grid_origin/term.divisor));
        return delta;
      };
      auto trial=fit;
      if(!constrain(trial,gridPoint(index,__int128(b)-a-quotientDelta()))) {
        // Discover one existing mixed-radix correction, then validate every
        // later actual record. Its divisor exceeds EVERY visited index, not
        // merely the preceding column-major record; old proofs remain exact.
        const auto divisor=(index/ *grid_columns)* *grid_columns;
        const auto previous=binding.quotient_terms.empty()?*grid_columns:binding.quotient_terms.back().divisor;
        if(!divisor || divisor<=grid_max_validated || divisor<=previous ||
            divisor%previous || !std::has_single_bit(divisor/previous) ||
            binding.quotient_terms.size()>=16)return {};
        const auto predicted=exact_literal?std::optional<std::int64_t>(a):indexedBindingValue(binding,index);
        if(!predicted)return {};
        const auto correction=__int128(b)-*predicted;
        if(correction<INT64_MIN || correction>INT64_MAX)return {};
        binding.quotient_terms.push_back({divisor,static_cast<std::int64_t>(correction)});
        trial=fit;
        if(!constrain(trial,gridPoint(index,__int128(b)-a-quotientDelta())))return {};
      }
      fit=std::move(trial);
      const auto coefficients_=coefficients(fit);if(!coefficients_)return {};
      const auto [row,col]=*coefficients_;
      const auto base=__int128(a)-__int128(row)*(grid_origin/ *grid_columns)-
          __int128(col)*(grid_origin% *grid_columns);
      if(base<INT64_MIN || base>INT64_MAX)return {};
      // Strengthening a coordinate proof need not change its affine expression.
      // Preserve immutable use identity so ancestors and use-scoped memo entries
      // survive. Fits are match-local proof state; Use coefficients stay immutable.
      // Until both axes are constrained, reuse only the current member's proof.
      if(old->kind==Kind::Binding && old->binding->base==base &&
          old->binding->stride==col && old->binding->period==*grid_columns &&
          old->binding->outer_stride==row && old->binding->quotient_terms==binding.quotient_terms) {
        grid_fits.at(old.get())=std::move(fit);
        SENTRYWARD_AUDIT(audit->event("SOURCE_BINDING_REUSE","unchanged_affine_grid_map",1,1));
        return bindingCertificate(old,index,fit.size==2);
      }
      binding.base=static_cast<std::int64_t>(base);binding.stride=col;
      binding.period=*grid_columns;binding.outer_stride=row;
      auto node=make(old->prototype,Kind::Binding,{},binding);
      const auto full_fit=fit.size==2;
      grid_fits.emplace(node.get(),std::move(fit));
      return bindingCertificate(std::move(node),index,full_fit);
    }
    std::optional<IndexedValueBinding> revised;
    if(old->kind==Kind::Binding) {
      const auto& binding=*old->binding;
      if(binding.base!=a || bool(binding.initial)!=bool(initial) || !sameProducer(binding) || binding.numerical_field!=field ||
          binding.power_of_two!=power_of_two ||
          (initial && binding.initial->region.allocation!=initial->region.allocation))return {};
      const auto expected=indexedBindingValue(binding,index);if(!expected)return {};
      if(*expected==b)return bindingCertificate(old,index);
      if(index<2 || index>member_count/2)return {};
      if(binding.period) {
        // Only a bounded dyadic refinement of the existing radix is supported.
        // floor(j/index) is zero for EVERY previously accepted j < index.
        // Reject arbitrary correction tables and validate the remaining input.
        if(index%binding.period || binding.quotient_terms.size()>=16)return {};
        const auto ratio=index/binding.period;
        if(ratio<2 || (ratio&(ratio-1)) ||
           (!binding.quotient_terms.empty() && index<=binding.quotient_terms.back().divisor))return {};
        const auto delta=static_cast<__int128>(b)-*expected;
        if(delta<INT64_MIN || delta>INT64_MAX)return {};
        revised=binding;revised->quotient_terms.push_back({index,static_cast<std::int64_t>(delta)});
      } else {
        const auto outer=static_cast<__int128>(b)-a;
        if(outer<INT64_MIN || outer>INT64_MAX)return {};
        revised=binding;revised->period=index;revised->outer_stride=static_cast<std::int64_t>(outer);
      }
    } else {
      if(!exact_literal)return {};
      revised=fresh();auto& binding=*revised;
      const auto delta=static_cast<__int128>(b)-a;
      if(delta<INT64_MIN || delta>INT64_MAX)return {};
      if(index==1)binding.stride=static_cast<std::int64_t>(delta);
      else if(delta) {
        // This exact use, not every use of its Value, was fixed over [0,index).
        if(index>member_count/2)return {};
        binding.period=index;binding.outer_stride=static_cast<std::int64_t>(delta);
      }
    }
    auto& binding=*revised;
    auto node=make(old->prototype,Kind::Binding,{},binding);
    return bindingCertificate(std::move(node),index);
  }
  const v::LoadOp* readThroughBits(Interpretation& info) {
    if(info.read_checked)return info.read;
    info.read_checked=true;
    const auto* cursor=&info.canonical;
    while(cursor->indices().empty() && !cursor->threadSelector() && cursor->conversion() &&
          cursor->conversion()->spec->kind==v::TypeConversionKind::Bitcast)
      cursor=&canonical(cursor->conversion()->source);
    if(!cursor->indices().empty() || cursor->threadSelector() || cursor->conversion())return nullptr;
    info.read=dynamic_cast<const v::LoadOp*>(&valueNode(*cursor).operation());
    return info.read;
  }
  std::optional<Certificate> forward(const Ptr& old,const v::Value& a,const v::Value& b,
      std::size_t index,unsigned depth,Kind kind) {
    Ptr child;
    if(old->kind==kind)child=old->children.at(0);
    else if(old->kind==Kind::Exact)child=exact(a);
    else if(old->kind==Kind::Binding && old->binding->producer==a &&
        !old->binding->base && !old->binding->stride && !old->binding->outer_stride && old->binding->quotient_terms.empty())
      child=exact(a); // Exact forwarded producer certified for the whole prefix.
    else return {};
    auto result=match(child,b,index,depth+1);
    if(!result)return {};
    auto node=(old->kind==kind || (old->kind==Kind::Exact && !result->spatial)) && result->result==child?old:
        make(old->prototype,kind,{result->result});
    return Certificate{std::move(node),index,result->end,result->spatial,result->index_stride,
        result->thread_invariant};
  }
  std::optional<Certificate> sourceBinding(const Ptr& old,const v::Value& b,std::size_t index,unsigned depth,
      Interpretation& a_info,Interpretation& b_info) {
    const auto& a=old->prototype;
    auto sa=slice(a,0,valueType(a_info).bitSize()/8),sb=slice(b,0,valueType(b_info).bitSize()/8);
    return sourceSlices(old,b,index,depth,sa,sb,{},&a_info,&b_info);
  }
  // Fixed interpretation edges do not change the spatial coordinate. Keep
  // them in the proof key rather than excluding every typed read from reuse.
  // This examines annotations only, never operands or expression ancestors.
  bool fixedProjection(v::Value value,bool bits_only=false) const {
    const auto node=value.nodeId(),instance=value.instanceIndex(),result=value.resultIndex();
    for(unsigned depth=0;depth<32;++depth) {
      if(value.threadSelector() || value.pointer())return false;
      for(const auto& coordinate:value.indices()) {
        if(coordinate.threadSelector() || coordinate.pointer() || coordinate.conversion() ||
           !coordinate.indices().empty() ||
           !dynamic_cast<const v::ConstantOp*>(&g.node(coordinate.nodeId()).operation()))return false;
      }
      if(!value.conversion())return true;
      const auto& conversion=*value.conversion();
      if((bits_only && conversion.spec->kind!=v::TypeConversionKind::Bitcast) ||
         conversion.source.nodeId()!=node || conversion.source.instanceIndex()!=instance ||
         conversion.source.resultIndex()!=result)return false;
      value=conversion.source;
    }
    return false;
  }
  bool sameFixedInterpretation(v::Value a,v::Value b) const {
    for(unsigned depth=0;depth<32;++depth) {
      if(a.nodeId()!=b.nodeId() || a.resultIndex()!=b.resultIndex() ||
         a.indices().size()!=b.indices().size() ||
         !std::equal(a.indices().begin(),a.indices().end(),b.indices().begin()) ||
         bool(a.conversion())!=bool(b.conversion()))return false;
      if(!a.conversion())return true;
      if(*a.conversion()->spec!=*b.conversion()->spec)return false;
      a=a.conversion()->source;b=b.conversion()->source;
    }
    return false;
  }
  bool templateEligible(Interpretation& facts) {
    if(facts.template_eligibility) {
      ++template_eligibility_hits;return facts.template_eligibility==2;
    }
    ++template_eligibility_evaluations;
    const auto& value=facts.canonical;
    const bool eligible=fixedProjection(value) && valueNode(facts).hasUniformThreadTemplate() &&
        valueType(facts).hasStaticShape() && !valueType(facts).element().isParameterized();
    facts.template_eligibility=eligible?2:1;
    return eligible;
  }
  // Common lifting for a certified affine member range. Memory callers
  // supply read-from byte strides; coordinate callers supply exact component
  // strides from immutable provenance. Neither infers a run from samples.
  void extendAffineMemberRange(Certificate& result,const v::Value& target,std::size_t index,
      std::size_t first,std::size_t count,std::int64_t byte_stride) {
    if(!result.result->binding || target.instanceIndex()<first ||
        target.instanceIndex()-first>=count)return;
    const auto& binding=*result.result->binding;
    if(!binding.quotient_terms.empty())return;
    if(grid_columns) {
      const auto fit=grid_fits.find(result.result.get());
      if(fit==grid_fits.end() || fit->second.size!=2)return;
    }
    const auto quotient=[](std::int64_t numerator,std::int64_t denominator)->std::size_t {
      if(!denominator)return 0;
      const auto q=static_cast<__int128>(numerator)/denominator;
      return q>0 && q<=SIZE_MAX && q*denominator==numerator?static_cast<std::size_t>(q):0;
    };
    std::size_t step=0;
    if(!binding.stride && !binding.outer_stride && !byte_stride)step=1;
    else if(!binding.period)step=quotient(byte_stride,binding.stride);
    else {
      const auto rows=quotient(byte_stride,binding.outer_stride);
      if(rows && rows<=SIZE_MAX/binding.period)step=rows*binding.period;
      else if(byte_stride==binding.stride)step=1;
    }
    if(!step)return;
    const auto available=count-(target.instanceIndex()-first),remaining=member_count-index;
    auto extent=available>=(remaining-1)/step+1?remaining:available*step;
    if(binding.period && step==1)extent=std::min<std::size_t>(extent,binding.period-index%binding.period);
    result.end=index+extent;result.spatial=true;result.index_stride=step;
    result.thread_invariant=false;
  }
  std::optional<Certificate> sourceSlices(const Ptr& old,const v::Value& b,std::size_t index,unsigned depth,
      const Slice* sa,const Slice* sb,std::optional<v::ElementType> field={},
      Interpretation* a_facts=nullptr,Interpretation* b_facts=nullptr) {
    const auto& a=old->prototype;
    const auto a_type=[&]() -> const v::ValueType& {return valueType(a_facts?*a_facts:interpretation(a));};
    const auto b_type=[&]() -> const v::ValueType& {return valueType(b_facts?*b_facts:interpretation(b));};
    if(sa && sb && sa->source.index()==sb->source.index()) {
      if(const auto* producer=std::get_if<v::Value>(&sa->source)) {
        const auto& other=std::get<v::Value>(sb->source);
        if(old->kind!=Kind::Forward && *producer==other && sa->offset<=INT64_MAX && sb->offset<=INT64_MAX) {
          auto result=bind(old,sa->offset,sb->offset,index,nullptr,producer,field);
          // The domain proof was built at read resolution from every actual
          // memory state. It certifies a range, not just two samples. Only a
          // direct, unselected read advances this exact instance coordinate;
          // arbitrary selections/packed fields keep their local proof bound.
          // The old prototype is an exact anchor, not a changing member. Only
          // the target domain needs a range proof. Requiring a range for the
          // anchor would reject elected-lane/singleton versus worker domains
          // even though the exact producer and offset were already resolved.
          if(result && !field && sb->pattern && !sb->pattern->source_instance_stride &&
             b.nodeId()==sb->pattern->first.node &&
             fixedProjection(a,true) && fixedProjection(b,true) &&
             sb->read_instance==b.instanceIndex() &&
             result->result->binding && result->result->binding->quotient_terms.empty()) {
            extendAffineMemberRange(*result,b,index,sb->pattern->first.instance,
                sb->pattern->count,sb->pattern->byte_stride);
          }
          SENTRYWARD_AUDIT(if(result && !result->spatial) {
            const auto* reason=!sb->pattern?"missing_read_from_range":
                (!fixedProjection(a,true) || !fixedProjection(b,true))?"nonfixed_projection":
                b.nodeId()!=sb->pattern->first.node?"indirect_read_context":
                field?"packed_numerical_field":grid_columns?"grid_coordinate_scope":
                !result->result->binding?"missing_index_binding":"source_stride_not_certified";
            audit->event("SOURCE_RANGE_PROOF_BOUNDARY",reason,index,1,b.nodeId());
            const auto witness_kind=sb->pattern?0u:valueNode(b).instanceCount()>1?2u:1u;
            if(range_witnesses[witness_kind]<8) {
              ++range_witnesses[witness_kind];
              const auto show=[&](const v::Value& x,const Slice* slice) {
                SENTRYWARD_DEBUG_PRINT(indexed_source_6);
                if(slice->pattern)SENTRYWARD_DEBUG_PRINT(indexed_source_7);
                else {
                  SENTRYWARD_DEBUG_PRINT(indexed_source_8);
                  const auto* direct=memory.readSlicePattern({x.nodeId(),x.instanceIndex()},0,0);
                  if(direct)SENTRYWARD_DEBUG_PRINT(indexed_source_9);
                }
              };
              SENTRYWARD_DEBUG_PRINT(indexed_source_10);
              show(a,sa);SENTRYWARD_DEBUG_PRINT(indexed_source_11);show(b,sb);SENTRYWARD_DEBUG_PRINT(indexed_source_12);
            }
          });
          // A concrete target can reuse its exact index-map certificate. This
          // is NOT a certificate for other members: ordinary memo checks the
          // full Value; matchTemplate additionally requires a spatial or
          // rebinding-invariant proof before accepting an entire thread domain.
          return result;
        }
        if(!field && (old->kind==Kind::Forward || *producer!=a || other!=b) && !sa->offset && !sb->offset &&
           valueType(*producer).isScalar() && valueType(other).isScalar() &&
           valueType(*producer).bitSize()==a_type().bitSize() &&
           valueType(other).bitSize()==b_type().bitSize()) {
          auto result=forward(old,*producer,other,index,depth,Kind::Forward);
          if(result && sb->pattern && sb->pattern->source_instance_stride) {
            // Lift the child's already-derived coordinate map through the
            // retained read-from map. Never rediscover writer membership by
            // comparing all reader instances or walking their ancestors.
            const auto& map=*sb->pattern;
            if(result->spatial && map.source_instance_stride==1 && !map.byte_stride &&
               b.nodeId()==map.first.node && b.instanceIndex()==sb->read_instance &&
               fixedProjection(b,true) && sb->read_offset==map.result_offset) {
              const auto available=map.count-(b.instanceIndex()-map.first.instance);
              const auto step=result->index_stride;
              const auto remaining=member_count-index;
              if(step && available<(remaining-1)/step+1)
                result->end=std::min(result->end,index+available*step);
            } else {
              // A reverse/strided writer map is preserved exactly, but needs
              // explicit index substitution before a child range can be used
              // in reader coordinates. Do not claim that range in its absence.
              result->end=index+1;result->spatial=false;
              result->thread_invariant=false;result->index_stride=1;
            }
          }
          return result;
        }
      } else {
        const auto& ia=std::get<InitialMemorySymbol>(sa->source);
        const auto& ib=std::get<InitialMemorySymbol>(sb->source);
        if(ia.region.allocation==ib.region.allocation && sa->offset<=INT64_MAX && sb->offset<=INT64_MAX &&
           ia.region.offset<=INT64_MAX-sa->offset && ib.region.offset<=INT64_MAX-sb->offset)
        {
          if(!grid_columns && sb->initial_reader && (b.instanceIndex()^coordinate_xor)>=index &&
              g.threadBindingRelation(*sb->initial_reader,b)==v::Graph::ThreadBindingRelation::SameDomain) {
            const auto unmapped=b.instanceIndex()^coordinate_xor;
            if(unmapped<index)return {};
            const auto origin=unmapped-index;
            if(coordinate_xor && (origin || !std::has_single_bit(member_count) || coordinate_xor>=member_count))return {};
            const InitialDomainKey domain_key{b.nodeId(),origin,member_count,sb->read_offset,coordinate_xor};
            if(const auto cached=initial_domain_bindings.find(domain_key);cached!=initial_domain_bindings.end() &&
                cached->second.base==static_cast<__int128>(ia.region.offset)+sa->offset &&
                cached->second.initial->region.allocation==ia.region.allocation &&
                cached->second.initial->region.bytes==ia.region.bytes && cached->second.initial->type==ia.type) {
              auto binding=cached->second;binding.leaf=old->prototype;binding.numerical_field=field;
              return Certificate{make(a,Kind::Binding,{},std::move(binding)),index,member_count,true,1,false};
            }
            const auto at=[&](std::size_t member)->std::optional<std::int64_t> {
              const std::size_t cost=std::bit_width(initial_ranges.size()+1)+1;
              if(cost>2000000-work)throw IndexedSourceLimit("indexed source range proof budget exceeded");
              work+=cost;
              if(member>SIZE_MAX-origin)return {};
              const auto mapped=(origin+member)^coordinate_xor;
              auto it=initial_ranges.upper_bound({b.nodeId(),mapped});
              if(it==initial_ranges.begin())return {};const auto& r=std::prev(it)->second;
              if(r.first.node!=b.nodeId() || mapped<r.first.instance ||
                  mapped-r.first.instance>=r.count ||
                  r.initial.region.allocation!=ia.region.allocation)return {};
              const auto value=static_cast<__int128>(r.initial.region.offset)+sb->read_offset+
                  static_cast<__int128>(r.byte_stride)*(mapped-r.first.instance);
              if(value<INT64_MIN || value>INT64_MAX)return {};
              return static_cast<std::int64_t>(value);
            };
            auto base=at(0),one=at(1);
            if(base && one && *base==static_cast<__int128>(ia.region.offset)+sa->offset) {
              const auto delta=static_cast<__int128>(*one)-*base;
              if(delta>=INT64_MIN && delta<=INT64_MAX) {
                IndexedValueBinding binding{old->prototype,*base,static_cast<std::int64_t>(delta),ia,false};
                binding.numerical_field=field;bool valid=true;
                // Discover a bounded radix candidate, then prove it over ALL
                // stored initial ranges below. These points are not a proof.
                for(std::size_t radix=2;radix<member_count;) {
                  auto actual=at(radix),predicted=indexedBindingValue(binding,radix);
                  if(!actual || !predicted){valid=false;break;}
                  const auto correction=static_cast<__int128>(*actual)-*predicted;
                  if(correction<INT64_MIN || correction>INT64_MAX || binding.quotient_terms.size()>=16) {
                    valid=false;break;
                  }
                  if(correction)binding.quotient_terms.push_back({radix,static_cast<std::int64_t>(correction)});
                  if(radix>SIZE_MAX/2)break;radix*=2;
                }
                auto native_binding=xorBinding(binding,coordinate_xor);
                auto native_target=g.node(b.nodeId()).result(b.resultIndex(),unmapped);
                auto native_slice=*sb;
                native_slice.initial_reader=native_target;native_slice.read_instance=unmapped;
                if(valid && native_binding && coversInitialBinding(*native_binding,native_slice,native_target,index,member_count)) {
                  initial_domain_bindings.emplace(domain_key,binding);
                  return Certificate{make(a,Kind::Binding,{},std::move(binding)),index,member_count,true,1,false};
                }
              }
            }
          }
          auto result=bind(old,ia.region.offset+sa->offset,ib.region.offset+sb->offset,index,&ia,nullptr,field);
          // A zero-stride initial range proves the full target read template
          // is a broadcast even when bind() quite correctly kept an Exact
          // use instead of manufacturing an index binding. A partial range
          // proves no such rebinding invariant.
          if(result && result->result->kind==Kind::Exact && !field && sb->initial_pattern && sb->initial_reader &&
             sb->read_instance==b.instanceIndex() && fixedProjection(b,true)) {
            const auto& range=*sb->initial_pattern;
            const auto& target_node=g.node(b.nodeId());
            const auto bytes=b_type().bitSize()/8;
            if(target_node.hasUniformThreadTemplate() &&
               range.first.node==b.nodeId() && range.first.instance==0 &&
               range.count==target_node.instanceCount() && range.byte_stride==0 &&
               b_type().bitSize()!=0 && b_type().bitSize()%8==0 &&
               sb->read_offset<=range.initial.region.bytes &&
               bytes<=range.initial.region.bytes-sb->read_offset &&
               g.threadBindingRelation(*sb->initial_reader,b)==v::Graph::ThreadBindingRelation::SameDomain)
              result->thread_invariant=true;
          }
          if(result && result->result->binding && sb->initial_pattern && sb->initial_reader &&
             sb->read_instance==b.instanceIndex() && fixedProjection(b) &&
             g.threadBindingRelation(*sb->initial_reader,b)==v::Graph::ThreadBindingRelation::SameDomain) {
            const auto& range=*sb->initial_pattern;
            extendAffineMemberRange(*result,b,index,range.first.instance,range.count,range.byte_stride);
          }
          return result;
        }
      }
    }
    rejected(sa && sb?"distinct_computed_sources_or_nonaffine_binding":"source_slice_unresolved",a,b,index);
    return {};
  }
  std::optional<Certificate> match(const Ptr& old,const v::Value& input,std::size_t index,unsigned depth=0,
      Interpretation* retained_facts=nullptr,Interpretation* prototype_facts=nullptr) {
    // Evaluate dependency pairs bottom-up. A long stored reduction chain is
    // not an invitation to recurse on the C++ stack or increase a depth cap.
    // Results (including failures) are local to this exact member query.
    // The pointer exists only during one outer match; recursive calls borrow
    // it. Matcher instances are thread-confined, so no global state is used.
    const auto jobKey=[](const Request& r){return JobKey{r.old.get(),r.input.nodeId(),
        r.input.instanceIndex(),r.input.resultIndex(),r.index,r.coordinate_xor};};
    const auto lookup=[&](MatchDriver& d,const Request& r)->std::optional<std::size_t> {
      auto found=d.keys.find(jobKey(r));if(found==d.keys.end())return {};
      for(auto id:found->second)if(d.results[id].request.input==r.input)return id;
      return {};
    };
    if(!match_driver) {
      MatchDriver driver;match_driver=&driver;
      struct Reset {MatchDriver*& p;~Reset(){p=nullptr;}} reset{match_driver};
      struct RestoreCoordinate {std::size_t& value;std::size_t saved;~RestoreCoordinate(){value=saved;}}
        restore_coordinate{coordinate_xor,coordinate_xor};
      const auto add=[&](Request r) {
        if(driver.results.size()>=2000000)throw IndexedSourceLimit("indexed source worklist budget exceeded");
        ++stored_requests;
        if(!r.facts){++request_endpoint_lookups;r.facts=&interpretation(r.input);}
        if(!r.prototype_facts){++request_endpoint_lookups;r.prototype_facts=&interpretation(r.old->prototype);}
        auto id=driver.results.size();driver.keys[jobKey(r)].push_back(id);
        driver.results.push_back({std::move(r),{},false});return id;
      };
      std::vector<std::size_t> pending{add({old,input,index,coordinate_xor})};
      while(!pending.empty()) {
        const auto id=pending.back();
        // Request copy keeps typed handles alive across vector growth.
        auto request=driver.results[id].request;
        driver.active=id;
        coordinate_xor=request.coordinate_xor;
        auto value=match(request.old,request.input,request.index,0,request.facts,request.prototype_facts);
        if(driver.needed) {
          auto needed=std::move(*driver.needed);driver.needed.reset();
          pending.push_back(add(std::move(needed)));
        } else {
          driver.results[id].value=std::move(value);driver.results[id].done=true;
          pending.pop_back();
        }
      }
      return driver.results[0].value;
    }
    SENTRYWARD_AUDIT((void)audit;match_stack[std::min<unsigned>(depth,128)]={old->prototype,input,index});
    if(retained_facts)++worklist_fact_reuses;
    auto& input_info=retained_facts?*retained_facts:interpretation(input);
    const auto& b=input_info.canonical;
    const bool template_target=templateEligible(input_info);
    const TemplateKey template_key{old.get(),b.nodeId(),b.resultIndex(),coordinate_xor};
    if(template_target)if(const auto it=template_memo.find(template_key);it!=template_memo.end()) {
      const auto& entry=it->second;
      if(index>=entry.certificate.begin && index<entry.certificate.end &&
         index%entry.certificate.index_stride==entry.certificate.begin%entry.certificate.index_stride &&
         (entry.certificate.thread_invariant ||
          static_cast<__int128>(b.instanceIndex()^coordinate_xor)==entry.instance_base+index/entry.certificate.index_stride) &&
         sameFixedInterpretation(entry.target,b)) {
        SENTRYWARD_AUDIT(audit->add(AuditCounter::source_template_proof_hits));
        return entry.certificate;
      }
    }
    const MatchKey key{old.get(),b.nodeId(),b.instanceIndex(),b.resultIndex(),coordinate_xor};
    auto found=memo.find(key);
    const auto covers=[&](const Memo& entry) {
      return entry.target==b && (entry.certificate.spatial?index==entry.certificate.begin:
          index>=entry.certificate.begin && index<entry.certificate.end &&
          index%entry.certificate.index_stride==entry.certificate.begin%entry.certificate.index_stride);
    };
    const Memo* cached=nullptr;
    if(found!=memo.end()) {
      if(covers(found->second.first))cached=&found->second.first;
      else if(found->second.collisions)for(const auto& entry:*found->second.collisions)
        if(covers(entry)){cached=&entry;break;}
    }
    if(cached) {
        SENTRYWARD_AUDIT(audit->event("SOURCE_PROOF_REUSE","immutable_use_and_member_interval",1,1));
        return cached->certificate;
    }
    // Cache hits must not allocate worklist records. Only a genuinely unknown
    // dependency suspends its parent; no ancestor recursion occurs here.
    if(match_driver && depth) {
      auto& driver=*match_driver;Request request{old,input,index,coordinate_xor,&input_info};
      if(auto id=lookup(driver,request)) {
        if(!driver.results[*id].done)throw IndexedSourceLimit("cyclic indexed source dependency", "source_dependency_cycle");
        return driver.results[*id].value;
      }
      driver.needed=std::move(request);return {};
    }
    SENTRYWARD_AUDIT(audit->add(AuditCounter::source_pair_visits));
    // The budget counts matching states, as in recursive evaluation. Resuming
    // the same state after a child completes is not a new dependency pair.
    // Actual resumed visits remain visible in source_pair_visits diagnostics.
    auto& job=match_driver->results[match_driver->active];
    if(!job.charged){charge(depth);job.charged=true;}
    if(prototype_facts)++prototype_fact_reuses;
    auto& prototype_info=prototype_facts?*prototype_facts:interpretation(old->prototype);
    auto result=matchImpl(old,prototype_info,input_info,index,depth);
    SENTRYWARD_AUDIT((void)audit;if(!result && !match_driver->needed && !grid_failure_witness)
      grid_failure_witness=Request{old,input,index,coordinate_xor};);
    if(result) {
      if(template_target && (result->spatial || result->thread_invariant) && result->end>index+1)
        template_memo.insert_or_assign(template_key,TemplateMemo{old,b,
            static_cast<__int128>(b.instanceIndex()^coordinate_xor)-index/result->index_stride,*result});
      // Retain the input pointer as well as the result: raw addresses in the
      // cache must not be recycled while their certificates are still live.
      // Recursive child matches may rehash memo, so obtain the slot anew.
      found=memo.find(key);
      // A conditional range may be shared by DAG parents at THIS coordinate.
      // covers() forbids reusing it at a different coordinate merely because
      // the concrete Value repeats; template_memo checks the instance map.
      if(found==memo.end())memo.emplace(key,MemoBucket{{old,b,*result},{}});
      else {
        auto& bucket=found->second;
        if(bucket.first.target==b)bucket.first.certificate=*result;
        else {
          if(!bucket.collisions)bucket.collisions=std::make_unique<std::vector<Memo>>();
          auto& entries=*bucket.collisions;
          auto entry=std::find_if(entries.begin(),entries.end(),[&](const auto& x){return x.target==b;});
          if(entry==entries.end())entries.push_back({old,b,*result});
          else entry->certificate=*result;
        }
      }
    }
    return result;
  }
  std::optional<Certificate> matchImpl(const Ptr& old,Interpretation& a_info,Interpretation& input_info,std::size_t index,unsigned depth) {
    const auto& input=input_info.canonical;
    const auto& a=old->prototype;
    // Equality of the same annotated value holds throughout the domain. If
    // this use later needs a binding, grid_prefix supplies every earlier
    // zero-difference constraint, including cached coordinates.
    if(old->kind==Kind::Exact && a==input)return Certificate{old,index,member_count};
    // Borrow each full typed fact once for this visit; asking for its scalar
    // fields must not perform another hash lookup/type-edge traversal.
    const auto& selected_a=selected(a_info);const auto& selected_b=selected(input_info);
    if(a.threadSelector() || input.threadSelector()) {
      if(grid_columns || finite_slice_domain) {
        // This is an already stored finite record domain, not the source
        // thread domain. Resolve each actual selector with its annotations,
        // then prove the resulting source coordinates against the whole grid.
        // No instance-range certificate is inferred from this one selection.
        if(selected_a.threadSelector() || selected_b.threadSelector())return {};
        auto result=forward(old,selected_a,selected_b,index,depth,Kind::Selected);
        if(result && result->spatial) {
          // Spatial proofs predict other target members. A nonspatial proof
          // only repeats THIS exact annotated target at certified coordinates;
          // ordinary memo checks that full target and can preserve its domain.
          result->begin=index;result->end=index+1;result->spatial=false;
          result->index_stride=1;result->thread_invariant=false;
        }
        if(result)result->thread_invariant=false;
        SENTRYWARD_AUDIT(if(result && !result->spatial && result->end>index+1)
          audit->event("SOURCE_GRID_SELECTOR_REUSE","exact_target_coordinate_domain",index,result->end,result->index_stride));
        return result;
      }
      const auto am=fixedXorSelector(a),bm=fixedXorSelector(input);
      if(!am || !bm || *am!=*bm)return {};
      const auto saved=coordinate_xor;coordinate_xor^=*am;
      struct Restore {std::size_t& value;std::size_t saved;~Restore(){value=saved;}} restore{coordinate_xor,saved};
      auto result=forward(old,selected_a,selected_b,index,depth,Kind::Selected);
      SENTRYWARD_AUDIT(if(!result && !match_driver->needed)
        audit->event("SOURCE_SELECTOR_REJECT","selected_child",index,coordinate_xor,a.nodeId()));
      return result;
    }
    if(old->kind==Kind::Selected || selected_a!=a)
      return forward(old,selected_a,selected_b,index,depth,Kind::Selected);
    const auto& b=selected_b;
    auto& b_info=selectedInfo(input_info);
    if(a.threadSelector() || b.threadSelector())return {};
    const auto& a_type=valueType(a_info);const auto& b_type=valueType(b_info);
    if(a_type!=b_type || a.resultIndex()!=b.resultIndex() ||
       bool(a.pointer())!=bool(b.pointer()) || (a.pointer() && *a.pointer()!=*b.pointer()))return {};
    const auto& pa=promotion(a_info);const auto& pb=promotion(b_info);
    if(pa && pb)return sourceSlices(old,b,index,depth,
        slice(pa->word,pa->byte_offset,2),slice(pb->word,pb->byte_offset,2),v::ElementType::floating(16,"bf16"),&a_info,&b_info);
    if(!a.indices().empty() || !b.indices().empty())return sourceBinding(old,b,index,depth,a_info,b_info);
    const auto* ar=readThroughBits(a_info);const auto* br=readThroughBits(b_info);
    if((ar && ar->attributes().guarded) || (br && br->attributes().guarded))return {};
    const auto ai=memory.symbolic_initial_reads.find({a.nodeId(),a.instanceIndex()});
    const auto bi=memory.symbolic_initial_reads.find({b.nodeId(),b.instanceIndex()});
    const bool initial_loads=ar && br && ai!=memory.symbolic_initial_reads.end() &&
        bi!=memory.symbolic_initial_reads.end() && ai->second==bi->second;
    if(!initial_loads && (ar || br || old->kind==Kind::Forward))return sourceBinding(old,b,index,depth,a_info,b_info);
    if(bool(a.conversion())!=bool(b.conversion()))return {};
    if(a.conversion()) {
      if(*a.conversion()->spec!=*b.conversion()->spec)return {};
      return forward(old,a.conversion()->source,b.conversion()->source,index,depth,Kind::Conversion);
    }
    const auto& a_node=valueNode(a_info);const auto& b_node=valueNode(b_info);
    const auto& x=a_node.operation();const auto& y=b_node.operation();
    const bool ideal_exp2=isIdealExp2(x) && isIdealExp2(y);
    if((!ideal_exp2 && (typeid(x)!=typeid(y) || x.name()!=y.name())) ||
       a_node.resultCount()!=b_node.resultCount())return {};
    if(const auto* p=dynamic_cast<const v::PtxScalarOp*>(&x);
       p && !ideal_exp2 && p->sourceType()!=dynamic_cast<const v::PtxScalarOp&>(y).sourceType())return {};
    if(const auto* ca=dynamic_cast<const v::ConstantOp*>(&x)) {
      const auto& cb=dynamic_cast<const v::ConstantOp&>(y);
      if(ca->pointer()!=cb.pointer())return {};
      // A uniform literal is member-independent, including integer/bit-like
      // literals. Do not route equality through affine fitting and lose that
      // fact before the template memo can consume it.
      if(old->kind==Kind::Exact && ca->literal()==cb.literal())
        return Certificate{old,index,member_count,false,1,b_node.hasUniformThreadTemplate()};
      const auto& t=a_type;
      if(t.isScalar() && t.element().isBitLike() && !t.element().isParameterized() &&
         t.element().width()<=64) {
        auto av=literal(ca->literal()),bv=literal(cb.literal());
        if(av && bv) {
          // One-hot masks vary exponentially in value but linearly in their
          // bit coordinate. Validate each actual literal and its bit width;
          // this is an exact scalar map, never a fitted address transform.
          if(grid_columns && (old->kind==Kind::OneHotLiteral ||
              (old->kind==Kind::Binding && old->binding->power_of_two)) && *av>0 && *bv>0 &&
             std::has_single_bit(std::uint64_t(*av)) && std::has_single_bit(std::uint64_t(*bv)) &&
             unsigned(std::bit_width(std::uint64_t(*av)))<=t.element().width() &&
             unsigned(std::bit_width(std::uint64_t(*bv)))<=t.element().width())
            return bind(old,std::countr_zero(std::uint64_t(*av)),std::countr_zero(std::uint64_t(*bv)),
                        index,nullptr,nullptr,{},true);
          return bind(old,*av,*bv,index);
        }
      }
      if(ca->literal()==cb.literal())return Certificate{old,index,member_count,false,1,
          b_node.hasUniformThreadTemplate()};
      return {};
    }
    if(!ideal_exp2 && attributes(x)!=attributes(y))return {};
    if(const auto* coordinate=dynamic_cast<const v::CoordinateOp*>(&x)) {
      auto av=a_node.instanceOrigin(a.instanceIndex()).coordinate_value,
           bv=b_node.instanceOrigin(b.instanceIndex()).coordinate_value;
      if(av && bv && *av<=INT64_MAX && *bv<=INT64_MAX) {
        auto result=bind(old,*av,*bv,index);
        // Graph's uniform thread rebinding copies CTA/cluster coordinates;
        // only ThreadIndex is recomputed from the new member. This proves
        // invariance of this target node, not equality across CTA occurrences.
        const bool cta_invariant=coordinate->kind()==v::CoordinateKind::BlockIndex ||
            coordinate->kind()==v::CoordinateKind::ClusterBlockIndex ||
            coordinate->kind()==v::CoordinateKind::ClusterRank;
        if(result && cta_invariant &&
            b_node.hasUniformThreadTemplate())result->thread_invariant=true;
        if(result && coordinate->kind()==v::CoordinateKind::ThreadIndex && fixedProjection(b)) {
          const auto* metadata=coordinateMetadata(b,static_cast<unsigned>(coordinate->axis()));
          const auto width=b_type.element().width();
          const auto type_name=b_type.element().toString();
          const bool unsigned_type=!type_name.empty() && (type_name[0]=='u' || type_name[0]=='b');
          const auto maximum=width && width<64?((std::uint64_t{1}<<width)-1):UINT64_MAX;
          if(metadata && metadata->count==b_node.instanceCount() &&
             unsigned_type && width && width<=64 && metadata->nodes[1].second<=maximum &&
             metadata->nodes[metadata->base+b.instanceIndex()].first==*bv) {
            selectorWork(std::bit_width(metadata->runs.size()+1));
            auto run=std::upper_bound(metadata->runs.begin(),metadata->runs.end(),b.instanceIndex(),
                [](std::size_t member,const auto& entry){return member<entry.first;});
            if(run!=metadata->runs.begin()) {
              --run;
              if(run->count>1 && b.instanceIndex()-run->first<run->count) {
                extendAffineMemberRange(*result,b,index,run->first,run->count,run->stride);
                if(run->first==0 && run->count==metadata->count && run->stride==0)
                  result->thread_invariant=true;
              }
            }
          }
        }
        return result;
      }
      return {};
    }
    if(dynamic_cast<const v::InputParameterOp*>(&x))return Certificate{old,index,member_count,false,1,
        b_node.hasUniformThreadTemplate()};
    if(!admittedPure(x) && !initial_loads)return {};
    struct Conditional {
      std::array<v::Value,3> arguments;
      std::optional<unsigned> bit;
      bool bit_template=false;
    };
    const auto nonzeroSelect=[&](Interpretation& info)->std::optional<Conditional> {
      if(!grid_columns || !dynamic_cast<const v::SelectOp*>(&valueNode(info).operation()))return {};
      const auto args=operands(info);if(args.size()!=3)return {};
      const auto predicate=args[0];
      if(predicate.conversion() || predicate.threadSelector() || predicate.pointer() || !predicate.indices().empty())return {};
      auto& pi=interpretation(predicate);
      const auto* compare=dynamic_cast<const v::CompareOp*>(&valueNode(pi).operation());
      if(!compare || compare->flushToZero() || compare->unorderedTrue())return {};
      const auto tests=operands(pi);if(tests.size()!=2)return {};
      // A constant bit in one concrete member is not a template proof.
      // Inspect only the bounded Compare/BitAnd skeleton: Fixed retains that
      // exact occurrence; SameDomain needs immutable uniform operand bindings.
      // constant() below admits only immutable literals plus conversion edges,
      // not coordinate/arithmetic evaluation. Thus the omitted boundary and
      // one-hot operands cannot vary under these uniform bindings. A broader
      // constant evaluator would require proving those operands invariant too.
      const auto templateEdge=[&](v::Value child,v::Value parent) {
        const auto relation=g.threadBindingRelation(child,parent);
        return relation==v::Graph::ThreadBindingRelation::Fixed ||
            (relation==v::Graph::ThreadBindingRelation::SameDomain &&
             valueNode(child).hasUniformThreadTemplate());
      };
      const bool predicate_template=templateEdge(predicate,info.canonical);
      const auto bitsSource=[&](v::Value value) {
        // Bit extraction ignores interpretation, not numeric conversion.
        while(value.conversion() && value.conversion()->spec->kind==v::TypeConversionKind::Bitcast &&
              value.indices().empty() && !value.threadSelector() && !value.pointer() &&
              valueType(value).isScalar() && valueType(value.conversion()->source).isScalar() &&
              valueType(value).bitSize()==valueType(value.conversion()->source).bitSize())
          value=value.conversion()->source;
        return value;
      };
      for(unsigned side=0;side<2;++side) {
        const auto& type=valueType(tests[side]);
        if(!type.isScalar() || type.element().isParameterized() || !type.element().isBitLike() ||
           !type.element().width() || type.element().width()>64 || valueType(tests[1-side])!=type)continue;
        const auto boundary=constant(tests[1-side]);if(!boundary)continue;
        auto predicate=compare->predicate();
        using P=v::ComparePredicate;
        if(side) {
          if(predicate==P::Greater)predicate=P::Less;
          else if(predicate==P::GreaterEqual)predicate=P::LessEqual;
          else if(predicate==P::Less)predicate=P::Greater;
          else if(predicate==P::LessEqual)predicate=P::GreaterEqual;
        }
        if(boundary->literal=="0" && (predicate==P::Equal || predicate==P::NotEqual)) {
          const bool equal=predicate==P::Equal;
          Conditional result{{tests[side],args[equal?2:1],args[equal?1:2]},std::nullopt};
          auto test=bitsSource(tests[side]);
          if(!test.conversion() && test.indices().empty() && !test.threadSelector() && !test.pointer()) {
            auto& ti=interpretation(test);
            const auto* operation=dynamic_cast<const v::BinaryOp*>(&valueNode(ti).operation());
            if(operation && operation->opcode()==v::BinaryOpcode::BitAnd) {
              const auto fields=operands(ti);
              for(unsigned field=0;fields.size()==2 && field<2;++field) {
                const auto c=constant(fields[field]);if(!c)continue;
                const auto value=literal(c->literal);if(!value || *value<=0)continue;
                const auto bits=std::uint64_t(*value);
                if(!std::has_single_bit(bits) || unsigned(std::bit_width(bits))>type.element().width())continue;
                result.arguments[0]=bitsSource(fields[1-field]);
                result.bit=unsigned(std::countr_zero(bits));
                result.bit_template=predicate_template && templateEdge(test,pi.canonical);
                return result;
              }
            }
          }
          return result;
        }
        if(type.element().kind()!=v::ElementKind::SignedInteger)continue;
        const bool sign_clear=(predicate==P::GreaterEqual && boundary->literal=="0") ||
            (predicate==P::Greater && boundary->literal=="-1");
        const bool sign_set=(predicate==P::Less && boundary->literal=="0") ||
            (predicate==P::LessEqual && boundary->literal=="-1");
        if(sign_clear || sign_set)return Conditional{
            {bitsSource(tests[side]),args[sign_set?1:2],args[sign_set?2:1]},
            type.element().width()-1,predicate_template};
      }
      return {};
    };
    const auto canonical_a=nonzeroSelect(a_info),canonical_b=nonzeroSelect(b_info);
    const bool canonical_select=canonical_a && canonical_b;
    if(bool(canonical_a)!=bool(canonical_b))return {};
    const bool bit_select=canonical_select && canonical_a->bit.has_value();
    if(canonical_select && bool(canonical_a->bit)!=bool(canonical_b->bit))return {};
    const auto result_kind=bit_select?Kind::BitTestSelect:canonical_select?Kind::NonzeroSelect:Kind::Operands;
    const std::span<const v::Value> aa=canonical_select?std::span<const v::Value>(canonical_a->arguments):operands(a_info);
    const std::span<const v::Value> bb=canonical_select?std::span<const v::Value>(canonical_b->arguments):operands(b_info);
    if(aa.size()!=bb.size() || (old->kind!=Kind::Exact && old->kind!=result_kind))return {};
    std::optional<IndexedValueBinding> bit_map;
    GridFit bit_fit;
    bool same_bit_map=true;
    if(bit_select) {
      bit_fit=old->kind==Kind::BitTestSelect?bit_test_fits.at(old.get()):grid_prefix;
      if(!constrain(bit_fit,gridPoint(index,__int128(*canonical_b->bit)-*canonical_a->bit)))return {};
      const auto fit=coefficients(bit_fit);if(!fit)return {};
      const auto [row,col]=*fit;
      const auto base=__int128(*canonical_a->bit)-__int128(row)*(grid_origin/ *grid_columns)-
          __int128(col)*(grid_origin% *grid_columns);
      if(base<INT64_MIN || base>INT64_MAX)return {};
      bit_map=IndexedValueBinding{aa[0],static_cast<std::int64_t>(base),col,std::nullopt,false};
      bit_map->period=*grid_columns;bit_map->outer_stride=row;
      same_bit_map=old->kind==Kind::BitTestSelect && old->binding->base==base &&
          old->binding->stride==col && old->binding->period==*grid_columns && old->binding->outer_stride==row;
    }
    std::vector<Ptr> children(aa.size());
    auto end=member_count;bool unchanged=(!canonical_select || old->kind==result_kind) && same_bit_map,spatial=false;
    std::size_t index_stride=0;
    bool template_proved=b_node.hasUniformThreadTemplate() && b_node.instanceCount()>1;
    bool unproved_spatial=false;
    std::vector<v::Value> scoped_selector_children;
    // Match the newly supplied terms before an accumulator's long prefix.
    // A rejected local binding then stops this match without walking the
    // entire reduction history. Operand slots themselves are never reordered.
    for(std::size_t next=aa.size();next>0;--next) {
      const auto i=next-1;
      auto child=old->kind==result_kind?old->children.at(i):exact(aa[i]);
      // Only mask-operand uses choose a bit-index coordinate. Ordinary integer
      // sequences1,2,3 must retain their linear coordinate rather than first
      // guessing an exponential map. The use kind separates proof-cache keys.
      if(grid_columns && child->kind==Kind::Exact) {
        const auto* bit_and=dynamic_cast<const v::BinaryOp*>(&x);
        const auto* literal_op=dynamic_cast<const v::ConstantOp*>(&valueNode(aa[i]).operation());
        if(bit_and && bit_and->opcode()==v::BinaryOpcode::BitAnd && literal_op)
          if(const auto number=literal(literal_op->literal());number && *number>0 &&
              std::has_single_bit(std::uint64_t(*number)))
            {
              auto [seed,inserted]=one_hot_seeds.try_emplace(child);
              if(inserted)seed->second=make(aa[i],Kind::OneHotLiteral);
              child=seed->second;
            }
      }
      auto matched=match(child,bb[i],index,depth+1);
      if(!matched)return {};
      unchanged=unchanged && matched->result==child;
      children[i]=matched->result;end=std::min(end,matched->end);
      // A range proved for the child's instance coordinate can be lifted to
      // this parent only when immutable membership storage proves the same
      // coordinate map. Sparse child roles/fixed-thread references are not
      // assumed to advance with the parent; retain a one-member certificate.
      spatial=spatial || matched->spatial;
      if(matched->index_stride>1 || matched->spatial) {
        if(index_stride && index_stride!=matched->index_stride) {
          // Concrete-target certificates are congruences anchored at this
          // same index. Their intersection has the least common multiple as
          // stride; differing periods do not imply a singleton. Spatial
          // certificates additionally encode a target-instance map, so they
          // retain the conservative path when those maps differ.
          const auto factor=index_stride/std::gcd(index_stride,matched->index_stride);
          if(!spatial && factor<=SIZE_MAX/matched->index_stride)
            index_stride=factor*matched->index_stride;
          else end=std::min(end,index+1);
        }
        else index_stride=matched->index_stride;
      }
      const auto relation=g.threadBindingRelation(bb[i],b);
      const auto& child_op=g.node(bb[i].nodeId()).operation();
      const bool constant_leaf=fixedProjection(bb[i]) &&
          (dynamic_cast<const v::ConstantOp*>(&child_op) || dynamic_cast<const v::InputParameterOp*>(&child_op));
      const bool proved=constant_leaf || (matched->spatial?relation==v::Graph::ThreadBindingRelation::SameDomain:
          (matched->thread_invariant && relation==v::Graph::ThreadBindingRelation::SameDomain) ||
          relation==v::Graph::ThreadBindingRelation::Fixed || bool(fixedInDomain(bb[i],b)));
      if(!proved) {
        if(matched->spatial)unproved_spatial=true;
        else scoped_selector_children.push_back(bb[i]);
      }
      SENTRYWARD_AUDIT(if(!proved && b_node.hasUniformThreadTemplate() && a.instanceIndex()!=b.instanceIndex() &&
          binding_witnesses<8) {
        ++binding_witnesses;
        audit->event("SOURCE_BINDING_BOUNDARY","child_member_map_not_certified",index,1,b.nodeId());
        SENTRYWARD_DEBUG_PRINT(indexed_source_13);
        if(matched->result->binding) {
          const auto& m=*matched->result->binding;
          SENTRYWARD_DEBUG_PRINT(indexed_source_14);
        }
      });
      template_proved=template_proved && proved;
      if(matched->spatial && !proved)end=std::min(end,index+1);
    }
    // Preserve which variable the proof depends on. An exact singleton target
    // stays exact; a broadcast uniform target is independent of its member;
    // only a same-domain spatial operand requires the member-coordinate map.
    // Unknown/partially overlapping domains cannot use either bulk proof.
    // Ordinary child certificates compare one fixed typed Value. If even one
    // child lacks a rebinding proof, another spatial operand cannot authorize
    // changing that child's instance with the parent.
    if(spatial && !template_proved && !unproved_spatial && !scoped_selector_children.empty() &&
       b_node.hasUniformThreadTemplate() && b_node.instanceCount()>1 && end>index+1) {
      const auto step=index_stride?index_stride:1;
      const auto available=b_node.instanceCount()-b.instanceIndex();
      const auto extent=end-index;
      const auto members=std::min(available,extent/step+bool(extent%step));
      const auto saved_begin=selector_scope_begin,saved_end=selector_scope_end;
      struct RestoreSelectorScope {
        std::size_t& begin;std::size_t& end;std::size_t old_begin,old_end;
        ~RestoreSelectorScope(){begin=old_begin;end=old_end;}
      } restore{selector_scope_begin,selector_scope_end,saved_begin,saved_end};
      selector_scope_begin=b.instanceIndex();selector_scope_end=selector_scope_begin+members;
      bool complete=members!=0;
      for(const auto& child:scoped_selector_children) {
        const auto proof=fixedInDomain(child,b);
        if(!proof || proof->end<=selector_scope_begin){complete=false;break;}
        selector_scope_end=std::min(selector_scope_end,proof->end);
      }
      if(complete) {
        const auto certified=selector_scope_end-selector_scope_begin;
        if(certified<extent/step+bool(extent%step))end=index+certified*step;
        template_proved=true;
      }
    }
    if(spatial && !template_proved)end=std::min(end,index+1);
    const bool thread_invariant=template_proved && !spatial;
    auto output=unchanged?old:make(a,result_kind,std::move(children),bit_map);
    if(bit_select) {
      bit_test_fits.insert_or_assign(output.get(),bit_fit);
      // Intersect, rather than discard, the existing child range proof with
      // the exact congruence on which this bit-index formula is unchanged.
      // Ordinary memo still requires the identical full target Value. Reuse
      // across its thread members additionally requires the local skeleton
      // proof above. The bit progression must CONTAIN the spatial child
      // progression; otherwise an LCM would change its instance advance.
      auto bits=bindingCertificate(output,index,bit_fit.size==2);
      end=std::min(end,bits.end);
      const auto child_stride=index_stride?index_stride:1;
      if(spatial && (!canonical_b->bit_template || child_stride%bits.index_stride))
        return Certificate{std::move(output),index,index+1};
      const auto factor=child_stride/std::gcd(child_stride,bits.index_stride);
      if(factor>SIZE_MAX/bits.index_stride)return Certificate{std::move(output),index,index+1};
      return Certificate{std::move(output),index,end,spatial,factor*bits.index_stride,
          thread_invariant && canonical_b->bit_template};
    }
    return Certificate{std::move(output),index,end,spatial,
        index_stride?index_stride:1,thread_invariant};
  }
 public:
  Matcher(const v::Graph& graph,const MemoryEliminationResult& r,AssembledPieceEnds* shared_piece_ends):g(graph),memory(r),specialization(graph),
      packed(graph,[this](auto value)->std::optional<std::int64_t> {
        auto known=specialization.known(value);return known?literal(known->literal):std::nullopt;
      }),own_piece_ends(shared_piece_ends?nullptr:std::make_unique<AssembledPieceEnds>()),
      piece_ends(shared_piece_ends?*shared_piece_ends:*own_piece_ends){indexInitialRanges();}
  ~Matcher() {
    sentryward::algorithm::ProfileScope facts("value_analysis.indexed_source.immutable_facts");
    facts.nodes=interpretation_hits;facts.edges=interpretation_misses;
    facts.pruned=interpretation_discards;
    sentryward::algorithm::ProfileScope domains("value_analysis.indexed_source.selector_domains");
    domains.nodes=coordinate_bounds.size();domains.edges=selector_member_visits;
    sentryward::algorithm::ProfileScope selectors("value_analysis.indexed_source.selector_facts");
    selectors.nodes=selector_new_states;selectors.edges=selector_collision_checks;selectors.pruned=selector_hits;
    sentryward::algorithm::ProfileScope metadata("value_analysis.indexed_source.match_metadata");
    metadata.nodes=template_eligibility_evaluations;metadata.edges=template_eligibility_hits;
    metadata.pruned=worklist_fact_reuses;
    sentryward::algorithm::ProfileScope endpoints("value_analysis.indexed_source.request_endpoints");
    endpoints.nodes=stored_requests;endpoints.edges=request_endpoint_lookups;
    endpoints.pruned=prototype_fact_reuses;
  }
  std::optional<v::Value> wholeScalarSlice(const IndexedSourceSlice& item,
                                          [[maybe_unused]] std::optional<std::size_t> grid_index={}) {
    charge(0);
    const auto resolved=slice(item.value,item.offset,item.bytes);
    const auto rejected=[&]([[maybe_unused]] const char* reason)->std::optional<v::Value> {
      SENTRYWARD_AUDIT({
        (void)audit;
        static thread_local unsigned witnesses=0;
        if(grid_index && witnesses<8) {
          ++witnesses;
          SENTRYWARD_DEBUG_PRINT(indexed_source_15);
          if(resolved)if(const auto* value=std::get_if<v::Value>(&resolved->source)) {
            SENTRYWARD_DEBUG_PRINT(indexed_source_16);
            if(const auto* c=dynamic_cast<const v::ConstantOp*>(&g.node(value->nodeId()).operation()))
              SENTRYWARD_DEBUG_PRINT(indexed_source_17);
          }
          if(resolved)if(const auto* initial=std::get_if<InitialMemorySymbol>(&resolved->source))
            SENTRYWARD_DEBUG_PRINT(indexed_source_18);
          SENTRYWARD_DEBUG_PRINT(indexed_source_19);
        }
      });
      return {};
    };
    if(!resolved)return rejected("unresolved");
    if(resolved->offset)return rejected("partial_offset");
    const auto* value=std::get_if<v::Value>(&resolved->source);
    if(!value)return rejected("initial_symbol");
    const auto& type=valueType(*value);
    if(!type.isScalar() || !type.hasStaticShape() || type.element().isParameterized() ||
       !type.bitSize() || type.bitSize()%8 || type.bitSize()/8!=item.bytes)return rejected("not_whole_scalar");
    return *value;
  }
  std::optional<IndexedSource> runGridSlices(std::span<const IndexedSourceSlice> values,
                                            std::uint64_t columns) {
    beginDomain();work=0;
    if(values.size()<3 || !values.front().bytes || columns<2 ||
       values.size()%columns || values.size()/columns<2 || values.size()>1000000)return {};
    std::vector<v::Value> prepared;prepared.reserve(values.size());
    for(const auto& item:values) {
      if(item.bytes!=values.front().bytes)return {};
      auto value=wholeScalarSlice(item,prepared.size());if(!value)return {};
      prepared.push_back(*value);
    }
    // All supplied records have been resolved exactly. masked() retains the
    // existing complete grid proof, selector handling and bounded fitting.
    return masked(prepared,columns,work);
  }
  std::optional<IndexedSource> runSlices(std::span<const IndexedSourceSlice> values) {
    beginDomain();
    struct FiniteSlices {
      bool& flag;bool previous;
      explicit FiniteSlices(bool& value):flag(value),previous(value){flag=true;}
      ~FiniteSlices(){flag=previous;}
    } finite_slices{finite_slice_domain};
    grid_failure_witness.reset();
    work=0;memo.clear();template_memo.clear();exact_nodes.clear();member_count=values.size();
    if(values.size()<3 || !values.front().bytes)return {};
    // A matrix read can resolve to whole scalar computed values. Seed the
    // existing matcher with those exact resolved values, not with a matrix
    // prototype whose width prevents forwarding the scalar producer. This is
    // one pass over supplied stored slices using the shared cached resolver;
    // partial fields and initial symbols retain the existing slice route.
    bool needs_slice_seed=false;
    for(const auto& item:values) {
      const auto& type=valueType(item.value);
      if(item.offset || !type.isScalar() || !type.hasStaticShape() ||
         type.element().isParameterized() || type.bitSize()%8 || type.bitSize()/8!=item.bytes) {
        needs_slice_seed=true;break;
      }
    }
    // Whole scalar operands already have the correct typed prototype, including
    // native matrix component selections. Retain that complete use identity.
    std::vector<IndexedSourceSlice> prepared;
    if(needs_slice_seed)prepared.reserve(values.size());
    if(needs_slice_seed)for(const auto& item:values) {
      if(item.bytes!=values.front().bytes)break;
      const auto value=wholeScalarSlice(item);
      if(!value)break;
      prepared.push_back({*value,0,item.bytes});
    }
    if(prepared.size()==values.size())values=prepared;
    auto root=exact(values.front().value);
    const auto& first=values.front();
    const auto sa=slice(first.value,first.offset,first.bytes);
    if(!sa)return {};
    for(std::size_t i=1;i<values.size();++i) {
      SENTRYWARD_AUDIT(audit->add(AuditCounter::source_members_validated));
      const auto& next=values[i];if(next.bytes!=first.bytes)return {};
      std::optional<Certificate> result;
      const auto sb=slice(next.value,next.offset,next.bytes);
      const auto* source=std::get_if<v::Value>(&sa->source);
      const bool resolved_pair=sb && sa->source.index()==sb->source.index() &&
          (!source || *source==std::get<v::Value>(sb->source));
      // This API validates an explicit finite slice domain. An authoritative
      // initial-memory pair or two slices of the SAME producer already supplies
      // the exact source/offset relation, including constant thread selections.
      // Do not send it through the template-only XOR-selector theorem instead.
      // Numeric conversions remain opaque Values in slice(); only exact bit
      // reinterpretations can be crossed by its existing resolver.
      if(resolved_pair)result=sourceSlices(root,next.value,i,0,sa,sb);
      else if(!first.offset && !next.offset && first.bytes==valueType(first.value).bitSize()/8 &&
          next.bytes==valueType(next.value).bitSize()/8)
        result=match(root,next.value,i);
      else result=sourceSlices(root,next.value,i,0,sa,sb);
      if(!result) {
        SENTRYWARD_AUDIT({
          (void)audit;
          if(slice_witnesses++<8) {
          const auto sb=slice(next.value,next.offset,next.bytes);
          SENTRYWARD_DEBUG_PRINT(indexed_source_20);
          if(auto* initial=std::get_if<InitialMemorySymbol>(&sa->source))
            SENTRYWARD_DEBUG_PRINT(indexed_source_21);
          if(sb)if(auto* initial=std::get_if<InitialMemorySymbol>(&sb->source))
            SENTRYWARD_DEBUG_PRINT(indexed_source_22);
          if(root->binding)SENTRYWARD_DEBUG_PRINT(indexed_source_23);
          SENTRYWARD_DEBUG_PRINT(indexed_source_24);
          if(grid_failure_witness) {
            const auto& witness=*grid_failure_witness;
            const auto left=witness.old->prototype,right=witness.input;
            const auto print=[&](const char* label,const v::Value& value) {
              const auto& node=g.node(value.nodeId());
              SENTRYWARD_DEBUG_PRINT(indexed_source_25);
            };
            SENTRYWARD_DEBUG_PRINT(indexed_source_26);
            print("left",left);print("right",right);SENTRYWARD_DEBUG_PRINT(indexed_source_27);
          }
          }
        });
        return {};
      }
      root=std::move(result->result);
    }
    SENTRYWARD_AUDIT(audit->event("SOURCE_SLICE_DOMAIN","native_operand_layout",values.size(),1,root->prototype.nodeId()));
    return IndexedSource{root->prototype,values.size(),root};
  }
  IndexedSource finishSource(v::Value prototype,std::size_t count,const Ptr& root) {
    SENTRYWARD_AUDIT(audit->event("SOURCE_USE_ROOT","retained_immutable_root",count,1,root->prototype.nodeId()));
    return IndexedSource{g.canonicalInterpretation(prototype),count,root};
  }
  v::Value fixedMember(v::Value prototype,std::size_t member) const {
    // fixedProjection has checked this annotation chain. Rebind its leaf,
    // preserving the order of component selection and numeric interpretation.
    // Cost is the stored annotation depth, not the producer expression DAG.
    if(!prototype.indices().empty()) {
      auto source=fixedMember(prototype.withoutSelection(),member);
      return g.component(source,{prototype.indices().begin(),prototype.indices().end()});
    }
    if(prototype.conversion()) {
      const auto& c=*prototype.conversion();
      auto source=fixedMember(c.source,member);
      return c.spec->kind==v::TypeConversionKind::Bitcast?g.bitcast(source,c.spec->target):
          g.convert(source,c.spec->target,c.spec->attributes);
    }
    return g.node(prototype.nodeId()).result(prototype.resultIndex(),member);
  }
  std::optional<IndexedSource> runTemplate(const IndexedSourceTemplate& domain) {
    beginDomain();grid_columns.reset();grid_fits.clear();
    work=0;memo.clear();template_memo.clear();exact_nodes.clear();member_count=domain.count;
    const auto& first=domain.prototype;
    if(domain.count<3 || !g.contains(first) || !fixedProjection(first))return {};
    const auto& node=g.node(first.nodeId());
    if(!node.hasUniformThreadTemplate() || first.instanceIndex()>node.instanceCount() ||
       domain.count>node.instanceCount()-first.instanceIndex())return {};
    auto root=exact(first);
    // These local matches construct an index map, not a sample-based proof.
    // Acceptance additionally requires the read-from/domain certificates to
    // cover EVERY requested member. No proof => reject, no member loop.
    for(std::size_t i=1;i<=2;++i) {
      auto found=match(root,fixedMember(first,first.instanceIndex()+i),i);
      if(!found || found->end<domain.count || found->index_stride!=1 ||
         (!found->spatial && !found->thread_invariant)) {
        SENTRYWARD_AUDIT(audit->event("TEMPLATE_SOURCE_UNSUPPORTED","read_from_domain_not_certified",
            domain.count,1,first.nodeId()));
        return {};
      }
      root=std::move(found->result);
    }
    return finishSource(first,domain.count,root);
  }
  std::optional<IndexedSource> run(std::span<const v::Value> values) {
    // The legacy finite-record interface must inspect its records, but it need
    // not inspect their expression ancestors. An already-template-shaped input
    // is sent to the same non-materializing public domain implementation.
    if(values.size()>=3) {
      const auto& first=values.front();
      if(g.contains(first) && g.node(first.nodeId()).hasUniformThreadTemplate()) {
        const auto eligible=[&](const v::Value& value,std::size_t index) {
          return value.graphIdentity()==first.graphIdentity() && value.nodeId()==first.nodeId() &&
              value.resultIndex()==first.resultIndex() && first.instanceIndex()<=SIZE_MAX-index &&
              value.instanceIndex()==first.instanceIndex()+index && fixedProjection(value) &&
              sameFixedInterpretation(first,value);
        };
        bool uniform=true;
        for(std::size_t i=0;i<values.size();++i)if(!eligible(values[i],i)){uniform=false;break;}
        if(uniform)if(auto result=runTemplate({first,values.size()}))return result;
      }
    }
    beginDomain();
    grid_columns.reset();grid_fits.clear();
    work=0;memo.clear();template_memo.clear();exact_nodes.clear();member_count=values.size();
    if(values.size()<3)return {};
    auto root=exact(values.front());std::size_t count=1;
    for(std::size_t i=1;i<values.size();++i) {
      SENTRYWARD_AUDIT(audit->add(AuditCounter::source_members_validated));
      auto matched=match(root,values[i],i);
      if(!matched) {
        SENTRYWARD_AUDIT(audit->event("SOURCE_USE_REJECTED","nonuniform_use_pattern",i,1,root->prototype.nodeId()));
        break;
      }
      root=std::move(matched->result);++count;
    }
    if(count<3)return {};
    // The summary walks each reachable use once. Failed speculative refinements
    // are unreachable and cannot contaminate the accepted prefix.
    return finishSource(values.front(),count,root);
  }
  std::optional<equivalence::ConstantValue> constant(v::Value value,unsigned depth=0) {
    if(depth>64)return {};
    value=g.canonicalInterpretation(value);
    if(!valueType(value).isScalar() || value.threadSelector() || !value.indices().empty())return {};
    const auto type=valueType(value).element().toString();
    if(value.conversion()) {
      const auto& c=*value.conversion();auto src=constant(c.source,depth+1);if(!src)return {};
      auto step=std::string(c.spec->kind==v::TypeConversionKind::Bitcast?"Bitcast":"Convert");
      if(c.spec->kind==v::TypeConversionKind::Convert)step+=' '+attrs(v::ConvertOp(c.spec->target,c.spec->attributes));
      return equivalence::constantValue(src->type,src->literal,
          std::string(step)+'('+src->type+" -> "+type+')',type);
    }
    if(const auto* c=dynamic_cast<const v::ConstantOp*>(&valueNode(value).operation()))
      return equivalence::constantValue(type,c->literal());
    return {};
  }
  std::optional<IndexedSource> masked(std::span<const v::Value> values,std::uint64_t columns,
                                     std::size_t preparation_charge=0) {
    // Retain the complete conditional computation while proving a grid. A
    // coordinate-varying known predicate must not turn identical Select
    // templates into unrelated chosen operands. Thread selection still resolves
    // exact occurrences; both branches and the predicate are matched below.
    struct PreserveSelect {
      bool& flag;bool previous;
      explicit PreserveSelect(bool& f):flag(f),previous(f){flag=true;}
      ~PreserveSelect(){flag=previous;}
    } preserve_select{preserve_data_select};
    beginDomain();
    grid_failure_witness.reset();
    const auto rejected=[&](const char* reason,std::size_t index) {
      SENTRYWARD_AUDIT((void)audit;if(values.size()>=3 && grid_failure_witnesses<4) {
        ++grid_failure_witnesses;
        SENTRYWARD_DEBUG_PRINT(indexed_source_28);
        if(index<values.size())SENTRYWARD_DEBUG_PRINT(indexed_source_29);
        if(grid_failure_witness) {
          const auto& w=*grid_failure_witness;
          SENTRYWARD_DEBUG_PRINT(indexed_source_30);
          if(w.old->binding) {
            const auto& b=*w.old->binding;
            SENTRYWARD_DEBUG_PRINT(indexed_source_31);
            if(b.producer)SENTRYWARD_DEBUG_PRINT(indexed_source_32);
          }
          if(w.input.threadSelector()) {
            const auto selector=*w.input.threadSelector();
            const auto known=specialization.known(selector);
            const auto& source=valueNode(w.input);
            SENTRYWARD_DEBUG_PRINT(indexed_source_33);
            for(std::size_t j=0;j<std::min<std::size_t>(source.instanceThreads().size(),8);++j)
              SENTRYWARD_DEBUG_PRINT(indexed_source_34);
            const auto resolved=selected(w.input);
            SENTRYWARD_DEBUG_PRINT(indexed_source_35);
            std::vector<v::Value> pending{selector};
            for(unsigned n=0;n<8 && n<pending.size();++n) {
              const auto v=pending[n];const auto& op=valueNode(v).operation();
              const auto k=specialization.known(v);
              SENTRYWARD_DEBUG_PRINT(indexed_source_36);
              if(!k && dynamic_cast<const v::BinaryOp*>(&op)) {
                std::ostringstream attrs;op.printAttributes(attrs);
                std::vector<equivalence::ConstantValue> constants;
                bool complete=true;
                SENTRYWARD_DEBUG_PRINT(indexed_source_37);
                for(const auto& operand:g.instance(v).operands) {
                  const auto c=specialization.known(operand);
                  SENTRYWARD_DEBUG_PRINT(indexed_source_38);
                  if(c)constants.push_back(*c);else complete=false;
                }
                if(complete) {
                  const auto direct=equivalence::constantOperation(std::string(op.name()),
                      g.type(v).element().toString(),attrs.str(),constants);
                  SENTRYWARD_DEBUG_PRINT(indexed_source_39);
                }
              }
              if(pending.size()<16)for(const auto& child:g.instance(v).operands)
                if(pending.size()<16)pending.push_back(child);
            }
          }
        }
        SENTRYWARD_DEBUG_PRINT(indexed_source_40);
      });
      (void)reason;(void)index;
    };
    // A bounded two-axis proof over existing records. Mask membership is exact,
    // including rejecting holes or arbitrary nonzero exceptions.
    if(columns<2 || values.size()%columns || values.size()/columns<2 || values.size()>1000000)return {};
    // Normalize only the already supplied finite roots. A resolved selection
    // is this exact input Value, not a prediction about another thread. Keep
    // every type/conversion/index annotation through the shared interpreter.
    std::vector<v::Value> prepared;prepared.reserve(values.size());
    std::size_t preparation_work=0;
    for(const auto& value:values) {
      ++preparation_work;prepared.push_back(selected(interpretation(value)));
    }
    values=prepared;
    SENTRYWARD_AUDIT({
      (void)audit;
      if(mapped_admission_witnesses<4 && !values.empty()) {
        ++mapped_admission_witnesses;
        std::size_t casts=0,selectors=0,components=0,pointers=0,node_changes=0,spec_changes=0;
        std::size_t first_change=SIZE_MAX,base_node=0,base_result=0,minimum=SIZE_MAX,maximum=0;
        std::vector<const v::TypeConversionSpec*> specs;
        for(auto x=values.front();x.conversion() && specs.size()<32;x=x.conversion()->source)
          specs.push_back(x.conversion()->spec.get());
        __int128 base=0,row=0,col=0;bool affine=true;
        for(std::size_t i=0;i<values.size();++i) {
          auto x=values[i];casts+=bool(x.conversion());
          selectors+=bool(x.threadSelector());components+=!x.indices().empty();pointers+=bool(x.pointer());
          std::size_t depth=0;
          while(x.conversion() && depth<32) {
            if(depth>=specs.size() || *specs[depth]!=*x.conversion()->spec)++spec_changes;
            x=x.conversion()->source;++depth;
          }
          if(depth!=specs.size() || x.conversion())++spec_changes;
          selectors+=bool(x.threadSelector());components+=!x.indices().empty();pointers+=bool(x.pointer());
          if(!i){base_node=x.nodeId();base_result=x.resultIndex();base=x.instanceIndex();}
          if(x.nodeId()!=base_node || x.resultIndex()!=base_result) {
            ++node_changes;if(first_change==SIZE_MAX)first_change=i;
          }
          minimum=std::min(minimum,x.instanceIndex());maximum=std::max(maximum,x.instanceIndex());
          if(i==1)col=__int128(x.instanceIndex())-base;
          if(i==columns)row=__int128(x.instanceIndex())-base;
          // Rows before the second are checked with row=0; row0 is unaffected.
          if(__int128(x.instanceIndex())!=base+row*(i/columns)+col*(i%columns))affine=false;
        }
        SENTRYWARD_DEBUG_PRINT(indexed_source_41);
      }
    });
    std::vector<std::size_t> active;
    std::int64_t min_live=INT64_MAX,max_live=INT64_MIN,min_zero=INT64_MAX,max_zero=INT64_MIN;
    for(std::size_t i=0;i<values.size();++i) {
      const auto t=valueType(values[i]);
      if(t!=valueType(values[0]) || !t.isScalar()){rejected("type",i);return {};}
      const auto c=constant(values[i]);const bool zero=c && c->literal=="0";
      const auto diagonal=static_cast<std::int64_t>(i/columns)-static_cast<std::int64_t>(i%columns);
      if(zero){min_zero=std::min(min_zero,diagonal);max_zero=std::max(max_zero,diagonal);}
      else {active.push_back(i);min_live=std::min(min_live,diagonal);max_live=std::max(max_live,diagonal);}
    }
    if(active.size()<3)return {};
    std::optional<IndexedSource::ZeroMask> mask;
    if(active.size()!=values.size()) {
      if(max_zero<min_live)mask=IndexedSource::ZeroMask{columns,min_live,true};
      else if(max_live<min_zero)mask=IndexedSource::ZeroMask{columns,min_zero,false};
      else {rejected("zero_mask",0);return {};}
    }
    work=preparation_charge+preparation_work;charge(0);
    memo.clear();template_memo.clear();exact_nodes.clear();grid_fits.clear();member_count=values.size();
    grid_columns=columns;grid_origin=active[0];grid_max_validated=grid_origin;grid_prefix={};
    struct Reset {std::optional<std::uint64_t>& x;~Reset(){x.reset();}} reset{grid_columns};
    // Establish a candidate's independent coordinate axes early, then walk
    // same-column records consecutively. Each record is still checked by
    // match()/template_memo against its actual full Value and member index.
    // The basis is only scheduling: acceptance requires every record's proof.
    std::vector<std::size_t> anchors{grid_origin};
    __int128 first_row=0,first_col=0;
    for(auto i:active) {
      const auto row=__int128(i/columns)-__int128(grid_origin/columns);
      const auto col=__int128(i%columns)-__int128(grid_origin%columns);
      if(anchors.size()==1 && (row || col)) {
        anchors.push_back(i);first_row=row;first_col=col;
      } else if(anchors.size()==2 && first_row*col-first_col*row) {
        anchors.push_back(i);break;
      }
    }
    std::vector<std::vector<std::size_t>> column_records(columns);
    for(auto i:active)column_records[i%columns].push_back(i);
    std::vector<std::size_t> order;order.reserve(active.size());
    order.insert(order.end(),anchors.begin(),anchors.end());
    for(const auto& records:column_records)for(auto i:records)
      if(std::find(anchors.begin(),anchors.end(),i)==anchors.end())order.push_back(i);
    auto root=exact(values[grid_origin]);
    for(auto i:order) {
      if(i==grid_origin)continue;
      SENTRYWARD_AUDIT(audit->add(AuditCounter::source_members_validated));
      auto found=match(root,values[i],i);
      if(!found) {
        rejected("member_match",i);
        SENTRYWARD_AUDIT(audit->event("GRID_SOURCE_REJECTED","nonuniform_grid_use",i,values.size(),root->prototype.nodeId()));
        return {};
      }
      root=std::move(found->result);
      // An exact use introduced at a later refinement must also obey every
      // previously observed (constant-valued) coordinate constraint.
      if(!constrain(grid_prefix,gridPoint(i)))throw std::logic_error("invalid grid prefix");
      grid_max_validated=std::max(grid_max_validated,i);
    }
    IndexedSource result{g.canonicalInterpretation(values[grid_origin]),values.size(),root};
    result.zero_mask=mask;
    SENTRYWARD_AUDIT({
      (void)audit;
      if(mapped_root_witnesses<4) {
        ++mapped_root_witnesses;
        // Only the direct retained root payload, never an ancestor walk.
        const auto* binding=root->binding.get();
        SENTRYWARD_DEBUG_PRINT(indexed_source_42);
      }
    });
    SENTRYWARD_AUDIT(audit->event("GRID_SOURCE_DOMAIN",mask?"validated_affine_diagonal_zero_mask":"validated_affine_grid",
        values.size(),mask?2:1,root->prototype.nodeId()));
    return result;
  }
};
}
struct IndexedSourceMatcher::Impl : Matcher { using Matcher::Matcher; };
std::optional<std::int64_t> indexedBindingValue(const IndexedValueBinding& b,std::uint64_t i) {
  __int128 result=b.base;
  const auto add=[&](std::int64_t coefficient,std::uint64_t coordinate) {
    const auto term=static_cast<__int128>(coefficient)*coordinate;
    return !__builtin_add_overflow(result,term,&result);
  };
  if(!add(b.stride,b.period?i%b.period:i) || !add(b.outer_stride,b.period?i/b.period:0))return {};
  for(const auto& t:b.quotient_terms)if(!t.divisor || !add(t.coefficient,i/t.divisor))return {};
  if(result<INT64_MIN || result>INT64_MAX)return {};
  return static_cast<std::int64_t>(result);
}
std::shared_ptr<const IndexedSourceUse> bindSourceUse(const v::Graph& graph,
    v::Value operand,v::Value consumer,std::int64_t base,std::int64_t stride) {
  using U=IndexedSourceUse;
  // Annotation depth is independent of represented thread count. Ordinary
  // operands remain lazy MemberTemplate leaves; do not descend their DAG.
  const auto bind=[&](auto&& self,v::Value value,unsigned depth)->std::shared_ptr<const U> {
    if(depth>=128)return {};
    value=graph.canonicalInterpretation(value);
    if(!value.indices().empty()) {
      std::vector<std::shared_ptr<const U>> children;
      auto matrix=self(self,value.withoutSelection(),depth+1);if(!matrix)return {};
      children.push_back(std::move(matrix));
      for(auto index:value.indices()) {
        auto child=self(self,index,depth+1);if(!child)return {};
        children.push_back(std::move(child));
      }
      return std::make_shared<U>(U{value,U::Kind::Component,{},std::move(children)});
    }
    if(value.conversion()) {
      // Selection on the converted value applies to its source as well.
      // Keep it in the child's receiver-domain binding; dropping this edge
      // would turn fromThread(convert(x), s) into convert(x). fromThread also
      // composes an existing source selection rather than replacing it.
      auto source=value.conversion()->source;
      if(value.threadSelector())source=graph.fromThread(source,*value.threadSelector());
      auto child=self(self,source,depth+1);if(!child)return {};
      return std::make_shared<U>(U{value,U::Kind::Conversion,{},{std::move(child)}});
    }
    const auto& node=graph.definingNode(value);
    if(node.isCollectiveOccurrence() && dynamic_cast<const v::WarpMmaOp*>(&node.operation()) &&
        !value.pointer() && !value.threadSelector())
      // WarpMma's result is the collective matrix, not a per-lane register.
      // The separate component children choose each receiver's elements.
      return std::make_shared<U>(U{value,U::Kind::Exact,{},{}});
    const auto map=graph.threadBindingMap(value,consumer);if(!map)return {};
    if(map->fixed)return std::make_shared<U>(U{value,U::Kind::Exact,{},{}});
    const auto first=__int128(map->base)+__int128(map->stride)*base;
    const auto step=__int128(map->stride)*stride;
    if(first<INT64_MIN || first>INT64_MAX || step<INT64_MIN || step>INT64_MAX)return {};
    auto binding=std::make_shared<IndexedValueBinding>(IndexedValueBinding{
        value,static_cast<std::int64_t>(first),static_cast<std::int64_t>(step),{}});
    return std::make_shared<U>(U{value,U::Kind::MemberTemplate,std::move(binding),{}});
  };
  return bind(bind,operand,0);
}
std::vector<IndexedValueBinding> indexedSourceBindings(const IndexedSource& source) {
  using Use=IndexedSourceUse;using Kind=Use::Kind;
  std::vector<IndexedValueBinding> result;
  std::set<const Use*> seen;
  std::vector<std::shared_ptr<const Use>> pending;
  if(source.root)pending.push_back(source.root);
  while(!pending.empty()) {
    auto use=std::move(pending.back());pending.pop_back();
    if(!seen.insert(use.get()).second)continue;
    if(use->binding)result.push_back(*use->binding);
    else if(use->kind==Kind::Exact)result.push_back({use->prototype,0,0,{},true});
    else if(use->kind==Kind::Forward || use->kind==Kind::Selected)
      result.push_back({use->prototype,0,0,{},false,use->children.at(0)->prototype,true});
    pending.insert(pending.end(),use->children.begin(),use->children.end());
  }
  return result;
}
IndexedSourceMatcher::IndexedSourceMatcher(const v::Graph& graph,const MemoryEliminationResult& memory)
    :impl_(std::make_unique<Impl>(graph,memory,nullptr)) {}
IndexedSourceMatcher::IndexedSourceMatcher(const v::Graph& graph,const MemoryEliminationResult& memory,
    AssembledPieceEnds& piece_ends):impl_(std::make_unique<Impl>(graph,memory,&piece_ends)) {}
IndexedSourceMatcher::~IndexedSourceMatcher()=default;
std::optional<IndexedSource> IndexedSourceMatcher::match(std::span<const v::Value> values) {
  return impl_->run(values);
}
std::optional<IndexedSource> IndexedSourceMatcher::matchTemplate(const IndexedSourceTemplate& domain) {
  return impl_->runTemplate(domain);
}
std::optional<IndexedSource> IndexedSourceMatcher::matchSlices(std::span<const IndexedSourceSlice> values) {
  return impl_->runSlices(values);
}
std::optional<IndexedSource> IndexedSourceMatcher::matchGrid(
    std::span<const v::Value> values,std::uint64_t columns) {
  return impl_->masked(values,columns);
}
std::optional<IndexedSource> IndexedSourceMatcher::matchGridSlices(
    std::span<const IndexedSourceSlice> values,std::uint64_t columns) {
  return impl_->runGridSlices(values,columns);
}
std::vector<IndexedSourceDomain> IndexedSourceMatcher::partition(std::span<const v::Value> values) {
  std::vector<IndexedSourceDomain> domains;
  for(std::size_t begin=0;begin<values.size();) {
    if(auto source=match(values.subspan(begin))) {
      const auto count=source->count;
      if(count<3 || count>values.size()-begin)throw std::logic_error("invalid indexed source domain");
      domains.push_back({begin,count,std::make_shared<const IndexedSource>(std::move(*source))});
      begin+=count;
    } else domains.push_back({begin++,1,{}});
    SENTRYWARD_AUDIT(audit->add(AuditCounter::source_domains_planned));
  }
  return domains;
}
std::optional<IndexedSource> matchIndexedSource(const v::Graph& graph,
    const MemoryEliminationResult& memory,std::span<const v::Value> values) {
  return Matcher(graph,memory,nullptr).run(values);
}
std::vector<IndexedSourceDomain> mapSourceTemplates(const v::Graph& graph,
    std::span<const v::Value> values,bool interleaved,
    std::span<const std::uint64_t> payload_offsets) {
  if(!payload_offsets.empty() && payload_offsets.size()!=values.size())
    throw std::invalid_argument("source payload offsets must cover stored records");
  const auto payload=[&](std::size_t i){return payload_offsets.empty()?0:payload_offsets[i];};
  // A MemberTemplate's absolute instance is in its binding, not its template
  // identity. Normalize the representative while retaining conversions and
  // constant component annotations. No operands or logical members are visited.
  const auto prototypeOf=[&](auto&& self,const v::Value& value)->v::Value {
    if(!graph.node(value.nodeId()).hasUniformThreadTemplate() ||
       value.pointer() || value.threadSelector())return value;
    if(!value.indices().empty())return graph.component(self(self,value.withoutSelection()),
        {value.indices().begin(),value.indices().end()});
    if(value.conversion()) {
      const auto& c=*value.conversion();auto source=self(self,c.source);
      return c.spec->kind==v::TypeConversionKind::Bitcast?graph.bitcast(source,c.spec->target):
          graph.convert(source,c.spec->target,c.spec->attributes);
    }
    return graph.node(value.nodeId()).result(value.resultIndex(),0);
  };
  const auto sameTemplate=[&](auto&& self,const v::Value& a,const v::Value& b)->bool {
    if(a.nodeId()!=b.nodeId() || a.resultIndex()!=b.resultIndex() ||
       a.pointer() || b.pointer() || a.threadSelector() || b.threadSelector())return false;
    // Repeated uses of one exact value need no thread-template certificate.
    // Only a changing member binding requires the uniform-template contract.
    if(a==b)return true;
    if(a.indices().size()!=b.indices().size())return false;
    if(!a.indices().empty()) {
      for(std::size_t i=0;i<a.indices().size();++i)
        if(a.indices()[i]!=b.indices()[i] ||
           !dynamic_cast<const v::ConstantOp*>(&graph.definingNode(a.indices()[i]).operation()))return false;
      return self(self,a.withoutSelection(),b.withoutSelection());
    }
    if(bool(a.conversion())!=bool(b.conversion()))return false;
    if(a.conversion()) {
      const auto& x=*a.conversion();const auto& y=*b.conversion();
      return x.spec->kind==y.spec->kind && x.spec->target==y.spec->target &&
          x.spec->attributes==y.spec->attributes && self(self,x.source,y.source);
    }
    return graph.node(a.nodeId()).hasUniformThreadTemplate();
  };
  if(interleaved) {
    // Numeric layout grouping only. The full annotation identity is checked
    // inside each bucket; hash/string coincidence is never an equality proof.
    struct Run {std::size_t first,last,count=1,step=1;std::int64_t member_step=0;};
    std::map<std::tuple<std::size_t,std::size_t,std::string,std::uint64_t,std::size_t>,std::vector<Run>> active;
    std::vector<IndexedSourceDomain> result;
    const auto emit=[&](const Run& run) {
      if(run.count==1){result.push_back({run.first,1,{}});return;}
      const auto original=graph.canonicalInterpretation(values[run.first]);
      auto prototype=prototypeOf(prototypeOf,original);
      auto binding=std::make_shared<IndexedValueBinding>(IndexedValueBinding{prototype,
          static_cast<std::int64_t>(original.instanceIndex()),run.member_step,{}});
      auto use=std::make_shared<IndexedSourceUse>(IndexedSourceUse{prototype,
          IndexedSourceUse::Kind::MemberTemplate,std::move(binding),{}});
      result.push_back({run.first,run.count,std::make_shared<IndexedSource>(
          IndexedSource{prototype,run.count,std::move(use)}),run.step});
    };
    for(std::size_t i=0;i<values.size();++i) {
      const auto& value=values[i];
      if(!sameTemplate(sameTemplate,value,value)){emit(Run{i,i});continue;}
      const auto fixed_member=graph.node(value.nodeId()).hasUniformThreadTemplate()?0:value.instanceIndex();
      auto& bucket=active[{value.nodeId(),value.resultIndex(),v::typeConversionLabel(value),payload(i),fixed_member}];
      auto found=std::find_if(bucket.begin(),bucket.end(),[&](const Run& run) {
        return sameTemplate(sameTemplate,values[run.first],value);
      });
      if(found==bucket.end()){bucket.push_back({i,i});continue;}
      const auto step=i-found->last;
      const auto delta=static_cast<__int128>(value.instanceIndex())-values[found->last].instanceIndex();
      if(delta>=INT64_MIN && delta<=INT64_MAX && (found->count==1 ||
          (step==found->step && delta==found->member_step))) {
        found->step=step;found->member_step=static_cast<std::int64_t>(delta);
        found->last=i;++found->count;
      } else {emit(*found);*found=Run{i,i};}
    }
    for(const auto& [key,bucket]:active){(void)key;for(const auto& run:bucket)emit(run);}
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.begin<b.begin;});
    return result;
  }
  std::vector<IndexedSourceDomain> result;
  for(std::size_t begin=0;begin<values.size();) {
    std::size_t end=begin+1;std::int64_t stride=0;
    if(end<values.size() && payload(begin)==payload(end) && sameTemplate(sameTemplate,values[begin],values[end])) {
      const auto delta=static_cast<__int128>(values[end].instanceIndex())-values[begin].instanceIndex();
      if(delta>=INT64_MIN && delta<=INT64_MAX) {
        stride=static_cast<std::int64_t>(delta);++end;
        while(end<values.size() && payload(begin)==payload(end) && sameTemplate(sameTemplate,values[begin],values[end]) &&
            static_cast<__int128>(values[begin].instanceIndex())+delta*(end-begin)==values[end].instanceIndex())++end;
      }
    }
    if(end==begin+1)result.push_back({begin,1,{}});
    else {
      const auto original=graph.canonicalInterpretation(values[begin]);
      auto prototype=prototypeOf(prototypeOf,original);
      auto binding=std::make_shared<IndexedValueBinding>(IndexedValueBinding{prototype,
          static_cast<std::int64_t>(original.instanceIndex()),stride,{}});
      auto use=std::make_shared<IndexedSourceUse>(IndexedSourceUse{prototype,
          IndexedSourceUse::Kind::MemberTemplate,std::move(binding),{}});
      auto source=std::make_shared<IndexedSource>(IndexedSource{prototype,end-begin,std::move(use)});
      result.push_back({begin,end-begin,std::move(source)});
    }
    begin=end;
  }
  return result;
}
}
