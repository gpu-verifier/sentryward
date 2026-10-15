#include "sentryward/algorithm/diagnostics/development_trace.h"
#pragma once
#include "sentryward/value_analysis/indexed_source.h"
#include "sentryward/algorithm/diagnostics/compression_audit.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/ptx_operation.h"
#include "sentryward/value_ir/type_conversion.h"
#include "sentryward/value_analysis/scalar_specialization.h"
#include "sentryward/value_analysis/packed_storage.h"
#include "sentryward/value_analysis/ideal_scalar_semantics.h"
#include <map>
#include <charconv>
#include <deque>
#include <tuple>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <cstdlib>
#include <iostream>

namespace sentryward::value_analysis {
// Pass-local bottom-up summaries. Each immutable annotated Value is visited
// once; parent construction reads only direct child summaries. Uses share
// their child DAGs rather than copying flattened bindings or matching trees.
// A flattened coordinate and exact row-quotient maps are supported. Other
// independent coordinates remain separate templates. Every supplied stored
// record is validated, rather than extrapolating from samples.
class PureSourceTemplates final {
  using V=value_ir::Value;
  using U=IndexedSourceUse;
  using Ptr=std::shared_ptr<const U>;
  struct Summary {std::size_t id;std::int64_t coordinate;bool varying;std::uint64_t row_bytes=0;};
  struct ChildMap {
    V value;
    std::size_t child;
    std::int64_t base,stride,outer;
    std::uint64_t period;
  };
  struct Template {
    V prototype;
    U::Kind kind;
    std::shared_ptr<const IndexedValueBinding> binding;
    std::vector<ChildMap> children;
    Ptr materialized={};
  };
  static auto planCost(const std::vector<IndexedSourceDomain>& domains) {
    // Count contiguous source runs, not merely stored AP records: sixteen
    // interleaved APs can force sixteen scalar K pieces on every matrix row.
    // This is a locality heuristic, not a proof of downstream complexity.
    std::size_t runs=0,singletons=0;std::set<const U*> uses;
    for(const auto& domain:domains) {
      runs+=domain.result_stride==1?1:domain.count;
      if(!domain.source){++singletons;continue;}
      const auto& root=domain.source->root;
      uses.insert(root && root->kind==U::Kind::CoordinateMap && root->children.size()==1?
          root->children[0].get():root.get());
    }
    // Equal AP counts need not mean equal work: shared parameterized bodies
    // allow the output consumer to combine rows into an existing 2-D domain.
    return std::tuple{runs,domains.size(),uses.size()+singletons};
  }
  enum class Form {Exact,Leaf,Literal,Forward,Conversion,Operands};
  struct Plan {
    Form form=Form::Exact;
    std::vector<V> children;
    std::string signature;
    std::optional<IndexedValueBinding> leaf;
    std::int64_t coordinate=0;
    std::uint64_t row_bytes=0;
    bool varying=true;
  };
  const value_ir::Graph& g;
  const MemoryEliminationResult& memory;
  ScalarSpecialization specialization;
  PackedStorageDecoder storage;
  struct SummaryKey {
    V value;
    unsigned policy;
    bool operator==(const SummaryKey&) const = default;
  };
  struct SummaryHash {
    static std::size_t valueHash(const V& value) noexcept {
      auto h=value.nodeId();
      const auto mix=[&](std::size_t x){h^=x+0x9e3779b9+(h<<6)+(h>>2);};
      mix(value.resultIndex());mix(value.instanceIndex());
      // Hash annotations, never operand ancestors. Equality checks the full
      // conversion spec and allocation metadata, not just this index hash.
      if(value.conversion()) {
        mix(std::size_t(value.conversion()->spec->kind)+1);
        mix(valueHash(value.conversion()->source));
      }
      for(const auto& index:value.indices())mix(valueHash(index));
      if(value.threadSelector())mix(valueHash(*value.threadSelector()));
      return h;
    }
    std::size_t operator()(const SummaryKey& k) const noexcept {
      return valueHash(k.value)^(std::size_t(k.policy)*0x9e3779b9);
    }
  };
  std::unordered_map<SummaryKey,Summary,SummaryHash> summaries;
  std::unordered_map<std::string,Summary> templates;
  std::unordered_map<std::size_t,std::string> operation_signatures;
  std::deque<Template> definitions;
  [[maybe_unused]] std::size_t witnesses=0;
  static std::string key(const V& x) {
    std::string s=std::to_string(x.nodeId())+":"+std::to_string(x.resultIndex())+":"+std::to_string(x.instanceIndex());
    if(x.conversion())s+="<"+value_ir::typeConversionLabel(x)+":"+key(x.conversion()->source)+">";
    for(auto index:x.indices())s+="["+key(index)+"]";
    if(x.threadSelector())s+="{"+key(*x.threadSelector())+"}";
    // Pointer-annotated leaves are opaque here; retain their owned annotation
    // identity rather than conflating allocation properties with arithmetic.
    if(x.pointer())s+="|"+std::to_string(reinterpret_cast<std::uintptr_t>(x.pointer().get()));
    return s;
  }
  static std::string field(const std::string& s){return std::to_string(s.size())+":"+s;}
  static bool difference(std::int64_t x,std::int64_t y,std::int64_t& out) {
    return !__builtin_sub_overflow(x,y,&out);
  }
  Ptr mapped(V value,const Summary& child,std::int64_t base,std::int64_t stride,
             std::uint64_t period=0,std::int64_t outer_stride=0) {
    auto binding=std::make_shared<IndexedValueBinding>(IndexedValueBinding{value,base,stride,{}});
    binding->period=period;binding->outer_stride=outer_stride;
    return std::make_shared<U>(U{value,U::Kind::CoordinateMap,std::move(binding),{materialize(child.id)}});
  }
  Ptr materialize(std::size_t id) {
    // Candidate descriptions contain IDs, not allocated use subgraphs.
    // Only a selected source domain needs these nodes. Visit each retained
    // template/direct edge once, without recursion or logical-member loops.
    struct Frame {std::size_t id,next=0;};
    std::vector<Frame> work{{id}};
    while(!work.empty()) {
      auto& frame=work.back();auto& t=definitions.at(frame.id-1);
      if(t.materialized){work.pop_back();continue;}
      if(frame.next<t.children.size()) {
        const auto child=t.children[frame.next++].child;
        if(!definitions.at(child-1).materialized)work.push_back({child});
        continue;
      }
      std::vector<Ptr> children;children.reserve(t.children.size());
      for(const auto& c:t.children) {
        auto binding=std::make_shared<IndexedValueBinding>(IndexedValueBinding{c.value,c.base,c.stride,{}});
        binding->period=c.period;binding->outer_stride=c.outer;
        children.push_back(std::make_shared<U>(U{c.value,U::Kind::CoordinateMap,
            std::move(binding),{definitions.at(c.child-1).materialized}}));
      }
      t.materialized=std::make_shared<U>(U{t.prototype,t.kind,t.binding,std::move(children)});
      work.pop_back();
    }
    return definitions.at(id-1).materialized;
  }
  template<class MakeUse>
  Summary intern(std::string signature,MakeUse make_use,bool varying,std::int64_t coordinate,std::uint64_t row_bytes=0) {
    signature+=field("row-bytes:"+std::to_string(row_bytes));
    auto found=templates.find(signature);
    if(found!=templates.end())return {found->second.id,coordinate,varying,row_bytes};
    definitions.push_back(make_use());
    Summary result{definitions.size(),coordinate,varying,row_bytes};
    templates.emplace(std::move(signature),result);return result;
  }
  Summary exact(V value) {
    // Policy 3 is the first-varying-row planner. An invariant use must not
    // reuse that policy's parameterized summary for the same Value.
    const auto id=summaryKey(value,4);
    if(auto found=summaries.find(id);found!=summaries.end())return found->second;
    definitions.push_back(Template{value,U::Kind::Exact,{},{}});
    Summary result{definitions.size(),0,false,0};
    summaries.emplace(id,result);return result;
  }
  std::optional<IndexedSourceSlice> storageField(V value,std::uint64_t width,std::uint64_t offset=0) {
    namespace v=value_ir;
    for(unsigned depth=0;depth<64;++depth) {
      value=specialization.selected(g.canonicalInterpretation(value));
      const auto& type=g.type(value);
      if(value.pointer() || value.threadSelector() ||
          !type.hasStaticShape() || type.bitSize()%8 || offset>type.bitSize()/8 ||
          width>type.bitSize()/8-offset)return {};
      if(!value.indices().empty()) {
        if(value.conversion())return IndexedSourceSlice{value,offset,width};
        auto base=value.withoutSelection();const auto& shape=g.type(base).shape();
        if(shape.size()!=value.indices().size())return {};
        std::uint64_t index=0;
        for(std::size_t i=0;i<shape.size();++i) {
          const auto c=specialization.known(value.indices()[i]);if(!c)return {};
          std::uint64_t n;
          try {if(c->literal.empty() || c->literal[0]=='-')return {};n=std::stoull(c->literal);}catch(const std::exception&){return {};}
          if(n>=shape[i] || index>(UINT64_MAX-n)/shape[i])return {};
          index=index*shape[i]+n;
        }
        const auto bytes=type.bitSize()/8;
        if(index>(UINT64_MAX-offset)/bytes)return {};
        offset+=index*bytes;value=base;continue;
      }
      if(type.isMatrix() && !dynamic_cast<const v::LoadOp*>(&g.definingNode(value).operation()))
        return IndexedSourceSlice{value,offset,width};
      if(value.conversion()) {
        const auto& c=*value.conversion();const auto& src=g.type(c.source);
        const bool bits=c.spec->kind==v::TypeConversionKind::Bitcast && src.bitSize()==type.bitSize();
        const bool low=c.spec->kind==v::TypeConversionKind::Convert &&
            c.spec->attributes==v::ConversionAttributes{} && src.isScalar() &&
            src.element().isBitLike() && src.element().kind()!=v::ElementKind::SignedInteger &&
            type.element().kind()==v::ElementKind::UnsignedInteger && type.bitSize()<=src.bitSize();
        if(bits || low){value=c.source;continue;}
        // A numerical conversion is not a byte view. Its defining node still
        // names the pre-conversion Load/ShiftRight/etc.; following that node
        // here would replace a narrow input plus conversion by a wider read.
        // Leave the typed handle for the ordinary Conversion consumer.
        return IndexedSourceSlice{value,offset,width};
      }
      const auto& op=g.definingNode(value).operation();
      if(auto* binary=dynamic_cast<const v::BinaryOp*>(&op);
          binary && binary->opcode()==v::BinaryOpcode::ShiftRight &&
          type.element().kind()!=v::ElementKind::SignedInteger &&
          binary->attributes().rounding==v::RoundingMode::None &&
          binary->attributes().multiply_mode==v::IntegerMultiplyMode::Low &&
          !binary->attributes().saturate && !binary->attributes().flush_to_zero &&
          !binary->attributes().propagate_nan) {
        const auto args=g.instance(value).operands;const auto amount=specialization.known(args[1]);
        if(amount) {
          std::uint64_t shift=UINT64_MAX;
          try {shift=std::stoull(amount->literal);}catch(const std::exception&){}
          if(shift<type.bitSize() && shift%8==0 && offset+width<=(type.bitSize()-shift)/8) {
            offset+=shift/8;value=args[0];continue;
          }
        }
      }
      if(auto fields=storage.borrowComposed(value)) {
        bool moved=false;
        for(const auto& f:*fields)if(f.start%8==0 && f.width%8==0 &&
            offset>=f.start/8 && offset+width<=std::uint64_t(f.start+f.width)/8 && f.value!=value) {
          offset-=f.start/8;value=f.value;moved=true;break;
        }
        if(moved)continue;
      }
      if(dynamic_cast<const v::LoadOp*>(&op)) {
        // A symbolic initial read deliberately maps to itself. It is an
        // external-memory leaf, not a forwarding edge to chase repeatedly.
        if(memory.symbolic_initial_reads.contains({value.nodeId(),value.instanceIndex()}))
          return IndexedSourceSlice{value,offset,width};
        const auto* read=memory.readReplacement({value.nodeId(),value.instanceIndex()});
        if(read) {
          if(auto x=std::get_if<V>(read)){value=*x;continue;}
          if(auto x=std::get_if<ReinterpretedRead>(read))if(auto source=std::get_if<V>(&x->source)) {
            if(offset>UINT64_MAX-x->byte_offset)return {};
            offset+=x->byte_offset;value=*source;continue;
          }
          if(auto x=std::get_if<AssembledRead>(read)) {
            std::uint64_t begin=0;bool moved=false;
            for(const auto& piece:x->pieces) {
              const auto size=piece.type.bitSize()/8;
              if(offset>=begin && offset-begin<=size && width<=size-(offset-begin)) {
                if(auto source=std::get_if<V>(&piece.source)) {
                  if(offset-begin>UINT64_MAX-piece.byte_offset)return {};
                  offset=offset-begin+piece.byte_offset;value=*source;moved=true;
                }
                break;
              }
              if(begin>UINT64_MAX-size)return {};
              begin+=size;
            }
            if(moved)continue;
          }
        }
      }
      return IndexedSourceSlice{value,offset,width};
    }
    return {};
  }
  Plan plan(V value) {
    namespace v=value_ir;
    Plan p;
    if(value.pointer())return p;
    const auto& type=g.type(value);
    if(!type.isScalar() || type.element().isParameterized() || !type.bitSize())return p;
    const auto selected=specialization.selected(value);
    if(selected!=value) {
      p.form=Form::Forward;p.children={selected};p.signature="selected:"+field(type.toString());return p;
    }
    if(value.threadSelector())return p;
    if(!value.conversion() && value.indices().empty())
      if(const auto* coordinate=dynamic_cast<const v::CoordinateOp*>(&g.definingNode(value).operation())) {
        const auto origin=g.definingNode(value).instanceOrigin(value.instanceIndex());
        if(!origin.coordinate_value || *origin.coordinate_value>std::uint64_t(INT64_MAX))return p;
        p.form=Form::Leaf;p.coordinate=*origin.coordinate_value;p.row_bytes=1;
        p.signature="coordinate:"+field(std::to_string(unsigned(coordinate->kind())))+
            field(std::to_string(unsigned(coordinate->axis())))+field(type.toString());
        p.leaf=IndexedValueBinding{value,0,1,{}};
        // Thread coordinates parameterize spatial members. CTA coordinates
        // are launch-bound invariants within a source template, not another
        // copy of the thread axis. Include their value in the identity so
        // different CTAs cannot silently acquire the same binding.
        if(coordinate->kind()!=v::CoordinateKind::ThreadIndex) {
          p.signature+=field(std::to_string(p.coordinate));
          p.leaf->base=p.coordinate;p.leaf->stride=0;
          p.coordinate=0;p.row_bytes=0;p.varying=false;
        }
        return p;
      }
    // Preserve exact storage selection before generic conversion propagation.
    // A low/high word extraction denotes a byte field, not two different
    // mathematical operations. Only bit-preserving operations are traversed.
    // A plain floating arithmetic result is already one numeric field. The
    // storage decoder would only cache that same handle, allocating a field
    // vector for every arithmetic instance. Only storage-bearing syntax can
    // change this result; retain its full decoding path.
    const bool storage_syntax=value.conversion() || !value.indices().empty() ||
        type.element().isBitLike() || dynamic_cast<const v::LoadOp*>(&g.definingNode(value).operation());
    if(storage_syntax && type.bitSize()%8==0)if(auto slice=storageField(value,type.bitSize()/8)) {
      const auto& t=g.type(slice->value);
      if(t.isMatrix() && slice->offset<=std::uint64_t(INT64_MAX)) {
        // The immutable read-from map already identifies initial bytes.
        // Do not turn separate intermediate Load nodes back into unrelated
        // matrix producers and rediscover their relation in projection.
        if(auto initial=memory.initialRead({slice->value.nodeId(),slice->value.instanceIndex()});
            initial && slice->offset<=initial->region.bytes &&
            type.bitSize()/8<=initial->region.bytes-slice->offset &&
            initial->region.offset<=std::uint64_t(INT64_MAX)-slice->offset) {
          p.form=Form::Leaf;p.coordinate=initial->region.offset+slice->offset;
          p.signature="initial:"+field(initial->region.allocation)+field(type.toString());
          // Replacing a matrix producer by its immutable initial allocation
          // must retain the producer's coordinate stride. Otherwise row-wise
          // operands become unrelated scalar offsets in the parent template.
          if(t.hasStaticShape() && t.shape().size()==2 && t.element().width()>=8 &&
              t.element().width()%8==0 &&
              t.shape().back()<=std::uint64_t(INT64_MAX)/(t.element().width()/8))
            p.row_bytes=t.shape().back()*(t.element().width()/8);
          p.leaf=IndexedValueBinding{value,0,1,*initial};return p;
        }
        p.form=Form::Leaf;p.coordinate=slice->offset;
        p.signature="producer:"+field(key(slice->value))+field(type.toString());
        p.leaf=IndexedValueBinding{value,0,1,{}};p.leaf->producer=slice->value;return p;
      }
      if(slice->value!=value && slice->offset==0 && t.bitSize()==type.bitSize()) {
        p.form=Form::Forward;p.children={slice->value};p.signature="field:"+field(type.toString());return p;
      }
    }
    if(!value.indices().empty() && !value.conversion()) {
      auto base=value.withoutSelection();const auto& t=g.type(base);
      if(!t.hasStaticShape() || t.shape().size()!=value.indices().size())return p;
      std::uint64_t index=0;
      for(std::size_t i=0;i<t.shape().size();++i) {
        auto x=g.resolve(value.indices()[i]);
        auto c=dynamic_cast<const v::ConstantOp*>(&g.definingNode(x).operation());
        if(!c || x.conversion() || x.pointer() || x.threadSelector() || !x.indices().empty())return p;
        std::size_t used=0;std::uint64_t n;
        try {if(c->literal().empty() || c->literal()[0]=='-')return p;n=std::stoull(c->literal(),&used,0);}
        catch(const std::exception&){return p;}
        if(used!=c->literal().size() || n>=t.shape()[i] || index>(UINT64_MAX-n)/t.shape()[i])return p;
        index=index*t.shape()[i]+n;
      }
      const auto width=type.bitSize()/8;
      if(index>std::uint64_t(INT64_MAX)/width)return p;
      p.form=Form::Leaf;p.coordinate=index*width;
      p.signature="producer:"+field(key(base))+field(type.toString());
      p.leaf=IndexedValueBinding{value,0,1,{}};p.leaf->producer=base;return p;
    }
    if(value.conversion()) {
      p.form=Form::Conversion;p.children={g.canonicalInterpretation(value.conversion()->source)};
      p.signature="conversion:"+field(value_ir::typeConversionLabel(value));return p;
    }
    const auto& op=g.definingNode(value).operation();
    if(const auto* constant=dynamic_cast<const v::ConstantOp*>(&op);
        constant && !constant->pointer() && type.element().kind()==v::ElementKind::UnsignedInteger &&
        type.bitSize()<=64) {
      // Literal coordinates are exact stored facts, just like resolved byte
      // offsets. Keep their type; the domain mapper validates every member.
      // Unsupported spellings and values outside the signed coordinate range
      // remain ordinary invariant constants.
      const auto& literal=constant->literal();std::uint64_t n=0;
      const auto [end,error]=std::from_chars(literal.data(),literal.data()+literal.size(),n);
      if(error==std::errc{} && end==literal.data()+literal.size() && n<=std::uint64_t(INT64_MAX) &&
          (type.bitSize()==64 || n<(std::uint64_t(1)<<type.bitSize()))) {
        p.form=Form::Literal;p.coordinate=std::int64_t(n);
        p.signature="unsigned-literal:"+field(type.toString());
        p.leaf=IndexedValueBinding{value,0,1,{}};return p;
      }
    }
    if(const auto* parameter=dynamic_cast<const v::InputParameterOp*>(&op);
        parameter && !parameter->pointer()) {
      // Scalar lowering already treats this launch-bound parameter as a
      // literal. Distinct ld.param occurrences must not hide that invariant
      // from template planning. Unbound parameters use their symbolic identity,
      // just like Graph::inputParameter in scalar lowering; no property of their
      // unknown value is inferred. Memory loads and pointers are not parameters.
      p.form=Form::Operands;
      p.signature=parameter->specializedLiteral()?
          "specialized-parameter:"+field(type.toString())+field(*parameter->specializedLiteral()):
          "symbolic-parameter:"+field(type.toString())+field(parameter->symbol());
      return p;
    }
    if(dynamic_cast<const v::LoadOp*>(&op)) {
      // Preserve the load/address/occurrence identity. No concrete initial
      // offset or relationship between separate dynamic reads is inferred.
      if(memory.symbolic_initial_reads.contains({value.nodeId(),value.instanceIndex()}))return p;
      const auto* replacement=memory.readReplacement({value.nodeId(),value.instanceIndex()});
      if(!replacement)return p;
      std::optional<V> source;std::uint64_t offset=0;
      const InitialMemorySymbol* initial=nullptr;
      if(auto x=std::get_if<V>(replacement))source=*x;
      else if(auto x=std::get_if<InitialMemorySymbol>(replacement))initial=x;
      else if(auto x=std::get_if<ReinterpretedRead>(replacement)) {
        offset=x->byte_offset;
        if(auto a=std::get_if<V>(&x->source))source=*a;
        else initial=&std::get<InitialMemorySymbol>(x->source);
      } else return p;
      if(initial) {
        if(offset>initial->region.bytes || type.bitSize()/8>initial->region.bytes-offset ||
           initial->region.offset>std::uint64_t(INT64_MAX) || offset>std::uint64_t(INT64_MAX)-initial->region.offset)return p;
        p.form=Form::Leaf;p.coordinate=initial->region.offset+offset;
        p.signature="initial:"+field(initial->region.allocation)+field(type.toString());
        p.leaf=IndexedValueBinding{value,0,1,*initial};return p;
      }
      if(source) {
        *source=g.canonicalInterpretation(*source);const auto& t=g.type(*source);
        if(!t.hasStaticShape() || offset>t.bitSize()/8 || type.bitSize()/8>t.bitSize()/8-offset)return p;
        if(t.isScalar() && offset==0 && t.bitSize()==type.bitSize()) {
          p.form=Form::Forward;p.children={*source};p.signature="forward:"+field(type.toString());return p;
        }
        if(t.isMatrix() && offset<=std::uint64_t(INT64_MAX)) {
          p.form=Form::Leaf;p.coordinate=offset;
          p.signature="producer:"+field(key(*source))+field(type.toString());
          p.leaf=IndexedValueBinding{value,0,1,{}};p.leaf->producer=*source;return p;
        }
      }
      return p;
    }
    const bool pure=dynamic_cast<const v::UnaryOp*>(&op) || dynamic_cast<const v::BinaryOp*>(&op) ||
        dynamic_cast<const v::FmaOp*>(&op) || dynamic_cast<const v::IdealExp2Op*>(&op) ||
        dynamic_cast<const v::CompareOp*>(&op) || dynamic_cast<const v::SelectOp*>(&op) ||
        dynamic_cast<const v::ConstantOp*>(&op) ||
        (dynamic_cast<const v::PtxScalarOp*>(&op) && static_cast<const v::PtxScalarOp&>(op).opcode()>=v::PtxScalarOpcode::Sin &&
         !static_cast<const v::PtxScalarOp&>(op).attributes().carry_in &&
         !static_cast<const v::PtxScalarOp&>(op).attributes().carry_out);
    if(!pure || op.resultTypes().size()!=1)return p;
    p.form=Form::Operands;
    if(isIdealExp2(op)) {
      // Use the same admitted abstraction as indexed-source matching and
      // scalar projection; hardware/software spellings are not extra domains.
      p.signature="op:"+field("Exp2")+field("ideal_arithmetic")+field(type.toString());
      for(auto x:g.instance(value).operands)p.children.push_back(g.canonicalInterpretation(x));
      return p;
    }
    auto found=operation_signatures.find(value.nodeId());
    if(found==operation_signatures.end()) {
      std::ostringstream attrs;op.printAttributes(attrs);
      found=operation_signatures.emplace(value.nodeId(),field(std::string(op.name()))+field(attrs.str())).first;
    }
    p.signature="op:"+found->second+field(type.toString());
    for(auto x:g.instance(value).operands)p.children.push_back(g.canonicalInterpretation(x));
    return p;
  }
  static SummaryKey summaryKey(const V& value,unsigned policy) {
    return {value,policy};
  }
  Summary build(V value,const Plan& p,unsigned policy) {
    if(p.form==Form::Exact)return exact(value);
    // A proved selection/read forwarding is an identity, not a new template
    // shape. Reuse its already compiled summary so direct and forwarded uses
    // share coordinates. Equal storage width alone is not a type identity.
    if(p.form==Form::Forward && p.children.size()==1 && g.type(value)==g.type(p.children.front()))
      return summaries.at(summaryKey(p.children.front(),policy));
    if(p.form==Form::Literal && policy!=5)
      return intern(p.signature+field(std::to_string(p.coordinate)),
          [&]{return Template{value,U::Kind::Operands,{},{}};},false,0);
    if(p.form==Form::Leaf || p.form==Form::Literal) {
      std::uint64_t row_bytes=p.row_bytes;
      if(p.leaf->producer) {
        const auto& type=g.type(*p.leaf->producer);
        if(type.hasStaticShape() && type.shape().size()==2 && type.element().width()>=8 && type.element().width()%8==0 &&
           type.shape().back()<=std::uint64_t(INT64_MAX)/(type.element().width()/8))
          row_bytes=type.shape().back()*(type.element().width()/8);
      }
      return intern(p.signature,[&]{return Template{value,U::Kind::Binding,
          std::make_shared<IndexedValueBinding>(*p.leaf),{}};},p.varying,p.coordinate,row_bytes);
    }
    bool varying=false;std::int64_t coordinate=0;std::size_t anchor=p.children.size();std::uint64_t row_bytes=0;
    for(std::size_t i=0;i<p.children.size();++i) {
      const auto& s=summaries.at(summaryKey(p.children[i],policy));
      if(s.varying && (anchor==p.children.size() || (policy==2 && s.row_bytes>row_bytes))) {
        varying=true;coordinate=s.coordinate;row_bytes=s.row_bytes;anchor=i;
        if(policy!=2)break;
      }
    }
    auto signature=p.signature;std::vector<ChildMap> maps;
    for(std::size_t i=0;i<p.children.size();++i) {
      const auto child=p.children[i];
      // A whole immutable subtree may be invariant in this use even though
      // another use can parameterize it. Exact identity preserves that fact;
      // never substitute the parent's coordinate into the invariant subtree.
      const auto& compiled=summaries.at(summaryKey(child,policy));
      const auto s=policy==1 && i!=anchor && compiled.varying?exact(child):compiled;
      std::int64_t delta=0,stride=s.varying?1:0,outer=0;std::uint64_t period=0;
      if(policy>=2 && i!=anchor && s.varying && row_bytes && s.row_bytes && coordinate>=0 &&
         s.id!=summaries.at(summaryKey(p.children[anchor],policy)).id) {
        // A row-dependent operand can share a flattened matrix coordinate.
        // Validate its residual at every supplied value; this is a coordinate
        // map, not an assumption that arbitrary operands are row invariant.
        std::int64_t origin;
        if(__builtin_mul_overflow(coordinate/std::int64_t(row_bytes),std::int64_t(s.row_bytes),&origin) ||
           !difference(s.coordinate,origin,delta))return exact(value);
        period=row_bytes;outer=s.row_bytes;stride=0;
      } else if(s.varying && !difference(s.coordinate,coordinate,delta))return exact(value);
      signature+=field(std::to_string(s.id))+field(std::to_string(delta))+field(s.varying?"varying":"fixed")+
          field(std::to_string(period))+field(std::to_string(outer));
      maps.push_back({child,s.id,s.varying?delta:0,stride,outer,period});
    }
    // Look up the structural template before allocating its immutable child
    // maps. Repeated instances carry coordinates, not duplicate child DAGs.
    signature+=field("row-bytes:"+std::to_string(row_bytes));
    if(auto found=templates.find(signature);found!=templates.end())
      return {found->second.id,coordinate,varying,row_bytes};
    auto kind=p.form==Form::Forward?U::Kind::Forward:p.form==Form::Conversion?U::Kind::Conversion:U::Kind::Operands;
    definitions.push_back(Template{value,kind,{},std::move(maps)});
    Summary result{definitions.size(),coordinate,varying,row_bytes};
    templates.emplace(std::move(signature),result);return result;
  }
  Summary summarize(V value,unsigned policy) {
    value=g.canonicalInterpretation(value);
    if(auto found=summaries.find(summaryKey(value,policy));found!=summaries.end())return found->second;
    struct Frame {V value;Plan plan;std::size_t next=0;};
    std::vector<Frame> work{{value,plan(value)}};std::unordered_set<SummaryKey,SummaryHash> active{summaryKey(value,policy)};
    while(!work.empty()) {
      auto& f=work.back();
      if(f.next<f.plan.children.size()) {
        auto child=f.plan.children[f.next++];const auto id=summaryKey(child,policy);
        if(summaries.contains(id))continue;
        if(active.contains(id)) {f.plan=Plan{};continue;}
        active.insert(id);work.push_back({child,plan(child)});continue;
      }
      const auto id=summaryKey(f.value,policy);auto result=build(f.value,f.plan,policy);
      summaries.emplace(id,std::move(result));active.erase(id);work.pop_back();
    }
    return summaries.at(summaryKey(value,policy));
  }
 public:
  // Test-only observation; production planning never scans the arena.
  std::size_t materializedTemplateCountForTesting() const {
    std::size_t count=0;
    for(const auto& definition:definitions)count+=bool(definition.materialized);
    return count;
  }
  // Exact query normalization shared with scalar projection. No source
  // matching, numerical rewriting, or logical-instance materialization.
  std::optional<IndexedSourceSlice> storageSlice(V value,std::uint64_t offset,std::uint64_t bytes) {
    return storageField(value,bytes,offset);
  }
  PureSourceTemplates(const value_ir::Graph& graph,const MemoryEliminationResult& m)
      :g(graph),memory(m),specialization(graph),storage(graph){}
  std::vector<IndexedSourceDomain> mapStrategy(std::span<const V> values,bool interleaved,
                                     std::span<const std::uint64_t> offsets,unsigned policy,
                                     const std::vector<IndexedSourceDomain>& members) {
    if(!offsets.empty() && offsets.size()!=values.size())throw std::invalid_argument("source offsets do not cover values");
    // No pure-operation propagation can reduce an already-single source
    // domain. Keep its existing template instead of visiting its ancestors.
    if(members.size()<=1)return members;
    struct Run {std::size_t begin,last,count=1,step=1;Summary summary;std::int64_t last_coordinate,stride=0;};
    std::map<std::pair<std::size_t,std::uint64_t>,Run> active;
    std::vector<IndexedSourceDomain> result;
    const auto emit=[&](const Run& run) {
      if(run.count==1){result.push_back({run.begin,1,{}});return;}
      auto root=mapped(definitions.at(run.summary.id-1).prototype,run.summary,run.summary.coordinate,run.stride);
      result.push_back({run.begin,run.count,std::make_shared<IndexedSource>(
          IndexedSource{root->prototype,run.count,std::move(root)}),run.step});
    };
    for(std::size_t i=0;i<values.size();++i) {
      auto summary=summarize(values[i],policy);const auto identity=std::pair{summary.id,offsets.empty()?0:offsets[i]};
      SENTRYWARD_AUDIT((void)audit;const auto minimum=sentryward::debug::environment("SENTRYWARD_PURE_TEMPLATE_MIN_NODE");
        const auto maximum=sentryward::debug::environment("SENTRYWARD_PURE_TEMPLATE_MAX_NODE");
        if(sentryward::debug::environment("SENTRYWARD_PURE_TEMPLATE_WITNESS") && witnesses<4 &&
          (!minimum || values[i].nodeId()>=std::strtoull(minimum,nullptr,10)) &&
          (!maximum || values[i].nodeId()<=std::strtoull(maximum,nullptr,10)) &&
          (minimum || maximum || g.definingNode(values[i]).operation().name()=="PtxExp2")) {
        ++witnesses;
        std::vector<std::pair<V,unsigned>> todo{{g.canonicalInterpretation(values[i]),0}};
        for(std::size_t n=0;n<todo.size() && n<96;++n) {
          const auto [value,depth]=todo[n];const auto found=summaries.find(summaryKey(value,policy));
          if(found==summaries.end())continue;
          const auto p=plan(value);const auto& s=found->second;
          SENTRYWARD_DEBUG_PRINT(pure_source_templates_0);
          if(depth<10)for(auto child:p.children)todo.emplace_back(child,depth+1);
        }
      });
      if(!interleaved && !active.empty() && active.begin()->first!=identity) {emit(active.begin()->second);active.clear();}
      auto found=active.find(identity);
      if(found==active.end()){active.emplace(identity,Run{i,i,1,1,summary,summary.coordinate,0});continue;}
      auto& run=found->second;std::int64_t delta;
      if(!difference(summary.coordinate,run.last_coordinate,delta) ||
         (run.count>1 && (i-run.last!=run.step || delta!=run.stride))) {
        emit(run);run=Run{i,i,1,1,summary,summary.coordinate,0};continue;
      }
      run.step=i-run.last;run.stride=delta;run.last=i;run.last_coordinate=summary.coordinate;++run.count;
    }
    for(const auto& [id,run]:active){(void)id;emit(run);}
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.begin<b.begin;});
    SENTRYWARD_AUDIT({
      (void)audit;
      const auto minimum=sentryward::debug::environment("SENTRYWARD_SOURCE_PLAN_MIN_NODE");
      const auto maximum=sentryward::debug::environment("SENTRYWARD_SOURCE_PLAN_MAX_NODE");
      if(policy>=2 && values.size()>256 && sentryward::debug::environment("SENTRYWARD_SOURCE_PLAN_WITNESS") && witnesses<2 &&
         (!minimum || values.front().nodeId()>=std::strtoull(minimum,nullptr,10)) &&
         (!maximum || values.front().nodeId()<=std::strtoull(maximum,nullptr,10))) {
        ++witnesses;
        std::vector<std::pair<V,V>> pairs{{g.canonicalInterpretation(values[0]),g.canonicalInterpretation(values[256])}};
        for(std::size_t n=0;n<pairs.size() && n<96;++n) {
          const auto [a,b]=pairs[n];
          const auto x=summaries.find(summaryKey(a,policy)),y=summaries.find(summaryKey(b,policy));
          if(x==summaries.end() || y==summaries.end() || x->second.id==y->second.id)continue;
          const auto ap=plan(a),bp=plan(b);
          SENTRYWARD_DEBUG_PRINT(pure_source_templates_1);
          if(ap.children.size()==bp.children.size())
            for(std::size_t i=0;i<ap.children.size();++i)pairs.emplace_back(ap.children[i],bp.children[i]);
        }
        std::vector<std::pair<V,unsigned>> todo{{g.canonicalInterpretation(values.front()),0}};
        for(std::size_t n=0;n<todo.size() && n<128;++n) {
          const auto [value,depth]=todo[n];const auto found=summaries.find(summaryKey(value,policy));
          if(found==summaries.end())continue;
          const auto p=plan(value);const auto& s=found->second;
          SENTRYWARD_DEBUG_PRINT(pure_source_templates_2);
          if(depth<20)for(auto child:p.children)todo.emplace_back(child,depth+1);
        }
      }
    });
    // Existing member templates may represent a domain that this one-coordinate
    // math summary cannot express. Reuse that exact mapping, not a tree matcher.
    if(planCost(members)<planCost(result))return members;
    return result;
  }
  std::vector<IndexedSourceDomain> map(std::span<const V> values,bool interleaved=false,
                                     std::span<const std::uint64_t> offsets={}) {
    // Four bounded whole-DAG policies, not combinations of arbitrary tree matches.
    // Joint coordinates support A(i)+B(i); fixed-tail supports f(A(i), bias);
    // row maps support A(row,col)+bias(row) with producer-defined row widths.
    // Each policy visits each Value/direct edge once and shares immutable uses.
    // Member-domain planning is identical for all policies. Build it once,
    // rather than repeating annotation formatting/grouping and use allocation.
    const auto members=mapSourceTemplates(g,values,interleaved,offsets);
    if(members.size()<=1)return members;
    // Producer row width is a stride, not evidence of which operand varies
    // within the output row. A row reduction can retain a wider producer than
    // the pointwise tensor it scales. Try the first varying operand as anchor;
    // every supplied source still validates the resulting coordinate relation.
    auto first_rows=mapStrategy(values,interleaved,offsets,3,members);
    if(first_rows.size()<=1)return first_rows;
    auto rows=mapStrategy(values,interleaved,offsets,2,members);
    if(planCost(first_rows)<planCost(rows))rows=std::move(first_rows);
    if(rows.size()<=1)return rows;
    // A complete dense-run plan already improves the member-only storage.
    // Keep that certified plan rather than rebuilding two alternative summary
    // DAGs solely to search for a possibly smaller presentation. Irregular
    // or singleton plans still use the remaining bounded strategies.
    if(planCost(rows)<planCost(members) && std::all_of(rows.begin(),rows.end(),
        [](const auto& d){return d.source && d.count>1 && d.result_stride==1;}))return rows;
    auto fixed=mapStrategy(values,interleaved,offsets,true,members);
    if(fixed.size()<=1)return fixed;
    auto joint=mapStrategy(values,interleaved,offsets,false,members);
    auto result=planCost(joint)<planCost(fixed)?std::move(joint):std::move(fixed);
    if(result.size()>1 && planCost(rows)<planCost(result))result=std::move(rows);
    // Constant coordinates are a bounded fallback for otherwise uncompressed
    // unsigned expressions (including scalar words later sliced/repacked).
    // Keeping literals invariant in the ordinary policies preserves their
    // existing floating-point template anchors and sharing.
    if(std::all_of(result.begin(),result.end(),[](const auto& d){return d.count==1;}) &&
        std::all_of(values.begin(),values.end(),[&](V x) {
          return g.type(x).element().kind()==value_ir::ElementKind::UnsignedInteger;
        })) {
      auto literals=mapStrategy(values,interleaved,offsets,5,members);
      if(planCost(literals)<planCost(result))return literals;
    }
    return result;
  }
};
}
