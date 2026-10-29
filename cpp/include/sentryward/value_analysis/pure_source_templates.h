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
#include <algorithm>
#include <array>
#include <map>
#include <charconv>
#include <deque>
#include <span>
#include <stdexcept>
#include <string_view>
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
  // Policies 0-3 and 5 are map strategies; 4 is the shared exact identity.
  static constexpr unsigned policies=6;
  // One record per distinct annotated Value (deep Value==) owns its summary
  // slots. Its first handle is only a lookup key: plans and templates keep the
  // caller's own handles, whose annotation identities the caches key on.
  struct Record {
    std::size_t active=0;                       // summarize() visit stamp
    std::array<std::size_t,policies> summary{}; // 1 + summary_pool index; 0 absent
  };
  struct Child {V value;Record* record=nullptr;};
  struct Plan {
    Form form=Form::Exact;
    std::vector<Child> children;
    std::size_t signature=0; // interned signature text; unused by Exact
    std::optional<IndexedValueBinding> leaf;
    std::int64_t coordinate=0;
    std::uint64_t row_bytes=0;
    bool varying=true;
  };
  const value_ir::Graph& g;
  const MemoryEliminationResult& memory;
  // The pass's shared AssembledRead extents; null scans the pieces instead.
  AssembledPieceEnds* assembled_ends;
  ScalarSpecialization specialization;
  PackedStorageDecoder storage;
  bool lift_block_index=false;
  struct ValueHash {
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
    // Not noexcept: libstdc++ then keeps this recursive hash in each node, so
    // bucket walks and rehashing compare stored codes instead of recomputing.
    std::size_t operator()(const V& value) const {return valueHash(value);}
  };
  std::unordered_map<V,Record,ValueHash> records;
  std::deque<Summary> summary_pool; // stable references while exact() appends
  // A template key is a path tag (0 leaf, 1 literal, 2 operands), the interned
  // plan signature and fixed-position integer fields. Two keys are equal
  // exactly when their tags, signatures and fields are.
  struct TemplateKey {std::size_t hash;std::vector<std::uint64_t> words;};
  struct TemplateProbe {std::size_t hash;std::span<const std::uint64_t> words;};
  struct TemplateHash {
    using is_transparent=void;
    template<class K> std::size_t operator()(const K& key) const noexcept {return key.hash;}
  };
  struct TemplateEqual {
    using is_transparent=void;
    template<class A,class B> bool operator()(const A& a,const B& b) const noexcept {
      return a.hash==b.hash && std::equal(a.words.begin(),a.words.end(),b.words.begin(),b.words.end());
    }
  };
  std::unordered_map<TemplateKey,std::size_t,TemplateHash,TemplateEqual> templates;
  std::unordered_map<std::string,std::size_t> plan_signatures;
  std::unordered_map<std::size_t,std::string> operation_signatures;
  std::deque<Template> definitions;
  // Reused scratch; plan(), build() and summarize() are not reentrant.
  std::string signature_text,key_text;
  std::vector<std::uint64_t> key_words;
  struct PendingMap {std::size_t index,child;std::int64_t base,stride,outer;std::uint64_t period;};
  std::vector<PendingMap> pending_maps;
  // A frame's Value is its parent's plan child (or the root), never a copy.
  struct SummaryFrame {Record* record;std::size_t parent,child;Plan plan;std::size_t next=0;};
  std::vector<SummaryFrame> frames;
  std::size_t active_stamp=0;
  [[maybe_unused]] std::size_t witnesses=0;
  static std::size_t wordsHash(std::span<const std::uint64_t> words) noexcept {
    std::uint64_t h=words.size();
    for(const auto word:words){h^=word;h*=0x9e3779b97f4a7c15ULL;h^=h>>29;}
    return std::size_t(h);
  }
  // Same decimal digits as std::to_string, appended without temporaries.
  static void appendNumber(std::string& out,auto n) {
    char digits[24];out.append(digits,std::to_chars(digits,digits+sizeof digits,n).ptr);
  }
  static void appendField(std::string& out,std::string_view text) {
    appendNumber(out,text.size());out+=':';out.append(text);
  }
  static void appendNumberField(std::string& out,auto n) {
    char digits[24];const auto end=std::to_chars(digits,digits+sizeof digits,n).ptr;
    appendField(out,{digits,std::size_t(end-digits)});
  }
  static void appendKey(std::string& out,const V& x) {
    appendNumber(out,x.nodeId());out+=':';appendNumber(out,x.resultIndex());out+=':';appendNumber(out,x.instanceIndex());
    if(x.conversion()) {
      out+='<';out+=value_ir::typeConversionLabel(x);out+=':';appendKey(out,x.conversion()->source);out+='>';
    }
    for(const auto& index:x.indices()){out+='[';appendKey(out,index);out+=']';}
    if(x.threadSelector()){out+='{';appendKey(out,*x.threadSelector());out+='}';}
    // Pointer-annotated leaves are opaque here; retain their owned annotation
    // identity rather than conflating allocation properties with arithmetic.
    // A template retains the producer named by its key, so a stored address
    // always denotes a live annotation: equality is identity, not reuse.
    if(x.pointer()){out+='|';appendNumber(out,reinterpret_cast<std::uintptr_t>(x.pointer().get()));}
  }
  void appendKeyField(const V& x) {
    key_text.clear();appendKey(key_text,x);appendField(signature_text,key_text);
  }
  std::size_t signatureId() {
    return plan_signatures.try_emplace(signature_text,plan_signatures.size()).first->second;
  }
  Record& record(const V& value) {return records.try_emplace(value).first->second;}
  const Summary* summaryOf(const Record& r,unsigned policy) const {
    return r.summary[policy]?&summary_pool[r.summary[policy]-1]:nullptr;
  }
  const Summary& summaryAt(const Record& r,unsigned policy) const {
    if(!r.summary[policy])throw std::out_of_range("pure source summary is absent");
    return summary_pool[r.summary[policy]-1];
  }
  // The first stored summary of a policy stays; a later store returns it.
  const Summary& store(Record& r,unsigned policy,const Summary& s) {
    if(!r.summary[policy]){summary_pool.push_back(s);r.summary[policy]=summary_pool.size();}
    return summary_pool[r.summary[policy]-1];
  }
  // Audit-only lookup; it never inserts a record.
  const Summary* findSummary(const V& value,unsigned policy) const {
    const auto found=records.find(value);
    return found==records.end()?nullptr:summaryOf(found->second,policy);
  }
  static bool difference(std::int64_t x,std::int64_t y,std::int64_t& out) {
    return !__builtin_sub_overflow(x,y,&out);
  }
  Ptr mapped(const V& value,const Summary& child,std::int64_t base,std::int64_t stride,
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
  Summary intern(std::span<const std::uint64_t> key,MakeUse make_use,bool varying,std::int64_t coordinate,std::uint64_t row_bytes=0) {
    const auto hash=wordsHash(key);
    if(auto found=templates.find(TemplateProbe{hash,key});found!=templates.end())
      return {found->second,coordinate,varying,row_bytes};
    definitions.push_back(make_use());
    templates.emplace(TemplateKey{hash,{key.begin(),key.end()}},definitions.size());
    return {definitions.size(),coordinate,varying,row_bytes};
  }
  Summary exact(const V& value,Record& r) {
    // Policy 3 is the first-varying-row planner. An invariant use must not
    // reuse that policy's parameterized summary for the same Value.
    if(const auto* found=summaryOf(r,4))return *found;
    definitions.push_back(Template{value,U::Kind::Exact,{},{}});
    return store(r,4,{definitions.size(),0,false,0});
  }
  std::optional<IndexedSourceSlice> storageField(V value,std::uint64_t width,std::uint64_t offset=0) {
    namespace v=value_ir;
    for(unsigned depth=0;depth<64;++depth) {
      value=specialization.selected(g.canonicalInterpretation(value));
      const auto& type=g.type(value);
      if(value.pointer() || value.threadSelector() || !type.hasStaticShape())return {};
      // One out-of-line size per step (type is immutable); it throws for a
      // parameterized element.
      const auto bits=type.bitSize();
      if(bits%8 || offset>bits/8 || width>bits/8-offset)return {};
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
        const auto bytes=bits/8;
        if(index>(UINT64_MAX-offset)/bytes)return {};
        offset+=index*bytes;value=base;continue;
      }
      if(type.isMatrix() && !dynamic_cast<const v::LoadOp*>(&g.definingNode(value).operation()))
        return IndexedSourceSlice{value,offset,width};
      if(value.conversion()) {
        const auto& c=*value.conversion();const auto& src=g.type(c.source);
        const bool same_bits=c.spec->kind==v::TypeConversionKind::Bitcast && src.bitSize()==bits;
        const bool low=c.spec->kind==v::TypeConversionKind::Convert &&
            c.spec->attributes==v::ConversionAttributes{} && src.isScalar() &&
            src.element().isBitLike() && src.element().kind()!=v::ElementKind::SignedInteger &&
            type.element().kind()==v::ElementKind::UnsignedInteger && bits<=src.bitSize();
        if(same_bits || low){value=c.source;continue;}
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
        const auto args=g.instanceOperands(value);const auto amount=specialization.known(args[1]);
        if(amount) {
          std::uint64_t shift=UINT64_MAX;
          try {shift=std::stoull(amount->literal);}catch(const std::exception&){}
          if(shift<bits && shift%8==0 && offset+width<=(bits-shift)/8) {
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
            bool moved=false;
            // A zero-width query also matches a piece ending at offset; scan it.
            if(const auto* ends=(width && assembled_ends)?assembled_ends->reusedEnds(*x):nullptr) {
              // All ends are representable, so the scan below could match only
              // the first piece ending after offset, and nothing else returns.
              const auto it=std::upper_bound(ends->begin(),ends->end(),offset);
              if(it!=ends->end() && width<=*it-offset) {
                const auto i=static_cast<std::size_t>(it-ends->begin());
                const auto begin=i?(*ends)[i-1]:0;const auto& piece=x->pieces[i];
                if(auto source=std::get_if<V>(&piece.source)) {
                  if(offset-begin>UINT64_MAX-piece.byte_offset)return {};
                  offset=offset-begin+piece.byte_offset;value=*source;moved=true;
                }
              }
            } else {
              std::uint64_t begin=0;
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
            }
            if(moved)continue;
          }
        }
      }
      return IndexedSourceSlice{value,offset,width};
    }
    return {};
  }
  // Policy independent. The signature text is interned once per plan, and each
  // child once here, not hashed again by every summary query of its parent.
  Plan plan(const V& value) {
    signature_text.clear();
    auto p=planForm(value);
    if(p.form!=Form::Exact)p.signature=signatureId();
    for(auto& child:p.children)child.record=&record(child.value);
    return p;
  }
  Plan planForm(const V& value) {
    namespace v=value_ir;
    Plan p;
    if(value.pointer())return p;
    const auto& type=g.type(value);
    if(!type.isScalar() || type.element().isParameterized() || !type.bitSize())return p;
    auto selected=specialization.selected(value);
    if(selected!=value) {
      p.form=Form::Forward;p.children.push_back({std::move(selected)});
      signature_text="selected:";appendField(signature_text,type.toString());return p;
    }
    if(value.threadSelector())return p;
    // At most one definingNode lookup, made on first use; the graph is immutable.
    const v::Node* defining=nullptr;
    const auto node=[&]()->const v::Node& {if(!defining)defining=&g.definingNode(value);return *defining;};
    if(!value.conversion() && value.indices().empty())
      if(const auto* coordinate=dynamic_cast<const v::CoordinateOp*>(&node().operation())) {
        // Lifting mode: a BlockIndex leaf is the unit parameter, not a value
        // of this CTA; decline it so the generic path lowers the parameter.
        if(coordinate->kind()==v::CoordinateKind::BlockIndex) {
          NoteUnitSite(lift_block_index?UnitSite::FoldTemplateLeaf:UnitSite::ValueTemplateLeaf);
          if(lift_block_index)return p;
        }
        const auto origin=node().instanceOrigin(value.instanceIndex());
        if(!origin.coordinate_value || *origin.coordinate_value>std::uint64_t(INT64_MAX))return p;
        p.form=Form::Leaf;p.coordinate=*origin.coordinate_value;p.row_bytes=1;
        signature_text="coordinate:";appendNumberField(signature_text,unsigned(coordinate->kind()));
        appendNumberField(signature_text,unsigned(coordinate->axis()));appendField(signature_text,type.toString());
        p.leaf=IndexedValueBinding{value,0,1,{}};
        // Thread coordinates parameterize spatial members. CTA coordinates
        // are launch-bound invariants within a source template, not another
        // copy of the thread axis. Include their value in the identity so
        // different CTAs cannot silently acquire the same binding.
        if(coordinate->kind()!=v::CoordinateKind::ThreadIndex) {
          appendNumberField(signature_text,p.coordinate);
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
        type.element().isBitLike() || dynamic_cast<const v::LoadOp*>(&node().operation());
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
          signature_text="initial:";appendField(signature_text,initial->region.allocation);
          appendField(signature_text,type.toString());
          // Replacing a matrix producer by its immutable initial allocation
          // must retain the producer's coordinate stride. Otherwise row-wise
          // operands become unrelated scalar offsets in the parent template.
          if(t.hasStaticShape() && t.shape().size()==2 && t.element().width()>=8 &&
              t.element().width()%8==0 &&
              t.shape().back()<=std::uint64_t(INT64_MAX)/(t.element().width()/8))
            p.row_bytes=t.shape().back()*(t.element().width()/8);
          if(p.row_bytes)UnitRowModuli()[initial->region.allocation].insert(p.row_bytes); // audited row widths (unit_lifting.h)
          p.leaf=IndexedValueBinding{value,0,1,*initial};return p;
        }
        p.form=Form::Leaf;p.coordinate=slice->offset;
        signature_text="producer:";appendKeyField(slice->value);appendField(signature_text,type.toString());
        p.leaf=IndexedValueBinding{value,0,1,{}};p.leaf->producer=std::move(slice->value);return p;
      }
      if(slice->value!=value && slice->offset==0 && t.bitSize()==type.bitSize()) {
        p.form=Form::Forward;p.children.push_back({std::move(slice->value)});
        signature_text="field:";appendField(signature_text,type.toString());return p;
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
      signature_text="producer:";appendKeyField(base);appendField(signature_text,type.toString());
      p.leaf=IndexedValueBinding{value,0,1,{}};p.leaf->producer=std::move(base);return p;
    }
    if(value.conversion()) {
      p.form=Form::Conversion;p.children.push_back({g.canonicalInterpretation(value.conversion()->source)});
      signature_text="conversion:";appendField(signature_text,value_ir::typeConversionLabel(value));return p;
    }
    const auto& op=node().operation();
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
        signature_text="unsigned-literal:";appendField(signature_text,type.toString());
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
      signature_text=parameter->specializedLiteral()?"specialized-parameter:":"symbolic-parameter:";
      appendField(signature_text,type.toString());
      appendField(signature_text,parameter->specializedLiteral()?*parameter->specializedLiteral():parameter->symbol());
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
        signature_text="initial:";appendField(signature_text,initial->region.allocation);
        appendField(signature_text,type.toString());
        p.leaf=IndexedValueBinding{value,0,1,*initial};return p;
      }
      if(source) {
        *source=g.canonicalInterpretation(std::move(*source));const auto& t=g.type(*source);
        if(!t.hasStaticShape() || offset>t.bitSize()/8 || type.bitSize()/8>t.bitSize()/8-offset)return p;
        if(t.isScalar() && offset==0 && t.bitSize()==type.bitSize()) {
          p.form=Form::Forward;p.children.push_back({std::move(*source)});
          signature_text="forward:";appendField(signature_text,type.toString());return p;
        }
        if(t.isMatrix() && offset<=std::uint64_t(INT64_MAX)) {
          p.form=Form::Leaf;p.coordinate=offset;
          signature_text="producer:";appendKeyField(*source);appendField(signature_text,type.toString());
          p.leaf=IndexedValueBinding{value,0,1,{}};p.leaf->producer=std::move(*source);return p;
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
      signature_text="op:";appendField(signature_text,"Exp2");appendField(signature_text,"ideal_arithmetic");
      appendField(signature_text,type.toString());
      const auto operands=g.instanceOperands(value);p.children.reserve(operands.size());
      for(const auto& x:operands)p.children.push_back({g.canonicalInterpretation(x)});
      return p;
    }
    auto found=operation_signatures.find(value.nodeId());
    if(found==operation_signatures.end()) {
      std::ostringstream attrs;op.printAttributes(attrs);
      std::string text;appendField(text,op.name());appendField(text,attrs.str());
      found=operation_signatures.emplace(value.nodeId(),std::move(text)).first;
    }
    signature_text="op:";signature_text+=found->second;appendField(signature_text,type.toString());
    const auto operands=g.instanceOperands(value);p.children.reserve(operands.size());
    for(const auto& x:operands)p.children.push_back({g.canonicalInterpretation(x)});
    return p;
  }
  Summary build(const V& value,Record& r,const Plan& p,unsigned policy) {
    if(p.form==Form::Exact)return exact(value,r);
    // A proved selection/read forwarding is an identity, not a new template
    // shape. Reuse its already compiled summary so direct and forwarded uses
    // share coordinates. Equal storage width alone is not a type identity.
    if(p.form==Form::Forward && p.children.size()==1 && g.type(value)==g.type(p.children.front().value))
      return summaryAt(*p.children.front().record,policy);
    if(p.form==Form::Literal && policy!=5) {
      const std::uint64_t key[]{1,p.signature,std::uint64_t(p.coordinate)};
      return intern(key,[&]{return Template{value,U::Kind::Operands,{},{}};},false,0);
    }
    if(p.form==Form::Leaf || p.form==Form::Literal) {
      std::uint64_t row_bytes=p.row_bytes;
      if(p.leaf->producer) {
        const auto& type=g.type(*p.leaf->producer);
        if(type.hasStaticShape() && type.shape().size()==2 && type.element().width()>=8 && type.element().width()%8==0 &&
           type.shape().back()<=std::uint64_t(INT64_MAX)/(type.element().width()/8))
          row_bytes=type.shape().back()*(type.element().width()/8);
      }
      const std::uint64_t key[]{0,p.signature,row_bytes};
      return intern(key,[&]{return Template{value,U::Kind::Binding,
          std::make_shared<IndexedValueBinding>(*p.leaf),{}};},p.varying,p.coordinate,row_bytes);
    }
    const auto n=p.children.size();
    bool varying=false;std::int64_t coordinate=0;std::size_t anchor=n,anchor_id=0;std::uint64_t row_bytes=0;
    for(std::size_t i=0;i<n;++i) {
      const auto& s=summaryAt(*p.children[i].record,policy);
      if(s.varying && (anchor==n || (policy==2 && s.row_bytes>row_bytes))) {
        varying=true;coordinate=s.coordinate;row_bytes=s.row_bytes;anchor=i;anchor_id=s.id;
        if(policy!=2)break;
      }
    }
    key_words.assign({2,p.signature});pending_maps.clear();
    for(std::size_t i=0;i<n;++i) {
      const auto& child=p.children[i];
      // A whole immutable subtree may be invariant in this use even though
      // another use can parameterize it. Exact identity preserves that fact;
      // never substitute the parent's coordinate into the invariant subtree.
      const auto& compiled=summaryAt(*child.record,policy);
      const auto s=policy==1 && i!=anchor && compiled.varying?exact(child.value,*child.record):compiled;
      std::int64_t delta=0,stride=s.varying?1:0,outer=0;std::uint64_t period=0;
      // A nonzero row width implies an anchor, whose summary is fixed here.
      if(policy>=2 && i!=anchor && s.varying && row_bytes && s.row_bytes && coordinate>=0 && s.id!=anchor_id) {
        // A row-dependent operand can share a flattened matrix coordinate.
        // Validate its residual at every supplied value; this is a coordinate
        // map, not an assumption that arbitrary operands are row invariant.
        std::int64_t origin;
        if(__builtin_mul_overflow(coordinate/std::int64_t(row_bytes),std::int64_t(s.row_bytes),&origin) ||
           !difference(s.coordinate,origin,delta))return exact(value,r);
        period=row_bytes;outer=s.row_bytes;stride=0;
      } else if(s.varying && !difference(s.coordinate,coordinate,delta))return exact(value,r);
      key_words.insert(key_words.end(),{std::uint64_t(s.id),std::uint64_t(delta),std::uint64_t(s.varying),
          period,std::uint64_t(outer)});
      pending_maps.push_back({i,s.id,s.varying?delta:0,stride,outer,period});
    }
    key_words.push_back(row_bytes);
    // Look up the structural template before copying child Values into its
    // immutable maps. Repeated instances carry coordinates, not child DAGs.
    const auto kind=p.form==Form::Forward?U::Kind::Forward:p.form==Form::Conversion?U::Kind::Conversion:U::Kind::Operands;
    return intern(key_words,[&] {
      std::vector<ChildMap> maps;maps.reserve(pending_maps.size());
      for(const auto& m:pending_maps)maps.push_back({p.children[m.index].value,m.child,m.base,m.stride,m.outer,m.period});
      return Template{value,kind,{},std::move(maps)};
    },varying,coordinate,row_bytes);
  }
  Summary summarize(const V& input,unsigned policy) {
    const auto value=g.canonicalInterpretation(input);
    auto& root=record(value);
    if(const auto* found=summaryOf(root,policy))return *found;
    // Same DFS and skip order as a hashed active set: summarized, then active
    // (a cycle makes the parent exact), then visit. Stamps never repeat.
    constexpr auto none=std::size_t(-1);
    struct Clear {std::vector<SummaryFrame>& frames;~Clear(){frames.clear();}} clear{frames};
    const auto stamp=++active_stamp;
    frames.push_back({&root,none,0,plan(value)});root.active=stamp;
    while(!frames.empty()) {
      const auto top=frames.size()-1;auto& f=frames[top];
      if(f.next<f.plan.children.size()) {
        const auto k=f.next++;auto& child=*f.plan.children[k].record;
        if(child.summary[policy])continue;
        if(child.active==stamp) {f.plan=Plan{};continue;}
        child.active=stamp;
        auto next=plan(f.plan.children[k].value); // before push_back moves f
        frames.push_back({&child,top,k,std::move(next)});continue;
      }
      const auto& frame_value=f.parent==none?value:frames[f.parent].plan.children[f.child].value;
      const auto result=build(frame_value,*f.record,f.plan,policy);
      store(*f.record,policy,result);f.record->active=0;frames.pop_back();
    }
    return summaryAt(root,policy);
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
    return storageField(std::move(value),bytes,offset);
  }
  // Pass the owner's shared AssembledRead extents when it has them; it must
  // outlive this object. Without them, storage queries scan assembled pieces.
  // `lift_block_index`: the scalar-fold lowering's instance in lifting mode
  // (unit_lifting.h) declines BlockIndex leaves and literals.
  PureSourceTemplates(const value_ir::Graph& graph,const MemoryEliminationResult& m,
                      AssembledPieceEnds* shared_ends=nullptr,bool lift_block_index=false)
      :g(graph),memory(m),assembled_ends(shared_ends),specialization(graph,lift_block_index),storage(graph),
       lift_block_index(lift_block_index){}
  std::vector<IndexedSourceDomain> mapStrategy(std::span<const V> values,bool interleaved,
                                     std::span<const std::uint64_t> offsets,unsigned policy,
                                     const std::vector<IndexedSourceDomain>& members) {
    if(!offsets.empty() && offsets.size()!=values.size())throw std::invalid_argument("source offsets do not cover values");
    // No pure-operation propagation can reduce an already-single source
    // domain. Keep its existing template instead of visiting its ancestors.
    if(members.size()<=1)return members;
    if(policy>=policies)throw std::invalid_argument("source template policy is out of range");
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
          const auto [value,depth]=todo[n];const auto* found=findSummary(value,policy);
          if(!found)continue;
          const auto p=plan(value);const auto& s=*found;
          SENTRYWARD_DEBUG_PRINT(pure_source_templates_0);
          if(depth<10)for(const auto& child:p.children)todo.emplace_back(child.value,depth+1);
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
          const auto* x=findSummary(a,policy);const auto* y=findSummary(b,policy);
          if(!x || !y || x->id==y->id)continue;
          const auto ap=plan(a),bp=plan(b);
          SENTRYWARD_DEBUG_PRINT(pure_source_templates_1);
          if(ap.children.size()==bp.children.size())
            for(std::size_t i=0;i<ap.children.size();++i)pairs.emplace_back(ap.children[i].value,bp.children[i].value);
        }
        std::vector<std::pair<V,unsigned>> todo{{g.canonicalInterpretation(values.front()),0}};
        for(std::size_t n=0;n<todo.size() && n<128;++n) {
          const auto [value,depth]=todo[n];const auto* found=findSummary(value,policy);
          if(!found)continue;
          const auto p=plan(value);const auto& s=*found;
          SENTRYWARD_DEBUG_PRINT(pure_source_templates_2);
          if(depth<20)for(const auto& child:p.children)todo.emplace_back(child.value,depth+1);
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
        std::all_of(values.begin(),values.end(),[&](const V& x) {
          return g.type(x).element().kind()==value_ir::ElementKind::UnsignedInteger;
        })) {
      auto literals=mapStrategy(values,interleaved,offsets,5,members);
      if(planCost(literals)<planCost(result))return literals;
    }
    return result;
  }
};
}
