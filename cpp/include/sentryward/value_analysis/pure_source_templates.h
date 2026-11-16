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
#include <bit>
#include <map>
#include <functional>
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
// A TMA region fragment Load read as a symbolic initial read (IndexedValueBinding::region_fragment):
// the frontend's ResolvedTma form Load(pointer(Convert.u64(Add.s64(origin, place))) [, guard]) with
// the constant place equal to MemoryAttributes::region_offset, at payload offset 0 of the read.
// Its bytes are the allocation's cells at origin + place (no action writes the allocation), zero
// where the guard is false. Every fragment of one region has the same origin and guard, so the
// fragments of a region differ only in their place. O(1) node lookups; no DAG traversal.
struct RegionFragmentLoad {
  value_ir::Value origin;
  std::optional<value_ir::Value> guard;
  std::string allocation;
  std::uint64_t place = 0;
};
inline std::optional<RegionFragmentLoad> regionFragmentLoad(const value_ir::Graph& g,
    const MemoryEliminationResult& memory,const value_ir::Value& value) {
  namespace v=value_ir;
  const auto plain=[](const v::Value& x){return !x.conversion() && x.indices().empty() && !x.threadSelector();};
  if(value.pointer() || !plain(value))return std::nullopt;
  const auto& node=g.definingNode(value);
  const auto* load=dynamic_cast<const v::LoadOp*>(&node.operation());
  if(!load || !load->attributes().region_offset || load->attributes().layout || load->attributes().descriptor ||
     load->attributes().interpretation!=v::MemoryAttributes::Interpretation::Ordinary ||
     value!=node.result(value.resultIndex(),value.instanceIndex()))return std::nullopt;
  const auto read=memory.symbolic_initial_reads.find({value.nodeId(),value.instanceIndex()});
  if(read==memory.symbolic_initial_reads.end() || read->second.offset)return std::nullopt;
  const auto operands=g.instanceOperands(value);
  const bool guarded=load->attributes().guarded;
  if(operands.size()!=(guarded?2U:1U))return std::nullopt;
  // The u64 conversion is a type edge of the resolved address (Graph::lowerTypeConversions).
  const auto address=g.resolve(operands[0]);
  const auto u64=v::ValueType::scalar(v::ElementType::unsignedInteger(64));
  if(!address.pointer() || !address.indices().empty() || address.threadSelector() || !address.conversion() ||
     address.conversion()->spec->kind!=v::TypeConversionKind::Convert || address.conversion()->spec->target!=u64 ||
     !(address.conversion()->spec->attributes==v::ConversionAttributes{}))return std::nullopt;
  const auto sum=g.resolve(address.conversion()->source);
  if(sum.pointer() || !plain(sum) || g.type(sum)!=v::ValueType::scalar(v::ElementType::signedInteger(64)))return std::nullopt;
  const auto* add=dynamic_cast<const v::BinaryOp*>(&g.definingNode(sum).operation());
  if(!add || add->opcode()!=v::BinaryOpcode::Add || add->attributes().saturate)return std::nullopt;
  const auto terms=g.instanceOperands(sum);
  if(terms.size()!=2)return std::nullopt;
  const auto offset=g.resolve(terms[1]);
  if(!plain(offset) || offset.pointer())return std::nullopt;
  const auto* place=dynamic_cast<const v::ConstantOp*>(&g.definingNode(offset).operation());
  if(!place || place->pointer())return std::nullopt;
  std::uint64_t n=0;
  const auto& literal=place->literal();
  const auto [end,error]=std::from_chars(literal.data(),literal.data()+literal.size(),n);
  if(error!=std::errc{} || end!=literal.data()+literal.size() || n!=*load->attributes().region_offset ||
     n>std::uint64_t(INT64_MAX))return std::nullopt;
  RegionFragmentLoad fragment{g.canonicalInterpretation(terms[0]),std::nullopt,read->second.allocation,n};
  if(guarded)fragment.guard=g.canonicalInterpretation(operands[1]);
  return fragment;
}
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
  // Slice and Fragment are made only for grid families (grid_forms): a byte slice of a wider
  // scalar (child: that scalar; coordinate: the byte offset), and bytes of a TMA region fragment
  // whose origin and guard are children (coordinate: place + byte offset), so that fragments of
  // different regions of one allocation (gathered rows) share a template.
  // BitSelect (grid families only): a Select on one bit of a mask word, whatever its spelling
  // ((m & 2^b) != 0, == 0 with exchanged arms, a sign test); children [mask word, value when
  // the bit is set, value when clear], coordinate: the bit index in the mask word.
  // Read (grid families only): a Load assembled from several stored pieces; children: the pieces'
  // source Values; the signature names every piece's size, payload offset and initial bytes. It
  // only says when two such Loads read the same bytes (equal children): it has no mapped form.
  // Offset (grid families only): an integer add/subtract chain base + k with literal steps (an
  // induction value); child: the base; coordinate: k. A family position that holds such chains
  // also takes a value that is no chain as its own base with k = 0.
  enum class Form {Exact,Leaf,Literal,Forward,Conversion,Operands,Slice,Fragment,BitSelect,Read,Offset};
  // Policies 0-3, 5 and 6 are map strategies; 4 is the shared exact identity.
  // Policy 6 is the joint policy 0 with every shared subtree (sharedSubtrees) exact.
  static constexpr unsigned policies=7;
  // One record per distinct annotated Value (deep Value==) owns its summary
  // slots. Its first handle is only a lookup key: plans and templates keep the
  // caller's own handles, whose annotation identities the caches key on.
  struct Record {
    std::size_t active=0;                       // summarize() visit stamp
    std::array<std::size_t,policies> summary{}; // 1 + summary_pool index; 0 absent
    // sharedSubtrees: the call stamp and value index of the last visit, and whether some
    // call reached this Value from two of its values (then it is one node in both).
    std::size_t share_stamp=0,share_root=0;
    // The sharedSubtrees call stamp when this Value's plan DAG reaches a shared Value.
    std::size_t touch_stamp=0;
    bool shared=false;
    // The records of its plan children, once any plan of this Value was made (plans are
    // deterministic), and its lookup key: sharedSubtrees walks these, not new plans.
    bool children_known=false;
    std::vector<Record*> children;
    const value_ir::Value* key=nullptr;
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
  // Set while a grid family's plans are made (gridPlan): planForm also makes Slice and Fragment
  // forms and Bit-kind literals. The summary policies never see such plans.
  bool grid_forms=false;
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
  // A frame's plan is shared with the plan cache (a frame never changes it); a cycle makes
  // the frame exact instead (planOf), as an empty plan would.
  struct SummaryFrame {Record* record;std::size_t parent,child;std::shared_ptr<const Plan> plan;std::size_t next=0;bool exact=false;};
  static const Plan& exactPlan() {static const Plan empty;return empty;}
  static const Plan& planOf(const SummaryFrame& f) {return f.exact?exactPlan():*f.plan;}
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
  Record& record(const V& value) {
    const auto it=records.try_emplace(value).first;
    if(!it->second.key)it->second.key=&it->first;
    return it->second;
  }
  // Remembers the plan children of `r` (their records) the first time a plan of it is made.
  static void noteChildren(Record& r,const Plan& p) {
    if(r.children_known)return;
    r.children.reserve(p.children.size());
    for(const auto& child:p.children)r.children.push_back(child.record);
    r.children_known=true;
  }
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
  struct SliceKey {
    std::uint64_t owner;std::size_t node,result,instance;std::uint64_t offset,bytes;
    bool operator==(const SliceKey&) const=default;
  };
  struct SliceKeyHash {
    std::size_t operator()(const SliceKey& k) const noexcept {
      std::size_t h=k.node;
      for(const std::size_t v:{std::size_t(k.owner),k.result,k.instance,std::size_t(k.offset),std::size_t(k.bytes)})
        h^=v+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
      return h;
    }
  };
  static constexpr std::size_t slice_memo_limit=1u<<18;
  std::unordered_map<SliceKey,std::optional<IndexedSourceSlice>,SliceKeyHash> slices;
  // Whether two handles are the same handle: equal fields and the same annotation objects
  // (conversion, pointer), recursively for selectors and indices. plan() is a function of
  // the handle, the immutable graph and memory facts and of caches it fills idempotently,
  // so the same handle has the same plan. Deep-equal handles with other annotation objects
  // may differ (signature keys name pointer annotations by identity) and are not the same.
  static bool sameHandle(const V& a,const V& b) {
    if(a.graphIdentity()!=b.graphIdentity() || a.nodeId()!=b.nodeId() || a.resultIndex()!=b.resultIndex() ||
       a.instanceIndex()!=b.instanceIndex() || a.pointer()!=b.pointer() || a.conversion()!=b.conversion())return false;
    const auto* s=a.threadSelector();const auto* t=b.threadSelector();
    if(bool(s)!=bool(t) || (s && s!=t && !sameHandle(*s,*t)))return false;
    const auto x=a.indices(),y=b.indices();
    if(x.size()!=y.size())return false;
    for(std::size_t i=0;i<x.size();++i)if(!sameHandle(x[i],y[i]))return false;
    return true;
  }
  // The plans of one map() call, shared by its strategies: each strategy summarizes the
  // same new nodes under its own policy and would plan each of them again. Keyed by record,
  // valid for the handle that made it (sameHandle); another handle plans afresh. Cleared
  // when map() returns, so it holds at most the plans of one call's new nodes.
  // The plan cache's lifetime: the outermost scope (one map() call) enables and then clears it.
  struct PlanScope {
    PureSourceTemplates& t;bool outer;
    explicit PlanScope(PureSourceTemplates& t):t(t),outer(!t.plan_caching){t.plan_caching=true;}
    ~PlanScope(){if(outer){t.plan_caching=false;t.plan_cache.clear();}}
  };
  struct CachedPlan {V handle;std::shared_ptr<const Plan> plan;};
  std::unordered_map<const Record*,CachedPlan> plan_cache;
  bool plan_caching=false;
  std::shared_ptr<const Plan> cachedPlan(const V& value,const Record& r) {
    if(!plan_caching)return std::make_shared<const Plan>(plan(value));
    if(const auto found=plan_cache.find(&r);found!=plan_cache.end() && sameHandle(found->second.handle,value))
      return found->second.plan;
    auto p=std::make_shared<const Plan>(plan(value));
    plan_cache.insert_or_assign(&r,CachedPlan{value,p});
    return p;
  }
  void offsetSignature(const value_ir::ValueType& type) {
    signature_text="offset:";appendField(signature_text,type.toString());
  }
  // base + k for an integer add/subtract chain with literal steps: wrapping adds of one type
  // associate, so the chain is its base plus the sum. Signed types take any literal steps;
  // unsigned types only non-negative sums. Nothing for a value that is no such add.
  std::optional<std::pair<V,std::int64_t>> offsetChain(const V& value,const value_ir::ValueType& type) {
    namespace v=value_ir;
    if(!type.isScalar() || type.element().isParameterized() || !type.bitSize() || type.bitSize()>64)return {};
    const auto kind=type.element().kind();
    if(kind!=v::ElementKind::SignedInteger && kind!=v::ElementKind::UnsignedInteger)return {};
    const auto literal=[&](const V& operand)->std::optional<std::int64_t> {
      const auto x=g.resolve(operand);
      if(x.conversion() || x.pointer() || x.threadSelector() || !x.indices().empty() || g.type(x)!=type)return {};
      const auto* constant=dynamic_cast<const v::ConstantOp*>(&g.definingNode(x).operation());
      if(!constant || constant->pointer())return {};
      std::int64_t n=0;const auto& text=constant->literal();
      const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),n);
      if(error!=std::errc{} || end!=text.data()+text.size() || n<=-(std::int64_t(1)<<30) || n>=(std::int64_t(1)<<30))return {};
      return n;
    };
    V base=value;std::int64_t total=0;bool any=false;
    for(unsigned depth=0;depth<64;++depth) {
      const auto b=g.canonicalInterpretation(base);
      if(b.conversion() || b.pointer() || b.threadSelector() || !b.indices().empty() || g.type(b)!=type)break;
      const auto* step=dynamic_cast<const v::BinaryOp*>(&g.definingNode(b).operation());
      if(!step || (step->opcode()!=v::BinaryOpcode::Add && step->opcode()!=v::BinaryOpcode::Subtract))break;
      const auto& a=step->attributes();
      if(a.rounding!=v::RoundingMode::None || a.multiply_mode!=v::IntegerMultiplyMode::Low || a.saturate ||
         a.flush_to_zero || a.propagate_nan)break;
      const auto args=g.instanceOperands(b);
      if(args.size()!=2)break;
      std::optional<std::int64_t> c;unsigned other=0;
      if((c=literal(args[1])))other=0;
      else if(step->opcode()==v::BinaryOpcode::Add && (c=literal(args[0])))other=1;
      if(!c || g.type(args[other])!=type)break;
      const auto next=total+(step->opcode()==v::BinaryOpcode::Subtract?-*c:*c);
      if(next<=-(std::int64_t(1)<<30) || next>=(std::int64_t(1)<<30))break;
      total=next;base=args[other];any=true;
    }
    if(!any || (kind==v::ElementKind::UnsignedInteger && total<0))return {};
    return std::pair{base,total};
  }
  // The word whose bit `bit` is bit `bit` of `mask`, through steps that keep that bit: equal-size
  // bitcasts, integer truncations (default Convert), right shifts by a constant and bfe fields.
  // Each step is exact for the tested bit only; at most 16 steps.
  void bitRoot(V& mask,std::uint64_t& bit) {
    namespace v=value_ir;
    const auto integer=[](const v::ValueType& t) {
      return t.isScalar() && !t.element().isParameterized() && t.element().isBitLike() && t.bitSize() && t.bitSize()<=64;
    };
    for(unsigned depth=0;depth<16;++depth) {
      mask=g.canonicalInterpretation(mask);
      if(mask.pointer() || mask.threadSelector() || !mask.indices().empty() || !integer(g.type(mask)))return;
      if(mask.conversion()) {
        const auto& c=*mask.conversion();const auto& source=g.type(c.source);const auto& target=g.type(mask);
        if(!integer(source))return;
        const bool same=c.spec->kind==v::TypeConversionKind::Bitcast && source.bitSize()==target.bitSize();
        const bool convert=c.spec->kind==v::TypeConversionKind::Convert && c.spec->attributes==v::ConversionAttributes{};
        const bool truncation=convert && target.bitSize()<=source.bitSize();
        // A widening keeps the source's bits; from a signed source it copies the sign bit into
        // every higher bit (from another source the higher bits are zero: not a source bit).
        const bool widening=convert && target.bitSize()>source.bitSize();
        const bool sign=widening && source.element().kind()==v::ElementKind::SignedInteger;
        if(!same && !truncation && !(widening && (sign || bit<source.bitSize())))return;
        if(sign && bit>=source.bitSize())bit=source.bitSize()-1;
        mask=c.source;continue;
      }
      const auto& op=g.definingNode(mask).operation();
      const auto width=g.type(mask).bitSize();
      const auto amount=[&](const V& x)->std::optional<std::uint64_t> {
        const auto c=specialization.known(x);if(!c)return {};
        std::uint64_t n=0;const auto& text=c->literal;
        const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),n);
        if(error!=std::errc{} || end!=text.data()+text.size())return {};
        return n;
      };
      if(const auto* shift=dynamic_cast<const v::BinaryOp*>(&op);shift && shift->opcode()==v::BinaryOpcode::ShiftRight &&
          shift->attributes().rounding==v::RoundingMode::None && !shift->attributes().saturate) {
        const auto args=g.instanceOperands(mask);
        const auto k=args.size()==2?amount(args[1]):std::nullopt;
        if(!k || *k>=width || bit>=width-*k || g.type(args[0])!=g.type(mask))return;
        bit+=*k;mask=args[0];continue;
      }
      if(const auto* field=dynamic_cast<const v::PtxScalarOp*>(&op);field && field->opcode()==v::PtxScalarOpcode::BitExtract) {
        const auto args=g.instanceOperands(mask);
        if(args.size()!=3 || !integer(g.type(args[0])))return;
        const auto pos=amount(args[1]),len=amount(args[2]);
        if(!pos || !len || bit>=*len || *pos>=g.type(args[0]).bitSize() || *len>g.type(args[0]).bitSize()-*pos)return;
        bit+=*pos;mask=args[0];continue;
      }
      return;
    }
  }
  struct BitSelection {V mask,set,clear;std::uint64_t bit;};
  // A Select whose predicate tests one bit of an integer word (the idioms of indexed_source.cpp's
  // conditional matcher): (m & 2^b) compared with 0, or the sign of a signed word.
  std::optional<BitSelection> bitSelection(const V& value) {
    namespace v=value_ir;
    const auto args=g.instanceOperands(value);
    if(args.size()!=3)return {};
    const auto predicate=g.canonicalInterpretation(args[0]);
    if(predicate.conversion() || predicate.threadSelector() || predicate.pointer() || !predicate.indices().empty())return {};
    const auto* compare=dynamic_cast<const v::CompareOp*>(&g.definingNode(predicate).operation());
    if(!compare || compare->flushToZero() || compare->unorderedTrue())return {};
    const auto tests=g.instanceOperands(predicate);
    if(tests.size()!=2)return {};
    const auto plain=[](const V& x){return !x.conversion() && !x.threadSelector() && !x.pointer() && x.indices().empty();};
    const auto bits=[&](V x) {
      // Bit extraction ignores interpretation, not numeric conversion.
      for(;;) {
        x=g.canonicalInterpretation(x);
        if(!x.conversion() || x.conversion()->spec->kind!=v::TypeConversionKind::Bitcast || !x.indices().empty() ||
           x.threadSelector() || x.pointer() || !g.type(x).isScalar() || !g.type(x.conversion()->source).isScalar() ||
           g.type(x).bitSize()!=g.type(x.conversion()->source).bitSize())return x;
        x=x.conversion()->source;
      }
    };
    for(unsigned side=0;side<2;++side) {
      const auto& type=g.type(tests[side]);
      if(!type.isScalar() || type.element().isParameterized() || !type.element().isBitLike() ||
         !type.bitSize() || type.bitSize()>64 || g.type(tests[1-side])!=type)continue;
      const auto boundary=specialization.known(tests[1-side]);
      if(!boundary)continue;
      using P=v::ComparePredicate;
      auto relation=compare->predicate();
      if(side) {
        if(relation==P::Greater)relation=P::Less;
        else if(relation==P::GreaterEqual)relation=P::LessEqual;
        else if(relation==P::Less)relation=P::Greater;
        else if(relation==P::LessEqual)relation=P::GreaterEqual;
      }
      if(boundary->literal=="0" && (relation==P::Equal || relation==P::NotEqual)) {
        const bool equal=relation==P::Equal;
        const auto test=bits(tests[side]);
        if(!plain(test))return {};
        const auto* operation=dynamic_cast<const v::BinaryOp*>(&g.definingNode(test).operation());
        if(!operation || operation->opcode()!=v::BinaryOpcode::BitAnd)return {};
        const auto fields=g.instanceOperands(test);
        for(unsigned field=0;fields.size()==2 && field<2;++field) {
          const auto c=specialization.known(fields[field]);
          if(!c)continue;
          std::uint64_t n=0;const auto& text=c->literal;
          const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),n);
          if(error!=std::errc{} || end!=text.data()+text.size() || !std::has_single_bit(n) ||
             std::uint64_t(std::bit_width(n))>type.bitSize())continue;
          BitSelection result{fields[1-field],args[equal?2:1],args[equal?1:2],std::uint64_t(std::countr_zero(n))};
          bitRoot(result.mask,result.bit);
          return result;
        }
        return {};
      }
      if(type.element().kind()!=v::ElementKind::SignedInteger)continue;
      const bool sign_clear=(relation==P::GreaterEqual && boundary->literal=="0") || (relation==P::Greater && boundary->literal=="-1");
      const bool sign_set=(relation==P::Less && boundary->literal=="0") || (relation==P::LessEqual && boundary->literal=="-1");
      if(!sign_clear && !sign_set)continue;
      BitSelection result{tests[side],args[sign_set?1:2],args[sign_set?2:1],type.bitSize()-1};
      bitRoot(result.mask,result.bit);
      return result;
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
    // The fragments of one TMA region: one leaf template whose coordinate is the place
    // (IndexedValueBinding::region_fragment). The signature names the allocation, the
    // fragment's type, the region's origin and guard, so equal signatures read one
    // region of one allocation; the summary's coordinate progression is validated per
    // stored record by the domain planner. A fragment row is a byte matrix: this is its
    // only template form (other matrices stay exact).
    if(const auto fragment=regionFragmentLoad(g,memory,value)) {
      // A fragment read with a symbolic position (symbolic_read_positions) has an exact cell that
      // class lifting moves by the unit's offset; the Fragment form addresses the bytes relative
      // to the region's origin, which lifting cannot move. Such a read keeps its previous forms.
      if(grid_forms && !memory.symbolic_read_positions.contains({value.nodeId(),value.instanceIndex()})) {
        p.form=Form::Fragment;p.coordinate=std::int64_t(fragment->place);
        p.children.push_back({fragment->origin});
        if(fragment->guard)p.children.push_back({*fragment->guard});
        signature_text="fragment:";appendField(signature_text,fragment->allocation);
        appendField(signature_text,type.toString());appendField(signature_text,fragment->guard?"guarded":"unguarded");
        // An exact region (MemoryAttributes::region_exact, unguarded: its origin is the literal first
        // byte) and a rebuilt one never share a family: the scalar fold reads a family's cells by its
        // leaf's kind (exact: moved by the allocation's δ; rebuilt: refused under lifting unless the
        // allocation is anchored), so a mixed family would move a rebuilt member's cell twice.
        const auto* load=dynamic_cast<const v::LoadOp*>(&g.definingNode(value).operation());
        appendField(signature_text,load && load->attributes().region_exact && !fragment->guard?"exact":"rebuilt");
        p.leaf=IndexedValueBinding{value,0,1,{}};p.leaf->region_fragment=true;
        return p;
      }
      p.form=Form::Leaf;p.coordinate=std::int64_t(fragment->place);p.row_bytes=0;
      signature_text="region-fragment:";appendField(signature_text,fragment->allocation);
      appendField(signature_text,type.toString());appendKeyField(fragment->origin);
      if(fragment->guard)appendKeyField(*fragment->guard);
      else appendField(signature_text,"unguarded");
      p.leaf=IndexedValueBinding{value,0,1,{}};p.leaf->region_fragment=true;
      return p;
    }
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
    // A vote over the lanes of a warp (a BinaryOp flagged lane_own)
    // is one value for every member that reads it, and the scalar fold lowers it as one
    // reduction over the lanes. It stays exact: no template descends into the lanes' terms.
    if(!value.conversion() && value.indices().empty())
      if(const auto* vote=dynamic_cast<const v::BinaryOp*>(&node().operation());vote && vote->attributes().lane_own)
        return p;
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
        if(grid_forms && !memory.symbolic_read_positions.contains({slice->value.nodeId(),slice->value.instanceIndex()}))
          if(const auto fragment=regionFragmentLoad(g,memory,slice->value);
            fragment && fragment->place<=std::uint64_t(INT64_MAX)-slice->offset) {
          p.form=Form::Fragment;p.coordinate=std::int64_t(fragment->place+slice->offset);
          p.children.push_back({fragment->origin});
          if(fragment->guard)p.children.push_back({*fragment->guard});
          signature_text="fragment-slice:";appendField(signature_text,fragment->allocation);
          appendField(signature_text,type.toString());appendField(signature_text,fragment->guard?"guarded":"unguarded");
          p.leaf=IndexedValueBinding{value,0,1,{}};p.leaf->region_fragment=true;
          p.leaf->producer=std::move(slice->value);
          return p;
        }
        p.form=Form::Leaf;p.coordinate=slice->offset;
        signature_text="producer:";appendKeyField(slice->value);appendField(signature_text,type.toString());
        p.leaf=IndexedValueBinding{value,0,1,{}};p.leaf->producer=std::move(slice->value);return p;
      }
      if(slice->value!=value && slice->offset==0 && t.bitSize()==type.bitSize()) {
        p.form=Form::Forward;p.children.push_back({std::move(slice->value)});
        signature_text="field:";appendField(signature_text,type.toString());return p;
      }
      // Bytes of a wider scalar (a word of a packed 128-bit store): the slice of that scalar.
      if(grid_forms && slice->value!=value && t.isScalar() && !slice->value.pointer() &&
          slice->offset<=std::uint64_t(INT64_MAX)) {
        p.form=Form::Slice;p.coordinate=std::int64_t(slice->offset);
        signature_text="slice:";appendField(signature_text,type.toString());appendField(signature_text,t.toString());
        p.leaf=IndexedValueBinding{value,0,1,{}};p.leaf->producer=slice->value;
        p.children.push_back({std::move(slice->value)});
        return p;
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
        grid_forms && constant && !constant->pointer() && type.bitSize()<=64 &&
        type.element().kind()==v::ElementKind::SignedInteger) {
      // A signed literal of a grid family: its value is the coordinate (negative values too).
      const auto& literal=constant->literal();std::int64_t n=0;
      const auto [end,error]=std::from_chars(literal.data(),literal.data()+literal.size(),n);
      const auto limit=std::int64_t(1)<<(type.bitSize()-1 < 62 ? type.bitSize()-1 : 62);
      if(error==std::errc{} && end==literal.data()+literal.size() && n>=-limit && n<limit) {
        p.form=Form::Literal;p.coordinate=n;
        signature_text="signed-literal:";appendField(signature_text,type.toString());
        p.leaf=IndexedValueBinding{value,0,1,{}};return p;
      }
    }
    if(const auto* constant=dynamic_cast<const v::ConstantOp*>(&op);
        constant && !constant->pointer() && type.bitSize()<=64 &&
        (type.element().kind()==v::ElementKind::UnsignedInteger ||
         (grid_forms && type.element().kind()==v::ElementKind::Bit))) {
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
      } else if(const auto* assembled=grid_forms?std::get_if<AssembledRead>(replacement):nullptr) {
        p.form=Form::Read;
        signature_text="read:";appendField(signature_text,type.toString());
        for(const auto& piece:assembled->pieces) {
          appendNumberField(signature_text,piece.type.bitSize());appendNumberField(signature_text,piece.byte_offset);
          if(const auto* from=std::get_if<V>(&piece.source)) {
            appendField(signature_text,"value");p.children.push_back({g.canonicalInterpretation(*from)});
          } else {
            const auto& bytes=std::get<InitialMemorySymbol>(piece.source);
            appendField(signature_text,bytes.region.allocation);
            appendNumberField(signature_text,bytes.region.offset);appendNumberField(signature_text,bytes.region.bytes);
          }
        }
        return p;
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
    if(grid_forms)if(auto chain=offsetChain(value,type)) {
      p.form=Form::Offset;p.coordinate=chain->second;
      offsetSignature(type);
      p.leaf=IndexedValueBinding{value,0,1,{}};
      p.children.push_back({g.canonicalInterpretation(chain->first)});
      return p;
    }
    if(grid_forms && dynamic_cast<const v::SelectOp*>(&op) && op.resultTypes().size()==1)
      if(auto selection=bitSelection(value);selection && selection->bit<=std::uint64_t(INT64_MAX)) {
        p.form=Form::BitSelect;p.coordinate=std::int64_t(selection->bit);
        signature_text="bit-select:";appendField(signature_text,type.toString());
        appendField(signature_text,g.type(selection->mask).toString());
        p.leaf=IndexedValueBinding{value,0,1,{}};
        p.children.push_back({g.canonicalInterpretation(selection->mask)});
        p.children.push_back({g.canonicalInterpretation(selection->set)});
        p.children.push_back({g.canonicalInterpretation(selection->clear)});
        return p;
      }
    const bool pure=dynamic_cast<const v::UnaryOp*>(&op) || dynamic_cast<const v::BinaryOp*>(&op) ||
        dynamic_cast<const v::FmaOp*>(&op) || dynamic_cast<const v::IdealExp2Op*>(&op) ||
        dynamic_cast<const v::CompareOp*>(&op) || dynamic_cast<const v::SelectOp*>(&op) ||
        dynamic_cast<const v::ConstantOp*>(&op) ||
        (dynamic_cast<const v::PtxScalarOp*>(&op) &&
         (grid_forms || static_cast<const v::PtxScalarOp&>(op).opcode()>=v::PtxScalarOpcode::Sin) &&
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
    // Policy 6 combines as the joint policy 0; only its summaries (shared subtrees exact) differ.
    const unsigned rule=policy==6?0:policy;
    // A proved selection/read forwarding is an identity, not a new template
    // shape. Reuse its already compiled summary so direct and forwarded uses
    // share coordinates. Equal storage width alone is not a type identity.
    if(p.form==Form::Forward && p.children.size()==1 && g.type(value)==g.type(p.children.front().value))
      return summaryAt(*p.children.front().record,policy);
    if(p.form==Form::Literal && rule!=5) {
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
      if(s.varying && (anchor==n || (rule==2 && s.row_bytes>row_bytes))) {
        varying=true;coordinate=s.coordinate;row_bytes=s.row_bytes;anchor=i;anchor_id=s.id;
        if(rule!=2)break;
      }
    }
    key_words.assign({2,p.signature});pending_maps.clear();
    for(std::size_t i=0;i<n;++i) {
      const auto& child=p.children[i];
      // A whole immutable subtree may be invariant in this use even though
      // another use can parameterize it. Exact identity preserves that fact;
      // never substitute the parent's coordinate into the invariant subtree.
      const auto& compiled=summaryAt(*child.record,policy);
      const auto s=rule==1 && i!=anchor && compiled.varying?exact(child.value,*child.record):compiled;
      std::int64_t delta=0,stride=s.varying?1:0,outer=0;std::uint64_t period=0;
      // A nonzero row width implies an anchor, whose summary is fixed here.
      if(rule>=2 && i!=anchor && s.varying && row_bytes && s.row_bytes && coordinate>=0 && s.id!=anchor_id) {
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
    if(policy==6 && root.touch_stamp!=share_stamp && root.summary[0]){root.summary[6]=root.summary[0];return summaryAt(root,6);}
    // Same DFS and skip order as a hashed active set: summarized, then active
    // (a cycle makes the parent exact), then visit. Stamps never repeat.
    constexpr auto none=std::size_t(-1);
    struct Clear {std::vector<SummaryFrame>& frames;~Clear(){frames.clear();}} clear{frames};
    const auto stamp=++active_stamp;
    frames.push_back({&root,none,0,cachedPlan(value,root)});root.active=stamp;noteChildren(root,*frames.back().plan);
    while(!frames.empty()) {
      const auto top=frames.size()-1;auto& f=frames[top];
      if(f.next<planOf(f).children.size()) {
        const auto k=f.next++;auto& child=*planOf(f).children[k].record;
        if(child.summary[policy])continue;
        if(policy==6 && child.shared) {store(child,6,exact(planOf(f).children[k].value,child));continue;}
        // An untouched Value's policy-6 summary is its policy-0 summary (no shared Value below).
        if(policy==6 && child.touch_stamp!=share_stamp && child.summary[0]){child.summary[6]=child.summary[0];continue;}
        if(child.active==stamp) {f.exact=true;continue;}
        child.active=stamp;
        auto next=cachedPlan(planOf(f).children[k].value,child); // before push_back moves f
        noteChildren(child,*next);
        frames.push_back({&child,top,k,std::move(next)});continue;
      }
      const auto& frame_value=f.parent==none?value:planOf(frames[f.parent]).children[f.child].value;
      const auto result=build(frame_value,*f.record,planOf(f),policy);
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
  // A handle without annotations (conversion, pointer, selection, thread selector) is its
  // node, result and instance, and storageField is a function of the handle (its caches fill
  // idempotently), so such queries are memoized: a projection asks again for every request,
  // including those its own cache answers. The memo is dropped at a bound; it never changes
  // a result. Expected O(1) per repeated query.
  std::optional<IndexedSourceSlice> storageSlice(V value,std::uint64_t offset,std::uint64_t bytes) {
    if(value.pointer() || value.conversion() || !value.indices().empty() || value.threadSelector())
      return storageField(std::move(value),bytes,offset);
    const SliceKey key{value.graphIdentity(),value.nodeId(),value.resultIndex(),value.instanceIndex(),offset,bytes};
    if(const auto found=slices.find(key);found!=slices.end())return found->second;
    auto result=storageField(std::move(value),bytes,offset);
    if(slices.size()>=slice_memo_limit)slices.clear();
    slices.emplace(key,result);
    return result;
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
  // ---- Grid families: two-axis source templates by anti-unification of member plans ----
  // The one-coordinate summaries above give one template to values whose varying leaves all move
  // by one step per member (or by a producer's row width). Members of a two-axis family (heads x
  // columns of an output, tokens x heads of an MMA operand built from per-thread registers) move
  // each leaf by its own step per axis: a matrix element by its row and column strides, a
  // per-head scalar by one element per head and nothing per column. gridDomains finds such
  // families among the one-axis domains of a plan: rows of equal domains at one record step. The
  // template is the plan DAG of the first member with, per leaf, coordinate = c + column*i + row*j
  // taken from three samples (first member, next column, next row); a position holding one Value
  // in all three is held exact. Every member of every row is then checked against it node by node
  // (form, signature, row width, leaf coordinate, exact Value), so nothing is extrapolated: a row
  // that differs ends the family. A family is made only where its template has a gathered
  // fragment or a mask bit test at a varying position (Grid::special); other rows are left to
  // the one-axis policies.
  // The result is an ordinary use DAG (Operands / Conversion /
  // Forward / Binding / Exact nodes under CoordinateMap edges, the leaf maps in the edges as
  // base + stride*(m % period) + outer_stride*(m / period)), projected by the scalar fold as any
  // other indexed source.
  // Cost: the template walk visits each node triple once; checking visits each distinct
  // (template node, member Value) pair once (a per-row subtree is checked once per row, not per
  // column); one plan per distinct Value per call. A call's work is bounded by grid_budget node
  // visits, after which the remaining domains stay one-axis.
 public:
  // plain: templates left as row domains (no fragment or bit-test position).
  struct GridStats {std::size_t attempts=0,families=0,members=0,domains=0,work=0,plain=0;};
  GridStats grid_stats;
 private:
  static constexpr std::size_t kGridMinRows=4,kGridPartners=8;
  struct GridNode {
    V value;                       // this position's Value in the first member
    Record* record=nullptr;
    Form form=Form::Exact;
    std::size_t signature=0;
    std::uint64_t row_bytes=0;
    bool same=false;               // one Value in all three samples: held exact
    bool leaf=false,mapped=false;  // a Leaf or Literal; whether its coordinate is mapped
    bool payload=false;            // a Slice or Fragment: children and a mapped coordinate
    bool power=false;              // a literal 1 << coordinate (the map is over the bit index)
    bool constant=false;           // one value in every member: no mapped coordinate below it
    std::int64_t coordinate=0,column=0,row=0;
    std::optional<IndexedValueBinding> binding;
    std::vector<std::size_t> children;
    std::vector<V> child_values;   // the children as this node's plan names them (edge prototypes)
    bool depends_column=false,depends_row=false;
    Ptr use;
  };
  struct GridCheckHash {
    std::size_t operator()(const std::pair<std::size_t,const Record*>& k) const noexcept {
      return std::hash<const Record*>{}(k.second)^(k.first*0x9e3779b97f4a7c15ULL);
    }
  };
  struct Grid {
    std::vector<GridNode> nodes;std::size_t root=0;
    // Whether some varying position is a gathered fragment or a mask bit test (Fragment,
    // BitSelect): the forms a family is made for (gridDomains). The other grid-only forms (Slice,
    // Read, Offset) take part in such a template but do not start one.
    bool special=false;
    // (template node, member Value) pairs already checked, with the member coordinates.
    std::unordered_map<std::pair<std::size_t,const Record*>,std::pair<std::int64_t,std::int64_t>,GridCheckHash> checked;
  };
  std::unordered_map<const Record*,std::shared_ptr<const Plan>> grid_plans;
  std::size_t grid_budget=0;
  std::shared_ptr<const Plan> gridPlan(const V& value,Record& r) {
    const auto [it,added]=grid_plans.try_emplace(&r);
    if(added) {
      struct Forms {bool& flag;~Forms(){flag=false;}} forms{grid_forms};
      grid_forms=true;
      it->second=std::make_shared<const Plan>(plan(value));
    }
    return it->second;
  }
  // A value that is no add chain, at a position where other members are: its own base, k = 0.
  std::shared_ptr<const Plan> asOffset(const V& value) {
    Plan p;p.form=Form::Offset;p.coordinate=0;
    offsetSignature(g.type(value));p.signature=signatureId();
    p.leaf=IndexedValueBinding{value,0,1,{}};
    p.children.push_back({value,&record(value)});
    return std::make_shared<const Plan>(std::move(p));
  }
  static bool powerOfTwo(std::int64_t x){return x>0 && !(x&(x-1));}
  static std::int64_t bitIndex(std::int64_t x){return std::int64_t(std::countr_zero(std::uint64_t(x)));}
  // The template of the family whose first member is `a`, next column `b`, next row `c`.
  // power_literals: literals that are powers of two in all three samples (not all equal) map
  // their bit index (1, 2, 4, ... along an axis) instead of their value.
  std::optional<Grid> gridTemplate(const V& a,const V& b,const V& c,bool power_literals) {
    Grid grid;
    constexpr auto open=std::size_t(-1);
    std::map<std::array<const Record*,3>,std::size_t> made;
    struct Frame {
      V x;Record* record;std::array<const Record*,3> key;
      std::shared_ptr<const Plan> px,py,pz;std::size_t next=0;std::vector<std::size_t> children;
    };
    std::vector<Frame> stack;
    // The codes name the reason for a debugger; every refusal leaves the domains one-axis.
    const auto no=[](std::size_t,const V&,const V&,const V&){return 0;};
    // 0: no template; 1: `out` is the node; 2: a frame was pushed.
    const auto enter=[&](const V& x0,const V& y0,const V& z0,std::size_t& out)->int {
      const auto x=g.canonicalInterpretation(x0),y=g.canonicalInterpretation(y0),z=g.canonicalInterpretation(z0);
      auto& rx=record(x);auto& ry=record(y);auto& rz=record(z);
      const std::array<const Record*,3> key{&rx,&ry,&rz};
      if(const auto found=made.find(key);found!=made.end()) {
        if(found->second==open)return no(1,x,y,z);
        out=found->second;return 1;
      }
      if(++grid_stats.work>grid_budget)return no(2,x,y,z);
      GridNode n;n.value=x;n.record=&rx;
      if(&rx==&ry && &rx==&rz) {
        n.same=true;n.constant=true;grid.nodes.push_back(std::move(n));out=grid.nodes.size()-1;made.emplace(key,out);return 1;
      }
      auto px=gridPlan(x,rx),py=gridPlan(y,ry),pz=gridPlan(z,rz);
      if(px->form==Form::Offset || py->form==Form::Offset || pz->form==Form::Offset) {
        const auto integer=[&](const V& value){return g.type(value)==g.type(x);};
        if(!integer(y) || !integer(z))return no(4,x,y,z);
        if(px->form!=Form::Offset)px=asOffset(x);
        if(py->form!=Form::Offset)py=asOffset(y);
        if(pz->form!=Form::Offset)pz=asOffset(z);
      }
      if(px->form==Form::Exact)return no(3,x,y,z);
      if(py->form!=px->form || pz->form!=px->form)return no(4,x,y,z);
      if(py->signature!=px->signature)return no(5,x,y,z);
      if(pz->signature!=px->signature)return no(6,x,y,z);
      if(py->children.size()!=px->children.size() || pz->children.size()!=px->children.size())return no(7,x,y,z);
      if(px->form==Form::Leaf || px->form==Form::Literal) {
        if(py->row_bytes!=px->row_bytes || pz->row_bytes!=px->row_bytes ||
           py->varying!=px->varying || pz->varying!=px->varying || !px->leaf)return no(8,x,y,z);
        n.form=px->form;n.signature=px->signature;n.row_bytes=px->row_bytes;
        n.leaf=true;n.coordinate=px->coordinate;n.binding=px->leaf;
        auto cy=py->coordinate,cz=pz->coordinate;
        // A bit-pattern literal (b8 ... b64) is one constant in every member. One that moves along
        // an axis is no mapped coordinate: its consumers may reinterpret the bits (a float written
        // as b32 whose exponent steps along the family), and bits of an index have no numerical
        // expression. Integer literals map as before.
        if(px->form==Form::Literal && (cy!=n.coordinate || cz!=n.coordinate) &&
           g.type(x).element().kind()==value_ir::ElementKind::Bit)return no(11,x,y,z);
        if(power_literals && px->form==Form::Literal && powerOfTwo(n.coordinate) && powerOfTwo(cy) && powerOfTwo(cz) &&
           (cy!=n.coordinate || cz!=n.coordinate)) {
          n.power=true;n.coordinate=bitIndex(n.coordinate);cy=bitIndex(cy);cz=bitIndex(cz);
        }
        if(!difference(cy,n.coordinate,n.column) || !difference(cz,n.coordinate,n.row))return no(9,x,y,z);
        // An invariant leaf (a CTA coordinate, whose value its signature names) has no map.
        if(!px->varying && (n.column || n.row))return no(10,x,y,z);
        n.mapped=px->form==Form::Leaf?px->varying:(n.column || n.row);
        n.depends_column=n.column!=0;n.depends_row=n.row!=0;
        n.constant=!n.column && !n.row;
        grid.nodes.push_back(std::move(n));out=grid.nodes.size()-1;made.emplace(key,out);return 1;
      }
      made.emplace(key,open);
      stack.push_back(Frame{x,&rx,key,std::move(px),std::move(py),std::move(pz),0,{}});
      return 2;
    };
    std::size_t root=0;
    switch(enter(a,b,c,root)) {
      case 0:return std::nullopt;
      case 1:grid.root=root;return grid;
      default:break;
    }
    while(!stack.empty()) {
      if(stack.back().next<stack.back().px->children.size()) {
        const auto i=stack.back().next++;
        // Copies: enter() may push a frame and move the stack.
        const auto x=stack.back().px->children[i].value,y=stack.back().py->children[i].value,z=stack.back().pz->children[i].value;
        std::size_t child=0;
        const auto state=enter(x,y,z,child);
        if(state==0)return std::nullopt;
        if(state==1)stack.back().children.push_back(child);
        continue;
      }
      auto f=std::move(stack.back());stack.pop_back();
      GridNode n;n.value=f.x;n.record=f.record;n.form=f.px->form;n.signature=f.px->signature;n.row_bytes=f.px->row_bytes;
      n.children=std::move(f.children);
      grid.special=grid.special || n.form==Form::Fragment || n.form==Form::BitSelect;
      if(n.form==Form::Slice || n.form==Form::Fragment || n.form==Form::BitSelect || n.form==Form::Offset) {
        if(!f.px->leaf)return std::nullopt;
        n.payload=true;n.coordinate=f.px->coordinate;n.binding=f.px->leaf;
        if(!difference(f.py->coordinate,f.px->coordinate,n.column) || !difference(f.pz->coordinate,f.px->coordinate,n.row))
          return std::nullopt;
        n.depends_column=n.column!=0;n.depends_row=n.row!=0;
      }
      n.child_values.reserve(f.px->children.size());
      for(const auto& child:f.px->children)n.child_values.push_back(child.value);
      n.constant=!n.depends_column && !n.depends_row;
      for(const auto child:n.children) {
        n.depends_column=n.depends_column || grid.nodes[child].depends_column;
        n.depends_row=n.depends_row || grid.nodes[child].depends_row;
        n.constant=n.constant && grid.nodes[child].constant;
      }
      // A Load assembled from pieces has no mapped form: its pieces must be one in every member.
      if(n.form==Form::Read && !n.constant)return std::nullopt;
      grid.nodes.push_back(std::move(n));
      const auto index=grid.nodes.size()-1;
      made[f.key]=index;
      if(stack.empty())grid.root=index;else stack.back().children.push_back(index);
    }
    return grid;
  }
  // Whether two Values are one value by their plans: equal forms, signatures, row widths and
  // coordinates, and equal children (an add chain and its base with k = 0 are compared as
  // chains). Unplannable Values are equal only when they are one Value. Each pair is compared
  // once per gridDomains call (grid_equal); the walk stops at the first difference.
  struct GridPairHash {
    std::size_t operator()(const std::pair<const Record*,const Record*>& k) const noexcept {
      return std::hash<const Record*>{}(k.first)*0x9e3779b97f4a7c15ULL^std::hash<const Record*>{}(k.second);
    }
  };
  std::unordered_map<std::pair<const Record*,const Record*>,bool,GridPairHash> grid_equal;
  bool gridEqual(const V& a,const V& b) {
    std::vector<std::pair<V,V>> work;work.emplace_back(a,b);
    std::vector<std::pair<const Record*,const Record*>> visited;
    bool equal=true;
    while(!work.empty() && equal) {
      const auto x=g.canonicalInterpretation(work.back().first),y=g.canonicalInterpretation(work.back().second);
      work.pop_back();
      auto& rx=record(x);auto& ry=record(y);
      if(&rx==&ry)continue;
      const std::pair<const Record*,const Record*> key{&rx,&ry};
      if(const auto found=grid_equal.find(key);found!=grid_equal.end()) {equal=found->second;continue;}
      if(++grid_stats.work>grid_budget) {equal=false;break;}
      auto px=gridPlan(x,rx),py=gridPlan(y,ry);
      if(px->form==Form::Offset || py->form==Form::Offset) {
        if(g.type(x)!=g.type(y)) {equal=false;break;}
        if(px->form!=Form::Offset)px=asOffset(x);
        if(py->form!=Form::Offset)py=asOffset(y);
        // Two values that are their own bases are compared as they are.
        if(px->children[0].record==&rx && py->children[0].record==&ry) {equal=false;break;}
      }
      if(px->form==Form::Exact || px->form!=py->form || px->signature!=py->signature || px->row_bytes!=py->row_bytes ||
         px->coordinate!=py->coordinate || px->varying!=py->varying || px->children.size()!=py->children.size()) {
        equal=false;break;
      }
      // Provisionally equal: a revisit inside this walk (a shared subtree) is not walked twice.
      grid_equal.emplace(key,true);visited.push_back(key);
      for(std::size_t i=0;i<px->children.size();++i)work.emplace_back(px->children[i].value,py->children[i].value);
    }
    if(!equal)for(const auto& key:visited)grid_equal.erase(key);
    if(!equal) {
      const auto x=g.canonicalInterpretation(a),y=g.canonicalInterpretation(b);
      grid_equal.insert_or_assign({&record(x),&record(y)},false);
    }
    return equal;
  }
  // Whether `member` is the template at (column, row): the same forms and signatures, every
  // mapped leaf at its mapped coordinate, every exact position the template's own Value.
  bool gridMatches(Grid& grid,const V& member,std::int64_t column,std::int64_t row) {
    std::vector<std::pair<std::size_t,V>> work;work.emplace_back(grid.root,member);
    const auto no=[](int,const GridNode&,const V&,std::int64_t=0,std::int64_t=0){return false;};
    while(!work.empty()) {
      const auto index=work.back().first;
      const auto x=g.canonicalInterpretation(work.back().second);work.pop_back();
      auto& r=record(x);
      const auto& n=grid.nodes[index];
      const std::pair<std::size_t,const Record*> key{index,&r};
      if(const auto found=grid.checked.find(key);found!=grid.checked.end()) {
        // The same Value under other member coordinates has the leaves it had: no second map.
        if((n.depends_column && found->second.first!=column) || (n.depends_row && found->second.second!=row))
          return no(1,n,x,found->second.first,found->second.second);
        continue;
      }
      if(++grid_stats.work>grid_budget)return no(2,n,x);
      if(n.same) {
        // Another Value at an exact position must be that value (e.g. another ld.param of one
        // parameter, another Load of the same stored bytes).
        if(&r!=n.record && !gridEqual(n.value,x))return no(3,n,x);
      } else {
        auto p=gridPlan(x,r);
        if(n.form==Form::Offset && p->form!=Form::Offset) {
          if(g.type(x)!=g.type(n.value))return no(4,n,x);
          p=asOffset(x);
        }
        if(p->form!=n.form)return no(5,n,x,std::int64_t(n.form),std::int64_t(p->form));
        if(p->signature!=n.signature)return no(6,n,x);
        if(p->row_bytes!=n.row_bytes || p->children.size()!=n.children.size())return no(7,n,x);
        if(n.leaf || n.payload) {
          std::int64_t expected=n.coordinate,term=0;
          if(__builtin_mul_overflow(n.column,column,&term) || __builtin_add_overflow(expected,term,&expected) ||
             __builtin_mul_overflow(n.row,row,&term) || __builtin_add_overflow(expected,term,&expected))return no(8,n,x);
          auto actual=p->coordinate;
          if(n.power){if(!powerOfTwo(actual))return no(9,n,x,expected,actual);actual=bitIndex(actual);}
          if(actual!=expected)return no(10,n,x,expected,actual);
          if(n.leaf && p->varying!=(n.form==Form::Leaf?n.mapped:true))return no(11,n,x);
        }
        for(std::size_t i=0;i<n.children.size();++i)work.emplace_back(n.children[i],p->children[i].value);
      }
      grid.checked.emplace(key,std::pair{column,row});
    }
    return true;
  }
  // The use DAG of a checked template. Member index m = outer*period + inner; rows_outer: the
  // rows are the outer axis (period = columns), else the columns are (period = rows).
  Ptr gridUses(Grid& grid,std::uint64_t period,bool rows_outer) {
    // Every edge is the identity map: each node sees the member index, and a mapped leaf, slice
    // or fragment maps it in its own binding (base + stride*(m % period) + outer_stride*(m / period)).
    const auto edge=[&](std::size_t index,const V& handle) {
      auto binding=std::make_shared<IndexedValueBinding>(IndexedValueBinding{handle,0,1,{}});
      return std::make_shared<U>(U{handle,U::Kind::CoordinateMap,std::move(binding),{grid.nodes[index].use}});
    };
    const auto mappedBinding=[&](const GridNode& n) {
      auto binding=std::make_shared<IndexedValueBinding>(*n.binding);
      binding->base=n.coordinate;binding->period=period;
      binding->stride=rows_outer?n.column:n.row;binding->outer_stride=rows_outer?n.row:n.column;
      binding->power_of_two=n.power;binding->addend=n.form==Form::Offset;
      return binding;
    };
    // Children were completed before their parents: one pass in node order.
    for(auto& n:grid.nodes) {
      // A constant position has the first member's value in every member (checked member by
      // member: equal forms, signatures, coordinates and constant children): held exact.
      if(n.same || n.constant)n.use=std::make_shared<U>(U{n.value,U::Kind::Exact,{},{}});
      else if(n.leaf) {
        if(n.form==Form::Literal && !n.mapped)n.use=std::make_shared<U>(U{n.value,U::Kind::Operands,{},{}});
        else if(n.mapped)n.use=std::make_shared<U>(U{n.value,U::Kind::Binding,mappedBinding(n),{}});
        else n.use=std::make_shared<U>(U{n.value,U::Kind::Binding,std::make_shared<IndexedValueBinding>(*n.binding),{}});
      } else {
        std::vector<Ptr> children;children.reserve(n.children.size());
        for(std::size_t i=0;i<n.children.size();++i)children.push_back(edge(n.children[i],n.child_values[i]));
        if(n.payload)n.use=std::make_shared<U>(U{n.value,
            n.form==Form::BitSelect?U::Kind::BitTestSelect:U::Kind::Binding,mappedBinding(n),std::move(children)});
        else {
          const auto kind=n.form==Form::Forward?U::Kind::Forward:n.form==Form::Conversion?U::Kind::Conversion:U::Kind::Operands;
          n.use=std::make_shared<U>(U{n.value,kind,{},std::move(children)});
        }
      }
    }
    return edge(grid.root,grid.nodes[grid.root].value);
  }
  // Merges rows of equal one-axis domains of `plan` into two-axis domains (IndexedSourceDomain
  // outer_count/outer_stride), or one-axis runs when the domains are single records. A domain
  // tries the next kGridPartners equal domains as its second row; a family needs kGridMinRows
  // rows. Domains that join no family are returned unchanged (the result then has the plan's
  // size). Public for a caller that keeps the one-axis plan (mapLine) as its fallback.
 public:
  // runs=false: single records are not chained into one-axis runs (only rows of several records
  // become families); a read asks for this.
  std::vector<IndexedSourceDomain> gridDomains(std::span<const V> values,std::span<const std::uint64_t> offsets,
                                               std::vector<IndexedSourceDomain> plan,bool runs=true) {
    if(plan.size()<kGridMinRows)return plan;
    struct Clear {PureSourceTemplates& t;~Clear(){t.grid_plans.clear();t.grid_equal.clear();}} clear{*this};
    grid_budget=grid_stats.work+256*values.size()+(std::size_t(1)<<20);
    std::unordered_map<std::size_t,std::size_t> at;at.reserve(plan.size());
    for(std::size_t k=0;k<plan.size();++k)at.emplace(plan[k].begin,k);
    std::vector<char> used(plan.size());
    std::vector<IndexedSourceDomain> result;result.reserve(plan.size());
    const auto compatible=[&](const IndexedSourceDomain& x,const IndexedSourceDomain& y) {
      return x.count==y.count && x.result_stride==y.result_stride && x.outer_count==1 && y.outer_count==1 &&
          bool(x.source)==bool(y.source) && (offsets.empty() || offsets[x.begin]==offsets[y.begin]);
    };
    for(std::size_t k=0;k<plan.size();++k) {
      if(used[k])continue;
      const auto& first=plan[k];bool merged=false;
      if(first.count==1 && !runs){used[k]=true;result.push_back(first);continue;}
      const auto inner_extent=(first.count-1)*first.result_stride;
      std::size_t tried=0;
      for(std::size_t j=k+1;j<plan.size() && tried<kGridPartners && !merged && grid_stats.work<grid_budget;++j) {
        const auto& next=plan[j];
        if(used[j] || !compatible(first,next) || next.begin<=first.begin)continue;
        ++tried;
        const auto step=next.begin-first.begin;
        bool reaches=true;
        for(std::size_t r=2;r<kGridMinRows && reaches;++r) {
          const auto found=at.find(first.begin+r*step);
          reaches=found!=at.end() && !used[found->second] && compatible(first,plan[found->second]);
        }
        if(!reaches)continue;
        ++grid_stats.attempts;
        std::optional<Grid> grid;std::vector<std::size_t> rows;
        // Literals that double along an axis first as bit indexes, then (a short family) as values.
        for(const bool power_literals:{true,false}) {
          grid=gridTemplate(values[first.begin],first.count>1?values[first.begin+first.result_stride]:values[first.begin],
              values[next.begin],power_literals);
          rows.clear();
          if(!grid || !grid->special)break;
          for(std::size_t r=0;;++r) {
            const auto found=at.find(first.begin+r*step);
            if(found==at.end() || used[found->second] || !compatible(first,plan[found->second]))break;
            bool matches=true;
            for(std::size_t c=0;c<first.count && matches;++c)
              matches=gridMatches(*grid,values[plan[found->second].begin+c*first.result_stride],std::int64_t(c),std::int64_t(r));
            if(!matches)break;
            rows.push_back(found->second);
          }
          if(rows.size()>=kGridMinRows || !std::any_of(grid->nodes.begin(),grid->nodes.end(),[](const GridNode& n){return n.power;}))break;
        }
        // Scope: a family is made only where its template has a gathered fragment or a mask bit
        // test at a varying position (rows gathered by per-member origins, per-member bits of a
        // mask word): what the one-axis policies cannot express and what this planner was built
        // for. Other rows stay row domains: output consolidation merges such rows afterwards where
        // that is wanted (strided rows), other consumers compare them row by row, and the fold
        // projects them from its per-row caches. Measured: with Slice, Read or Offset positions
        // starting families too, cake_vsa's lowering (no fragments) ran past the 1000 s limit
        // under node pressure; with this scope its lowering is as without grid families (same
        // peak, collections and families). Another partner
        // step has the same forms, so this domain is not tried again, and neither are the equal
        // domains at this step after it (the rows of the same would-be family): one template walk
        // per chain of rows, not one per row. They are returned unchanged.
        if(grid && !grid->special) {
          ++grid_stats.plain;
          for(std::size_t r=1;;++r) {
            const auto found=at.find(first.begin+r*step);
            if(found==at.end() || used[found->second] || !compatible(first,plan[found->second]))break;
            used[found->second]=true;result.push_back(plan[found->second]);
          }
          break;
        }
        if(!grid || rows.size()<kGridMinRows)continue;
        const bool rows_outer=step>inner_extent;
        if(!rows_outer && !(first.result_stride>(rows.size()-1)*step))continue;
        IndexedSourceDomain domain;domain.begin=first.begin;
        if(first.count==1) {
          domain.count=rows.size();domain.result_stride=step;
        } else if(rows_outer) {
          domain.count=first.count;domain.result_stride=first.result_stride;
          domain.outer_count=rows.size();domain.outer_stride=step;
        } else {
          domain.count=rows.size();domain.result_stride=step;
          domain.outer_count=first.count;domain.outer_stride=first.result_stride;
        }
        auto root=gridUses(*grid,rows_outer?first.count:rows.size(),rows_outer);
        const auto prototype=root->prototype;
        domain.source=std::make_shared<IndexedSource>(IndexedSource{prototype,first.count*rows.size(),std::move(root)});
        for(const auto row:rows)used[row]=true;
        ++grid_stats.families;grid_stats.members+=first.count*rows.size();grid_stats.domains+=rows.size();
        result.push_back(std::move(domain));merged=true;
      }
      if(!merged){used[k]=true;result.push_back(first);}
    }
    std::sort(result.begin(),result.end(),[](const auto& x,const auto& y){return x.begin<y.begin;});
    return result;
  }
 public:
  // Marks every Value that the plan DAGs of two of `values` both reach (Record::shared): one
  // node in both, so a template over them holds it fixed. Sound for every later call too: a
  // template that keeps a Value exact requires that very Value in each member. One visit per
  // distinct reachable Value per call, and a Value already shared is not descended: O(new DAG).
  // Returns whether the plan DAGs of `values` reach a shared Value (new or from an earlier call);
  // otherwise policy 6 summarizes them exactly as the joint policy 0 does.
  bool sharedSubtrees(std::span<const V> values) {
    const auto stamp=++share_stamp;
    std::vector<Record*> work;bool reached=false;
    // (child, parent) record edges of the walk, for the touched marks below.
    std::vector<std::pair<Record*,Record*>> edges;
    for(std::size_t i=0;i<values.size();++i) {
      work.assign(1,&record(g.canonicalInterpretation(values[i])));
      while(!work.empty()) {
        auto& r=*work.back();work.pop_back();
        if(r.share_stamp==stamp) {
          if(r.share_root!=i+1)r.shared=true;
          reached=reached || r.shared;
          continue;
        }
        r.share_stamp=stamp;r.share_root=i+1;
        // Shared in an earlier call: held exact as a whole, so its subtree is not walked again
        // (a later call costs only its new Values).
        if(r.shared){reached=true;continue;}
        // The children records a plan of this Value already produced (summarize or an earlier
        // walk); a plan is made only for a Value never planned before.
        if(!r.children_known){const auto p=plan(*r.key);noteChildren(r,p);}
        for(auto* child:r.children){work.push_back(child);edges.emplace_back(child,&r);}
      }
    }
    if(!reached)return false;
    // Touched: a shared Value and every Value whose plan DAG reaches one. Policy 6 summarizes
    // an untouched Value exactly as policy 0 (summarize reuses that summary). Upward search from
    // the shared Values over the sorted edges: O(E log E) for the E edges of this walk.
    std::sort(edges.begin(),edges.end());
    std::vector<Record*> frontier;
    for(const auto& [child,parent]:edges)if(child->shared && child->touch_stamp!=stamp){child->touch_stamp=stamp;frontier.push_back(child);}
    while(!frontier.empty()) {
      auto* x=frontier.back();frontier.pop_back();
      for(auto it=std::lower_bound(edges.begin(),edges.end(),std::pair<Record*,Record*>{x,nullptr});
          it!=edges.end() && it->first==x;++it)
        if(it->second->touch_stamp!=stamp){it->second->touch_stamp=stamp;frontier.push_back(it->second);}
    }
    return true;
  }
  std::size_t share_stamp=0;
  // Whether the last map() returned the shared-subtree plan.
  bool last_map_shared=false;
  // shared_subtrees: also try the shared-subtree policy (policy 6). View normalization's read
  // and output-view planning ask for it; the scalar fold's own planning (scalar output runs,
  // scalar reduction chains, packed record fields) keeps the previous policies.
  // grid: also merge rows of equal one-axis domains into two-axis domains (gridDomains); the
  // caller then handles IndexedSourceDomain::outer_count.
  std::vector<IndexedSourceDomain> map(std::span<const V> values,bool interleaved=false,
                                     std::span<const std::uint64_t> offsets={},bool shared_subtrees=false,
                                     bool grid=false,bool grid_runs=true) {
    auto planned=mapLine(values,interleaved,offsets,shared_subtrees);
    return grid?gridDomains(values,offsets,std::move(planned),grid_runs):planned;
  }
  std::vector<IndexedSourceDomain> mapLine(std::span<const V> values,bool interleaved,
                                         std::span<const std::uint64_t> offsets,bool shared_subtrees) {
    PlanScope plan_scope(*this);  // the shared-subtree strategy reuses the other strategies' plans
    // Member-domain planning is shared by every policy, the shared-subtree policy included.
    const auto members=mapSourceTemplates(g,values,interleaved,offsets);
    auto planned=mapPolicies(values,interleaved,offsets,members);
    // Values that differ only in leaves inside one shared subtree (a row scale every column
    // multiplies by) are no joint template, and the fixed-tail policy keeps their second
    // varying operand exact: hold exactly the shared subtrees fixed instead (policy 6).
    // Taken only when it gives a strictly smaller plan.
    last_map_shared=false;
    if(planned.size()<=1 || !shared_subtrees)return planned;
    // Without a shared Value in reach, policy 6 is policy 0, which mapPolicies already tried.
    if(!sharedSubtrees(values))return planned;
    auto shared=mapStrategy(values,interleaved,offsets,6,members);
    if(!(planCost(shared)<planCost(planned)))return planned;
    last_map_shared=true;
    return shared;
  }
  std::vector<IndexedSourceDomain> mapPolicies(std::span<const V> values,bool interleaved,
                                               std::span<const std::uint64_t> offsets) {
    return mapPolicies(values,interleaved,offsets,mapSourceTemplates(g,values,interleaved,offsets));
  }
  std::vector<IndexedSourceDomain> mapPolicies(std::span<const V> values,bool interleaved,
                                               std::span<const std::uint64_t> offsets,
                                               const std::vector<IndexedSourceDomain>& members) {
    // Four bounded whole-DAG policies, not combinations of arbitrary tree matches.
    // Joint coordinates support A(i)+B(i); fixed-tail supports f(A(i), bias);
    // row maps support A(row,col)+bias(row) with producer-defined row widths.
    // Each policy visits each Value/direct edge once and shares immutable uses.
    // Member-domain planning is identical for all policies. Build it once,
    // rather than repeating annotation formatting/grouping and use allocation.
    if(members.size()<=1)return members;
    PlanScope plan_scope(*this);
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
