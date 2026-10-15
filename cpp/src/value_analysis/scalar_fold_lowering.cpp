#include "sentryward/algorithm/diagnostics/development_trace.h"
#ifdef SENTRYWARD_ENABLE_COMPRESSION_AUDIT
#include "sentryward/scalar_fold/printer.h"
#endif
#include "sentryward/value_analysis/scalar_fold_lowering.h"
#include "sentryward/value_analysis/indexed_source.h"
#include "sentryward/value_analysis/pure_source_templates.h"
#include "sentryward/algorithm/base/affine_index.h"
#include "sentryward/value_analysis/indexed_source_identity.h"
#include "sentryward/value_analysis/packed_storage.h"
#include "sentryward/value_analysis/scalar_specialization.h"
#include "sentryward/value_analysis/ideal_scalar_semantics.h"
#include "sentryward/algorithm/diagnostics/compression_audit.h"
#include "sentryward/value_analysis/projection_ledger.h"
#include "sentryward/scalar_fold/reduction_builder.h"
#include "sentryward/scalar_fold/real_expression_builder.h"
#include "sentryward/equivalence/constant_values.h"
#include "sentryward/scalar_fold/consolidate_outputs.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/native_matrix.h"
#include "sentryward/value_ir/ptx_operation.h"
#include "sentryward/frontend/ptx/layouts.h"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <limits>
#include <functional>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace sentryward::value_analysis {
namespace {
namespace v = value_ir;
namespace s = scalar_fold;
struct Failure : std::runtime_error {
  std::string code;
  Failure(std::string c, std::string message):std::runtime_error(std::move(message)),code(std::move(c)){}
};
// Exact source-boundary refinement, not a scalar fallback or speculative
// bisection. Cuts are inverse images in the current output coordinate domain.
struct OutputDomainCuts { std::vector<std::uint64_t> cuts; unsigned axis=0; };
struct OutputDomainRadix { unsigned axis; std::uint64_t period; };
struct RetryFoldDomain {std::vector<std::string> keep;};
struct CollectProjectionFrontier {};
[[noreturn]] void fail(std::string c,std::string m){throw Failure(std::move(c),std::move(m));}
std::int64_t add(std::int64_t a,std::int64_t b){
  std::int64_t out; if(__builtin_add_overflow(a,b,&out)) fail("scalar_fold_index_overflow",
      "index addition exceeds signed 64-bit range: "+std::to_string(a)+" + "+std::to_string(b));
  return out;
}
std::int64_t mul(std::int64_t a,std::int64_t b){
  std::int64_t out; if(__builtin_mul_overflow(a,b,&out)) fail("scalar_fold_index_overflow",
      "index multiplication exceeds signed 64-bit range: "+std::to_string(a)+" * "+std::to_string(b));
  return out;
}
std::int64_t integer(std::uint64_t x){
  if(x>std::uint64_t(INT64_MAX)) fail("index_representation_overflow","index exceeds signed 64-bit domain"); return std::int64_t(x);
}
struct OffsetExpression;
struct Offset {
  std::int64_t base=0;
  std::map<unsigned,std::int64_t> terms;
  std::shared_ptr<const OffsetExpression> expression;
  Offset()=default;
  Offset(std::int64_t b):base(b){}
  std::string key()const;
  std::uint64_t binderMask()const;
};
struct OffsetExpression {
  enum Kind {Add,Scale,Quotient,Remainder} kind;
  Offset left,right;
  std::int64_t factor;
  std::string key;
  std::uint64_t binders;
};
std::uint64_t Offset::binderMask()const {
  if(expression)return expression->binders;
  std::uint64_t mask=0;
  for(auto [level,c]:terms)if(c) {
    if(level>=64)return UINT64_MAX;
    mask|=std::uint64_t{1}<<level;
  }
  return mask;
}
std::string Offset::key()const {
  if(expression)return expression->key;
  std::string k=std::to_string(base);
  for(auto [level,c]:terms) k+=","+std::to_string(level)+":"+std::to_string(c);
  return k;
}
Offset compound(OffsetExpression::Kind kind,Offset left,Offset right={},std::int64_t factor=0) {
  auto key="("+std::to_string(kind)+":"+std::to_string(factor)+":"+left.key()+":"+right.key()+")";
  const auto binders=left.binderMask()|right.binderMask();
  Offset result;
  result.expression=std::make_shared<const OffsetExpression>(OffsetExpression{kind,std::move(left),std::move(right),factor,std::move(key),binders});
  return result;
}
Offset plus(Offset a,const Offset& b){
  if(a.expression || b.expression) {
    if(!a.expression && a.base==0 && a.terms.empty())return b;
    if(!b.expression && b.base==0 && b.terms.empty())return a;
    const auto constant=[](const Offset& x){return !x.expression && x.terms.empty();};
    if(a.expression && a.expression->kind==OffsetExpression::Add && constant(a.expression->left)) {
      const auto& e=*a.expression;
      if(constant(b))return plus(e.right,Offset(add(e.left.base,b.base)));
      if(b.expression && b.expression->kind==OffsetExpression::Add && constant(b.expression->left))
        return plus(Offset(add(e.left.base,b.expression->left.base)),plus(e.right,b.expression->right));
    }
    if(constant(a) && b.expression && b.expression->kind==OffsetExpression::Add && constant(b.expression->left))
      return plus(b.expression->right,Offset(add(a.base,b.expression->left.base)));
    // Keep a constant displacement at the outer Add. This is a local shape
    // rule, not a traversal to rediscover affine terms in an expression tree.
    if(a.expression && a.expression->kind==OffsetExpression::Add &&
       !a.expression->left.expression && a.expression->left.terms.empty() && b.expression)
      return compound(OffsetExpression::Add,a.expression->left,
          compound(OffsetExpression::Add,a.expression->right,b));
    if(a.expression && !b.expression && b.terms.empty())return compound(OffsetExpression::Add,b,std::move(a));
    return compound(OffsetExpression::Add,std::move(a),b);
  }
  a.base=add(a.base,b.base);
  for(auto [level,c]:b.terms){a.terms[level]=add(a.terms[level],c);if(!a.terms[level])a.terms.erase(level);}
  return a;
}
Offset times(Offset a,std::int64_t c){
  if(!c)return Offset{};
  if(c==1)return a;
  if(a.expression && a.expression->kind==OffsetExpression::Add &&
     !a.expression->left.expression && a.expression->left.terms.empty()) {
    const auto& e=*a.expression;
    return plus(Offset(mul(e.left.base,c)),times(e.right,c));
  }
  if(a.expression)return compound(OffsetExpression::Scale,std::move(a),{},c);
  a.base=mul(a.base,c);
  for(auto& [level,v]:a.terms){(void)level;v=mul(v,c);}
  return a;
}
std::uint64_t alignment(const Offset& x) {
  const auto magnitude=[](std::int64_t n){return n<0?std::uint64_t{0}-std::uint64_t(n):std::uint64_t(n);};
  if(x.expression) {
    const auto& e=*x.expression;auto left=alignment(e.left);
    if(e.kind==OffsetExpression::Add)return std::gcd(left,alignment(e.right));
    if(e.kind==OffsetExpression::Scale) {
      auto factor=magnitude(e.factor);
      if(factor && left>UINT64_MAX/factor)return 1;
      return left*factor;
    }
    const auto divisor=static_cast<std::uint64_t>(e.factor);
    if(e.kind==OffsetExpression::Remainder)return std::gcd(left,divisor);
    return left%divisor==0?left/divisor:1;
  }
  auto divisor=magnitude(x.base);
  for(auto [level,c]:x.terms){(void)level;divisor=std::gcd(divisor,magnitude(c));}
  return divisor;
}
std::optional<std::int64_t> literal(std::string text){
  if(text.empty() || text.starts_with("0f") || text.starts_with("0d"))return {};
  if(text.back()=='U' || text.back()=='u')text.pop_back();
  try{std::size_t used=0;auto n=std::stoll(text,&used,0);if(used==text.size())return n;}catch(const std::exception&){}
  return {};
}
std::string attributes(const v::Operation& op){
  std::ostringstream out;op.printAttributes(out);return out.str();
}
std::string pointerLabel(const v::PointerProperties& p){
  std::string label="pointer={identity="+std::to_string(p.identity.size())+":"+p.identity+
    ",addressing="+std::to_string(p.addressing.size())+":"+p.addressing;
  label+=",cta="+(p.owner_cta?std::to_string(*p.owner_cta):"none");
  label+=",thread="+(p.owner_thread?std::to_string(*p.owner_thread):"none");
  return label+"}";
}
std::optional<std::int64_t> knownInteger(std::int64_t value,const v::ElementType& type){
  if(!type.isBitLike() || type.kind()==v::ElementKind::Predicate ||
     type.isParameterized() || !type.width() || type.width()>64)return {};
  const auto width=type.width();
  const auto mask=width==64?UINT64_MAX:((std::uint64_t{1}<<width)-1);
  auto bits=std::uint64_t(value)&mask;
  if(type.kind()==v::ElementKind::SignedInteger){
    if(bits&(std::uint64_t{1}<<(width-1)))bits|=~mask;
    return std::bit_cast<std::int64_t>(bits);
  }
  if(bits>std::uint64_t(INT64_MAX))return {};
  return std::int64_t(bits);
}
std::string valueKey(const v::Value& x){
  std::string key=std::to_string(x.nodeId())+":"+std::to_string(x.instanceIndex())+":"+std::to_string(x.resultIndex());
  if(x.conversion())key+="<"+v::typeConversionLabel(x)+":"+valueKey(x.conversion()->source)+">";
  for(auto i:x.indices())key+="["+valueKey(i)+"]";
  if(x.threadSelector())key+="{"+valueKey(*x.threadSelector())+"}";
  if(x.pointer())key+="|"+pointerLabel(*x.pointer());
  return key;
}
std::optional<unsigned> scalarBytes(const std::string& type) {
  for(const auto& [name,bytes]:std::initializer_list<std::pair<const char*,unsigned>>{
      {"b8",1},{"b16",2},{"b32",4},{"b64",8},{"u8",1},{"u16",2},{"u32",4},{"u64",8},
      {"s8",1},{"s16",2},{"s32",4},{"s64",8},{"f16",2},{"bf16",2},{"f32",4},{"f64",8}})
    if(type==name)return bytes;
  return {};
}
void coalesceFragments(s::Graph& graph,std::vector<ScalarFoldOutput>& outputs) {
  std::sort(outputs.begin(),outputs.end(),[](const auto& a,const auto& b){
    return std::tie(a.region.allocation,a.region.offset)<std::tie(b.region.allocation,b.region.offset);
  });
  std::vector<ScalarFoldOutput> compact;
  for(std::size_t i=0;i<outputs.size();) {
    const auto& first=outputs[i];
    const auto& node=graph.node(first.value);
    if(node.kind!=s::Kind::Operation || node.name!="ExtractBits" || node.operands.size()!=2 ||
       !first.value.conversion.empty() || !first.value.pointer_provenance.empty()) {
      compact.push_back(first);++i;continue;
    }
    const auto source=node.operands[0];
    const auto width=scalarBytes(graph.type(source));
    std::uint64_t covered=0;
    auto j=i;
    if(width)for(;j<outputs.size() && covered<*width;++j) {
      const auto& output=outputs[j];
      const auto& part=graph.node(output.value);
      if(output.region.allocation!=first.region.allocation ||
         covered>UINT64_MAX-first.region.offset || output.region.offset!=first.region.offset+covered ||
         !output.region.bytes || output.region.bytes>*width-covered ||
         !output.value.conversion.empty() || !output.value.pointer_provenance.empty() ||
         part.kind!=s::Kind::Operation || part.name!="ExtractBits" || part.operands.size()!=2 ||
         part.operands[0]!=source || part.attributes!="width="+std::to_string(output.region.bytes*8) ||
         scalarBytes(part.type)!=std::optional<unsigned>(output.region.bytes) ||
         !part.operands[1].conversion.empty() || !part.operands[1].pointer_provenance.empty())break;
      const auto& offset=graph.node(part.operands[1]);
      if(offset.kind!=s::Kind::Constant || offset.type!="u64" || !offset.attributes.empty() ||
         offset.name!=std::to_string(covered*8))break;
      covered+=output.region.bytes;
    }
    if(width && covered==*width && j>i+1) {
      auto combined=first;combined.region.bytes=*width;combined.value=source;
      compact.push_back(std::move(combined));i=j;
    } else {compact.push_back(first);++i;}
  }
  outputs=std::move(compact);
}
class Lower {
  const v::Graph& input;
  const MemoryEliminationResult& memory;
  std::map<std::string,v::Value> allocation_bases;
  // Contiguous stored initial-read domains; built once, never by a projection
  // query enumerating logical members. The address remains a Value operand.
  std::map<MemoryInstance,std::size_t> symbolic_read_domains;
  std::map<std::size_t,v::ThreadMembers> member_coordinates;
  struct SelectorRun {std::size_t end;std::int64_t thread;};
  std::map<std::string,std::map<std::size_t,SelectorRun>> member_selectors;
  const ScalarFoldOptions& options;
  ScalarSpecialization specialization{input};
  ScalarFoldResult result;
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
  std::unique_ptr<debug::ProjectionLedger> ledger=debug::ProjectionLedger::enabled()?
      std::make_unique<debug::ProjectionLedger>():nullptr;
#endif
  struct OutputDomain {
    std::vector<std::uint64_t> shape,strides;
    std::uint64_t count;
    OutputDomain(std::uint64_t n):shape{n},count(n){}
    OutputDomain(std::vector<std::uint64_t> sizes,std::vector<std::uint64_t> steps)
        :shape(std::move(sizes)),strides(std::move(steps)),count(1) {
      if(shape.empty() || strides.size()!=shape.size())fail("scalar_fold_prerequisite","invalid output layout rank");
      for(auto n:shape) {
        if(!n || count>UINT64_MAX/n)fail("scalar_fold_prerequisite","invalid output layout extent");
        count*=n;
      }
    }
    s::OutputFamily family(const ScalarFoldOutput& output) const {
      if(!count || output.region.bytes%count)fail("scalar_fold_prerequisite","invalid output payload extent");
      const auto width=output.region.bytes/count;
      return {output.region.allocation,output.region.offset,width,shape,
          strides.empty()?std::vector<std::uint64_t>{width}:strides,output.value};
    }
  };
  std::map<std::size_t,OutputDomain> output_domains;
  s::ReductionBuilder reductions{result.graph};
  s::RealExpressionBuilder real_expressions{result.graph,reductions,[this]{limit();}};
  std::unordered_map<std::string,s::Ref> cache;
  std::unordered_map<std::string,std::pair<Failure,std::optional<MemoryInstance>>> failed_projections;
  struct ViewInterval {
    std::uint64_t begin,end,prefix_end;
    std::size_t block;
  };
  std::unordered_map<const MemoryView*,std::vector<ViewInterval>> view_intervals;
  std::set<std::string> visiting;
  std::vector<std::uint64_t> domains;
  unsigned output_refinement_depth=0;
  unsigned output_domain_rank=0;
  struct IndexedContext {
    Offset index;
    std::shared_ptr<const IndexedSourceUse> use = {};
  };
  std::shared_ptr<const IndexedContext> indexed_context;
  // Lazily project the already-existing template graph. The key describes a
  // use's instance map, not an equality conjecture about two expressions.
  std::map<std::tuple<std::string,std::int64_t,std::int64_t>,
      std::shared_ptr<const IndexedSourceUse>> member_uses;
  std::map<const IndexedSourceUse*,std::shared_ptr<const IndexedSourceUse>> expanded_member_uses;
  struct InitialByteMap {
    std::size_t first,count;
    ConstantRegion initial;
    std::int64_t stride;
    std::size_t period=0;
    std::int64_t outer_stride=0;
  };
  std::map<std::size_t,std::vector<InitialByteMap>> initial_byte_maps;
  const InitialByteMap* initialByteMap(std::size_t node,std::size_t first,std::size_t last) {
    auto [entry,inserted]=initial_byte_maps.try_emplace(node);
    if(inserted) {
      // Initial-symbol IDs describe retained read records, not different
      // mathematical arrays. Coalesce their certified byte maps without
      // changing IDs/provenance in MemoryEliminationResult. Every stored run
      // is checked, never just representative logical members.
      for(auto it=memory.initial_read_families.lower_bound({node,0});
          it!=memory.initial_read_families.end() && it->first.node==node;++it) {
        const auto& f=it->second;
        if(!entry->second.empty()) {
          auto& previous=entry->second.back();
          const auto delta=__int128(f.initial.region.offset)-previous.initial.offset;
          const auto stride=previous.count==1?delta:__int128(previous.stride);
          if(previous.first<=SIZE_MAX-previous.count && previous.first+previous.count==f.first.instance &&
              previous.initial.allocation==f.initial.region.allocation && previous.initial.bytes==f.initial.region.bytes &&
              stride>=INT64_MIN && stride<=INT64_MAX &&
              delta==stride*previous.count && (f.count==1 || f.byte_stride==stride) &&
              f.count<=SIZE_MAX-previous.count) {
            previous.stride=static_cast<std::int64_t>(stride);previous.count+=f.count;continue;
          }
        }
        entry->second.push_back({f.first.instance,f.count,f.initial.region,f.byte_stride});
      }
      // Compile exact first-difference runs. A periodic carry can straddle
      // the arbitrary boundaries of stored affine records (e.g. an elected
      // lane removed every warp). Validate every run and junction; do not
      // enumerate members or require the storage segmentation to align.
      std::vector<InitialByteMap> periodic;
      const auto& runs=entry->second;
      for(std::size_t begin=0;begin<runs.size();) {
        auto end=begin+1,anchor=begin;
        while(end<runs.size() && runs[end-1].first<=SIZE_MAX-runs[end-1].count &&
            runs[end-1].first+runs[end-1].count==runs[end].first &&
            runs[end].initial.allocation==runs[begin].initial.allocation &&
            runs[end].initial.bytes==runs[begin].initial.bytes) {
          if(runs[end].count>runs[anchor].count)anchor=end;
          ++end;
        }
        const auto baseline=runs[anchor].stride;
        std::size_t period=0,anomalies=0;
        __int128 carry=0;
        const auto edge=[&](std::size_t at,std::size_t count,__int128 slope) {
          if(!count || slope==baseline)return true;
          if(count!=1)return false;
          const auto jump=at-runs[begin].first+1;
          if(!period){period=jump;carry=slope-baseline;}
          ++anomalies;
          return __int128(jump)==__int128(period)*anomalies && slope-baseline==carry;
        };
        bool valid=true;
        for(std::size_t i=begin;valid && i<end;++i) {
          const auto& r=runs[i];
          valid=edge(r.first,r.count-1,r.stride);
          if(valid && i+1<end)
            valid=edge(r.first+r.count-1,1,__int128(runs[i+1].initial.offset)-
                (__int128(r.initial.offset)+__int128(r.stride)*(r.count-1)));
        }
        const auto count=runs[end-1].first-runs[begin].first+runs[end-1].count;
        const auto outer=__int128(baseline)*period+carry;
        if(valid && period && anomalies==(count-1)/period && outer>=INT64_MIN && outer<=INT64_MAX) {
          auto map=runs[begin];map.count=count;map.stride=baseline;
          map.period=period;map.outer_stride=static_cast<std::int64_t>(outer);
          periodic.push_back(std::move(map));
        } else for(auto i=begin;i<end;++i)periodic.push_back(runs[i]);
        begin=end;
      }
      entry->second=std::move(periodic);
    }
    const auto& maps=entry->second;
    auto it=std::upper_bound(maps.begin(),maps.end(),first,[](auto i,const auto& map){return i<map.first;});
    if(it==maps.begin())return nullptr;
    --it;
    return first<=last && last-it->first<it->count?&*it:nullptr;
  }
  struct PeriodicReadMap {
    const ReadSlicePattern* first;
    std::size_t count,period;
    std::int64_t byte_outer,member_outer;
  };
  std::map<std::pair<std::size_t,std::uint64_t>,std::vector<PeriodicReadMap>> periodic_read_maps;
  const PeriodicReadMap* periodicReadMap(std::size_t node,std::size_t first,
      std::size_t last,std::uint64_t offset,std::uint64_t width) {
    auto bucket=memory.read_slice_patterns.upper_bound({node,offset});
    if(bucket==memory.read_slice_patterns.begin())return nullptr;
    --bucket;if(bucket->first.first!=node)return nullptr;
    auto [cached,inserted]=periodic_read_maps.try_emplace(bucket->first);
    if(inserted) {
      // Compile the already stored affine segments once. This compares only
      // source identity and numeric maps, never operands or logical members.
      const auto& patterns=bucket->second;
      for(auto begin=patterns.begin();begin!=patterns.end();) {
        const auto& p=begin->second;auto next=std::next(begin);
        std::int64_t byte_outer=0,member_outer=0;std::size_t blocks=1;
        while(next!=patterns.end()) {
          const auto& q=next->second;
          if(!p.count || q.count!=p.count || !p.sameSourceTemplate(q.source) ||
              q.bytes!=p.bytes || q.result_offset!=p.result_offset ||
              q.byte_stride!=p.byte_stride || q.source_instance_stride!=p.source_instance_stride ||
              static_cast<unsigned __int128>(p.first.instance)+
                  static_cast<unsigned __int128>(blocks)*p.count!=q.first.instance)break;
          const auto bytes=static_cast<__int128>(q.source_offset)-p.source_offset;
          const auto members=static_cast<__int128>(q.source.instanceIndex())-p.source.instanceIndex();
          if(blocks==1) {
            if(bytes<INT64_MIN || bytes>INT64_MAX || members<INT64_MIN || members>INT64_MAX)break;
            byte_outer=static_cast<std::int64_t>(bytes);member_outer=static_cast<std::int64_t>(members);
          } else if(bytes!=static_cast<__int128>(blocks)*byte_outer ||
                    members!=static_cast<__int128>(blocks)*member_outer)break;
          ++blocks;++next;
        }
        if(blocks>1 && p.count<=SIZE_MAX/blocks)
          cached->second.push_back({&p,p.count*blocks,p.count,byte_outer,member_outer});
        begin=next;
      }
    }
    const auto& maps=cached->second;
    auto found=std::upper_bound(maps.begin(),maps.end(),first,
        [](auto member,const auto& map){return member<map.first->first.instance;});
    if(found==maps.begin())return nullptr;
    const auto& map=*std::prev(found);const auto& p=*map.first;
    if(last<first || first<p.first.instance || last-p.first.instance>=map.count ||
        offset<p.result_offset || offset-p.result_offset>p.bytes ||
        width>p.bytes-(offset-p.result_offset))return nullptr;
    return &map;
  }
  std::shared_ptr<const IndexedSourceUse> memberUse(v::Value value,
      std::int64_t base,std::int64_t stride) {
    value=input.canonicalInterpretation(value);
    auto key=std::tuple{valueKey(value),base,stride};
    if(auto found=member_uses.find(key);found!=member_uses.end())return found->second;
    auto binding=std::make_shared<IndexedValueBinding>(IndexedValueBinding{value,base,stride,{}});
    auto use=std::make_shared<IndexedSourceUse>(IndexedSourceUse{value,
        IndexedSourceUse::Kind::MemberTemplate,std::move(binding),{}});
    member_uses.emplace(std::move(key),use);return use;
  }
  IndexedSourceIdentity indexed_use_identity{valueKey};
  // A refinement may turn an existing binder into the singleton [0,1).
  // Its value is then exactly zero. Retaining its old coefficient makes
  // equivalent queries have different cache keys, and repeated coordinate
  // composition can overflow coefficients that contribute nothing at all.
  // Inspect only affected index sub-DAGs, never source/math ancestors.
  Offset specializeSingletons(Offset x) const {
    std::uint64_t fixed=0;
    for(unsigned i=0;i<domains.size() && i<64;++i)
      if(domains[i]==1)fixed|=std::uint64_t{1}<<i;
    if(!(x.binderMask()&fixed))return x;
    std::unordered_map<const OffsetExpression*,Offset> memo;
    const auto visit=[&](auto&& self,Offset at)->Offset {
      if(!(at.binderMask()&fixed))return at;
      if(!at.expression) {
        for(auto it=at.terms.begin();it!=at.terms.end();)
          if(it->first<domains.size() && domains[it->first]==1)it=at.terms.erase(it);
          else ++it;
        return at;
      }
      const auto* e=at.expression.get();
      if(auto found=memo.find(e);found!=memo.end())return found->second;
      auto left=self(self,e->left);
      Offset out;
      switch(e->kind) {
        case OffsetExpression::Add:out=plus(std::move(left),self(self,e->right));break;
        case OffsetExpression::Scale:out=times(std::move(left),e->factor);break;
        case OffsetExpression::Quotient:out=quotient(left,e->factor);break;
        case OffsetExpression::Remainder:out=remainder(left,e->factor);break;
      }
      memo.emplace(e,out);return out;
    };
    return visit(visit,std::move(x));
  }
  const IndexedValueBinding* indexedBinding(v::Value value) {
    if(!indexed_context)return nullptr;
    const auto& use=indexed_context->use;
    if(use && use->kind==IndexedSourceUse::Kind::Binding &&
       use->prototype==input.canonicalInterpretation(value) && use->binding)return &*use->binding;
    return nullptr;
  }
  // Context cleanup must not catch/rethrow a suspended projection at every
  // indexed edge. One throw belongs to the outer projection worklist; scoped
  // restoration visits each frame once during unwinding.
  template<class F> auto inContext(std::shared_ptr<const IndexedContext> next,F&& fn) {
    struct Restore {
      std::shared_ptr<const IndexedContext>& current;
      std::shared_ptr<const IndexedContext> saved;
      ~Restore(){current=std::move(saved);}
    } restore{indexed_context,indexed_context};
    if(next && next->index.binderMask())
      next=std::make_shared<IndexedContext>(IndexedContext{specializeSingletons(next->index),next->use});
    indexed_context=std::move(next);
    return fn();
  }
  // Enter a child by its use edge. Identity of the concrete Value alone does
  // not specify which coordinate substitution is in force.
  template<class F> auto inUse(std::shared_ptr<const IndexedSourceUse> use,F&& fn) {
    auto next=use?std::make_shared<const IndexedContext>(IndexedContext{
        indexed_context->index,std::move(use)}):nullptr;
    return inContext(std::move(next),std::forward<F>(fn));
  }
  // Keep binding identities alive for projection cache keys for the whole pass.
  std::vector<std::shared_ptr<const IndexedSource>> output_source_patterns;
  IndexedSourceMatcher output_source_matcher{input,memory};
  PureSourceTemplates pure_source_templates{input,memory};
  // One finite native-fragment proof per collective occurrence, reused across
  // output coordinates and worklist resumptions. These are operand views,
  // never a table of precomputed output expressions.
  std::map<MemoryInstance,std::array<PartitionedRead,3>> warp_operand_views;
  // The immutable partition owns the read-from occurrence. A cached lattice
  // proves every byte slice in its extent, independently of the caller's
  // output coordinates; the caller retains those coordinates in its Offset.
  using PartitionLatticeKey=std::tuple<const PartitionedRead*,std::string,std::string,
      std::uint64_t,std::uint64_t,std::uint64_t>;
  std::map<PartitionLatticeKey,std::optional<MemoryView>> partition_lattices;
  using PartitionGridKey=std::tuple<const PartitionedRead*,std::string,std::string>;
  std::map<PartitionGridKey,std::optional<MemoryView>> partition_grids;
  using FoldPieceKey=std::pair<std::string,std::uint64_t>;
  struct FoldOwner;
  struct Projection {
    v::Value value;
    Offset offset;
    v::ElementType type;
    std::vector<std::uint64_t> domains;
    std::string key;
    std::unordered_set<std::string> dependencies = {};
    std::shared_ptr<const IndexedContext> indexed = {};
    std::shared_ptr<const FoldOwner> fold = {};
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
    std::uint64_t audit_attempts=0;
#endif
  };
  const Projection* active_projection=nullptr;
  // Inline frames transferred on dependency suspension, innermost first.
  // Keep their direct dependency roots and coordinate context instead of
  // rediscovering the entire inline chain when its leaf becomes available.
  std::vector<Projection> suspended_projections;
  std::optional<Projection> refinement_resume;
  struct FoldOwner {
    FoldPieceKey key;
    unsigned axis;
    std::uint64_t segment,first,count;
    std::shared_ptr<const FoldOwner> parent;
    std::shared_ptr<const Projection> resume;
  };
  std::shared_ptr<const FoldOwner> fold_owner;
  std::map<FoldPieceKey,std::vector<std::pair<std::uint64_t,std::uint64_t>>> fold_pieces;
  struct FoldScope {
    Lower& owner;
    std::shared_ptr<const FoldOwner> saved;
    FoldScope(Lower& o,FoldPieceKey key,unsigned axis,std::size_t segment,
        std::uint64_t first,std::uint64_t count):owner(o),saved(o.fold_owner) {
      if(!o.active_projection)throw std::logic_error("Fold without projection continuation");
      const auto& task=*o.active_projection;
      // Snapshot only the bounded request, never its accumulated dependency set.
      auto resume=std::make_shared<Projection>(Projection{task.value,task.offset,task.type,task.domains,task.key});
      resume->indexed=task.indexed;resume->fold=task.fold;
      owner.fold_owner=std::make_shared<FoldOwner>(FoldOwner{std::move(key),axis,segment,first,count,saved,std::move(resume)});
    }
    ~FoldScope(){owner.fold_owner=std::move(saved);}
  };
  void refineFold(unsigned axis,const std::vector<std::uint64_t>& cuts) {
    SENTRYWARD_LEDGER(ledger->add("fold_refinements");ledger->add("fold_refinement_cuts",cuts.size()));
    auto owner=fold_owner;while(owner && owner->axis!=axis)owner=owner->parent;
    if(!owner)fail("compact_view_unsupported","read partition requires an unowned reduction-domain split");
    auto& pieces=fold_pieces.at(owner->key);
    if(owner->segment>=pieces.size() || pieces[owner->segment]!=std::pair{owner->first,owner->count})
      fail("scalar_fold_internal_binding","stale reduction-domain refinement");
    std::vector<std::pair<std::uint64_t,std::uint64_t>> replacement;
    std::uint64_t start=0;
    for(auto cut:cuts)if(cut>start && cut<owner->count) {
      replacement.emplace_back(owner->first+start,cut-start);start=cut;
    }
    if(!start)fail("compact_view_unsupported","read partition supplied no reduction boundary");
    replacement.emplace_back(owner->first+start,owner->count-start);
    pieces.erase(pieces.begin()+owner->segment);
    pieces.insert(pieces.begin()+owner->segment,replacement.begin(),replacement.end());
    RetryFoldDomain retry;
    refinement_resume=*owner->resume;
    for(auto p=owner;p;p=p->parent)retry.keep.push_back(p->key.first);
    throw retry;
  }
  std::unordered_set<std::string>* requested = nullptr;
  // A suspended reduction owns only its accumulator and the current domain's
  // dependencies. Completed domains are never reconstructed on resumption.
  // References live in cache so collectFrontier can remap them; no raw graph
  // handles survive collection inside the continuation record.
  struct ReductionProgress {
    std::uint64_t next=0;
    std::string accumulator_key;
    std::unordered_set<std::string> pending;
    std::uint64_t sub_next=0;
  };
  std::map<std::string,ReductionProgress> reduction_progress;
  std::map<std::string,std::vector<algorithm::AffineDomain>> reduction_domain_plans;
  const std::string* projection_key=nullptr;
  struct ReductionRequests {
    std::unordered_set<std::string>*& current;
    std::unordered_set<std::string>* previous;
    ReductionRequests(std::unordered_set<std::string>*& target,ReductionProgress& progress)
        :current(target),previous(target){current=&progress.pending;}
    ~ReductionRequests(){current=previous;}
  };
  template<class Init> ReductionProgress& resumeReduction(Init&& initialize) {
    if(!projection_key)throw std::logic_error("reduction outside a projection task");
    const auto key=*projection_key;
    if(auto found=reduction_progress.find(key);found!=reduction_progress.end()) {
      SENTRYWARD_AUDIT(audit->add(AuditCounter::reduction_resumes));
      return found->second;
    }
    // Initialization may itself request a child. Publish the frame only after
    // its initial accumulator has actually been constructed.
    auto initial=initialize();
    auto [found,inserted]=reduction_progress.emplace(key,
        ReductionProgress{0,"$reduction:"+key,{}});
    if(!inserted)throw std::logic_error("duplicate reduction continuation");
    cache.emplace(found->second.accumulator_key,initial);
    return found->second;
  }
  void commitReduction(ReductionProgress& progress,std::uint64_t next,s::Ref accumulator) {
    if(next<=progress.next)throw std::logic_error("reduction continuation did not advance");
    cache.at(progress.accumulator_key)=accumulator;
    progress.next=next;
    progress.sub_next=0;
    // The accumulator now contains the semantic dependencies. Operand cache
    // entries replaced by a Fold are no longer strong roots of this task.
    progress.pending.clear();
    SENTRYWARD_AUDIT(audit->add(AuditCounter::reduction_domains_committed));
  }
  void finishReduction(const std::string& key) {
    if(auto found=reduction_progress.find(key);found!=reduction_progress.end()) {
      cache.erase(found->second.accumulator_key);reduction_progress.erase(found);
    }
    auto piece=fold_pieces.lower_bound({key,0});
    while(piece!=fold_pieces.end() && piece->first.first==key)piece=fold_pieces.erase(piece);
  }
  void discardReductions() {
    for(const auto& [key,progress]:reduction_progress){(void)key;cache.erase(progress.accumulator_key);}
    reduction_progress.clear();
    fold_pieces.clear();
  }
  std::size_t next_collection = 0;
  std::size_t next_completed_collection = 0;
  bool projecting=false;
  bool collecting=false;
  std::size_t projection_depth=0,inline_projection_depth=0;
  std::optional<MemoryInstance> current;
  std::size_t spatial_count=0,spatial_serial=0;
  struct SpatialValues {
    const v::Graph* graph=nullptr;
    v::Value base;
    std::size_t count=0;
    std::int64_t stride=0;
    std::vector<v::Value> explicit_values;
    bool affine()const{return explicit_values.empty();}
    v::Value at(std::size_t i)const {
      if(i>=count)throw std::out_of_range("spatial value instance");
      if(!affine())return explicit_values[i];
      const auto index=static_cast<__int128>(base.instanceIndex())+static_cast<__int128>(stride)*i;
      if(index<0 || index>SIZE_MAX)throw std::logic_error("invalid spatial instance family");
      return graph->node(base.nodeId()).result(base.resultIndex(),static_cast<std::size_t>(index));
    }
    bool equals(const SpatialValues& other)const {
      if(count!=other.count)return false;
      if(affine() && other.affine())return base==other.base && stride==other.stride;
      for(std::size_t i=0;i<count;++i)if(at(i)!=other.at(i))return false;
      return true;
    }
  };
  std::unordered_map<std::string,SpatialValues> spatial_values;
  std::unordered_map<std::string,std::int64_t> spatial_instance_strides;
  void spatialReject(){fail("scalar_fold_spatial_projection","spatial values do not share an exact scalar template");}
  void registerSpatialFamily(SpatialValues family) {
    const auto key=valueKey(family.base);
    if(auto found=spatial_values.find(key);found!=spatial_values.end()) {
      if(!found->second.equals(family))spatialReject();
      return;
    }
    if(family.affine())spatial_instance_strides.insert_or_assign(key,family.stride);
    spatial_values.emplace(key,std::move(family));
  }
  void registerSpatial(std::vector<v::Value> values) {
    SENTRYWARD_AUDIT(audit->add(AuditCounter::spatial_members_enumerated,values.size());
      if(values.size()>1)audit->event("INSTANCE_ENUMERATION","rediscover_spatial_template_from_members",
          values.size(),1,values.front().nodeId(),true));
    if(values.size()!=spatial_count)spatialReject();
    for(auto& x:values)x=input.resolve(x);
    const auto& first=values.front();
    for(const auto& x:values) {
      if(x.nodeId()!=first.nodeId() || x.resultIndex()!=first.resultIndex() ||
          input.type(x)!=input.type(first) || !input.type(x).isScalar() ||
          !x.indices().empty() || x.threadSelector() || x.pointer() ||
          bool(x.conversion())!=bool(first.conversion()) ||
          (x.conversion() && v::typeConversionLabel(x)!=v::typeConversionLabel(first)))spatialReject();
    }
    auto key=valueKey(first);
    if(values.size()>1) {
      const auto stride=static_cast<__int128>(values[1].instanceIndex())-first.instanceIndex();
      bool affine=stride>=INT64_MIN && stride<=INT64_MAX;
      for(std::size_t i=0;affine && i<values.size();++i)
        affine=static_cast<__int128>(first.instanceIndex())+stride*i==values[i].instanceIndex();
      if(affine)spatial_instance_strides.insert_or_assign(key,static_cast<std::int64_t>(stride));
      else spatial_instance_strides.erase(key);
    }
    const auto stride=spatial_instance_strides.find(key);
    if(!first.conversion() && stride!=spatial_instance_strides.end())
      registerSpatialFamily({&input,first,spatial_count,stride->second,{}});
    else {
      const auto base=first;
      registerSpatialFamily({&input,base,spatial_count,0,std::move(values)});
    }
  }
  const SpatialValues& spatialValues(v::Value value) {
    auto found=spatial_values.find(valueKey(value));
    if(found==spatial_values.end())spatialReject();
    return found->second;
  }
  Offset spatialOffsets(const std::vector<std::uint64_t>& offsets) {
    if(offsets.size()!=spatial_count || offsets.front()>std::uint64_t(INT64_MAX))spatialReject();
    const auto step=static_cast<__int128>(offsets[1])-offsets[0];
    if(step<INT64_MIN || step>INT64_MAX)spatialReject();
    for(std::size_t i=0;i<offsets.size();++i)
      if(static_cast<__int128>(offsets[0])+step*i!=offsets[i] || offsets[i]>std::uint64_t(INT64_MAX))spatialReject();
    Offset result(static_cast<std::int64_t>(offsets[0]));
    if(step)result.terms[0]=static_cast<std::int64_t>(step);
    return result;
  }
  void prepareSpatial(v::Value value) {
    const auto& members=spatialValues(value);
    if(value.conversion()) {
      std::vector<v::Value> sources;
      SENTRYWARD_AUDIT(if(members.count>1)audit->event("INSTANCE_ENUMERATION",
        "spatial_conversion_sources",members.count,1,value.nodeId(),true));
      for(std::size_t i=0;i<members.count;++i)sources.push_back(members.at(i).conversion()->source);
      registerSpatial(std::move(sources));return;
    }
    const auto& node=input.definingNode(value);
    const auto& op=node.operation();
    if(scalarOperation(op)) {
      if(members.affine() && node.hasUniformThreadTemplate()) {
        std::vector<SpatialValues> children;
        bool supported=true;
        for(const auto& operand:node.operands()) {
          const auto resolved=input.resolve(operand);
          if(resolved!=operand || !operand.indices().empty() || operand.threadSelector() ||
             operand.pointer() || operand.conversion() || !input.type(operand).isScalar()) {supported=false;break;}
          const auto& child=input.definingNode(operand);
          if(child.storedInstanceCount()!=1){supported=false;break;}
          auto base=operand;std::int64_t stride=0;
          const auto& threads=child.instanceThreads();
          if(child.origin().cta==node.origin().cta && !threads.empty() &&
             threads.at(operand.instanceIndex())==node.origin().thread) {
            if(threads.storageIdentity()!=node.instanceThreads().storageIdentity()) {supported=false;break;}
            base=child.result(operand.resultIndex(),value.instanceIndex());stride=members.stride;
          }
          children.push_back({&input,base,spatial_count,stride,{}});
        }
        if(supported) {
          for(auto& child:children)registerSpatialFamily(std::move(child));
          ++result.spatial_template_operations;
          return;
        }
      }
      const auto first=input.instance(value);
      std::vector<std::vector<v::Value>> operands(first.operands.size());
      SENTRYWARD_AUDIT(if(members.count>1)audit->event("INSTANCE_ENUMERATION",
        "spatial_operand_fallback",members.count,1,value.nodeId(),true));
      for(std::size_t member=0;member<members.count;++member) {
        const auto item=input.instance(members.at(member));
        if(item.operands.size()!=operands.size())spatialReject();
        for(std::size_t i=0;i<operands.size();++i)operands[i].push_back(item.operands[i]);
      }
      for(auto& items:operands)registerSpatial(std::move(items));
    } else if(!dynamic_cast<const v::LoadOp*>(&op) && !dynamic_cast<const v::ConstantOp*>(&op) &&
              !dynamic_cast<const v::InputParameterOp*>(&op) && !dynamic_cast<const v::CoordinateOp*>(&op))spatialReject();
  }
  static v::ElementType bit(unsigned n){return v::ElementType::bit(n);}
  void limit(){
    SENTRYWARD_AUDIT(audit->maximum(AuditCounter::max_graph_nodes,result.graph.size()));
    result.peak_construction_nodes=std::max(result.peak_construction_nodes,result.graph.size());
    if(!collecting && result.graph.size()>options.max_nodes) {
      // Raw Ref locals must unwind before a moving collection. The explicit
      // worklist and reduction cache own every value needed to resume.
      if(projecting && options.construction_collection_interval)
        throw CollectProjectionFrontier{};
      fail("expression_node_limit_exceeded","scalar graph node limit exceeded");
    }
    if(domains.size()>options.max_recursion)
      fail("expression_nesting_limit_exceeded","scalar/Fold binder nesting budget exceeded");
  }
  std::pair<std::int64_t,std::int64_t> bounds(const Offset& x,std::size_t depth=0)const{
    if(depth>options.max_recursion)fail("layout_nesting_limit_exceeded","layout expression nesting budget exceeded");
    if(x.expression) {
      const auto& e=*x.expression;auto [lo,hi]=bounds(e.left,depth+1);
      if(e.kind==OffsetExpression::Add) {
        auto [rl,rh]=bounds(e.right,depth+1);return {add(lo,rl),add(hi,rh)};
      }
      if(e.kind==OffsetExpression::Scale) {
        auto l=mul(lo,e.factor),h=mul(hi,e.factor);return {std::min(l,h),std::max(l,h)};
      }
      if(lo<0 || e.factor<=0)fail("scalar_fold_invalid_index","invalid mixed-radix layout coordinate");
      if(e.kind==OffsetExpression::Quotient)return {lo/e.factor,hi/e.factor};
      if(lo/e.factor==hi/e.factor)return {lo%e.factor,hi%e.factor};
      const auto step=std::gcd(alignment(e.left),static_cast<std::uint64_t>(e.factor));
      return {0,e.factor-static_cast<std::int64_t>(step)};
    }
    auto lo=x.base,hi=x.base;
    for(auto [level,c]:x.terms){
      if(level>=domains.size())fail("scalar_fold_internal_binding","index is outside its Fold");
      auto amount=mul(c,integer(domains[level]?domains[level]-1:0));
      if(amount<0)lo=add(lo,amount);else hi=add(hi,amount);
    }
    return {lo,hi};
  }
  // Split an affine mixed-radix coordinate only when its residual is proved
  // to stay within one digit. E.g. (128*r+c)/128=r for 0<=c<128.
  // Visits the stored index expression, never the members of its domains.
  // Compound indices may require subtree bounds; the nesting budget bounds
  // this work independently of the number of tensor elements.
  std::optional<std::pair<Offset,Offset>> digit(const Offset& x,std::int64_t d,std::size_t depth=0)const {
    if(depth>options.max_recursion)fail("layout_nesting_limit_exceeded","layout digit nesting budget exceeded");
    const auto normalize=[&](Offset q,Offset r)->std::optional<std::pair<Offset,Offset>> {
      const auto [lo,hi]=bounds(r);
      const auto floor=[&](std::int64_t v){return v/d-(v%d<0);};
      const auto carry=floor(lo);
      if(carry!=floor(hi))return {};
      const auto shift=-static_cast<__int128>(carry)*d;
      if(shift<INT64_MIN || shift>INT64_MAX)return {};
      return std::pair{plus(q,Offset(carry)),plus(r,Offset(static_cast<std::int64_t>(shift)))};
    };
    if(x.expression) {
      const auto& e=*x.expression;
      if(e.kind==OffsetExpression::Scale && e.factor>0) {
        if(e.factor%d==0)return std::pair{times(e.left,e.factor/d),Offset{}};
        if(d%e.factor==0)if(auto inner=digit(e.left,d/e.factor,depth+1))
          return std::pair{inner->first,times(inner->second,e.factor)};
      }
      if(e.kind==OffsetExpression::Add) {
        auto a=digit(e.left,d,depth+1),b=digit(e.right,d,depth+1);
        if(a && b) {
          auto r=plus(a->second,b->second);
          return normalize(plus(a->first,b->first),std::move(r));
        }
      }
      const auto [lo,hi]=bounds(x);
      if(lo>=0 && hi<d)return std::pair{Offset{},x};
      return {};
    }
    Offset q(x.base/d),r(x.base%d);
    for(auto [level,c]:x.terms) {
      if(c/d)q.terms[level]=c/d;
      if(c%d)r.terms[level]=c%d;
    }
    return normalize(std::move(q),std::move(r));
  }
  Offset quotient(const Offset& x,std::uint64_t width)const{
    if(!width)fail("scalar_fold_invalid_index","zero index divisor");
    if(width==1)return x;
    const auto d=integer(width);
    auto [lo,hi]=bounds(x);
    if(lo<0)fail("scalar_fold_invalid_index","negative view coordinate");
    if(auto split=digit(x,d))return std::move(split->first);
    if(lo/d==hi/d)return Offset(lo/d);
    if(!x.expression && x.base>=d) {
      auto residual=x;residual.base%=d;
      if(bounds(residual).first>=0)
        return plus(Offset(x.base/d),quotient(residual,width));
    }
    if(x.expression && x.expression->kind==OffsetExpression::Add) {
      const auto& e=*x.expression;
      if(!e.left.expression && e.left.terms.empty() && e.left.base>=d) {
        auto residual=plus(Offset(e.left.base%d),e.right);
        if(bounds(residual).first>=0)
          return plus(Offset(e.left.base/d),quotient(residual,width));
      }
    }
    return compound(OffsetExpression::Quotient,x,{},d);
  }
  Offset remainder(const Offset& x,std::uint64_t width)const{
    if(!width)fail("scalar_fold_invalid_index","zero index divisor");
    const auto [lo,hi]=bounds(x);
    if(lo<0)fail("scalar_fold_invalid_index","negative view coordinate");
    if(std::uint64_t(hi)<width)return x;
    if(x.expression && x.expression->kind==OffsetExpression::Remainder &&
       std::uint64_t(x.expression->factor)%width==0)
      return remainder(x.expression->left,width);
    if(auto split=digit(x,integer(width)))return std::move(split->second);
    if(!x.expression && x.base>=integer(width)) {
      auto residual=x;residual.base%=integer(width);
      if(bounds(residual).first>=0)return remainder(residual,width);
    }
    if(x.expression && x.expression->kind==OffsetExpression::Add) {
      const auto& e=*x.expression;
      // Removing an exact nonnegative multiple preserves the residue. Bounds
      // reject signed/wrapping cases; no replica equality is assumed.
      if(!e.left.expression && e.left.terms.empty() && e.left.base>=0) {
        auto reduced=plus(Offset(std::uint64_t(e.left.base)%width),e.right);
        const auto [rl,rh]=bounds(reduced);
        if(rl>=0)return std::uint64_t(rh)<width?reduced:
            compound(OffsetExpression::Remainder,std::move(reduced),{},integer(width));
      }
    }
    auto q=quotient(x,width);
    if(alignment(x)%width==0)return Offset{};
    if(!x.expression && !q.expression)return plus(x,times(q,-integer(width)));
    return compound(OffsetExpression::Remainder,x,{},integer(width));
  }
  // One exact construction path for both PTX arithmetic and operations added
  // by projection (notably ExtractBits). Inspect only direct literal operands
  // and their conversion edges; never search operand ancestors. The shared
  // constant evaluator preserves bit width, signedness and admitted modifiers.
  s::Ref emitOperation(std::string name,std::string type,std::vector<s::Ref> operands,
      std::string attr={}) {
    const auto decoded=[&](const s::Ref& value)->std::optional<equivalence::ConstantValue> {
      const auto& expression=result.graph.node(value);
      if(expression.kind!=s::Kind::Constant || !value.pointer_provenance.empty())return {};
      // Predicate literals are not integer bit encodings in constantValue.
      // Only accept the unconverted canonical Boolean literals here.
      if(expression.type=="pred" && value.conversion.empty() &&
          (value.converted_type.empty() || value.converted_type=="pred") &&
          (expression.name=="0" || expression.name=="1"))
        return equivalence::ConstantValue{"pred",expression.name};
      return equivalence::constantValue(expression.type,expression.name,value.conversion,value.converted_type);
    };
    if(name=="Select" && attr.empty() && operands.size()==3) {
      if(auto predicate=decoded(operands[0]);predicate && predicate->type=="pred" &&
          (predicate->literal=="0" || predicate->literal=="1")) {
        const auto& selected=operands[predicate->literal=="1"?1:2];
        if(result.graph.type(selected)==type)return selected;
      }
    }
    // The usual dynamic expression must not allocate a temporary literal
    // array. This pass reads only the fixed-arity operand headers twice.
    for(const auto& operand:operands)
      if(result.graph.node(operand).kind!=s::Kind::Constant || !operand.pointer_provenance.empty())
        return result.graph.operation(std::move(name),std::move(type),std::move(operands),std::move(attr));
    std::vector<equivalence::ConstantValue> constants;
    constants.reserve(operands.size());
    for(const auto& operand:operands) {
      auto value=decoded(operand);if(!value)break;
      constants.push_back(std::move(*value));
    }
    if(constants.size()==operands.size())
      if(auto value=equivalence::constantOperation(name,type,attr,constants))
        return result.graph.constant(std::move(value->type),std::move(value->literal));
    return result.graph.operation(std::move(name),std::move(type),std::move(operands),std::move(attr));
  }
  s::Ref offsetValue(const Offset& x){
    if(x.expression) {
      const auto& e=*x.expression;
      auto left=offsetValue(e.left);
      if(e.kind==OffsetExpression::Add)return emitOperation("Add","u64",{left,offsetValue(e.right)});
      const auto name=e.kind==OffsetExpression::Scale?"Multiply":e.kind==OffsetExpression::Quotient?"Divide":"Remainder";
      return emitOperation(name,"u64",{left,result.graph.constant("u64",std::to_string(e.factor))});
    }
    auto r=result.graph.constant("u64",std::to_string(x.base));
    for(auto [level,c]:x.terms){
      // Exact local specialization; do not recursively recompute bounds here.
      if(level>=domains.size())fail("scalar_fold_internal_binding","offset binder is absent");
      if(domains[level]==1)continue;
      auto t=result.graph.index(level);
      if(c!=1)t=emitOperation("Multiply","u64",{t,result.graph.constant("u64",std::to_string(c))});
      r=emitOperation("Add","u64",{r,t});
    }
    return r;
  }
  std::optional<std::int64_t> known(v::Value x,unsigned depth=0)const{
    return knownAt(x,depth,indexed_context?indexed_context->use.get():nullptr);
  }
  std::optional<std::int64_t> knownAt(v::Value x,unsigned depth,const IndexedSourceUse* use)const{
    if(depth>32)return {};
    x=input.resolve(x);
    if(use && use->prototype!=input.canonicalInterpretation(x))return {};
    if(use && (use->kind==IndexedSourceUse::Kind::Exact || use->kind==IndexedSourceUse::Kind::OneHotLiteral))use=nullptr;
    if(use && (use->kind==IndexedSourceUse::Kind::Forward || use->kind==IndexedSourceUse::Kind::Selected))
      return knownAt(use->children.at(0)->prototype,depth+1,use->children.at(0).get());
    if(use && (use->kind==IndexedSourceUse::Kind::BitTestSelect ||
               use->kind==IndexedSourceUse::Kind::MemberTemplate ||
               use->kind==IndexedSourceUse::Kind::Component ||
               use->kind==IndexedSourceUse::Kind::CoordinateMap))return {};
    if(use && use->binding) {
        const auto& binding=*use->binding;
        if(binding.fixed)return {};
        if(binding.initial || binding.producer)return {};
        const auto decode=[&](std::optional<std::int64_t> coordinate)->std::optional<std::int64_t> {
          if(!coordinate || !binding.power_of_two)return coordinate;
          const auto width=input.type(x).element().width();
          if(*coordinate<0 || *coordinate>=63 || *coordinate>=width)return {};
          return knownInteger(std::int64_t{1}<<*coordinate,input.type(x).element());
        };
        if(!binding.stride && !binding.outer_stride && binding.quotient_terms.empty())return decode(binding.base);
        const auto [lo,hi]=bounds(indexed_context->index);
        if(lo==hi && lo>=0) {
          return decode(indexedBindingValue(binding,static_cast<std::uint64_t>(lo)));
        }
        return {};
      }
    if(!x.indices().empty() || x.threadSelector() || !input.type(x).isScalar())return {};
    if(x.conversion()){
      const auto& conversion=*x.conversion();
      const auto& source_type=input.type(conversion.source);
      const auto& target_type=input.type(x);
      if(!source_type.isScalar() || !source_type.element().isBitLike() ||
         !target_type.element().isBitLike())return {};
      if(conversion.spec->kind==v::TypeConversionKind::Convert &&
         conversion.spec->attributes!=v::ConversionAttributes{})return {};
      const auto* child=use?use->children.at(0).get():nullptr;
      auto value=knownAt(child?child->prototype:conversion.source,depth+1,child);
      return value?knownInteger(*value,target_type.element()):std::nullopt;
    }
    const auto& n=input.definingNode(x);
    if(const auto* c=dynamic_cast<const v::ConstantOp*>(&n.operation())){
      auto value=literal(c->literal());
      return value?knownInteger(*value,input.type(x).element()):std::nullopt;
    }
    if(dynamic_cast<const v::CoordinateOp*>(&n.operation())){
      if(auto val=input.instance(x).origin.coordinate_value)
        if(*val<=std::uint64_t(INT64_MAX))return knownInteger(std::int64_t(*val),input.type(x).element());
    }
    if(dynamic_cast<const v::BinaryOp*>(&n.operation()) && input.type(x).element().isBitLike()) {
      std::vector<equivalence::ConstantValue> operands;
      std::size_t operand_index=0;
      for(auto operand:input.instance(x).operands) {
        auto value=knownAt(operand,depth+1,use?use->children.at(operand_index).get():nullptr);
        ++operand_index;
        if(!value)return {};
        operands.push_back({input.type(operand).element().toString(),std::to_string(*value)});
      }
      if(auto value=equivalence::constantOperation(std::string(n.operation().name()),
          input.type(x).element().toString(),attributes(n.operation()),operands)) {
        auto parsed=literal(value->literal);
        return parsed?knownInteger(*parsed,input.type(x).element()):std::nullopt;
      }
    }
    return {};
  }
  s::Ref cast(s::Ref x,const v::ElementType& type,std::string step="Bitcast"){
    auto target=type.toString();if(result.graph.type(x)==target && step=="Bitcast")return x;
    return result.graph.convert(std::move(x),target,std::move(step));
  }
  v::Value selected(v::Value value){
    if(!value.threadSelector())return value;
    const auto thread=known(selected(*value.threadSelector()));
    if(!thread || *thread<0) {
      const auto selector=*value.threadSelector();
      const auto fixed=specialization.known(selector);
      fail("scalar_fold_thread_selector","thread selector is not constant; selector_node="+
          std::to_string(selector.nodeId())+", selector_instance="+std::to_string(selector.instanceIndex())+
          ", operation="+std::string(input.definingNode(selector).operation().name())+
          ", use_kind="+(indexed_context && indexed_context->use?
              std::to_string(static_cast<int>(indexed_context->use->kind)):"none")+
          ", fixed_instance_value="+(fixed?fixed->literal:"unknown"));
    }
    if(value.conversion()) {
      const auto& conversion=*value.conversion();
      auto source=selected(input.fromThread(conversion.source,*value.threadSelector()));
      auto result=conversion.spec->kind==v::TypeConversionKind::Bitcast
          ?input.bitcast(source,conversion.spec->target)
          :input.convert(source,conversion.spec->target,conversion.spec->attributes);
      if(!value.indices().empty())result=input.component(result,{value.indices().begin(),value.indices().end()});
      if(value.pointer())result=input.asPointer(result,*value.pointer());
      return result;
    }
    const auto& n=input.definingNode(value);
    const auto cta=input.instance(value).origin.cta;
    if(n.origin().cta==cta) {
      auto j=n.instanceThreads().indexOf(static_cast<std::uint64_t>(*thread));
      if(n.instanceThreads().empty() && n.origin().thread==std::optional<std::uint64_t>(*thread))j=0;
      if(j) {
        auto result=n.result(value.resultIndex(),*j);
        if(!value.indices().empty())result=input.component(result,{value.indices().begin(),value.indices().end()});
        if(value.pointer())result=input.asPointer(result,*value.pointer());
        return result;
      }
    }
    fail("scalar_fold_thread_selector","selected thread "+std::to_string(*thread)+
        " has no matching value instance; source thread="+
        (n.origin().thread?std::to_string(*n.origin().thread):"none")+
        ", members="+std::to_string(n.instances().size()));
  }
  s::Ref scalar(v::Value x){
    const auto& type=input.type(x);
    if(!type.isScalar())fail("scalar_fold_matrix_operand","scalar operator received an unprojected matrix");
    return project(x,Offset{},type.element());
  }
  s::Ref initial(const InitialMemorySymbol& x,const Offset& offset,const v::ElementType& type){
    auto [lo,hi]=bounds(offset);
    const auto bytes=type.width()/8;
    if(lo<0 || hi<0 || std::uint64_t(hi)>x.region.bytes || bytes>x.region.bytes-std::uint64_t(hi))
      fail("scalar_fold_invalid_index","initial-memory slice exceeds its recorded region");
    return result.graph.initialCell(x.region.allocation,offsetValue(plus(offset,Offset(integer(x.region.offset)))),type.toString());
  }
  s::Ref source(const ViewSource& x,const Offset& offset,const v::ElementType& type){
    if(const auto* i=std::get_if<InitialMemorySymbol>(&x))return initial(*i,offset,type);
    return project(std::get<v::Value>(x),offset,type);
  }
  const std::vector<ViewInterval>& intervalsFor(const MemoryView& view) {
    auto [entry,inserted]=view_intervals.try_emplace(&view);
    if(inserted) {
      for(std::size_t i=0;i<view.blocks.size();++i) {
        const auto& block=view.blocks[i];
        unsigned __int128 end=static_cast<unsigned __int128>(block.result_offset)+block.bytes;
        for(const auto& axis:block.axes)if(axis.count) {
          end+=static_cast<unsigned __int128>(axis.count-1)*axis.result_stride;
          end=std::min(end,static_cast<unsigned __int128>(UINT64_MAX));
        }
        entry->second.push_back({block.result_offset,end>UINT64_MAX?UINT64_MAX:static_cast<std::uint64_t>(end),0,i});
      }
      auto& intervals=entry->second;
      std::sort(intervals.begin(),intervals.end(),[](const auto& a,const auto& b){return a.begin<b.begin;});
      std::uint64_t end=0;
      for(auto& interval:intervals)interval.prefix_end=end=std::max(end,interval.end);
    }
    return entry->second;
  }
  std::vector<std::size_t> coveringBlocks(const MemoryView& view,std::uint64_t begin,std::uint64_t end) {
    const auto& intervals=intervalsFor(view);
    std::vector<std::size_t> candidates;
    auto it=std::upper_bound(intervals.begin(),intervals.end(),begin,
        [](auto value,const auto& interval){return value<interval.begin;});
    while(it!=intervals.begin()) {
      --it;
      if(it->prefix_end<end)break;
      if(it->end>=end)candidates.push_back(it->block);
    }
    std::sort(candidates.begin(),candidates.end());
    return candidates;
  }
  // Plan an affine read interval without constructing expression nodes.
  std::uint64_t readSpan(v::Value value,const Offset& offset,std::uint64_t stride,
                         const v::ElementType& type,std::uint64_t count,unsigned depth=0) {
    SENTRYWARD_LEDGER(ledger->add("read_span_calls"));
    SENTRYWARD_AUDIT(if(count>1 && (depth>=32 || spatial_count))
      audit->event("SPAN_DEGRADED",spatial_count?"spatial_projection_disables_span":"read_span_depth_limit",
          count,1,value.nodeId()));
    if(count<=1 || depth>=32 || spatial_count)return 1;
    value=input.canonicalInterpretation(value);
    const auto single=[&]([[maybe_unused]] const char* reason) {
      SENTRYWARD_AUDIT(audit->event("SPAN_DEGRADED",reason,count,1,value.nodeId()));
      SENTRYWARD_AUDIT({
        (void)audit;
        static thread_local std::set<std::string> reported;
        const auto& op=input.definingNode(value).operation();
        const auto label=std::string(reason)+":"+std::string(op.name());
        if(reported.size()<16 && reported.insert(label).second)
          SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_0);
      });
      return std::uint64_t{1};
    };
    if(value.threadSelector() || !value.indices().empty())return single("read_span_value_selection");
    const auto [lo,hi]=bounds(offset);
    if(lo<0 || hi<lo)return 1;
    const auto width=type.width()/8;
    const auto fit=[&](std::uint64_t bytes,std::uint64_t start) {
      if(start>bytes || width>bytes-start)return std::uint64_t{1};
      return stride?std::min(count,1+(bytes-start-width)/stride):count;
    };
    const auto& value_type=input.type(value);
    if(!value_type.hasStaticShape() || value_type.element().isParameterized() ||
       value_type.bitSize()%8)return single("read_span_unresolved_type");
    // A computed matrix has a known numerical element domain. A packed scalar
    // only has a byte domain: its independent numeric fields need their own
    // source mapping before treating them as one parametric numerical domain.
    const auto& operation=input.definingNode(value).operation();
    // Use the same identity-transfer rule as projectImpl. A layout-preserving
    // native copy must not turn a known matrix domain into K scalar queries.
    if(const auto* copy=dynamic_cast<const v::NativeMatrixOp*>(&operation);
       copy && copy->opcode()==v::NativeMatrixOpcode::TcgenCopyValue) {
      const auto source=input.instance(value).operands.at(0);
      if(input.type(source)==value_type)
        return readSpan(source,offset,stride,type,count,depth+1);
    }
    if(value_type.isMatrix() && (value.conversion() || scalarOperation(operation) ||
        dynamic_cast<const v::MatMulOp*>(&operation) ||
        dynamic_cast<const v::TcgenMmaOp*>(&operation)))
      return fit(value_type.bitSize()/8,std::uint64_t(hi));
    if(value.conversion() || !dynamic_cast<const v::LoadOp*>(&operation))return single("read_span_not_matrix_or_load");
    if(auto initial=memory.initialRead({value.nodeId(),value.instanceIndex()}))
      return fit(initial->region.bytes,std::uint64_t(hi));
    const auto found=memory.normalizedRead({value.nodeId(),value.instanceIndex()});
    if(!found)return single("read_span_missing_substitution");
    if(const auto* ref=std::get_if<const v::Value*>(&*found))
      return readSpan(**ref,offset,stride,type,count,depth+1);
    if(const auto* initial=std::get_if<const InitialMemorySymbol*>(&*found))
      return fit((*initial)->region.bytes,std::uint64_t(hi));
    const MemoryView* selected_view=nullptr;
    if(const auto* direct=std::get_if<const MemoryView*>(&*found))selected_view=*direct;
    else if(const auto* split=std::get_if<const PartitionedRead*>(&*found)) {
      for(const auto& region:(*split)->regions)
        if(std::uint64_t(lo)>=region.begin && std::uint64_t(hi)+width<=region.begin+region.bytes) {
          selected_view=&region.view;break;
        }
    }
    if(!selected_view) {
      if(const auto* split=std::get_if<const PartitionedRead*>(&*found))
        if(const auto* grid=partitionGrid(**split,value,type))
          return fit(grid->blocks.front().bytes*grid->indexed_source->count,std::uint64_t(hi));
      // A residue map certifies this column, not adjacent reduction columns.
      // Reuse/precompute its proof but do not manufacture a wider read span.
      if(const auto* split=std::get_if<const PartitionedRead*>(&*found))
        if(!(*split)->template_domains)(void)partitionLattice(**split,value,offset,type);
      return single("read_span_crosses_source_domains");
    }
    const auto& view=*selected_view;
    if(view.indexed_source) {
      const auto& block=view.blocks.at(0);
      if(block.axes.size()>1) {
        auto range=offset;Offset iteration;iteration.terms[domains.size()]=integer(stride);
        range=plus(range,iteration);domains.push_back(count);
        const auto mapped=viewOffset(block,range,type);domains.pop_back();
        return mapped?count:single("read_span_crosses_source_domains");
      }
      const auto total=block.bytes*view.indexed_source->count;
      const auto unit=std::min<std::uint64_t>(width,block.bytes);
      if(!unit || std::max<std::uint64_t>(width,block.bytes)%unit ||
         alignment(plus(offset,Offset(-integer(block.result_offset))))%unit || stride%unit)return 1;
      return fit(total,std::uint64_t(hi)-block.result_offset);
    }
    const auto candidates=coveringBlocks(view,std::uint64_t(lo),std::uint64_t(hi)+width);
    if(candidates.empty())return single("read_span_no_covering_block");
    const auto& block=view.blocks[candidates.front()];
    if(!block.axes.empty()) {
      if(view.blocks.size()!=1)return 1;
      const auto& source=view.sources.at(block.source);
      const auto* initial=std::get_if<InitialMemorySymbol>(&source);
      const auto extent=initial?initial->region.bytes:input.type(std::get<v::Value>(source)).bitSize()/8;
      auto range=offset;
      Offset iteration;iteration.terms[domains.size()]=integer(stride);
      range=plus(range,iteration);
      domains.push_back(count);
      const auto mapped=viewOffset(block,range,type);
      bool valid=false;
      if(mapped) {
        const auto [sl,sh]=bounds(*mapped);
        valid=sl>=0 && sh>=sl && std::uint64_t(sh)<=extent &&
            width<=extent-std::uint64_t(sh);
      }
      domains.pop_back();
      return valid?count:single("read_span_axis_mapping");
    }
    auto span=fit(block.bytes,std::uint64_t(hi)-block.result_offset);
    // A new overlapping block may change source precedence.
    const auto& intervals=intervalsFor(view);
    auto next=std::upper_bound(intervals.begin(),intervals.end(),std::uint64_t(lo),
        [](auto x,const auto& interval){return x<interval.begin;});
    if(stride && next!=intervals.end()) {
      if(next->begin<std::uint64_t(hi)+width)return 1;
      span=std::min(span,1+(next->begin-std::uint64_t(hi)-width)/stride);
    }
    auto source_offset=plus(offset,Offset(add(integer(block.source_offset),-integer(block.result_offset))));
    if(const auto* initial=std::get_if<InitialMemorySymbol>(&view.sources.at(block.source))) {
      const auto [sl,sh]=bounds(source_offset);
      if(sl<0 || sh<sl)return 1;
      return std::min(span,fit(initial->region.bytes,std::uint64_t(sh)));
    }
    return std::min(span,readSpan(std::get<v::Value>(view.sources.at(block.source)),
        source_offset,stride,type,span,depth+1));
  }
  std::optional<Offset> viewOffset(const ViewBlock& block,const Offset& offset,const v::ElementType& type){
    auto pos=plus(offset,Offset(-integer(block.result_offset)));
    auto [lo,hi]=bounds(pos);if(lo<0 || hi<0)return {};
    auto src=Offset(integer(block.source_offset));
    std::vector<ViewAxis> axes=block.axes;
    std::sort(axes.begin(),axes.end(),[](const auto& a,const auto& b){return a.result_stride>b.result_stride;});
    try{
      for(const auto& axis:axes){
        auto coord=quotient(pos,axis.result_stride);
        auto [cl,ch]=bounds(coord);
        if(cl<0 || std::uint64_t(ch)>=axis.count)return {};
        pos=remainder(pos,axis.result_stride);
        src=plus(src,times(coord,integer(axis.source_stride)));
      }
    }catch(const Failure& e){
      if(e.code=="scalar_fold_non_affine_projection")return {};
      throw;
    }
    auto [pl,ph]=bounds(pos);auto width=type.width()/8;
    if(pl<0 || ph<0 || std::uint64_t(ph)>block.bytes || width>block.bytes-std::uint64_t(ph))return {};
    return plus(src,pos);
  }
  std::optional<s::Ref> viewBlock(const MemoryView& view,const ViewBlock& block,
                                 const Offset& offset,const v::ElementType& type){
    auto mapped=viewOffset(block,offset,type);
    if(!mapped)return {};
    return source(view.sources.at(block.source),*mapped,type);
  }
  s::Ref view(const MemoryView& view,const Offset& offset,const v::ElementType& type){
    if(view.sources.size()!=1 || view.blocks.size()!=1 || view.blocks[0].source!=0)
      fail("compact_view_unsupported","a view must contain one source rule; use disjoint result domains");
    if(view.indexed_source)return indexedView(view,offset,type);
    if(view.blocks.size()<=4) {
      for(const auto& b:view.blocks)if(auto r=viewBlock(view,b,offset,type))return *r;
    } else {
      const auto [lo,hi]=bounds(offset);
      std::vector<std::size_t> candidates;
      if(lo>=0 && hi>=lo)candidates=coveringBlocks(view,std::uint64_t(lo),std::uint64_t(hi)+type.width()/8);
      for(auto index:candidates)if(auto r=viewBlock(view,view.blocks[index],offset,type))return *r;
    }
    fail("compact_view_unsupported","read spans independent source domains or has unsupported alignment");
  }
  Offset mappedCoordinate(const IndexedValueBinding& binding,const Offset& index) {
    auto position=Offset(binding.base);
    if(binding.period) {
      position=plus(position,times(remainder(index,binding.period),binding.stride));
      position=plus(position,times(quotient(index,binding.period),binding.outer_stride));
    } else position=plus(position,times(index,binding.stride));
    for(const auto& term:binding.quotient_terms)
      position=plus(position,times(quotient(index,term.divisor),term.coefficient));
    if(binding.quotient_terms.empty())return position;
    // Use independent radix digits to keep bounds precise. Work is over the
    // stored coordinate coefficients, never over represented source members.
    auto base=binding.base;
    std::vector<std::pair<Offset,std::int64_t>> digits;
    const auto append=[&](Offset digit,std::int64_t coefficient,std::int64_t upper) {
      if(!coefficient)return;
      if(coefficient<0) {
        base=add(base,mul(coefficient,upper));
        digit=plus(Offset(upper),times(std::move(digit),-1));coefficient=mul(coefficient,-1);
      }
      digits.emplace_back(std::move(digit),coefficient);
    };
    auto divisor=binding.period?binding.period:1;
    auto coefficient=binding.period?binding.outer_stride:binding.stride;
    if(binding.period)append(remainder(index,binding.period),binding.stride,integer(binding.period-1));
    for(const auto& term:binding.quotient_terms) {
      if(!term.divisor || term.divisor%divisor)return position;
      const auto ratio=term.divisor/divisor;
      append(remainder(quotient(index,divisor),ratio),coefficient,integer(ratio-1));
      coefficient=add(mul(coefficient,integer(ratio)),term.coefficient);divisor=term.divisor;
    }
    const auto [lo,hi]=bounds(index);
    if(lo<0)return position;
    append(quotient(index,divisor),coefficient,hi/integer(divisor));
    if(base<0)return position;
    position=Offset(base);
    for(const auto& [digit,scale]:digits)position=plus(position,times(digit,scale));
    return position;
  }
  s::Ref indexedView(const MemoryView& view,const Offset& offset,const v::ElementType& type) {
    if(spatial_count)fail("compact_view_unsupported","simultaneous spatial and indexed-source binding is not supported");
    const auto& pattern=*view.indexed_source;
    const auto& block=view.blocks.at(0);
    const auto result_step=block.axes.empty()?block.bytes:block.axes.front().result_stride;
    const auto width=type.width()/8;
    const auto unit=std::min<std::uint64_t>(width,block.bytes);
    auto relative=plus(offset,Offset(-integer(block.result_offset)));
    const auto [lo,hi]=bounds(relative);
    const bool grid=block.axes.size()>1;
    std::uint64_t extent=block.bytes;
    for(const auto& axis:block.axes)extent+=(axis.count-1)*axis.result_stride;
    if(!unit || std::max<std::uint64_t>(width,block.bytes)%unit || alignment(relative)%unit ||
       lo<0 || hi<lo || !result_step ||
       std::uint64_t(hi)>extent || width>extent-std::uint64_t(hi))
      fail("compact_view_unsupported","indexed read requires aligned, integer-multiple access widths within its domain");
    if(!grid && result_step!=block.bytes) {
      const auto [pl,ph]=bounds(remainder(relative,result_step));
      if(pl<0 || ph<pl || std::uint64_t(ph)>block.bytes || width>block.bytes-std::uint64_t(ph))
        fail("compact_view_unsupported","indexed read crosses disjoint template residues");
    }
    const auto part=[&](const Offset& at,const v::ElementType& desired) {
      auto index=quotient(at,result_step);
      auto source_offset=plus(Offset(integer(block.source_offset)),remainder(at,result_step));
      if(grid) {
        auto pos=at;index=Offset(0);
        source_offset=Offset(integer(block.source_offset));
        for(auto it=block.axes.rbegin();it!=block.axes.rend();++it) {
          const auto coordinate=quotient(pos,it->result_stride);
          const auto [low,high]=bounds(coordinate);
          if(low<0 || high<low || std::uint64_t(high)>=it->count)
            fail("compact_view_unsupported","indexed row coordinate exceeds its domain");
          index=plus(times(index,integer(it->count)),coordinate);
          source_offset=plus(source_offset,times(coordinate,integer(it->source_stride)));
          pos=remainder(pos,it->result_stride);
        }
        const auto [low,high]=bounds(pos);
        if(low<0 || high<low || std::uint64_t(high)>block.bytes ||
            desired.width()/8>block.bytes-std::uint64_t(high))
          fail("compact_view_unsupported","indexed read crosses row residues");
        source_offset=plus(source_offset,pos);
      }
      auto r=inContext(std::make_shared<const IndexedContext>(IndexedContext{index,pattern.root}),
          [&]{return project(pattern.prototype,source_offset,desired);});
        if(pattern.zero_mask) {
          const auto& mask=*pattern.zero_mask;
          const auto row=quotient(index,mask.columns),col=remainder(index,mask.columns);
          // Nonnegative index arithmetic avoids unsigned subtraction below
          // the diagonal. Projection still checks every source's full bounds;
          // unsupported extrapolations are not silently hidden by the mask.
          auto lhs=row,rhs=col;
          if(mask.diagonal>=0)rhs=plus(rhs,Offset(mask.diagonal));
          else lhs=plus(lhs,Offset(-mask.diagonal));
          const auto pred=emitOperation("Compare","pred",{offsetValue(lhs),offsetValue(rhs)},
              mask.lower?"predicate=ge":"predicate=lt");
          r=emitOperation("Select",desired.toString(),
              {pred,r,result.graph.constant(desired.toString(),"0")});
        }
      return r;
    };
    if(width<=block.bytes)return part(relative,type);
    if(width>8)fail("compact_view_unsupported","indexed scalar packing exceeds 64 bits");
    const auto target="b"+std::to_string(type.width());
    auto packed=result.graph.constant(target,"0");
    for(std::uint64_t j=0;j<width/block.bytes;++j) {
      auto value=part(plus(relative,Offset(integer(j*block.bytes))),bit(unsigned(8*block.bytes)));
      value=result.graph.convert(value,target,"ZeroExtend");
      if(j)value=emitOperation("ShiftLeft",target,
          {value,result.graph.constant("u32",std::to_string(8*j*block.bytes))});
      packed=emitOperation("BitOr",target,{packed,value});
    }
    return cast(packed,type);
  }
  struct PartitionLattice {const MemoryView* view;Offset offset;};
  const MemoryView* partitionGrid(const PartitionedRead& split,v::Value reader,
                                  const v::ElementType& type) {
    const auto& reader_type=input.type(reader);
    if(type.isParameterized() || !type.width() || type.width()%8 ||
       !reader_type.isMatrix() || !reader_type.hasStaticShape() ||
       reader_type.element().isParameterized() || reader_type.element().width()!=type.width() ||
       reader_type.shape().size()!=2)return nullptr;
    const auto width=std::uint64_t(type.width()/8);
    const auto columns=reader_type.shape().back();
    const PartitionGridKey key{&split,valueKey(reader),type.toString()};
    auto [entry,inserted]=partition_grids.try_emplace(key);
    if(inserted) {
      const auto* replacement=memory.readReplacement({reader.nodeId(),reader.instanceIndex()});
      const auto* raw=replacement?std::get_if<AssembledRead>(replacement):nullptr;
      if(!raw)return nullptr;
      SENTRYWARD_AUDIT({
        (void)audit;
        std::map<std::uint64_t,std::size_t> histogram;
        for(const auto& piece:raw->pieces)if(piece.type.hasStaticShape() && !piece.type.element().isParameterized())
          ++histogram[piece.type.bitSize()];
        SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_1);
        for(auto [bits,n]:histogram)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_2);
        SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_3);
      });
      std::uint64_t fields=0;
      for(const auto& piece:raw->pieces) {
        // piece.type describes the stored byte fragment, not the producer's
        // type: elimination publishes matrix<Nxb8> even for packed registers.
        // Bound the stored fragment to64bits; every resulting slice must still
        // resolve to an exact whole numeric field through the shared decoder.
        if(!piece.type.hasStaticShape() ||
           piece.type.element().isParameterized() || !piece.type.bitSize() ||
           piece.type.bitSize()>64 || piece.type.bitSize()%type.width())return nullptr;
        const auto count=piece.type.bitSize()/type.width();
        if(count>1000000-fields)return nullptr;
        fields+=count;
      }
      if(fields<3 || columns<2 || fields%columns || fields/columns<2 ||
         reader_type.bitSize()%8 || fields*width!=reader_type.bitSize()/8)return nullptr;
      std::vector<IndexedSourceSlice> slices;slices.reserve(fields);
      std::uint64_t offset=0;
      // At most8 byte-aligned fields per stored fragment, independent of source
      // matrix extent or thread count. Wider compact pieces were rejected above.
      for(const auto& piece:raw->pieces) {
        for(std::uint64_t field=0;field<piece.type.bitSize()/type.width();++field) {
          slices.push_back({reader,offset,width});offset+=width;
        }
      }
      if(auto pattern=output_source_matcher.matchGridSlices(slices,columns)) {
        if(pattern->count!=slices.size())return nullptr;
        auto source=std::make_shared<const IndexedSource>(std::move(*pattern));
        entry->second=MemoryView{reader_type,MemoryViewKind::Layout,{source->prototype},
            {{0,0,0,width,{{source->count,0,width}}}},{},std::move(source)};
      }
    }
    return entry->second?&*entry->second:nullptr;
  }
  std::optional<PartitionLattice> partitionLattice(
      const PartitionedRead& split,v::Value reader,const Offset& offset,const v::ElementType& type) {
    if(type.isParameterized() || !type.width() || type.width()%8)return {};
    const auto width=std::uint64_t(type.width()/8);
    const auto [lo,hi]=bounds(offset);
    if(lo<0 || hi<=lo)return {};
    // Exact congruence summary, not a fit to representative evaluations.
    // Quotient/remainder values are integral but otherwise unconstrained.
    // Outer scaling therefore retains e.g. 32*floor((512*i+j)/128).
    using Congruence=std::pair<std::int64_t,std::uint64_t>;
    std::unordered_map<const OffsetExpression*,Congruence> congruences;
    const auto congruence=[&](auto&& self,const Offset& x,unsigned depth)->Congruence {
      if(!x.expression) {
        std::uint64_t step=0;
        for(auto [axis,c]:x.terms) {
          (void)axis;
          const auto magnitude=c<0?std::uint64_t{0}-std::uint64_t(c):std::uint64_t(c);
          step=std::gcd(step,magnitude);
        }
        return {x.base,step};
      }
      if(depth>=options.max_recursion)return {0,1};
      if(auto found=congruences.find(x.expression.get());found!=congruences.end())return found->second;
      const auto& e=*x.expression;
      if(e.kind==OffsetExpression::Quotient || e.kind==OffsetExpression::Remainder)return {0,1};
      const auto a=self(self,e.left,depth+1);
      if(e.kind==OffsetExpression::Add) {
        const auto b=self(self,e.right,depth+1);
        const Congruence result{add(a.first,b.first),std::gcd(a.second,b.second)};
        congruences.emplace(x.expression.get(),result);return result;
      }
      const auto magnitude=e.factor<0?std::uint64_t{0}-std::uint64_t(e.factor):std::uint64_t(e.factor);
      if(magnitude && a.second>std::uint64_t(INT64_MAX)/magnitude)return {0,1};
      const Congruence result{mul(a.first,e.factor),a.second*magnitude};
      congruences.emplace(x.expression.get(),result);return result;
    };
    const auto [base,step]=congruence(congruence,offset,0);
    if(step<width || step>std::uint64_t(INT64_MAX))return {};
    const auto modulus=std::int64_t(step);
    const auto raw_residue=base%modulus;
    const auto residue=raw_residue<0?raw_residue+modulus:raw_residue;
    const auto first=std::uint64_t(lo)+
        (std::uint64_t(residue)+step-std::uint64_t(lo)%step)%step;
    if(first>std::uint64_t(hi))return {};
    const auto count=(std::uint64_t(hi)-first)/step+1;
    // At most one selected position per stored region is admitted here. This
    // bound forbids constructing a logical-member vector from a compact region.
    if(count<2 || count>split.regions.size() || count>UINT64_MAX/width)return {};
    const PartitionLatticeKey key{&split,valueKey(reader),type.toString(),first,step,count};
    auto [entry,inserted]=partition_lattices.try_emplace(key);
    if(inserted) {
      const bool raw_reader=memory.readReplacement({reader.nodeId(),reader.instanceIndex()})!=nullptr;
      const auto reader_type=input.type(reader);
      if(raw_reader && (!reader_type.hasStaticShape() || reader_type.element().isParameterized()))return {};
      const auto reader_bits=raw_reader?reader_type.bitSize():0;
      std::vector<std::pair<std::uint64_t,IndexedSourceSlice>> pieces;
      pieces.reserve(std::min<std::size_t>(split.regions.size(),count));
      bool valid=true;
      std::uint64_t previous_end=0;
      for(const auto& region:split.regions) {
        if(!region.bytes || region.begin<previous_end || region.begin>UINT64_MAX-region.bytes){valid=false;break;}
        const auto end=region.begin+region.bytes;
        previous_end=end;
        if(end<=first || region.begin>std::uint64_t(hi))continue;
        const auto begin=std::max(first,region.begin);
        const auto q=(begin-first)/step+((begin-first)%step!=0);
        if(q>=count)continue;
        const auto at=first+q*step;
        if(at>=end)continue;
        if(width>end-at || (q+1<count && step<end-at)) {valid=false;break;}
        // Prefer the authoritative immutable read substitution. slice() in
        // the shared matcher already resolves assembled pieces and read aliases
        // with cached piece-end indices, preserving the actual writer/use.
        // In particular never strip an indexed region down to its prototype.
        if(raw_reader) {
          const auto bits=reader_bits;
          if(bits%8 || at>bits/8 || width>bits/8-at){valid=false;break;}
          pieces.push_back({q,{reader,at,width}});
          continue;
        }
        const auto& source_view=region.view;
        // Existing indexed regions are already handled by the direct covering
        // path. Never flatten one here to satisfy matchSlices' finite API.
        if(source_view.indexed_source || source_view.blocks.size()!=1 ||
           source_view.sources.size()!=1 || source_view.blocks[0].source!=0) {valid=false;break;}
        const auto* value=std::get_if<v::Value>(&source_view.sources[0]);
        if(!value){valid=false;break;}
        const auto mapped=viewOffset(source_view.blocks[0],Offset(integer(at)),type);
        if(!mapped || mapped->expression || !mapped->terms.empty() || mapped->base<0) {valid=false;break;}
        const auto bits=input.type(*value).bitSize();
        if(bits%8 || std::uint64_t(mapped->base)>bits/8 || width>bits/8-std::uint64_t(mapped->base)) {valid=false;break;}
        pieces.push_back({q,{*value,std::uint64_t(mapped->base),width}});
      }
      if(valid && pieces.size()==count) {
        std::sort(pieces.begin(),pieces.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        std::vector<IndexedSourceSlice> slices;slices.reserve(pieces.size());
        for(std::size_t i=0;i<pieces.size();++i) {
          if(pieces[i].first!=i){valid=false;break;}
          slices.push_back(pieces[i].second);
        }
        if(valid)if(auto pattern=output_source_matcher.matchSlices(slices)) {
          auto source=std::make_shared<const IndexedSource>(std::move(*pattern));
          if(source->count==count)
            entry->second=MemoryView{v::ValueType::matrix(type,{count}),MemoryViewKind::Layout,
                {source->prototype},{{0,0,0,width,{{count,0,width}}}},{},std::move(source)};
        }
      }
    }
    if(!entry->second)return {};
    auto index=quotient(plus(offset,Offset(-integer(first))),integer(step));
    return PartitionLattice{&*entry->second,times(index,integer(width))};
  }
  s::Ref partition(const PartitionedRead& split,v::Value reader,const Offset& offset,const v::ElementType& type) {
    SENTRYWARD_LEDGER(ledger->add("partition_queries");ledger->add("partition_regions_available",split.regions.size()));
    const auto [lo,hi]=bounds(offset);
    if(lo>=0 && hi>=lo)for(const auto& region:split.regions)
      if(std::uint64_t(lo)>=region.begin && std::uint64_t(hi)<=region.begin+region.bytes &&
         type.width()/8<=region.begin+region.bytes-std::uint64_t(hi) &&
         (!split.template_domains || viewOffset(region.view.blocks.at(0),offset,type))) {
        return view(region.view,offset,type);
      }
    // A template-domain partition is not necessarily a complete rectangular
    // formula: independent row/column operands and causal zero masks can span
    // many such regions. Reuse the cached grid proof over retained raw pieces
    // before splitting the caller's output domain. partitionGrid never expands
    // compact source members; its finite input is the stored read fragments.
    if(const auto* grid=partitionGrid(split,reader,type))return view(*grid,offset,type);
    if(!split.template_domains)
      if(auto lattice=partitionLattice(split,reader,offset,type))return view(*lattice->view,lattice->offset,type);
    refinePartitionDomain(split,offset,type);
    SENTRYWARD_AUDIT({
      (void)audit;
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_4);
      for(auto d:domains)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_5);
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_6);
      for(std::size_t i=0;i<std::min<std::size_t>(8,split.regions.size());++i) {
        const auto& r=split.regions[i];SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_7);
        for(const auto& b:r.view.blocks) {SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_8);for(const auto& a:b.axes)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_9);}
        SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_10);
      }
    });
    fail("compact_view_unsupported","projection crosses independent read-result domains");
  }
  // Shared domain-only preflight: no Value projection and no scalar nodes.
  // The same exact boundaries serve ordinary reads and MMA operand planning.
  void refinePartitionDomain(const PartitionedRead& split,const Offset& offset,const v::ElementType& type) {
    SENTRYWARD_LEDGER(ledger->add("partition_refinement_queries");ledger->add("partition_refinement_regions",split.regions.size()));
    if(!split.regions.empty() && !split.regions[0].view.blocks.at(0).axes.empty()) {
      const auto outer=split.regions[0].view.blocks[0].axes.back();
      const bool grid=outer.result_stride && std::all_of(split.regions.begin(),split.regions.end(),
          [&](const auto& r){return r.view.blocks.size()==1 && !r.view.blocks[0].axes.empty() &&
            r.view.blocks[0].axes.back().result_stride==outer.result_stride;});
      if(grid) {
        // A single strided axis also has row gaps (e.g. interleaved columns).
        // Introduce the row radix once, then split only the within-row domain.
        // Never turn repeated row boundaries into one cut per logical row.
        if(output_refinement_depth && !offset.expression && offset.terms.size()==1) {
          const auto [axis,step]=*offset.terms.begin();
          if(axis<output_domain_rank && step>0 && outer.result_stride%std::uint64_t(step)==0) {
            const auto period=outer.result_stride/std::uint64_t(step);
            if(period>1 && domains[axis]>period && domains[axis]%period==0)
              throw OutputDomainRadix{axis,period};
          }
        }
        const auto row_index=quotient(offset,outer.result_stride);
        const auto [row_lo,row_hi]=bounds(row_index);
        auto row=split;row.regions.clear();
        auto bands=split;bands.regions.clear();
        bool crosses=false;
        for(const auto& region:split.regions) {
          const auto& block=region.view.blocks[0];
          const auto first=region.begin/outer.result_stride;
          const auto count=block.axes.back().count;
          auto band=region;band.begin=first;band.bytes=count;
          band.view.blocks[0].axes.clear();bands.regions.push_back(std::move(band));
          if(row_hi<integer(first) || row_lo>=integer(first+count))continue;
          if(row_lo<integer(first) || row_hi>=integer(first+count)) {crosses=true;continue;}
          auto local=region;local.begin%=outer.result_stride;
          local.bytes-=(count-1)*outer.result_stride;
          local.view.blocks[0].result_offset=local.begin;
          local.view.blocks[0].axes.pop_back();row.regions.push_back(std::move(local));
        }
        // Split only at stored source-band boundaries, never at every row.
        // Once inside one band, all columns are checked in the local row domain.
        if(crosses){refinePartitionDomain(bands,row_index,bit(8));return;}
        if(!row.regions.empty()) {
          refinePartitionDomain(row,remainder(offset,outer.result_stride),type);return;
        }
      }
    }
    const auto [lo,hi]=bounds(offset);
    // Positive affine/add/scale/quotient maps are monotone in every binder.
    // Invert source boundaries using lower/upper envelopes for the remaining
    // coordinates: O(index axes * regions * log(domain) * index expression),
    // not Cartesian members. Varying remainder maps still need a radix domain.
    std::set<unsigned> varying_axes;
    const auto monotone=[&](auto&& self,const Offset& x)->bool {
      if(!x.expression) {
        for(auto [level,coefficient]:x.terms) {
          if(domains.at(level)==1)continue;
          if(coefficient<0)return false;
          varying_axes.insert(level);
        }
        return true;
      }
      const auto& e=*x.expression;
      return self(self,e.left) &&
        (e.kind==OffsetExpression::Add ? self(self,e.right) :
         (e.kind==OffsetExpression::Scale || e.kind==OffsetExpression::Quotient) && e.factor>0);
    };
    if((output_refinement_depth || fold_owner) && lo>=0 && monotone(monotone,offset))
    for(const auto axis_value:varying_axes) {
      const std::optional<unsigned> axis=axis_value;
      const auto at=[&](auto&& self,const Offset& x,std::uint64_t i,bool upper)->std::int64_t {
        if(!x.expression) {
          auto n=x.base;
          for(auto [level,c]:x.terms)
            n=add(n,mul(c,integer(level==*axis?i:(upper?domains[level]-1:0))));
          return n;
        }
        const auto& e=*x.expression;const auto n=self(self,e.left,i,upper);
        if(e.kind==OffsetExpression::Add)return add(n,self(self,e.right,i,upper));
        if(e.kind==OffsetExpression::Scale)return mul(n,e.factor);
        return n/e.factor;
      };
      OutputDomainCuts refinement;
      refinement.axis=*axis;
      const auto inverse=[&](std::uint64_t boundary) {
        // Bound the other index axes at both corners. These monotone envelopes
        // isolate source-boundary crossings without enumerating Cartesian
        // members; a subsequent query can refine another remaining axis.
        for(const bool upper:{false,true}) {
        std::uint64_t first=0,last=domains[*axis];
        while(first<last) {
          const auto middle=first+(last-first)/2;
          if(at(at,offset,middle,upper)<integer(boundary))first=middle+1;else last=middle;
        }
        if(first && first<domains[*axis])refinement.cuts.push_back(first);
        }
      };
      for(const auto& region:split.regions) {
        inverse(region.begin);
        // Split also before an element would straddle a source boundary.
        if(region.bytes>=type.width()/8)
          inverse(region.begin+region.bytes-type.width()/8+1);
      }
      std::sort(refinement.cuts.begin(),refinement.cuts.end());
      refinement.cuts.erase(std::unique(refinement.cuts.begin(),refinement.cuts.end()),refinement.cuts.end());
      if(!refinement.cuts.empty()) {
        if(*axis>=output_domain_rank)refineFold(*axis,refinement.cuts);
        SENTRYWARD_AUDIT({
          (void)audit;static unsigned witnesses=0;
          if(sentryward::debug::environment("SENTRYWARD_OUTPUT_DOMAIN_WITNESS") && witnesses++<8) {
            SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_11);
            for(std::size_t i=0;i<std::min<std::size_t>(4,split.regions.size());++i) {
              const auto& r=split.regions[i];SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_12);
              SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_13);
              if(r.view.indexed_source) {
                const auto& source=*r.view.indexed_source;
                SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_14);
                if(source.root) {
                  SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_15);
                  if(source.root->binding)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_16);
                  if(!source.root->children.empty())SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_17);
                }
              } else for(const auto& source:r.view.sources)
                if(const auto* value=std::get_if<v::Value>(&source))
                  SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_18);
              for(const auto& b:r.view.blocks) {SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_19);
                for(const auto& a:b.axes)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_20);}
              SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_21);
            }
          }
        });
        throw refinement;
      }
    }
    // Periodic source selection stays rectangular in (quotient,remainder)
    // coordinates. Reshape the *output domain*, not every repeated row.
    const auto radix=[&](auto&& self,const Offset& x)->std::optional<OutputDomainRadix> {
      if(!x.expression)return {};
      const auto& e=*x.expression;
      if(e.kind==OffsetExpression::Remainder && e.factor>1 && !e.left.expression &&
         e.left.base==0 && e.left.terms.size()==1) {
        const auto [level,c]=*e.left.terms.begin();
        if(level<output_domain_rank && c==1 && domains[level]>std::uint64_t(e.factor) &&
           domains[level]%e.factor==0)return OutputDomainRadix{level,std::uint64_t(e.factor)};
      }
      if(auto found=self(self,e.left))return found;
      return self(self,e.right);
    };
    if(output_refinement_depth)if(auto split=radix(radix,offset))throw *split;
    SENTRYWARD_AUDIT({
      audit->event("DOMAIN_SPLIT_REQUIRED","read_projection_crosses_source_partition",split.regions.size(),1);
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_22);
      for(auto n:domains)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_23);
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_24);
      for(std::size_t i=0;i<std::min<std::size_t>(split.regions.size(),12);++i) {
        const auto& r=split.regions[i];SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_25);
        if(r.view.indexed_source)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_26);
        SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_27);
      }
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_28);
    });
  }
  s::Ref read(v::Value value,const Offset& offset,const v::ElementType& type){
    if(auto found=memory.symbolic_initial_reads.find({value.nodeId(),value.instanceIndex()});
        found!=memory.symbolic_initial_reads.end()) {
      const auto& region=found->second;
      auto [lo,hi]=bounds(offset);
      const auto bytes=(type.width()+7)/8;
      if(lo<0 || hi<0 || std::uint64_t(hi)>region.bytes || bytes>region.bytes-std::uint64_t(hi))
        fail("scalar_fold_invalid_index","symbolic initial-memory slice exceeds its recorded payload");
      const auto base=allocation_bases.find(region.allocation);
      if(base==allocation_bases.end())
        fail("symbolic_initial_base_unsupported","No unique pointer root for symbolic initial-memory allocation "+region.allocation);
      // Preserve the exact integer address, including page-table loads. This is
      // a subtraction of the real object base, not substitution of base=0.
      const auto address=input.instance(value).operands.at(0);
      auto address_value=inUse(indexed_context && indexed_context->use &&
          indexed_context->use->kind==IndexedSourceUse::Kind::Operands?
          indexed_context->use->children.at(0):nullptr,[&]{return scalar(address);});
      auto base_value=inUse(nullptr,[&]{return scalar(base->second);});
      auto relative=emitOperation("Subtract","u64",{address_value,base_value});
      relative=emitOperation("Add","u64",{relative,offsetValue(plus(offset,Offset(integer(region.offset))))});
      return result.graph.initialCell(region.allocation,relative,type.toString());
    }
    if(spatial_count) {
      const auto stride=spatial_instance_strides.find(valueKey(value));
      if(stride!=spatial_instance_strides.end()) {
        const MemoryInstance key{value.nodeId(),value.instanceIndex()};
        auto found=memory.initial_read_families.upper_bound(key);
        if(found!=memory.initial_read_families.begin()) {
          const auto& family=std::prev(found)->second;
          const auto last=static_cast<__int128>(key.instance)+static_cast<__int128>(stride->second)*(spatial_count-1);
          if(family.first.node==key.node && key.instance>=family.first.instance &&
             key.instance-family.first.instance<family.count && last>=family.first.instance &&
             last-static_cast<__int128>(family.first.instance)<family.count) {
            const auto first=memory.initialRead(key);
            const auto step=static_cast<__int128>(stride->second)*family.byte_stride;
            if(first && first->region.offset<=INT64_MAX && step>=INT64_MIN && step<=INT64_MAX) {
              const auto [lo,hi]=bounds(offset);
              if(lo>=0 && hi>=0 && std::uint64_t(hi)<=first->region.bytes &&
                 type.width()/8<=first->region.bytes-std::uint64_t(hi)) {
                Offset address(static_cast<std::int64_t>(first->region.offset));
                if(step)address.terms[0]=static_cast<std::int64_t>(step);
                return result.graph.initialCell(first->region.allocation,offsetValue(plus(address,offset)),type.toString());
              }
            }
          }
        }
      }
      std::vector<v::Value> replacements;
      std::vector<std::uint64_t> offsets;
      std::vector<InitialMemorySymbol> initials;
      const auto& members=spatialValues(value);
      for(std::size_t member=0;member<members.count;++member) {
        const auto x=members.at(member);
        auto found=memory.normalizedRead({x.nodeId(),x.instanceIndex()});
        auto family_initial=memory.initialRead({x.nodeId(),x.instanceIndex()});
        if(family_initial)found=NormalizedReadReference{&*family_initial};
        if(!found)spatialReject();
        const v::Value* replacement=nullptr;
        const InitialMemorySymbol* initial=nullptr;
        std::uint64_t source_offset=0,source_bytes=0;
        if(const auto* item=std::get_if<const v::Value*>(&*found))replacement=*item;
        else if(const auto* item=std::get_if<const InitialMemorySymbol*>(&*found))initial=*item;
        else {
          if(!std::holds_alternative<const MemoryView*>(*found))spatialReject();
          const auto& view=**std::get_if<const MemoryView*>(&*found);
          if(view.indexed_source)spatialReject();
          if(view.blocks.size()!=1)spatialReject();
          const auto& block=view.blocks[0];
          if(block.result_offset || !block.axes.empty() || block.bytes<view.type.bitSize()/8 ||
             block.source>=view.sources.size())spatialReject();
          const auto& source=view.sources[block.source];
          source_offset=block.source_offset;source_bytes=block.bytes;
          replacement=std::get_if<v::Value>(&source);
          initial=std::get_if<InitialMemorySymbol>(&source);
        }
        if(replacement) {
          if(source_offset)spatialReject();
          replacements.push_back(*replacement);
        } else if(initial) {
          auto item=*initial;
          if(source_offset>item.region.bytes || source_bytes>item.region.bytes-source_offset ||
             source_offset>UINT64_MAX-item.region.offset)spatialReject();
          item.region.offset+=source_offset;
          if(source_bytes)item.region.bytes=source_bytes;
          if(!initials.empty() && (item.type!=initials[0].type ||
             item.region.allocation!=initials[0].region.allocation || item.region.bytes!=initials[0].region.bytes))spatialReject();
          offsets.push_back(item.region.offset);initials.push_back(std::move(item));
        } else spatialReject();
      }
      if(replacements.size()==spatial_count) {
        const auto source=replacements.front();registerSpatial(std::move(replacements));
        return project(source,offset,type);
      }
      if(initials.empty() || offsets.size()!=spatial_count)spatialReject();
      const auto* first=&initials[0];
      const auto [lo,hi]=bounds(offset);
      if(lo<0 || hi<0 || std::uint64_t(hi)>first->region.bytes || type.width()/8>first->region.bytes-std::uint64_t(hi))spatialReject();
      return result.graph.initialCell(first->region.allocation,
          offsetValue(plus(offset,spatialOffsets(offsets))),type.toString());
    }
    auto found=memory.normalizedRead({value.nodeId(),value.instanceIndex()});
    auto family_initial=memory.initialRead({value.nodeId(),value.instanceIndex()});
    if(family_initial)found=NormalizedReadReference{&*family_initial};
    if(!found)fail("scalar_fold_missing_read","read has no normalized memory substitution");
    if(const auto* r=std::get_if<const v::Value*>(&*found))return project(**r,offset,type);
    if(const auto* r=std::get_if<const InitialMemorySymbol*>(&*found))return initial(**r,offset,type);
    if(const auto* r=std::get_if<const PartitionedRead*>(&*found))return partition(**r,value,offset,type);
    return view(*std::get<const MemoryView*>(*found),offset,type);
  }
  bool scalarOperation(const v::Operation& op)const{
    return dynamic_cast<const v::UnaryOp*>(&op) || dynamic_cast<const v::BinaryOp*>(&op) ||
      dynamic_cast<const v::FmaOp*>(&op) || dynamic_cast<const v::CompareOp*>(&op) ||
      dynamic_cast<const v::SelectOp*>(&op) || dynamic_cast<const v::IdealExp2Op*>(&op) ||
      dynamic_cast<const v::PtxScalarOp*>(&op);
  }
  struct ScalarReductionPlan {
    s::Reducer reducer;
    std::vector<v::Value> terms;
    std::vector<IndexedSourceDomain> pieces;
  };
  std::unordered_map<std::string,std::optional<ScalarReductionPlan>> scalar_reduction_plans;
  std::optional<s::Ref> plannedScalarReduction(v::Value value,const v::ElementType& type) {
    // Plan existing scalar instruction chains before projecting their terms.
    // No logical member materialization and no comparison of scalar DAGs.
    if(indexed_context || spatial_count || type.kind()!=v::ElementKind::Floating)return {};
    const auto* binary=dynamic_cast<const v::BinaryOp*>(&input.definingNode(value).operation());
    if(!binary || binary->attributes().saturate || binary->attributes().propagate_nan)return {};
    std::optional<s::Reducer> reducer;
    switch(binary->opcode()) {
      case v::BinaryOpcode::Add:reducer=s::Reducer::Sum;break;
      case v::BinaryOpcode::Maximum:reducer=s::Reducer::Max;break;
      case v::BinaryOpcode::Minimum:reducer=s::Reducer::Min;break;
      default:return {};
    }
    const auto key=valueKey(value);
    auto found=scalar_reduction_plans.find(key);
    if(found==scalar_reduction_plans.end()) {
      ScalarReductionPlan plan{*reducer,{},{}};
      auto cursor=value;
      for(;;) {
        cursor=specialization.selected(input.canonicalInterpretation(cursor));
        const auto* op=dynamic_cast<const v::BinaryOp*>(&input.definingNode(cursor).operation());
        if(cursor.conversion() || !cursor.indices().empty() || cursor.threadSelector() ||
           !op || op->opcode()!=binary->opcode() || op->attributes().saturate || op->attributes().propagate_nan ||
           input.type(cursor)!=v::ValueType::scalar(type))break;
        const auto args=input.instance(cursor).operands;
        if(args.size()!=2)break;
        plan.terms.push_back(args[1]);cursor=args[0];
      }
      plan.terms.push_back(cursor);std::reverse(plan.terms.begin(),plan.terms.end());
      if(plan.terms.size()>=3)plan.pieces=pure_source_templates.map(plan.terms);
      const bool compact=std::any_of(plan.pieces.begin(),plan.pieces.end(),
          [](const auto& d){return d.source && d.count>=3;});
      found=scalar_reduction_plans.emplace(key,compact?std::optional(std::move(plan)):std::nullopt).first;
    }
    if(!found->second)return {};
    const auto& plan=*found->second;
    const auto level=domains.size();
    const auto identity=*reducer==s::Reducer::Sum?"0":*reducer==s::Reducer::Max?"-inf":"inf";
    auto& progress=resumeReduction([&]{return result.graph.constant(type.toString(),identity);});
    ReductionRequests requests(requested,progress);
    auto accumulated=cache.at(progress.accumulator_key);
    for(auto i=progress.next;i<plan.pieces.size();++i) {
      const auto& piece=plan.pieces[i];
      const FoldPieceKey piece_key{*projection_key,i};
      auto [parts,inserted]=fold_pieces.try_emplace(piece_key,
          std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,piece.count}});
      for(auto sub=progress.sub_next;sub<parts->second.size();++sub) {
      const auto [first,count]=parts->second[sub];
      FoldScope scope(*this,piece_key,level,sub,first,count);
      s::Ref term;
      {
        struct RestoreDomains {
          std::vector<std::uint64_t>& current;std::vector<std::uint64_t> saved;
          ~RestoreDomains(){current=std::move(saved);}
        } restore{domains,domains};
        if(piece.source && piece.count>1) {
          Offset coordinate(integer(first));
          if(count>1){domains.push_back(count);coordinate.terms[level]=1;}
          term=inContext(std::make_shared<IndexedContext>(IndexedContext{coordinate,piece.source->root}),
              [&]{return project(piece.source->prototype,Offset(0),type);});
        } else term=scalar(plan.terms[piece.begin]);
      }
      if(piece.source && count>1)
        accumulated=result.graph.fold(*reducer,accumulated,term,level,count,type.toString());
      else accumulated=emitOperation(*reducer==s::Reducer::Sum?"Add":
          *reducer==s::Reducer::Max?"Maximum":"Minimum",type.toString(),{accumulated,term},"ideal_arithmetic");
      cache.at(progress.accumulator_key)=accumulated;progress.sub_next=sub+1;progress.pending.clear();
      }
      commitReduction(progress,i+1,accumulated);limit();
    }
    return accumulated;
  }
  s::Ref pure(v::Value value,const Offset& index,const v::ElementType& type){
    if(auto compact=plannedScalarReduction(value,type))return *compact;
    const auto& node=input.definingNode(value);const auto& op=node.operation();
    const auto input_operands=input.instance(value).operands;
    std::vector<s::Ref> operands(input_operands.size());
    const auto projectOperand=[&](std::size_t i) {
      auto operand=input_operands[i];
      const auto& t=input.type(operand);
      const auto projectChild=[&] {
      if(t.isScalar())return scalar(operand);
      else {
        if(!t.hasStaticShape())fail("scalar_fold_parameterized_shape","matrix operand shape is descriptor-dependent");
        return project(operand,times(index,integer(t.element().width()/8)),t.element());
      }
      };
      if(indexed_context && indexed_context->use && indexed_context->use->kind==IndexedSourceUse::Kind::Operands)
        operands[i]=inUse(indexed_context->use->children.at(i),projectChild);
      else operands[i]=projectChild();
    };
    if(dynamic_cast<const v::FmaOp*>(&op) && input_operands.size()==3) {
      projectOperand(2);projectOperand(0);projectOperand(1);
    } else if(const auto* binary=dynamic_cast<const v::BinaryOp*>(&op);
        binary && binary->opcode()==v::BinaryOpcode::Multiply && input_operands.size()==2 &&
        input_operands[1].nodeId()<input_operands[0].nodeId()) {
      projectOperand(1);projectOperand(0);
    } else for(std::size_t i=0;i<input_operands.size();++i)projectOperand(i);
    if (type.kind()==v::ElementKind::Floating) {
      const auto eligible=[](const v::ArithmeticAttributes& a) {
        // This path constructs ideal-real mathematics, consistently with the
        // equivalence normalizer: rounding and FTZ are not scalar operations.
        // The original Value IR still retains the hardware attributes. Do not
        // discard saturation or special NaN behavior under this abstraction.
        return !a.saturate && !a.propagate_nan;
      };
      if (const auto* binary=dynamic_cast<const v::BinaryOp*>(&op);
          binary && eligible(binary->attributes())) {
        std::optional<s::Reducer> reducer;
        switch(binary->opcode()) {
          case v::BinaryOpcode::Add: reducer=s::Reducer::Sum; break;
          case v::BinaryOpcode::Multiply: reducer=s::Reducer::Product; break;
          case v::BinaryOpcode::Maximum: reducer=s::Reducer::Max; break;
          case v::BinaryOpcode::Minimum: reducer=s::Reducer::Min; break;
          default: break;
        }
        if(reducer || binary->opcode()==v::BinaryOpcode::Subtract || binary->opcode()==v::BinaryOpcode::Divide)
          return real_expressions.operation(std::string(op.name()),type.toString(),std::move(operands),"ideal_arithmetic",domains.size());
      }
      // Explicit PTX rounding is needed by bit-level numerical idioms (for
      // example exponent construction). Preserve the fused operation until a
      // consumer proves a complete abstraction, rather than guessing it from
      // constants after the metadata has been erased. Unspecified rounding is
      // the existing ideal-mathematics reduction contract.
      if (const auto* fma=dynamic_cast<const v::FmaOp*>(&op);fma && eligible(fma->attributes()) &&
          fma->attributes().rounding==v::RoundingMode::None) {
        auto product=real_expressions.operation("Multiply",type.toString(),
                                            {operands.at(0),operands.at(1)},"ideal_arithmetic",domains.size());
        return real_expressions.operation("Add",type.toString(),{operands.at(2),product},"ideal_arithmetic",domains.size());
      }
      if(const auto* unary=dynamic_cast<const v::UnaryOp*>(&op);
          unary && unary->opcode()==v::UnaryOpcode::Negate && eligible(unary->attributes()))
        return real_expressions.operation("Negate",type.toString(),std::move(operands),"ideal_arithmetic",domains.size());
      if(isIdealExp2(op))
        return real_expressions.operation("Exp2",type.toString(),std::move(operands),"ideal_arithmetic",domains.size());
      if(const auto* ptx=dynamic_cast<const v::PtxScalarOp*>(&op);
          ptx && ptx->opcode()==v::PtxScalarOpcode::Reciprocal) {
        // rcp's standard approximation belongs to the admitted ideal-real
        // numerical abstraction, like native/software Exp2 and FTZ arithmetic.
        // The original Value IR still preserves mnemonic and modifiers.
        const auto& modifiers=ptx->attributes().modifiers;
        const bool admitted=std::all_of(modifiers.begin(),modifiers.end(),[](const auto& modifier) {
          return modifier=="approx" || modifier=="ftz" || modifier=="rn" ||
            modifier=="rz" || modifier=="rm" || modifier=="rp" ||
            modifier=="f32" || modifier=="f64";
        });
        if(admitted)
          return real_expressions.operation("Reciprocal",type.toString(),std::move(operands),"ideal_arithmetic",domains.size());
      }
    }
    auto attr=attributes(op);
    if(node.resultCount()>1)attr+=" result="+std::to_string(value.resultIndex());
    if(const auto* p=dynamic_cast<const v::PtxScalarOp*>(&op))attr+=" source_type="+p->sourceType().toString();
    return emitOperation(std::string(op.name()),type.toString(),std::move(operands),attr);
  }
  // Pull a reduction's index progression through stored read domains. This
  // never asks whether the source expressions have a common mathematical form.
  std::optional<std::vector<algorithm::AffineDomain>> reductionDomains(
      v::Value value,const Offset& offset,std::uint64_t stride,
      const v::ElementType& type,std::uint64_t count) {
    SENTRYWARD_LEDGER(ledger->add("reduction_domain_queries"));
    const auto [lo,hi]=bounds(offset);
    if(lo<0 || hi<lo || !stride || !count || !type.width() || type.width()%8)return {};
    const auto read=memory.normalizedRead({value.nodeId(),value.instanceIndex()});
    const auto* split=read?std::get_if<const PartitionedRead*>(&*read):nullptr;
    if(split)if(const auto* grid=partitionGrid(**split,value,type)) {
      const auto end=static_cast<unsigned __int128>(hi)+
          static_cast<unsigned __int128>(count-1)*stride+type.width()/8;
      if(end<=grid->type.bitSize()/8)
        return std::vector<algorithm::AffineDomain>{{0,1,count}};
    }
    if(!split || !(*split)->template_domains) {
      if(readSpan(value,offset,stride,type,count)==count)
        return std::vector<algorithm::AffineDomain>{{0,1,count}};
      return {};
    }
    std::vector<algorithm::AffineDomain> result;
    std::uint64_t covered=0;
    const auto width=type.width()/8;
    for(const auto& region:(*split)->regions) {
      // Filter whole stored row groups before interpreting their local axes.
      // Different source templates may own disjoint row intervals.
      const auto query_end=static_cast<unsigned __int128>(hi)+
          static_cast<unsigned __int128>(count-1)*stride+width;
      if(query_end<=region.begin || std::uint64_t(lo)>=region.begin+region.bytes)continue;
      auto b=region.view.blocks.at(0);
      // This is a coverage plan, not a rewrite of the source mapping. Adjacent
      // payload repetitions cover one interval even when their source stride
      // differs. Keep the original view for projection of the Fold body.
      // Collapse only inside this one read-result region/occurrence.
      std::erase_if(b.axes,[](const auto& axis){return axis.count==1;});
      while(!b.axes.empty() && b.axes.front().result_stride==b.bytes) {
        const auto count=b.axes.front().count;
        if(!count || b.bytes>UINT64_MAX/count)return {};
        b.bytes*=count;b.axes.erase(b.axes.begin());
      }
      auto local_offset=offset;auto local_lo=lo,local_hi=hi;
      while(b.axes.size()>1) {
        const auto outer=b.axes.back();
        if(!outer.result_stride || !outer.count)return {};
        const auto origin=b.result_offset-b.result_offset%outer.result_stride;
        const auto relative=plus(local_offset,Offset(-integer(origin)));
        const auto row=bounds(quotient(relative,outer.result_stride));
        if(row.first<0 || row.second<row.first || std::uint64_t(row.second)>=outer.count)return {};
        local_offset=remainder(relative,outer.result_stride);
        b.result_offset-=origin;
        std::tie(local_lo,local_hi)=bounds(local_offset);
        if(local_lo<0 || static_cast<unsigned __int128>(local_hi)+
            static_cast<unsigned __int128>(count-1)*stride+width>outer.result_stride)return {};
        b.axes.pop_back();
      }
      if(b.axes.size()>1 || b.bytes<width)return {};
      const auto lo=local_lo,hi=local_hi;
      if(b.axes.empty()) {
        // A single contiguous source region can cover a consecutive K range.
        const auto lower=lo>=integer(b.result_offset)?0:
            (b.result_offset-std::uint64_t(lo)+stride-1)/stride;
        const auto end=b.result_offset+b.bytes-width;
        if(std::uint64_t(hi)>end || lower>=count)continue;
        const auto upper=std::min(count-1,(end-std::uint64_t(hi))/stride);
        if(lower<=upper){result.push_back({lower,1,upper-lower+1});covered+=upper-lower+1;}
        continue;
      }
      const auto& axis=b.axes[0];
      if(!axis.count || !axis.result_stride)return {};
      const auto residue=bounds(remainder(local_offset,axis.result_stride));
      if(residue.first!=residue.second)return {};
      // One contraction remains within one repetition period, even while the
      // output binder selects many periods. Intersect the contiguous payload
      // with K as one interval, not one AP per payload byte/element.
      const auto r=std::uint64_t(residue.first);
      const auto start=b.result_offset%axis.result_stride;
      if(b.bytes<=axis.result_stride-start &&
         static_cast<unsigned __int128>(r)+static_cast<unsigned __int128>(count-1)*stride+width<=axis.result_stride) {
        const auto lower=r>=start?0:(start-r+stride-1)/stride;
        if(start+b.bytes>=r+width) {
          const auto upper=std::min(count-1,(start+b.bytes-width-r)/stride);
          const auto first=static_cast<unsigned __int128>(lo)+static_cast<unsigned __int128>(lower)*stride;
          const auto last=static_cast<unsigned __int128>(hi)+static_cast<unsigned __int128>(upper)*stride+width;
          const auto end=static_cast<unsigned __int128>(b.result_offset)+
              static_cast<unsigned __int128>(axis.count-1)*axis.result_stride+b.bytes;
          if(lower<=upper && first>=b.result_offset && last<=end) {
            result.push_back({lower,1,upper-lower+1});covered+=upper-lower+1;continue;
          }
        } else continue;
      }
      const auto inner=(std::uint64_t(residue.first)+axis.result_stride-
          b.result_offset%axis.result_stride)%axis.result_stride;
      // If K changes the residue, intersect its exact progression with each
      // payload position. The loop is over bytes of one stored fragment, not
      // over threads, matrix elements, or reduction members.
      const bool varying_residue=stride%axis.result_stride!=0;
      const auto positions=varying_residue?b.bytes-width+1:1;
      for(std::uint64_t p=0;p<positions;++p) {
        const auto within=varying_residue?p:inner;
        if(within>b.bytes-width)continue;
        const algorithm::AffineDomain addresses{b.result_offset+within,axis.result_stride,axis.count};
        const auto intersection=algorithm::IntersectAffine(
            {std::uint64_t(lo),stride,count},addresses);
        if(!intersection.count)continue;
        const auto end=addresses.last()+b.bytes-within-width;
        if(std::uint64_t(hi)>end)continue;
        const auto upper=std::min(count-1,(end-std::uint64_t(hi))/stride);
        if(intersection.left>upper)continue;
        const auto step=intersection.count>1?intersection.left_step:1;
        const auto n=std::min(intersection.count,1+(upper-intersection.left)/step);
        result.push_back({intersection.left,step,n});covered+=n;
      }
    }
    if(covered!=count)return {};
    return result;
  }
  s::Ref warpMma(v::Value value,const Offset& index){
    auto [lo,hi]=bounds(index);
    if(lo<0 || hi>=128)fail("scalar_fold_invalid_index","warp MMA output exceeds 16x8 matrix");
    const auto& op=dynamic_cast<const v::WarpMmaOp&>(input.definingNode(value).operation());
    const auto type=v::ElementType::floating(16,op.bfloat()?"bf16":"f16");
    const MemoryInstance key{value.nodeId(),value.instanceIndex()};
    auto found=warp_operand_views.find(key);
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
    if(found==warp_operand_views.end()) {
      const auto args=input.instance(value).operands;
      if(args.size()!=320)fail("scalar_fold_mma_shape","warp MMA fragment operand count is not 320");
      std::array<std::vector<IndexedSourceSlice>,3> slices;
      slices[0].reserve(256);slices[1].reserve(128);slices[2].reserve(128);
      for(unsigned row=0;row<16;++row)for(unsigned k=0;k<16;++k)
        slices[0].push_back({args.at(((row%8)*4+(k%8)/2)*4+(row/8)+2*(k/8)),2*(k%2),2});
      for(unsigned col=0;col<8;++col)for(unsigned k=0;k<16;++k)
        slices[1].push_back({args.at(128+(col*4+(k%8)/2)*2+k/8),2*(k%2),2});
      for(unsigned row=0;row<16;++row)for(unsigned col=0;col<8;++col)
        slices[2].push_back({args.at(192+((row%8)*4+col/2)*4+(row/8)*2+col%2),0,4});
      const auto make=[&](unsigned role) {
        auto pattern=output_source_matcher.matchSlices(slices[role]);
        const auto width=role==2?4u:2u;
        const auto matrix_type=v::ValueType::matrix(role==2?v::ElementType::floating(32):type,{slices[role].size()});
        PartitionedRead partition{matrix_type,{},true};
        if(pattern) {
          auto source=std::make_shared<const IndexedSource>(std::move(*pattern));
          partition.regions.push_back({0,source->count*width,MemoryView{matrix_type,
              MemoryViewKind::Layout,{source->prototype},{{0,0,0,width,{{source->count,0,width}}}},{},source}});
          return partition;
        }
        // Preserve existing register/component domains instead of requiring
        // unrelated C registers to match one expression. This finite layout
        // contains at most 256 records, independently of output/kernel size.
        std::vector<v::Value> values;std::vector<std::uint64_t> offsets;
        for(const auto& slice:slices[role]){values.push_back(slice.value);offsets.push_back(slice.offset);}
        for(const auto& domain:mapSourceTemplates(input,values,true,offsets)) {
          const auto begin=domain.begin*width;
          const auto bytes=((domain.count-1)*domain.result_stride+1)*width;
          const auto prototype=domain.source?domain.source->prototype:values[domain.begin];
          partition.regions.push_back({begin,bytes,MemoryView{matrix_type,MemoryViewKind::Layout,
              {prototype},{{0,offsets[domain.begin],begin,width,{{domain.count,0,domain.result_stride*width}}}},{},domain.source}});
        }
        return partition;
      };
      found=warp_operand_views.emplace(key,std::array<PartitionedRead,3>{make(0),make(1),make(2)}).first;
    }
    const auto& views=found->second;
    const auto operand=[&](unsigned role,const Offset& offset,const v::ElementType& element) {
      if(views[role].regions.size()==1)return view(views[role].regions.front().view,offset,element);
      return partition(views[role],value,offset,element);
    };
    const auto row=quotient(index,8),col=remainder(index,8);
    const auto level=static_cast<unsigned>(domains.size());
    struct Restore { std::vector<std::uint64_t>& domains;std::size_t size;
      ~Restore(){domains.resize(size);} } restore{domains,level};
    auto& progress=resumeReduction([&]{
      return operand(2,times(index,4),v::ElementType::floating(32));
    });
    ReductionRequests requests(requested,progress);
    auto sum=cache.at(progress.accumulator_key);
    if(progress.next)return sum;
    // Native warp contractions own their reduction axis just like other MMA
    // contractions. Read partitions may refine only the current uncommitted
    // piece; dependency suspension and collection retain completed sums.
    const FoldPieceKey piece_key{*projection_key,0};
    auto [parts,inserted]=fold_pieces.try_emplace(piece_key,
        std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,16}});
    for(auto sub=progress.sub_next;sub<parts->second.size();++sub) {
      const auto [first,count]=parts->second[sub];
      domains.resize(level);
      FoldScope scope(*this,piece_key,level,sub,first,count);
      Offset k(integer(first));
      if(count>1){domains.push_back(count);k.terms[level]=1;}
      auto a=operand(0,times(plus(times(row,16),k),2),type);
      auto b=operand(1,times(plus(times(col,16),k),2),type);
      a=result.graph.convert(a,"f32","NumericInterpretation");
      b=result.graph.convert(b,"f32","NumericInterpretation");
      const auto body=emitOperation("Multiply","f32",{a,b},"ideal_arithmetic");
      domains.resize(level);
      sum=count==1?reductions.append(s::Reducer::Sum,sum,body,"f32",level):
          reductions.appendFold(s::Reducer::Sum,sum,body,level,count,"f32");
      cache.at(progress.accumulator_key)=sum;
      progress.sub_next=sub+1;progress.pending.clear();
      limit();
    }
    commitReduction(progress,1,sum);
    return sum;
  }
  s::Ref scaledMma(v::Value value,const Offset& index,const v::TcgenMmaAttributes& attr) {
    const auto& args=input.instance(value).operands;
    const auto& dst=input.type(value);
    std::string kind;
    unsigned group=1;
    for(const auto& mod:attr.modifiers) {
      if(mod.starts_with("kind::"))kind=mod.substr(6);
      if(mod=="cta_group::2")group=2;
    }
    if(attr.sparse || attr.scale_input_d || (kind!="mxf4" && kind!="mxf4nvf4" && kind!="mxf8f6f4"))
      fail("scalar_fold_mma_variant","block-scaled scalar lowering requires dense MMA without scaled D");
    auto desc=known(args.at(2));
    if(!desc || *desc<0 || std::uint64_t(*desc)>UINT32_MAX)
      fail("scalar_fold_mma_descriptor","MMA instruction descriptor is not constant");
    const auto bits=std::uint64_t(*desc);
    const auto d=frontend::ptx::DecodeBlockScaledDescriptor(bits,kind);
    if((d.a_element_bits!=4 && d.a_element_bits!=8) ||
       (d.b_element_bits!=4 && d.b_element_bits!=8))
      fail("scalar_fold_mma_variant","block-scaled scalar lowering requires FP4 or FP8 operands");
    if(bits&(3ULL<<13))fail("scalar_fold_mma_flags","MMA operand negation is not lowered");
    const auto vector=kind=="mxf4"?2u:kind=="mxf4nvf4"?4u:1u, block=d.k/vector;
    const auto scale_index=3u+unsigned(attr.accumulate);
    const auto scale_type=v::ElementType::floating(8,(bits&(1ULL<<23))?"ue8m0":"ue4m3");
    const auto byte=v::ElementType::bit(8);
    const auto checkShape=[&](v::Value operand,std::vector<std::uint64_t> shape) {
      const auto& type=input.type(operand);
      if(!type.hasStaticShape() || !std::ranges::equal(type.shape(),shape) || type.element().width()!=8)
        fail("scalar_fold_mma_shape","packed FP4 or replicated scale shape disagrees with descriptor");
    };
    const unsigned unit=(d.a_element_bits==4 || d.b_element_bits==4)?2:1;
    const unsigned astride=unit*d.a_element_bits/8,bstride=unit*d.b_element_bits/8;
    checkShape(args.at(0),{d.m,d.k*d.a_element_bits/8});
    checkShape(args.at(1),{d.n,d.k*d.b_element_bits/8});
    checkShape(args.at(scale_index),{group,4,d.m/group,vector});
    checkShape(args.at(scale_index+1),{group,4,d.n,vector});
    if(!std::ranges::equal(dst.shape(),std::vector<std::uint64_t>{d.m,d.n}))
      fail("scalar_fold_mma_shape","FP4 output shape disagrees with descriptor");
    const auto row=quotient(index,d.n),col=remainder(index,d.n);
    const auto dtype=v::ElementType::floating(32);
    auto& progress=resumeReduction([&]{return attr.accumulate?
        project(args.at(3),times(index,4),dtype):result.graph.constant("f32","0");});
    ReductionRequests requests(requested,progress);
    auto sum=cache.at(progress.accumulator_key);
    const auto level=static_cast<unsigned>(domains.size());
    // PTX scale replicas: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-2x
    const auto scale=[&](bool b,unsigned chunk) {
      const auto rows=b?d.n:d.m/group;
      const auto part=b?Offset(0):quotient(row,rows);
      const auto mn=b?col:remainder(row,rows);
      std::optional<s::Ref> canonical;
      for(unsigned peer=0;peer<(b?group:1);++peer)for(unsigned replica=0;replica<4;++replica) {
        const auto p=b?Offset(peer):part;
        const auto position=plus(times(plus(times(plus(times(p,4),Offset(replica)),rows),mn),vector),Offset(chunk));
        auto value=project(args.at(scale_index+unsigned(b)),position,scale_type);
        if(canonical && *canonical!=value) {
#ifdef SENTRYWARD_ENABLE_COMPRESSION_AUDIT
          if(sentryward::debug::environment("SENTRYWARD_SCALE_REPLICA_DIAGNOSTIC")) {
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/blocks/scalar_fold_lowering_trace_2.inc"
#endif
}
#endif
          fail("scalar_fold_scale_replicas","scale replicas do not have the same proven value");
        }
        canonical=value;
      }
      return result.graph.convert(*canonical,"f32","NumericInterpretation");
    };
    const auto fp4=[&](s::Ref packed,unsigned half) {
      auto field=emitOperation("ExtractBits","b4",
          {packed,result.graph.constant("u64",std::to_string(half*4))},"width=4");
      return result.graph.convert(result.graph.convert(field,"e2m1","Bitcast"),"f32","NumericInterpretation");
    };
    const auto element=[&](bool b,const Offset& position,unsigned component) {
      const auto width=b?d.b_element_bits:d.a_element_bits;
      auto packed=project(args[unsigned(b)],plus(position,Offset(width==8?component:0)),byte);
      if(width==4)return fp4(packed,component);
      const auto code=(bits>>(b?10:7))&7;
      return result.graph.convert(result.graph.convert(packed,code?"e5m2":"e4m3","Bitcast"),"f32","NumericInterpretation");
    };
    for(unsigned chunk=static_cast<unsigned>(progress.next/(block/unit));chunk<vector;++chunk) {
      domains.resize(level);
      auto sa=scale(false,chunk),sb=scale(true,chunk);
      for(std::uint64_t begin=progress.next;begin<(chunk+1)*block/unit;) {
        domains.resize(level);
        auto ai=plus(times(row,d.k*d.a_element_bits/8),Offset(integer(begin*astride)));
        auto bi=plus(times(col,d.k*d.b_element_bits/8),Offset(integer(begin*bstride)));
        const auto remaining=(chunk+1)*block/unit-begin;
        auto count=std::min(readSpan(args[0],ai,astride,byte,remaining),readSpan(args[1],bi,bstride,byte,remaining));
        // Mixed FP4/FP8 contractions consume two FP8 bytes per packed unit.
        // Check both source spans without expanding the reduction domain.
        if(astride==2)count=std::min(count,readSpan(args[0],plus(ai,Offset(1)),astride,byte,remaining));
        if(bstride==2)count=std::min(count,readSpan(args[1],plus(bi,Offset(1)),bstride,byte,remaining));
        // Extent comes from the instruction/source map, not repetition
        // inference. A certified two-element domain is already a legal Fold.
        if(count>1){
          domains.push_back(count);Offset iterator;iterator.terms[level]=1;
          // ai/bi may be compound quotient/remainder expressions. Compose the
          // new binder through the Offset algebra; writing their affine terms
          // directly would leave those terms hidden behind expression.
          ai=plus(ai,times(iterator,astride));bi=plus(bi,times(iterator,bstride));
        }
        std::array<s::Ref,2> terms;
        for(unsigned half=0;half<unit;++half) {
          auto a=emitOperation("Multiply","f32",{element(false,ai,half),sa},"ideal_arithmetic");
          auto b=emitOperation("Multiply","f32",{element(true,bi,half),sb},"ideal_arithmetic");
          terms[half]=emitOperation("Multiply","f32",{a,b},"ideal_arithmetic");
        }
        auto body=unit==1?terms[0]:emitOperation("Add","f32",{terms[0],terms[1]},"ideal_arithmetic");
        domains.resize(level);
        sum=count==1?reductions.append(s::Reducer::Sum,sum,body,"f32",level):
            reductions.appendFold(s::Reducer::Sum,sum,body,level,count,"f32");
        begin+=count;commitReduction(progress,begin,sum);limit();
      }
    }
    return cast(sum,dst.element());
  }
  s::Ref mma(v::Value value,const Offset& index){
    const auto& node=input.definingNode(value);const auto& op=node.operation();
    const auto& args=input.instance(value).operands;
    const auto& dst=input.type(value);
    if(!dst.hasStaticShape() || dst.shape().size()!=2)
      fail("scalar_fold_parameterized_mma","MMA result shape is not a fixed rank-two matrix");
    const auto rows=dst.shape()[0],cols=dst.shape()[1];
    auto row=quotient(index,cols),col=remainder(index,cols);
    auto [rl,rh]=bounds(row);if(rl<0 || std::uint64_t(rh)>=rows)fail("scalar_fold_invalid_index","MMA output row exceeds shape");
    const bool native=dynamic_cast<const v::TcgenMmaOp*>(&op);
    bool accumulate=false;
    std::size_t accumulator=2;
    auto atype=input.type(args[0]).element(),btype=input.type(args[1]).element(),dtype=dst.element();
    if(native){
      const auto& attr=dynamic_cast<const v::TcgenMmaOp&>(op).attributes();
      if(attr.scaled)return scaledMma(value,index,attr);
      if(attr.scaled || attr.sparse || attr.scale_input_d)
        fail("scalar_fold_mma_variant","scaled, sparse or scaled-D MMA needs scalar semantics");
      auto desc=known(args[2]);
      if(!desc || *desc<0 || std::uint64_t(*desc)>UINT32_MAX)
        fail("scalar_fold_mma_descriptor","MMA instruction descriptor is not constant");
      const auto bits=std::uint64_t(*desc);
      // PTX instruction descriptor: flags 0..3 and negate bits need separate rules.
      if(bits & (15ULL | (3ULL<<13)))
        fail("scalar_fold_mma_flags","MMA sparsity, saturation or negate flags are not lowered");
      std::string kind;
      for(const auto& mod:attr.modifiers)if(mod.starts_with("kind::"))kind=mod.substr(6);
      if(kind!="f16" && kind!="tf32" && kind!="f8f6f4")
        fail("scalar_fold_mma_variant","MMA arithmetic kind is not supported");
      auto d=frontend::ptx::DecodeMmaDescriptor(kind,bits);
      const auto interpret=[&](unsigned code,unsigned width){
        if(kind=="f16")return v::ElementType::floating(16,code?"bf16":"f16");
        if(kind=="tf32")return v::ElementType::floating(32,"tf32");
        if(width!=8)fail("scalar_fold_subbyte_mma","sub-byte MMA elements need bit-index iterators");
        return v::ElementType::floating(8,code?"e5m2":"e4m3");
      };
      atype=interpret((bits>>7)&7,d.a_element_bits);
      btype=interpret((bits>>10)&7,d.b_element_bits);
      dtype=v::ElementType::floating(d.d_element_bits);
      accumulate=attr.accumulate;accumulator=3;
    }else{
      const auto* mat=dynamic_cast<const v::MatMulOp*>(&op);
      if(!mat)fail("scalar_fold_mma_variant","native fragment/scaled MMA is not lowered");
      accumulate=mat->accumulate();
    }
    const auto& a=input.type(args[0]);const auto& b=input.type(args[1]);
    if(!a.hasStaticShape() || !b.hasStaticShape() || a.shape().size()!=2 || b.shape().size()!=2)
      fail("scalar_fold_parameterized_mma","MMA operand shape is not fixed rank two");
    const auto k=a.shape()[1];
    if(a.shape()[0]!=rows || (native?(b.shape()[0]!=cols || b.shape()[1]!=k):(b.shape()[0]!=k || b.shape()[1]!=cols)))
      fail("scalar_fold_mma_shape","MMA operand shapes disagree with scalar contraction");
    const auto level=static_cast<unsigned>(domains.size());
    // Publish only a complete plan. Domain refinement may throw; leaving an
    // empty cache entry then would silently disable planning on the retry.
    auto plan_it=reduction_domain_plans.find(*projection_key);
    if(plan_it==reduction_domain_plans.end()) {
      std::vector<algorithm::AffineDomain> computed_plan;
      const auto ao=times(times(row,integer(k)),integer(atype.width()/8));
      const auto bo=times(native?times(col,integer(k)):col,integer(btype.width()/8));
      const auto ad=reductionDomains(args[0],ao,atype.width()/8,atype,k);
      const auto bd=reductionDomains(args[1],bo,(native?1:cols)*(btype.width()/8),btype,k);
      const auto preflight=[&](v::Value operand,const Offset& at,const v::ElementType& type) {
        const auto read=memory.normalizedRead({operand.nodeId(),operand.instanceIndex()});
        if(read)if(const auto* split=std::get_if<const PartitionedRead*>(&*read))
          refinePartitionDomain(**split,at,type);
      };
      if(!ad)preflight(args[0],ao,atype);
      if(!bd)preflight(args[1],bo,btype);
      SENTRYWARD_AUDIT({
        (void)audit;
        static thread_local std::set<std::uint64_t> reported;
        const bool fragmented=(ad && ad->size()>1) || (bd && bd->size()>1);
        if((!ad || !bd || fragmented) && reported.size()<16 && reported.insert(value.nodeId()).second) {
          SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_30);
          for(auto operand:{args[0],args[1]}) {
            const auto read=memory.normalizedRead({operand.nodeId(),operand.instanceIndex()});
            SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_31);
            if(read)if(const auto* split=std::get_if<const PartitionedRead*>(&*read)) {
              SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_32);
              for(std::size_t i=0;i<std::min<std::size_t>(4,(*split)->regions.size());++i) {
                const auto& block=(*split)->regions[i].view.blocks.at(0);
                SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_33);
                for(const auto& axis:block.axes)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_34);
                SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_35);
                const auto& region=(*split)->regions[i].view;
                if(const auto* source=std::get_if<v::Value>(&region.sources.at(block.source))) {
                  SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_36);
                  for(const auto& arg:input.instance(*source).operands)
                    SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_37);
                }
              }
            }
          }
        }
      });
      if(ad && bd) {
        SENTRYWARD_LEDGER(ledger->add("reduction_domain_candidate_pairs",ad->size()*bd->size()));
        for(const auto& a_domain:*ad)for(const auto& b_domain:*bd) {
          const auto common=algorithm::IntersectAffine(a_domain,b_domain);
          if(common.count)computed_plan.push_back({a_domain.first+a_domain.step*common.left,
              common.count>1?a_domain.step*common.left_step:1,common.count});
        }
        std::uint64_t covered=0;for(const auto& d:computed_plan)covered+=d.count;
        if(covered!=k)fail("scalar_fold_internal_binding","reduction source domains do not cover K");
      }
      plan_it=reduction_domain_plans.emplace(*projection_key,std::move(computed_plan)).first;
      SENTRYWARD_LEDGER(ledger->add(plan_it->second.empty()?"mma_unplanned":"mma_planned");
        ledger->add("mma_plan_pieces",plan_it->second.size()));
    }
    auto& progress=resumeReduction([&]{return accumulate?
        project(args.at(accumulator),times(index,integer(dst.element().width()/8)),dtype):
        result.graph.constant(dtype.toString(),"0");});
    ReductionRequests requests(requested,progress);
    auto folded=cache.at(progress.accumulator_key);
    const auto& plan=plan_it->second;
    const bool planned=!plan.empty();
    for(std::uint64_t cursor=progress.next;cursor<(planned?plan.size():k);) {
      const auto begin=planned?plan[cursor].first:cursor;
      const auto step=planned?plan[cursor].step:1;
      domains.resize(level);
      const auto a_start=times(plus(times(row,integer(k)),Offset(integer(begin))),integer(atype.width()/8));
      const auto b_start=times(native?plus(times(col,integer(k)),Offset(integer(begin))):
          plus(Offset(mul(integer(begin),integer(cols))),col),integer(btype.width()/8));
      auto count=planned?plan[cursor].count:std::min(readSpan(args[0],a_start,atype.width()/8,atype,k-begin),
          readSpan(args[1],b_start,mul(native?1:integer(cols),integer(btype.width()/8)),btype,k-begin));
      const auto whole_count=count;
      const FoldPieceKey piece_key{*projection_key,cursor};
      auto [parts,inserted]=fold_pieces.try_emplace(piece_key,
          std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,whole_count}});
      for(auto sub=progress.sub_next;sub<parts->second.size();++sub) {
      const auto [first,part_count]=parts->second[sub];count=part_count;
      SENTRYWARD_LEDGER(ledger->add("mma_piece_attempts");ledger->add(count==1?"mma_singleton_attempts":"mma_fold_attempts"));
      domains.resize(level);
      FoldScope scope(*this,piece_key,level,sub,first,count);
      // The >=3 threshold belongs to recognizing repetition in scalar chains;
      // a native contraction's certified extent does not need rediscovery.
      SENTRYWARD_AUDIT(if(count==1 && k-begin>1)
        audit->event("INSTANCE_ENUMERATION","mma_scalar_terms_then_fold_detection",k-begin,1,value.nodeId(),true);
        if(count==1)audit->add(AuditCounter::scalar_reduction_terms));
      if(count>1)domains.push_back(count);
      Offset iv(add(integer(begin),mul(integer(step),integer(first))));if(count>1)iv.terms[level]=integer(step);
      auto ai=plus(times(row,integer(k)),iv);
      auto bi=native?plus(times(col,integer(k)),iv):plus(times(iv,integer(cols)),col);
      auto av=project(args[0],times(ai,integer(atype.width()/8)),atype);
      auto bv=project(args[1],times(bi,integer(btype.width()/8)),btype);
      av=result.graph.convert(av,dtype.toString(),"NumericInterpretation");
      bv=result.graph.convert(bv,dtype.toString(),"NumericInterpretation");
      auto body=emitOperation("Multiply",dtype.toString(),{av,bv},"ideal_arithmetic");
      domains.resize(level);
      // Storage-domain boundaries do not imply different mathematical term
      // templates. Submit each emitted singleton to the same incremental
      // builder, preserving its exact operands and existing domain order.
      // Recognition starts after three terms; a mismatch remains an Add.
      if(count==1)folded=reductions.append(s::Reducer::Sum,folded,body,dtype.toString(),level);
      else folded=reductions.appendFold(s::Reducer::Sum,folded,body,level,count,dtype.toString());
      cache.at(progress.accumulator_key)=folded;progress.sub_next=sub+1;progress.pending.clear();
      SENTRYWARD_LEDGER(ledger->add("mma_pieces_committed"));
      }
      cursor+=planned?1:whole_count;
      commitReduction(progress,cursor,folded);
      limit();
    }
    return cast(folded,dst.element());
  }
  s::Ref whole(v::Value value,const Offset& index){
    const auto& op=input.definingNode(value).operation();
    const auto& type=input.type(value);
    if(const auto* c=dynamic_cast<const v::ConstantOp*>(&op)){
      if(!type.isScalar())fail("scalar_fold_matrix_constant","matrix literal needs component decoding");
      return real_expressions.constant(type.element().toString(),c->literal());
    }
    if(const auto* p=dynamic_cast<const v::InputParameterOp*>(&op)){
      // A supplied scalar launch value is a concrete specialization fact,
      // including float bits loaded through b32/b64 registers. Do not turn
      // pointer/object identity or tensor memory contents into numeric constants.
      if(type.isScalar() && p->specializedLiteral() && !p->pointer() && !value.pointer())
        return real_expressions.constant(type.element().toString(),*p->specializedLiteral());
      auto r=result.graph.parameter(type.element().toString(),p->symbol());
      if(!type.isScalar())r=emitOperation("ParameterElement",type.element().toString(),{r,offsetValue(index)});
      return r;
    }
    if(dynamic_cast<const v::CoordinateOp*>(&op)){
      if(spatial_count) {
        std::vector<std::uint64_t> values;
        const auto& members=spatialValues(value);
        for(std::size_t i=0;i<members.count;++i) {
          auto coordinate=input.instance(members.at(i)).origin.coordinate_value;
          if(!coordinate)spatialReject();
          values.push_back(*coordinate);
        }
        return result.graph.convert(offsetValue(spatialOffsets(values)),type.element().toString(),"Convert");
      }
      const auto& origin=input.instance(value).origin;
      if(!origin.coordinate_value)fail("scalar_fold_coordinate","execution coordinate is not bound");
      return result.graph.constant(type.element().toString(),std::to_string(*origin.coordinate_value));
    }
    if(dynamic_cast<const v::WarpMmaOp*>(&op))return warpMma(value,index);
    if(dynamic_cast<const v::TcgenMmaOp*>(&op) || dynamic_cast<const v::MatMulOp*>(&op))return mma(value,index);
    if(scalarOperation(op))return pure(value,index,type.element());
    fail("scalar_fold_operation_unsupported","scalar semantics unavailable for "+std::string(op.name()));
  }
  std::optional<PackedStorageDecoder::Fields> packedFields(v::Value value) const {
    // `known` depends on this projection's binding environment. Keep its
    // decoder local; view normalization owns a pass-wide context-free decoder.
    return PackedStorageDecoder(input,[this](auto x){return known(x);}).decode(value);
  }
  v::Value memberValue(v::Value value,std::size_t member) const {
    if(value.threadSelector()) {
      auto selector=*value.threadSelector();
      const auto map=input.threadBindingMap(selector,value);
      if(!map)fail("compact_view_unsupported","nested selector requires a receiver-domain member mapping");
      if(!map->fixed) {
        const auto at=__int128(map->base)+__int128(map->stride)*member;
        if(at<0 || at>SIZE_MAX)fail("scalar_fold_invalid_index","nested selector member is outside its domain");
        selector=memberValue(selector,static_cast<std::size_t>(at));
      }
      auto base=value.withoutSelection();
      if(!value.indices().empty())base=input.component(base,{value.indices().begin(),value.indices().end()});
      return input.fromThread(memberValue(base,member),selector);
    }
    if(!value.indices().empty())return input.component(memberValue(value.withoutSelection(),member),
        {value.indices().begin(),value.indices().end()});
    if(value.conversion()) {
      const auto& c=*value.conversion();auto source=memberValue(c.source,member);
      return c.spec->kind==v::TypeConversionKind::Bitcast?input.bitcast(source,c.spec->target):
          input.convert(source,c.spec->target,c.spec->attributes);
    }
    return input.node(value.nodeId()).result(value.resultIndex(),member);
  }
  s::Ref projectMemberTemplate(v::Value value,const Offset& offset,const v::ElementType& desired) {
    const auto original=indexed_context->use;
    if(auto found=expanded_member_uses.find(original.get());found!=expanded_member_uses.end())
      // Expanding a stored member binding is the implementation of THIS
      // request, not another dependency on the same value and byte range.
      // Keep one continuation/cache identity per request. The expanded use
      // is Conversion/Operands; only its actual operand queries enqueue work.
      return inUse(found->second,[&]{return projectImpl(value,offset,desired);});
    const auto map=*indexed_context->use->binding;
    const auto position=plus(Offset(map.base),times(indexed_context->index,map.stride));
    const auto [first,last]=bounds(position);
    const auto& node=input.definingNode(value);
    if(!value.indices().empty() && !value.threadSelector()) {
      auto use=bindSourceUse(input,value,value,map.base,map.stride);
      if(!use)fail("compact_view_unsupported","component requires independent producer and index mappings");
      expanded_member_uses.emplace(original.get(),use);
      return inUse(std::move(use),[&]{return projectImpl(value,offset,desired);});
    }
    if(node.isCollectiveOccurrence() && dynamic_cast<const v::WarpMmaOp*>(&node.operation()) &&
        !value.conversion() && !value.threadSelector() && !value.pointer()) {
      if(first<0 || last<first || std::uint64_t(last)>=node.instances().size())
        fail("scalar_fold_invalid_index","collective use exceeds its member domain");
      return inUse({},[&]{return project(value,offset,desired);});
    }
    // A selection is indexed in the receiver's domain, not the selected
    // producer's domain (a warp broadcast commonly selects a singleton).
    if(value.threadSelector()) {
      const auto selector=input.canonicalInterpretation(*value.threadSelector());
      const auto count=input.definingNode(selector).instances().size();
      if(first<0 || last<first || (count!=1 && std::uint64_t(last)>=count))
        fail("scalar_fold_invalid_index","selector member map exceeds its domain");
      auto [entry,inserted]=member_selectors.try_emplace(valueKey(selector));
      if(inserted) {
        // Validate existing finite selector bindings once. This does enumerate
        // the selector's members, but never source values, payloads or outputs.
        std::size_t begin=0;std::optional<std::int64_t> previous;
        for(std::size_t i=0;i<count;++i) {
          const auto current=inUse({},[&]{return known(selected(memberValue(selector,i)));});
          if(!current || *current<0) {entry->second.clear();break;}
          if(previous && *current!=*previous) {
            entry->second.emplace(begin,SelectorRun{i,*previous});begin=i;
          }
          previous=current;
          if(i+1==count)entry->second.emplace(begin,SelectorRun{count,*previous});
        }
      }
      const auto at=count==1?0:std::size_t(first);
      auto run=entry->second.upper_bound(at);
      if(run==entry->second.begin())fail("compact_view_unsupported","thread selector has no static member domain");
      --run;
      if(count!=1 && std::uint64_t(last)>=run->second.end) {
        // Selection facts already define disjoint index domains. Reuse the
        // same domain refinement as reads; do not project each receiver or
        // assume that distinct selected producers carry equal values.
        PartitionedRead selection{v::ValueType::matrix(bit(8),{count}),{},true};
        for(const auto& [begin,item]:entry->second) {
          if(integer(item.end)<=first || integer(begin)>last)continue;
          selection.regions.push_back({begin,item.end-begin,
              MemoryView{selection.type,MemoryViewKind::Slice,{},
                  {{0,0,begin,item.end-begin,{}}},{}}});
        }
        refinePartitionDomain(selection,position,bit(8));
        fail("compact_view_unsupported","thread selection crosses distinct producer domains");
      }
      auto source=value.withoutSelection();
      if(!value.indices().empty())source=input.component(source,{value.indices().begin(),value.indices().end()});
      const auto selected_value=inUse({},[&]{return selected(input.fromThread(source,memberValue(selector,at)));});
      return inUse({},[&]{return project(selected_value,offset,desired);});
    }
    if(value.conversion()) {
      auto child=memberUse(value.conversion()->source,map.base,map.stride);
      auto use=std::make_shared<IndexedSourceUse>(IndexedSourceUse{value,
          IndexedSourceUse::Kind::Conversion,{}, {std::move(child)}});
      expanded_member_uses.emplace(original.get(),use);
      return inUse(std::move(use),[&]{return projectImpl(value,offset,desired);});
    }
    if(first<0 || last<first || std::uint64_t(last)>=node.instances().size())
      fail("scalar_fold_invalid_index","source template member map exceeds its domain");
    if(first==last)return inUse({},[&]{return project(memberValue(value,std::size_t(first)),offset,desired);});
    // A pointer annotation identifies an allocation; it does not change this
    // scalar address computation or select a different spatial member.
    if(value.threadSelector())
      fail("compact_view_unsupported","template source requires an explicit selector mapping");
    const auto& op=node.operation();
    if(dynamic_cast<const v::CoordinateOp*>(&op)) {
      auto [entry,inserted]=member_coordinates.try_emplace(node.id());
      if(inserted) {
        // The launch binding is an existing finite table. Validate it once
        // per coordinate node, not once per projected output or ancestor.
        for(std::size_t i=0;i<node.instances().size();++i) {
          const auto x=node.instanceOrigin(i).coordinate_value;
          if(!x || *x>INT64_MAX){entry->second=v::ThreadMembers{};break;}
          entry->second.push_back(*x);
        }
      }
      const auto& coordinates=entry->second;
      const auto segment=coordinates.affineSegment(std::size_t(first));
      if(!segment)fail("compact_view_unsupported","coordinate member domain has no known values");
      if(std::uint64_t(last)>=segment->begin+segment->count) {
        PartitionedRead selection{v::ValueType::matrix(bit(8),{coordinates.size()}),{},true};
        for(std::size_t begin=std::size_t(first);begin<=std::size_t(last);) {
          const auto run=*coordinates.affineSegment(begin);
          selection.regions.push_back({run.begin,run.count,
              MemoryView{selection.type,MemoryViewKind::Slice,{},{{0,0,run.begin,run.count,{}}},{}}});
          begin=run.begin+run.count;
        }
        refinePartitionDomain(selection,position,bit(8));
        fail("compact_view_unsupported","coordinate query crosses affine domains");
      }
      if(bounds(offset)!=std::pair<std::int64_t,std::int64_t>{0,0} ||
          desired.width()!=input.type(value).element().width())
        fail("scalar_fold_invalid_index","coordinate projection requires the full scalar");
      return result.graph.convert(offsetValue(plus(Offset(integer(coordinates.at(segment->begin))),
          times(plus(position,Offset(-integer(segment->begin))),segment->stride))),desired.toString(),"Convert");
    }
    if(dynamic_cast<const v::LoadOp*>(&op)) {
      const auto [lo,hi]=bounds(offset);const auto width=desired.width()/8;
      if(lo<0 || hi<lo || !width)fail("scalar_fold_invalid_index","invalid template read slice");
      if(const auto* initial=initialByteMap(value.nodeId(),std::size_t(first),std::size_t(last))) {
        if(std::uint64_t(hi)<=initial->initial.bytes && width<=initial->initial.bytes-std::uint64_t(hi)) {
          auto delta=plus(position,Offset(-integer(initial->first)));
          auto relative=initial->period?
              plus(times(remainder(delta,initial->period),initial->stride),
                  times(quotient(delta,initial->period),initial->outer_stride)):
              times(delta,initial->stride);
          auto address=plus(Offset(integer(initial->initial.offset)),plus(offset,relative));
          return result.graph.initialCell(initial->initial.allocation,offsetValue(address),desired.toString());
        }
      }
      auto domain=symbolic_read_domains.upper_bound({value.nodeId(),std::size_t(first)});
      if(domain!=symbolic_read_domains.begin()) {
        --domain;
        const auto prototype=memory.symbolic_initial_reads.find({value.nodeId(),value.instanceIndex()});
        const auto region=memory.symbolic_initial_reads.find(domain->first);
        if(domain->first.node==value.nodeId() && std::size_t(last)<domain->second &&
            prototype!=memory.symbolic_initial_reads.end() && region!=memory.symbolic_initial_reads.end() &&
            prototype->second.allocation==region->second.allocation &&
            prototype->second.offset==region->second.offset && prototype->second.bytes==region->second.bytes &&
            node.hasUniformThreadTemplate()) {
          std::vector<std::shared_ptr<const IndexedSourceUse>> children;
          for(auto operand:input.instance(value).operands) {
            operand=input.canonicalInterpretation(operand);
            auto use=bindSourceUse(input,operand,value,map.base,map.stride);
            if(!use)fail("compact_view_unsupported","symbolic read address requires a cross-domain member mapping");
            children.push_back(std::move(use));
          }
          auto use=std::make_shared<IndexedSourceUse>(IndexedSourceUse{value,
              IndexedSourceUse::Kind::Operands,{},std::move(children)});
          // Coverage depends on this query's binder bounds. Unlike a pure
          // operand expansion, do not cache it as valid for every later domain.
          return inUse(std::move(use),[&]{return read(value,offset,desired);});
        }
      }
      const auto* slice=memory.readSlicePattern({value.nodeId(),std::size_t(first)},
          std::uint64_t(lo),std::uint64_t(hi-lo)+width);
      if(slice && std::uint64_t(last)-slice->first.instance<slice->count) {
        const auto delta=plus(position,Offset(-integer(slice->first.instance)));
        auto source_offset=plus(Offset(integer(slice->source_offset)-integer(slice->result_offset)),
            plus(offset,times(delta,slice->byte_stride)));
        const auto base=add(integer(slice->source.instanceIndex()),
            mul(map.base-integer(slice->first.instance),slice->source_instance_stride));
        auto use=memberUse(slice->source,base,mul(map.stride,slice->source_instance_stride));
        return inUse(std::move(use),[&]{return project(slice->source,source_offset,desired);});
      }
      if(const auto* periodic=periodicReadMap(value.nodeId(),std::size_t(first),std::size_t(last),
          std::uint64_t(lo),std::uint64_t(hi-lo)+width)) {
        const auto& p=*periodic->first;
        const auto delta=plus(position,Offset(-integer(p.first.instance)));
        const auto outer=quotient(delta,periodic->period),inner=remainder(delta,periodic->period);
        const auto source_offset=plus(offset,plus(Offset(integer(p.source_offset)-integer(p.result_offset)),
            plus(times(outer,periodic->byte_outer),times(inner,p.byte_stride))));
        const auto members=plus(Offset(integer(p.source.instanceIndex())),
            plus(times(outer,periodic->member_outer),times(inner,p.source_instance_stride)));
        return inContext(std::make_shared<IndexedContext>(IndexedContext{members,memberUse(p.source,0,1)}),
            [&]{return project(p.source,source_offset,desired);});
      }
      auto found=memory.initial_read_families.upper_bound({value.nodeId(),std::size_t(first)});
      if(found!=memory.initial_read_families.begin()) {
        const auto& family=std::prev(found)->second;
        if(family.first.node==value.nodeId() && std::uint64_t(first)>=family.first.instance &&
            std::uint64_t(last)-family.first.instance<family.count) {
          const auto initial=memory.initialRead({value.nodeId(),family.first.instance});
          if(initial && std::uint64_t(hi)<=initial->region.bytes && width<=initial->region.bytes-std::uint64_t(hi)) {
            auto address=plus(Offset(integer(initial->region.offset)),plus(offset,
                times(plus(position,Offset(-integer(family.first.instance))),family.byte_stride)));
            return result.graph.initialCell(initial->region.allocation,offsetValue(address),desired.toString());
          }
        }
      }
      // Existing read-from patterns may have distinct writers on adjacent
      // member ranges. Refine the output coordinate at those stored boundaries;
      // do not compare writer expressions or enumerate members to find them.
      SENTRYWARD_AUDIT((void)audit;
        static unsigned witnesses=0;
        if(sentryward::debug::environment("SENTRYWARD_READ_DOMAIN_WITNESS") && witnesses++<4) {
          SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_38);
          if(auto initial=initial_byte_maps.find(value.nodeId());initial!=initial_byte_maps.end()) {
            unsigned shown=0;
            for(const auto& map:initial->second) {
              if(shown++==12)break;
              SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_39);
            }
          }
          auto bucket=memory.read_slice_patterns.upper_bound({value.nodeId(),std::uint64_t(lo)});
          if(bucket!=memory.read_slice_patterns.begin()) {
            --bucket;
            if(bucket->first.first==value.nodeId()) {
              auto p=bucket->second.upper_bound(std::size_t(first));
              if(p!=bucket->second.begin())--p;
              for(unsigned shown=0;p!=bucket->second.end() && shown<12;++p,++shown) {
                const auto& x=p->second;
                SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_40);
              }
            }
          }
        });
      if(lo==hi) {
          PartitionedRead partition{v::ValueType::matrix(bit(8),{node.instances().size()}),{},true};
          const auto addDomain=[&](std::uint64_t begin,std::uint64_t count) {
            if(begin>std::uint64_t(last) || begin+count<=std::uint64_t(first))return;
            partition.regions.push_back({begin,count,MemoryView{partition.type,MemoryViewKind::Slice,{},
                {{0,0,begin,count,{}}},{}}});
          };
          // Initial-memory reads have the same member-domain contract as
          // writer-backed slices. Refine at stored family boundaries; never
          // enumerate members or extrapolate one family's address stride.
          auto initial=memory.initial_read_families.upper_bound({value.nodeId(),std::size_t(first)});
          if(initial!=memory.initial_read_families.begin())--initial;
          for(;initial!=memory.initial_read_families.end();++initial) {
            const auto& p=initial->second;
            if(p.first.node<value.nodeId())continue;
            if(p.first.node!=value.nodeId() || p.first.instance>std::uint64_t(last))break;
            if(p.first.instance+p.count<=std::uint64_t(first))continue;
            addDomain(p.first.instance,p.count);
          }
          auto domain=memory.read_slice_patterns.upper_bound({value.nodeId(),std::uint64_t(lo)});
          if(domain!=memory.read_slice_patterns.begin()) {
            --domain;
            if(domain->first.first==value.nodeId())for(const auto& [start,p]:domain->second) {
              (void)start;
              if(std::uint64_t(lo)<p.result_offset || std::uint64_t(lo)-p.result_offset>p.bytes ||
                  width>p.bytes-(std::uint64_t(lo)-p.result_offset))continue;
              addDomain(p.first.instance,p.count);
            }
          }
          refinePartitionDomain(partition,position,bit(8));
      }
      SENTRYWARD_AUDIT((void)audit;SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_41));
      fail("compact_view_unsupported","template read crosses read-from member or payload domains");
    }
    if(dynamic_cast<const v::ConstantOp*>(&op) || dynamic_cast<const v::InputParameterOp*>(&op))
      return inUse({},[&]{return project(value,offset,desired);});
    if(!scalarOperation(op) || !node.hasUniformThreadTemplate())
      fail("compact_view_unsupported","operation requires an explicit template member mapping");
    std::vector<std::shared_ptr<const IndexedSourceUse>> children;
    // One stored operand list. Graph owns the SameDomain/Fixed distinction;
    // do not enumerate members to rediscover it or search operand ancestry.
    for(auto operand:input.instance(value).operands) {
      operand=input.canonicalInterpretation(operand);
      auto use=bindSourceUse(input,operand,value,map.base,map.stride);
      if(use)children.push_back(std::move(use));
      else {
        const auto& producer=input.definingNode(operand);
        const auto& pm=producer.instanceThreads();const auto& cm=node.instanceThreads();
        fail("compact_view_unsupported","operand requires an explicit cross-domain member mapping: producer="+
            std::to_string(operand.nodeId())+" members="+std::to_string(pm.size())+
            " step="+(pm.affineStep()?std::to_string(*pm.affineStep()):"nonaffine")+
            " consumer="+std::to_string(value.nodeId())+" members="+std::to_string(cm.size())+
            " step="+(cm.affineStep()?std::to_string(*cm.affineStep()):"nonaffine")+
            " annotations="+std::to_string(bool(operand.conversion()))+","+
            std::to_string(bool(operand.threadSelector()))+","+std::to_string(operand.indices().size()));
      }
    }
    auto use=std::make_shared<IndexedSourceUse>(IndexedSourceUse{value,
        IndexedSourceUse::Kind::Operands,{},std::move(children)});
    expanded_member_uses.emplace(original.get(),use);
    return inUse(std::move(use),[&]{return projectImpl(value,offset,desired);});
  }
  // Decode an already projected integer index into the shared range algebra.
  // One worklist over this index DAG, not over matrix/math ancestors or member
  // values. Every arithmetic/conversion step must preserve its exact value on
  // the current domain; bit operations are accepted only with proven bounds.
  Offset componentOffset(s::Ref root) {
    std::unordered_map<s::Ref,Offset,s::RefHash> done;
    struct Frame {s::Ref ref;std::size_t child=0;};
    std::vector<Frame> work{{root}};
    const auto fits=[&](const Offset& x,const std::string& type) {
      if(type.size()<2 || std::string("ubs").find(type[0])==std::string::npos)return false;
      unsigned width=0;const auto [end,error]=std::from_chars(type.data()+1,type.data()+type.size(),width);
      if(error!=std::errc{} || end!=type.data()+type.size() || !width || width>64)return false;
      const auto [lo,hi]=bounds(x);
      const __int128 low=type[0]=='s'?-(__int128(1)<<(width-1)):0;
      const __int128 high=type[0]=='s'?(__int128(1)<<(width-1))-1:(__int128(1)<<width)-1;
      return lo>=low && hi<=high;
    };
    while(!work.empty()) {
      auto& frame=work.back();const auto& n=result.graph.node(frame.ref);
      if(n.kind!=s::Kind::Constant && n.kind!=s::Kind::Index &&
          !(n.kind==s::Kind::Operation && n.operands.size()==2 &&
            (n.name=="Add" || n.name=="Subtract" || n.name=="Multiply" ||
             n.name=="Divide" || n.name=="Remainder" || n.name=="ShiftLeft" ||
             n.name=="ShiftRight" || n.name=="BitAnd")))
        fail("compact_view_unsupported","component index requires an unsupported coordinate operation: "+n.name);
      if(frame.child<n.operands.size()) {
        auto child=n.operands[frame.child++];
        if(!done.contains(child))work.push_back({std::move(child)});
        continue;
      }
      std::optional<Offset> value;
      if(n.kind==s::Kind::Constant) {if(auto x=literal(n.name))value=Offset(*x);}
      else if(n.kind==s::Kind::Index && n.index_level<domains.size()) {
        value=Offset{};value->terms[n.index_level]=1;
      } else if(n.kind==s::Kind::Operation && n.operands.size()==2) {
        const auto& a=done.at(n.operands[0]);const auto& b=done.at(n.operands[1]);
        const auto [al,ah]=bounds(a);const auto [bl,bh]=bounds(b);
        if(n.name=="Add")value=plus(a,b);
        else if(n.name=="Subtract")value=plus(a,times(b,-1));
        else if(n.name=="Multiply") {
          if(bl==bh)value=times(a,bl);else if(al==ah)value=times(b,al);
        } else if(bl==bh && bl>0 && al>=0 && n.name=="Divide")value=quotient(a,std::uint64_t(bl));
        else if(bl==bh && bl>0 && al>=0 && n.name=="Remainder")value=remainder(a,std::uint64_t(bl));
        else if(bl==bh && bl>=0 && bl<63 && al>=0 && n.name=="ShiftLeft")value=times(a,std::int64_t{1}<<bl);
        else if(bl==bh && bl>=0 && bl<63 && al>=0 && n.name=="ShiftRight")value=quotient(a,std::uint64_t{1}<<bl);
        else if(bl==bh && bl>=0 && bl<INT64_MAX && al>=0 && n.name=="BitAnd" &&
            std::has_single_bit(std::uint64_t(bl)+1))value=remainder(a,std::uint64_t(bl)+1);
      }
      if(!value || !fits(*value,n.type))
        fail("compact_view_unsupported","component index is outside exact affine/radix range algebra: "+n.name);
      // Validate each conversion edge, including intermediate narrowing, not
      // merely the final type. Never erase an FP or saturating conversion.
      std::string_view steps=frame.ref.conversion;
      while(!steps.empty()) {
        const auto split=steps.find("; ");const auto step=steps.substr(0,split);
        const auto open=step.find('('),arrow=step.find(" -> ");
        if(open==std::string_view::npos || arrow==std::string_view::npos || step.back()!=')' ||
            (step.substr(0,open)!="Convert" && step.substr(0,open)!="Bitcast") ||
            !fits(*value,std::string(step.substr(open+1,arrow-open-1))) ||
            !fits(*value,std::string(step.substr(arrow+4,step.size()-arrow-5))))
          fail("compact_view_unsupported","component index conversion is not range-preserving");
        if(split==std::string_view::npos)break;
        steps.remove_prefix(split+2);
      }
      done.emplace(frame.ref,std::move(*value));work.pop_back();
    }
    return done.at(root);
  }
  s::Ref projectImpl(v::Value value,Offset offset,const v::ElementType& desired){
    if(indexed_context && indexed_context->use) {
      const auto use=indexed_context->use;
      if(use->prototype!=input.canonicalInterpretation(value))
        fail("scalar_fold_prerequisite","indexed projection lost its use-position binding");
      if(use->kind==IndexedSourceUse::Kind::MemberTemplate)
        return projectMemberTemplate(value,offset,desired);
      if(use->kind==IndexedSourceUse::Kind::Component) {
        const auto matrix=use->children.at(0);
        const auto& shape=input.type(matrix->prototype).shape();
        if(use->children.size()!=shape.size()+1)
          fail("scalar_fold_component_rank","component use has inconsistent index mappings");
        Offset element;
        for(std::size_t axis=0;axis<shape.size();++axis) {
          const auto child=use->children[axis+1];
          auto coordinate=componentOffset(inUse(child,[&]{return scalar(child->prototype);}));
          const auto [lo,hi]=bounds(coordinate);
          if(lo<0 || hi<lo || std::uint64_t(hi)>=shape[axis])
            fail("scalar_fold_component_index","mapped component index exceeds matrix shape");
          element=plus(times(element,integer(shape[axis])),coordinate);
        }
        const auto width=input.type(matrix->prototype).element().width();
        if(!width || width%8)fail("scalar_fold_subbyte_value","component payload must be byte aligned");
        return inUse(matrix,[&]{return project(matrix->prototype,
            plus(offset,times(element,integer(width/8))),desired);});
      }
      if(use->kind==IndexedSourceUse::Kind::Exact || use->kind==IndexedSourceUse::Kind::OneHotLiteral)
        return inUse({},[&]{return project(value,offset,desired);});
      if(use->kind==IndexedSourceUse::Kind::Forward || use->kind==IndexedSourceUse::Kind::Selected)
        return inUse(use->children.at(0),[&]{return project(use->children.at(0)->prototype,offset,desired);});
      if(use->kind==IndexedSourceUse::Kind::NonzeroSelect || use->kind==IndexedSourceUse::Kind::BitTestSelect) {
        const auto& type=input.type(value);
        if(bounds(offset)!=std::pair<std::int64_t,std::int64_t>{0,0} ||
           !type.isScalar() || type.element().width()!=desired.width())
          fail("scalar_fold_conversion_projection","conditional source requires its complete scalar");
        const auto child=[&](std::size_t i) {
          const auto source=use->children.at(i);
          return inUse(source,[&]{return scalar(source->prototype);});
        };
        auto test=child(0),yes=child(1),no=child(2);
        auto test_type=input.type(use->children.at(0)->prototype).element().toString();
        if(use->kind==IndexedSourceUse::Kind::BitTestSelect) {
          const auto& map=*use->binding;
          auto bit=plus(Offset(map.base),plus(times(remainder(indexed_context->index,map.period),map.stride),
              times(quotient(indexed_context->index,map.period),map.outer_stride)));
          const auto [first,last]=bounds(bit);
          const auto width=input.type(use->children.at(0)->prototype).element().width();
          if(first<0 || last>=width)fail("scalar_fold_invalid_index","bit-test index exceeds mask width");
          test=result.graph.convert(test,"b"+std::to_string(width),"Bitcast");
          test=emitOperation("ExtractBits","u32",{test,offsetValue(bit)},"width=1");
          test_type="u32";
        }
        auto predicate=emitOperation("Compare","pred",
            {test,result.graph.constant(test_type,"0")},
            attributes(v::CompareOp(v::ComparePredicate::NotEqual,
                v::ValueType::scalar(v::ElementType::predicate()))));
        return cast(emitOperation("Select",type.element().toString(),{predicate,yes,no}),desired);
      }
    }
      if(const auto* found=indexedBinding(value)) {
        const auto& binding=*found;
        if(binding.fixed) {
          return inUse({},[&]{return project(value,offset,desired);});
        }
        auto position=mappedCoordinate(binding,indexed_context->index);
        const auto& t=input.type(value);
        const auto [lo,hi]=bounds(offset);
        if(lo<0 || hi<lo || std::uint64_t(hi)>t.bitSize()/8 || desired.width()/8>t.bitSize()/8-std::uint64_t(hi))
          fail("compact_view_unsupported","indexed source projection exceeds its value");
        if(binding.numerical_field && (lo!=0 || hi!=0 || desired!=t.element()))
          fail("scalar_fold_conversion_projection","decoded floating field requires its complete numerical scalar");
        const auto field=binding.numerical_field.value_or(desired);
        if(binding.initial) {
          auto r=result.graph.initialCell(binding.initial->region.allocation,
              offsetValue(plus(position,offset)),field.toString());
          return binding.numerical_field?result.graph.convert(r,desired.toString(),"NumericInterpretation"):r;
        }
        if(binding.producer) {
          // The affine offset belongs to this indexed use, but the shared
          // producer retains its own leaves and occurrence identity.
          auto r=inContext(binding.producer_uses_bindings?indexed_context:nullptr,
              [&]{return project(*binding.producer,plus(position,offset),field);});
          return binding.numerical_field?result.graph.convert(r,desired.toString(),"NumericInterpretation"):r;
        }
        auto scalar=result.graph.convert(offsetValue(position),t.element().toString(),"Convert");
        if(binding.power_of_two) {
          const auto [first,last]=bounds(position);
          if(first<0 || last>=t.element().width() || binding.initial || binding.producer)
            fail("scalar_fold_invalid_index","one-hot scalar bit index exceeds its exact width");
          scalar=emitOperation("ShiftLeft",t.element().toString(),
              {result.graph.constant(t.element().toString(),"1"),scalar});
        }
        if(lo==0 && hi==0 && desired.width()==t.element().width())return cast(scalar,desired);
        return emitOperation("ExtractBits",desired.toString(),{scalar,offsetValue(times(offset,8))},
            "width="+std::to_string(desired.width()));
      }
    if(spatial_count)prepareSpatial(value);
    // A conversion use already binds its source (including any cross-thread
    // selection). Resolve that child under its own map below. Evaluating the
    // selector under the conversion's map confuses two different use sites.
    const bool mapped_conversion=value.conversion() && indexed_context && indexed_context->use &&
        indexed_context->use->kind==IndexedSourceUse::Kind::Conversion;
    if(!mapped_conversion)value=selected(value);
    const auto& type=input.type(value);
    if(type.element().isParameterized() || !type.hasStaticShape())
      fail("scalar_fold_parameterized_type","value type or shape depends on an unresolved descriptor");
    if(!value.conversion() && type.isScalar() && type.element().kind()==v::ElementKind::Predicate &&
       desired.kind()==v::ElementKind::Predicate && !offset.expression && offset.base==0 && offset.terms.empty()){
      return whole(value,Offset{});
    }
    const bool subbyte = !desired.width() || desired.width()%8 ||
        !type.element().width() || type.element().width()%8;
    if(subbyte) {
      // A complete numeric scalar (e.g. FP4 conversion or its u4 bitcast)
      // has no byte index. Only storage projection requires byte alignment.
      // Never route a Load through whole(): memory still requires whole bytes.
      const auto [lo,hi]=bounds(offset);
      if(!type.isScalar() || !value.indices().empty() || lo!=0 || hi!=0 ||
          !desired.width() || desired.width()!=type.element().width() ||
          dynamic_cast<const v::LoadOp*>(&input.definingNode(value).operation()))
        fail("scalar_fold_subbyte_value","projection requires byte-aligned scalar elements");
      if(!value.conversion())return cast(whole(value,Offset{}),desired);
      // Conversion handling below retains its exact conversion attributes and
      // recursively requests the entire source scalar, not a byte slice.
    }
    if(!indexed_context && type.isScalar() && desired.width()<type.element().width()) {
      const auto [lo,hi]=bounds(offset);
      if(lo>=0 && lo==hi && std::uint64_t(lo)<type.element().width()/8)
        if(auto fields=packedFields(value);fields && fields->size()>1)
          for(const auto& field:*fields)
            if(field.start==std::uint64_t(lo)*8 && field.width==desired.width())
              return project(field.value,Offset{},desired);
    }
    if(value.conversion()){
      if(!value.indices().empty())fail("scalar_fold_conversion_projection","component selection after conversion is not lowered");
      if(!type.isScalar())fail("scalar_fold_conversion_projection","matrix conversion requires per-element rules");
      const auto& c=*value.conversion();
      const auto child=indexed_context && indexed_context->use?indexed_context->use->children.at(0):nullptr;
      auto base=inUse(child,[&]{return scalar(child?child->prototype:c.source);});
      std::string step=c.spec->kind==v::TypeConversionKind::Bitcast?"Bitcast":"Convert";
      if(c.spec->kind==v::TypeConversionKind::Convert)step+=" "+attributes(v::ConvertOp(c.spec->target,c.spec->attributes));
      auto converted=result.graph.convert(base,type.element().toString(),step);
      auto [lo,hi]=bounds(offset);
      if(lo==0 && hi==0 && desired.width()==type.element().width())return cast(converted,desired);
      if(lo<0 || hi<0 || add(hi,desired.width()/8)>integer(type.bitSize()/8))
        fail("scalar_fold_invalid_index","converted scalar slice exceeds value");
      return emitOperation("ExtractBits",desired.toString(),{converted,offsetValue(times(offset,8))},
        "width="+std::to_string(desired.width()));
    }
    if(!value.indices().empty()){
      auto base=value.withoutSelection();const auto& shape=input.type(base).shape();
      if(shape.size()!=value.indices().size())fail("scalar_fold_component_rank","component index rank mismatch");
      Offset index;
      for(std::size_t j=0;j<shape.size();++j){
        auto k=known(value.indices()[j]);
        if(!k || *k<0 || std::uint64_t(*k)>=shape[j])fail("scalar_fold_component_index","component index is not a fixed valid coordinate");
        index=plus(times(index,integer(shape[j])),Offset(*k));
      }
      return project(base,plus(times(index,integer(type.element().width()/8)),offset),desired);
    }
    const auto& op=input.definingNode(value).operation();
    if(dynamic_cast<const v::LoadOp*>(&op))return read(value,offset,desired);
    if(const auto* copy=dynamic_cast<const v::NativeMatrixOp*>(&op);
        copy && copy->opcode()==v::NativeMatrixOpcode::TcgenCopyValue && type.hasStaticShape()) {
      const auto source=input.instance(value).operands.at(0);
      if(input.type(source)==type)return project(source,offset,desired);
    }
    const auto width=type.element().width()/8;
    auto [lo,hi]=bounds(offset);
    if(lo<0 || hi<0 || std::uint64_t(hi)>type.bitSize()/8 || desired.width()/8>type.bitSize()/8-std::uint64_t(hi))
      fail("scalar_fold_invalid_index","scalar projection exceeds producer value");
    auto index=quotient(offset,width),inner=remainder(offset,width);
    auto [il,ih]=bounds(inner);
    if(il<0 || ih<0 || std::uint64_t(ih)+desired.width()/8>width)
      fail("scalar_fold_cross_element","scalar projection crosses producer element boundaries");
    auto result=whole(value,index);
    if(il==0 && ih==0 && desired.width()==type.element().width())return cast(result,desired);
    return this->emitOperation("ExtractBits",desired.toString(),{result,offsetValue(times(inner,8))},
      "width="+std::to_string(desired.width()));
  }
  s::Ref evaluateProjection(Projection& task) {
    SENTRYWARD_LEDGER(ledger->evaluate(task.key));
    // A task owns its direct requested results, not every request made by
    // inlined descendants. A completed child's value already retains its
    // semantic operands. Rooting the child's temporary construction history
    // prevents Fold replacement from releasing concrete intermediate terms.
    struct RestoreRequests {
      std::unordered_set<std::string>*& target;
      std::unordered_set<std::string>* saved;
      ~RestoreRequests(){target=saved;}
    } restore_requests{requested,requested};
    // Explicitly suspended child results remain roots until this task has
    // completed. Clearing them on retry can repeatedly collect the very child
    // that the worklist just finished before an inline parent consumes it.
    requested=&task.dependencies;
    struct RestoreTask {
      const Projection*& current;const Projection* saved;
      ~RestoreTask(){current=saved;}
    } restore_task{active_projection,active_projection};
    active_projection=&task;
    struct RestoreFold {
      Lower& owner;std::shared_ptr<const FoldOwner> saved;
      ~RestoreFold(){owner.fold_owner=std::move(saved);}
    } restore_fold{*this,fold_owner};
    fold_owner=task.fold;
    SENTRYWARD_AUDIT(audit->add(AuditCounter::projection_evaluations);
      audit->maximum(AuditCounter::max_projection_depth,projection_depth));
    struct Restore {
      std::shared_ptr<const IndexedContext>& target;
      std::shared_ptr<const IndexedContext> saved;
      ~Restore(){target=std::move(saved);}
    } restore{indexed_context,indexed_context};
    struct RestoreKey {
      const std::string*& target;
      const std::string* saved;
      ~RestoreKey(){target=saved;}
    } restore_key{projection_key,projection_key};
    projection_key=&task.key;
    indexed_context=task.indexed;
    current=MemoryInstance{task.value.nodeId(),task.value.instanceIndex()};
    limit();
    auto r=projectImpl(task.value,task.offset,task.type);
    const auto& op=input.definingNode(task.value).operation();
    const auto* pointer=task.value.pointer().get();
    if(!pointer) {
      if(const auto* c=dynamic_cast<const v::ConstantOp*>(&op);c && c->pointer())pointer=&*c->pointer();
      else if(const auto* p=dynamic_cast<const v::InputParameterOp*>(&op);p && p->pointer())pointer=&*p->pointer();
    }
    if(pointer)r=result.graph.withPointer(std::move(r),pointerLabel(*pointer));
    limit();
    cache.emplace(task.key,r);
    SENTRYWARD_LEDGER(ledger->complete(task.key));
    finishReduction(task.key);
    return r;
  }
  s::Ref project(v::Value value,Offset offset,const v::ElementType& type){
    SENTRYWARD_AUDIT(audit->add(AuditCounter::projection_requests));
    offset=specializeSingletons(std::move(offset));
    value=input.canonicalInterpretation(value);
    struct RestoreContext {
      std::shared_ptr<const IndexedContext>& target;
      std::shared_ptr<const IndexedContext> saved;
      ~RestoreContext(){target=std::move(saved);}
    } restore_context{indexed_context,indexed_context};
    // Coordinate maps change the query, not the value. Compose them before
    // cache lookup/worklist insertion; equivalent mapped uses share one task.
    while(indexed_context && indexed_context->use &&
          indexed_context->use->kind==IndexedSourceUse::Kind::CoordinateMap) {
      const auto& use=indexed_context->use;
      if(use->prototype!=value)
        fail("scalar_fold_prerequisite","indexed projection lost its use-position binding");
      const auto& map=*use->binding;const auto child=use->children.at(0);
      auto coordinate=mappedCoordinate(map,indexed_context->index);
      indexed_context=std::make_shared<IndexedContext>(IndexedContext{std::move(coordinate),child});
      value=child->prototype;
    }
    // Indexed/spatial prototypes must use the already-validated family map,
    // not the concrete predicate of its first member. Outside those scopes,
    // fixed input facts can select an operand before constructing either arm.
    if(!indexed_context && !spatial_count)value=specialization.selected(value);
    bool shared_view=false;
    if(!indexed_context && !spatial_count && !offset.expression && offset.terms.empty() &&
        offset.base>=0 && type.width() && type.width()%8==0) {
      if(auto field=pure_source_templates.storageSlice(value,std::uint64_t(offset.base),type.width()/8)) {
        // The direct storage-slice path already canonicalizes a normalized
        // read to its source. Count a subsequent cache hit as view reuse too;
        // do not require the old Load spelling to survive that optimization.
        if(dynamic_cast<const v::LoadOp*>(&input.definingNode(value).operation()))
          shared_view=memory.normalizedRead({value.nodeId(),value.instanceIndex()}).has_value();
        value=field->value;offset=Offset(integer(field->offset));
      }
    }
    auto identity=valueKey(value);
    if(!value.pointer() && !value.conversion() && value.indices().empty() && !value.threadSelector() &&
       dynamic_cast<const v::LoadOp*>(&input.definingNode(value).operation())) {
      auto read=memory.canonicalRead({value.nodeId(),value.instanceIndex()});
      if(const auto alias=memory.normalized_read_aliases.find(read);alias!=memory.normalized_read_aliases.end())
        read=alias->second;
      if(const auto canonical=memory.normalized_reads.find(read);
          canonical!=memory.normalized_reads.end()) {
        identity="view:"+std::to_string(read.node)+":"+std::to_string(read.instance);
        shared_view=true;
      }
    }
    auto key=(spatial_count?"spatial:"+std::to_string(spatial_serial)+":":"")+identity+"@"+offset.key()+"/"+type.toString()+"#";
    auto used=offset.binderMask();
    if(indexed_context) {
      used|=indexed_context->index.binderMask();
      key+="indexed:"+indexed_use_identity.context(indexed_context->use,indexed_context->index.key(),{});
    }
    // A projection depends on its selected Value/use and index bindings, not
    // the extents of unrelated enclosing Folds. Keep binder positions/rank
    // (inner Fold levels rely on them), and all used extents. Spatial legacy
    // bindings retain their full environment. No operand ancestors are read.
    for(std::size_t i=0;i<domains.size();++i)
      key+=(spatial_count || i>=64 || (used&(std::uint64_t{1}<<i)))?
          std::to_string(domains[i])+",":"_,";
    SENTRYWARD_LEDGER(ledger->request(key,projection_key,std::string(input.definingNode(value).operation().name())+
        (indexed_context?"/indexed":"/plain"));
      if(ledger->checkpoint())ledger->snapshot(result.graph,cache.size(),result.outputs.size()));
    SENTRYWARD_AUDIT(if(requested) {
      audit->add(AuditCounter::dependency_requests);
    });
    // Retain cache hits too: an unfinished parent needs these roots across GC.
    // Expected O(key length), rather than scanning every prior dependency.
    if(requested)requested->insert(key);
    if(auto i=cache.find(key);i!=cache.end()){
      SENTRYWARD_LEDGER(ledger->hit(key));
      SENTRYWARD_AUDIT(audit->add(AuditCounter::projection_cache_hits));
      if(shared_view)++result.shared_view_projection_hits;
      return i->second;
    }
    if(auto i=failed_projections.find(key);i!=failed_projections.end()) {
      current=i->second.second;throw i->second.first;
    }
    Projection request{value,std::move(offset),type,domains,key};
    request.indexed=indexed_context;
    request.fold=fold_owner;
    if(projecting) {
      if(projection_depth>=options.max_pending_projections)
        fail("projection_worklist_limit_exceeded","projection worklist budget exceeded");
      if(inline_projection_depth>=32) {
        SENTRYWARD_LEDGER(ledger->add("dependency_suspensions"));
        SENTRYWARD_AUDIT(audit->event("DEPENDENCY_SUSPEND","projection_exception_worklist",
          projection_depth,inline_projection_depth,value.nodeId()));
        throw request;
      }
      if(!visiting.insert(request.key).second)fail("scalar_fold_cycle","cyclic value or read substitution; requested="+
          request.key+"; parent="+(active_projection?active_projection->key:"none"));
      try {
      struct InlineFrame {
        Lower& owner;
        const std::string& key;
        std::vector<std::uint64_t> saved_domains;
        std::optional<MemoryInstance> saved_current;
        bool completed=false;
        ~InlineFrame() {
          --owner.projection_depth;--owner.inline_projection_depth;
          owner.visiting.erase(key);owner.domains=std::move(saved_domains);
          if(completed)owner.current=saved_current;
        }
      } frame{*this,request.key,domains,current};
      ++projection_depth;++inline_projection_depth;
      // A dependency suspension is handled once by the outer worklist. Do
      // not catch/rethrow it at every inline ancestor: that restarts exception
      // search repeatedly along the same path. Cleanup alone restores the
      // caller's domain, while failures retain their diagnostic location.
      auto r=evaluateProjection(request);
      frame.completed=true;
      return r;
      } catch(const Projection&) {
        // InlineFrame has already restored context and removed its visiting
        // key. Move, rather than copy, the accumulated dependency set.
        suspended_projections.push_back(std::move(request));
        throw;
      }
    }
    const auto outer_domains=domains;
    projecting=true;
    std::vector<Projection> work;
    std::optional<std::pair<std::string,std::size_t>> pressure_retry;
    work.push_back(std::move(request));
    visiting.insert(key);
    try {
      while(!work.empty()) {
        auto& task=work.back();
        SENTRYWARD_AUDIT(if(task.audit_attempts++) {
          audit->add(AuditCounter::parent_restarts);
          audit->event("DEPENDENCY_REWALK","parent_restarts_after_dependency",
            task.audit_attempts,task.dependencies.size(),task.value.nodeId());
        });
        requested=&task.dependencies;
        domains=task.domains;
        projection_depth=work.size();
        try {
          evaluateProjection(task);
          pressure_retry.reset();
          visiting.erase(task.key);
          work.pop_back();
          requested=nullptr;
          if(options.construction_collection_interval && result.graph.size()>=next_collection)
            collectFrontier(work,key);
        } catch(const CollectProjectionFrontier&) {
          SENTRYWARD_LEDGER(ledger->add("node_pressure_retries"));
          requested=nullptr;
          collectFrontier(work,key);
          const auto live=result.graph.size();
          // A task that cannot advance after reclaiming the same frontier
          // cannot be retried indefinitely under this construction budget.
          if(live>=options.max_nodes || (pressure_retry &&
              pressure_retry->first==task.key && live>=pressure_retry->second))
            fail("expression_node_limit_exceeded","scalar graph node limit exceeded");
          pressure_retry=std::pair{task.key,live};
        } catch(const RetryFoldDomain& retry) {
          SENTRYWARD_LEDGER(ledger->add("fold_domain_retries"));
          // Only the current, uncommitted Fold piece changed. Accumulators
          // retain completed pieces; cached completed projections remain valid.
          const std::unordered_set<std::string> keep(retry.keep.begin(),retry.keep.end());
          for(auto p=reduction_progress.begin();p!=reduction_progress.end();) {
            if(keep.contains(p->first)){p->second.pending.clear();++p;continue;}
            const auto abandoned=p->first;++p;finishReduction(abandoned);
          }
          if(!refinement_resume)throw std::logic_error("missing Fold continuation");
          auto resume=std::move(*refinement_resume);refinement_resume.reset();
          // Retain the enclosing explicit tasks. An inline owner gets its own
          // task here, rather than re-entering the entire output expression
          // for every discovered source boundary.
          std::size_t prefix=1;
          for(std::size_t i=0;i<work.size();++i)
            if(keep.contains(work[i].key))prefix=i+1;
          work.erase(work.begin()+prefix,work.end());visiting.clear();
          for(auto& task:work)visiting.insert(task.key);
          if(work.back().key!=resume.key) {
            work.back().dependencies.insert(resume.key);
            visiting.insert(resume.key);work.push_back(std::move(resume));
          }
          requested=nullptr;
        } catch(Projection& child) {
          requested=nullptr;
          const auto enqueue=[&](Projection&& next) {
            work.back().dependencies.insert(next.key);
            if(!visiting.insert(next.key).second)fail("scalar_fold_cycle","cyclic value or read substitution");
            if(work.size()>=options.max_pending_projections)
              fail("projection_worklist_limit_exceeded","projection worklist budget exceeded");
            work.push_back(std::move(next));
          };
          // Parent-before-child stack order. Each suspended frame is retried
          // only after its direct child completes; it no longer reconstructs
          // every discarded inline ancestor. These records contain no scalar
          // graph refs, and collectFrontier roots/remaps their cached results.
          for(auto i=suspended_projections.rbegin();i!=suspended_projections.rend();++i)
            enqueue(std::move(*i));
          suspended_projections.clear();
          enqueue(std::move(child));
        } catch(const Failure& error) {
          if(error.code!="scalar_fold_non_affine_projection")throw;
          failed_projections.insert_or_assign(task.key,std::make_pair(error,current));
          visiting.erase(task.key);work.pop_back();requested=nullptr;
          if(work.empty())throw;
        }
      }
      domains=outer_domains;
      requested=nullptr;
      projecting=false;
      discardReductions();
      return cache.at(key);
    } catch(...) {
      suspended_projections.clear();
      // Output-domain refinement intentionally unwinds this driver and later
      // retries another domain. The explicit work frames are being destroyed,
      // so none of their keys remains an active ancestor. Inline frame keys
      // have already been removed by InlineFrame's destructor. Remove only
      // the keys owned by this driver; ordinary in-driver cycle checks stay.
      for(const auto& task:work)visiting.erase(task.key);
      domains=outer_domains;
      requested=nullptr;
      projecting=false;
      discardReductions();
      throw;
    }
  }
  void scheduleCollection() {
    const auto cap=std::max<std::size_t>(1,options.max_nodes/4);
    const auto interval=std::min(cap,std::max(options.construction_collection_interval,
                                            result.graph.size()/2));
    next_collection=std::min(options.max_nodes,result.graph.size()+interval);
  }
  void collectFrontier(const std::vector<Projection>& work,const std::string& root) {
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
    auto measurement=ledger?std::make_unique<debug::ProjectionLedger::Collection>(*ledger,result.graph,cache.size()):nullptr;
#endif
    struct CollectionScope {bool& flag;bool saved;~CollectionScope(){flag=saved;}}
      scope{collecting,collecting};
    collecting=true;
    SENTRYWARD_AUDIT(audit->add(AuditCounter::gc_calls);
      audit->add(AuditCounter::gc_input_nodes,result.graph.size());
      audit->event("GRAPH_COLLECTION","projection_frontier_collection",result.graph.size(),cache.size()));
    std::unordered_set<std::string> needed{root};
    for(const auto& task:work)
      needed.insert(task.dependencies.begin(),task.dependencies.end());
    for(const auto& [key,progress]:reduction_progress) {
      (void)key;
      needed.insert(progress.accumulator_key);
      needed.insert(progress.pending.begin(),progress.pending.end());
    }
    std::vector<std::string> keys;
    std::vector<s::Ref> roots;
    for(const auto& output:result.outputs)roots.push_back(output.value);
    std::unordered_set<s::Id> live;
    std::vector<s::Id> pending;
    for(const auto& value:roots)pending.push_back(value.node);
    const auto trace=[&] {
      while(!pending.empty()) {
        const auto id=pending.back();pending.pop_back();
        if(!live.insert(id).second)continue;
        for(const auto& child:result.graph.node(id).operands)pending.push_back(child.node);
      }
    };
    trace();
    SENTRYWARD_AUDIT(audit->event("LIVE_GRAPH","completed_output_nodes",live.size(),result.outputs.size());
      if(sentryward::debug::environment("SENTRYWARD_OUTPUT_DOMAIN_WITNESS") && !live.empty()) {
        std::map<std::string,std::size_t> kinds,shapes;
        for(auto id:live) {
          const auto& n=result.graph.node(id);
          ++kinds[std::to_string(int(n.kind))+":"+(n.kind==s::Kind::Operation?n.name:"")];
        }
        for(const auto& [i,d]:output_domains) {
          (void)i;std::string shape;
          for(auto size:d.shape)shape+=std::to_string(size)+",";
          ++shapes[shape];
        }
        SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_42);
        for(const auto& [kind,count]:kinds)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_43);
        SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_44);for(const auto& [shape,count]:shapes)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_45);
        SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_46);
      });
    [[maybe_unused]] const auto output_live=live.size();
    for(const auto& key:needed)
      if(const auto entry=cache.find(key);entry!=cache.end())pending.push_back(entry->second.node);
    // Completed projections are memoized computations, including results
    // whose *first* reuse is in a later output. Keep the existing cache while
    // it fits within graph headroom; a hit-only policy loses those results
    // before their first reuse. At pressure retain only actual dependencies.
    const auto reserve=std::max<std::size_t>(1,options.max_nodes/4);
    if(result.graph.size()<options.max_nodes-reserve) {
      for(const auto& [key,value]:cache)pending.push_back(value.node);
      SENTRYWARD_LEDGER(ledger->add("headroom_projection_roots",cache.size()));
    } else {
      SENTRYWARD_LEDGER(ledger->add("pressure_weak_cache_collections"));
    }
    trace();
    SENTRYWARD_AUDIT(audit->event("LIVE_GRAPH","additional_frontier_nodes",live.size()-output_live,needed.size());
      audit->event("LIVE_GRAPH","unreachable_before_collection",result.graph.size()-live.size(),cache.size());
      if(sentryward::debug::environment("SENTRYWARD_COMPRESSION_AUDIT"))
        SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_47));
    for(auto entry=cache.begin();entry!=cache.end();) {
      if(!needed.contains(entry->first) && !live.contains(entry->second.node)) {
        SENTRYWARD_LEDGER(ledger->evict(entry->first));entry=cache.erase(entry);
      }
      else {keys.push_back(entry->first);roots.push_back(entry->second);++entry;}
    }
    result.nodes_reclaimed+=real_expressions.collect(roots,
        std::span<const s::Ref>(roots).subspan(result.outputs.size()));
    ++result.construction_collections;
    for(std::size_t i=0;i<result.outputs.size();++i)result.outputs[i].value=roots[i];
    for(std::size_t i=0;i<keys.size();++i)cache.at(keys[i])=roots[result.outputs.size()+i];
    if(result.graph.size()>options.max_nodes)
      fail("expression_node_limit_exceeded","scalar graph node limit exceeded");
    scheduleCollection();
  }
  void collectCompletedOutputs(bool force=false) {
    if(!force && result.graph.size()<next_completed_collection)return;
    // Output compaction uses weak observers and would undo construction
    // cache retention. In the automatic policy it is a pressure operation,
    // not an operation at every newly completed family. Explicit intervals
    // continue to exercise aggressive moving collection in component tests.
    if(!force && options.construction_collection_interval==std::numeric_limits<std::size_t>::max() &&
        result.graph.size()<options.max_nodes-std::max<std::size_t>(1,options.max_nodes/4))return;
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
    auto measurement=ledger?std::make_unique<debug::ProjectionLedger::Collection>(*ledger,result.graph,cache.size()):nullptr;
#endif
    SENTRYWARD_AUDIT(audit->add(AuditCounter::gc_calls);
      audit->add(AuditCounter::gc_input_nodes,result.graph.size());
      audit->event("CACHE_COLLECTION","completed_output_weak_projections",cache.size(),result.outputs.size()));
    std::vector<s::Ref> roots;
    for(const auto& out:result.outputs)roots.push_back(out.value);
    // Observe, never root, cached projections. Consolidation must be allowed to
    // discard concrete terms replaced by a Fold. Both collections remap only
    // survivors, in O(cache entries) extra work; no ancestry search per entry.
    std::vector<std::optional<s::Id>> weak_cache;
    weak_cache.reserve(cache.size());
    for(const auto& [key,value]:cache) {
      result.graph.node(value); // validate owner before retaining only the ID
      // The following collect deliberately drops completed Fold matcher seeds.
      // Do not cache an accumulator whose compact extension needs those seeds.
      if(reductions.hasMatchingState(value))weak_cache.emplace_back(std::nullopt);
      else weak_cache.emplace_back(value.node);
    }
    failed_projections.clear();
    result.nodes_reclaimed+=real_expressions.collect(roots, {}, weak_cache);
    for(std::size_t i=0;i<roots.size();++i)result.outputs[i].value=roots[i];
    if(!output_domains.empty()) {
      s::Artifact artifact;artifact.graph=std::move(result.graph);
      for(std::size_t i=0;i<result.outputs.size();++i) {
        const auto& output=result.outputs[i];
        if(auto domain=output_domains.find(i);domain!=output_domains.end()) {
          artifact.output_families.push_back(domain->second.family(output));
        } else artifact.outputs.push_back({output.region.allocation,output.region.offset,output.region.bytes,output.value});
      }
      const auto before=artifact.graph.size();
      result.output_families_merged+=s::consolidateOutputs(artifact,false,weak_cache);
      if(before>artifact.graph.size())result.nodes_reclaimed+=before-artifact.graph.size();
      result.graph=std::move(artifact.graph);result.outputs.clear();output_domains.clear();
      for(const auto& output:artifact.outputs)
        result.outputs.push_back({{output.allocation,output.offset,output.bytes},output.value});
      for(const auto& family:artifact.output_families) {
        OutputDomain domain(family.shape,family.strides_bytes);
        if(!family.element_bytes || domain.count>UINT64_MAX/family.element_bytes)
          fail("scalar_fold_prerequisite","collected output payload overflow");
        const auto bytes=domain.count*family.element_bytes;
        output_domains.emplace(result.outputs.size(),std::move(domain));
        result.outputs.push_back({{family.allocation,family.offset,bytes},family.body});
      }
    }
    std::size_t i=0;
    [[maybe_unused]] std::size_t discarded=0;
    // No insertion occurred between collecting observers and this traversal;
    // erasing an entry preserves all remaining unordered-map iterators.
    for(auto entry=cache.begin();entry!=cache.end();++i) {
      if(weak_cache[i]) {
        const auto retained=result.graph.ref(*weak_cache[i]);
        entry->second.node=retained.node;entry->second.owner=retained.owner;
        ++entry;
      }
      else {SENTRYWARD_LEDGER(ledger->evict(entry->first));entry=cache.erase(entry);++discarded;}
    }
    SENTRYWARD_AUDIT(audit->add(AuditCounter::cache_entries_discarded,discarded);
      audit->event("CACHE_RETENTION","live_projections_after_output_consolidation",cache.size(),discarded));
    ++result.construction_collections;
    scheduleCollection();
    const auto cap=std::max<std::size_t>(1,options.max_nodes/4);
    const auto growth=std::min(cap,std::max(options.construction_collection_interval,result.graph.size()/2));
    next_completed_collection=std::max(result.graph.size()+growth,
        std::min(std::max<std::size_t>(1,options.max_nodes/2),next_completed_collection+growth));
  }
  template<class Project>
  void emitOutputDomain(ConstantRegion region,OutputDomain layout,Offset offset,const Project& project) {
    SENTRYWARD_LEDGER(ledger->add("output_domain_attempts"));
    if(result.outputs.size()>=options.max_outputs)
      fail("output_domain_limit_exceeded","refined output domain budget exceeded");
    if(output_refinement_depth>=options.max_recursion)
      fail("domain_refinement_limit_exceeded","source-domain refinement depth exceeded");
    struct Depth {unsigned& n;Depth(unsigned& n):n(n){++n;}~Depth(){--n;}} depth(output_refinement_depth);
    struct Rank {unsigned& n;unsigned saved;~Rank(){n=saved;}} rank{output_domain_rank,output_domain_rank};
    output_domain_rank=layout.shape.size();
    domains=layout.shape;
    try {
      auto body=project(offset);domains.clear();
      SENTRYWARD_AUDIT({
        (void)audit;
        if(result.outputs.empty())if(const auto path=sentryward::debug::environment("SENTRYWARD_SCALAR_ROOT_WITNESS")) {
          std::ofstream out(path);
          out<<"# allocation="<<region.allocation<<" offset="<<region.offset<<" shape=";
          for(auto n:layout.shape)out<<n<<',';
          out<<'\n';
          const std::array<s::Ref,1> root{body};
          s::writeText(out,result.graph,root,output_domain_rank);
        }
      });
      output_domains.emplace(result.outputs.size(),layout);
      result.outputs.push_back({region,body});
      SENTRYWARD_LEDGER(ledger->add("output_domains_completed");ledger->add("output_elements_completed",layout.count));
    }
    catch(const OutputDomainCuts& refinement) {
      SENTRYWARD_LEDGER(ledger->add("output_domain_refinements");ledger->add("output_domain_cuts",refinement.cuts.size()));
      domains.clear();
      const auto shift=[&](auto&& self,Offset x,std::uint64_t begin)->Offset {
        if(!x.expression) {
          if(auto i=x.terms.find(refinement.axis);i!=x.terms.end())x.base=add(x.base,mul(i->second,integer(begin)));
          return x;
        }
        const auto& e=*x.expression;
        return compound(e.kind,self(self,e.left,begin),self(self,e.right,begin),e.factor);
      };
      const auto axis=refinement.axis;
      const auto stride=layout.strides.empty()?region.bytes/layout.count:layout.strides[axis];
      const auto payload=region.bytes/layout.shape[axis];
      auto cuts=refinement.cuts;cuts.push_back(layout.shape[axis]);
      std::uint64_t previous=0;
      for(const auto end:cuts) {
        auto subregion=region;subregion.offset+=previous*stride;subregion.bytes=(end-previous)*payload;
        auto sublayout=layout;sublayout.shape[axis]=end-previous;
        sublayout.count=layout.count/layout.shape[axis]*(end-previous);
        emitOutputDomain(subregion,std::move(sublayout),shift(shift,offset,previous),project);
        previous=end;
      }
    }
    catch(const OutputDomainRadix& radix) {
      SENTRYWARD_LEDGER(ledger->add("output_radix_refinements"));
      const auto axis=radix.axis;
      const auto period=radix.period;
      const auto remap=[&](auto&& self,Offset x)->Offset {
        if(x.expression) {
          const auto& e=*x.expression;
          return compound(e.kind,self(self,e.left),self(self,e.right),e.factor);
        }
        std::map<unsigned,std::int64_t> terms;
        for(auto [level,c]:x.terms) {
          if(level<axis)terms[level]=c;
          else if(level==axis){terms[level]=mul(c,integer(period));terms[level+1]=c;}
          else terms[level+1]=c;
        }
        x.terms=std::move(terms);return x;
      };
      if(layout.strides.empty())layout.strides={region.bytes/layout.count};
      const auto stride=layout.strides.at(axis);
      if(stride>UINT64_MAX/period)fail("index_representation_overflow","output radix stride overflow");
      layout.shape[axis]/=period;layout.shape.insert(layout.shape.begin()+axis+1,period);
      layout.strides[axis]*=period;layout.strides.insert(layout.strides.begin()+axis+1,stride);
      emitOutputDomain(region,std::move(layout),remap(remap,offset),project);
    }
  }
  void outputFamily(const OutputWrite& output,std::uint64_t begin,std::uint64_t count,
                    const v::ElementType& type,std::uint64_t bytes) {
    SENTRYWARD_AUDIT(audit->add(AuditCounter::output_elements_requested,count));
    if(result.outputs.size()>=options.max_outputs) {
      collectCompletedOutputs(true);
      if(result.outputs.size()>=options.max_outputs)
        fail("output_family_limit_exceeded","output family budget exceeded");
    }
      Offset offset(integer(output.value_byte_offset+begin));
      if(count>1)offset.terms[0]=integer(bytes);
      auto region=output.region;region.offset+=begin;region.bytes=count*bytes;
      emitOutputDomain(region,OutputDomain(count),offset,[&](const Offset& at){return project(output.value,at,type);});
      // The growth threshold is O(1); a family is not one scalar output.
      // Use the same boundary as normalized output domains, rather than
      // retaining intermediate projections until 128 families accumulate.
      if(options.collection_interval)collectCompletedOutputs();
      return;
  }
  bool spatialOutput(std::span<const OutputWrite> outputs) {
    if(outputs.empty())return false;
    std::vector<v::Value> values;values.reserve(outputs.size());
    for(const auto& output:outputs)values.push_back(output.value);
    const auto plan=pure_source_templates.map(values,true);
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/blocks/scalar_fold_lowering_trace_3.inc"
#endif
    // Validate addresses of the existing records, never synthesize members.
    // The resulting output stride and source-member stride are independent.
    for(const auto& domain:plan) {
      const auto& first=outputs[domain.begin];
      if(!first.region.bytes || domain.count>UINT64_MAX/first.region.bytes)return false;
      const auto step=domain.count>1?
          outputs[domain.begin+domain.result_stride].region.offset-first.region.offset:first.region.bytes;
      if(!step)return false;
      for(std::size_t i=0;i<domain.count;++i) {
        const auto& item=outputs[domain.begin+i*domain.result_stride];
        if(item.region.allocation!=first.region.allocation || item.region.bytes!=first.region.bytes ||
           item.value_byte_offset!=first.value_byte_offset ||
           static_cast<unsigned __int128>(first.region.offset)+
             static_cast<unsigned __int128>(step)*i!=item.region.offset)return false;
      }
    }
    struct Grid {std::size_t first,rows=1;std::uint64_t outer_stride=0;};
    std::vector<Grid> grids;
    std::map<std::tuple<std::size_t,std::size_t,std::size_t,std::int64_t,std::uint64_t>,std::size_t> open;
    for(std::size_t i=0;i<plan.size();++i) {
      const auto& d=plan[i];const auto& first=outputs[d.begin];
      const auto root=d.source?d.source->root:nullptr;
      if(!root || root->kind!=IndexedSourceUse::Kind::CoordinateMap || !root->binding ||
         root->binding->period || !root->binding->quotient_terms.empty() ||
         root->children.size()!=1 || d.source->zero_mask || d.count<2) {
        grids.push_back({i});continue;
      }
      const auto& map=*root->binding;
      const auto key=std::tuple{indexed_use_identity.use(root->children[0]),d.count,d.result_stride,
          map.stride,first.value_byte_offset};
      auto found=open.find(key);bool extended=false;
      if(found!=open.end()) {
        auto& grid=grids[found->second];const auto& base=plan[grid.first];
        const auto& origin=outputs[base.begin];
        const auto column_stride=outputs[base.begin+base.result_stride].region.offset-origin.region.offset;
        const auto current_stride=outputs[d.begin+d.result_stride].region.offset-first.region.offset;
        const auto outer=grid.rows==1?first.region.offset-origin.region.offset:grid.outer_stride;
        const auto expected=static_cast<__int128>(base.source->root->binding->base)+
            static_cast<__int128>(grid.rows)*d.count*map.stride;
        if(expected==map.base && current_stride==column_stride && origin.region.allocation==first.region.allocation &&
           origin.region.bytes==first.region.bytes &&
           static_cast<unsigned __int128>(outer)>=(static_cast<unsigned __int128>(d.count)-1)*column_stride+first.region.bytes &&
           static_cast<unsigned __int128>(origin.region.offset)+static_cast<unsigned __int128>(grid.rows)*outer==first.region.offset) {
          ++grid.rows;grid.outer_stride=outer;extended=true;
        }
      }
      if(!extended){open.insert_or_assign(key,grids.size());grids.push_back({i});}
    }
    SENTRYWARD_AUDIT((void)audit;if(sentryward::debug::environment("SENTRYWARD_OUTPUT_DOMAIN_WITNESS"))
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_51));
    for(const auto& grid:grids) {
      const auto& domain=plan[grid.first];
      const auto& first=outputs[domain.begin];
      if(!domain.source) {
        outputFamily(first,0,1,input.type(first.value).element(),first.region.bytes);continue;
      }
      const auto step=outputs[domain.begin+domain.result_stride].region.offset-first.region.offset;
      if(grid.rows>UINT64_MAX/domain.count || grid.rows*domain.count>UINT64_MAX/first.region.bytes)
        fail("index_representation_overflow","output grid payload overflow");
      const auto count=grid.rows*domain.count;
      auto region=first.region;region.bytes*=count;
      auto source=std::make_shared<IndexedSource>(*domain.source);source->count=count;
      output_source_patterns.push_back(source);
      Offset index;index.terms[0]=1;
      OutputDomain layout({domain.count},{step});
      if(grid.rows>1){index.terms[0]=integer(domain.count);index.terms[1]=1;
        layout=OutputDomain({grid.rows,domain.count},{grid.outer_stride,step});}
      emitOutputDomain(region,std::move(layout),index,[&](const Offset& at) {
        return inContext(std::make_shared<IndexedContext>(IndexedContext{at,source->root}),[&] {
          // The source plan owns the representative and its coordinate map;
          // the output record supplies only destination/payload information.
          return project(source->prototype,Offset(integer(first.value_byte_offset)),input.type(first.value).element());
        });
      });
      ++result.spatial_families_projected;result.spatial_outputs_projected+=count;
      if(options.collection_interval)collectCompletedOutputs();
    }
    return true;
  }
 public:
  Lower(const v::Graph& g,const MemoryEliminationResult& m,const ScalarFoldOptions& o):input(g),memory(m),options(o){
    for(auto it=memory.symbolic_initial_reads.begin();it!=memory.symbolic_initial_reads.end();) {
      const auto begin=it;auto end=it->first.instance+1;++it;
      while(it!=memory.symbolic_initial_reads.end() && it->first.node==begin->first.node &&
          it->first.instance==end && it->second.allocation==begin->second.allocation &&
          it->second.offset==begin->second.offset && it->second.bytes==begin->second.bytes) {++end;++it;}
      symbolic_read_domains.emplace(begin->first,end);
    }
    if(m.symbolic_initial_reads.empty())return;
    const auto field=[](const std::string& s){return std::to_string(s.size())+":"+s;};
    std::set<std::string> ambiguous;
    for(const auto& node:g.nodes()) {
      const auto* parameter=dynamic_cast<const v::InputParameterOp*>(&node.operation());
      if(!parameter || !parameter->pointer())continue;
      const auto& p=*parameter->pointer();
      if(p.owner_cta || p.owner_thread || p.addressing!="global")continue;
      const auto identity=field(p.addressing)+field(p.identity);
      const auto [it,inserted]=allocation_bases.emplace(identity,node.result());
      if(!inserted) {
        const auto& prior=dynamic_cast<const v::InputParameterOp&>(g.definingNode(it->second).operation());
        if(prior.symbol()!=parameter->symbol())ambiguous.insert(identity);
      }
    }
    for(const auto& identity:ambiguous)allocation_bases.erase(identity);
  }
  bool spatialStoreFamily(const OutputWriteFamily& family) {
    if(!options.compact_output_families || family.count<2 ||
       !family.region.bytes || family.count>UINT64_MAX/family.region.bytes ||
       family.region.offset>UINT64_MAX-family.count*family.region.bytes ||
       family.byte_stride<0 || std::uint64_t(family.byte_stride)!=family.region.bytes)return false;
    const auto& store=input.node(family.first.node);
    if(!store.hasUniformThreadTemplate())return false;
    const auto operand=store.operands()[1];
    if(input.resolve(operand)!=operand || operand.conversion() || operand.pointer() ||
       !operand.indices().empty() || operand.threadSelector() || !input.type(operand).isScalar())return false;
    const auto& child=input.definingNode(operand);
    if(child.storedInstanceCount()!=1)return false;
    auto base=operand;std::int64_t stride=0;
    const auto& threads=child.instanceThreads();
    if(child.origin().cta==store.origin().cta && !threads.empty() &&
       threads.at(operand.instanceIndex())==store.origin().thread) {
      if(threads.storageIdentity()!=store.instanceThreads().storageIdentity())return false;
      base=child.result(operand.resultIndex(),family.first.instance);stride=1;
    }
    spatial_count=family.count;++spatial_serial;
    spatial_values.clear();spatial_instance_strides.clear();domains={spatial_count};
    try {
      registerSpatialFamily({&input,base,spatial_count,stride,{}});
      const auto& type=input.type(base);
      auto body=project(base,Offset{},type.element());
      auto region=family.region;region.bytes*=family.count;
      output_domains.emplace(result.outputs.size(),family.count);
      result.outputs.push_back({region,body});
      ++result.spatial_families_projected;result.spatial_outputs_projected+=family.count;
    } catch(const Failure& error) {
      spatial_count=0;spatial_values.clear();spatial_instance_strides.clear();domains.clear();
      if(error.code=="scalar_fold_spatial_projection" || error.code=="scalar_fold_non_affine_projection" ||
         (error.code.ends_with("_limit_exceeded") || error.code=="index_representation_overflow" || error.code=="invalid_analysis_budget")) {
        visiting.clear();collectCompletedOutputs((error.code.ends_with("_limit_exceeded") || error.code=="index_representation_overflow" || error.code=="invalid_analysis_budget"));return false;
      }
      throw;
    }
    spatial_count=0;spatial_values.clear();spatial_instance_strides.clear();domains.clear();
    return true;
  }
  void validateRadixPartition(const PartitionedRead& split) {
    // First reduce coverage-only adjacent axes to a progression of intervals.
    // This does not alter the retained member/payload maps. Mixed row strides
    // need not share a radix (e.g. a missing source at one row breaks an AP).
    struct Span {std::uint64_t first,width,step,count;};
    std::vector<Span> spans;bool strided=true;
    unsigned __int128 span_coverage=0;
    const auto extent=split.type.bitSize()/8;
    for(const auto& region:split.regions) {
      if(region.view.blocks.size()!=1){strided=false;break;}
      const auto& b=region.view.blocks[0];
      if(!b.bytes || b.result_offset!=region.begin)fail("compact_view_unsupported","invalid strided domain origin");
      Span span{b.result_offset,b.bytes,0,1};
      for(const auto& axis:b.axes) {
        if(!axis.count)fail("compact_view_unsupported","empty strided domain axis");
        if(axis.count==1)continue;
        if(span.count==1 && axis.result_stride==span.width) {
          if(span.width>UINT64_MAX/axis.count)fail("compact_view_unsupported","strided domain width overflow");
          span.width*=axis.count;
        } else if(span.count==1 && axis.result_stride>=span.width) {
          span.step=axis.result_stride;span.count=axis.count;
        } else if(static_cast<unsigned __int128>(span.step)*span.count==axis.result_stride &&
                  span.count<=UINT64_MAX/axis.count)span.count*=axis.count;
        else {strided=false;break;}
      }
      if(!strided)break;
      const auto bounding=static_cast<unsigned __int128>(span.count-1)*span.step+span.width;
      if(bounding!=region.bytes || span.first+bounding>extent)
        fail("compact_view_unsupported","strided domain bounding extent mismatch");
      span_coverage+=static_cast<unsigned __int128>(span.width)*span.count;
      spans.push_back(span);
    }
    if(strided) {
      const auto overlaps=[](const Span& a,const Span& b) {
        using Wide=__int128;
        const auto end=[](const Span& x){return Wide(x.first)+Wide(x.count-1)*x.step+x.width;};
        if(end(a)<=b.first || end(b)<=a.first)return false;
        const auto interval=[](const Span& one,const Span& many) {
          const auto lo=Wide(one.first)-Wide(many.width-1),hi=Wide(one.first)+one.width-1;
          if(many.count==1)return Wide(many.first)>=lo && Wide(many.first)<=hi;
          const auto first=lo<=many.first?Wide(0):(lo-many.first+many.step-1)/many.step;
          const auto last=hi< many.first?Wide(-1):std::min(Wide(many.count-1),(hi-many.first)/many.step);
          return first<=last;
        };
        if(a.count==1)return interval(a,b);
        if(b.count==1)return interval(b,a);
        // Only shifts congruent to the start difference modulo gcd can
        // intersect. Usually one or two tests for aligned fragment strides;
        // worst case depends on stored payload width/gcd, not member count.
        const Wide g=std::gcd(a.step,b.step);
        const Wide low=-Wide(a.width-1),high=Wide(b.width-1);
        auto correction=(Wide(a.first)-b.first-low)%g;if(correction<0)correction+=g;
        for(auto delta=low+correction;delta<=high;delta+=g) {
          const Wide af=Wide(a.first)+(delta<0?-delta:0),bf=Wide(b.first)+(delta>0?delta:0);
          if(af>UINT64_MAX || bf>UINT64_MAX)continue;
          const auto ac=std::min(Wide(a.count),1+(Wide(UINT64_MAX)-af)/a.step);
          const auto bc=std::min(Wide(b.count),1+(Wide(UINT64_MAX)-bf)/b.step);
          if(algorithm::IntersectAffine({std::uint64_t(af),a.step,std::uint64_t(ac)},
              {std::uint64_t(bf),b.step,std::uint64_t(bc)}).count)return true;
        }
        return false;
      };
      for(std::size_t i=0;i<spans.size();++i)for(std::size_t j=0;j<i;++j)
        if(overlaps(spans[i],spans[j]))fail("compact_view_unsupported","overlapping strided result domains");
      if(span_coverage!=extent)fail("compact_view_unsupported","incomplete strided result domains");
      return;
    }
    // A stored mixed-radix box has one interval per digit. Coverage and
    // disjointness cost O(R^2 * D + A log A), not its represented cardinality.
    // R = source rules, D = distinct axis strides, A = stored axes.
    const auto total=split.type.bitSize()/8;
    const auto unit=split.regions.front().view.blocks.at(0).bytes;
    if(!unit || total%unit)fail("compact_view_unsupported","invalid radix view unit");
    std::vector<std::uint64_t> strides{unit,total};
    for(const auto& region:split.regions) {
      if(region.view.blocks.size()!=1)fail("compact_view_unsupported","radix domain needs one source rule");
      const auto& block=region.view.blocks[0];
      if(block.bytes!=unit || block.result_offset!=region.begin || region.begin%unit)
        fail("compact_view_unsupported","inconsistent radix domain origin or unit");
      for(const auto& axis:block.axes) {
        if(!axis.count || !axis.result_stride || axis.result_stride%unit || axis.result_stride>total)
          fail("compact_view_unsupported","invalid radix domain axis");
        strides.push_back(axis.result_stride);
      }
    }
    std::sort(strides.begin(),strides.end());strides.erase(std::unique(strides.begin(),strides.end()),strides.end());
    for(std::size_t i=1;i<strides.size();++i)if(strides[i]%strides[i-1])
      fail("compact_view_unsupported","view axes do not share a radix grid");
    using Interval=std::pair<std::uint64_t,std::uint64_t>;
    std::vector<std::vector<Interval>> boxes;unsigned __int128 covered=0;
    for(const auto& region:split.regions) {
      const auto& block=region.view.blocks[0];
      if(region.begin>=total)fail("compact_view_unsupported","radix domain exceeds read");
      std::vector<Interval> box;
      for(std::size_t d=0;d+1<strides.size();++d) {
        const auto first=(region.begin/strides[d])%(strides[d+1]/strides[d]);
        box.emplace_back(first,first+1);
      }
      unsigned __int128 extent=unit,volume=1;
      std::set<std::uint64_t> varying;
      for(const auto& axis:block.axes) {
        if(axis.count==1)continue;
        const auto d=std::size_t(std::lower_bound(strides.begin(),strides.end(),axis.result_stride)-strides.begin());
        if(d+1>=strides.size() || !varying.insert(axis.result_stride).second ||
            axis.count>strides[d+1]/strides[d]-box[d].first)
          fail("compact_view_unsupported","radix domain crosses a digit boundary");
        box[d].second=box[d].first+axis.count;
        extent+=static_cast<unsigned __int128>(axis.count-1)*axis.result_stride;
        volume*=axis.count;
      }
      if(extent!=region.bytes || static_cast<unsigned __int128>(region.begin)+extent>total)
        fail("compact_view_unsupported","radix domain bounding extent mismatch");
      for(const auto& prior:boxes) {
        bool overlaps=true;
        for(std::size_t d=0;d<box.size();++d)
          if(box[d].second<=prior[d].first || prior[d].second<=box[d].first){overlaps=false;break;}
        if(overlaps)fail("compact_view_unsupported","overlapping radix result domains");
      }
      covered+=volume;boxes.push_back(std::move(box));
    }
    if(covered!=total/unit)fail("compact_view_unsupported","incomplete radix result domains");
  }
  ScalarFoldResult run(){
    SENTRYWARD_AUDIT_STAGE("scalar_fold");
    SENTRYWARD_AUDIT(audit->add(AuditCounter::input_output_records,memory.outputs.size());
      audit->add(AuditCounter::input_output_families,memory.output_families.size());
      audit->event("INPUT_REPRESENTATION","scalar_fold_output_sources",
        memory.outputCount(),memory.output_families.size()));
    result.source_outputs=memory.outputCount();
    SENTRYWARD_AUDIT({(void)audit;if(sentryward::debug::environment("SENTRYWARD_OUTPUT_DOMAIN_WITNESS"))
      for(std::size_t i=0;i<std::min<std::size_t>(8,memory.outputs.size());++i) {
        const auto& x=memory.outputs[i];
        SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_52);
      }
    });
    try{
      if(!memory.complete || !memory.views_normalized)fail("scalar_fold_prerequisite","complete memory elimination and normalized views are required");
      for(const auto& [key,read]:memory.normalized_reads) {
        (void)key;
        if(const auto* split=std::get_if<PartitionedRead>(&read)) {
          if(split->template_domains && !split->regions.empty()) {
            const auto& first_block=split->regions.front().view.blocks.at(0);
            const auto outer=first_block.axes.empty()?0:first_block.axes.back().result_stride;
            if(std::any_of(split->regions.begin(),split->regions.end(),[&](const auto& r) {
                return r.view.blocks.size()==1 && (r.view.blocks[0].axes.size()>2 ||
                    (!r.view.blocks[0].axes.empty() && r.view.blocks[0].axes.back().result_stride!=outer));
              })) {
              try {validateRadixPartition(*split);}catch(const Failure&) {
                SENTRYWARD_AUDIT((void)audit;SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_53);
                  for(const auto& r:split->regions) {SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_54);
                    for(const auto& axis:r.view.blocks[0].axes)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_55);
                    SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_56);});
                throw;
              }
              continue;
            }
            // Check stored AP domains, not their members. A common byte grid
            // makes overlap exactly an AP intersection of cell indices.
            const auto unit=split->regions.front().view.blocks.at(0).bytes;
            const auto& first_axes=split->regions.front().view.blocks.at(0).axes;
            const bool row_grid=first_axes.size()==2;
            const auto row_stride=row_grid?first_axes.back().result_stride:split->type.bitSize()/8;
            const auto row_count=row_grid && row_stride?split->type.bitSize()/8/row_stride:1;
            const bool uniform=unit && std::all_of(split->regions.begin(),split->regions.end(),
              [&](const auto& r){return r.view.blocks.size()==1 &&
                (row_grid?(r.view.blocks[0].axes.size()==2 &&
                    r.view.blocks[0].axes.back().result_stride==row_stride &&
                    r.view.blocks[0].axes.back().count>0):r.view.blocks[0].axes.size()<=1) &&
                r.view.blocks[0].bytes==unit && r.begin%unit==0;});
            if(uniform) {
              if(!row_stride || !row_count || row_stride%unit ||
                  static_cast<unsigned __int128>(row_stride)*row_count!=split->type.bitSize()/8)
                fail("compact_view_unsupported","invalid template row coverage");
              std::vector<algorithm::AffineDomain> sets;
              std::vector<std::pair<std::uint64_t,std::uint64_t>> row_ranges;
              unsigned __int128 covered=0;
              for(const auto& region:split->regions) {
                const auto& b=region.view.blocks[0];
                if((!b.axes.empty() &&
                    (!b.axes[0].count || b.axes[0].result_stride%unit)))
                  fail("compact_view_unsupported","invalid template result progression");
                const auto first_row=row_grid?region.begin/row_stride:0;
                const auto rows=row_grid?b.axes.back().count:1;
                if(first_row>=row_count || rows>row_count-first_row)
                  fail("compact_view_unsupported","template row range exceeds read");
                algorithm::AffineDomain d{(region.begin%row_stride)/unit,b.axes.empty()?0:b.axes[0].result_stride/unit,
                    b.axes.empty()?1:b.axes[0].count};
                if(d.last()>=row_stride/unit || (!d.step && d.count>1))
                  fail("compact_view_unsupported","template result progression exceeds read");
                covered+=static_cast<unsigned __int128>(d.count)*rows;
                sets.push_back(d);
                row_ranges.emplace_back(first_row,first_row+rows);
              }
              // Row intervals may be disjoint even when column APs coincide;
              // AffineIndex requires disjoint one-dimensional keys. Sweep
              // stored row intervals and test only row-overlapping candidates.
              // O(R log R + C * AP-intersection), C <= R*(R-1)/2, no members.
              std::vector<std::size_t> order(sets.size());
              std::iota(order.begin(),order.end(),0);
              std::sort(order.begin(),order.end(),[&](auto a,auto b){return row_ranges[a]<row_ranges[b];});
              std::multimap<std::uint64_t,std::size_t> active;
              for(auto i:order) {
                active.erase(active.begin(),active.upper_bound(row_ranges[i].first));
                for(auto [end,j]:active) {
                  (void)end;
                  if(algorithm::IntersectAffine(sets[i],sets[j]).count)
                    fail("compact_view_unsupported","overlapping template result domains");
                }
                active.emplace(row_ranges[i].second,i);
              }
              if(covered!=split->type.bitSize()/8/unit || split->type.bitSize()/8%unit)
                fail("compact_view_unsupported","incomplete template result domains");
              continue;
            }
          }
          std::uint64_t cursor=0;
          for(const auto& region:split->regions) {
            if(!region.bytes || region.begin!=cursor || region.bytes>UINT64_MAX-cursor)
              fail("compact_view_unsupported","read-result domains must form a disjoint complete partition");
            cursor+=region.bytes;
          }
          if(cursor!=split->type.bitSize()/8)fail("compact_view_unsupported","incomplete read-result domains");
        }
      }
      if(!options.max_nodes || !options.max_outputs || !options.max_recursion || !options.max_pending_projections)fail("invalid_analysis_budget","scalar/Fold budgets must be positive");
      next_collection=std::min(options.construction_collection_interval,
                               std::max<std::size_t>(1,options.max_nodes/4));
      next_completed_collection=next_collection;
      for(const auto& family:memory.output_families) {
        if(spatialStoreFamily(family))continue;
        std::vector<OutputWrite> outputs;
        if(options.compact_output_families && family.count>1 && family.byte_stride>=0 &&
           std::uint64_t(family.byte_stride)==family.region.bytes) {
          outputs.reserve(family.count);
          SENTRYWARD_AUDIT(audit->event("INSTANCE_ENUMERATION","output_family_materialized_for_spatial_recovery",
            family.count,1,family.first.node,true));
          for(std::size_t i=0;i<family.count;++i)outputs.push_back(family.at(input,i));
          if(spatialOutput(outputs))continue;
        }
        SENTRYWARD_AUDIT(if(family.count>1)audit->event("INSTANCE_ENUMERATION","output_family_scalar_fallback",
          family.count,1,family.first.node,true));
        for(std::size_t i=0;i<family.count;++i) {
          const auto output=family.at(input,i);
          const auto& type=input.type(output.value);
          outputFamily(output,0,1,type.element(),output.region.bytes);
        }
      }
      std::size_t spatial_failed_end=0;
      if((!memory.normalized_output_groups.empty() || !memory.normalized_output_members.empty()) &&
         memory.normalized_output_members.size()!=memory.outputs.size())
        fail("scalar_fold_prerequisite","invalid normalized output membership");
      for(std::size_t output_index=0;output_index<memory.outputs.size();++output_index){
        if(options.compact_output_families) {
          if(const auto found=memory.normalized_output_groups.find(output_index);
             found!=memory.normalized_output_groups.end()) {
            const auto& group=found->second;
            if(group.end<=output_index || group.end>memory.outputs.size())
              fail("scalar_fold_prerequisite","invalid normalized output record range");
            for(const auto& output:group.domains) {
              const auto width=output.element_type.width()/8;
              if(!width || !output.region.bytes || output.region.bytes%width)
                fail("scalar_fold_prerequisite","invalid numerical output view extent");
              if(result.outputs.size()>=options.max_outputs) {
                collectCompletedOutputs(true);
                if(result.outputs.size()>=options.max_outputs)
                  fail("output_family_limit_exceeded","output family budget exceeded");
              }
              const auto count=output.region.bytes/width;
              OutputDomain layout(output.shape,output.strides_bytes);
              if(layout.count!=count)fail("scalar_fold_prerequisite","output view shape does not cover payload");
              SENTRYWARD_AUDIT(audit->add(AuditCounter::output_elements_requested,count));
              domains=output.shape;Offset offset(integer(output.source_byte_offset));
              std::uint64_t step=width;
              for(std::size_t axis=domains.size();axis-->0;) {
                if(domains[axis]>1)offset.terms[axis]=integer(step);
                if(step>UINT64_MAX/domains[axis])fail("scalar_fold_prerequisite","output projection extent overflow");
                step*=domains[axis];
              }
              emitOutputDomain(output.region,std::move(layout),offset,
                  [&](const Offset& at){return view(output.view,at,output.element_type);});
              ++result.spatial_families_projected;result.spatial_outputs_projected+=count;
              // A template can cover thousands of outputs. Poll the O(1)
              // node-growth guard per completed template, not per 128 templates.
              if(options.collection_interval)collectCompletedOutputs();
            }
            continue;
          }
          if(!memory.normalized_output_members.empty() && memory.normalized_output_members[output_index])continue;
        }
        const auto& output=memory.outputs[output_index];
        const auto& type=input.type(output.value);
        const auto element=output.element_type.value_or(type.element());
        if(element.isParameterized())fail("scalar_fold_parameterized_type","output element type is unresolved");
        const auto element_bytes=element.width()/8;
        if(!element_bytes)fail("scalar_fold_subbyte_value","sub-byte output requires bit projection");
        if(options.compact_output_families && output_index>=spatial_failed_end && type.isScalar() &&
           element==type.element() && output.region.bytes==element_bytes && element_bytes<=8) {
          auto end=output_index+1;
          const auto cta=input.instance(output.value).origin.cta;
          while(end<memory.outputs.size()) {
            const auto& next=memory.outputs[end];
            const auto offset=static_cast<__int128>(output.region.offset)+static_cast<__int128>(end-output_index)*element_bytes;
            if(next.region.allocation!=output.region.allocation || next.region.bytes!=element_bytes ||
               offset!=next.region.offset || input.type(next.value)!=type ||
               next.element_type.value_or(type.element())!=element ||
               next.value_byte_offset!=output.value_byte_offset || input.instance(next.value).origin.cta!=cta)break;
            ++end;
          }
          // Node identity is not a source-pattern boundary after memory
          // elimination. Distinct scalar producers may implement one affine
          // output domain. Plan/validate that domain before constructing any
          // scalar expressions; do not build N outputs and recover it later.
          if(end-output_index>1) {
            if(spatialOutput(std::span(memory.outputs).subspan(output_index,end-output_index))) {
              output_index=end-1;continue;
            }
            spatial_failed_end=end;
          }
        }
        if(options.compact_output_families && output.region.bytes &&
            output.region.bytes%element_bytes==0 && element_bytes<=8) {
          outputFamily(output,0,output.region.bytes/element_bytes,element,element_bytes);
          continue;
        }
        std::uint64_t done=0;
        while(done<output.region.bytes){
          if(result.outputs.size()>=options.max_outputs)fail("output_count_limit_exceeded","scalar output budget exceeded");
          auto bytes=std::min<std::uint64_t>(element_bytes,output.region.bytes-done);
          if(bytes>8)fail("scalar_fold_wide_scalar","output scalar wider than 64 bits requires a numeric type");
          auto target=bytes==element_bytes?element:bit(unsigned(8*bytes));
          auto projected=project(output.value,Offset(integer(output.value_byte_offset+done)),target);
          auto region=output.region;region.offset+=done;region.bytes=bytes;
          result.outputs.push_back({region,projected});done+=bytes;
          if(options.collection_interval && result.outputs.size()%options.collection_interval==0)
            collectCompletedOutputs();
        }
      }
      if(output_domains.empty())coalesceFragments(result.graph,result.outputs);
      std::vector<s::Ref> roots;for(const auto& out:result.outputs)roots.push_back(out.value);
      std::map<std::size_t,std::vector<s::Ref>> validation_roots;
      for(std::size_t i=0;i<roots.size();++i) {
        const auto domain=output_domains.find(i);
        const auto rank=domain==output_domains.end()?0:domain->second.shape.size();
        validation_roots[rank].push_back(roots[i]);
      }
      for(const auto& [rank,values]:validation_roots)result.graph.validateRoots(values,rank);
      result.nodes_before_final_collection=result.graph.size();
      result.nodes_reclaimed+=result.graph.collect(roots);
      for(std::size_t i=0;i<roots.size();++i)result.outputs[i].value=roots[i];
      if(!output_domains.empty()) {
        std::vector<ScalarFoldOutput> scalars;
        for(std::size_t i=0;i<result.outputs.size();++i) {
          const auto& out=result.outputs[i];
          if(auto domain=output_domains.find(i);domain!=output_domains.end()) {
            result.output_families.push_back(domain->second.family(out));
          } else scalars.push_back(out);
        }
        result.outputs=std::move(scalars);
        s::Artifact artifact;
        artifact.graph=std::move(result.graph);
        artifact.output_families=std::move(result.output_families);
        for(const auto& output:result.outputs)
          artifact.outputs.push_back({output.region.allocation,output.region.offset,output.region.bytes,output.value});
        const auto before=artifact.graph.size();
        result.output_families_merged+=s::consolidateOutputs(artifact,false);
        if(before>artifact.graph.size())result.nodes_reclaimed+=before-artifact.graph.size();
        result.graph=std::move(artifact.graph);
        result.output_families=std::move(artifact.output_families);
        for(std::size_t i=0;i<result.outputs.size();++i)result.outputs[i].value=artifact.outputs[i].value;
      }
      result.complete=true;
    }catch(const Failure& e){result.diagnostic=ScalarFoldDiagnostic{e.code,e.what(),current};}
     catch(const IndexedSourceLimit& e){result.diagnostic=ScalarFoldDiagnostic{e.code,e.what(),current};}
     catch(const frontend::ptx::LayoutError& e){result.diagnostic=ScalarFoldDiagnostic{"scalar_fold_mma_descriptor",e.what(),current};}
    if(sentryward::debug::environment("SENTRYWARD_SCALAR_NODE_HISTOGRAM")) {
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/blocks/scalar_fold_lowering_trace_4.inc"
#endif
}
    SENTRYWARD_LEDGER(ledger->snapshot(result.graph,cache.size(),result.outputs.size()));
    return std::move(result);
  }
};
}
ScalarFoldResult lowerScalarFold(const value_ir::Graph& graph,const MemoryEliminationResult& memory,
                                const ScalarFoldOptions& options){
  return Lower(graph,memory,options).run();
}
} // namespace sentryward::value_analysis
