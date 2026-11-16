#pragma once
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
#include "sentryward/equivalence/conversion_uses.h"
#include "sentryward/scalar_fold/consolidate_outputs.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/native_matrix.h"
#include "sentryward/value_ir/ptx_operation.h"
#include "sentryward/frontend/ptx/layouts.h"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <compare>
#include <concepts>
#include <limits>
#include <functional>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <ranges>
#include <set>
#include <sstream>
#include <string_view>
#include <tuple>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace sentryward::value_analysis {
namespace {
namespace v = value_ir;
namespace s = scalar_fold;
// Compression-audit builds also compute state that only their reports read.
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
constexpr bool audit_build=true;
#else
constexpr bool audit_build=false;
#endif
struct Failure : std::runtime_error {
  std::string code;
  bool literal=false; // the message is a string literal at its fail() site (first-item rule, run.cpp)
  Failure(std::string c, std::string message, bool l=false):std::runtime_error(std::move(message)),code(std::move(c)),literal(l){}
};
// Exact source-boundary refinement, not a scalar fallback or speculative
// bisection. Cuts are inverse images in the current output coordinate domain.
struct OutputDomainCuts { std::vector<std::uint64_t> cuts; unsigned axis=0; };
struct OutputDomainRadix { unsigned axis; std::uint64_t period; };
struct CollectProjectionFrontier {};
[[noreturn]] void fail(std::string c,std::string m){throw Failure(std::move(c),std::move(m));}
// A string-literal message binds here (exact match before std::string): the
// site's text carries no CTA, offset or node id (scalar_fold_failure_codes.h).
[[noreturn]] void fail(std::string c,const char* m){throw Failure(std::move(c),m,true);}
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
// Appends std::to_string(n)'s digits without a temporary string.
template<std::integral T> void appendDecimal(std::string& out,T n) {
  char digits[24];
  out.append(digits,std::to_chars(digits,digits+sizeof digits,n).ptr);
}
// The integer std::from_chars reads from the whole of `text`; nullopt when it
// fails or leaves characters over.
template<std::integral T> std::optional<T> exactInteger(std::string_view text) {
  T value{};const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
  if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size())return std::nullopt;
  return value;
}
// Affine coefficients by binder level, ascending: the std::map interface this
// file uses (ordered iteration, operator[] inserting 0, find, erase) over a
// sorted array. Up to four levels stay inline, so the Offset copied with each
// request, context and expression copies a few words instead of tree nodes.
// Unlike std::map, every insertion or erasure invalidates references and
// iterators into the same terms: never hold one across operator[] on a new
// level or across erase (use erase's return value and re-read end()).
class OffsetTerms {
 public:
  using value_type=std::pair<unsigned,std::int64_t>;
  using iterator=value_type*;
  using const_iterator=const value_type*;
  OffsetTerms()=default;
  OffsetTerms(const OffsetTerms& o):spill_(o.spill_),size_(o.size_) {
    if(spill_.empty())std::copy_n(o.inline_.begin(),size_,inline_.begin());
  }
  OffsetTerms(OffsetTerms&& o) noexcept:spill_(std::move(o.spill_)),size_(std::exchange(o.size_,0)) {
    if(spill_.empty())std::copy_n(o.inline_.begin(),size_,inline_.begin());
    o.spill_.clear();
  }
  OffsetTerms& operator=(const OffsetTerms& o) {
    if(this==&o)return *this;
    if(o.spill_.empty()){spill_.clear();std::copy_n(o.inline_.begin(),o.size_,inline_.begin());}
    else spill_=o.spill_;
    size_=o.size_;return *this;
  }
  OffsetTerms& operator=(OffsetTerms&& o) noexcept {
    if(this==&o)return *this;
    spill_=std::move(o.spill_);o.spill_.clear();size_=std::exchange(o.size_,0);
    if(spill_.empty())std::copy_n(o.inline_.begin(),size_,inline_.begin());
    return *this;
  }
  iterator begin(){return data();}
  iterator end(){return data()+size_;}
  const_iterator begin()const{return data();}
  const_iterator end()const{return data()+size_;}
  std::size_t size()const{return size_;}
  bool empty()const{return !size_;}
  iterator find(unsigned level){const auto i=lower(level);return i!=end() && i->first==level?i:end();}
  const_iterator find(unsigned level)const{return const_cast<OffsetTerms&>(*this).find(level);}
  std::int64_t& operator[](unsigned level) {
    const auto i=lower(level);
    if(i!=end() && i->first==level)return i->second;
    return insertAt(std::size_t(i-begin()),level)->second;
  }
  // A template, so that erase(0) erases level 0 as std::map's erase does,
  // instead of being ambiguous with a null iterator.
  template<class It> requires std::same_as<It,iterator> || std::same_as<It,const_iterator>
  iterator erase(It at) {
    const auto i=std::size_t(at-begin());
    std::move(begin()+i+1,end(),begin()+i);
    --size_;if(!spill_.empty())spill_.pop_back();
    return begin()+i;
  }
  std::size_t erase(unsigned level){const auto i=find(level);if(i==end())return 0;erase(i);return 1;}
 private:
  static constexpr std::size_t inline_capacity=4;
  value_type* data(){return spill_.empty()?inline_.data():spill_.data();}
  const value_type* data()const{return spill_.empty()?inline_.data():spill_.data();}
  iterator lower(unsigned level) {
    return std::lower_bound(begin(),end(),level,[](const value_type& t,unsigned l){return t.first<l;});
  }
  iterator insertAt(std::size_t i,unsigned level) {
    if(spill_.empty() && size_<inline_capacity) {
      std::move_backward(inline_.begin()+i,inline_.begin()+size_,inline_.begin()+size_+1);
      inline_[i]={level,0};++size_;return inline_.data()+i;
    }
    // Spill once: the vector holds every entry until the terms empty again.
    if(spill_.empty()){spill_.reserve(2*inline_capacity);spill_.assign(inline_.begin(),inline_.begin()+size_);}
    spill_.insert(spill_.begin()+i,value_type{level,0});++size_;return spill_.data()+i;
  }
  // Invariant: spill_ empty => entries in inline_[0,size_); else spill_.size()==size_.
  std::array<value_type,inline_capacity> inline_{};
  std::vector<value_type> spill_;
  std::uint32_t size_=0;
};
struct OffsetExpression;
struct Offset {
  std::int64_t base=0;
  OffsetTerms terms;
  std::shared_ptr<const OffsetExpression> expression;
  Offset()=default;
  Offset(std::int64_t b):base(b){}
  std::string key()const;
  void appendKey(std::string& out)const;
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
// key()'s text, appended in place.
void Offset::appendKey(std::string& out)const {
  if(expression){out+=expression->key;return;}
  appendDecimal(out,base);
  for(auto [level,c]:terms){out+=',';appendDecimal(out,level);out+=':';appendDecimal(out,c);}
}
std::string Offset::key()const {
  std::string k;appendKey(k);
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
// valueKey's text, appended in place.
void appendValueKey(std::string& out,const v::Value& x) {
  appendDecimal(out,x.nodeId());out+=':';appendDecimal(out,x.instanceIndex());out+=':';appendDecimal(out,x.resultIndex());
  if(x.conversion()){out+='<';out+=v::typeConversionLabel(x);out+=':';appendValueKey(out,x.conversion()->source);out+='>';}
  for(const auto& i:x.indices()){out+='[';appendValueKey(out,i);out+=']';}
  if(x.threadSelector()){out+='{';appendValueKey(out,*x.threadSelector());out+='}';}
  if(x.pointer()){out+='|';out+=pointerLabel(*x.pointer());}
}
std::string valueKey(const v::Value& x) {
  std::string key;appendValueKey(key,x);
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
// The bytes one binding tensor compares (scalar_fold::ComparedOutput): runs
// of run_ contiguous bytes (the element and the dense axes inside it) at
// origin_ + sum_k i_k*stride_k, i_k < size_k, over the other axes, outermost
// first. Only nested layouts are represented (each stride a multiple of the
// element width and at least the extent of the axes inside it; the comparison
// admits no other layout), with at most kAxes outer axes. Queries: O(axes) per
// run visited, never per byte or element.
class ComparedBytes {
 public:
  static constexpr std::size_t kAxes=16;
  static std::optional<ComparedBytes> of(const s::ComparedOutput& layout) {
    const auto width=layout.element_bytes;
    if(!width || width>UINT64_MAX-layout.offset || layout.shape.size()!=layout.strides_bytes.size())return {};
    std::vector<std::pair<std::uint64_t,std::uint64_t>> axes;
    for(std::size_t i=0;i<layout.shape.size();++i) {
      if(!layout.shape[i])return {};
      if(layout.shape[i]>1)axes.emplace_back(layout.strides_bytes[i],layout.shape[i]);
    }
    std::sort(axes.begin(),axes.end());
    ComparedBytes result;result.origin_=layout.offset;result.run_=width;
    unsigned __int128 extent=width;
    for(const auto& [stride,size]:axes) {
      if(stride<extent || stride%width)return {};
      if(result.outer_.empty() && stride==result.run_)result.run_*=size;
      else result.outer_.emplace_back(stride,size);
      extent+=static_cast<unsigned __int128>(stride)*(size-1);
      if(extent>UINT64_MAX-layout.offset)return {};
    }
    if(result.outer_.size()>kAxes)return {};
    std::reverse(result.outer_.begin(),result.outer_.end());
    return result;
  }
  // The first run that ends after `at`, as [begin,end); none past the last.
  std::optional<std::pair<std::uint64_t,std::uint64_t>> run(std::uint64_t at) const {
    if(at<origin_)return std::pair(origin_,origin_+run_);
    auto x=at-origin_;
    std::array<std::uint64_t,kAxes> digit{};
    std::size_t level=0;
    for(;level<outer_.size();++level) {
      const auto [stride,size]=outer_[level];
      digit[level]=x/stride;
      if(digit[level]>=size)break;
      x-=digit[level]*stride;
    }
    if(level==outer_.size() && x<run_)return std::pair(at-x,at-x+run_);
    // Past the inner part at `level`: the next element of the level above.
    while(level-->0)if(digit[level]+1<outer_[level].second) {
      auto begin=origin_+(digit[level]+1)*outer_[level].first;
      for(std::size_t k=0;k<level;++k)begin+=digit[k]*outer_[k].first;
      return std::pair(begin,begin+run_);
    }
    return std::nullopt;
  }
  bool covers(std::uint64_t begin,std::uint64_t end) const {
    for(auto at=begin;at<end;) {
      const auto r=run(at);
      if(!r || r->first>at)return false;
      at=r->second;
    }
    return true;
  }
  bool meets(std::uint64_t begin,std::uint64_t end) const {
    const auto r=run(begin);
    return begin<end && r && r->first<end;
  }
  // The compared bytes of [begin,end), as maximal intervals in order.
  template<class F> void pieces(std::uint64_t begin,std::uint64_t end,F&& f) const {
    std::optional<std::pair<std::uint64_t,std::uint64_t>> open;
    for(auto at=begin;at<end;) {
      const auto r=run(at);
      if(!r || r->first>=end)break;
      const auto lo=std::max(r->first,at),hi=std::min(r->second,end);
      if(open && open->second==lo)open->second=hi;
      else {if(open)f(open->first,open->second);open=std::pair(lo,hi);}
      at=hi;
    }
    if(open)f(open->first,open->second);
  }
 private:
  std::uint64_t origin_=0,run_=0;
  std::vector<std::pair<std::uint64_t,std::uint64_t>> outer_;
};
// Probe start in a power-of-two table: every hash bit reaches the mask.
std::size_t keySlot(std::size_t hash,std::size_t mask) {
  const auto x=static_cast<std::uint64_t>(hash)*0x9e3779b97f4a7c15ULL;
  return static_cast<std::size_t>(x^(x>>32))&mask;
}
// A projection key is its text, interned once per request: two live
// handles have equal IDs exactly when their texts are equal (a probe hit
// compares the whole text). Handles are counted; an ID is reused only after
// its last holder (handle, KeySet member or memo entry) is gone. The memoized
// result lives in the key's slot, so a request makes one probe.
// Single-threaded, like the lowering.
class ProjectionKeys;
class ProjectionKey {
 public:
  static constexpr std::uint32_t none=std::numeric_limits<std::uint32_t>::max();
  ProjectionKey() noexcept=default;
  ProjectionKey(const ProjectionKey& o) noexcept;
  ProjectionKey(ProjectionKey&& o) noexcept:keys_(std::exchange(o.keys_,nullptr)),id_(std::exchange(o.id_,none))
  {}
  ProjectionKey& operator=(ProjectionKey o) noexcept {
    std::swap(keys_,o.keys_);std::swap(id_,o.id_);
    return *this;
  }
  ~ProjectionKey();
  std::uint32_t id()const noexcept{return id_;}
  friend bool operator==(const ProjectionKey& a,const ProjectionKey& b) noexcept {return a.id()==b.id();}
  // Maps keyed by ProjectionKey are ordered by ID; their consumers are order-free.
  friend std::strong_ordering operator<=>(const ProjectionKey& a,const ProjectionKey& b) noexcept {
    return a.id()<=>b.id();
  }
 private:
  friend class ProjectionKeys;
  friend class KeySet;
  ProjectionKey(ProjectionKeys* keys,std::uint32_t id) noexcept;  // retains
  ProjectionKeys* keys_=nullptr;
  std::uint32_t id_=none;
};
class ProjectionKeys {
 public:
  ProjectionKeys()=default;
  ProjectionKeys(const ProjectionKeys&)=delete;
  ProjectionKeys& operator=(const ProjectionKeys&)=delete;
  ProjectionKey intern(std::string_view text);
  const std::string& text(const ProjectionKey& k)const{return slot(k.id()).text;}
  const std::string& textOf(std::uint32_t id)const{return slot(id).text;}
  // Memoized results with unordered_map's contract: emplace inserts only an
  // absent key; at() throws the std::out_of_range unordered_map::at throws.
  s::Ref* find(const ProjectionKey& k){return findId(k.id());}
  s::Ref& at(const ProjectionKey& k) {
    if(auto* value=find(k))return *value;
    throw std::out_of_range("unordered_map::at");
  }
  void emplace(const ProjectionKey& k,const s::Ref& value);
  void erase(const ProjectionKey& k);
  // Memoized entries by dense position. Erasure moves the last entry into the
  // erased position, so positions hold only while no entry is added or erased.
  std::size_t size()const{return cached_.size();}
  s::Ref& valueAt(std::size_t i){return slot(cached_[i]).value;}
  s::Ref* findId(std::uint32_t id);
  // Stable compaction: keep(id,value) is called once per entry, in order.
  template<class Keep> void retainIf(Keep&& keep);
 private:
  friend class ProjectionKey;
  friend class KeySet;
  static constexpr unsigned chunk_bits=12;
  struct Slot {
    std::string text;           // a free ID keeps the capacity for reuse
    std::uint32_t tag=0;
    std::uint32_t refs=0;
    std::uint32_t cache_pos=0;  // 1-based position in cached_; 0: not memoized
    s::Ref value;
  };
  // The folded text hash places and filters probes; a hit still compares the
  // whole text, so the tag never decides identity.
  struct IndexSlot {std::uint32_t tag;std::uint32_t id;};
  static std::uint32_t tagOf(std::size_t hash){return std::uint32_t(hash^(hash>>32));}
  Slot& slot(std::uint32_t id){return chunks_[id>>chunk_bits][id&((1u<<chunk_bits)-1)];}
  const Slot& slot(std::uint32_t id)const{return chunks_[id>>chunk_bits][id&((1u<<chunk_bits)-1)];}
  void retain(std::uint32_t id) noexcept{++slot(id).refs;}
  void release(std::uint32_t id) noexcept;
  void grow();
  void index(std::uint32_t tag,std::uint32_t id) noexcept;
  void unindex(std::uint32_t tag,std::uint32_t id) noexcept;
  // Fixed chunks: slots never move, and growth never copies the records.
  std::vector<std::unique_ptr<Slot[]>> chunks_;
  std::uint32_t used_=0;
  std::vector<std::uint32_t> free_;  // capacity covers every ID: release never allocates
  std::vector<IndexSlot> index_;     // text tag -> ID; linear probing at load <= 1/2
  std::size_t indexed_=0;
  std::vector<std::uint32_t> cached_;
};
ProjectionKey::ProjectionKey(ProjectionKeys* keys,std::uint32_t id) noexcept:keys_(keys),id_(id){keys_->retain(id_);}
ProjectionKey::ProjectionKey(const ProjectionKey& o) noexcept:keys_(o.keys_),id_(o.id())
{if(keys_)keys_->retain(id_);}
ProjectionKey::~ProjectionKey(){if(keys_)keys_->release(id_);}
ProjectionKey ProjectionKeys::intern(std::string_view text) {
  const auto tag=tagOf(std::hash<std::string_view>{}(text));
  if(!index_.empty()) {
    const auto mask=index_.size()-1;
    for(auto i=keySlot(tag,mask);index_[i].id!=ProjectionKey::none;i=(i+1)&mask)
      if(index_[i].tag==tag && slot(index_[i].id).text==text)return ProjectionKey(this,index_[i].id);
  }
  // Allocate before publishing anything: a throw leaves the table unchanged.
  if(2*(indexed_+1)>index_.size())grow();
  if(free_.empty()) {
    if(used_==ProjectionKey::none)throw std::length_error("projection key table is full");
    if((used_>>chunk_bits)==chunks_.size())chunks_.push_back(std::make_unique<Slot[]>(std::size_t{1}<<chunk_bits));
    if(free_.capacity()<=used_)free_.reserve(std::max<std::size_t>(64,2*std::size_t(used_)));
    free_.push_back(used_++);
  }
  const auto id=free_.back();
  auto& s=slot(id);
  s.text.assign(text);  // may throw: the ID stays free
  free_.pop_back();s.tag=tag;s.refs=0;index(tag,id);++indexed_;
  return ProjectionKey(this,id);
}
void ProjectionKeys::release(std::uint32_t id) noexcept {
  auto& s=slot(id);
  if(--s.refs)return;
  unindex(s.tag,id);--indexed_;s.text.clear();free_.push_back(id);
}
void ProjectionKeys::grow() {
  std::vector<IndexSlot> grown(index_.empty()?std::size_t{64}:2*index_.size(),IndexSlot{0,ProjectionKey::none});
  grown.swap(index_);
  for(const auto& entry:grown)if(entry.id!=ProjectionKey::none)index(entry.tag,entry.id);
}
void ProjectionKeys::index(std::uint32_t tag,std::uint32_t id) noexcept {
  const auto mask=index_.size()-1;
  auto i=keySlot(tag,mask);
  while(index_[i].id!=ProjectionKey::none)i=(i+1)&mask;
  index_[i]={tag,id};
}
// Backward-shift deletion: every remaining probe chain stays unbroken.
void ProjectionKeys::unindex(std::uint32_t tag,std::uint32_t id) noexcept {
  const auto mask=index_.size()-1;
  auto i=keySlot(tag,mask);
  for(;index_[i].id!=id;i=(i+1)&mask)if(index_[i].id==ProjectionKey::none)return;
  for(auto j=(i+1)&mask;index_[j].id!=ProjectionKey::none;j=(j+1)&mask)
    if(((j-keySlot(index_[j].tag,mask))&mask)>=((j-i)&mask)){index_[i]=index_[j];i=j;}
  index_[i]={0,ProjectionKey::none};
}
s::Ref* ProjectionKeys::findId(std::uint32_t id) {
  auto& s=slot(id);
  return s.cache_pos?&s.value:nullptr;
}
void ProjectionKeys::emplace(const ProjectionKey& k,const s::Ref& value) {
  const auto id=k.id();
  auto& s=slot(id);
  if(s.cache_pos)return;
  auto copy=value;
  cached_.push_back(id);  // may throw: nothing is published yet
  s.value=std::move(copy);s.cache_pos=std::uint32_t(cached_.size());retain(id);
}
void ProjectionKeys::erase(const ProjectionKey& k) {
  const auto id=k.id();
  auto& s=slot(id);
  if(!s.cache_pos)return;
  const auto at=s.cache_pos-1;
  cached_[at]=cached_.back();slot(cached_[at]).cache_pos=at+1;cached_.pop_back();
  s.cache_pos=0;s.value=s::Ref{};release(id);
}
template<class Keep> void ProjectionKeys::retainIf(Keep&& keep) {
  std::size_t out=0,i=0;
  try {
    for(;i<cached_.size();++i) {
      const auto id=cached_[i];
      auto& s=slot(id);
      if(keep(id,s.value)){cached_[out++]=id;s.cache_pos=std::uint32_t(out);continue;}
      s.cache_pos=0;s.value=s::Ref{};release(id);
    }
  } catch(...) {
    // Only an invariant violation throws here; the remaining entries stay.
    for(;i<cached_.size();++i){const auto id=cached_[i];cached_[out++]=id;slot(id).cache_pos=std::uint32_t(out);}
    cached_.resize(out);throw;
  }
  cached_.resize(out);
}
// Retained IDs by open addressing (load <= 1/2, linear probing, backward-
// shift erase): no node allocation per member. Iteration order is
// unspecified, as with unordered_set; every consumer is order-free.
class KeySet {
 public:
  KeySet()=default;
  KeySet(const KeySet& o):keys_(o.keys_),ids_(o.ids_),size_(o.size_)
  {for(const auto id:ids_)if(id!=ProjectionKey::none)keys_->retain(id);}
  KeySet(KeySet&& o) noexcept:keys_(o.keys_),ids_(std::move(o.ids_)),size_(std::exchange(o.size_,0))
  {
    o.ids_.clear();
  }
  KeySet& operator=(KeySet o) noexcept {
    std::swap(keys_,o.keys_);ids_.swap(o.ids_);std::swap(size_,o.size_);
    return *this;
  }
  ~KeySet(){clear();}
  bool insert(const ProjectionKey& k);  // true when newly inserted (retained)
  void insert(const KeySet& other);
  bool contains(const ProjectionKey& k)const;
  bool contains(std::uint32_t id)const;
  void erase(const ProjectionKey& k);   // releases an erased member
  void clear() noexcept;                // releases every member; keeps the table
  std::size_t size()const{return size_;}
  template<class F> void forEachId(F&& f)const {
    for(const auto id:ids_)if(id!=ProjectionKey::none)f(id);
  }
 private:
  bool find(std::uint32_t id)const;
  bool insertId(ProjectionKeys* keys,std::uint32_t id);
  void grow();
  ProjectionKeys* keys_=nullptr;
  std::vector<std::uint32_t> ids_;
  std::size_t size_=0;
};
bool KeySet::find(std::uint32_t id)const {
  if(!size_)return false;
  const auto mask=ids_.size()-1;
  for(auto i=keySlot(id,mask);ids_[i]!=ProjectionKey::none;i=(i+1)&mask)if(ids_[i]==id)return true;
  return false;
}
void KeySet::grow() {
  std::vector<std::uint32_t> grown(ids_.empty()?std::size_t{8}:2*ids_.size(),ProjectionKey::none);
  grown.swap(ids_);
  const auto mask=ids_.size()-1;
  for(const auto id:grown)if(id!=ProjectionKey::none) {
    auto i=keySlot(id,mask);
    while(ids_[i]!=ProjectionKey::none)i=(i+1)&mask;
    ids_[i]=id;
  }
}
bool KeySet::insertId(ProjectionKeys* keys,std::uint32_t id) {
  // At the load limit, grow (allocating first) only for a new member.
  if(2*(size_+1)>ids_.size()) {
    if(find(id))return false;
    grow();
  }
  const auto mask=ids_.size()-1;
  for(auto i=keySlot(id,mask);;i=(i+1)&mask) {
    if(ids_[i]==id)return false;
    if(ids_[i]==ProjectionKey::none){ids_[i]=id;++size_;keys_=keys;keys->retain(id);return true;}
  }
}
bool KeySet::insert(const ProjectionKey& k) {return insertId(k.keys_,k.id());}
void KeySet::insert(const KeySet& other) {
  for(const auto id:other.ids_)if(id!=ProjectionKey::none)insertId(other.keys_,id);
}
bool KeySet::contains(const ProjectionKey& k)const {return find(k.id());}
bool KeySet::contains(std::uint32_t id)const {return find(id);}
void KeySet::erase(const ProjectionKey& k) {
  const auto id=k.id();
  if(size_) {
    const auto mask=ids_.size()-1;
    auto i=keySlot(id,mask);
    while(ids_[i]!=id && ids_[i]!=ProjectionKey::none)i=(i+1)&mask;
    if(ids_[i]==id) {
      // Backward-shift deletion: every remaining probe chain stays unbroken.
      for(auto j=(i+1)&mask;ids_[j]!=ProjectionKey::none;j=(j+1)&mask)
        if(((j-keySlot(ids_[j],mask))&mask)>=((j-i)&mask)){ids_[i]=ids_[j];i=j;}
      ids_[i]=ProjectionKey::none;--size_;keys_->release(id);
    }
  }
}
void KeySet::clear() noexcept {
  if(!size_)return;
  for(auto& id:ids_)if(id!=ProjectionKey::none){keys_->release(id);id=ProjectionKey::none;}
  size_=0;
}
struct RetryFoldDomain {std::vector<ProjectionKey> keep;};
class Lower {
  const v::Graph& input;
  const MemoryEliminationResult& memory;
  std::map<std::string,v::Value> allocation_bases;
  // Global pointer parameters that name an allocation another parameter names first (a tensor and a
  // view of it, such as V and its transpose Vt): symbol -> that first parameter's symbol. The input
  // contract makes every pointer to an object equal to its base, so they denote one value.
  std::map<std::string,std::string> pointer_aliases;
  // Contiguous stored initial-read domains; built once, never by a projection
  // query enumerating logical members. The address remains a Value operand.
  std::map<MemoryInstance,std::size_t> symbolic_read_domains;
  // Consecutive positioned reads of one node with one base whose constant parts form a
  // progression (MemoryEliminationResult::symbolic_read_positions), keyed by the first.
  struct PositionRun {std::size_t count;std::int64_t step;const SymbolicPosition* first;};
  std::map<MemoryInstance,PositionRun> position_runs;
  std::map<std::size_t,v::ThreadMembers> member_coordinates;
  struct SelectorRun {std::size_t end;std::int64_t thread;};
  std::map<std::string,std::map<std::size_t,SelectorRun>> member_selectors;
  const ScalarFoldOptions& options;
  ScalarSpecialization specialization{input,options.unit_lifting!=nullptr};
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
  // Lifting mode (unit_lifting.h): outputs [instantiated_end, outputs.size())
  // are parametric bodies, each of the CTA (coordinates) in output_ctas at the
  // same index; earlier outputs are instantiated. output_cta is the CTA of the
  // output being lowered (the convention of the per-CTA output grouping below:
  // the stored value's instance origin); `writer` is the CTA of its store
  // (class mode: the unit whose class the output is copied for; empty when the
  // output's records lie in more than one unit).
  std::size_t instantiated_end=0;
  using CtaCoordinates=std::array<std::uint64_t,3>;
  struct OutputOrigin {std::optional<CtaCoordinates> value,writer;};
  std::vector<OutputOrigin> output_ctas;
  OutputOrigin output_cta;
  // Lifting mode: per allocation, whether every member unit's δ of it is 0 (anchoredRegion).
  std::map<std::string,bool> anchored_regions;
  const std::array<std::uint64_t,kUnitSiteCount> sites_at_start=UnitSiteCounts();
  s::ReductionBuilder reductions{result.graph};
  // Unwritten bytes (refuseUnwrittenOutputs) are opaque: no finiteness fact lets a rule drop them.
  s::RealExpressionBuilder real_expressions{result.graph,reductions,[this]{limit();},
      [this](const std::string& allocation){return memory.unwritten_read_allocations.contains(allocation);}};
  // Interned projection keys with their memoized results. Declared before
  // every key holder below, so it is destroyed after all of them.
  ProjectionKeys keys;
  struct ViewInterval {
    std::uint64_t begin,end,prefix_end;
    std::size_t block;
  };
  std::unordered_map<const MemoryView*,std::vector<ViewInterval>> view_intervals;
  KeySet visiting;
  // Reused buffers for spelling a key in project(); no text outlives a call.
  std::string key_text,index_text;
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
  const InitialByteMap* initialByteMap(std::size_t node,std::size_t first,std::size_t last);
  struct PeriodicReadMap {
    const ReadSlicePattern* first;
    std::size_t count,period;
    std::int64_t byte_outer,member_outer;
  };
  std::map<std::pair<std::size_t,std::uint64_t>,std::vector<PeriodicReadMap>> periodic_read_maps;
  const PeriodicReadMap* periodicReadMap(std::size_t node,std::size_t first,
      std::size_t last,std::uint64_t offset,std::uint64_t width);
  std::shared_ptr<const IndexedSourceUse> memberUse(v::Value value,
      std::int64_t base,std::int64_t stride);
  IndexedSourceIdentity indexed_use_identity{valueKey};
  Offset specializeSingletons(Offset x) const;
  const IndexedValueBinding* indexedBinding(v::Value value);
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
  // One owner of AssembledRead piece extents for both raw-read consumers.
  AssembledPieceEnds assembled_piece_ends;
  IndexedSourceMatcher output_source_matcher{input,memory,assembled_piece_ends};
  PureSourceTemplates pure_source_templates{input,memory,&assembled_piece_ends,options.unit_lifting!=nullptr};
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
  // Ordered by key ID, then piece: lower_bound({key,0}) walks one key's pieces.
  using FoldPieceKey=std::pair<ProjectionKey,std::uint64_t>;
  struct FoldOwner;
  struct Projection {
    v::Value value;
    Offset offset;
    v::ElementType type;
    std::vector<std::uint64_t> domains;
    ProjectionKey key;
    KeySet dependencies = {};
    std::shared_ptr<const IndexedContext> indexed = {};
    std::shared_ptr<const FoldOwner> fold = {};
    std::uint32_t select_context = 0;
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
    std::uint64_t audit_attempts=0;
#endif
  };
  const Projection* active_projection=nullptr;
  // Inline frames transferred on dependency suspension, innermost first.
  // Keep their direct dependency roots and coordinate context instead of
  // rediscovering the entire inline chain when its leaf becomes available.
  std::vector<Projection> suspended_projections;
  // Set only between the depth-32 throw in project() and the driver's catch:
  // InlineFrame destructors then transfer their requests, innermost first.
  bool suspending=false;
  std::size_t suspension_base=0,suspension_frames=0;
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
      resume->indexed=task.indexed;resume->fold=task.fold;resume->select_context=task.select_context;
      owner.fold_owner=std::make_shared<FoldOwner>(FoldOwner{std::move(key),axis,segment,first,count,saved,std::move(resume)});
    }
    ~FoldScope(){owner.fold_owner=std::move(saved);}
  };
  void refineFold(unsigned axis,const std::vector<std::uint64_t>& cuts);
  KeySet* requested = nullptr;
  // A suspended reduction owns only its accumulator and the current domain's
  // dependencies. Completed domains are never reconstructed on resumption.
  // References live in the memo so collectFrontier can remap them; no raw
  // graph handles survive collection inside the continuation record.
  struct ReductionProgress {
    std::uint64_t next=0;
    ProjectionKey accumulator_key;
    KeySet pending;
    std::uint64_t sub_next=0;
  };
  std::map<ProjectionKey,ReductionProgress> reduction_progress;
  std::map<ProjectionKey,std::vector<algorithm::AffineDomain>> reduction_domain_plans;
  const ProjectionKey* projection_key=nullptr;
  struct ReductionRequests {
    KeySet*& current;
    KeySet* previous;
    ReductionRequests(KeySet*& target,ReductionProgress& progress)
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
    // The accumulator key is the text "$reduction:"+key, interned.
    auto [found,inserted]=reduction_progress.emplace(key,
        ReductionProgress{0,keys.intern("$reduction:"+keys.text(key)),{}});
    if(!inserted)throw std::logic_error("duplicate reduction continuation");
    keys.emplace(found->second.accumulator_key,initial);
    return found->second;
  }
  void commitReduction(ReductionProgress& progress,std::uint64_t next,s::Ref accumulator);
  void finishReduction(const ProjectionKey& key);
  void discardReductions() {
    for(const auto& [key,progress]:reduction_progress){(void)key;keys.erase(progress.accumulator_key);}
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
  void registerSpatialFamily(SpatialValues family);
  void registerSpatial(std::vector<v::Value> values);
  const SpatialValues& spatialValues(v::Value value) {
    auto found=spatial_values.find(valueKey(value));
    if(found==spatial_values.end())spatialReject();
    return found->second;
  }
  Offset spatialOffsets(const std::vector<std::uint64_t>& offsets);
  void prepareSpatial(v::Value value);
  static v::ElementType bit(unsigned n){return v::ElementType::bit(n);}
  void limit();
  std::pair<std::int64_t,std::int64_t> bounds(const Offset& x,std::size_t depth=0)const;
  std::optional<std::pair<Offset,Offset>> digit(const Offset& x,std::int64_t d,std::size_t depth=0)const;
  Offset quotient(const Offset& x,std::uint64_t width)const;
  Offset remainder(const Offset& x,std::uint64_t width)const;
  s::Ref emitOperation(std::string name,std::string type,std::vector<s::Ref> operands,
      std::string attr={});
  s::Ref offsetValue(const Offset& x);
  // ---- Cross-CTA value lifting (unit_lifting.h) ----------------------------
  [[noreturn]] void liftRefuse(std::string message) {
    fail("scalar_fold_unit_parameter_unsupported",std::move(message));
  }
  // `op` as the BlockIndex coordinate it is in lifting mode; else null.
  const v::CoordinateOp* liftedCoordinate(const v::Operation& op) const {
    if(!options.unit_lifting)return nullptr;
    const auto* coordinate=dynamic_cast<const v::CoordinateOp*>(&op);
    return coordinate && coordinate->kind()==v::CoordinateKind::BlockIndex?coordinate:nullptr;
  }
  s::Ref unitCoordinate(v::Axis axis,std::uint64_t coordinate,std::string type);
  const v::CoordinateOp* liftedBindingCoordinate(const IndexedValueBinding& binding,const Offset& position);
  // The InitialCell sites, classified for δ (every InitialCell of the lowering
  // is built through one of these two). Representative-absolute: the offset
  // is an Offset of the read's slot-resolved region, so a member reads it at
  // +δ(allocation). κ-covariant: the offset comes from the address value
  // graph (unit parameters and its own cells are instantiated), so no δ.
  // Instantiation tells them apart by an input parameter (not a unit's) in the
  // offset's cone: a covariant offset keeps the pointer root it is relative to
  // (read), and a representative-absolute one must not contain one (refused
  // here). The kind is thus part of the cell's hash-consed identity and
  // survives every rebuild (merging, canonical forms), which a side table of
  // tagged cells would not. O(offset cone) in lifting mode, nothing otherwise.
  s::Ref repAbsoluteCell(std::string allocation,s::Ref offset,std::string type) {
    if(options.unit_lifting && parametricCone(offset))
      liftRefuse("representative-absolute offset over an input parameter");
    return result.graph.initialCell(std::move(allocation),std::move(offset),std::move(type));
  }
  bool parametricCone(const s::Ref& root) const;
  void refuseUnwrittenOutputs();
  s::Ref covariantCell(std::string allocation,s::Ref offset,std::string type) {
    return result.graph.initialCell(std::move(allocation),std::move(offset),std::move(type));
  }
  // Phase (e) item boundary (first-item rule, ScalarFoldResult::FailureSite): one
  // normalized output group, contiguous scalar run or record, keyed by its
  // first record. O(1) per item.
  void beginItem(std::size_t record) {
    result.failure_site.record=record;++result.failure_site.items;
  }
  void pushOutput(ConstantRegion region,s::Ref body);
  // The symbolic base of the output records being lowered (OutputWrite::base).
  std::vector<SymbolicAddressTerm> output_base;
  // The atom of a symbolic term: the representative-absolute cell of its element of the
  // input object (an exact element, which lifting moves by the member's δ, like the
  // kernel's own read of that element).
  s::Ref atomCell(const SymbolicAddressTerm& term) {
    if(term.element_offset<0)fail("symbolic_output_placement_unsupported","negative element offset");
    return repAbsoluteCell(AllocationIdentity("global",term.object),
        result.graph.constant("u64",std::to_string(term.element_offset)),term.element_type);
  }
  s::Ref placed(s::Ref body);
  void setOutputCta(const OutputWrite& output) {
    if(!options.unit_lifting)return;
    output_cta.value=input.instanceOrigin(output.value).cta_coordinates;
    output_cta.writer=input.node(output.writer.node).instanceOrigin(output.writer.instance).cta_coordinates;
  }
  void includeRecordUnits(std::ranges::input_range auto&& records);
  static bool integralType(const std::string& type) {
    return type=="pred" || (!type.empty() && (type[0]=='b' || type[0]=='u' || type[0]=='s'));
  }
  bool realConversion(const s::Ref& use) const;
  // The rebuilds lifting refuses over a unit parameter, the same in instantiate and
  // memberBody: a real-valued operation or a Fold (the real-expression and reduction
  // builders decide from literal operands), and a use with a real-valued conversion
  // (realConversion). The unlifted lowering would build each differently from the unit's
  // literal coordinates.
  void refuseUnliftableNode(const s::Node& node) {
    if(node.kind==s::Kind::Fold)liftRefuse("Fold over a unit parameter");
    if(node.kind==s::Kind::Operation && !integralType(node.type))
      liftRefuse("real-valued "+node.name+" over a unit parameter");
  }
  void refuseUnliftableUse(const s::Ref& use) {
    if(realConversion(use))liftRefuse("real-valued conversion over a unit parameter");
  }
  // One unit to instantiate bodies for: its origin (the unit parameters), δ
  // per global allocation of its representative-absolute offsets and the
  // representative's CTA linear ids renamed to its own (pointer provenance
  // "cta=N"). A representative's own instance has neither; class mode
  // supplies both for every member.
  struct UnitInstance {
    CtaCoordinates origin{};
    const std::map<std::string,std::int64_t>* delta=nullptr;
    const std::map<std::uint64_t,std::uint64_t>* ctas=nullptr;
  };
  // δ(allocation) in one δ table (a unit's, or a member box digit's); an allocation without
  // an entry refuses lifting.
  std::int64_t deltaIn(const std::map<std::string,std::int64_t>& delta,const std::string& allocation) {
    const auto found=delta.find(allocation);
    if(found==delta.end())liftRefuse("no δ for allocation "+allocation);
    return found->second;
  }
  std::int64_t deltaOf(const UnitInstance& unit,const std::string& allocation) {
    return unit.delta?deltaIn(*unit.delta,allocation):0;
  }
  // The graph size at which completed outputs are collected under node pressure: three quarters of
  // the node budget.
  std::size_t pressureNodes() const {
    return options.max_nodes-std::max<std::size_t>(1,options.max_nodes/4);
  }
  bool anchoredRegion(const std::string& allocation);
  // A pointerLabel's fields: identity, addressing and the position of its
  // "cta=" value (the labels are length-prefixed, so parsing is exact).
  struct ProvenanceFields {std::string_view identity,addressing;std::size_t cta_begin=0,cta_end=0;};
  static std::optional<ProvenanceFields> provenanceFields(std::string_view label);
  // The δ key of a slot constant (a numeric literal naming a global allocation,
  // SlotAddressValue) with provenance `fields`; nullopt for any other use. δ is keyed as a
  // region's allocation (AllocationIdentity).
  std::optional<std::string> slotAllocation(const s::Ref& use,const ProvenanceFields& fields) const {
    if(result.graph.node(use).kind!=s::Kind::Constant || fields.identity.empty() || fields.addressing!="global")
      return std::nullopt;
    return AllocationIdentity(fields.addressing,fields.identity);
  }
  s::Ref moveSlot(s::Ref use,std::int64_t delta);
  s::Ref instantiateUse(s::Ref use,const UnitInstance& unit);
  s::Ref shiftOffset(const s::Ref& offset,std::int64_t delta);
  // A unit parameter's axis and offset (ParseUnitCoordinateParameter); any other spelling
  // refuses lifting.
  UnitCoordinate unitParameter(const std::string& name) {
    const auto parameter=ParseUnitCoordinateParameter(name);
    if(!parameter)liftRefuse("malformed unit parameter "+name);
    return *parameter;
  }
  s::Ref instantiate(const s::Ref& root,const UnitInstance& unit,const std::vector<std::uint8_t>& depends,
                     std::unordered_map<s::Id,s::Ref>& memo);
  CtaCoordinates unitOrigin(const CtaCoordinates& cta) const;
  // Instantiation marks (one ascending pass over the graph; node IDs are
  // topological). kKappaDependent: a unit parameter below. kCellDependent: a
  // cell or a decorated use that moves by δ / is renamed below (only when
  // `moves`, i.e. some unit has δ or renaming). kCovariantCell: an InitialCell
  // whose offset contains an input parameter (the symbolic read's base).
  static constexpr std::uint8_t kKappaDependent=1,kCellDependent=2,kCovariantCell=4;
  std::vector<std::uint8_t> instantiationMarks(bool moves) const;
  // Instantiate every pending output (instantiateClasses).
  void instantiatePending() {
    if(!options.unit_lifting)return;
    if(instantiated_end>=result.outputs.size()){output_ctas.clear();return;}
    instantiateClasses();
  }
  // Class mode: a class whose units fill one box, the representative's origin + unit_shape x
  // [0, n) per axis, factored per axis into mixed-radix digits (innermost first) along which
  // every allocation's δ is linear: the unit at digit values v has δ(a) = sum v_d x delta_d(a)
  // and origin rep + sum v_d x step_d on each digit's axis. nullopt when the members do not
  // fill such a box (each unit once) or some δ is not of that form. O(units x δ entries x
  // log units): δ additivity over the axes is checked per unit, a digit's linearity and the
  // separability E(p q + r) = E(p q) + E(r) of the sequence left over per digit level.
  struct MemberDigit {unsigned axis=0;std::uint64_t count=1,step=0;std::map<std::string,std::int64_t> delta;};
  struct MemberBox {std::vector<MemberDigit> digits;};
  std::optional<MemberBox> memberBox(const CtaCoordinates& rep,const std::vector<UnitLifting::Member>& members) const;
  // A position in a member box: dense digits are the family's Index levels, the others are
  // fixed (arm) values; old levels (the representative's domain and Fold binders) move up by
  // `shift`, the number of dense digits.
  struct MemberPoint {
    std::vector<std::optional<std::uint32_t>> level;
    std::vector<std::uint64_t> value;
    std::uint32_t shift=0;
  };
  s::Ref addMemberTerms(s::Ref base,std::optional<std::int64_t> constant,const std::string& type,
      const MemberBox& box,const MemberPoint& p,const std::function<std::int64_t(const MemberDigit&)>& coefficient);
  // Rebuild marks for member families (one ascending pass; node IDs are topological):
  // kFamilyRebuild for an Index, a Fold, a unit parameter, a cell, a decorated use or
  // anything above one; kFamilyParametric for an input parameter (not a unit's) below;
  // kFamilyKappa for a unit parameter below (instantiationMarks' kKappaDependent).
  static constexpr std::uint8_t kFamilyRebuild=1,kFamilyParametric=2,kFamilyKappa=4;
  static constexpr std::uint8_t kFamilyUnitParameter=kFamilyRebuild|kFamilyKappa;
  std::vector<std::uint8_t> familyMarks() const;
  struct MemberFamilyUnavailable {};
  s::Ref memberBody(const s::Ref& root,const MemberBox& box,const MemberPoint& p,const CtaCoordinates& rep,
                    const std::vector<std::uint8_t>& marks,std::unordered_map<s::Id,s::Ref>& memo);
  // Lifted outputs stay within the lowering's node budget; no collection runs while they are
  // built. O(1), after each lifted family or copy.
  void liftBudget() {
    if(result.graph.size()>options.max_nodes)
      fail("expression_node_limit_exceeded","lifted member outputs exceed the scalar graph node budget");
  }
  // Exact canonical forms (RealExpressionBuilder::canonicalize) of the bodies of outputs
  // `owners`, before their families merge. Every merge types bf16 field reads, so outputs
  // that unpack the low and high fields of packed pairs become shifted copies; only
  // mergeRepresentativeFamilies, which says why, also makes ideal-real sums canonical.
  // O(reachable nodes), plus the sorts.
  void canonicalizeOutputs(const std::vector<std::size_t>& owners,s::CanonicalForms forms) {
    std::vector<s::Ref> bodies;bodies.reserve(owners.size());
    for(const auto i:owners)bodies.push_back(result.outputs[i].value);
    real_expressions.canonicalize(bodies,forms);
    for(std::size_t k=0;k<owners.size();++k)result.outputs[owners[k]].value=bodies[k];
  }
  // The outputs consolidation merges: those with an output domain.
  std::vector<std::size_t> familyOutputs() const {
    std::vector<std::size_t> owners;owners.reserve(output_domains.size());
    for(const auto& [i,domain]:output_domains)if(i<result.outputs.size())owners.push_back(i);
    return owners;
  }
  bool mergeRepresentativeFamilies(const std::map<CtaCoordinates,MemberBox>* classes,bool pending=false);
  bool copiesFit(const std::vector<UnitLifting::Member>& members,const std::vector<s::Ref>& roots,
                 const std::vector<std::uint8_t>& moving,std::uint64_t& planned) const;
  void instantiateClasses();
  bool memberFamilies(std::size_t i,const MemberBox& box,const CtaCoordinates& rep,const std::optional<OutputDomain>& domain,
      const std::vector<std::uint8_t>& marks,
      std::map<std::tuple<CtaCoordinates,std::vector<bool>,std::size_t>,std::unordered_map<s::Id,s::Ref>>& points);
  std::uint64_t shiftRegion(const ConstantRegion& region,const std::optional<OutputDomain>& domain,std::int64_t delta);
  std::optional<std::int64_t> known(v::Value x,unsigned depth=0)const{
    return knownAt(x,depth,indexed_context?indexed_context->use.get():nullptr);
  }
  std::optional<std::int64_t> knownAt(v::Value x,unsigned depth,const IndexedSourceUse* use)const;
  s::Ref cast(s::Ref x,const v::ElementType& type,std::string step="Bitcast"){
    auto target=type.toString();if(result.graph.type(x)==target && step=="Bitcast")return x;
    return result.graph.convert(std::move(x),target,std::move(step));
  }
  bool zeroMatrixLiteral(const v::Value& value) const;
  // Whether every byte of a Load is backed by zero matrix literals, through its
  // normalized read only (the substitution read() projects). Any other source,
  // initial memory or an uncertain view is not zero. Memoized per operand read
  // instance; nested source reads are not, so the bounded answer is order-free.
  struct MemoryInstanceHash {
    std::size_t operator()(const MemoryInstance& x)const noexcept {
      const auto h=std::hash<std::size_t>{}(x.node);
      return h^(std::hash<std::size_t>{}(x.instance)+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2));
    }
  };
  std::unordered_map<MemoryInstance,bool,MemoryInstanceHash> zero_reads;
  bool zeroRead(v::Value value,unsigned depth=0);
  std::optional<s::Ref> zeroProductMma(const v::OperandsRef& args,bool accumulate,std::size_t accumulator,
      const Offset& index,std::int64_t cell,const v::ElementType& dtype,const v::ElementType& element);
  v::Value selected(v::Value value);
  s::Ref scalar(v::Value x){
    const auto& type=input.type(x);
    if(!type.isScalar())fail("scalar_fold_matrix_operand","scalar operator received an unprojected matrix");
    return project(x,Offset{},type.element());
  }
  s::Ref initial(const InitialMemorySymbol& x,const Offset& offset,const v::ElementType& type);
  s::Ref source(const ViewSource& x,const Offset& offset,const v::ElementType& type){
    if(const auto* i=std::get_if<InitialMemorySymbol>(&x))return initial(*i,offset,type);
    return project(std::get<v::Value>(x),offset,type);
  }
  const std::vector<ViewInterval>& intervalsFor(const MemoryView& view);
  std::vector<std::size_t> coveringBlocks(const MemoryView& view,std::uint64_t begin,std::uint64_t end);
  std::uint64_t readSpan(v::Value value,const Offset& offset,std::uint64_t stride,
                         const v::ElementType& type,std::uint64_t count,unsigned depth=0);
  std::optional<Offset> viewOffset(const ViewBlock& block,const Offset& offset,const v::ElementType& type);
  std::optional<s::Ref> viewBlock(const MemoryView& view,const ViewBlock& block,
                                 const Offset& offset,const v::ElementType& type){
    auto mapped=viewOffset(block,offset,type);
    if(!mapped)return {};
    return source(view.sources.at(block.source),*mapped,type);
  }
  s::Ref view(const MemoryView& view,const Offset& offset,const v::ElementType& type);
  Offset mappedCoordinate(const IndexedValueBinding& binding,const Offset& index);
  s::Ref indexedView(const MemoryView& view,const Offset& offset,const v::ElementType& type);
  const ReadRegion* coveringRegion(const PartitionedRead& split,const Offset& offset,const v::ElementType& type);
  // A partition whose regions repeat with one period P (TMA zero fill of a
  // partial tile: per matrix row, initial bytes then zero bytes) re-encoded as
  // one strided region per phase: phase j of n periods gets the outer axis
  // (n, source step, P). Exact: every byte keeps its source byte; a zero-literal
  // phase reads its first period's literal, whose bytes are zero too. Only
  // maximal runs of at least two whole periods are re-encoded, with the next
  // region of the same shape as the one candidate period. Built once per
  // partition in O(R log R + checked phases), R = stored regions; consulted
  // only when the stored partition has no region for a query.
  std::map<const PartitionedRead*,std::optional<PartitionedRead>> periodic_partitions;
  const PartitionedRead* periodicPartition(const PartitionedRead& split);
  struct PartitionLattice {const MemoryView* view;Offset offset;};
  const MemoryView* partitionGrid(const PartitionedRead& split,v::Value reader,
                                  const v::ElementType& type);
  std::optional<PartitionLattice> partitionLattice(
      const PartitionedRead& split,v::Value reader,const Offset& offset,const v::ElementType& type);
  s::Ref partition(const PartitionedRead& split,v::Value reader,const Offset& offset,const v::ElementType& type);
  void refinePartitionDomain(const PartitionedRead& split,const Offset& offset,const v::ElementType& type);
  // Add/Subtract and bitcast steps followed from an address to its cvta root. A deeper
  // address is not decomposed; the caller's whole-address form is exact as well.
  static constexpr unsigned kCvtaTermDepth=16;
  std::optional<s::Ref> cvtaRelative(v::Value address,const v::Value& root,
      const std::shared_ptr<const IndexedSourceUse>& use,unsigned depth);
  bool samePointer(const v::Value& a,const v::Value& b) const;
  const std::pair<const MemoryInstance,PositionRun>* positionRun(std::size_t node,std::size_t first,std::size_t last) const;
  s::Ref positionedCell(const SymbolicPosition& position,const Offset& constant,const std::string& type);
  s::Ref integerValue(const Offset& position,const v::ElementType& type);
  std::optional<std::int64_t> exactRegionOrigin(const v::Value& leaf,const RegionFragmentLoad& fragment) const;
  // The cell of an exact region fragment: representative-absolute at origin + place, a plain
  // Offset, so that it merges and lifts like any slot-resolved initial read.
  s::Ref exactRegionCell(const std::string& allocation,std::int64_t origin,const Offset& place,
      const v::ElementType& type,const v::ElementType* numerical) {
    auto cell=repAbsoluteCell(allocation,offsetValue(plus(Offset(integer(std::uint64_t(origin))),place)),type.toString());
    return numerical?result.graph.convert(cell,numerical->toString(),"NumericInterpretation"):cell;
  }
  s::Ref regionFragmentCell(const v::Value& leaf,const Offset& place,const Offset& offset,
      const v::ElementType& type,const v::ElementType* numerical,
      std::shared_ptr<const IndexedSourceUse> origin_use=nullptr,
      std::shared_ptr<const IndexedSourceUse> guard_use=nullptr);
  s::Ref read(v::Value value,const Offset& offset,const v::ElementType& type);
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
  std::optional<s::Ref> plannedScalarReduction(v::Value value,const v::ElementType& type);
  // ---- Lane reductions --------------------------------------------------------------------------
  // A predicate BitOr / BitAnd flagged lane_own is the value view's vote over the lanes of a
  // warp: operand 1 is the chain of the lanes' terms, operand 0 the issuing lane's own term.
  // The flag is only a hint: here operand 0 must be found among the chain's terms (compared as
  // resolved values, each at the thread it is read from), and then the value is the chain's.
  // When the terms are also one family over the lane index (the planner's member template, as
  // for a chain of floating terms above), the value is
  //   Fold(Max | Min, initial = own, body = the k-th lane's term, binder at the current depth),
  // and no lane's term is lowered on its own. Per vote instance: O(lanes) for the spine, then
  // the planner's cost for the terms (its plans are cached per value); planned once.
  struct LaneReductionPlan {
    s::Reducer reducer;
    v::Value own,chain;
    std::vector<v::Value> terms;
    bool planned=false;
    std::vector<IndexedSourceDomain> pieces;  // empty: not one family, the value is the chain's
  };
  std::unordered_map<std::string,std::optional<LaneReductionPlan>> lane_reduction_plans;
  std::optional<v::Value> laneTerm(const v::Value& term);
  const v::BinaryOp* laneRoot(const v::Value& value,const v::ElementType& type) const;
  LaneReductionPlan* laneReduction(v::Value value,const v::ElementType& type);
  std::optional<s::Ref> plannedLaneReduction(LaneReductionPlan& plan,const v::ElementType& type);
  // The exact value of a scalar-graph constant seen through its exact (bit-preserving or
  // representable) conversions, e.g. an f64 parameter read as b64 bits.
  std::optional<equivalence::ConstantValue> exactConstant(const s::Ref& r) const {
    const auto& n=result.graph.node(r);
    if(n.kind!=s::Kind::Constant || !r.pointer_provenance.empty())return {};
    return equivalence::constantValue(n.type,n.name,r.conversion,r.converted_type);
  }
  bool contractibleProduct(v::Value x) const;
  std::optional<s::Ref> constantArithmetic(const v::Operation& op,std::span<const v::Value> args,
      const std::vector<s::Ref>& operands,const v::ElementType& type);
  std::optional<s::Ref> constantConversion(const s::Ref& base,const v::ConversionAttributes& attributes,
      const v::ElementType& target);
  s::Ref pure(v::Value value,const Offset& index,const v::ElementType& type);
  static std::uint8_t arithmeticModes(const v::ArithmeticAttributes& attributes);
  // An operand the builders kept, or a constant they respelled (constant normalization).
  bool sameOperand(const s::Ref& made,const s::Ref& given) const {
    return made==given || (result.graph.node(made).kind==s::Kind::Constant && result.graph.node(given).kind==s::Kind::Constant);
  }
  // Marks `made` as the image of a floating PTX instruction (scalar-graph provenance, not identity)
  // when the expression builders returned operation `name` on the instruction's own operands. A
  // Fold, a rewritten exponential factor, a folded constant or a returned operand is no image and
  // stays unmarked. O(operands).
  void markInstruction(const s::Ref& made,const char* name,const std::vector<s::Ref>& operands,std::uint8_t modes) {
    const auto& node=result.graph.node(made);
    if(node.kind!=s::Kind::Operation || node.name!=name || node.operands.size()!=operands.size())return;
    for(std::size_t i=0;i<operands.size();++i)if(!sameOperand(node.operands[i],operands[i]))return;
    result.graph.markInstruction(made,modes);
  }
  void markArithmeticInstruction(const s::Ref& made,const v::BinaryOp& binary,const std::vector<s::Ref>& operands,
      std::span<const v::Value> values);
  std::optional<std::vector<algorithm::AffineDomain>> regionDomains(const PartitionedRead& split,
      const Offset& offset,std::uint64_t stride,const v::ElementType& type,std::uint64_t count);
  std::optional<std::vector<algorithm::AffineDomain>> reductionDomains(
      v::Value value,const Offset& offset,std::uint64_t stride,
      const v::ElementType& type,std::uint64_t count);
  // Lane selections and same-width bitcasts constantBits crosses; past them a value counts
  // as not constant, which only keeps it symbolic.
  static constexpr unsigned kConstantBitsDepth=8;
  std::optional<equivalence::ConstantValue> constantBits(v::Value value);
  std::optional<v::Value> uniformConstant(std::span<const IndexedSourceSlice> slices,std::uint64_t width);
  s::Ref warpMma(v::Value value,const Offset& index);
  // tcgen05.mma.sp metadata validity (PTX ISA 9.7.18.10.9.1-.3): each 4-bit field of the
  // metadata matrix must hold one of the listed meaningful index pairs; any other value is
  // undefined behavior. The metadata operand is the bytes of its Tensor Memory footprint
  // (TcgenMma, LinkMmaMemory), every 4-bit field of which this MMA consumes. Its fields are
  // projected through memory elimination to their sources: an input object's cell ranges
  // over the field values its tensor_properties declare (every value when it declares
  // none), a literal (TMA out-of-bounds fill or a kernel constant) is itself, and Tensor Memory
  // that no store wrote before the MMA reads it holds any value. A field that
  // can hold a value outside the list is reachable undefined behavior, with that value as
  // the witness (the input contract lets that field take it); a field whose source is
  // anything else leaves validity unproven. Work: per sparse MMA instance, one projection
  // and one classification per source span of its metadata (readSpan), at most
  // 128 x cta_group x 8 bytes; it stops at the first witness.
  static constexpr std::array<unsigned,7> kSparseMetadataMeaningful{0b0100,0b1000,0b1100,0b1001,0b1101,0b0110,0b1110};
  static std::bitset<16> meaningfulSparseMetadata() {
    std::bitset<16> set;for(const auto value:kSparseMetadataMeaningful)set.set(value);return set;
  }
  static std::optional<std::string> globalObject(std::string_view identity);
  static std::bitset<16> nibbleDomain(const TensorPropertyFact* fact,unsigned position);
  struct MetadataByteSource {
    bool known=false;
    std::array<std::bitset<16>,2> fields{}; // possible values of the low and high 4-bit field
    std::string source,object,reason;
    const TensorPropertyFact* fact=nullptr;
    std::optional<std::uint64_t> object_byte_offset;
  };
  MetadataByteSource metadataByteSource(const s::Ref& byte);
  std::optional<UndefinedBehaviorWitness> sparse_metadata_witness;
  static void sparseMetadataWording(const std::string& kind,UndefinedBehaviorWitness& w);
  void checkSparseMetadata();
  s::Ref scaledMma(v::Value value,const Offset& index,const v::TcgenMmaAttributes& attr);
  s::Ref mma(v::Value value,const Offset& index);
  // ---- Selection contexts ------------------------------------------------------
  // select(c, a, b) is a where c holds and b elsewhere, so inside arm a, c is true and inside arm
  // b it is false: a Select there whose condition c fixes has one possible arm. Lowering a Select
  // in the unspecified cone (below) gives each arm the predicates its condition's value fixes,
  // its selection context; a Select under it whose condition the context fixes lowers only the
  // arm that runs: select(c, a, f(select(c, u, v))) = select(c, a, f(v)) and the symmetric forms,
  // also when u has unspecified bits. Predicates are identified by the scalar graph's hash-consed
  // Refs: equal Refs are one function of the inputs, so equal in every execution (and in every
  // thread that computes them). An arm's context holds only its own Select's literals: it ends at
  // the next Select it does not fix, so a cone node is lowered at most once per arm of its nearest
  // such Select (contexts never multiply), and a projection outside the cone is context-free.
  static constexpr std::size_t kSelectContextLiterals=16;
  // BitNot and constant-armed Select wrappers decidedPredicate strips from one predicate; a
  // deeper predicate stays undecided, so both arms are lowered.
  static constexpr std::size_t kPredicateSteps=16;
  struct SelectContext {std::vector<std::pair<s::Ref,bool>> literals;};
  // The live contexts by serial (0: no context), interned by literal set. Serials are never
  // reused, since projection keys name them: an output collection retires (erases) every
  // context, and a later context takes the next serial.
  std::unordered_map<std::uint32_t,SelectContext> select_contexts;
  std::unordered_map<std::string,std::uint32_t> select_context_ids;
  std::uint32_t next_select_context=1;
  std::uint32_t select_context=0;
  struct SelectScope {
    Lower& owner;std::uint32_t saved;
    SelectScope(Lower& o,std::uint32_t context):owner(o),saved(o.select_context){o.select_context=context;}
    ~SelectScope(){owner.select_context=saved;}
  };
  // Value nodes from which an OpaqueValueOp is reachable through operands (conversions, component
  // indices and thread selections included; a load is a leaf). Built on first use from users of
  // each node in compressed rows: O(nodes + stored operand references) time and memory, nothing
  // when the graph has no OpaqueValueOp.
  std::vector<char> unspecified_cone;
  bool unspecified_cone_built=false;
  void buildUnspecifiedCone();
  bool inUnspecifiedCone(const v::Value& value);
  std::optional<bool> predicateConstant(const s::Ref& x) const;
  // The predicate operation that is x (a use without conversion or provenance), if any.
  const s::Node* predicateOperation(const s::Ref& x) const {
    if(!x.conversion.empty() || !x.converted_type.empty() || !x.pointer_provenance.empty())return nullptr;
    const auto& node=result.graph.node(x);
    return node.kind==s::Kind::Operation && node.type=="pred" && node.attributes.empty()?&node:nullptr;
  }
  std::optional<bool> decidedPredicate(s::Ref x) const;
  std::uint32_t selectContextFor(const s::Ref& condition,bool value);
  static std::string selectContextText(const std::vector<std::pair<s::Ref,bool>>& literals);
  // An OpaqueValue (a register read before its first assignment, an edited tensor map) has bits that
  // neither PTX nor the model determines, hence no scalar semantics: name it wherever a projection reaches it.
  [[noreturn]] void unspecified(v::Value value,const v::OpaqueValueOp& opaque) {
    const auto line=input.definingNode(value).origin().source_line;
    fail("unspecified_value_dependence","the projected value depends on bits that are unspecified: "+
        opaque.provenance()+(line?" (PTX line "+std::to_string(*line)+")":""));
  }
  s::Ref whole(v::Value value,const Offset& index);
  std::optional<std::pair<PackedStorageDecoder::Fields,v::ElementType>> wideRecordFields(v::Value value) const;
  std::optional<std::vector<OutputWrite>> wideRecordRun(std::size_t begin,std::size_t& end);
  std::optional<PackedStorageDecoder::Fields> packedFields(v::Value value) const {
    // `known` depends on this projection's binding environment. Keep its
    // decoder local; view normalization owns a pass-wide context-free decoder.
    return PackedStorageDecoder(input,[this](auto x){return known(x);}).decode(value);
  }
  v::Value memberValue(v::Value value,std::size_t member) const;
  static bool sameMemberTemplate(const v::Value& a,const v::Value& b);
  // A template read of load `value` at members position = base + step x Index(level) (one
  // binder, step > 0) and constant result byte `lo`, where each member has its own stored
  // read-slice pattern and no stored run or periodic map spans them (e.g. a lane reading the
  // value another warp's thread wrote). The members are grouped into maximal runs (of at least
  // kMinAffineRun members, since any two are affine) along which the patterns name one source
  // template whose source member and source byte offset are affine in the binder, checked at
  // every member. One run: the read is that source, projected as a member
  // template at those members and offsets. Several runs, one longer than one member: the output
  // coordinate is refined at the run boundaries (refinePartitionDomain). Otherwise nothing, and
  // the caller refines at every stored pattern. Exact: every member's own stored mapping is
  // validated; nothing is extrapolated. O(members of the query) pattern lookups.
  static constexpr std::uint64_t kMinAffineRun=3;
  std::optional<s::Ref> memberAffineRead(const v::Value& value,const Offset& position,std::int64_t lo,
      std::uint64_t width,const v::ElementType& desired);
  s::Ref projectMemberTemplate(v::Value value,const Offset& offset,const v::ElementType& desired);
  Offset componentOffset(s::Ref root);
  static bool integerKind(const v::ElementType& e) {
    return e.kind()==v::ElementKind::Bit || e.kind()==v::ElementKind::UnsignedInteger ||
        e.kind()==v::ElementKind::SignedInteger;
  }
  // One byte-slice step (sliceStep): the slice of `source` at `offset` that equals the request.
  // For an operation node, `operand` is the operand it reads (its child under an Operands use).
  struct SliceStep {v::Value source;Offset offset;std::size_t operand=0;};
  std::optional<SliceStep> sliceStep(const v::Value& value,const Offset& offset,const v::ElementType& desired,
      std::optional<v::Value> conversion_source={}) const;
  bool reachesLoad(v::Value x) const;
  std::optional<s::Ref> projectSlice(const v::Value& value,const Offset& offset,const v::ElementType& desired);
  s::Ref projectImpl(v::Value value,Offset offset,const v::ElementType& desired);
  s::Ref evaluateProjection(Projection& task);
  s::Ref project(v::Value value,Offset offset,const v::ElementType& type);
  void scheduleCollection() {
    const auto cap=std::max<std::size_t>(1,options.max_nodes/4);
    const auto interval=std::min(cap,std::max(options.construction_collection_interval,
                                            result.graph.size()/2));
    next_collection=std::min(options.max_nodes,result.graph.size()+interval);
  }
  void collectFrontier(const std::vector<Projection>& work,const ProjectionKey& root);
  std::vector<std::optional<s::Id>> observedCache();
  std::size_t retainObserved(const std::vector<std::optional<s::Id>>& weak_cache);
  void collectBeforeFinalLifting();
  void collectCompletedOutputs(bool force=false);
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
      pushOutput(region,body);
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
        OffsetTerms terms;
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
  // Partial results held across top-level projections, which may collect the graph and
  // expire raw Refs: collectFrontier roots and remaps them. A stack of PinnedRoot entries;
  // none is held across an output collection.
  std::vector<s::Ref> pinned_roots;
  struct PinnedRoot {
    Lower& owner;std::size_t slot;
    PinnedRoot(Lower& o,s::Ref value):owner(o),slot(o.pinned_roots.size()){o.pinned_roots.push_back(std::move(value));}
    ~PinnedRoot(){owner.pinned_roots.pop_back();}
    PinnedRoot(const PinnedRoot&)=delete;
    PinnedRoot& operator=(const PinnedRoot&)=delete;
    s::Ref& value(){return owner.pinned_roots[slot];}
  };
  void reducedOutput(const OutputWrite& output);
  // Equivalence runs: the bytes the binding compares, per allocation; empty
  // in lifting mode, whose regions are the representative's.
  std::map<std::string,ComparedBytes> compared_bytes;
  enum class Compared {All,None,Some};
  Compared comparedIn(const std::string& allocation,std::uint64_t offset,std::uint64_t width,
                      const std::vector<std::uint64_t>& shape={},const std::vector<std::uint64_t>& strides={}) const;
  Compared comparedIn(const OutputWriteFamily& family) const;
  void projectComparedBytes(const OutputWrite& output,const v::ElementType& element);
  void outputFamily(const OutputWrite& output,std::uint64_t begin,std::uint64_t count,
                    const v::ElementType& type,std::uint64_t bytes);
  // The widest scalar an output record holds (64-bit integers and floats).
  static constexpr std::uint64_t kWidestScalarBytes=8;
  // An integer or bit record element wider than 64 bits (a u128 built from two u64 halves for
  // st.shared.b128, then copied out by a TMA store) has no scalar type: its bytes are output as
  // one-element families of unsigned 64-bit fields at constant offsets, which the output
  // partition reads as packed lanes. Exact: each field is a bit slice of the stored value.
  // O(record bytes / 8) families.
  void wideOutput(const OutputWrite& output,Compared coverage) {
    const auto field=v::ElementType::unsignedInteger(8*kWidestScalarBytes);
    if(coverage==Compared::Some){projectComparedBytes(output,field);return;}
    for(std::uint64_t done=0;done<output.region.bytes;done+=kWidestScalarBytes)
      outputFamily(output,done,1,field,kWidestScalarBytes);
  }
  bool spatialOutput(std::span<const OutputWrite> outputs);
 public:
  Lower(const v::Graph& g,const MemoryEliminationResult& m,const ScalarFoldOptions& o):input(g),memory(m),options(o){
    if(options.compared_outputs && !options.unit_lifting)
      for(const auto& [allocation,layout]:*options.compared_outputs)
        if(auto bytes=ComparedBytes::of(layout))compared_bytes.emplace(allocation,std::move(*bytes));
    for(auto it=memory.symbolic_initial_reads.begin();it!=memory.symbolic_initial_reads.end();) {
      const auto begin=it;auto end=it->first.instance+1;++it;
      while(it!=memory.symbolic_initial_reads.end() && it->first.node==begin->first.node &&
          it->first.instance==end && it->second.allocation==begin->second.allocation &&
          it->second.offset==begin->second.offset && it->second.bytes==begin->second.bytes) {++end;++it;}
      symbolic_read_domains.emplace(begin->first,end);
    }
    for(auto it=memory.symbolic_read_positions.begin();it!=memory.symbolic_read_positions.end();) {
      const auto begin=it;PositionRun run{1,0,&it->second};++it;
      while(it!=memory.symbolic_read_positions.end() && it->first.node==begin->first.node &&
          it->first.instance==begin->first.instance+run.count && it->second.base==run.first->base &&
          it->second.constant.allocation==run.first->constant.allocation &&
          it->second.constant.bytes==run.first->constant.bytes) {
        const auto delta=static_cast<__int128>(it->second.constant.offset)-run.first->constant.offset;
        if(run.count==1 && delta>=INT64_MIN && delta<=INT64_MAX)run.step=static_cast<std::int64_t>(delta);
        if(delta!=static_cast<__int128>(run.step)*run.count)break;
        ++run.count;++it;
      }
      position_runs.emplace(begin->first,run);
    }
    if(m.symbolic_initial_reads.empty())return;
    for(const auto& node:g.nodes()) {
      const auto* parameter=dynamic_cast<const v::InputParameterOp*>(&node.operation());
      if(!parameter || !parameter->pointer())continue;
      const auto& p=*parameter->pointer();
      if(p.owner_cta || p.owner_thread || p.addressing!="global")continue;
      const auto [it,inserted]=allocation_bases.emplace(AllocationIdentity(p.addressing,p.identity),node.result());
      if(!inserted) {
        const auto& prior=dynamic_cast<const v::InputParameterOp&>(g.definingNode(it->second).operation());
        if(prior.symbol()!=parameter->symbol())pointer_aliases.emplace(parameter->symbol(),prior.symbol());
      }
    }
  }
  bool spatialStoreFamily(const OutputWriteFamily& family);
  void storeFamily(const OutputWriteFamily& family);
  void comparedStoreFamily(const OutputWriteFamily& family);
  void validateRadixPartition(const PartitionedRead& split);
  ScalarFoldResult run();
};
}

} // namespace sentryward::value_analysis

#include "scalar_lowering_sources.hpp"
#include "scalar_lowering_coordinates.hpp"
#include "scalar_lowering_members.hpp"
#include "scalar_lowering_views.hpp"
#include "scalar_lowering_partitions.hpp"
#include "scalar_lowering_reads.hpp"
#include "scalar_lowering_arithmetic.hpp"
#include "scalar_lowering_mma.hpp"
#include "scalar_lowering_templates.hpp"
#include "scalar_lowering_projection.hpp"
#include "scalar_lowering_collection.hpp"
#include "scalar_lowering_outputs.hpp"
#include "scalar_lowering_driver.hpp"
