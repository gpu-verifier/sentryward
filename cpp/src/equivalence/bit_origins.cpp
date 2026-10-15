#include "sentryward/equivalence/bit_origins.h"
#include "sentryward/equivalence/constant_values.h"
#include "sentryward/equivalence/conversion_uses.h"
#include "sentryward/equivalence/integer_interval.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <climits>
#include <limits>
#include <map>
#include <tuple>

namespace sentryward::equivalence {
unsigned bitWidth(const std::string& t) {
  if(t.size()>128)return 0;
  if(t=="e4m3" || t=="e5m2")return 8;
  if(t=="bf16")return 16;
  if(t.size()<2 || std::string("bsuf").find(t[0])==std::string::npos)return 0;
  unsigned n=0;auto r=std::from_chars(t.data()+1,t.data()+t.size(),n);
  return r.ec==std::errc{} && r.ptr==t.data()+t.size() &&
      (n==8 || n==16 || n==32 || n==64)?n:0;
}
namespace {
namespace s=scalar_fold;
namespace ii=integer_interval;
using Origin=SymbolicBitOrigin;
using Origins=SymbolicBitOrigins;
Origin constant(bool value=false) {return {Origin::Kind::Constant,0,{},value};}
bool metadata(const s::Ref& r) {
  return r.conversion.size()<=128 && r.converted_type.size()<=128 &&
      r.pointer_provenance.empty();
}
bool integerBitsType(const std::string& type) {
  // bf16 is a floating format despite its leading 'b'. Raw bN/uN/sN types
  // must have a decimal width immediately after their tag.
  return type.size()>=2 && type[1]>='0' && type[1]<='9' &&
      std::string("bsu").find(type[0])!=std::string::npos && bitWidth(type);
}
std::optional<std::uint64_t> literal(const s::Graph& g,s::Ref r) {
  const auto& n=g.node(r);
  if(n.kind!=s::Kind::Constant || !metadata(r) || n.name.size()>128)return {};
  if(!integerBitsType(n.type))return {};
  auto current=n.type;auto text=normalizeConversions(current,g.type(r),r.conversion);if(!text)return {};
  while(!text->empty()) {
    const auto end=text->find("; "),begin=text->find('('),arrow=text->find(" -> ");
    if(begin==std::string::npos || arrow==std::string::npos)return {};
    const auto step=text->substr(0,end),op=step.substr(0,begin),target=step.substr(arrow+4,step.size()-arrow-5);
    if(!integerBitsType(target) || (op!="Convert" && (op!="Bitcast" || bitWidth(current)!=bitWidth(target))))return {};
    current=target;if(end==std::string::npos)break;text->erase(0,end+2);
  }
  auto value=constantValue(n.type,n.name,r.conversion,r.converted_type);
  if(!value)return {};
  std::uint64_t bits=0;auto p=std::from_chars(value->literal.data(),value->literal.data()+value->literal.size(),bits);
  if(p.ec!=std::errc{} || p.ptr!=value->literal.data()+value->literal.size())return {};
  return bits;
}
struct RefKey {
  s::Ref value;std::size_t scope=0;
  bool operator<(const RefKey& b) const {
    return std::tie(value.node,value.conversion,value.converted_type,value.owner,scope)<
        std::tie(b.value.node,b.value.conversion,b.value.converted_type,b.value.owner,b.scope);
  }
};
struct Index {
  enum class Transform { Identity, Divide, Remainder };
  s::Ref input;
  Transform transform=Transform::Identity;
  std::uint64_t divisor=1,lower=0,upper=0;
  std::size_t scope=0;
};
struct Affine {
  SymbolicInteger value;
  ii::Interval range;
  bool symbolic=true;
};
bool add(std::uint64_t a,std::uint64_t b,std::uint64_t& c) {
  return !__builtin_add_overflow(a,b,&c);
}
bool multiply(std::uint64_t a,std::uint64_t b,std::uint64_t& c) {
  return !__builtin_mul_overflow(a,b,&c);
}
}

struct BitOriginAnalysis::Impl {
  const s::Graph& graph;
  std::vector<BitDomains> scopes;
  std::vector<BitSource> sources;
  std::vector<Index> indices;
  std::map<RefKey,std::size_t> source_ids;
  std::map<std::pair<s::Id,std::size_t>,std::optional<Origins>> nodes;
  std::map<std::pair<s::Id,std::size_t>,std::optional<Affine>> integers;
  std::size_t work=0;unsigned depth=0;bool exhausted=false;
  std::uint32_t exhausted_limits=0;
  explicit Impl(const s::Graph& g):graph(g) {}
  void exhaust(BitOriginLimit limit) {
    exhausted=true;exhausted_limits|=static_cast<std::uint32_t>(limit);
  }
  bool charge(std::size_t n=1) {
    constexpr std::size_t limit=1u<<20;
    if(exhausted)return false;
    if(n>limit-work){exhaust(BitOriginLimit::Work);return false;}work+=n;return true;
  }
  std::optional<std::size_t> scope(const BitDomains& domain) {
    if(domain.size()>64 || !charge(domain.size()+1))return {};
    for(auto [level,count]:domain) {
      (void)level;
      if(!count || count-1>std::uint64_t(INT64_MAX))return {};
    }
    for(std::size_t i=0;i<scopes.size();++i) {
      if(!charge(domain.size()+1))return {};
      if(scopes[i]==domain)return i;
    }
    if(scopes.size()>=128){exhaust(BitOriginLimit::Scopes);return {};}
    scopes.push_back(domain);return scopes.size()-1;
  }
  std::optional<std::size_t> index(Index value) {
    for(std::size_t i=0;i<indices.size();++i) {
      if(!charge())return {};
      const auto& x=indices[i];
      if(x.input==value.input && x.transform==value.transform && x.divisor==value.divisor &&
          x.scope==value.scope && x.lower==value.lower && x.upper==value.upper)return i;
    }
    if(indices.size()>=512){exhaust(BitOriginLimit::Indices);return {};}
    indices.push_back(std::move(value));return indices.size()-1;
  }
  std::optional<std::size_t> source(s::Ref value,std::size_t scope,BitSource::Kind kind) {
    if(!metadata(value) || !charge(value.conversion.size()+value.converted_type.size()+1))return {};
    RefKey key{value,scope};
    if(auto i=source_ids.find(key);i!=source_ids.end())return i->second;
    if(sources.size()>=4096){exhaust(BitOriginLimit::Sources);return {};}
    const auto width=bitWidth(graph.type(value));if(!width)return {};
    const auto id=sources.size();sources.push_back({kind,value,width});source_ids.emplace(std::move(key),id);return id;
  }
  Origins opaque(s::Ref ref,std::size_t scope,BitSource::Kind kind=BitSource::Kind::OpaqueValue) {
    const auto width=bitWidth(graph.type(ref));Origins result(width);
    if(!charge(width))return result;
    if(auto id=source(ref,scope,kind))for(unsigned bit=0;bit<width;++bit)
      result[bit]={Origin::Kind::Source,*id,{{},0,bit},false};
    return result;
  }
  // Unknown occurrences are independent variables. This may lose identities,
  // never gain one. Known source atoms retain their exact bit correlations.
  Origin logic(const std::array<Origin,3>& input,unsigned arity,unsigned lut) {
    std::array<Origin,3> vars{};std::array<unsigned,3> which{};unsigned count=0;
    for(unsigned i=0;i<arity;++i)if(input[i].kind!=Origin::Kind::Constant) {
      unsigned j=0;
      if(input[i].kind==Origin::Kind::Source)for(;j<count;++j) {
        auto a=input[i],b=vars[j];a.inverted=b.inverted=false;
        if(a==b)break;
      } else j=count;
      if(j==count){vars[count]=input[i];vars[count].inverted=false;++count;}which[i]=j;
    }
    if(!charge((1u<<count)*arity+count))return {};
    std::array<bool,8> table{};
    for(unsigned assignment=0;assignment<(1u<<count);++assignment) {
      unsigned column=0;
      for(unsigned i=0;i<arity;++i) {
        const auto& a=input[i];
        const bool v=a.kind==Origin::Kind::Constant?false:bool((assignment>>which[i])&1);
        column=(column<<1)|unsigned(v^a.inverted);
      }
      table[assignment]=(lut>>column)&1;
    }
    bool same=true;for(unsigned a=1;a<(1u<<count);++a)same &= table[a]==table[0];
    if(same)return constant(table[0]);
    for(unsigned v=0;v<count;++v) {
      bool matches=true;for(unsigned a=0;a<(1u<<count);++a)
        matches &= table[a]==(bool((a>>v)&1)^table[0]);
      if(matches && vars[v].kind==Origin::Kind::Source) {
        auto result=vars[v];result.inverted=table[0];return result;
      }
    }
    return {};
  }
  std::optional<Affine> integerUse(s::Ref ref,std::size_t scope) {
    if(!metadata(ref) || !charge(ref.conversion.size()+1))return {};
    auto result=integerNode(ref.node,scope);if(!result)return {};
    auto current=graph.node(ref).type;
    auto text=normalizeConversions(current,graph.type(ref),ref.conversion);if(!text)return {};
    while(!text->empty()) {
      const auto end=text->find("; "),begin=text->find('('),arrow=text->find(" -> ");
      if(begin==std::string::npos || arrow==std::string::npos)return {};
      const auto step=text->substr(0,end),op=step.substr(0,begin),target=step.substr(arrow+4,step.size()-arrow-5);
      // A raw bN destination of an ordinary integer conversion stores the
      // same low N bits as uN. Its scalar value is retained only when the
      // interval proves no truncation. This does not reinterpret floats.
      const auto numeric=[](std::string t){if(t.starts_with("b"))t[0]='u';return t;};
      const auto from=numeric(current),to=numeric(target);
      auto range=integerConversionRange(from,to,op+"("+from+" -> "+to+")",result->range);
      if(!range || (target.starts_with("b") && !current.starts_with("u") && op!="Bitcast"))return {};
      result->range=*range;current=target;
      if(end==std::string::npos)break;text->erase(0,end+2);
    }
    return result;
  }
  std::optional<Affine> integerNode(s::Id id,std::size_t scope) {
    const auto key=std::pair{id,scope};if(auto i=integers.find(key);i!=integers.end())return i->second;
    if(depth>=256){exhaust(BitOriginLimit::Depth);return {};}
    if(integers.size()>=4096){exhaust(BitOriginLimit::IntegerNodes);return {};}
    if(!charge())return {};
    ++depth;auto value=integerCompute(id,scope);--depth;integers.emplace(key,value);return value;
  }
  std::optional<Affine> integerCompute(s::Id id,std::size_t scope) {
    const auto& n=graph.node(id);auto type=ii::parseType(n.type);
    if(!type || n.name.size()>128 || n.attributes.size()>128)return {};
    if(n.kind==s::Kind::Constant) {
      auto value=ii::literal(*type,n.name);if(!value)return {};
      if(value->lower<0)return Affine{{},*value,false};
      const auto v=std::uint64_t(value->lower);return Affine{{{},0,v,v,v},*value};
    }
    if(n.kind==s::Kind::Index) {
      const auto bound=scopes[scope].find(n.index_level);if(bound==scopes[scope].end())return {};
      auto range=ii::externalIndex(*type,bound->second);if(!range)return {};
      auto symbol=index({graph.ref(id),Index::Transform::Identity,1,0,std::uint64_t(range->upper),scope});
      if(!symbol)return {};return Affine{{symbol,1,0,0,std::uint64_t(range->upper)},*range};
    }
    const bool typed_leaf=!n.fold &&
        ((n.kind==s::Kind::Parameter && n.operands.empty()) ||
         (n.kind==s::Kind::InitialCell && n.operands.size()==1 && metadata(n.operands[0]) &&
          graph.type(n.operands[0])=="u64"));
    if(n.attributes.empty() && typed_leaf) {
      if(auto range=ii::typeRange(*type))return Affine{{},*range,false};return {};
    }
    if(n.kind!=s::Kind::Operation || !n.attributes.empty() || n.operands.size()!=2)return {};
    auto a=integerUse(n.operands[0],scope),b=integerUse(n.operands[1],scope);
    if(!a || !b || a->range.type!=*type || b->range.type!=*type)return {};
    ii::Result range;
    if(n.name=="Add") {
      range=ii::apply(ii::Operation::Add,a->range,b->range);
    } else if(n.name=="Subtract") {
      range=ii::apply(ii::Operation::Subtract,a->range,b->range);
    } else if(n.name=="Multiply") {
      range=ii::apply(ii::Operation::Multiply,a->range,b->range);
    } else if(n.name=="BitAnd" || n.name=="BitXor") {
      range=ii::apply(n.name=="BitAnd"?ii::Operation::BitAnd:ii::Operation::BitXor,a->range,b->range);
    } else if((n.name=="Divide" || n.name=="Remainder") && b->range.lower==b->range.upper && b->range.lower>0) {
      range=n.name=="Divide"?ii::divideByPositiveLiteral(a->range,b->range.lower):ii::remainderByPositiveLiteral(a->range,b->range.lower);
    } else if(n.name=="ShiftRight" && b->range.lower==b->range.upper) {
      range=ii::shiftRightByLiteral(a->range,b->range.lower);
    }
    if(!range)return {};
    // Range-only facts are useful for exact integer identities, but do not
    // assert that an opaque/multi-index value is one affine bit-origin atom.
    if(!a->symbolic || !b->symbolic || range->lower<0)return Affine{{},*range,false};
    auto av=a->value,bv=b->value;SymbolicInteger v;
    if(n.name=="Add") {
      if(av.index && bv.index && av.index!=bv.index)return Affine{{},*range,false};
      v.index=av.index?av.index:bv.index;
      if(!add(av.coefficient,bv.coefficient,v.coefficient) || !add(av.constant,bv.constant,v.constant))return {};
    } else if(n.name=="Multiply") {
      if(av.index && bv.index)return Affine{{},*range,false};
      if(!av.index)std::swap(av,bv);
      v.index=av.index;
      if(!multiply(av.coefficient,bv.constant,v.coefficient) || !multiply(av.constant,bv.constant,v.constant))return {};
    } else if((n.name=="Divide" || n.name=="Remainder") && !bv.index && bv.constant && bv.constant<=std::uint64_t(INT64_MAX)) {
      const auto divisor=bv.constant;
      if(!av.index) v.constant=n.name=="Divide"?av.constant/divisor:av.constant%divisor;
      else if(av.constant==0 && av.coefficient && divisor%av.coefficient==0) {
        // Checked range arithmetic above proves the multiplier did not wrap.
        // floor(c*i/d)=floor(i/(d/c)); (c*i)%d=c*(i%(d/c)).
        const auto& old=indices[*av.index];const auto d=divisor/av.coefficient;
        if(old.transform!=Index::Transform::Identity)return Affine{{},*range,false};
        auto symbol=index({old.input,n.name=="Divide"?Index::Transform::Divide:Index::Transform::Remainder,
            d,n.name=="Divide"?old.lower/d:0,n.name=="Divide"?old.upper/d:std::min(old.upper,d-1),scope});
        if(!symbol)return {};v.index=symbol;v.coefficient=n.name=="Divide"?1:av.coefficient;
      } else return Affine{{},*range,false};
    } else return Affine{{},*range,false};
    // Interval remainder bounds may include residues excluded by the exact
    // affine certificate (e.g. (2*i)%4 is {0,2}, not all of [0,3]). Tighten
    // only from that certified expression, after proving typed non-overflow.
    auto lower=v.constant,upper=v.constant;
    if(v.index) {
      const auto& symbol=indices[*v.index];std::uint64_t lo=0,hi=0;
      if(!multiply(v.coefficient,symbol.lower,lo) || !add(lower,lo,lower) ||
          !multiply(v.coefficient,symbol.upper,hi) || !add(upper,hi,upper))return {};
    }
    if(lower<std::uint64_t(range->lower) || upper>std::uint64_t(range->upper))return {};
    auto exact=ii::interval(range->type,lower,upper);if(!exact)return {};
    v.lower=lower;v.upper=upper;return Affine{v,*exact};
  }
  std::optional<Origins> use(s::Ref r,std::size_t scope) {
    if(!metadata(r) || !charge(r.conversion.size()+1))return {};
    auto bits=node(r.node,scope);if(!bits)return {};
    auto current=graph.node(r).type;
    auto normalized=normalizeConversions(current,graph.type(r),r.conversion);
    if(!normalized)return opaque(r,scope);
    auto text=*normalized;std::string prefix;
    while(!text.empty()) {
      const auto end=text.find("; ");const auto step=text.substr(0,end);
      const auto begin=step.find('('),arrow=step.find(" -> ");
      if(begin==std::string::npos || arrow==std::string::npos)return opaque(r,scope);
      const auto op=step.substr(0,begin),target=step.substr(arrow+4,step.size()-arrow-5);
      const auto width=bitWidth(target);if(!width)return {};
      if(!charge(width))return Origins(width);
      if(!prefix.empty())prefix+="; ";prefix+=step;
      if(op=="Bitcast") {if(width!=bits->size())return opaque(r,scope);}
      else if(op=="Convert" && integerBitsType(current) && integerBitsType(target) &&
          (current.starts_with("u") || current.starts_with("s")))
        bits->resize(width,current.starts_with("s")?bits->back():constant());
      else {
        // A validated numeric conversion is an opaque VALUE of its declared
        // result width, not a reason to hide later exact zero extension and
        // packing. Preserve the complete conversion prefix on its source;
        // do not infer a bit relation through this conversion or erase its
        // rounding/saturation modifiers. Unknown conversion syntax failed
        // normalizeConversions above. The <=128-byte use cap and
        // <=64-bit hardware widths bound this work independently of ancestry.
        auto converted=r;converted.conversion=prefix;converted.converted_type=target;
        bits=opaque(std::move(converted),scope);
      }
      current=target;
      if(end==std::string::npos)break;text.erase(0,end+2);
    }
    return bits;
  }
  std::optional<Origins> node(s::Id id,std::size_t scope) {
    const auto key=std::pair{id,scope};if(auto i=nodes.find(key);i!=nodes.end()) {
      if(!charge(i->second?i->second->size():1))return {};return i->second;
    }
    if(depth>=256){exhaust(BitOriginLimit::Depth);return {};}
    if(nodes.size()>=4096){exhaust(BitOriginLimit::BitNodes);return {};}
    if(!charge())return {};
    ++depth;auto result=compute(id,scope);--depth;nodes.emplace(key,result);return result;
  }
  Origin variableByte(const Origins& input,const SymbolicInteger& selector,unsigned bit) {
    if(!selector.index || !selector.coefficient)return input[8*(selector.constant&7)+((selector.constant&8)?7:bit)];
    const auto& symbol=indices[*selector.index];
    if(symbol.upper-symbol.lower>15)return {};
    Origin first;bool initialized=false,all_constant=true;std::uint64_t first_q=0,first_offset=0,delta=0;
    for(auto q=symbol.lower;;++q) {
      if(!charge())return {};
      std::uint64_t select=0;if(!multiply(selector.coefficient,q,select) || !add(select,selector.constant,select) || select>15)return {};
      auto origin=input[8*(select&7)+((select&8)?7:bit)];
      if(origin.kind==Origin::Kind::Unknown)return {};
      if(!initialized) {first=origin;first_q=q;first_offset=origin.offset.constant;initialized=true;}
      else {
        if(origin.kind!=first.kind || origin.inverted!=first.inverted)return {};
        if(origin.kind==Origin::Kind::Source) {
          if(origin.source!=first.source || origin.offset.index || first.offset.index)return {};
          if(q==first_q+1) {
            if(origin.offset.constant<first_offset)return {};delta=origin.offset.constant-first_offset;
          }
          std::uint64_t expected=0;
          if(!multiply(delta,q-first_q,expected) || !add(first_offset,expected,expected) || origin.offset.constant!=expected)return {};
        }
      }
      all_constant &= origin.kind==Origin::Kind::Constant;
      if(q==symbol.upper)break;
    }
    if(all_constant || symbol.lower==symbol.upper)return first;
    std::uint64_t base=0;if(!multiply(delta,first_q,base) || base>first_offset)return {};
    first.offset={selector.index,delta,first_offset-base};return first;
  }
  std::optional<Origins> compute(s::Id id,std::size_t scope) {
    const auto& n=graph.node(id);const auto width=bitWidth(n.type);if(!width)return {};
    if(!charge(width) || n.type.size()>128 || n.name.size()>128 || n.attributes.size()>128)return Origins(width);
    if(n.kind==s::Kind::Constant) {
      auto raw=graph.ref(id);const auto target="b"+std::to_string(width);
      auto value=constantValue(n.type,n.name,n.type==target?"":"Bitcast("+n.type+" -> "+target+")",target);
      if(!value)return opaque(raw,scope);
      std::uint64_t bits=0;auto p=std::from_chars(value->literal.data(),value->literal.data()+value->literal.size(),bits);
      if(p.ec!=std::errc{} || p.ptr!=value->literal.data()+value->literal.size())return opaque(raw,scope);
      Origins result(width);for(unsigned i=0;i<width;++i)result[i]=constant((bits>>i)&1);return result;
    }
    if(n.kind==s::Kind::InitialCell)return opaque(graph.ref(id),scope,BitSource::Kind::Memory);
    if(n.kind!=s::Kind::Operation)return opaque(graph.ref(id),scope);
    if((n.name=="ShiftRight" || n.name=="ShiftLeft") && n.attributes.empty() && n.operands.size()==2) {
      auto a=use(n.operands[0],scope);auto shift=literal(graph,n.operands[1]);
      if(!a || !shift || a->size()!=width || graph.type(n.operands[0])!=n.type || !integerBitsType(n.type) ||
          !integerBitsType(graph.type(n.operands[1])) ||
          (n.type[0]!='u' && n.type[0]!='b' && n.type[0]!='s'))return opaque(graph.ref(id),scope);
      Origins bits(width,constant());
      for(unsigned i=0;i<width;++i) {
        if(n.name=="ShiftRight")bits[i]=*shift<width-i?(*a)[i+*shift]:(n.type[0]=='s'?a->back():constant());
        else if(*shift<=i)bits[i]=(*a)[i-*shift];
      }
      return bits;
    }
    if(n.name=="ExtractBits" && n.operands.size()==2 && n.attributes=="width="+std::to_string(width)) {
      auto a=use(n.operands[0],scope);auto start=literal(graph,n.operands[1]);
      const auto& source_type=graph.type(n.operands[0]);
      if(!a || !start || !integerBitsType(source_type) ||
          !integerBitsType(graph.type(n.operands[1])) ||
          *start>a->size() || width>a->size()-*start)return opaque(graph.ref(id),scope);
      return Origins(a->begin()+*start,a->begin()+*start+width);
    }
    if(n.name=="PtxBytePermute" && n.type=="b32" && n.operands.size()==3 &&
        n.attributes=="source=b32.b32 source_type=b32" && graph.type(n.operands[0])=="b32" && graph.type(n.operands[1])=="b32") {
      if(graph.type(n.operands[2])!="u32" && graph.type(n.operands[2])!="b32")return opaque(graph.ref(id),scope);
      auto a=use(n.operands[0],scope),b=use(n.operands[1],scope);
      if(!a || !b || a->size()!=32 || b->size()!=32)return opaque(graph.ref(id),scope);
      a->insert(a->end(),b->begin(),b->end());Origins bits(32);
      if(auto control=literal(graph,n.operands[2])) {
        for(unsigned i=0;i<4;++i)for(unsigned j=0;j<8;++j) {
          const auto select=(*control>>(4*i))&15;bits[8*i+j]=(*a)[8*(select&7)+((select&8)?7:j)];
        }
        return bits;
      }
      auto control=integerUse(n.operands[2],scope);
      if(!control || !control->symbolic || !control->value.index || control->value.upper>UINT32_MAX)return opaque(graph.ref(id),scope);
      const auto& expression=control->value;const auto& symbol=indices[*expression.index];
      // Each nibble's endpoint fits in 0..15: no inter-nibble carry can alter
      // the extracted affine selectors. High control bits are not consumed.
      for(unsigned i=0;i<4;++i) {
        const auto c=(expression.constant>>(4*i))&15,k=(expression.coefficient>>(4*i))&15;
        std::uint64_t high=0;if(!multiply(k,symbol.upper,high) || !add(c,high,high) || high>15)return opaque(graph.ref(id),scope);
        SymbolicInteger selector{expression.index,k,c,0,high};
        for(unsigned j=0;j<8;++j)bits[8*i+j]=variableByte(*a,selector,j);
      }
      return bits;
    }
    // Unsigned/bit-vector multiplication by a power of two is a logical shift
    // modulo the declared width. This is not a floating or numeric alias.
    if(n.name=="Multiply" && n.attributes.empty() && n.operands.size()==2 && integerBitsType(n.type) && (n.type[0]=='u' || n.type[0]=='b')) {
      for(unsigned side=0;side<2;++side)if(auto factor=literal(graph,n.operands[side]);factor && *factor && (*factor&(*factor-1))==0) {
        auto a=use(n.operands[1-side],scope);if(!a || a->size()!=width ||
            graph.type(n.operands[side])!=n.type || graph.type(n.operands[1-side])!=n.type)break;
        unsigned shift=0;for(auto f=*factor;f>1;f>>=1)++shift;
        Origins bits(width,constant());for(unsigned i=shift;i<width;++i)bits[i]=(*a)[i-shift];return bits;
      }
    }
    if(n.name=="Add" && n.attributes.empty() && n.operands.size()==2 && integerBitsType(n.type) && (n.type[0]=='u' || n.type[0]=='b')) {
      auto a=use(n.operands[0],scope),b=use(n.operands[1],scope);
      if(!a || !b || a->size()!=width || b->size()!=width || graph.type(n.operands[0])!=n.type || graph.type(n.operands[1])!=n.type)return opaque(graph.ref(id),scope);
      Origins bits(width);auto carry=constant();
      for(unsigned i=0;i<width;++i) {
        const std::array inputs{(*a)[i],(*b)[i],carry};
        bits[i]=logic(inputs,3,0x96);carry=logic(inputs,3,0xe8);
      }
      return bits;
    }
    unsigned lut=0;std::size_t arity=0;
    if(!integerBitsType(n.type))return opaque(graph.ref(id),scope);
    if(n.attributes.empty() && n.name=="BitAnd"){lut=8;arity=2;}
    if(n.attributes.empty() && n.name=="BitOr"){lut=14;arity=2;}
    if(n.attributes.empty() && n.name=="BitXor"){lut=6;arity=2;}
    if(n.attributes.empty() && n.name=="BitNot"){lut=1;arity=1;}
    if(n.name=="PtxLogic3" && n.type=="b32" && n.operands.size()==4 &&
        n.attributes=="source=b32.b32 source_type=b32") {
      auto value=literal(graph,n.operands[3]);if(!value || *value>255)return opaque(graph.ref(id),scope);
      if(graph.type(n.operands[3])!="u32")return opaque(graph.ref(id),scope);
      lut=*value;arity=3;
    } else if(n.operands.size()!=arity)return opaque(graph.ref(id),scope);
    if(!arity)return opaque(graph.ref(id),scope);
    std::array<Origins,3> args;
    for(std::size_t i=0;i<arity;++i) {
      auto a=use(n.operands[i],scope);if(!a || a->size()!=width || graph.type(n.operands[i])!=n.type)return opaque(graph.ref(id),scope);args[i]=std::move(*a);
    }
    Origins bits(width);
    for(unsigned i=0;i<width;++i) {
      std::array<Origin,3> column{};for(unsigned j=0;j<arity;++j)column[j]=args[j][i];
      bits[i]=logic(column,arity,lut);
    }
    return bits;
  }
};
BitOriginAnalysis::BitOriginAnalysis(const scalar_fold::Graph& graph):impl_(std::make_unique<Impl>(graph)) {}
BitOriginAnalysis::~BitOriginAnalysis()=default;
std::optional<SymbolicBitOrigins> BitOriginAnalysis::trace(scalar_fold::Ref ref,const BitDomains& domains) {
  auto scope=impl_->scope(domains);if(!scope)return {};return impl_->use(std::move(ref),*scope);
}
std::optional<SymbolicInteger> BitOriginAnalysis::integer(scalar_fold::Ref ref,const BitDomains& domains) {
  auto scope=impl_->scope(domains);if(!scope)return {};auto result=impl_->integerUse(std::move(ref),*scope);
  if(!result || !result->symbolic)return {};return result->value;
}
integer_interval::Result BitOriginAnalysis::integerRange(scalar_fold::Ref ref,const BitDomains& domains) {
  auto scope=impl_->scope(domains);if(!scope)return {};auto result=impl_->integerUse(std::move(ref),*scope);
  if(!result)return {};return result->range;
}
const BitSource* BitOriginAnalysis::source(std::size_t id) const {
  return id<impl_->sources.size()?&impl_->sources[id]:nullptr;
}
std::optional<std::pair<std::uint64_t,std::uint64_t>> BitOriginAnalysis::bounds(const SymbolicBitOffset& offset) const {
  if(!impl_->charge())return {};
  if(!offset.index) {
    if(offset.coefficient)return {};return std::pair{offset.constant,offset.constant};
  }
  if(*offset.index>=impl_->indices.size())return {};
  const auto& index=impl_->indices[*offset.index];std::uint64_t lo=0,hi=0;
  if(!multiply(offset.coefficient,index.lower,lo) || !add(lo,offset.constant,lo) ||
      !multiply(offset.coefficient,index.upper,hi) || !add(hi,offset.constant,hi))return {};
  return std::pair{lo,hi};
}
bool BitOriginAnalysis::chargeWork(std::size_t work) {return impl_->charge(work);}
std::optional<scalar_fold::Ref> BitOriginAnalysis::emit(scalar_fold::Graph& graph,const SymbolicBitOffset& offset,std::uint64_t divisor) const {
  if(&graph!=&impl_->graph || !divisor || offset.constant%divisor || offset.coefficient%divisor ||
      !bounds(offset) || !impl_->charge(8))return {};
  auto c=offset.constant/divisor,k=offset.coefficient/divisor;
  auto result=graph.constant("u64",std::to_string(c));
  if(offset.index && k) {
    if(*offset.index>=impl_->indices.size())return {};const auto& symbol=impl_->indices[*offset.index];
    auto input=symbol.input;if(graph.type(input)!="u64")input=graph.convert(input,"u64","Convert");
    if(symbol.transform!=Index::Transform::Identity)input=graph.operation(
        symbol.transform==Index::Transform::Divide?"Divide":"Remainder","u64",{input,graph.constant("u64",std::to_string(symbol.divisor))});
    if(k!=1)input=graph.operation("Multiply","u64",{input,graph.constant("u64",std::to_string(k))});
    result=c?graph.operation("Add","u64",{input,result}):input;
  }
  return result;
}
BitOriginStatistics BitOriginAnalysis::statistics() const {
  return {impl_->nodes.size()+impl_->integers.size(),impl_->scopes.size(),impl_->sources.size(),impl_->indices.size(),impl_->work,impl_->exhausted,
          impl_->nodes.size(),impl_->integers.size(),impl_->exhausted_limits};
}
std::optional<BitOrigins> traceBitOrigins(const scalar_fold::Graph& graph,scalar_fold::Ref ref) {
  BitOriginAnalysis analysis(graph);auto bits=analysis.trace(ref,{});if(!bits)return {};
  BitOrigins result;result.reserve(bits->size());
  for(const auto& bit:*bits) {
    if(bit.kind==Origin::Kind::Unknown)return {};
    if(bit.kind==Origin::Kind::Constant){result.push_back({{},0,bit.inverted});continue;}
    const auto* source=analysis.source(bit.source);
    if(!source || source->kind!=BitSource::Kind::Memory || bit.offset.index || bit.offset.constant>UINT_MAX)return {};
    result.push_back({source->value.node,unsigned(bit.offset.constant),bit.inverted});
  }
  return result;
}
}
