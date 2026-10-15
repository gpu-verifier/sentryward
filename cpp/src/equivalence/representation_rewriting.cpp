#include "sentryward/equivalence/representation_rewriting.h"
#include "sentryward/equivalence/packed_values.h"
#include "sentryward/equivalence/constant_values.h"
#include "sentryward/equivalence/conversion_uses.h"
#include "sentryward/equivalence/egraph.h"
#include "sentryward/equivalence/bit_origins.h"
#include <array>
#include <stdexcept>
#include <unordered_map>
#include <limits>
namespace sentryward::equivalence {
namespace {
namespace s=scalar_fold;
namespace ii=integer_interval;
bool unannotated(const s::Ref& ref) {
  return ref.conversion.empty() && ref.converted_type.empty() && ref.pointer_provenance.empty();
}
s::Ref keepUseTail(s::Ref result,const s::Ref& incoming) {
  if(!incoming.conversion.empty()) {
    if(!result.conversion.empty())result.conversion+="; ";
    result.conversion+=incoming.conversion;result.converted_type=incoming.converted_type;
  }
  return result;
}
std::optional<std::int64_t> exactIntegerLiteral(const s::Graph& graph,const s::Ref& ref) {
  if(!unannotated(ref))return {};
  const auto& n=graph.node(ref);
  if(n.kind!=s::Kind::Constant || !n.attributes.empty() || !n.operands.empty())return {};
  auto value=ii::literal(n.type,n.name);if(!value)return {};return value->lower;
}
bool convertedSelectCandidate(const s::Graph& graph,const s::Ref& ref) {
  const auto& source=graph.node(ref);
  return source.kind==s::Kind::Operation && source.name=="Select" && source.attributes.empty() &&
      source.operands.size()==3 && ii::parseType(source.type) && !ref.conversion.empty() &&
      ref.pointer_provenance.empty() && (graph.type(ref)=="f32" || graph.type(ref)=="f64");
}
bool scopedIntegerCandidate(const s::Graph& graph,const s::Ref& ref) {
  const auto& source=graph.node(ref);
  return source.kind==s::Kind::Operation && source.attributes.empty() &&
      source.operands.size()==2 && ii::parseType(source.type) &&
      (source.name=="Divide" || source.name=="Remainder");
}
bool zeroProductCandidate(const s::Graph& graph,const s::Ref& ref) {
  const auto& source=graph.node(ref);
  return source.kind==s::Kind::Operation && source.type=="pred" && source.operands.size()==2 &&
      ((source.attributes.empty() && (source.name=="Equal" || source.name=="NotEqual")) ||
       (source.name=="Compare" && (source.attributes=="predicate=eq" || source.attributes=="predicate=ne")));
}
bool bf16WidenedCandidate(const s::Graph& graph,const s::Ref& ref) {
  if(!ref.pointer_provenance.empty() || ref.conversion.size()>128 || bitWidth(graph.node(ref).type)!=32)return false;
  const auto prefix=leadingBitcast(graph.node(ref).type,ref.conversion);
  return prefix && prefix->target=="f32";
}
// Move an ORDINARY integer->float edge through one exact Select only when
// every arm value is representable. The actual bare/RN edge is retained on
// each arm; no default rounding alias, integer precision drop or modifier
// admission is introduced. No predicate/arm descendant is rewritten here.
s::Ref rewriteConvertedSelect(s::Graph& graph,s::Ref ref,
    BitOriginAnalysis& analysis,const BitDomains& domains) {
  const auto& source=graph.node(ref);
  if(!convertedSelectCandidate(graph,ref))return ref;
  const auto target=graph.type(ref);if(target!="f32" && target!="f64")return ref;
  const auto normalized=normalizeConversions(source.type,target,ref.conversion);
  const auto edge="("+source.type+" -> "+target+")";
  if(!normalized || (*normalized!="Convert"+edge &&
      *normalized!="Convert rounding=nearest-even"+edge))return ref;
  const auto n=source;
  if(graph.type(n.operands[0])!="pred" || !unannotated(n.operands[0]))return ref;
  const std::int64_t exact_limit=std::int64_t{1}<<(target=="f32"?24:53);
  for(unsigned arm=1;arm<3;++arm) {
    if(graph.type(n.operands[arm])!=n.type || !n.operands[arm].pointer_provenance.empty() ||
        n.operands[arm].conversion.size()+ref.conversion.size()+2>128)return ref;
    const auto range=analysis.integerRange(n.operands[arm],domains);
    if(!range || range->lower < -exact_limit || range->upper > exact_limit)return ref;
  }
  auto yes=n.operands[1],no=n.operands[2];
  yes=keepUseTail(std::move(yes),ref);no=keepUseTail(std::move(no),ref);
  return graph.operation("Select",target,{n.operands[0],std::move(yes),std::move(no)});
}
// One fixed local Euclidean identity. All participating integer operations
// have compatible types, no modifiers/provenance, nonnegative interval facts,
// and a proof that the original typed arithmetic cannot wrap. Facts share the
// session's bounded cache; only immediate heads are inspected/emitted.
s::Ref rewriteScopedInteger(s::Graph& graph,s::Ref ref,
    BitOriginAnalysis& analysis,const BitDomains& domains) {
  const auto& source=graph.node(ref);
  if(!scopedIntegerCandidate(graph,ref))return ref;
  const auto n=source;const auto type=n.type;
  if(graph.type(n.operands[0])!=type || graph.type(n.operands[1])!=type ||
      !unannotated(n.operands[0]) || !unannotated(n.operands[1]))return ref;
  const auto d=exactIntegerLiteral(graph,n.operands[1]);
  if((n.name!="Divide" && n.name!="Remainder") || !d || *d<=0)return ref;
  const auto range=analysis.integerRange(graph.ref(ref.node),domains);
  const auto input_range=analysis.integerRange(n.operands[0],domains);
  if(!range || !input_range || input_range->lower<0)return ref;
  const auto constant=[&](std::int64_t value){return graph.constant(type,std::to_string(value));};
  const auto divide=[&](s::Ref value) {
    const auto r=analysis.integerRange(value,domains);
    if(r && r->upper<*d)return constant(0);
    return graph.operation("Divide",type,{std::move(value),constant(*d)});
  };
  const auto remainder=[&](s::Ref value) {
    const auto r=analysis.integerRange(value,domains);
    if(r && r->upper<*d)return value;
    return graph.operation("Remainder",type,{std::move(value),constant(*d)});
  };
  if(input_range->upper<*d)return keepUseTail(n.name=="Divide"?constant(0):n.operands[0],ref);
  // Return q only if this complete immediate value is exactly d*q.
  const auto multiple=[&](const s::Ref& value)->std::optional<s::Ref> {
    auto r=analysis.integerRange(value,domains);if(!r || r->lower<0)return {};
    if(auto literal=exactIntegerLiteral(graph,value);literal && *literal%*d==0)
      return constant(*literal / *d);
    const auto p=graph.node(value);
    if(p.kind!=s::Kind::Operation || p.name!="Multiply" || p.type!=type ||
        !p.attributes.empty() || p.operands.size()!=2)return {};
    for(unsigned side=0;side<2;++side) {
      if(!unannotated(p.operands[side]) || !unannotated(p.operands[1-side]) ||
          graph.type(p.operands[side])!=type || graph.type(p.operands[1-side])!=type)continue;
      auto c=exactIntegerLiteral(graph,p.operands[side]);
      auto x=analysis.integerRange(p.operands[1-side],domains);
      if(!c || *c<0 || *c%*d || !x || x->lower<0)continue;
      if(*c / *d==1)return p.operands[1-side];
      return graph.operation("Multiply",type,{p.operands[1-side],constant(*c / *d)});
    }
    return {};
  };
  const auto input=graph.node(n.operands[0]);
  if(input.kind!=s::Kind::Operation || input.type!=type || !input.attributes.empty() || input.operands.size()!=2)return ref;
  for(const auto& child:input.operands)if(!unannotated(child) || graph.type(child)!=type)return ref;
  if(input.name=="Add")for(unsigned side=0;side<2;++side) {
    auto a=analysis.integerRange(input.operands[side],domains);
    auto b=analysis.integerRange(input.operands[1-side],domains);
    if(!a || !b || a->lower<0 || b->lower<0)continue;
    if(auto quotient=multiple(input.operands[side])) {
      auto result=n.name=="Remainder"?remainder(input.operands[1-side]):
          graph.operation("Add",type,{*quotient,divide(input.operands[1-side])});
      return keepUseTail(std::move(result),ref);
    }
  }
  if(input.name=="Multiply")for(unsigned side=0;side<2;++side) {
    auto c=exactIntegerLiteral(graph,input.operands[side]);
    auto x=analysis.integerRange(input.operands[1-side],domains);
    if(!c || *c<=0 || !x || x->lower<0)continue;
    if(*c%*d==0) {
      auto result=n.name=="Remainder"?constant(0):
          (*c / *d==1?input.operands[1-side]:graph.operation("Multiply",type,{input.operands[1-side],constant(*c / *d)}));
      return keepUseTail(std::move(result),ref);
    }
    if(*d%*c==0) {
      auto reduced=graph.operation(n.name,type,{input.operands[1-side],constant(*d / *c)});
      if(n.name=="Remainder" && *c!=1)reduced=graph.operation("Multiply",type,{std::move(reduced),constant(*c)});
      return keepUseTail(std::move(reduced),ref);
    }
  }
  return ref;
}
s::Ref rewriteZeroProductComparison(s::Graph& graph,s::Ref ref,
    BitOriginAnalysis& analysis,const BitDomains& domains) {
  const auto& source=graph.node(ref);
  if(!zeroProductCandidate(graph,ref))return ref;
  const auto n=source;
  for(unsigned side=0;side<2;++side) {
    auto zero=exactIntegerLiteral(graph,n.operands[side]);if(!zero || *zero!=0)continue;
    const auto& product_ref=n.operands[1-side];if(!unannotated(product_ref))continue;
    const auto p=graph.node(product_ref);
    if(p.kind!=s::Kind::Operation || p.name!="Multiply" || !p.attributes.empty() ||
        p.operands.size()!=2 || !ii::parseType(p.type) || graph.type(n.operands[side])!=p.type)continue;
    const auto range=analysis.integerRange(product_ref,domains);if(!range)continue;
    for(unsigned factor=0;factor<2;++factor) {
      const auto c=exactIntegerLiteral(graph,p.operands[factor]);
      if(!c || *c==0 || !unannotated(p.operands[1-factor]) || graph.type(p.operands[1-factor])!=p.type)continue;
      auto children=n.operands;children[1-side]=p.operands[1-factor];
      return keepUseTail(graph.operation(n.name,"pred",std::move(children),n.attributes),ref);
    }
  }
  return ref;
}
s::Ref rewriteConvertedSelect(s::Graph& graph,s::Ref ref) {
  BitOriginAnalysis analysis(graph);return rewriteConvertedSelect(graph,ref,analysis,{});
}
s::Ref rewriteScopedInteger(s::Graph& graph,s::Ref ref) {
  BitOriginAnalysis analysis(graph);return rewriteScopedInteger(graph,ref,analysis,{});
}
s::Ref rewriteZeroProductComparison(s::Graph& graph,s::Ref ref) {
  BitOriginAnalysis analysis(graph);return rewriteZeroProductComparison(graph,ref,analysis,{});
}
scalar_fold::Ref rewriteFloatingStorageRead(scalar_fold::Graph& graph,scalar_fold::Ref ref) {
  const auto prefix=floatingStorageReadPrefix(graph,ref);
  if(!prefix)return rewriteLocalFloatingMemoryRead(graph,std::move(ref));
  // Copy only the bounded local cell fields before interning invalidates Node
  // references. The address and every unsupported tail retain their exact use.
  auto allocation=graph.node(ref).name;
  auto address=graph.node(ref).operands[0];
  auto value=graph.initialCell(std::move(allocation),std::move(address),prefix->target);
  if(!prefix->remainder.empty()) {
    value.conversion=prefix->remainder;value.converted_type=ref.converted_type;
  }
  return value;
}
// Decode a BF16 field widened by shifting its bits into an FP32 register.
scalar_fold::Ref rewriteBf16WidenedRead(scalar_fold::Graph& graph,scalar_fold::Ref ref,
    BitOriginAnalysis& analysis,const BitDomains& domains) {
  namespace s=scalar_fold;
  const auto& source=graph.node(ref);
  if(!bf16WidenedCandidate(graph,ref))return ref;
  const auto prefix=leadingBitcast(source.type,ref.conversion);
  if(!prefix || prefix->target!="f32")return ref;
  auto bits=analysis.trace(graph.ref(ref.node),domains);if(!bits || bits->size()!=32)return ref;
  for(unsigned i=0;i<16;++i)if((*bits)[i].kind!=SymbolicBitOrigin::Kind::Constant || (*bits)[i].inverted)return ref;
  auto field=recoverBitField(graph,SymbolicBitOrigins(bits->begin()+16,bits->end()),"bf16",analysis);
  if(!field)return ref;
  auto value=graph.convert(*field,"f32","NumericInterpretation");
  if(!prefix->remainder.empty()) {
    value.conversion+="; "+prefix->remainder;
    value.converted_type=ref.converted_type;
  }
  return value;
}
scalar_fold::Ref rewriteBf16WidenedRead(scalar_fold::Graph& graph,scalar_fold::Ref ref) {
  BitOriginAnalysis analysis(graph);return rewriteBf16WidenedRead(graph,ref,analysis,{});
}

const std::array rules{
  RepresentationRule{"floating-storage-read",RewriteSemantics::ExactBits,
      "An unchanged byte-aligned memory view; remaining conversions stay explicit",rewriteFloatingStorageRead},
  RepresentationRule{"exact-use",RewriteSemantics::ExactNumeric,
      "Exact conversion and packed-use identities",[](scalar_fold::Graph& g,scalar_fold::Ref r){return normalizeExactUse(g,r);}},
  RepresentationRule{"converted-select",RewriteSemantics::ExactNumeric,
      "One ordinary integer-to-float edge through a Select whose arms are exactly representable",rewriteConvertedSelect,rewriteConvertedSelect,convertedSelectCandidate},
  RepresentationRule{"euclidean-integer",RewriteSemantics::ExactNumeric,
      "Literal-factor quotient/remainder identities certified nonnegative and nonwrapping",rewriteScopedInteger,rewriteScopedInteger,scopedIntegerCandidate},
  RepresentationRule{"zero-product-predicate",RewriteSemantics::ExactNumeric,
      "A nonwrapping integer product with one nonzero literal factor compares to zero",rewriteZeroProductComparison,rewriteZeroProductComparison,zeroProductCandidate},
  RepresentationRule{"contiguous-read",RewriteSemantics::ExactBits,
      "One unchanged contiguous source field; opaque fields retain their original type",rewriteContiguousRead,rewriteContiguousRead,contiguousReadCandidate},
  RepresentationRule{"packed-extract",RewriteSemantics::ExactBits,
      "A 16-bit floating field in a 32-bit value; scoped discrete selectors",rewritePackedExtract,rewritePackedExtract,packedExtractCandidate},
  RepresentationRule{"signed-nibble",RewriteSemantics::ExactNumeric,
      "All 16 possible nibble values decode to the signed INT4 value",rewriteSignedNibble,rewriteSignedNibble,signedNibbleCandidate},
  RepresentationRule{"bf16-widened-read",RewriteSemantics::ExactNumeric,
      "A BF16 read widened by a 16-bit exponent-aligned shift",rewriteBf16WidenedRead,rewriteBf16WidenedRead,bf16WidenedCandidate}
};
bool scopedCandidate(const s::Graph& graph,const s::Ref& ref) {
  for(const auto& rule:rules)if(rule.apply_scoped && (!rule.candidate || rule.candidate(graph,ref)))return true;
  return false;
}
s::Ref rewriteCheap(s::Graph& graph,s::Ref ref,std::vector<std::string_view>& applied) {
  // The same registry, only its existing budget-independent rules. A cheap
  // typed read may reveal a different candidate; scheduling checks AFTER this
  // bounded fixed point. No child/ancestor walk or separate normalizer exists.
  for(unsigned round=0;round<16;++round) {
    bool changed=false;
    for(const auto& rule:rules)if(!rule.apply_scoped) {
      if(rule.candidate && !rule.candidate(graph,ref))continue;
      auto next=rule.apply(graph,ref);
      if(next!=ref){applied.push_back(rule.id);ref=std::move(next);changed=true;}
    }
    if(!changed)break;
  }
  return ref;
}
}
std::span<const RepresentationRule> representationRules(){return rules;}
scalar_fold::Ref rewriteRepresentation(scalar_fold::Graph& graph,scalar_fold::Ref ref,
    std::vector<std::string_view>* applied_rules) {
  RepresentationRewriter rewriter(graph);return rewriter.rewrite(ref,{},applied_rules);
}
scalar_fold::Ref rewriteRepresentation(scalar_fold::Graph& graph,scalar_fold::Ref ref,
    BitOriginAnalysis& analysis,const BitDomains& domains,std::vector<std::string_view>* applied_rules) {
  if(ref.conversion.size()>128 || ref.converted_type.size()>128 || !ref.pointer_provenance.empty())return ref;
  for(unsigned round=0;round<16;++round) {
    bool changed=false;
    for(const auto& rule:rules) {
      if(rule.candidate && !rule.candidate(graph,ref))continue;
      // Exhausting the NEW bit-analysis budget must not disable the existing
      // exact-use normalizer for later nodes in a large streamed artifact.
      if(rule.apply_scoped && !analysis.chargeWork(1))continue;
      auto next=rule.apply_scoped?rule.apply_scoped(graph,ref,analysis,domains):rule.apply(graph,ref);
      if(next!=ref) {
        if(applied_rules)applied_rules->push_back(rule.id);
        ref=std::move(next);changed=true;
      }
    }
    if(!changed)return ref;
  }
  throw std::invalid_argument("representation_rewrite_budget");
}
struct RepresentationRewriter::Impl {
  struct Key {
    scalar_fold::Ref value;std::size_t scope=0;
    bool operator==(const Key&) const = default;
  };
  struct KeyHash {
    std::size_t operator()(const Key& key) const {
      // RefHash covers node, complete conversion/type, owner and provenance;
      // collisions are resolved by complete Ref + validated-scope equality.
      auto hash=scalar_fold::RefHash{}(key.value);
      return hash^(key.scope+0x9e3779b9u+(hash<<6)+(hash>>2));
    }
  };
  struct Result {scalar_fold::Ref value;std::vector<std::string_view> rules;};
  scalar_fold::Graph& graph;
  BitOriginAnalysis bits;
  std::vector<BitDomains> scopes;
  std::unordered_map<Key,Result,KeyHash> memo;
  std::uint64_t hits=0,misses=0,lookup_work=0;
  std::uint64_t candidate_uses=0,candidate_bypasses=0;
  explicit Impl(scalar_fold::Graph& g):graph(g),bits(g) {}
  static void increment(std::uint64_t& value) {
    if(value!=std::numeric_limits<std::uint64_t>::max())++value;
  }
  void chargeLookup(std::size_t amount) {
    lookup_work=amount>std::numeric_limits<std::uint64_t>::max()-lookup_work?
        std::numeric_limits<std::uint64_t>::max():lookup_work+amount;
  }
  std::optional<std::size_t> scope(const BitDomains& domains) {
    if(domains.size()>64)return {};
    chargeLookup(domains.size()+1);
    for(auto [level,count]:domains) {
      (void)level;if(!count || count-1>std::uint64_t(INT64_MAX))return {};
    }
    // Lookups do not create numerical facts. Keep access to established cache
    // entries after the separately capped bit-analysis budget has exhausted.
    for(std::size_t i=0;i<scopes.size();++i) {
      chargeLookup(domains.size()+1);if(scopes[i]==domains)return i;
    }
    if(scopes.size()>=128)return {};scopes.push_back(domains);return scopes.size()-1;
  }
};
RepresentationRewriter::RepresentationRewriter(scalar_fold::Graph& graph):impl_(std::make_unique<Impl>(graph)) {}
RepresentationRewriter::~RepresentationRewriter()=default;
scalar_fold::Ref RepresentationRewriter::rewrite(scalar_fold::Ref ref,const BitDomains& domains,
    std::vector<std::string_view>* applied_rules) {
  if(ref.conversion.size()>128 || ref.converted_type.size()>128 || ref.pointer_provenance.size()>128)
    return rewriteRepresentation(impl_->graph,std::move(ref),impl_->bits,domains,applied_rules);
  if(!ref.pointer_provenance.empty())return ref;
  const auto incoming=ref;
  std::vector<std::string_view> cheap_rules;
  ref=rewriteCheap(impl_->graph,std::move(ref),cheap_rules);
  if(!scopedCandidate(impl_->graph,ref)) {
    impl_->increment(impl_->candidate_bypasses);
    if(applied_rules)applied_rules->insert(applied_rules->end(),cheap_rules.begin(),cheap_rules.end());
    return ref;
  }
  impl_->increment(impl_->candidate_uses);
  const auto scope=impl_->scope(domains);
  if(!scope) {
    if(applied_rules)applied_rules->insert(applied_rules->end(),cheap_rules.begin(),cheap_rules.end());
    return rewriteRepresentation(impl_->graph,std::move(ref),impl_->bits,domains,applied_rules);
  }
  impl_->chargeLookup(incoming.conversion.size()+incoming.converted_type.size()+incoming.pointer_provenance.size()+1);
  Impl::Key key{incoming,*scope};
  if(auto cached=impl_->memo.find(key);cached!=impl_->memo.end()) {
    Impl::increment(impl_->hits);
    if(applied_rules)applied_rules->insert(applied_rules->end(),cached->second.rules.begin(),cached->second.rules.end());
    return cached->second.value;
  }
  Impl::increment(impl_->misses);
  std::vector<std::string_view> rules=std::move(cheap_rules);
  auto result=rewriteRepresentation(impl_->graph,ref,impl_->bits,domains,&rules);
  if(applied_rules)applied_rules->insert(applied_rules->end(),rules.begin(),rules.end());
  if(impl_->memo.size()<4096 && rules.size()<=80 && result.conversion.size()<=128 &&
      result.converted_type.size()<=128 && result.pointer_provenance.size()<=128)
    impl_->memo.emplace(std::move(key),Impl::Result{result,std::move(rules)});
  return result;
}
RepresentationRewriteStatistics RepresentationRewriter::statistics() const {
  return {impl_->memo.size(),impl_->scopes.size(),impl_->hits,impl_->misses,impl_->lookup_work,impl_->bits.statistics(),
          impl_->candidate_uses,impl_->candidate_bypasses};
}
}
