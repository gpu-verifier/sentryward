#include "sentryward/algorithm/diagnostics/development_trace.h"
#ifdef SENTRYWARD_ENABLE_COMPRESSION_AUDIT
#include "sentryward/scalar_fold/printer.h"
#endif
#include "sentryward/scalar_fold/real_expression_builder.h"
#include "sentryward/equivalence/float_from_chars.h"
#include "sentryward/equivalence/egraph.h"
#include "sentryward/equivalence/conversion_uses.h"
#include "sentryward/equivalence/constant_values.h"
#include "sentryward/algorithm/diagnostics/compression_audit.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#define UPDATE_STAT(code) do { code; } while(false)
#else
#define UPDATE_STAT(code) ((void)0)
#endif

namespace sentryward::scalar_fold {
namespace {
bool realType(const std::string& t) {
  return t=="real" || t=="e4m3" || t=="e5m2" || t=="f16" ||
    t=="bf16" || t=="f32" || t=="f64";
}
bool clean(const Ref& r) {
  return r.conversion.empty() && r.converted_type.empty() && r.pointer_provenance.empty();
}
bool ideal(const Node& n) {
  return realType(n.type) && n.kind==Kind::Operation &&
    (n.attributes.empty() || n.attributes=="ideal_arithmetic");
}
bool finiteOperation(const std::string& n) {
  return n=="Add" || n=="Subtract" || n=="Multiply" || n=="Divide" ||
    n=="Negate" || n=="Exp2" || n=="Maximum" || n=="Minimum" || n=="Reciprocal";
}
constexpr std::size_t kSummaryAtoms=8,kCacheEntries=256,kLocalNodes=64;
struct RefLess {
  bool operator()(const Ref& a,const Ref& b) const {
    return std::tie(a.node,a.owner,a.conversion,a.converted_type,a.pointer_provenance)<
      std::tie(b.node,b.owner,b.conversion,b.converted_type,b.pointer_provenance);
  }
};
using Powers=std::map<Ref,std::int64_t,RefLess>;
struct Factor {Powers powers;std::vector<Ref> weights;bool exponential=false;};
bool merge(Powers& target,const Powers& source,std::int64_t sign) {
  for(const auto& [atom,n]:source) {
    std::int64_t delta,next;
    if(__builtin_mul_overflow(n,sign,&delta) || __builtin_add_overflow(target[atom],delta,&next))return false;
    if(next)target[atom]=next;else target.erase(atom);
  }
  return target.size()<=kSummaryAtoms;
}
}
struct RealExpressionBuilder::Impl {
  Graph& graph;
  ReductionBuilder& reductions;
  std::function<void()> poll;
  Statistics stats;
  struct Facts {
    bool finite=false,positive=false,nonnegative=false;
    std::shared_ptr<const Powers> exponent;
    std::shared_ptr<const Factor> factor;
  };
  std::vector<Facts> facts;
  std::uint64_t owner=0;
  struct Compiled {
    std::unique_ptr<equivalence::FrozenUpdateEgraph> index;
    bool proved=false;
  };
  std::unordered_map<std::string,Compiled> updates;
  void check() const {if(poll)poll();}
  bool preservesFinite(const Ref& r) const {
    if(clean(r))return true;
    if(!r.pointer_provenance.empty() || r.conversion.empty())return false;
    const auto& source=graph.node(r).type;
    const auto& target=graph.type(r);
    const auto normalized=equivalence::normalizeConversions(source,target,r.conversion);
    return normalized && ((normalized->empty() && source==target) ||
      (equivalence::exactFloatingWidening(source,target) &&
       *normalized=="NumericInterpretation("+source+" -> "+target+")"));
  }
  bool finite(const Ref& r) const{return r.node<facts.size() && facts[r.node].finite && preservesFinite(r);}
  bool positive(const Ref& r) const{return r.node<facts.size() && facts[r.node].positive && preservesFinite(r);}
  bool nonnegative(const Ref& r) const{return r.node<facts.size() && facts[r.node].nonnegative && preservesFinite(r);}
  bool literal(const Ref& r,const std::string& type,const char* text) const {
    const auto& n=graph.node(r);
    return clean(r) && n.kind==Kind::Constant && n.type==type && n.attributes.empty() && n.name==text;
  }
  Powers exponentOf(const Ref& r) const {
    if(clean(r) && r.node<facts.size() && facts[r.node].exponent)return *facts[r.node].exponent;
    return {{r,1}};
  }
  Factor factorOf(const Ref& r) const {
    if(clean(r) && r.node<facts.size() && facts[r.node].factor)return *facts[r.node].factor;
    return {{},{r},false};
  }
  bool hasExponentialFactor(const Ref& r) const {
    return clean(r) && r.node<facts.size() && facts[r.node].factor && facts[r.node].factor->exponential;
  }
  // One topological pass over newly created nodes. Summaries are bounded
  // metadata, never a recursive ancestry query and never an expanded Fold.
  void refresh() {
    if(!graph.size()){facts.clear();owner=0;return;}
    if(graph.size() && owner!=graph.ref(0).owner){owner=graph.ref(0).owner;facts.clear();}
    while(facts.size()<graph.size()) {
      if(!(facts.size()%256))check();
      const auto& n=graph.node(facts.size());Facts f;
      if(realType(n.type)) {
        // InitialCell is the legacy alias of InputValue. This fact covers
        // typed numerical input reads, not their exact address operands.
        if(n.kind==Kind::Parameter || n.kind==Kind::InputValue)f.finite=true;
        else if(n.kind==Kind::Constant) {
          // A raw PTX literal is not a decimal string. Reuse exact decoding;
          // NaN/infinity deliberately do not acquire a finite fact.
          const auto decoded=equivalence::constantValue(n.type,n.name,{},{});
          const auto& text=decoded?decoded->literal:n.name;
          double x=0;const auto p=equivalence::floatFromChars(text.data(),text.data()+text.size(),x);
          f.finite=p.ec==std::errc{} && p.ptr==text.data()+text.size() && std::isfinite(x);
          f.positive=f.finite && x>0;f.nonnegative=f.finite && x>=0;
          if(f.finite && x==0)f.exponent=std::make_shared<Powers>();
          if(f.finite && x==1)f.factor=std::make_shared<Factor>();
        } else if(ideal(n) && n.name=="Select" && n.operands.size()==3) {
          // Selecting between finite values stays finite for either predicate.
          // The Select remains an opaque value: this does not erase a mask,
          // establish a branch choice or propagate the predicate's expression.
          f.finite=finite(n.operands[1]) && finite(n.operands[2]);
          f.positive=f.finite && positive(n.operands[1]) && positive(n.operands[2]);
          f.nonnegative=f.finite && nonnegative(n.operands[1]) && nonnegative(n.operands[2]);
        } else if((ideal(n) && finiteOperation(n.name)) || n.kind==Kind::Fold) {
          f.finite=std::all_of(n.operands.begin(),n.operands.end(),[&](const auto& r){return finite(r);});
          if(ideal(n)) {
            if(n.name=="Divide")f.finite=f.finite && positive(n.operands.at(1));
            if(n.name=="Reciprocal")f.finite=f.finite && positive(n.operands.at(0));
            if(n.name=="Maximum" || n.name=="Minimum")for(unsigned side=0;side<2;++side)
              if(literal(n.operands[side],n.type,n.name=="Maximum"?"-inf":"inf") && finite(n.operands[1-side]))
                f.finite=true;
            if(n.name=="Exp2" || n.name=="Reciprocal")f.positive=f.nonnegative=f.finite;
            else if(n.name=="Add" || n.name=="Multiply") {
              f.nonnegative=f.finite && nonnegative(n.operands[0]) && nonnegative(n.operands[1]);
              f.positive=f.nonnegative && (n.name=="Add"?
                positive(n.operands[0]) || positive(n.operands[1]):positive(n.operands[0]) && positive(n.operands[1]));
            }
            if(f.finite && (n.name=="Add" || n.name=="Subtract" || n.name=="Negate")) {
              auto p=exponentOf(n.operands[0]);
              bool ok=true;
              if(n.name=="Negate")for(auto& [atom,coefficient]:p) {
                (void)atom;
                if(coefficient==std::numeric_limits<std::int64_t>::min()){ok=false;break;}
                coefficient=-coefficient;
              } else ok=merge(p,exponentOf(n.operands[1]),n.name=="Subtract"?-1:1);
              if(ok)f.exponent=std::make_shared<Powers>(std::move(p));
            }
            if(f.finite && n.name=="Exp2")
              f.factor=std::make_shared<Factor>(Factor{exponentOf(n.operands[0]),{},true});
            // Ordinary products are opaque factors. Flatten only an actual
            // exponential update; don't allocate/copy weights for every MMA
            // product merely because a later consumer could be an Exp2.
            if(f.finite && n.name=="Multiply" &&
                (hasExponentialFactor(n.operands[0]) || hasExponentialFactor(n.operands[1]))) {
              auto a=factorOf(n.operands[0]),b=factorOf(n.operands[1]);
              if(a.weights.size()+b.weights.size()<=kSummaryAtoms && merge(a.powers,b.powers,1)) {
                a.weights.insert(a.weights.end(),b.weights.begin(),b.weights.end());a.exponential|=b.exponential;
                f.factor=std::make_shared<Factor>(std::move(a));
              }
            }
          } else {
            if(n.fold->count==0){f=facts.at(n.operands[0].node);if(!clean(n.operands[0]))f={};}
            else if(n.fold->reducer==Reducer::Sum) {
              f.nonnegative=f.finite && nonnegative(n.operands[0]) && nonnegative(n.operands[1]);
              f.positive=f.nonnegative && (positive(n.operands[0]) || positive(n.operands[1]));
            } else if((n.fold->reducer==Reducer::Max || n.fold->reducer==Reducer::Min) &&
                literal(n.operands[0],n.type,n.fold->reducer==Reducer::Max?"-inf":"inf") && finite(n.operands[1]))
              f.finite=true;
          }
        }
      }
      facts.push_back(std::move(f));UPDATE_STAT(++stats.summary_nodes);
      SENTRYWARD_AUDIT(audit->add(AuditCounter::update_summary_nodes));
    }
  }
  Ref constant(const std::string& type,std::string text) {check();return graph.constant(type,std::move(text));}
  Ref raw(std::string name,const std::string& type,std::vector<Ref> args) {
    check();return graph.operation(std::move(name),type,std::move(args),"ideal_arithmetic");
  }
  Ref normalizedConstant(Ref value,const std::string& type) {
    const auto& node=graph.node(value);
    if(node.kind!=Kind::Constant || !value.pointer_provenance.empty())return value;
    if(const auto decoded=equivalence::constantValue(node.type,node.name,value.conversion,value.converted_type);
        decoded && decoded->type==type)return constant(type,decoded->literal);
    unsigned width=0;std::uint64_t infinity=0;
    if(type=="f16"){width=16;infinity=0x7c00;}else if(type=="bf16"){width=16;infinity=0x7f80;}
    else if(type=="f32"){width=32;infinity=0x7f800000;}else if(type=="f64"){width=64;infinity=0x7ff0000000000000ULL;}
    if(!width)return value;
    auto conversion=value.conversion;if(!conversion.empty())conversion+="; ";
    conversion+="Bitcast("+type+" -> b"+std::to_string(width)+")";
    const auto bits=equivalence::constantValue(node.type,node.name,conversion,"b"+std::to_string(width));
    if(!bits)return value;
    std::uint64_t integer=0;auto p=std::from_chars(bits->literal.data(),bits->literal.data()+bits->literal.size(),integer);
    if(p.ec!=std::errc{} || p.ptr!=bits->literal.data()+bits->literal.size())return value;
    if(integer==infinity)return constant(type,"inf");
    if(integer==(infinity|(std::uint64_t{1}<<(width-1))))return constant(type,"-inf");
    return value;
  }
  Ref body(const std::vector<Ref>& weights,const std::string& type) {
    auto result=constant(type,"1");
    for(const auto& r:weights)result=literal(result,type,"1")?r:raw("Multiply",type,{result,r});
    return result;
  }
  Ref materialize(const Factor& factor,const std::string& type) {
    auto result=body(factor.weights,type);if(factor.powers.empty())return result;
    auto power=constant(type,"0");
    for(const auto& [atom,n]:factor.powers) {
      auto term=atom;
      if(n!=1)term=n==-1?raw("Negate",type,{atom}):raw("Multiply",type,{constant(type,std::to_string(n)),atom});
      power=literal(power,type,"0")?term:raw("Add",type,{power,term});
    }
    auto scale=raw("Exp2",type,{power});
    return literal(result,type,"1")?scale:raw("Multiply",type,{scale,result});
  }
  // A local parameter graph: old Fold/state/body are opaque slots. Repeated
  // actual operands bind to the same slot; no coordinate or instance ID enters
  // the key. Finite/positive facts are checked before this conditional proof.
  struct Local {
    Impl& impl;Graph graph;std::string type;std::vector<Ref> slots;
    std::vector<std::string> positives;
    Ref slot(const Ref& r) {
      auto found=std::find(slots.begin(),slots.end(),r);
      const auto index=found==slots.end()?slots.size():std::size_t(found-slots.begin());
      if(found==slots.end())slots.push_back(r);
      const auto name="p"+std::to_string(index);
      if(impl.positive(r) && std::find(positives.begin(),positives.end(),name)==positives.end())positives.push_back(name);
      return graph.parameter(type,name);
    }
    Ref op(std::string name,std::vector<Ref> operands) {
      return graph.operation(std::move(name),type,std::move(operands),"ideal_arithmetic");
    }
    Ref number(const char* n){return graph.constant(type,n);}
    Ref exp(const Powers& powers) {
      auto power=number("0");
      for(const auto& [atom,n]:powers) {
        auto term=slot(atom);
        if(n!=1)term=n==-1?op("Negate",{term}):op("Multiply",{graph.constant(type,std::to_string(n)),term});
        power=impl.literalLocal(graph,power,"0")?term:op("Add",{power,term});
      }
      return op("Exp2",{power});
    }
    Ref factors(const Factor& f) {
      auto value=number("1");
      for(const auto& weight:f.weights)value=impl.literalLocal(graph,value,"1")?slot(weight):op("Multiply",{value,slot(weight)});
      if(!f.powers.empty())value=impl.literalLocal(graph,value,"1")?exp(f.powers):op("Multiply",{exp(f.powers),value});
      return value;
    }
  };
  bool literalLocal(const Graph& g,const Ref& r,const char* value) const {
    const auto& n=g.node(r);return n.kind==Kind::Constant && n.name==value;
  }
  bool prove(Local& local,Ref source,Ref target,std::uint32_t level) {
    if(local.graph.size()>kLocalNodes) {
      UPDATE_STAT(++stats.budget_rejections);SENTRYWARD_AUDIT(audit->add(AuditCounter::update_budget_rejections));return false;
    }
    std::ostringstream text;text<<"update-rules-v1:"<<level<<':';
    for(const auto& n:local.graph.nodes()) {
      text<<int(n.kind)<<':'<<n.type.size()<<':'<<n.type<<':'<<n.name.size()<<':'<<n.name<<':'<<n.attributes<<'[';
      for(const auto& r:n.operands)text<<r.node<<',';
      text<<']';
    }
    text<<source.node<<'/'<<target.node<<':';for(const auto& p:local.positives)text<<p<<',';
    auto key=text.str();auto found=updates.find(key);
    if(found!=updates.end()) {
      UPDATE_STAT(++stats.cache_hits);SENTRYWARD_AUDIT(audit->add(AuditCounter::update_cache_hits));return found->second.proved;
    }
    UPDATE_STAT(++stats.cache_misses);SENTRYWARD_AUDIT(audit->add(AuditCounter::update_cache_misses));
    if(updates.size()>=kCacheEntries) {
      UPDATE_STAT(++stats.cache_capacity_rejections);
      SENTRYWARD_AUDIT(audit->add(AuditCounter::update_capacity_rejections));return false;
    }
    equivalence::Options options;options.max_nodes=2048;options.max_iterations=6;
    options.max_work=100000;options.timeout=std::chrono::milliseconds(200);
    const std::array roots{source,target};
    auto index=std::make_unique<equivalence::FrozenUpdateEgraph>(local.graph,roots,options,local.positives);
    UPDATE_STAT(++stats.compiles;stats.compile_nodes+=index->compilation().statistics.nodes);
    SENTRYWARD_AUDIT(audit->add(AuditCounter::update_compiles);
      audit->add(AuditCounter::update_compile_nodes,index->compilation().statistics.nodes));
    // Import already certified the normalized meaning of these exact roots.
    // Looking up their pre-normalized syntax can legitimately miss a removed
    // Multiply(...,1) or equivalent reciprocal spelling in the frozen index.
    auto a=index->importedRoot(0),b=index->importedRoot(1);
    UPDATE_STAT(stats.query_work+=a.work+b.work);
    SENTRYWARD_AUDIT(audit->add(AuditCounter::update_query_work,a.work+b.work));
    const bool proved=a.eclass && b.eclass && a.eclass==b.eclass;
    SENTRYWARD_AUDIT((void)audit;if(!proved) {
      SENTRYWARD_DEBUG_PRINT(real_expression_builder_0);
      for(const auto& name:local.positives)SENTRYWARD_DEBUG_PRINT(real_expression_builder_1);
      SENTRYWARD_DEBUG_PRINT(real_expression_builder_2);writeText(std::cerr,local.graph,roots);
    });
    if(index->compilation().reason!="saturated") {
      UPDATE_STAT(++stats.budget_rejections);SENTRYWARD_AUDIT(audit->add(AuditCounter::update_budget_rejections));
    }
    updates.emplace(std::move(key),Compiled{std::move(index),proved});UPDATE_STAT(stats.templates=updates.size());
    return proved;
  }
  Ref operation(std::string name,std::string type,std::vector<Ref> operands,
                std::string attributes,std::uint32_t level) {
    check();
    if(!realType(type) || (!attributes.empty() && attributes!="ideal_arithmetic") || !finiteOperation(name))
      return graph.operation(std::move(name),std::move(type),std::move(operands),std::move(attributes));
    const std::size_t arity=name=="Exp2" || name=="Negate" || name=="Reciprocal"?1:2;
    if(operands.size()!=arity)throw std::invalid_argument("real expression operand count differs");
    for(const auto& r:operands)if(graph.type(r)!=type)throw std::invalid_argument("real expression operand types differ");
    for(auto& r:operands)r=normalizedConstant(r,type);
    refresh();
    if(name=="Subtract") {
      if(literal(operands[0],type,"-inf") && finite(operands[1]))return constant(type,"-inf");
      return operation("Add",type,{operands[0],operation("Negate",type,{operands[1]},attributes,level)},attributes,level);
    }
    if(name=="Divide")return operation("Multiply",type,{operands[0],operation("Reciprocal",type,{operands[1]},attributes,level)},attributes,level);
    if(name=="Exp2" && literal(operands[0],type,"-inf"))return constant(type,"0");
    if(name=="Maximum" || name=="Minimum")for(unsigned side=0;side<2;++side)
      if(literal(operands[side],type,name=="Maximum"?"-inf":"inf") && finite(operands[1-side]))return operands[1-side];
    if(name=="Multiply")for(unsigned side=0;side<2;++side)
      if(literal(operands[side],type,"0") && finite(operands[1-side]))return constant(type,"0");
    if(name=="Exp2" && finite(operands[0])) {
      const auto powers=exponentOf(operands[0]);
      if(powers.empty())return constant(type,"1"); // certified additive summary, no walk
    }
    if((name=="Add" || name=="Multiply") && finite(operands[0]) && finite(operands[1]) &&
        (hasExponentialFactor(operands[0]) || hasExponentialFactor(operands[1]))) {
      auto a=factorOf(operands[0]),b=factorOf(operands[1]);
      if(a.exponential || b.exponential) {
        Local local{*this,{},type,{},{}};
        auto source=local.op(name,{local.factors(a),local.factors(b)});
        if(name=="Multiply" && a.weights.size()+b.weights.size()<=kSummaryAtoms) {
          auto combined=a;
          if(merge(combined.powers,b.powers,1)) {
            combined.weights.insert(combined.weights.end(),b.weights.begin(),b.weights.end());
            if(prove(local,source,local.factors(combined),level))return materialize(combined,type);
          }
        } else if(name=="Add") {
          Powers common;
          for(const auto& [atom,n]:a.powers)if(auto p=b.powers.find(atom);p!=b.powers.end() && p->second==n)common.emplace(atom,n);
          if(!common.empty() && merge(a.powers,common,-1) && merge(b.powers,common,-1)) {
            auto target=local.op("Multiply",{local.exp(common),local.op("Add",{local.factors(a),local.factors(b)})});
            if(prove(local,source,target,level)) {
              auto value=reductions.append(Reducer::Sum,materialize(a,type),materialize(b,type),type,level);
              return materialize(Factor{std::move(common),{value},true},type);
            }
          }
        }
      }
    }
    if(name=="Reciprocal" && positive(operands[0])) {
      auto factor=factorOf(operands[0]);
      if(factor.exponential) {
        Powers opposite;
        if(merge(opposite,factor.powers,-1)) {
          auto weight=body(factor.weights,type);refresh();
          if(positive(weight)) {
            Local local{*this,{},type,{},{}};
            auto source=local.op("Reciprocal",{local.factors(factor)});
            auto target=local.op("Multiply",{local.exp(opposite),local.op("Reciprocal",{local.slot(weight)})});
            if(prove(local,source,target,level))
              return materialize(Factor{std::move(opposite),{raw("Reciprocal",type,{weight})},true},type);
          }
        }
      }
    }
    std::optional<Reducer> reducer;
    if(name=="Add")reducer=Reducer::Sum;else if(name=="Multiply")reducer=Reducer::Product;
    else if(name=="Maximum")reducer=Reducer::Max;else if(name=="Minimum")reducer=Reducer::Min;
    if(reducer)return reductions.append(*reducer,operands[0],operands[1],std::move(type),level);
    return raw(std::move(name),type,std::move(operands));
  }
  void remap(std::span<const Id> mapping,std::uint64_t next_owner) {
    std::vector<Facts> retained;
    for(Id id=0;id<facts.size();++id)if(mapping[id]!=std::numeric_limits<Id>::max()) {
      if(retained.size()<=mapping[id])retained.resize(mapping[id]+1);
      auto f=std::move(facts[id]);
      const auto remapRef=[&](Ref& r) {
        if(r.node>=mapping.size() || mapping[r.node]==std::numeric_limits<Id>::max())return false;
        r.node=mapping[r.node];r.owner=next_owner;return true;
      };
      if(f.exponent) {
        Powers p;bool ok=true;
        for(const auto& [atom,n]:*f.exponent){auto r=atom;if(!remapRef(r)){ok=false;break;}p.emplace(std::move(r),n);}
        f.exponent=ok?std::make_shared<Powers>(std::move(p)):nullptr;
      }
      if(f.factor) {
        Factor p=*f.factor;bool ok=true;Powers powers;
        for(const auto& [atom,n]:p.powers){auto r=atom;if(!remapRef(r)){ok=false;break;}powers.emplace(std::move(r),n);}
        for(auto& r:p.weights)ok=remapRef(r) && ok;
        p.powers=std::move(powers);f.factor=ok?std::make_shared<Factor>(std::move(p)):nullptr;
      }
      retained[mapping[id]]=std::move(f);UPDATE_STAT(++stats.summary_remaps);
    }
    facts=std::move(retained);owner=next_owner;
  }
};
RealExpressionBuilder::RealExpressionBuilder(Graph& graph,ReductionBuilder& reductions,std::function<void()> poll)
  :impl_(std::make_unique<Impl>(Impl{graph,reductions,std::move(poll),{}, {},0,{}})){}
RealExpressionBuilder::~RealExpressionBuilder()=default;
Ref RealExpressionBuilder::constant(std::string type,std::string literal) {
  return impl_->normalizedConstant(impl_->constant(type,std::move(literal)),type);
}
Ref RealExpressionBuilder::operation(std::string name,std::string type,std::vector<Ref> operands,
    std::string attributes,std::uint32_t level) {
  return impl_->operation(std::move(name),std::move(type),std::move(operands),std::move(attributes),level);
}
std::size_t RealExpressionBuilder::collect(std::span<Ref> roots){return collect(roots,roots);}
std::size_t RealExpressionBuilder::collect(std::span<Ref> roots,std::span<const Ref> extendable,
    std::span<std::optional<Id>> weak) {
  impl_->refresh();
  return impl_->reductions.collect(roots,extendable,weak,[this](auto map,auto owner){impl_->remap(map,owner);});
}
const RealExpressionBuilder::Statistics& RealExpressionBuilder::statistics() const{return impl_->stats;}
}
#undef UPDATE_STAT
