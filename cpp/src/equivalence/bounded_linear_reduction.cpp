#include "sentryward/equivalence/bounded_linear_reduction.h"
#include "sentryward/equivalence/epoch_work_allowance.h"

#include <algorithm>
#include <bit>
#include <map>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>

namespace sentryward::equivalence::bounded_linear_reduction {
namespace p = polynomial_certificate;
namespace {
bool zero(const Coefficient& value) { return value.terms().empty(); }
std::size_t words(const p::Integer& value) {
  return value == 0 ? 0 : (boost::multiprecision::msb(value < 0 ? -value : value) + 64) / 64;
}
std::size_t width(const Coefficient& value) {
  if(zero(value))return 1;
  const auto& c=value.terms()[0].coefficient;
  return 1+words(c.numerator)+words(c.denominator);
}
Limits bounded(Limits limits) {
  limits.max_atoms=std::min<std::size_t>(limits.max_atoms,512);
  limits.max_nodes=std::min(limits.max_nodes,epoch_work_allowance::maximum_retained_headers);
  limits.max_cached_terms=std::min<std::size_t>(limits.max_cached_terms,32768);
  limits.max_intervals=std::min<std::size_t>(limits.max_intervals,1024);
  return limits;
}
p::Limits coefficientLimits(const Limits& limits) {
  auto result=limits.coefficients;result.max_terms=1;
  result.max_work=std::min(result.max_work,limits.max_work);
  return result;
}
bool realType(std::string_view type) {
  return type=="real" || type=="f16" || type=="bf16" || type=="f32" || type=="f64";
}
std::size_t arity(NodeKind kind) {
  if(kind==NodeKind::Atom || kind==NodeKind::Literal)return 0;
  if(kind==NodeKind::Negate || kind==NodeKind::Identity)return 1;
  if(kind==NodeKind::Add || kind==NodeKind::Subtract || kind==NodeKind::Multiply || kind==NodeKind::Divide)return 2;
  return std::numeric_limits<std::size_t>::max();
}
} // namespace

std::optional<std::vector<bool>> rankedDagHeads(std::span<const NodeId> owners,
    const std::function<std::span<const NodeId>(NodeId)>& children,
    std::size_t max_nodes,std::size_t max_work,RankedDagStatistics* statistics) {
  RankedDagStatistics local;if(!statistics)statistics=&local;*statistics={};
  // Bound the entire stored snapshot, not logical Fold instances. Engine
  // callers supply their remaining optional proof budget explicitly.
  max_nodes=std::min<std::size_t>(max_nodes,1000000);
  max_work=std::min<std::size_t>(max_work,4000000);
  if(owners.size()>max_nodes || !max_work || !children)return {};
  const auto charge=[&](std::size_t amount) {
    if(amount>max_work-statistics->work)return false;
    statistics->work+=amount;return true;
  };
  // Account for initialized ranks and result mask as well as both scans.
  if(!charge(owners.size()))return {};
  std::vector<NodeId> rank(owners.size(),owners.size());
  std::vector<bool> result(owners.size(),true);
  for(NodeId id=0;id<owners.size();++id) {
    if(!charge(1) || owners[id]>=owners.size() || owners[owners[id]]!=owners[id])return {};
    ++statistics->nodes;rank[owners[id]]=std::min(rank[owners[id]],id);
  }
  for(NodeId id=0;id<owners.size();++id) {
    if(!charge(1))return {};
    for(auto child:children(id)) {
      if(!charge(1) || child>=owners.size())return {};
      ++statistics->edges;
      if(rank[owners[child]]>=rank[owners[id]])result[id]=false;
    }
  }
  return result;
}

Builder::Builder(Limits limits,WorkCharge work_charge)
    : limits_(bounded(limits)),work_charge_(work_charge?
        std::make_shared<WorkCharge>(std::move(work_charge)):nullptr),
      coefficients_(coefficientLimits(limits_),work_charge_?
        WorkCharge{[hook=work_charge_](std::size_t n){return (*hook)(n);}}:WorkCharge{}) {}
std::size_t Builder::work() const { return local_work_+coefficients_.work(); }
bool Builder::exhausted() const { return exhausted_ || coefficients_.exhausted(); }
bool Builder::charge(std::size_t amount) {
  if(exhausted() || work()>limits_.max_work || amount>limits_.max_work-work() ||
      (work_charge_ && !(*work_charge_)(amount))) {
    exhausted_=true;return false;
  }
  local_work_+=amount;return true;
}
bool Builder::constantCoefficient(const Coefficient& value) {
  if(value.terms().size()>1 || (!zero(value) && !value.terms()[0].monomial.powers.empty()))return false;
  if(!charge(width(value)))return false;
  // Builder-created immutable coefficients have already passed the exact
  // integer size/normalization checks. This also checks caller-supplied keys.
  const Coefficient zero_value;
  return coefficients_.apply(p::Operation::Add,std::array{&value,&zero_value}).has_value() && charge(0);
}
bool Builder::admitted(const LinearForm& value) {
  if(value.terms_.size()>limits_.max_atoms || !constantCoefficient(value.constant_))return false;
  std::optional<Atom> previous;
  for(const auto& term:value.terms_) {
    if((previous && *previous>=term.atom) || zero(term.weight) || !constantCoefficient(term.weight))return false;
    previous=term.atom;
  }
  return charge(value.terms_.size());
}
std::optional<Coefficient> Builder::coefficient(std::string_view literal) {
  if(!charge(1))return {};
  auto result=coefficients_.literal(literal);
  if(!result || !charge(0))return {};
  return result;
}
Result Builder::literal(std::string_view literal) {
  auto c=coefficient(literal);if(!c)return {};
  LinearForm result;result.constant_=std::move(*c);return result;
}
Result Builder::atom(Atom atom) {
  if(!limits_.max_atoms || !charge(1))return {};
  auto one=coefficient("1");if(!one)return {};
  LinearForm result;result.terms_.push_back({atom,std::move(*one)});return result;
}
std::optional<Coefficient> Builder::combineCoefficient(const Coefficient& a,const Coefficient& b,bool subtract) {
  if(!charge(1))return {};
  auto result=coefficients_.apply(subtract?p::Operation::Subtract:p::Operation::Add,std::array{&a,&b});
  if(!result || !charge(0))return {};
  return result;
}
std::optional<Coefficient> Builder::productCoefficient(const Coefficient& a,const Coefficient& b) {
  if(!charge(1))return {};
  auto result=coefficients_.apply(p::Operation::Multiply,std::array{&a,&b});
  if(!result || !charge(0))return {};
  return result;
}
std::optional<Coefficient> Builder::reciprocalCoefficient(const Coefficient& value) {
  if(zero(value) || !constantCoefficient(value))return {};
  const auto& rational=value.terms()[0].coefficient;
  // The existing literal helper owns parse/BigInt limits. At the largest bit
  // bound, decimal conversion may exceed its character cap: conservatively
  // return Unknown, never bypass that cap to accept a derived denominator.
  if(!charge(width(value)))return {};
  const auto numerator=rational.numerator.convert_to<std::string>();
  const auto denominator=rational.denominator.convert_to<std::string>();
  auto result=coefficient(denominator);if(!result)return {};
  result=coefficients_.divideByLiteral(*result,numerator);
  if(!result || !charge(0))return {};
  return result;
}
Result Builder::combine(const LinearForm& a,const LinearForm& b,bool subtract) {
  if(!charge(a.terms_.size()+b.terms_.size()))return {};
  LinearForm result;
  auto constant=combineCoefficient(a.constant_,b.constant_,subtract);if(!constant)return {};
  result.constant_=std::move(*constant);
  std::size_t i=0,j=0;
  while(i<a.terms_.size() || j<b.terms_.size()) {
    if(!charge(1))return {};
    Atom atom;std::optional<Coefficient> weight;
    if(j==b.terms_.size() || (i<a.terms_.size() && a.terms_[i].atom<b.terms_[j].atom)) {
      atom=a.terms_[i].atom;
      if(!charge(width(a.terms_[i].weight)))return {};
      weight=a.terms_[i++].weight;
    } else if(i==a.terms_.size() || b.terms_[j].atom<a.terms_[i].atom) {
      atom=b.terms_[j].atom;
      weight=combineCoefficient(Coefficient{},b.terms_[j++].weight,subtract);
    } else {
      atom=a.terms_[i].atom;
      weight=combineCoefficient(a.terms_[i++].weight,b.terms_[j++].weight,subtract);
    }
    if(!weight)return {};
    if(!zero(*weight)) {
      if(result.terms_.size()>=limits_.max_atoms)return {};
      result.terms_.push_back({atom,std::move(*weight)});
    }
  }
  return result;
}
Result Builder::scale(const LinearForm& value,const Coefficient& factor) {
  if(!charge(value.terms_.size()) || !constantCoefficient(factor))return {};
  LinearForm result;
  auto constant=productCoefficient(value.constant_,factor);if(!constant)return {};
  result.constant_=std::move(*constant);
  for(const auto& term:value.terms_) {
    auto weight=productCoefficient(term.weight,factor);if(!weight)return {};
    if(!zero(*weight))result.terms_.push_back({term.atom,std::move(*weight)});
  }
  return result;
}
Result Builder::negate(const LinearForm& value) {
  auto minus=coefficient("-1");if(!minus)return {};
  return scale(value,*minus);
}
Result Builder::apply(Operation operation,std::span<const LinearForm* const> operands) {
  const auto expected=operation==Operation::Negate?1U:2U;
  if(operands.size()!=expected || !charge(1))return {};
  for(auto operand:operands)if(!operand || !admitted(*operand))return {};
  const auto& a=*operands[0];
  if(operation==Operation::Negate)return negate(a);
  const auto& b=*operands[1];
  switch(operation) {
    case Operation::Add:return combine(a,b,false);
    case Operation::Subtract:return combine(a,b,true);
    case Operation::Multiply:
      if(a.terms_.empty())return scale(b,a.constant_);
      if(b.terms_.empty())return scale(a,b.constant_);
      return {};
    case Operation::Divide:
      if(!b.terms_.empty())return {};
      if(auto reciprocal=reciprocalCoefficient(b.constant_))return scale(a,*reciprocal);
      return {};
    case Operation::Negate:break;
  }
  return {};
}

std::optional<bool> Builder::intervalsEqual(std::span<const WeightedInterval> left,
                                          std::span<const WeightedInterval> right) {
  if(left.size()>limits_.max_intervals || right.size()>limits_.max_intervals-left.size())return {};
  auto a=canonicalIntervals(left);if(!a)return {};
  auto b=canonicalIntervals(right);if(!b)return {};
  if(a->size()!=b->size())return false;
  for(std::size_t i=0;i<a->size();++i)
    if((*a)[i].begin!=(*b)[i].begin || (*a)[i].end!=(*b)[i].end || (*a)[i].weight!=(*b)[i].weight)return false;
  return true;
}
std::optional<std::vector<WeightedInterval>> Builder::canonicalIntervals(
    std::span<const WeightedInterval> pieces) {
  if(pieces.size()>limits_.max_intervals)return {};
  struct Event {std::uint64_t point;const Coefficient* weight;bool subtract;};
  std::vector<Event> events;events.reserve(2*pieces.size());
  for(const auto& piece:pieces) {
    if(!charge(1) || piece.end<piece.begin || !constantCoefficient(piece.weight))return {};
    if(piece.begin==piece.end || zero(piece.weight))continue;
    events.push_back({piece.begin,&piece.weight,false});
    events.push_back({piece.end,&piece.weight,true});
  }
  if(!charge(events.size()*(1+std::bit_width(events.size()))))return {};
  std::sort(events.begin(),events.end(),[](const Event& a,const Event& b){return a.point<b.point;});
  Coefficient active;std::uint64_t previous=0;
  std::vector<WeightedInterval> result;
  for(const auto& event:events) {
    if(event.point!=previous && !zero(active)) {
      if(!charge(width(active)))return {};
      if(!result.empty() && result.back().end==previous && result.back().weight==active)result.back().end=event.point;
      else result.push_back({previous,event.point,active});
    }
    auto next=combineCoefficient(active,*event.weight,event.subtract);if(!next)return {};
    active=std::move(*next);previous=event.point;
  }
  if(!zero(active))return {};
  return result;
}

DagSession::DagSession(Builder& builder,Resolver resolve,Limits limits)
    : builder_(builder),resolve_(std::move(resolve)),limits_(bounded(limits)),
      initial_work_(builder.work()) {
  limits_.max_atoms=std::min(limits_.max_atoms,builder_.limits_.max_atoms);
  limits_.max_nodes=std::min(limits_.max_nodes,builder_.limits_.max_nodes);
  limits_.max_cached_terms=std::min(limits_.max_cached_terms,builder_.limits_.max_cached_terms);
  limits_.max_work=std::min(limits_.max_work,builder_.limits_.max_work);
}

Statistics DagSession::statistics() const {
  return {entries_.size(),peak_terms_,builder_.work()-initial_work_,edges_,queries_};
}
bool DagSession::charge(std::size_t amount) {
  if(exhausted_)return false;
  if(builder_.exhausted() || builder_.work()-initial_work_>limits_.max_work ||
      amount>limits_.max_work-(builder_.work()-initial_work_) || !builder_.charge(amount)) {
    exhausted_=true;cap_=Cap::Work;return false;
  }
  return true;
}
bool DagSession::live(std::size_t query_terms) {
  // Include a fixed allowance for the current adjoint and exact-operation
  // temporaries; coefficient words themselves have their existing bit/work cap.
  constexpr std::size_t scratch_terms=4;
  if(constant_terms_>limits_.max_cached_terms ||
      scratch_terms>limits_.max_cached_terms-constant_terms_ ||
      query_terms>limits_.max_cached_terms-constant_terms_-scratch_terms) {
    exhausted_=true;cap_=Cap::CachedTerms;return false;
  }
  peak_terms_=std::max(peak_terms_,constant_terms_+query_terms+scratch_terms);return charge(0);
}
bool DagSession::coefficientFailure() {
  exhausted_=true;cap_=builder_.exhausted()?Cap::Work:Cap::Coefficients;return false;
}
bool DagSession::catalog(NodeId root) {
  struct Task {NodeId id;bool ready;};
  std::vector<Task> pending{{root,false}};
  while(!pending.empty()) {
    if(!charge(1))return false;
    const auto [id,ready]=pending.back();pending.pop_back();
    auto found=entries_.find(id);
    if(!ready) {
      if(found!=entries_.end()) {
        if(found->second.state!=State::Ready)return false;
        continue;
      }
      if(entries_.size()>=limits_.max_nodes){exhausted_=true;cap_=Cap::Nodes;return false;}
      auto node=resolve_?resolve_(id):std::nullopt;
      if(!node || !node->admitted || !realType(node->type) || node->child_count>2 || node->child_count!=arity(node->kind)) {
        // Cache the denial too: do not repeat a resolver's unknown source work.
        Entry unknown;if(node)unknown.node=*node;unknown.state=State::Unknown;
        entries_.emplace(id,std::move(unknown));return false;
      }
      Entry entry;entry.node=*node;entries_.emplace(id,std::move(entry));
      edges_+=node->child_count;pending.push_back({id,true});
      for(std::size_t i=0;i<node->child_count;++i)pending.push_back({node->children[i],false});
      continue;
    }
    if(found==entries_.end() || found->second.state!=State::Active)return false;
    auto& entry=found->second;const auto& node=entry.node;
    std::array<const Entry*,2> children{};
    for(std::size_t i=0;i<node.child_count;++i) {
      const auto child=entries_.find(node.children[i]);
      if(child==entries_.end() || child->second.state!=State::Ready || child->second.node.type!=node.type) {
        entry.state=State::Unknown;return false;
      }
      children[i]=&child->second;
    }
    const auto scalar=[&](std::size_t i){return children[i]->constant.has_value();};
    std::optional<Coefficient> constant;
    bool valid=true,coefficient_unknown=false;
    switch(node.kind) {
      case NodeKind::Literal:
        constant=builder_.coefficient(node.literal);valid=constant.has_value();coefficient_unknown=!valid;break;
      case NodeKind::Atom:break;
      case NodeKind::Identity:
        if(scalar(0)) {if(!charge(width(*children[0]->constant)))return false;constant=children[0]->constant;}
        break;
      case NodeKind::Negate:
        if(scalar(0))constant=builder_.combineCoefficient(Coefficient{},*children[0]->constant,true);
        valid=!scalar(0) || constant.has_value();coefficient_unknown=!valid;break;
      case NodeKind::Add:case NodeKind::Subtract:
        if(scalar(0) && scalar(1))constant=builder_.combineCoefficient(
            *children[0]->constant,*children[1]->constant,node.kind==NodeKind::Subtract);
        valid=!(scalar(0) && scalar(1)) || constant.has_value();coefficient_unknown=!valid;break;
      case NodeKind::Multiply:
        valid=scalar(0) || scalar(1);
        if(scalar(0) && scalar(1))constant=builder_.productCoefficient(*children[0]->constant,*children[1]->constant);
        else if(valid)entry.linear_child=node.children[scalar(0)?1:0];
        coefficient_unknown=scalar(0) && scalar(1) && !constant;
        valid=valid && !coefficient_unknown;break;
      case NodeKind::Divide:
        valid=scalar(1) && !zero(*children[1]->constant);
        if(valid) {
          auto reciprocal=builder_.reciprocalCoefficient(*children[1]->constant);
          if(!reciprocal){valid=false;coefficient_unknown=true;}
          else if(scalar(0)) {
            constant=builder_.productCoefficient(*children[0]->constant,*reciprocal);
            valid=constant.has_value();coefficient_unknown=!valid;
          }
          else {entry.scale=std::move(*reciprocal);entry.linear_child=node.children[0];}
        }
        break;
    }
    if(!valid || !charge(0)) {
      entry.state=State::Unknown;if(coefficient_unknown)coefficientFailure();return false;
    }
    entry.constant=std::move(constant);entry.state=State::Ready;
    constant_terms_+=(entry.constant && !zero(*entry.constant)?1U:0U)+(entry.scale?1U:0U);
    if(!live(0))return false;
  }
  return true;
}

Result DagSession::summarize(NodeId root) {
  ++queries_;if(!charge(1) || !catalog(root))return {};
  // The persistent catalog has no expanded forms. Collect this root's stored
  // selected DAG once; cached Ready headers were already type/cycle checked.
  struct Task {NodeId id;bool ready;};
  std::vector<Task> pending{{root,false}};std::vector<NodeId> order;
  std::unordered_set<NodeId> seen;
  while(!pending.empty()) {
    if(!charge(1))return {};
    const auto [id,ready]=pending.back();pending.pop_back();
    const auto& entry=entries_.at(id);
    if(ready){order.push_back(id);continue;}
    if(!seen.insert(id).second)continue;
    pending.push_back({id,true});
    if(entry.constant)continue;
    if(entry.node.kind==NodeKind::Multiply || entry.node.kind==NodeKind::Divide)
      pending.push_back({entry.linear_child,false});
    else for(std::size_t i=0;i<entry.node.child_count;++i)pending.push_back({entry.node.children[i],false});
  }
  auto one=builder_.coefficient("1");
  if(!one){coefficientFailure();return {};}
  if(!charge(0))return {};
  std::unordered_map<NodeId,Coefficient> weights;weights.emplace(root,std::move(*one));
  std::map<Atom,Coefficient> atoms;Coefficient constant;
  if(!live(weights.size()))return {};
  const auto accumulate=[&](NodeId id,const Coefficient& value)->bool {
    if(!charge(1))return false;
    if(zero(value))return true;
    auto previous=weights.find(id);
    if(previous==weights.end()) {
      if(!charge(width(value)))return false;
      weights.emplace(id,value);
    } else {
      auto next=builder_.combineCoefficient(previous->second,value,false);if(!next)return coefficientFailure();
      if(zero(*next))weights.erase(previous);else previous->second=std::move(*next);
    }
    return live(weights.size()+atoms.size()+(!zero(constant)?1U:0U));
  };
  for(auto it=order.rbegin();it!=order.rend();++it) {
    if(!charge(1))return {};
    auto found=weights.find(*it);if(found==weights.end())continue;
    auto weight=std::move(found->second);weights.erase(found);
    const auto& entry=entries_.at(*it);const auto& node=entry.node;
    if(entry.constant) {
      auto value=builder_.productCoefficient(weight,*entry.constant);if(!value){coefficientFailure();return {};}
      auto next=builder_.combineCoefficient(constant,*value,false);if(!next){coefficientFailure();return {};}
      constant=std::move(*next);
    } else if(node.kind==NodeKind::Atom) {
      if(!charge(1+std::bit_width(atoms.size())))return {};
      auto previous=atoms.find(node.atom);
      if(previous==atoms.end()) {
        if(atoms.size()>=limits_.max_atoms){exhausted_=true;cap_=Cap::Atoms;return {};}
        atoms.emplace(node.atom,std::move(weight));
      } else {
        auto next=builder_.combineCoefficient(previous->second,weight,false);if(!next){coefficientFailure();return {};}
        if(zero(*next))atoms.erase(previous);else previous->second=std::move(*next);
      }
    } else if(node.kind==NodeKind::Multiply || node.kind==NodeKind::Divide) {
      const auto coefficient=node.kind==NodeKind::Divide?&*entry.scale:
          &*entries_.at(node.children[node.children[0]==entry.linear_child?1:0]).constant;
      auto next=builder_.productCoefficient(weight,*coefficient);
      if(!next){coefficientFailure();return {};}
      if(!accumulate(entry.linear_child,*next))return {};
    } else {
      for(std::size_t i=0;i<node.child_count;++i) {
        if(node.kind==NodeKind::Negate || (node.kind==NodeKind::Subtract && i==1)) {
          auto next=builder_.combineCoefficient(Coefficient{},weight,true);
          if(!next){coefficientFailure();return {};}
          if(!accumulate(node.children[i],*next))return {};
        } else if(!accumulate(node.children[i],weight))return {};
      }
    }
    if(!live(weights.size()+atoms.size()+(!zero(constant)?1U:0U)))return {};
  }
  if(!weights.empty() || !charge(atoms.size()) || !charge(0))return {};
  LinearForm result;result.constant_=std::move(constant);result.terms_.reserve(atoms.size());
  for(auto& [atom,weight]:atoms)result.terms_.push_back({atom,std::move(weight)});
  return result;
}

Result summarizeDag(NodeId root,const std::function<std::optional<Node>(NodeId)>& resolve,
                    Limits limits,Statistics* statistics) {
  Builder builder(limits);DagSession session(builder,resolve,limits);
  auto result=session.summarize(root);if(statistics)*statistics=session.statistics();return result;
}

} // namespace sentryward::equivalence::bounded_linear_reduction
