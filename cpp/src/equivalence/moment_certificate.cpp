#include "sentryward/equivalence/moment_certificate.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace sentryward::equivalence::moment_certificate {
namespace p=polynomial_certificate;
namespace l=bounded_linear_reduction;
namespace {
Limits bounded(Limits v) {
  v.max_means=std::min<std::size_t>(v.max_means,512);
  v.max_support_entries=std::min<std::size_t>(v.max_support_entries,32768);
  v.max_intervals=std::min<std::size_t>(v.max_intervals,1024);
  v.max_square_terms=std::min<std::size_t>(v.max_square_terms,512);
  v.max_work=std::min<std::size_t>(v.max_work,1000000);
  v.coefficients.max_degree=std::min<std::uint32_t>(v.coefficients.max_degree,2);
  v.coefficients.max_coefficient_bits=std::min<std::size_t>(v.coefficients.max_coefficient_bits,4096);
  v.coefficients.max_intermediate_bits=std::min<std::size_t>(v.coefficients.max_intermediate_bits,8192);
  v.coefficients.max_decimal_chars=std::min<std::size_t>(v.coefficients.max_decimal_chars,1152);
  return v;
}
p::Limits polynomialLimits(const Limits& v) {
  auto c=v.coefficients;c.max_terms=std::min<std::size_t>(c.max_terms,8);
  c.max_work=std::min(c.max_work,v.max_work);return c;
}
l::Limits intervalLimits(const Limits& v) {
  l::Limits c;c.max_intervals=v.max_intervals;c.max_work=v.max_work;
  c.coefficients=v.coefficients;return c;
}
bool zero(const Coefficient& c){return c.terms().empty();}
std::size_t coefficientWords(const Coefficient& c) {
  if(zero(c))return 1;
  // cpp_int backend size is an immediate limb count; no abs/copy of BigInts.
  const auto& q=c.terms()[0].coefficient;
  return 1+q.numerator.backend().size()+q.denominator.backend().size();
}
}
bool equal(const Signature& a,const Signature& b) {
  if(!a.owner || a.owner!=b.owner || a.group!=b.group || a.context!=b.context ||
      a.mean_squares!=b.mean_squares || a.second_moment.size()!=b.second_moment.size())return false;
  for(std::size_t i=0;i<a.second_moment.size();++i) {
    const auto& x=a.second_moment[i];const auto& y=b.second_moment[i];
    if(x.begin!=y.begin || x.end!=y.end || x.weight!=y.weight)return false;
  }
  return true;
}
Builder::Builder(Limits v):limits_(bounded(v)),coefficients_(polynomialLimits(limits_)),
    intervals_(intervalLimits(limits_)) {
  static std::atomic<std::uint64_t> next{1};auto candidate=next.load(std::memory_order_relaxed);
  for(;;) {
    if(!candidate)throw std::overflow_error("moment certificate owner overflow");
    if(next.compare_exchange_weak(candidate,candidate+1,std::memory_order_relaxed)){owner_=candidate;break;}
  }
}
std::size_t Builder::work() const{return local_work_+coefficients_.work()+intervals_.work();}
bool Builder::exhausted() const{return exhausted_ || coefficients_.exhausted() || intervals_.exhausted();}
bool Builder::charge(std::size_t n) {
  if(exhausted() || work()>limits_.max_work || n>limits_.max_work-work()) {exhausted_=true;return false;}
  local_work_+=n;return true;
}
bool Builder::constant(const Coefficient& c) {
  if(c.terms().size()>1 || (!zero(c) && !c.terms()[0].monomial.powers.empty()))return false;
  const Coefficient z;return coefficients_.apply(p::Operation::Add,std::array{&c,&z}).has_value() && charge();
}
std::optional<Coefficient> Builder::coefficient(std::string_view text) {
  if(!charge())return {};auto c=coefficients_.literal(text);if(!c || !charge(0))return {};return c;
}
std::optional<Coefficient> Builder::apply(p::Operation op,const Coefficient& a,const Coefficient& b) {
  if(!charge())return {};auto c=coefficients_.apply(op,std::array{&a,&b});
  if(!c || !charge(0))return {};return c;
}
std::optional<MeanId> Builder::mean(const MeanRequest& request) {
  if(!request.finite || !zero(request.constant) || !charge(request.weighted_samples.size()))return {};
  auto intervals=intervals_.canonicalIntervals(request.weighted_samples);if(!intervals || !charge(0))return {};
  Mean key{request.group,request.context,0,{}};key.support.reserve(intervals->size());
  for(const auto& x:*intervals) {
    if(x.end<=x.begin || x.end-x.begin>UINT64_MAX-key.count || !charge())return {};
    key.count+=x.end-x.begin;key.support.emplace_back(x.begin,x.end);
  }
  if(!key.count)return {};
  auto one=coefficient("1");if(!one)return {};
  auto weight=coefficients_.divideByLiteral(*one,std::to_string(key.count));if(!weight || !charge(0))return {};
  for(const auto& x:*intervals)if(!charge(coefficientWords(x.weight)+coefficientWords(*weight)) || x.weight!=*weight)return {};
  // CanonicalIntervals already joins adjoining intervals with equal weights.
  // Exact keys, not a hash or sample-count-only match, establish identity.
  if(!charge(6*(1+std::bit_width(means_.size()+1))*(key.support.size()+1)))return {};
  MeanKey lookup{key.group,key.context,key.count,key.support};
  if(auto found=mean_index_.find(lookup);found!=mean_index_.end())return found->second;
  // Two retained support arrays (record + index key), not hidden uncapped copies.
  if(means_.size()>=limits_.max_means || key.support.size()>(limits_.max_support_entries-support_entries_)/2)return {};
  support_entries_+=2*key.support.size();const auto id=means_.size();
  means_.push_back(std::move(key));mean_index_.emplace(std::move(lookup),id);return id;
}
bool Builder::identity(std::uint64_t a,std::uint64_t b,const Coefficient& k) {
  if(!a || !b || b>UINT64_MAX-a || !constant(k))return false;
  const auto counts=std::pair{a,b};auto na=coefficient(std::to_string(a)),nb=coefficient(std::to_string(b));
  if(!na || !nb)return false;auto product=apply(p::Operation::Multiply,*na,*nb);
  if(!product)return false;auto expected=coefficients_.divideByLiteral(*product,std::to_string(a+b));
  if(!expected || !charge(coefficientWords(*expected)+coefficientWords(k)) || *expected!=k)return false;
  if(identities_.contains(counts))return identities_.at(counts);
  // At most two formal mean variables and degree two: never one variable per
  // source element, a whole-expression polynomial, or a distributed graph.
  auto x=coefficients_.atom(1),y=coefficients_.atom(2);if(!x || !y)return false;
  auto d=apply(p::Operation::Subtract,*x,*y);if(!d)return false;
  auto d2=apply(p::Operation::Multiply,*d,*d);if(!d2)return false;
  auto lhs=apply(p::Operation::Multiply,*expected,*d2);
  auto xx=apply(p::Operation::Multiply,*x,*x),yy=apply(p::Operation::Multiply,*y,*y);
  auto ax=apply(p::Operation::Multiply,*na,*x),by=apply(p::Operation::Multiply,*nb,*y);
  if(!lhs || !xx || !yy || !ax || !by)return false;
  auto numerator=apply(p::Operation::Add,*ax,*by);if(!numerator)return false;
  auto mu=coefficients_.divideByLiteral(*numerator,std::to_string(a+b));if(!mu || !charge(0))return false;
  auto mu2=apply(p::Operation::Multiply,*mu,*mu);
  auto aa=apply(p::Operation::Multiply,*na,*xx),bb=apply(p::Operation::Multiply,*nb,*yy);
  auto n=coefficient(std::to_string(a+b));if(!mu2 || !aa || !bb || !n)return false;
  auto c=apply(p::Operation::Multiply,*n,*mu2),s=apply(p::Operation::Add,*aa,*bb);
  if(!c || !s)return false;auto rhs=apply(p::Operation::Subtract,*s,*c);
  if(!rhs || !charge(0))return false;const bool valid=*lhs==*rhs;identities_.emplace(counts,valid);return valid;
}
std::optional<Builder::Correction> Builder::correction(MeanId a,MeanId b,const Coefficient& k) {
  if(a>=means_.size() || b>=means_.size() || !charge())return {};
  // Copy only these bounded support keys before mean() can reallocate means_.
  if(!charge(means_[a].support.size()+means_[b].support.size()))return {};
  const auto x=means_[a],y=means_[b];
  if(x.group!=y.group || x.context!=y.context || !identity(x.count,y.count,k) ||
      x.support.size()+y.support.size()>limits_.max_intervals || !charge(x.support.size()+y.support.size()))return {};
  std::vector<std::pair<std::uint64_t,std::uint64_t>> support=x.support;
  support.insert(support.end(),y.support.begin(),y.support.end());
  if(!charge(support.size()*(1+std::bit_width(support.size()))))return {};
  std::sort(support.begin(),support.end());
  for(std::size_t i=1;i<support.size();++i)if(support[i].first<support[i-1].second)return {};
  auto one=coefficient("1");if(!one)return {};
  auto weight=coefficients_.divideByLiteral(*one,std::to_string(x.count+y.count));if(!weight || !charge(0))return {};
  std::vector<Interval> pieces;pieces.reserve(support.size());
  for(auto [begin,end]:support)pieces.push_back({begin,end,*weight});
  auto merged=mean({x.group,x.context,{},pieces,true});if(!merged)return {};
  auto na=coefficient(std::to_string(x.count)),nb=coefficient(std::to_string(y.count));
  auto negative=coefficient("-"+std::to_string(x.count+y.count));if(!na || !nb || !negative)return {};
  Correction result;result.owner=owner_;result.means={a,b,*merged};result.weights={*na,*nb,*negative};return result;
}
bool Builder::addSquare(Form& form,MeanId id,const Coefficient& weight) {
  if(id>=means_.size() || !constant(weight) || !charge())return false;const auto& m=means_[id];
  if(form.owner && form.owner!=owner_)return false;form.owner=owner_;
  const auto scope=std::pair{m.group,m.context};if(form.scope && *form.scope!=scope)return false;form.scope=scope;
  if(zero(weight))return true;
  if(m.count==1) {
    if(form.second_moment.size()>=limits_.max_intervals)return false;
    form.second_moment.push_back({m.support.at(0).first,m.support.at(0).second,weight});return true;
  }
  auto found=form.mean_squares.find(id);Coefficient old;if(found!=form.mean_squares.end())old=found->second;
  auto next=apply(p::Operation::Add,old,weight);if(!next)return false;
  if(zero(*next)){form.mean_squares.erase(id);return true;}
  if(found==form.mean_squares.end() && form.mean_squares.size()>=limits_.max_square_terms)return false;
  form.mean_squares[id]=std::move(*next);return true;
}
bool Builder::add(Form& form,const Correction& correction,const Coefficient& scale) {
  const auto fail=[&]{form.valid=false;return false;};
  if(!form.valid || correction.owner!=owner_ || !constant(scale))return fail();
  for(std::size_t i=0;i<3;++i) {
    auto w=apply(p::Operation::Multiply,correction.weights[i],scale);
    if(!w || !addSquare(form,correction.means[i],*w))return fail();
  }
  return charge(0) || fail();
}
bool Builder::addCentered(Form& form,MeanId id,std::span<const Interval> pieces,bool finite) {
  const auto fail=[&]{form.valid=false;return false;};
  if(!form.valid || (form.owner && form.owner!=owner_))return fail();
  auto signature=centered(id,pieces,finite);if(!signature)return fail();
  const auto scope=std::pair{signature->group,signature->context};
  if(form.scope && *form.scope!=scope)return fail();form.scope=scope;form.owner=owner_;
  if(signature->second_moment.size()>limits_.max_intervals-form.second_moment.size() ||
      !charge(signature->second_moment.size()+signature->mean_squares.size()))return fail();
  form.second_moment.insert(form.second_moment.end(),signature->second_moment.begin(),signature->second_moment.end());
  for(const auto& term:signature->mean_squares)if(!addSquare(form,term.mean,term.weight))return fail();
  return charge(0) || fail();
}
std::optional<Signature> Builder::finish(const Form& form) {
  if(!form.valid || form.owner!=owner_ || !form.scope || !charge(form.mean_squares.size()))return {};
  auto second=intervals_.canonicalIntervals(form.second_moment);if(!second || !charge(0))return {};
  Signature out;out.owner=owner_;out.group=form.scope->first;out.context=form.scope->second;
  out.second_moment=std::move(*second);
  for(const auto& [mean,weight]:form.mean_squares)out.mean_squares.push_back({mean,weight});return out;
}
std::optional<Signature> Builder::centered(MeanId id,std::span<const Interval> pieces,bool finite) {
  if(!finite || id>=means_.size() || !charge(pieces.size()))return {};
  auto second=intervals_.canonicalIntervals(pieces);if(!second || second->empty() || !charge(0))return {};
  const auto& mean=means_[id];if(second->size()!=mean.support.size())return {};
  const auto& scale=second->at(0).weight;if(!constant(scale) || zero(scale))return {};
  for(std::size_t i=0;i<second->size();++i) {
    if(!charge(coefficientWords((*second)[i].weight)+coefficientWords(scale)))return {};const auto& x=(*second)[i];
    if(x.begin!=mean.support[i].first || x.end!=mean.support[i].second || x.weight!=scale)return {};
  }
  auto n=coefficient("-"+std::to_string(mean.count));if(!n)return {};
  auto weight=apply(p::Operation::Multiply,*n,scale);if(!weight)return {};
  Form form;form.owner=owner_;form.scope=std::pair{mean.group,mean.context};form.second_moment=std::move(*second);
  if(!addSquare(form,id,*weight))return {};
  return finish(form);
}
std::optional<std::size_t> Builder::familyTemplate(const Mean& mean) {
  if(mean.support.empty() || !charge(2*mean.support.size()))return {};
  const auto origin=mean.support.front().first;
  std::vector<std::pair<std::uint64_t,std::uint64_t>> normalized;normalized.reserve(mean.support.size());
  for(auto [begin,end]:mean.support) {
    if(begin<origin || end<=begin)return {};
    normalized.emplace_back(begin-origin,end-origin);
  }
  if(!charge(6*(1+std::bit_width(family_templates_.size()+1))*(normalized.size()+1)))return {};
  MeanKey key{mean.group,mean.context,mean.count,std::move(normalized)};
  if(auto found=family_templates_.find(key);found!=family_templates_.end())return found->second;
  if(family_templates_.size()>=limits_.max_square_terms ||
      std::get<3>(key).size()>limits_.max_support_entries-support_entries_)return {};
  support_entries_+=std::get<3>(key).size();const auto id=family_templates_.size();
  family_templates_.emplace(std::move(key),id);return id;
}
bool Builder::appendFamilyTerm(FamilyForm& form,std::size_t key,FamilyTerm term) {
  if(!form.valid || !constant(term.weight) || !charge())return false;
  if(!term.count || zero(term.weight))return true;
  if(!term.step && term.count>1) {
    auto n=coefficient(std::to_string(term.count));if(!n)return false;
    auto weight=apply(p::Operation::Multiply,term.weight,*n);if(!weight)return false;
    term.weight=std::move(*weight);term.count=1;
  }
  if(term.count==1)term.step=1;
  if(!term.step || term.begin==UINT64_MAX ||
      term.count-1>(UINT64_MAX-1-term.begin)/term.step ||
      form.term_count>=limits_.max_intervals ||
      !charge(coefficientWords(term.weight)+2*(1+std::bit_width(form.terms.size()+1))))return false;
  form.terms[key].push_back(std::move(term));++form.term_count;return true;
}
bool Builder::appendFamily(FamilyForm& form,MeanId id,Translation axis,const Coefficient& weight) {
  if(id>=means_.size() || !constant(weight) || !charge())return false;
  const auto& mean=means_[id];
  if(mean.support.empty() || (form.owner && form.owner!=owner_))return false;
  form.owner=owner_;const auto scope=std::pair{mean.group,mean.context};
  if(form.scope && *form.scope!=scope)return false;form.scope=scope;
  if(!axis.count || zero(weight))return true;
  // The complete last translated window must fit, not just its origin.
  const auto end=mean.support.back().second;
  if(axis.begin>UINT64_MAX-end || (axis.step &&
      axis.count-1>(UINT64_MAX-end-axis.begin)/axis.step))return false;
  auto key=familyTemplate(mean);if(!key)return false;
  return appendFamilyTerm(form,*key,{axis.begin+mean.support.front().first,axis.step,axis.count,weight});
}
bool Builder::addFamily(FamilyForm& form,const Correction& correction,Translation axis,
                        const Coefficient& scale,bool finite) {
  const auto fail=[&]{form.valid=false;return false;};
  if(!form.valid || correction.owner!=owner_ || !finite || !constant(scale))return fail();
  for(std::size_t i=0;i<3;++i) {
    auto weight=apply(p::Operation::Multiply,correction.weights[i],scale);
    if(!weight || !appendFamily(form,correction.means[i],axis,*weight))return fail();
  }
  return charge(0) || fail();
}
bool Builder::addFixed(FamilyForm& form,const Correction& correction,const Coefficient& scale) {
  return addFamily(form,correction,{0,1,1},scale,true);
}
bool Builder::addCentered(FamilyForm& form,MeanId id,std::span<const Interval> pieces,bool finite) {
  const auto fail=[&]{form.valid=false;return false;};
  if(!form.valid || id>=means_.size() || (form.owner && form.owner!=owner_))return fail();
  auto signature=centered(id,pieces,finite);if(!signature)return fail();
  const auto scope=std::pair{signature->group,signature->context};
  if(form.scope && *form.scope!=scope)return fail();form.scope=scope;form.owner=owner_;
  if(!signature->second_moment.empty()) {
    // A singleton normalized mean is precisely the same finite point square.
    Mean point{signature->group,signature->context,1,{{0,1}}};
    auto key=familyTemplate(point);if(!key)return fail();
    for(const auto& interval:signature->second_moment) {
      if(interval.end<=interval.begin || !appendFamilyTerm(form,*key,
          {interval.begin,1,interval.end-interval.begin,interval.weight}))return fail();
    }
  }
  for(const auto& term:signature->mean_squares)
    if(!appendFamily(form,term.mean,{0,1,1},term.weight))return fail();
  return charge(0) || fail();
}
std::optional<Builder::FamilySignature> Builder::finishFamily(const FamilyForm& form) {
  if(!form.valid || form.owner!=owner_ || !form.scope || form.term_count>limits_.max_intervals ||
      !charge(form.terms.size()+form.term_count))return {};
  for(const auto& [key,terms]:form.terms) {
    (void)key;for(const auto& term:terms)if(!charge(coefficientWords(term.weight)))return {};
  }
  FamilySignature out;out.owner=owner_;out.scope=*form.scope;out.term_count=form.term_count;
  out.terms=form.terms;return out;
}
std::optional<bool> Builder::equalFamilies(const FamilySignature& a,const FamilySignature& b) {
  if(a.owner!=owner_ || b.owner!=owner_ || a.scope!=b.scope)return false;
  std::size_t fragments=0;
  const auto compare=[&](std::span<const FamilyTerm> left,std::span<const FamilyTerm> right)->std::optional<bool> {
    std::uint64_t modulus=1;
    for(auto terms:{left,right})for(const auto& term:terms) {
      if(!charge())return {};
      if(term.count<=1)continue;
      if(!term.step || !charge(2*(1+std::bit_width(term.step)+std::bit_width(modulus))))return {};
      const auto reduced=modulus/std::gcd(modulus,term.step);
      if(term.step>UINT64_MAX/reduced)return {};modulus=reduced*term.step;
    }
    std::map<std::uint64_t,std::vector<Interval>> residues;
    const auto append=[&](const FamilyTerm& term,bool subtract)->bool {
      if(!term.count || zero(term.weight))return true;
      if(!charge(coefficientWords(term.weight)))return false;
      Coefficient weight=term.weight;
      if(subtract) {Coefficient z;auto negative=apply(p::Operation::Subtract,z,weight);if(!negative)return false;weight=std::move(*negative);}
      const auto fanout=term.count==1?std::uint64_t{1}:modulus/term.step;
      if(!fanout || fanout>limits_.max_intervals || fanout>limits_.max_intervals-fragments)return false;
      // This loop is over capped lattice residue metadata, NEVER over count.
      for(std::uint64_t residue_index=0;residue_index<fanout;++residue_index) {
        if(!charge())return false;if(residue_index>=term.count)continue;
        const auto first=term.begin+residue_index*term.step;
        const auto count=1+(term.count-1-residue_index)/fanout;
        const auto begin=first/modulus;
        if(count>UINT64_MAX-begin ||
            !charge(coefficientWords(weight)+2*(1+std::bit_width(residues.size()+1))))return false;
        residues[first%modulus].push_back({begin,begin+count,weight});++fragments;
      }
      return true;
    };
    for(const auto& term:left)if(!append(term,false))return {};
    for(const auto& term:right)if(!append(term,true))return {};
    for(const auto& [residue,intervals]:residues) {
      (void)residue;auto canonical=intervals_.canonicalIntervals(intervals);
      if(!canonical || !charge(0))return {};
      if(!canonical->empty())return false;
    }
    return true;
  };
  auto x=a.terms.begin(),y=b.terms.begin();
  while(x!=a.terms.end() || y!=b.terms.end()) {
    if(!charge())return {};
    std::span<const FamilyTerm> left,right;
    if(y==b.terms.end() || (x!=a.terms.end() && x->first<y->first)){left=x->second;++x;}
    else if(x==a.terms.end() || y->first<x->first){right=y->second;++y;}
    else {left=x->second;right=y->second;++x;++y;}
    auto result=compare(left,right);if(!result || !*result)return result;
  }
  return true;
}
} // namespace sentryward::equivalence::moment_certificate
