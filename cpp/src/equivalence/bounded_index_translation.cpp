#include "sentryward/equivalence/bounded_index_translation.h"
#include "sentryward/equivalence/integer_interval.h"

#include <algorithm>
#include <bit>
#include <compare>
#include <limits>
#include <map>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sentryward::equivalence::bounded_index_translation {
namespace {
namespace i = integer_interval;
constexpr std::uint64_t maximum = std::numeric_limits<std::int64_t>::max();
using Coefficients = std::map<std::uint32_t,std::uint64_t>;
struct Affine {
  std::uint64_t constant = 0;
  Coefficients indices;
  auto operator<=>(const Affine&) const = default;
};
struct Atom {
  bool remainder = false;
  std::uint64_t divisor = 1;
  Affine residual;
  auto operator<=>(const Atom&) const = default;
};
struct Form {
  Affine linear;
  std::map<std::size_t,std::uint64_t> atoms;
  bool operator==(const Form&) const = default;
  std::size_t terms() const { return linear.indices.size()+atoms.size(); }
};
struct Summary { i::Interval range; Form form; bool projected=false,literal=false; };
struct CacheKey {
  NodeId node=0;
  bool right=false;
  bool operator==(const CacheKey&) const = default;
};
struct CacheHash {
  std::size_t operator()(CacheKey key) const { return (key.node*2)^key.right; }
};
bool add(std::uint64_t& a,std::uint64_t b) {
  if(b>maximum-a)return false;a+=b;return true;
}
bool multiply(std::uint64_t& a,std::uint64_t b) {
  if(a && b>maximum/a)return false;a*=b;return true;
}
} // namespace

struct Session::Impl {
  struct Entry { unsigned char state=0; std::optional<Summary> summary; };
  Resolve resolve;
  Charge external_charge;
  Limits limits;
  Statistics stats;
  std::map<std::uint32_t,i::Interval> outer;
  std::uint32_t binder;
  std::uint64_t shift,count;
  bool valid=true;
  std::unordered_map<CacheKey,Entry,CacheHash> cache;
  std::map<Atom,std::size_t> atoms;

  Impl(Resolve r,std::span<const IndexExtent> extents,std::uint32_t b,
       std::uint64_t s,std::uint64_t c,Limits l,Charge charge)
      :resolve(std::move(r)),external_charge(std::move(charge)),limits(l),binder(b),shift(s),count(c) {
    if(!count || shift>maximum || count-1>maximum-shift ||
        extents.size()>=limits.max_indices || !limits.max_nodes || !limits.max_terms ||
        !limits.max_cached_terms || !limits.max_atoms || !limits.max_work ||
        limits.max_nodes>4096 || limits.max_terms>64 || limits.max_cached_terms>32768 ||
        limits.max_atoms>512 || limits.max_indices>64 || limits.max_work>1048576)valid=false;
    if(!valid)return;
    for(const auto& extent:extents) {
      if(!work(1))break;
      auto range=i::externalIndex(i::Type{false,64},extent.count);
      if(!range || extent.level==binder || !outer.try_emplace(extent.level,*range).second)valid=false;
    }
  }
  bool work(std::size_t amount) {
    if(stats.exhausted)return false;
    if(amount>limits.max_work-stats.work || (external_charge && !external_charge(amount))) {
      stats.exhausted=true;return false;
    }
    stats.work+=amount;return true;
  }
  bool fits(const Form& form) {
    return form.terms()<=limits.max_terms && form.linear.indices.size()<=limits.max_indices;
  }
  std::optional<Form> combine(const Form& a,const Form& b) {
    if(!work((a.terms()+b.terms()+1)*2*(std::bit_width(limits.max_terms)+1)))return {};
    Form result=a;
    if(!add(result.linear.constant,b.linear.constant))return {};
    for(const auto& [level,value]:b.linear.indices)if(!add(result.linear.indices[level],value))return {};
    for(const auto& [atom,value]:b.atoms)if(!add(result.atoms[atom],value))return {};
    if(!fits(result))return {};return result;
  }
  std::optional<Form> scale(const Form& input,std::uint64_t coefficient) {
    if(!work(input.terms()+1))return {};
    if(!coefficient)return Form{};
    Form result=input;
    if(!multiply(result.linear.constant,coefficient))return {};
    for(auto& [level,value]:result.linear.indices)if(!multiply(value,coefficient))return {};
    for(auto& [atom,value]:result.atoms)if(!multiply(value,coefficient))return {};
    return result;
  }
  std::optional<Form> euclidean(const Form& numerator,std::uint64_t divisor,bool remainder) {
    if(!divisor || !numerator.atoms.empty() ||
        !work((numerator.terms()*2+1)*2*(std::bit_width(limits.max_indices)+1)))return {};
    Form result;
    Atom atom;atom.divisor=divisor;atom.remainder=remainder;
    atom.residual.constant=numerator.linear.constant%divisor;
    if(!remainder)result.linear.constant=numerator.linear.constant/divisor;
    for(const auto& [level,value]:numerator.linear.indices) {
      if(value%divisor)atom.residual.indices.emplace(level,value%divisor);
      if(!remainder && value/divisor)result.linear.indices.emplace(level,value/divisor);
    }
    if(atom.residual.indices.empty()) {
      if(remainder)result.linear.constant=atom.residual.constant;
      return result;
    }
    // Atom identity uses complete residual coefficients, divisor, and operation.
    // std::map equality/comparison costs at most max_indices; charge that bound
    // for every tree comparison, not merely the one outward API invocation.
    const auto height=std::bit_width(atoms.size()+1);
    if(!work((limits.max_indices+1)*2*(height+1)))return {};
    auto found=atoms.find(atom);
    if(found==atoms.end()) {
      if(atoms.size()>=limits.max_atoms || atom.residual.indices.size()>
          limits.max_cached_terms-stats.cached_terms) {stats.exhausted=true;return {};}
      stats.cached_terms+=atom.residual.indices.size();
      found=atoms.emplace(std::move(atom),atoms.size()).first;
      stats.atoms=atoms.size();
    }
    result.atoms.emplace(found->second,1);
    if(!fits(result))return {};return result;
  }
  std::optional<Summary> make(const Node& node,bool right) {
    const bool projected=node.type=="exact:u64";
    if(!node.admitted || (node.type!="u64" && !projected) || node.child_count>2 ||
        node.literal.size()>21)return {};
    const i::Type type{false,64};
    if(node.kind==Kind::Literal) {
      if(node.child_count)return {};
      auto value=i::literal(type,node.literal);if(!value)return {};
      return Summary{*value,Form{Affine{static_cast<std::uint64_t>(value->lower),{}},{}},projected,true};
    }
    if(node.kind==Kind::Index) {
      if(node.child_count)return {};
      i::Result range;
      Form form;
      if(node.index_level==binder) {
        const auto begin=right?0:shift;
        range=i::interval(type,static_cast<std::int64_t>(begin),static_cast<std::int64_t>(begin+count-1));
        form.linear.constant=begin;
      } else if(auto bound=outer.find(node.index_level);bound!=outer.end())range=bound->second;
      if(!range)return {};
      form.linear.indices.emplace(node.index_level,1);
      return Summary{*range,std::move(form),projected};
    }
    if(node.child_count!=2)return {};
    const auto& a=cache.at({node.children[0],right}).summary;
    const auto& b=cache.at({node.children[1],right}).summary;
    if(!a || !b || a->projected!=projected || b->projected!=projected)return {};
    i::Result range;std::optional<Form> form;
    if(node.kind==Kind::Add) {
      range=i::apply(i::Operation::Add,a->range,b->range);
      if(range)form=combine(a->form,b->form);
    } else if(node.kind==Kind::Multiply) {
      range=i::apply(i::Operation::Multiply,a->range,b->range);
      if(range) {
        if(!a->form.terms())form=scale(b->form,a->form.linear.constant);
        else if(!b->form.terms())form=scale(a->form,b->form.linear.constant);
      }
    } else if(node.kind==Kind::Divide || node.kind==Kind::Remainder) {
      // Require a genuine literal node, not an inferred singleton of a computed
      // expression. Its cached summary retains that distinction without a
      // second resolver call or header reconstruction.
      if(!b->literal || b->range.lower<=0 || b->range.lower!=b->range.upper)return {};
      range=node.kind==Kind::Divide?i::divideByPositiveLiteral(a->range,b->range.lower):
          i::remainderByPositiveLiteral(a->range,b->range.lower);
      if(range)form=euclidean(a->form,static_cast<std::uint64_t>(b->range.lower),node.kind==Kind::Remainder);
    }
    if(!range || range->lower<0 || !form)return {};
    return Summary{*range,std::move(*form),projected};
  }
  const std::optional<Summary>* summarize(NodeId root,bool right) {
    struct Task {CacheKey key;bool ready=false;Node node;};
    const CacheKey root_key{root,right};
    std::vector<Task> pending{{root_key,false,{}}};
    while(!pending.empty()) {
      if(!work(1))return nullptr;
      auto task=std::move(pending.back());pending.pop_back();
      if(task.ready) {
        auto value=make(task.node,right);
        auto& entry=cache.at(task.key);
        if(value) {
          const auto terms=value->form.terms();
          if(terms>limits.max_cached_terms-stats.cached_terms){stats.exhausted=true;return nullptr;}
          stats.cached_terms+=terms;
        }
        entry.summary=std::move(value);entry.state=2;continue;
      }
      if(auto found=cache.find(task.key);found!=cache.end()) {
        if(found->second.state==1)return nullptr; // Active DAG cycle is Unknown.
        continue;
      }
      if(cache.size()>=limits.max_nodes){stats.exhausted=true;return nullptr;}
      auto node=resolve(task.key.node);
      cache.emplace(task.key,Entry{1,{}});stats.nodes=cache.size();
      if(!node || !node->admitted || node->child_count>2 ||
          (node->type!="u64" && node->type!="exact:u64") || node->literal.size()>21) {
        cache.at(task.key).state=2;continue;
      }
      if(!work(node->child_count+node->type.size()+node->literal.size()))return nullptr;
      pending.push_back({task.key,true,*node});
      for(std::size_t c=0;c<node->child_count;++c)pending.push_back({{node->children[c],right},false,{}});
    }
    return &cache.at(root_key).summary;
  }
  bool equivalent(NodeId left,NodeId right) {
    if(!valid || stats.exhausted)return false;
    const auto* a=summarize(left,false);
    const auto* b=summarize(right,true);
    if(!a || !b || !*a || !*b || (*a)->projected!=(*b)->projected ||
        !work((*a)->form.terms()+(*b)->form.terms()+1))return false;
    return (*a)->form==(*b)->form;
  }
};

Session::Session(Resolve resolve,std::span<const IndexExtent> extents,std::uint32_t binder,
                 std::uint64_t shift,std::uint64_t count,Limits limits,Charge charge)
    :impl_(std::make_unique<Impl>(std::move(resolve),extents,binder,shift,count,limits,std::move(charge))) {}
Session::~Session()=default;
Session::Session(Session&&) noexcept=default;
Session& Session::operator=(Session&&) noexcept=default;
bool Session::equivalent(NodeId a,NodeId b) {return impl_->equivalent(a,b);}
Statistics Session::statistics() const {return impl_->stats;}

} // namespace sentryward::equivalence::bounded_index_translation
