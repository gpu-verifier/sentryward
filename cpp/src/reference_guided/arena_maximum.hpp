#pragma once
#include "core.hpp"

namespace rg {

// first_argmax(v_0..v_{n-1}; d_0 < ... < d_{n-1}) is d_j for the least j with v_j = max_k v_k: the
// index a lexicographic (value descending, index ascending) selection returns. Values are finite.
inline auto Arena::first_argmax(std::vector<std::pair<Q,Id>> members) -> Id {
  std::sort(members.begin(),members.end());
  std::vector<Id> values;std::vector<Q> indices;
  for(const auto &[index,value]:members){indices.push_back(index);values.push_back(value);}
  const Q low=indices.front(),high=indices.back();
  return intern("first_argmax",values,indices,"",{true,low>0,low>=0},std::pair(low,high));
}

// The leaves of a nested max node. O(nodes of the max shell).
inline auto Arena::max_leaves(Id value) const -> std::set<Id> {
  std::set<Id> leaves;std::vector<Id> todo{value};
  while(!todo.empty()) {
    const Id x=todo.back();todo.pop_back();
    if(nodes[x].op=="max" || nodes[x].op=="masked_max")todo.insert(todo.end(),nodes[x].args.begin(),nodes[x].args.end());
    else leaves.insert(x);
  }
  return leaves;
}

// A selection arm as tournament members (index, value): a constant index d whose value is the
// compared operand, a first_argmax node or c + first_argmax whose values' max is the compared
// operand, or (nested) a two-constant leaf selection. Empty when the arm is none of these.
inline auto Arena::tournament_members(Id arm,Id compared) -> std::vector<std::pair<Q,Id>> {
  if(!nodes[compared].facts.finite)return {};
  if(auto d=value(arm))return d->denominator()==1 ? std::vector<std::pair<Q,Id>>{{*d,compared}} : std::vector<std::pair<Q,Id>>{};
  Q shift=0;Id core=arm;
  if(nodes[arm].op=="add" && nodes[arm].args.size()==2) {
    for(size_t j=0;j<2;++j)
      if(nodes[arm].args[j]==one && nodes[arm].data[1-j]==1 && nodes[nodes[arm].args[1-j]].op=="first_argmax") {
        shift=nodes[arm].data[j];core=nodes[arm].args[1-j];
      }
    if(core==arm)return {};
  }
  if(nodes[core].op=="select" && shift==0)
    if(auto leaf=tournament_select(nodes[core].args[0],nodes[core].args[1],nodes[core].args[2],true))core=*leaf;
  if(nodes[core].op!="first_argmax" || shift.denominator()!=1)return {};
  const auto &values=nodes[core].args;const auto &indices=nodes[core].data;
  if(max_leaves(compared)!=std::set<Id>(values.begin(),values.end()))return {};
  std::vector<std::pair<Q,Id>> members;
  for(size_t j=0;j<values.size();++j)members.emplace_back(indices[j]+shift,values[j]);
  return members;
}

// A predicate whose only non-constant atom is one first_argmax node F (integer, bounds [lo, hi]) is a
// function on a finite domain: evaluate it at every index of F. Returns the point set or nothing.
// Integer arithmetic, words, comparisons and selections only; O(index count x predicate nodes).
inline auto Arena::argmax_point_set(Id predicate) -> std::optional<std::pair<Id,std::vector<Q>>> {
  static constexpr size_t predicate_node_limit=64;
  Id atom=invalid;std::vector<Id> order,todo{predicate};std::set<Id> seen;
  while(!todo.empty()) {
    const Id x=todo.back();todo.pop_back();
    if(!seen.insert(x).second)continue;
    if(seen.size()>predicate_node_limit)return {};
    const auto &n=nodes[x];
    if(n.op=="const")continue;
    if(n.op=="first_argmax"){if(atom!=invalid && atom!=x)return {};atom=x;continue;}
    if(n.op!="add" && n.op!="mul" && n.op!="idiv" && n.op!="imod" && n.op!="word" && n.op!="compare" && n.op!="select")return {};
    order.push_back(x);todo.insert(todo.end(),n.args.begin(),n.args.end());
  }
  if(atom==invalid)return {};
  std::vector<Q> points;
  for(const Q &index:nodes[atom].data) {
    std::map<Id,Q> v{{atom,index}};
    std::function<std::optional<Q>(Id)> eval=[&](Id x)->std::optional<Q> {
      if(auto c=value(x))return c;
      if(auto it=v.find(x);it!=v.end())return it->second;
      const auto &n=nodes[x];std::vector<Q> xs;
      for(Id arg:n.args){auto r=eval(arg);if(!r)return {};xs.push_back(*r);}
      Q r;
      if(n.op=="add"){r=0;for(size_t j=0;j<xs.size();++j)r+=n.data[j]*xs[j];}
      else if(n.op=="mul"){r=n.data[0];for(const Q &q:xs)r*=q;}
      else if(n.op=="idiv" || n.op=="imod") {
        if(xs[0].denominator()!=1)return {};
        const Z d=n.data[0].numerator();
        r=n.op=="idiv"?Q(floorz(xs[0].numerator(),d)):Q(modz(xs[0].numerator(),d));
      } else if(n.op=="word") {
        if(xs[0].denominator()!=1)return {};
        r=Q(modz(xs[0].numerator(),pow2(n.data[0].numerator().convert_to<unsigned>())));
      } else if(n.op=="compare") {
        const auto &t=n.tag;const Q &l=xs[0],&h=xs[1];
        r=Q((t=="lt"&&l<h)||(t=="le"&&l<=h)||(t=="eq"&&l==h)||(t=="ne"&&l!=h)||(t=="gt"&&l>h)||(t=="ge"&&l>=h)?1:0);
      } else r=xs[0]!=0?xs[1]:xs[2];
      v.emplace(x,r);return r;
    };
    const auto truth=eval(predicate);
    if(!truth || (*truth!=0 && *truth!=1))return {};
    if(*truth==1)points.push_back(index);
  }
  return std::pair(atom,points);
}

// select(p, select(q, X, Y), Y) is select(p and q, X, Y). Taken only when p and q are predicates of one
// first_argmax node F whose conjunction holds at exactly one index j: then it is select(F == j, X, Y).
inline auto Arena::argmax_point_select(Id p,Id a,Id b) -> std::optional<Id> {
  if(nodes[a].op!="select" || nodes[a].args.size()!=3 || nodes[a].args[2]!=b)return {};
  const Id q=nodes[a].args[0],x=nodes[a].args[1];
  const auto left=argmax_point_set(p),right=argmax_point_set(q);
  if(!left || !right || left->first!=right->first)return {};
  std::vector<Q> both;
  std::set_intersection(left->second.begin(),left->second.end(),right->second.begin(),right->second.end(),std::back_inserter(both));
  if(both.size()!=1)return {};
  ++stats["argmax_point_selections"];
  return select(compare("eq",left->first,constant(both.front())),x,b);
}

// The compared values (X of the else arm, Y of the then arm) of a selection that takes the then arm when
// Y > X and the else arm when X > Y: p = (X <= Y), (X < Y), or (X < Y) or Q with Q zero unless X == Y
// (Q = (X == Y) or a 0/1 product containing it). Ties are not constrained.
inline auto Arena::max_selection_operands(Id p) const -> std::optional<std::pair<Id,Id>> {
  if(nodes[p].op=="select" && nodes[p].args.size()==3 && nodes[p].args[1]==one && nodes[p].args[2]==zero)p=nodes[p].args[0];
  Id strict=p,tie=invalid;
  if(nodes[p].op=="select" && nodes[p].args.size()==3 && nodes[p].args[1]==one){strict=nodes[p].args[0];tie=nodes[p].args[2];}
  const auto &c=nodes[strict];
  if(c.op!="compare" || c.args.size()!=2 || (c.tag!="lt" && c.tag!="le") || (c.tag=="le" && tie!=invalid))return {};
  const Id x=c.args[0],y=c.args[1];
  auto equality=[&](Id f){const auto &e=nodes[f];return e.op=="compare" && e.tag=="eq" && e.args.size()==2 &&
      ((e.args[0]==x && e.args[1]==y) || (e.args[0]==y && e.args[1]==x));};
  if(tie!=invalid) {
    const auto &t=nodes[tie];
    const bool product=t.op=="mul" && t.data.size()==1 && t.data[0]==1 &&
        std::any_of(t.args.begin(),t.args.end(),equality) &&
        std::all_of(t.args.begin(),t.args.end(),[&](Id f){return nodes[f].bounds && nodes[f].bounds->first>=0 &&
                                                                   nodes[f].bounds->second<=1;});
    if(!product && !equality(tie))return {};
  }
  return std::pair(x,y);
}

// max(x, y) of a tournament. When every leaf is masked (select(G == d, c, u), one G, distinct d) the maximum
// is kept as masked_max(leaves in d order; d): a canonical spelling of the same maximum that does not depend on
// the order in which the selections combined it (an explicit max nests in tournament order). O(leaves).
inline auto Arena::tournament_maximum(Id x,Id y) -> Id {
  std::set<Id> leaves=max_leaves(x);
  for(Id leaf:max_leaves(y))leaves.insert(leaf);
  if(auto masked=masked_maximum(leaves))return *masked;
  return maximum({x,y});
}

inline auto Arena::masked_maximum(const std::set<Id> &leaves) -> std::optional<Id> {
  if(leaves.size()<2)return {};
  Id argmax=invalid;std::map<Q,Id> ordered;bool finite=true,pos=true,nn=true;
  for(Id leaf:leaves) {
    const auto masked=masked_leaf(leaf);
    if(!masked || (argmax!=invalid && argmax!=masked->argmax) || !ordered.emplace(masked->index,leaf).second)return {};
    argmax=masked->argmax;finite=finite && nodes[leaf].facts.finite;
    pos=pos && nodes[masked->payload].facts.pos;nn=nn && nodes[masked->payload].facts.nn;
  }
  std::vector<Id> args;std::vector<Q> indices;
  for(const auto &[index,leaf]:ordered){indices.push_back(index);args.push_back(leaf);}
  // G equals at most one of the distinct d, so at least one leaf is its payload u: the maximum is at least
  // that u (positive or nonnegative when every payload is).
  return intern("masked_max",args,indices,"",{finite,finite && pos,finite && nn});
}

inline auto Arena::masked_leaf(Id w) -> std::optional<MaskedLeaf> {
  const auto &n=nodes[w];
  if(n.op!="select" || n.args.size()!=3)return {};
  const auto &t=nodes[n.args[0]];
  const auto mask=value(n.args[1]);
  if(!mask || t.op!="compare" || t.tag!="eq" || t.args.size()!=2 || nodes[t.args[0]].op!="first_argmax")return {};
  const auto index=value(t.args[1]);
  if(!index)return {};
  return MaskedLeaf{t.args[0],*index,*mask,n.args[2]};
}

// A payload tournament: selections compare the (masked) values w and carry payloads u, where every w_j is u_j
// or select(G == d_j, c, u_j) (one first_argmax G, one constant c < 0, distinct d_j) and every u_j >= 0.
// At most one member is masked at run time and it is below every other member, so the carried payload of a
// set of two or more members is its maximum w. select(p, PY, PX) of disjoint member sets is max(X, Y) when each
// payload is its set's maximum already (PY == Y) or, for one member, that member's u. Exact; O(members).
inline auto Arena::payload_tournament_select(Id p,Id a,Id b) -> std::optional<Id> {
  const auto operands=max_selection_operands(p);
  if(!operands)return {};
  const auto [x,y]=*operands;
  if(a==y && b==x)return {};
  const auto then_leaves=max_leaves(y),else_leaves=max_leaves(x);
  for(Id leaf:then_leaves)if(else_leaves.contains(leaf))return {};
  auto carried=[&](Id payload,Id compared,const std::set<Id> &leaves) {
    // The same maximum spelled as an explicit max or as masked_max: equal leaf sets.
    if(payload==compared || ((nodes[payload].op=="max" || nodes[payload].op=="masked_max") &&
                              max_leaves(payload)==leaves))return true;
    if(leaves.size()!=1)return false;
    const auto masked=masked_leaf(compared);
    return masked && masked->payload==payload;
  };
  if(!carried(a,y,then_leaves) || !carried(b,x,else_leaves))return {};
  Id argmax=invalid;std::optional<Q> mask;std::set<Q> indices;
  for(const auto *leaves:{&then_leaves,&else_leaves})
    for(Id leaf:*leaves) {
      if(const auto masked=masked_leaf(leaf)) {
        if((argmax!=invalid && argmax!=masked->argmax) || (mask && *mask!=masked->mask) || masked->mask>=0 ||
           !nodes[masked->payload].facts.finite || !nodes[masked->payload].facts.nn ||
           !indices.insert(masked->index).second)return {};
        argmax=masked->argmax;mask=masked->mask;
      } else if(!nodes[leaf].facts.finite || !nodes[leaf].facts.nn)return {};
    }
  if(argmax==invalid)return {};
  ++stats["payload_tournament_maxima"];
  return tournament_maximum(x,y);
}

// select(p, a, b) of a lexicographic tournament. With X the compared value of b and Y that of a:
//  p = (X <= Y): a wins ties (a's indices must all precede b's);
//  p = (X < Y): b wins ties (b's indices must precede a's);
//  p = (X < Y) or (X == Y and Ia - Ib <= -1): explicit tie-break, any disjoint indices;
//  p = (X < Y) or (X == Y): a wins ties (a's indices must all precede b's).
// Arms equal to the compared values give max(X, Y) (p differs from X < Y only where X == Y); index
// arms give the first argmax of both member sets. A two-constant leaf is converted only when nested
// (`leaf`), so a lone two-way selection keeps its form. Exact; O(members).
inline auto Arena::tournament_select(Id p,Id a,Id b,bool leaf) -> std::optional<Id> {
  if(nodes[p].op=="select" && nodes[p].args.size()==3 && nodes[p].args[1]==one && nodes[p].args[2]==zero)p=nodes[p].args[0];
  Id strict=p,tie=invalid;
  if(nodes[p].op=="select" && nodes[p].args.size()==3 && nodes[p].args[1]==one) {strict=nodes[p].args[0];tie=nodes[p].args[2];}
  const auto &c=nodes[strict];
  if(c.op!="compare" || c.args.size()!=2 || (c.tag!="lt" && c.tag!="le") || (c.tag=="le" && tie!=invalid))return {};
  const Id x=c.args[0],y=c.args[1];
  if(!nodes[x].facts.finite || !nodes[y].facts.finite)return {};
  // The tie conjunction: a product of compare(eq, X, Y) and compare(le, Ia - Ib, -1) (or an equivalent
  // form). The predicate is the AND of every factor, so every index-shaped factor must name the same
  // difference; any other one (a runtime flag, say) is refused rather than dropped.
  bool tie_values=false;Id tie_index=invalid;
  if(tie!=invalid) {
    const auto &t=nodes[tie];
    const bool product=t.op=="mul" && t.data.size()==1 && t.data[0]==1;
    if(!product && t.op!="compare")return {};
    for(Id factor:product ? t.args : std::vector<Id>{tie}) {
      const auto &f=nodes[factor];
      Id index=invalid;
      if(f.op=="compare" && f.tag=="eq" && f.args.size()==2 &&
         ((f.args[0]==x && f.args[1]==y) || (f.args[0]==y && f.args[1]==x)))tie_values=true;
      else if(f.op=="compare" && f.args.size()==2 && ((f.tag=="le" && f.args[1]==constant(-1)) ||
              (f.tag=="lt" && f.args[1]==zero)))index=f.args[0];
      else if(f.op=="compare" && f.args.size()==2 && ((f.tag=="ge" && f.args[1]==one) ||
              (f.tag=="gt" && f.args[1]==zero)))index=neg(f.args[0]);
      else return {};
      if(index!=invalid) {
        if(tie_index!=invalid && tie_index!=index)return {};
        tie_index=index;
      }
    }
    if(!tie_values)return {};
  }
  if(a==y && b==x) {
    ++stats["tournament_value_maxima"];
    return tournament_maximum(x,y);
  }
  if(!leaf && value(a) && value(b))return {};
  auto ma=tournament_members(a,y),mb=tournament_members(b,x);
  if(ma.empty() || mb.empty())return {};
  Q a_low=ma.front().first,a_high=ma.front().first,b_low=mb.front().first,b_high=mb.front().first;
  for(const auto &m:ma){a_low=std::min(a_low,m.first);a_high=std::max(a_high,m.first);}
  for(const auto &m:mb){b_low=std::min(b_low,m.first);b_high=std::max(b_high,m.first);}
  if(tie!=invalid && tie_index!=invalid) {
    if(tie_index!=sub(a,b))return {};
    std::set<Q> seen;
    for(const auto &m:ma)seen.insert(m.first);
    for(const auto &m:mb)if(!seen.insert(m.first).second)return {};
  } else if(c.tag=="le" || tie!=invalid ? !(a_high<b_low) : !(b_high<a_low))return {};
  ma.insert(ma.end(),mb.begin(),mb.end());
  ++stats["tournament_first_argmax_merges"];
  return first_argmax(std::move(ma));
}

inline auto Arena::maximum(std::vector<Id> xs) -> Id {
  std::vector<Id> flat;
  for (Id x : xs) {
    if(!neginf(x)) flat.push_back(x);
  }
  std::optional<Q> largest;
  xs.clear();
  for(Id x:flat) {
    if(auto v=value(x)) {if(!largest || *v>*largest) largest=*v;}
    else xs.push_back(x);
  }
  if(auto pulled=common_positive_scale(xs,largest)) return *pulled;
  if(largest) xs.push_back(constant(*largest));
  std::sort(xs.begin(),xs.end());xs.erase(std::unique(xs.begin(),xs.end()),xs.end());
  // One stored-operand scan, not pairwise comparison. A proved lower bound
  // of one operand dominates any finite operand whose upper bound is below
  // it. On equal lower bounds prefer the larger upper bound, so a point
  // interval cannot hide the nonconstant witness that dominates that point.
  // Keep one witness even when all intervals are the same point.
  Id witness=invalid;Q greatest_lower=0;
  for(Id x:xs)if(nodes[x].facts.finite && nodes[x].bounds &&
      (witness==invalid || nodes[x].bounds->first>greatest_lower ||
       (nodes[x].bounds->first==greatest_lower &&
        nodes[x].bounds->second>nodes[witness].bounds->second))) {
    witness=x;greatest_lower=nodes[x].bounds->first;
  }
  if(witness!=invalid) {
    auto end=std::remove_if(xs.begin(),xs.end(),[&](Id x) {
      return x!=witness && nodes[x].facts.finite && nodes[x].bounds &&
             nodes[x].bounds->second<=greatest_lower;
    });
    stats["maximum_dominated_operands"]+=std::distance(end,xs.end());xs.erase(end,xs.end());
  }
  if(largest && std::none_of(xs.begin(),xs.end(),[&](Id x){return nodes[x].op=="const";}))largest.reset();
  if(xs.empty()) return infinity(-1);
  if(xs.size()==1) return xs[0];
  if(xs.size()==2 && !largest) {
    for(unsigned i=0;i<2;++i) {
      const auto &n=nodes[xs[i]];
      if(n.op=="mul" && n.data[0]==-1 && n.args.size()==1 && n.args[0]==xs[1-i] && nodes[xs[1-i]].facts.finite)
        return absolute(xs[1-i]);
    }
  }
  if(xs.size()==2 && largest) {
    Id other=nodes[xs[0]].op=="const" ? xs[1] : xs[0];
    if(*largest==0 && nodes[other].facts.finite && nodes[other].facts.nn) return other;
    const auto &negated=nodes[other];
    if(negated.op=="mul" && negated.data[0]==-1 && negated.args.size()==1) {
      const auto &inner=nodes[negated.args[0]];
      if(inner.op=="max" && inner.args.size()==2) {
        for(unsigned i=0;i<2;++i) if(auto c=value(inner.args[i]);c && *largest<=-*c) {
          Id value=inner.args[1-i];Q hi=-*c,lo=*largest;
          if(nodes[value].facts.finite) return clamp(neg(value),lo,hi);
        }
      }
    }
  }
  Facts f{true,false,false};Bounds bounds;
  bool extended=true,finite_witness=false;
  bool all_bounds=true;Q lo,hi;bool first=true;
  for(Id x:xs) {
    f.finite &= nodes[x].facts.finite;f.pos |= nodes[x].facts.pos;f.nn |= nodes[x].facts.nn;
    extended &= nodes[x].facts.finite_or_negative_infinity();
    finite_witness |= nodes[x].facts.finite;
    if(auto b=nodes[x].bounds) {
      if(first) {lo=b->first;hi=b->second;first=false;}
      else {lo=std::max(lo,b->first);hi=std::max(hi,b->second);}
    } else all_bounds=false;
  }
  f.finite=extended && finite_witness;
  f.finite_or_neginf=extended;
  if(all_bounds) bounds=std::pair(lo,hi);
  return intern("max",xs,{},"",f,bounds);
}

} // namespace rg
