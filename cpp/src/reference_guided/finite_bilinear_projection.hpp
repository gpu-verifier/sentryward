#pragma once
#include "checker.hpp"

namespace rg {
struct FiniteBilinearProof { Id replacement = invalid; J receipt; };
struct FiniteBilinearLimits {
  size_t work = 200000, depth = 64, swaps = 2048, groups = 256, factors = 8;
};

// Exact finite-sum identities. An optional independently derived state/output
// descriptor directs interchange; singleton sums do not require one.
// No kernel/input names and no logical-index loops.
// A projection read supplies only the orientation of Fubini: its component
// coordinate remains the inner reduction. All other read coordinates match
// the supplied prototype exactly. Every rewritten identity is independently
// valid; neither a read anchor nor a grouping fingerprint proves an equality.
class FiniteBilinearProjection {
  Checker &checker; Arena &a; FiniteBilinearLimits limits;
  Id projection_read = invalid, projection_axis = invalid;
  size_t component = 0, work = 0, swaps = 0, pulls = 0, masks = 0, fusions = 0, singletons = 0;
  uint64_t serial;
  struct Miss { const char *why; };
  struct Fold { Id axis, seed, body; Z count; };
  using Parameter = std::tuple<std::string, size_t, Id>;
  struct FusionKey {
    Z count; Id weight; std::vector<Parameter> parameters;
    bool operator<(const FusionKey &o) const {
      return std::tie(count,weight,parameters) < std::tie(o.count,o.weight,o.parameters);
    }
  };
  struct FusionPiece { FusionKey key; Id axis, body; };
  std::unordered_map<Id,Id> memo;
  // Orientation queries inspect only stored product/Fold shells. They neither
  // establish coverage nor compare numerical bodies, and are memoized per
  // (shell, component axis, prospective time axis).
  std::map<std::tuple<Id,Id,Id>,bool> projection_cache;
  std::map<std::tuple<Id,Id,Id>,Id> renames;
  std::map<std::tuple<Id,Id,Id>,Id> point_instances;
  std::map<Z,Id> time_axes;
  std::unordered_map<Id,FusionPiece> fusion_pieces;
  bool oriented = false;

  void charge(size_t n=1) {
    a.check();
    if(n>limits.work-std::min(work,limits.work))throw Miss{"work_budget"};
    work+=n;
  }
  void depth(size_t d) { charge();if(d>limits.depth)throw Miss{"depth_budget"}; }
  bool finite(Id id) const { return a.nodes[id].facts.finite; }
  std::optional<Fold> fold(Id id) const {
    const auto &n=a.nodes[id];
    if(n.op!="fold" || n.tag!="Sum" || n.args.size()!=2 || n.data.size()!=2 ||
       n.data[0].denominator()!=1 || n.data[0]<0 || n.data[0]>=Q(a.nodes.size()) ||
       n.data[1].denominator()!=1 || n.data[1]<=0 || n.data[1]>Q(65536) || !n.facts.finite)
      return {};
    Id v=n.data[0].numerator().convert_to<Id>();const auto &axis=a.nodes[v];
    if(axis.op!="var" || !axis.bounds || axis.bounds->first!=0 ||
       axis.bounds->second!=n.data[1]-1 || !axis.facts.finite ||
       !finite(n.args[0]) || !finite(n.args[1]) || a.depends(n.args[0],v))return {};
    return Fold{v,n.args[0],n.args[1],n.data[1].numerator()};
  }
  Id make_fold(Id axis,const Z &count,Id seed,Id body) {
    charge(3);
    if(body==a.zero)return seed;
    if(!finite(seed)||!finite(body)||a.depends(seed,axis))throw Miss{"fold_scope"};
    return a.intern("fold",{seed,body},{Q(axis),Q(count)},"Sum",{true,false,false});
  }
  Id product(std::vector<Id> xs) {
    size_t edges=0;
    for(Id x:xs) { if(!finite(x))throw Miss{"nonfinite_product"};edges+=a.nodes[x].op=="mul"?a.nodes[x].args.size()+1:1; }
    charge(edges);return a.mul(std::move(xs));
  }
  Id sum_plain(std::vector<Id> xs) {
    size_t edges=0;
    for(Id x:xs) { if(!finite(x))throw Miss{"nonfinite_sum"};edges+=a.nodes[x].op=="add"?a.nodes[x].args.size():1; }
    charge(edges);return a.add(std::move(xs));
  }
  std::vector<Id> factors(Id id) {
    const auto &n=a.nodes[id];
    if(n.op!="mul")return {id};
    if(n.args.size()>limits.factors || n.data.size()!=1)throw Miss{"factor_budget"};
    charge(n.args.size()+1);auto result=n.args;
    if(n.data[0]!=1)result.push_back(a.constant(n.data[0]));
    return result;
  }
  bool read_matches(Id id,Id axis,Id excluded=invalid) {
    if(!oriented)return false;
    const auto &n=a.nodes[id],&p=a.nodes[projection_read];
    if(n.op!="read" || n.tag!=p.tag || n.data!=p.data || n.args.size()!=p.args.size() || !n.facts.finite)return false;
    for(size_t k=0;k<n.args.size();++k)if(k!=component && n.args[k]!=p.args[k])return false;
    Id index=n.args[component];const auto &x=a.nodes[index];
    if(x.args.size()>4 || !x.bounds || x.bounds->first<a.nodes[projection_axis].bounds->first ||
       x.bounds->second>a.nodes[projection_axis].bounds->second)return false;
    if(excluded!=invalid && a.depends(index,excluded))return false;
    Q coefficient(0),offset(0);
    charge(x.args.size()+1);
    for(auto [term,c]:a.terms(index)) {
      if(c.denominator()!=1)return false;
      if(term==axis)coefficient+=c;
      else if(term==a.one)offset+=c;
      else {
        // A blocked component coordinate can use several existing integer
        // binders, e.g. 16*outer+inner. Retain those exact terms; this check is
        // only an orientation witness, never a dense-domain or bijection proof.
        const auto &other=a.nodes[term];
        if(other.op!="var" || !other.facts.finite || !other.bounds ||
           other.bounds->first.denominator()!=1 || other.bounds->second.denominator()!=1 ||
           c<=0)return false;
      }
    }
    return coefficient>0 && offset.denominator()==1;
  }
  bool projection_factor(Id body,Id axis,Id excluded=invalid) {
    auto key=std::tuple(body,axis,excluded);
    if(auto it=projection_cache.find(key);it!=projection_cache.end()) {
      ++a.stats["finite_bilinear_projection_cache_hits"];return it->second;
    }
    // Inner component/time interchange moves the read below a component Fold.
    // Follow only the product/Fold shell so an outer component binder can
    // continue the same directed interchange. Numerical operands (Exp, calls,
    // Select predicates, read-index DAGs) are not recursively searched here.
    std::vector<Id> todo{body};std::unordered_set<Id> seen;size_t edges=0;
    bool found=false;
    while(!todo.empty()) {
      charge();Id id=todo.back();todo.pop_back();
      if(!seen.insert(id).second || !a.depends(id,axis))continue;
      if(seen.size()>64)throw Miss{"projection_shell_budget"};
      ++a.stats["finite_bilinear_projection_shell_visits"];
      if(read_matches(id,axis,excluded)){found=true;break;}
      const auto &n=a.nodes[id];
      if(n.op=="mul") {
        if(n.args.size()>limits.factors || n.data.size()!=1)throw Miss{"factor_budget"};
        if(n.args.size()>128-std::min<size_t>(edges,128))throw Miss{"projection_shell_budget"};
        edges+=n.args.size();charge(n.args.size());
        todo.insert(todo.end(),n.args.begin(),n.args.end());
      } else if(auto f=fold(id)) {
        if(f->axis==axis || f->axis==excluded)continue; // No shadowed witness.
        if(++edges>128)throw Miss{"projection_shell_budget"};
        todo.push_back(f->body);
      }
    }
    projection_cache.emplace(key,found);return found;
  }
  bool setup(Id read,Id axis) {
    if(read==invalid && axis==invalid)return true;
    if(read==invalid || axis==invalid || read>=a.nodes.size() || axis>=a.nodes.size())return false;
    const auto &p=a.nodes[read],&v=a.nodes[axis];
    if(p.op!="read" || !p.facts.finite || p.args.empty() || p.args.size()>4 ||
       v.op!="var" || !v.bounds || v.bounds->first.denominator()!=1 ||
       v.bounds->second.denominator()!=1 || v.bounds->first>v.bounds->second)return false;
    size_t found=0;
    for(size_t k=0;k<p.args.size();++k)if(a.depends(p.args[k],axis)){component=k;++found;}
    if(found!=1)return false;
    projection_read=read;projection_axis=axis;oriented=true;
    return read_matches(read,axis);
  }

  // Rename one free integer binder to a fresh variable with the SAME domain.
  // This is alpha renaming, not parameter generalization, so retained scalar
  // facts/bounds stay valid. A target occurring free or bound would capture.
  Id rename(Id id,Id from,Id to,size_t d=0) {
    depth(d);if(from==to || !a.depends(id,from))return id;
    if(a.nodes[from].op!="var" || a.nodes[to].op!="var" ||
       a.nodes[from].bounds!=a.nodes[to].bounds)throw Miss{"rename_domain"};
    if(id==from)return to;
    if(a.depends(id,to))throw Miss{"rename_capture"};
    auto key=std::tuple(id,from,to);
    if(auto it=renames.find(key);it!=renames.end())return it->second;
    const auto &n=a.nodes[id];charge(n.args.size());
    if(n.op=="fold" && n.data.size()==2 && (n.data[0]==Q(to) || n.data[0]==Q(from)))
      throw Miss{"rename_capture"}; // A shadowing Fold binds its body, not its seed.
    std::vector<Id> args;args.reserve(n.args.size());
    for(Id child:n.args)args.push_back(rename(child,from,to,d+1));
    Id value;
    if(n.op=="add") {
      std::vector<Id> terms;
      for(size_t k=0;k<args.size();++k)terms.push_back(product({a.constant(n.data[k]),args[k]}));
      value=sum_plain(std::move(terms));
    } else if(n.op=="mul") {args.push_back(a.constant(n.data[0]));value=product(std::move(args));}
    else if(n.op=="signed") {
      if(args.size()!=1 || n.data.size()!=1 || n.data[0].denominator()!=1 ||
         n.data[0]<=0 || n.data[0]>256) throw Miss{"signed_word_width"};
      value=a.signed_word(args[0],n.data[0].numerator().convert_to<unsigned>());
    }
    else value=a.intern(n.op,std::move(args),n.data,n.tag,n.facts,n.bounds);
    renames.emplace(key,value);return value;
  }
  Id time_axis(const Z &count) {
    if(auto it=time_axes.find(count);it!=time_axes.end())return it->second;
    if(time_axes.size()==limits.groups)throw Miss{"axis_budget"};
    const size_t before=a.nodes.size();
    Id id=a.var("finite-bilinear:"+std::to_string(serial)+":time:"+count.str(),0,Q(count-1));
    if(size_t(id)<before)throw Miss{"axis_name_collision"};
    time_axes.emplace(count,id);return id;
  }
  std::optional<FusionPiece> fusion_piece(Id axis,const Z &count,Id body) {
    Id canonical=time_axis(count);
    if(axis!=canonical && a.depends(body,canonical))throw Miss{"rename_capture"};
    Id renamed=rename(body,axis,canonical);
    Id state=invalid;std::vector<Id> outside;
    for(Id factor:factors(renamed)) {
      auto f=fold(factor);
      if(f && f->seed==a.zero && projection_factor(f->body,f->axis)) {
        if(state!=invalid)return {};
        state=factor;
      } else outside.push_back(factor);
    }
    if(state==invalid)return {};
    const auto sf=*fold(state);
    FusionKey key{count,product(std::move(outside)),{}};
    std::vector<Id> todo{sf.body};std::unordered_set<Id> seen;size_t edges=0;
    while(!todo.empty()) {
      charge();Id id=todo.back();todo.pop_back();if(!seen.insert(id).second)continue;
      if(seen.size()>64)throw Miss{"fingerprint_budget"};
      const auto &n=a.nodes[id];
      if(n.args.size()>128-std::min<size_t>(edges,128))throw Miss{"fingerprint_budget"};
      edges+=n.args.size();charge(n.args.size());
      if(n.op=="read") {
        for(size_t k=0;k<n.args.size();++k)if(!a.depends(n.args[k],sf.axis))
          key.parameters.emplace_back(n.tag,k,n.args[k]);
      } else if(n.op=="fold")return {};
      else todo.insert(todo.end(),n.args.begin(),n.args.end());
    }
    std::sort(key.parameters.begin(),key.parameters.end());
    key.parameters.erase(std::unique(key.parameters.begin(),key.parameters.end()),key.parameters.end());
    return FusionPiece{std::move(key),canonical,renamed};
  }
  Id time_piece(Id axis,const Z &count,Id body) {
    Id value=make_fold(axis,count,a.zero,body);
    if(value!=a.zero)if(auto piece=fusion_piece(axis,count,body))fusion_pieces.emplace(value,std::move(*piece));
    return value;
  }
  Id sum(std::vector<Id> xs) {
    Id value=sum_plain(std::move(xs));
    const auto &n=a.nodes[value];
    if(!oriented || n.op!="add")return value;
    charge(n.args.size());
    struct Group { Id axis;std::vector<Id> bodies,originals; };
    std::map<FusionKey,Group> groups;std::vector<Id> remaining;
    for(size_t k=0;k<n.args.size();++k) {
      Id original=product({a.constant(n.data[k]),n.args[k]});
      auto it=fusion_pieces.find(n.args[k]);
      if(it==fusion_pieces.end()) {remaining.push_back(original);continue;}
      auto key=it->second.key;key.weight=product({a.constant(n.data[k]),key.weight});
      auto [g,fresh]=groups.try_emplace(key,Group{it->second.axis,{},{}});
      if(groups.size()>limits.groups)throw Miss{"fusion_group_budget"};
      g->second.bodies.push_back(product({a.constant(n.data[k]),it->second.body}));
      g->second.originals.push_back(original);
    }
    for(auto &[key,group]:groups) {
      if(group.bodies.size()==1) {remaining.push_back(group.originals[0]);continue;}
      Id joined=sum_plain(std::move(group.bodies));
      // Same finite weight is common to every term; factoring creates the
      // inner Add of state pieces before the enclosing time binder is closed.
      charge(a.nodes[joined].args.size());joined=a.factor_sum(joined);
      Id merged=make_fold(group.axis,key.count,a.zero,joined);
      fusion_pieces.emplace(merged,FusionPiece{key,group.axis,joined});
      remaining.push_back(merged);++fusions;
    }
    return sum_plain(std::move(remaining));
  }

  // Find an exact one-point support of a finite zero mask on [0,count).
  // Only an existing affine comparison against literals is inspected. This
  // does not search tensor contents, enumerate indices, or strengthen facts.
  std::optional<Z> singleton_mask_point(Id body,Id axis,const Z &count) {
    for(Id factor:factors(body)) {
      charge();const auto &mask=a.nodes[factor];
      if(mask.op!="select" || mask.args.size()!=3 || !mask.facts.finite)continue;
      bool invert=mask.args[1]==a.zero;
      if(!invert && mask.args[2]!=a.zero)continue;
      const auto &predicate=a.nodes[mask.args[0]];
      if(predicate.op!="compare" || predicate.args.size()!=2)continue;
      std::string op=predicate.tag;
      if(invert) {
        if(op=="eq")op="ne";else if(op=="ne")op="eq";
        else if(op=="lt")op="ge";else if(op=="le")op="gt";
        else if(op=="gt")op="le";else if(op=="ge")op="lt";else continue;
      }
      Z coefficient=0,offset=0;bool affine=true;
      for(size_t side=0;side<2 && affine;++side) {
        const auto &n=a.nodes[predicate.args[side]];
        if(!n.facts.finite || n.args.size()>4){affine=false;break;}
        charge(n.args.size()+1);
        for(auto [term,c]:a.terms(predicate.args[side])) {
          if(c.denominator()!=1){affine=false;break;}
          Z value=c.numerator();if(side)value=-value;
          if(term==axis)coefficient+=value;
          else if(term==a.one)offset+=value;
          else {affine=false;break;}
        }
      }
      if(!affine)continue;
      if(coefficient<0) {
        coefficient=-coefficient;offset=-offset;
        if(op=="lt")op="gt";else if(op=="le")op="ge";
        else if(op=="gt")op="lt";else if(op=="ge")op="le";
      }
      Z low=0,high=count-1;
      if(coefficient==0) {
        bool truth=op=="eq"?offset==0:op=="ne"?offset!=0:op=="lt"?offset<0:
                   op=="le"?offset<=0:op=="gt"?offset>0:op=="ge" && offset>=0;
        if(truth && count==1)return Z(0);
        continue;
      }
      Z floor=floorz(-offset,coefficient),ceil=-floorz(offset,coefficient);
      if(op=="eq") {
        if(modz(-offset,coefficient)==0 && floor>=0 && floor<count)return floor;
        continue;
      }
      if(op=="ne") {
        bool excluded=modz(-offset,coefficient)==0 && floor>=0 && floor<count;
        if(count==1 && !excluded)return Z(0);
        if(count==2 && excluded)return Z(1)-floor;
        continue;
      }
      if(op=="lt")high=std::min(high,Z(ceil-1));
      else if(op=="le")high=std::min(high,floor);
      else if(op=="gt")low=std::max(low,Z(floor+1));
      else if(op=="ge")low=std::max(low,ceil);
      else continue;
      if(low==high)return low;
    }
    return {};
  }
  Id instantiate_point(Id root,Id axis,const Z &point) {
    const auto &domain=a.nodes[axis];
    if(domain.op!="var" || !domain.bounds || Q(point)<domain.bounds->first || Q(point)>domain.bounds->second)
      throw Miss{"point_domain"};
    Id literal=a.constant(Q(point));
    struct Task {Id id;bool ready;size_t depth;};
    std::vector<Task> todo{{root,false,0}};
    while(!todo.empty()) {
      auto task=todo.back();todo.pop_back();depth(task.depth);
      auto key=std::tuple(task.id,axis,literal);
      if(point_instances.contains(key))continue;
      if(task.id==axis){point_instances.emplace(key,literal);continue;}
      if(!a.depends(task.id,axis)){point_instances.emplace(key,task.id);continue;}
      const auto &n=a.nodes[task.id];
      if(n.args.size()>128)throw Miss{"point_arity"};
      charge(n.args.size());
      bool shadow=n.op=="fold" && n.data.size()==2 && n.data[0]==Q(axis);
      if(!task.ready) {
        todo.push_back({task.id,true,task.depth});
        for(size_t k=0;k<n.args.size();++k)
          if(!(shadow && k==1) && !point_instances.contains(std::tuple(n.args[k],axis,literal)))
            todo.push_back({n.args[k],false,task.depth+1});
        continue;
      }
      std::vector<Id> args;args.reserve(n.args.size());
      for(size_t k=0;k<n.args.size();++k)
        args.push_back(shadow && k==1?n.args[k]:point_instances.at(std::tuple(n.args[k],axis,literal)));
      Id value;
      if((n.op=="idiv" || n.op=="imod") && args.size()==1 && n.data.size()==1 &&
         n.data[0].denominator()==1 && n.data[0]>0)
        value=n.op=="idiv"?a.idiv(args[0],n.data[0].numerator()):a.imod(args[0],n.data[0].numerator());
      // Recompute arithmetic/predicate facts through the shared constructors.
      // Opaque reads keep their declared facts: the literal is a member of
      // the original binder domain, not an enlargement of its assumptions.
      else value=checker.reapply(task.id,args);
      point_instances.emplace(key,value);
      ++a.stats["finite_bilinear_point_node_visits"];
    }
    return point_instances.at(std::tuple(root,axis,literal));
  }

  Id normalize_fold(Id axis,const Z &count,Id seed,Id body,size_t d) {
    depth(d);
    if(!finite(seed)||!finite(body)||a.depends(seed,axis))throw Miss{"fold_scope"};
    if(body==a.zero)return seed;
    if(auto point=singleton_mask_point(body,axis,count)) {
      Id value=instantiate_point(body,axis,*point);
      ++singletons;return sum({seed,value});
    }
    auto fs=factors(body);std::vector<Id> invariant,dynamic;
    for(Id f:fs)(a.depends(f,axis)?dynamic:invariant).push_back(f);
    Id scale=product(std::move(invariant));
    if(dynamic.empty()) {++pulls;return sum({seed,product({a.constant(Q(count)),scale})});}
    if(scale!=a.one) {
      ++pulls;Id inner=normalize_fold(axis,count,a.zero,product(std::move(dynamic)),d+1);
      return sum({seed,product({scale,inner})});
    }
    if(oriented) {
      // Distribute one small Add hidden by projection factors. No repeated
      // polynomial expansion and no products of independent sum families.
      Id additive=invalid;std::vector<Id> outside;
      for(Id f:fs) {
        if(a.nodes[f].op=="add") {if(additive!=invalid){additive=invalid;outside.clear();break;}additive=f;}
        else outside.push_back(f);
      }
      bool distribute=false;
      if(additive!=invalid && a.nodes[additive].args.size()<=4) {
        distribute=projection_factor(product(outside),axis);
        // A seeded inner interchange already separates its carried seed from
        // its time contribution. Both can contain the projection under their
        // own Fold shells, rather than in an outside multiplier. Apply the
        // same finite linearity identity so the outer component can continue;
        // do not turn unrelated arithmetic Adds into a polynomial search.
        if(!distribute) {
          distribute=true;
          for(Id term:a.nodes[additive].args)
            if(!projection_factor(term,axis)){distribute=false;break;}
        }
      }
      if(distribute) {
        const auto &add=a.nodes[additive];charge(add.args.size());std::vector<Id> pieces{seed};
        for(size_t k=0;k<add.args.size();++k) {
          auto term=outside;term.push_back(add.args[k]);term.push_back(a.constant(add.data[k]));
          pieces.push_back(normalize_fold(axis,count,a.zero,product(std::move(term)),d+1));
        }
        return sum(std::move(pieces));
      }
      Id inner=invalid;outside.clear();
      for(Id f:fs) {
        if(fold(f)) {if(inner!=invalid){inner=invalid;outside.clear();break;}inner=f;}
        else outside.push_back(f);
      }
      if(inner!=invalid) {
        const Fold next=*fold(inner);Id multiplier=product(outside);
        const bool projection=projection_factor(multiplier,axis,next.axis) ||
                              projection_factor(next.body,axis,next.axis);
        if(!projection)return make_fold(axis,count,seed,body);
        if(next.axis==axis || a.depends(multiplier,next.axis))throw Miss{"interchange_capture"};
        if(++swaps>limits.swaps)throw Miss{"swap_budget"};
        Id carried=next.seed==a.zero?a.zero:
          normalize_fold(axis,count,a.zero,product({multiplier,next.seed}),d+1);
        Id projected=normalize_fold(axis,count,a.zero,product({multiplier,next.body}),d+1);
        return sum({seed,carried,time_piece(next.axis,next.count,projected)});
      }
    }
    return make_fold(axis,count,seed,body);
  }
  Id visit(Id id,size_t d=0) {
    depth(d);if(auto it=memo.find(id);it!=memo.end())return it->second;
    const auto &n=a.nodes[id];Id result=id;
    if((n.op=="add" || n.op=="mul" || n.op=="select" || n.op=="fold") && n.facts.finite) {
      charge(n.args.size());std::vector<Id> args;args.reserve(n.args.size());
      for(Id child:n.args)args.push_back(visit(child,d+1));
      if(n.op=="add") {
        std::vector<Id> terms;for(size_t k=0;k<args.size();++k)terms.push_back(product({a.constant(n.data[k]),args[k]}));
        result=sum(std::move(terms));
      } else if(n.op=="mul") {args.push_back(a.constant(n.data[0]));result=product(std::move(args));}
      else if(n.op=="select" && args.size()==3) {
        // Select polarity is canonicalized independently of which arm is
        // active. Preserve that polarity in the indicator instead of requiring
        // the zero arm to remain false after predicate normalization.
        bool zero_true=args[1]==a.zero,zero_false=args[2]==a.zero;
        Id payload=zero_true?args[2]:args[1];
        if(zero_true!=zero_false && payload!=a.one && finite(args[0]) && finite(payload)) {
          Id indicator=a.select(args[0],zero_true?a.zero:a.one,zero_true?a.one:a.zero);
          result=product({indicator,payload});++masks;
        } else result=a.select(args[0],args[1],args[2]);
      } else if(n.op=="fold") {
        if(auto f=fold(id))result=normalize_fold(f->axis,f->count,args[0],args[1],d+1);
        else if(args!=n.args)result=checker.reapply(id,args);
      }
    }
    memo.emplace(id,result);return result;
  }
public:
  std::string last_failure;
  explicit FiniteBilinearProjection(Checker &owner,FiniteBilinearLimits budget={})
      :checker(owner),a(owner.a),limits(budget),serial(++a.stats["finite_bilinear_banks"]) {
    limits.work=std::min<size_t>(limits.work,200000);limits.depth=std::min<size_t>(limits.depth,64);
    limits.swaps=std::min<size_t>(limits.swaps,2048);limits.groups=std::min<size_t>(limits.groups,256);
    limits.factors=std::min<size_t>(limits.factors,8);
  }
  std::optional<FiniteBilinearProof> rewrite(Id root,Id read=invalid,Id axis=invalid) {
    work=swaps=pulls=masks=fusions=singletons=0;oriented=false;projection_read=projection_axis=invalid;
    // Canonical binders remain bank-owned across calls. Newly obtained names
    // must be fresh; later calls can reuse only IDs recorded by this bank.
    memo.clear();projection_cache.clear();renames.clear();point_instances.clear();fusion_pieces.clear();last_failure.clear();
    try {
      if(root>=a.nodes.size() || !finite(root))throw Miss{"nonfinite_root"};
      if(!setup(read,axis))throw Miss{"projection_descriptor"};
      Id replacement=visit(root);
      a.stats["finite_bilinear_swaps"]+=swaps;a.stats["finite_bilinear_fusions"]+=fusions;
      a.stats["finite_bilinear_singleton_sums"]+=singletons;
      a.stats["finite_bilinear_masks"]+=masks;a.stats["finite_bilinear_work"]+=work;
      return FiniteBilinearProof{replacement,{{"method","finite_distributivity_and_sum_interchange"},
        {"projection_read",read},{"projection_axis",axis},{"swaps",swaps},{"fusions",fusions},
        {"invariant_factor_pulls",pulls},{"zero_masks",masks},{"singleton_sums",singletons},{"stored_work",work},
        {"coverage","required_by_subsequent_native_reduction_check"}}};
    } catch(const Miss &miss) {last_failure=miss.why;}
      catch(const Unsupported &) {last_failure="unsupported";}
    ++a.stats["finite_bilinear_decline_"+last_failure];a.stats["finite_bilinear_work"]+=work;return {};
  }
};
} // namespace rg
