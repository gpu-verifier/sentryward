#pragma once
#include "core.hpp"

namespace rg {

inline auto Arena::var(std::string key, Q lo, Q hi) -> Id {
  // Integer index variable, used by family coordinates and Fold binders.
  // Native callers must use read/atom for symbolic real-valued inputs.
  return intern("var", {}, {lo, hi}, std::move(key), {true, lo > 0, lo >= 0},
                std::pair(lo, hi));
}

inline auto Arena::terms(Id n) -> std::vector<std::pair<Id, Q>> {
  const auto &x = nodes[n];
  if (x.op == "const")
    return {{one, x.data[0]}};
  if (x.op == "add") {
    std::vector<std::pair<Id, Q>> r;
    r.reserve(x.args.size());
    for (size_t i = 0; i < x.args.size(); ++i)
      r.emplace_back(x.args[i], x.data[i]);
    return r;
  }
  if (x.op == "mul" && x.data[0] != 1)
    return {{mul(x.args), x.data[0]}};
  return {{n, Q(1)}};
}

inline auto Arena::add(std::vector<Id> xs) -> Id {
  xs.erase(std::remove(xs.begin(), xs.end(), zero), xs.end());
  if (xs.empty())
    return zero;
  if (xs.size() == 1)
    return xs[0];
  std::set<int> inf;
  bool all = true;
  for (Id i : xs) {
    auto &n = nodes[i];
    if (n.op == "infinity")
      inf.insert(n.data[0].numerator().convert_to<int>());
    else if (!n.facts.finite)
      all = false;
  }
  if (inf.size() > 1)
    return intern("undefined", {}, {}, "opposite infinities");
  if (!inf.empty() && all)
    return infinity(*inf.begin());
  bool finite_or_neginf = std::all_of(xs.begin(), xs.end(), [&](Id i) {
    return nodes[i].facts.finite_or_negative_infinity();
  });
  for (Id i : xs)
    if (!nodes[i].facts.finite)
      return intern("add_raw", xs, std::vector<Q>(xs.size(), Q(1)), "",
                    {false,false,false,finite_or_neginf});
  std::map<Id, Q> ts;
  for (Id i : xs)
    for (const auto &[n, c] : terms(i)) {
      ts[n] += c;
      ++counter(Hot::add_term_visits);
      check();
    }
  std::vector<Id> keys;
  for (auto &[n, c] : ts)
    if (c != 0)
      keys.push_back(n);
  for (Id n : keys) {
    const auto &x = nodes[n];
    auto it = ts.find(n);
    if (x.op != "imod" || it == ts.end() || it->second == 0)
      continue;
    auto d = find_node("idiv", x.args, x.data);
    if (d && ts.contains(*d) && ts[*d] == it->second * x.data[0]) {
      Q c = it->second;
      ts.erase(n);
      ts.erase(*d);
      for (const auto &[t, k] : terms(x.args[0]))
        ts[t] += c * k;
    }
  }
  for (auto it = ts.begin(); it != ts.end();)
    if (it->second == 0)
      it = ts.erase(it);
    else
      ++it;
  if (ts.empty())
    return zero;
  if (ts.size() == 1)
    return mul({constant(ts.begin()->second), ts.begin()->first});
  std::vector<Id> as;
  std::vector<Q> ds;
  as.reserve(ts.size());
  ds.reserve(ts.size());
  for (auto &[n, c] : ts) {
    as.push_back(n);
    ds.push_back(std::move(c));
  }
  // Facts and bounds read only stored nodes and these terms, in key order.
  return intern_lazy("add", as, ds, "", [&](size_t hashed) {
    Facts f{true, false, true};
    Bounds b = std::pair(Q(0), Q(0));
    for (size_t i = 0; i < as.size(); ++i) {
      const Q &c = ds[i];
      const auto &x = nodes[as[i]];
      f.finite &= x.facts.finite;
      f.nn &= c >= 0 && x.facts.nn;
      f.pos |= c > 0 && x.facts.pos;
      if (b && x.bounds) {
        Q l = x.bounds->first * c, h = x.bounds->second * c;
        b->first += std::min(l, h);
        b->second += std::max(l, h);
      } else
        b = {};
    }
    f.pos &= f.nn;
    return insert_node(hashed, "add", std::move(as), std::move(ds), "", f, std::move(b));
  });
}

inline auto Arena::mul(std::vector<Id> xs) -> Id {
  Q scalar(1);
  std::vector<Id> factors, es, natural_es, rest;
  for (Id i : xs) {
    const auto &n = nodes[i];
    if (n.op == "const")
      scalar *= n.data[0];
    else if (n.op == "mul") {
      scalar *= n.data[0];
      factors.insert(factors.end(), n.args.begin(), n.args.end());
    } else
      factors.push_back(i);
  }
  if (scalar == 0 && std::all_of(factors.begin(), factors.end(),
                                 [&](Id i) { return nodes[i].facts.finite; }))
    return zero;
  for (Id i : factors) {
    const auto &n = nodes[i];
    if (n.op == "exp2" && nodes[n.args[0]].facts.finite)
      es.push_back(n.args[0]);
    else if (n.op == "exp" && nodes[n.args[0]].facts.finite)
      natural_es.push_back(n.args[0]);
    else
      rest.push_back(i);
  }
  if (exact_ln2 != invalid) natural_log2_factors(rest);
  Id binary = es.empty() ? zero : add(es), natural = natural_es.empty() ? zero : add(natural_es);
  if (auto bases = exponent_bases(natural, binary)) std::tie(natural, binary) = *bases;
  if (binary != zero)
    rest.push_back(intern("exp2", {binary}, {}, "", {true, true, true}));
  if (natural != zero)
    rest.push_back(intern("exp", {natural}, {}, "", {true, true, true}));
  if (rest.empty())
    return constant(scalar);
  // Distribute only a single -infinity mask over a strictly positive scale.
  // The other branch must already be finite. Never form products of branch
  // combinations, and never treat zero * -infinity as zero.
  if (scalar > 0) {
    size_t masked = rest.size(); bool positive_others = true;
    for (size_t i=0;i<rest.size();++i) {
      const auto &n=nodes[rest[i]];
      bool candidate=n.op=="select" && n.args.size()==3 &&
        ((neginf(n.args[1]) && nodes[n.args[2]].facts.finite) ||
         (neginf(n.args[2]) && nodes[n.args[1]].facts.finite));
      if (candidate && masked==rest.size()) masked=i;
      else if (!n.facts.finite || !n.facts.pos) positive_others=false;
    }
    if (masked<rest.size() && positive_others &&
        (rest.size()>1 || scalar!=1)) {
      auto args=nodes[rest[masked]].args;
      bool negative_true=neginf(args[1]);
      std::vector<Id> positive{constant(scalar)};
      for(size_t i=0;i<rest.size();++i)if(i!=masked)positive.push_back(rest[i]);
      Id factor=mul(std::move(positive));
      Id value=mul({factor,args[negative_true?2:1]});
      ++stats["positive_mask_scale_normalizations"];
      return select(args[0],negative_true?args[1]:value,negative_true?value:args[2]);
    }
  }
  if (rest.size() == 1) {
    const auto &n = nodes[rest[0]];
    if (n.op == "infinity" && scalar != 0)
      return infinity(n.data[0].numerator().convert_to<int>() *
                      (scalar > 0 ? 1 : -1));
    if (scalar == 1)
      return rest[0];
    if (scalar == -1 && n.op == "clamp") {
      Id x=n.args[0];Q lo=-n.data[1],hi=-n.data[0];
      return clamp(neg(x),lo,hi);
    }
    if (n.op == "add") {
      std::vector<Id> ys;
      for (size_t i = 0; i < n.args.size(); ++i)
        ys.push_back(mul({constant(scalar * n.data[i]), n.args[i]}));
      return add(ys);
    }
  }
  std::sort(rest.begin(), rest.end());
  // Facts and bounds read only stored factor nodes and the scalar.
  return intern_lazy("mul", rest, std::span<const Q>(&scalar, 1), "", [&](size_t hashed) {
    Facts f{true, scalar > 0, scalar >= 0};
    bool extended_product=scalar>0; unsigned extended_factors=0;
    Bounds b = std::pair(scalar, scalar);
    for (Id i : rest) {
      const auto &n = nodes[i];
      f.finite &= n.facts.finite;
      f.pos &= n.facts.pos;
      f.nn &= n.facts.nn;
      if (!n.facts.finite) {
        ++extended_factors;
        extended_product &= n.facts.finite_or_negative_infinity();
      } else extended_product &= n.facts.pos;
      if (b && n.bounds) {
        std::array<Q, 4> vs{
            b->first * n.bounds->first, b->first * n.bounds->second,
            b->second * n.bounds->first, b->second * n.bounds->second};
        b = std::pair(*std::min_element(vs.begin(), vs.end()),
                      *std::max_element(vs.begin(), vs.end()));
      } else
        b = {};
    }
    f.finite_or_neginf=extended_product && extended_factors==1;
    if(f.finite && scalar>=0 && !f.nn) {
      bool nonnegative=true;
      for(size_t i=0;i<rest.size();) {
        size_t end=i+1;while(end<rest.size() && rest[end]==rest[i])++end;
        if((end-i)%2 && !nodes[rest[i]].facts.nn) {nonnegative=false;break;}
        i=end;
      }
      f.nn=nonnegative;
    }
    return insert_node(hashed, "mul", std::move(rest), {scalar}, "", f, std::move(b));
  });
}

inline auto Arena::floor_value(Id n, Bounds established) -> Id {
  if(auto v=value(n))return constant(Q(floorq(*v)));
  if(!nodes[n].facts.finite)throw Unsupported("floor requires finite input");
  Bounds bounds=established;
  if(nodes[n].bounds) {
    auto b=std::pair(Q(floorq(nodes[n].bounds->first)),Q(floorq(nodes[n].bounds->second)));
    if(bounds) b={std::max(b.first,bounds->first),std::min(b.second,bounds->second)};
    if(b.first>b.second)throw Unsupported("inconsistent floor bounds");
    bounds=b;
  }
  return intern("floor",{n},{},"",{true,bounds && bounds->first>0,bounds && bounds->first>=0},bounds);
}

inline auto Arena::exponential(Id n, bool base2) -> Id {
  if (n == zero)
    return one;
  if (neginf(n)) return zero;
  if (nodes[n].facts.finite)
    if (auto bases = base2 ? exponent_bases(zero, n) : exponent_bases(n, zero)) {
      ++stats["exact_exponential_base_changes"];
      return mul({exponential(bases->first, false), exponential(bases->second, true)});
    }
  // Only normalize a mask with an immediately eliminable -infinity arm.
  // Nested arbitrary Selects remain shared atoms; there is no path splitting.
  if (nodes[n].op=="select" && nodes[n].args.size()==3) {
    auto xs=nodes[n].args;
    bool nt=neginf(xs[1]), nf=neginf(xs[2]);
    if ((nt && nodes[xs[2]].facts.finite) ||
        (nf && nodes[xs[1]].facts.finite)) {
      Id e=exponential(xs[nt?2:1],base2);
      ++stats["masked_exponential_normalizations"];
      return select(xs[0],nt?zero:e,nt?e:zero);
    }
  }
  // Exp(x+c)=Exp(x)*Exp(c) remains defined for x in R union {-infinity}
  // and finite c. Match just this additive shell, not x's producer DAG.
  // The additive-shell rules below, for a raw sum given by its operands and coefficients (the
  // shell's own, or scaled by a positive constant for Exp(k * shell)).
  const auto raw_sum=[&](std::vector<Id> xs,std::vector<Q> coefficients)->std::optional<Id> {
    size_t extended=xs.size(); bool supported=true;
    for(size_t i=0;i<xs.size();++i)if(!nodes[xs[i]].facts.finite) {
      if(extended!=xs.size() || coefficients[i]<=0 ||
         !nodes[xs[i]].facts.finite_or_negative_infinity()) {supported=false;break;}
      extended=i;
    }
    // Do not chase an arbitrarily deep extended-real addition history.
    // Such a child remains an opaque, finitely summarized exponential.
    if(supported && extended<xs.size() && xs.size()>1 &&
       nodes[xs[extended]].op!="add_raw") {
      std::vector<Id> offsets;
      for(size_t i=0;i<xs.size();++i)if(i!=extended)
        offsets.push_back(mul({constant(coefficients[i]),xs[i]}));
      Id body=mul({constant(coefficients[extended]),xs[extended]});
      Id offset=add(std::move(offsets));
      Id masked=exponential(body,base2), scale=exponential(offset,base2);
      ++stats["masked_exponential_finite_offsets"];
      return mul({masked,scale});
    }
    // Several extended terms that are Selects on one condition p, each arm finite or -infinity,
    // with positive coefficients: the sum is select(p, sum of the true arms, sum of the false arms)
    // plus the finite terms, pointwise (no +infinity arises), so Exp of it is Exp of that one
    // Select, which the mask rule above normalizes. Inspects only this additive shell. Exact.
    Id condition=invalid;size_t selects=0;bool same=true;
    for(size_t i=0;i<xs.size() && same;++i) {
      if(nodes[xs[i]].facts.finite)continue;
      const auto &term=nodes[xs[i]];
      same=coefficients[i]>0 && term.op=="select" && term.args.size()==3 &&
           (condition==invalid || term.args[0]==condition) &&
           nodes[term.args[1]].facts.finite_or_negative_infinity() &&
           nodes[term.args[2]].facts.finite_or_negative_infinity();
      if(same){condition=term.args[0];++selects;}
    }
    if(same && selects>=2) {
      std::vector<Id> yes,no;
      for(size_t i=0;i<xs.size();++i) {
        const Id c=constant(coefficients[i]);
        if(nodes[xs[i]].facts.finite){yes.push_back(mul({c,xs[i]}));no.push_back(mul({c,xs[i]}));continue;}
        const auto arms=nodes[xs[i]].args;
        yes.push_back(mul({c,arms[1]}));no.push_back(mul({c,arms[2]}));
      }
      ++stats["masked_exponential_same_condition_sums"];
      return exponential(select(condition,add(std::move(yes)),add(std::move(no))),base2);
    }
    return std::nullopt;
  };
  if (nodes[n].op=="add_raw" && nodes[n].args.size()==nodes[n].data.size())
    if (auto result=raw_sum(nodes[n].args,nodes[n].data)) return *result;
  // Exp(k * W) for a constant k > 0 and W one of the two shells above. Pointwise on R union
  // {-infinity}: k * select(p, a, b) = select(p, k a, k b), and k * (sum_i c_i x_i) = sum_i (k c_i) x_i
  // (k * -infinity = -infinity for k > 0; the extended terms keep positive coefficients, so no
  // +infinity and no 0 * infinity arises). Side conditions: k is a positive constant; W is the
  // immediate operand; -infinity sits in one arm of a Select whose other arm is finite, or in the
  // extended terms of a raw sum that the rules above accept (one finite-or--infinity term with a
  // positive coefficient that is not itself a raw sum, or several Selects on one condition with
  // finite-or--infinity arms and positive coefficients); every other term is finite. Inspects only
  // this shell. Exact.
  if (nodes[n].op=="mul" && nodes[n].args.size()==1 && nodes[n].data.size()==1 && nodes[n].data[0]>0 &&
      !nodes[nodes[n].args[0]].facts.finite) {
    const Q k=nodes[n].data[0];
    const Id inner=nodes[n].args[0];
    const std::string shell=nodes[inner].op;
    const auto xs=nodes[inner].args;
    const auto coefficients=nodes[inner].data;
    if (shell=="select" && xs.size()==3) {
      const bool nt=neginf(xs[1]), nf=neginf(xs[2]);
      if ((nt && nodes[xs[2]].facts.finite) || (nf && nodes[xs[1]].facts.finite)) {
        const Id e=exponential(mul({constant(k),xs[nt?2:1]}),base2);
        ++stats["masked_exponential_scaled_selects"];
        return select(xs[0],nt?zero:e,nt?e:zero);
      }
    } else if (shell=="add_raw" && xs.size()==coefficients.size()) {
      std::vector<Q> scaled;
      for (const Q &c:coefficients) scaled.push_back(c*k);
      if (auto result=raw_sum(xs,std::move(scaled))) {
        ++stats["masked_exponential_scaled_sums"];
        return *result;
      }
    }
  }
  // Exp(k * max(select(p, -inf, a_1), ..., select(p, -inf, a_n))), k > 0, each a_i finite, one
  // condition p and -infinity on the same arm (or the mirror). Pointwise the max is
  // select(p, -infinity, max(a_1, ..., a_n)) (a max with a finite operand is already finite and
  // never reaches this point), and k * (-infinity) = -infinity, so Exp of it is Exp of that one
  // Select, which the mask rule above normalizes. Inspects only this shell. Exact.
  {
    Q scale(1);Id inner=n;
    if(nodes[n].op=="mul" && nodes[n].args.size()==1 && nodes[n].data.size()==1 && nodes[n].data[0]>0) {
      scale=nodes[n].data[0];inner=nodes[n].args[0];
    }
    if(!nodes[inner].facts.finite && nodes[inner].op=="max" && nodes[inner].args.size()>=2) {
      const auto &operands=nodes[inner].args;
      Id condition=invalid;int infinite_arm=0;std::vector<Id> arms;
      for(Id x:operands) {
        const auto &term=nodes[x];
        const int arm=term.op=="select" && term.args.size()==3 ?
            (neginf(term.args[1]) && nodes[term.args[2]].facts.finite ? 1 :
             neginf(term.args[2]) && nodes[term.args[1]].facts.finite ? 2 : 0) : 0;
        if(arm==0 || (condition!=invalid && (term.args[0]!=condition || arm!=infinite_arm))) {arms.clear();break;}
        condition=term.args[0];infinite_arm=arm;arms.push_back(term.args[arm==1?2:1]);
      }
      if(arms.size()==operands.size()) {
        const Id finite=mul({constant(scale),maximum(std::move(arms))}), ninf=infinity(-1);
        ++stats["masked_exponential_same_condition_maxima"];
        return exponential(infinite_arm==1 ? select(condition,ninf,finite) : select(condition,finite,ninf),base2);
      }
    }
  }
  // Keep the identity and square forms consistent with Power. Inspect only
  // the immediate exponent: do not distribute or search its producer DAG.
  const auto &exponent = nodes[n];
  const char *logop = base2 ? "log2" : "log";
  if (exponent.op == logop && exponent.facts.finite)
    return exponent.args[0];
  if (exponent.op == "mul" && exponent.data[0] == 2 &&
      exponent.args.size() == 1) {
    const auto &log = nodes[exponent.args[0]];
    if (log.op == logop && log.facts.finite) {
      Id base = log.args[0];
      return mul({base, base});
    }
  }
  bool f = nodes[n].facts.finite;
  bool extended=nodes[n].facts.finite_or_negative_infinity();
  return intern(base2 ? "exp2" : "exp", {n}, {}, "", {extended, f, extended});
}

} // namespace rg
