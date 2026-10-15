#pragma once
#include <algorithm>
#include <bit>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/rational.hpp>
#include <chrono>
#include <climits>
#include <cmath>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace rg {
using J = nlohmann::json;
using Z = boost::multiprecision::cpp_int;
using Q = boost::rational<Z>;
using Id = uint32_t;
constexpr Id invalid = UINT32_MAX;
struct Unsupported : std::runtime_error {
  std::string code = "expression_form_unsupported";
  J details = J::object();
  using runtime_error::runtime_error;
  Unsupported(std::string message, std::string diagnostic_code, J evidence)
      : runtime_error(std::move(message)), code(std::move(diagnostic_code)), details(std::move(evidence)) {}
};
struct Timeout : std::runtime_error {
  Timeout() : runtime_error("wall clock deadline") {};
};
inline Z floorz(Z n, const Z &d) {
  Z q = n / d, r = n % d;
  if (r < 0)
    --q;
  return q;
}
inline Z modz(const Z &n, const Z &d) { return n - floorz(n, d) * d; }
inline Z floorq(const Q &q) { return floorz(q.numerator(), q.denominator()); }
inline Z pow2(unsigned w) { return Z(1) << w; }
inline std::string str(const Q &q) {
  return q.numerator().str() +
         (q.denominator() == 1 ? "" : "/" + q.denominator().str());
}
inline void mix(size_t &h, size_t v) {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
}
inline size_t hashq(const Q &q) {
  size_t h = boost::multiprecision::hash_value(q.numerator());
  mix(h, boost::multiprecision::hash_value(q.denominator()));
  return h;
}
inline Q decimal(std::string s) {
  auto pos = s.find('/');
  if (pos != s.npos)
    return Q(Z(s.substr(0, pos)), Z(s.substr(pos + 1)));
  int e = 0;
  pos = s.find_first_of("eE");
  if (pos != s.npos) {
    e = std::stoi(s.substr(pos + 1));
    s.resize(pos);
  }
  pos = s.find('.');
  if (pos != s.npos) {
    e -= int(s.size() - pos - 1);
    s.erase(pos, 1);
  }
  bool neg = false;
  if (!s.empty() && (s[0] == '-' || s[0] == '+')) {
    neg = s[0] == '-';
    s.erase(0, 1);
  }
  Z n = 0;
  for (char c : s) {
    if (c < '0' || c > '9')
      throw Unsupported("invalid decimal");
    n = n * 10 + (c - '0');
  }
  Z d = 1;
  while (e < 0) {
    d *= 10;
    ++e;
  }
  while (e > 0) {
    n *= 10;
    --e;
  }
  return Q(neg ? -n : n, d);
}
inline Q jq(const J &j) {
  return decimal(j.is_string() ? j.get<std::string>() : j.dump());
}
using Bounds = std::optional<std::pair<Q, Q>>;
struct Facts {
  bool finite = false, pos = false, nn = false;
  // A value is either a finite real or negative infinity. This is not a
  // branch/path set: unknown, NaN, positive infinity and undefined are excluded.
  bool finite_or_neginf = false;
  bool finite_or_negative_infinity() const { return finite || finite_or_neginf; }
};
struct Node {
  std::string op, tag;
  std::vector<Id> args;
  std::vector<Q> data;
  Facts facts;
  Bounds bounds;
  std::vector<Id> free;
};
struct Arena {
  // Immutable node-local facts, modulo 2^256. Unknown bits are not zero bits.
  struct KnownBits { Z zero=0, one=0; bool integer=false; };
  std::unordered_map<Id,KnownBits> known_bits_cache;
  // Resolved one-bit queries and OR slices. Read-only comparison probes only
  // consume this cache; ordinary multi-byte payload slicing adds no entries.
  std::map<std::tuple<Id,unsigned,unsigned>,Id> resolved_bit_slice_cache;
  std::map<std::pair<Id,Id>,Id> coordinate_witness_cache;
  std::map<std::tuple<Id,Id,bool>,bool> reduction_witness_cache;
  std::deque<Node> nodes;
  std::unordered_map<size_t, std::vector<Id>> table;
  std::map<std::string, uint64_t> stats;
  std::chrono::steady_clock::time_point deadline;
  Id zero, one;
  uint64_t ticks = 0;
  Arena(double seconds = 300)
      : deadline(std::chrono::steady_clock::now() +
                 std::chrono::milliseconds(int64_t(seconds * 1000))) {
    zero = constant(0);
    one = constant(1);
    stats["select_max_probe_temporary_nodes"] = 0;
    stats["select_max_probe_ancestor_nodes"] = 0;
  }
  void check() {
    if ((++ticks & 4095) == 0 && std::chrono::steady_clock::now() > deadline)
      throw Timeout();
  }
  static size_t hash(const std::string &op, const std::vector<Id> &as,
                     const std::vector<Q> &ds, const std::string &tag) {
    size_t h = std::hash<std::string>{}(op);
    mix(h, std::hash<std::string>{}(tag));
    for (auto x : as)
      mix(h, x);
    for (const auto &x : ds)
      mix(h, hashq(x));
    return h;
  }
  Id intern(std::string op, std::vector<Id> as = {}, std::vector<Q> ds = {},
            std::string tag = "", Facts f = {}, Bounds b = {}) {
    check();
    ++stats["intern_requests"];
    size_t h = hash(op, as, ds, tag);
    auto &bucket = table[h];
    for (Id i : bucket) {
      const auto &n = nodes[i];
      if (n.op == op && n.args == as && n.data == ds && n.tag == tag)
        return i;
    }
    if (nodes.size() >= invalid)
      throw Unsupported("arena ID capacity");
    Id id = Id(nodes.size());
    std::vector<Id> fs;
    for (auto x : as) {
      std::vector<Id> tmp;
      std::set_union(fs.begin(), fs.end(), nodes[x].free.begin(),
                     nodes[x].free.end(), std::back_inserter(tmp));
      fs.swap(tmp);
    }
    if (op == "var")
      fs = {id};
    if (op == "fold") {
      fs = nodes[as[1]].free;
      Id v = ds[0].numerator().convert_to<Id>();
      fs.erase(std::remove(fs.begin(), fs.end(), v), fs.end());
      std::vector<Id> tmp;
      std::set_union(fs.begin(), fs.end(), nodes[as[0]].free.begin(),
                     nodes[as[0]].free.end(), std::back_inserter(tmp));
      fs.swap(tmp);
    }
    nodes.push_back({std::move(op), std::move(tag), std::move(as),
                     std::move(ds), f, std::move(b), std::move(fs)});
    bucket.push_back(id);
    return id;
  }
  std::optional<Id> lookup(const std::string &o, const std::vector<Id> &as,
                           const std::vector<Q> &ds = {}) const {
    auto it = table.find(hash(o, as, ds, ""));
    if (it == table.end())
      return {};
    for (Id i : it->second) {
      auto &n = nodes[i];
      if (n.op == o && n.args == as && n.data == ds && n.tag.empty())
        return i;
    }
    return {};
  }
  bool depends(Id n, Id v) const {
    return std::binary_search(nodes[n].free.begin(), nodes[n].free.end(), v);
  }
  // One coordinate witness, not an element enumeration or a numerical-DAG
  // substitution. Invariant nodes are reused by identity. All callers use
  // this same bounded interpreter before constructing the reduction node.
  Id coordinate_at_zero(Id root,Id axis) {
    auto key=std::pair(root,axis);
    if(auto it=coordinate_witness_cache.find(key);it!=coordinate_witness_cache.end())return it->second;
    std::unordered_map<Id,Id> memo;
    std::vector<std::pair<Id,bool>> todo{{root,false}};
    size_t work=0;
    while(!todo.empty()) {
      check();auto [id,ready]=todo.back();todo.pop_back();
      if(memo.contains(id))continue;
      if(++work>512)return coordinate_witness_cache.emplace(key,invalid).first->second;
      if(id==axis){memo[id]=zero;continue;}
      if(!depends(id,axis)){memo[id]=id;continue;}
      const auto &n=nodes[id];const auto &op=n.op;
      bool supported=op=="add" || op=="add_raw" || op=="mul" ||
        op=="compare" || op=="select" || op=="max" || op=="idiv" ||
        op=="imod" || op=="word_or" || op=="word_shr" || op=="signed" || op=="extractbits";
      if(!supported || n.args.size()>32){memo[id]=invalid;continue;}
      if(!ready) {
        todo.emplace_back(id,true);
        for(Id child:n.args)if(!memo.contains(child))todo.emplace_back(child,false);
        continue;
      }
      std::vector<Id> xs;
      for(Id child:n.args)xs.push_back(memo.at(child));
      if(std::find(xs.begin(),xs.end(),invalid)!=xs.end()){memo[id]=invalid;continue;}
      Id result=invalid;
      try {
        if(op=="add" || op=="add_raw") {
          if(xs.size()==n.data.size()) {
            for(size_t i=0;i<xs.size();++i)xs[i]=mul({constant(n.data[i]),xs[i]});
            result=add(xs);
          }
        } else if(op=="mul" && n.data.size()==1) {
          xs.push_back(constant(n.data[0]));result=mul(xs);
        } else if(op=="compare" && xs.size()==2)result=compare(n.tag,xs[0],xs[1]);
        else if(op=="select" && xs.size()==3)result=select(xs[0],xs[1],xs[2],true);
        else if(op=="max")result=maximum(xs);
        else if((op=="idiv" || op=="imod") && xs.size()==1 && n.data.size()==1 &&
                n.data[0].denominator()==1 && n.data[0]>0)
          result=op=="idiv"?idiv(xs[0],n.data[0].numerator()):imod(xs[0],n.data[0].numerator());
        else if(op=="word_or" && xs.size()==2 && n.data.size()==1)
          result=word_or(xs[0],xs[1],n.data[0].numerator().convert_to<unsigned>());
        else if(op=="word_shr" && xs.size()==2 && n.data.size()==2)
          result=word_shr(xs[0],xs[1],n.data[0].numerator().convert_to<unsigned>(),n.data[1]!=0);
        else if(op=="signed" && xs.size()==1 && n.data.size()==1)
          result=signed_word(xs[0],n.data[0].numerator().convert_to<unsigned>());
        else if(op=="extractbits" && xs.size()==1 && n.data.size()==2)
          result=bit_slice(xs[0],n.data[0].numerator().convert_to<unsigned>(),n.data[1].numerator().convert_to<unsigned>());
      } catch(const Unsupported &) {result=invalid;}
      memo[id]=result;++stats["sum_witness_coordinate_nodes"];
    }
    return coordinate_witness_cache.emplace(key,memo.at(root)).first->second;
  }
  bool reduction_witness_at_zero(Id root,Id axis,const Z &count,bool positive) {
    if(count<=0 || axis==invalid)return false;
    auto key=std::tuple(root,axis,positive);
    if(auto it=reduction_witness_cache.find(key);it!=reduction_witness_cache.end())return it->second;
    Id id=root;bool result=false;
    for(size_t work=0;work<128;++work) {
      check();const auto &n=nodes[id];
      if(n.facts.finite && (!positive || n.facts.pos)){result=true;break;}
      if(positive?(!n.facts.finite || !n.facts.nn):!n.facts.finite_or_negative_infinity())break;
      if(n.op!="select" || n.args.size()!=3)break;
      Id p=coordinate_at_zero(n.args[0],axis);
      if(p!=one && p!=zero)break;
      id=n.args[p==one?1:2];
    }
    if(result)++stats[positive?"sum_positive_coordinate_witnesses":"max_finite_coordinate_witnesses"];
    return reduction_witness_cache.emplace(key,result).first->second;
  }
  bool positive_at_zero(Id body,Id axis,const Z &count) {
    return reduction_witness_at_zero(body,axis,count,true);
  }
  Facts fold_facts(Id initial,Id body,const Q &count,bool maximum,Id axis=invalid) {
    if(count==0)return nodes[initial].facts;
    if(count<0)return {};
    const auto &s=nodes[initial].facts,&b=nodes[body].facts;
    bool extended=s.finite_or_negative_infinity() && b.finite_or_negative_infinity();
    if(maximum) {
      bool finite=extended && (s.finite || b.finite);
      if(!finite && extended && count.denominator()==1)
        finite=reduction_witness_at_zero(body,axis,count.numerator(),false);
      return {finite,finite && (s.pos || b.pos),finite && (s.nn || b.nn),extended};
    }
    bool nn=s.nn && b.nn;
    bool finite=s.finite && b.finite,positive=nn && (s.pos || b.pos);
    if(finite && nn && !positive && count.denominator()==1)
      positive=positive_at_zero(body,axis,count.numerator());
    return {finite,positive,nn,extended};
  }
  Id constant(Q q) {
    return intern("const", {}, {q}, "", {true, q > 0, q >= 0}, std::pair(q, q));
  }
  std::optional<Q> value(Id n) const {
    if (nodes[n].op == "const")
      return nodes[n].data[0];
    return {};
  }
  Id var(std::string key, Q lo, Q hi) {
    // Integer index variable, used by family coordinates and Fold binders.
    // Native callers must use read/atom for symbolic real-valued inputs.
    return intern("var", {}, {lo, hi}, std::move(key), {true, lo > 0, lo >= 0},
                  std::pair(lo, hi));
  }
  Id infinity(int sign) { return intern("infinity", {}, {Q(sign)}, "", {false,false,false,sign<0}); }
  bool neginf(Id n) const {
    return nodes[n].op == "infinity" && nodes[n].data[0] == -1;
  }
  std::vector<std::pair<Id, Q>> terms(Id n) {
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
  Id add(std::vector<Id> xs) {
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
      for (auto [n, c] : terms(i)) {
        ts[n] += c;
        ++stats["add_term_visits"];
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
      auto d = lookup("idiv", x.args, x.data);
      if (d && ts.contains(*d) && ts[*d] == it->second * x.data[0]) {
        Q c = it->second;
        ts.erase(n);
        ts.erase(*d);
        for (auto [t, k] : terms(x.args[0]))
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
    Facts f{true, false, true};
    Bounds b = std::pair(Q(0), Q(0));
    for (auto &[n, c] : ts) {
      as.push_back(n);
      ds.push_back(c);
      auto &x = nodes[n];
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
    return intern("add", std::move(as), std::move(ds), "", f, b);
  }
  Id neg(Id n) { return mul({constant(-1), n}); }
  Id sub(Id a, Id b) { return add({a, neg(b)}); }
  Id mul(std::vector<Id> xs) {
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
    if (!es.empty()) {
      Id e = add(es);
      if (e != zero)
        rest.push_back(intern("exp2", {e}, {}, "", {true, true, true}));
    }
    if (!natural_es.empty()) {
      Id e = add(natural_es);
      if (e != zero)
        rest.push_back(intern("exp", {e}, {}, "", {true, true, true}));
    }
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
    return intern("mul", std::move(rest), {scalar}, "", f, b);
  }
  Id floor_value(Id n, Bounds established = {}) {
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
  Id exponential(Id n, bool base2) {
    if (n == zero)
      return one;
    if (neginf(n)) return zero;
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
    if (nodes[n].op=="add_raw" && nodes[n].args.size()==nodes[n].data.size()) {
      auto xs=nodes[n].args; auto coefficients=nodes[n].data;
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
  Id exp2(Id n) { return exponential(n, true); }
  Id exp(Id n) { return exponential(n, false); }
  static bool valid_comparison(const std::string &op) {
    return op=="eq" || op=="ne" || op=="lt" || op=="le" || op=="gt" || op=="ge";
  }
  static std::string reverse_comparison(const std::string &op) {
    if(op=="lt")return "gt";
    if(op=="le")return "ge";
    if(op=="gt")return "lt";
    if(op=="ge")return "le";
    return op;
  }
  static std::string complement_comparison(const std::string &op) {
    if(op=="eq")return "ne";
    if(op=="ne")return "eq";
    if(op=="lt")return "ge";
    if(op=="le")return "gt";
    if(op=="gt")return "le";
    return "lt";
  }
  struct SignComparison {
    Id source; unsigned bit; bool negative;
  };
  std::optional<SignComparison> sign_comparison(const std::string &op,Id a,Id b) const {
    Id id=a;auto boundary=value(b);std::string order=op;
    if(!boundary && value(a)){id=b;boundary=value(a);order=reverse_comparison(op);}
    const auto &n=nodes[id];
    if(!boundary || n.op!="signed" || n.args.size()!=1 || n.data.size()!=1 ||
       n.data[0].denominator()!=1 || n.data[0]<1 || n.data[0]>256 || !n.facts.finite)
      return {};
    const bool negative=(*boundary==0 && order=="lt") || (*boundary==-1 && order=="le");
    const bool nonnegative=(*boundary==0 && order=="ge") || (*boundary==-1 && order=="gt");
    if(!negative && !nonnegative)return {};
    return SignComparison{n.args[0],n.data[0].numerator().convert_to<unsigned>()-1,negative};
  }
  // A virtual unit-coefficient product is a bounded vector of existing factor
  // IDs, NOT a newly interned Mul. This replaces terms() for comparison queries:
  // terms() is a constructor and can intern a stripped product.
  using ComparisonAtom = std::vector<Id>;
  struct ComparisonDifference {
    std::map<ComparisonAtom,Q> coefficients;
    Q offset=0;
  };
  static constexpr size_t max_comparison_terms=32;
  std::optional<ComparisonDifference> comparison_difference(Id a,Id b) const {
    if(!nodes[a].facts.finite || !nodes[b].facts.finite)return {};
    ComparisonDifference form;size_t terms_count=0,factor_count=0;
    auto atom=[&](Id id,const Q &coefficient) {
      if(++terms_count>max_comparison_terms)return false;
      const auto &n=nodes[id];
      if(n.op=="const"){form.offset+=coefficient*n.data[0];return true;}
      ComparisonAtom factors;Q c=coefficient;
      if(n.op=="mul") {
        if(n.data.size()!=1 || n.args.size()>max_comparison_terms-factor_count)return false;
        factor_count+=n.args.size();factors=n.args;c*=n.data[0];
      } else {
        if(++factor_count>max_comparison_terms)return false;
        factors.push_back(id);
      }
      if(factors.empty())form.offset+=c;
      else form.coefficients[std::move(factors)]+=c;
      return true;
    };
    auto side=[&](Id id,const Q &sign) {
      const auto &n=nodes[id];
      if(n.op!="add")return atom(id,sign);
      if(n.args.size()>max_comparison_terms-terms_count || n.data.size()!=n.args.size())return false;
      for(size_t i=0;i<n.args.size();++i)if(!atom(n.args[i],sign*n.data[i]))return false;
      return true;
    };
    if(!side(a,Q(1)) || !side(b,Q(-1)))return {};
    for(auto it=form.coefficients.begin();it!=form.coefficients.end();)
      if(it->second==0)it=form.coefficients.erase(it);else ++it;
    return form;
  }
  struct ComparisonDescription {
    enum Kind { Raw, Affine, Constant, NotPrepared } kind=Raw;
    std::string op;
    Id lhs=invalid,rhs=invalid;
    ComparisonDifference form;
    Q threshold=0;
    bool truth=false;
  };
  struct BooleanComparisonStep {
    Id resolved=invalid;
    bool scaled=false,complemented=false;
  };
  BooleanComparisonStep normalize_boolean_comparison(std::string &op,Id &a,Id &b) const {
    BooleanComparisonStep out;
    if(op!="eq" && op!="ne")return out;
    if(value(a) && !value(b))std::swap(a,b);
    const auto &scaled=nodes[a];
    if(b==zero && scaled.op=="mul" && scaled.args.size()==1 &&
       scaled.data.size()==1 && scaled.data[0]!=0 && nodes[scaled.args[0]].op=="compare") {
      a=scaled.args[0];out.scaled=true;
    }
    const auto &inner=nodes[a];
    if(inner.op!="compare" || inner.args.size()!=2 || !inner.data.empty() ||
       !valid_comparison(inner.tag) || (b!=zero && b!=one))return out;
    if((op=="ne" && b==zero)||(op=="eq" && b==one)) {out.resolved=a;return out;}
    if(nodes[inner.args[0]].facts.finite && nodes[inner.args[1]].facts.finite) {
      op=complement_comparison(inner.tag);a=inner.args[0];b=inner.args[1];out.complemented=true;
    }
    return out;
  }
  // Both real comparison construction and Max recognition use this exact
  // normal form. Only already-known integer facts are read here. At most 32
  // stored terms/factors are inspected; no numerical ancestors are visited.
  ComparisonDescription normalize_comparison(std::string op,Id a,Id b,
      std::optional<ComparisonDifference> form) const {
    ComparisonDescription out;out.op=op;out.lhs=a;out.rhs=b;
    if(!valid_comparison(op)){out.kind=ComparisonDescription::NotPrepared;return out;}
    if(!form)return out;
    for(const auto &[factors,c]:form->coefficients) {
      (void)c;
      for(Id id:factors) {
        auto it=known_bits_cache.find(id);
        if(!nodes[id].bounds || it==known_bits_cache.end() || !it->second.integer)return out;
      }
    }
    if(form->coefficients.empty()) {
      out.kind=ComparisonDescription::Constant;
      const Q &c=form->offset;
      out.truth=op=="eq"?c==0:op=="ne"?c!=0:op=="lt"?c<0:op=="le"?c<=0:op=="gt"?c>0:c>=0;
      return out;
    }
    Z denominator=1;
    for(const auto &[_,c]:form->coefficients)
      denominator=denominator/boost::multiprecision::gcd(denominator,c.denominator())*c.denominator();
    Z divisor=0;
    for(const auto &[_,c]:form->coefficients) {
      Z v=c.numerator()*(denominator/c.denominator());
      divisor=boost::multiprecision::gcd(divisor,boost::multiprecision::abs(v));
    }
    Q scale(denominator,divisor);
    if(form->coefficients.begin()->second<0){scale=-scale;op=reverse_comparison(op);}
    Q threshold=-form->offset*scale;
    if(op=="lt"){op="le";threshold=Q(-floorq(-threshold)-1);}
    else if(op=="gt"){op="ge";threshold=Q(floorq(threshold)+1);}
    else if(op=="le")threshold=Q(floorq(threshold));
    else if(op=="ge")threshold=Q(-floorq(-threshold));
    else if(threshold.denominator()!=1) {
      out.kind=ComparisonDescription::Constant;out.truth=op=="ne";return out;
    }
    for(auto &[_,c]:form->coefficients)c*=scale;
    form->offset=0;
    out.kind=ComparisonDescription::Affine;out.op=std::move(op);
    out.form=std::move(*form);out.threshold=std::move(threshold);
    return out;
  }
  ComparisonDescription describe_comparison(std::string op,Id a,Id b) const {
    if(!valid_comparison(op)) {
      ComparisonDescription out;out.kind=ComparisonDescription::NotPrepared;return out;
    }
    // Fixed local Boolean shells only. Resolved bit queries may themselves be
    // comparisons (e.g. a bit of an all-ones parameterized right shift).
    for(unsigned shell=0;shell<4;++shell) {
      if(auto sign=sign_comparison(op,a,b)) {
        auto bits=known_bits_cache.find(sign->source);
        auto slice=resolved_bit_slice_cache.find(std::tuple(sign->source,sign->bit,1u));
        if(bits==known_bits_cache.end() || !bits->second.integer || slice==resolved_bit_slice_cache.end()) {
          ComparisonDescription out;out.kind=ComparisonDescription::NotPrepared;return out;
        }
        op=sign->negative?"ne":"eq";a=slice->second;b=zero;continue;
      }
      auto boolean=normalize_boolean_comparison(op,a,b);
      if(boolean.resolved!=invalid) {
        const auto &n=nodes[boolean.resolved];op=n.tag;a=n.args[0];b=n.args[1];continue;
      }
      if(boolean.complemented)continue;
      return normalize_comparison(op,a,b,comparison_difference(a,b));
    }
    ComparisonDescription out;out.kind=ComparisonDescription::NotPrepared;return out;
  }
  static bool same_comparison(const ComparisonDescription &a,const ComparisonDescription &b) {
    if(a.kind!=b.kind || a.kind==ComparisonDescription::NotPrepared)return false;
    if(a.kind==ComparisonDescription::Constant)return a.truth==b.truth;
    if(a.kind==ComparisonDescription::Affine)
      return a.op==b.op && a.threshold==b.threshold && a.form.coefficients==b.form.coefficients;
    return (a.op==b.op && a.lhs==b.lhs && a.rhs==b.rhs) ||
           (a.op==reverse_comparison(b.op) && a.lhs==b.rhs && a.rhs==b.lhs);
  }
  bool select_is_max(Id p,Id a,Id b) const {
    const auto &n=nodes[p];
    if(n.op!="compare" || n.args.size()!=2 || !n.data.empty() || !valid_comparison(n.tag) ||
       !nodes[a].facts.finite || !nodes[b].facts.finite)return false;
    if(((n.tag=="gt" || n.tag=="ge") && n.args[0]==a && n.args[1]==b) ||
       ((n.tag=="lt" || n.tag=="le") && n.args[0]==b && n.args[1]==a))return true;
    // Ordinary equality-controlled decoders are not order predicates. Retain
    // the signed-word boundary case, whose real comparison can become eq/ne.
    if((n.tag=="eq" || n.tag=="ne") &&
       !sign_comparison("gt",a,b) && !sign_comparison("ge",a,b))return false;
    auto predicate=describe_comparison(n.tag,n.args[0],n.args[1]);
    return same_comparison(predicate,describe_comparison("gt",a,b)) ||
           same_comparison(predicate,describe_comparison("ge",a,b));
  }
  Id compare(std::string p, Id a, Id b) {
    if (!valid_comparison(p))
      throw Unsupported("comparison " + p);
    auto av = value(a), bv = value(b);
    std::optional<int> c;
    if(a==b && nodes[a].facts.finite)c=0;
    else if (av && bv)
      c = *av < *bv ? -1 : *av > *bv ? 1 : 0;
    else if (nodes[a].op == "infinity" && nodes[b].facts.finite)
      c = nodes[a].data[0].numerator().convert_to<int>();
    else if (nodes[b].op == "infinity" && nodes[a].facts.finite)
      c = -nodes[b].data[0].numerator().convert_to<int>();
    if (!c && nodes[a].facts.finite && nodes[b].facts.finite && nodes[a].bounds && nodes[b].bounds) {
      const auto &ab=*nodes[a].bounds, &bb=*nodes[b].bounds;
      if (ab.second < bb.first) c=-1;
      else if (ab.first > bb.second) c=1;
      else if (ab.first==ab.second && ab==bb) c=0;
      else if (p=="lt" && ab.first>=bb.second) return zero;
      else if (p=="ge" && ab.first>=bb.second) return one;
      else if (p=="gt" && ab.second<=bb.first) return zero;
      else if (p=="le" && ab.second<=bb.first) return one;
    }
    if (c) {
      bool r = p == "eq"   ? *c == 0
               : p == "ne" ? *c != 0
               : p == "lt" ? *c < 0
               : p == "le" ? *c <= 0
               : p == "gt" ? *c > 0
                           : *c >= 0;
      return constant(int(r));
    }
    if (b == zero && nodes[a].facts.pos &&
        (p == "ne" || p == "gt" || p == "ge"))
      return one;
    // Predicate constructors are shared by import, substitution and reference
    // instantiation. Inspect only this comparison's immediate Boolean shell;
    // never distribute Selects or search numerical ancestors.
    // A signed word crossing its sign boundary is not an unknown real value.
    // Sign-only comparisons consume the same parametric bit query used by
    // extraction. Preserve the declared width; do not split its member domain.
    if(auto query=sign_comparison(p,a,b);query && known_bits(query->source).integer) {
      Id sign=bit_slice(query->source,query->bit,1);
      ++stats["signed_word_sign_queries"];
      return compare(query->negative?"ne":"eq",sign,zero);
    }
    auto boolean=normalize_boolean_comparison(p,a,b);
    if(boolean.scaled)++stats["predicate_scaled_boolean_zero"];
    if(boolean.complemented)++stats["predicate_boolean_complements"];
    if(boolean.resolved!=invalid)return boolean.resolved;

    // Actual construction may prepare integer facts. The shared normalizer
    // and its read-only consumers never do so, and never call terms().
    Id original_a=a,original_b=b;std::string original_p=p;
    auto form=comparison_difference(a,b);
    if(form) {
      bool ready=true;
      for(const auto &[factors,c]:form->coefficients) {
        (void)c;
        for(Id id:factors)if(!nodes[id].bounds || !known_bits(id).integer){ready=false;break;}
        if(!ready)break;
      }
      auto description=normalize_comparison(p,a,b,std::move(form));
      if(description.kind==ComparisonDescription::Constant)return description.truth?one:zero;
      if(description.kind==ComparisonDescription::Affine) {
        std::vector<Id> parts;
        for(const auto &[factors,c]:description.form.coefficients) {
          std::vector<Id> xs=factors;
          if(c!=1)xs.push_back(constant(c));
          parts.push_back(xs.size()==1?xs[0]:mul(std::move(xs)));
        }
        a=add(std::move(parts));b=constant(description.threshold);p=description.op;
        if(a!=original_a || b!=original_b || p!=original_p) {
          ++stats["predicate_integer_affine_normalizations"];
          return compare(p,a,b);
        }
      }
    }
    return intern("compare", {a, b}, {}, p, {true, false, true},
                  std::pair(Q(0), Q(1)));
  }
  Id select(Id p, Id a, Id b, bool infer_bounds = false) {
    if (p == one)
      return a;
    if (p == zero)
      return b;
    if (a == b)
      return a;
    // A complementary condition with swapped arms denotes the same Select,
    // including when an arm is -infinity. Only the comparison operands must
    // be proved finite/ordered; requiring finite arms would lose masked Max.
    // Reuse compare() for integer strictness and affine normalization. The
    // preferred polarities (eq/le/lt) never switch back on the next rebuild.
    const auto &predicate = nodes[p];
    if (predicate.op == "compare" && predicate.args.size() == 2 &&
        predicate.data.empty() && nodes[predicate.args[0]].facts.finite &&
        nodes[predicate.args[1]].facts.finite &&
        (predicate.tag == "ne" || predicate.tag == "ge" || predicate.tag == "gt")) {
      const std::string opposite = predicate.tag == "ne" ? "eq"
                                   : predicate.tag == "ge" ? "lt" : "le";
      p = compare(opposite, predicate.args[0], predicate.args[1]);
      std::swap(a, b);
      ++stats["select_complementary_polarities"];
      if (p == one) return a;
      if (p == zero) return b;
    }
    // Recognition is a const query over bounded stored shells and prepared
    // facts. In particular it cannot call compare/add/mul/bit_slice/known_bits.
    ++stats["select_max_probe_calls"];
    const size_t probe_nodes=nodes.size(),probe_facts=known_bits_cache.size();
    const bool is_max=select_is_max(p,a,b);
    if(nodes.size()!=probe_nodes)stats["select_max_probe_temporary_nodes"]+=nodes.size()-probe_nodes;
    if(known_bits_cache.size()!=probe_facts)stats["select_max_probe_ancestor_nodes"]+=known_bits_cache.size()-probe_facts;
    if(is_max) {
      ++stats["select_max_probe_hits"];
      return maximum({a,b});
    }
    const auto &condition=nodes[p];
    // Refine only the selected occurrence of the compared value. No inherited
    // path context is added to the Arena, and no fact escapes to that value.
    Facts af=nodes[a].facts,bf=nodes[b].facts;
    if(condition.op=="compare" && condition.args.size()==2 &&
       (condition.tag=="eq" || condition.tag=="ne")) {
      Id compared=invalid;
      if(neginf(condition.args[0]))compared=condition.args[1];
      else if(neginf(condition.args[1]))compared=condition.args[0];
      if(compared!=invalid && nodes[compared].facts.finite_or_negative_infinity()) {
        if(condition.tag=="ne" && a==compared)af.finite=true;
        if(condition.tag=="eq" && b==compared)bf.finite=true;
      }
    }
    const Id exponential_arm = a == one ? b : b == one ? a : invalid;
    if (exponential_arm != invalid && nodes[exponential_arm].op == "exp2" &&
        nodes[nodes[exponential_arm].args[0]].facts.finite) {
      // Keep the identity independent of predicate polarity. Canonicalizing
      // a complementary Select must not disable the existing rescale rule.
      const Id one_predicate = a == one ? p : compare("eq", p, zero);
      Id d = nodes[exponential_arm].args[0];
      const auto &n = nodes[d];
      if (n.op == "add" && n.args.size() == 2 && n.data[0] == -n.data[1]) {
        Id x = n.args[0], y = n.args[1];
        Q c = n.data[0];
        if (c < 0) {
          std::swap(x, y);
          c = -c;
        }
        if (nodes[x].facts.finite && nodes[y].facts.finite) {
          ++stats["select_rescaling_identity"];
          return exp2(mul({constant(c), sub(x, select(one_predicate, x, y))}));
        }
      }
      return exp2(select(one_predicate, zero, d));
    }
    Bounds bounds;
    // Demand-driven: propagating every floating Select interval makes all
    // downstream add/mul rebuilds perform otherwise unused rational arithmetic.
    if (infer_bounds && nodes[a].bounds && nodes[b].bounds)
      bounds = std::pair(std::min(nodes[a].bounds->first, nodes[b].bounds->first),
                         std::max(nodes[a].bounds->second, nodes[b].bounds->second));
    return intern("select", {p, a, b}, {}, "",
                  {af.finite && bf.finite, af.pos && bf.pos, af.nn && bf.nn,
                   af.finite_or_negative_infinity() && bf.finite_or_negative_infinity()}, bounds);
  }
  Id clamp(Id x, Q lo, Q hi) {
    if (lo > hi) throw Unsupported("invalid clamp interval");
    if (auto v=value(x)) return constant(std::min(hi,std::max(lo,*v)));
    if (lo == hi && nodes[x].facts.finite) return constant(lo);
    if (auto b=nodes[x].bounds; b && nodes[x].facts.finite) {
      if(b->second <= lo) return constant(lo);
      if(b->first >= hi) return constant(hi);
      if(b->first >= lo && b->second <= hi) return x;
    }
    return intern("clamp",{x},{lo,hi},"",{nodes[x].facts.finite,lo>0,lo>=0},std::pair(lo,hi));
  }
  Id maximum(std::vector<Id> xs) {
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
  // Integer operands are decoded at the import boundary. This operation uses
  // unbounded nonnegative integer XOR; no logical-domain enumeration.
  Id bitxor(Id x, Id y) {
    auto xb = nodes[x].bounds, yb = nodes[y].bounds;
    if (!xb || !yb || xb->first < 0 || yb->first < 0)
      throw Unsupported("XOR requires nonnegative integer operands");
    auto xv = value(x), yv = value(y);
    if ((xv && xv->denominator() != 1) || (yv && yv->denominator() != 1))
      throw Unsupported("XOR requires integer constants");
    if (xv && yv)
      return constant(Q(Z(xv->numerator() ^ yv->numerator())));
    if (x == y)
      return zero;
    if (x == zero)
      return y;
    if (y == zero)
      return x;
    if (xv) {
      std::swap(x, y);
      std::swap(xv, yv);
      std::swap(xb, yb);
    }
    if (yv) {
      const auto &n = nodes[x];
      if (n.op == "xor" && n.args[1] == y)
        return n.args[0];
      Z mask = yv->numerator();
      if (mask > 0 && (mask & (mask - 1)) == 0 && xb->second < Q(mask * 2)) {
        // Toggle one bit, valid only over the proved two-half interval.
        return sub(add({x, y}), mul({constant(Q(mask * 2)), idiv(x, mask)}));
      }
    }
    if (!yv && y < x)
      std::swap(x, y);
    Z hi = std::max(floorq(xb->second), floorq(yb->second));
    unsigned bits = hi == 0 ? 0 : boost::multiprecision::msb(hi) + 1;
    return intern("xor", {x, y}, {}, "", {true, false, true},
                  std::pair(Q(0), Q(pow2(bits) - 1)));
  }
  Id absolute(Id x) {
    const auto &n = nodes[x];
    if (auto v = value(x))
      return constant(*v < 0 ? -*v : *v);
    if (n.op == "abs")
      return x;
    if (n.op == "infinity")
      return infinity(1);
    if (n.facts.finite && n.facts.nn)
      return x;
    if (n.bounds && n.bounds->second <= 0)
      return neg(x);
    Bounds b;
    if (n.bounds)
      b = std::pair(Q(0), std::max(-n.bounds->first, n.bounds->second));
    bool finite = n.facts.finite;
    return intern("abs", {x}, {}, "", {finite, finite && n.facts.pos, finite},
                  b);
  }
  Id logarithm(Id x, bool base2 = false) {
    if (x == one) return zero;
    // Preserve a partial real function when positivity is not established.
    bool finite = nodes[x].facts.finite && nodes[x].facts.pos;
    return intern(base2 ? "log2" : "log", {x}, {}, "", {finite, false, false});
  }
  Id power(Id base, Id exponent) {
    if(nodes[base].facts.finite) {
      if(exponent==one) return base;
      if(auto v=value(exponent);v && *v==2) return mul({base,base});
    }
    // Positive finite base and finite exponent suffice for a real result.
    // No floating approximation or power expansion is needed for import.
    bool positive = nodes[base].facts.finite && nodes[base].facts.pos &&
                    nodes[exponent].facts.finite;
    if (positive && exponent == zero) return one;
    if (positive && exponent == one) return base;
    if (positive && base == one) return one;
    // A positive-base real power and the compiler's lg2/mul/ex2 lowering
    // share an exact symbolic form. Storage constants retain their exact
    // imported values; no numerical log approximation is performed here.
    if (positive)
      return exp2(mul({logarithm(base, true), exponent}));
    return intern("power", {base, exponent}, {}, "", {positive, positive, positive});
  }
  bool nonzero(Id n) {
    auto v = value(n);
    return (v && *v != 0) || (nodes[n].facts.finite && nodes[n].facts.pos);
  }
  Id recip(Id n) {
    auto v = value(n);
    if (v && *v != 0)
      return constant(Q(1) / *v);
    bool p = nodes[n].facts.finite && nodes[n].facts.pos;
    const auto &node = nodes[n];
    if (node.op == "sqrt" && p) {
      Id x=node.args[0];
      return intern("reciprocalsqrt",{x},{},"",{true,true,true});
    }
    if ((node.op == "exp2" || node.op == "exp") && p) {
      bool base2 = node.op == "exp2";
      Id exponent = node.args[0];
      return exponential(neg(exponent), base2);
    }
    // Only expose exponential factors when every denominator factor is known
    // finite and nonzero. Never cancel a possibly-zero or nonfinite denominator.
    if (node.op == "mul" && node.data[0] != 0 &&
        std::all_of(node.args.begin(), node.args.end(),
                    [&](Id x) { return nonzero(x); })) {
      std::vector<Id> other, powers;
      Q scalar = node.data[0];
      for (Id x : node.args) {
        const auto &factor = nodes[x];
        if ((factor.op == "exp2" || factor.op == "exp") &&
            nodes[factor.args[0]].facts.finite)
          powers.push_back(x);
        else other.push_back(x);
      }
      if (!powers.empty()) {
        other.push_back(constant(scalar));
        Id remaining = mul(other);
        std::vector<Id> inverse{recip(remaining)};
        for (Id x : powers) inverse.push_back(recip(x));
        ++stats["reciprocal_exponential_factors"];
        return mul(inverse);
      }
    }
    return intern("recip", {n}, {}, "", {p, p, p});
  }
  using Factors = std::pair<std::map<Id, int>, Q>;
  Factors factors(Id n) {
    const auto &x = nodes[n];
    if (x.op == "const")
      return {{}, x.data[0]};
    std::map<Id, int> fs;
    if (x.op == "mul") {
      for (Id a : x.args)
        ++fs[a];
      return {fs, x.data[0]};
    }
    return {{{n, 1}}, Q(1)};
  }
  static std::map<Id, int> common(const std::map<Id, int> &a,
                                  const std::map<Id, int> &b) {
    std::map<Id, int> r;
    for (auto [x, k] : a) {
      auto it = b.find(x);
      if (it != b.end())
        r[x] = std::min(k, it->second);
    }
    return r;
  }
  Id product(Factors f, const std::map<Id, int> &remove = {}) {
    std::vector<Id> xs{constant(f.second)};
    for (auto [n, k] : f.first) {
      auto it = remove.find(n);
      if (it != remove.end())
        k -= it->second;
      for (int j = 0; j < k; ++j)
        xs.push_back(n);
    }
    return mul(xs);
  }
  Id divide(Id x, Id y) {
    if (x == y && nonzero(y))
      return one;
    auto af = factors(x), bf = factors(y);
    auto c = common(af.first, bf.first);
    for (auto it = c.begin(); it != c.end();)
      if (!nonzero(it->first))
        it = c.erase(it);
      else
        ++it;
    if (!c.empty()) {
      Id a = product(af, c), b = product(bf, c);
      if (nonzero(b)) {
        ++stats["cancel_nonzero_factor"];
        return mul({a, recip(b)});
      }
    }
    return mul({x, recip(y)});
  }
  Id factor_sum(Id n) {
    const auto &x = nodes[n];
    if (x.op != "add")
      return n;
    std::vector<Factors> ps;
    for (Id a : x.args)
      ps.push_back(factors(a));
    auto c = ps[0].first;
    for (size_t i = 1; i < ps.size(); ++i)
      c = common(c, ps[i].first);
    if (c.empty())
      return n;
    std::vector<Id> rs;
    for (size_t i = 0; i < ps.size(); ++i) {
      ps[i].second *= x.data[i];
      rs.push_back(product(ps[i], c));
    }
    return mul({product({c, Q(1)}), add(rs)});
  }
  KnownBits known_bits(Id root) {
    const Z mask=pow2(256)-1;
    auto exact=[&](Z v) {v=modz(v,pow2(256));return KnownBits{mask^v,v,true};};
    auto scale=[&](KnownBits k,const Q& c) {
      if(!k.integer || c.denominator()!=1)return KnownBits{};
      Z v=modz(c.numerator(),pow2(256));
      if(v==0)return exact(0);
      if((k.zero|k.one)==mask)return exact(k.one*v);
      if((v&(v-1))==0) {
        unsigned s=boost::multiprecision::lsb(v);
        return KnownBits{Z(((k.zero<<s)|(pow2(s)-1))&mask),Z((k.one<<s)&mask),true};
      }
      unsigned s=boost::multiprecision::lsb(v);
      return KnownBits{pow2(s)-1,0,true};
    };
    auto sum=[&](KnownBits l,KnownBits r) {
      if(!l.integer || !r.integer)return KnownBits{};
      KnownBits out{0,0,true};unsigned carries=1;
      for(unsigned b=0;b<256;++b) {
        Z bit=pow2(b);
        unsigned ls=(l.zero&bit)!=0?1:(l.one&bit)!=0?2:3;
        unsigned rs=(r.zero&bit)!=0?1:(r.one&bit)!=0?2:3;
        unsigned values=0,next=0;
        for(unsigned x=0;x<2;++x)for(unsigned y=0;y<2;++y)for(unsigned c=0;c<2;++c)
          if((ls&(1u<<x)) && (rs&(1u<<y)) && (carries&(1u<<c))) {
            unsigned v=x+y+c;values|=1u<<(v&1);next|=1u<<(v>>1);
          }
        if(values==1)out.zero|=bit;else if(values==2)out.one|=bit;
        carries=next;
      }
      return out;
    };
    auto descends=[](const Node& n) {
      return n.op=="add" || n.op=="mul" || n.op=="select" || n.op=="word_or" ||
             n.op=="word" || n.op=="signed" || n.op=="extractbits" ||
             n.op=="constant_bit_mask" || n.op=="idiv" || n.op=="imod" || n.op=="word_shr" ||
             n.op=="max" || n.op=="clamp";
    };
    std::vector<std::pair<Id,bool>> todo{{root,false}};
    while(!todo.empty()) {
      auto [id,ready]=todo.back();todo.pop_back();check();
      if(known_bits_cache.contains(id)){++stats["known_bits_cache_hits"];continue;}
      const auto& n=nodes[id];
      if(!ready && descends(n)) {
        todo.emplace_back(id,true);
        for(Id child:n.args)if(!known_bits_cache.contains(child))todo.emplace_back(child,false);
        continue;
      }
      auto child=[&](size_t j){return known_bits_cache.at(n.args.at(j));};
      KnownBits out;
      if(n.op=="const" && n.data[0].denominator()==1)out=exact(n.data[0].numerator());
      else if(n.op=="var" || n.op=="readbits" || n.op=="packedbits" || n.op=="floatbits" ||
              n.op=="compare" || n.op=="floor")out.integer=true;
      else if(n.op=="read") {
        auto split=n.tag.rfind('\n');std::string type=split==std::string::npos?"":n.tag.substr(split+1);
        out.integer=type.size()>1 && (type[0]=='u'||type[0]=='s'||type[0]=='b') &&
          std::all_of(type.begin()+1,type.end(),[](char c){return c>='0'&&c<='9';});
      }
      else if(n.op=="add") {
        out=exact(0);
        for(size_t j=0;j<n.args.size();++j)out=sum(out,scale(child(j),n.data[j]));
      } else if(n.op=="mul") {
        out=exact(1);
        for(size_t j=0;j<n.args.size();++j) {
          auto k=child(j);
          if(!k.integer || !out.integer){out={};break;}
          if((out.zero|out.one)==mask)out=scale(k,Q(out.one));
          else if((k.zero|k.one)==mask)out=scale(out,Q(k.one));
          else out={0,0,true};
        }
        out=scale(out,n.data[0]);
      } else if(n.op=="select") {
        auto l=child(1),r=child(2);out={Z(l.zero&r.zero),Z(l.one&r.one),l.integer&&r.integer};
      } else if(n.op=="word_or") {
        auto l=child(0),r=child(1);out={Z(l.zero&r.zero),Z(l.one|r.one),true};
      } else if(n.op=="word" || n.op=="signed" || n.op=="extractbits" || n.op=="constant_bit_mask") {
        out=child(0);out.integer=true;
        if(n.op=="constant_bit_mask") {
          Z m=modz(n.data[0].numerator(),pow2(256));out.zero|=mask^m;out.one&=m;
        } else {
          unsigned lo=n.op=="extractbits"?n.data[0].numerator().convert_to<unsigned>():0;
          unsigned w=n.data[n.op=="extractbits"?1:0].numerator().convert_to<unsigned>();
          if(w>0 && w<=256 && lo<=256-w) {
            Z m=pow2(w)-1;out.zero=(out.zero>>lo)&m;out.one=(out.one>>lo)&m;
            if(n.op!="signed")out.zero|=mask^m;
            else if((out.zero&pow2(w-1))!=0)out.zero|=mask^m;
            else if((out.one&pow2(w-1))!=0)out.one|=mask^m;
          } else out={0,0,true};
        }
      } else if(n.op=="idiv" || n.op=="imod") {
        out.integer=n.op=="idiv" || child(0).integer;
        Z d=n.data[0].numerator();
        if(out.integer && child(0).integer && d>0 && (d&(d-1))==0) {
          unsigned s=boost::multiprecision::msb(d);
          if(s<256) {
            if(n.op=="imod") {Z m=pow2(s)-1;out={Z((child(0).zero&m)|(mask^m)),Z(child(0).one&m),true};}
            else out={Z(child(0).zero>>s),Z(child(0).one>>s),true};
          }
        }
      } else if(n.op=="word_shr")out.integer=true;
      else if(n.op=="max" || n.op=="clamp") {
        out.integer=std::all_of(n.args.begin(),n.args.end(),[&](Id x){return known_bits_cache.at(x).integer;});
        for(const auto& d:n.data)out.integer &= d.denominator()==1;
      }
      // Intersect with the shared high prefix of a proven integer interval.
      if(out.integer && n.bounds) {
        Z lo=floorq(n.bounds->first),hi=floorq(n.bounds->second);
        if(lo<=hi && floorz(lo,pow2(256))==floorz(hi,pow2(256))) {
          lo=modz(lo,pow2(256));hi=modz(hi,pow2(256));Z differing=lo^hi;
          Z prefix=differing==0?mask:Z(mask^(pow2(boost::multiprecision::msb(differing)+1)-1));
          out.one|=lo&prefix;out.zero|=(mask^lo)&prefix;
        }
      }
      if(!out.integer)out={};
      known_bits_cache.emplace(id,std::move(out));++stats["known_bits_nodes"];
      stats["known_bits_edges"]+=descends(n)?n.args.size():0;
    }
    return known_bits_cache.at(root);
  }
  Id word_or(Id lhs,Id rhs,unsigned width) {
    if(width==0 || width>256)throw Unsupported("word OR width");
    Z mask=pow2(width)-1;
    // Typed integer inputs; substitution may enlarge their coordinate range.
    for(Id* operand:{&lhs,&rhs}) {
      auto b=nodes[*operand].bounds;
      if(!b || b->first<0 || b->second>Q(mask))*operand=imod(*operand,pow2(width));
    }
    if(lhs==rhs || rhs==zero)return lhs;
    if(lhs==zero)return rhs;
    auto l=known_bits(lhs),r=known_bits(rhs);
    Z ones=(l.one|r.one)&mask, zeros=(l.zero&r.zero)&mask;
    if((ones|zeros)==mask)return constant(Q(ones));
    if(((mask^l.zero)&(mask^r.zero)&mask)==0) {
      ++stats["disjoint_integer_or_fields"];return add({lhs,rhs});
    }
    if(rhs<lhs)std::swap(lhs,rhs);
    return intern("word_or",{lhs,rhs},{Q(width)},"",{true,false,true},std::pair(Q(ones),Q(mask^zeros)));
  }
  // PTX shr: the u32 shift count is clamped, never taken modulo width.
  // The source is already interpreted in the declared signed/unsigned type.
  Id word_shr(Id source, Id shift, unsigned width, bool sign) {
    if(width==0 || width>256) throw Unsupported("right shift width");
    auto sb=nodes[shift].bounds;
    if(!sb || sb->first<0) throw Unsupported("right shift requires unsigned count");
    // Compose only the immediate, same-width unsigned operation. The input
    // word and both counts must still have their typed integer semantics.
    // The synthesized count is an ideal integer sum: wrapping it to u32 would
    // turn e.g. UINT32_MAX + 1 into zero, contrary to two clamped PTX shifts.
    const auto &inner=nodes[source];
    if(!sign && inner.op=="word_shr" && inner.args.size()==2 && inner.data.size()==2 &&
       inner.data[0]==Q(width) && inner.data[1]==0) {
      const auto args=inner.args;
      const auto original_bounds=nodes[args[0]].bounds, first_bounds=nodes[args[1]].bounds;
      if(original_bounds && original_bounds->first>=0 && original_bounds->second<Q(pow2(width)) &&
         first_bounds && first_bounds->first>=0 && nodes[args[0]].facts.finite &&
         nodes[args[1]].facts.finite && nodes[shift].facts.finite &&
         known_bits(args[0]).integer && known_bits(args[1]).integer && known_bits(shift).integer) {
        source=args[0]; shift=add({args[1],shift}); sb=nodes[shift].bounds;
        ++stats["unsigned_nested_shift_compositions"];
      }
    }
    if(auto s=value(shift)) {
      if(s->denominator()!=1) throw Unsupported("noninteger right shift count");
      unsigned amount=*s>=Q(width)?width:s->numerator().convert_to<unsigned>();
      if(auto v=value(source);v && v->denominator()==1)
        return constant(Q(floorz(v->numerator(),pow2(amount))));
      if(!sign && nodes[source].op=="readbits")
        return amount>=width?zero:bit_slice(source,amount,width-amount);
      return idiv(source,pow2(amount));
    }
    if(source==zero)return zero;
    if(sb->first>=Q(width))return sign?idiv(source,pow2(width)):zero;
    Q type_lo=sign?Q(-pow2(width-1)):Q(0),type_hi=sign?Q(pow2(width-1)-1):Q(pow2(width)-1);
    Bounds result=std::pair(type_lo,type_hi);
    auto source_bounds=nodes[source].bounds;
    if(source_bounds && nodes[source].facts.finite && source_bounds->first>=type_lo && source_bounds->second<=type_hi) {
      Z lo=-floorq(-source_bounds->first),hi=floorq(source_bounds->second);
      Z first=-floorq(-sb->first),last=floorq(sb->second);
      if(lo<=hi && first<=last) {
        unsigned s0=first>=Z(width)?width:first.convert_to<unsigned>();
        unsigned s1=last>=Z(width)?width:last.convert_to<unsigned>();
        Z a=floorz(lo,pow2(s0)),b=floorz(lo,pow2(s1));
        Z c=floorz(hi,pow2(s0)),d=floorz(hi,pow2(s1));
        result=std::pair(Q(std::min(a,b)),Q(std::max(c,d)));
        ++stats["word_shift_range_transfers"];
      }
    }
    return intern("word_shr",{source,shift},{Q(width),Q(sign)},"",
                  {nodes[source].facts.finite,result->first>0,result->first>=0},result);
  }
  // Shared signed reinterpretation for import and all reconstruction paths.
  // Operands are integer values; modulo width is retained unless range proves
  // it irrelevant. No recursive polynomial/modulo expansion is necessary.
  Id signed_word(Id source,unsigned width) {
    if(width==0 || width>256)throw Unsupported("signed word width");
    Z modulus=pow2(width),half=pow2(width-1);
    if(auto v=value(source);v && v->denominator()==1) {
      Z bits=modz(v->numerator(),modulus);
      return constant(Q(bits<half?bits:Z(bits-modulus)));
    }
    auto b=nodes[source].bounds;
    if(nodes[source].facts.finite && b && b->first>=Q(-half) && b->second<Q(half))return source;
    Id bits=source;
    if(!b || b->first<0 || b->second>=Q(modulus)) {
      bits=intern("word",{source},{Q(width)},"",{nodes[source].facts.finite,false,true},
                  std::pair(Q(0),Q(modulus-1)));
      b=nodes[bits].bounds;
    }
    if(nodes[bits].facts.finite && b && b->second<Q(half))return bits;
    if(nodes[bits].facts.finite && b && b->first>=Q(half))return sub(bits,constant(Q(modulus)));
    return intern("signed",{bits},{Q(width)},"",{nodes[bits].facts.finite,false,false},
                  std::pair(Q(-half),Q(half-1)));
  }
  Id bit_slice(Id source,unsigned lo,unsigned count) {
    // Every actual one-bit producer, including direct PTX extraction, prepares
    // the same result consumed by signed-comparison probes. Key before local
    // slice composition. Do not memoize every multi-byte FP decoder slice.
    if(count!=1)return compute_bit_slice(source,lo,count);
    auto key=std::tuple(source,lo,count);
    if(auto it=resolved_bit_slice_cache.find(key);it!=resolved_bit_slice_cache.end())return it->second;
    Id result=compute_bit_slice(source,lo,count);
    // Local composition can have computed only the child's facts. Prepare
    // the original query source here, never later in a speculative Max probe.
    if(!known_bits_cache.contains(source))known_bits(source);
    resolved_bit_slice_cache.emplace(key,result);
    return result;
  }
  Id compute_bit_slice(Id source,unsigned lo,unsigned count) {
    if(count==0)return zero;
    if(count>256)throw Unsupported("bit slice width");
    // Preserve truncation when composing slices. This is one local shell,
    // not an ancestor search. A slice past an unsigned field reads zero; a
    // partially overlapping slice retains only the remaining low bits.
    const auto &original=nodes[source];
    bool slice=original.op=="extractbits" && original.args.size()==1 && original.data.size()==2;
    bool truncation=original.op=="word" && original.args.size()==1 && original.data.size()==1;
    if((slice || truncation) && original.facts.finite) {
      Q start=slice?original.data[0]:Q(0), field_width=original.data[slice?1:0];
      if(start.denominator()==1 && start>=0 && start<=Q(UINT_MAX) &&
         field_width.denominator()==1 && field_width>=0 && field_width<=256) {
        unsigned w=field_width.numerator().convert_to<unsigned>();
        if(lo>=w){++stats["bit_slice_truncation_zeros"];return zero;}
        const uint64_t combined=uint64_t(lo)+start.numerator().convert_to<unsigned>();
        if(combined<=UINT_MAX) {
          count=std::min(count,w-lo);lo=unsigned(combined);source=original.args[0];
          ++stats["bit_slice_local_compositions"];
        }
      }
    }
    if(auto v=value(source);v && v->denominator()==1) {
      const Z integer=v->numerator(), magnitude=integer<0?-integer:integer;
      // Avoid allocating 2^lo for a large offset once its result is known.
      if(magnitude==0 || (lo>0 && Z(boost::multiprecision::msb(magnitude))<Z(lo)))
        return integer<0?constant(Q(pow2(count)-1)):zero;
      return constant(Q(modz(floorz(integer,pow2(lo)),pow2(count))));
    }
    const auto source_bounds=nodes[source].bounds;
    const auto bits=known_bits(source);
    if(bits.integer && nodes[source].facts.finite && source_bounds) {
      // All supported word payloads fit 256 bits. Compare safely without
      // unsigned lo+count arithmetic, or a huge allocation for a large lo.
      unsigned cutoff=std::min(lo,256u);
      if(source_bounds->first>=0 && source_bounds->second<Q(pow2(cutoff))) {
        ++stats["bit_slice_range_zeros"];return zero;
      }
      if(lo>=256 && source_bounds->first>=Q(-pow2(256)) && source_bounds->second<0)
        return constant(Q(pow2(count)-1));
    }
    if(lo<256 && count<=256-lo) {
      const Z mask=pow2(count)-1;
      if(bits.integer && (((bits.zero|bits.one)>>lo)&mask)==mask) {
        ++stats["bit_slice_known_values"];
        return constant(Q((bits.one>>lo)&mask));
      }
    }
    if(nodes[source].op=="word_or") {
      // Slice the stored OR DAG once per requested window, without recursive
      // descent or enumerating the member domain.
      std::vector<std::pair<Id,bool>> todo{{source,false}};
      while(!todo.empty()) {
        auto [id,ready]=todo.back();todo.pop_back();check();
        auto key=std::tuple(id,lo,count);if(resolved_bit_slice_cache.contains(key))continue;
        const auto& n=nodes[id];
        if(n.op!="word_or") {resolved_bit_slice_cache.emplace(key,bit_slice(id,lo,count));continue;}
        if(!ready) {todo.emplace_back(id,true);for(Id arg:n.args)todo.emplace_back(arg,false);continue;}
        Id l=resolved_bit_slice_cache.at(std::tuple(n.args[0],lo,count)),r=resolved_bit_slice_cache.at(std::tuple(n.args[1],lo,count));
        resolved_bit_slice_cache.emplace(key,word_or(l,r,count));++stats["word_or_slice_nodes"];
      }
      return resolved_bit_slice_cache.at(std::tuple(source,lo,count));
    }
    if(nodes[source].op=="word_shr") {
      // Copy scalar fields before interning; Arena storage can reallocate.
      auto args=nodes[source].args;auto data=nodes[source].data;
      unsigned width=data[0].numerator().convert_to<unsigned>();
      auto v=value(args[0]);
      if(data[1]==0 && v && *v==Q(pow2(width)-1)) {
        std::vector<Id> bits;
        for(unsigned b=0;lo<width && b<count && b<width-lo;++b)
          bits.push_back(mul({constant(Q(pow2(b))),compare("lt",args[1],constant(Q(width-lo-b)))}));
        return add(bits);
      }
    }
    const auto &n=nodes[source];
    if(n.op=="readbits" && lo%8==0 && count%8==0 && count>0 && Q(lo)+Q(count)<=n.data[0]) {
      std::string tag=n.tag;Id address=n.args[0];
      return intern("readbits",{add({address,constant(Q(lo/8))})},{Q(count)},tag,
                    {true,false,true},std::pair(Q(0),Q(pow2(count)-1)));
    }
    return intern("extractbits",{source},{Q(lo),Q(count)},"",{true,false,true},
                  std::pair(Q(0),Q(pow2(count)-1)));
  }
  Id idiv(Id n, Z d) {
    if (d <= 0)
      throw Unsupported("index divisor must be positive");
    if (d == 1)
      return n;
    if((nodes[n].op=="word_or" || (nodes[n].op=="word_shr" && nodes[n].data[1]==0)) && (d&(d-1))==0) {
      unsigned lo=boost::multiprecision::msb(d);
      unsigned width=nodes[n].data[0].numerator().convert_to<unsigned>();
      return lo>=width?zero:bit_slice(n,lo,width-lo);
    }
    if(nodes[n].op=="idiv") return idiv(nodes[n].args[0],d*nodes[n].data[0].numerator());
    if(nodes[n].op=="readbits" && (d & (d-1))==0) {
      unsigned shift=boost::multiprecision::msb(d);
      unsigned bits=nodes[n].data[0].numerator().convert_to<unsigned>();
      if(shift>=bits) return zero;
      if(shift%8==0 && bits%8==0) return bit_slice(n,shift,bits-shift);
    }
    auto ts = terms(n);
    Z g = d;
    for (auto &[_, c] : ts) {
      if (c.denominator() != 1) {
        g = 1;
        break;
      }
      g = boost::multiprecision::gcd(g,
                                     boost::multiprecision::abs(c.numerator()));
    }
    if (g > 1) {
      std::vector<Id> rs;
      for (auto &[t, c] : ts)
        rs.push_back(mul({constant(c / Q(g)), t}));
      return idiv(add(rs), d / g);
    }
    std::vector<Id> qs, rs;
    for (auto &[t, c] : ts) {
      if (c.denominator() != 1)
        return intern("idiv", {n}, {Q(d)}, "", {true, false, false});
      Z q = floorz(c.numerator(), d), r = modz(c.numerator(), d);
      if (q != 0)
        qs.push_back(mul({constant(Q(q)), t}));
      if (r != 0)
        rs.push_back(mul({constant(Q(r)), t}));
    }
    Id r = add(rs);
    auto b = nodes[r].bounds;
    if (b && floorq(b->first / Q(d)) == floorq(b->second / Q(d))) {
      qs.push_back(constant(Q(floorq(b->first / Q(d)))));
      return add(qs);
    }
    // Packed decoding often contains (d-small)*bit. A centered remainder
    // exposes its carry without enumerating bit values. Try only short affine
    // sums and prove the remaining quotient is constant from exact bounds.
    if(ts.size()<=8) {
      Q low(0), high(0);bool bounded=true,changed=false;
      for(const auto &[term,coefficient]:ts) {
        Z rem=modz(coefficient.numerator(),d);
        if(rem>d/2) {rem-=d;changed=true;}
        if(rem==0)continue;
        auto range=nodes[term].bounds;
        if(!range || !nodes[term].facts.finite) {bounded=false;break;}
        Q l=Q(rem)*range->first,h=Q(rem)*range->second;
        low+=std::min(l,h);high+=std::max(l,h);
      }
      if(changed && bounded && floorq(low/Q(d))==floorq(high/Q(d))) {
        std::vector<Id> centered{constant(Q(floorq(low/Q(d))))};
        for(const auto &[term,coefficient]:ts) {
          Z rem=modz(coefficient.numerator(),d);if(rem>d/2)rem-=d;
          Z quotient=(coefficient.numerator()-rem)/d;
          if(quotient!=0)centered.push_back(mul({constant(Q(quotient)),term}));
        }
        ++stats["bounded_centered_quotients"];
        return add(centered);
      }
    }
    Bounds bound;
    if (b)
      bound =
          std::pair(Q(floorq(b->first / Q(d))), Q(floorq(b->second / Q(d))));
    qs.push_back(intern("idiv", {r}, {Q(d)}, "", {true, false, false}, bound));
    return add(qs);
  }
  Id imod(Id n, Z d) {
    if (d <= 0)
      throw Unsupported("index modulus must be positive");
    if((nodes[n].op=="word_shr" || nodes[n].op=="word_or") && (d&(d-1))==0)
      return bit_slice(n,0,boost::multiprecision::msb(d));
    if(nodes[n].op=="imod" && modz(nodes[n].data[0].numerator(),d)==0)
      return imod(nodes[n].args[0],d);
    if(nodes[n].op=="readbits" && (d & (d-1))==0) {
      unsigned count=boost::multiprecision::msb(d);
      unsigned bits=nodes[n].data[0].numerator().convert_to<unsigned>();
      if(count>=bits) return n;
      if(count>0 && count%8==0 && bits%8==0) return bit_slice(n,0,count);
    }
    if (nodes[n].op == "word" && modz(pow2(nodes[n].data[0].numerator().convert_to<unsigned>()), d) == 0)
      return imod(nodes[n].args[0], d);
    auto ts = terms(n);
    Z g = d;
    for (auto &[_, c] : ts) {
      if (c.denominator() != 1) {
        g = 1;
        break;
      }
      g = boost::multiprecision::gcd(g,
                                     boost::multiprecision::abs(c.numerator()));
    }
    if (g > 1) {
      std::vector<Id> rs;
      for (auto &[t, c] : ts)
        rs.push_back(mul({constant(c / Q(g)), t}));
      if (d / g == 1)
        return zero;
      return mul({constant(Q(g)), imod(add(rs), d / g)});
    }
    Id q = idiv(n, d);
    if (nodes[q].op != "idiv" && nodes[q].op != "add")
      return sub(n, mul({constant(Q(d)), q}));
    std::vector<Id> rs;
    for (auto &[t, c] : ts) {
      if (c.denominator() != 1)
        return intern("imod", {n}, {Q(d)}, "", {true, false, true},
                      std::pair(Q(0), Q(d - 1)));
      Z rem = modz(c.numerator(), d);
      if (rem != 0)
        rs.push_back(mul({constant(Q(rem)), t}));
    }
    Id r = add(rs);
    auto b = nodes[r].bounds;
    if (b && floorq(b->first / Q(d)) == floorq(b->second / Q(d)))
      return sub(r, constant(Q(floorq(b->first / Q(d)) * d)));
    // Coefficient reduction can expose a word beneath an aligned constant.
    // Only peel it when truncation preserves this modulus; never wrap addresses.
    if (r != n && nodes[r].op == "word" &&
        modz(pow2(nodes[r].data[0].numerator().convert_to<unsigned>()), d) == 0)
      return imod(nodes[r].args[0], d);
    return intern("imod", {r}, {Q(d)}, "", {true, false, true},
                  std::pair(Q(0), Q(d - 1)));
  }
  Id read(std::string alloc, std::vector<Id> coords, std::string type,
          Bounds b = {}) {
    bool finite = type.starts_with("real") || bool(b);
    return intern("read", coords, {}, alloc + "\n" + type,
                  {finite, b && b->first > 0, b && b->first >= 0}, b);
  }
  std::string render(Id id, int depth = 4) const {
    const auto &n = nodes[id];
    if (n.op == "const")
      return str(n.data[0]);
    if (depth <= 0)
      return n.op + "#" + std::to_string(id);
    std::string s = n.op + "[" + n.tag;
    for (const auto &q : n.data)
      s += "," + str(q);
    s += "](";
    for (Id x : n.args)
      s += render(x, depth - 1) + ",";
    return s + ")";
  }
};
} // namespace rg
