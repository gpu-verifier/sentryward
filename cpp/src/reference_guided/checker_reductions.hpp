#pragma once
#include "checker.hpp"

namespace rg {

// case_restriction: (output coordinate, value) of a case part (match by cases on an output coordinate); the
// bodies instantiated here are restricted as the target was. Null elsewhere.
inline auto Checker::match_term_target(Id sid, Id n, const std::vector<Id> &initial, bool fixed, bool masked_reference,
                                  const std::pair<Id, Id> *case_restriction) -> std::set<Match> {
  const auto &s = specs[sid];
  const auto &p = patterns.nodes[s.pattern];
  const auto &node = a.nodes[n];
  std::vector<std::pair<std::vector<Id>, Id>> rs;
  for (auto cap : matches(s.pattern, n, initial, fixed))
    rs.emplace_back(std::move(cap), a.one);
  auto expmatches = [&](Id ep, Id target, const std::vector<Id> &cap,
                        Id factor) {
    const auto &t = a.nodes[target];
    if (t.op != patterns.nodes[ep].op || !a.nodes[t.args[0]].facts.finite)
      return;
    const auto &sum = a.nodes[t.args[0]];
    if (sum.op != "add")
      return;
    for (size_t j = 0; j < sum.args.size(); ++j) {
      Id candidate = a.mul({a.constant(sum.data[j]), sum.args[j]});
      auto caps = matches(patterns.nodes[ep].args[0], candidate, cap, fixed);
      if (caps.empty())
        continue;
      std::vector<Id> rest;
      for (size_t k = 0; k < sum.args.size(); ++k)
        if (k != j)
          rest.push_back(a.mul({a.constant(sum.data[k]), sum.args[k]}));
      Id f = a.mul({factor, a.exponential(a.add(rest), t.op == "exp2")});
      for (auto c : caps)
        rs.emplace_back(std::move(c), f);
    }
  };
  if (rs.empty() && (p.op == "exp2" || p.op == "exp") && node.op == p.op)
    expmatches(s.pattern, n, initial, a.one);
  if (rs.empty() && p.op == "mul" && node.op == "mul" &&
      p.data == node.data && p.args.size() == 2 && node.args.size() == 2) {
    for (size_t j = 0; j < 2; ++j) {
      Id ep = p.args[j], other = p.args[1 - j];
      if (patterns.nodes[ep].op != "exp2" && patterns.nodes[ep].op != "exp")
        continue;
      for (size_t k = 0; k < 2; ++k)
        for (auto cap : matches(other, node.args[1 - k], initial, fixed))
          expmatches(ep, node.args[k], cap, a.one);
    }
  }
  if (exponential_mode && rs.empty())
    for(auto cap:body_candidates(s,n,initial))rs.emplace_back(std::move(cap),a.one);
  // Rule (missing positive scale factor): a product body P = g * R whose factor g is an
  // exponential of a stored input (a block scale such as 2^(byte - 127)), against a target that is
  // a product of one factor fewer and holds no exponential: candidates of R alone propose captures;
  // each is proposed with coefficient 1/g at that capture (g is finite and positive there), and
  // the ordinary verification below proves target = P * (1/g) for the whole body P. The reduction
  // consumer (bind) then requires that coefficient to be independent of the binder.
  // (Fresh copies: the matchers above may have grown both node stores.)
  const std::string pattern_op = patterns.nodes[s.pattern].op, target_op = a.nodes[n].op;
  const auto pattern_args = patterns.nodes[s.pattern].args;
  const auto pattern_data = patterns.nodes[s.pattern].data;
  const auto pattern_tag = patterns.nodes[s.pattern].tag;
  const auto target_args = a.nodes[n].args;
  if (exponential_mode && rs.empty() && pattern_op == "mul" && pattern_data.size() == 1 &&
      pattern_args.size() >= 2 && pattern_args.size() <= 8 && target_op == "mul" &&
      target_args.size() + 1 == pattern_args.size() &&
      std::none_of(target_args.begin(), target_args.end(), [&](Id f) {
        return a.nodes[f].op == "exp" || a.nodes[f].op == "exp2"; })) {
    for (size_t j = 0; j < pattern_args.size() && rs.empty(); ++j) {
      const auto &g = patterns.nodes[pattern_args[j]];
      if ((g.op != "exp" && g.op != "exp2") || !stored_input_scale(pattern_args[j])) continue;
      std::vector<Id> rest;
      for (size_t k = 0; k < pattern_args.size(); ++k) if (k != j) rest.push_back(pattern_args[k]);
      Spec reduced = s;
      reduced.pattern = patterns.intern("mul", std::move(rest), pattern_data, pattern_tag);
      Captures caps = matches(reduced.pattern, n, initial, fixed);
      if (caps.empty()) caps = body_candidates(reduced, n, initial);
      for (auto &cap : caps) {
        if (cap.size() != s.slots || std::find(cap.begin(), cap.end(), invalid) != cap.end()) continue;
        Id scale = instantiate_body(pattern_args[j], cap);
        if (case_restriction && scale != invalid) scale = restrict_axis(scale, case_restriction->first, case_restriction->second);
        if (scale == invalid || !a.nodes[scale].facts.finite || !a.nodes[scale].facts.pos) continue;
        rs.emplace_back(std::move(cap), a.recip(scale));
        ++a.stats["missing_scale_factor_candidates"];
      }
    }
  }
  std::set<Match> unique;
  for (auto &[cap, f] : rs) {
    if (cap.size() != s.slots ||
        std::find(cap.begin(), cap.end(), invalid) != cap.end())
      continue;
    // EP anchors propose candidates; they cannot override parameters fixed
    // by the enclosing reduction query.
    if (fixed) {
      bool same=true;
      for (size_t k=0;k<cap.size();++k)
        if(k!=s.axis && cap[k]!=initial[k])same=false;
      if(!same)continue;
    }
    if(exponential_mode) {
      Id expected;
      {ExponentialPolynomial::Timer timer(scalar.profiling?&scalar.instantiate_ns:nullptr);expected=instantiate_body(s.pattern,cap);}
      if(case_restriction && expected!=invalid) {expected=restrict_axis(expected,case_restriction->first,case_restriction->second);++a.stats["case_restricted_bodies"];}
      if(expected==invalid)++a.stats["ep_body_instantiation_declines"];
      bool equal=false;
      // Second pass only when the body as instantiated is missing: the body instantiated against this
      // target (twin mode) with its remaining reads aligned to the target's reads (align_reads). A
      // pair-local copy; the proof below is the same complete equality.
      for(int pass=0;pass<2 && !equal;++pass) {
      if(pass==0 && expected==invalid)continue;
      if(pass==1) {
        // Only a body the Arena cannot instantiate as such (a read it does not hold) is instantiated
        // against the target; a body that was instantiated keeps its reads.
        if(expected!=invalid)break;
        Id twin=instantiate_body(s.pattern,cap,n);
        if(case_restriction && twin!=invalid) twin=restrict_axis(twin,case_restriction->first,case_restriction->second);
        if(twin==invalid)break;
        ++a.stats["ep_body_twin_bodies"];
        expected=align_reads(twin,n);
      }
      if(!admit_payload_coordinates(a.mul({expected,f}),n))continue;
      ++a.stats["ep_body_obligations"];
      {ExponentialPolynomial::Timer timer(scalar.profiling?&scalar.body_scalar_ns:nullptr);
        // This theorem proves the complete masked product, not just its
        // positive branch. An opaque EP atom must not erase the mask.
        if (auto coefficient = masked_reference ? masked_body_coefficient(expected, n, true) : std::optional<Id>{}) {
          f = *coefficient; equal = true;
        } else if (missing_required_call(a.mul({expected,f}),n)) {
          // The body needs a reduction Call the target does not hold: no EP equality.
          equal = false; ++a.stats["ep_body_missing_call_refutations"];
        } else {
          equal=scalar.equal(a.mul({expected,f}),n);
          // Rule (constant ratio of two products): f * body and the target are finite products of the
          // same factors with scalars e and d; then target = (d / e) * f * body. Exact. O(factors).
          if(!equal) {
            const Id lhs=a.mul({expected,f});
            const auto &l=a.nodes[lhs], &r=a.nodes[n];
            if(l.op=="mul" && r.op=="mul" && l.data.size()==1 && r.data.size()==1 && l.data[0]!=0 &&
               l.facts.finite && r.facts.finite && l.args==r.args && l.data[0]!=r.data[0]) {
              f=a.mul({f,a.constant(r.data[0]/l.data[0])});equal=true;++a.stats["ep_body_constant_ratios"];
            }
          }
          // Rule (cases of a stored byte, byte_cases_equal): the body and the target as functions of one
          // stored byte (an e2m1 nibble decoded by different formulas).
          if(!equal) equal=byte_cases_equal(a.mul({expected,f}),n);
        }
      }
      }
      if(!equal)continue;
      ++a.stats["ep_body_proofs"];
    }
    std::vector<Id> params;
    for (size_t j = 0; j < cap.size(); ++j)
      if (j != s.axis)
        params.push_back(cap[j]);
    unique.insert({std::move(params), cap[s.axis], canonical_coefficient(f)});
  }
  return unique;
}

// A pattern exponential whose argument reads a stored input (a read, readbits or packedbits node
// occurs in it): a data-dependent scale, not a softmax weight. O(pattern nodes of the argument).
inline auto Checker::stored_input_scale(Id pattern_node) -> bool {
  std::vector<Id> todo{pattern_node};
  std::unordered_set<Id> seen{pattern_node};
  while (!todo.empty()) {
    const auto &n = patterns.nodes[todo.back()]; todo.pop_back();
    if (n.op == "read" || n.op == "readbits" || n.op == "packedbits") return true;
    if (n.op == "call" || seen.size() > 64) return false;
    for (Id x : n.args) if (seen.insert(x).second) todo.push_back(x);
  }
  return false;
}

inline auto Checker::combine(const std::vector<Summary> &parts) -> Summary {
  Summary result;
  bool first = true;
  for (const auto &s : parts) {
    if (s.empty())
      continue;
    if (first) {
      result = s;
      first = false;
      continue;
    }
    using Prefix = std::pair<Id, std::vector<Id>>;
    std::map<Prefix, std::vector<std::pair<Key, Id>>> index;
    for (const auto &[k, p] : s)
      index[{k.sid, k.params}].emplace_back(k, p);
    Summary next;
    for (const auto &[k, p] : result) {
      auto it = index.find({k.sid, k.params});
      if (it == index.end())
        continue;
      for (const auto &[other, p2] : it->second) {
        a.check();
        if (!equal_coefficients(k.coef, other.coef)) continue;
        Key key = k;
        key.coef = canonical_coefficient(key.coef);
        key.residual = plus(k.sid, {k.residual, other.residual});
        next[key] = concat(p, p2);
      }
    }
    result = std::move(next);
  }
  return result;
}

inline auto Checker::scale(const Summary &s, Id c) -> Summary {
  Summary r;
  for (const auto &[k, p] : s) {
    Key key = k;
    key.coef = canonical_coefficient(a.mul({k.coef, c}));
    key.residual = a.mul({k.residual, c});
    r[key] = p;
  }
  return r;
}

inline auto Checker::residual(const Summary &s, Id n) -> Summary {
  Summary r;
  for (const auto &[k, p] : s) {
    Key key = k;
    key.residual = plus(k.sid, {k.residual, n});
    r[key] = p;
  }
  return r;
}

inline auto Checker::bind(const Summary &s, Id var, const Z &count) -> Summary {
  Summary r;
  for (const auto &[k, p] : s) {
    if (a.depends(k.coef, var) ||
        std::any_of(k.params.begin(), k.params.end(),
                    [&](Id x) { return a.depends(x, var); })) {
      ++failures["binder_in_fixed_parameter_or_factor"];
      continue;
    }
    if (a.depends(k.residual, var)) {
      ++failures["unmatched_binder_dependent_residual"];
      continue;
    }
    Key key = k;
    if (!specs[k.sid].maximum)
      key.residual = a.mul({a.constant(Q(count)), k.residual});
    r[key] = bind_piece(p, var, count);
  }
  return r;
}

inline auto Checker::box(Id index,
                       const std::vector<std::pair<Id, Z>> &domains) -> std::optional<Box> {
  std::map<Id, Q> ts;
  for (auto [t, c] : a.terms(index))
    ts[t] = c;
  Q start = 0;
  if (ts.contains(a.one)) {
    start = ts[a.one];
    ts.erase(a.one);
  }
  Box b;
  for (const auto &[v, n] : domains) {
    Q c = 0;
    if (ts.contains(v)) {
      c = ts[v];
      ts.erase(v);
    }
    if (c.denominator() != 1 || n < 0)
      return {};
    if (c == 0) {
      b.mult *= n;
      continue;
    }
    if (c < 0) {
      start += c * Q(n - 1);
      c = -c;
    }
    b.axes.emplace_back(c.numerator(), n);
  }
  for (auto it = ts.begin(); it != ts.end();)
    if (it->second == 0)
      it = ts.erase(it);
    else
      ++it;
  if (!ts.empty() || start.denominator() != 1)
    return {};
  b.start = start.numerator();
  return b;
}

inline auto Checker::boxes(Id p, const Z &count, bool &permuted) -> std::optional<std::vector<Box>> {
  struct W {
    Id id;
    bool leave;
  };
  std::vector<W> todo{{p, false}};
  std::vector<std::pair<Id, Z>> domains;
  std::vector<Box> out;
  std::optional<Id> common_mask;
  bool plain = false;
  permuted = false;
  while (!todo.empty()) {
    a.check();
    auto w = todo.back();
    todo.pop_back();
    const auto &n = pieces[w.id];
    if (w.leave) {
      domains.pop_back();
      continue;
    }
    if (n.kind == PieceNode::Concat) {
      todo.push_back({n.right, false});
      todo.push_back({n.left, false});
    } else if (n.kind == PieceNode::Bind) {
      domains.emplace_back(n.var, n.count);
      todo.push_back({w.id, true});
      todo.push_back({n.left, false});
    } else {
      Id index = n.index;
      const auto &coordinate = a.nodes[index];
      if (coordinate.op == "index_xor") {
        if (plain || coordinate.args.size() != 2 || coordinate.data.size() != 1 ||
            (coordinate.data[0] != 6 && coordinate.data[0] != 7) ||
            count != pow2(coordinate.data[0].numerator().convert_to<unsigned>())) return {};
        Id mask = coordinate.args[1];
        if (!bounded_coordinate_integer(a, coordinate.args[0], count) ||
            !bounded_coordinate_integer(a, mask, count) ||
            (common_mask && *common_mask != mask)) return {};
        for (const auto &[variable, length] : domains) {
          (void)length;
          if (a.depends(mask, variable)) return {};
        }
        common_mask = mask; permuted = true; index = coordinate.args[0];
      } else {
        if (common_mask) return {};
        plain = true;
      }
      auto b = box(index, domains);
      if (!b)
        return {};
      out.push_back(std::move(*b));
      ++a.stats["coverage_stored_piece_visits"];
    }
  }
  return out;
}

inline auto Checker::close(const Summary &s) -> std::optional<Id> {
  std::set<Id> closed;
  for (const auto &[k, p] : s) {
    bool permuted = false;
    auto bs = boxes(p, specs[k.sid].count, permuted);
    if (bs && cover_boxes(*bs, specs[k.sid].count, a)) {
      closed.insert(
          plus(k.sid, {k.residual, a.mul({k.coef, call(k.sid, k.params)})}));
      ++a.stats["completed_reductions"];
      if (permuted) ++a.stats["xor_permutation_complete_domains"];
    }
  }
  if (closed.size() > 1) {
    const auto representative=*closed.begin();
    for(auto alternative:closed)if(alternative!=representative)
      complete_reduction_equalities.emplace_back(representative,alternative);
    ++a.stats["multiple_complete_reduction_proofs"];
  }
  if (closed.empty())
    return {};
  return *closed.begin();
}

} // namespace rg
