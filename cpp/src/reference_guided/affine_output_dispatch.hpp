#pragma once
#include "affine_state_pattern.hpp"

namespace rg {
struct AffineOutputProof {
  Id replacement = invalid, projection_read = invalid, projection_axis = invalid;
  J receipt;
};

// A finite dispatcher is checked against its stored arms. This is not a sample
// inference: each reachable arm must be the certified state prefix times the
// same typed projection read at its exact output coordinate.
class AffineOutputDispatch {
  Checker &checker;
  Arena &a;
  AffineStatePattern &states;
  struct Decline { const char *reason; };
  uint64_t work = 0;
  bool projection_sum = false;
  static constexpr size_t max_arms = 4096, max_factors = 8;
  static constexpr uint64_t max_work = 200000;
  void tick(size_t n = 1) {
    a.check(); if (n > max_work - work) throw Decline{"work_budget"};
    work += n;
  }
  struct Dot {
    Id state = invalid, read = invalid, binder = invalid;
    size_t segment = 0, prefix = 0;
    Z count = 0;
  };
  std::optional<Z> selected_point(Id condition, Id axis) {
    tick(); const auto &cmp = a.nodes[condition];
    if (cmp.op != "compare" || cmp.tag != "eq" || cmp.args.size() != 2) return {};
    Q scale(0), offset(0);
    for (size_t side=0;side<2;++side) {
      const auto &operand=a.nodes[cmp.args[side]];
      if (operand.args.size()>4) return {};
      if (operand.op=="add") {
        for(Id child:operand.args)if(child!=axis && child!=a.one)return {};
      } else if (operand.op=="mul") {
        if(operand.args.size()!=1 || operand.args[0]!=axis)return {};
      } else if (operand.op!="const" && cmp.args[side]!=axis) return {};
      tick(1+operand.args.size());
      for (auto [term, coefficient] : a.terms(cmp.args[side])) {
        tick(); if(side)coefficient=-coefficient;
        if (term == axis) scale += coefficient;
        else if (term == a.one) offset += coefficient;
        else return {};
      }
    }
    if (scale != 1 && scale != -1) return {};
    Q point = -offset / scale;
    if (point.denominator() != 1) return {};
    return point.numerator();
  }
  Dot dot(Id root) {
    tick(); const auto &fold = a.nodes[root];
    if (fold.op != "fold" || fold.tag != "Sum" || fold.args.size() != 2 ||
        fold.data.size() != 2 || fold.data[0].denominator() != 1 ||
        fold.data[0]<0 || fold.data[0]>=Q(a.nodes.size()) ||
        fold.data[1].denominator() != 1 || fold.data[1] <= 0 || fold.args[0] != a.zero)
      throw Decline{"projection_sum_form"};
    Id binder = fold.data[0].numerator().convert_to<Id>();
    if (binder >= a.nodes.size()) throw Decline{"projection_binder"};
    const auto &bound = a.nodes[binder];
    if (bound.op != "var" || !bound.bounds || bound.bounds->first != 0 ||
        bound.bounds->second != fold.data[1]-1) throw Decline{"projection_domain"};
    Id body = fold.args[1]; const auto &product = a.nodes[body];
    if (product.op != "mul" || product.data.size() != 1 ||
        product.args.size() > max_factors || !product.facts.finite)
      throw Decline{"projection_product"};
    std::optional<Dot> found;
    for (size_t i = 0; i < product.args.size(); ++i) {
      tick(); Id read = product.args[i]; const auto &r = a.nodes[read];
      if (r.op != "read" || !r.facts.finite || r.bounds || r.args.size()>4 || !a.depends(read,binder)) continue;
      size_t coordinate = 0;
      for (Id x : r.args) coordinate += a.depends(x,binder);
      if (coordinate != 1) continue;
      std::vector<Id> other{a.constant(product.data[0])};
      for (size_t j = 0; j < product.args.size(); ++j) if (i != j) other.push_back(product.args[j]);
      Id state = a.mul(std::move(other)); auto location = states.lookup(state);
      if (!location) continue;
      Dot candidate{state,read,binder,location->segment,location->index,fold.data[1].numerator()};
      if (found && (found->state != state || found->read != read)) throw Decline{"ambiguous_projection"};
      found = candidate;
    }
    if (!found) throw Decline{"uncertified_projection_state"};
    return *found;
  }
  // Coordinates of two reads of one tensor at points p and q, fitted as one
  // read in the axis: equal coordinates stay, constant ones become the integer
  // affine line through both points. Declines with the caller's reasons.
  std::vector<Id> fit_coordinates(const Node &r, const Z &p, const Node &s, const Z &q, Id axis,
                                  const char *fit, const char *noninteger) {
    std::vector<Id> coordinates;
    for (size_t i = 0; i < r.args.size(); ++i) {
      tick();
      if (r.args[i] == s.args[i]) { coordinates.push_back(r.args[i]); continue; }
      auto left = a.value(r.args[i]), right = a.value(s.args[i]);
      if (!left || !right || left->denominator() != 1 || right->denominator() != 1) throw Decline{fit};
      const Q slope = (*right - *left) / Q(q - p);
      if (slope.denominator() != 1) throw Decline{noninteger};
      coordinates.push_back(a.add({r.args[i], a.mul({a.constant(slope), a.sub(axis, a.constant(Q(p)))})}));
    }
    return coordinates;
  }
  Id read_template(const Dot &first, const Z &first_point,
                   const Dot &second, const Z &second_point, Id axis) {
    const auto &r = a.nodes[first.read], &s = a.nodes[second.read];
    if (r.tag != s.tag || r.args.size() != s.args.size() || first.binder != second.binder ||
        first_point == second_point) throw Decline{"projection_template_shape"};
    const auto coordinates = fit_coordinates(r, first_point, s, second_point, axis,
                                             "projection_coordinate_fit", "projection_coordinate_noninteger");
    auto split = r.tag.rfind('\n');
    if (split == std::string::npos || !r.tag.substr(split+1).starts_with("real"))
      throw Decline{"projection_read_type"};
    return a.read(r.tag.substr(0,split),coordinates,r.tag.substr(split+1));
  }
  // Dispatcher arms of the form shell_p(Sum_p): the Sums take the ordinary
  // dispatch below, and the shell becomes one expression in the output
  // coordinate. Each arm is split at its unique projection Sum (a path of
  // Add/Mul nodes with exactly one Sum-containing operand), the Sum replaced
  // by a placeholder: an opaque finite real atom, so every identity proved
  // with it holds for the Sum's real value. The shells of two arms fit a template
  // (equal nodes, or typed reads whose integer coordinates are affine in the
  // point, as read_template); the template at every arm's point must equal
  // that arm's shell by the exact scalar engine. Then the dispatched Sum fills
  // the placeholder. O(arms * shell) plus one bounded EP query per arm.
  struct ArmShell { Id hole = invalid, shell = invalid; };
  static constexpr unsigned max_shell_depth = 4;
  Id shell_hole = invalid;
  // Memoized per arm_shells call: split_shell asks every child at every level.
  std::map<std::pair<Id, unsigned>, bool> sum_memo;
  bool contains_sum(Id id, unsigned depth) {
    if (auto it = sum_memo.find({id, depth}); it != sum_memo.end()) return it->second;
    tick(); const auto &n = a.nodes[id];
    bool found = false;
    if (n.op == "fold") found = n.tag == "Sum";
    else if (depth < max_shell_depth && (n.op == "add" || n.op == "mul") && n.args.size() <= max_factors)
      for (Id child : n.args) if (contains_sum(child, depth + 1)) { found = true; break; }
    sum_memo.emplace(std::pair(id, depth), found);
    return found;
  }
  Id split_shell(Id id, Id &sum, unsigned depth) {
    tick(); const auto &n = a.nodes[id];
    if (n.op == "fold" && n.tag == "Sum") { sum = id; return shell_hole; }
    if (depth >= max_shell_depth || (n.op != "add" && n.op != "mul") || !n.facts.finite ||
        n.args.size() > max_factors) throw Decline{"projection_sum_form"};
    size_t chosen = n.args.size();
    for (size_t i = 0; i < n.args.size(); ++i)
      if (contains_sum(n.args[i], depth + 1)) {
        if (chosen != n.args.size()) throw Decline{"shell_sum_ambiguous"};
        chosen = i;
      }
    if (chosen == n.args.size()) throw Decline{"projection_sum_form"};
    std::vector<Id> args = n.args;
    args[chosen] = split_shell(n.args[chosen], sum, depth + 1);
    return checker.reapply(id, args);
  }
  Id shell_template(Id x, const Z &px, Id y, const Z &py, Id axis, std::map<std::pair<Id, Id>, Id> &memo) {
    tick();
    if (x == y) return x;
    if (auto it = memo.find({x, y}); it != memo.end()) return it->second;
    const auto &m = a.nodes[x], &n = a.nodes[y];
    const bool commutative = m.op == "add" || m.op == "mul";
    if (m.op != n.op || m.tag != n.tag || m.args.size() != n.args.size() ||
        (!commutative && m.data != n.data) || (m.op == "mul" && m.data != n.data) ||
        m.op == "const" || m.op == "var" || m.op == "fold" || m.op == "call")
      throw Decline{"shell_template"};
    Id result;
    if (commutative) {
      // Operand order follows Arena creation order, which may differ between
      // arms: pair equal operands first, then the rest by a unique structural
      // signature (operator, tag, arity, coefficient).
      const bool sum = m.op == "add";
      auto operands = [&](const Node &node) {
        std::vector<std::pair<Id, Q>> out;
        for (size_t i = 0; i < node.args.size(); ++i) out.emplace_back(node.args[i], sum ? node.data[i] : Q(1));
        return out;
      };
      auto left = operands(m), right = operands(n);
      auto signature = [&](const std::pair<Id, Q> &term) {
        const auto &t = a.nodes[term.first];
        return std::tuple(t.op, t.tag, t.args.size(), term.second);
      };
      std::vector<Id> terms;
      if (!sum) terms.push_back(a.constant(m.data[0]));
      std::vector<bool> used(right.size(), false);
      for (const auto &l : left) {
        tick();
        size_t match = right.size();
        for (size_t j = 0; j < right.size() && match == right.size(); ++j)
          if (!used[j] && right[j] == l) match = j;
        if (match == right.size())
          for (size_t j = 0; j < right.size(); ++j)
            if (!used[j] && signature(right[j]) == signature(l)) {
              if (match != right.size()) throw Decline{"shell_template_ambiguous"};
              match = j;
            }
        if (match == right.size()) throw Decline{"shell_template"};
        used[match] = true;
        const Id fitted = shell_template(l.first, px, right[match].first, py, axis, memo);
        terms.push_back(sum ? a.mul({a.constant(l.second), fitted}) : fitted);
      }
      result = sum ? a.add(std::move(terms)) : a.mul(std::move(terms));
    } else if (m.op == "read") {
      const auto coordinates = fit_coordinates(m, px, n, py, axis, "shell_coordinate_fit", "shell_coordinate_noninteger");
      const auto split = m.tag.rfind('\n');
      if (split == std::string::npos || m.bounds || !m.tag.substr(split + 1).starts_with("real"))
        throw Decline{"shell_read_type"};
      result = a.read(m.tag.substr(0, split), coordinates, m.tag.substr(split + 1));
    } else {
      std::vector<Id> args;
      for (size_t i = 0; i < m.args.size(); ++i) args.push_back(shell_template(m.args[i], px, n.args[i], py, axis, memo));
      result = checker.reapply(x, args);
    }
    memo.emplace(std::pair(x, y), result);
    return result;
  }
  // The shell with one node (the output axis variable or the placeholder
  // atom) replaced by a value. A subtree without the axis is kept unchanged,
  // as is every Fold for the placeholder: split_shell puts the placeholder
  // above every Fold it keeps, and an opaque atom has no free variable to
  // test. Only a Fold over the axis declines, so no binder is captured.
  // Operators are rebuilt by the shared reapply. O(changed shell nodes).
  Id replace(Id root, Id needle, Id value) {
    std::unordered_map<Id, Id> memo;
    const bool placeholder = needle == shell_hole;
    std::function<Id(Id)> visit = [&](Id id) -> Id {
      tick();
      if (id == needle) return value;
      if (auto it = memo.find(id); it != memo.end()) return it->second;
      const auto &n = a.nodes[id];
      if (placeholder ? n.op == "fold" : !a.depends(id, needle)) return id;
      if (n.op == "fold") throw Decline{"shell_fold"};
      std::vector<Id> args;
      bool changed = false;
      for (Id child : n.args) { args.push_back(visit(child)); changed |= args.back() != child; }
      return memo.emplace(id, changed ? checker.reapply(id, args) : id).first->second;
    };
    return visit(root);
  }
  // Replaces each arm by its Sum and returns the fitted shell; nothing when
  // the arms are Sums already (the ordinary dispatcher form).
  std::optional<ArmShell> arm_shells(std::map<Z, Id> &arms, Id axis) {
    if (a.nodes[arms.begin()->second].op == "fold") return {};
    if (shell_hole == invalid)
      shell_hole = a.intern("shell_hole", {}, {}, "affine-output-dispatch", {true, false, false});
    sum_memo.clear();
    std::map<Z, Id> sums, shells;
    for (const auto &[point, arm] : arms) {
      Id sum = invalid;
      const Id shell = split_shell(arm, sum, 0);
      if (sum == invalid) throw Decline{"projection_sum_form"};
      sums.emplace(point, sum); shells.emplace(point, shell);
    }
    auto f = shells.begin(), s = std::next(f);
    std::map<std::pair<Id, Id>, Id> memo;
    const Id shell = shell_template(f->second, f->first, s->second, s->first, axis, memo);
    for (const auto &[point, actual] : shells) {
      tick();
      const Id expected = replace(shell, axis, a.constant(Q(point)));
      if (!checker.scalar.equal(expected, actual)) throw Decline{"shell_alignment"};
    }
    arms = std::move(sums);
    ++a.stats["affine_output_shell_dispatches"];
    return ArmShell{shell_hole, shell};
  }
  std::optional<AffineOutputProof> single_projection(Id root) {
    const auto &fold=a.nodes[root];
    if(fold.op!="fold" || fold.tag!="Sum" || fold.args.size()!=2 ||
       fold.data.size()!=2 || fold.args[0]!=a.zero || !fold.facts.finite ||
       fold.data[0].denominator()!=1 || fold.data[0]<0 || fold.data[0]>=Q(a.nodes.size()) ||
       fold.data[1].denominator()!=1 || fold.data[1]<=0)return {};
    Id axis=fold.data[0].numerator().convert_to<Id>();
    const auto &v=a.nodes[axis];
    if(v.op!="var" || !v.bounds || v.bounds->first!=0 || v.bounds->second!=fold.data[1]-1)return {};
    const auto &body=a.nodes[fold.args[1]];
    if(body.op!="mul" || !body.facts.finite || body.args.size()>max_factors)return {};
    Id projection=invalid;
    for(Id factor:body.args) {
      tick();const auto &read=a.nodes[factor];
      auto split=read.tag.rfind('\n');
      if(read.op!="read" || !read.facts.finite || read.bounds || read.args.size()>4 ||
         split==std::string::npos || !read.tag.substr(split+1).starts_with("real"))continue;
      size_t coordinates=0;bool exact=true;
      for(Id index:read.args)if(a.depends(index,axis)) {++coordinates;exact &= index==axis;}
      if(coordinates!=1 || !exact)continue;
      projection_sum=true;
      if(projection!=invalid)throw Decline{"singleton_projection_ambiguous"};
      projection=factor;
    }
    if(projection==invalid)return {};
    // Reuse already-certified recurrence prefixes inside this stored Fold.
    // The outermost single state need not itself be a multi-step segment.
    auto certified=states.rewrite(root);
    Id replacement=certified?certified->replacement:root;
    ++a.stats["affine_output_single_projection_descriptors"];
    return AffineOutputProof{replacement,projection,axis,
      {{"method","finite_single_projection"},
       {"state_replacement",certified?certified->receipt:J()},
       {"coverage","unchanged_original_projection_domain"}}};
  }
public:
  std::string last_failure;
  // A cheap shape classification, not a proof or an orientation choice.
  // Consumers may apply independently valid finite-Sum identities even when
  // more than one direct typed read prevents selecting a projection axis role.
  bool has_projection_sum() const { return projection_sum; }
  AffineOutputDispatch(Checker &owner, AffineStatePattern &state_proofs)
      : checker(owner), a(owner.a), states(state_proofs) {}
  std::optional<AffineOutputProof> rewrite(Id root) {
    last_failure.clear(); work = 0;projection_sum=false;
    try {
      if (!states.collect(root)) { last_failure=states.last_failure; return {}; }
      if (auto location = states.lookup(root)) {
        Id result = states.closed_prefix(location->segment,a.constant(Q(location->index)));
        if (result == invalid) throw Decline{"final_state_construction"};
        ++a.stats["affine_output_final_state_proofs"];
        return AffineOutputProof{result,invalid,invalid,
          {{"method","certified_affine_final_state"},{"segment",location->segment},{"prefix",location->index}}};
      }
      if(a.nodes[root].op=="fold")return single_projection(root);
      const auto &r = a.nodes[root];
      if (r.op != "select" || r.free.size() != 1) return {};
      Id axis = r.free[0]; const auto &v = a.nodes[axis];
      if (v.op != "var" || !v.bounds || v.bounds->first.denominator() != 1 ||
          v.bounds->second.denominator() != 1) throw Decline{"output_axis_domain"};
      Z lo = v.bounds->first.numerator(), hi = v.bounds->second.numerator();
      if (lo > hi || hi-lo+1 > max_arms) throw Decline{"dispatcher_size"};
      std::map<Z,Id> arms; Id cursor = root;
      while (a.nodes[cursor].op == "select") {
        tick(); const auto &n = a.nodes[cursor];
        if (n.args.size() != 3 || arms.size() >= max_arms) throw Decline{"dispatcher_form"};
        auto point = selected_point(n.args[0],axis);
        if (!point || *point < lo || *point > hi) throw Decline{"dispatcher_predicate"};
        if (!arms.emplace(*point,n.args[1]).second) throw Decline{"dispatcher_overlap"};
        cursor = n.args[2];
      }
      // A last default arm may cover exactly one remaining point. Larger
      // defaults decline instead of enumerating an unrepresented domain.
      if (Z(arms.size()) == hi-lo) {
        Z missing = lo;
        for (const auto &[point,unused] : arms) { (void)unused; if (point == missing) ++missing; else break; }
        if (missing > hi || !arms.emplace(missing,cursor).second) throw Decline{"dispatcher_default"};
      }
      if (Z(arms.size()) != hi-lo+1 || arms.size() < 2) throw Decline{"dispatcher_coverage"};
      const auto shell = arm_shells(arms, axis);
      auto f = arms.begin(), s = std::next(f);
      Dot first = dot(f->second);
      if(first.binder==axis)throw Decline{"projection_output_capture"};
      // An imported default arm may reuse an independently scoped, already
      // stored projection Fold. Alpha-align only its bound component variable;
      // neither sample fitting nor equal-looking state updates justify reuse.
      // Re-collect the renamed actual state through the existing recognizer so
      // every update, incoming value and prefix remains independently checked.
      std::map<Z,Dot> aligned;
      for (const auto &[point,arm] : arms) {
        tick(); Dot actual=dot(arm);
        if (a.depends(arm,axis)) throw Decline{"dispatcher_state_alignment"};
        if (actual.binder!=first.binder) {
          if (actual.binder==axis || actual.count!=first.count ||
              a.nodes[actual.binder].bounds!=a.nodes[first.binder].bounds ||
              a.depends(actual.state,first.binder) || a.depends(actual.read,first.binder))
            throw Decline{"projection_alpha_capture"};
          Id state=states.substitute(actual.state,actual.binder,first.binder);
          Id read=states.substitute(actual.read,actual.binder,first.binder);
          if (state==invalid || read==invalid || !states.collect(state))
            throw Decline{"projection_alpha_state"};
          actual.state=state;actual.read=read;actual.binder=first.binder;
          ++a.stats["affine_output_alpha_aligned_arms"];
        }
        aligned.emplace(point,actual);
      }
      // Collecting a renamed final state may extend an existing segment. Query
      // the authoritative locations after all such extensions, not old copies.
      for (auto &[point,actual] : aligned) {
        (void)point;auto location=states.lookup(actual.state);
        if (!location) throw Decline{"projection_alpha_uncertified"};
        actual.segment=location->segment;actual.prefix=location->index;
      }
      first=aligned.at(f->first);Dot second=aligned.at(s->first);
      Id projection = read_template(first,f->first,second,s->first,axis);
      Z prefix_offset = Z(first.prefix)-f->first;
      const auto &segment = states.segments().at(first.segment);
      if (lo+prefix_offset < 0 || hi+prefix_offset >= Z(segment.count)) throw Decline{"prefix_domain"};
      for (const auto &[point,actual] : aligned) {
        tick();
        if (actual.segment != first.segment || actual.binder != first.binder || actual.count != first.count ||
            Z(actual.prefix) != point+prefix_offset) throw Decline{"dispatcher_state_alignment"};
        Id expected_read = states.substitute(projection,axis,a.constant(Q(point)));
        if (expected_read != actual.read) throw Decline{"dispatcher_projection_alignment"};
      }
      Id index = a.add({axis,a.constant(Q(prefix_offset))});
      Id state = states.closed_prefix(first.segment,index);
      if (state == invalid) throw Decline{"prefix_construction"};
      Id body = a.mul({projection,state});
      Id result = a.intern("fold",{a.zero,body},{Q(first.binder),Q(first.count)},"Sum",
                           {a.nodes[body].facts.finite,false,false});
      if (shell) result = replace(shell->shell, shell->hole, result);
      a.stats["affine_output_dispatch_arms"] += arms.size();
      ++a.stats["affine_output_dispatch_proofs"];
      return AffineOutputProof{result,projection,first.binder,
        {{"method","complete_affine_prefix_dispatch"},{"arms",arms.size()},
         {"output_domain",J::array({lo.str(),hi.str()})},{"segment",first.segment},
         {"coverage","exactly_once_per_output_coordinate"},{"work",work}}};
    } catch (const Decline &d) {
      last_failure = d.reason; ++a.stats["affine_output_decline_"+last_failure]; return {};
    }
  }
};
} // namespace rg
