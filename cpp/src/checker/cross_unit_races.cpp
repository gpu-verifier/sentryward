#include "sentryward/algorithm/checker/cross_unit_races.h"

#include "sentryward/algorithm/memory/model.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sentryward::algorithm {
namespace {
using I = address_solver::Int;
using Status = CrossUnitOutcome::Status;

struct Interval {
  I lo, hi; // [lo, hi)
};
// Sorted, disjoint, merged intervals.
std::vector<Interval> Merge(std::vector<Interval> intervals) {
  std::sort(intervals.begin(), intervals.end(), [](const Interval &a, const Interval &b) { return a.lo < b.lo; });
  std::vector<Interval> merged;
  for (const auto &x : intervals)
    if (!merged.empty() && x.lo <= merged.back().hi) merged.back().hi = std::max(merged.back().hi, x.hi);
    else merged.push_back(x);
  return merged;
}
// Whether a and (b shifted by d) share a byte; both sorted and disjoint (linear merge).
bool Meet(const std::vector<Interval> &a, const std::vector<Interval> &b, I d) {
  std::size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    const I blo = b[j].lo + d, bhi = b[j].hi + d;
    if (a[i].hi <= blo) ++i;
    else if (bhi <= a[i].lo) ++j;
    else return true;
  }
  return false;
}
// Parts of one class on one allocation, sorted by lo, with a max-hi tree (interval queries).
struct PartIndex {
  std::vector<std::size_t> parts; // indexes into the global part list
  std::vector<Interval> spans;
  std::vector<I> maximum;
  std::size_t leaves = 1;
  void Build() {
    std::vector<std::size_t> order(parts.size());
    for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return spans[a].lo < spans[b].lo; });
    std::vector<std::size_t> p(order.size());
    std::vector<Interval> s(order.size());
    for (std::size_t k = 0; k < order.size(); ++k) {
      p[k] = parts[order[k]];
      s[k] = spans[order[k]];
    }
    parts = std::move(p);
    spans = std::move(s);
    while (leaves < spans.size()) leaves *= 2;
    maximum.assign(2 * leaves, -(I(1) << 126));
    for (std::size_t k = 0; k < spans.size(); ++k) maximum[leaves + k] = spans[k].hi;
    for (std::size_t k = leaves; k-- > 1;) maximum[k] = std::max(maximum[2 * k], maximum[2 * k + 1]);
  }
  // Visits every k with spans[k] ∩ [lo, hi) ≠ ∅; the visitor returns false to stop.
  template <class Visit> bool Query(I lo, I hi, Visit &&visit) const {
    const auto walk = [&](auto &&self, std::size_t node, std::size_t begin, std::size_t end) -> bool {
      if (begin >= spans.size() || spans[begin].lo >= hi || maximum[node] <= lo) return true;
      if (end - begin == 1) return spans[begin].hi > lo ? visit(begin) : true;
      const auto middle = begin + (end - begin) / 2;
      return self(self, 2 * node, begin, middle) && self(self, 2 * node + 1, middle, end);
    };
    return walk(walk, 1, 0, leaves);
  }
};
Interval Span(const CrossUnitPart &part) {
  const auto &f = part.family;
  const auto &p = part.footprint;
  const I a = p.offset + p.step * I(f.first_member), b = p.offset + p.step * I(f.first_member + f.count - 1);
  return {std::min(a, b), std::max(a, b) + I(p.bytes)};
}
} // namespace

bool UnitPrivateSpace(std::string_view space) noexcept {
  return space == "shared" || space == "local" || space == "tensor" || space == "register" || space == "param" ||
         space == "const" || space.starts_with("shared::");
}

CrossUnitOutcome CheckCrossUnit(std::span<const CrossUnitPart> parts, const UnitQuotient &quotient,
                                MemoryModel *model, const CrossUnitLimits &limits) {
  std::uint64_t unit_pairs = 0, refinements = 0, decisions = 0; // the work counted against `limits`
  CrossUnitOutcome outcome;
  const auto unknown = [](std::string reason) {
    CrossUnitOutcome result;
    result.status = Status::kUnknown;
    result.reason = std::move(reason);
    return result;
  };
  // No member unit: every unit pair is a representative pair (the normal checker's).
  if (!quotient.Shared()) return outcome;
  const auto shared_class = [&](std::uint32_t klass) { return quotient.classes.at(klass).units.size() > 1; };

  // Global parts, by allocation; per alias the allocations that use it.
  struct ClassParts {
    PartIndex all;
    std::vector<std::size_t> writes;
    std::vector<Interval> write_hull, all_hull;
  };
  std::map<std::string, std::map<std::uint32_t, ClassParts>> by_allocation;
  // Lookup parts (no constant span), by allocation and class.
  std::map<std::string, std::map<std::uint32_t, std::vector<std::size_t>>> lookups;
  std::map<std::string, std::set<std::string>> alias_allocations;
  std::map<std::string, bool> alias_write, alias_shared;
  bool unplaced = false, unplaced_write = false, unplaced_shared = false, any_shared_global = false,
       any_global_write = false;
  for (std::size_t k = 0; k < parts.size(); ++k) {
    const auto &part = parts[k];
    const auto &fp = part.footprint;
    if (fp.empty || part.family.count == 0) continue;
    if (UnitPrivateSpace(fp.space)) continue;
    const auto klass = quotient.units.at(quotient.UnitOf(part.cta)).klass;
    const bool shared = shared_class(klass);
    const bool lookup = model && !fp.symbolic.empty() && !fp.gathers.empty();
    if (fp.space != "global" || fp.allocation.empty() || (!fp.symbolic.empty() && !lookup) || !fp.affine) {
      // Not placeable against a translated copy: it may meet any other unit's global access.
      unplaced = true;
      unplaced_write |= part.write;
      unplaced_shared |= shared;
      continue;
    }
    any_shared_global |= shared;
    any_global_write |= part.write;
    if (lookup) {
      lookups[fp.allocation][klass].push_back(k);
    } else {
      auto &entry = by_allocation[fp.allocation][klass];
      entry.all.parts.push_back(k);
      entry.all.spans.push_back(Span(part));
      if (part.write) entry.writes.push_back(k);
    }
    if (!fp.alias.empty()) {
      alias_allocations[fp.alias].insert(fp.allocation);
      alias_write[fp.alias] = alias_write[fp.alias] || part.write;
      alias_shared[fp.alias] = alias_shared[fp.alias] || shared;
    }
  }
  // An unplaceable global access paired with a member unit's access, one of them a write (an
  // Unknown involving a member restarts per unit).
  if (unplaced && (unplaced_shared || any_shared_global) && (unplaced_write || any_global_write))
    return unknown("unplaced_global_access");
  // Different allocations sharing an alias are Unknown to ClassifyFamilyOverlap (family_access.cpp).
  for (const auto &[alias, allocations] : alias_allocations)
    if (allocations.size() > 1 && alias_write[alias] && alias_shared[alias])
      return unknown("alias_group:" + alias);

  for (auto &[allocation, classes] : by_allocation) {
    // δ of every unit of every class with parts here, sorted (the window search).
    std::map<std::uint32_t, std::vector<std::pair<std::int64_t, std::uint32_t>>> deltas;
    for (auto &[klass, entry] : classes) {
      entry.all.Build();
      std::vector<Interval> writes;
      for (const auto k : entry.writes) writes.push_back(Span(parts[k]));
      entry.write_hull = Merge(std::move(writes));
      entry.all_hull = Merge(entry.all.spans);
      const auto &c = quotient.classes.at(klass);
      auto &list = deltas[klass];
      list.reserve(c.units.size());
      for (std::size_t i = 0; i < c.units.size(); ++i) {
        if (i == 0) {
          list.emplace_back(0, 0);
          continue;
        }
        if (i >= c.delta.size()) return unknown("delta_missing:" + allocation);
        const auto found = c.delta[i].find(allocation);
        if (found == c.delta[i].end()) return unknown("delta_missing:" + allocation);
        list.emplace_back(found->second, static_cast<std::uint32_t>(i));
      }
      std::sort(list.begin(), list.end());
    }
    for (const auto &[a, A] : classes) {
      if (A.write_hull.empty()) continue;
      for (const auto &[b, B] : classes) {
        if (B.all_hull.empty() || (!shared_class(a) && !shared_class(b))) continue;
        // d = δ_v - δ_u must lie strictly between these for the hulls to meet.
        const I low = A.write_hull.front().lo - B.all_hull.back().hi;
        const I high = A.write_hull.back().hi - B.all_hull.front().lo;
        const auto &du = deltas.at(a), &dv = deltas.at(b);
        std::unordered_map<std::int64_t, bool> memo; // d -> hulls or families meet (never true: we stop)
        for (const auto &[delta_u, i] : du) {
          const I lo = I(delta_u) + low, hi = I(delta_u) + high;
          auto first = std::upper_bound(dv.begin(), dv.end(), lo, [](I x, const auto &e) { return x < I(e.first); });
          for (auto it = first; it != dv.end() && I(it->first) < hi; ++it) {
            const auto [delta_v, j] = *it;
            if ((a == b && i == j) || (i == 0 && j == 0)) continue; // same unit; representative pair
            if (++unit_pairs > limits.unit_pairs) {
              CrossUnitOutcome cap;
              cap.status = Status::kCap;
              cap.reason = "unit_pairs";
              return cap;
            }
            const std::int64_t d = delta_v - delta_u;
            if (!memo.try_emplace(d, false).second) continue;
            if (!Meet(A.write_hull, B.all_hull, I(d))) continue;
            // Exact: every write part of A against the parts of B whose translated span meets it.
            for (const auto k : A.writes) {
              const auto &p = parts[k];
              const auto span = Span(p);
              std::optional<CrossUnitOutcome> found;
              B.all.Query(span.lo - I(d), span.hi - I(d), [&](std::size_t index) {
                const auto &q = parts[B.all.parts[index]];
                if (++refinements > limits.refinements) {
                  CrossUnitOutcome cap;
                  cap.status = Status::kCap;
                  cap.reason = "refinements";
                  found = cap;
                  return false;
                }
                auto shifted = q.footprint;
                shifted.offset += I(d);
                auto relation = ClassifyFamilyOverlap(p.family, p.footprint, q.family, shifted);
                if (!relation.unknown.empty()) {
                  CrossUnitOutcome u;
                  u.status = Status::kUnknown;
                  u.reason = "family_overlap_unknown:" + allocation;
                  found = u;
                  return false;
                }
                for (const auto &r : relation.overlap)
                  if (const auto witness = r.witness()) {
                    CrossUnitOutcome race;
                    race.status = Status::kRace;
                    race.first_part = k;
                    race.second_part = B.all.parts[index];
                    race.first_member = witness->first;
                    race.second_member = witness->second;
                    const auto &ca = quotient.classes.at(a), &cb = quotient.classes.at(b);
                    race.first_unit = ca.units.at(i);
                    race.second_unit = cb.units.at(j);
                    race.first_delta = delta_u;
                    race.second_delta = delta_v;
                    race.allocation = allocation;
                    const I x = p.footprint.offset + p.footprint.step * I(witness->first) + I(delta_u);
                    const I y = q.footprint.offset + q.footprint.step * I(witness->second) + I(delta_v);
                    const I begin = std::max(x, y), end = std::min(x + I(p.footprint.bytes), y + I(q.footprint.bytes));
                    race.region_offset = begin > 0 ? static_cast<std::uint64_t>(begin) : 0;
                    race.region_bytes = end > begin ? static_cast<std::uint64_t>(end - begin) : 0;
                    found = race;
                    return false;
                  }
                return true;
              });
              if (found) return std::move(*found);
            }
          }
        }
      }
    }
  }

  // Lookup parts: every pair with at least one lookup part, over pairs of unit groups. The parts of a
  // class are bucketed by their lookup terms (constant parts: the bucket without terms). Two buckets
  // whose translated terms are equal cancel them: their residual spans are compared exactly, as in
  // the constant pass. Other bucket pairs are decided once on their residual hulls, then pair by pair.
  for (const auto &[allocation, lookup_classes] : lookups) {
    struct Group {
      std::vector<std::int64_t> delta; // over On::objects
      std::vector<std::uint32_t> units; // indexes into the class's units
    };
    struct Bucket {
      bool lookup = false;
      std::vector<GatherTerm> terms; // sorted by (symbol, arg)
      PartIndex all;                 // residual spans (the footprint without its lookups)
      std::vector<std::size_t> writes;
      Interval write_hull{0, 0}, all_hull{0, 0};
    };
    struct On {
      std::vector<Bucket> buckets;
      std::vector<std::string> objects; // the allocation, then the lookups' objects
      std::vector<Group> groups;
      std::string alias;
      bool write = false, uniform = true;
    };
    std::map<std::uint32_t, On> on;
    for (const auto &[klass, list] : lookup_classes) {
      auto &entry = on[klass];
      std::map<std::vector<std::tuple<std::string, I, I>>, std::size_t> index;
      for (const auto k : list) {
        auto terms = parts[k].footprint.gathers;
        std::sort(terms.begin(), terms.end(), [](const GatherTerm &x, const GatherTerm &y) {
          return std::tie(x.symbol, x.arg) < std::tie(y.symbol, y.arg);
        });
        std::vector<std::tuple<std::string, I, I>> key;
        for (const auto &t : terms) key.emplace_back(t.symbol, t.arg, t.coefficient);
        const auto [it, added] = index.try_emplace(std::move(key), entry.buckets.size());
        if (added) entry.buckets.push_back({true, std::move(terms), {}, {}, {0, 0}, {0, 0}});
        auto &bucket = entry.buckets[it->second];
        bucket.all.parts.push_back(k);
      }
    }
    if (const auto found = by_allocation.find(allocation); found != by_allocation.end())
      for (const auto &[klass, entry] : found->second) {
        Bucket bucket;
        bucket.all.parts = entry.all.parts;
        on[klass].buckets.push_back(std::move(bucket));
      }
    for (auto &[klass, entry] : on) {
      entry.objects.push_back(allocation);
      for (auto &bucket : entry.buckets) {
        for (const auto k : bucket.all.parts) {
          bucket.all.spans.push_back(Span(parts[k]));
          if (parts[k].write) bucket.writes.push_back(k);
          if (entry.alias.empty()) entry.alias = parts[k].footprint.alias;
        }
        bucket.all.Build();
        const auto hull = [&](const auto &list) {
          Interval h{0, 0};
          bool first = true;
          for (const auto k : list) {
            const auto span = Span(parts[k]);
            h = first ? span : Interval{std::min(h.lo, span.lo), std::max(h.hi, span.hi)};
            first = false;
          }
          return h;
        };
        bucket.write_hull = hull(bucket.writes);
        bucket.all_hull = hull(bucket.all.parts);
        entry.write |= !bucket.writes.empty();
        for (const auto &g : bucket.terms) {
          entry.uniform &= g.uniform;
          if (std::find(entry.objects.begin(), entry.objects.end(), g.object) == entry.objects.end())
            entry.objects.push_back(g.object);
        }
      }
      const auto &c = quotient.classes.at(klass);
      std::map<std::vector<std::int64_t>, std::size_t> index;
      for (std::uint32_t i = 0; i < c.units.size(); ++i) {
        std::vector<std::int64_t> delta(entry.objects.size(), 0);
        if (i > 0) {
          if (i >= c.delta.size()) return unknown("delta_missing:" + allocation);
          for (std::size_t o = 0; o < entry.objects.size(); ++o) {
            const auto found = c.delta[i].find(entry.objects[o]);
            if (found == c.delta[i].end()) return unknown("delta_missing:" + entry.objects[o]);
            delta[o] = found->second;
          }
        }
        const auto [it, added] = index.try_emplace(delta, entry.groups.size());
        if (added) entry.groups.push_back({std::move(delta), {}});
        entry.groups[it->second].units.push_back(i);
      }
    }
    // Lookup terms translated to a unit with δ `delta` over `objects`.
    const auto shift_terms = [](std::vector<GatherTerm> terms, const std::vector<std::string> &objects,
                                const std::vector<std::int64_t> &delta) {
      for (auto &g : terms) g.arg += I(delta[std::find(objects.begin(), objects.end(), g.object) - objects.begin()]);
      return terms;
    };
    const auto same_terms = [](const std::vector<GatherTerm> &x, const std::vector<GatherTerm> &y) {
      return std::equal(x.begin(), x.end(), y.begin(), y.end(), [](const GatherTerm &a, const GatherTerm &b) {
        return a.symbol == b.symbol && a.arg == b.arg && a.coefficient == b.coefficient;
      });
    };
    MemoryAccessFamily single;
    single.count = 1;
    // One region covering `hull` of a bucket placed in a unit: its translated terms over [lo, hi).
    const auto region = [&](const On &side, const Bucket &bucket, const Group &group,
                            const Interval &hull) -> std::optional<FamilyFootprint> {
      const I width = hull.hi - hull.lo;
      if (width <= 0 || width > I(UINT64_MAX)) return std::nullopt;
      FamilyFootprint f;
      f.space = "global";
      f.allocation = allocation;
      f.alias = side.alias;
      f.offset = hull.lo + I(group.delta.front());
      f.bytes = static_cast<std::uint64_t>(width);
      f.affine = true;
      f.gathers = shift_terms(bucket.terms, side.objects, group.delta);
      if (bucket.lookup) f.symbolic = "lookups";
      return f;
    };
    const auto placed = [&](const CrossUnitPart &part, const On &side, const Group &group) {
      auto footprint = part.footprint;
      footprint.offset += I(group.delta.front());
      footprint.gathers = shift_terms(footprint.gathers, side.objects, group.delta);
      return footprint;
    };
    for (const auto &[a, A] : on) {
      if (!A.write) continue;
      for (const auto &[b, B] : on) {
        if (!shared_class(a) && !shared_class(b)) continue;
        // Objects both classes look up, with the relative shifts that make some pair of their
        // lookups coincide (arg_p + δ_u = arg_q + δ_v); the key holds any other shift as "apart".
        std::vector<std::pair<std::size_t, std::size_t>> common; // (index in A.objects, in B.objects)
        for (std::size_t x = 1; x < A.objects.size(); ++x)
          if (const auto y = std::find(B.objects.begin() + 1, B.objects.end(), A.objects[x]); y != B.objects.end())
            common.emplace_back(x, static_cast<std::size_t>(y - B.objects.begin()));
        std::vector<std::set<std::int64_t>> meets(common.size());
        for (std::size_t c = 0; c < common.size(); ++c) {
          const auto &object = A.objects[common[c].first];
          for (const auto &P : A.buckets)
            for (const auto &x : P.terms)
              if (x.object == object)
                for (const auto &Q : B.buckets)
                  for (const auto &y : Q.terms)
                    if (y.symbol == x.symbol) meets[c].insert(std::int64_t(x.arg - y.arg));
        }
        const bool relative = A.uniform && B.uniform;
        std::set<std::vector<std::int64_t>> memo;
        for (std::size_t g = 0; g < A.groups.size(); ++g)
          for (std::size_t h = 0; h < B.groups.size(); ++h) {
            const auto &G = A.groups[g], &H = B.groups[h];
            // Unit pairs realized by the group pair, without same-unit and representative pairs.
            const std::uint64_t pairs = std::uint64_t(G.units.size()) * H.units.size();
            const bool zero = G.units.front() == 0 && H.units.front() == 0;
            const std::uint64_t excluded = a == b ? (g == h ? G.units.size() : 0) : (zero ? 1 : 0);
            if (pairs == excluded) continue;
            if (++unit_pairs > limits.unit_pairs) {
              CrossUnitOutcome cap;
              cap.status = Status::kCap;
              cap.reason = "unit_pairs";
              return cap;
            }
            std::int64_t d;
            if (__builtin_sub_overflow(H.delta.front(), G.delta.front(), &d)) return unknown("delta_overflow:" + allocation);
            std::vector<std::int64_t> key;
            if (relative) {
              key.push_back(d);
              for (std::size_t c = 0; c < common.size(); ++c) {
                std::int64_t t;
                if (__builtin_sub_overflow(H.delta[common[c].second], G.delta[common[c].first], &t))
                  return unknown("delta_overflow:" + allocation);
                const bool meet = meets[c].contains(t);
                key.push_back(meet);
                key.push_back(meet ? t : 0);
              }
            } else {
              key = {std::int64_t(g), std::int64_t(h)};
            }
            if (!memo.insert(std::move(key)).second) continue;
            for (const auto &P : A.buckets) {
              if (P.writes.empty()) continue;
              const auto x_terms = shift_terms(P.terms, A.objects, G.delta);
              for (const auto &Q : B.buckets) {
                if (!P.lookup && !Q.lookup) continue; // the constant pass
                if (P.lookup && Q.lookup && same_terms(x_terms, shift_terms(Q.terms, B.objects, H.delta))) {
                  // Equal lookups cancel: residual spans, exactly.
                  for (const auto k : P.writes) {
                    const auto &p = parts[k];
                    const auto span = Span(p);
                    std::optional<CrossUnitOutcome> found;
                    Q.all.Query(span.lo - I(d), span.hi - I(d), [&](std::size_t index) {
                      const auto &q = parts[Q.all.parts[index]];
                      if (++refinements > limits.refinements) {
                        CrossUnitOutcome cap;
                        cap.status = Status::kCap;
                        cap.reason = "refinements";
                        found = cap;
                        return false;
                      }
                      auto x = p.footprint, y = q.footprint;
                      x.symbolic.clear();
                      x.gathers.clear();
                      y.symbolic.clear();
                      y.gathers.clear();
                      y.offset += I(d);
                      const auto relation = ClassifyFamilyOverlap(p.family, x, q.family, y);
                      if (!relation.overlap.empty() || !relation.unknown.empty()) {
                        found = unknown("symbolic_overlap:" + allocation);
                        return false;
                      }
                      return true;
                    });
                    if (found) return std::move(*found);
                  }
                  continue;
                }
                const auto x = region(A, P, G, P.write_hull), y = region(B, Q, H, Q.all_hull);
                if (x && y && model->translatedFamilyOverlap(single, *x, single, *y).status == OverlapStatus::kDisjoint)
                  continue;
                // The hulls may meet: part by part.
                for (const auto k : P.writes) {
                  const auto &p = parts[k];
                  const auto px = placed(p, A, G);
                  for (const auto l : Q.all.parts) {
                    const auto &q = parts[l];
                    if (++decisions > limits.symbolic_decisions) {
                      CrossUnitOutcome cap;
                      cap.status = Status::kCap;
                      cap.reason = "symbolic_decisions";
                      return cap;
                    }
                    if (model->translatedFamilyOverlap(p.family, px, q.family, placed(q, B, H)).status !=
                        OverlapStatus::kDisjoint)
                      return unknown("symbolic_overlap:" + allocation);
                  }
                }
              }
            }
          }
      }
    }
  }
  return outcome;
}
} // namespace sentryward::algorithm
