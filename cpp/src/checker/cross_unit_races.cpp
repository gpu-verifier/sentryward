#include "sentryward/algorithm/checker/cross_unit_races.h"

#include "sentryward/algorithm/memory/model.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_set>
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
// Parts on one allocation, sorted by lo, with a max-hi tree (interval queries).
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
      if (end - begin == 1) return visit(begin); // maximum[node] == spans[begin].hi > lo
      const auto middle = begin + (end - begin) / 2;
      return self(self, 2 * node, begin, middle) && self(self, 2 * node + 1, middle, end);
    };
    return walk(walk, 1, 0, leaves);
  }
};
Interval Span(const CrossUnitPart &part) {
  const auto [lo, hi] = FootprintSpan(part.family, part.footprint);
  return {lo, hi};
}
// A part group's units on one allocation: (δ, unit index in the class), sorted.
using Deltas = std::vector<std::pair<std::int64_t, std::uint32_t>>;
// The runs of equal δ of a sorted list whose distinct values are first + step * k: run k holds
// positions [starts[k], starts[k + 1]). Nullopt when the distinct values are no progression.
struct Progression {
  std::vector<std::size_t> starts;
  I first = 0, step = 0; // step 0: one value
};
std::optional<Progression> AsProgression(const Deltas &list) {
  if (list.empty()) return std::nullopt;
  Progression result;
  result.first = list.front().first;
  for (std::size_t k = 0; k < list.size(); ++k) {
    if (k > 0 && list[k].first == list[k - 1].first) continue;
    const auto value = I(list[k].first);
    if (result.starts.size() == 1) result.step = value - result.first;
    else if (result.starts.size() > 1 && value - I(list[result.starts.back()].first) != result.step)
      return std::nullopt;
    result.starts.push_back(k);
  }
  result.starts.push_back(list.size());
  return result;
}
I FloorDiv(I a, I b) { // b > 0
  I q = a / b;
  if (a % b != 0 && a < 0) --q;
  return q;
}
// The window search over two progressions with one step s (a one-value list takes the other's):
// δ_v - δ_u = base + s * t for value indexes l - k = t, so the window low < d < high is one range
// of t for every u. Per t, its first unit pair in the search order (u by position, then v by
// position) that is neither one unit of one class nor the two representatives; the pairs in that
// order. O(|du| + |dv| + T log T) for the T < |du| + |dv| translations (t = 0 of one class scans
// its runs for two distinct units).
std::vector<std::pair<std::size_t, std::size_t>> FirstPairs(const Deltas &du, const Progression &pu, const Deltas &dv,
                                                            const Progression &pv, I step, I low, I high,
                                                            bool same_class) {
  const I n = I(pu.starts.size() - 1), m = I(pv.starts.size() - 1), base = pv.first - pu.first;
  I lowest = -(n - 1), highest = m - 1;
  if (step > 0) {
    lowest = std::max(lowest, FloorDiv(low - base, step) + 1);
    highest = std::min(highest, -FloorDiv(base - high, step) - 1);
  } else if (!(low < base && base < high)) {
    return {};
  }
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  for (I t = lowest; t <= highest; ++t) {
    bool found = false;
    for (I k = t < 0 ? -t : I(0); !found && k < n && k + t < m; ++k) {
      const auto u = static_cast<std::size_t>(k), v = static_cast<std::size_t>(k + t);
      for (auto p = pu.starts[u]; !found && p < pu.starts[u + 1]; ++p)
        for (auto q = pv.starts[v]; q < pv.starts[v + 1]; ++q) {
          const auto i = du[p].second, j = dv[q].second;
          if ((same_class && i == j) || (i == 0 && j == 0)) continue;
          pairs.emplace_back(p, q);
          found = true;
          break;
        }
    }
  }
  std::sort(pairs.begin(), pairs.end());
  return pairs;
}
// Unit-private spaces (owner-qualified or per thread): never shared across units.
bool UnitPrivateSpace(std::string_view space) noexcept {
  return space == "shared" || space == "local" || space == "tensor" || space == "register" || space == "param" ||
         space == "const" || space.starts_with("shared::");
}
} // namespace

CrossUnitOutcome CheckCrossUnit(std::span<const CrossUnitPart> parts, const UnitQuotient &quotient,
                                MemoryModel &model, const CrossUnitLimits &limits) {
  std::uint64_t unit_pairs = 0, refinements = 0, decisions = 0; // the work counted against `limits`
  const auto outcome = [](Status status, std::string reason) {
    CrossUnitOutcome result;
    result.status = status;
    result.reason = std::move(reason);
    return result;
  };
  const auto unknown = [&](std::string reason) { return outcome(Status::kUnknown, std::move(reason)); };
  const auto cap = [&](std::string reason) { return outcome(Status::kCap, std::move(reason)); };
  // No member unit: every unit pair is a representative pair (the normal checker's).
  if (!quotient.Shared()) return {};
  const auto shared_class = [&](std::uint32_t klass) { return quotient.classes.at(klass).units.size() > 1; };

  // Global parts, by allocation and part group; per alias the allocations that use it. A part group
  // is a class, or one piece of a class's piecewise δ on the allocation (UnitQuotient::PiecewiseDelta).
  struct PartGroup {
    std::uint32_t klass = 0, piece = 0; // piece kWhole: the class's one δ
    auto operator<=>(const PartGroup &) const = default;
  };
  constexpr std::uint32_t kWhole = UINT32_MAX;
  struct GroupParts {
    PartIndex all;
    std::vector<std::size_t> writes;
    std::vector<Interval> write_hull, all_hull;
  };
  std::map<std::string, std::map<PartGroup, GroupParts>> by_allocation;
  // Lookup parts (no constant span), by allocation and class.
  std::map<std::string, std::map<std::uint32_t, std::vector<std::size_t>>> lookups;
  std::map<std::string, std::set<std::string>> alias_allocations;
  std::map<std::string, bool> alias_write, alias_shared;
  bool unplaced = false, unplaced_write = false, unplaced_shared = false, any_shared_global = false,
       any_global_write = false;
  // The allocations some placeable global part writes: a pair on any other one holds no write.
  std::set<std::string> written;
  for (const auto &part : parts)
    if (part.write && !part.footprint.empty && part.family.count && part.footprint.space == "global" &&
        !part.footprint.allocation.empty())
      written.insert(part.footprint.allocation);
  for (std::size_t k = 0; k < parts.size(); ++k) {
    const auto &part = parts[k];
    const auto &fp = part.footprint;
    if (fp.empty || part.family.count == 0) continue;
    if (UnitPrivateSpace(fp.space)) continue;
    const auto klass = quotient.units.at(quotient.UnitOf(part.cta)).klass;
    const bool shared = shared_class(klass);
    const bool lookup = !fp.symbolic.empty() && !fp.gathers.empty();
    if (fp.space != "global" || fp.allocation.empty() || (!fp.symbolic.empty() && !lookup) || !fp.affine) {
      // Not placeable against a translated copy: it may meet any other unit's global access.
      unplaced = true;
      unplaced_write |= part.write;
      unplaced_shared |= shared;
      continue;
    }
    any_shared_global |= shared;
    any_global_write |= part.write;
    if (!fp.alias.empty()) {
      alias_allocations[fp.alias].insert(fp.allocation);
      alias_write[fp.alias] = alias_write[fp.alias] || part.write;
      alias_shared[fp.alias] = alias_shared[fp.alias] || shared;
    }
    if (!written.contains(fp.allocation)) continue;
    const auto &pieces = quotient.classes.at(klass).pieces;
    if (lookup) {
      // A lookup part is placed by one δ per object (the groups of the lookup pass).
      if (pieces.contains(fp.allocation)) return unknown("piecewise_lookup:" + fp.allocation);
      for (const auto &g : fp.gathers)
        if (pieces.contains(g.object)) return unknown("piecewise_lookup:" + g.object);
      lookups[fp.allocation][klass].push_back(k);
      continue;
    }
    PartGroup group{klass, kWhole};
    if (const auto found = pieces.find(fp.allocation); found != pieces.end()) {
      const auto piece = part.site ? found->second.PieceOf(*part.site) : std::nullopt;
      if (!piece) return unknown("piece_unknown:" + fp.allocation);
      group.piece = *piece;
    }
    auto &entry = by_allocation[fp.allocation][group];
    entry.all.parts.push_back(k);
    entry.all.spans.push_back(Span(part));
    if (part.write) entry.writes.push_back(k);
  }
  // An unplaceable global access paired with a member unit's access, one of them a write (an
  // Unknown involving a member restarts per unit).
  if (unplaced && (unplaced_shared || any_shared_global) && (unplaced_write || any_global_write))
    return unknown("unplaced_global_access");
  // Different allocations sharing an alias are Unknown to ClassifyFamilyOverlap (family_access.cpp).
  for (const auto &[alias, allocations] : alias_allocations)
    if (allocations.size() > 1 && alias_write[alias] && alias_shared[alias])
      return unknown("alias_group:" + alias);

  for (auto &[allocation, groups] : by_allocation) {
    // δ of every unit of every part group with parts here, sorted (the window search), and its runs
    // when the distinct values form a progression.
    std::map<PartGroup, Deltas> deltas;
    std::map<PartGroup, std::optional<Progression>> progressions;
    for (auto &[group, entry] : groups) {
      entry.all.Build();
      std::vector<Interval> writes;
      for (const auto k : entry.writes) writes.push_back(Span(parts[k]));
      entry.write_hull = Merge(std::move(writes));
      entry.all_hull = Merge(entry.all.spans);
      const auto &c = quotient.classes.at(group.klass);
      auto &list = deltas[group];
      list.reserve(c.units.size());
      for (std::size_t i = 0; i < c.units.size(); ++i) {
        if (i == 0) {
          list.emplace_back(0, 0);
          continue;
        }
        if (group.piece != kWhole) {
          list.emplace_back(c.pieces.at(allocation).delta.at(group.piece).at(i), static_cast<std::uint32_t>(i));
          continue;
        }
        if (i >= c.delta.size()) return unknown("delta_missing:" + allocation);
        const auto found = c.delta[i].find(allocation);
        if (found == c.delta[i].end()) return unknown("delta_missing:" + allocation);
        list.emplace_back(found->second, static_cast<std::uint32_t>(i));
      }
      std::sort(list.begin(), list.end());
      progressions[group] = AsProgression(list);
    }
    // Two pieces of one class in one of its member units: the representative's own check placed them
    // at d = 0, the member moves them apart by d = δ_u(Q) - δ_u(P). Their pairs may be ordered within
    // the unit, which this pass does not know, so any overlap with a write is Unknown. Each d once.
    for (const auto &[g, P] : groups) {
      if (g.piece == kWhole || P.write_hull.empty()) continue;
      const auto &table = quotient.classes.at(g.klass).pieces.at(allocation);
      for (const auto &[h, Q] : groups) {
        if (h.klass != g.klass || h.piece == g.piece) continue;
        std::unordered_set<std::int64_t> memo;
        for (std::size_t i = 1; i < table.delta.at(g.piece).size(); ++i) {
          std::int64_t d;
          if (__builtin_sub_overflow(table.delta.at(h.piece).at(i), table.delta.at(g.piece).at(i), &d))
            return unknown("delta_overflow:" + allocation);
          if (++unit_pairs > limits.unit_pairs) return cap("unit_pairs");
          if (d == 0 || !memo.insert(d).second) continue;
          if (!Meet(P.write_hull, Q.all_hull, I(d))) continue;
          for (const auto k : P.writes) {
            const auto &p = parts[k];
            const auto span = Span(p);
            std::optional<CrossUnitOutcome> found;
            Q.all.Query(span.lo - I(d), span.hi - I(d), [&](std::size_t index) {
              const auto &q = parts[Q.all.parts[index]];
              if (++refinements > limits.refinements) {
                found = cap("refinements");
                return false;
              }
              auto shifted = q.footprint;
              shifted.offset += I(d);
              const auto relation = ClassifyFamilyOverlap(p.family, p.footprint, q.family, shifted);
              if (!relation.overlap.empty() || !relation.unknown.empty()) {
                found = unknown("piece_overlap:" + allocation);
                return false;
              }
              return true;
            });
            if (found) return std::move(*found);
          }
        }
      }
    }
    for (const auto &[a, A] : groups) {
      if (A.write_hull.empty()) continue;
      for (const auto &[b, B] : groups) {
        if (B.all_hull.empty() || (!shared_class(a.klass) && !shared_class(b.klass))) continue;
        // d = δ_v - δ_u must lie strictly between these for the hulls to meet.
        const I low = A.write_hull.front().lo - B.all_hull.back().hi;
        const I high = A.write_hull.back().hi - B.all_hull.front().lo;
        const auto &du = deltas.at(a), &dv = deltas.at(b);
        // The translation d = δ_v - δ_u of the unit pair at positions (p_u, p_v); nullopt on overflow.
        const auto translation = [&](std::size_t p_u, std::size_t p_v) -> std::optional<std::int64_t> {
          std::int64_t d;
          if (__builtin_sub_overflow(dv[p_v].first, du[p_u].first, &d)) return std::nullopt;
          return d;
        };
        // Decides translation d for the unit pair at positions (p_u, p_v): nullopt when disjoint.
        const auto decide = [&](std::size_t p_u, std::size_t p_v, std::int64_t d) -> std::optional<CrossUnitOutcome> {
          const auto [delta_u, i] = du[p_u];
          const auto [delta_v, j] = dv[p_v];
          if (!Meet(A.write_hull, B.all_hull, I(d))) return std::nullopt;
          // Exact: every write part of A against the parts of B whose translated span meets it.
          for (const auto k : A.writes) {
            const auto &p = parts[k];
            const auto span = Span(p);
            std::optional<CrossUnitOutcome> found;
            B.all.Query(span.lo - I(d), span.hi - I(d), [&](std::size_t index) {
              const auto &q = parts[B.all.parts[index]];
              if (++refinements > limits.refinements) {
                found = cap("refinements");
                return false;
              }
              auto shifted = q.footprint;
              shifted.offset += I(d);
              auto relation = ClassifyFamilyOverlap(p.family, p.footprint, q.family, shifted);
              if (!relation.unknown.empty()) {
                found = unknown("family_overlap_unknown:" + allocation);
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
                  const auto &ca = quotient.classes.at(a.klass), &cb = quotient.classes.at(b.klass);
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
            if (found) return found;
          }
          return std::nullopt;
        };
        // Progressions with one step: each translation once, at the pair where the search below
        // first decides it, in the same order (the same first race).
        const auto &pu = progressions.at(a), &pv = progressions.at(b);
        if (pu && pv && (pu->step == 0 || pv->step == 0 || pu->step == pv->step)) {
          for (const auto &[p_u, p_v] :
               FirstPairs(du, *pu, dv, *pv, std::max(pu->step, pv->step), low, high, a.klass == b.klass)) {
            if (++unit_pairs > limits.unit_pairs) return cap("unit_pairs");
            const auto d = translation(p_u, p_v);
            if (!d) return unknown("delta_overflow:" + allocation);
            if (auto outcome = decide(p_u, p_v, *d)) return std::move(*outcome);
          }
          continue;
        }
        std::unordered_set<std::int64_t> memo; // d already decided disjoint (a meeting d returns)
        for (std::size_t p_u = 0; p_u < du.size(); ++p_u) {
          const auto [delta_u, i] = du[p_u];
          const I lo = I(delta_u) + low, hi = I(delta_u) + high;
          auto p_v = static_cast<std::size_t>(
              std::upper_bound(dv.begin(), dv.end(), lo, [](I x, const auto &e) { return x < I(e.first); }) - dv.begin());
          for (; p_v < dv.size() && I(dv[p_v].first) < hi; ++p_v) {
            const auto j = dv[p_v].second;
            if ((a.klass == b.klass && i == j) || (i == 0 && j == 0)) continue; // same unit; representative pair
            if (++unit_pairs > limits.unit_pairs) return cap("unit_pairs");
            const auto d = translation(p_u, p_v);
            if (!d) return unknown("delta_overflow:" + allocation);
            if (!memo.insert(*d).second) continue;
            if (auto outcome = decide(p_u, p_v, *d)) return std::move(*outcome);
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
      for (const auto &[group, entry] : found->second) {
        Bucket bucket;
        bucket.all.parts = entry.all.parts;
        on[group.klass].buckets.push_back(std::move(bucket));
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
            if (++unit_pairs > limits.unit_pairs) return cap("unit_pairs");
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
                    auto x = p.footprint;
                    x.symbolic.clear();
                    x.gathers.clear();
                    std::optional<CrossUnitOutcome> found;
                    Q.all.Query(span.lo - I(d), span.hi - I(d), [&](std::size_t index) {
                      const auto &q = parts[Q.all.parts[index]];
                      if (++refinements > limits.refinements) {
                        found = cap("refinements");
                        return false;
                      }
                      auto y = q.footprint;
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
                if (x && y && model.translatedFamilyOverlap(single, *x, single, *y).status == OverlapStatus::kDisjoint)
                  continue;
                // The hulls may meet: part by part.
                for (const auto k : P.writes) {
                  const auto &p = parts[k];
                  const auto px = placed(p, A, G);
                  for (const auto l : Q.all.parts) {
                    const auto &q = parts[l];
                    if (++decisions > limits.symbolic_decisions) return cap("symbolic_decisions");
                    if (model.translatedFamilyOverlap(p.family, px, q.family, placed(q, B, H)).status !=
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
  return {};
}
} // namespace sentryward::algorithm
