#pragma once
// Cross-CTA class mode: the question/answer recorder, the member replay of
// the logged questions (every query event of unit_facts.h except BindingRead
// and ReportBearing) and class-mode admission. Class mode admits a
// whole executed unit before any of its CTAs is consumed; members are never
// consumed. The replay runs against scratch tables, so admission itself
// writes nothing the analysis reads.
//
// Cost: the recorder does O(1) hashing per event of a recorded
// CTA (one virtual call per event; no record is copied: answers are interned
// per unit). Replay re-asks every logged question once per candidate class:
// value reads grouped by record (one Meta or ReadSlice per record; a Full read is
// re-read as a ReadSlice over its logged registers), slots one resolver call per
// logged slot on a scratch Resources per (consumer, Resources id).
// The structure key is O(paths x records + lanes) per CTA.
#include "sentryward/algorithm/checker/unit_quotient.h"
#include "sentryward/frontend/ptx/unit_classes.h"
#include "sentryward/frontend/ptx/unit_facts.h"

#include <cstdint>
#include <map>
#include <set>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sentryward::frontend::ptx {
// δ per global allocation of one unit relative to its class representative: one δ for an
// allocation that every slot moves alike, else its (site, δ) list, sorted by site
// (algorithm::PiecewiseDelta once the class is complete).
struct UnitDelta {
  std::map<std::string, std::int64_t> uniform;
  std::map<std::string, std::vector<std::pair<algorithm::UnitSite, std::int64_t>>> pieces;
  // Value lifting's δ: the unit's exact global offsets (plain ones and table-atom footprints' and
  // regions' values at its own elements) against the representative's, one per allocation; nullopt
  // when they are not (the unit's values cannot be lifted from the representative's).
  std::optional<std::map<std::string, std::int64_t>> valued = std::map<std::string, std::int64_t>{};
  // A containment member (unit_classes.h ContainmentProfile): lane by lane its records are its
  // representative's without some plain global accesses and compute, plus some taken branches, so
  // its events are the representative's minus those accesses. `occurrences`, per rank: each
  // instruction's executed-record count minus the representative's (lowering's opcode counts).
  bool contained = false;
  // A marked containment member: in some lane a tcgen05.ld of the member writes a register that a
  // removed record of its representative had read or overwritten while an asynchronous write of it was
  // pending. The representative's record ordered that pending write before the new one; the member
  // has no such order, so it may have two unordered writes of one register by one thread (and
  // unordered later accesses of that register) that its representative lacks. Only such register
  // pairs differ. The mark over-approximates (ContainMap::Align); a class with a marked member never
  // reports a clean memory result (unit_quotient.h ContainmentRestartReason), and a containment
  // member never lifts values.
  bool hazard = false;
  std::vector<std::map<std::uint32_t, std::int64_t>> occurrences;
  // A containment member's missing accesses, per rank and lane (the lane's index in its CTA): the
  // representative's active plain global accesses that the member removed or has predicated off, as
  // sorted (instruction, occurrence) pairs; lanes on one pair of paths share a list.
  std::vector<std::vector<std::shared_ptr<const std::vector<std::pair<std::uint32_t, std::uint32_t>>>>> removed;
  // A containment member's TMA loads that matched their representative's only without where they read
  // in global memory: the allocations they read, which no unit may write (the memory check restarts
  // per unit otherwise), and whose δ is not recorded.
  std::set<std::string> relaxed;
};

// One lane of a containment admission (query_log.cpp ContainMap::Align), exposed for unit tests:
// whether a member's record path is its representative's with removable records removed and taken
// branches added; whether the member is then marked (UnitDelta::hazard); and the representative's
// plain global accesses the member does not perform, as sorted (instruction, occurrence) pairs.
// `rep_origins` and the register lists are the paths' ContainmentProfile::origins / registers.
struct ContainmentAlignment {
  bool admitted = false, hazard = false;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> removed;
};
[[nodiscard]] ContainmentAlignment AlignContainmentPaths(
    const std::vector<ContainmentRecord> &rep, const std::vector<std::pair<std::uint32_t, std::uint32_t>> &rep_origins,
    const std::vector<std::uint32_t> &rep_registers, const std::vector<ContainmentRecord> &member,
    const std::vector<std::uint32_t> &member_registers);

// Founder planning for containment classes (the driver's pre-pass over every unit's profiles, before
// the class loop): the order in which the class loop takes the units, so that a unit that contains
// others by structure founds its class before them. Units with equal containment keys are taken by
// decreasing count of active records; one that no chosen founder contains lane by lane
// (ContainMap's record matching, without the founder's lowering state, which no unit has yet)
// becomes a planned founder. The order is the planned founders in unit order, then every other unit
// in unit order. It only orders: admission decides, exactly, what joins what.
// Paths are interned by content, so the plan holds each distinct path once: O(distinct paths x
// records) memory, O(units x lanes) to order plus one record matching per distinct pair of paths.
class ContainmentPlan {
public:
  ContainmentPlan();
  ~ContainmentPlan();
  ContainmentPlan(const ContainmentPlan &) = delete;
  ContainmentPlan &operator=(const ContainmentPlan &) = delete;
  // `profiles`: every rank's ComputeContainmentProfile of `unit`. One thread at a time.
  void Add(std::uint32_t unit, std::vector<ContainmentProfile> profiles);
  // The class loop's order of units 0..units-1, how many of its first entries are planned founders,
  // and whether some unit is planned to join a founder whose paths are not all its own (a containment
  // member, which never lifts values).
  struct Order {
    std::vector<std::uint32_t> units;
    std::size_t founders = 0;
    bool contains = false;
  };
  [[nodiscard]] Order Plan(std::uint32_t units) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The classes of one launch in class mode (the driver's unit-major loop).
class UnitClasses {
public:
  // `log_budget_bytes`: the logs' budget, min(2 GiB, 10% of --rss-limit-mib).
  UnitClasses(const UnitPlan &plan, std::uint64_t log_budget_bytes);
  ~UnitClasses();
  UnitClasses(const UnitClasses &) = delete;
  UnitClasses &operator=(const UnitClasses &) = delete;

  // Look-ahead admission (run.cpp's workers): `classes` classes existed (in creation order)
  // when PreAdmit ran; `joined` is the first of them admitting the unit, with its δ.
  struct PreAdmission {
    std::uint32_t classes = 0;
    std::optional<std::uint32_t> joined;
    UnitDelta delta;
  };
  // Admit's checks (below) of a unit against the classes founded so far, off the main thread (class
  // logs never change once founded; the snapshot shares them). Admit takes its result:
  // those classes are decided by it, later ones replayed there. `previous`: an earlier PreAdmit
  // of these facts (empty before the first), whose classes stay decided (its admission, if any,
  // stands).
  // `profiles`: every rank's ComputeContainmentProfile, or empty (no containment admission).
  [[nodiscard]] PreAdmission PreAdmit(std::span<const CtaFacts *const> facts,
                                      const std::vector<StructureKey> &keys, PreAdmission previous,
                                      std::span<const ContainmentProfile> profiles = {}) const;
  // Every rank of `unit` executed (`facts` in rank order, accounting done), none
  // consumed: equal structure keys on every rank, then equal answers to the
  // logged questions, equal slot outcomes up to δ (one per allocation, or one per
  // slot site in a class that admits a piecewise δ) and the logged alignment
  // checks (AbsoluteCheck) against each candidate class in creation order.
  // Returns the class the unit joins, or nullopt: the unit founds a class, each
  // rank consumed between Record() and Recorded().
  // A class founded after AdmitContainment with its founder's profiles also admits, when the
  // structure keys differ, a containment member (UnitDelta::contained): equal containment keys,
  // lane-by-lane containment, and the logged questions re-asked at the matching records of every
  // member path that a representative path's lanes follow (unit_classes.h ContainmentProfile).
  // `keys`: every rank's ComputeStructureKey; `pre`: PreAdmit's result on these facts; `profiles`:
  // as for PreAdmit (a founder keeps them).
  std::optional<std::uint32_t> Admit(std::uint32_t unit, std::span<const CtaFacts *const> facts,
                                     std::vector<StructureKey> keys, PreAdmission pre,
                                     std::vector<ContainmentProfile> profiles = {});
  // Admit of a unit the look-ahead admitted (`pre.joined`), which reads neither its facts nor its
  // keys: the unit joins that class (its facts may already be released).
  void Join(std::uint32_t unit, PreAdmission pre);
  // Classes founded from now on admit a unit whose slots move one allocation by different
  // amounts at different sites (a piecewise δ, UnitQuotient::PiecewiseDelta). The driver calls it
  // before the first unit, in every class-mode run. The cross-unit race pass decides piecewise δ;
  // value lifting does not represent it, so a value stage refuses lifting on such a quotient.
  void AdmitPiecewise() noexcept;
  // Classes founded from now on admit a unit whose exact global offsets are not its representative's
  // moved by one δ per allocation (UnitDelta::valued is nullopt; class mode's table atoms admit it on
  // the forms of its addresses). Its values cannot be lifted, so the driver calls it once no value
  // stage could complete anyway: the run lifts none, a consumed CTA's value view has a diagnostic, or
  // the per-unit value analysis is over the per-unit re-run budget (run.cpp PerUnitRestartBudget).
  void AdmitFormOnly() noexcept;
  // Some admitted member's exact global offsets are not its representative's moved by one δ per
  // allocation (UnitDelta::valued is nullopt; a member with a piecewise δ is counted by
  // PiecewiseAdmitted instead): no value stage can lift values. O(1).
  [[nodiscard]] bool ValuedUnliftable() const noexcept;
  // Some admitted member has a piecewise δ (UnitDelta::pieces non-empty): value lifting does not
  // represent it, so no value stage lifts values. O(1).
  [[nodiscard]] bool PiecewiseAdmitted() const noexcept;
  // Classes founded from now on (with their founder's profiles) admit containment members. A
  // member's values are not its representative's, so the driver calls it where it calls
  // AdmitFormOnly: no value stage could complete anyway.
  void AdmitContainment() noexcept;
  // Whether classes founded now admit containment members (the driver then computes profiles). O(1).
  [[nodiscard]] bool ContainmentOn() const noexcept;
  // The class loop takes `founders` planned founders first (ContainmentPlan): the futility rule (16
  // classes without an admission) counts from the classes after them.
  void ExpectFounders(std::size_t founders) noexcept;
  // Some admitted member is a containment member: no value stage lifts values. O(1).
  [[nodiscard]] bool ContainmentAdmitted() const noexcept;
  // A necessary condition for founder planning (ContainmentPlan) to change the classes: the loop has
  // admitted a containment member, or founded two classes whose founders have one containment key
  // (equal lanes and fixed records; they differ only in records a member may lack). Without either,
  // no unit is contained in another and the plan would be the unit order. O(1).
  [[nodiscard]] bool PlanUseful() const noexcept;
  // Some admitted member is marked (UnitDelta::hazard): its class never reports a clean memory
  // result. O(1).
  [[nodiscard]] bool MarkedAdmitted() const noexcept;
  // The founding unit's rank is about to be consumed: the recorder to attach
  // (null when no log is kept: futility).
  QuerySink *Record(std::uint32_t unit, std::uint32_t rank, const CtaFacts &facts);
  // ... and was consumed, the recorder detached. After the last rank the class
  // exists (its log sealed, or closed: it can admit no member).
  void Recorded(std::uint32_t unit, std::uint32_t rank);
  // After the last unit: the quotient tables of the analysed program (unit order).
  [[nodiscard]] algorithm::UnitQuotient Quotient() const;
  // The tensormap locality rule: every unit's tensormap.cp_fenceproxy regions are
  // constant, disjoint across units and overlap no input tensor map.
  [[nodiscard]] bool TensormapLocality() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace sentryward::frontend::ptx
