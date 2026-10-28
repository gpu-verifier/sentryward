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
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sentryward::frontend::ptx {
// δ per global allocation of one unit relative to its class representative.
using UnitDelta = std::map<std::string, std::int64_t>;

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
  // those classes are decided by it, later ones replayed there.
  [[nodiscard]] PreAdmission PreAdmit(std::span<const CtaFacts *const> facts,
                                      const std::vector<StructureKey> &keys) const;
  // Every rank of `unit` executed (`facts` in rank order, accounting done), none
  // consumed: equal structure keys on every rank, then equal answers to the
  // logged questions, equal slot outcomes up to δ and the logged alignment
  // checks (AbsoluteCheck) against each candidate class in creation order.
  // Returns the class the unit joins, or nullopt: the unit founds a class, each
  // rank consumed between Record() and Recorded().
  // `keys`: every rank's ComputeStructureKey; `pre`: PreAdmit's result on these facts.
  std::optional<std::uint32_t> Admit(std::uint32_t unit, std::span<const CtaFacts *const> facts,
                                     std::vector<StructureKey> keys, PreAdmission pre);
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
