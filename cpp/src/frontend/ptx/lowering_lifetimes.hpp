#pragma once
#include "lowering_impl.hpp"

namespace sentryward::frontend::ptx {
void ModelLowerer::Impl::Lifetime(Thread &t, const RecordView &e) {
  ProfileScope profile("frontend.model.lifetime");
  if (!tmem)
    throw SemanticError("tmem_allocation_identity", "no lifetime analysis");
  if (const auto *issue = BlockingTmemIssue(tmem->issues))
    ThrowTmemIssue(*issue);
  const auto &i = function.instructions[e.instruction_id()];
  auto key = Key(t.actor.warp.value(), e);
  if (i.semantics.operation == "alloc") {
    const TmemAllocation *allocation = nullptr;
    for (const auto &a : tmem->provenance.allocations)
      if (Key(a.site) == key) {
        if (allocation)
          throw SemanticError("tmem_allocation_identity",
                              "multiple lifetimes match allocation");
        allocation = &a;
      }
    if (!allocation)
      throw SemanticError("tmem_allocation_identity",
                          "no matching logical lifetime");
    std::vector<AsyncEffect> effects;
    if (t.actor.lane == 0) {
      auto a = resources->Bind("shared", "physical.shared.cta.0",
                               Unsigned(allocation->shared_slot,
                                        "TMEM publication slot", UINT32_MAX),
                               4);
      effects.push_back({{MemoryEffect{AccessMode::kWrite, a.first, 4}}});
      allocations[key] = Next();
    }
    auto id = Async(t, e, std::move(effects));
    auto tail = Emit(t, e, Await{{Ref(id)}});
    auto &align = aligned[key];
    align.calls[t.actor.lane] = {id, tail};
    align.completions = true;
  } else {
    auto id = Emit(t, e, Await{});
    aligned[key].calls[t.actor.lane] = {id, id};
  }
}
void ModelLowerer::Impl::LifetimeOrders() {
  if (!tmem)
    return;
  ProfileScope profile("frontend.model.lifetime_orders");
  // One immutable selection per source/allocation, shared by every target.
  // The target conjunction is unchanged; only its repeated payload is shared.
  using Effects = SharedOptional<std::vector<std::uint32_t>>;
  std::map<std::pair<InstId, std::string>, Effects> selected;
  std::map<std::string, SharedString> strings;
  const auto intern = [&](const std::string& text) -> const SharedString& {
    const auto [entry, inserted] = strings.try_emplace(text);
    if (inserted) entry->second = SharedString(text);
    return entry->second;
  };
  for (const auto &obligation : tmem->ordering_obligations) {
    const auto& kind = intern(obligation.kind);
    const auto& message = intern(obligation.message);
    std::vector<InstId> from, to;
    auto source_key = Key(obligation.source),
         target_key = Key(obligation.target);
    if (allocations.contains(source_key))
      from.push_back(allocations.at(source_key));
    else if (sites.contains(source_key))
      for (auto id : sites.at(source_key))
        if (source.program.instructions.holdsOperation<AsyncOp>(id.value()))
          from.push_back(id);
    std::set<ActorId> seen;
    if (sites.contains(target_key))
      for (auto id : sites.at(target_key))
        if (seen.insert(source.program.instructions.actor(id.value())).second)
          to.push_back(id);
    if (from.empty() || to.empty()) {
      Diagnostic("lifecycle_order_endpoint_missing", obligation.message, nullptr, nullptr,
          {{"constraint",obligation.kind},{"source_missing",from.empty()},{"target_missing",to.empty()},
           {"allocation",obligation.allocation.ResourceName()},
           {"source",{{"ptx_line",obligation.source.ptx_line},{"warp",obligation.source.warp},{"occurrence",obligation.source.occurrence}}},
           {"target",{{"ptx_line",obligation.target.ptx_line},{"warp",obligation.target.warp},{"occurrence",obligation.target.occurrence}}}});
      continue;
    }
    std::vector<MilestoneRef> targets;
    targets.reserve(to.size());
    for (auto id : to) targets.push_back(Ref(id, Milestone::kIssue));
    const SharedOptional<std::vector<MilestoneRef>> target_domain(std::move(targets));
    for (auto id : from) {
      Effects effects;
      if (obligation.kind == "access_completion_before_dealloc" &&
          !plans.contains(id.value())) {
        auto allocation =
            tmem->provenance.AllocationById(obligation.allocation);
        if (!allocation)
          throw SemanticError("unresolved_lifetime_footprint",
                              "lifetime allocation missing");
        auto name = resources->frame.Format(AllocationKey::Tensor(resources->frame.self(),
            allocation->columns == 512 ? std::string("physical.tmem.cta.0")
                                       : obligation.allocation.ResourceName())).allocation;
        auto [it, added] = selected.try_emplace({id, name});
        if (added) {
          std::vector<std::uint32_t> indices;
          bool tensor = false;
          const auto instruction = source.instruction(id);
          const auto &op = std::get<AsyncOp>(instruction.op);
          for (std::uint32_t k = 0; k < op.effects.size(); ++k) {
            bool match = false;
            for (const auto &access : op.effects[k].accesses)
              if (access.bytes)
                if (auto p = std::get_if<AddressId>(&access.endpoint)) {
                  const auto &binding = resources->Binding(*p);
                  if (binding.address_space == "tensor") {
                    tensor = true;
                    match |= binding.allocation == name;
                  }
                }
            if (match)
              indices.push_back(k);
          }
          if (tensor && indices.empty())
            throw SemanticError("unresolved_lifetime_footprint",
                                "retirement does not match allocation");
          it->second = std::move(indices);
        }
        effects = it->second;
      }
      orders.push_back({kind, Ref(id), target_domain->front(), message,
                        effects, target_domain});
    }
    Poll();
  }
}
InstId ModelLowerer::Impl::Copy(Thread &t, const RecordView &e,
                                std::vector<Fragment> reads,
                                std::vector<Fragment> writes,
                                std::optional<CompletionAction> completion) {
  auto id = Next();
  std::vector<AsyncEffect> effects;
  std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint32_t>> ranges;
  std::uint64_t cursor = 0, total = 0;
  for (auto [a, n] : writes) {
    (void)a;
    if (!n) throw SemanticError("copy_fragment_mapping", "copy byte extent is unknown");
    total += *n;
  }
  for (auto [a, n] : reads) {
    if (!n) throw SemanticError("copy_fragment_mapping", "copy byte extent is unknown");
    ranges.emplace_back(cursor, cursor + *n, effects.size());
    effects.push_back({{MemoryEffect{AccessMode::kRead, a, n}}});
    cursor += *n;
  }
  if (cursor > total)
    throw SemanticError("copy_fragment_mapping",
                        "copy source larger than destination");
  cursor = 0;
  for (auto [a, n] : writes) {
    const auto binding = resources->Binding(a);
    auto end = cursor + *n;
    std::set<std::uint64_t> points{cursor, end};
    for (auto [begin, stop, index] : ranges) {
      (void)index;
      if (cursor < begin && begin < end)
        points.insert(begin);
      if (cursor < stop && stop < end)
        points.insert(stop);
    }
    for (auto it = points.begin(), next = std::next(it); next != points.end();
         ++it, ++next) {
      std::vector<MilestoneRef> refs;
      for (auto [begin, stop, index] : ranges)
        if (begin <= *it && *it < stop)
          refs.push_back(Ref(id, Milestone::kEffect, index));
      // Interned even for an expression binding.
      auto shifted = resources->interner().Apply(
          ConstantStep{AllocationKey::Opaque(binding.address_space, binding.allocation, binding.alias_scope),
                       binding.offset + *it - cursor});
      if (!binding.expression.empty()) {
        auto expression = Json::parse(binding.expression);
        expression["address"]["offset"] = {{"operator", "add"},
            {"operands", Json::array({expression["address"]["offset"], *it - cursor})}};
        expression["byte_count"] = *next - *it;
        shifted = resources->interner().Apply(SerializedExpressionStep(binding.address_space, expression.dump(),
                                                                       binding.alias_scope, binding.allocation));
      }
      effects.push_back(
          {{MemoryEffect{AccessMode::kWrite, shifted, *next - *it}},
           {},
           std::move(refs)});
    }
    cursor = end;
  }
  if (completion) {
    std::vector<MilestoneRef> refs;
    for (std::uint32_t k = 0; k < effects.size(); ++k)
      refs.push_back(Ref(id, Milestone::kEffect, k));
    effects.push_back({{}, completion, std::move(refs)});
  }
  return Async(t, e, std::move(effects));
}
} // namespace sentryward::frontend::ptx
