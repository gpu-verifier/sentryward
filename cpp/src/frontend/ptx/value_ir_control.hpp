#pragma once
#include "value_ir_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
bool ValueTemplateExtractor::RunCollective(bool (ValueTemplateExtractor::*run)(), const char* code, bool& pending) {
  try { if (!(this->*run)()) return false; }
  catch (const std::exception& error) { Fail(code, error.what()); Invalidate(); }
  pending = false; ++cursor_; return true;
}

void ValueTemplateExtractor::BeginOrigin(const RecordView& record) {
  origin_ = {};
  origin_.cta = cta_;
  const auto coordinates = frame_.SelfCoordinates();
  origin_.cta_coordinates = std::array<std::uint64_t, 3>{coordinates[0], coordinates[1], coordinates[2]};
  SetThread(origin_, facts_, lane_);
  origin_.instruction_index = record.instruction_id();
  origin_.occurrence = record.occurrence();
  origin_.source_file = source_file_;
  origin_.function = function_name_;
}

bool ValueTemplateExtractor::Step() {
  if (Done()) return false;
  if (facts_.Expanded(lane_)) record_ = facts_.Meta({lane_, cursor_});
  else {
    const bool live = (*live_)[cursor_];
    current_record_ = facts_.Keep(live ? facts_.ReadFull({lane_, cursor_}) : facts_.CopyMeta({lane_, cursor_}));
    record_ = live ? facts_.ViewFull(current_record_, {lane_, cursor_})
                   : facts_.ViewPrototypeCopy(current_record_, {lane_, cursor_});
  }
  const auto record = *record_;
  if (merged_ && (record.instruction_id() <= merged_->branch || record.instruction_id() >= merged_->join)) FinishMergedArm();
  if (warp_mma_pending_) return RunCollective(&ValueTemplateExtractor::WarpMma, "unsupported_warp_mma", warp_mma_pending_);
  if (reduction_pending_) return RunCollective(&ValueTemplateExtractor::LaneReduction, LaneReductionCode(), reduction_pending_);
  if (matrix_pending_) return RunCollective(&ValueTemplateExtractor::MatrixMemory, "unsupported_matrix_memory", matrix_pending_);
  if (shuffle_source_) {
    if (!Shuffle()) return false;
    shuffle_source_.reset(); shuffle_selector_.reset(); ++cursor_; return true;
  }
  suboperation_ = 0;
  {
    BeginOrigin(record);
    for (const auto& [header, end] : loop_ends_) {
      if (record.instruction_id() < header || record.instruction_id() > end) continue;
      if (cursor_) {
        const auto previous = facts_.Meta({lane_, cursor_ - 1});
        if (previous.instruction_id() < header || previous.instruction_id() > end) iterations_[header] = 0;
        else if (record.instruction_id() == header && previous.instruction_id() >= header &&
                 function_.instructions[previous.instruction_id()].branch_target == header) ++iterations_[header];
      }
      origin_.unroll_context.emplace_back(header, iterations_[header]);
    }
    if (record.instruction_id() >= function_.instructions.size()) {
      Fail("invalid_execution_record", "Recorded instruction index is out of range"); ++cursor_; return true;
    }
    instruction_ = &function_.instructions[record.instruction_id()];
    origin_.source_line = instruction_->line;
    const auto& static_origin = instruction_origins_[record.instruction_id()];
    origin_.source_text = static_origin.source_text;
    origin_.scope = static_origin.scope;
    origin_.predicate = static_origin.predicate;
    origin_.executed = record.predicate_value().value_or(true);
    const auto reached_group = graph_.observeOccurrence(origin_);
    if (!(*live_)[cursor_]) { ++cursor_; return true; }
    if (const auto join = record.merged_arm_join()) {
      // The arm runs when the branch is not taken.
      std::optional<Expression> runs;
      if (instruction_->predicate) {
        runs = Operand(*instruction_->predicate, Pred());
        if (runs) runs = Reinterpret(*runs, Pred());
        if (runs && !instruction_->predicate_negated)
          runs = Expression{Operation("ptx.not", "not.pred", {Pred()}, {*runs}).result(), Pred()};
      }
      if (runs) merged_.emplace(MergedArmValues{instruction_->id, *join, *runs, {}, {}});
      else Fail("two_arm_merge_values", "The predicate of a merged one-armed diamond has no value");
      ++cursor_; return true;
    }
    if (!record.opaque_compute_diamond_indices().empty()) {
      const auto branch = instruction_;
      const auto origin = origin_;
      bool lifted = false;
      lifting_ = true;
      try { lifted = LiftRegion(); }
      catch (const std::invalid_argument& error) { Fail("invalid_value_operation", error.what()); }
      lifting_ = false; instruction_ = branch; origin_ = origin;
      if (lifted) { ++cursor_; return true; }
      for (const auto &name : record.opaque_compute_register_writes()) {
        const auto key = Resolve(name, instruction_->scope);
        registers_.erase(key);
        invalidated_.insert(key);
      }
      for (auto index : record.opaque_compute_diamond_indices())
        if (index < function_.instructions.size()) {
          const auto &skipped = function_.instructions[index];
          for (const auto &name : skipped.destinations) {
            const auto key = Resolve(name, skipped.scope);
            registers_.erase(key);
            invalidated_.insert(key);
          }
        }
      carry_.reset();
      ++cursor_; return true;
    }
    if (record.collapsed_wait_loop() || record.canonical_wait_probe()) {
      if (!record.normalized_wait_result()) {
        Fail("missing_wait_summary", "Normalized polling loop lacks its return-value summary");
        Invalidate();
      } else {
        for (const auto& destination : instruction_->destinations)
          Bind(destination, Constant(Pred(), *record.normalized_wait_result() ? "true" : "false"));
      }
      ++cursor_; return true;
    }
    if (record.predicate_value() == false) { ++cursor_; return true; }
    if (instruction_->predicate && !record.predicate_value()) {
      Fail("unknown_execution_predicate", "The active instruction path is not fixed"); Invalidate(); ++cursor_; return true;
    }
    if (branch_capture_ && record.branch_scope_decision()) {
      try {
        if (instruction_->control != "direct_branch" || !instruction_->predicate)
          throw std::invalid_argument("branch capture does not name a conditional direct branch");
        const auto name = *instruction_->predicate;
        const auto value = registers_.find(Resolve(name, instruction_->scope));
        const auto producer = facts_.Origin(record, name);
        if (value == registers_.end() || value->second.missing_definition ||
            !value->second.components.empty() || !producer)
          throw std::invalid_argument("pre-decision predicate definition is unavailable");
        branch_capture_->observe({value->second.value, reached_group, origin_, name,
            *producer, *record.branch_scope_decision(), instruction_->predicate_negated});
      } catch (const std::exception& error) {
        // Optional proof evidence must not change ordinary value semantics.
        // An incomplete capture is never published or used as an assumption.
        branch_capture_->unavailable(error.what());
      }
      ++cursor_; return true;
    }
    if (instruction_->opcode=="mma")
      return RunCollective(&ValueTemplateExtractor::WarpMma, "unsupported_warp_mma", warp_mma_pending_);
    if (In(instruction_->opcode, {"ldmatrix", "stmatrix"}))
      return RunCollective(&ValueTemplateExtractor::MatrixMemory, "unsupported_matrix_memory", matrix_pending_);
    if (instruction_->opcode == "shfl") {
      try {
        if (!Shuffle()) return false;
      } catch (const std::exception& error) { Fail("unsupported_shuffle", error.what()); Invalidate(); }
      shuffle_source_.reset(); shuffle_selector_.reset(); ++cursor_; return true;
    }
    if (In(instruction_->opcode, {"vote", "redux"}) && shuffle_values && !demand_mode)
      return RunCollective(&ValueTemplateExtractor::LaneReduction, LaneReductionCode(), reduction_pending_);
    if (merged_)
      for (const auto& name : instruction_->destinations)
        if (name != "_")
          if (const auto key = Resolve(name, instruction_->scope); !merged_->before.contains(key))
            merged_->before.emplace(key, CurrentValue(key));
    try { InstructionValue(); }
    catch (const std::invalid_argument &error) {
      Fail("invalid_value_operation", error.what()); Invalidate();
    }
  }
  ++cursor_; return true;
}
void ValueTemplateExtractor::FinishMergedArm() {
  const auto arm = std::move(*merged_);
  merged_.reset();
  for (const auto& [key, prior] : arm.before) {
    const auto found = registers_.find(key);
    if (found == registers_.end()) continue;
    if (const auto merged = prior ? Choose(arm.runs, found->second, *prior) : std::nullopt) found->second = *merged;
    else { registers_.erase(found); invalidated_.insert(key); }
  }
}
std::optional<Expression> ValueTemplateExtractor::Choose(const Expression& condition, const Expression& value,
                                                         const Expression& other) {
  const bool scalar = value.components.empty() && other.components.empty() && !value.missing_definition &&
      !other.missing_definition;
  if (scalar && value.value == other.value) return value;
  auto updated = scalar ? std::optional<Expression>(value) : Bits(value);
  auto previous = scalar ? std::optional<Expression>(other) : Bits(other);
  // The alternatives meet in the typed one of the two (b32 and f32 meet in f32).
  if (updated && previous && previous->type.value != updated->type.value) {
    if (updated->type.value.element().kind() == v::ElementKind::Bit) updated = Reinterpret(*updated, previous->type);
    else previous = Reinterpret(*previous, updated->type);
  }
  if (!updated || !previous) return {};
  auto selected = Add(std::make_unique<v::SelectOp>(updated->type.value),
      {condition.value, updated->value, previous->value}, origin_).result();
  if (const auto& pointer = updated->value.pointer(); pointer && previous->value.pointer() && *pointer == *previous->value.pointer())
    selected = graph_.asPointer(selected, *pointer);
  return Expression{selected, updated->type};
}
// Execution skipped the region from a forward branch to its join on an unresolved predicate;
// every edge inside is forward. One program-order pass gives each instruction its path: the
// fall-through enters on the negated branch guard, a conditional edge adds its guard, a join
// takes the disjunction and a guarded instruction adds its own guard. Each instruction is
// lowered as usual on the values its sources have on its path. A register the region assigns
// has, on a path, its last assignment's value where that assignment covers the path, else
// select(assignment path, value, the earlier value on the rest of the path). A register
// without a value before the region is read only where its assignments cover the path, and
// keeps a value after the join only if they cover every path.
// Size limits of a lifted region: guard terms of one instruction's path, and path splits of
// all its register values.
constexpr std::size_t kLiftedPathTerms = 64;
constexpr std::size_t kLiftedValueSplits = 4096;
bool ValueTemplateExtractor::LiftRegion() {
  const auto& branch = *instruction_;
  const auto& indices = record_->opaque_compute_diamond_indices();
  const auto join = indices.back() + 1;
  const auto refuse = [&](const char* code, const std::string& message) { Fail(code, message); return false; };
  if (!branch.predicate || !branch.branch_target || indices.front() != branch.id + 1 ||
      indices.size() != join - indices.front() || *branch.branch_target > join || join > function_.instructions.size())
    return refuse("lifted_region_shape", "Opaque compute region is not the contiguous forward region of its branch");
  struct Assignment { GuardPath path; Expression value; };
  struct Lifted { std::optional<Expression> prior; std::vector<Assignment> assignments; };
  std::map<Key, Lifted> lifted;
  std::map<std::uint32_t, Expression> guards;
  std::map<std::tuple<Key, std::size_t, GuardPath>, std::optional<Expression>> values;
  std::size_t splits = 0;
  bool limited = false;
  // `key` on a nonempty `path` after its first `count` assignments: the last assignment that
  // covers the path, the earlier value where it is disjoint, else a select on the first guard
  // the path leaves open (the region's own branch structure, gated-SSA form).
  const auto value = [&](auto&& self, const Key& key, const Lifted& entry, std::size_t count,
                         const GuardPath& path) -> std::optional<Expression> {
    if (!count) return entry.prior;
    const auto& last = entry.assignments[count - 1];
    if (Implies(path, last.path)) return last.value;
    if (Disjoint(path, last.path)) return self(self, key, entry, count - 1, path);
    const auto cached = std::tuple{key, count, path};
    if (const auto found = values.find(cached); found != values.end()) return found->second;
    const auto open = Open(path, last.path);
    if (!open || ++splits > kLiftedValueSplits) { limited = true; return {}; }
    const auto taken = self(self, key, entry, count, Conjoin(path, {*open, true}));
    const auto other = taken ? self(self, key, entry, count, Conjoin(path, {*open, false})) : std::nullopt;
    const auto result = other ? Choose(guards.at(*open), *taken, *other) : std::nullopt;
    return values.emplace(cached, result).first->second;
  };
  // A nonempty path's value of `key` (lifted), or a refusal.
  const auto read = [&](const Key& key, const GuardPath& path, const std::string& name) -> std::optional<Expression> {
    const auto diagnostics = diagnostics_.size();
    auto result = value(value, key, lifted.at(key), lifted.at(key).assignments.size(), path);
    if (!result && diagnostics_.size() == diagnostics)
      refuse(limited ? "lifted_region_path_limit" : "lifted_region_unassigned_register",
             limited ? "A path or value of the lifted region exceeds its size limit"
                     : "Register '" + name + "' may be unassigned on a path of the lifted region that reads it");
    return result;
  };
  // The guard of `instruction` (instruction_) on `path`: the value of literal (instruction, true).
  const auto guard = [&](const Instruction& instruction, const GuardPath& path) {
    const auto key = Resolve(*instruction.predicate, instruction.scope);
    auto truth = lifted.contains(key) ? read(key, path, *instruction.predicate) : Operand(*instruction.predicate, Pred());
    if (truth) truth = Reinterpret(*truth, Pred());
    if (!truth) return false;
    guards.insert_or_assign(instruction.id, *truth);
    return true;
  };
  std::map<std::uint32_t, GuardPath> reach;
  const auto flow = [&](std::uint32_t target, const GuardPath& path) {
    if (path.empty() || target >= join) return true;
    auto& entry = reach[target];
    entry = Disjoin(std::move(entry), path);
    return entry.size() <= kLiftedPathTerms ||
           refuse("lifted_region_path_limit", "A path of the lifted region exceeds " + std::to_string(kLiftedPathTerms) +
                  " guard terms");
  };
  // A st.local the executor proved (LocalRegionStore): its path, byte offset and data. After the
  // walk every element slot j of the array is read once, takes each store in program order as
  // select(offset == (j - c) * e, component c, the slot) under the gated form of the store's path,
  // and is written once: the one read and one write of the array lowering records for the region.
  // O(stores x components x slots) nodes.
  struct LocalStoreEffect {
    const Instruction* instruction;
    const LocalRegionStore* store;
    GuardPath path;
    Expression offset;
    std::vector<Expression> data;
  };
  std::vector<LocalStoreEffect> local_effects;
  const auto store_local = [&](const Instruction& instruction, const GuardPath& executed) {
    const auto* store = local_stores_.Shape(instruction.id);
    std::optional<Type> type;
    for (const auto& modifier : instruction.modifiers)
      if (auto parsed = PtxType(modifier)) type = *parsed;
    auto text = Trim(module_.view(instruction.operands.at(1)));
    if (text.starts_with('{') && text.ends_with('}')) text = text.substr(1, text.size() - 2);
    const auto parts = Split(text, ',');
    const auto* first = local_effects.empty() ? store : local_effects.front().store;
    if (!store || !type || type->elements != 1 || parts.size() != store->components ||
        type->bits != 8 * store->element_bytes || first->declaration.resource != store->declaration.resource ||
        first->element_bytes != store->element_bytes)
      return refuse("lifted_region_local_store", "The st.local inside the lifted region has no element-slot shape of "
                    "the region's one .local array");
    const auto diagnostics = diagnostics_.size();
    const auto failed = [&] {
      if (diagnostics_.size() == diagnostics)
        refuse("lifted_region_local_store", "The st.local inside the lifted region has no value for its offset or data");
      return false;
    };
    const auto u64 = U64();
    auto offset = Constant(u64, Integer(store->offset & ((Integer(1) << 64) - 1)).convert_to<std::string>());
    for (const auto& term : store->index_terms) {
      const auto key = Resolve(term, instruction.scope);
      auto part = lifted.contains(key) ? read(key, executed, term) : Operand(term, u64);
      if (part) part = Reinterpret(*part, u64);
      if (!part) return failed();
      offset = {Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::Add, u64.value), {offset.value, part->value}, origin_).result(), u64};
    }
    std::vector<Expression> data;
    for (const auto& part : parts) {
      auto component = Operand(Trim(part), *type);
      if (component && type->value.isScalar() && type->value.element().isBitLike() &&
          component->type.value.bitSize() > type->value.bitSize())
        component = Part(*component, *type, 0);
      if (component) component = Reinterpret(*component, *type);
      if (!component) return failed();
      data.push_back(*component);
    }
    local_effects.push_back({&instruction, store, executed, offset, std::move(data)});
    return true;
  };
  const auto apply_local = [&]() {
    const auto& array = *local_effects.front().store;
    instruction_ = local_effects.front().instruction;
    const auto type = local_effects.front().data.front().type;
    const auto bytes = array.element_bytes;
    for (std::uint64_t slot = 0; slot < array.Slots(); ++slot) {
      auto address = Address("[" + array.declaration.name + "+" + std::to_string(slot * bytes) + "]");
      if (!address) return false;
      Expression current{Add(std::make_unique<v::LoadOp>(type.value), {address->value}, origin_).result(), type};
      for (std::size_t n = 0; n < local_effects.size(); ++n) {
        const auto& effect = local_effects[n];
        for (unsigned component = 0; component < effect.store->components && component <= slot; ++component) {
          const auto target = Constant(U64(), std::to_string((slot - component) * bytes));
          const auto hit = Add(std::make_unique<v::CompareOp>(v::ComparePredicate::Equal, Pred().value),
                               {effect.offset.value, target.value}, origin_).result();
          auto data = Reinterpret(effect.data[component], type);
          if (!data) return false;
          const Expression updated{Add(std::make_unique<v::SelectOp>(type.value), {hit, data->value, current.value}, origin_).result(),
                                   type};
          const Lifted entry{current, {{effect.path, updated}}};
          const Key key{effect.instruction->scope, "local-slot:" + std::to_string(n) + ':' + std::to_string(component) +
                                                   ':' + std::to_string(slot)};
          auto next = value(value, key, entry, 1, GuardPath{GuardTerm{}});
          if (next) next = Reinterpret(*next, type);
          if (!next)
            return limited ? refuse("lifted_region_path_limit", "A path or value of the lifted region exceeds its size limit")
                           : refuse("lifted_region_local_store", "A slot of the region's .local array has no merged value");
          current = *next;
        }
      }
      Add(std::make_unique<v::StoreOp>(), {address->value, current.value}, origin_);
    }
    return true;
  };
  if (!guard(branch, GuardPath{GuardTerm{}}) ||
      !flow(*branch.branch_target, GuardPath{GuardTerm{GuardLiteral{branch.id, !branch.predicate_negated}}}) ||
      !flow(branch.id + 1, GuardPath{GuardTerm{GuardLiteral{branch.id, branch.predicate_negated}}}))
    return false;
  for (auto index = branch.id + 1; index < join; ++index) {
    const auto at = reach.find(index);
    if (at == reach.end()) continue;
    const auto path = std::move(at->second);
    reach.erase(at);
    const auto& instruction = function_.instructions[index];
    instruction_ = &instruction;
    origin_.source_line = instruction.line;
    origin_.source_text = instruction_origins_[index].source_text;
    origin_.predicate = instruction_origins_[index].predicate;
    if (instruction.control == "direct_branch") {
      const auto target = instruction.branch_target;
      if (!target || *target <= index || *target > join)
        return refuse("lifted_region_control", "A branch inside the lifted region is not forward to at most its join");
      if (!instruction.predicate) { if (!flow(*target, path)) return false; continue; }
      if (!guard(instruction, path) || !flow(*target, Conjoin(path, {index, !instruction.predicate_negated})) ||
          !flow(index + 1, Conjoin(path, {index, instruction.predicate_negated})))
        return false;
      continue;
    }
    if (instruction.control != "none")
      return refuse("lifted_region_control", "Control instruction '" + instruction.opcode + "' inside the lifted region");
    const auto& semantics = instruction.semantics;
    const bool parameter = semantics.category == "memory" && In(instruction.opcode, {"ld", "ldu"}) &&
        semantics.memory_space == "param";
    const bool local_store = LocalStoreInstruction(instruction);
    if (semantics.category == "memory" && !parameter && !local_store)
      return refuse("lifted_region_memory_access", "'" + instruction.opcode + "' inside the lifted region has no recorded "
                    "memory access; only parameter reads and .local stores are lowered");
    // Warp-level operations read other lanes or the active mask, which the region's paths do not determine.
    if (!parameter && !local_store &&
        (semantics.category != "compute" || semantics.memory_effect || WarpLevelOpcode(instruction.opcode)))
      return refuse("lifted_region_unsupported_instruction", "'" + instruction.opcode + "' inside the lifted region is not "
                    "per-thread register computation");
    if (Has(instruction, "cc") || In(instruction.opcode, {"addc", "subc", "madc"}))
      return refuse("lifted_region_carry", "Carry-flag arithmetic inside the lifted region");
    auto executed = path;
    if (instruction.predicate) {
      if (!guard(instruction, path)) return false;
      executed = Conjoin(path, {index, !instruction.predicate_negated});
    }
    if (!flow(index + 1, path)) return false;
    if (executed.empty()) continue;
    // Region-assigned registers are absent from registers_ (and invalidated) between
    // instructions; the sources take their value on this path while the instruction lowers.
    std::vector<Key> present;
    for (std::size_t operand = instruction.destinations.empty() ? 0 : 1; operand < instruction.operands.size(); ++operand)
      for (const auto& name : Registers(module_.view(instruction.operands[operand]))) {
        const auto key = Resolve(name, instruction.scope);
        if (!lifted.contains(key) || registers_.contains(key)) continue;
        auto source = read(key, executed, name);
        if (!source) return false;
        registers_.insert_or_assign(key, *source);
        present.push_back(key);
      }
    std::vector<Key> written;
    for (const auto& name : instruction.destinations)
      if (name != "_") {
        const auto key = Resolve(name, instruction.scope);
        if (!lifted.contains(key)) lifted[key].prior = CurrentValue(key);
        written.push_back(key);
      }
    const auto diagnostics = diagnostics_.size();
    if (local_store) { if (!store_local(instruction, executed)) return false; }
    else InstructionValue();
    std::vector<std::optional<Expression>> results;
    for (const auto& key : written) {
      const auto found = registers_.find(key);
      results.push_back(found == registers_.end() ? std::nullopt : std::optional<Expression>(found->second));
    }
    for (const auto* keys : {&present, &written})
      for (const auto& key : *keys) { registers_.erase(key); invalidated_.insert(key); }
    if (diagnostics_.size() != diagnostics) return false;
    for (std::size_t n = 0; n < written.size(); ++n) {
      if (!results[n]) return refuse("lifted_region_value", "Lowering defined no value for '" + written[n].second + "'");
      lifted.at(written[n]).assignments.push_back({executed, *results[n]});
    }
  }
  if (!local_effects.empty() && !apply_local()) return false;
  for (const auto& [key, entry] : lifted) {
    const auto result = value(value, key, entry, entry.assignments.size(), GuardPath{GuardTerm{}});
    if (limited) return refuse("lifted_region_path_limit", "A path or value of the lifted region exceeds its size limit");
    if (result) { registers_.insert_or_assign(key, *result); invalidated_.erase(key); }
  }
  return true;
}
} // namespace
} // namespace sentryward::frontend::ptx
