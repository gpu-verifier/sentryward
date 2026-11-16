#pragma once
#include "execution_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
const std::vector<std::vector<std::uint32_t>> &Executor::RegisterUses() const {
  if (register_uses_.empty()) {
    register_uses_.resize(names_.size());
    // Reads only: the registers of every operand but a destination (operand 0 unless it is an
    // address), and the guard predicate.
    for (std::uint32_t index = 0; index < decoded_.size(); ++index) {
      const auto &d = decoded_[index];
      std::set<std::uint32_t> reads;
      for (std::size_t k = 0; k < d.operands.size(); ++k)
        if (k > 0 || d.operands[k].address || d.destinations.empty())
          reads.insert(d.operands[k].registers.begin(), d.operands[k].registers.end());
      if (d.predicate != kMissing) reads.insert(d.predicate);
      for (const auto reg : reads)
        if (reg < register_uses_.size()) register_uses_[reg].push_back(index);
    }
  }
  return register_uses_;
}
bool Executor::SharedFedPredicate(std::uint32_t pc) const {
  const auto predicate = decoded_[pc].predicate;
  if (predicate == kMissing) return false;
  if (shared_fed_.empty()) {
    const auto &uses = RegisterUses();
    shared_fed_.assign(names_.size(), 0);
    std::vector<std::uint32_t> work;
    const auto mark = [&](std::uint32_t reg) {
      if (reg < shared_fed_.size() && !shared_fed_[reg]) { shared_fed_[reg] = 1; work.push_back(reg); }
    };
    // Floating-point instructions end the chain: the shared-memory model forwards integers (constants,
    // pointers, bounded integers); a floating-point value read from shared memory is not one it can
    // decide a branch with, and routing such a CTA costs a shared-model run for nothing.
    const auto floating = [](const Instruction &i) {
      return std::any_of(i.modifiers.begin(), i.modifiers.end(), [](const std::string &m) {
        return In(m, {"f16", "f16x2", "bf16", "bf16x2", "f32", "f64", "tf32", "ftz", "e4m3", "e5m2",
                      "e4m3x2", "e5m2x2", "e2m1x2", "e2m3x2", "e3m2x2", "ue8m0x2"});
      });
    };
    for (const auto &d : decoded_) {
      const auto &i = *d.instruction;
      if (i.opcode == "ld" && (Has(i, "shared") || Has(i, "shared::cta") || Has(i, "shared::cluster")) && !floating(i))
        for (const auto reg : d.destinations) mark(reg);
    }
    // Each register enters the worklist once; each use edge is followed once per register.
    while (!work.empty()) {
      const auto reg = work.back();
      work.pop_back();
      for (const auto index : uses[reg])
        if (!floating(*decoded_[index].instruction))
          for (const auto out : decoded_[index].destinations) mark(out);
    }
  }
  return predicate < shared_fed_.size() && shared_fed_[predicate];
}
void Executor::RouteToSharedModel(const Instruction &instruction) const {
  ExecutionError error("shared_controlled_branch", instruction.line,
                       "undecided branch on a shared-memory value: run with the shared-memory model");
  error.shared_control = true;
  error.shared_routed = true;
  throw error;
}
bool Executor::DataOpaqueAdmitted(std::uint32_t pc) const {
  if (data_opaque_disabled) return false;
  if (const auto found = data_opaque_.find(pc); found != data_opaque_.end()) return found->second;
  data_opaque_[pc] = false; // a cycle through this branch refuses it
  RegisterUses();
  const auto &branch = *decoded_[pc].instruction;
  if (branch.control != "direct_branch" || !branch.branch_target || *branch.branch_target <= pc + 1) return false;
  const auto join = OpaqueRegionJoin(pc, true);
  if (!join) return false;
  std::set<std::uint32_t> tainted, seen_branches{pc};
  std::vector<std::uint32_t> work;
  for (auto index = pc + 1; index < *join; ++index)
    for (const auto reg : decoded_[index].destinations)
      if (reg < register_uses_.size() && tainted.insert(reg).second) work.push_back(reg);
  constexpr std::size_t kTaintLimit = 1u << 16;
  while (!work.empty()) {
    if (tainted.size() > kTaintLimit) return false;
    const auto reg = work.back();
    work.pop_back();
    for (const auto index : register_uses_[reg]) {
      if (index > pc && index < *join) continue; // inside the region
      const auto &d = decoded_[index];
      const auto &i = *d.instruction;
      if (d.predicate == reg) {
        // A forward branch on it: admitted only as an admitted region of its own.
        if (i.control == "direct_branch" && seen_branches.insert(index).second && DataOpaqueAdmitted(index)) continue;
        if (const auto found = data_opaque_.find(index);
            i.control == "direct_branch" && found != data_opaque_.end() && found->second)
          continue;
        return false;
      }
      if (i.control == "none" && i.semantics.category == "compute" && OpaqueRegionInstruction(i)) {
        for (const auto out : d.destinations)
          if (out < register_uses_.size() && tainted.insert(out).second) work.push_back(out);
        continue;
      }
      // The stored value of a store to memory (st, tcgen05.st, stmatrix; never its address).
      if ((i.opcode == "st" || i.opcode == "stmatrix" || (i.opcode == "tcgen05" && i.semantics.operation == "store")) &&
          !d.operands.empty() && d.operands[0].address &&
          std::find(d.operands[0].registers.begin(), d.operands[0].registers.end(), reg) == d.operands[0].registers.end())
        continue;
      return false;
    }
  }
  data_opaque_[pc] = true;
  return true;
}
std::optional<std::uint32_t> Executor::OpaqueRegionJoin(std::uint32_t pc, bool arms) const {
  const auto &branch = *decoded_[pc].instruction;
  if (!branch.branch_target || *branch.branch_target <= pc + (arms ? 0 : 1)) return {};
  auto join = *branch.branch_target;
  const auto depth = branch.scope.size();
  // Stores to one private .local array are admitted once their slots are proved; the first
  // unproved one is remembered for the region when nothing else refuses it.
  std::optional<std::string> store_refusal;
  const LocalRegionStore *array = nullptr;
  for (auto index = pc + 1; index < join; ++index) {
    if (index >= decoded_.size()) return {};
    const auto &candidate = *decoded_[index].instruction;
    if (arms) {
      if (candidate.scope.size() < depth || !std::equal(branch.scope.begin(), branch.scope.end(), candidate.scope.begin()))
        return {};
      bool shadows = false;
      for (const auto *list : {&decoded_[index].uses, &decoded_[index].destinations})
        for (auto reg : *list)
          if (ScopeLocal(candidate, reg, branch.scope, &shadows) && shadows) return {};
      if (candidate.control == "direct_branch") {
        if (!candidate.branch_target || *candidate.branch_target <= index) return {};
        join = std::max(join, *candidate.branch_target);
        continue;
      }
    }
    if (OpaqueRegionInstruction(candidate)) continue;
    if (!LocalStoreInstruction(candidate) || candidate.scope != branch.scope) return {};
    std::string refusal;
    const auto *store = ProvenLocalStore(index, refusal);
    if (store && array && (array->declaration.resource != store->declaration.resource ||
                           array->element_bytes != store->element_bytes)) {
      refusal = "the region stores to more than one .local array or element width";
      store = nullptr;
    }
    if (store) array = store;
    else if (!store_refusal) store_refusal = "st.local at PTX line " + std::to_string(candidate.line) + ": " + refusal;
  }
  const auto admit = [&]() -> std::optional<std::uint32_t> {
    if (!store_refusal) return join;
    region_store_refusals_.insert_or_assign(pc, *store_refusal);
    return {};
  };
  if (!arms) return admit();
  if (join >= decoded_.size()) return {};
  const auto &last = decoded_[join - 1].instruction->scope, &after = decoded_[join].instruction->scope;
  if (last.size() > depth && after.size() > depth && last[depth] == after[depth]) return {};
  return admit();
}
bool Executor::StrictlyDominates(std::uint32_t definition, std::uint32_t use) const {
  PrepareControl();
  if (control_irreducible_ || definition >= control_block_for_pc_.size() || use >= control_block_for_pc_.size())
    return false;
  auto block = control_block_for_pc_[use];
  const auto target = control_block_for_pc_[definition];
  if (block == target) return definition < use;
  while (block != kMissing && block != target && control_dom_[block] != block) block = control_dom_[block];
  return block == target;
}
const LocalRegionStore *Executor::ProvenLocalStore(std::uint32_t pc, std::string &refusal) const {
  if (!local_stores_) local_stores_ = std::make_unique<LocalRegionStores>(m_, f_);
  const auto *store = local_stores_->Shape(pc);
  auto [entry, added] = local_store_proofs_.try_emplace(pc);
  if (added)
    entry->second = store ? LocalStoreRefusal(pc, *store)
        : "its address is not a .local array of at most " + std::to_string(kLiftedLocalSlots) +
          " element slots plus integer registers, each with one definition";
  refusal = entry->second;
  return refusal.empty() ? store : nullptr;
}
// The byte offset of the store is offset + sum(index terms). Each term's value is evaluated
// once over its static definitions on a lane-independent scratch state: thread coordinates and
// the lane are bounded by the launch, a selp on an unknown predicate is the union of its
// alternatives and a shuffle any lane's value of its source, each a bounded value of its own
// register; every other instruction is executed by Data(). A definition contributes only when
// it is the register's one definition, unpredicated, and every definition it reads precedes it
// on every path (so it reads a defined value of the same last instance). Values bounded on one
// such register keep a table over it, so correlated terms (e - (e & -4)) stay exact.
std::string Executor::LocalStoreRefusal(std::uint32_t pc, const LocalRegionStore &store) const {
  if (register_definitions_.empty()) {
    register_definitions_.resize(names_.size());
    for (std::uint32_t index = 0; index < decoded_.size(); ++index)
      for (auto reg : decoded_[index].destinations) register_definitions_[reg].push_back(index);
  }
  const auto unique = [&](std::uint32_t reg) -> std::optional<std::uint32_t> {
    const auto &list = register_definitions_[reg];
    if (list.size() != 1) return {};
    return list.front();
  };
  // The registers an instruction reads (after its destination operand; `uses` also names the
  // destinations), or none when it reads one it writes.
  const auto sources = [&](const Decoded &d) -> std::optional<std::vector<std::uint32_t>> {
    std::vector<std::uint32_t> result;
    for (std::size_t k = d.destinations.empty() ? 0 : 1; k < d.operands.size(); ++k)
      for (auto reg : d.operands[k].registers) {
        if (std::find(d.destinations.begin(), d.destinations.end(), reg) != d.destinations.end()) return {};
        if (std::find(result.begin(), result.end(), reg) == result.end()) result.push_back(reg);
      }
    return result;
  };
  // The address chain: one definition each, every register it reads defined before it, and the
  // address register defined before the store.
  for (auto index : store.chain) {
    const auto read = sources(decoded_[index]);
    if (!read || decoded_[index].predicate != kMissing)
      return "its address chain is not defined before its use on every path";
    for (auto reg : *read)
      if (const auto definition = unique(reg); definition && !StrictlyDominates(*definition, index))
        return "its address chain is not defined before its use on every path";
  }
  if (store.chain.empty() || !StrictlyDominates(store.chain.back(), pc))
    return "its address register is not defined before the store on every path";
  if (c_.warps.empty()) return "the launch has no warp";
  State scratch = Initial(c_.warps.front(), 0);
  for (const auto &[name, value] : c_.warps.front().initial_registers) scratch.registers[ids_.at(name)] = Scalar{};
  std::set<std::uint32_t> coordinates;
  const auto bound = [&](std::string_view name, std::uint64_t count) {
    const auto reg = ids_.at(std::string(name));
    coordinates.insert(reg);
    scratch.registers[reg] = count <= 1 ? Scalar(Integer(0))
        : MakeBounded(Integer(0), Integer(count - 1), std::string(name), {});
  };
  bound("%laneid", c_.warp_size);
  bound("%warpid", c_.warps.size());
  bound("%tid.x", c_.block_dim[0]);
  bound("%tid.y", c_.block_dim[1]);
  bound("%tid.z", c_.block_dim[2]);
  const auto integral = [](const Scalar &value) {
    return std::holds_alternative<Integer>(value) || std::holds_alternative<BoundedInteger>(value) ||
           std::holds_alternative<bool>(value);
  };
  const auto range = [](const Scalar &value) -> std::optional<std::pair<Integer, Integer>> {
    if (const auto n = ScalarInteger(value); n && !std::holds_alternative<bool>(value)) return std::pair{*n, *n};
    if (const auto *b = std::get_if<BoundedInteger>(&value)) return std::pair{(*b)->lower, (*b)->upper};
    return {};
  };
  // A value in [lower, upper] that is named by `reg` alone.
  const auto fresh = [&](std::uint32_t reg, Integer lower, Integer upper) -> Scalar {
    if (lower == upper) return lower;
    return MakeBounded(std::move(lower), std::move(upper), "local-slot:" + names_[reg], {});
  };
  std::map<std::uint32_t, Scalar> values;
  std::size_t evaluated = 0;
  // One proof's caps: the definitions it evaluates (each once), which bound its cost on a large
  // slice, and the recursion depth. Past either, a value stays unknown and its term refuses.
  constexpr std::size_t kSliceLimit = 512;
  constexpr unsigned kSliceDepth = 64;
  const auto evaluate = [&](auto &&self, std::uint32_t reg, unsigned depth) -> Scalar {
    if (coordinates.contains(reg)) return scratch.registers[reg];
    if (const auto found = values.find(reg); found != values.end()) return found->second;
    Scalar result;
    const auto definition = unique(reg);
    if (definition && depth <= kSliceDepth && evaluated++ < kSliceLimit) {
      const auto &d = decoded_[*definition];
      const auto &i = *d.instruction;
      const bool compute = i.semantics.category == "compute" && !i.semantics.memory_effect &&
          (!WarpLevelOpcode(i.opcode) || i.opcode == "shfl");
      const auto read = sources(d);
      if (compute && read && d.predicate == kMissing && i.control == "none") {
        for (auto source : *read) {
          if (coordinates.contains(source)) continue;
          const auto from = unique(source);
          scratch.registers[source] = from && StrictlyDominates(*from, *definition)
              ? self(self, source, depth + 1) : Scalar{};
        }
        const auto operand = [&](std::size_t n) { return n < d.operands.size() ? Value(d.operands[n], scratch) : Scalar{}; };
        if (i.opcode == "selp" && d.operands.size() == 4 && !Truth(operand(3))) {
          const auto a = range(operand(1)), b = range(operand(2));
          if (a && b) result = fresh(reg, std::min(a->first, b->first), std::max(a->second, b->second));
        } else if (i.opcode == "shfl") {
          if (const auto source = range(operand(1)); source && !d.destinations.empty() && d.destinations.front() == reg)
            result = fresh(reg, source->first, source->second);
        } else {
          try {
            WarpContext context;
            Data(d, scratch, 0, 0, 0, Json::object(), context);
            result = scratch.registers[reg];
          } catch (const std::bad_alloc &) {
            throw;
          } catch (const std::exception &) {} // an instruction Data() refuses: the value stays unknown
        }
        if (!integral(result)) result = Scalar{};
      }
    }
    values.insert_or_assign(reg, result);
    return result;
  };
  Scalar offset = Integer(store.offset);
  for (const auto &term : store.index_terms) {
    const auto found = ids_.find(term);
    const auto value = found == ids_.end() ? Scalar{} : evaluate(evaluate, found->second, 0);
    if (!range(value)) return "index term " + term + " has no bound";
    offset = IntegerBinary("add", offset, value, nullptr);
  }
  const Integer width = Integer(store.components) * store.element_bytes, extent = store.declaration.bytes;
  // Byte offsets enumerated from a range (one check each); a wider range refuses the store.
  constexpr std::uint64_t kByteOffsetLimit = 256;
  std::vector<Integer> offsets;
  if (const auto n = ScalarInteger(offset)) offsets.push_back(*n);
  else if (const auto *b = std::get_if<BoundedInteger>(&offset)) {
    if (const auto table = TableOf(*b)) offsets = table->values;
    else if (store.element_bytes == 1 && (*b)->upper - (*b)->lower < Integer(kByteOffsetLimit))
      for (Integer n = (*b)->lower; n <= (*b)->upper; ++n) offsets.push_back(n);
  }
  if (offsets.empty()) return "its byte offset has no exact set of values";
  for (const auto &n : offsets)
    if (n < 0 || n + width > extent || n % store.element_bytes != 0)
      return "byte offset " + n.convert_to<std::string>() + " is not an aligned " + width.convert_to<std::string>() +
             "-byte access inside the " + std::to_string(store.declaration.bytes) + "-byte array " + store.declaration.name;
  return {};
}
void Executor::OpaqueRegionNames(std::uint32_t pc, std::uint32_t join, std::vector<std::string> &reads,
                                 std::vector<std::string> &writes) const {
  const auto &scope = decoded_[pc].instruction->scope;
  std::set<std::string> read, written;
  for (auto index = pc + 1; index < join; ++index) {
    const auto &skipped = decoded_[index];
    for (auto reg : skipped.destinations)
      if (!ScopeLocal(*skipped.instruction, reg, scope)) written.insert(names_[reg]);
    if (skipped.instruction->control == "direct_branch") {
      // A branch inside a multi-arm region reads only its guard; the target is a label.
      if (skipped.predicate != kMissing && !ScopeLocal(*skipped.instruction, skipped.predicate, scope))
        read.insert(names_[skipped.predicate]);
      continue;
    }
    for (auto reg : skipped.uses) {
      if (ScopeLocal(*skipped.instruction, reg, scope)) continue;
      bool source = skipped.predicate == reg;
      for (std::size_t operand = skipped.destinations.empty() ? 0 : 1; operand < skipped.operands.size(); ++operand)
        source |= std::find(skipped.operands[operand].registers.begin(), skipped.operands[operand].registers.end(), reg) !=
                  skipped.operands[operand].registers.end();
      if (source) read.insert(names_[reg]);
    }
  }
  reads.assign(read.begin(), read.end());
  writes.assign(written.begin(), written.end());
}
const Executor::MergedArm *Executor::MergedArmAt(std::uint32_t pc) const {
  auto [entry, inserted] = merged_arms_.try_emplace(pc);
  if (!inserted) return entry->second ? &*entry->second : nullptr;
  const auto &branch = *decoded_[pc].instruction;
  if (branch.control != "direct_branch" || decoded_[pc].predicate == kMissing || !branch.branch_target ||
      *branch.branch_target <= pc + 1 || *branch.branch_target > decoded_.size())
    return nullptr;
  MergedArm arm;
  std::set<std::uint32_t> writes;
  for (auto index = pc + 1; index < *branch.branch_target; ++index) {
    const auto &decoded = decoded_[index];
    const auto &candidate = *decoded.instruction;
    const auto &s = candidate.semantics;
    if (candidate.control != "none") return nullptr;
    // "compute" is also the default category; only these opcodes are register-only.
    const bool pure = s.category == "compute" && In(candidate.opcode, {"mov", "add", "sub", "mul", "mad", "fma",
        "div", "rem", "neg", "abs", "min", "max", "and", "or", "xor", "not", "cnot", "shl", "shr", "shf", "lop3",
        "prmt", "bfe", "bfi", "bfind", "brev", "clz", "popc", "cvt", "cvta", "selp", "setp", "set", "slct", "rcp",
        "sqrt", "rsqrt", "ex2", "lg2", "sin", "cos", "tanh", "copysign", "testp", "mul24", "mad24", "sad", "dp2a",
        "dp4a", "fns"});
    const bool private_read = s.category == "memory" && s.memory_mode == "read" && s.memory_space &&
        In(*s.memory_space, {"param", "local", "const"});
    const bool tmem = candidate.opcode == "tcgen05" && In(s.operation, {"load", "store", "wait_load", "wait_store"});
    if (!pure && !private_read && !tmem) return nullptr;
    arm.aligned |= tmem;
    writes.insert(decoded.destinations.begin(), decoded.destinations.end());
  }
  if (reference_ranges_.empty()) {
    reference_ranges_.assign(names_.size(), {UINT32_MAX, 0});
    for (std::uint32_t index = 0; index < decoded_.size(); ++index)
      for (const auto *list : {&decoded_[index].uses, &decoded_[index].destinations})
        for (auto reg : *list) {
          auto &range = reference_ranges_.at(reg);
          range = {std::min(range.first, index), std::max(range.second, index)};
        }
  }
  for (auto reg : writes)
    if (reference_ranges_.at(reg).first <= pc || reference_ranges_.at(reg).second >= *branch.branch_target)
      return nullptr;
  arm.writes.assign(writes.begin(), writes.end());
  arm.uniform = StaticWarpUniform(pc, decoded_[pc].predicate, 0);
  entry->second = std::move(arm);
  return &*entry->second;
}
// The static proof that `reg` is warp-uniform at `use`: its reaching definition lies in use's
// basic block (no label entry in between) and is an unpredicated setp/selp/not/and/or/xor/mov/cvt
// of constants and such registers, or a vote.sync (any/all/uni/ballot) whose mask is the full
// warp.
bool Executor::StaticWarpUniform(std::uint32_t use, std::uint32_t reg, unsigned depth) const {
  if (depth > 16) return false;
  for (auto index = use; index-- > 0;) {
    if (label_entries_.contains(index + 1)) return false;
    const auto &d = decoded_[index];
    if (std::find(d.destinations.begin(), d.destinations.end(), reg) == d.destinations.end()) continue;
    const auto &i = *d.instruction;
    if (d.predicate != kMissing) return false;
    if (i.opcode == "vote") {
      if (i.modifiers.size() != 3 || i.modifiers[0] != "sync" || !In(i.modifiers[1], {"any", "all", "uni", "ballot"}) ||
          d.operands.size() != 3)
        return false;
      const auto &mask = d.operands[2];
      if (mask.reg == kMissing && mask.registers.empty()) return FullWarpMask(mask.constant);
      if (mask.reg == kMissing) return false;
      // A register mask: its reaching definition is a mov of a full constant.
      for (auto at = index; at-- > 0;) {
        if (label_entries_.contains(at + 1)) return false;
        const auto &m = decoded_[at];
        if (std::find(m.destinations.begin(), m.destinations.end(), mask.reg) == m.destinations.end()) continue;
        return m.predicate == kMissing && m.instruction->opcode == "mov" && m.operands.size() == 2 &&
               m.operands[1].reg == kMissing && m.operands[1].registers.empty() && FullWarpMask(m.operands[1].constant);
      }
      return false;
    }
    if (!In(i.opcode, {"setp", "selp", "not", "and", "or", "xor", "mov", "cvt"}) || d.operands.empty()) return false;
    for (std::size_t operand = 1; operand < d.operands.size(); ++operand) {
      std::vector<std::uint32_t> sources = d.operands[operand].registers;
      if (d.operands[operand].reg != kMissing) sources.push_back(d.operands[operand].reg);
      for (auto source : sources)
        if (!StaticWarpUniform(index, source, depth + 1)) return false;
    }
    return true;
  }
  return false;
}

} // namespace
} // namespace sentryward::frontend::ptx
