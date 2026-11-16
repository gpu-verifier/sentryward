#pragma once
#include "execution_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
void Executor::PrepareControl() const {
  if (control_ready_)
    return;
  control_ready_ = true;
  const auto size = static_cast<std::uint32_t>(decoded_.size());
  std::set<std::uint32_t> leaders{0, size};
  for (std::uint32_t pc = 0; pc < size; ++pc) {
    const auto &i = *decoded_[pc].instruction;
    if (i.branch_target)
      leaders.insert(*i.branch_target);
    if (i.control != "none" && i.control != "call")
      leaders.insert(pc + 1);
  }
  control_block_for_pc_.resize(size);
  for (auto it = leaders.begin(); std::next(it) != leaders.end(); ++it) {
    ControlBlock block;
    block.begin = *it;
    block.end = *std::next(it);
    if (block.end > size) {
      control_irreducible_ = true;
      return;
    }
    for (auto pc = block.begin; pc < block.end; ++pc)
      control_block_for_pc_[pc] = control_blocks_.size();
    control_blocks_.push_back(std::move(block));
  }
  const auto exit = static_cast<std::uint32_t>(control_blocks_.size());
  control_blocks_.push_back(ControlBlock{size, size, {}, {}, {}});
  for (std::uint32_t b = 0; b < exit; ++b) {
    auto &block = control_blocks_[b];
    const auto &i = *decoded_[block.end - 1].instruction;
    auto edge = [&](std::uint32_t to) {
      auto &out = block.successors;
      if (std::find(out.begin(), out.end(), to) == out.end()) {
        out.push_back(to);
        control_blocks_[to].predecessors.push_back(b);
      }
    };
    if (i.control == "direct_branch" && i.branch_target &&
        *i.branch_target < size) {
      edge(control_block_for_pc_[*i.branch_target]);
      if (i.predicate)
        edge(b + 1);
    } else if (In(i.control, {"return", "exit", "trap"})) {
      edge(exit);
      if (i.predicate)
        edge(b + 1);
    } else {
      edge(b + 1);
      if (i.control == "indirect_branch")
        control_irreducible_ = true;
    }
  }
  auto dominators = [&](std::uint32_t root, bool reverse) {
    const auto count = control_blocks_.size();
    auto outgoing = [&](std::uint32_t b) -> const std::vector<std::uint32_t> & {
      return reverse ? control_blocks_[b].predecessors
                     : control_blocks_[b].successors;
    };
    std::vector<bool> seen(count);
    std::vector<std::pair<std::uint32_t, std::size_t>> stack{{root, 0}};
    std::vector<std::uint32_t> order;
    seen[root] = true;
    while (!stack.empty()) {
      auto &[b, next] = stack.back();
      const auto &edges = outgoing(b);
      if (next == edges.size()) {
        order.push_back(b);
        stack.pop_back();
      } else {
        auto n = edges[next++];
        if (!seen[n]) {
          seen[n] = true;
          stack.emplace_back(n, 0);
        }
      }
    }
    std::reverse(order.begin(), order.end());
    std::vector<std::uint32_t> rank(count, kMissing), parent(count, kMissing);
    for (std::uint32_t n = 0; n < order.size(); ++n)
      rank[order[n]] = n;
    parent[root] = root;
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto b : order) {
        if (b == root)
          continue;
        auto value = kMissing;
        const auto &incoming = reverse ? control_blocks_[b].successors
                                       : control_blocks_[b].predecessors;
        for (auto p : incoming)
          if (parent[p] != kMissing) {
            if (value == kMissing)
              value = p;
            else {
              auto x = p;
              while (x != value) {
                while (rank[x] > rank[value])
                  x = parent[x];
                while (rank[value] > rank[x])
                  value = parent[value];
              }
            }
          }
        if (value != parent[b]) {
          parent[b] = value;
          changed = true;
        }
      }
    }
    return parent;
  };
  control_dom_ = dominators(0, false);
  const auto &dom = control_dom_;
  control_postdom_ = dominators(exit, true);
  auto dominates = [&](std::uint32_t h, std::uint32_t b) {
    while (b != kMissing && b != h && dom[b] != b)
      b = dom[b];
    return b == h;
  };
  // Keep strictly nested natural loops even when the compiler gives them the
  // same header (for (...) { while (wait) {} ... }). Combining by header would
  // incorrectly turn the inner polling latch into an outer computation latch.
  for (std::uint32_t b = 0; b < exit; ++b)
    for (auto h : control_blocks_[b].successors) {
      if (dom[b] == kMissing || !dominates(h, b))
        continue;
      NaturalLoop loop;
      loop.header = h;
      loop.latches.insert(b);
      loop.blocks.insert(h);
      std::vector<std::uint32_t> pending{b};
      while (!pending.empty()) {
        auto n = pending.back();
        pending.pop_back();
        if (!loop.blocks.insert(n).second)
          continue;
        for (auto p : control_blocks_[n].predecessors)
          if (dom[p] != kMissing)
            pending.push_back(p);
      }
      control_loops_.push_back(std::move(loop));
    }
  bool merged = true;
  while (merged) {
    merged = false;
    for (std::size_t a = 0; a < control_loops_.size() && !merged; ++a)
      for (std::size_t b = a + 1; b < control_loops_.size(); ++b) {
        auto &left = control_loops_[a];
        auto &right = control_loops_[b];
        if (left.header != right.header)
          continue;
        const bool l_contains =
            std::includes(left.blocks.begin(), left.blocks.end(),
                          right.blocks.begin(), right.blocks.end());
        const bool r_contains =
            std::includes(right.blocks.begin(), right.blocks.end(),
                          left.blocks.begin(), left.blocks.end());
        if (l_contains != r_contains)
          continue; // Strict nesting.
        left.blocks.insert(right.blocks.begin(), right.blocks.end());
        left.latches.insert(right.latches.begin(), right.latches.end());
        control_loops_.erase(control_loops_.begin() + b);
        merged = true;
        break;
      }
  }
  // Every DFS backedge in a reducible graph must target a dominator.
  std::vector<unsigned char> color(control_blocks_.size());
  std::vector<std::pair<std::uint32_t, std::size_t>> stack{{0, 0}};
  color[0] = 1;
  while (!stack.empty()) {
    auto &[b, next] = stack.back();
    const auto &edges = control_blocks_[b].successors;
    if (next == edges.size()) {
      color[b] = 2;
      stack.pop_back();
      continue;
    }
    auto n = edges[next++];
    if (color[n] == 1 && !dominates(n, b))
      control_irreducible_ = true;
    if (!color[n]) {
      color[n] = 1;
      stack.emplace_back(n, 0);
    }
  }
  for (std::uint32_t id = 0; id < control_loops_.size(); ++id)
    for (auto b : control_loops_[id].blocks)
      control_blocks_[b].loops.push_back(id);
  for (auto &block : control_blocks_)
    std::sort(block.loops.begin(), block.loops.end(), [&](auto a, auto b) {
      return control_loops_[a].blocks.size() > control_loops_[b].blocks.size();
    });
  for (auto &loop : control_loops_) {
    std::map<std::uint32_t, std::vector<std::uint32_t>> definitions;
    std::vector<std::uint32_t> tests;
    bool retry_only = true;
    for (auto b : loop.blocks) {
      const auto &block = control_blocks_[b];
      for (auto pc = block.begin; pc < block.end; ++pc) {
        const auto& i = *decoded_[pc].instruction;
        // A fixed-address read is also used to receive the next scheduler
        // task. It cannot prove that an effectful outer loop is a wait.
        // Keep such loops Unknown, never promote them to bounded computation.
        if (!(i.semantics.category == "compute" || i.control == "direct_branch" ||
              i.opcode == "nanosleep" || IsWait(i) ||
              (i.semantics.category == "memory" && i.semantics.memory_mode == "read")))
          retry_only = false;
        for (auto reg : decoded_[pc].destinations)
          definitions[reg].push_back(pc);
      }
      for (auto out : block.successors)
        if (!loop.blocks.contains(out)) {
          auto pred = decoded_[block.end - 1].predicate;
          if (pred != kMissing)
            tests.push_back(pred);
        }
    }
    bool induction = false, polling = false, unknown = false;
    std::set<std::uint32_t> visited;
    while (!tests.empty()) {
      auto reg = tests.back();
      tests.pop_back();
      if (reg == kMissing || !visited.insert(reg).second)
        continue;
      const auto found = definitions.find(reg);
      if (found == definitions.end())
        continue;
      for (auto pc : found->second) {
        const auto &d = decoded_[pc];
        const auto &i = *d.instruction;
        if (i.semantics.category != "compute") {
          bool invariant = true;
          for (const auto &operand : d.operands)
            if (operand.address)
              for (auto address_reg : operand.registers)
                if (definitions.contains(address_reg))
                  invariant = false;
          if (IsWait(i) || (i.semantics.category == "memory" &&
                            i.semantics.memory_mode == "read" && invariant))
            polling = true;
          else
            unknown = true;
          continue;
        }
        if (!i.predicate && found->second.size() == 1 &&
            (i.opcode == "add" || i.opcode == "sub") &&
            d.operands.size() == 3 && d.operands[1].reg == reg) {
          auto step = ParseInteger(d.operands[2].text);
          if (step && *step != 0)
            induction = true;
        }
        for (std::size_t n = 1; n < d.operands.size(); ++n)
          for (auto input : d.operands[n].registers)
            tests.push_back(input);
        if (d.predicate != kMissing)
          tests.push_back(d.predicate);
      }
    }
    loop.kind = polling     ? (retry_only ? ControlLoop::Polling : ControlLoop::Unknown)
                : unknown   ? ControlLoop::Unknown
                : induction ? ControlLoop::Computation
                            : ControlLoop::Unknown;
  }
}
Executor::ControlChoice Executor::ClassifyControl(std::uint32_t pc) const {
  PrepareControl();
  if (auto it = control_choices_.find(pc); it != control_choices_.end())
    return it->second;
  auto classify = [&]() -> ControlChoice {
    if (control_irreducible_)
      return {};
    const auto b = control_block_for_pc_[pc];
    const auto &block = control_blocks_[b];
    const auto &i = *decoded_[pc].instruction;
    const bool terminal = In(i.control, {"return", "exit", "trap"});
    if ((!terminal && (!i.branch_target || *i.branch_target >= decoded_.size())) ||
        block.successors.size() != 2 || pc + 1 >= decoded_.size())
      return {};
    auto target = terminal ? static_cast<std::uint32_t>(control_blocks_.size() - 1)
                           : control_block_for_pc_[*i.branch_target];
    auto fall = control_block_for_pc_[pc + 1];
    if (!block.loops.empty()) {
      const auto id = block.loops.back();
      const auto &loop = control_loops_[id];
      bool t = loop.blocks.contains(target), f = loop.blocks.contains(fall);
      if (t != f) {
        if (loop.kind == ControlLoop::Polling)
          return {ControlLoop::Polling};
        if (loop.latches.size() != 1)
          return {};
        const auto stays = t ? target : fall;
        bool post = loop.latches.contains(b) && stays == loop.header;
        // A conditional exit followed by a dedicated unconditional backedge
        // is also a post-test. Do not cross body effects, joins, or continues.
        if (!post && loop.latches.contains(stays)) {
          const auto &latch = control_blocks_[stays];
          if (latch.end == latch.begin + 1 &&
              latch.predecessors.size() == 1 && latch.predecessors.front() == b &&
              latch.successors.size() == 1 && latch.successors.front() == loop.header) {
            const auto &jump = *decoded_[latch.begin].instruction;
            post = jump.opcode == "bra" && !jump.predicate;
          }
        }
        const bool pre = b == loop.header && !post;
        if (!pre && !post)
          return {}; // Unknown break is not a representative if.
        // A pre-test block must not already execute body effects before its
        // condition; otherwise a numeric bound would execute an extra prefix.
        if (pre)
          for (auto k = block.begin; k < pc; ++k) {
            const auto &op = *decoded_[k].instruction;
            if (op.semantics.category != "compute" &&
                !(op.semantics.category == "memory" &&
                  op.semantics.memory_mode == "read"))
              return {};
          }
        return {loop.kind, id, t, post};
      }
      if (target == loop.header || fall == loop.header)
        return {}; // Unknown continue.
    }
    // Only ordinary diamonds with a join in the same loop nesting are eligible.
    auto join = control_postdom_[b];
    if (join == kMissing || join == b ||
        control_blocks_[join].loops != block.loops)
      return {};
    for (auto id : block.loops)
      if (control_loops_[id].header == join)
        return {};
    return {ControlLoop::None};
  };
  auto result = classify();
  control_choices_.emplace(pc, result);
  return result;
}
std::optional<std::pair<std::string, std::uint32_t>>
Executor::WaitLoop(const Instruction &i) const {
  if (!IsWait(i) || i.destinations.empty())
    return {};
  // A retry may branch to a short, idempotent address/parity setup before
  // the wait. Every value overwritten by the loop must be defined before
  // being read in this prefix: no carried counter, address, or probe result.
  const auto stable_prefix=[&](std::uint32_t target,std::uint32_t branch) {
    if(target==i.id)return true;
    if(target>i.id || i.id-target>8)return false;
    const auto entry=label_entries_.upper_bound(target);
    if(entry!=label_entries_.end() && *entry<=i.id)return false;
    std::set<std::string> overwritten,defined;
    for(auto pc=target;pc<=branch;++pc)
      for(const auto &name:f_.instructions[pc].destinations)overwritten.insert(name);
    for(auto pc=target;pc<i.id;++pc) {
      const auto &setup=f_.instructions[pc];
      if(setup.predicate || setup.control!="none" || Has(setup,"cc") || setup.destinations.size()!=1 ||
         !In(setup.opcode,{"mov","cvt","cvta","add","sub","shl","shr","and","or","xor"}))return false;
      for(std::size_t k=1;k<setup.operands.size();++k) {
        const auto value=std::string(TrimView(m_.view(setup.operands[k])));
        if(value.find_first_of("!{}[]| ")!=std::string::npos)return false;
        // Special registers sampled inside the loop need their own stability
        // proof. Ordinary declared registers are invariant unless overwritten.
        if(value.starts_with("%clock") || value.starts_with("%globaltimer") ||
           value.starts_with("%pm") || value=="%smid")return false;
        if(overwritten.contains(value) && !defined.contains(value))return false;
      }
      defined.insert(setup.destinations.front());
    }
    return true;
  };
  // NVVM also encodes !ready as selp(flag,1,0,ready); setp.eq(retry,flag,0).
  // Accept only this side-effect-free, same-target loop. Replay the conversion
  // instructions after successful waiting so live-out flag/predicate values
  // and their value origins are retained in both execution paths.
  if(i.id+3<f_.instructions.size()) {
    const auto &select=f_.instructions[i.id+1];
    const auto &compare=f_.instructions[i.id+2];
    const auto &branch=f_.instructions[i.id+3];
    const auto operand=[&](const Instruction &inst,std::size_t index) {
      return TrimView(m_.view(inst.operands.at(index)));
    };
    if(select.opcode=="selp" && Has(select,"b32") && !select.predicate && select.operands.size()==4 &&
       compare.opcode=="setp" && Has(compare,"eq") && Has(compare,"b32") && !compare.predicate && compare.operands.size()==3 &&
       branch.control=="direct_branch" && branch.branch_target && stable_prefix(*branch.branch_target,i.id+3) && branch.predicate && !branch.predicate_negated &&
       select.destinations.size()==1 && compare.destinations.size()==1 &&
       operand(select,1)=="1" && operand(select,2)=="0" && operand(select,3)==i.destinations.front() &&
       operand(compare,1)==select.destinations.front() && operand(compare,2)=="0" &&
       *branch.predicate==compare.destinations.front())
      return {{"pc:"+std::to_string(i.id+4),i.id+4}};
  }
  std::optional<std::pair<std::string, std::uint32_t>> success;
  for (std::uint32_t offset = 1; offset <= 8; ++offset) {
    auto pc = i.id + offset;
    if (pc >= f_.instructions.size())
      return {};
    const auto &candidate = f_.instructions[pc];
    if (candidate.control == "direct_branch") {
      if (!candidate.branch_target)
        return {};
      auto target = *candidate.branch_target;
      if (candidate.predicate && *candidate.predicate == i.destinations[0] &&
          !candidate.predicate_negated && target > pc) {
        if (success)
          return {};
        success = {{candidate.branch_label, target}};
        continue;
      }
      if (stable_prefix(target,pc)) {
        if (candidate.predicate && *candidate.predicate == i.destinations[0] &&
            candidate.predicate_negated)
          return {{"pc:" + std::to_string(pc + 1), pc + 1}};
        if (!candidate.predicate && success)
          return success;
      }
      return {};
    }
    if (candidate.opcode == "nanosleep") {
      if (candidate.predicate && (*candidate.predicate != i.destinations[0] ||
                                  !candidate.predicate_negated))
        return {};
      continue;
    }
    return {};
  }
  return {};
}
} // namespace
} // namespace sentryward::frontend::ptx
