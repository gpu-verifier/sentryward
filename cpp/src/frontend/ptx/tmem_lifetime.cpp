#include "sentryward/frontend/ptx/tmem_lifetime.h"

#include <algorithm>
#include <cctype>
#include <deque>
#include <functional>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <unordered_set>

namespace sentryward::frontend::ptx {
namespace {
using PointerMap = std::unordered_map<std::string_view, TmemPointer>;
struct Atom {
  TmemAtomSite site;
  TmemLaneExecutions lanes;
  std::deque<RecordHolder> held; // the Full reads `lanes` views (ExpansionSite::Tmem); stable addresses
};
using Atoms = std::map<TmemAllocationId, Atom>;
// One lane's records for the analysis passes. BlockScaleTmem expands the lane;
// Tmem never does: every pass walks metadata, and a step that reads values gets
// the record read at that position (Full, or a Slice of the registers it
// names). On an expanded lane, whichever site expanded it, the Expanded
// element serves both, as lowering's own reads do.
class LaneRecords {
public:
  LaneRecords(const CtaFacts &facts, LaneRef lane, ExpansionSite site) : facts_(facts), lane_(lane) {
    if (site != ExpansionSite::Tmem) (void)facts.ExpandLane(lane, site);
    expanded_ = facts.Expanded(lane);
  }
  [[nodiscard]] std::size_t size() const { return facts_.Size(lane_); }
  [[nodiscard]] RecordView Meta(std::size_t pos) const { return facts_.Meta({lane_, pos}); }
  // The record at `pos` with every value; valid until Values reads another position.
  [[nodiscard]] RecordView Values(std::size_t pos) {
    if (expanded_) return Meta(pos);
    if (!held_ || held_pos_ != pos) {
      held_ = facts_.ReadFull({lane_, pos});
      held_pos_ = pos;
    }
    return held_.view();
  }
  [[nodiscard]] RecordHolder Hold(std::size_t pos) const { return facts_.ReadFull({lane_, pos}); }
  [[nodiscard]] bool expanded() const noexcept { return expanded_; }
  // The record at `pos` with the values an operand's resolution reads: the
  // registers its text and its split base name (ResolveExpression reads no
  // other); valid until the next Operand call.
  [[nodiscard]] RecordView Operand(std::size_t pos, std::string_view operand);

private:
  const CtaFacts &facts_;
  LaneRef lane_;
  bool expanded_ = false;
  RecordHolder held_, operand_;
  std::size_t held_pos_ = 0;
  std::vector<RegId> slice_;
};
std::string_view Trim(std::string_view text) {
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.front())))
    text.remove_prefix(1);
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
    text.remove_suffix(1);
  return text;
}
std::string_view Strip(std::string_view text) {
  text = Trim(text);
  return text.size() >= 2 && text.front() == '[' && text.back() == ']'
             ? Trim(text.substr(1, text.size() - 2))
             : text;
}
std::string Opcode(const Instruction &instruction) {
  auto result = instruction.opcode;
  for (const auto &modifier : instruction.modifiers)
    result += "." + modifier;
  return result;
}
TmemAtomSite Site(const FactSource &facts, std::uint32_t warp,
                  const RecordView &execution) {
  const auto &instruction =
      facts.function().instructions.at(execution.instruction_id());
  return {warp, instruction.id, execution.occurrence(), instruction.line,
          Opcode(instruction)};
}
void Poll(const TmemAnalysisOptions &options, std::size_t &count) {
  if ((++count & 4095) == 0 && options.poll)
    options.poll();
}
Atoms MakeAtoms(const CtaFacts &facts, ExpansionSite site,
                const TmemAnalysisOptions &options) {
  const auto &function = facts.function();
  Atoms atoms;
  std::size_t count = 0;
  for (std::size_t index = 0; index < facts.LaneCount(); ++index) {
    const auto lane = facts.Lane(index);
    const auto warp = facts.Warp(lane), lane_id = facts.LaneId(lane);
    LaneRecords records(facts, lane, site);
    for (std::size_t pos = 0; pos < records.size(); ++pos) {
      Poll(options, count);
      const auto execution = records.Meta(pos);
      if (function.instructions[execution.instruction_id()]
              .semantics.category != "tmem_lifetime")
        continue;
      const TmemAllocationId key{warp, execution.instruction_id(),
                                 execution.occurrence()};
      auto [found, inserted] = atoms.try_emplace(key);
      if (inserted)
        found->second.site = Site(facts, warp, execution);
      if (records.expanded())
        found->second.lanes.emplace_back(lane_id, execution);
      else
        found->second.lanes.emplace_back(lane_id, found->second.held.emplace_back(records.Hold(pos)).view());
    }
  }
  return atoms;
}
bool FullWarp(const Atom &atom) {
  std::uint32_t mask = 0;
  for (const auto &[lane, _] : atom.lanes) {
    if (lane >= 32)
      return false;
    mask |= std::uint32_t{1} << lane;
  }
  return mask == std::numeric_limits<std::uint32_t>::max() &&
         atom.lanes.size() == 32;
}
std::optional<unsigned> CtaGroup(const Instruction &instruction) {
  std::optional<unsigned> group;
  for (const auto &modifier : instruction.modifiers) {
    if (modifier == "cta_group::1" || modifier == "cta_group::2") {
      if (group)
        return {};
      group = modifier.back() - '0';
    }
  }
  return group;
}
std::optional<Integer> Known(const FactSource &facts,
                             const RecordView &execution,
                             std::string_view token) {
  token = Trim(token);
  if (auto integer = ParseInteger(token))
    return integer;
  if (const auto *value = facts.Value(execution, token)) {
    if (const auto *integer = std::get_if<Integer>(value))
      return *integer;
  }
  const auto &parameters = facts.launch().parameters();
  const auto fixed = parameters.find(std::string(token));
  if (fixed != parameters.end())
    return fixed->second;
  return {};
}
template <class Lookup>
ResolvedTmemOperand ResolveExpression(const FactSource &facts,
                                      const RecordView &execution,
                                      std::string_view token,
                                      Lookup &&pointer) {
  token = Trim(token);
  if (auto integer = ParseInteger(token))
    return *integer;
  if (auto found = pointer(token))
    return *found;
  if (auto integer = Known(facts, execution, token))
    return *integer;
  if (auto split = SplitAddress(token)) {
    if (auto found = pointer(split->first)) {
      if (auto added = found->AddEncoded(split->second))
        return *added;
    } else if (auto integer = Known(facts, execution, split->first))
      return Integer(*integer + split->second);
  }
  return {};
}
std::optional<TmemPointer> FindPointer(const PointerMap &pointers,
                                       std::string_view name) {
  auto found = pointers.find(name);
  return found == pointers.end() ? std::nullopt
                                 : std::optional(found->second);
}
ResolvedTmemOperand ResolveExpression(const FactSource &facts,
                                      const RecordView &execution,
                                      std::string_view token,
                                      const PointerMap &pointers) {
  return ResolveExpression(
      facts, execution, token,
      [&](std::string_view name) { return FindPointer(pointers, name); });
}
// The TmemPointer alternative of ResolveExpression(token, pointers), found in
// the same order without reading a value. The one exception: a pointer split
// base is taken only after Known(token) failed, because an integer answer at
// that step takes precedence over it.
std::optional<TmemPointer> ResolvePointer(const FactSource &facts,
                                          const RecordView &execution,
                                          std::string_view token,
                                          const PointerMap &pointers) {
  token = Trim(token);
  if (ParseInteger(token))
    return {};
  if (auto found = FindPointer(pointers, token))
    return found;
  // Without a '+' or '-' after its first character, the token splits to
  // itself or not at all, and FindPointer(token) has just failed.
  if (token.find_first_of("+-", 1) == std::string_view::npos)
    return {};
  if (auto split = SplitAddress(token)) {
    if (auto found = FindPointer(pointers, split->first)) {
      if (Known(facts, execution, token)) {
        return {};
      }
      return found->AddEncoded(split->second);
    }
  }
  return {};
}
std::optional<Integer> UniformInteger(const FactSource &facts,
                                      const Atom &atom,
                                      std::string_view operand) {
  std::optional<Integer> result;
  const PointerMap empty;
  for (const auto &[_, execution] : atom.lanes) {
    const auto value =
        ResolveExpression(facts, execution, Strip(operand), empty);
    const auto *integer = std::get_if<Integer>(&value);
    if (!integer || (result && *result != *integer))
      return {};
    result = *integer;
  }
  return result;
}
std::string Hex(const Integer &integer) {
  std::ostringstream result;
  if (integer < 0)
    result << '-' << std::hex << -integer;
  else
    result << std::hex << integer;
  return result.str();
}
void AllocationRecords(const FactSource &facts, const Atoms &atoms,
                       const TmemAnalysisOptions &options,
                       TmemProvenanceIndex &result) {
  for (const auto &[key, atom] : atoms) {
    const auto &instruction =
        facts.function().instructions[key.instruction_index];
    if (instruction.semantics.operation != "alloc")
      continue;
    const auto issue = [&](std::string code, std::string message) {
      result.issues.push_back({std::move(code), std::move(message), atom.site});
    };
    if (!FullWarp(atom)) {
      issue("tmem_noncollective_lifetime",
            "tcgen05.alloc must be executed by all 32 threads of its warp");
      continue;
    }
    const auto group = CtaGroup(instruction);
    if (group != 1 && !(options.allow_cta_group2 && group == 2)) {
      issue("tmem_cta_group_2",
            "cta_group::2 allocation spans two CTAs and is not projected into "
            "the one-CTA lifetime model");
      continue;
    }
    if (instruction.operands.size() != 2) {
      issue("tmem_malformed_alloc",
            "tcgen05.alloc must provide a shared destination and nCols");
      continue;
    }
    const auto slot = UniformInteger(
        facts, atom, facts.module().view(instruction.operands[0]));
    const auto columns = UniformInteger(
        facts, atom, facts.module().view(instruction.operands[1]));
    if (!slot) {
      issue("tmem_unresolved_alloc_slot",
            "allocation result shared-memory slot is not lane-uniform and "
            "statically known");
      continue;
    }
    if (!columns) {
      issue("tmem_unresolved_alloc_size",
            "allocation nCols is not lane-uniform and statically known");
      continue;
    }
    if (*columns < 32 || *columns > 512 || (*columns & (*columns - 1)) != 0) {
      issue("tmem_invalid_alloc_size",
            "allocation nCols=" + columns->str() +
                " is not a power of two in [32, 512]");
      continue;
    }
    if (*slot % 4 != 0) {
      issue("tmem_misaligned_alloc_slot", "allocation result slot 0x" +
                                              Hex(*slot) +
                                              " is not 4-byte aligned");
      continue;
    }
    result.allocations.push_back(
        {key, atom.site, *slot, columns->convert_to<std::uint32_t>(), *group});
  }
}
// Operands are classified by ResolvePointer; the integer partner of a pointer
// (the Integer alternative of ResolveExpression) is read only where the result
// uses it. This returns what resolving both operands eagerly would return:
// every skipped lookup is a pure const read whose result cannot change the
// return value. An operand classified as a pointer is exactly one
// whose ResolveExpression is a TmemPointer, so it has no Integer alternative
// and is never read as a partner.
std::optional<TmemPointer> CopyPointer(const FactSource &facts,
                                       const RecordView &execution,
                                       const PointerMap &pointers) {
  const auto &instruction =
      facts.function().instructions[execution.instruction_id()];
  if (instruction.operands.size() < 2 ||
      (instruction.opcode != "mov" && instruction.opcode != "cvt" &&
       instruction.opcode != "add" && instruction.opcode != "sub" &&
       instruction.opcode != "selp" && instruction.opcode != "and" &&
       instruction.opcode != "xor"))
    return {};
  const auto operand = [&](std::size_t i) {
    return facts.module().view(instruction.operands[i]);
  };
  const auto classify = [&](std::size_t i) {
    return ResolvePointer(facts, execution, operand(i), pointers);
  };
  const auto partner = [&](std::size_t i) -> std::optional<Integer> {
    const auto value =
        ResolveExpression(facts, execution, operand(i), pointers);
    if (const auto *integer = std::get_if<Integer>(&value))
      return *integer;
    return {};
  };
  const auto integer32 = [](std::string_view type) {
    return type == "b32" || type == "u32" || type == "s32";
  };
  const auto &modifiers = instruction.modifiers;
  if ((instruction.opcode == "mov" && !modifiers.empty() &&
       std::all_of(modifiers.begin(), modifiers.end(), integer32)) ||
      (instruction.opcode == "cvt" && modifiers.size() == 2 &&
       std::all_of(modifiers.begin(), modifiers.end(), integer32))) {
    if (auto pointer = classify(1))
      return pointer;
  }
  if (instruction.operands.size() < 3 || modifiers.size() != 1)
    return {};
  const auto left = classify(1), right = classify(2);
  const auto *lp = left ? &*left : nullptr;
  const auto *rp = right ? &*right : nullptr;
  if ((instruction.opcode == "xor" || instruction.opcode == "and") && modifiers[0] == "b32") {
    const bool exclusive = instruction.opcode == "xor";
    for (const auto &[pointer, other] :
         {std::pair{lp, std::size_t{2}}, std::pair{rp, std::size_t{1}}}) {
      if (!pointer || (other == 2 ? rp : lp))
        continue;
      if (const auto integer = partner(other)) {
        const auto mask = (*integer & Integer(0xffffffffU)).convert_to<std::uint32_t>();
        // Preserve the allocation when xor changes only subword-selector bits,
        // or and clears only bits among the low five.
        if (exclusive ? (mask & 0x3fffffffU) == 0 : (mask & 0xffffffe0U) == 0xffffffe0U)
          return TmemPointer{pointer->allocation,
                             exclusive ? pointer->encoded_offset ^ mask : pointer->encoded_offset & mask};
      }
    }
  }
  if ((instruction.opcode == "add" || instruction.opcode == "sub") &&
      (modifiers[0] == "u32" || modifiers[0] == "s32")) {
    // Each case needs exactly one pointer side: the other one must have an
    // Integer alternative.
    if (lp && !rp) {
      if (const auto ri = partner(2))
        return lp->AddEncoded(instruction.opcode == "sub" ? Integer(-*ri) : *ri);
    } else if (instruction.opcode == "add" && rp && !lp) {
      if (const auto li = partner(1))
        return rp->AddEncoded(*li);
    }
  }
  if (instruction.opcode == "selp" && integer32(modifiers[0]) && lp && rp &&
      *lp == *rp)
    return *lp;
  return {};
}
// `execution` is the record's metadata; `values` reads it with its values,
// only for a 4-byte shared load.
std::optional<Integer> SharedLoadSlot(const FactSource &facts,
                                      const RecordView &execution,
                                      const std::function<RecordView()> &values) {
  const auto &instruction =
      facts.function().instructions[execution.instruction_id()];
  if ((instruction.opcode != "ld" && instruction.opcode != "ldu") ||
      (instruction.semantics.memory_space != "shared" &&
       instruction.semantics.memory_space != "shared.cta") ||
      instruction.semantics.memory_mode != "read" ||
      instruction.memory_width != 4 || instruction.destinations.size() != 1 ||
      instruction.operands.size() < 2)
    return {};
  const PointerMap empty;
  auto value = ResolveExpression(
      facts, values(), Strip(facts.module().view(instruction.operands[1])),
      empty);
  if (const auto *integer = std::get_if<Integer>(&value))
    return *integer;
  return {};
}
// Whether some operand after the first names a pointer register, itself or as
// its split base: otherwise CopyPointer classifies no pointer, reads no value
// and returns nothing.
bool NamesPointer(const FactSource &facts, const Instruction &instruction,
                  const PointerMap &pointers) {
  for (std::size_t k = 1; k < instruction.operands.size(); ++k) {
    const auto token = Trim(facts.module().view(instruction.operands[k]));
    if (pointers.contains(token)) return true;
    if (const auto split = SplitAddress(token); split && pointers.contains(std::string_view(split->first))) return true;
  }
  return false;
}
RecordView LaneRecords::Operand(std::size_t pos, std::string_view operand) {
  if (expanded_) return Meta(pos);
  const auto token = Strip(operand);
  slice_.clear();
  if (const auto reg = facts_.Reg(token)) slice_.push_back(*reg);
  if (const auto split = SplitAddress(token))
    if (const auto reg = facts_.Reg(split->first); reg && std::find(slice_.begin(), slice_.end(), *reg) == slice_.end())
      slice_.push_back(*reg);
  std::sort(slice_.begin(), slice_.end());
  operand_ = facts_.ReadSlice({lane_, pos}, slice_);
  return operand_.view();
}
class LocalPointerSpills {
  struct Object { std::string symbol; std::vector<std::uint32_t> scope; Integer size; };
  struct Address { std::size_t object; Integer offset; };
  const FactSource &facts;
  std::vector<Object> objects;
  std::map<std::string,Address> registers;
  std::map<std::pair<std::size_t,Integer>,std::pair<TmemPointer,unsigned>> bytes;
  std::optional<Address> Resolve(std::string_view text,const Instruction &instruction) const {
    auto split=SplitAddress(Strip(text)); if (!split) return {};
    auto found=registers.find(split->first);
    if (found!=registers.end()) { auto address=found->second; address.offset+=split->second; return address; }
    std::optional<Address> result;
    for (std::size_t k=0;k<objects.size();++k) {
      const auto &object=objects[k];
      if (object.symbol==split->first && object.scope.size()<=instruction.scope.size() &&
          std::equal(object.scope.begin(),object.scope.end(),instruction.scope.begin()) &&
          (!result || objects[result->object].scope.size()<object.scope.size())) result=Address{k,split->second};
    }
    return result;
  }
public:
  explicit LocalPointerSpills(const FactSource &f):facts(f) {
    static const std::regex declaration(R"(^\s*\.local\s+(?:\.align\s+\d+\s+)?\.([bus]\d+)\s+([A-Za-z_$][A-Za-z0-9_$]*)(?:\[(\d+)\])?\s*;?\s*$)");
    for (const auto &d : facts.function().directives) {
      auto text=std::string(facts.module().view(d.source)); std::smatch match;
      if (std::regex_match(text,match,declaration)) {
        const auto width=std::stoul(match[1].str().substr(1))/8;
        const Integer count=match[3].matched?Integer(match[3].str()):Integer(1);
        objects.push_back({match[2].str(),d.scope,width*count});
      }
    }
  }
  // `e` is the record's metadata; `values` reads the record with its values.
  PointerMap Step(const RecordView &e,const std::function<RecordView()> &values,const PointerMap &pointers) {
    if (objects.empty()) return {};
    const auto &i=facts.function().instructions[e.instruction_id()];
    std::vector<std::string_view> ops;
    for (auto operand : i.operands) ops.push_back(Trim(facts.module().view(operand)));
    PointerMap recovered;
    if (i.semantics.memory_space=="generic" &&
        (i.semantics.memory_mode=="write" || i.semantics.memory_mode=="atomic")) bytes.clear();
    const bool local=std::find(i.modifiers.begin(),i.modifiers.end(),"local")!=i.modifiers.end();
    if ((i.opcode=="ld" || i.opcode=="st") && local && ops.size()==2) {
      const bool store=i.opcode=="st";
      auto address=Resolve(ops[store?0:1],i); const auto width=i.memory_width;
      const bool valid=address && width && address->offset>=0 && address->offset+*width<=objects[address->object].size;
      if (store) {
        if (!valid) bytes.clear();
        else {
          for (unsigned k=0;k<*width;++k) bytes.erase({address->object,address->offset+k});
          auto text=ops[1]; if (text.starts_with('{') && text.ends_with('}')) text=text.substr(1,text.size()-2);
          std::vector<std::string_view> parts;
          for (;;) { auto comma=text.find(','); parts.push_back(Trim(text.substr(0,comma))); if (comma==text.npos) break; text.remove_prefix(comma+1); }
          if (*width==4*parts.size())
            for (std::size_t k=0;k<parts.size();++k) {
              const auto value=ResolveExpression(facts,values(),parts[k],pointers);
              if (const auto *pointer=std::get_if<TmemPointer>(&value))
                for (unsigned byte=0;byte<4;++byte) bytes[{address->object,address->offset+4*k+byte}]={*pointer,byte};
            }
        }
      } else if (valid && *width==4*i.destinations.size())
        for (std::size_t k=0;k<i.destinations.size();++k) {
          auto base=bytes.find({address->object,address->offset+4*k}); if (base==bytes.end()) continue;
          bool intact=true;
          for (unsigned byte=0;byte<4;++byte) {
            auto part=bytes.find({address->object,address->offset+4*k+byte});
            intact &= part!=bytes.end() && part->second==std::pair{base->second.first,byte};
          }
          if (intact) recovered[i.destinations[k]]=base->second.first;
        }
    }
    std::optional<Address> propagated;
    const auto &mods=i.modifiers;
    if (ops.size()>=2 && i.destinations.size()==1) {
      if ((i.opcode=="mov" && mods.size()==1 && (mods[0]=="b64" || mods[0]=="u64" || mods[0]=="s64")) ||
          (i.opcode=="cvta" && local && std::find(mods.begin(),mods.end(),"u64")!=mods.end())) propagated=Resolve(ops[1],i);
      else if ((i.opcode=="add" || i.opcode=="sub") && ops.size()==3 && mods.size()==1 && (mods[0]=="u64" || mods[0]=="s64")) {
        auto base=Resolve(ops[1],i); auto delta=Known(facts,values(),ops[2]);
        if (!base && i.opcode=="add") { base=Resolve(ops[2],i); delta=Known(facts,values(),ops[1]); }
        if (base && delta) { base->offset+=i.opcode=="sub"?Integer(-*delta):*delta; propagated=base; }
      }
    }
    for (const auto &dest : i.destinations) { registers.erase(dest); if (propagated) registers.emplace(dest,*propagated); }
    return recovered;
  }
};
TmemProvenanceIndex BuildProvenance(const CtaFacts &facts, ExpansionSite site,
                                    const Atoms &atoms,
                                    const TmemAnalysisOptions &options) {
  const auto &function = facts.function();
  TmemProvenanceIndex result;
  result.facts = &facts;
  result.positioned = site == ExpansionSite::Tmem;
  AllocationRecords(facts, atoms, options, result);
  std::map<TmemAllocationId, const TmemAllocation *> by_site;
  std::map<Integer, std::vector<const TmemAllocation *>> by_slot;
  for (const auto &allocation : result.allocations) {
    by_site.emplace(allocation.identifier, &allocation);
    by_slot[allocation.shared_slot].push_back(&allocation);
  }
  std::vector<std::vector<std::string>> referenced_registers(
      function.instructions.size());
  for (const auto &instruction : function.instructions) {
    if (instruction.opcode != "tcgen05" && instruction.opcode != "cvt")
      continue;
    auto &names = referenced_registers[instruction.id];
    names = instruction.registers;
    for (const auto &operand : instruction.operands) {
      if (auto split = SplitAddress(Strip(facts.module().view(operand)));
          split &&
          std::find(names.begin(), names.end(), split->first) == names.end())
        names.push_back(split->first);
    }
  }
  std::set<TmemPublicationRead> reads;
  std::set<TmemLifetimeIssue> issues(result.issues.begin(),
                                     result.issues.end());
  std::size_t count = 0;
  for (std::size_t index = 0; index < facts.LaneCount(); ++index) {
    const auto lane = facts.Lane(index);
    const auto warp = facts.Warp(lane);
    PointerMap pointers;
    LocalPointerSpills spills(facts);
    std::map<Integer, const TmemAllocation *> local_slots;
    LaneRecords records(facts, lane, site);
    for (std::size_t pos = 0; pos < records.size(); ++pos) {
      Poll(options, count);
      const auto execution = records.Meta(pos);
      const std::function<RecordView()> values = [&] { return records.Values(pos); };
      const auto &instruction =
          function.instructions[execution.instruction_id()];
      if ((instruction.opcode == "tcgen05" || instruction.opcode == "cvt") && !pointers.empty()) {
        TmemProvenanceIndex::Snapshot snapshot;
        for (const auto &name : referenced_registers[instruction.id]) {
          if (auto found = pointers.find(name); found != pointers.end())
            snapshot.emplace_back(found->first, found->second);
        }
        if (!snapshot.empty()) {
          result.register_state_before.emplace(*execution.ref(), std::move(snapshot));
        }
      }
      const TmemAllocationId key{warp, instruction.id,
                                 execution.occurrence()};
      if (auto allocation = by_site.find(key); allocation != by_site.end())
        local_slots[allocation->second->shared_slot] = allocation->second;
      if (execution.assumed_equality() && execution.assumed_equality()->second == 0)
        if (auto p = pointers.find(execution.assumed_equality()->first);
            p != pointers.end() && p->second.encoded_offset == 0)
          result.zero_based_allocations.insert(p->second.allocation);
      const auto recovered=spills.Step(execution,values,pointers);
      if (instruction.opcode=="ld" && instruction.semantics.memory_space=="local") {
        for (const auto &dest : instruction.destinations) {
          pointers.erase(dest);
          if (auto found=recovered.find(dest);found!=recovered.end()) pointers.emplace(dest,found->second);
        }
        continue;
      }
      if (instruction.destinations.empty())
        continue;
      const auto loaded_slot = SharedLoadSlot(facts, execution, values);
      std::optional<TmemPointer> propagated;
      auto slot = loaded_slot ? by_slot.find(*loaded_slot) : by_slot.end();
      if (slot != by_slot.end()) {
        const TmemAllocation *selected = nullptr;
        bool provisional = false;
        if (auto local = local_slots.find(*loaded_slot);
            local != local_slots.end())
          selected = local->second;
        const auto &choices = slot->second;
        if (!selected && choices.size() == 1)
          selected = choices.front();
        if (!selected && choices.size() > 1) {
          if (auto resolved = options.publication_resolution.find(key);
              resolved != options.publication_resolution.end()) {
            for (const auto *choice : choices)
              if (choice->identifier == resolved->second)
                selected = choice;
          }
          if (!selected && options.allow_ambiguous_publications) {
            selected = choices.front();
            provisional = true;
          }
        }
        if (selected) {
          propagated = TmemPointer{selected->identifier, 0};
          if (!provisional)
            reads.insert({Site(facts, warp, execution),
                          selected->identifier});
        } else {
          issues.insert({"tmem_ambiguous_publication_slot",
                         "shared slot 0x" + Hex(*loaded_slot) +
                             " is reused by " + std::to_string(choices.size()) +
                             " allocation occurrences; this cross-warp load "
                             "has no unique producer",
                         Site(facts, warp, execution)});
        }
      } else if (!pointers.empty() && NamesPointer(facts, instruction, pointers))
        propagated = CopyPointer(facts, values(), pointers);
      for (const auto &destination : instruction.destinations) {
        if (propagated)
          pointers[destination] = *propagated;
        else
          pointers.erase(destination);
      }
    }
  }
  result.publication_reads.assign(reads.begin(), reads.end());
  result.issues.assign(issues.begin(), issues.end());
  return result;
}
std::vector<std::string_view> TmemOperands(const FactSource &facts,
                                           const Instruction &instruction) {
  std::vector<std::string_view> bracketed;
  for (const auto &range : instruction.operands) {
    const auto operand = Trim(facts.module().view(range));
    if (operand.size() >= 2 && operand.front() == '[' && operand.back() == ']')
      bracketed.push_back(operand);
  }
  const auto &op = instruction.semantics.operation;
  if (bracketed.empty())
    return bracketed;
  if (op == "load")
    return {bracketed.back()};
  if (op == "store")
    return {bracketed.front()};
  if (op == "mma" || op == "cp" || op == "shift")
    return bracketed;
  return {};
}
struct LanePositions {
  std::uint32_t warp{};
  std::map<DynamicKey, std::size_t> positions;
};
std::vector<LanePositions>
RelevantPositions(const CtaFacts &facts,
                  const std::set<TmemAllocationId> &relevant,
                  const TmemAnalysisOptions &options) {
  std::vector<LanePositions> result;
  std::size_t count = 0;
  for (std::size_t index = 0; index < facts.LaneCount(); ++index) {
    const auto trace = facts.Lane(index);
    auto &lane = result.emplace_back();
    lane.warp = facts.Warp(trace);
    for (std::size_t i = 0; i < facts.Size(trace); ++i) {
      Poll(options, count);
      const auto execution = facts.Meta({trace, i});
      if (relevant.contains(
              {lane.warp, execution.instruction_id(), execution.occurrence()}))
        lane.positions.emplace(execution.Key(), i);
    }
  }
  return result;
}
bool Precedes(const std::vector<LanePositions> &positions,
              const TmemAtomSite &source, const TmemAtomSite &target) {
  if (source.warp != target.warp)
    return false;
  bool saw_target = false;
  for (const auto &lane : positions) {
    if (lane.warp != target.warp)
      continue;
    const auto to =
        lane.positions.find({target.instruction_index, target.occurrence});
    if (to == lane.positions.end())
      continue;
    saw_target = true;
    const auto from =
        lane.positions.find({source.instruction_index, source.occurrence});
    if (from == lane.positions.end() || from->second >= to->second)
      return false;
  }
  return saw_target;
}
std::map<TmemAtomSite, TmemAtomSite>
RetirementSites(const CtaFacts &facts, ExpansionSite site,
                const TmemAnalysisOptions &options) {
  const auto &function = facts.function();
  std::map<TmemAtomSite, std::set<std::optional<TmemAtomSite>>> candidates;
  std::size_t count = 0;
  for (std::size_t index = 0; index < facts.LaneCount(); ++index) {
    const auto lane = facts.Lane(index);
    const auto warp = facts.Warp(lane);
    std::vector<TmemAtomSite> loads, stores;
    LaneRecords records(facts, lane, site);
    for (std::size_t pos = 0; pos < records.size(); ++pos) {
      Poll(options, count);
      const auto execution = records.Meta(pos);
      const auto &instruction =
          function.instructions[execution.instruction_id()];
      if (instruction.opcode != "tcgen05" ||
          instruction.semantics.category != "async")
        continue;
      const auto &operation = instruction.semantics.operation;
      if (operation == "load")
        loads.push_back(Site(facts, warp, execution));
      else if (operation == "store")
        stores.push_back(Site(facts, warp, execution));
      else if (operation == "wait_load" || operation == "wait_store") {
        auto &pending = operation == "wait_load" ? loads : stores;
        if (!pending.empty()) {
          const auto site = Site(facts, warp, execution);
          for (const auto &item : pending)
            candidates[item].insert(site);
          pending.clear();
        }
      }
    }
    for (const auto &pending : loads)
      candidates[pending].insert(std::nullopt);
    for (const auto &pending : stores)
      candidates[pending].insert(std::nullopt);
  }
  std::map<TmemAtomSite, TmemAtomSite> result;
  for (const auto &[site, waits] : candidates)
    if (waits.size() == 1 && *waits.begin())
      result.emplace(site, **waits.begin());
  return result;
}
} // namespace

std::string TmemAllocationId::ResourceName() const {
  return "tmem.alloc.W" + std::to_string(warp) + ".I" +
         std::to_string(instruction_index) + ".O" + std::to_string(occurrence);
}
std::optional<TmemPointer> TmemPointer::AddEncoded(const Integer &delta) const {
  const Integer value = (Integer(encoded_offset) + delta) & Integer(0xffffffffU);
  return TmemPointer{allocation, value.convert_to<std::uint32_t>()};
}
ResolvedTmemOperand
TmemProvenanceIndex::Resolve(std::uint32_t,
                             const RecordView &execution,
                             std::string_view operand, std::uint32_t) const {
  if (!facts)
    return {};
  const Snapshot *snapshot = nullptr;
  const auto kind = execution.kind();
  if (kind == RecordKind::Expanded ||
      (positioned && (kind == RecordKind::Full || kind == RecordKind::Slice)))
    if (const auto at = execution.ref())
      if (const auto found = register_state_before.find(*at); found != register_state_before.end())
        snapshot = &found->second;
  return ResolveExpression(
      *facts, execution, Strip(operand),
      [&](std::string_view name) -> std::optional<TmemPointer> {
        if (snapshot) {
          for (const auto &[reg, pointer] : *snapshot)
            if (reg == name)
              return pointer;
        }
        return {};
      });
}
ResolvedTmemOperand
TmemProvenanceIndex::Resolve(std::uint32_t warp,
                             const RecordView &execution,
                             std::size_t operand, std::uint32_t lane) const {
  if (!facts)
    return {};
  const auto &instruction =
      facts->function().instructions[execution.instruction_id()];
  if (operand >= instruction.operands.size())
    return {};
  return Resolve(warp, execution,
                 facts->module().view(instruction.operands[operand]), lane);
}
std::optional<TmemPointer>
TmemProvenanceIndex::NormalizeTaddr(const ResolvedTmemOperand &value) const {
  if (const auto *pointer = std::get_if<TmemPointer>(&value))
    return *pointer;
  if (const auto *integer = std::get_if<Integer>(&value);
      integer && allocations.size() == 1 &&
      (allocations.front().columns == 512 ||
       zero_based_allocations.contains(allocations.front().identifier)) && *integer >= 0 &&
      *integer <= std::numeric_limits<std::uint32_t>::max()) {
    return TmemPointer{allocations.front().identifier,
                       integer->convert_to<std::uint32_t>()};
  }
  return {};
}
const TmemAllocation *
TmemProvenanceIndex::AllocationById(TmemAllocationId id) const {
  const auto found = std::lower_bound(
      allocations.begin(), allocations.end(), id,
      [](const TmemAllocation &allocation, const TmemAllocationId &key) {
        return allocation.identifier < key;
      });
  return found != allocations.end() && found->identifier == id ? &*found
                                                               : nullptr;
}
TmemLifetimeAnalysis AnalyzeTmemLifetimes(const CtaFacts &facts, ExpansionSite site,
                                          const TmemAnalysisOptions &options) {
  const auto &function = facts.function();
  const auto atoms = MakeAtoms(facts, site, options);
  TmemLifetimeAnalysis result;
  result.provenance = BuildProvenance(facts, site, atoms, options);
  auto &provenance = result.provenance;
  std::set<TmemLifetimeIssue> issues(provenance.issues.begin(),
                                     provenance.issues.end());
  std::set<TmemUse> uses;
  std::set<std::pair<TmemAtomSite, TmemAllocationId>> deallocations;
  std::map<TmemAllocationId, TmemAllocationId> deallocation_by_site;
  std::map<TmemAllocationId, std::set<TmemAtomSite>> dealloc_by_allocation;
  std::set<TmemOrderObligation> obligations;
  for (const auto &[key, atom] : atoms) {
    const auto &instruction =
        function.instructions[key.instruction_index];
    const auto &operation = instruction.semantics.operation;
    const auto issue = [&](std::string code, std::string message) {
      issues.insert({std::move(code), std::move(message), atom.site});
    };
    if (operation != "alloc" && !FullWarp(atom))
      issue("tmem_noncollective_lifetime",
            "tcgen05." + operation +
                " must be executed by all 32 threads of its warp");
    if (operation != "dealloc" || !FullWarp(atom))
      continue;
    const auto group = CtaGroup(instruction);
    if (group != 1 && !(options.allow_cta_group2 && group == 2)) {
      issue("tmem_cta_group_2",
            "cta_group::2 TMEM lifetime operation spans two CTAs");
      continue;
    }
    if (instruction.operands.size() != 2) {
      issue("tmem_malformed_dealloc",
            "tcgen05.dealloc must provide taddr and nCols");
      continue;
    }
    std::set<TmemPointer> pointers;
    bool unresolved = false;
    for (const auto &[lane, execution] : atom.lanes) {
      auto pointer = provenance.NormalizeTaddr(
          provenance.Resolve(atom.site.warp, execution, std::size_t{0}, lane));
      if (pointer)
        pointers.insert(*pointer);
      else
        unresolved = true;
    }
    if (unresolved) {
      issue("tmem_unresolved_dealloc_pointer",
            "deallocation taddr has no lane-uniform allocation provenance");
      continue;
    }
    if (pointers.size() != 1) {
      issue("tmem_nonuniform_dealloc_pointer",
            "deallocation lanes select different allocation pointers");
      continue;
    }
    const auto &pointer = *pointers.begin();
    const auto *allocation = provenance.AllocationById(pointer.allocation);
    if (!allocation) {
      issue("tmem_unknown_deallocation",
            "deallocation refers to an unknown allocation identity");
      continue;
    }
    if (pointer.encoded_offset) {
      issue("tmem_dealloc_nonbase",
            "deallocation taddr is not the allocation base");
      continue;
    }
    std::set<Integer> columns;
    for (const auto &[lane, execution] : atom.lanes) {
      const auto value =
          provenance.Resolve(atom.site.warp, execution, std::size_t{1}, lane);
      if (const auto *integer = std::get_if<Integer>(&value))
        columns.insert(*integer);
      else
        unresolved = true;
    }
    if (unresolved) {
      issue("tmem_unresolved_dealloc_size",
            "deallocation nCols is not statically known in every lane");
      continue;
    }
    if (columns.size() != 1) {
      issue("tmem_nonuniform_dealloc_size",
            "deallocation lanes use different nCols values");
      continue;
    }
    if (*columns.begin() != allocation->columns) {
      issue("tmem_dealloc_size_mismatch",
            "deallocation nCols=" + columns.begin()->str() +
                " does not match allocation nCols=" +
                std::to_string(allocation->columns));
      continue;
    }
    deallocation_by_site[key] = pointer.allocation;
    dealloc_by_allocation[pointer.allocation].insert(atom.site);
    deallocations.emplace(atom.site, pointer.allocation);
  }
  std::size_t count = 0;
  for (std::size_t index = 0; index < facts.LaneCount(); ++index) {
    const auto lane = facts.Lane(index);
    const auto warp = facts.Warp(lane), lane_id = facts.LaneId(lane);
    std::set<TmemAllocationId> freed;
    LaneRecords records(facts, lane, site);
    for (std::size_t pos = 0; pos < records.size(); ++pos) {
      Poll(options, count);
      const auto execution = records.Meta(pos);
      const auto &instruction =
          function.instructions[execution.instruction_id()];
      const auto &semantics = instruction.semantics;
      if (semantics.category == "tmem_lifetime") {
        if (semantics.operation != "dealloc") {
          const auto group = CtaGroup(instruction);
          if (group != 1 && !(options.allow_cta_group2 && group == 2))
            issues.insert(
                {"tmem_cta_group_2",
                 "cta_group::2 TMEM lifetime operation spans two CTAs",
                 Site(facts, warp, execution)});
        } else {
          const auto found = deallocation_by_site.find(
              {warp, instruction.id, execution.occurrence()});
          if (found != deallocation_by_site.end())
            freed.insert(found->second);
        }
        continue;
      }
      if (instruction.opcode != "tcgen05" || semantics.category != "async")
        continue;
      const auto site = Site(facts, warp, execution);
      if (CtaGroup(instruction) == 2 && !options.allow_cta_group2) {
        issues.insert({"tmem_cta_group_2",
                       "cta_group::2 tcgen05 access spans two CTAs", site});
        continue;
      }
      for (const auto operand : TmemOperands(facts, instruction)) {
        const auto pointer = provenance.NormalizeTaddr(provenance.Resolve(
            warp, records.Operand(pos, operand), operand, lane_id));
        if (!pointer) {
          issues.insert({"tmem_unresolved_use_pointer",
                         "tcgen05." + semantics.operation +
                             " taddr has no unique allocation provenance",
                         site});
          continue;
        }
        uses.insert({site, pointer->allocation, pointer->encoded_offset,
                     semantics.operation});
        if (freed.contains(pointer->allocation))
          issues.insert({"tmem_use_after_free",
                         "tcgen05." + semantics.operation + " uses " +
                             pointer->ResourceName() +
                             " after its deallocation",
                         site});
      }
    }
  }
  for (const auto &allocation : provenance.allocations) {
    const auto found = dealloc_by_allocation.find(allocation.identifier);
    if (found == dealloc_by_allocation.end() || found->second.empty())
      issues.insert(
          {"tmem_missing_dealloc",
           allocation.identifier.ResourceName() + " is never deallocated",
           allocation.site});
    else if (found->second.size() > 1)
      issues.insert({"tmem_multiple_dealloc",
                     allocation.identifier.ResourceName() + " has " +
                         std::to_string(found->second.size()) +
                         " distinct deallocation occurrences",
                     *found->second.begin()});
  }
  const auto retirement = RetirementSites(facts, site, options);
  std::set<TmemAllocationId> relevant;
  for (const auto &[key, _] : atoms)
    relevant.insert(key);
  for (const auto &publication : provenance.publication_reads)
    relevant.insert(publication.site.Key());
  for (const auto &use : uses)
    relevant.insert(use.site.Key());
  for (const auto &[_, wait] : retirement)
    relevant.insert(wait.Key());
  const auto positions = RelevantPositions(facts, relevant, options);
  const auto before = [&](const TmemAtomSite &source,
                          const TmemAtomSite &target) {
    return Precedes(positions, source, target);
  };
  for (std::size_t i = 0; i < provenance.allocations.size(); ++i) {
    const auto &left = provenance.allocations[i];
    for (std::size_t j = i + 1; j < provenance.allocations.size(); ++j) {
      const auto &right = provenance.allocations[j];
      if (left.columns == right.columns)
        continue;
      const auto &larger = left.columns > right.columns ? left : right;
      const auto &smaller = left.columns > right.columns ? right : left;
      if (before(larger.site, smaller.site))
        continue;
      if (before(smaller.site, larger.site))
        issues.insert({"tmem_increasing_alloc_size",
                       "allocation nCols increases from " +
                           std::to_string(smaller.columns) + " to " +
                           std::to_string(larger.columns),
                       larger.site});
      else
        obligations.insert({"larger_allocation_before_smaller",
                            larger.identifier, larger.site, smaller.site,
                            "the " + std::to_string(larger.columns) +
                                "-column allocation must happen before the " +
                                std::to_string(smaller.columns) +
                                "-column allocation so nCols never increases"});
    }
  }
  for (const auto &allocation : provenance.allocations) {
    for (const auto &[key, atom] : atoms) {
      if (function.instructions[key.instruction_index]
              .semantics.operation != "relinquish_alloc_permit")
        continue;
      if (before(allocation.site, atom.site))
        continue;
      if (before(atom.site, allocation.site))
        issues.insert({"tmem_alloc_after_relinquish",
                       "tcgen05.alloc executes after allocation permission was "
                       "relinquished",
                       allocation.site});
      else
        obligations.insert({"allocation_before_relinquish",
                            allocation.identifier, allocation.site, atom.site,
                            "allocation must happen before the cross-warp "
                            "relinquish_alloc_permit occurrence"});
    }
  }
  for (const auto &publication : provenance.publication_reads) {
    const auto &allocation = *provenance.AllocationById(publication.allocation);
    if (!before(allocation.site, publication.site))
      obligations.insert({"alloc_publication_before_load",
                          publication.allocation, allocation.site,
                          publication.site,
                          "the allocation's 4-byte shared taddr write must "
                          "happen before the consumer shared load"});
  }
  for (const auto &use : uses) {
    const auto *allocation = provenance.AllocationById(use.allocation);
    if (!allocation)
      throw std::logic_error("TMEM provenance refers to missing allocation");
    if (!before(allocation->site, use.site))
      obligations.insert({"allocation_before_use", use.allocation,
                          allocation->site, use.site,
                          use.allocation.ResourceName() +
                              " must be allocated before tcgen05." +
                              use.operation + " uses it"});
  }
  for (const auto &[site, id] : deallocations) {
    const auto &allocation = *provenance.AllocationById(id);
    if (before(allocation.site, site))
      continue;
    if (before(site, allocation.site))
      issues.insert({"tmem_dealloc_before_alloc",
                     "tcgen05.dealloc executes before its allocation", site});
    else
      obligations.insert(
          {"allocation_before_dealloc", id, allocation.site, site,
           "allocation must happen before its cross-warp deallocation"});
  }
  for (const auto &use : uses) {
    const auto sites = dealloc_by_allocation.find(use.allocation);
    if (sites == dealloc_by_allocation.end())
      continue;
    for (const auto &site : sites->second) {
      const auto wait = retirement.find(use.site);
      if (wait != retirement.end() && before(wait->second, site))
        continue;
      obligations.insert(
          {"access_completion_before_dealloc", use.allocation, use.site, site,
           "tcgen05." + use.operation +
               " must complete before the allocation is deallocated"});
    }
  }
  result.uses.assign(uses.begin(), uses.end());
  result.deallocations.assign(deallocations.begin(), deallocations.end());
  result.issues.assign(issues.begin(), issues.end());
  result.ordering_obligations.assign(obligations.begin(), obligations.end());
  result.cluster_pairing_required = std::any_of(
      provenance.allocations.begin(), provenance.allocations.end(),
      [](const auto &allocation) { return allocation.cta_group == 2; });
  return result;
}
} // namespace sentryward::frontend::ptx
