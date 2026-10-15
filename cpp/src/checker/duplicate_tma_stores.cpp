#include "sentryward/algorithm/checker/duplicate_tma_stores.h"

#include <string_view>
#include <map>
#include <tuple>
#include <vector>
#include <algorithm>
#include <set>

namespace sentryward::algorithm {
namespace {

bool isDenseTmaStore(std::string_view opcode) {
  constexpr std::string_view prefix = "cp.async.bulk.tensor.";
  if (!opcode.starts_with(prefix)) return false;
  opcode.remove_prefix(prefix.size());
  enum Modifier : unsigned {
    kRank = 1, kGlobal = 2, kShared = 4, kBulkGroup = 8,
    kTile = 16, kWeak = 32, kCacheHint = 64,
  };
  unsigned seen = 0;
  while (!opcode.empty()) {
    const auto separator = opcode.find('.');
    const auto token = opcode.substr(0, separator);
    unsigned modifier;
    if (token.size() == 2 && token[0] >= '1' && token[0] <= '5' && token[1] == 'd')
      modifier = kRank;
    else if (token == "global") modifier = kGlobal;
    else if (token == "shared::cta") {
      if (!(seen & kGlobal)) return false;
      modifier = kShared;
    } else if (token == "bulk_group") modifier = kBulkGroup;
    else if (token == "tile") modifier = kTile;
    else if (token == "weak") modifier = kWeak;
    else if (token == "L2::cache_hint") modifier = kCacheHint;
    else return false;
    if (seen & modifier) return false;
    seen |= modifier;
    if (separator == std::string_view::npos) break;
    opcode.remove_prefix(separator + 1);
    if (opcode.empty()) return false;
  }
  constexpr unsigned required = kRank | kGlobal | kShared | kBulkGroup;
  return (seen & required) == required;
}

bool eligible(const SourceInstruction& instruction) {
  if (!instruction.ptx_origin || instruction.ptx_origin->file.empty()) return false;
  if (!isDenseTmaStore(instruction.ptx_origin->opcode.view())) return false;
  const auto* operation = std::get_if<AsyncOp>(&instruction.op);
  if (!operation || !operation->completion_requires.empty()) return false;
  bool reads = false, writes = false;
  for (const auto& effect : operation->effects) {
    if (effect.completion) return false;
    for (const auto& predecessor : effect.predecessors) {
      if (predecessor.instruction != instruction.id) return false;
    }
    for (const auto& access : effect.accesses) {
      if (!std::holds_alternative<AddressId>(access.endpoint)) return false;
      if (access.bytes == 0) continue;
      reads |= access.mode == AccessMode::kRead;
      writes |= access.mode == AccessMode::kWrite;
    }
  }
  return reads && writes;
}

bool sameCopy(const AsyncOp& first, const AsyncOp& second) {
  if (first.effects.size() != second.effects.size()) return false;
  for (std::size_t i = 0; i < first.effects.size(); ++i) {
    const auto& a = first.effects[i];
    const auto& b = second.effects[i];
    if (a.accesses != b.accesses || a.predecessors.size() != b.predecessors.size())
      return false;
    for (std::size_t j = 0; j < a.predecessors.size(); ++j) {
      if (a.predecessors[j].milestone != b.predecessors[j].milestone ||
          a.predecessors[j].effect_index != b.predecessors[j].effect_index)
        return false;
    }
  }
  return true;
}

}  // namespace

DuplicateTmaStores::DuplicateTmaStores(const SourceProgram& source) {
  std::unordered_map<std::uint32_t, const ActorInfo*> actors;
  for (const auto& actor : source.actors) actors.emplace(actor.id.value(), &actor);
  using Site = std::tuple<std::uint32_t, std::uint32_t, std::string_view,
                          std::uint32_t, std::uint32_t, std::string_view>;
  std::map<Site, std::vector<Entry>> sites;
  for (std::size_t i=0;i<source.instructions.size();++i) {
    if (!isDenseTmaStore(source.instructions.opcode(i).view())) continue;
    const auto instruction=source.instructions[i];
    const auto actor = actors.find(instruction.actor.value());
    if (actor == actors.end() || !eligible(instruction)) continue;
    const auto& origin = *instruction.ptx_origin;
    sites[{actor->second->cta.value(), actor->second->warp.value(), origin.file.view(),
           origin.instruction_index, origin.occurrence, origin.opcode.view()}]
        .push_back({std::make_shared<SourceInstruction>(instruction), actor->second});
  }
  std::size_t next_class = 0;
  for (const auto& [site, copies] : sites) {
    std::vector<std::vector<Entry>> classes;
    for (const auto& copy : copies) {
      const auto found = std::find_if(classes.begin(), classes.end(), [&](const auto& group) {
        return sameCopy(std::get<AsyncOp>(copy.instruction->op),
                        std::get<AsyncOp>(group.front().instruction->op));
      });
      if (found == classes.end()) classes.push_back({copy});
      else found->push_back(copy);
    }
    for (const auto& group : classes) {
      if (std::none_of(group.begin(), group.end(), [&](const auto& copy) {
            return copy.actor->lane != group.front().actor->lane;
          })) continue;
      std::set<std::uint32_t> lanes;
      for(const auto& copy:group)lanes.insert(copy.actor->lane);
      for (auto copy : group) {
        copy.copy_class = next_class;
        copy.distinct_lanes = lanes.size()==group.size();
        entries_.emplace(copy.instruction->id.value(), copy);
      }
      ++next_class;
    }
  }
}

bool DuplicateTmaStores::contains(const MemoryAccess& first,
                                 const MemoryAccess& second) const {
  if (!mayContain(first) || !mayContain(second) ||
      first.actor == second.actor || first.address != second.address ||
      first.bytes != second.bytes ||
      first.effect_index != second.effect_index ||
      first.access_index != second.access_index) return false;
  const auto a = entries_.find(first.instruction.value());
  const auto b = entries_.find(second.instruction.value());
  if (a == entries_.end() || b == entries_.end()) return false;
  const auto& x = a->second;
  const auto& y = b->second;
  return x.copy_class == y.copy_class && x.actor->lane != y.actor->lane;
}

bool DuplicateTmaStores::mayContain(const MemoryAccess& access) const {
  return access.mode == AccessMode::kWrite &&
         access.operand == AccessOperand::kEffect && access.bytes &&
         *access.bytes != 0 && entries_.contains(access.instruction.value());
}

std::optional<std::size_t> DuplicateTmaStores::copyClass(const MemoryAccess& access) const {
  if (!mayContain(access)) return {};
  const auto& entry=entries_.at(access.instruction.value());
  if(!entry.distinct_lanes)return {};
  return entry.copy_class;
}

}  // namespace sentryward::algorithm
