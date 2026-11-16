#pragma once
#include "execution_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
State Executor::Initial(const WarpContract &w, std::uint32_t lane) const {
  State s;
  s.registers.resize(names_.size());
  s.origins.resize(names_.size());
  s.occurrences.resize(decoded_.size());
  s.branch_visits.resize(decoded_.size());
  auto set = [&](std::string_view name, Integer value) {
    s.registers[ids_.at(std::string(name))] = std::move(value);
  };
  auto thread = w.warp_id * c_.warp_size + lane;
  set("%laneid", lane);
  set("%warpid", w.warp_id);
  set("%nwarpid", c_.warps.size());
  set("%tid.x", thread % c_.block_dim[0]);
  set("%tid.y", thread / c_.block_dim[0] % c_.block_dim[1]);
  set("%tid.z", thread / (c_.block_dim[0] * c_.block_dim[1]));
  for (std::size_t j = 0; j < 3; ++j) {
    set(std::string("%ntid.") + "xyz"[j], c_.block_dim[j]);
    set(std::string("%ctaid.") + "xyz"[j], 0);
    if (c_.cluster_dim)
      set(std::string("%cluster_nctaid.") + "xyz"[j], (*c_.cluster_dim)[j]);
  }
  set("%cluster_ctarank", 0);
  for (const auto &[name, v] : c_.special_registers)
    set(name, v);
  for (const auto &[name, v] : w.initial_registers)
    set(name, v);
  return s;
}
Scalar Executor::ValueText(std::string_view text, const State &s) const {
  text = TrimView(text);
  if (auto n = ParseInteger(text))
    return *n;
  if (c_.parameter_regions.contains(std::string(text)))
    return ParameterPointer{std::string(text), 0};
  if (auto it = ids_.find(std::string(text));
      it != ids_.end() &&
      (!Unknown(s.registers[it->second]) || s.origins[it->second]))
    return s.registers[it->second];
  if (auto it = c_.parameters.find(std::string(text));
      it != c_.parameters.end())
    return it->second;
  if (text == "true" || text == "p")
    return true;
  if (text == "false" || text == "!p")
    return false;
  return {};
}
void Executor::Set(const Decoded &d, State &s,
                   std::vector<Scalar> values) const {
  for (std::size_t n = 0; n < d.destination_slots.size(); ++n)
    if (d.destination_slots[n] != kMissing)
      s.registers[d.destination_slots[n]] =
          n < values.size() ? std::move(values[n]) : Scalar{};
}
void Executor::Origins(const Decoded &d, State &s,
                       std::uint64_t occurrence) const {
  DynamicKey origin{d.instruction->id, occurrence};
  std::vector<unsigned> widths;
  if (d.instruction->opcode == "cvt")
    for (const auto &modifier : d.instruction->modifiers)
      if (modifier.size() > 1 && In(modifier.substr(0,1), {"b","u","s"}) &&
          std::all_of(modifier.begin()+1,modifier.end(), [](unsigned char c) { return std::isdigit(c); }))
        widths.push_back(std::stoul(modifier.substr(1)));
  const bool narrowing = widths.size() == 2 && widths[0] < widths[1];
  if (!narrowing && In(d.instruction->opcode, {"mov", "cvt"}) && d.operands.size() > 1 &&
      d.operands[1].registers.size() == 1)
    if (auto known = s.origins[d.operands[1].registers[0]])
      origin = *known;
  for (auto id : d.destinations)
    s.origins[id] = origin;
}
ExecutedInstruction Executor::Record(const Decoded &d, const State &s,
                                     std::uint64_t occurrence,
                                     std::optional<bool> predicate) const {
  ExecutedInstruction out;
  auto &metadata = out.mutableMetadata();
  auto &bindings = out.mutableKnownRegisters();
  metadata.instruction_id = d.instruction->id;
  metadata.occurrence = occurrence;
  metadata.predicate_value = predicate;
  bindings.reserve(d.uses.size());
  metadata.register_origins.reserve(d.uses.size());
  for (auto id : d.uses) {
    if (!Unknown(s.registers[id]))
      bindings.emplace_back(id, s.registers[id]);
    if (s.origins[id])
      metadata.register_origins.emplace_back(id, *s.origins[id]);
  }
  return out;
}
std::optional<Executor::ShuffleSelection> Executor::SelectShuffle(
    const Decoded& d,const State& state,std::uint32_t warp,std::uint32_t lane) const {
  const auto& i=*d.instruction;
  auto source=ScalarInteger(Value(d.operands[2],state));
  auto clamp=ScalarInteger(Value(d.operands[3],state));
  auto mask=ScalarInteger(Value(d.operands[4],state));
  if(!source || !clamp || !mask)return {};
  const auto members=(*mask&0xffffffff).convert_to<std::uint32_t>();
  if(!(members&(1u<<lane)))Error(i,warp,lane,"executing lane absent from shuffle membermask");
  const auto b=(*source&31).convert_to<unsigned>();
  const auto c=(*clamp&31).convert_to<unsigned>();
  const auto segment=((*clamp>>8)&31).convert_to<unsigned>();
  const auto inverse=(~segment)&31;
  const auto maximum=(lane&segment)|(c&inverse);
  int from=(lane&segment)|(b&inverse);
  if(i.modifiers[1]=="up")from=int(lane)-int(b);
  if(i.modifiers[1]=="down")from=lane+b;
  if(i.modifiers[1]=="bfly")from=lane^b;
  const bool valid=i.modifiers[1]=="up"?from>=int(maximum):from<=int(maximum);
  if(!valid)from=lane;
  if(!(members&(1u<<from)))Error(i,warp,lane,"selected shuffle source absent from membermask");
  return ShuffleSelection{static_cast<std::uint32_t>(from),valid};
}
} // namespace
} // namespace sentryward::frontend::ptx
