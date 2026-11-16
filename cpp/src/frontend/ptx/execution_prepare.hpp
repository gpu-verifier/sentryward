#pragma once
#include "execution_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
void Executor::UnresolvedControl(const Instruction &instruction, const State &state,
                                  const LaneTrace &trace, std::uint32_t reg) const {
  std::map<DynamicKey,const ExecutedInstruction *> definitions;
  for (const auto &e : detail::ExecutionStorage::Records(trace.instructions)) definitions[e.Key()] = &e;
  std::set<std::pair<DynamicKey, std::uint32_t>> visited;
  std::set<std::tuple<std::string,std::uint64_t,unsigned>> missing;
  Json required = Json::array();
  bool shared_control = false;
  std::function<void(DynamicKey, std::uint32_t)> visit = [&](DynamicKey key, std::uint32_t result) {
    if (!visited.emplace(key, result).second || !definitions.contains(key)) return;
    const auto &e = *definitions.at(key);
    const auto &d = decoded_[e.instruction_id()];
    const auto &i = *d.instruction;
    shared_control |= i.opcode == "ld" &&
        (Has(i,"shared") || Has(i,"shared::cta") || Has(i,"shared::cluster"));
    if (i.opcode == "ld" && Has(i,"param") && i.memory_width && !d.destinations.empty() &&
        d.operands.size() == 2 && d.operands[1].address) {
      auto [name,offset] = *d.operands[1].address;
      if (auto id = ids_.find(name); id != ids_.end())
        for (const auto &[r,v] : detail::ExecutionStorage::Bindings(e)) if (r == id->second)
          if (auto p = std::get_if<ParameterPointer>(&v)) { name = p->region; offset += p->offset; }
      auto region = c_.parameter_regions.find(name);
      const auto width = *i.memory_width / d.destinations.size();
      if (region != c_.parameter_regions.end())
        for (std::size_t j = 0; j < d.destinations.size(); ++j) {
          if (d.destinations[j] != result) continue;
          const Integer begin = offset + j * width;
          if (begin < 0 || begin > INT64_MAX || begin + width > region->second.size_bytes) continue;
          const auto off = begin.convert_to<std::uint64_t>();
          if (!pointers_.contains({name,off,width}) && !region->second.ReadInteger(off,width,false) &&
              missing.emplace(name,off,width).second)
            required.push_back({{"parameter",name}, {"offset",off}, {"bytes",width}, {"ptx_line",i.line}});
        }
    }
    for (const auto &[operand,origin] : e.register_origins()) visit(origin, operand);
  };
  if (reg < state.origins.size() && state.origins[reg]) visit(*state.origins[reg], reg);
  const std::string code = required.empty() ? "branch_not_static" : "missing_parameter_value";
  ExecutionError error(code, instruction.line, "control predicate is not statically determined", std::move(required));
  error.shared_control = shared_control;
  throw error;
}
std::optional<std::pair<std::uint32_t, Integer>>
Executor::Assertion(const Instruction &branch, std::uint32_t target) const {
  if (!branch.predicate || branch.predicate_negated || branch.id == 0 ||
      target <= branch.id + 1 || target > decoded_.size()) return {};
  for (auto k = branch.id + 1; k < target; ++k)
    if (decoded_[k].instruction->predicate) return {};
  const auto &terminal = *decoded_[target - 1].instruction;
  if (terminal.control == "trap") {
    if (target != branch.id + 2) return {};
  } else if (terminal.control == "call" && !terminal.operands.empty() &&
             TrimView(m_.view(terminal.operands[0])) == "__assertfail") {
    for (auto k = branch.id + 1; k + 1 < target; ++k) {
      const auto &i = *decoded_[k].instruction;
      if (!(In(i.opcode, {"mov", "cvta"}) || (i.opcode == "st" && Has(i, "param")))) return {};
    }
  } else return {};
  const auto &compare = decoded_[branch.id - 1];
  const auto &i = *compare.instruction;
  if (i.opcode != "setp" || !Has(i, "eq") || i.predicate ||
      compare.operands.size() != 3 || compare.operands[0].text != *branch.predicate) return {};
  for (auto [r, literal] : {std::pair{1u,2u}, std::pair{2u,1u}})
    if (compare.operands[r].reg != kMissing)
      if (auto n = ParseInteger(compare.operands[literal].text)) return {{compare.operands[r].reg, *n}};
  return {};
}
void Executor::Prepare() {
  const auto local_declarations = LocalMemoryDeclarations(m_, f_);
  local_memory_ = !local_declarations.empty();
  ProfileScope prepare("frontend.execute.prepare");
  for (const auto &entry : c_.fixed_input.value("global_memory_inputs", Json::array())) {
    const auto name = entry.at("resource").get<std::string>();
    const auto bits = entry.at("element_bits").get<unsigned>();
    if (bits != 8 && bits != 16 && bits != 32 && bits != 64)
      throw std::invalid_argument("constant input has unsupported element width");
    ParameterRegion region;
    region.name = name;
    region.source = {{"kind", "typed_fields"}, {"fields", Json::array()}};
    auto offset = entry.value("byte_offset", std::uint64_t(0));
    for (const auto &value : entry.at("values")) {
      Integer n = JsonInteger(value) & ((Integer(1) << bits) - 1);
      region.offset_values[offset] = {bits / 8, n};
      if (offset > UINT64_MAX - bits / 8) throw std::invalid_argument("constant input overflows address range");
      offset += bits / 8;
    }
    region.size_bytes = offset;
    if (!constant_inputs_.emplace(name, std::move(region)).second)
      throw std::invalid_argument("duplicate constant input");
    constant_elements_[name] = {bits / 8, AllocationIdentity(AllocationPointer{name, 0, std::nullopt}, c_),
                                entry.value("row_elements", std::uint64_t(0))};
  }
  table_atoms_ = c_.fixed_input.value("table_atoms", false);
  for (const auto &table : c_.fixed_input.value("pointer_array_inputs", Json::array()))
    for (const auto &entry : table.at("entries")) {
      auto &slot = pointer_tables_[table.at("resource").get<std::string>()][entry.at("slot_offset").get<std::uint64_t>()];
      if (entry.value("null", false)) slot = Integer(0);
      else
        slot = AllocationPointer{entry.at("resource").get<std::string>(), Integer(entry.at("offset").get<std::uint64_t>()),
                                 entry.contains("extent_bytes") ? std::optional(entry.at("extent_bytes").get<std::uint64_t>())
                                                                : std::nullopt};
    }
  for (const auto &entry : c_.fixed_input.value("tensor_property_inputs", Json::array()))
    tensor_properties_[entry.at("resource").get<std::string>()] = {entry.at("object").get<std::string>(),
        entry.at("element_bits").get<unsigned>(), entry.at("signed").get<bool>(),
        JsonInteger(entry.at("lower_bound")), JsonInteger(entry.at("upper_bound"))};
  if (!f_.has_body || f_.instructions.empty())
    throw std::runtime_error("PTX entry has no executable body");
  ProfileScope registers("frontend.execute.prepare.registers");
  for (const auto &i : f_.instructions) {
    if (i.id >= f_.instructions.size())
      throw std::runtime_error("non-dense PTX instruction ids");
    convergence_ |= i.opcode == "activemask";
    for (const auto &r : i.registers)
      Intern(r);
    for (const auto &r : i.destinations)
      Intern(r);
    if (i.predicate)
      Intern(*i.predicate);
    for (auto range : i.operands) {
      auto text = TrimView(m_.view(range));
      for (auto r : RegistersIn(text))
        Intern(r);
      if (text.starts_with('[') && text.ends_with(']'))
        if (auto split = SplitAddress(text.substr(1, text.size() - 2)))
          if (!ParseInteger(split->first))
            Intern(split->first);
    }
  }
  for (const auto &[r, v] : c_.special_registers)
    Intern(r);
  for (const auto &w : c_.warps)
    for (const auto &[r, v] : w.initial_registers)
      Intern(r);
  for (auto r : {"%laneid", "%warpid", "%nwarpid", "%tid.x", "%tid.y", "%tid.z",
                 "%ntid.x", "%ntid.y", "%ntid.z", "%ctaid.x", "%ctaid.y",
                 "%ctaid.z", "%cluster_ctarank", "%cluster_nctaid.x",
                 "%cluster_nctaid.y", "%cluster_nctaid.z"})
    Intern(r);
  std::map<std::string, std::set<std::uint32_t>> raw_labels;
  for (const auto &label : f_.labels) {
    label_entries_.insert(label.instruction);
    raw_labels[label.name].insert(label.instruction);
    std::string qualified;
    for (auto id : label.scope) {
      if (!qualified.empty())
        qualified += '/';
      qualified += std::to_string(id);
    }
    if (!qualified.empty())
      qualified += "::";
    qualified += label.name;
    labels_[qualified] = label.instruction;
  }
  for (const auto &[name, targets] : raw_labels)
    if (targets.size() == 1)
      labels_[name] = *targets.begin();
  registers.nodes = names_.size();
  registers.edges = labels_.size();
  registers.Stop();
  ProfileScope decode("frontend.execute.prepare.decode");
  decoded_.resize(f_.instructions.size());
  for (const auto &i : f_.instructions) {
    auto &d = decoded_[i.id];
    d.instruction = &i;
    if (i.predicate)
      d.predicate = ids_.at(*i.predicate);
    for (const auto &r : i.destinations)
      d.destinations.push_back(ids_.at(r));
    for (auto range : i.operands) {
      Operand o;
      o.text = std::string(TrimView(m_.view(range)));
      for (const auto &r : RegistersIn(o.text))
        o.registers.push_back(ids_.at(r));
      if (o.text.starts_with('[') && o.text.ends_with(']'))
        o.address =
            SplitAddress(std::string_view(o.text).substr(1, o.text.size() - 2));
      if (o.address) {
        auto found = ids_.find(o.address->first);
        if (found != ids_.end() &&
            std::find(o.registers.begin(), o.registers.end(), found->second) ==
                o.registers.end())
          o.registers.push_back(found->second);
      }
      if (auto n = ParseInteger(o.text))
        o.constant = *n;
      else if (o.text.size() >= 3 && o.text[0] == '0' &&
               In(o.text.substr(1, 1), {"f", "F", "d", "D"}) &&
               o.text.size() == (std::tolower(o.text[1]) == 'f' ? 10u : 18u)) {
        auto n = ParseInteger("0x" + o.text.substr(2));
        if (n)
          o.constant = *n;
      } else if (o.text == "true" || o.text == "p")
        o.constant = true;
      else if (o.text == "false" || o.text == "!p")
        o.constant = false;
      else if (c_.parameter_regions.contains(o.text))
        o.constant = ParameterPointer{o.text, 0};
      else {
        if (auto found = ids_.find(o.text); found != ids_.end())
          o.reg = found->second;
        if (auto found = c_.parameters.find(o.text);
            found != c_.parameters.end())
          o.constant = found->second;
      }
      const LocalMemoryDeclaration* local = nullptr;
      for (const auto& declaration : local_declarations)
        if (declaration.name == o.text && declaration.scope.size() <= i.scope.size() &&
            std::equal(declaration.scope.begin(), declaration.scope.end(), i.scope.begin()) &&
            (!local || declaration.scope.size() > local->scope.size())) local = &declaration;
      if (local) o.constant = AllocationPointer{local->resource, 0, local->bytes};
      for (auto id : o.registers)
        if (std::find(d.uses.begin(), d.uses.end(), id) == d.uses.end())
          d.uses.push_back(id);
      d.operands.push_back(std::move(o));
    }
    if (d.predicate != kMissing &&
        std::find(d.uses.begin(), d.uses.end(), d.predicate) == d.uses.end())
      d.uses.push_back(d.predicate);
    if (!d.operands.empty() &&
        d.operands[0].text.find('|') != std::string::npos) {
      auto text = d.operands[0].text;
      std::size_t begin = 0;
      while (true) {
        auto end = text.find('|', begin);
        auto regs = RegistersIn(std::string_view(text).substr(
            begin,
            end == std::string::npos ? text.size() - begin : end - begin));
        d.destination_slots.push_back(regs.empty() ? kMissing
                                                   : ids_.at(regs[0]));
        if (end == std::string::npos)
          break;
        begin = end + 1;
      }
    } else
      d.destination_slots = d.destinations;
    d.wait_loop = WaitLoop(i);
    d.replay_wait_result = d.wait_loop && i.id+1<f_.instructions.size() &&
        f_.instructions[i.id+1].opcode=="selp";
    decode.edges += d.operands.size();
  }
  decode.nodes = decoded_.size();
  decode.Stop();
  ProfileScope bindings("frontend.execute.prepare.pointer_bindings");
  std::map<std::string, std::vector<std::pair<std::uint64_t, std::uint64_t>>>
      ranges;
  for (const auto &p : c_.pointer_parameters) {
    const Parameter *parameter = nullptr;
    if (p.parameter_index) {
      if (*p.parameter_index >= f_.parameters.size())
        throw std::runtime_error("pointer parameter index outside PTX ABI");
      parameter = &f_.parameters[*p.parameter_index];
    } else
      for (const auto &item : f_.parameters)
        if (item.name == p.parameter) {
          parameter = &item;
          break;
        }
    if (!parameter || !parameter->name)
      throw std::runtime_error("pointer parameter missing from PTX ABI");
    std::optional<std::uint64_t> size;
    if (parameter->array) {
      auto text = *parameter->array;
      if (text.size() > 2 && text.front() == '[' && text.back() == ']') {
        auto n =
            ParseInteger(std::string_view(text).substr(1, text.size() - 2));
        if (n && *n >= 0 && *n <= UINT64_MAX)
          size = n->convert_to<std::uint64_t>();
      }
    } else
      for (const auto &type : parameter->types)
        if (type.size() > 2 && type.front() == '.' &&
            In(type.substr(1, 1), {"b", "u", "s", "f"})) {
          auto n = ParseInteger(type.substr(2));
          if (n && *n > 0 && *n % 8 == 0)
            size = (*n / 8).convert_to<std::uint64_t>();
        }
    if (!size || p.byte_offset > *size || p.width_bytes > *size - p.byte_offset)
      throw std::runtime_error(
          "pointer field exceeds PTX ABI parameter storage");
    auto end = p.byte_offset + p.width_bytes;
    for (auto [a, b] : ranges[*parameter->name])
      if (p.byte_offset < b && a < end)
        throw std::runtime_error("overlapping resolved pointer fields");
    ranges[*parameter->name].push_back({p.byte_offset, end});
    pointers_.emplace(
        std::tuple{*parameter->name, p.byte_offset, p.width_bytes}, &p);
  }
  bindings.nodes = pointers_.size();
  prepare.nodes = decoded_.size();
  prepare.edges = names_.size();
}
// Build a basic-block CFG once per executor. Dominators use immediate-parent
// chains, rather than a quadratic matrix over PTX instructions.
} // namespace
} // namespace sentryward::frontend::ptx
