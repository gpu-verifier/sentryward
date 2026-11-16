#pragma once
#include "value_ir_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
const PartitionSpec* ValueTemplateExtractor::PartitionRule(const Instruction& i) const {
  // Only an opcode some rule can whitelist gets an entry (the cache stays
  // sparse); its flags and rule are computed once per function.
  if (!In(i.opcode,{"cp","ld","mma","ldmatrix","stmatrix","shfl","elect","tcgen05","vote","redux"})) return nullptr;
  if (!partition_specs) throw std::logic_error("Partition without the value view's rule cache");
  auto& rules=partition_specs->rules;
  auto found=rules.find(i.id);
  if (found==rules.end()) {
    std::unique_ptr<const PartitionSpec> rule;
    const auto kind=PartitionKindsOf(i);
    if (kind.bulk || kind.copy || kind.mma || kind.matrix || kind.shuffle || kind.election || kind.allocation ||
        kind.tc_mma || kind.parameter || kind.reduction)
      if (auto spec=BuildPartitionSpec(module_,i,kind)) rule=std::make_unique<const PartitionSpec>(std::move(*spec));
    found=rules.emplace(i.id,std::move(rule)).first;
  }
  return found->second.get();
}

std::vector<std::vector<LaneRef>> ValueTemplateExtractor::Partition() const {
  if (Done() || members.size()<2) return {};
  const auto meta=facts_.Meta({lane_,cursor_});
  const auto& i=function_.instructions.at(meta.instruction_id());
  if (!(*live_)[cursor_] || meta.predicate_value()==false) return {};
  const auto* spec=PartitionRule(i);
  if (!spec) return {};
  // Per member: the fact-stage class, then the structural keys of its matrix
  // peers, so a veto or an exception arises at the member that causes it.
  // Parts are numbered by first appearance, so they are ordered by first
  // member and part 0 holds the leader.
  auto stage1=facts_.BeginPartition(members,cursor_,*spec);
  std::vector<std::vector<LaneRef>> parts;
  std::unordered_map<std::string,std::uint32_t> numbering; // "<class>:" ++ stage-2 key -> part
  std::string key;
  MemberFacts derived;
  try {
  while (!stage1.Done()) {
    const auto member=stage1.Member();
    // The member's stage-1 key enters the part key through its class.
    stage1.Next(derived);
    key.assign(std::to_string(stage1.Class(stage1.Evaluated()-1)));
    key+=':';
    if (matrix_pending_) {
      const auto found=matrix_addresses->find({facts_.Warp(member),i.id,facts_.Meta({member,cursor_}).occurrence(),0,
                                               origin_.unroll_context});
      if (found==matrix_addresses->end() || found->second.arrived!=UINT32_MAX) {
        stage1.Finish(query::PartitionEnd::Vetoed);
        return {};
      }
      const auto shape=DecodeMatrixShape(module_,i,i.opcode=="stmatrix");
      for (unsigned m=0;m<shape.count;++m) for (unsigned half=0;half<(shape.transpose?2u:1u);++half) {
        const unsigned peer=8*m+(shape.transpose?2*(facts_.LaneId(member)%4)+half:facts_.LaneId(member)/4);
        key+="source:";
        key+=graph_.structuralKey(found->second.addresses[peer]->value,origin_);
      }
    }
    const auto [part,added]=numbering.try_emplace(key,static_cast<std::uint32_t>(parts.size()));
    if (added) parts.emplace_back();
    parts[part->second].push_back(member);
  }
  } catch (...) {
    // An exception ends the partition stage (recorded as Threw), then propagates.
    stage1.Finish(query::PartitionEnd::Threw);
    throw;
  }
  stage1.Finish(query::PartitionEnd::Complete);
  if (parts.size()<2) return {};
  return parts;
}

FunctionContext::FunctionContext(const Module& module, const Function& function)
    : source_file(module.source_name), function_name(function.name), local_stores(module, function) {
  instruction_origins.reserve(function.instructions.size());
  for (const auto& instruction : function.instructions) {
    auto& origin = instruction_origins.emplace_back();
    origin.source_text = std::string(module.view(instruction.source));
    origin.scope = ScopeText(instruction.scope);
    if (instruction.predicate)
      origin.predicate = (instruction.predicate_negated ? "!" : "") + *instruction.predicate;
  }
  static const std::regex type_regex("\\.(pred|bf[0-9]+(?:x[0-9]+)?|[busf][0-9]+(?:x[0-9]+)?)\\b");
  static const std::regex name_regex("(%?[A-Za-z_$][A-Za-z0-9_$]*)(?:<([0-9]+)>)?");
  const auto parse = [&](const Directive &directive) {
    if (!In(directive.name, {".reg", ".shared", ".global", ".local", ".const", ".param"})) return;
    const auto text = std::string(module.view(directive.arguments));
    std::smatch type_match;
    if (!std::regex_search(text, type_match, type_regex)) return;
    auto type = PtxType(type_match[1].str());
    if (!type) return;
    const auto names = text.substr(static_cast<std::size_t>(type_match.position() + type_match.length()));
    for (const auto &part : Split(names, ',')) {
      std::smatch name;
      if (!std::regex_search(part, name, name_regex)) continue;
      if (directive.name != ".reg") {
        symbols.emplace(directive.scope, name[1].str());
        symbol_spaces[{directive.scope, name[1].str()}] = directive.name.substr(1);
        continue;
      }
      if (name[2].matched) {
        const auto count = std::stoul(name[2]);
        if (count > 100000) throw std::invalid_argument("excessive PTX register declaration");
        for (std::size_t n = 0; n < count; ++n)
          declarations.emplace(Key{directive.scope, name[1].str() + std::to_string(n)}, *type);
      } else declarations.emplace(Key{directive.scope, name[1].str()}, *type);
    }
  };
  for (const auto &directive : module.directives) parse(directive);
  for (const auto &directive : function.directives) parse(directive);
  for (const auto &parameter : function.parameters)
    if (parameter.name) symbols.emplace(Scope{}, *parameter.name);
  for (const auto& instruction : function.instructions)
    if (instruction.branch_target && *instruction.branch_target <= instruction.id)
      loop_ends[*instruction.branch_target] = std::max(loop_ends[*instruction.branch_target], instruction.id);
}
} // namespace
} // namespace sentryward::frontend::ptx
