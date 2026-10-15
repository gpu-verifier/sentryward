#include "sentryward/frontend/ptx/public_input.h"
#include "sentryward/frontend/ptx/execution.h"
#include "sentryward/frontend/ptx/operands.h"

#include <algorithm>
#include <functional>
#include <set>
#include <variant>

namespace sentryward::frontend::ptx {
namespace {
struct Numeric {};
struct Field { std::string parameter; std::uint64_t offset; };
using Value = std::variant<std::monostate, Numeric, Integer, Field, ParameterPointer>;
bool Has(const Instruction &i, const std::string &m) {
  return std::find(i.modifiers.begin(), i.modifiers.end(), m) != i.modifiers.end();
}
std::string_view Unbracket(std::string_view text) {
  while (!text.empty() && (text.front() == '[' || std::isspace(static_cast<unsigned char>(text.front())))) text.remove_prefix(1);
  while (!text.empty() && (text.back() == ']' || std::isspace(static_cast<unsigned char>(text.back())))) text.remove_suffix(1);
  return text;
}
bool Number(const Value &v) { return std::holds_alternative<Integer>(v) || std::holds_alternative<Numeric>(v); }
} // namespace

std::vector<InferredPointer> InferPointerFields(const Module &module,
    const Function &function, const FrontendConfig &config) {
  const auto count = function.instructions.size();
  if (count == 0) return {};
  std::set<std::size_t> leaders{0};
  for (const auto &label : function.labels) if (label.instruction < count) leaders.insert(label.instruction);
  for (const auto &i : function.instructions) {
    if (i.branch_target && *i.branch_target < count) leaders.insert(*i.branch_target);
    if (i.control != "none" && i.control != "call" && i.id + 1 < count) leaders.insert(i.id + 1);
  }
  const std::vector<std::size_t> starts(leaders.begin(), leaders.end());
  std::vector<std::size_t> block_of(count);
  std::vector<std::vector<std::size_t>> predecessors(starts.size());
  std::map<std::pair<std::string, std::size_t>, std::vector<std::size_t>> definitions;
  for (std::size_t b = 0; b < starts.size(); ++b) {
    const auto end = b + 1 < starts.size() ? starts[b+1] : count;
    for (auto id = starts[b]; id < end; ++id) {
      block_of[id] = b;
      for (const auto &r : function.instructions[id].destinations) definitions[{r,b}].push_back(id);
    }
  }
  for (std::size_t b = 0; b < starts.size(); ++b) {
    const auto end = b + 1 < starts.size() ? starts[b+1] : count;
    const auto &i = function.instructions[end-1];
    if (i.branch_target && *i.branch_target < count) predecessors[block_of[*i.branch_target]].push_back(b);
    const bool terminator = i.control == "direct_branch" || i.control == "indirect_branch" ||
                            i.control == "return" || i.control == "exit" || i.control == "trap";
    if ((!terminator || i.predicate) && b + 1 < starts.size()) predecessors[b+1].push_back(b);
  }
  using Site = std::pair<std::string, std::size_t>;
  std::map<Site, std::optional<std::size_t>> reaching;
  auto definition = [&](const std::string &token, std::size_t before) -> std::optional<std::size_t> {
    const Site key{token,before};
    if (auto it = reaching.find(key); it != reaching.end()) return it->second;
    std::vector<std::pair<std::size_t,std::size_t>> pending{{block_of[before],before}};
    std::set<std::pair<std::size_t,std::size_t>> visited;
    std::set<std::size_t> found;
    while (!pending.empty() && found.size() < 2) {
      const auto [block,limit] = pending.back(); pending.pop_back();
      if (!visited.insert({block,limit}).second) continue;
      const auto &sites = definitions[{token,block}];
      auto position = std::lower_bound(sites.begin(), sites.end(), limit);
      bool stopped = false;
      while (position != sites.begin()) {
        const auto site = *--position;
        found.insert(site);
        if (!function.instructions[site].predicate) { stopped = true; break; }
      }
      if (!stopped) {
        if (block == 0 || predecessors[block].empty()) found.insert(count);
        for (auto prior : predecessors[block]) pending.emplace_back(prior,count);
      }
    }
    std::optional<std::size_t> result;
    if (found.size() == 1 && *found.begin() != count) result = *found.begin();
    reaching.emplace(key,result);
    return result;
  };
  std::map<std::tuple<std::string,std::size_t,unsigned>,Value> cache;
  std::function<Value(const std::string &,std::size_t,unsigned)> value;
  value = [&](const std::string &token, std::size_t before, unsigned depth) -> Value {
    if (depth > 96) return {};
    const auto key = std::tuple{token,before,depth};
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    auto calculate = [&]() -> Value {
      if (auto literal = ParseInteger(token)) return *literal;
      if (config.parameter_regions.contains(token)) return ParameterPointer{token,0};
      for (const auto *prefix : {"%tid.","%ntid.","%ctaid.","%nctaid.","%laneid","%warpid"})
        if (token.starts_with(prefix)) return Numeric{};
      auto site = definition(token,before);
      if (!site) return {};
      const auto &i = function.instructions[*site];
      if (i.predicate || i.operands.empty()) return {};
      std::vector<std::string> operands;
      for (auto r : i.operands) operands.emplace_back(module.view(r));
      if (i.opcode == "ld" && Has(i,"param") && operands.size() >= 2) {
        auto split = SplitAddress(Unbracket(operands[1]));
        if (!split) return {};
        auto base = value(split->first,*site,depth+1);
        const auto *pointer = std::get_if<ParameterPointer>(&base);
        if (!pointer) return {};
        const auto &region = config.parameter_regions.at(pointer->region);
        const auto width = i.memory_width;
        const auto &destinations = i.destinations;
        auto dest = std::find(destinations.begin(),destinations.end(),token);
        if (!width || destinations.empty() || *width % destinations.size() || dest == destinations.end()) return {};
        auto per_register = *width / destinations.size();
        Integer offset = pointer->offset + split->second + (dest - destinations.begin()) * per_register;
        if (offset < 0 || offset + per_register > region.size_bytes || offset > INT64_MAX) return {};
        if (auto known = region.ReadInteger(offset.convert_to<std::int64_t>(),per_register,false)) return *known;
        if (per_register == 8) return Field{pointer->region,offset.convert_to<std::uint64_t>()};
        return Numeric{};
      }
      if ((i.opcode == "mov" || i.opcode == "cvta") && operands.size() == 2) return value(operands[1],*site,depth+1);
      if (i.opcode == "cvt" && operands.size() == 2) {
        auto v = value(operands[1],*site,depth+1);
        const std::set<std::string> small{"s8","u8","s16","u16","s32","u32"};
        if (!i.modifiers.empty() && small.contains(i.modifiers.back()))
          return std::holds_alternative<Integer>(v) ? v : Value(Numeric{});
        return Number(v) ? v : Value{};
      }
      if ((i.opcode == "add" || i.opcode == "sub") && operands.size() == 3) {
        auto left = value(operands[1],*site,depth+1), right = value(operands[2],*site,depth+1);
        auto *l = std::get_if<Integer>(&left), *r = std::get_if<Integer>(&right);
        const bool add = i.opcode == "add";
        if (l && r) return Integer(add ? Integer(*l + *r) : Integer(*l - *r));
        if (auto *p = std::get_if<ParameterPointer>(&left); p && r)
          return ParameterPointer{p->region, p->offset + (add ? *r : Integer(-*r))};
        if (auto *p = std::get_if<ParameterPointer>(&right); add && p && l)
          return ParameterPointer{p->region,p->offset + *l};
        if (std::holds_alternative<Field>(left) && Number(right)) return left;
        if (add && std::holds_alternative<Field>(right) && Number(left)) return right;
        if (Number(left) && Number(right)) return Numeric{};
      }
      const std::set<std::string> arithmetic{"mul","mad","shl","shr","and","or","xor","bfe","div","rem"};
      if (arithmetic.contains(i.opcode)) {
        for (std::size_t n = 1; n < operands.size(); ++n)
          if (!Number(value(operands[n],*site,depth+1))) return {};
        return Numeric{};
      }
      return {};
    };
    auto result = calculate(); cache.emplace(key,result); return result;
  };
  std::set<std::pair<std::string,std::uint64_t>> bound, found;
  for (const auto &p : config.pointer_parameters) {
    auto name = p.parameter.value_or("");
    if (p.parameter_index) name = *function.parameters[*p.parameter_index].name;
    bound.emplace(name,p.byte_offset);
  }
  for (const auto &i : function.instructions) {
    std::optional<std::size_t> address;
    if (Has(i,"global")) {
      if (i.opcode == "ld" && i.operands.size() >= 2) address = 1;
      else if (i.opcode == "st" && !i.operands.empty()) address = 0;
      else if (i.opcode == "tensormap" && Has(i,"cp_fenceproxy") && !i.operands.empty()) address = 0;
      else if (i.opcode == "cp" && Has(i,"async") && !Has(i,"tensor") && !Has(i,"bulk") && i.operands.size() >= 2) address = 1;
    }
    if (!address) continue;
    auto split = SplitAddress(Unbracket(module.view(i.operands[*address])));
    if (!split) continue;
    const auto candidate = value(split->first,i.id,0);
    const auto *f = std::get_if<Field>(&candidate);
    if (!f || bound.contains({f->parameter,f->offset})) continue;
    bool scalar = false;
    for (const auto &entry : config.raw.value("parameter_field_types",nlohmann::json::array()))
      if (entry["parameter"] == f->parameter && Integer(f->offset) < JsonInteger(entry["offset"]) + JsonInteger(entry["bytes"]) && JsonInteger(entry["offset"]) < Integer(f->offset) + 8) scalar = true;
    if (!scalar) found.emplace(f->parameter,f->offset);
  }
  std::vector<InferredPointer> result;
  for (const auto &[name,offset] : found) result.push_back({name,offset,8});
  return result;
}
} // namespace sentryward::frontend::ptx
