#include "sentryward/frontend/ptx/config.h"
#include "sentryward/frontend/ptx/module.h"
#include "sentryward/algorithm/source_ir/ptx_assumptions.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <openssl/evp.h>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace sentryward::frontend::ptx {
namespace {
using Json = nlohmann::json;
[[noreturn]] void Bad(const std::string &message) {
  throw std::invalid_argument("PTX config: " + message);
}
std::uint64_t Nonnegative(const Json &v, std::string_view where) {
  if (!v.is_number_integer() ||
      (!v.is_number_unsigned() && v.get<std::int64_t>() < 0))
    Bad(std::string(where) + " must be a non-negative integer");
  return v.get<std::uint64_t>();
}
std::uint64_t Positive(const Json &v, std::string_view where) {
  auto n = Nonnegative(v, where);
  if (!n)
    Bad(std::string(where) + " must be positive");
  return n;
}
std::uint32_t Bounded32(const Json &v, std::string_view where,
                        bool positive = false) {
  const auto n = positive ? Positive(v, where) : Nonnegative(v, where);
  if (n > UINT32_MAX)
    Bad(std::string(where) + " exceeds the native uint32 index domain");
  return static_cast<std::uint32_t>(n);
}
Coordinates Dimensions(const Json &v, std::string_view where) {
  if (!v.is_array() || v.size() != 3)
    Bad(std::string(where) + " must contain three positive integers");
  Coordinates out{};
  for (std::size_t i = 0; i < 3; ++i) {
    auto n = Positive(v[i], where);
    if (n > UINT32_MAX)
      Bad("dimension exceeds uint32");
    out[i] = n;
  }
  return out;
}
void RejectOracle(const Json &v, std::string path = "$") {
  static const std::set<std::string> forbidden{"generation",
                                               "generation_before",
                                               "generation_after",
                                               "intended_generation",
                                               "intended_generation_after",
                                               "phase",
                                               "phase_id",
                                               "phase_members",
                                               "wait_match",
                                               "contributors",
                                               "wait_enable"};
  if (v.is_object())
    for (auto it = v.begin(); it != v.end(); ++it) {
      auto lower = it.key();
      std::transform(lower.begin(), lower.end(), lower.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (forbidden.contains(lower))
        Bad(path + "." + it.key() + ": solver-oracle field is forbidden");
      RejectOracle(it.value(), path + "." + it.key());
    }
  else if (v.is_array())
    for (std::size_t i = 0; i < v.size(); ++i)
      RejectOracle(v[i], path + "[" + std::to_string(i) + "]");
}
std::map<std::string, Integer> Integers(const Json &v, std::string_view where) {
  if (!v.is_object())
    Bad(std::string(where) + " must map names to integers");
  std::map<std::string, Integer> out;
  for (auto it = v.begin(); it != v.end(); ++it)
    out.emplace(it.key(), JsonInteger(it.value()));
  return out;
}
std::string Sha256(const std::vector<std::uint8_t> &bytes) {
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned len = 0;
  if (!EVP_Digest(bytes.data(), bytes.size(), digest, &len, EVP_sha256(),
                  nullptr))
    Bad("SHA256 failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(64);
  for (unsigned i = 0; i < len; ++i) {
    result.push_back(hex[digest[i] >> 4]);
    result.push_back(hex[digest[i] & 15]);
  }
  return result;
}
const Json &EmptyObject() {
  static const Json empty = Json::object();
  return empty;
}
} // namespace
Integer JsonInteger(const Json &value) {
  if (value.is_boolean())
    return value.get<bool>() ? 1 : 0;
  if (value.is_number_unsigned())
    return Integer(value.get<std::uint64_t>());
  if (value.is_number_integer())
    return Integer(value.get<std::int64_t>());
  Bad("expected integer, got " + value.dump());
}
std::optional<Integer> ParameterRegion::ReadInteger(std::int64_t offset,
                                                    std::uint32_t width,
                                                    bool is_signed) const {
  if (offset < 0 || !width || std::uint64_t(offset) > size_bytes ||
      width > size_bytes - std::uint64_t(offset))
    Bad("parameter region " + name + " read is outside size");
  Integer bits = 0;
  if (source.value("kind", "") == "typed_fields") {
    for (std::uint32_t byte = 0; byte < width; ++byte) {
      const auto position = static_cast<std::uint64_t>(offset) + byte;
      auto field = offset_values.upper_bound(position);
      if (field == offset_values.begin()) return {};
      --field;
      if (position - field->first >= field->second.first) return {};
      bits |= ((field->second.second >> (8 * (position - field->first))) & 255) << (8 * byte);
    }
    if (is_signed && (bits & (Integer(1) << (8 * width - 1))) != 0)
      bits -= Integer(1) << (8 * width);
    return bits;
  }
  auto it = offset_values.find(offset);
  if (it != offset_values.end()) {
    if (it->second.first != width)
      Bad("parameter region override byte width mismatch");
    bits = it->second.second & ((Integer(1) << (8 * width)) - 1);
  } else if (data)
    for (std::uint32_t i = 0; i < width; ++i)
      bits |= Integer((*data)[offset + i]) << (8 * i);
  else
    return {};
  if (is_signed && (bits & (Integer(1) << (8 * width - 1))) != 0)
    bits -= Integer(1) << (8 * width);
  return bits;
}
std::uint64_t FrontendConfig::Threads() const {
  Integer threads = Integer(block_dim[0]) * block_dim[1] * block_dim[2];
  if (threads > UINT64_MAX)
    Bad("block thread count exceeds uint64");
  return threads.convert_to<std::uint64_t>();
}
const LoopBound *FrontendConfig::FindLoopBound(std::string_view header,
                                               std::uint32_t warp) const {
  const LoopBound *found = nullptr;
  for (const auto &bound : loop_bounds)
    if (bound.header == header &&
        (!bound.warp_ids ||
         std::find(bound.warp_ids->begin(), bound.warp_ids->end(), warp) !=
             bound.warp_ids->end())) {
      if (found)
        Bad("multiple loop bounds match");
      found = &bound;
    }
  return found;
}
std::optional<bool> FrontendConfig::BranchOutcome(std::uint32_t line,
                                                  std::uint32_t warp,
                                                  std::uint32_t lane,
                                                  std::uint64_t visit) const {
  const BranchOverride *found = nullptr;
  for (const auto &item : branch_overrides)
    if (item.line == line && (!item.warp_id || *item.warp_id == warp) &&
        (!item.lane_id || *item.lane_id == lane)) {
      if (found)
        Bad("multiple branch overrides match");
      found = &item;
    }
  if (!found)
    return {};
  if (found->outcomes.size() == 1)
    return found->outcomes[0];
  if (visit >= found->outcomes.size())
    Bad("branch override exhausted");
  return found->outcomes[visit];
}
const Json &FrontendConfig::InstructionContract(
    std::uint32_t line, std::uint32_t instruction,
    std::optional<std::uint32_t> warp,
    std::optional<std::uint64_t> occurrence) const {
  if (instruction_contracts.empty())
    return EmptyObject();
  std::vector<std::string> suffixes;
  if (warp && occurrence)
    suffixes.push_back("@warp:" + std::to_string(*warp) +
                       "@occurrence:" + std::to_string(*occurrence));
  if (occurrence)
    suffixes.push_back("@occurrence:" + std::to_string(*occurrence));
  if (warp)
    suffixes.push_back("@warp:" + std::to_string(*warp));
  suffixes.emplace_back();
  for (const auto &suffix : suffixes) {
    auto a = instruction_contracts.find("index:" + std::to_string(instruction) +
                                        suffix);
    auto b =
        instruction_contracts.find("line:" + std::to_string(line) + suffix);
    if (a != instruction_contracts.end() && b != instruction_contracts.end() &&
        *a != *b)
      Bad("conflicting index/line instruction contracts");
    if (a != instruction_contracts.end())
      return *a;
    if (b != instruction_contracts.end())
      return *b;
  }
  return EmptyObject();
}
FrontendConfig ParseConfig(Json value, const std::filesystem::path &base_path) {
  if (!value.is_object() ||
      value.value("schema", "") != "spverify.ptx-front-end-config/v1")
    Bad("schema must be spverify.ptx-front-end-config/v1");
  RejectOracle(value);
  const Json original = value;
  // Optional fields use the reference API's null-as-absent convention. Keep
  // the original sidecar for provenance while normalizing only typed inputs.
  const auto erase_null = [](Json &object,
                             std::initializer_list<std::string_view> fields) {
    if (!object.is_object())
      return;
    for (const auto field : fields) {
      const auto it = object.find(std::string(field));
      if (it != object.end() && it->is_null())
        object.erase(it);
    }
  };
  if (value.contains("cta"))
    erase_null(value["cta"], {"cluster_dim"});
  if (value.contains("resources") && value["resources"].is_array())
    for (auto &resource : value["resources"]) {
      erase_null(resource, {"address", "match", "allocation"});
      if (resource.is_object() && resource.contains("init"))
        erase_null(resource["init"], {"expected_arrivals"});
    }
  if (value.contains("parameter_regions") &&
      value["parameter_regions"].is_object())
    for (auto &region : value["parameter_regions"])
      erase_null(region, {"data_file", "data_hex", "sha256"});
  if (value.contains("instruction_contracts") &&
      value["instruction_contracts"].is_object())
    for (auto &contract : value["instruction_contracts"]) {
      erase_null(contract, {"footprint_contract", "resource", "tmem_resource",
                            "pointer_resource", "destination_resource",
                            "source_resource", "barrier_resource",
                            "active_lanes", "wait_usage", "descriptor"});
      for (const auto kind : {"reads", "writes"})
        if (contract.is_object() && contract.contains(kind) &&
            contract[kind].is_array())
          for (auto &access : contract[kind])
            erase_null(access, {"role"});
    }
  if (value.contains("tma_descriptors") && value["tma_descriptors"].is_object())
    for (auto &descriptor : value["tma_descriptors"])
      erase_null(descriptor,
                 {"transfer_bytes", "box_shape", "element_bytes",
                  "coordinate_strides_bytes", "global_shape", "encoded_hex"});
  auto assumptions = value.value("semantic_assumptions", Json::object());
  if (!assumptions.is_object())
    Bad("semantic_assumptions must be an object");
  auto flags = algorithm::DefaultPtxAssumptions();
  for (auto it = assumptions.begin(); it != assumptions.end(); ++it) {
    if (!flags.contains(it.key()) || !it->is_boolean())
      Bad("invalid semantic assumption");
    flags.at(it.key()) = it->get<bool>();
  }
  for (auto field : {"warps", "loop_bounds", "branch_overrides", "resources",
                     "pointer_parameters"})
    if (value.contains(field) && !value[field].is_array())
      Bad(std::string(field) + " must be a list");
  FrontendConfig c;
  c.raw = original;
  c.raw["semantic_assumptions"] = flags;
  c.entry = value.at("entry").get<std::string>();
  if (c.entry.empty())
    Bad("empty entry");
  auto cta = value.at("cta");
  c.block_dim = Dimensions(cta.at("block_dim"), "cta.block_dim");
  if (cta.contains("cluster_dim"))
    c.cluster_dim = Dimensions(cta["cluster_dim"], "cta.cluster_dim");
  auto warp_size = Positive(cta.value("warp_size", Json(32)), "cta.warp_size");
  if (warp_size != 32)
    Bad("CUDA warp size must be32");
  c.warp_size = warp_size;
  if (c.warp_size != 32 || c.Threads() % 32)
    Bad("CUDA block must have complete 32-thread warps");
  if (c.Threads() > UINT32_MAX)
    Bad("CTA thread count exceeds the native uint32 thread domain");
  auto warp_count = c.Threads() / 32;
  if (warp_count > UINT32_MAX)
    Bad("too many warps");
  Json warps = value.value("warps", Json::array());
  if (!value.contains("warps"))
    for (std::uint32_t w = 0; w < warp_count; ++w)
      warps.push_back({{"warp_id", w}});
  if (!warps.is_array() || warps.empty())
    Bad("warps must be nonempty list");
  std::set<std::uint32_t> seen;
  for (const auto &w : warps) {
    WarpContract item;
    auto warp_id = Nonnegative(w.at("warp_id"), "warp_id");
    if (warp_id > UINT32_MAX)
      Bad("warp_id overflow");
    item.warp_id = warp_id;
    if (item.warp_id >= warp_count || !seen.insert(item.warp_id).second)
      Bad("invalid/duplicate warp_id");
    item.role = w.value("role", "warp " + std::to_string(item.warp_id));
    item.purpose = w.value("purpose", "");
    if (w.contains("entry_label") && !w["entry_label"].is_null())
      item.entry_label = w["entry_label"].get<std::string>();
    item.initial_registers = Integers(
        w.value("initial_registers", Json::object()), "initial_registers");
    c.warps.push_back(std::move(item));
  }
  if (seen.size() != warp_count)
    Bad("one contract required for every physical warp");
  std::sort(c.warps.begin(), c.warps.end(),
            [](const auto &a, const auto &b) { return a.warp_id < b.warp_id; });
  for (const auto &b : value.value("loop_bounds", Json::array())) {
    LoopBound item;
    item.header = b.at("header").get<std::string>();
    if (item.header.empty())
      Bad("empty loop header");
    item.trip_count = Positive(b.at("trip_count"), "trip_count");
    if (b.contains("warp_ids") && !b["warp_ids"].is_null()) {
      if (!b["warp_ids"].is_array())
        Bad("loop warp_ids must be a list");
      item.warp_ids.emplace();
      for (const auto &w : b["warp_ids"]) {
        auto id = Bounded32(w, "loop warp_id");
        if (!seen.contains(id))
          Bad("invalid loop warp_id");
        item.warp_ids->push_back(id);
      }
      std::sort(item.warp_ids->begin(), item.warp_ids->end());
      item.warp_ids->erase(
          std::unique(item.warp_ids->begin(), item.warp_ids->end()),
          item.warp_ids->end());
    }
    if (b.contains("backedge_line") && !b["backedge_line"].is_null())
      item.backedge_line = Bounded32(b["backedge_line"], "backedge_line", true);
    c.loop_bounds.push_back(std::move(item));
  }
  for (const auto &b : value.value("branch_overrides", Json::array())) {
    BranchOverride item;
    item.line = Bounded32(b.at("line"), "branch line", true);
    auto outcomes = b.at("outcomes");
    if (outcomes.is_boolean())
      item.outcomes.push_back(outcomes.get<bool>());
    else if (outcomes.is_array() && !outcomes.empty())
      for (const auto &v : outcomes) {
        if (!v.is_boolean())
          Bad("branch outcomes must be bool");
        item.outcomes.push_back(v.get<bool>());
      }
    else
      Bad("invalid branch outcomes");
    if (b.contains("warp_id") && !b["warp_id"].is_null()) {
      auto n = Bounded32(b["warp_id"], "branch warp");
      if (!seen.contains(n))
        Bad("invalid branch warp");
      item.warp_id = n;
    }
    if (b.contains("lane_id") && !b["lane_id"].is_null()) {
      auto n = Nonnegative(b["lane_id"], "branch lane");
      if (n >= 32)
        Bad("invalid branch lane");
      item.lane_id = n;
    }
    c.branch_overrides.push_back(std::move(item));
  }
  c.parameters =
      Integers(value.value("parameters", Json::object()), "parameters");
  c.special_registers = Integers(
      value.value("special_registers", Json::object()), "special_registers");
  auto regions = value.value("parameter_regions", Json::object());
  if (!regions.is_object())
    Bad("parameter_regions must be object");
  for (auto it = regions.begin(); it != regions.end(); ++it) {
    if (it.key().empty())
      Bad("parameter region name must be non-empty");
    const auto &r = it.value();
    ParameterRegion region;
    region.name = it.key();
    region.size_bytes = Positive(r.at("size_bytes"), "parameter size");
    if (r.value("byte_order", "little") != "little")
      Bad("only little-endian parameter bytes supported");
    if (r.contains("data_file") && r.contains("data_hex"))
      Bad("both data_file and data_hex");
    region.source = {{"kind", "offset_values"}};
    if (r.contains("data_file")) {
      auto path = std::filesystem::path(r["data_file"].get<std::string>());
      if (path.is_relative())
        path = base_path / path;
      std::ifstream in(path, std::ios::binary);
      if (!in)
        Bad("cannot read parameter data " + path.string());
      region.data =
          std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
      if (!r.contains("sha256"))
        Bad("data_file requires sha256");
      region.source = {
          {"kind", "file"},
          {"configured_path", r["data_file"]},
          {"resolved_path", std::filesystem::absolute(path).string()}};
    }
    if (r.contains("data_hex")) {
      region.data.emplace();
      auto hex = r["data_hex"].get<std::string>();
      std::string compact;
      for (unsigned char ch : hex)
        if (!std::isspace(ch))
          compact.push_back(ch);
      if (compact.size() % 2)
        Bad("odd-length data_hex");
      for (std::size_t i = 0; i < compact.size(); i += 2) {
        if (!std::isxdigit(compact[i]) || !std::isxdigit(compact[i + 1]))
          Bad("invalid data_hex");
        region.data->push_back(std::stoul(compact.substr(i, 2), nullptr, 16));
      }
      region.source = {{"kind", "inline_hex"}};
    }
    if (region.data && region.data->size() != region.size_bytes)
      Bad("parameter data size mismatch");
    if (r.contains("sha256")) {
      if (!region.data)
        Bad("sha256 requires data");
      auto expected = r["sha256"].get<std::string>();
      std::transform(expected.begin(), expected.end(), expected.begin(),
                     [](unsigned char ch) { return std::tolower(ch); });
      auto actual = Sha256(*region.data);
      if (expected != actual)
        Bad("parameter sha256 mismatch");
      region.source["sha256"] = actual;
    }
    auto offsets = r.value("offset_values", Json::object());
    if (!offsets.is_object())
      Bad("offset_values must be object");
    for (auto f = offsets.begin(); f != offsets.end(); ++f) {
      std::size_t consumed = 0;
      auto offset = std::stoull(f.key(), &consumed, 0);
      if (consumed != f.key().size() || f.key().starts_with('-'))
        Bad("invalid parameter offset");
      auto width = Positive(f.value().at("bytes"), "parameter override width");
      if (width > UINT32_MAX || offset > region.size_bytes ||
          width > region.size_bytes - offset)
        Bad("parameter override outside region");
      region.offset_values[offset] = {width,
                                      JsonInteger(f.value().at("value"))};
    }
    c.parameter_regions.emplace(it.key(), std::move(region));
  }
  c.resources = value.value("resources", Json::array());
  if (!c.resources.is_array())
    Bad("resources must be list");
  std::map<std::string, const Json *> resources;
  for (const auto &r : c.resources) {
    auto id = r.at("id").get<std::string>();
    auto kind = r.at("kind").get<std::string>();
    if (id.empty() || !resources.emplace(id, &r).second)
      Bad("invalid/duplicate resource id");
    if (kind == "memory") {
      static const std::set<std::string> spaces{
          "global",      "generic",         "shared",
          "shared::cta", "shared::cluster", "tmem"};
      if (!spaces.contains(r.value("space", "")))
        Bad("unmodeled memory space");
    } else if (kind == "mbarrier") {
      if (r.value("layout", "layout::v0") != "layout::v0")
        Bad("unsupported mbarrier layout");
      auto init = r.value("init", Json::object());
      if (!init.is_object())
        Bad("barrier init must be object");
      if (init.contains("expected_arrivals") &&
          Positive(init["expected_arrivals"], "expected_arrivals") >=
              (1u << 20))
        Bad("invalid expected arrivals");
    } else
      Bad("invalid resource kind");
    for (auto field : {"address", "match"})
      if (r.contains(field) && !r[field].is_object())
        Bad("resource address/match must be object");
  }
  std::set<std::string> pointer_resources, allocation_objects;
  for (const auto &p : value.value("pointer_parameters", Json::array())) {
    PointerParameter item;
    auto src = p.at("source");
    if (!src.is_object())
      Bad("pointer source must be object");
    for (auto it = src.begin(); it != src.end(); ++it)
      if (it.key() != "parameter" && it.key() != "parameter_index" &&
          it.key() != "byte_offset")
        Bad("unknown pointer source field");
    bool name = src.contains("parameter") && !src["parameter"].is_null(),
         index = src.contains("parameter_index") &&
                 !src["parameter_index"].is_null();
    if (name == index)
      Bad("pointer source needs exactly one parameter selector");
    if (name) {
      item.parameter = src["parameter"].get<std::string>();
      if (item.parameter->empty())
        Bad("empty pointer parameter");
    } else {
      auto n = Nonnegative(src["parameter_index"], "parameter_index");
      if (n > UINT32_MAX)
        Bad("parameter_index overflow");
      item.parameter_index = n;
    }
    item.byte_offset =
        Nonnegative(src.value("byte_offset", Json(0)), "byte_offset");
    auto width = Positive(p.value("width_bytes", Json(8)), "width_bytes");
    if (width > UINT32_MAX || item.byte_offset > UINT64_MAX - width)
      Bad("pointer field overflow");
    item.width_bytes = width;
    item.resource = p.at("resource").get<std::string>();
    auto ri = resources.find(item.resource);
    if (ri == resources.end() || ri->second->value("kind", "") != "memory" ||
        (ri->second->value("space", "") != "global" &&
         ri->second->value("space", "") != "generic"))
      Bad("pointer resource must be global/generic memory");
    if (!pointer_resources.insert(item.resource).second)
      Bad("pointer resources must be pairwise nonaliasing");
    auto object = item.resource;
    if (ri->second->contains("allocation")) {
      const auto &a = (*ri->second)["allocation"];
      if (a.value("resolution", "") != "concrete")
        Bad("pointer allocation must be concrete");
      object = a.at("object").get<std::string>();
      if (object.empty())
        Bad("empty allocation object");
    }
    if (!allocation_objects.insert(object).second &&
        value.value("fixed_input", Json::object()).value("public_input_schema", "") != "spverify.kernel-input/v1")
      Bad("pointer allocation objects alias");
    auto extent = p.value("extent_bytes", Json("unbounded"));
    if (extent != "unbounded")
      item.extent_bytes = Positive(extent, "extent_bytes");
    for (const auto &other : c.pointer_parameters)
      if (item.parameter == other.parameter &&
          item.parameter_index == other.parameter_index &&
          item.byte_offset < other.byte_offset + other.width_bytes &&
          other.byte_offset < item.byte_offset + item.width_bytes)
        Bad("overlapping pointer fields");
    c.pointer_parameters.push_back(std::move(item));
  }
  c.instruction_contracts =
      value.value("instruction_contracts", Json::object());
  c.tma_descriptors = value.value("tma_descriptors", Json::object());
  if (!c.instruction_contracts.is_object() || !c.tma_descriptors.is_object())
    Bad("contracts and descriptors must be objects");
  const std::regex selector(
      R"((?:line:[1-9]\d*|index:(?:0|[1-9]\d*))(?:@warp:(?:0|[1-9]\d*))?(?:@occurrence:(?:0|[1-9]\d*))?)");
  for (auto it = c.instruction_contracts.begin();
       it != c.instruction_contracts.end(); ++it) {
    const auto &r = it.value();
    if (!std::regex_match(it.key(), selector) || !r.is_object())
      Bad("invalid instruction contract selector");
    if (r.contains("footprint_contract") && r["footprint_contract"] != "exact")
      Bad("footprint contract must be exact");
    for (auto key :
         {"resource", "tmem_resource", "pointer_resource",
          "destination_resource", "source_resource", "barrier_resource"})
      if (r.contains(key) && !resources.contains(r[key].get<std::string>()))
        Bad("unknown contract resource");
    for (auto kind : {"reads", "writes"})
      if (r.contains(kind)) {
        if (!r[kind].is_array())
          Bad("contract accesses must be array");
        for (const auto &a : r[kind]) {
          if (!a.is_object() ||
              !resources.contains(a.at("resource").get<std::string>()))
            Bad("unknown access resource");
          if (a.contains("guard") && !a["guard"].is_boolean())
            Bad("access guard must be bool");
          if (a.contains("role") &&
              (!a["role"].is_string() || a["role"].get<std::string>().empty()))
            Bad("access role invalid");
        }
      }
    if (r.contains("active_lanes")) {
      if (!r["active_lanes"].is_array())
        Bad("active_lanes must be array");
      for (const auto &lane : r["active_lanes"])
        if (Nonnegative(lane, "active lane") >= 32)
          Bad("active lane outside warp");
    }
    if (r.contains("wait_usage") && r["wait_usage"] != "logical_wait" &&
        r["wait_usage"] != "discarded_probe")
      Bad("invalid wait_usage");
    if (r.contains("descriptor") &&
        !c.tma_descriptors.contains(r["descriptor"].get<std::string>()))
      Bad("unknown TMA descriptor");
  }
  for (auto it = c.tma_descriptors.begin(); it != c.tma_descriptors.end();
       ++it) {
    const auto &d = it.value();
    if (!d.is_object() ||
        !resources.contains(d.at("resource").get<std::string>()))
      Bad("invalid descriptor resource");
    for (auto key : {"transfer_bytes", "element_bytes"})
      if (d.contains(key))
        Positive(d[key], key);
    for (auto key : {"box_shape", "coordinate_strides_bytes", "global_shape"})
      if (d.contains(key)) {
        if (!d[key].is_array() || d[key].empty())
          Bad("descriptor dimensions must be nonempty array");
        for (std::size_t i = 0; i < d[key].size(); ++i)
          if (std::string_view(key) == "coordinate_strides_bytes" && i != 0)
            Nonnegative(d[key][i], key);
          else
            Positive(d[key][i], key);
      }
    if (!d.contains("transfer_bytes") &&
        (!d.contains("box_shape") || !d.contains("element_bytes")))
      Bad("descriptor needs transfer size");
    Nonnegative(d.value("base_byte_offset", Json(0)), "base_byte_offset");
    if (d.contains("coordinate_strides_bytes")) {
      auto rank = d["coordinate_strides_bytes"].size();
      if (!d.contains("global_shape") || d["global_shape"].size() != rank ||
          !d.contains("box_shape") || d["box_shape"].size() != rank ||
          !d.contains("element_bytes"))
        Bad("descriptor rank mismatch");
      Integer bytes = JsonInteger(d["element_bytes"]);
      for (const auto &n : d["box_shape"])
        bytes *= JsonInteger(n);
      if (d.contains("transfer_bytes") &&
          bytes != JsonInteger(d["transfer_bytes"]))
        Bad("descriptor transfer size mismatch");
    }
  }
  c.fixed_input = value.value("fixed_input", Json::object());
  if (!c.fixed_input.is_object())
    Bad("fixed_input must be object");
  if (value.contains("elect_lane") && !value["elect_lane"].is_null()) {
    auto n = Nonnegative(value["elect_lane"], "elect_lane");
    if (n >= 32)
      Bad("elect_lane outside warp");
    c.elect_lane = n;
  }
  c.max_steps_per_lane = Positive(
      value.value("max_steps_per_lane", Json(100000)), "max_steps_per_lane");
  c.compress_compute = value.value("compress_compute", true);
  c.compress_thread_records = value.value("compress_thread_records", true);
  return c;
}
FrontendConfig BindLaunchObjects(const Module &module, const Function &function,
                                 const FrontendConfig &config) {
  FrontendConfig out = config;
  struct SharedObject { Integer alignment, bytes; bool dynamic; };
  std::map<std::string, SharedObject> shared;
  std::vector<std::string> shared_order;
  const std::regex shared_pattern(R"((?:^|\s)\.shared(?:\s|$))"),
      symbol_pattern(R"(\.([busf]\d+|bf16)\s+([A-Za-z_$][A-Za-z0-9_$]*)(\s*(?:\[[^\]]*\])*))"),
      alignment_pattern(R"(\.align\s+(\d+))"), vector_pattern(R"(\.v([248])\b)"),
      dimension_pattern(R"(\[\s*(\d*)\s*\])");
  auto scan = [&](const std::vector<Directive> &directives) {
    for (const auto &d : directives) {
      std::string raw(module.view(d.source));
      std::smatch match;
      if (!std::regex_search(raw, shared_pattern) ||
          !std::regex_search(raw, match, symbol_pattern)) continue;
      std::string type = match[1], name = match[2], dimensions = match[3];
      Integer width = type == "bf16" ? 2 : std::stoul(type.substr(1)) / 8;
      if (std::regex_search(raw, match, vector_pattern)) width *= std::stoul(match[1]);
      Integer alignment = width;
      if (std::regex_search(raw, match, alignment_pattern)) alignment = Integer(match[1].str());
      if (alignment <= 0 || (alignment & (alignment - 1)) != 0) Bad("invalid shared object alignment");
      Integer bytes = width;
      bool dynamic = false;
      std::size_t count = 0;
      for (auto d = std::sregex_iterator(dimensions.begin(), dimensions.end(), dimension_pattern);
           d != std::sregex_iterator(); ++d) {
        ++count;
        if ((*d)[1].length() == 0) dynamic = true;
        else bytes *= Integer((*d)[1].str());
      }
      if (count != std::size_t(std::count(dimensions.begin(), dimensions.end(), '[')))
        Bad("shared object has unresolved extent");
      if (dynamic) {
        if (raw.find(".extern") == std::string::npos) Bad("unsized static shared object");
        bytes = JsonInteger(config.fixed_input.value("dynamic_shared_bytes", Json(0)));
      }
      if (!shared.contains(name)) shared_order.push_back(name);
      shared[name] = {alignment, bytes, dynamic};
    }
  };
  scan(module.directives);
  scan(function.directives);
  std::set<std::string> referenced;
  const std::regex symbol_reference(R"([A-Za-z_$][A-Za-z0-9_$]*)");
  for (const auto &instruction : function.instructions)
    for (auto operand : instruction.operands) {
      const std::string text(module.view(operand));
      for (auto it = std::sregex_iterator(text.begin(), text.end(), symbol_reference);
           it != std::sregex_iterator(); ++it) referenced.insert(it->str());
    }
  if (std::any_of(shared_order.begin(), shared_order.end(), [&](const auto &s) { return referenced.contains(s); }))
    std::erase_if(shared_order, [&](const auto &s) { return !referenced.contains(s); });
  std::vector<std::pair<Integer,Integer>> reserved;
  Integer cursor = 0;
  for (const auto &name : shared_order) {
    const auto &object = shared.at(name);
    if (object.dynamic || !out.parameters.contains(name)) continue;
    const auto &base = out.parameters.at(name);
    if (base < 0 || base % object.alignment != 0 || std::any_of(reserved.begin(), reserved.end(),
        [&](const auto &r) { return base < r.second && r.first < base + object.bytes; }))
      Bad("shared object has inconsistent placement");
    reserved.emplace_back(base, base + object.bytes);
  }
  for (const auto &name : shared_order) {
    const auto &object = shared.at(name);
    if (object.dynamic) continue;
    if (!out.parameters.contains(name)) {
      for (;;) {
        cursor = (cursor + object.alignment - 1) / object.alignment * object.alignment;
        auto hit = std::find_if(reserved.begin(), reserved.end(), [&](const auto &r) {
          return cursor < r.second && r.first < cursor + object.bytes;
        });
        if (hit == reserved.end()) break;
        cursor = hit->second;
      }
      out.parameters[name] = cursor;
      reserved.emplace_back(cursor, cursor + object.bytes);
    }
    cursor = std::max(cursor, Integer(out.parameters.at(name) + object.bytes));
  }
  Integer dynamic_alignment = 1;
  std::set<Integer> dynamic_bases;
  for (const auto &name : shared_order) if (shared.at(name).dynamic) {
    dynamic_alignment = std::max(dynamic_alignment, shared.at(name).alignment);
    if (out.parameters.contains(name)) dynamic_bases.insert(out.parameters.at(name));
  }
  if (dynamic_bases.size() > 1) Bad("extern shared declarations must alias one dynamic region");
  const Integer dynamic_base = dynamic_bases.empty()
      ? Integer((cursor + dynamic_alignment - 1) / dynamic_alignment * dynamic_alignment)
      : *dynamic_bases.begin();
  if (dynamic_base < cursor || dynamic_base % dynamic_alignment != 0)
    Bad("dynamic shared placement overlaps static objects or violates alignment");
  Json layout = Json::array();
  for (const auto &name : shared_order) {
    const auto &object = shared.at(name);
    if (object.dynamic) out.parameters[name] = dynamic_base;
    for (const auto &n : {out.parameters[name], object.alignment, object.bytes})
      if (n < 0 || n > UINT64_MAX) Bad("shared layout exceeds 64-bit address range");
    layout.push_back({{"symbol", name}, {"base", out.parameters[name].convert_to<std::uint64_t>()},
        {"alignment", object.alignment.convert_to<std::uint64_t>()},
        {"bytes", object.bytes.convert_to<std::uint64_t>()}, {"dynamic", object.dynamic}});
  }
  out.raw["shared_object_layout"] = std::move(layout);
  for (auto it = out.tma_descriptors.begin(); it != out.tma_descriptors.end();
       ++it) {
    auto &descriptor = it.value();
    std::vector<std::uint64_t> indices, offsets;
    for (auto name : {"abi_parameter_index", "parameter_index"})
      if (descriptor.contains(name))
        indices.push_back(
            Nonnegative(descriptor[name], "tensor map ABI index"));
    for (auto name : {"abi_parameter_byte_offset", "parameter_byte_offset"})
      if (descriptor.contains(name))
        offsets.push_back(
            Nonnegative(descriptor[name], "tensor map ABI offset"));
    if ((indices.size() > 1 && indices[0] != indices[1]) ||
        (offsets.size() > 1 && offsets[0] != offsets[1]))
      Bad("conflicting tensor map ABI locations");
    std::optional<std::pair<std::string, std::uint64_t>> location;
    if (!indices.empty() || !offsets.empty()) {
      if (indices.empty() || offsets.empty() ||
          indices[0] >= function.parameters.size() ||
          !function.parameters[indices[0]].name)
        Bad("invalid tensor map ABI location");
      const auto &parameter = function.parameters[indices[0]];
      location = {{*parameter.name, offsets[0]}};
      if (parameter.array) {
        std::smatch match;
        const std::regex size_pattern(R"(\[(\d+)\])");
        if (std::regex_match(*parameter.array, match, size_pattern)) {
          auto size = std::stoull(match[1]);
          if (offsets[0] > size || 128 > size - offsets[0])
            Bad("tensor map exceeds PTX parameter storage");
        }
      }
    }
    if (descriptor.contains("encoded_hex")) {
      if (!descriptor["encoded_hex"].is_string())
        Bad("tensor map encoded_hex must be string");
      std::string hex = descriptor["encoded_hex"].get<std::string>(), compact;
      for (unsigned char ch : hex)
        if (!std::isspace(ch))
          compact.push_back(ch);
      if (compact.size() != 256)
        Bad("tensor map requires 128 encoded bytes");
      std::vector<std::uint8_t> bytes;
      for (std::size_t i = 0; i < compact.size(); i += 2) {
        if (!std::isxdigit(compact[i]) || !std::isxdigit(compact[i + 1]))
          Bad("invalid tensor map hex");
        bytes.push_back(std::stoul(compact.substr(i, 2), nullptr, 16));
      }
      std::vector<std::pair<std::string, std::uint64_t>> matches;
      for (const auto &[name, region] : config.parameter_regions) {
        if (!region.data ||
            std::none_of(function.parameters.begin(), function.parameters.end(),
                         [&name = name](const auto &p) { return p.name == name; }))
          continue;
        auto at = region.data->begin();
        while (at != region.data->end()) {
          at = std::search(at, region.data->end(), bytes.begin(), bytes.end());
          if (at == region.data->end())
            break;
          auto offset = std::uint64_t(at - region.data->begin());
          bool overlap = false;
          for (const auto &[start, override] : region.offset_values)
            if (offset < start + override.first && start < offset + 128) {
              overlap = true;
              break;
            }
          if (!overlap)
            matches.emplace_back(name, offset);
          ++at;
        }
      }
      if (location) {
        if (std::find(matches.begin(), matches.end(), *location) ==
            matches.end())
          Bad("tensor map bytes disagree with ABI location");
      } else if (matches.size() == 1)
        location = matches.front();
      else if (matches.size() > 1)
        Bad("tensor map found at multiple parameter locations");
      else if (!descriptor.contains("address"))
        Bad("tensor map not found in captured parameters");
    }
    if (location)
      descriptor["parameter_location"] = {location->first, location->second};
    if (descriptor.contains("address"))
      Nonnegative(descriptor["address"], "tensor map address");
  }
  return out;
}
std::pair<Coordinates, Coordinates> LaunchDimensions(const FrontendConfig &c) {
  const std::set<std::string> lane{"%laneid", "%warpid", "%nwarpid",
                                   "%tid.x",  "%tid.y",  "%tid.z",
                                   "%ntid.x", "%ntid.y", "%ntid.z"};
  for (const auto &[name, v] : c.special_registers)
    if (lane.contains(name))
      Bad("hardware thread coordinates cannot be overridden");
  for (const auto &w : c.warps) {
    if (w.entry_label)
      Bad("full-thread specialization starts at entry");
    for (const auto &[name, v] : w.initial_registers)
      if (lane.contains(name) || name.starts_with("%ctaid.") ||
          name.starts_with("%cluster_") || name.starts_with("%nctaid.") ||
          name.starts_with("%clusterid.") || name.starts_with("%nclusterid."))
        Bad("hardware coordinates cannot be overridden");
  }
  Json raw = c.fixed_input.value("grid_dim", Json());
  if (raw.is_null()) {
    raw = Json::array();
    for (char axis : std::string("xyz")) {
      auto it = c.special_registers.find(std::string("%nctaid.") + axis);
      if (it == c.special_registers.end())
        Bad("full-thread specialization requires fixed_input.grid_dim");
      raw.push_back(it->second.convert_to<std::uint64_t>());
    }
  }
  auto grid = Dimensions(raw, "grid_dim");
  for (std::size_t i = 0; i < 3; ++i) {
    auto it = c.special_registers.find(std::string("%nctaid.") + "xyz"[i]);
    if (it != c.special_registers.end() && it->second != grid[i])
      Bad("grid disagrees with special registers");
  }
  auto cluster = c.cluster_dim.value_or(Coordinates{1, 1, 1});
  if (c.fixed_input.contains("cluster_shape") &&
      !c.fixed_input["cluster_shape"].is_null()) {
    auto fixed = Dimensions(c.fixed_input["cluster_shape"], "cluster_shape");
    if (c.cluster_dim && *c.cluster_dim != fixed)
      Bad("cluster dimension disagreement");
    cluster = fixed;
  }
  for (std::size_t i = 0; i < 3; ++i)
    if (grid[i] % cluster[i])
      Bad("grid must contain whole clusters");
  if (Integer(cluster[0]) * cluster[1] * cluster[2] > UINT32_MAX)
    Bad("cluster rank domain exceeds uint32");
  return {grid, cluster};
}
FrontendConfig ConfigForCta(const FrontendConfig &c, Coordinates xyz) {
  auto [grid, cluster] = LaunchDimensions(c);
  FrontendConfig out = c;
  Coordinates local{};
  for (std::size_t i = 0; i < 3; ++i) {
    if (xyz[i] >= grid[i])
      Bad("CTA coordinate outside grid");
    local[i] = xyz[i] % cluster[i];
    const auto axis = std::string(1, "xyz"[i]);
    out.special_registers["%ctaid." + axis] = xyz[i];
    out.special_registers["%nctaid." + axis] = grid[i];
    out.special_registers["%cluster_ctaid." + axis] = local[i];
    out.special_registers["%cluster_nctaid." + axis] = cluster[i];
    out.special_registers["%clusterid." + axis] = xyz[i] / cluster[i];
    out.special_registers["%nclusterid." + axis] = grid[i] / cluster[i];
  }
  auto rank =
      std::uint64_t(local[0]) +
      std::uint64_t(cluster[0]) *
          (std::uint64_t(local[1]) + std::uint64_t(cluster[1]) * local[2]);
  out.special_registers["%cluster_ctarank"] = rank;
  out.special_registers["%cluster_nctarank"] =
      std::uint64_t(cluster[0]) * cluster[1] * cluster[2];
  out.cluster_dim = cluster;
  out.fixed_input["cta_coordinate"] = xyz;
  out.fixed_input["cluster_ctarank"] = rank;
  out.raw["fixed_input"] = out.fixed_input;
  out.raw["special_registers"] = Json::object();
  for (const auto &[name, v] : out.special_registers)
    out.raw["special_registers"][name] =
        v < 0 ? Json(v.convert_to<std::int64_t>())
              : Json(v.convert_to<std::uint64_t>());
  out.raw["cta"]["cluster_dim"] = cluster;
  return out;
}
} // namespace sentryward::frontend::ptx
