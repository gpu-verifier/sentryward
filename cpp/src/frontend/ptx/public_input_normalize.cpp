#include "sentryward/frontend/ptx/public_input.h"
#include "sentryward/frontend/ptx/module.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <openssl/evp.h>
#include <regex>
#include <set>

namespace sentryward::frontend::ptx {
namespace {
using Json = nlohmann::json;
using Key = std::pair<std::size_t, std::uint64_t>;
[[noreturn]] void Fail(const std::string &code, const std::string &path,
                       const std::string &message,
                       const std::string &status = "invalid_configuration") {
  throw InputError(code, path, message, status);
}
std::string Type(const Parameter &p) {
  const std::regex pattern(R"(\.[busf](8|16|32|64|128))");
  for (const auto &s : p.types) if (std::regex_match(s, pattern)) return s.substr(1);
  return "";
}
unsigned Width(const std::string &type) {
  if (type == "pointer") return 8;
  return type == "bool" ? 1 : std::stoul(type.substr(1)) / 8;
}
bool Has(const Parameter &p, const std::string &type) {
  return std::find(p.types.begin(), p.types.end(), type) != p.types.end();
}
std::string Resource(Key key) {
  return "input.arg" + std::to_string(key.first) + "." + std::to_string(key.second);
}
std::uint64_t Bits(const std::string &type, const Json &value, const std::string &path) {
  if (type == "pointer") return 0;
  if (type.starts_with('f')) {
    const double number = value.get<double>();
    if (type == "f64") return std::bit_cast<std::uint64_t>(number);
    if (type == "f32") {
      const float f = static_cast<float>(number);
      if (!std::isfinite(f)) Fail("invalid_value", path, "value overflows f32");
      return std::bit_cast<std::uint32_t>(f);
    }
    const auto round_even = [](double n) {
      auto lower = static_cast<std::uint64_t>(std::floor(n));
      return lower + unsigned(n - lower > .5 || (n - lower == .5 && (lower & 1)));
    };
    const double magnitude = std::abs(number);
    const std::uint64_t sign = std::signbit(number) ? 0x8000 : 0;
    if (magnitude < std::ldexp(1., -14))
      return sign | round_even(std::ldexp(magnitude, 24));
    if (magnitude >= 65520.) Fail("invalid_value", path, "value overflows f16");
    int exponent = 0;
    const double mantissa = std::frexp(magnitude, &exponent);
    const auto fraction = round_even(mantissa * 2048) - 1024;
    return sign | ((static_cast<std::uint64_t>(exponent + 14) << 10) + fraction);
  }
  const Integer mask = (Integer(1) << (Width(type) * 8)) - 1;
  return (JsonInteger(value) & mask).convert_to<std::uint64_t>();
}
const Function &Entry(const Json &input, const Module &module) {
  const Function *found = nullptr;
  for (const auto &f : module.functions) {
    if (f.kind != "entry" || !f.has_body) continue;
    if (input.contains("entry") && input["entry"] != f.name) continue;
    if (found) Fail("entry_required", "$.entry", "select one kernel entry");
    found = &f;
  }
  if (!found) Fail(input.contains("entry") ? "unknown_entry" : "entry_required", "$.entry", "entry is not defined in this PTX");
  return *found;
}
} // namespace

FrontendConfig NormalizeKernelInput(const KernelInput &supplied, const Module &module,
                                    const InputOptions &options,
                                    const std::vector<InferredPointer> &inferred) {
  const auto input = ParseKernelInput(supplied.fields).fields;
  const auto &function = Entry(input, module);
  if (!options.max_steps_per_thread || options.elect_lane >= 32)
    Fail("invalid_analysis_option", "analysis", "invalid step budget or elected lane");
  Json launch = input["launch"];
  const auto block = launch["block_dim"];
  Integer threads = 1;
  for (const auto &n : block) threads *= JsonInteger(n);
  if (threads > 1024) Fail("invalid_launch", "$.launch.block_dim", "a CTA exceeds 1024 threads");
  if (threads % 32 != 0) Fail("partial_warp_launch", "$.launch.block_dim", "partial-warp launches are not supported", "ptx_execution_unsupported");
  for (const auto &d : function.directives) {
    if (d.name != ".reqntid" && d.name != ".maxntid" && d.name != ".reqnctapercluster") continue;
    auto text = std::string(module.view(d.arguments));
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream stream(text);
    Json dims = Json::array();
    std::uint64_t n;
    while (stream >> n) dims.push_back(n);
    while (dims.size() < 3) dims.push_back(1);
    if (dims.size() != 3) continue;
    if (d.name == ".reqnctapercluster") {
      if (launch.contains("cluster_dim") && launch["cluster_dim"] != dims)
        Fail("inconsistent_launch", "$.launch.cluster_dim", "contradicts PTX cluster dimensions");
      launch["cluster_dim"] = dims;
    } else if (d.name == ".reqntid" && block != dims)
      Fail("inconsistent_launch", "$.launch.block_dim", "contradicts PTX required dimensions");
    else if (d.name == ".maxntid" && threads > JsonInteger(dims[0]) * JsonInteger(dims[1]) * JsonInteger(dims[2]))
      Fail("inconsistent_launch", "$.launch.block_dim", "exceeds PTX maximum dimensions");
  }
  const auto cluster = launch.value("cluster_dim", Json::array({1, 1, 1}));
  for (unsigned i = 0; i < 3; ++i)
    if (JsonInteger(launch["grid_dim"][i]) % JsonInteger(cluster[i]) != 0)
      Fail("invalid_launch", "$.launch.grid_dim", "grid must contain complete clusters");

  std::vector<std::uint64_t> sizes;
  Json regions = Json::object();
  for (const auto &p : function.parameters) {
    const auto type = Type(p);
    if (type.empty() || (p.array && !std::regex_match(*p.array, std::regex(R"(\[\d+\])"))))
      Fail("unknown_parameter_layout", "$.declarations", "parameter storage size cannot be derived", "ptx_execution_unsupported");
    if (!p.name) Fail("unnamed_parameter", "$.declarations", "entry parameter has no name", "ptx_execution_unsupported");
    Integer bytes = Width(type);
    if (p.array) bytes *= Integer(p.array->substr(1, p.array->size() - 2));
    if (bytes < 1 || bytes > UINT64_MAX) Fail("unknown_parameter_layout", "$.declarations", "parameter size is outside supported range", "ptx_execution_unsupported");
    sizes.push_back(bytes.convert_to<std::uint64_t>());
    regions[*p.name] = {{"size_bytes", sizes.back()}, {"offset_values", Json::object()}};
  }
  auto resolve = [&](const Json &selector, const std::string &path) -> std::size_t {
    if (selector.is_number_integer() && JsonInteger(selector) < function.parameters.size())
      return selector.get<std::size_t>();
    for (std::size_t i = 0; i < function.parameters.size(); ++i)
      if (selector.is_string() && selector == *function.parameters[i].name) return i;
    Fail("unknown_parameter", path + ".parameter", "parameter does not exist");
  };
  struct Field { std::uint64_t begin, end; std::string type, path; };
  struct Pointer { unsigned width; std::string object, source; };
  std::map<std::size_t, std::vector<Field>> fields;
  std::map<Key, Pointer> pointers;
  std::set<Key> nulls, shape_keys, descriptor_keys;
  Json opaque = Json::array(), shape_inputs = Json::array(), descriptors = Json::object();
  auto field = [&](const Json &selector, std::uint64_t offset, std::uint64_t width,
                   const std::string &type, const std::string &path, bool merge = false) {
    const auto i = resolve(selector, path);
    const auto &p = function.parameters[i];
    if (!width || offset > sizes[i] || width > sizes[i] - offset)
      Fail("field_out_of_bounds", path, "field is outside the PTX parameter");
    const auto base = Type(p);
    if (!p.array) {
      const bool pointer = type == "pointer" || type == "tensor_map";
      const bool compatible = pointer ? (base == "b64" || base == "u64" || base == "s64") :
          (base.starts_with('b') || (type == "bool" ? "u8" : type) == base);
      if (offset != 0 || width != sizes[i] || !compatible)
        Fail("inconsistent_parameter_type", path + ".type", "type contradicts the PTX parameter declaration");
    }
    if (Has(p, ".ptr") && type != "pointer" && type != "tensor_map")
      Fail("physical_pointer_value", path, "pointer inputs use symbolic objects");
    for (const auto &prior : fields[i]) if (offset < prior.end && prior.begin < offset + width) {
      if (merge && prior.begin == offset && prior.end == offset + width && prior.type == type) return i;
      Fail("conflicting_declaration", path, "field overlaps " + prior.path);
    }
    fields[i].push_back({offset, offset + width, type, path});
    return i;
  };
  auto add_map = [&](std::size_t i, std::uint64_t offset, const std::string &object) {
    const Json location = Json::array({*function.parameters[i].name, offset});
    for (const auto &entry : opaque) if (entry["parameter_location"] == location) return;
    opaque.push_back({{"parameter_location", location}, {"object", object}});
  };
  for (std::size_t n = 0; n < input["declarations"].size(); ++n) {
    const auto &d = input["declarations"][n];
    const auto path = "$.declarations[" + std::to_string(n) + "]";
    const auto i = resolve(d["parameter"], path);
    const auto type = d["type"].get<std::string>();
    const auto offset = d["offset"].get<std::uint64_t>();
    const auto width = type == "tensor_map" ? (function.parameters[i].array ? 128 : sizes[i]) : Width(type);
    field(d["parameter"], offset, width, type, path);
    if (type == "pointer") pointers[{i,offset}] = {static_cast<unsigned>(width), d.value("object", ""), "declaration"};
    if (type == "tensor_map") {
      auto object = d.value("object", "tensor_map.arg" + std::to_string(i) + "." + std::to_string(offset));
      add_map(i, offset, object);
      if (!function.parameters[i].array && width == 8) pointers[{i,offset}] = {8, object, "tensor_map"};
    }
  }
  for (std::size_t n = 0; n < input["tma_descriptors"].size(); ++n) {
    const auto &d = input["tma_descriptors"][n];
    const auto path = "$.tma_descriptors[" + std::to_string(n) + "]";
    const auto i = resolve(d["parameter"], path);
    const auto offset = d["offset"].get<std::uint64_t>();
    const auto width = function.parameters[i].array ? 128 : sizes[i];
    field(d["parameter"], offset, width, "tensor_map", path, true);
    if (!descriptor_keys.insert({i,offset}).second) Fail("duplicate_tma_descriptor", path, "descriptor field is already configured");
    auto object = "tensor_map.arg" + std::to_string(i) + "." + std::to_string(offset);
    if (!function.parameters[i].array && width == 8 && !pointers.contains({i,offset})) pointers[{i,offset}] = {8, object, "tensor_map"};
    add_map(i, offset, object);
  }
  for (std::size_t n = 0; n < input["arguments"].size(); ++n) {
    const auto &a = input["arguments"][n];
    const auto path = "$.arguments[" + std::to_string(n) + "]";
    const auto type = a["type"].get<std::string>();
    const auto offset = a["offset"].get<std::uint64_t>();
    const auto width = Width(type);
    const auto i = field(a["parameter"], offset, width, type, path, true);
    if (type == "pointer") {
      if (pointers.contains({i,offset}) && !pointers.at({i,offset}).object.empty())
        Fail("inconsistent_pointer_declaration", path, "a null pointer cannot name an allocated object");
      pointers.erase({i,offset}); nulls.insert({i,offset});
    }
    auto &values = regions[*function.parameters[i].name]["offset_values"];
    const auto key = std::to_string(offset);
    if (values.contains(key)) Fail("duplicate_argument", path, "parameter field already has a value");
    values[key] = {{"bytes", width}, {"value", Bits(type, a["value"], path + ".value")}};
  }
  for (std::size_t n = 0; n < input["shape_arrays"].size(); ++n) {
    const auto &s = input["shape_arrays"][n];
    const auto path = "$.shape_arrays[" + std::to_string(n) + "]";
    const auto offset = s["offset"].get<std::uint64_t>();
    const auto i = field(s["parameter"], offset, 8, "pointer", path, true);
    const Key key{i,offset};
    if (nulls.contains(key)) Fail("inconsistent_shape_pointer", path, "a shape array cannot use a null pointer");
    if (!shape_keys.insert(key).second) Fail("duplicate_shape_array", path, "pointer already has a shape array");
    pointers.try_emplace(key, Pointer{8, "", "shape_array"});
    Json values = Json::array();
    for (const auto &shape : s["shapes"]) for (const auto &v : shape) values.push_back(v);
    shape_inputs.push_back({{"resource", Resource(key)}, {"element_bits", Width(s["element_type"]) * 8}, {"values", values}});
  }
  for (std::size_t i = 0; i < function.parameters.size(); ++i) {
    const auto &p = function.parameters[i];
    if (!Has(p, ".ptr")) continue;
    if (Has(p, ".shared") || Has(p, ".local") || Has(p, ".const"))
      Fail("unsupported_pointer_space", "$.declarations", "only global input objects are supported", "ptx_execution_unsupported");
    if (nulls.contains({i,0}) || pointers.contains({i,0})) continue;
    field(i, 0, sizes[i], "pointer", "$.declarations");
    pointers[{i,0}] = {static_cast<unsigned>(sizes[i]), "", "ptx_ptr"};
  }
  for (const auto &p : inferred) {
    const auto i = resolve(p.parameter, "$.declarations");
    const Key key{i,p.offset};
    if (nulls.contains(key)) continue;
    if (pointers.contains(key)) {
      if (pointers.at(key).width != p.width_bytes)
        Fail("inconsistent_pointer_declaration", "$.declarations", "declaration contradicts inferred pointer");
      continue;
    }
    field(p.parameter, p.offset, p.width_bytes, "pointer", "$.declarations");
    pointers[key] = {p.width_bytes, "", "address_use"};
  }
  Json resources = Json::array(), pointer_inputs = Json::array(), objects = Json::array();
  std::map<std::string, std::string> object_resources;
  auto memory = [&](const std::string &id, const std::string &object) {
    resources.push_back({{"id", id}, {"kind", "memory"}, {"space", "global"},
        {"allocation", {{"resolution", "concrete"}, {"object", object}, {"base_byte_offset", 0}}}});
    object_resources[object] = id;
  };
  for (const auto &[key,p] : pointers) {
    const auto id = Resource(key), object = p.object.empty() ? id : p.object;
    memory(id, object);
    pointer_inputs.push_back({{"source", {{"parameter", *function.parameters[key.first].name}, {"byte_offset", key.second}}},
        {"resource", id}, {"width_bytes", p.width}, {"extent_bytes", "unbounded"}});
    objects.push_back({{"parameter", *function.parameters[key.first].name}, {"offset", key.second},
        {"bytes", p.width}, {"object", object}, {"source", p.source}});
  }
  static const std::map<std::string, unsigned> element_bytes{
      {"u8",1},{"s8",1},{"e4m3",1},{"e5m2",1},{"u16",2},{"s16",2},{"f16",2},{"bf16",2},
      {"u32",4},{"s32",4},{"f32",4},{"tf32",4},{"u64",8},{"s64",8},{"f64",8}};
  for (auto d : input["tma_descriptors"]) {
    const auto i = resolve(d["parameter"], "$.tma_descriptors");
    const auto offset = d["offset"].get<std::uint64_t>();
    d.erase("parameter"); d.erase("offset");
    d["parameter_location"] = Json::array({*function.parameters[i].name, offset});
    if (pointers.contains({i,offset})) d["allocation_location"] = Json::array({Resource({i,offset}), 0});
    if (d.contains("element_type")) {
      const auto type = d["element_type"].get<std::string>();
      if (type.starts_with("16u4")) {
        d["element_bytes"] = 1; d["packed_element_bits"] = 4;
        d["packed_shared_alignment"] = type == "16u4_align16b" ? 16 : 8;
        for (const auto *name : {"box_shape", "global_shape"}) if (d.contains(name)) {
          auto n = d[name][0].get<std::uint64_t>();
          if (n % 2) Fail("inconsistent_tma_descriptor", "$.tma_descriptors", "packed dimension zero must contain whole bytes");
          d[name][0] = n / 2;
        }
      } else d["element_bytes"] = element_bytes.at(type);
      d.erase("element_type");
      if (d.contains("global_strides_bytes")) {
        Json strides = Json::array({d["element_bytes"]});
        for (const auto &s : d["global_strides_bytes"]) strides.push_back(s);
        d["coordinate_strides_bytes"] = std::move(strides); d.erase("global_strides_bytes");
      }
    }
    if (d.contains("object")) {
      const auto object = d["object"].get<std::string>();
      if (!object_resources.contains(object)) memory("tma.input." + std::to_string(object_resources.size()), object);
      d["resource"] = object_resources.at(object); d.erase("object");
    }
    descriptors["input.tma." + std::to_string(i) + "." + std::to_string(offset)] = std::move(d);
  }
  Json decisions = Json::array(), overrides = Json::array();
  for (auto b : input["branch_decisions"]) {
    const auto line = b["line"].get<std::uint64_t>();
    std::vector<const Instruction *> matches;
    for (const auto &inst : function.instructions) if (inst.line == line) matches.push_back(&inst);
    if (matches.size() != 1 || matches[0]->control != "direct_branch" || !matches[0]->predicate)
      Fail("invalid_branch_target", "$.branch_decisions", "expected a unique conditional direct branch in the selected entry");
    bool taken;
    if (b["taken"] == "random") {
      const auto text = "spverify.branch.v1:" + b["seed"].dump() + ":" + std::to_string(line);
      unsigned char digest[EVP_MAX_MD_SIZE]; unsigned size = 0;
      if (!EVP_Digest(text.data(), text.size(), digest, &size, EVP_sha256(), nullptr))
        throw std::runtime_error("SHA256 failed");
      taken = (digest[0] & 1) != 0;
      b["resolved_taken"] = taken; b["sampling"] = "fixed_per_branch_sha256_v1";
    } else taken = b["taken"].get<bool>();
    decisions.push_back(b); overrides.push_back({{"line", line}, {"outcomes", taken}});
  }
  Json types = Json::array(), null_fields = Json::array(), shape_fields = Json::array();
  for (auto &[i, entries] : fields) {
    std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) { return a.begin < b.begin; });
    for (const auto &f : entries) if (f.type != "pointer" && f.type != "tensor_map")
      types.push_back({{"parameter", *function.parameters[i].name}, {"offset", f.begin}, {"bytes", f.end-f.begin}, {"type", f.type}});
  }
  for (auto [i,offset] : nulls) null_fields.push_back({{"parameter", *function.parameters[i].name}, {"offset", offset}});
  for (auto [i,offset] : shape_keys) shape_fields.push_back({{"parameter", *function.parameters[i].name}, {"offset", offset}});
  Json raw{{"schema", "spverify.ptx-front-end-config/v1"}, {"entry", function.name},
      {"cta", {{"block_dim", block}, {"cluster_dim", cluster}}}, {"parameter_regions", regions},
      {"pointer_parameters", pointer_inputs}, {"resources", resources},
      {"branch_overrides", overrides}, {"user_branch_decisions", decisions},
      {"opaque_tensor_maps", opaque}, {"parameter_field_types", types}, {"input_objects", objects},
      {"null_pointer_fields", null_fields}, {"semantic_assumptions", supplied.fields.value("assumptions", Json::object())},
      {"address_invariants", supplied.fields.value("address_invariants", Json::array())},
      {"data_branch_loop_iterations", supplied.fields.value("data_branch_loop_iterations", Json(4))},
      {"elect_lane", options.elect_lane}, {"max_steps_per_lane", options.max_steps_per_thread},
      {"compress_compute", options.compress_compute},
      {"compress_thread_records", options.compress_thread_records},
      {"fixed_input", {{"grid_dim", launch["grid_dim"]}, {"dynamic_shared_bytes", launch["dynamic_shared_bytes"]},
          {"kernel_pointer_contract", {{"aliasing", "distinct_objects_noalias"}, {"extent_bytes", "unbounded"}}},
          {"launch_contract", {{"external_grids", "no_overlap"}}}, {"cluster_launch_cancellation", false},
          {"assume_trap_assertions", true}, {"public_input_schema", "spverify.kernel-input/v1"}}}};
  if (!shape_inputs.empty()) {
    raw["fixed_input"]["global_memory_inputs"] = shape_inputs;
    raw["fixed_input"]["shape_array_inputs"] = shape_fields;
    Json protected_regions = Json::array();
    for (const auto &shape : shape_inputs) {
      const auto &resource = *std::find_if(resources.begin(), resources.end(),
          [&](const auto &r) { return r["id"] == shape["resource"]; });
      const auto &allocation = resource["allocation"];
      protected_regions.push_back({{"resource", shape["resource"]},
          {"allocation", allocation["object"]},
          {"byte_offset", allocation["base_byte_offset"]},
          {"bytes", shape["values"].size() * shape["element_bits"].get<unsigned>() / 8}});
    }
    raw["fixed_input"]["shape_input_contract"] = {
        {"kind", "fixed_read_only"}, {"origin", "shape_arrays"},
        {"text", "Configured shape bytes retain their supplied values and are not written during the analyzed execution."},
        {"regions", protected_regions}};
  }
  auto result = ParseConfig(raw);
  result.tma_descriptors = descriptors; result.raw["tma_descriptors"] = descriptors;
  for (auto &[name,region] : result.parameter_regions) { (void)name; region.source = {{"kind", "typed_fields"}}; }
  return result;
}
} // namespace sentryward::frontend::ptx
