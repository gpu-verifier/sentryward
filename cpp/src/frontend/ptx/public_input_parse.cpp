#include "sentryward/frontend/ptx/public_input.h"
#include "sentryward/algorithm/source_ir/ptx_assumptions.h"

#include <cmath>
#include <fstream>
#include <set>

namespace sentryward::frontend::ptx {
using Json = nlohmann::json;
InputError::InputError(std::string c, std::string p, std::string message,
                       std::string s)
    : std::runtime_error(p + ": " + message), code(std::move(c)),
      path(std::move(p)), status(std::move(s)) {}
namespace {
[[noreturn]] void Fail(const std::string &code, const std::string &path,
                       const std::string &message) {
  throw InputError(code, path, message);
}
void Object(const Json &v, const std::set<std::string> &keys,
            const std::string &path) {
  if (!v.is_object()) Fail("invalid_type", path, "expected an object");
  for (auto it = v.begin(); it != v.end(); ++it)
    if (!keys.contains(it.key()))
      Fail("unknown_field", path + "." + it.key(), "field is not accepted");
}
void Array(const Json &v, const std::string &path) {
  if (!v.is_array()) Fail("invalid_type", path, "expected an array");
}
void String(const Json &v, const std::string &path) {
  if (!v.is_string() || v.get_ref<const std::string &>().empty())
    Fail("invalid_type", path, "expected a nonempty string");
}
std::uint64_t Natural(const Json &v, const std::string &path,
                      std::uint64_t minimum = 0) {
  if (!v.is_number_integer() ||
      (!v.is_number_unsigned() && v.get<std::int64_t>() < 0) ||
      v.get<std::uint64_t>() < minimum)
    Fail("invalid_value", path, "expected a nonnegative integer in range");
  return v.get<std::uint64_t>();
}
void Selector(const Json &v, const std::string &path) {
  if (v.is_number_integer()) (void)Natural(v, path);
  else String(v, path);
}
bool ScalarType(const std::string &type) {
  static const std::set<std::string> types{
      "u8", "u16", "u32", "u64", "s8", "s16", "s32", "s64",
      "b8", "b16", "b32", "b64", "f16", "f32", "f64", "bool"};
  return types.contains(type);
}
void Scalar(const Json &v, const std::string &type, const std::string &path) {
  if (type == "pointer") {
    if (!v.is_null()) Fail("physical_pointer_value", path, "only null is accepted");
  } else if (type == "bool") {
    if (!v.is_boolean()) Fail("invalid_type", path, "expected a Boolean");
  } else if (type.starts_with('f')) {
    if (!v.is_number() || !std::isfinite(v.get<double>()))
      Fail("invalid_value", path, "expected a finite number");
  } else {
    if (!v.is_number_integer()) Fail("invalid_value", path, "expected an integer");
    const auto bits = std::stoul(type.substr(1));
    const Integer number = JsonInteger(v);
    const Integer lower = type.starts_with('s') ? -(Integer(1) << (bits - 1)) : Integer(0);
    const Integer upper = Integer(1) << (bits - unsigned(type.starts_with('s')));
    if (number < lower || number >= upper)
      Fail("invalid_value", path, "value is outside " + type);
  }
}
void Dimensions(const Json &v, const std::string &path) {
  if (!v.is_array() || v.size() != 3)
    Fail("invalid_value", path, "expected [x, y, z]");
  for (const auto &n : v) (void)Natural(n, path, 1);
}
std::string ItemPath(const std::string &name, std::size_t index) {
  return "$." + name + "[" + std::to_string(index) + "]";
}
void TensorMaps(Json &root) {
  auto &records = root["tma_descriptors"];
  Array(records, "$.tma_descriptors");
  const std::set<std::string> vectors{"box_shape", "global_shape", "coordinate_strides_bytes", "element_strides"};
  const std::set<std::string> element_types{
      "u8", "s8", "e4m3", "e5m2", "u16", "s16", "f16", "bf16",
      "u32", "s32", "f32", "tf32", "u64", "s64", "f64",
      "16u4_align16b", "16u4_align8b"};
  for (std::size_t i = 0; i < records.size(); ++i) {
    auto &record = records[i];
    const auto path = ItemPath("tma_descriptors", i);
    Object(record, {"parameter", "offset", "object", "element_type", "rank", "base_byte_offset",
                    "swizzle", "interleave", "oob_fill", "global_strides_bytes",
                    "box_shape", "global_shape", "coordinate_strides_bytes", "element_strides"}, path);
    Selector(record.value("parameter", Json()), path + ".parameter");
    record["offset"] = Natural(record.value("offset", Json(0)), path + ".offset");
    for (const auto *key : {"object", "element_type", "swizzle", "interleave", "oob_fill"})
      if (record.contains(key)) String(record[key], path + "." + key);
    if (record.contains("element_type") && !element_types.contains(record["element_type"]))
      throw InputError("unsupported_tma_element_type", path + ".element_type", "unsupported tensor-map element type", "ptx_execution_unsupported");
    const std::map<std::string, std::set<std::string>> modes{
        {"swizzle", {"none", "32B", "64B", "128B", "128B_atom32B", "128B_atom64B"}},
        {"interleave", {"none", "16B", "32B"}}, {"oob_fill", {"zero", "nan"}}};
    for (const auto &[key, choices] : modes)
      if (record.contains(key) && !choices.contains(record[key]))
        Fail("invalid_tma_descriptor", path + "." + key, "unknown tensor-map mode");
    std::set<std::size_t> ranks;
    for (const auto &key : vectors) if (record.contains(key)) {
      const auto &v = record[key];
      Array(v, path + "." + key);
      if (v.empty() || v.size() > 5)
        Fail("invalid_tma_descriptor", path + "." + key, "expected one to five dimensions");
      for (std::size_t j = 0; j < v.size(); ++j)
        (void)Natural(v[j], path + "." + key,
                      key == "coordinate_strides_bytes" && j != 0 ? 0 : 1);
      ranks.insert(v.size());
    }
    if (record.contains("rank")) {
      auto rank = Natural(record["rank"], path + ".rank", 1);
      if (rank > 5) Fail("invalid_tma_descriptor", path, "rank exceeds five");
      ranks.insert(rank);
    }
    if (ranks.size() > 1)
      Fail("inconsistent_tma_descriptor", path, "shape and stride ranks disagree");
    if (record.contains("global_strides_bytes")) {
      if (record.contains("coordinate_strides_bytes"))
        Fail("inconsistent_tma_descriptor", path, "choose one stride representation");
      const auto &v = record["global_strides_bytes"];
      Array(v, path + ".global_strides_bytes");
      for (const auto &n : v) (void)Natural(n, path + ".global_strides_bytes", 0);
      if (!ranks.empty() && v.size() + 1 != *ranks.begin())
        Fail("inconsistent_tma_descriptor", path, "outer stride count must be rank minus one");
    }
    if (record.value("element_type", "").starts_with("16u4") && record.contains("coordinate_strides_bytes"))
      Fail("inconsistent_tma_descriptor", path, "packed types require outer global_strides_bytes");
    if (record.contains("base_byte_offset")) (void)Natural(record["base_byte_offset"], path + ".base_byte_offset");
    static const std::map<std::string, unsigned> bytes{
        {"u8",1},{"s8",1},{"e4m3",1},{"e5m2",1},{"u16",2},{"s16",2},{"f16",2},{"bf16",2},
        {"u32",4},{"s32",4},{"f32",4},{"tf32",4},{"u64",8},{"s64",8},{"f64",8}};
    auto type = record.value("element_type", "");
    if (record.value("interleave", "") == "none" && record.contains("coordinate_strides_bytes") &&
        bytes.contains(type) && record["coordinate_strides_bytes"][0] != bytes.at(type))
      Fail("inconsistent_tma_descriptor", path, "dimension zero must have the element byte stride");
  }
}
} // namespace

KernelInput ParseKernelInput(const Json &value) {
  Object(value, {"schema", "entry", "launch", "arguments", "declarations", "shape_arrays", "tma_descriptors", "branch_decisions", "assumptions", "address_invariants", "data_branch_loop_iterations", "pointer_arrays", "tensor_properties"}, "$");
  if (value.value("schema", Json()) != "spverify.kernel-input/v1")
    Fail("invalid_schema", "$.schema", "expected spverify.kernel-input/v1");
  Json root = value;
  if (root.contains("assumptions")) {
    if (!root["assumptions"].is_object())
      Fail("invalid_assumptions", "$.assumptions", "expected Boolean assumption overrides");
    const auto allowed = algorithm::DefaultPtxAssumptions();
    for (const auto& item : root["assumptions"].items()) {
      const auto path = "$.assumptions." + item.key();
      if (!allowed.contains(item.key())) {
        if (item.key() == "immediate_visibility")
          Fail("unknown_assumption", path,
               "immediate_visibility is no longer configurable: immediate visibility is a fixed model premise; remove this field only if that premise matches the requested analysis");
        if (item.key() == "warp_tmem_load_completion")
          Fail("unknown_assumption", path,
               "warp_tmem_load_completion was removed; remove a false override, but a true override requires reviewing TMEM completion semantics");
        Fail("unknown_assumption", path, "unknown semantic assumption: " + item.key());
      }
      if (!item.value().is_boolean())
        Fail("invalid_assumption_type", path, "expected a Boolean");
    }
  }
  if (root.contains("data_branch_loop_iterations"))
    // Representative control flow loop bound; it should exceed the deepest pipeline the kernel wraps around.
    root["data_branch_loop_iterations"] = Natural(root["data_branch_loop_iterations"], "$.data_branch_loop_iterations", 1);
  if (root.contains("address_invariants")) {
    // Constraint contents are validated by the symbolic address model.
    Array(root["address_invariants"], "$.address_invariants");
    for (std::size_t i = 0; i < root["address_invariants"].size(); ++i)
      if (!root["address_invariants"][i].is_object())
        Fail("invalid_address_invariant", ItemPath("address_invariants", i), "expected an object");
  }
  auto &launch = root["launch"];
  Object(launch, {"grid_dim", "block_dim", "cluster_dim", "dynamic_shared_bytes"}, "$.launch");
  for (const auto *key : {"grid_dim", "block_dim"}) {
    if (!launch.contains(key)) Fail("missing_launch_dimension", "$.launch." + std::string(key), "launch dimensions are required");
    Dimensions(launch[key], "$.launch." + std::string(key));
  }
  if (launch.contains("cluster_dim")) Dimensions(launch["cluster_dim"], "$.launch.cluster_dim");
  launch["dynamic_shared_bytes"] = Natural(launch.value("dynamic_shared_bytes", Json(0)), "$.launch.dynamic_shared_bytes");
  if (root.contains("entry")) String(root["entry"], "$.entry");
  for (const auto *key : {"declarations", "arguments", "shape_arrays", "tma_descriptors", "branch_decisions", "pointer_arrays",
                          "tensor_properties"}) {
    if (!root.contains(key)) root[key] = Json::array();
    Array(root[key], "$." + std::string(key));
  }
  std::map<std::string, std::vector<Json>> names;
  for (std::size_t i = 0; i < root["declarations"].size(); ++i) {
    auto &d = root["declarations"][i];
    auto path = ItemPath("declarations", i);
    Object(d, {"parameter", "offset", "type", "space", "object", "name"}, path);
    if (!d.value("type", Json()).is_string()) Fail("invalid_declaration_type", path + ".type", "unknown parameter type");
    const auto type = d["type"].get<std::string>();
    if (!ScalarType(type) && type != "pointer" && type != "tensor_map")
      Fail("invalid_declaration_type", path + ".type", "unknown parameter type");
    Selector(d.value("parameter", Json()), path + ".parameter");
    d["offset"] = Natural(d.value("offset", Json(0)), path + ".offset");
    if (d.contains("space") && (type != "pointer" || d["space"] != "global"))
      Fail("invalid_value", path + ".space", "only global input pointers are supported");
    if (d.contains("object")) {
      if (type != "pointer" && type != "tensor_map") Fail("invalid_value", path + ".object", "scalar declarations do not name memory objects");
      String(d["object"], path + ".object");
    }
    if (d.contains("name")) { String(d["name"], path + ".name"); names[d["name"]].push_back(d); }
  }
  Json arguments = Json::array();
  std::set<std::string> used_names;
  for (std::size_t i = 0; i < root["arguments"].size(); ++i) {
    const auto &a = root["arguments"][i];
    auto path = ItemPath("arguments", i);
    if (!a.is_object() || !a.contains("field")) { arguments.push_back(a); continue; }
    if (a.size() != 2 || !a.contains("value")) Fail("invalid_named_argument", path, "expected field and value");
    if (!a["field"].is_string() || !names.contains(a["field"])) Fail("unknown_argument_field", path + ".field", "no matching ABI declaration");
    auto name = a["field"].get<std::string>();
    if (!used_names.insert(name).second) Fail("duplicate_argument", path, "named field already has a value");
    for (const auto &d : names.at(name)) {
      if (d["type"] == "tensor_map") Fail("invalid_named_argument", path, "descriptor contents are not accepted");
      arguments.push_back({{"parameter", d["parameter"]}, {"offset", d["offset"]}, {"type", d["type"]}, {"value", a["value"]}});
    }
  }
  root["arguments"] = std::move(arguments);
  for (std::size_t i = 0; i < root["arguments"].size(); ++i) {
    auto &a = root["arguments"][i];
    auto path = ItemPath("arguments", i);
    Object(a, {"parameter", "offset", "type", "value"}, path);
    if (!a.value("type", Json()).is_string()) Fail("invalid_scalar_type", path + ".type", "expected a scalar type or null pointer");
    const auto type = a["type"].get<std::string>();
    if (!ScalarType(type) && type != "pointer") Fail("invalid_scalar_type", path + ".type", "expected a scalar type or null pointer");
    if (!a.contains("value")) Fail("missing_argument_value", path + ".value", "value is required");
    Selector(a.value("parameter", Json()), path + ".parameter");
    a["offset"] = Natural(a.value("offset", Json(0)), path + ".offset");
    Scalar(a["value"], type, path + ".value");
  }
  for (std::size_t i = 0; i < root["shape_arrays"].size(); ++i) {
    auto &s = root["shape_arrays"][i];
    auto path = ItemPath("shape_arrays", i);
    Object(s, {"parameter", "offset", "element_type", "shapes"}, path);
    Selector(s.value("parameter", Json()), path + ".parameter");
    s["offset"] = Natural(s.value("offset", Json(0)), path + ".offset");
    auto type = s.value("element_type", Json("s32"));
    if (type != "s32" && type != "u32" && type != "s64" && type != "u64") Fail("invalid_shape_type", path, "expected a 32- or 64-bit integer type");
    s["element_type"] = type;
    Array(s["shapes"], path + ".shapes");
    if (s["shapes"].empty()) Fail("invalid_shape_array", path, "expected at least one shape");
    std::size_t rank = 0;
    for (const auto &shape : s["shapes"]) {
      Array(shape, path + ".shapes");
      if (shape.empty() || (rank && shape.size() != rank)) Fail("invalid_shape_array", path, "expected nonempty shapes of equal rank");
      rank = shape.size();
      for (const auto &dim : shape) { (void)Natural(dim, path); Scalar(dim, type, path); }
    }
  }
  // Pointer tables: entry k is the 8-byte pointer at table byte 8k, naming `object` at
  // `byte_offset` (optional `extent_bytes`), or null.
  for (std::size_t i = 0; i < root["pointer_arrays"].size(); ++i) {
    auto &p = root["pointer_arrays"][i];
    auto path = ItemPath("pointer_arrays", i);
    Object(p, {"parameter", "offset", "entries"}, path);
    Selector(p.value("parameter", Json()), path + ".parameter");
    p["offset"] = Natural(p.value("offset", Json(0)), path + ".offset");
    Array(p["entries"], path + ".entries");
    if (p["entries"].empty()) Fail("invalid_pointer_array", path, "expected at least one entry");
    for (std::size_t k = 0; k < p["entries"].size(); ++k) {
      auto &entry = p["entries"][k];
      const auto entry_path = path + ".entries[" + std::to_string(k) + "]";
      if (entry.is_null()) continue;
      Object(entry, {"object", "byte_offset", "extent_bytes"}, entry_path);
      String(entry["object"], entry_path + ".object");
      entry["byte_offset"] = Natural(entry.value("byte_offset", Json(0)), entry_path + ".byte_offset");
      if (entry.contains("extent_bytes") &&
          (entry["extent_bytes"] = Natural(entry["extent_bytes"], entry_path + ".extent_bytes")) == 0)
        Fail("invalid_pointer_array", entry_path + ".extent_bytes", "expected a positive extent");
    }
  }
  // Trusted element properties of an input object: one property per record
  // (injective, lower_bound or upper_bound) under the stated integer element type.
  for (std::size_t i = 0; i < root["tensor_properties"].size(); ++i) {
    auto &t = root["tensor_properties"][i];
    auto path = ItemPath("tensor_properties", i);
    Object(t, {"object", "element_type", "property", "value"}, path);
    String(t.value("object", Json()), path + ".object");
    const auto type = t.value("element_type", Json());
    if (type != "s32" && type != "u32" && type != "s64" && type != "u64")
      Fail("invalid_tensor_property", path + ".element_type", "expected s32, u32, s64 or u64");
    const auto property = t.value("property", Json());
    if (property != "injective" && property != "lower_bound" && property != "upper_bound")
      Fail("invalid_tensor_property", path + ".property", "expected injective, lower_bound or upper_bound");
    if ((property == "injective") == t.contains("value"))
      Fail("invalid_tensor_property", path + ".value", property == "injective" ? "injective takes no value" : "a bound requires a value");
    if (t.contains("value")) Scalar(t["value"], type.get<std::string>(), path + ".value");
  }
  TensorMaps(root);
  std::set<std::uint64_t> lines;
  for (std::size_t i = 0; i < root["branch_decisions"].size(); ++i) {
    auto &b = root["branch_decisions"][i];
    auto path = ItemPath("branch_decisions", i);
    Object(b, {"line", "taken", "note", "seed"}, path);
    auto line = Natural(b.value("line", Json()), path + ".line", 1);
    if (!lines.insert(line).second) Fail("duplicate_branch_decision", path, "branch line is already configured");
    if (b.value("taken", Json()) == "random") (void)Natural(b.value("seed", Json()), path + ".seed");
    else {
      Scalar(b.value("taken", Json()), "bool", path + ".taken");
      if (b.contains("seed")) Fail("invalid_branch_seed", path + ".seed", "seed requires taken=random");
    }
    if (b.contains("note")) String(b["note"], path + ".note");
  }
  return {std::move(root)};
}

KernelInput LoadKernelInput(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) throw InputError("invalid_input_file", "$", "cannot open " + path.string());
  std::vector<std::set<std::string>> keys;
  auto callback = [&](int, Json::parse_event_t event, Json &value) {
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    else if (event == Json::parse_event_t::object_end) keys.pop_back();
    else if (event == Json::parse_event_t::key && !keys.back().insert(value.get<std::string>()).second)
      throw InputError("duplicate_field", "$", "duplicate JSON field");
    return true;
  };
  try { return ParseKernelInput(Json::parse(input, callback)); }
  catch (const Json::exception &error) { throw InputError("invalid_input_file", "$", error.what()); }
}
} // namespace sentryward::frontend::ptx
