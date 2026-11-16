#pragma once
#include "core.hpp"
namespace rg {
// The bytes of an object a premise covers. Offsets are canonical byte offsets of the object (its
// shared layout: target offset plus coordinate times target stride, identical on both sides).
// whole_object covers every element of the object. record_bytes covers, in records of
// record_stride_bytes bytes starting at offset_bytes, the byte_count bytes from first_byte of each
// record: byte o is covered exactly when o >= offset_bytes and
// (o - offset_bytes) mod record_stride_bytes is in [first_byte, first_byte + byte_count). An element
// read is covered only when all its bytes are. Example: a cache of 528-byte tokens whose last 16
// bytes are scales is {offset_bytes 0, record_stride_bytes 528, first_byte 512, byte_count 16}.
struct InputRegion {
  struct Records { Z offset, stride, first, count; };
  std::optional<Records> records; // empty: whole_object
  static InputRegion parse(const J &json) {
    if (json.is_string() && json.get<std::string>() == "whole_object") return {};
    if (!json.is_object() || json.value("kind", std::string()) != "record_bytes")
      throw Unsupported("input contract region must be whole_object or a record_bytes object");
    for (auto it = json.begin(); it != json.end(); ++it)
      if (it.key() != "kind" && it.key() != "offset_bytes" && it.key() != "record_stride_bytes" &&
          it.key() != "first_byte" && it.key() != "byte_count")
        throw Unsupported("unknown input contract region field: " + it.key());
    auto natural = [&](const char *key) {
      if (!json.contains(key)) throw Unsupported(std::string("input contract record region requires ") + key);
      Q value = jq(json.at(key));
      if (value.denominator() != 1 || value < 0)
        throw Unsupported(std::string("input contract record region field is not a natural number: ") + key);
      return value.numerator();
    };
    Records r{natural("offset_bytes"), natural("record_stride_bytes"), natural("first_byte"), natural("byte_count")};
    if (r.stride <= 0 || r.count <= 0 || r.first + r.count > r.stride)
      throw Unsupported("input contract record region must select a nonempty byte range inside each record");
    return InputRegion{r};
  }
  bool whole() const { return !records; }
  bool operator==(const InputRegion &other) const {
    if (bool(records) != bool(other.records)) return false;
    return !records || (records->offset == other.records->offset && records->stride == other.records->stride &&
                        records->first == other.records->first && records->count == other.records->count);
  }
  J declaration() const {
    if (!records) return "whole_object";
    auto number = [](const Z &value) { return J::parse(value.convert_to<std::string>()); };
    return J{{"kind", "record_bytes"}, {"offset_bytes", number(records->offset)},
             {"record_stride_bytes", number(records->stride)}, {"first_byte", number(records->first)},
             {"byte_count", number(records->count)}};
  }
};
// Caller-supplied premises on canonical initial-memory objects, not derived facts.
// Exact type matching prevents constraints on decoded FP8 from bounding raw u8.
struct InputContract {
  struct Rule {
    std::string object, element_type, source;
    bool finite = false;
    Bounds range;
    std::optional<Q> lower, upper;
    InputRegion region;
  };
  std::map<std::pair<std::string,std::string>, Rule> rules;
  J declaration = J::array();
  mutable std::set<std::tuple<std::string,std::string,std::string>> applied;
  static InputContract parse(const J &json) {
    if (!json.is_array()) throw Unsupported("input contract must be an array");
    InputContract result;
    for (const auto &item : json) {
      if (!item.is_object()) throw Unsupported("input contract entry must be object");
      for (auto it=item.begin();it!=item.end();++it)
        if (it.key()!="object" && it.key()!="element_type" && it.key()!="source" &&
            it.key()!="region" && it.key()!="finite" && it.key()!="min_inclusive" && it.key()!="max_inclusive")
          throw Unsupported("unknown input contract field: " + it.key());
      Rule r{item.at("object").get<std::string>(),item.at("element_type").get<std::string>(),item.at("source").get<std::string>(), false, {}, {}, {}, {}};
      if(r.object.empty() || r.source.empty() || !item.contains("region"))
        throw Unsupported("input contract requires named object, source and region");
      r.region=InputRegion::parse(item.at("region"));
      static const std::set<std::string> types{"u8","u16","u32","u64","s8","s16","s32","s64","f16","bf16","f32","f64","e4m3","e5m2","e8m0"};
      if (!types.contains(r.element_type)) throw Unsupported("unsupported input contract element type");
      r.finite=item.value("finite",false);
      if(item.contains("min_inclusive") != item.contains("max_inclusive"))
        throw Unsupported("input contract range requires both inclusive endpoints");
      if(item.contains("min_inclusive")) {
        Q lo=jq(item.at("min_inclusive")), hi=jq(item.at("max_inclusive"));
        if(lo>hi) throw Unsupported("empty input contract range");
        if(r.element_type[0]=='s' || r.element_type[0]=='u') {
          unsigned w=std::stoul(r.element_type.substr(1));
          Q lower=r.element_type[0]=='s'?Q(-pow2(w-1)):Q(0);
          Q upper=r.element_type[0]=='s'?Q(pow2(w-1)-1):Q(pow2(w)-1);
          if(lo.denominator()!=1 || hi.denominator()!=1 || lo<lower || hi>upper)
            throw Unsupported("input contract integer range outside declared type");
        }
        r.range=std::pair(lo,hi);r.finite=true;
      }
      if(!r.finite && !r.range) throw Unsupported("empty input contract premise");
      if(!result.rules.emplace(std::pair(r.object,r.element_type),r).second)
        throw Unsupported("duplicate input contract object/type");
    }
    result.declaration=json;
    return result;
  }
  const Rule *find(const std::string &object,const std::string &type) const {
    auto it=rules.find({object,type});return it==rules.end()?nullptr:&it->second;
  }
  void validate_mapping(const J &mapping) const {
    for(const auto &[key,rule]:rules) for(const auto *side:{"left_inputs","right_inputs"}) {
      bool found=false;
      for(const auto &name:mapping.at(side).value("allocations",J::object()))
        found |= name==rule.object;
      if(!found) throw Unsupported("input contract object not mapped on both sides: "+rule.object);
    }
  }
};
}
