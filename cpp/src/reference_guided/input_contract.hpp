#pragma once
#include "core.hpp"
namespace rg {
// Caller-supplied premises on canonical initial-memory objects, not derived facts.
// Exact type matching prevents constraints on decoded FP8 from bounding raw u8.
struct InputContract {
  struct Rule {
    std::string object, element_type, source;
    bool finite = false;
    Bounds range;
    std::optional<Q> lower, upper;
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
      Rule r{item.at("object").get<std::string>(),item.at("element_type").get<std::string>(),item.at("source").get<std::string>(), false, {}, {}, {}};
      if(r.object.empty() || r.source.empty() || item.at("region")!="whole_object")
        throw Unsupported("input contract requires named object, source and whole_object region");
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
