#pragma once
#include "input_contract.hpp"
#include "scoped_predicate.hpp"
#include <variant>
namespace rg {
// Arena-scoped predicates and immutable context: use one context per comparison.
struct Interval {
  std::optional<Q> lower, upper;
  bool lower_inclusive = true, upper_inclusive = true;
};
struct Finite {};
struct InputElementAssumption {
  std::string input, interpretation;
  std::variant<Interval, Finite> property;
  std::string reason = "caller-provided premise";
};
struct PredicateAssumption { Id predicate = invalid; }; // Boolean node in this comparison's Arena
using Assumption = std::variant<InputElementAssumption, PredicateAssumption, ScopedPredicateAssumption>;
using AssumptionSet = std::vector<Assumption>;
struct InputCorrespondence {
  std::string reference, optimized, shared;
};
// Exact values of selected initial-memory elements, in the common logical
// coordinate system. These are explicit premises, not expression/node rewrites.
// The caller establishes the input correspondence and read-only lifetime.
struct FixedInputElement {
  std::string input, interpretation;
  std::vector<Z> coordinates;
  Q value;
  std::string reason;
};
struct FixedInputKey {
  std::string input, interpretation;
  std::vector<Z> coordinates;
  bool operator==(const FixedInputKey &) const = default;
};
struct FixedInputKeyHash {
  size_t operator()(const FixedInputKey &key) const {
    size_t h=std::hash<std::string>{}(key.input);
    mix(h,std::hash<std::string>{}(key.interpretation));
    for(const Z &coordinate:key.coordinates)
      mix(h,boost::multiprecision::hash_value(coordinate));
    return h;
  }
};
enum class NumericalSemantics { IdealRealExactWord };
struct EquivalenceContext {
  std::vector<InputCorrespondence> inputs;
  AssumptionSet assumptions;
  std::vector<FixedInputElement> fixed_inputs;
  NumericalSemantics semantics = NumericalSemantics::IdealRealExactWord;
};
// Prepared once. JSON belongs to the legacy mapping adapter, not the public context.
struct PreparedContext {
  InputContract contract;
  std::unordered_set<Id> predicates;
  mutable std::unordered_set<Id> applied_predicates;
  PreparedScopedPredicates scoped_predicates;
  std::map<std::pair<std::string,std::string>,Interval> intervals;
  std::vector<FixedInputElement> fixed_inputs;
  std::unordered_map<FixedInputKey,size_t,FixedInputKeyHash> fixed_lookup;
  std::unordered_map<std::string,std::unordered_set<std::string>> fixed_input_types;
  mutable std::vector<size_t> applied_fixed_inputs;
  mutable std::vector<bool> fixed_input_applied;
  PreparedContext(Arena &a, const EquivalenceContext &context) {
    if(context.semantics != NumericalSemantics::IdealRealExactWord)
      throw Unsupported("unsupported numerical semantics");
    fixed_inputs.reserve(context.fixed_inputs.size());
    fixed_lookup.reserve(context.fixed_inputs.size());
    for(const auto &entry:context.fixed_inputs) {
      static const std::set<std::string> types{"u8","u16","u32","u64","s8","s16","s32","s64"};
      if(entry.input.empty() || entry.reason.empty() || entry.coordinates.empty() ||
         !types.contains(entry.interpretation))
        throw Unsupported("fixed input requires identity, integer type, coordinates and provenance");
      if(std::any_of(entry.coordinates.begin(),entry.coordinates.end(),[](const Z &v){return v<0;}))
        throw Unsupported("negative fixed input coordinate");
      unsigned bits=std::stoul(entry.interpretation.substr(1));
      Q lo=entry.interpretation[0]=='s'?Q(-pow2(bits-1)):Q(0);
      Q hi=entry.interpretation[0]=='s'?Q(pow2(bits-1)-1):Q(pow2(bits)-1);
      if(entry.value.denominator()!=1 || entry.value<lo || entry.value>hi)
        throw Unsupported("fixed input literal outside declared integer type");
      FixedInputKey key{entry.input,entry.interpretation,entry.coordinates};
      if(!fixed_lookup.emplace(std::move(key),fixed_inputs.size()).second)
        throw Unsupported("duplicate fixed input coordinate/type");
      fixed_input_types[entry.input].insert(entry.interpretation);
      fixed_inputs.push_back(entry);
    }
    fixed_input_applied.resize(fixed_inputs.size(),false);
    for(const auto &entry:context.assumptions) {
      if(const auto *p=std::get_if<PredicateAssumption>(&entry)) {
        if(p->predicate>=a.nodes.size()) throw Unsupported("predicate outside comparison Arena");
        const auto &n=a.nodes[p->predicate];
        if(p->predicate==a.zero) throw Unsupported("inconsistent assumptions: false predicate");
        if(p->predicate!=a.one && n.op!="compare") throw Unsupported("assumption requires a Boolean comparison");
        std::vector<Id> work{p->predicate};std::unordered_set<Id> seen;
        while(!work.empty()) {
          Id id=work.back();work.pop_back();if(!seen.insert(id).second)continue;
          const auto &node=a.nodes[id];
          if(node.op=="var" || node.op=="fold") throw Unsupported("predicate assumption has an unscoped binder");
          work.insert(work.end(),node.args.begin(),node.args.end());
        }
        if(n.op=="compare") {
          static const std::map<std::string,std::string> inverse{{"eq","ne"},{"ne","eq"},{"lt","ge"},{"ge","lt"},{"le","gt"},{"gt","le"}};
          Id opposite=a.compare(inverse.at(n.tag),n.args[0],n.args[1]);
          if(predicates.contains(opposite)) throw Unsupported("inconsistent assumptions: opposite predicates");
        }
        predicates.insert(p->predicate);
      } else if (const auto *p=std::get_if<ScopedPredicateAssumption>(&entry)) {
        scoped_predicates.add(a,*p);
      } else {
        const auto &e=std::get<InputElementAssumption>(entry);
        InputContract::Rule r{e.input,e.interpretation,e.reason,false,{},{},{}};
        if(e.input.empty() || e.reason.empty()) throw Unsupported("input premise needs input and provenance");
        static const std::set<std::string> types{"u8","u16","u32","u64","s8","s16","s32","s64","f16","bf16","f32","f64","e4m3","e5m2","e8m0"};
        if(!types.contains(e.interpretation)) throw Unsupported("unsupported input premise interpretation");
        r.finite=true;
        if(const auto *interval=std::get_if<Interval>(&e.property)) {
          auto key=std::pair(e.input,e.interpretation);
          Interval combined=*interval;
          if(auto old=intervals.find(key);old!=intervals.end()) {
            const auto &prior=old->second;
            if(prior.lower && (!combined.lower || *prior.lower>*combined.lower)) {
              combined.lower=prior.lower;combined.lower_inclusive=prior.lower_inclusive;
            } else if(prior.lower && combined.lower && *prior.lower==*combined.lower)
              combined.lower_inclusive &= prior.lower_inclusive;
            if(prior.upper && (!combined.upper || *prior.upper<*combined.upper)) {
              combined.upper=prior.upper;combined.upper_inclusive=prior.upper_inclusive;
            } else if(prior.upper && combined.upper && *prior.upper==*combined.upper)
              combined.upper_inclusive &= prior.upper_inclusive;
          }
          intervals[key]=combined;
          interval=&intervals.at(key);
          auto lo=interval->lower, hi=interval->upper;
          if(!lo && !hi) throw Unsupported("empty interval premise");
          if(lo && hi && (*lo>*hi || (*lo==*hi && (!interval->lower_inclusive || !interval->upper_inclusive))))
            throw Unsupported("inconsistent assumptions: empty interval");
          bool integer=e.interpretation[0]=='s' || e.interpretation[0]=='u';
          if(integer) {
            unsigned bits=std::stoul(e.interpretation.substr(1));
            Q bottom=e.interpretation[0]=='s'?Q(-pow2(bits-1)):Q(0);
            Q top=e.interpretation[0]=='s'?Q(pow2(bits-1)-1):Q(pow2(bits)-1);
            if(lo) { Z v=-floorq(-*lo); if(!interval->lower_inclusive && Q(v)==*lo) ++v; lo=Q(v); }
            if(hi) { Z v=floorq(*hi); if(!interval->upper_inclusive && Q(v)==*hi) --v; hi=Q(v); }
            lo=lo?std::max(*lo,bottom):bottom;hi=hi?std::min(*hi,top):top;
            if(*lo>*hi) throw Unsupported("inconsistent assumptions: no representable integer");
          }
          // Closed hull is sound for open real bounds. One-sided bounds are
          // retained below and intersected with any bounds already known at reads.
          r.lower=lo;r.upper=hi;
          if(lo && hi) r.range=std::pair(*lo,*hi);
        }
        auto key=std::pair(r.object,r.element_type);
        auto [it,inserted]=contract.rules.emplace(key,r);
        if(!inserted) {
          auto &old=it->second;
          if(r.lower) old.lower=old.lower?std::max(*old.lower,*r.lower):r.lower;
          if(r.upper) old.upper=old.upper?std::min(*old.upper,*r.upper):r.upper;
          if(old.lower && old.upper) {
            if(*old.lower>*old.upper) throw Unsupported("inconsistent assumptions: disjoint intervals");
            old.range=std::pair(*old.lower,*old.upper);
          }
        }
      }
    }
    for(const auto &entry:fixed_inputs)
      if(const auto *rule=contract.find(entry.input,entry.interpretation))
        if((rule->lower && entry.value<*rule->lower) ||
           (rule->upper && entry.value>*rule->upper) ||
           (rule->range && (entry.value<rule->range->first || entry.value>rule->range->second)))
          throw Unsupported("fixed input conflicts with input interval");
  }
  Id apply(Arena &a, Id node) const {
    if(predicates.contains(node)) { applied_predicates.insert(node);return a.one; }
    return scoped_predicates.apply(a,node);
  }
  // O(rank) hashing and expected O(1) lookup; never scans the configured array.
  // Symbolic coordinates remain symbolic. An exact singleton bound is sufficient
  // because it proves the coordinate independently of this premise.
  std::optional<Id> fixed_read(Arena &a,const std::string &input,
                              const std::string &interpretation,
                              const std::vector<Id> &coordinates) const {
    if(fixed_lookup.empty())return {};
    const auto input_types=fixed_input_types.find(input);
    if(input_types==fixed_input_types.end() || !input_types->second.contains(interpretation))return {};
    FixedInputKey key{input,interpretation,{}};
    key.coordinates.reserve(coordinates.size());
    for(Id coordinate:coordinates) {
      auto value=a.value(coordinate);
      const auto &bounds=a.nodes[coordinate].bounds;
      if(!value && bounds && bounds->first==bounds->second)value=bounds->first;
      if(!value || value->denominator()!=1 || *value<0)return {};
      key.coordinates.push_back(value->numerator());
    }
    auto found=fixed_lookup.find(key);
    if(found==fixed_lookup.end())return {};
    const auto &entry=fixed_inputs[found->second];
    if(!fixed_input_applied[found->second]) {
      fixed_input_applied[found->second]=true;
      applied_fixed_inputs.push_back(found->second);
    }
    ++a.stats["fixed_input_reads_substituted"];
    return a.constant(entry.value);
  }
};
inline void bind_context_inputs(J &mapping,const EquivalenceContext &context) {
  std::set<std::pair<std::string,std::string>> seen;
  for(const auto &binding:context.inputs) {
    if(binding.shared.empty()) throw Unsupported("empty shared input identity");
    for(const auto &[side,raw]:{std::pair("left_inputs",binding.reference),std::pair("right_inputs",binding.optimized)}) {
      if(raw.empty() || !seen.emplace(side,raw).second) throw Unsupported("duplicate or empty input binding");
      auto &table=mapping[side]["allocations"];
      if(table.contains(raw) && table[raw]!=binding.shared) throw Unsupported("conflicting input correspondence");
      table[raw]=binding.shared;
    }
  }
}
} // namespace rg
