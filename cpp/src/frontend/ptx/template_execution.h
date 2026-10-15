#pragma once
#include "sentryward/frontend/ptx/execution.h"
#include "sentryward/algorithm/base/shared_array.h"
#include <unordered_map>
#include <algorithm>
#include <map>
#include <unordered_set>

namespace sentryward::frontend::ptx::detail {
struct TemplateFallback {};
struct TemplatePending {};
struct TemplateExchange {
  struct Producer {
    algorithm::SharedArray<std::size_t> members;
    std::size_t value;
  };
  std::vector<Producer> producers;
  std::optional<std::size_t> Source(std::size_t member) const {
    for(const auto& producer:producers)
      if(std::binary_search(producer.members.begin(),producer.members.end(),member)) return producer.value;
    return {};
  }
};
struct TemplateVote {
  std::shared_ptr<TemplateExchange> predicates=std::make_shared<TemplateExchange>();
  std::shared_ptr<TemplateExchange> masks=std::make_shared<TemplateExchange>();
  std::shared_ptr<const std::map<std::uint32_t,std::size_t>> members;
  std::string mode;
  std::function<void(std::uint32_t,const std::string&)> error;
};
struct TemplateValue {
  std::optional<Scalar> uniform;
  std::vector<Scalar> coordinates;
  std::vector<std::size_t> operands;
  std::function<Scalar(const std::vector<Scalar>&)> evaluate;
  std::shared_ptr<TemplateExchange> exchange{};
  std::function<Scalar(std::size_t)> member_value{};
  bool coordinate_only=false;
  std::shared_ptr<TemplateVote> vote{};
};
struct TemplateRecord {
  ExecutedInstruction record;
  std::vector<std::pair<std::uint32_t,std::size_t>> bindings;
};
struct TemplateArena {
  struct ComputationKey {
    std::uint32_t instruction,destination;
    std::vector<std::size_t> operands;
    bool operator==(const ComputationKey&) const=default;
  };
  struct ComputationHash {
    std::size_t operator()(const ComputationKey& key) const {
      std::size_t hash=key.instruction;
      const auto add=[&](std::size_t value){hash^=value+0x9e3779b9+(hash<<6)+(hash>>2);};
      add(key.destination);for(auto operand:key.operands)add(operand);
      return hash;
    }
  };
  std::vector<TemplateValue> values;
  std::vector<TemplateRecord> records;
  std::vector<std::unordered_map<std::size_t, Scalar>> caches;
  std::function<void()> poll;
  std::size_t coordinate_folds=0;
  std::map<Integer,std::size_t> integer_constants;
  std::unordered_map<ComputationKey,std::size_t,ComputationHash> computations;
  std::unordered_set<std::size_t> reused_values;
  std::optional<std::size_t> FindComputation(const ComputationKey& key) {
    const auto found=computations.find(key);
    if(found==computations.end())return {};
    if(reused_values.size()<4096)reused_values.insert(found->second);
    return found->second;
  }
  void RememberComputation(ComputationKey key,std::size_t value) {
    if(computations.size()<100000)computations.emplace(std::move(key),value);
  }
  void FoldCoordinates(TemplateValue& node) {
    if(!node.evaluate || caches.empty())return;
    for(auto operand:node.operands)
      if(!values[operand].uniform && values[operand].coordinates.empty() &&
          !values[operand].member_value && !values[operand].coordinate_only)return;
    node.coordinate_only=true;
    if(coordinate_folds>=256)return;
    ++coordinate_folds;
    std::optional<Scalar> first;
    for(std::size_t thread=0;thread<caches.size();++thread) {
      if((thread&127)==0 && poll)poll();
      try {
        std::vector<Scalar> arguments;arguments.reserve(node.operands.size());
        for(auto operand:node.operands) {
          arguments.push_back(Value(operand,thread));
        }
        auto output=node.evaluate(arguments);
        if(first && output!=*first)return;
        first=std::move(output);
      } catch(const std::exception&) { return; }
    }
    node.uniform=std::move(first);
  }
  const Scalar &Value(std::size_t id, std::size_t thread) {
    if (values[id].uniform) return *values[id].uniform;
    std::vector<std::pair<std::size_t,std::size_t>> pending{{id,thread}};
    std::size_t steps=0;
    while (!pending.empty()) {
      if ((steps++&1023)==0 && poll) poll();
      const auto [current,member]=pending.back();
      auto& cache=caches.at(member);
      if (cache.contains(current)) {pending.pop_back(); continue;}
      const auto &node=values[current];
      if (node.uniform) cache[current]=*node.uniform;
      else if(node.member_value)cache[current]=node.member_value(member);
      else if (!node.coordinates.empty()) cache[current]=node.coordinates.at(member);
      else {
        bool ready=true;
        for (auto operand:node.operands) if (!values[operand].uniform && !cache.contains(operand)) {pending.emplace_back(operand,member); ready=false;}
        if (!ready) continue;
        std::vector<Scalar> inputs; inputs.reserve(node.operands.size());
        for (auto operand:node.operands) inputs.push_back(values[operand].uniform ? *values[operand].uniform : cache.at(operand));
        if(node.vote) {
          const auto actor=std::get<Integer>(inputs.at(0)).convert_to<std::uint32_t>();
          const auto integer=[](const Scalar& value)->std::optional<Integer> {
            if(auto n=std::get_if<Integer>(&value))return *n;
            if(auto b=std::get_if<bool>(&value))return Integer(*b);
            return {};
          };
          auto raw_mask=integer(inputs.at(1));
          if(!raw_mask)cache[current]=Scalar{};
          else {
            const auto mask=(*raw_mask&0xffffffff).convert_to<std::uint32_t>();
            if(!(mask&(1u<<(actor%32))))node.vote->error(actor,"executing lane absent from vote membermask");
            std::uint32_t bits=0;
            bool complete=true,unknown=false;
            for(unsigned lane=0;lane<32;++lane)if(mask&(1u<<lane)) {
              const auto found=node.vote->members->find(actor-actor%32+lane);
              if(found==node.vote->members->end())throw TemplateFallback{};
              const auto peer=found->second;
              auto predicate=node.vote->predicates->Source(peer), peer_mask=node.vote->masks->Source(peer);
              if(!predicate || !peer_mask)throw TemplatePending{};
              const auto lookup=[&](std::size_t value)->const Scalar* {
                if(values[value].uniform)return &*values[value].uniform;
                auto entry=caches[peer].find(value);
                if(entry!=caches[peer].end())return &entry->second;
                pending.emplace_back(value,peer);complete=false;return nullptr;
              };
              const auto mask_value=lookup(*peer_mask),predicate_value=lookup(*predicate);
              if(!mask_value || !predicate_value)continue;
              const auto other=integer(*mask_value);
              if(!other || ((*other&0xffffffff).convert_to<std::uint32_t>()!=mask))
                node.vote->error(actor,"vote participants have different membermasks");
              const auto truth=integer(*predicate_value);
              if(!truth)unknown=true;
              else if(*truth!=0)bits|=1u<<lane;
            }
            if(!complete)continue;
            if(unknown)cache[current]=Scalar{};
            else if(node.vote->mode=="ballot")cache[current]=Integer(bits);
            else cache[current]=node.vote->mode=="all"?bits==mask:
              node.vote->mode=="any"?bits!=0:bits==0 || bits==mask;
          }
        } else if(node.exchange) {
          const auto* selected=std::get_if<Integer>(&inputs.at(0));
          if(!selected)cache[current]=Scalar{};
          else {
            if(*selected<0 || *selected>=caches.size())throw std::out_of_range("template exchange member");
            const auto source_member=selected->convert_to<std::size_t>();
            const auto source=node.exchange->Source(source_member);
            if(!source)throw TemplatePending{};
            if(values[*source].uniform)cache[current]=*values[*source].uniform;
            else if(auto found=caches[source_member].find(*source);found!=caches[source_member].end())cache[current]=found->second;
            else {pending.emplace_back(*source,source_member);continue;}
          }
        } else cache[current]=node.evaluate(inputs);
      }
      pending.pop_back();
    }
    return caches.at(thread).at(id);
  }
  std::size_t Constant(Scalar value) {
    if (std::holds_alternative<std::monostate>(value) && !values.empty()) return 0;
    if(const auto* integer=std::get_if<Integer>(&value)) {
      const auto found=integer_constants.find(*integer);
      if(found!=integer_constants.end())return found->second;
      if(integer_constants.size()<65536)integer_constants.emplace(*integer,values.size());
    }
    values.push_back({std::move(value),{},{},{}}); return values.size()-1;
  }
  std::size_t SharedResult(std::span<const std::size_t> domain,
      std::span<const Scalar> outputs, std::span<const std::uint32_t> actors,
      std::size_t coordinate, std::uint32_t warp_size, Coordinates dimensions) {
    if(domain.empty() || domain.size()!=outputs.size())
      throw std::logic_error("invalid shared result domain");
    if(std::all_of(outputs.begin(),outputs.end(),[&](const auto& value){return value==outputs.front();}))
      return Constant(outputs.front());
    const bool integers=std::all_of(outputs.begin(),outputs.end(),[](const auto& value){return std::holds_alternative<Integer>(value);});
    const auto index=[warp_size,dimensions](std::uint32_t actor,unsigned axis)->std::uint32_t {
      switch(axis) {
        case 0:return actor;
        case 1:return actor%warp_size;
        case 2:return actor/warp_size;
        case 3:return actor%dimensions[0];
        case 4:return actor/dimensions[0]%dimensions[1];
        default:return actor/(dimensions[0]*dimensions[1]);
      }
    };
    if(integers)for(unsigned axis=0;axis<6;++axis) {
      const Integer x0=index(actors[domain.front()],axis);
      const auto& y0=std::get<Integer>(outputs.front());
      std::size_t second=1;
      while(second<domain.size() && Integer(index(actors[domain[second]],axis))==x0)++second;
      if(second==domain.size())continue;
      const Integer dx=Integer(index(actors[domain[second]],axis))-x0;
      const Integer dy=std::get<Integer>(outputs[second])-y0;
      bool exact=true;
      for(std::size_t slot=0;slot<domain.size();++slot) {
        if((slot&127)==0 && poll)poll();
        if((std::get<Integer>(outputs[slot])-y0)*dx !=
            (Integer(index(actors[domain[slot]],axis))-x0)*dy) {exact=false;break;}
      }
      if(!exact)continue;
      TemplateValue node;node.operands={coordinate};node.coordinate_only=true;
      node.evaluate=[index,axis,x0,y0,dx,dy](const std::vector<Scalar>& args)->Scalar {
        const auto actor=std::get<Integer>(args.at(0)).convert_to<std::uint32_t>();
        return Integer(y0+(Integer(index(actor,axis))-x0)*dy/dx);
      };
      const auto id=values.size();values.push_back(std::move(node));return id;
    }
    constexpr std::size_t kMaxRuns=8;
    std::vector<std::pair<std::size_t,Scalar>> runs;
    if(std::is_sorted(domain.begin(),domain.end())) {
      for(std::size_t slot=0;slot<domain.size();++slot) {
        if(runs.empty() || runs.back().second!=outputs[slot]) {
          runs.emplace_back(domain[slot],outputs[slot]);
          if(runs.size()>kMaxRuns)break;
        }
      }
    }
    if(!runs.empty() && runs.size()<=kMaxRuns) {
      TemplateValue node;
      node.member_value=[runs=std::move(runs)](std::size_t member)->Scalar {
        auto end=std::upper_bound(runs.begin(),runs.end(),member,
            [](auto value,const auto& run){return value<run.first;});
        return (end==runs.begin()?end:std::prev(end))->second;
      };
      const auto id=values.size();values.push_back(std::move(node));return id;
    }
    TemplateValue node;node.coordinates.resize(actors.size());
    for(std::size_t slot=0;slot<domain.size();++slot)node.coordinates[domain[slot]]=outputs[slot];
    const auto id=values.size();values.push_back(std::move(node));return id;
  }
  ExecutedInstruction Read(std::size_t id, std::size_t thread,
      std::optional<std::span<const std::uint32_t>> registers = {}) {
    if (poll) poll();
    // Recomputable concrete values are bounded independently of trace length.
    auto& cache=caches.at(thread);
    if (cache.size() > 8192) {
      std::vector<std::size_t> ids;
      ids.reserve(cache.size());
      for(const auto& [id,value]:cache) { (void)value; if(!reused_values.contains(id))ids.push_back(id); }
      if(ids.size()>4096) {
        const auto retained=ids.end()-4096;
        std::nth_element(ids.begin(),retained,ids.end());
        const auto cutoff=*retained;
        std::erase_if(cache,[&](const auto& entry){return !reused_values.contains(entry.first) && entry.first<cutoff;});
      }
    }
    const auto &node = records.at(id);
    auto record = node.record;
    const auto selected=[&](std::uint32_t reg) {
      return !registers || std::find(registers->begin(),registers->end(),reg)!=registers->end();
    };
    if (!node.bindings.empty()) record.mutableKnownRegisters().clear();
    else if(registers)std::erase_if(record.mutableKnownRegisters(),[&](const auto& binding) {
      return !selected(binding.first);
    });
    for (auto [reg, value] : node.bindings) {
      if(!selected(reg))continue;
      const auto &concrete = Value(value, thread);
      if (!std::holds_alternative<std::monostate>(concrete))
        record.mutableKnownRegisters().emplace_back(reg, concrete);
    }
    return record;
  }
  InstructionSequence Expand(std::shared_ptr<TemplateArena> owner,
      std::shared_ptr<const std::vector<std::size_t>> path,std::size_t thread) {
    return InstructionSequence(path->size(),[owner,path,thread] {
      InstructionSequence::Storage result; result.reserve(path->size());
      for (auto id:*path) {
        if ((result.size()&1023)==0 && owner->poll) owner->poll();
        const auto &node=owner->records[id];
        auto record=node.record;
        if (!node.bindings.empty()) record.mutableKnownRegisters().clear();
        for (auto [reg,value]:node.bindings) {
          const auto &concrete=owner->Value(value,thread);
          if (!std::holds_alternative<std::monostate>(concrete)) record.mutableKnownRegisters().emplace_back(reg,concrete);
        }
        result.push_back(std::move(record));
      }
      // A completed expansion owns its concrete values.
      owner->caches[thread].clear();

      return result;
    }, [owner,path](std::size_t index) -> const ExecutedInstruction & {
      return owner->records.at(path->at(index)).record;
    }, [owner,path,thread](std::size_t index) {
      return owner->Read(path->at(index), thread);
    }, path.get(), [owner,path,thread](std::size_t index,std::span<const std::uint32_t> registers) {
      return owner->Read(path->at(index),thread,registers);
    }, [owner,path](std::size_t index,std::uint32_t reg)->std::optional<std::size_t> {
      for(const auto& [candidate,id]:owner->records.at(path->at(index)).bindings)
        if(candidate==reg)return id;
      return {};
    });
  }
};
}
