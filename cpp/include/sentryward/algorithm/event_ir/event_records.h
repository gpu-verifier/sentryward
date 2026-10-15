#pragma once
#include "sentryward/algorithm/base/template_records.h"
#include "sentryward/algorithm/event_ir/event.h"
#include <array>
#include <algorithm>
#include <vector>
namespace sentryward::algorithm {
struct EventRecordCodec {
  struct Binding {
    EventId id; InstId origin; ActorId actor; AddressId address;
    AsyncOpId async_id{}; AddressId destination{};
    SharedArray<std::uint32_t> extra_endpoints{};
    friend bool operator==(const Binding&,const Binding&)=default;
  };
  struct Delta {
    std::array<std::int64_t, 6> fields{};
    std::vector<std::int64_t> extra_endpoints;
    Delta() = default;
    Delta(std::initializer_list<std::int64_t> values) {
      if (values.size() != fields.size()) throw std::invalid_argument("event binding delta width");
      std::copy(values.begin(), values.end(), fields.begin());
    }
    std::int64_t operator[](std::size_t i) const { return fields[i]; }
    friend bool operator==(const Delta&, const Delta&) = default;
  };
  static std::optional<Delta> difference(const Binding& a,const Binding& b) {
    Delta delta{std::int64_t(b.id.value())-a.id.value(),
      std::int64_t(b.origin.value())-a.origin.value(),
      std::int64_t(b.actor.value())-a.actor.value(),
      std::int64_t(b.address.value())-a.address.value(),
      std::int64_t(b.async_id.value())-a.async_id.value(),
      std::int64_t(b.destination.value())-a.destination.value()};
    if (a.extra_endpoints.size() != b.extra_endpoints.size()) return {};
    for (std::size_t i = 0; i < a.extra_endpoints.size(); ++i)
      delta.extra_endpoints.push_back(std::int64_t(b.extra_endpoints[i]) - a.extra_endpoints[i]);
    return delta;
  }
  static std::optional<Binding> translate(const Binding& base,const Delta& delta,std::size_t index) {
    std::array<std::uint32_t,6> fields{base.id.value(),base.origin.value(),
                                     base.actor.value(),base.address.value(),
                                     base.async_id.value(),base.destination.value()};
    for(std::size_t i=0;i<fields.size();++i) {
      if(delta[i]>=0) {
        auto step=static_cast<std::uint64_t>(delta[i]);
        if(step && index>(UINT32_MAX-fields[i])/step)return {};
        fields[i]+=static_cast<std::uint32_t>(step*index);
      } else {
        auto step=static_cast<std::uint64_t>(-(delta[i]+1))+1;
        if(index>fields[i]/step)return {};
        fields[i]-=static_cast<std::uint32_t>(step*index);
      }
    }
    auto extra = base.extra_endpoints;
    if (!delta.extra_endpoints.empty()) {
      if (delta.extra_endpoints.size() != extra.size()) return {};
      for (std::size_t i = 0; i < extra.size(); ++i) {
        const auto step = delta.extra_endpoints[i];
        if (!step || !index) continue;
        const auto magnitude = step > 0 ? std::uint64_t(step) : std::uint64_t(-(step + 1)) + 1;
        if (index > (step > 0 ? UINT32_MAX - extra[i] : extra[i]) / magnitude) return {};
      }
      if (index && std::any_of(delta.extra_endpoints.begin(), delta.extra_endpoints.end(),
                              [](auto step) { return step != 0; }))
        extra = extra.transformed([&](std::size_t i, std::uint32_t value) {
          return static_cast<std::uint32_t>(value + delta.extra_endpoints[i] * std::int64_t(index));
        });
    }
    return Binding{EventId(fields[0]),InstId(fields[1]),ActorId(fields[2]),AddressId(fields[3]),
        AsyncOpId(fields[4]),AddressId(fields[5]), std::move(extra)};
  }
  static bool split(Event& event,Binding& binding){
    auto* issue=std::get_if<ImmediateMemoryIssueEvent>(&event.payload);
    auto* exec=std::get_if<ImmediateMemoryExecEvent>(&event.payload);
    binding={event.id,event.origin.value_or(InstId{}),event.actor.value_or(ActorId{}),issue?issue->address:exec?exec->address:AddressId{}};
    std::visit([&](auto& payload) {
      if constexpr(requires { payload.async_id; }) {
        binding.async_id=payload.async_id;payload.async_id=AsyncOpId{};
      }
      if constexpr(requires { payload.source; payload.destination; }) {
        binding.address=payload.source;binding.destination=payload.destination;
        payload.source=AddressId{};payload.destination=AddressId{};
      }
    },event.payload);
    event.id=EventId{};
    if (auto* effect = std::get_if<GenericAsyncEffectEvent>(&event.payload)) {
      std::vector<std::uint32_t> extra;
      effect->accesses = effect->accesses.transformed([&](std::size_t i, MemoryEffect access) {
        std::uint32_t endpoint;
        if (auto* address = std::get_if<AddressId>(&access.endpoint)) {
          endpoint = address->value();
          *address = AddressId{};
        } else {
          auto& reg = std::get<RegisterAddress>(access.endpoint);
          endpoint = reg.actor.value();
          reg.actor = ActorId{};
        }
        if (i < 2) (i == 0 ? binding.address : binding.destination) = AddressId(endpoint);
        else extra.push_back(endpoint);
        return access;
      });
      binding.extra_endpoints = std::move(extra);
    }
    if(event.origin)event.origin=InstId{};
    if(event.actor)event.actor=ActorId{};
    if(issue)issue->address=AddressId{};else if(exec)exec->address=AddressId{};
    return true;
  }
  static void bind(Event& event,const Binding& binding){
    event.id=binding.id;
    if(event.origin)event.origin=binding.origin;
    if(event.actor)event.actor=binding.actor;
    std::visit([&](auto& payload) {
      if constexpr(requires { payload.async_id; })payload.async_id=binding.async_id;
      if constexpr(requires { payload.source; payload.destination; }) {
        payload.source=binding.address;payload.destination=binding.destination;
      }
    },event.payload);
    if(auto* issue=std::get_if<ImmediateMemoryIssueEvent>(&event.payload))issue->address=binding.address;
    else if(auto* exec=std::get_if<ImmediateMemoryExecEvent>(&event.payload))exec->address=binding.address;
    else if (auto* effect = std::get_if<GenericAsyncEffectEvent>(&event.payload)) {
      effect->accesses = effect->accesses.transformed([&](std::size_t i, MemoryEffect access) {
        const auto endpoint = i < 2 ? (i == 0 ? binding.address : binding.destination).value()
                                    : binding.extra_endpoints.at(i - 2);
        if (auto* address = std::get_if<AddressId>(&access.endpoint)) {
          *address = AddressId(endpoint);
        } else std::get<RegisterAddress>(access.endpoint).actor = ActorId(endpoint);
        return access;
      });
    }
  }
  static std::size_t hash(const Event& event){
    auto h=event.payload.index();HashCombine(h,static_cast<unsigned>(event.scheduling_class));
    HashCombine(h,event.origin.has_value());HashCombine(h,event.actor.has_value());
    std::visit([&](const auto& payload) {
      if constexpr(requires { payload.effect_index; })HashCombine(h,payload.effect_index);
      if constexpr(requires { payload.detached_accesses; })HashCombine(h,payload.detached_accesses);
      if constexpr(requires { payload.checks_issue_initialization; })HashCombine(h,payload.checks_issue_initialization);
      if constexpr(requires { payload.detached_access_count; })HashCombine(h,payload.detached_access_count);
      if constexpr(requires { payload.detached_transfer; })HashCombine(h,payload.detached_transfer);
      if constexpr(requires { payload.barriers; })
        for(auto barrier:payload.barriers)HashCombine(h,barrier.value());
      if constexpr(requires { payload.accesses; })for(const auto& access:payload.accesses) {
        HashCombine(h,static_cast<unsigned>(access.mode));
        HashCombine(h,access.endpoint.index());
        std::visit([&](const auto& endpoint) {
          if constexpr(requires { endpoint.actor; }) {
            HashCombine(h,endpoint.actor.value());HashCombine(h,std::hash<std::string>{}(endpoint.name));
          } else HashCombine(h,endpoint.value());
        },access.endpoint);
        HashCombine(h,access.bytes.has_value());
        if(access.bytes)HashCombine(h,*access.bytes);
        HashCombine(h,access.atomic_scope.has_value());
        if(access.atomic_scope)HashCombine(h,static_cast<unsigned>(*access.atomic_scope));
      }
      if constexpr(requires { payload.completion; }) {
        HashCombine(h,payload.completion.has_value());
        if(payload.completion) {
          const auto effect=[&](const auto& action) {
            HashCombine(h,action.barrier.value());
            if constexpr(requires { action.bytes; })HashCombine(h,action.bytes);
            else HashCombine(h,action.arrivals);
          };
          if constexpr(requires { payload.completion->index(); }) {
            HashCombine(h,payload.completion->index());std::visit(effect,*payload.completion);
          } else effect(*payload.completion);
        }
      }
      if constexpr(requires { payload.direction; })HashCombine(h,static_cast<unsigned>(payload.direction));
      if constexpr(requires { payload.bytes; })HashCombine(h,payload.bytes);
      if constexpr(requires { payload.barrier; })HashCombine(h,payload.barrier.value());
      if constexpr(requires { payload.state; })HashCombine(h,payload.state.value());
      if constexpr(requires { payload.sync_id; })HashCombine(h,payload.sync_id.value());
      if constexpr(requires { payload.kind; })HashCombine(h,static_cast<unsigned>(payload.kind));
      if constexpr(requires { payload.expected_arrivals; })HashCombine(h,payload.expected_arrivals);
      if constexpr(requires { payload.arrival_decrement; })HashCombine(h,payload.arrival_decrement);
      if constexpr(requires { payload.transaction_increment; })HashCombine(h,payload.transaction_increment);
      if constexpr(requires { payload.arrivals; })HashCombine(h,payload.arrivals);
      if constexpr(requires { payload.phase; })HashCombine(h,payload.phase);
      if constexpr(requires { payload.parity; })HashCombine(h,payload.parity);
      if constexpr(requires { payload.blocking; })HashCombine(h,payload.blocking);
      if constexpr(requires { payload.produced_state; }) {
        HashCombine(h,payload.produced_state.has_value());
        if(payload.produced_state)HashCombine(h,payload.produced_state->value());
      }
    },event.payload);
    return h;
  }
};
using EventRecords=TemplateRecords<Event,EventRecordCodec>;
} // namespace sentryward::algorithm
