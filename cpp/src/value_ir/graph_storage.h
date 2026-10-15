#pragma once
#include "sentryward/value_ir/graph.h"

#include <atomic>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <map>
#include <set>
#include <sstream>
#include <typeinfo>
#include <cctype>
#include <iomanip>
#include <tuple>
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/ptx_operation.h"


namespace sentryward::value_ir {
namespace graph_detail {
inline Instance makeInstance(SourceOrigin origin, std::vector<Value> operands, const InstanceTypes& types) {
  return {std::move(origin), std::move(operands), types.results, types.elements};
}

inline std::uint64_t nextIdentity() {
  static std::atomic<std::uint64_t> next{1};
  auto candidate = next.load(std::memory_order_relaxed);
  for (;;) {
    if (candidate == std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("value graph identity space exhausted");
    }
    if (next.compare_exchange_weak(candidate, candidate + 1, std::memory_order_relaxed)) {
      return candidate;
    }
  }
}

inline bool hasOccurrence(const SourceOrigin& origin) {
  return origin.cta && origin.thread && origin.instruction_index && origin.occurrence;
}

inline void addMember(ThreadGroup& group, std::uint64_t thread) {
  auto position = std::lower_bound(group.threads.begin(), group.threads.end(), thread);
  group.threads.insert(position, thread);
}

}
struct Graph::Storage final {
  bool rewriting = false;
  std::vector<SourceOrigin> template_threads;
  ThreadMembers template_instance_members;
  ThreadMembers template_members;
  std::optional<ThreadGroupId> template_cursor;
  std::set<std::uint64_t> sealed_ctas;
  void requireOpen(const SourceOrigin& origin) const {
    if (origin.cta && sealed_ctas.contains(*origin.cta))
      throw std::logic_error("Cannot append an event to a sealed CTA");
  }
  std::deque<InstanceTypes> type_storage;
  std::map<std::pair<std::size_t,std::size_t>,const InstanceTypes*> bound_types;
  std::map<std::string, std::vector<const InstanceTypes*>> static_types;

  const InstanceTypes& typesFor(std::span<const ValueType> types) {
    const bool fixed = std::all_of(types.begin(), types.end(), [](const auto& type) {
      return type.hasStaticShape() && !type.element().isParameterized();
    });
    std::string key;
    if (fixed) {
      for (const auto& type : types) key += type.toString() + ";";
      if (auto found = static_types.find(key); found != static_types.end())
        for (const auto* existing : found->second)
          if (std::ranges::equal(existing->results, types)) return *existing;
    }
    auto& entry = type_storage.emplace_back();
    entry.results.assign(types.begin(), types.end());
    for (const auto& type : types) entry.elements.push_back(ValueType::scalar(type.element()));
    if (fixed) static_types[std::move(key)].push_back(&entry);
    return entry;
  }

  std::map<std::string, std::size_t, std::less<>> source_text_ids;
  std::size_t textId(std::string_view text) {
    const auto found = source_text_ids.find(text);
    if (found != source_text_ids.end()) return found->second;
    return source_text_ids.emplace(text, source_text_ids.size()).first->second;
  }
  std::string occurrenceKey(const SourceOrigin& origin) {
    std::ostringstream key;
    key << *origin.cta << ':' << textId(origin.source_file ? origin.source_file->view() : "") << ':'
        << textId(origin.function ? origin.function->view() : "") << ':'
        << textId(origin.scope ? origin.scope->view() : "") << ':' << *origin.instruction_index << ':' << *origin.occurrence;
    if (origin.executed) key << ":executed=" << *origin.executed;
    for (const auto& [header, iteration] : origin.unroll_context) key << '/' << header << ':' << iteration;
    return key.str();
  }
  std::uint64_t identity = graph_detail::nextIdentity();
  std::deque<std::unique_ptr<Node>> nodes;
  struct TypeAlias {
    ThreadMembers threads;
    std::optional<std::uint64_t> cta;
    struct Entry {
      Value source;
      std::shared_ptr<const TypeConversionSpec> spec;
      bool resolved = false;
    };
    std::size_t count = 0;
    std::map<std::size_t, Entry> entries;
  };
  mutable std::map<std::size_t, TypeAlias> aliases;
  std::size_t live_nodes = 0;
  std::deque<ThreadGroup> groups;
  mutable std::vector<std::pair<std::size_t,std::size_t>> group_intervals;
  void indexGroups() const {
    if(group_intervals.size()==groups.size())return;
    const auto missing=groups.size();
    std::vector<std::size_t> first_child(groups.size(),missing),next_sibling(groups.size(),missing);
    // Parents precede children. Construction validates each split's domain.
    for(std::size_t i=groups.size();i>0;--i)if(const auto parent=groups[i-1].parent) {
      next_sibling[i-1]=first_child.at(*parent);first_child.at(*parent)=i-1;
    }
    group_intervals.resize(groups.size());
    std::vector<std::pair<std::size_t,std::size_t>> stack;
    std::size_t position=0;
    for(const auto& root:groups)if(!root.parent) {
      group_intervals[root.id].first=position++;
      stack.emplace_back(root.id,first_child[root.id]);
      while(!stack.empty()) {
        auto& [id,child]=stack.back();
        if(child==missing) {
          group_intervals[id].second=position;stack.pop_back();
        } else {
          const auto next=child;child=next_sibling[next];
          group_intervals[next].first=position++;
          stack.emplace_back(next,first_child[next]);
        }
      }
    }
  }
  std::map<std::uint64_t, ThreadGroupId> roots;
  std::map<std::pair<std::uint64_t,std::uint64_t>,SourceOrigin> thread_bindings;
  std::map<std::pair<std::uint64_t, std::uint64_t>, ThreadGroupId> cursors;
  std::map<std::pair<ThreadGroupId, std::string>, ThreadGroupId> transitions;
  std::map<ThreadGroupId, std::size_t> shared;
  std::map<std::string, std::vector<ThreadGroupId>> occurrences;
  std::map<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>, std::vector<ThreadGroupId>> execution_occurrences;

  ThreadGroupId advance(const SourceOrigin& origin, std::string key) {
    const auto cta = *origin.cta, thread = *origin.thread;
    if(!template_threads.empty()) {
      if(origin.cta!=template_threads.front().cta || origin.thread!=template_threads.front().thread)
        throw std::invalid_argument("thread template requires its representative origin");
      const auto next=groups.size();
      groups.push_back({next,*template_cursor,cta,template_members});
      template_cursor=next;
      return *template_cursor;
    }
    auto cursor = cursors.find({cta, thread});
    if (cursor == cursors.end()) {
      auto root = roots.find(cta);
      if (root == roots.end()) {
        const auto id = groups.size();
        groups.push_back({id, std::nullopt, cta, {}});
        root = roots.emplace(cta, id).first;
      }
      graph_detail::addMember(groups[root->second], thread);
      cursor = cursors.emplace(std::pair{cta, thread}, root->second).first;
    }
    auto [edge, inserted] = transitions.try_emplace({cursor->second, std::move(key)}, groups.size());
    if (inserted) groups.push_back({edge->second, cursor->second, cta, {}});
    const auto next = edge->second;
    graph_detail::addMember(groups[next], thread);
    cursor->second = next;
    return next;
  }
};
}
