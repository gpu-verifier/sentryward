#include "sentryward/frontend/ptx/execution.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace sentryward::frontend::ptx {
namespace {
std::shared_ptr<ExecutionMetadata> EmptyMetadata() {
  static auto value = std::make_shared<ExecutionMetadata>();
  return value;
}
std::shared_ptr<ExecutionBindings> EmptyBindings() {
  static auto value = std::make_shared<ExecutionBindings>();
  return value;
}
class RecordHash {
public:
  std::size_t value = 0;
  void Mix(std::size_t part) {
    value ^= part + std::size_t{0x9e3779b9} + (value << 6) + (value >> 2);
  }
  template <typename T> void Add(const T &part) { Mix(std::hash<T>{}(part)); }
  void Add(const Integer &part) {
    const auto &backend = part.backend();
    Add(backend.sign());
    Add(backend.size());
    for (unsigned i = 0; i < backend.size(); ++i) Add(backend.limbs()[i]);
  }
  template <typename T> void Add(const std::optional<T> &part) {
    Add(part.has_value());
    if (part) Add(*part);
  }
  template <typename A, typename B> void Add(const std::pair<A, B> &part) {
    Add(part.first);
    Add(part.second);
  }
  template <typename T> void Add(const std::vector<T> &part) {
    Add(part.size());
    for (const auto &item : part) Add(item);
  }
  void Add(const DynamicKey &part) {
    Add(part.instruction);
    Add(part.occurrence);
  }
  void Add(const ParameterPointer &part) {
    Add(part.region);
    Add(part.offset);
  }
  void Add(const AllocationPointer &part) {
    Add(part.resource);
    Add(part.offset);
    Add(part.extent_bytes);
  }
  void Add(const ClusterSharedPointer &part) {
    Add(part.local_address);
    Add(part.target_rank);
  }
  void Add(const PointerBits &part) {
    Add(bool(part.pointer));
    if (part.pointer) {
      Add(part.pointer->index());
      std::visit([&](const auto &item) { Add(item); }, *part.pointer);
    }
    Add(part.source_bit);
    Add(part.width);
    Add(part.destination_bit);
  }
  void Add(const DescriptorFragment &part) {
    Add(part.descriptor);
    Add(part.offset);
    Add(part.bytes);
  }
  void Add(std::monostate) { Mix(0); }
  void Add(const Scalar &part) {
    Add(part.index());
    std::visit([&](const auto &item) { Add(item); }, part);
  }
  void Add(const ExecutionMetadata &part) {
    Add(part.instruction_id);
    Add(part.occurrence);
    Add(part.predicate_value);
    Add(part.branch_scope_decision.has_value());
    if (part.branch_scope_decision) {
      Add(part.branch_scope_decision->taken);
      Add(part.branch_scope_decision->visit);
      Add(part.branch_scope_decision->user_decision);
    }
    Add(part.register_origins);
    Add(part.collapsed_wait_loop);
    Add(part.canonical_wait_probe);
    Add(part.normalized_wait_result);
    Add(part.normalized_clc_result);
    Add(part.wait_exit_label);
    Add(part.schedule_seed_outcome);
    Add(part.probe_fallback_key);
    Add(part.opaque_compute_diamond_indices);
    Add(part.collapsed_wait_indices);
    Add(part.opaque_compute_join_label);
    Add(part.assumed_equality);
    Add(part.opaque_compute_register_reads);
    Add(part.opaque_compute_register_writes);
  }
};
bool EqualScalar(const Scalar &left, const Scalar &right) {
  if (left.index() != right.index()) return false;
  if (const auto *bits = std::get_if<PointerBits>(&left)) {
    const auto &other = std::get<PointerBits>(right);
    const bool same_pointer = bits->pointer && other.pointer
        ? *bits->pointer == *other.pointer : bits->pointer == other.pointer;
    return same_pointer && bits->source_bit == other.source_bit &&
        bits->width == other.width && bits->destination_bit == other.destination_bit;
  }
  return left == right;
}
bool EqualBindings(const ExecutionBindings &left, const ExecutionBindings &right) {
  return left.size() == right.size() &&
      std::equal(left.begin(), left.end(), right.begin(), [](const auto &a, const auto &b) {
        return a.first == b.first && EqualScalar(a.second, b.second);
      });
}
std::uint64_t StringStorage(const std::string &value) { return value.capacity() + 1; }
std::uint64_t IntegerStorage(const Integer &value) {
  return value.backend().size() * sizeof(*value.backend().limbs());
}
std::uint64_t PointerStorage(const PointerValue &value) {
  return std::visit([](const auto &pointer) -> std::uint64_t {
    using T = std::decay_t<decltype(pointer)>;
    if constexpr (std::is_same_v<T, ParameterPointer>)
      return StringStorage(pointer.region) + IntegerStorage(pointer.offset);
    else if constexpr (std::is_same_v<T, AllocationPointer>)
      return StringStorage(pointer.resource) + IntegerStorage(pointer.offset);
    else
      return IntegerStorage(pointer.local_address);
  }, value);
}
std::uint64_t ScalarStorage(const Scalar &value) {
  return std::visit([](const auto &part) -> std::uint64_t {
    using T = std::decay_t<decltype(part)>;
    if constexpr (std::is_same_v<T, Integer>) return IntegerStorage(part);
    else if constexpr (std::is_same_v<T, ParameterPointer>)
      return StringStorage(part.region) + IntegerStorage(part.offset);
    else if constexpr (std::is_same_v<T, AllocationPointer>)
      return StringStorage(part.resource) + IntegerStorage(part.offset);
    else if constexpr (std::is_same_v<T, ClusterSharedPointer>)
      return IntegerStorage(part.local_address);
    else if constexpr (std::is_same_v<T, PointerBits>)
      return part.pointer ? sizeof(PointerValue) + PointerStorage(*part.pointer) : 0;
    else if constexpr (std::is_same_v<T, DescriptorFragment>) return StringStorage(part.descriptor);
    else return 0;
  }, value);
}
std::uint64_t MetadataStorage(const ExecutionMetadata &value) {
  std::uint64_t bytes = sizeof(value) + 2 * sizeof(void *) +
      value.register_origins.capacity() * sizeof(value.register_origins.front()) +
      value.opaque_compute_diamond_indices.capacity() * sizeof(std::uint32_t) +
      value.collapsed_wait_indices.capacity() * sizeof(std::uint32_t);
  for (const auto *field : {&value.wait_exit_label, &value.schedule_seed_outcome,
                             &value.opaque_compute_join_label})
    if (*field) bytes += StringStorage(**field);
  if (value.assumed_equality)
    bytes += StringStorage(value.assumed_equality->first) +
             IntegerStorage(value.assumed_equality->second);
  for (const auto *fields : {&value.opaque_compute_register_reads,
                              &value.opaque_compute_register_writes}) {
    bytes += fields->capacity() * sizeof(std::string);
    for (const auto &field : *fields) bytes += StringStorage(field);
  }
  return bytes;
}
std::uint64_t BindingsStorage(const ExecutionBindings &value) {
  std::uint64_t bytes = sizeof(value) + 2 * sizeof(void *) +
      value.capacity() * sizeof(value.front());
  for (const auto &[reg, scalar] : value) {
    (void)reg;
    bytes += ScalarStorage(scalar);
  }
  return bytes;
}
} // namespace

ExecutedInstruction::ExecutedInstruction()
    : metadata_(EmptyMetadata()), bindings_(EmptyBindings()) {}
ExecutionMetadata &ExecutedInstruction::mutableMetadata() {
  if (metadata_.use_count() != 1) metadata_ = std::make_shared<ExecutionMetadata>(*metadata_);
  return *metadata_;
}
ExecutionBindings &ExecutedInstruction::mutableKnownRegisters() {
  if (bindings_.use_count() != 1) bindings_ = std::make_shared<ExecutionBindings>(*bindings_);
  return *bindings_;
}

struct ThreadRecordPool::Impl {
  struct Binding {
    std::size_t hash;
    std::shared_ptr<ExecutionBindings> values;
  };
  struct Template {
    std::uint64_t parent;
    std::shared_ptr<ExecutionMetadata> metadata;
    std::vector<Binding> bindings;
  };
  std::vector<Template> templates;
  std::unordered_multimap<std::size_t, std::uint64_t> index;
  ThreadRecordStatistics statistics;
  bool has_empty_binding = false;
};

ThreadRecordPool::ThreadRecordPool() : impl_(std::make_unique<Impl>()) {}
ThreadRecordPool::~ThreadRecordPool() = default;
ThreadRecordPool::ThreadRecordPool(ThreadRecordPool &&) noexcept = default;
ThreadRecordPool &ThreadRecordPool::operator=(ThreadRecordPool &&) noexcept = default;

void ThreadRecordPool::Intern(ExecutedInstruction &record, std::uint64_t &cursor) {
  if (cursor > impl_->templates.size()) throw std::invalid_argument("invalid thread prefix");
  ++impl_->statistics.record_count;
  RecordHash key;
  key.Add(cursor);
  key.Add(*record.metadata_);
  const auto [begin, end] = impl_->index.equal_range(key.value);
  std::uint64_t found = 0;
  for (auto it = begin; it != end; ++it) {
    const auto &candidate = impl_->templates[it->second - 1];
    if (candidate.parent == cursor && *candidate.metadata == *record.metadata_) {
      found = it->second;
      break;
    }
  }
  if (!found) {
    impl_->templates.push_back({cursor, record.metadata_, {}});
    found = impl_->templates.size();
    impl_->index.emplace(key.value, found);
    ++impl_->statistics.template_count;
    impl_->statistics.estimated_owned_bytes += MetadataStorage(*record.metadata_);
  }
  auto &node = impl_->templates[found - 1];
  record.metadata_ = node.metadata;
  cursor = found;
  if (record.bindings_->empty()) {
    record.bindings_ = EmptyBindings();
    if (!impl_->has_empty_binding) {
      impl_->has_empty_binding = true;
      ++impl_->statistics.unique_binding_count;
      impl_->statistics.estimated_owned_bytes += BindingsStorage(*record.bindings_);
    }
    return;
  }
  RecordHash bindings_key;
  bindings_key.Add(*record.bindings_);
  for (const auto &binding : node.bindings)
    if (binding.hash == bindings_key.value &&
        EqualBindings(*binding.values, *record.bindings_)) {
      record.bindings_ = binding.values;
      return;
    }
  const auto previous_capacity = node.bindings.capacity();
  node.bindings.push_back({bindings_key.value, record.bindings_});
  ++impl_->statistics.unique_binding_count;
  impl_->statistics.estimated_owned_bytes += BindingsStorage(*record.bindings_) +
      (node.bindings.capacity() - previous_capacity) * sizeof(Impl::Binding);
}
ThreadRecordStatistics ThreadRecordPool::Statistics() const {
  auto result = impl_->statistics;
  result.estimated_owned_bytes += sizeof(Impl) +
      impl_->templates.capacity() * sizeof(Impl::Template) +
      impl_->index.bucket_count() * sizeof(void *) +
      impl_->index.size() * (sizeof(std::pair<const std::size_t, std::uint64_t>) + 2 * sizeof(void *));
  return result;
}
} // namespace sentryward::frontend::ptx
