#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sentryward::value_ir {

class ValueType;
struct TypeConversion;
struct PointerProperties final {
  std::string identity;
  std::string addressing;
  std::optional<std::uint64_t> owner_cta;
  std::optional<std::uint64_t> owner_thread;
  bool operator==(const PointerProperties&) const = default;
};

class Graph;
class Node;

// A handle is resolved through its owner; it never dereferences stale storage.
class Value final {
 public:
  Value() = default;
  [[nodiscard]] std::uint64_t graphIdentity() const noexcept { return owner_; }
  [[nodiscard]] std::size_t nodeId() const noexcept { return node_id_; }
  [[nodiscard]] std::size_t resultIndex() const noexcept { return result_index_; }
  [[nodiscard]] std::size_t instanceIndex() const noexcept { return instance_index_; }
  [[nodiscard]] std::span<const Value> indices() const noexcept { return indices_; }
  [[nodiscard]] const Value* threadSelector() const noexcept { return thread_selector_.get(); }
  [[nodiscard]] const std::shared_ptr<const PointerProperties>& pointer() const noexcept { return pointer_; }
  [[nodiscard]] const std::shared_ptr<const TypeConversion>& conversion() const noexcept { return conversion_; }
  [[nodiscard]] Value withoutSelection() const {
    auto result = *this;
    result.indices_.clear(); result.thread_selector_.reset();
    return result;
  }
  [[nodiscard]] const ValueType& type(const Graph& owner) const;
  friend bool operator==(const Value&, const Value&);

 private:
  friend class Graph;
  friend class Node;
  friend class CheckpointCodec;
  Value(std::uint64_t owner, std::size_t node, std::size_t result, std::size_t instance = 0)
      : owner_(owner), node_id_(node), result_index_(result), instance_index_(instance) {}
  std::uint64_t owner_ = 0;
  std::size_t node_id_ = 0;
  std::size_t result_index_ = 0;
  std::size_t instance_index_ = 0;
  std::vector<Value> indices_;
  std::shared_ptr<const Value> thread_selector_;
  std::shared_ptr<const PointerProperties> pointer_;
  std::shared_ptr<const TypeConversion> conversion_;
};


}  // namespace sentryward::value_ir
