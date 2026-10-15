#pragma once

#include <compare>
#include <memory>
#include <optional>
#include <utility>

namespace sentryward::algorithm {

// An optional immutable value. Copies share storage; comparisons compare the
// contents, never the allocation address. An absent value differs from a
// present empty container. There is deliberately no mutable dereference.
template <class T>
class SharedOptional final {
 public:
  SharedOptional() = default;
  SharedOptional(std::nullopt_t) noexcept {}
  SharedOptional(T value) : value_(std::make_shared<const T>(std::move(value))) {}

  bool has_value() const noexcept { return static_cast<bool>(value_); }
  explicit operator bool() const noexcept { return has_value(); }
  const T& operator*() const noexcept { return *value_; }
  const T* operator->() const noexcept { return value_.get(); }
  const void* storage_identity() const noexcept { return value_.get(); }

  friend bool operator==(const SharedOptional& a, const SharedOptional& b) {
    return a.value_ == b.value_ ||
        (a.value_ && b.value_ && *a.value_ == *b.value_);
  }
  friend auto operator<=>(const SharedOptional& a, const SharedOptional& b) {
    if constexpr (std::three_way_comparable<T>) {
      using Result = decltype(std::declval<const T&>() <=> std::declval<const T&>());
      if (a.value_ == b.value_) return Result::equivalent;
      if (!a.value_) return Result::less;
      if (!b.value_) return Result::greater;
      return *a.value_ <=> *b.value_;
    } else {
      // Standard libraries predating container operator<=> (Apple clang 15)
      // still order containers with operator<.
      using Result = std::weak_ordering;
      if (a.value_ == b.value_) return Result::equivalent;
      if (!a.value_) return Result::less;
      if (!b.value_) return Result::greater;
      if (*a.value_ < *b.value_) return Result::less;
      if (*b.value_ < *a.value_) return Result::greater;
      return Result::equivalent;
    }
  }

 private:
  std::shared_ptr<const T> value_;
};

}  // namespace sentryward::algorithm
