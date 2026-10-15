#pragma once

#include <type_traits>
#include <utility>
#include <vector>

namespace sentryward::algorithm {

// Rebuild a finished vector with exactly the live storage. Unlike
// shrink_to_fit, this does not leave retention of a large deleted tail up to
// the standard library. Existing element references are invalidated.
template <class T>
void CompactVector(std::vector<T>& values) {
  static_assert(std::is_nothrow_move_constructible_v<T> ||
                std::is_copy_constructible_v<T>);
  if (values.capacity() == values.size()) return;
  std::vector<T> compact;
  compact.reserve(values.size());
  for (auto& value : values) {
    compact.emplace_back(std::move_if_noexcept(value));
  }
  compact.swap(values);
}

}  // namespace sentryward::algorithm
