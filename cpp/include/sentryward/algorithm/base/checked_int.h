#pragma once

#include <concepts>
#include <limits>
#include <type_traits>
#include <utility>

#include "sentryward/algorithm/base/result.h"

namespace sentryward::algorithm {

enum class CheckedArithmeticError {
  kOverflow,
  kUnderflow,
  kOutOfRange,
};

template <std::integral T>
  requires(!std::same_as<std::remove_cv_t<T>, bool>)
[[nodiscard]] constexpr Result<T, CheckedArithmeticError> CheckedAdd(T left,
                                                                     T right) {
  if constexpr (std::is_unsigned_v<T>) {
    if (right > std::numeric_limits<T>::max() - left) {
      return Result<T, CheckedArithmeticError>::Failure(
          CheckedArithmeticError::kOverflow);
    }
  } else {
    if (right > 0 && left > std::numeric_limits<T>::max() - right) {
      return Result<T, CheckedArithmeticError>::Failure(
          CheckedArithmeticError::kOverflow);
    }
    if (right < 0 && left < std::numeric_limits<T>::min() - right) {
      return Result<T, CheckedArithmeticError>::Failure(
          CheckedArithmeticError::kUnderflow);
    }
  }
  return Result<T, CheckedArithmeticError>::Success(
      static_cast<T>(left + right));
}

template <std::integral T>
  requires(!std::same_as<std::remove_cv_t<T>, bool>)
[[nodiscard]] constexpr Result<T, CheckedArithmeticError> CheckedSubtract(
    T left, T right) {
  if constexpr (std::is_unsigned_v<T>) {
    if (left < right) {
      return Result<T, CheckedArithmeticError>::Failure(
          CheckedArithmeticError::kUnderflow);
    }
  } else {
    if (right > 0 && left < std::numeric_limits<T>::min() + right) {
      return Result<T, CheckedArithmeticError>::Failure(
          CheckedArithmeticError::kUnderflow);
    }
    if (right < 0 && left > std::numeric_limits<T>::max() + right) {
      return Result<T, CheckedArithmeticError>::Failure(
          CheckedArithmeticError::kOverflow);
    }
  }
  return Result<T, CheckedArithmeticError>::Success(
      static_cast<T>(left - right));
}

template <std::integral To, std::integral From>
  requires(!std::same_as<std::remove_cv_t<To>, bool> &&
           !std::same_as<std::remove_cv_t<From>, bool>)
[[nodiscard]] constexpr Result<To, CheckedArithmeticError> CheckedCast(
    From value) {
  if (!std::in_range<To>(value)) {
    return Result<To, CheckedArithmeticError>::Failure(
        CheckedArithmeticError::kOutOfRange);
  }
  return Result<To, CheckedArithmeticError>::Success(static_cast<To>(value));
}

}  // namespace sentryward::algorithm
