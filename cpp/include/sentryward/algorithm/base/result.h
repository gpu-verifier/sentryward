#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace sentryward::algorithm {

template <typename T, typename E>
class Result final {
 public:
  [[nodiscard]] static Result Success(T value) {
    return Result(Value{std::move(value)});
  }

  [[nodiscard]] static Result Failure(E error) {
    return Result(Error{std::move(error)});
  }

  [[nodiscard]] bool has_value() const noexcept {
    return std::holds_alternative<Value>(storage_);
  }

  [[nodiscard]] explicit operator bool() const noexcept { return has_value(); }

  [[nodiscard]] T& value() & {
    requireValue();
    return std::get<Value>(storage_).value;
  }

  [[nodiscard]] const T& value() const& {
    requireValue();
    return std::get<Value>(storage_).value;
  }

  [[nodiscard]] T&& value() && {
    requireValue();
    return std::move(std::get<Value>(storage_).value);
  }

  [[nodiscard]] E& error() & {
    requireError();
    return std::get<Error>(storage_).error;
  }

  [[nodiscard]] const E& error() const& {
    requireError();
    return std::get<Error>(storage_).error;
  }

  [[nodiscard]] E&& error() && {
    requireError();
    return std::move(std::get<Error>(storage_).error);
  }

 private:
  struct Value final {
    T value;
  };

  struct Error final {
    E error;
  };

  explicit Result(Value value) : storage_(std::move(value)) {}
  explicit Result(Error error) : storage_(std::move(error)) {}

  void requireValue() const {
    if (!has_value()) {
      throw std::logic_error("Result does not contain a value");
    }
  }

  void requireError() const {
    if (has_value()) {
      throw std::logic_error("Result does not contain an error");
    }
  }

  std::variant<Value, Error> storage_;
};

template <typename E>
class Result<void, E> final {
 public:
  [[nodiscard]] static Result Success() { return Result(Value{}); }

  [[nodiscard]] static Result Failure(E error) {
    return Result(Error{std::move(error)});
  }

  [[nodiscard]] bool has_value() const noexcept {
    return std::holds_alternative<Value>(storage_);
  }

  [[nodiscard]] explicit operator bool() const noexcept { return has_value(); }

  void value() const {
    if (!has_value()) {
      throw std::logic_error("Result does not contain a value");
    }
  }

  [[nodiscard]] E& error() & {
    requireError();
    return std::get<Error>(storage_).error;
  }

  [[nodiscard]] const E& error() const& {
    requireError();
    return std::get<Error>(storage_).error;
  }

  [[nodiscard]] E&& error() && {
    requireError();
    return std::move(std::get<Error>(storage_).error);
  }

 private:
  struct Value final {};

  struct Error final {
    E error;
  };

  explicit Result(Value value) : storage_(value) {}
  explicit Result(Error error) : storage_(std::move(error)) {}

  void requireError() const {
    if (has_value()) {
      throw std::logic_error("Result does not contain an error");
    }
  }

  std::variant<Value, Error> storage_;
};

}  // namespace sentryward::algorithm
