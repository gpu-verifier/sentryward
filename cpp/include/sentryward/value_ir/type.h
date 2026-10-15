#pragma once

#include <cstdint>
#include <optional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include "sentryward/value_ir/value.h"

namespace sentryward::value_ir {

struct DescriptorQuery final {
  Value descriptor;
  std::string interpretation;
  [[nodiscard]] std::string toString(std::string_view query) const;
  bool operator==(const DescriptorQuery&) const = default;
};

enum class ElementKind { Bit, UnsignedInteger, SignedInteger, Floating, Predicate, Parameterized };

class ElementType final {
 public:
  static ElementType bit(std::uint32_t width);
  static ElementType unsignedInteger(std::uint32_t width);
  static ElementType signedInteger(std::uint32_t width);
  static ElementType floating(std::uint32_t width, std::string precise_type = {});
  static ElementType predicate();
  static ElementType fromDescriptor(Value descriptor, std::string interpretation);

  [[nodiscard]] ElementKind kind() const noexcept { return kind_; }
  [[nodiscard]] std::uint32_t width() const;
  [[nodiscard]] bool isParameterized() const noexcept { return bool(query_); }
  [[nodiscard]] const std::shared_ptr<const DescriptorQuery>& query() const noexcept { return query_; }
  [[nodiscard]] const std::string& preciseType() const noexcept { return precise_type_; }
  [[nodiscard]] bool isInteger() const noexcept;
  [[nodiscard]] bool isNumeric() const noexcept;
  [[nodiscard]] bool isBitLike() const noexcept;
  [[nodiscard]] std::string toString() const;
  friend bool operator==(const ElementType&, const ElementType&);

 private:
  ElementType(ElementKind kind, std::uint32_t width, std::string precise_type);
  ElementKind kind_;
  std::uint32_t width_;
  std::string precise_type_;
  std::shared_ptr<const DescriptorQuery> query_;
};

class ValueType final {
 public:
  static ValueType scalar(ElementType element);
  static ValueType matrix(ElementType element, std::vector<std::uint64_t> shape);
  static ValueType parameterizedMatrix(ElementType element, Value descriptor, std::string interpretation);

  [[nodiscard]] const ElementType& element() const noexcept { return element_; }
  [[nodiscard]] bool isScalar() const noexcept { return shape_.empty() && !shape_query_; }
  [[nodiscard]] bool isMatrix() const noexcept { return !isScalar(); }
  [[nodiscard]] bool hasStaticShape() const noexcept { return !shape_query_; }
  [[nodiscard]] const std::shared_ptr<const DescriptorQuery>& shapeQuery() const noexcept { return shape_query_; }
  [[nodiscard]] std::span<const std::uint64_t> shape() const;
  [[nodiscard]] std::uint64_t elementCount() const;
  [[nodiscard]] std::uint64_t bitSize() const;
  [[nodiscard]] std::string toString() const;
  friend bool operator==(const ValueType&, const ValueType&);

 private:
  ValueType(ElementType element, std::vector<std::uint64_t> shape);
  ElementType element_;
  std::vector<std::uint64_t> shape_;
  std::shared_ptr<const DescriptorQuery> shape_query_;
};

[[nodiscard]] bool sameShape(const ValueType& left, const ValueType& right) noexcept;

}  // namespace sentryward::value_ir
