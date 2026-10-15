#include "sentryward/value_ir/type.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <sstream>
#include <functional>

namespace sentryward::value_ir {

namespace {
bool sameQuery(const std::shared_ptr<const DescriptorQuery>& left,
               const std::shared_ptr<const DescriptorQuery>& right) {
  return left == right || (left && right && *left == *right);
}
}
bool operator==(const ElementType& left, const ElementType& right) {
  return left.kind_ == right.kind_ && left.width_ == right.width_ &&
      left.precise_type_ == right.precise_type_ && sameQuery(left.query_, right.query_);
}
bool operator==(const ValueType& left, const ValueType& right) {
  return left.element_ == right.element_ && left.shape_ == right.shape_ &&
      sameQuery(left.shape_query_, right.shape_query_);
}

std::string DescriptorQuery::toString(std::string_view query) const {
  std::ostringstream out;
  const std::function<void(const Value&)> reference = [&](const Value& value) {
    out << '%' << value.nodeId() << ".r" << value.resultIndex() << "@instance" << value.instanceIndex();
    for (const auto& index : value.indices()) { out << '['; reference(index); out << ']'; }
    if (value.threadSelector()) { out << "@thread("; reference(*value.threadSelector()); out << ')'; }
  };
  out << query << '(';
  reference(descriptor);
  out << ", " << interpretation << ')';
  return out.str();
}

ElementType ElementType::fromDescriptor(Value descriptor, std::string interpretation) {
  if (interpretation.empty()) throw std::invalid_argument("descriptor interpretation must not be empty");
  ElementType result(ElementKind::Parameterized, 0, {});
  result.query_ = std::make_shared<const DescriptorQuery>(DescriptorQuery{std::move(descriptor), std::move(interpretation)});
  return result;
}
std::uint32_t ElementType::width() const {
  if (isParameterized()) throw std::logic_error("element width depends on a descriptor");
  return width_;
}

ElementType::ElementType(ElementKind kind, std::uint32_t width, std::string precise_type)
    : kind_(kind), width_(width), precise_type_(std::move(precise_type)) {
  if (width == 0 && kind != ElementKind::Parameterized) throw std::invalid_argument("element width must be positive");
}

ElementType ElementType::bit(std::uint32_t width) { return {ElementKind::Bit, width, {}}; }
ElementType ElementType::unsignedInteger(std::uint32_t width) {
  return {ElementKind::UnsignedInteger, width, {}};
}
ElementType ElementType::signedInteger(std::uint32_t width) {
  return {ElementKind::SignedInteger, width, {}};
}
ElementType ElementType::floating(std::uint32_t width, std::string precise_type) {
  if (precise_type.empty()) precise_type = "f" + std::to_string(width);
  if (std::any_of(precise_type.begin(), precise_type.end(), [](unsigned char c) {
        return c <= 32 || c >= 127;
      })) {
    throw std::invalid_argument("floating type spelling must be a nonempty printable token");
  }
  return {ElementKind::Floating, width, std::move(precise_type)};
}
ElementType ElementType::predicate() { return {ElementKind::Predicate, 1, {}}; }
bool ElementType::isInteger() const noexcept {
  return kind_ == ElementKind::UnsignedInteger || kind_ == ElementKind::SignedInteger;
}
bool ElementType::isNumeric() const noexcept { return isInteger() || kind_ == ElementKind::Floating; }
bool ElementType::isBitLike() const noexcept {
  return isInteger() || kind_ == ElementKind::Bit || kind_ == ElementKind::Predicate;
}
std::string ElementType::toString() const {
  switch (kind_) {
    case ElementKind::Bit: return "b" + std::to_string(width_);
    case ElementKind::UnsignedInteger: return "u" + std::to_string(width_);
    case ElementKind::SignedInteger: return "s" + std::to_string(width_);
    case ElementKind::Floating: return precise_type_;
    case ElementKind::Predicate: return "pred";
    case ElementKind::Parameterized: return query_->toString("ElementTypeOf");
  }
  throw std::logic_error("invalid element kind");
}

ValueType::ValueType(ElementType element, std::vector<std::uint64_t> shape)
    : element_(std::move(element)), shape_(std::move(shape)) {
  std::uint64_t count = 1;
  for (const auto extent : shape_) {
    if (extent == 0) throw std::invalid_argument("matrix extents must be positive");
    if (count > std::numeric_limits<std::uint64_t>::max() / extent) {
      throw std::overflow_error("matrix element count overflow");
    }
    count *= extent;
  }
  if (!element_.isParameterized() && count > std::numeric_limits<std::uint64_t>::max() / element_.width()) {
    throw std::overflow_error("value bit size overflow");
  }
}
ValueType ValueType::scalar(ElementType element) { return {std::move(element), {}}; }
ValueType ValueType::matrix(ElementType element, std::vector<std::uint64_t> shape) {
  if (shape.empty()) throw std::invalid_argument("a matrix must have at least one dimension");
  return {std::move(element), std::move(shape)};
}
ValueType ValueType::parameterizedMatrix(ElementType element, Value descriptor, std::string interpretation) {
  if (interpretation.empty()) throw std::invalid_argument("descriptor interpretation must not be empty");
  ValueType result(std::move(element), {});
  result.shape_query_ = std::make_shared<const DescriptorQuery>(DescriptorQuery{std::move(descriptor), std::move(interpretation)});
  return result;
}
std::span<const std::uint64_t> ValueType::shape() const {
  if (shape_query_) throw std::logic_error("matrix shape depends on a descriptor");
  return shape_;
}
std::uint64_t ValueType::elementCount() const {
  if (shape_query_) throw std::logic_error("matrix element count depends on a descriptor");
  std::uint64_t count = 1;
  for (const auto extent : shape_) count *= extent;
  return count;
}
std::uint64_t ValueType::bitSize() const { return elementCount() * element_.width(); }
std::string ValueType::toString() const {
  if (isScalar()) return element_.toString();
  if (shape_query_) return "matrix<" + shape_query_->toString("ShapeOf") + ", " + element_.toString() + ">";
  std::string result = "matrix<";
  for (const auto extent : shape_) result += std::to_string(extent) + "x";
  return result + element_.toString() + ">";
}
bool sameShape(const ValueType& left, const ValueType& right) noexcept {
  if (!left.hasStaticShape() || !right.hasStaticShape())
    return left.shapeQuery() && right.shapeQuery() && sameQuery(left.shapeQuery(), right.shapeQuery());
  return std::ranges::equal(left.shape(), right.shape());
}

}  // namespace sentryward::value_ir
