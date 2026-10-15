#pragma once

#include <compare>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace sentryward::algorithm {

// Immutable text with value comparisons and shared ownership. Build one handle
// per static path/diagnostic, then copy it into dynamic records. There is no
// process-global intern table retaining text after its model is destroyed.
class SharedString final {
 public:
  SharedString() = default;
  SharedString(std::string value)
      : value_(value.empty() ? nullptr
                            : std::make_shared<const std::string>(std::move(value))) {}
  SharedString(std::string_view value) : SharedString(std::string(value)) {}
  SharedString(const char* value) : SharedString(std::string(value)) {}

  [[nodiscard]] const std::string& str() const noexcept {
    static const std::string empty;
    return value_ ? *value_ : empty;
  }
  [[nodiscard]] std::string_view view() const noexcept { return str(); }
  [[nodiscard]] bool empty() const noexcept { return str().empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return str().size(); }
  [[nodiscard]] const char* data() const noexcept { return str().data(); }
  [[nodiscard]] const char* c_str() const noexcept { return str().c_str(); }
  [[nodiscard]] bool starts_with(std::string_view prefix) const noexcept {
    return view().starts_with(prefix);
  }
  [[nodiscard]] const void* identity() const noexcept { return value_.get(); }
  operator const std::string&() const noexcept { return str(); }

  friend bool operator==(const SharedString& a, const SharedString& b) noexcept {
    return a.value_ == b.value_ || a.str() == b.str();
  }
  friend bool operator==(const SharedString& a, std::string_view b) noexcept {
    return a.view() == b;
  }
  friend bool operator==(const SharedString& a, const std::string& b) noexcept {
    return a.str() == b;
  }
  friend bool operator==(const SharedString& a, const char* b) noexcept {
    return a.view() == b;
  }
  friend auto operator<=>(const SharedString& a, const SharedString& b) noexcept {
    if (a.value_ == b.value_) return std::strong_ordering::equal;
    return a.str() <=> b.str();
  }
  friend auto operator<=>(const SharedString& a, std::string_view b) noexcept {
    return a.view() <=> b;
  }
  friend auto operator<=>(const SharedString& a, const std::string& b) noexcept {
    return a.str() <=> b;
  }
  friend auto operator<=>(const SharedString& a, const char* b) noexcept {
    return a.view() <=> std::string_view(b);
  }
  friend std::string operator+(const SharedString& a, const char* b) {
    return a.str() + b;
  }
  friend std::ostream& operator<<(std::ostream& stream, const SharedString& value) {
    return stream << value.str();
  }
  // Keep the existing JSON wire representation without coupling this base
  // header to nlohmann/json or to a particular basic_json specialization.
  template <class Json>
  friend void to_json(Json& json, const SharedString& value) { json = value.str(); }

 private:
  std::shared_ptr<const std::string> value_;
};

}  // namespace sentryward::algorithm
