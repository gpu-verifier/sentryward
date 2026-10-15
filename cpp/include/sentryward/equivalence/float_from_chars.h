#pragma once
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <version>
namespace sentryward::equivalence {
// std::from_chars for float and double. libc++ may parse only integers; there
// the same grammar is matched by hand and strtof/strtod convert the match.
// The process never changes its locale, so both read the "C" radix point.
template<class T> std::from_chars_result floatFromChars(const char* first, const char* last, T& value) {
#if defined(__cpp_lib_to_chars) || !defined(_LIBCPP_VERSION)
  return std::from_chars(first, last, value);
#else
  const char* p = first;
  const bool negative = p != last && *p == '-';
  if(negative) ++p;
  const auto digits = [&] {
    const char* begin = p;
    while(p != last && *p >= '0' && *p <= '9') ++p;
    return p != begin;
  };
  const auto word = [&](std::string_view expected) {
    if(static_cast<std::size_t>(last - p) < expected.size()) return false;
    for(std::size_t i = 0; i < expected.size(); ++i)
      if((p[i] | 0x20) != expected[i]) return false;
    p += expected.size();
    return true;
  };
  const bool whole = digits();
  bool fraction = false;
  if(p != last && *p == '.') {
    const char* point = p++;
    fraction = digits();
    if(!whole && !fraction) p = point;
  }
  if(whole || fraction) {
    if(p != last && (*p == 'e' || *p == 'E')) {
      const char* exponent = p++;
      if(p != last && (*p == '+' || *p == '-')) ++p;
      if(!digits()) p = exponent;
    }
  } else if(word("inf")) {
    word("inity");
    value = negative ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
    return {p, std::errc{}};
  } else if(word("nan")) {
    if(p != last && *p == '(') {
      const char* payload = p++;
      while(p != last && (std::isalnum(static_cast<unsigned char>(*p)) || *p == '_')) ++p;
      if(p != last && *p == ')') ++p; else p = payload;
    }
    value = negative ? -std::numeric_limits<T>::quiet_NaN() : std::numeric_limits<T>::quiet_NaN();
    return {p, std::errc{}};
  } else {
    return {first, std::errc::invalid_argument};
  }
  const std::string text(first, p);
  const int saved = errno;
  errno = 0;
  char* end = nullptr;
  const T parsed = std::is_same_v<T, float> ? static_cast<T>(std::strtof(text.c_str(), &end))
                                            : static_cast<T>(std::strtod(text.c_str(), &end));
  const bool range = errno == ERANGE && (parsed == 0 || std::isinf(parsed));
  errno = saved;
  if(end != text.c_str() + text.size()) return {first, std::errc::invalid_argument};
  if(range) return {p, std::errc::result_out_of_range};
  value = parsed;
  return {p, std::errc{}};
#endif
}
}
