#pragma once
#include "core.hpp"

namespace rg {
inline bool floating(const std::string &t) {
  return t == "e2m1" || t == "ue4m3" || t == "ue8m0" || t == "e8m0" || t.starts_with("f") || t.starts_with("bf") || t.starts_with("e4") ||
         t.starts_with("e5") || t.starts_with("tf");
}
inline unsigned width(const std::string &t) {
  if (t == "e2m1") return 4;
  if (t == "e4m3" || t == "ue4m3" || t == "e5m2" || t == "e8m0" || t == "ue8m0")
    return 8;
  auto p = t.find_first_of("0123456789");
  if (p == std::string::npos || p == 0 || t.size() - p > 3)
    throw Unsupported("type width: " + t);
  const std::string prefix = t.substr(0, p);
  if (prefix != "b" && prefix != "u" && prefix != "s" &&
      prefix != "f" && prefix != "bf" && prefix != "tf")
    throw Unsupported("unrecognized numeric type: " + t);
  unsigned w = 0;
  for (size_t i = p; i < t.size(); ++i) {
    if (t[i] < '0' || t[i] > '9')
      throw Unsupported("unrecognized compound type: " + t);
    w = w * 10 + unsigned(t[i] - '0');
  }
  if (w == 0 || w > 256)
    throw Unsupported("type width out of supported range");
  return w;
}
// Lookup one alias without copying the complete allocation/parameter map.
inline std::string mapped_name(const J &mapping, const char *section,
                               const std::string &name) {
  auto table = mapping.find(section);
  if (table == mapping.end())
    return name;
  auto value = table->find(name);
  return value == table->end() ? name : value->get<std::string>();
}
} // namespace rg
