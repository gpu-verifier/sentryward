#include "sentryward/frontend/ptx/operands.h"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <limits>
#include <stdexcept>

namespace sentryward::frontend::ptx {
namespace {
bool Head(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$';
}
bool Part(char c) {
  return Head(c) || std::isdigit(static_cast<unsigned char>(c));
}
std::string_view Trim(std::string_view s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
    s.remove_prefix(1);
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
    s.remove_suffix(1);
  return s;
}
std::optional<std::uint32_t> Number(std::string_view s) {
  if (s.empty())
    return std::nullopt;
  std::uint32_t v;
  auto result = std::from_chars(s.data(), s.data() + s.size(), v);
  if (result.ec != std::errc{} || result.ptr != s.data() + s.size())
    return std::nullopt;
  return v;
}
std::optional<std::uint32_t> Bits(std::string_view s) {
  if (s.starts_with("bf"))
    return Number(s.substr(2));
  if (!s.empty() &&
      std::string_view("busf").find(s.front()) != std::string_view::npos)
    return Number(s.substr(1));
  return std::nullopt;
}
std::uint32_t Narrow(std::uint64_t n) {
  if (n > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("PTX memory width exceeds supported range");
  return static_cast<std::uint32_t>(n);
}
} // namespace
std::vector<std::string> Registers(std::string_view text) {
  std::vector<std::string> result;
  text = Trim(text);
  if(const auto separator=text.find('|');separator!=text.npos) {
    auto first=Registers(text.substr(0,separator));
    auto second=Registers(text.substr(separator+1));
    first.insert(first.end(),second.begin(),second.end());
    return first;
  }
  if (text.size() >= 2 && text.front() == '{' && text.back() == '}') {
    text.remove_prefix(1); text.remove_suffix(1);
    for (;;) {
      const auto comma=text.find(',');
      auto part=Registers(text.substr(0,comma));
      result.insert(result.end(),part.begin(),part.end());
      if (comma==text.npos) break;
      text.remove_prefix(comma+1);
    }
    return result;
  }
  for (std::size_t at = 0; at < text.size(); ++at)
    if (text[at] == '%' && at + 1 < text.size() && Head(text[at + 1])) {
      auto end = at + 2;
      while (end < text.size() && Part(text[end]))
        ++end;
      result.emplace_back(text.substr(at, end - at));
      at = end - 1;
    }
  if (result.empty()) {
    text = Trim(text);
    if (!text.empty() && Head(text.front()) && text != "_" &&
        std::all_of(text.begin(), text.end(), Part))
      result.emplace_back(text);
  }
  return result;
}
std::vector<std::string> Destinations(const Module &m, const Instruction &i) {
  const auto &s = i.semantics;
  if (s.category == "collective" && s.operation == "named_reduce" &&
      !i.operands.empty())
    return Registers(m.view(i.operands.front()));
  if (s.category == "fence" || s.category == "control" ||
      s.category == "collective" || s.category == "tmem_lifetime" ||
      i.operands.empty())
    return {};
  if (s.category == "async") {
    if (i.opcode != "tcgen05" || s.operation != "load")
      return {};
    auto result = Registers(m.view(i.operands.front()));
    if (i.operands.size() > 1 &&
        std::find(i.modifiers.begin(), i.modifiers.end(), "red") !=
            i.modifiers.end()) {
      auto extra = Registers(m.view(i.operands[1]));
      result.insert(result.end(), extra.begin(), extra.end());
    }
    return result;
  }
  if (s.category == "memory" &&
      (s.memory_mode == "write" || s.memory_mode == "atomic")) {
    if (s.memory_mode != "atomic" || i.opcode == "red")
      return {};
  }
  if (s.category == "mbarrier") {
    const auto &op = s.operation;
    if (op != "test_wait_parity" && op != "try_wait_parity" &&
        op != "test_wait_state" && op != "try_wait_state" &&
        op != "pending_count") {
      if (op != "arrive" && op != "arrive_drop" && op != "arrive_expect_tx")
        return {};
      auto first = Trim(m.view(i.operands.front()));
      if (first == "_" || first.starts_with("["))
        return {};
    }
  }
  return Registers(m.view(i.operands.front()));
}
std::optional<std::uint32_t> MemoryWidth(const Instruction &i) {
  if (i.opcode == "ldmatrix" || i.opcode == "stmatrix") {
    std::optional<std::uint32_t> rows, columns, bits;
    bool packed = false;
    for (const auto &mod : i.modifiers) {
      std::string_view s = mod;
      if (s.starts_with("m") && !rows) {
        auto n = s.find('n');
        if (n != s.npos) {
          rows = Number(s.substr(1, n - 1));
          columns = Number(s.substr(n + 1));
        }
      }
      if (!bits)
        bits = Bits(s);
      packed |= s == "b4x16_p64" || s == "b6x16_p32";
    }
    if (!rows || !columns)
      return std::nullopt;
    std::uint64_t bytes;
    if (packed) {
      if (*columns != 16)
        return std::nullopt;
      bytes = 16;
    } else {
      if (!bits || (std::uint64_t(*columns) * *bits) % 8)
        return std::nullopt;
      bytes = std::uint64_t(*columns) * *bits / 8;
    }
    bytes *= *rows;
    if (bytes % 8)
      return std::nullopt;
    return Narrow(bytes / 8);
  }
  std::uint32_t vector = 1;
  std::optional<std::uint32_t> bits;
  for (const auto &mod : i.modifiers) {
    if (mod.starts_with("v"))
      if (auto n = Number(std::string_view(mod).substr(1)))
        vector = *n;
    if (auto n = Bits(mod))
      bits = n;
  }
  if (!bits || *bits % 8)
    return std::nullopt;
  return Narrow(std::uint64_t(vector) * *bits / 8);
}
std::optional<std::uint32_t> AddressLanes(const Instruction &i) {
  if (i.opcode != "ldmatrix" && i.opcode != "stmatrix")
    return std::nullopt;
  for (const auto &mod : i.modifiers)
    if (mod.starts_with("x"))
      if (auto n = Number(std::string_view(mod).substr(1))) {
        if (*n == 1 || *n == 2 || *n == 4)
          return 8 * *n;
        return std::nullopt;
      }
  return std::nullopt;
}
} // namespace sentryward::frontend::ptx
