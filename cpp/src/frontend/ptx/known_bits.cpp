#include "sentryward/frontend/ptx/resources.h"
#include <algorithm>
#include <sstream>

namespace sentryward::frontend::ptx {
Resources::KnownBits Resources::FixedBits(const ExecutedInstruction &execution,
    std::string_view operand, unsigned width, const BitsResolver &resolver) const {
  std::set<DynamicKey> visiting;
  std::function<KnownBits(const ExecutedInstruction &, std::string_view, unsigned)> query;
  query = [&](const ExecutedInstruction &e, std::string_view text, unsigned bits) -> KnownBits {
    if (!bits || bits > 64) return {};
    const auto full = UINT64_MAX >> (64 - bits);
    const auto token = Trim(text);
    if (auto n = ScalarInteger(Value(e, token)))
      return {(*n & Integer(full)).convert_to<std::uint64_t>(), full};
    if (resolver)
      if (auto answer = resolver(e, token, bits))
        return {answer->first & answer->second & full, answer->second & full};
    const auto *origin = program.RegisterOrigin(e, token);
    if (!origin || !active_index_ || visiting.contains(*origin)) return {};
    const auto found = Definition(*origin);
    if (!found || found->predicate_value() != true) return {};
    const auto &definition = *found;
    const auto &i = program.function->instructions.at(definition.instruction_id());
    const auto args = OperandTexts(*program.module, i);
    if (args.empty() || args[0] != token) return {};
    visiting.insert(*origin);
    struct Erase {
      std::set<DynamicKey> &set; DynamicKey key;
      ~Erase() { set.erase(key); }
    } erase{visiting, *origin};
    auto get = [&](std::size_t index, unsigned size = 0) {
      return query(definition, args.at(index), size ? size : bits);
    };
    const auto &mods = i.modifiers;
    const auto type_width = [](const std::string &s) -> unsigned {
      if (s.size() < 2 || (s[0] != 'u' && s[0] != 's')) return 0;
      auto parsed = ParseInteger(std::string_view(s).substr(1));
      return parsed && *parsed > 0 && *parsed <= 64 ? parsed->convert_to<unsigned>() : 0;
    };
    if (i.opcode == "cvt" && args.size() == 2 && mods.size() == 2 &&
        type_width(mods[0]) == bits && type_width(mods[1])) {
      const auto src = type_width(mods[1]);
      auto [v, k] = get(1, src);
      if (bits <= src) return {v & full, k & full};
      const auto upper = full ^ (UINT64_MAX >> (64 - src));
      if (mods[1][0] == 'u') return {v, k | upper};
      const auto sign = 1ULL << (src - 1);
      return (k & sign) ? KnownBits{v | ((v & sign) ? upper : 0), k | upper} : KnownBits{v, k};
    }
    const auto suffix = std::to_string(bits);
    if ((i.opcode == "shr" || i.opcode == "shl") && args.size() == 3 &&
        (Has(i, "b" + suffix) || Has(i, "u" + suffix) || Has(i, "s" + suffix))) {
      auto [shift, known] = get(2, 32);
      if (known != UINT32_MAX) return {};
      shift = std::min<std::uint64_t>(shift, bits);
      auto [v, k] = get(1);
      if (i.opcode == "shl")
        return shift == 64 ? KnownBits{0, full} :
            KnownBits{(v << shift) & full, ((k << shift) | ((1ULL << shift) - 1)) & full};
      const auto upper = full ^ (shift == 64 ? 0 : full >> shift);
      const auto sign = 1ULL << (bits - 1);
      const auto sv = shift == 64 ? 0 : v >> shift;
      const auto sk = shift == 64 ? 0 : k >> shift;
      if (Has(i, "s" + suffix))
        return (k & sign) ? KnownBits{sv | ((v & sign) ? upper : 0), sk | upper} : KnownBits{sv, sk};
      return {sv, sk | upper};
    }
    if (!Has(i, "b" + suffix)) return {};
    if (i.opcode == "mov" && args.size() == 2 && args[1].starts_with('{') && args[1].ends_with('}')) {
      std::vector<std::string> parts;
      std::stringstream values(args[1].substr(1, args[1].size() - 2));
      for (std::string value; std::getline(values, value, ',');) parts.push_back(Trim(value));
      if ((parts.size() != 2 && parts.size() != 4) || bits % parts.size()) return {};
      std::uint64_t value = 0, known = 0;
      const auto part_width = bits / parts.size();
      for (std::size_t n = 0; n < parts.size(); ++n) {
        auto [v, k] = query(definition, parts[n], part_width);
        value |= v << (n * part_width); known |= k << (n * part_width);
      }
      return {value & full, known & full};
    }
    if (i.opcode == "selp" && args.size() == 4) {
      if (auto p = ScalarInteger(Value(definition, args[3]))) return get(*p == 0 ? 2 : 1);
      const auto [a, ka] = get(1); const auto [b, kb] = get(2);
      const auto known = ka & kb & ~(a ^ b) & full;
      return {a & known, known};
    }
    if (args.size() == 3 && (i.opcode == "and" || i.opcode == "or" || i.opcode == "xor")) {
      const auto [a, ka] = get(1); const auto [b, kb] = get(2);
      const auto known = (i.opcode == "and" ? (ka & kb) | (ka & ~a) | (kb & ~b) :
                          i.opcode == "or" ? (ka & kb) | (ka & a) | (kb & b) : ka & kb) & full;
      const auto value = i.opcode == "and" ? a & b : i.opcode == "or" ? a | b : a ^ b;
      return {value & known, known};
    }
    return {};
  };
  return query(execution, operand, width);
}
} // namespace sentryward::frontend::ptx
