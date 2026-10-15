#pragma once
#include <stdexcept>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sentryward::frontend::ptx {
// Offsets reference Module::text. Static text and decoded semantics are stored
// once, independently of the number of lanes, CTAs or dynamic occurrences.
struct ParserError : std::invalid_argument {
  std::string file;
  std::uint32_t line, column;
  ParserError(std::string f, std::uint32_t l, std::uint32_t c, const std::string& message)
      : std::invalid_argument(f + ":" + std::to_string(l) + ":" + std::to_string(c) + ": " + message),
        file(std::move(f)), line(l), column(c) {}
};

struct Range {
  std::size_t begin{}, end{};
};
struct Semantics {
  std::string category, operation;
  bool modeled = true;
  std::optional<std::string> memory_space, memory_mode;
  bool asynchronous = false, control_relevant = false, memory_effect = false;
  bool atomic_add = false;
};
struct Directive {
  std::string name;
  Range source, arguments;
  std::vector<std::uint32_t> scope;
};
struct DebugLocation {
  std::uint32_t file{}, line{}, column{};
  std::string attributes;
};
struct Instruction {
  std::uint32_t id{}, line{}, column{};
  Range source;
  std::string opcode;
  std::vector<std::string> modifiers;
  std::vector<Range> operands;
  std::optional<std::uint32_t> branch_target;
  std::string branch_label;
  std::optional<std::string> predicate;
  bool predicate_negated = false;
  std::string control = "none";
  std::optional<DebugLocation> debug_location;
  std::vector<std::uint32_t> scope;
  std::vector<std::string> registers;
  std::vector<std::uint32_t> indirect_targets;
  Semantics semantics;
  std::vector<std::string> destinations;
  std::optional<std::uint32_t> memory_width, address_lanes;
};
struct Label {
  std::string name;
  std::vector<std::uint32_t> scope;
  std::uint32_t instruction{};
};
struct Parameter {
  std::optional<std::string> name, storage, array;
  std::optional<std::uint32_t> alignment;
  std::vector<std::string> types;
  Range source;
};
struct Function {
  std::string name, kind;
  Range header, body;
  std::vector<Parameter> parameters, returns;
  bool has_body = false;
  std::vector<Instruction> instructions;
  std::vector<Label> labels;
  std::vector<Directive> directives;
};
struct Module {
  std::string text, source_name;
  std::vector<Function> functions;
  std::vector<Directive> directives;
  std::map<std::uint32_t, std::string> file_table;
  [[nodiscard]] std::string_view view(Range range) const {
    return std::string_view(text).substr(range.begin, range.end - range.begin);
  }
};
[[nodiscard]] Semantics Classify(const Instruction &instruction);
[[nodiscard]] Module Parse(std::string text, std::string source_name);
} // namespace sentryward::frontend::ptx
