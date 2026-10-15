#include "sentryward/frontend/ptx/module.h"
#include "sentryward/frontend/ptx/operands.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace sentryward::frontend::ptx {
namespace {
bool Space(char c) { return std::isspace(static_cast<unsigned char>(c)); }
bool Head(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$';
}
bool Ident(char c) {
  return Head(c) || std::isdigit(static_cast<unsigned char>(c));
}
class Parser {
public:
  explicit Parser(Module &module) : m(module), masked(m.text) {
    lines.push_back(0);
    for (std::size_t i = 0; i < m.text.size(); ++i)
      if (m.text[i] == '\n')
        lines.push_back(i + 1);
    Mask();
  }
  void Run() {
    std::unordered_set<std::string> names;
    std::size_t at = 0;
    while (at < masked.size()) {
      if (masked[at] == '"') {
        at = Quoted(at);
        continue;
      }
      if (masked.compare(at, 6, ".entry") == 0 && Boundary(at + 6)) {
        at = FunctionAt(at, 6, "entry", names);
        continue;
      }
      if (masked.compare(at, 5, ".func") == 0 && Boundary(at + 5)) {
        at = FunctionAt(at, 5, "func", names);
        continue;
      }
      if (masked[at] == '.' && (at == 0 || !Ident(masked[at - 1]))) {
        auto name_end = at + 1;
        while (name_end < masked.size() && Ident(masked[name_end]))
          ++name_end;
        const auto name = masked.substr(at, name_end - at);
        if (name == ".version" || name == ".target" || name == ".address_size") {
          auto stop=masked.find('\n',name_end);
          if (stop==std::string::npos) stop=masked.size();
          m.directives.push_back({name,{at,stop},Trim(name_end,stop),{}});
          at=stop; continue;
        }
        if (name == ".extern") {
          const auto storage_at = Ws(name_end, masked.size());
          if (masked.compare(storage_at, 7, ".shared") == 0 && Boundary(storage_at + 7)) {
            const auto stop = Statement(at, masked.size());
            m.directives.push_back({".shared", {at, stop}, Trim(storage_at + 7, stop), {}});
            at = stop + 1;
            continue;
          }
        }
        if (name == ".file") {
          auto stop = masked.find('\n', name_end);
          if (stop == std::string::npos)
            stop = masked.size();
          auto quote = masked.find('"', name_end);
          if (quote < stop) {
            auto end = Quoted(quote);
            std::uint32_t id{};
            std::istringstream fields(
                masked.substr(name_end, quote - name_end));
            if (fields >> id)
              m.file_table[id] = masked.substr(quote + 1, end - quote - 2);
          }
          at = stop;
          continue;
        }
        if (name == ".shared" || name == ".global" || name == ".const") {
          auto stop = Statement(at, masked.size());
          m.directives.push_back({name, {at, stop}, Trim(name_end, stop), {}});
          at = stop + 1;
          continue;
        }
      }
      ++at;
    }
  }

private:
  Module &m;
  std::string masked;
  std::vector<std::size_t> lines;
  [[noreturn]] void Error(std::string message, std::size_t at) const {
    const auto row =
        std::upper_bound(lines.begin(), lines.end(), at) - lines.begin() - 1;
    throw ParserError(m.source_name, row + 1, at - lines[row] + 1, message);
  }
  bool Boundary(std::size_t at) const {
    return at == masked.size() || !Ident(masked[at]);
  }
  std::size_t Ws(std::size_t at, std::size_t end) const {
    while (at < end && Space(masked[at]))
      ++at;
    return at;
  }
  Range Trim(std::size_t a, std::size_t b) const {
    a = Ws(a, b);
    while (b > a && Space(masked[b - 1]))
      --b;
    return {a, b};
  }
  std::size_t Quoted(std::size_t at) const {
    for (auto i = at + 1; i < masked.size(); ++i) {
      if (masked[i] == '\\') {
        ++i;
        continue;
      }
      if (masked[i] == '"')
        return i + 1;
    }
    Error("unterminated string", at);
  }
  void Mask() {
    for (std::size_t at = 0; at < masked.size();) {
      if (masked[at] == '"') {
        at = Quoted(at);
        continue;
      }
      bool line = masked.compare(at, 2, "//") == 0,
           block = masked.compare(at, 2, "/*") == 0;
      if (!line && !block) {
        ++at;
        continue;
      }
      auto end = masked.find(line ? "\n" : "*/", at + 2);
      if (end == std::string::npos) {
        if (block)
          Error("unterminated block comment", at);
        end = masked.size();
      } else if (block)
        end += 2;
      for (; at < end; ++at)
        if (masked[at] != '\n')
          masked[at] = ' ';
    }
  }
  std::size_t Match(std::size_t at, std::size_t end) const {
    std::vector<char> stack{masked[at]};
    for (auto i = at + 1; i < end; ++i) {
      const char c = masked[i];
      if (c == '"') {
        i = Quoted(i) - 1;
        continue;
      }
      if (c == '(' || c == '[' || c == '{')
        stack.push_back(c);
      if (c == ')' || c == ']' || c == '}') {
        const char expected = c == ')' ? '(' : c == ']' ? '[' : '{';
        if (stack.empty() || stack.back() != expected)
          Error("mismatched delimiter", i);
        stack.pop_back();
        if (stack.empty())
          return i;
      }
    }
    Error("unterminated delimiter", at);
  }
  std::size_t Statement(std::size_t at, std::size_t end) const {
    for (auto i = at; i < end; ++i) {
      if (masked[i] == '"') {
        i = Quoted(i) - 1;
        continue;
      }
      if (masked[i] == '(' || masked[i] == '[' || masked[i] == '{') {
        i = Match(i, end);
        continue;
      }
      if (masked[i] == ';')
        return i;
      if (masked[i] == '}')
        Error("missing instruction semicolon", i);
    }
    Error("missing instruction semicolon", at);
  }
  std::vector<Parameter> Parameters(std::size_t begin, std::size_t end) const {
    std::vector<Parameter> result;
    for (auto at = begin; at <= end; ++at) {
      if (at < end && (masked[at] == '[' || masked[at] == '(')) {
        at = Match(at, end);
        continue;
      }
      if (at < end && masked[at] != ',')
        continue;
      auto range = Trim(begin, at);
      begin = at + 1;
      if (range.begin == range.end)
        continue;
      const auto text = masked.substr(range.begin, range.end - range.begin);
      Parameter p;
      p.source = range;
      static const std::regex token(R"(\.[A-Za-z_][A-Za-z0-9_]*)");
      for (std::sregex_iterator it(text.begin(), text.end(), token), stop;
           it != stop; ++it) {
        auto value = it->str();
        if (!p.storage && it->position() == 0 &&
            (value == ".param" || value == ".reg"))
          p.storage = value;
        else if (value != ".align")
          p.types.push_back(value);
      }
      static const std::regex align(R"(\.align\s+(\d+))"),
          name(R"(([A-Za-z_$][A-Za-z0-9_$]*)(\[[^\]]*\])?\s*$)");
      std::smatch match;
      if (std::regex_search(text, match, align))
        p.alignment = static_cast<std::uint32_t>(std::stoul(match[1]));
      if (std::regex_search(text, match, name)) {
        p.name = match[1];
        if (match[2].matched)
          p.array = match[2];
      }
      result.push_back(std::move(p));
    }
    return result;
  }
  std::size_t FunctionAt(std::size_t start, std::size_t prefix,
                         const std::string &kind,
                         std::unordered_set<std::string> &names) {
    Function f;
    f.kind = kind;
    auto at = Ws(start + prefix, masked.size());
    if (at < masked.size() && masked[at] == '(') {
      if (kind != "func")
        Error("entry cannot have return parameters", at);
      auto end = Match(at, masked.size());
      f.returns = Parameters(at + 1, end);
      at = Ws(end + 1, masked.size());
    }
    auto begin = at;
    if (at == masked.size() || !Head(masked[at]))
      Error("missing function name", at);
    while (at < masked.size() && Ident(masked[at]))
      ++at;
    f.name = masked.substr(begin, at - begin);
    if (!names.insert(f.name).second)
      Error("duplicate function", begin);
    at = Ws(at, masked.size());
    if (at < masked.size() && masked[at] == '(') {
      auto end = Match(at, masked.size());
      f.parameters = Parameters(at + 1, end);
      at = end + 1;
    }
    for (; at < masked.size(); ++at) {
      if (masked[at] == '.') {
        const auto begin = at;
        ++at;
        while (at < masked.size() && Ident(masked[at])) ++at;
        const auto name = masked.substr(begin, at - begin);
        if (name == ".reqntid" || name == ".maxntid" || name == ".reqnctapercluster") {
          const auto arguments = Ws(at, masked.size());
          auto stop = arguments;
          while (stop < masked.size() && (std::isdigit(static_cast<unsigned char>(masked[stop])) ||
                 std::isspace(static_cast<unsigned char>(masked[stop])) || masked[stop] == ',')) ++stop;
          f.directives.push_back({name, {begin, stop}, {arguments, stop}, {}});
          at = stop - 1;
        } else --at;
        continue;
      }
      if (masked[at] == '"') {
        at = Quoted(at) - 1;
        continue;
      }
      if (masked[at] == '(' || masked[at] == '[') {
        at = Match(at, masked.size());
        continue;
      }
      if (masked[at] == ';') {
        f.header = {start, at + 1};
        m.functions.push_back(std::move(f));
        return at + 1;
      }
      if (masked[at] == '{') {
        auto end = Match(at, masked.size());
        f.header = {start, at};
        f.body = {at + 1, end};
        f.has_body = true;
        Body(f);
        m.functions.push_back(std::move(f));
        return end + 1;
      }
    }
    Error("unterminated function", start);
  }
  void Body(Function &f) {
    auto at = f.body.begin;
    std::vector<std::uint32_t> scope;
    std::map<std::pair<std::vector<std::uint32_t>, std::string>, std::vector<std::string>> tables;
    std::uint32_t next_scope = 0;
    std::optional<DebugLocation> location;
    while ((at = Ws(at, f.body.end)) < f.body.end) {
      if (masked[at] == ';') {
        ++at;
        continue;
      }
      if (masked[at] == '{') {
        scope.push_back(++next_scope);
        ++at;
        continue;
      }
      if (masked[at] == '}') {
        if (scope.empty())
          Error("unmatched scope", at);
        scope.pop_back();
        ++at;
        continue;
      }
      if (Head(masked[at])) {
        auto end = at + 1;
        while (end < f.body.end && Ident(masked[end]))
          ++end;
        auto colon = end;
        while (colon < f.body.end &&
               (masked[colon] == ' ' || masked[colon] == '\t'))
          ++colon;
        if (colon < f.body.end && masked[colon] == ':') {
          f.labels.push_back(
              {masked.substr(at, end - at), scope,
               static_cast<std::uint32_t>(f.instructions.size())});
          at = colon + 1;
          continue;
        }
      }
      if (masked[at] == '.') {
        auto end = at + 1;
        while (end < f.body.end && Ident(masked[end]))
          ++end;
        auto directive = masked.substr(at, end - at);
        std::size_t stop;
        if (directive == ".loc" || directive == ".file" ||
            directive == ".maxnreg" || directive == ".maxntid" ||
            directive == ".reqntid" || directive == ".noreturn") {
          stop = masked.find('\n', end);
          if (stop == std::string::npos || stop > f.body.end)
            stop = f.body.end;
        } else
          stop = Statement(at, f.body.end);
        auto arguments = Trim(end, stop);
        if (arguments.end > arguments.begin && masked[arguments.end - 1] == ';')
          arguments = Trim(arguments.begin, arguments.end - 1);
        f.directives.push_back({directive, {at, stop}, arguments, scope});
        if (directive == ".branchtargets") {
          if (f.labels.empty() || f.labels.back().scope != scope)
            Error("branch table requires a label in the same scope", at);
          std::vector<std::string> targets;
          std::istringstream names(masked.substr(arguments.begin, arguments.end-arguments.begin));
          std::string name;
          while (std::getline(names,name,',')) {
            const auto first=name.find_first_not_of(" \t\r\n"), last=name.find_last_not_of(" \t\r\n");
            if (first==name.npos) Error("empty indirect branch target",at);
            targets.push_back(name.substr(first,last-first+1));
          }
          if (!tables.emplace(std::pair{scope,f.labels.back().name},std::move(targets)).second)
            Error("duplicate branch table",at);
        }
        if (directive == ".loc") {
          DebugLocation debug;
          std::istringstream fields(
              masked.substr(arguments.begin, arguments.end - arguments.begin));
          if (!(fields >> debug.file >> debug.line >> debug.column))
            Error("invalid debug location", at);
          std::getline(fields, debug.attributes);
          auto first = debug.attributes.find_first_not_of(" \t");
          debug.attributes =
              first == std::string::npos ? "" : debug.attributes.substr(first);
          location = std::move(debug);
        }
        at = stop + 1;
        continue;
      }
      const auto end = Statement(at, f.body.end);
      Instruction i;
      i.id = static_cast<std::uint32_t>(f.instructions.size());
      i.source = {at, end + 1};
      i.scope = scope;
      i.debug_location = location;
      auto row =
          std::upper_bound(lines.begin(), lines.end(), at) - lines.begin() - 1;
      i.line = static_cast<std::uint32_t>(row + 1);
      i.column = static_cast<std::uint32_t>(at - lines[row] + 1);
      if (masked[at] == '@') {
        ++at;
        if (masked[at] == '!') {
          i.predicate_negated = true;
          ++at;
        }
        const auto begin = at;
        if (masked[at] == '%')
          ++at;
        if (at >= end || !Head(masked[at]))
          Error("invalid predicate", at);
        while (at < end && Ident(masked[at]))
          ++at;
        i.predicate = masked.substr(begin, at - begin);
        at = Ws(at, end);
      }
      auto begin = at;
      if (at >= end || !(std::isalpha(static_cast<unsigned char>(masked[at])) ||
                         masked[at] == '_'))
        Error("expected opcode", at);
      while (at < end &&
             (Ident(masked[at]) || masked[at] == '.' || masked[at] == ':'))
        ++at;
      auto spelling = masked.substr(begin, at - begin);
      std::size_t part = 0;
      do {
        auto next = spelling.find('.', part);
        auto token = spelling.substr(
            part, next == std::string::npos ? next : next - part);
        if (token.empty())
          Error("empty opcode component", begin + part);
        if (part == 0)
          i.opcode = token;
        else
          i.modifiers.push_back(token);
        if (next == std::string::npos)
          break;
        part = next + 1;
      } while (true);
      if (at < end && !Space(masked[at]) && masked[at] != '[' &&
          masked[at] != '{')
        Error("unexpected character after opcode", at);
      begin = Ws(at, end);
      for (auto cursor = begin; cursor < end; ++cursor) {
        if (masked[cursor] == '"') {
          cursor = Quoted(cursor) - 1;
          continue;
        }
        if (masked[cursor] == '(' || masked[cursor] == '[' ||
            masked[cursor] == '{') {
          cursor = Match(cursor, end);
          continue;
        }
        if (masked[cursor] == ',') {
          auto range = Trim(begin, cursor);
          if (range.begin == range.end)
            Error("empty operand", begin);
          i.operands.push_back(range);
          begin = cursor + 1;
        }
      }
      if (begin < end) {
        auto range = Trim(begin, end);
        if (range.begin == range.end)
          Error("empty operand", begin);
        i.operands.push_back(range);
      }
      if (i.opcode == "bra") {
        if (i.operands.size() != 1)
          Error("direct branch needs one target", i.source.begin);
        i.control = "direct_branch";
      } else if (i.opcode == "brx") {
        if (i.operands.size() < 2)
          Error("indirect branch needs target table", i.source.begin);
        i.control = "indirect_branch";
      } else if (i.opcode == "ret")
        i.control = "return";
      else if (i.opcode == "exit")
        i.control = "exit";
      else if (i.opcode == "trap")
        i.control = "trap";
      else if (i.opcode == "call")
        i.control = "call";
      std::unordered_set<std::string> seen;
      for (auto range : i.operands)
        for (auto cursor = range.begin; cursor < range.end; ++cursor)
          if (masked[cursor] == '%' && cursor + 1 < range.end &&
              Head(masked[cursor + 1])) {
            auto stop = cursor + 2;
            while (stop < range.end && Ident(masked[stop]))
              ++stop;
            auto reg = masked.substr(cursor, stop - cursor);
            if (seen.insert(reg).second)
              i.registers.push_back(std::move(reg));
            cursor = stop - 1;
          }
      i.semantics = Classify(i);
      i.destinations = Destinations(m, i);
      i.memory_width = MemoryWidth(i);
      i.address_lanes = AddressLanes(i);
      f.instructions.push_back(std::move(i));
      at = end + 1;
    }
    if (!scope.empty())
      Error("unterminated scope", f.body.end);
    std::map<std::pair<std::vector<std::uint32_t>, std::string>, std::uint32_t>
        labels;
    for (const auto &label : f.labels)
      if (!labels
               .emplace(std::make_pair(label.scope, label.name),
                        label.instruction)
               .second)
        Error("duplicate label in scope: " + label.name, f.body.begin);
    for (auto &instruction : f.instructions) {
      if (instruction.control == "indirect_branch") {
        const auto name=std::string(m.view(instruction.operands.back()));
        auto search=instruction.scope;
        for (;;) {
          const auto table=tables.find({search,name});
          if (table!=tables.end()) {
            for (const auto &target : table->second) {
              auto target_scope=search;
              for (;;) {
                const auto found=labels.find({target_scope,target});
                if (found!=labels.end()) {
                  if (found->second>=f.instructions.size()) Error("branch target has no instruction",instruction.source.begin);
                  instruction.indirect_targets.push_back(found->second); break;
                }
                if (target_scope.empty()) Error("undefined indirect target: "+target,instruction.source.begin);
                target_scope.pop_back();
              }
            }
            break;
          }
          if (search.empty()) Error("undefined branch table: "+name,instruction.source.begin);
          search.pop_back();
        }
      }
      if (instruction.control == "direct_branch") {
        const auto name = std::string(m.view(instruction.operands.front()));
        if (name.empty() || !Head(name.front()) ||
            !std::all_of(name.begin(), name.end(), Ident))
          Error("invalid direct branch label", instruction.source.begin);
        auto search = instruction.scope;
        while (true) {
          const auto found = labels.find({search, name});
          if (found != labels.end()) {
            instruction.branch_target = found->second;
            for (auto id : search) {
              if (!instruction.branch_label.empty())
                instruction.branch_label += '/';
              instruction.branch_label += std::to_string(id);
            }
            if (!search.empty())
              instruction.branch_label += "::";
            instruction.branch_label += name;
            break;
          }
          if (search.empty())
            Error("undefined branch label: " + name, instruction.source.begin);
          search.pop_back();
        }
      }
    }
  }
};
} // namespace
Module Parse(std::string text, std::string source_name) {
  Module result;
  result.text = std::move(text);
  result.source_name = std::move(source_name);
  Parser(result).Run();
  return result;
}
} // namespace sentryward::frontend::ptx
