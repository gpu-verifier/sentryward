#include "execution_impl.hpp"

namespace sentryward::frontend::ptx {
std::optional<Integer> ScalarInteger(const Scalar &value) {
  if (auto n = std::get_if<Integer>(&value))
    return *n;
  if (auto b = std::get_if<bool>(&value))
    return Integer(*b);
  if (const auto *pinned = PinnedInteger(value))
    return *pinned->exact;
  return {};
}
bool TableAtom(const Scalar &value) noexcept { return Pinned(value); }
Scalar PlainValue(const Scalar &value) {
  if (DataPredicate(value)) return Scalar(*PinnedInteger(value)->exact != 0);
  if (const auto *pinned = PinnedInteger(value)) return *pinned->exact;
  if (const auto *pointer = PinnedPointer(value))
    return AllocationPointer{pointer->resource, *pointer->exact, pointer->extent_bytes};
  return value;
}
std::optional<PointerBits> PointerFragment(const Scalar &value,unsigned source,unsigned width) {
  if (!width || source+width>64) return {};
  if (const auto *bits=std::get_if<PointerBits>(&value)) {
    if (bits->destination_bit || source+width>bits->width) return {};
    auto result=*bits; result.source_bit+=source; result.width=width; return result;
  }
  std::shared_ptr<const PointerValue> pointer;
  if (auto p=std::get_if<AllocationPointer>(&value)) pointer=std::make_shared<PointerValue>(*p);
  else if (auto p=std::get_if<ParameterPointer>(&value)) pointer=std::make_shared<PointerValue>(*p);
  else if (auto p=std::get_if<ClusterSharedPointer>(&value)) pointer=std::make_shared<PointerValue>(*p);
  if (!pointer) return {};
  return PointerBits{std::move(pointer),source,width,0};
}
Scalar RestorePointer(const PointerBits &bits) {
  if (bits.source_bit==0 && bits.width==64 && bits.destination_bit==0)
    return std::visit([](const auto &pointer)->Scalar{return pointer;},*bits.pointer);
  return bits;
}
bool WarpLevelOpcode(std::string_view opcode) {
  return In(opcode, {"elect", "shfl", "vote", "match", "activemask", "redux", "mma", "wmma"});
}
// PTX generic addressing: an address maps to global memory unless it falls in the const, local or
// shared window; the .param window lies inside the global window (left unknown here).
std::optional<bool> GenericAddressInSpace(std::string_view space, const Scalar &address,
                                          const nlohmann::json &resources) {
  if (TableAtom(address)) return GenericAddressInSpace(space, PlainValue(address), resources);
  std::string resource;
  Integer lower, upper;
  std::optional<std::uint64_t> extent;
  if (const auto *p = std::get_if<AllocationPointer>(&address)) {
    resource = p->resource; lower = upper = p->offset; extent = p->extent_bytes;
  } else if (const auto *q = std::get_if<BoundedAllocationPointer>(&address)) {
    resource = (*q)->resource; lower = (*q)->lower; upper = (*q)->upper; extent = (*q)->extent_bytes;
  } else return std::nullopt;
  if (lower < 0 || (extent && upper > Integer(*extent))) return std::nullopt;
  const auto record = std::find_if(resources.begin(), resources.end(),
                                   [&](const auto &r) { return r.value("id", "") == resource; });
  if (record == resources.end() || record->value("kind", "") != "memory" || record->value("space", "") != "global")
    return std::nullopt;
  if (space == "global") return true;
  if (In(space, {"shared", "shared::cta", "shared::cluster", "local", "const"})) return false;
  return std::nullopt;
}
std::optional<Integer> ParseInteger(std::string_view text) {
  text = TrimView(text);
  if (text.empty())
    return {};
  const bool unsigned_literal=text.back()=='U';
  if(unsigned_literal)text.remove_suffix(1);
  if(text.empty())return {};
  bool negative = false;
  if (text.front() == '+' || text.front() == '-') {
    negative = text.front() == '-';
    text.remove_prefix(1);
  }
  if (text.empty())
    return {};
  unsigned base = 10;
  if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    base = 16;
    text.remove_prefix(2);
  } else if (text.size()>2 && text[0]=='0' && (text[1]=='b' || text[1]=='B')) {
    base=2;
    text.remove_prefix(2);
  } else if(text.size()>1 && text[0]=='0') {
    base=8;
  }
  if (text.empty())
    return {};
  Integer value = 0;
  for (unsigned char ch : text) {
    unsigned digit;
    if (ch >= '0' && ch <= '9')
      digit = ch - '0';
    else if (ch >= 'a' && ch <= 'f')
      digit = ch - 'a' + 10;
    else if (ch >= 'A' && ch <= 'F')
      digit = ch - 'A' + 10;
    else
      return {};
    if (digit >= base)
      return {};
    value *= base;
    value += digit;
  }
  // PTX ISA 4.5.1: U selects unsigned 64-bit literal arithmetic.
  if(negative && (unsigned_literal || value>std::numeric_limits<std::int64_t>::max()))
    return Integer((-value)&((Integer(1)<<64)-1));
  return negative ? Integer(-value) : value;
}
std::optional<std::pair<std::string, Integer>>
SplitAddress(std::string_view text) {
  text = TrimView(text);
  if (text.empty())
    return {};
  if (auto number = ParseInteger(text))
    return {{std::string(text), 0}};
  std::size_t split = std::string_view::npos;
  for (std::size_t i = 1; i < text.size(); ++i)
    if (text[i] == '+' || text[i] == '-') {
      split = i;
      break;
    }
  if (split == std::string_view::npos) {
    if (text.find_first_of(" \t,[]{}") != std::string_view::npos)
      return {};
    return {{std::string(text), 0}};
  }
  auto base = TrimView(text.substr(0, split));
  std::string offset;
  for (unsigned char ch : text.substr(split))
    if (!std::isspace(ch))
      offset.push_back(ch);
  if (offset.starts_with("+-")) offset.erase(0,1);
  auto n = ParseInteger(offset);
  if (base.empty() || !n)
    return {};
  return {{std::string(base), *n}};
}
std::vector<std::string> RegistersIn(std::string_view text) {
  std::vector<std::string> result;
  text=TrimView(text);
  if(const auto separator=text.find('|');separator!=text.npos) {
    auto first=RegistersIn(text.substr(0,separator));
    auto second=RegistersIn(text.substr(separator+1));
    first.insert(first.end(),second.begin(),second.end());
    return first;
  }
  if (text.size()>=2 && text.front()=='{' && text.back()=='}') {
    text.remove_prefix(1); text.remove_suffix(1);
    for (;;) {
      const auto comma=text.find(',');
      auto part=RegistersIn(text.substr(0,comma)); result.insert(result.end(),part.begin(),part.end());
      if (comma==text.npos) break;
      text.remove_prefix(comma+1);
    }
    return result;
  }
  auto head = [](unsigned char c) {
    return std::isalpha(c) || c == '_' || c == '$';
  };
  auto ident = [&](unsigned char c) { return head(c) || std::isdigit(c); };
  for (std::size_t i = 0; i < text.size(); ++i)
    if (text[i] == '%' && i + 1 < text.size() && head(text[i + 1])) {
      auto begin = i++;
      while (i < text.size() && ident(text[i]))
        ++i;
      result.emplace_back(text.substr(begin, i - begin));
      if (i)
        --i;
    }
  auto token = TrimView(text);
  if (result.empty() && !token.empty() && token != "_" && head(token.front()) &&
      std::all_of(token.begin(), token.end(), ident))
    result.emplace_back(token);
  return result;
}
namespace {
ExecutionProgram SpecializeFunctionOnce(const Module &, const Function &, const FrontendConfig &,
                                        std::function<void()>, SpecializationPurpose);
// Representative control flow is a last resort: it applies only after ordinary execution failed on an
// unresolved branch, so it never changes a kernel the fixed input already determines.
template <class Run>
auto WithDataBranches(const FrontendConfig &c, bool defer_shared_control, Run run) {
  auto undetermined = [&](const ExecutionError &error) {
    // Shared-memory control is first retried by the caller as a whole cluster
    // (SpecializeCluster), which applies these passes itself if that also fails.
    return !exact_control_pass && !(defer_shared_control && error.shared_control) &&
           (error.code == "branch_not_static" || error.code == "missing_parameter_value");
  };
  auto pass = [&](bool data) {
    exact_control_pass = true;
    data_branch_pass = data;
    try { auto result = run(); exact_control_pass = data_branch_pass = false; return result; }
    catch (...) { exact_control_pass = data_branch_pass = false; throw; }
  };
  try { return run(); }
  catch (const ExecutionError &error) { if (!undetermined(error)) throw; }
  try { return pass(false); }
  catch (const ExecutionError &error) {
    if (!DataBranchesAssumed(c) ||
        (error.code != "branch_not_static" && error.code != "missing_parameter_value"))
      throw;
  }
  return pass(true);
}
} // namespace
// Data predicates' opaque regions (table atoms) are admitted statically (DataOpaqueAdmitted); a run
// that took one and then failed on control is re-run taking none, by the exact outcomes.
template <class Run> auto WithDataOpaqueFallback(Run run) {
  data_opaque_used = false;
  try { return run(); }
  catch (const ExecutionError &error) {
    if (!data_opaque_used || (error.code != "branch_not_static" && error.code != "missing_parameter_value")) throw;
  }
  data_opaque_disabled = true;
  try { auto result = run(); data_opaque_disabled = false; return result; }
  catch (...) { data_opaque_disabled = false; throw; }
}
ExecutionProgram SpecializeFunction(const Module &m, const Function &f,
                                    const FrontendConfig &c,
                                    std::function<void()> poll,
                                    SpecializationPurpose purpose) {
  return WithDataOpaqueFallback([&] {
    return WithDataBranches(c, true, [&] { return SpecializeFunctionOnce(m, f, c, poll, purpose); });
  });
}
namespace {
ExecutionProgram SpecializeFunctionOnce(const Module &m, const Function &f,
                                        const FrontendConfig &c,
                                        std::function<void()> poll,
                                        SpecializationPurpose purpose) {
  try {
    auto executor = std::make_shared<Executor>(m, f, c, poll, purpose);
    auto compressed = executor->RunTemplates();
    auto result = compressed ? std::move(*compressed) : executor->Run();
    if (purpose == SpecializationPurpose::Verification) RecoverTensorMaps(result);
    return result;
  }
  catch (const ExecutionError &error) {
    if (!error.shared_control) throw;
    const auto cluster=c.cluster_dim.value_or(Coordinates{1,1,1});
    if (cluster[0]*cluster[1]*cluster[2]!=1) throw;
    std::vector<ExecutionProgram> programs;
    try { SharedRoutingAttempt attempt(error); programs=Executor::RunCluster(m,f,{c},poll,purpose); }
    catch (const ExecutionError &) {
      if (!error.shared_routed) throw;
      // The routed run failed: this pass runs without routing, as before routing existed (its
      // result, or its error). A later pass of WithDataBranches (exact control, data branches)
      // routes again.
      struct Restore { bool previous; ~Restore() { shared_routing_disabled = previous; } } restore{shared_routing_disabled};
      shared_routing_disabled = true;
      return SpecializeFunctionOnce(m,f,c,poll,purpose);
    }
    if (purpose == SpecializationPurpose::Verification) RecoverTensorMaps(programs.front());
    return std::move(programs.front());
  }
}
} // namespace
SharedRoutingAttempt::SharedRoutingAttempt(const ExecutionError &routed) : previous_(routed_branch_line) {
  routed_branch_line = routed.shared_routed ? std::optional<std::uint32_t>(routed.line) : std::nullopt;
}
SharedRoutingAttempt::~SharedRoutingAttempt() { routed_branch_line = previous_; }
ExecutionProgram SpecializeFunctionUnrouted(const Module &m, const Function &f, const FrontendConfig &c,
                                            std::function<void()> poll, SpecializationPurpose purpose) {
  struct Restore { bool previous; ~Restore() { shared_routing_disabled = previous; } } restore{shared_routing_disabled};
  shared_routing_disabled = true;
  return SpecializeFunction(m, f, c, std::move(poll), purpose);
}
std::vector<ExecutionProgram> SpecializeCluster(const Module &m, const Function &f,
    const std::vector<FrontendConfig> &configs, std::function<void()> poll,
    SpecializationPurpose purpose) {
  auto programs=WithDataOpaqueFallback([&]{
    return WithDataBranches(configs.front(),false,[&]{return Executor::RunCluster(m,f,configs,poll,purpose);});
  });
  if (purpose == SpecializationPurpose::Verification)
    for (auto &program : programs) RecoverTensorMaps(program);
  return programs;
}
Json ScalarJson(const Scalar &value) {
  if (TableAtom(value)) return ScalarJson(PlainValue(value));
  if (const auto *bounded=std::get_if<BoundedInteger>(&value))
    return {{"kind","bounded_integer"},{"lower",ScalarJson(Scalar((*bounded)->lower))},
            {"upper",ScalarJson(Scalar((*bounded)->upper))},{"identity",(*bounded)->identity}};
  if (const auto *pointer=std::get_if<BoundedAllocationPointer>(&value))
    return {{"kind","bounded_allocation_pointer"},{"resource",(*pointer)->resource},
            {"lower",ScalarJson(Scalar((*pointer)->lower))},{"upper",ScalarJson(Scalar((*pointer)->upper))}};
  if (auto fragment=std::get_if<DescriptorFragment>(&value))
    return {{"kind","descriptor_fragment"},{"descriptor",fragment->descriptor},
            {"offset",fragment->offset},{"bytes",fragment->bytes}};
  if (const auto *bits=std::get_if<PointerBits>(&value))
    return {{"kind","pointer_bits"},{"pointer",std::visit([](const auto &p){return ScalarJson(Scalar(p));},*bits->pointer)},
        {"source_bit",bits->source_bit},{"width",bits->width},{"destination_bit",bits->destination_bit}};
  if (auto n = std::get_if<Integer>(&value)) {
    if (*n >= INT64_MIN && *n <= INT64_MAX)
      return n->convert_to<std::int64_t>();
    if (*n >= 0 && *n <= UINT64_MAX)
      return n->convert_to<std::uint64_t>();
    return {{"integer", n->convert_to<std::string>()}};
  }
  if (auto b = std::get_if<bool>(&value))
    return *b;
  if (auto p = std::get_if<ParameterPointer>(&value))
    return {{"kind", "parameter_pointer"},
            {"region", p->region},
            {"offset", ScalarJson(Scalar(p->offset))}};
  if (auto p = std::get_if<AllocationPointer>(&value))
    return {
        {"kind", "allocation_pointer"},
        {"resource", p->resource},
        {"offset", ScalarJson(Scalar(p->offset))},
        {"extent_bytes", p->extent_bytes ? Json(*p->extent_bytes) : Json()}};
  if (auto p = std::get_if<ClusterSharedPointer>(&value))
    return {{"kind", "cluster_shared_pointer"},
            {"local_address", ScalarJson(Scalar(p->local_address))},
            {"target_rank", p->target_rank}};
  return nullptr;
}
Json ExecutionRecordJson(const ExecutionProgram &program,
                         const ExecutedInstruction &r) {
  Json known = Json::object(), origins = Json::object();
  for (const auto &[id, v] : detail::ExecutionStorage::Bindings(r))
    known[program.register_names[id]] = ScalarJson(v);
  for (const auto &[id, v] : r.register_origins())
    origins[program.register_names[id]] = {v.instruction, v.occurrence};
  return {{"instruction", r.instruction_id()},
          {"occurrence", r.occurrence()},
          {"predicate_value",
           r.predicate_value() ? Json(*r.predicate_value()) : Json()},
          {"known_registers", known},
          {"register_origins", origins},
          {"collapsed_wait_loop", r.collapsed_wait_loop()},
          {"normalized_wait_result", r.normalized_wait_result() ? Json(*r.normalized_wait_result()) : Json()},
          {"normalized_clc_result", r.normalized_clc_result() ? Json(*r.normalized_clc_result()) : Json()},
          {"canonical_wait_probe", r.canonical_wait_probe()},
          {"wait_exit_label",
           r.wait_exit_label() ? Json(*r.wait_exit_label()) : Json()},
          {"schedule_seed_outcome",
           r.schedule_seed_outcome() ? Json(*r.schedule_seed_outcome()) : Json()},
          {"probe_fallback_key", r.probe_fallback_key()
                                     ? Json{r.probe_fallback_key()->instruction,
                                            r.probe_fallback_key()->occurrence}
                                     : Json()},
          {"opaque_compute_diamond_indices", r.opaque_compute_diamond_indices()},
          {"opaque_compute_join_label", r.opaque_compute_join_label()
                                            ? Json(*r.opaque_compute_join_label())
                                            : Json()},
          {"merged_arm_join", r.merged_arm_join() ? Json(*r.merged_arm_join()) : Json()},
          {"assumed_equality", r.assumed_equality()
              ? Json{r.assumed_equality()->first, ScalarJson(Scalar(r.assumed_equality()->second))} : Json()},
          {"opaque_compute_register_reads", r.opaque_compute_register_reads()},
          {"opaque_compute_register_writes", r.opaque_compute_register_writes()},
          {"collapsed_wait_indices", r.collapsed_wait_indices()}};
}
Json ExecutionJson(const ExecutionProgram &program) {
  Json lanes = Json::array();
  for (const auto &lane : program.lane_traces) {
    Json records = Json::array();
    for (std::size_t index=0;index<lane.instructions.size();++index)
      records.push_back(ExecutionRecordJson(program, detail::ExecutionStorage::Read(lane.instructions, index)));
    Json branches = Json::object();
    for (auto [k, v] : lane.branch_visits)
      branches[std::to_string(k)] = v;
    lanes.push_back({{"warp_id", lane.warp_id},
                     {"lane_id", lane.lane_id},
                     {"instructions", records},
                     {"branch_visits", branches},
                     {"loop_backedges", lane.loop_backedges}});
  }
  return {{"lane_traces", lanes},
          {"static_execution_counts", program.static_execution_counts}};
}
PrivateTemplateValues::PrivateTemplateValues() : overlay_(std::make_unique<detail::TemplateOverlay>()) {
  if (detail::template_overlay) throw std::logic_error("nested PrivateTemplateValues");
  detail::template_overlay = overlay_.get();
}
PrivateTemplateValues::~PrivateTemplateValues() { detail::template_overlay = nullptr; }
} // namespace sentryward::frontend::ptx
