#include "sentryward/algorithm/diagnostics/development_trace.h"
#include <cstdlib>
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/diagnostics/ir_size_profile.h"
#include "sentryward/algorithm/diagnostics/memory_profile.h"
#include "sentryward/algorithm/io/source_json.h"
#include "sentryward/algorithm/io/source_storage.h"

#include <algorithm>
#include <initializer_list>
#include <istream>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>

#include <nlohmann/json.hpp>

#include "sentryward/algorithm/analysis/reachability.h"
#include "sentryward/algorithm/analysis/family_relation_query.h"
#include "sentryward/algorithm/checker/unit_quotient.h"
#include "sentryward/algorithm/checker/verifier.h"
#include "sentryward/algorithm/event_ir/lowering.h"
#include "sentryward/algorithm/memory/access.h"
#include "sentryward/algorithm/source_ir/validator.h"
#include "sentryward/algorithm/source_ir/initialization_recovery.h"

namespace sentryward::algorithm {
namespace {
using Json = nlohmann::json;

struct SourceIrError : std::invalid_argument { using std::invalid_argument::invalid_argument; };
[[noreturn]] void Fail(const std::string& message) {
  throw SourceIrError(message);
}

void Fields(const Json& value, std::initializer_list<const char*> names,
            std::initializer_list<const char*> optional = {}) {
  if (!value.is_object()) Fail("expected an object");
  std::set<std::string> expected;
  for (const auto* name : names) expected.insert(name);
  std::set<std::string> allowed;
  for (const auto* name : optional) allowed.insert(name);
  for (const auto& [key, unused] : value.items()) {
    (void)unused;
    if (!expected.erase(key) && !allowed.contains(key)) Fail("unexpected field: " + key);
  }
  if (!expected.empty()) Fail("missing field: " + *expected.begin());
}

const Json& Array(const Json& value) {
  if (!value.is_array()) Fail("expected an array");
  return value;
}

std::string String(const Json& value) {
  if (!value.is_string()) Fail("expected a string");
  return value.get<std::string>();
}

template <typename T>
T UInt(const Json& value) {
  if (!value.is_number_unsigned() &&
      !(value.is_number_integer() && value.get<std::int64_t>() >= 0)) {
    Fail("expected an unsigned integer");
  }
  const auto number = value.get<std::uint64_t>();
  if (number > std::numeric_limits<T>::max()) Fail("integer out of range");
  return static_cast<T>(number);
}

template <typename T>
T Id(const Json& value) { return T{UInt<std::uint32_t>(value)}; }

template <typename T>
std::optional<T> OptionalId(const Json& value) {
  return value.is_null() ? std::nullopt : std::optional<T>{Id<T>(value)};
}

using SharedTextPool = std::map<std::string, SharedString, std::less<>>;
SharedString SharedText(const Json& value, SharedTextPool& texts) {
  const auto text = String(value);
  auto [entry, inserted] = texts.try_emplace(text);
  if (inserted) entry->second = text;
  return entry->second;
}

SourceLocation Location(const Json& value, SharedTextPool& texts) {
  Fields(value, {"file", "line", "column"});
  return {SharedText(value.at("file"), texts), UInt<std::uint32_t>(value.at("line")),
          UInt<std::uint32_t>(value.at("column"))};
}

MilestoneRef Reference(const Json& value) {
  Fields(value, {"instruction", "milestone", "effect_index"});
  const auto name = String(value.at("milestone"));
  const auto milestone = name == "issue" ? Milestone::kIssue
      : name == "reads_complete" ? Milestone::kReadsComplete
      : name == "complete" ? Milestone::kComplete
      : name == "effect" ? Milestone::kEffect
      : throw std::invalid_argument("unknown milestone: " + name);
  const auto effect = UInt<std::uint32_t>(value.at("effect_index"));
  if (milestone != Milestone::kEffect && effect != 0) {
    Fail("effect_index requires an effect milestone");
  }
  return {Id<InstId>(value.at("instruction")), milestone, effect};
}

std::vector<MilestoneRef> References(const Json& value) {
  std::vector<MilestoneRef> result;
  for (const auto& item : Array(value)) result.push_back(Reference(item));
  return result;
}

MemoryEndpoint Endpoint(const Json& value) {
  const auto kind = String(value.at("kind"));
  if (kind == "memory") {
    Fields(value, {"kind", "address"});
    return Id<AddressId>(value.at("address"));
  }
  if (kind == "register") {
    Fields(value, {"kind", "actor", "name"});
    return RegisterAddress{Id<ActorId>(value.at("actor")), String(value.at("name"))};
  }
  Fail("unknown memory endpoint: " + kind);
}

MemoryEffect Access(const Json& value) {
  Fields(value, {"mode", "endpoint", "bytes"}, {"atomic_scope"});
  const auto mode = String(value.at("mode"));
  if (mode != "read" && mode != "write") Fail("unknown access mode: " + mode);
  std::optional<AtomicScope> atomic;
  if (value.contains("atomic_scope") && !value.at("atomic_scope").is_null()) {
    const auto scope = String(value.at("atomic_scope"));
    if (scope == "cta") atomic = AtomicScope::kCta;
    else if (scope == "gpu") atomic = AtomicScope::kGpu;
    else if (scope == "sys") atomic = AtomicScope::kSys;
    else Fail("unsupported atomic scope: " + scope);
  }
  return {mode == "read" ? AccessMode::kRead : AccessMode::kWrite,
          Endpoint(value.at("endpoint")), value.at("bytes").is_null() ? std::nullopt
              : std::optional{UInt<std::uint64_t>(value.at("bytes"))}, atomic};
}

std::optional<CompletionAction> Action(const Json& value) {
  if (value.is_null()) return std::nullopt;
  Fields(value, {"kind", "barrier", "arrivals", "bytes"});
  const auto kind = String(value.at("kind"));
  const auto barrier = Id<BarrierLifetimeId>(value.at("barrier"));
  const auto arrivals = UInt<std::uint32_t>(value.at("arrivals"));
  const auto bytes = UInt<std::uint64_t>(value.at("bytes"));
  if (kind == "arrive" && bytes == 0) return ArriveEffect{barrier, arrivals};
  if (kind == "complete_tx" && arrivals == 0) return CompleteTxEffect{barrier, bytes};
  Fail("invalid completion action");
}

SourceOp Operation(const Json& value) {
  const auto kind = String(value.at("kind"));
  if (kind == "load" || kind == "store") {
    Fields(value, {"kind", "address", "bytes"});
    const auto address = Id<AddressId>(value.at("address"));
    const auto bytes = UInt<std::uint32_t>(value.at("bytes"));
    if (kind == "load") return ImmediateLoad{address, bytes};
    return ImmediateStore{address, bytes};
  }
  if (kind == "async_op") {
    Fields(value, {"kind", "async_id", "effects", "completion_requires"});
    AsyncOp result{Id<AsyncOpId>(value.at("async_id")), {},
                   References(value.at("completion_requires"))};
    for (const auto& effect : Array(value.at("effects"))) {
      Fields(effect, {"accesses", "completion", "predecessors"});
      AsyncEffect item{{}, Action(effect.at("completion")), References(effect.at("predecessors"))};
      const auto& encoded_accesses = Array(effect.at("accesses"));
      std::vector<MemoryEffect> accesses;
      accesses.reserve(encoded_accesses.size());
      for (const auto& access : encoded_accesses) accesses.push_back(Access(access));
      item.accesses = std::move(accesses);
      result.effects.push_back(std::move(item));
    }
    return result;
  }
  if (kind == "await") {
    Fields(value, {"kind", "requirements"});
    return Await{References(value.at("requirements"))};
  }
  if (kind == "sync_threads") {
    Fields(value, {"kind", "sync_id"});
    return SyncThreads{Id<SyncId>(value.at("sync_id"))};
  }
  if (kind == "sync_warp") {
    Fields(value, {"kind", "sync_id", "lane_mask"}, {"memory_ordering"});
    return SyncWarp{Id<SyncId>(value.at("sync_id")), UInt<std::uint32_t>(value.at("lane_mask")),
                    value.value("memory_ordering", true)};
  }
  const auto barrier = Id<BarrierLifetimeId>(value.at("barrier"));
  if (kind == "mbarrier.init") {
    Fields(value, {"kind", "barrier", "expected_arrivals"});
    return MBarrierInit{barrier, UInt<std::uint32_t>(value.at("expected_arrivals"))};
  }
  if (kind == "named_barrier.arrive" || kind == "named_barrier.sync") {
    Fields(value, {"kind", "barrier", "expected_arrivals"});
    const auto count = UInt<std::uint32_t>(value.at("expected_arrivals"));
    if (kind == "named_barrier.arrive") return NamedBarrierArrive{barrier, count};
    return NamedBarrierSync{barrier, count};
  }
  if (kind == "mbarrier.arrive") {
    Fields(value, {"kind", "barrier", "arrivals", "result"});
    return MBarrierArrive{barrier, UInt<std::uint32_t>(value.at("arrivals")), OptionalId<StateId>(value.at("result"))};
  }
  if (kind == "mbarrier.add_arrivals") {
    Fields(value, {"kind", "barrier", "arrivals"});
    return MBarrierAddArrivals{barrier, UInt<std::uint32_t>(value.at("arrivals"))};
  }
  if (kind == "mbarrier.arrive_expect_tx") {
    Fields(value, {"kind", "barrier", "arrivals", "bytes", "result"});
    return MBarrierArriveExpectTx{barrier, UInt<std::uint32_t>(value.at("arrivals")),
        UInt<std::uint64_t>(value.at("bytes")), OptionalId<StateId>(value.at("result"))};
  }
  if (kind == "mbarrier.expect_tx" || kind == "mbarrier.complete_tx") {
    Fields(value, {"kind", "barrier", "bytes"});
    const auto bytes = UInt<std::uint64_t>(value.at("bytes"));
    if (kind == "mbarrier.expect_tx") return MBarrierExpectTx{barrier, bytes};
    return MBarrierCompleteTx{barrier, bytes};
  }
  if (kind == "mbarrier.wait_state") {
    Fields(value, {"kind", "barrier", "state"});
    return MBarrierWaitState{barrier, Id<StateId>(value.at("state"))};
  }
  if (kind == "mbarrier.wait_parity") {
    Fields(value, {"kind", "barrier", "parity"});
    return MBarrierWaitParity{barrier, UInt<std::uint8_t>(value.at("parity"))};
  }
  Fail("unknown source operation: " + kind);
}

enum class RecordKind {
  kNone, kActor, kInstruction, kEdge, kCollective, kDependency, kBinding,
  kRequiredOrder, kVerificationLimitation, kSourceViolation,
};

void DecodeRecord(SourceArtifact& result, RecordKind kind, const Json& item,
                  SharedTextPool& texts);

struct ParsedArtifact {
  Json root;
  SourceArtifact records;
  std::string record_error;
  bool cached_records = false;
};

// Keep the small envelope and one record, not a second full program DOM.
class StrictJsonSax final : public nlohmann::json_sax<Json> {
 public:
  ParsedArtifact result;

  bool null() override { Append(nullptr); return true; }
  bool boolean(bool value) override { Append(value); return true; }
  bool number_integer(number_integer_t value) override { Append(value); return true; }
  bool number_unsigned(number_unsigned_t value) override { Append(value); return true; }
  bool number_float(number_float_t value, const string_t&) override { Append(value); return true; }
  bool string(string_t& value) override { Append(std::move(value)); return true; }
  bool binary(binary_t& value) override { Append(Json::binary(std::move(value))); return true; }
  bool start_object(std::size_t) override {
    const auto role = DiscardNext() ? Role::kDiscard : stack_.empty() ? Role::kRoot
        : stack_.back().role == Role::kRoot && stack_.back().key == "program" ? Role::kProgram
        : stack_.back().role == Role::kRoot && stack_.back().key == "memory_model" ? Role::kMemory
        : stack_.back().role == Role::kRoot && stack_.back().key == "metadata" ? Role::kMetadata
        : Role::kOther;
    stack_.push_back({Append(Json::object(), true), {}, role, RecordKind::kNone, {}});
    return true;
  }
  bool start_array(std::size_t) override {
    const auto kind = ArrayKind();
    const auto role = DiscardNext() ? Role::kDiscard : Role::kOther;
    stack_.push_back({Append(Json::array(), true), {}, role, kind, {}});
    return true;
  }
  bool key(string_t& value) override {
    auto& frame = stack_.back();
    const bool duplicate = frame.role == Role::kDiscard || frame.role == Role::kMetadata
        ? !frame.keys.insert(value).second : frame.value->contains(value);
    if (duplicate) Fail("duplicate JSON field: " + value);
    frame.key = std::move(value);
    return true;
  }
  bool end_object() override { EndContainer(); return true; }
  bool end_array() override { EndContainer(); return true; }
  bool parse_error(std::size_t, const std::string&, const Json::exception& error) override {
    Fail(error.what());
  }

 private:
  enum class Role { kOther, kRoot, kProgram, kMemory, kMetadata, kDiscard };
  struct Frame {
    Json* value;
    std::string key;
    Role role = Role::kOther;
    RecordKind records = RecordKind::kNone;
    std::set<std::string> keys;
  };
  std::vector<Frame> stack_;
  Json record_;
  SharedTextPool texts_;

  bool DiscardNext() const {
    return !stack_.empty() && (stack_.back().role == Role::kDiscard
        || (stack_.back().role == Role::kMetadata && stack_.back().key != "semantic_assumptions"
            && stack_.back().key != "frontend_assumptions"
            && stack_.back().key != "mma_operand_ordering"));
  }

  RecordKind ArrayKind() const {
    if (stack_.empty()) return RecordKind::kNone;
    const auto& frame = stack_.back();
    if (frame.role == Role::kProgram) {
      if (frame.key == "actors") return RecordKind::kActor;
      if (frame.key == "instructions") return RecordKind::kInstruction;
      if (frame.key == "edges") return RecordKind::kEdge;
      if (frame.key == "collectives") return RecordKind::kCollective;
      if (frame.key == "dependencies") return RecordKind::kDependency;
    }
    if (frame.role == Role::kMemory && frame.key == "bindings") return RecordKind::kBinding;
    if (frame.role == Role::kRoot) {
      if (frame.key == "required_orders") return RecordKind::kRequiredOrder;
      if (frame.key == "verification_limitations") return RecordKind::kVerificationLimitation;
      if (frame.key == "source_violations") return RecordKind::kSourceViolation;
    }
    return RecordKind::kNone;
  }

  void FinishRecord() {
    // Unsupported frontend envelopes retain their diagnostics-only API behavior.
    if (result.record_error.empty()) {
      if ((stack_.back().records == RecordKind::kInstruction || stack_.back().records == RecordKind::kEdge) &&
          record_.is_object() && record_.contains("cache_block")) result.cached_records = true;
      try { DecodeRecord(result.records, stack_.back().records, record_, texts_); }
      catch (const Json::exception& error) { result.record_error = error.what(); }
      catch (const std::invalid_argument& error) { result.record_error = error.what(); }
    }
    record_ = nullptr;
  }

  void EndContainer() {
    stack_.pop_back();
    if (!stack_.empty() && stack_.back().records != RecordKind::kNone) FinishRecord();
  }

  Json* Append(Json value, bool container = false) {
    if (stack_.empty()) { result.root = std::move(value); return &result.root; }
    if (DiscardNext()) return nullptr;
    auto& frame = stack_.back();
    if (frame.records != RecordKind::kNone) {
      record_ = std::move(value);
      if (!container) FinishRecord();
      return &record_;
    }
    if (frame.value->is_array()) {
      frame.value->push_back(std::move(value));
      return &frame.value->back();
    }
    auto& slot = (*frame.value)[frame.key];
    slot = std::move(value);
    return &slot;
  }
};

template <typename Input>
ParsedArtifact Parse(Input&& input) {
  StrictJsonSax reader;
  if (!Json::sax_parse(std::forward<Input>(input), &reader)) Fail("invalid JSON");
  return std::move(reader.result);
}

void DecodeRecord(SourceArtifact& result, RecordKind kind, const Json& item,
                  SharedTextPool& texts) {
  if ((kind == RecordKind::kInstruction || kind == RecordKind::kEdge) && item.contains("cache_block")) {
    Fields(item, {"cache_block", "width", "members", "cohort", "columns"});
    if (item.at("cache_block") != true) Fail("invalid cache block marker");
    std::optional<std::pair<std::uint64_t,std::uintptr_t>> cohort;
    if (!item.at("cohort").is_null()) {
      const auto& values = Array(item.at("cohort"));
      if (values.size() != 2) Fail("invalid cache cohort");
      cohort = {UInt<std::uint64_t>(values[0]), UInt<std::uintptr_t>(values[1])};
    }
    const auto restore = [&](auto& records, auto get, auto decodeDelta) {
      using Records = std::decay_t<decltype(records)>;
      std::vector<typename Records::CachedColumn> columns;
      for (const auto& run : Array(item.at("columns"))) {
        Fields(run, {"column", "begin", "count", "delta", "value"});
        if (run.at("value").contains("cache_block")) Fail("nested cache block");
        SourceArtifact one;
        DecodeRecord(one, kind, run.at("value"), texts);
        columns.push_back({UInt<std::size_t>(run.at("column")), UInt<std::size_t>(run.at("begin")),
            UInt<std::size_t>(run.at("count")), get(one), decodeDelta(run.at("delta"))});
      }
      records.appendCachedBlock(UInt<std::size_t>(item.at("width")),
          UInt<std::size_t>(item.at("members")), columns, cohort);
    };
    if (kind == RecordKind::kInstruction)
      restore(result.program.instructions, [](const auto& s) { return s.program.instructions.at(0); },
          [](const Json& d) {
            Fields(d, {"fields", "operands"});
            SourceRecordCodec::Delta delta;
            delta.fields = d.at("fields").get<std::array<std::int64_t,4>>();
            delta.operands = d.at("operands").get<std::vector<std::int64_t>>();
            return delta;
          });
    else restore(result.program.edges, [](const auto& s) { return s.program.edges.at(0); },
        [](const Json& d) { return d.get<SourceEdgeCodec::Delta>(); });
    return;
  }
  switch (kind) {
  case RecordKind::kActor:
    Fields(item, {"id", "cta", "warp", "lane"});
    result.program.actors.push_back({Id<ActorId>(item.at("id")), Id<CtaId>(item.at("cta")),
        Id<WarpId>(item.at("warp")), UInt<std::uint32_t>(item.at("lane"))});
    return;
  case RecordKind::kInstruction: {
    Fields(item, {"id", "actor", "location", "op"}, {"ptx_origin"});
    std::optional<PtxOrigin> origin;
    if (item.contains("ptx_origin") && !item.at("ptx_origin").is_null()) {
      const auto& value = item.at("ptx_origin");
      Fields(value, {"file", "instruction_index", "occurrence", "opcode"});
      origin = PtxOrigin{SharedText(value.at("file"), texts),
          UInt<std::uint32_t>(value.at("instruction_index")),
          UInt<std::uint32_t>(value.at("occurrence")), SharedText(value.at("opcode"), texts)};
      if (origin->file.empty() || origin->opcode.empty()) {
        Fail("PTX origin file and opcode must be nonempty");
      }
    }
    result.program.instructions.push_back({Id<InstId>(item.at("id")), Id<ActorId>(item.at("actor")),
        Location(item.at("location"), texts), Operation(item.at("op")), std::move(origin)});
    return;
  }
  case RecordKind::kEdge: {
    Fields(item, {"from_instruction", "to_instruction", "kind"});
    const auto kind = String(item.at("kind"));
    if (kind != "program_order" && kind != "complete_before_issue") Fail("unknown source edge kind");
    result.program.edges.push_back({Id<InstId>(item.at("from_instruction")), Id<InstId>(item.at("to_instruction")),
        kind == "program_order" ? SourceEdgeKind::kProgramOrder : SourceEdgeKind::kCompleteBeforeIssue});
    return;
  }
  case RecordKind::kCollective: {
    Fields(item, {"sync_id", "expected_participants", "lane_mask"});
    CollectiveSpec spec{Id<SyncId>(item.at("sync_id")), {}, std::nullopt};
    for (const auto& actor : Array(item.at("expected_participants"))) spec.expected_participants.push_back(Id<ActorId>(actor));
    if (!item.at("lane_mask").is_null()) spec.lane_mask = UInt<std::uint32_t>(item.at("lane_mask"));
    result.program.collectives.push_back(std::move(spec));
    return;
  }
  case RecordKind::kDependency:
    Fields(item, {"from_ref", "to_ref"});
    result.program.dependencies.push_back({Reference(item.at("from_ref")), Reference(item.at("to_ref"))});
    return;
  case RecordKind::kBinding:
    Fields(item, {"address", "address_space", "allocation", "offset"}, {"alias_scope", "expression"});
    result.addresses.push_back({Id<AddressId>(item.at("address")), String(item.at("address_space")),
        String(item.at("allocation")), UInt<std::uint64_t>(item.at("offset")),
        item.contains("alias_scope") ? String(item.at("alias_scope")) : "",
        item.contains("expression") && !item.at("expression").is_null()
            ? item.at("expression").dump() : ""});
    return;
  case RecordKind::kRequiredOrder: {
    Fields(item, {"kind", "from_ref", "to_ref", "message"}, {"from_effects", "target_domain"});
    result.required_orders.push_back({SharedText(item.at("kind"), texts), Reference(item.at("from_ref")),
        Reference(item.at("to_ref")), SharedText(item.at("message"), texts)});
    if (item.contains("from_effects") && !item.at("from_effects").is_null()) {
      auto& order = result.required_orders.back();
      if (order.from.milestone != Milestone::kComplete)
        Fail("grouped effect order requires the complete milestone");
      std::vector<std::uint32_t> effects;
      std::set<std::uint32_t> seen;
      for (const auto& value : Array(item.at("from_effects"))) {
        const auto index = UInt<std::uint32_t>(value);
        if (!seen.insert(index).second) Fail("duplicate required effect index");
        effects.push_back(index);
      }
      order.from_effects = std::move(effects);
    }
    if(item.contains("target_domain") && !item.at("target_domain").is_null()) {
      std::vector<MilestoneRef> targets;
      std::set<std::tuple<InstId,Milestone,std::uint32_t>> seen;
      for(const auto& value:Array(item.at("target_domain"))) {
        auto ref=Reference(value);
        if(!seen.emplace(ref.instruction,ref.milestone,ref.effect_index).second)
          Fail("duplicate required target endpoint");
        targets.push_back(ref);
      }
      if(targets.empty() || targets.front()!=result.required_orders.back().to)
        Fail("required target domain must start with to_ref");
      result.required_orders.back().target_domain=std::move(targets);
    }
    return;
  }
  case RecordKind::kVerificationLimitation: {
      Fields(item, {"code", "message"});
      VerificationLimitation limitation{String(item.at("code")), String(item.at("message"))};
      if (limitation.code.empty() || limitation.message.empty()) Fail("empty verification limitation");
      result.verification_limitations.push_back(std::move(limitation));
      return;
  }
  case RecordKind::kSourceViolation: {
    Fields(item, {"code", "message"});
    SourceViolation violation{String(item.at("code")), String(item.at("message"))};
    if (violation.code.empty() || violation.message.empty()) Fail("empty source violation");
    result.source_violations.push_back(std::move(violation));
    return;
  }
  case RecordKind::kNone: Fail("invalid streamed record kind");
  }
}

void ValidateRequiredEffectGroups(const SourceArtifact& source, bool check_memory = true) {
  std::unordered_map<std::uint32_t, std::optional<AsyncOp>> operations;
  for (const auto& order : source.required_orders)
    if (order.from_effects) operations.emplace(order.from.instruction.value(), std::nullopt);
  if (operations.empty()) return;
  for (std::size_t i = 0; i < source.program.instructions.size(); ++i) {
    const auto found = operations.find(source.program.instructions.id(i).value());
    if (found == operations.end()) continue;
    const auto instruction = source.program.instructions[i];
    if (const auto* op = std::get_if<AsyncOp>(&instruction.op)) found->second = *op;
  }
  std::unordered_map<std::uint32_t, std::string_view> spaces;
  std::unordered_set<std::uint32_t> checked_empty;
  for (const auto& order : source.required_orders) {
    if (!order.from_effects) continue;
    const auto& operation = operations.at(order.from.instruction.value());
    if (!operation) Fail("grouped effect order requires an async_op instruction");
    for (const auto index : *order.from_effects)
      if (index >= operation->effects.size()) Fail("required effect index out of range");
    if (!check_memory || !order.from_effects->empty() ||
        !checked_empty.insert(order.from.instruction.value()).second) continue;
    if (spaces.empty())
      for (const auto& binding : source.addresses)
        spaces.emplace(binding.address.value(), binding.address_space);
    for (const auto& effect : operation->effects) {
      for (const auto& access : effect.accesses) {
        const auto* address = std::get_if<AddressId>(&access.endpoint);
        if (!address || access.bytes == 0) continue;
        const auto binding = spaces.find(address->value());
        if (binding == spaces.end()) Fail("empty effect group has an unknown memory binding");
        if (binding->second == "tensor") Fail("empty effect group omits a nonzero TMEM footprint");
      }
    }
  }
}

SourceArtifact Decode(ParsedArtifact parsed) {
  const auto& root = parsed.root;
  Fields(root, {"schema", "status", "program", "memory_model", "diagnostics", "metadata", "required_orders"},
         {"verification_limitations", "source_violations"});
  if (String(root.at("schema")) != "spverify.source-program/v1" &&
      String(root.at("schema")) != "spverify.source-cache/v1") Fail("unsupported schema");
  if (parsed.cached_records && String(root.at("schema")) != "spverify.source-cache/v1")
    Fail("compressed blocks require the internal source-cache schema");
  if (String(root.at("status")) != "ready" || !Array(root.at("diagnostics")).empty()) {
    Fail("frontend did not produce a complete supported program");
  }
  if (!root.at("metadata").is_object()) Fail("metadata must be an object");
  const auto& metadata = root.at("metadata");
  parsed.records.frontend_assumptions_json = metadata.value("frontend_assumptions", Json::object()).dump();
  parsed.records.mma_operand_ordering_json = metadata.value("mma_operand_ordering", Json::object()).dump();
  if (metadata.contains("semantic_assumptions")) {
    const auto& flags = metadata.at("semantic_assumptions");
    if (!flags.is_object()) Fail("semantic_assumptions must be an object");
    PtxAssumptions overrides;
    for (const auto& [name, value] : flags.items()) {
      if (!value.is_boolean()) Fail("semantic assumption flags must be Boolean");
      overrides.emplace(name, value.get<bool>());
    }
    OverridePtxAssumptions(parsed.records.semantic_assumptions, overrides);
  }
  const auto& program = root.at("program");
  Fields(program, {"actors", "instructions", "edges", "collectives", "dependencies"});
  for (const auto* name : {"actors", "instructions", "edges", "collectives", "dependencies"}) {
    (void)Array(program.at(name));
  }
  const auto& memory = root.at("memory_model");
  Fields(memory, {"kind", "bindings"}, {"address_invariants", "solver"});
  const auto kind = String(memory.at("kind"));
  if (kind != "constant" && kind != "symbolic") Fail("unsupported memory model");
  (void)Array(memory.at("bindings"));
  if (memory.contains("address_invariants")) {
    if (kind != "symbolic") Fail("address invariants require the symbolic memory model");
    parsed.records.address_invariants_json = Json(Array(memory.at("address_invariants"))).dump();
  }
  if (memory.contains("solver")) {
    const auto& solver = memory.at("solver");
    Fields(solver, {"timeout_ms"});
    parsed.records.address_solver.timeout_ms = UInt<unsigned>(solver.at("timeout_ms"));
    if (parsed.records.address_solver.timeout_ms == 0) Fail("solver timeout_ms must be positive");
  }
  (void)Array(root.at("required_orders"));
  if (root.contains("verification_limitations")) (void)Array(root.at("verification_limitations"));
  if (root.contains("source_violations")) (void)Array(root.at("source_violations"));
  if (!parsed.record_error.empty()) Fail(parsed.record_error);
  ValidateRequiredEffectGroups(parsed.records);
  return std::move(parsed.records);
}

EventId Resolve(const OriginMap& origins, MilestoneRef ref) {
  if (const auto event = origins.resolve(ref)) return *event;
  Fail("required order references an absent milestone");
}
EventId Resolve(const LoweredProgram& program, MilestoneRef ref) {
  return Resolve(program.origins(),ref);
}

Json Source(const MemoryAccess& access) {
  Json result = {{"instruction", access.instruction.value()}, {"actor", access.actor.value()},
      {"event", access.event.value()}, {"mode", access.mode == AccessMode::kRead ? "read" : "write"},
      {"bytes", access.bytes ? Json(*access.bytes) : Json(nullptr)}, {"effect_index", access.effect_index}, {"access_index", access.access_index},
      {"location", {{"file", access.location.file}, {"line", access.location.line}, {"column", access.location.column}}}};
  if (access.atomic_scope)
    result["atomic_scope"] = *access.atomic_scope == AtomicScope::kCta ? "cta" :
                              *access.atomic_scope == AtomicScope::kGpu ? "gpu" : "sys";
  if (const auto* address = std::get_if<AddressId>(&access.address)) {
    result["endpoint"] = {{"kind", "memory"}, {"address", address->value()}};
  } else {
    const auto& reg = std::get<RegisterAddress>(access.address);
    result["endpoint"] = {{"kind", "register"}, {"actor", reg.actor.value()}, {"name", reg.name}};
  }
  return result;
}
}  // namespace

SourceArtifact ReadSourceArtifact(std::string_view text) {
  try { return Decode(Parse(text)); }
  catch (const Json::exception& error) { throw std::invalid_argument(error.what()); }
}

SourceArtifact ReadSourceArtifact(std::istream& input) {
  try { return Decode(Parse(input)); }
  catch (const Json::exception& error) { throw std::invalid_argument(error.what()); }
}

namespace {
Json LocationJson(const SourceLocation& location) {
  return {{"file", location.file}, {"line", location.line}, {"column", location.column}};
}

template <typename Code, std::size_t N>
const char* CodeName(Code code, const char* const (&names)[N]) {
  const auto index = static_cast<std::size_t>(code);
  return index < N ? names[index] : "unknown";
}

const char* CodeName(SimulationErrorCode code) {
  static constexpr const char* names[] = {
      "malformed_program", "deadlock", "use_before_initialization", "duplicate_initialization",
      "zero_expected_arrivals", "arrival_underflow", "transaction_overflow", "transaction_underflow",
      "phase_overflow", "missing_successful_wait_handoff", "missing_state_token", "state_barrier_mismatch",
      "stale_state_wait", "invalid_parity", "stale_parity_wait", "wait_done_before_issue",
      "wait_target_incomplete", "collective_mismatch", "unexpected_urgent_event", "arrival_overflow",
      "named_barrier_count_mismatch"};
  return CodeName(code, names);
}

const char* CodeName(PhaseBoundaryErrorCode code) {
  static constexpr const char* names[] = {
      "unknown_phase", "wrong_phase_kind", "invalid_graph", "invalid_effect",
      "arrival_arithmetic_overflow", "transaction_arithmetic_overflow", "arrival_balance_mismatch",
      "transaction_balance_mismatch", "negative_transaction_prefix", "premature_phase_completion",
      "open_phase_arrival_overflow", "open_phase_not_strictly_positive", "ideal_solver_failure",
      "negative_arrival_prefix", "arrival_prefix_overflow"};
  return CodeName(code, names);
}

const char* CodeName(WaitCheckErrorCode code) {
  static constexpr const char* names[] = {
      "unknown_event", "not_wait_issue", "missing_wait_target", "missing_wait_origin",
      "missing_wait_milestones", "mismatched_wait_done", "missing_initialization",
      "initialization_not_ordered", "unknown_target_phase", "target_phase_incomplete",
      "missing_target_completion", "target_completion_not_ordered_before_done",
      "issue_not_ordered_before_done", "missing_previous_phase", "previous_phase_not_ordered_before_issue",
      "missing_previous_phase_completion", "phase_number_overflow", "missing_next_phase",
      "issue_not_ordered_before_next_phase", "missing_state_definition", "invalid_state_definition",
      "state_barrier_mismatch", "missing_state_token", "state_phase_mismatch",
      "state_definition_not_ordered_before_issue", "parity_mismatch", "happens_before_graph_mutation_failed"};
  return CodeName(code, names);
}

const char* CodeName(LifecycleCheckErrorCode code) {
  static constexpr const char* names[] = {"missing_previous_phase", "previous_phase_incomplete",
      "invalid_positive_arrival", "missing_successful_wait_handoff"};
  return CodeName(code, names);
}

const char* CodeName(HappensBeforeGraphMutationErrorCode code) {
  static constexpr const char* names[] = {"frozen", "missing_event", "invalid_reason", "cycle",
      "duplicate_phase_completion", "event_id_overflow"};
  return CodeName(code, names);
}

Json PhaseJson(PhaseKey phase) {
  return {{"barrier", phase.barrier.value()}, {"generation", phase.phase}};
}

// Bound witness lists and resolve only the event IDs included in the report.
class RejectionReport final {
 public:
  void Write(Json& report, const RejectionWitness& witness, const RejectionContext* context) {
    static constexpr const char* names[] = {"reference_deadlock", "semantic_violation",
        "initialization_order", "previous_phase_separation", "phase_boundary",
        "wait_condition", "barrier_lifecycle", "hb_graph_mutation", "hb_cycle"};
    report["rejection_kind"] = names[witness.index()];
    report["rejection"] = Visit(witness);
    report["message"] = report["rejection"].at("message");
    if (context && context->counterexample) {
      const auto& execution = *context->counterexample;
      Json replay{{"candidate_order", Events(execution.candidate_order)},
          {"replayed_candidate_order", Events(execution.replayed_candidate_order)},
          {"executed_prefix", Events(execution.executable_prefix.executedEvents())},
          {"unexecuted_candidate", Events(execution.unexecuted_candidate)},
          {"candidate_order_preserved", execution.candidate_order_preserved}};
      if (execution.stall) replay["first_unexecuted"] = Event(execution.stall->first_unexecuted);
      if (execution.stop) replay["stop"] = Visit(*execution.stop);
      report["counterexample"] = std::move(replay);
    }
    report["source_event_scope"] = "reported_witness_events";
    report["witness_event_list_limit"] = kEventLimit;
    report["witness_event_lists_truncated"] = truncated_;
    report["provenance_events_total"] = context ? context->provenance.events.size() : 0;
    report["source_events"] = Json::array();
    report["synthetic_events"] = Json::array();
    report["unmapped_witness_events"] = Json::array();
    for (const auto event : events_) {
      const auto* provenance = context ? context->provenance.find(event) : nullptr;
      if (!provenance) {
        report["unmapped_witness_events"].push_back(event.value());
      } else if (const auto* origin = std::get_if<SourceEventProvenance>(provenance)) {
        report["source_events"].push_back({{"event", event.value()},
            {"instruction", origin->instruction.value()}, {"file", origin->location.file},
            {"line", origin->location.line}, {"column", origin->location.column}});
      } else {
        const auto& synthetic = std::get<SyntheticEventProvenance>(*provenance);
        static constexpr const char* kinds[] = {"collective_completion", "mbarrier_phase_completion", "internal_derived"};
        Json item{{"event", event.value()}, {"kind", CodeName(synthetic.kind, kinds)}};
        if (synthetic.sync_id) item["sync_id"] = synthetic.sync_id->value();
        if (synthetic.phase) item["phase"] = PhaseJson(*synthetic.phase);
        report["synthetic_events"].push_back(std::move(item));
      }
    }
  }

 private:
  static constexpr std::size_t kEventLimit = 32;
  std::set<EventId> events_;
  bool truncated_ = false;

  std::uint32_t Event(EventId event) { events_.insert(event); return event.value(); }
  Json Event(std::optional<EventId> event) { return event ? Json(Event(*event)) : Json(nullptr); }

  template <typename Range>
  Json Events(const Range& events) {
    Json items = Json::array();
    for (const auto event : events) {
      if (items.size() == kEventLimit) break;
      items.push_back(Event(event));
    }
    const bool truncated = items.size() < events.size();
    truncated_ |= truncated;
    return {{"events", std::move(items)}, {"total", events.size()}, {"truncated", truncated}};
  }

  template <typename... Types>
  Json Visit(const std::variant<Types...>& value) {
    return std::visit([&](const auto& item) { return Detail(item); }, value);
  }

  Json Detail(const ReferenceDeadlock& value) {
    return {{"message", "The reference execution cannot advance; events remain blocked."},
        {"executed", Events(value.executed)}, {"blocked", Events(value.blocked)}};
  }
  Json Detail(const SemanticViolation& value) {
    Json result{{"code", CodeName(value.code)}, {"message", value.message}, {"event", Event(value.event)}};
    if (value.barrier) result["barrier"] = value.barrier->value();
    if (value.sync_id) result["sync_id"] = value.sync_id->value();
    if (value.state) result["state"] = value.state->value();
    return result;
  }
  Json Detail(const InitializationOrderWitness& value) {
    return {{"message", "A barrier effect is not ordered after its initialization."},
        {"phase", PhaseJson(value.phase)}, {"effect", Event(value.effect)},
        {"initialization", Event(value.initialization)}, {"counterexample_schedule", Events(value.counterexample_schedule)}};
  }
  Json Detail(const PreviousPhaseSeparationWitness& value) {
    return {{"message", "A barrier effect is not ordered after all previous-phase effects."},
        {"phase", PhaseJson(value.phase)}, {"effect", Event(value.effect)},
        {"previous_phase", PhaseJson(value.previous_phase)},
        {"unordered_previous_effects", Events(value.unordered_previous_effects)},
        {"counterexample_schedule", Events(value.counterexample_schedule)}};
  }
  Json Detail(const PhaseBoundaryError& value) {
    Json result = Visit(value.witness);
    result["code"] = CodeName(value.code);
    result["message"] = std::string("Phase boundary check failed: ") + CodeName(value.code) + ".";
    return result;
  }
  Json Detail(const std::monostate&) { return Json::object(); }
  Json Detail(const PhaseBalanceWitness& value) {
    return {{"phase", PhaseJson(value.phase)}, {"expected_arrivals", value.expected_arrivals},
        {"actual_arrivals", value.actual_arrivals}, {"transaction_total", value.transaction_total},
        {"arrival_increment_total", value.arrival_increment_total}};
  }
  Json Detail(const PhaseIdealWitness& value) {
    Json vertices = Json::array();
    for (auto i = value.ideal.find_first(); i != DynamicBitset::npos && vertices.size() < kEventLimit;
         i = value.ideal.find_next(i)) vertices.push_back(i);
    const auto total = value.ideal.count();
    const bool truncated = vertices.size() < total;
    truncated_ |= truncated;
    return {{"phase", PhaseJson(value.phase)}, {"arrival_total", value.arrival_total},
        {"transaction_total", value.transaction_total}, {"arrival_increment_total", value.arrival_increment_total},
        {"omitted_effect", Event(value.omitted_effect)},
        {"ideal", {{"local_vertices", std::move(vertices)}, {"total", total}, {"truncated", truncated}}},
        {"topological_schedule", Events(value.topological_schedule)}};
  }
  Json Detail(const PhaseSolverWitness& value) {
    static constexpr const char* names[] = {"ok", "infeasible", "invalid_input", "arithmetic_overflow", "flow_failure"};
    return {{"phase", PhaseJson(value.phase)}, {"solver_status", CodeName(value.status, names)}};
  }
  Json Detail(const WaitCheckError& value) {
    Json result = Visit(value.witness);
    result["code"] = CodeName(value.code);
    result["message"] = std::string("Wait check failed: ") + CodeName(value.code) + ".";
    return result;
  }
  Json Detail(const WaitEventWitness& value) {
    Json result{{"issue", Event(value.issue)}, {"related_event", Event(value.related_event)}};
    if (value.phase) result["phase"] = PhaseJson(*value.phase);
    return result;
  }
  Json Detail(const WaitReachabilityWitness& value) {
    return {{"issue", Event(value.issue)}, {"phase", PhaseJson(value.phase)}, {"required_events", Events(value.required_events)}};
  }
  Json Detail(const WaitStateWitness& value) {
    Json result{{"issue", Event(value.issue)}, {"state", value.state.value()},
        {"target", PhaseJson(value.target)}, {"definition", Event(value.definition)}};
    if (value.token) result["token"] = PhaseJson({value.token->barrier, value.token->phase});
    return result;
  }
  Json Detail(const WaitParityWitness& value) {
    return {{"issue", Event(value.issue)}, {"target", PhaseJson(value.target)},
        {"actual_parity", value.actual}, {"expected_parity", value.expected}};
  }
  Json Detail(const LifecycleCheckError& value) {
    return {{"code", CodeName(value.code)},
        {"message", std::string("Barrier lifecycle check failed: ") + CodeName(value.code) + "."},
        {"phase", PhaseJson(value.witness.phase)}, {"positive_arrival", Event(value.witness.positive_arrival)},
        {"previous_phase", PhaseJson(value.witness.previous_phase)},
        {"candidate_waits", Events(value.witness.candidate_waits)}};
  }
  Json Detail(const HappensBeforeGraphMutationError& value) {
    Json result{{"code", CodeName(value.code)},
        {"message", std::string("Happens-before graph update failed: ") + CodeName(value.code) + "."},
        {"from", Event(value.from)}, {"to", Event(value.to)}};
    if (value.phase) result["phase"] = PhaseJson(*value.phase);
    return result;
  }
  Json Detail(const FinalHappensBeforeCycleWitness& value) {
    return {{"message", "The constructed happens-before graph contains a cycle."},
        {"partial_topological_order", Events(value.partial_topological_order)}};
  }
};

std::string RenderReport(Json& report) {
  const auto status = report.value("status", "");
  report["memory_analysis_complete"] = report.contains("memory_analysis") &&
      (status == "race_free" || status == "potential_race") &&
      report.value("unresolved_count", std::size_t{0}) == 0;
  Json usage = Json::array();
  const auto flags = report.value("semantic_assumptions", Json::object());
  const auto frontend = report.value("frontend_assumptions", Json::object());
  for (const auto& entry : AssumptionPool()) {
    const bool enabled = flags.value(entry.key, false);
    Json triggered = enabled ? Json(nullptr) : Json(false);
    if (enabled && entry.key == "duplicate_tma_store_benign" && report.contains("warning_count"))
      triggered = report["warning_count"] > 0;
    if (enabled && entry.key == "representative_election" && frontend.contains("election_uses"))
      triggered = frontend["election_uses"] > 0;
    if (enabled && entry.key == "clc_always_fails" && frontend.contains("clc_uses"))
      triggered = frontend["clc_uses"] > 0;
    if (enabled && entry.key == "representative_branches" && frontend.contains("representative_branch_uses"))
      triggered = frontend["representative_branch_uses"] > 0;
    if (enabled && entry.key == "bounded_computation_loops" && frontend.contains("bounded_loop_uses"))
      triggered = frontend["bounded_loop_uses"] > 0;
    if (enabled && entry.key == "order_independent_atomics" && frontend.contains("atomic_uses"))
      triggered = frontend["atomic_uses"] > 0;
    usage.push_back({{"label", entry.label}, {"key", entry.key}, {"category", "analysis assumption"},
        {"text", entry.text}, {"enabled", enabled}, {"triggered", triggered}});
  }
  report["assumptions"] = std::move(usage);
  report["assumption_reporting"] = "triggered=null means applicability was not measured or analysis was not reached";
  if (!report.contains("stage")) {
    if (status == "not_wellsynced") report["stage"]="synchronization_check";
    else if (status == "potential_race" || status == "memory_analysis_inconclusive") report["stage"]="memory_check";
    else if (status == "invalid_event_view" || status == "event_view_unsupported") report["stage"]="event_view_validation";
    else if (status == "shared_memory_value_unproven") report["stage"]="shared_memory_value_check";
    else if (status == "unsafe_lifecycle") report["stage"]="lifecycle_check";
    else if (status == "internal_error" || status == "allocation_failed") report["stage"]="verification";
  }
  return report.dump();
}

using LifecycleDomain = std::map<std::size_t, MemberSet>;
struct LifecycleAnswer {
  LifecycleDomain failed;
  std::size_t count() const {
    std::size_t n=0;for(const auto& [family,members]:failed)n+=members.count();return n;
  }
};

// No concrete source/target pair list. Source endpoints and shared target
// domains are resolved once; only family partitions form query products.
std::vector<LifecycleAnswer> CheckRequiredOrders(
    FamilyRelationQuery& reachability,const SourceArtifact& input,
    const OriginMap& lowered,Json* statistics=nullptr) {
  ProfileScope assembly("lifecycle.relation_requests");
  const auto& layout=reachability.families();
  const auto add=[&](LifecycleDomain& domain,EventId event) {
    const auto [family,member]=layout.locate(event);
    auto [it,fresh]=domain.try_emplace(family,layout.families()[family].count);
    it->second.set(member);
  };
  std::map<const void*,LifecycleDomain> target_cache;
  std::map<std::pair<InstId,const void*>,LifecycleDomain> effect_cache;
  std::vector<LifecycleAnswer> answers(input.required_orders.size());
  std::size_t endpoint_reads=0,logical_orders=0,requests=0,owner_bindings=0;
  struct Owner {std::size_t order;MemberSet left;};
  struct Pending {
    FamilyRelationBatch batch;
    std::vector<std::vector<Owner>> owners;
    std::map<std::pair<std::size_t,std::size_t>,std::vector<std::size_t>> index;
  };
  std::map<std::size_t,Pending> pending;
  const auto enqueue=[&](std::size_t owner,const LifecycleDomain& sources,
                         const LifecycleDomain& targets) {
    for(const auto& [source,left]:sources) {
      auto [entry,fresh]=pending.try_emplace(source);
      auto& p=entry->second;p.batch.source=source;
      for(const auto& [target,right]:targets) {
        if(right.empty())continue;
        auto& candidates=p.index[{target,right.hash()}];
        std::size_t q=SIZE_MAX;
        for(auto candidate:candidates)
          if(p.batch.requests[candidate].requested.front().right==right){q=candidate;break;}
        if(q==SIZE_MAX) {
          q=p.batch.requests.size();candidates.push_back(q);
          p.batch.requests.push_back({target,{MemberRelation::Rectangle(left,right)}});
          p.owners.emplace_back();++requests;assembly.nodes=requests;
        } else p.batch.requests[q].requested.front().left.unite(left);
        p.owners[q].push_back({owner,left});++owner_bindings;assembly.edges=owner_bindings;
      }
    }
  };
  const auto run=[&] {
    std::vector<FamilyRelationBatch> batch;
    std::vector<Pending*> metadata;
    const auto flush=[&] {
      if(batch.empty())return;
      // Exactly the race engine, but lifecycle requires strict forward order.
      reachability.checkOrderBatch(batch,true,true);
      for(std::size_t b=0;b<batch.size();++b)
        for(std::size_t q=0;q<batch[b].requests.size();++q) {
          const auto& request=batch[b].requests[q];
          const auto& right=request.requested.front().right;
          for(const auto& owner:metadata[b]->owners[q]) {
            const auto failed=request.uncoveredTargets(&owner.left);
            if(!failed.empty()) {
              auto [it,fresh]=answers[owner.order].failed.try_emplace(request.target,right.extent());
              it->second.unite(failed);
            }
          }
        }
      batch.clear();metadata.clear();
    };
    for(auto& [source,p]:pending) {
      if(batch.size()==64)flush();
      batch.push_back(std::move(p.batch));metadata.push_back(&p);
    }
    flush();pending.clear();
  };
  for(std::size_t i=0;i<input.required_orders.size();++i) {
    const auto& order=input.required_orders[i];
    LifecycleDomain singleton;
    const LifecycleDomain* targets=&singleton;
    if(order.target_domain) {
      auto [it,fresh]=target_cache.try_emplace(order.target_domain.storage_identity());
      if(fresh)for(auto ref:order.targets()){add(it->second,Resolve(lowered,ref));++endpoint_reads;}
      targets=&it->second;
    } else {add(singleton,Resolve(lowered,order.to));++endpoint_reads;}
    logical_orders+=order.targets().size();
    LifecycleDomain source;add(source,Resolve(lowered,order.from));++endpoint_reads;
    enqueue(i,source,*targets);
  }
  ProfileScope domain_profile("lifecycle.request_domains");
  domain_profile.nodes=input.required_orders.size();domain_profile.edges=logical_orders;
  domain_profile.pruned=endpoint_reads;
  run();
  for(std::size_t i=0;i<input.required_orders.size();++i) {
    const auto& order=input.required_orders[i];
    if(answers[i].failed.empty() || !order.from_effects)continue;
    auto [it,fresh]=effect_cache.try_emplace(
        std::make_pair(order.from.instruction,order.from_effects.storage_identity()));
    if(fresh)for(auto effect:*order.from_effects) {
      add(it->second,Resolve(lowered,{order.from.instruction,Milestone::kEffect,effect}));++endpoint_reads;
    }
    auto targets=std::move(answers[i].failed);answers[i].failed.clear();
    // An empty certified selection is a vacuous conjunction, as before.
    enqueue(i,it->second,targets);
  }
  run();
  assembly.nodes=requests;assembly.edges=owner_bindings;
  if(statistics)*statistics={{"stored_orders",input.required_orders.size()},
      {"logical_orders",logical_orders},{"family_requests",requests},
      {"owner_bindings",owner_bindings},{"endpoint_reads",endpoint_reads},
      {"concrete_pair_materializations",0}};
  return answers;
}

std::optional<std::string> ValidateSharedConstantWrites(const SourceArtifact& input,
    const PassCertificate& certificate,const OriginMap& lowered,
    HbReachability& hb) {
  ProfileScope profile("frontend_values.interference");
  std::unordered_set<std::uint32_t> wanted;
  for(std::size_t i=0;i<input.required_orders.size();++i)
    if(input.required_orders[i].kind.view()=="shared_constant_read_from") {
      const auto& order=input.required_orders[i];
      wanted.insert(Resolve(lowered,order.from).value());
      for(auto target:order.targets())wanted.insert(Resolve(lowered,target).value());
    }
  if(wanted.empty())return {};
  std::unordered_map<std::uint32_t,const ConstantAddressBinding*> bindings;
  for(const auto& binding:input.addresses)bindings.emplace(binding.address.value(),&binding);
  struct Span { std::string allocation; std::uint64_t begin,end; };
  auto span=[&](const MemoryAccess& access)->std::optional<Span> {
    const auto* id=std::get_if<AddressId>(&access.address);
    if(!id || !access.bytes || !*access.bytes)return {};
    const auto found=bindings.find(id->value());
    if(found==bindings.end() || found->second->address_space!="shared" ||
        !found->second->expression.empty() || *access.bytes>UINT64_MAX-found->second->offset)return {};
    return Span{found->second->allocation,found->second->offset,found->second->offset+*access.bytes};
  };
  std::map<EventId,std::vector<MemoryAccess>> endpoints;
  auto scan=VisitMemoryAccesses(certificate.happensBeforeGraph.events(),input.program,
      [&](const MemoryAccess& access)->std::optional<MemoryAnalysisError> {
        if(wanted.contains(access.event.value()) && span(access))endpoints[access.event].push_back(access);
        return {};
      });
  if(!scan)return scan.error().message;
  struct Query { std::uint64_t begin,end; EventId write,read; };
  struct Index { std::map<std::uint64_t,std::vector<Query>> starts; std::uint64_t max_width=0; };
  std::map<std::string,Index> queries;
  for(std::size_t i=0;i<input.required_orders.size();++i) {
    if(input.required_orders[i].kind.view()!="shared_constant_read_from")continue;
    const auto& order=input.required_orders[i];
    const auto write=Resolve(lowered,order.from);
    for(auto target:order.targets()) {
    const auto read=Resolve(lowered,target);bool found=false;
    for(const auto& a:endpoints[write])if(a.mode==AccessMode::kWrite)
      for(const auto& b:endpoints[read])if(b.mode==AccessMode::kRead) {
        const auto x=*span(a),y=*span(b);
        const auto begin=std::max(x.begin,y.begin),end=std::min(x.end,y.end);
        if(x.allocation!=y.allocation || begin>=end)continue;
        auto& index=queries[x.allocation]; index.max_width=std::max(index.max_width,end-begin);
        index.starts[begin].push_back({begin,end,write,read}); found=true;
      }
    if(!found)return "A shared constant's supplying write and read lack resolved overlapping shared-memory footprints.";
    }
  }
  std::map<std::string,std::set<std::string>> queried_aliases;
  for(const auto& binding:input.addresses)
    if(binding.address_space=="shared" && !binding.alias_scope.empty() && queries.contains(binding.allocation))
      queried_aliases[binding.alias_scope].insert(binding.allocation);
  // Each candidate write must be ordered before the chosen source or after its read. The checks are
  // batched (reachesBatch answers up to 64 sources per traversal) and flushed in visit order, so the
  // reported failure is the one the first failing check, or an earlier structural error, gives.
  static const std::string intervene="Another shared-memory write may intervene between the chosen constant source and its read.";
  struct Pending { EventId access,write,read; };
  std::vector<Pending> pending;
  const auto flush=[&]()->bool {
    if(pending.empty())return true;
    std::vector<HbQuery> before;before.reserve(pending.size());
    for(const auto& p:pending)before.push_back({p.access,p.write});
    const auto ordered=hb.reachesBatch(before);
    std::vector<HbQuery> after;
    for(std::size_t i=0;i<pending.size();++i)if(!ordered[i])after.push_back({pending[i].read,pending[i].access});
    pending.clear();
    if(after.empty())return true;
    const auto answers=hb.reachesBatch(after);
    return std::all_of(answers.begin(),answers.end(),[](auto answer){return answer!=0;});
  };
  const auto fail=[&](const MemoryAccess& access,const char* message)->MemoryAnalysisError {
    return {flush()?std::string(message):intervene,access.event,access.address};
  };
  scan=VisitMemoryAccesses(certificate.happensBeforeGraph.events(),input.program,
      [&](const MemoryAccess& access)->std::optional<MemoryAnalysisError> {
        if(access.mode!=AccessMode::kWrite || access.bytes==0)return {};
        const auto* id=std::get_if<AddressId>(&access.address);
        if(!id)return {};
        const auto binding=bindings.find(id->value());
        if(binding==bindings.end())return fail(access,"Shared constant validation found an unbound write.");
        if(binding->second->address_space!="shared")return {};
        const auto aliases=queried_aliases.find(binding->second->alias_scope);
        if(binding->second->allocation=="unresolved" || (aliases!=queried_aliases.end() &&
            (aliases->second.size()>1 || !aliases->second.contains(binding->second->allocation))))
          return fail(access,"Shared constant validation does not support writes with unresolved shared-memory aliasing.");
        const auto index=queries.find(binding->second->allocation);
        if(index==queries.end())return {};
        const auto region=span(access);
        if(!region)return fail(access,"Shared constant validation requires a constant footprint for a potentially interfering write.");
        auto it=index->second.starts.lower_bound(region->begin>=index->second.max_width?
            region->begin-index->second.max_width+1:0);
        for(;it!=index->second.starts.end() && it->first<region->end;++it)
          for(const auto& query:it->second) {
            if(query.end<=region->begin || access.event==query.write)continue;
            pending.push_back({access.event,query.write,query.read});
          }
        if(pending.size()>=(std::size_t(1)<<16) && !flush())
          return MemoryAnalysisError{intervene,access.event,access.address};
        return {};
      });
  if(!scan)return scan.error().message;
  if(!flush())return intervene;
  return {};
}

// Cross-CTA class mode: per-occurrence evidence of the quotient program stands
// for every unit of its CTA's class.
class QuotientEvidence {
 public:
  QuotientEvidence(UnitQuotient& quotient, const SourceProgram& program)
      : quotient_(quotient), program_(program), actors_(quotient, program.actors) {}
  // The CTA of an instruction (its actor's); instruction ids are dense in emission order.
  [[nodiscard]] std::uint32_t Cta(InstId id) const {
    if (id.value() >= program_.instructions.size()) throw std::logic_error("instruction outside the program");
    const auto instruction = program_.instructions[id.value()];
    if (instruction.id != id) throw std::logic_error("instruction ids are not dense");
    return actors_.Cta(instruction.actor);
  }
  [[nodiscard]] std::uint64_t Weight(std::uint32_t cta) const { return quotient_.Weight(cta); }
  [[nodiscard]] UnitQuotient& quotient() const noexcept { return quotient_; }

 private:
  UnitQuotient& quotient_;
  const SourceProgram& program_;
  ActorWeights actors_;
};

// The CTA of an instruction, and CTA-specific ids taken relative to their CTA (repeated entries): an
// instruction id relative to the CTA's first instruction and an actor id to its first actor, both only
// inside the run of consecutive ids that starts there (what the CTA's own lowering emitted; instructions
// appended later, such as cluster completion joins, are not relative), and a barrier lifetime id
// relative to the lowest lifetime the CTA initializes. Built on first use: one pass over instruction
// headers, plus the barrier initializers and an O(actors log actors) pass over the actors.
class CtaRelativeIds {
 public:
  static constexpr std::uint32_t kNoCta = UINT32_MAX;
  explicit CtaRelativeIds(const SourceProgram& program) : program_(program) {}
  [[nodiscard]] std::uint32_t Cta(InstId id) {
    Build();
    if (id.value() >= program_.instructions.size() || program_.instructions.id(id.value()) != id) return kNoCta;
    const auto actor = program_.instructions.actor(id.value()).value();
    return actor < actor_cta_.size() ? actor_cta_[actor] : kNoCta;
  }
  [[nodiscard]] std::optional<std::int64_t> Relative(MessageIdKind kind, std::uint32_t cta, std::uint32_t value) {
    Build();
    if (cta >= instructions_.size()) return std::nullopt;
    const auto in_range = [&](const Run& run) -> std::optional<std::int64_t> {
      if (value < run.first || value >= run.end) return std::nullopt;
      return std::int64_t(value) - run.first;
    };
    switch (kind) {
      case MessageIdKind::kInstruction: return in_range(instructions_[cta]);
      case MessageIdKind::kActor: return in_range(actors_[cta]);
      case MessageIdKind::kBarrierLifetime:
        if (barrier_base_[cta] == UINT32_MAX) return std::nullopt;
        return std::int64_t(value) - barrier_base_[cta];
      case MessageIdKind::kOther: break;
    }
    return std::nullopt;
  }

 private:
  // The run [first, end) of consecutive ids from a CTA's first one; ids are added in increasing order.
  struct Run {
    std::uint32_t first = UINT32_MAX;
    std::uint64_t end = 0;
    bool open = true;
    void Add(std::uint32_t value) {
      if (first == UINT32_MAX) {
        first = value;
        end = std::uint64_t(value) + 1;
      } else if (open && value == end) ++end;
      else open = false;
    }
  };
  void Build() {
    if (built_) return;
    built_ = true;
    std::uint32_t ctas = 0;
    for (const auto& actor : program_.actors) {
      if (actor.id.value() >= actor_cta_.size()) actor_cta_.resize(actor.id.value() + 1, kNoCta);
      actor_cta_[actor.id.value()] = actor.cta.value();
      ctas = std::max(ctas, actor.cta.value() + 1);
    }
    instructions_.assign(ctas, {});
    actors_.assign(ctas, {});
    barrier_base_.assign(ctas, UINT32_MAX);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> actors; // (id, CTA), in id order
    actors.reserve(program_.actors.size());
    for (const auto& actor : program_.actors) actors.emplace_back(actor.id.value(), actor.cta.value());
    std::sort(actors.begin(), actors.end());
    for (const auto& [id, cta] : actors) actors_[cta].Add(id);
    const auto cta_of = [&](std::size_t index) {
      const auto actor = program_.instructions.actor(index).value();
      return actor < actor_cta_.size() ? actor_cta_[actor] : kNoCta;
    };
    for (std::size_t index = 0; index < program_.instructions.size(); ++index) {
      const auto cta = cta_of(index);
      if (cta == kNoCta) continue;
      instructions_[cta].Add(program_.instructions.id(index).value());
      if (program_.instructions.holdsOperation<MBarrierInit>(index)) {
        const auto barrier = std::get<MBarrierInit>(program_.instructions[index].op).barrier.value();
        barrier_base_[cta] = std::min(barrier_base_[cta], barrier);
      }
    }
  }
  const SourceProgram& program_;
  bool built_ = false;
  std::vector<std::uint32_t> actor_cta_;
  std::vector<Run> instructions_, actors_;
  std::vector<std::uint32_t> barrier_base_;
};

// Repeated entries (docs/output-json.md "Repeated entries"), on both paths: entries of one list that
// are identical apart from the launch unit they occur in are reported once with `multiplicity` and
// `members`. Identical apart from the unit: the two CTAs have the same rank in their units, and the
// entries are equal once every CTA-specific id (an id field or an id rendered in the message) is
// taken relative to its CTA (CtaRelativeIds). An entry with a non-relative id, or without a CTA,
// stands alone. The entry shown is the occurrence in the lowest unit, with its own ids; entries are
// listed in the order of those occurrences. The list's first entry is that occurrence alone, followed
// by an entry for its other occurrences.
//
// In class mode a representative's occurrence stands for every unit of its class, so both paths group
// the same occurrences and render the same list. Where class mode cannot render the per-unit list
// exactly (a non-relative entry of a class with members, an entry without a CTA unless the analysed
// program is numbered as the per-unit one, or a shown occurrence whose CTA the analysed program numbers
// differently, UnitQuotient::NumberedAsPerUnit) it restarts per unit.
//
// Cost: each entry is keyed once (O(entry size)) and hashed; the CTA tables are built once
// for a non-empty list; members are sorted once per group: O(entries log entries) per unit, and
// O(representative entries + class sizes per group) in class mode.
class RepeatedEntries {
 public:
  struct IdField {
    const char* name;
    MessageIdKind kind;
    std::uint32_t value;
  };
  // `layout`: the launch's units (null: units are not defined and entries stay as listed);
  // `quotient`: class mode (then `layout` is it).
  RepeatedEntries(const LaunchUnits* layout, UnitQuotient* quotient, CtaRelativeIds& ids)
      : layout_(layout), quotient_(quotient), ids_(ids) {}
  void Add(Json item, std::uint32_t cta, std::initializer_list<IdField> fields,
           const std::vector<MessageId>& message_ids = {}) {
    const auto position = next_++;
    if (!layout_) {
      groups_.push_back({std::move(item), position, 0, cta, {}});
      return;
    }
    if (cta == CtaRelativeIds::kNoCta) {
      // Without a CTA the entry cannot stand for a class, nor be checked against the per-unit numbering.
      if (quotient_ && (quotient_->Shared() || quotient_->unit_shape[1] != 1 || quotient_->unit_shape[2] != 1))
        throw UnitClassRestart{"repeated_entries"};
      groups_.push_back({std::move(item), position, 0, cta, {}});
      return;
    }
    const auto unit = layout_->UnitOf(cta);
    std::uint32_t source = unit, lowest = unit;
    if (quotient_) {
      source = quotient_->units.at(unit).klass;
      lowest = quotient_->classes.at(source).units.front();
    }
    auto key = Key(item, cta, fields, message_ids);
    if (!key) {
      if (quotient_ && quotient_->Weight(cta) > 1) throw UnitClassRestart{"repeated_entries"};
      groups_.push_back({std::move(item), position, lowest, cta, {source}});
      return;
    }
    const auto [found, fresh] = index_.try_emplace(std::move(*key), groups_.size());
    if (fresh) {
      groups_.push_back({std::move(item), position, lowest, cta, {source}});
      return;
    }
    auto& group = groups_[found->second];
    group.sources.push_back(source);
    if (lowest < group.unit) {
      group.item = std::move(item);
      group.position = position;
      group.unit = lowest;
      group.cta = cta;
    }
  }
  void Render(Json& list) {
    std::sort(groups_.begin(), groups_.end(),
              [](const Group& a, const Group& b) { return a.position < b.position; });
    const auto first_member = quotient_ ? quotient_->FirstMemberUnit() : 0;
    for (auto& group : groups_) {
      if (group.sources.empty()) {
        list.push_back(std::move(group.item));
        continue;
      }
      if (quotient_ && !quotient_->NumberedAsPerUnit(group.cta, first_member))
        throw UnitClassRestart{"repeated_entries"};
      std::vector<std::uint32_t> units;
      if (quotient_)
        for (const auto klass : group.sources) {
          const auto& members = quotient_->classes.at(klass).units;
          units.insert(units.end(), members.begin(), members.end());
        }
      else units = std::move(group.sources);
      std::sort(units.begin(), units.end());
      if (list.empty()) {
        // The shown occurrence alone, then its other occurrences.
        list.push_back(group.item);
        units.erase(std::lower_bound(units.begin(), units.end(), group.unit));
        if (units.empty()) continue;
      } else if (units.size() == 1) {
        list.push_back(std::move(group.item));
        continue;
      }
      group.item["multiplicity"] = units.size();
      auto runs = Json::array();
      for (std::size_t i = 0; i < units.size();) {
        std::size_t j = i;
        while (j + 1 < units.size() && units[j + 1] <= units[j] + 1) ++j;
        runs.push_back({units[i], units[j]});
        i = j + 1;
      }
      group.item["members"] = std::move(runs);
      list.push_back(std::move(group.item));
    }
  }

 private:
  struct Group {
    Json item;
    std::size_t position;
    std::uint32_t unit, cta;
    std::vector<std::uint32_t> sources; // units, or classes in class mode; empty: never merged
  };
  // The entry with its CTA-specific ids relative to `cta`, prefixed by the CTA's rank; none when an id
  // is not relative.
  std::optional<std::string> Key(const Json& item, std::uint32_t cta, std::initializer_list<IdField> fields,
                                 const std::vector<MessageId>& message_ids) {
    Json relative = item;
    for (const auto& field : fields) {
      const auto value = ids_.Relative(field.kind, cta, field.value);
      if (!value) return std::nullopt;
      relative[field.name] = *value;
    }
    if (!message_ids.empty()) {
      const auto& text = item.at("message").get_ref<const std::string&>();
      std::string message;
      std::size_t at = 0;
      for (const auto& id : message_ids) {
        const auto value = ids_.Relative(id.kind, cta, id.value);
        if (!value || id.begin < at || id.begin + id.size > text.size()) return std::nullopt;
        message.append(text, at, id.begin - at);
        message += '\x1f';
        message += std::to_string(static_cast<int>(id.kind)) + ':' + std::to_string(*value);
        message += '\x1f';
        at = id.begin + id.size;
      }
      message.append(text, at);
      relative["message"] = std::move(message);
    }
    return std::to_string(layout_->RankOf(cta)) + ' ' + relative.dump(-1, ' ', false, Json::error_handler_t::replace);
  }
  const LaunchUnits* layout_;
  UnitQuotient* quotient_;
  CtaRelativeIds& ids_;
  std::size_t next_ = 0;
  std::vector<Group> groups_;
  std::unordered_map<std::string, std::size_t> index_;
};

template <typename Input>
std::string Analyze(Input&& source, bool verify, MemoryRaceOptions memory_options,
                    const PtxAssumptions& assumptions, ReachabilityOptions reachability_options,
                    std::optional<double> wellsync_timeout_seconds,
                    const VerifiedProgramObserver& on_verified,
                    const MemoryInputProvider& memory_input = {}) {
  const char* profile_env = sentryward::debug::environment("SENTRYWARD_PROFILE");
  ProfileSession profile_session(profile_env && std::string_view(profile_env) == "1");
  ProfileScope native_profile("native.total");
  Json report{{"schema", "spverify.native-result/v1"}, {"mode", verify ? "verify" : "validate"}};
  bool report_rendered = false;
  bool decoding_source_ir = true;
  try {
    SourceArtifact input;
    if constexpr (std::is_same_v<std::remove_cvref_t<Input>, SourceArtifact>) {
      input = std::forward<Input>(source);
      ValidateRequiredEffectGroups(input, !memory_input);
    } else {
      auto parsed = ProfileCall("input.parse", [&] { return Parse(std::forward<Input>(source)); });
      if (parsed.root.value("status", "") == "unsupported") {
        report["status"] = "event_view_unsupported";
        report["diagnostics"] = parsed.root.at("diagnostics");
        return RenderReport(report);
      }
      input = ProfileCall("input.decode", [&] { return Decode(std::move(parsed)); });
    }
    decoding_source_ir = false;
    OverridePtxAssumptions(input.semantic_assumptions, assumptions);
    report["frontend_assumptions"] = Json::parse(input.frontend_assumptions_json);
    report["mma_operand_ordering"] = Json::parse(input.mma_operand_ordering_json);
    report["semantic_assumptions"] = input.semantic_assumptions;
    const auto address_invariants = Json::parse(input.address_invariants_json);
    report["address_model"] = {{"kind", "symbolic"}, {"layers", {0, 2}},
        {"solver", SymbolicAddressModel::SolverName()},
        {"solver_timeout_ms", input.address_solver.timeout_ms},
        {"address_invariants", address_invariants}};
    report["assumption_dependent"] = !address_invariants.empty() ||
        report["frontend_assumptions"].contains("tensor_properties") ||
        std::any_of(input.semantic_assumptions.begin(), input.semantic_assumptions.end(),
                    [](const auto& item) { return item.second; });
    memory_options.warn_duplicate_tma_stores = input.semantic_assumptions.at("duplicate_tma_store_benign");
    std::optional<QuotientEvidence> units;
    if (memory_options.units) units.emplace(*memory_options.units, input.program);
    // Repeated entries are reported per launch unit on both paths (RepeatedEntries).
    const LaunchUnits* launch_units = memory_options.units ? memory_options.units : memory_options.launch_units;
    CtaRelativeIds relative_ids(input.program);
    const auto recovery = ProfileCall("input.recover_initializers", [&] { return RecoverDuplicateInitializers(input.program); });
    report["warnings"] = Json::array();
    if (recovery.applied()) {
      report["analysis_target"] = "recovered_program";
      report["recovery_policy"] = "last_ordered_init";
      std::unordered_set<std::uint32_t> replaced;
      RepeatedEntries warnings(launch_units, memory_options.units, relative_ids);
      for (const auto& warning : recovery.warnings) {
        replaced.insert(warning.replaced_instruction.value());
        Json item{
            {"code", "duplicate-barrier-initialization"},
            {"message", "Earlier initialization is ignored; the last program-ordered initialization is retained. Analysis applies to the recovered program."},
            {"barrier", warning.barrier.value()},
            {"replaced_instruction", warning.replaced_instruction.value()},
            {"retained_instruction", warning.retained_instruction.value()},
            {"replaced_location", LocationJson(warning.replaced_location)},
            {"retained_location", LocationJson(warning.retained_location)},
            {"replaced_expected_arrivals", warning.replaced_expected_arrivals},
            {"retained_expected_arrivals", warning.retained_expected_arrivals}};
        warnings.Add(std::move(item), relative_ids.Cta(warning.replaced_instruction),
                     {{"barrier", MessageIdKind::kBarrierLifetime, warning.barrier.value()},
                      {"replaced_instruction", MessageIdKind::kInstruction, warning.replaced_instruction.value()},
                      {"retained_instruction", MessageIdKind::kInstruction, warning.retained_instruction.value()}});
      }
      warnings.Render(report["warnings"]);
      // Original init Issue and Complete denote the same event.
      std::map<const void*,SharedOptional<std::vector<MilestoneRef>>> rewritten_targets;
      for (auto& order : input.required_orders) {
        const auto rewrite=[&](MilestoneRef& ref) {
          if (ref.milestone == Milestone::kIssue && ref.effect_index == 0 &&
              replaced.contains(ref.instruction.value())) {
            ref.milestone = Milestone::kComplete;
          }
        };
        rewrite(order.from);rewrite(order.to);
        if(order.target_domain) {
          auto [it,fresh]=rewritten_targets.try_emplace(order.target_domain.storage_identity());
          if(fresh) {
            auto refs=*order.target_domain;
            for(auto& ref:refs)rewrite(ref);
            it->second=std::move(refs);
          }
          order.target_domain=it->second;
        }
      }
    }
    const auto& diagnostics = recovery.diagnostics;
    if (!diagnostics.empty()) {
      report["status"] = "invalid_event_view";
      report["stage"] = "event_view_validation";
      report["reason"] = ToString(diagnostics.front().code);
      report["diagnostics"] = Json::array();
      RepeatedEntries entries(launch_units, memory_options.units, relative_ids);
      for (const auto& diagnostic : diagnostics) {
        Json item{{"code", ToString(diagnostic.code)}, {"message", diagnostic.message}};
        if (!diagnostic.instruction) {
          entries.Add(std::move(item), CtaRelativeIds::kNoCta, {}, diagnostic.message_ids);
          continue;
        }
        item["instruction"] = diagnostic.instruction->value();
        entries.Add(std::move(item), relative_ids.Cta(*diagnostic.instruction),
                    {{"instruction", MessageIdKind::kInstruction, diagnostic.instruction->value()}},
                    diagnostic.message_ids);
      }
      entries.Render(report["diagnostics"]);
      return RenderReport(report);
    }
    ProfileCall("input.compact_source", [&] { CompactSourceStorage(input); });
    ReportRecordGraphSize("source.after_recovery", input.program.instructions, input.program.edges);
    ReportSourceMemory(input, "input.after_recovery");
    std::unique_ptr<SymbolicAddressModel> model;
    std::optional<MemoryAccessRecords> compute_accesses;
    const auto prepare_memory = [&](const EventRecords& events) -> SymbolicAddressModel& {
      if (!model) {
        ProfileScope memory_profile("memory.prepare");
        if (memory_input) input.addresses = memory_input();
        ValidateRequiredEffectGroups(input);
        std::unordered_set<std::uint32_t> addresses;
        addresses.reserve(input.addresses.size());
        for (const auto& binding : input.addresses) addresses.insert(binding.address.value());
        auto bound = ProfileCall("memory.bind_compute_accesses", [&] {
          return ExtractMemoryAccesses(events, input.program);
        });
        if (!bound) Fail(bound.error().message);
        compute_accesses.emplace(std::move(bound).value());
        ProfileCall("memory.validate_bindings", [&] {
          for (const auto& access : *compute_accesses) {
            if (const auto* address = std::get_if<AddressId>(&access.address);
                access.bytes != 0 && address && !addresses.contains(address->value()))
              Fail("memory access has no address binding");
          }
        });
        report["counts"]["memory_accesses"] = compute_accesses->size();
        model = std::make_unique<SymbolicAddressModel>(
            input.addresses, input.address_invariants_json, input.address_solver, 4096,
            input.tensor_properties_json);
        ReportAddressMemory(model->constantModel(), "memory.address_model");
        if (const auto error = model->validate()) Fail(error->message);
      }
      return *model;
    };
    std::optional<LoweredProgram> prepared;
    {
      ProfileScope metadata_profile("input.metadata");
      auto lowered = ProfileCall("input.metadata_lower", [&] { return Lowerer{}.Run(input.program, true, CollectiveOrdering::kExecution); });
      if (!lowered) Fail(lowered.error().message);
      prepared.emplace(std::move(lowered).value());
      // Validate endpoint domains without ever constructing their pair product.
      std::unordered_set<const void*> checked_domains;
      for(const auto& order:input.required_orders) {
        (void)Resolve(*prepared,order.from);
        if(order.target_domain && !checked_domains.insert(order.target_domain.storage_identity()).second)continue;
        if(order.targets().empty() || order.targets().front()!=order.to)
          Fail("invalid required-order target domain");
        for(auto ref:order.targets())(void)Resolve(*prepared,ref);
      }
      const auto& graph = prepared->graph();
      report["counts"] = {{"threads", input.program.actors.size()}, {"instructions", input.program.instructions.size()},
          {"events", graph.eventCount()}, {"edges", graph.edgeCount()}};
    }
    std::size_t logical_required_orders=0;
    for(const auto& order:input.required_orders)logical_required_orders+=order.targets().size();
    report["required_orders"] = logical_required_orders;
    report["required_order_domains"] = input.required_orders.size();
    report["synchronization_input"] = {{"kind", "ordering_endpoints"},
        {"memory_address_payloads", false}, {"compute_access_binding", "after_happens_before"}};
    const auto shared_value_orders=std::count_if(input.required_orders.begin(),input.required_orders.end(),
        [](const auto& order){return order.kind.view()=="shared_constant_read_from";});
    if(shared_value_orders)report["frontend_value_validation"]={{"status","pending"},{"required_orders",shared_value_orders}};
    ReportRecordGraphSize("event.ready", prepared->graph().events(), prepared->graph().edges());
    ReportGraphMemory(*prepared, "input.graph_ready");
    report["verification_limitations"] = Json::array();
    for (const auto& limitation : input.verification_limitations) {
      report["verification_limitations"].push_back({{"code", limitation.code}, {"message", limitation.message}});
    }
    report["source_violations"] = Json::array();
    for (const auto& violation : input.source_violations) {
      report["source_violations"].push_back({{"code", violation.code}, {"message", violation.message}});
    }
    if (!verify) {
      if (memory_input) throw std::logic_error("Deferred memory input requires verification");
      (void)prepare_memory(prepared->graph().events());
      report["status"] = "validated";
      return RenderReport(report);
    }
    if (!input.verification_limitations.empty()) {
      report["status"] = "event_view_incomplete";
      report["stage"] = "event_view_construction";
      report["reason"] = input.verification_limitations.front().code;
      return RenderReport(report);
    }
    CheckerOptions checker_options;
    checker_options.reachability = reachability_options;
    checker_options.timeout_seconds = wellsync_timeout_seconds;
    auto synchronization = WellSyncChecker{}.RunPrepared(input.program, *prepared, checker_options);
    auto lifecycle_origins=std::move(*prepared).takeOrigins();
    prepared.reset();
    if (synchronization.passed() && !input.source_violations.empty()) {
      report["status"] = "source_lifecycle_violation";
      report["stage"] = "lifecycle_check";
      report["reason"] = input.source_violations.front().code;
      return RenderReport(report);
    }
    VerificationResult result{std::move(synchronization), std::nullopt, std::nullopt};
    std::optional<StaticHbQueries> static_hb;
    if (result.synchronization.passed())
      static_hb.emplace(result.synchronization.certificate()->happensBeforeGraph,
          memory_options.reachability, memory_options.reachability_factory);
    const auto record_hb = [&](const char* stage, const HbReachabilityStats& before) {
      const auto stats = HbStatsSince(static_hb->stats(), before);
      report["hb_queries"][stage] = {{"algorithm", std::string(static_hb->name())},
          {"queries", stats.queries}, {"filtered_queries", stats.filtered_queries},
          {"batches", stats.batches}, {"edge_visits", stats.edge_visits},
          {"cache_hits", stats.cache_hits}, {"dfs_queries", stats.dfs_queries},
          {"shared_peak_index_bytes", stats.peak_index_bytes}};
    };
    if (result.synchronization.passed() && !input.required_orders.empty()) {
      ProfileScope required_profile("lifecycle.required_orders");
      const auto relation_before=static_hb->relations().stats();
      Json request_stats;
      const auto ordered = CheckRequiredOrders(static_hb->relations(), input, lifecycle_origins, &request_stats);
      const auto relation_after=static_hb->relations().stats();
      report["hb_queries"]["required_orders"]={{"algorithm","member-relations"},
          {"batches",relation_after.traversals-relation_before.traversals},
          {"edge_visits",relation_after.relation_visits-relation_before.relation_visits},
          {"member_enumerations",relation_after.member_enumerations-relation_before.member_enumerations},
          {"request_construction",std::move(request_stats)}};
      Json unresolved = Json::array();
      // Reporting must not materialize millions of failed relations. The check
      // and classification still cover every requirement; only witnesses are capped.
      constexpr std::size_t failure_sample_limit = 32;
      constexpr std::size_t effect_sample_limit = 32;
      std::size_t failure_count = 0;
      bool shared_value_unproven=false;
      std::map<std::string,Json> lifecycle_diagnostics;
      const auto code_of=[](const std::string& kind) -> std::string {
        return kind=="access_completion_before_dealloc"?"access_completion_before_deallocation_not_proven":
            kind=="mbarrier_final_invalidation"?"barrier_completion_before_invalidation_not_proven":
            kind=="allocation_before_dealloc"?"allocation_before_deallocation_not_proven":
            kind=="tmem_group2_access_before_dealloc"?"paired_access_completion_before_deallocation_not_proven":
            kind=="shared_constant_read_from"?"shared_constant_dependency_unproven":kind+"_not_proven";
      };
      // Class mode: a failed order of a representative CTA fails in every unit of its class
      // (counts weighted; samples drawn from that expansion in today's order).
      std::vector<std::pair<std::size_t,std::uint32_t>> class_failures; // (order, its CTA)
      for (std::size_t index = 0; index < input.required_orders.size(); ++index) {
        const auto& order = input.required_orders[index];
        if (const auto failed=ordered[index].count()) {
          const auto kind=order.kind.str();
          const std::string code=code_of(kind);
          auto& diagnostic=lifecycle_diagnostics[code];
          if(diagnostic.is_null())diagnostic={{"code",code},{"stage","lifecycle_check"},{"message",order.message.str()},
             {"details",{{"constraint",kind},{"count",0},{"witnesses",Json::array()}}}};
          auto& evidence=diagnostic["details"];
          shared_value_unproven|=order.kind.view()=="shared_constant_read_from";
          if(units) {
            const auto cta=units->Cta(order.from.instruction);
            const auto weight=units->Weight(cta);
            failure_count+=failed*weight;
            evidence["count"]=evidence["count"].get<std::size_t>()+failed*weight;
            class_failures.emplace_back(index,cta);
            continue;
          }
          failure_count+=failed;
          evidence["count"]=evidence["count"].get<std::size_t>()+failed;
          if(evidence["witnesses"].size()<failure_sample_limit || unresolved.size()<failure_sample_limit) {
            const auto& [family,members]=*ordered[index].failed.begin();
            const auto& f=static_hb->relations().families().families()[family];
            const auto event=EventId(f.first.value()+f.stride*members.first());
            const auto target=lifecycle_origins.origin(event);
            if(!target)throw std::logic_error("required-order witness lacks source origin");
            Json witness={{"kind",kind},{"message",order.message.str()},
              {"from_instruction",order.from.instruction.value()},{"to_instruction",target->value()},
              {"failed_target_count",failed}};
            if(order.from_effects) {
              auto effects=Json::array();
              const auto count=order.from_effects->size();
              for(std::size_t i=0;i<std::min(count,effect_sample_limit);++i)effects.push_back((*order.from_effects)[i]);
              witness["from_effects"]=std::move(effects);
              witness["from_effects_count"]=count;
              witness["from_effects_truncated"]=count>effect_sample_limit;
            }
            if(unresolved.size()<failure_sample_limit)unresolved.push_back(witness);
            if(evidence["witnesses"].size()<failure_sample_limit)evidence["witnesses"].push_back(std::move(witness));
          }
          evidence["witnesses_truncated"]=evidence["count"].get<std::size_t>()>evidence["witnesses"].size();
        }
      }
      if(units && !class_failures.empty()) {
        // The expansion in today's order: orders emitted while a CTA was lowered, CTA by CTA in launch
        // order (each CTA's are its representative CTA's), then those emitted at Finish, unit by unit.
        // Each instance carries the representative's witness (its numbering). Bounded: it stops once
        // every sample list is full.
        const auto& q=units->quotient();
        std::map<std::uint32_t,std::vector<std::size_t>> by_cta, by_unit;
        for(const auto& [index,cta]:class_failures)
          (index<q.add_time_orders?by_cta[cta]:by_unit[q.UnitOf(cta)]).push_back(index);
        std::map<std::size_t,Json> witnesses;
        std::size_t open=1+lifecycle_diagnostics.size();
        const auto emit=[&](std::size_t index) {
          auto [cached,fresh]=witnesses.try_emplace(index);
          if(fresh) {
            const auto& order=input.required_orders[index];
            const auto& [family,members]=*ordered[index].failed.begin();
            const auto& f=static_hb->relations().families().families()[family];
            const auto event=EventId(f.first.value()+f.stride*members.first());
            const auto target=lifecycle_origins.origin(event);
            if(!target)throw std::logic_error("required-order witness lacks source origin");
            Json witness={{"kind",order.kind.str()},{"message",order.message.str()},
              {"from_instruction",order.from.instruction.value()},{"to_instruction",target->value()},
              {"failed_target_count",ordered[index].count()}};
            if(order.from_effects) {
              auto effects=Json::array();
              const auto count=order.from_effects->size();
              for(std::size_t i=0;i<std::min(count,effect_sample_limit);++i)effects.push_back((*order.from_effects)[i]);
              witness["from_effects"]=std::move(effects);
              witness["from_effects_count"]=count;
              witness["from_effects_truncated"]=count>effect_sample_limit;
            }
            cached->second=std::move(witness);
          }
          auto& list=lifecycle_diagnostics[code_of(input.required_orders[index].kind.str())]["details"]["witnesses"];
          if(unresolved.size()<failure_sample_limit && unresolved.size()+1==failure_sample_limit)--open;
          if(unresolved.size()<failure_sample_limit)unresolved.push_back(cached->second);
          if(list.size()<failure_sample_limit && list.size()+1==failure_sample_limit)--open;
          if(list.size()<failure_sample_limit)list.push_back(cached->second);
          return open!=0;
        };
        bool more=true;
        for(std::uint32_t cta=0;more && cta<q.grid[0]*q.grid[1]*q.grid[2];++cta)
          if(const auto found=by_cta.find(q.RepresentativeCta(cta));found!=by_cta.end())
            for(const auto index:found->second)if(!(more=emit(index)))break;
        for(std::uint32_t unit=0;more && unit<q.units.size();++unit)
          if(const auto found=by_unit.find(q.classes.at(q.units[unit].klass).units.front());found!=by_unit.end())
            for(const auto index:found->second)if(!(more=emit(index)))break;
        for(auto& [code,diagnostic]:lifecycle_diagnostics) {
          auto& evidence=diagnostic["details"];
          evidence["witnesses_truncated"]=evidence["count"].get<std::size_t>()>evidence["witnesses"].size();
        }
      }
      if (failure_count != 0) {
        report["diagnostics"]=Json::array();
        for(auto& [code,diagnostic]:lifecycle_diagnostics)report["diagnostics"].push_back(std::move(diagnostic));
        report["failed_required_orders_count"] = failure_count;
        report["failed_required_orders_sample_limit"] = failure_sample_limit;
        report["failed_required_orders_truncated"] = failure_count > unresolved.size();
        report["status"] = shared_value_unproven?"shared_memory_value_unproven":"unsafe_lifecycle";
        report["error"] = {{"code",report["diagnostics"].front()["code"]},
                           {"message",report["diagnostics"].front()["message"]}};
        if(shared_value_unproven) {
          report["reason"]="shared_constant_dependency_unproven";
          report["frontend_value_validation"]["status"]="unproven";
        }
        report["failed_required_orders"] = std::move(unresolved);
        return RenderReport(report);
      }
    }
    if(!shared_value_orders)lifecycle_origins=OriginMap{};
    if (result.synchronization.passed()) {
      auto& memory_model = prepare_memory(result.synchronization.certificate()->happensBeforeGraph.events());
      const auto before = static_hb->stats();
      auto families = ProfileCall("memory.bind_compute_families", [&] {
        return BindMemoryAccessFamilies(result.synchronization.certificate()->happensBeforeGraph.events(), *compute_accesses);
      });
      auto memory = ProfileCall("memory.check", [&] { return MemoryRaceChecker{}.RunWithAccesses(
          input.program, *result.synchronization.certificate(), *compute_accesses, families,
          memory_model, memory_options, &*static_hb); });
      record_hb("memory", before);
      if (memory) result.memory = std::move(memory.value());
      else result.memory_error = memory.error();
    }
    switch (result.status()) {
      case VerificationStatus::kSynchronizationRejected:
        report["status"] = "not_wellsynced";
        break;
      case VerificationStatus::kUnsupported: report["status"] = "unsupported"; break;
      case VerificationStatus::kInvalidInput: report["status"] = "invalid_event_view"; break;
      case VerificationStatus::kRaceFound:
        report["status"] = "potential_race";
        report["diagnostics"]=Json::array({{{"code","conflicting_accesses_unordered"},{"stage","memory_check"},
          {"message","Conflicting accesses have no proven required ordering"},
          {"details",{{"witnesses_path","races"},{"count",result.memory->race_count}}}}});
        break;
      case VerificationStatus::kRaceFree: report["status"] = "race_free"; break;
      case VerificationStatus::kUnresolved: report["status"] = "memory_analysis_inconclusive"; break;
    }
    if (const auto* diagnostic = result.synchronization.invalidInput()) report["message"] = diagnostic->message;
    if (const auto* diagnostic = result.synchronization.unsupported()) report["message"] = diagnostic->message;
    if (result.memory_error) report["message"] = result.memory_error->message;
    if (const auto* rejection = result.synchronization.rejection()) {
      RejectionReport{}.Write(report, *rejection, result.synchronization.rejectionContext());
    }
    if (result.memory) {
      report["races"] = Json::array();
      const auto finding_json = [](const RaceFinding& race) {
        Json finding{{"first", Source(race.first)}, {"second", Source(race.second)},
            {"overlap", race.overlap.description}};
        if (race.overlap.constant_region) {
          const auto& region = *race.overlap.constant_region;
          finding["region"] = {{"address_space", region.address_space}, {"allocation", region.allocation},
                               {"offset", region.offset}, {"bytes", region.bytes}};
        }
        return finding;
      };
      for (const auto& race : result.memory->races) {
        report["races"].push_back(finding_json(race));
      }
      if (const auto& cross = result.memory->cross_unit_race; cross && !report["races"].empty()) {
        // Cross-CTA class mode: the representatives' accesses, translated into these units.
        report["races"][0]["cross_unit"] = {{"first_unit", cross->first_unit}, {"first_cta", cross->first_cta},
            {"second_unit", cross->second_unit}, {"second_cta", cross->second_cta},
            {"allocation", cross->allocation}, {"first_delta", cross->first_delta},
            {"second_delta", cross->second_delta}};
      }
      for (const auto& warning : result.memory->duplicate_tma_store_warnings) {
        Json finding = finding_json(warning);
        finding["code"] = "duplicate-tma-store";
        finding["message"] = "Unordered same-warp TMA writes have matching copy footprints and PTX site-visit indices. Value equality is not proven; analysis continues with all accesses retained.";
        report["warnings"].push_back(std::move(finding));
      }
      report["warning_count"] = result.memory->warning_count;
      report["race_count"] = result.memory->race_count;
      report["race_samples_truncated"] = result.memory->race_count > result.memory->races.size();
      report["warning_samples_truncated"] = result.memory->warning_count >
          result.memory->duplicate_tma_store_warnings.size();
      report["unresolved_count"] = result.memory->unresolved_count;
      if (!result.memory->unresolved.empty()) {
        const auto& pair = result.memory->unresolved.front();
        report["reason"] = pair.code;
        report["message"] = pair.reason;
        report["details"]={{"witness_path","unsupported_memory_pair"}};
        report["unsupported_memory_pair"] = {
            {"first", Source(pair.first)}, {"second", Source(pair.second)},
            {"reason", pair.reason}};
        for (const auto& [name, access] : {
                 std::pair{"first", &pair.first}, std::pair{"second", &pair.second}}) {
          const auto* address = std::get_if<AddressId>(&access->address);
          if (!address) continue;
          for (const auto& binding : input.addresses) {
            if (binding.address != *address) continue;
            auto& endpoint = report["unsupported_memory_pair"][name]["endpoint"];
            endpoint["allocation"] = binding.allocation;
            endpoint["address_space"] = binding.address_space;
            if (!binding.expression.empty()) {
              endpoint["expression"] = Json::parse(binding.expression);
              // The reason comes from the address solver, not expression presence.
            }
            break;
          }
        }
        report["memory_analysis_complete"] = false;
      }
      report["search_complete"] = result.memory->search_complete;
      const auto& hb = result.memory->reachability_stats;
      report["memory_analysis"] = {
          {"exhaustive", memory_options.exhaustive},
          {"strategy", result.memory->search_strategy},
          {"candidate_generation_fallback", result.memory->candidate_generation_fallback},
          {"hb_algorithm", result.memory->reachability_algorithm},
          {"address_candidates", result.memory->address_candidates},
          {"unordered_candidates_examined", result.memory->candidates_examined},
          {"overlap_queries", result.memory->overlap_queries},
          {"hb_queries", hb.queries}, {"hb_filtered_queries", hb.filtered_queries},
          {"hb_cache_hits", hb.cache_hits},
          {"hb_batches", hb.batches}, {"hb_edge_visits", hb.edge_visits},
          {"hb_peak_index_bytes", hb.peak_index_bytes}, {"hb_dfs_queries", hb.dfs_queries},
          {"hb_query_seconds", result.memory->reachability_seconds}};
      const auto address = model->stats();
      if(result.memory->family_stats) {
        const auto& family=*result.memory->family_stats;
        report["memory_analysis"]["family_analysis"]={
          {"logical_accesses",family.logical_accesses},{"access_families",family.access_families},
          {"candidate_relations",family.candidate_relations},{"affine_member_rows",family.affine_member_rows},
          {"hb_member_enumerations",family.hb_member_enumerations},{"hb_frontier_pieces",family.hb_frontier_pieces}};
      }
      report["memory_analysis"]["address_model"] = {
          {"refined_pairs", address.refined}, {"cache_hits", address.cache_hits},
          {"layer0_disjoint", address.layer0_disjoint}, {"ground_disjoint", address.ground_disjoint},
          {"ground_overlap", address.ground_overlap}, {"congruence_disjoint", address.congruence_disjoint},
          {"injective_disjoint", address.injective_disjoint}, {"solver_queries", address.solver_queries},
          {"solver_unsat", address.solver_unsat}, {"solver_sat", address.solver_sat},
          {"solver_unknown", address.solver_unknown}};
      if (result.memory->candidate_generation_fallback) {
        report["memory_analysis"]["strategy_note"] = {
            {"code", "memory_conflict_candidate_generation_unavailable"},
            {"message", kMemoryCandidateFallback}};
      }
      if (result.memory->first_race_seconds) {
        report["memory_analysis"]["first_race_seconds"] = *result.memory->first_race_seconds;
      }
      if (memory_options.continue_after_race) {
        report["race_count_scope"] = "retained_witnesses";
        report["race_count_complete"] = false;
        report["memory_analysis"]["continue_after_race"] = true;
        report["memory_analysis"]["search_scope"] = "ordering_constraints";
        report["memory_analysis"]["violated_order_constraints"] = result.memory->violated_order_constraints;
      }
    }
    if(shared_value_orders && result.status()==VerificationStatus::kRaceFree &&
        result.memory && result.memory->search_complete) {
      const auto before = static_hb->stats();
      const auto error = ValidateSharedConstantWrites(input,*result.synchronization.certificate(),lifecycle_origins,static_hb->get());
      lifecycle_origins=OriginMap{};
      record_hb("shared_constant_interference", before);
      if(error) {
        report["status"]="shared_memory_value_unproven";
        report["reason"]="shared_constant_interference_unproven";
        report["error"]={{"code","shared_constant_interference_unproven"},{"message",*error}};
        report["frontend_value_validation"]["status"]="unproven";
        return RenderReport(report);
      }
      report["frontend_value_validation"]["status"]="validated";
    }
    if (on_verified && result.status() == VerificationStatus::kRaceFree &&
        result.memory && result.memory->search_complete) {
      auto rendered = RenderReport(report);
      report_rendered = true;
      on_verified(input, *result.synchronization.certificate(), *result.memory, *compute_accesses, rendered);
      return rendered;
    }
  } catch (const SourceIrError& error) {
    report["status"] = "invalid_event_view";
    report["stage"] = "event_view_validation";
    report["reason"] = "invalid_event_view_structure";
    report["message"] = error.what();
  } catch (const Json::exception& error) {
    report["status"] = decoding_source_ir ? "invalid_event_view" : "internal_error";
    report["message"] = error.what();
  } catch (const std::invalid_argument& error) {
    report["status"] = decoding_source_ir ? "invalid_event_view" : "internal_error";
    report["message"] = error.what();
  } catch (const std::bad_alloc& error) {
    report["status"] = "allocation_failed";
    report["resource"] = {{"kind","memory"},{"origin","allocator"},{"cause","unknown"}};
    report["message"] = error.what();
  } catch (const std::exception& error) {
    report["status"] = "internal_error";
    report["message"] = error.what();
  }
  // RenderReport normalizes warning counts; do not apply it twice if the
  // numerical observer throws after receiving the completed verification.
  return report_rendered ? report.dump() : RenderReport(report);
}
}  // namespace

std::string AnalyzeSourceArtifact(SourceArtifact input, bool verify, const MemoryRaceOptions& memory,
                                  const PtxAssumptions& assumptions, ReachabilityOptions reachability,
                                  std::optional<double> wellsync_timeout_seconds,
                                  const VerifiedProgramObserver& on_verified,
                                  const MemoryInputProvider& memory_input) {
  return Analyze(std::move(input), verify, memory, assumptions, reachability, wellsync_timeout_seconds, on_verified, memory_input);
}

std::string AnalyzeSourceArtifact(std::string_view text, bool verify, const MemoryRaceOptions& memory,
                                  const PtxAssumptions& assumptions, ReachabilityOptions reachability,
                                  std::optional<double> wellsync_timeout_seconds,
                                  const VerifiedProgramObserver& on_verified) {
  return Analyze(text, verify, memory, assumptions, reachability, wellsync_timeout_seconds, on_verified);
}

std::string AnalyzeSourceArtifact(std::istream& input, bool verify, const MemoryRaceOptions& memory,
                                  const PtxAssumptions& assumptions, ReachabilityOptions reachability,
                                  std::optional<double> wellsync_timeout_seconds,
                                  const VerifiedProgramObserver& on_verified) {
  return Analyze(input, verify, memory, assumptions, reachability, wellsync_timeout_seconds, on_verified);
}
}  // namespace sentryward::algorithm
