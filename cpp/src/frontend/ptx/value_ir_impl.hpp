#pragma once
#include "sentryward/frontend/ptx/value_ir.h"
#include "sentryward/algorithm/base/analysis_scope.h"
#include "sentryward/frontend/ptx/branch_scope_capture.h"
#include "sentryward/frontend/ptx/address_expression.h"
#include "sentryward/frontend/ptx/operands.h"
#include "sentryward/frontend/ptx/layouts.h"
#include "sentryward/frontend/ptx/resources.h"
#include "sentryward/frontend/ptx/tmem_lifetime.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/ptx_operation.h"
#include "sentryward/value_ir/native_matrix.h"
#include "sentryward/algorithm/diagnostics/memory_profile.h"
#include "sentryward/equivalence/float_from_chars.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <iomanip>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace sentryward::frontend::ptx {
namespace {
namespace v = value_ir;
using Scope = std::vector<std::uint32_t>;
using Key = std::pair<Scope, std::string>;

bool In(std::string_view x, std::initializer_list<std::string_view> values) {
  return std::find(values.begin(), values.end(), x) != values.end();
}
std::vector<std::string> Split(std::string_view text, char separator) {
  std::vector<std::string> result;
  for (;;) {
    const auto at = text.find(separator);
    result.push_back(Trim(text.substr(0, at)));
    if (at == text.npos) return result;
    text.remove_prefix(at + 1);
  }
}
std::string ScopeText(const Scope &scope) {
  std::string out;
  for (const auto item : scope) out += '/' + std::to_string(item);
  return out.empty() ? "/" : out;
}
struct Type {
  v::ValueType value;
  std::string spelling;
  std::uint32_t bits;
  std::uint32_t elements = 1;
};
std::optional<Type> PtxType(std::string_view spelling) {
  const std::string text(spelling);
  if (text == "pred") return Type{v::ValueType::scalar(v::ElementType::predicate()), text, 1};
  for (const std::string format : {"e2m1", "e4m3", "e5m2", "ue8m0"}) {
    const std::uint32_t width = format == "e2m1" ? 4 : 8;
    if (text == format) return Type{v::ValueType::scalar(v::ElementType::floating(width, format)), text, width};
    if (text == format + "x2") return Type{v::ValueType::matrix(v::ElementType::floating(width, format), {2}), text, width, 2};
  }
  static const std::regex pattern("^(bf|[busf])([0-9]+)(x([0-9]+))?$");
  std::smatch match;
  if (!std::regex_match(text, match, pattern)) return {};
  const auto width = static_cast<std::uint32_t>(std::stoul(match[2]));
  const auto count = match[4].matched ? static_cast<std::uint32_t>(std::stoul(match[4])) : 1;
  if (!In(std::to_string(width), {"8", "16", "32", "64", "128"}) || count == 0 || count > 16)
    return {};
  const auto kind = match[1].str();
  auto element = kind == "b" ? v::ElementType::bit(width)
      : kind == "u" ? v::ElementType::unsignedInteger(width)
      : kind == "s" ? v::ElementType::signedInteger(width)
      : v::ElementType::floating(width, kind + std::to_string(width));
  return Type{count == 1 ? v::ValueType::scalar(element)
                        : v::ValueType::matrix(element, {count}), text, width, count};
}
Type U32() { return *PtxType("u32"); }
Type U64() { return *PtxType("u64"); }
Type Pred() { return *PtxType("pred"); }
struct Expression {
  v::Value value;
  Type type;
  std::vector<Expression> components;
  std::optional<std::string> missing_definition;
  // With missing_definition: the register had not been assigned on this thread's path when it was
  // read, so the value is its unspecified initial one (Unassigned), built once a consumer needs it.
  std::optional<Key> unassigned;
  Expression(v::Value value, Type type, std::vector<Expression> components = {})
      : value(std::move(value)), type(std::move(type)), components(std::move(components)) {}
};

struct FunctionContext {
  algorithm::SharedString source_file, function_name;
  std::vector<v::SourceOrigin> instruction_origins;
  std::map<Key, Type> declarations;
  std::set<Key> symbols;
  std::map<Key, std::string> symbol_spaces;
  std::map<std::uint32_t, std::uint32_t> loop_ends;
  // The st.local shapes a lifted region's stores follow (execution.h).
  LocalRegionStores local_stores;

  FunctionContext(const Module&, const Function&);
};

// The thread fields of `origin` for `lane`: its linear thread id, warp, lane
// and block coordinates. Returns the linear thread id.
std::uint64_t SetThread(v::SourceOrigin& origin, const FactSource& facts, LaneRef lane) {
  const auto thread=std::uint64_t(facts.Warp(lane))*facts.launch().warp_size()+facts.LaneId(lane);
  const auto& dim=facts.launch().block_dim();
  origin.thread=thread; origin.warp=facts.Warp(lane); origin.lane=facts.LaneId(lane);
  origin.thread_coordinates={thread%dim[0],thread/dim[0]%dim[1],thread/dim[0]/dim[1]};
  return thread;
}

// Path predicates of an if-converted forward region in disjunctive normal form: a literal is
// (guarding instruction, value of its guard), a term a sorted conjunction, a path a
// disjunction of terms ({} on no path, {{}} on every path through the region).
using GuardLiteral = std::pair<std::uint32_t, bool>;
using GuardTerm = std::vector<GuardLiteral>;
using GuardPath = std::vector<GuardTerm>;
bool Weaker(const GuardTerm& term, const GuardTerm& stronger) {
  return std::includes(stronger.begin(), stronger.end(), term.begin(), term.end());
}
// Absorption (t | t&x = t) and resolution (t&l | t&!l = t) until neither applies.
GuardPath Simplify(GuardPath path) {
  for (bool changed = true; changed;) {
    changed = false;
    for (std::size_t a = 0; a < path.size() && !changed; ++a)
      for (std::size_t b = 0; b < path.size() && !changed; ++b) {
        if (a == b) continue;
        if (Weaker(path[a], path[b])) { path.erase(path.begin() + b); changed = true; continue; }
        if (path[a].size() != path[b].size()) continue;
        std::size_t differ = path[a].size();
        bool single = true;
        for (std::size_t n = 0; n < path[a].size() && single; ++n)
          if (path[a][n] != path[b][n]) {
            single = differ == path[a].size() && path[a][n].first == path[b][n].first;
            differ = n;
          }
        if (!single || differ == path[a].size()) continue;
        path[a].erase(path[a].begin() + differ);
        path.erase(path.begin() + b);
        changed = true;
      }
  }
  std::sort(path.begin(), path.end());
  return path;
}
GuardPath Conjoin(const GuardPath& path, GuardLiteral literal) {
  GuardPath result;
  for (auto term : path) {
    if (std::binary_search(term.begin(), term.end(), GuardLiteral{literal.first, !literal.second})) continue;
    const auto at = std::lower_bound(term.begin(), term.end(), literal);
    if (at == term.end() || *at != literal) term.insert(at, literal);
    result.push_back(std::move(term));
  }
  return Simplify(std::move(result));
}
GuardPath Disjoin(GuardPath path, const GuardPath& other) {
  path.insert(path.end(), other.begin(), other.end());
  return Simplify(std::move(path));
}
// Sufficient conditions only: every term of `path` strengthens a term of `other`; every pair
// of terms holds complementary literals.
bool Implies(const GuardPath& path, const GuardPath& other) {
  return std::all_of(path.begin(), path.end(), [&](const auto& term) {
    return std::any_of(other.begin(), other.end(), [&](const auto& weaker) { return Weaker(weaker, term); });
  });
}
bool Disjoint(const GuardPath& path, const GuardPath& other) {
  return std::all_of(path.begin(), path.end(), [&](const auto& term) {
    return std::all_of(other.begin(), other.end(), [&](const auto& second) {
      return std::any_of(term.begin(), term.end(), [&](const auto& literal) {
        return std::binary_search(second.begin(), second.end(), GuardLiteral{literal.first, !literal.second});
      });
    });
  });
}
// The first guard among the literals of `path` and `other` whose value `path` leaves open.
std::optional<std::uint32_t> Open(const GuardPath& path, const GuardPath& other) {
  std::set<std::uint32_t> guards;
  for (const auto* terms : {&path, &other})
    for (const auto& term : *terms)
      for (const auto& literal : term) guards.insert(literal.first);
  for (const auto guard : guards) {
    std::optional<bool> fixed;
    bool open = false;
    for (const auto& term : path) {
      const auto at = std::lower_bound(term.begin(), term.end(), GuardLiteral{guard, false});
      if (at == term.end() || at->first != guard || (fixed && *fixed != at->second)) { open = true; break; }
      fixed = at->second;
    }
    if (open) return guard;
  }
  return {};
}

class ValueTemplateExtractor {
public:
  ValueTemplateExtractor(v::Graph &graph, std::vector<ValueIrDiagnostic> &diagnostics,
                const FactSource &facts, LaneRef lane,
                std::uint32_t cta, const UnitFrame& frame, const FunctionContext& context, bool prune,
                std::shared_ptr<std::vector<std::uint8_t>> live, BranchScopeCapture* branch_capture = nullptr)
      : branch_capture_(branch_capture), graph_(graph), diagnostics_(diagnostics), facts_(facts),
        lane_(lane),
        module_(facts.module()), function_(facts.function()), cta_(cta), frame_(frame),
        declarations_(context.declarations), symbols_(context.symbols), symbol_spaces_(context.symbol_spaces),
        loop_ends_(context.loop_ends), source_file_(context.source_file), function_name_(context.function_name),
        instruction_origins_(context.instruction_origins), local_stores_(context.local_stores) {
    live_ = std::move(live);
    if (!live_->empty()) return;
    live_->assign(facts_.Size(lane_), true);
    if (!prune) return;
    // Bit 1: ordinary value semantics. Bit 2: optional branch evidence only.
    // One backward pass, one stored byte per template record. A failed
    // evidence-only producer cannot invalidate unrelated numerical outputs.
    std::map<Key,std::uint8_t> needed;
    for (std::size_t cursor = facts_.Size(lane_); cursor-- > 0;) {
      const auto record = facts_.Meta({lane_, cursor});
      const auto& instruction = function_.instructions.at(record.instruction_id());
      if (record.predicate_value() == false) continue;
      if (branch_capture_ && record.branch_scope_decision() && instruction.predicate) {
        needed[Resolve(*instruction.predicate, instruction.scope)] |= 2;
        (*live_)[cursor] = 2;
        continue;
      }
      if (!record.opaque_compute_diamond_indices().empty()) {
        // A lifted region assigns its writes on some paths only: they stay needed, and its
        // reads and guard are needed once any write is, or always when it stores to .local memory.
        std::uint8_t used = 0;
        for (const auto& name : record.opaque_compute_register_writes())
          if (auto found = needed.find(Resolve(name, instruction.scope)); found != needed.end()) used |= found->second;
        for (const auto index : record.opaque_compute_diamond_indices())
          if (index < function_.instructions.size() && LocalStoreInstruction(function_.instructions[index])) used |= 1;
        (*live_)[cursor] = used;
        if (!used) continue;
        auto reads = record.opaque_compute_register_reads();
        if (instruction.predicate) reads.push_back(*instruction.predicate);
        for (const auto& name : reads)
          if (const auto key = Resolve(name, instruction.scope); declarations_.contains(key)) needed[key] |= used;
        continue;
      }
      // A merged branch's predicate selects the values after its join.
      if (record.merged_arm_join() && instruction.predicate) {
        needed[Resolve(*instruction.predicate, instruction.scope)] |= 1;
        continue;
      }
      if (instruction.control != "none" && instruction.control != "call") continue;
      const bool pure = instruction.semantics.category == "compute" &&
          !instruction.semantics.memory_effect && instruction.control == "none" &&
          !In(instruction.opcode, {"shfl", "vote", "redux", "mma", "addc", "subc", "madc"}) && !Has(instruction, "cc");
      std::uint8_t used=0;
      for(const auto& name:instruction.destinations)
        if(auto found=needed.find(Resolve(name,instruction.scope));found!=needed.end())used|=found->second;
      if (pure && !used) { (*live_)[cursor] = false; continue; }
      const std::uint8_t demand=pure?used:std::uint8_t(used|1);
      (*live_)[cursor]=demand;
      // mbarrier.inval also writes the object's bytes (MbarrierInvalidate): kept like a store.
      if (instruction.opcode == "mbarrier" && instruction.semantics.operation != "inval") {
        (*live_)[cursor] = used;
        for (const auto& name : instruction.destinations) needed.erase(Resolve(name, instruction.scope));
        continue;
      }
      for (const auto& name : instruction.destinations) needed.erase(Resolve(name, instruction.scope));
      for (std::size_t operand = instruction.destinations.empty() ? 0 : 1;
           operand < instruction.operands.size(); ++operand) {
        if ((In(instruction.opcode, {"ld", "ldu", "st"}) ||
             (instruction.opcode == "cp" && Has(instruction, "bulk") && Has(instruction, "tensor"))) &&
            Has(instruction, "L2::cache_hint") && operand + 1 == instruction.operands.size()) continue;
        const auto text = module_.view(instruction.operands[operand]);
        for (std::size_t at = 0; at < text.size();) {
          const auto first = static_cast<unsigned char>(text[at]);
          if (!(std::isalpha(first) || text[at] == '_' || text[at] == '%' || text[at] == '$')) { ++at; continue; }
          const auto begin = at++;
          while (at < text.size() && (std::isalnum(static_cast<unsigned char>(text[at])) ||
                 text[at] == '_' || text[at] == '$' || text[at] == '.')) ++at;
          const auto key = Resolve(text.substr(begin, at - begin), instruction.scope);
          if (declarations_.contains(key)) needed[key] |= demand;
        }
      }
    }
  }
  bool cohort_template = false;
  std::uint64_t publications=0;
  std::vector<LaneRef> members;
  std::unique_ptr<ValueTemplateExtractor> Fork(std::vector<LaneRef> selected) const {
    auto child=std::make_unique<ValueTemplateExtractor>(*this);
    child->lane_=selected.front();
    child->members=std::move(selected);
    child->record_.reset();
    const auto from=std::uint64_t(facts_.Warp(lane_))*facts_.launch().warp_size()+facts_.LaneId(lane_);
    const auto to=SetThread(child->origin_,facts_,child->lane_);
    const auto rebind=[&](auto&& self, Expression& value)->void {
      if (!value.missing_definition && value.components.empty())
        value.value=graph_.bindThread(value.value,cta_,from,to);
      for (auto& component:value.components) self(self,component);
    };
    for (auto& [key,value]:child->registers_) rebind(rebind,value);
    for (auto& [key,value]:child->inputs_) rebind(rebind,value);
    if (child->carry_) rebind(rebind,*child->carry_);
    if (child->shuffle_source_) rebind(rebind,*child->shuffle_source_);
    if (child->shuffle_selector_) rebind(rebind,*child->shuffle_selector_);
    if (child->merged_) {
      rebind(rebind,child->merged_->runs);
      for (auto& [key,value]:child->merged_->before) if (value) rebind(rebind,*value);
      for (auto& loaded:child->merged_->loads) {
        loaded.base=graph_.bindThread(loaded.base,cta_,from,to);
        for (auto& value:loaded.registers) rebind(rebind,value);
      }
    }
    return child;
  }
  std::vector<std::vector<LaneRef>> Partition() const;
  bool demand_mode = false;
  AddressValueExtractor::Definition demand_definition;
  std::function<std::optional<DynamicKey>(const RecordView&, const Key&)> demand_producer;
  std::function<std::optional<Expression>(std::uint32_t, DynamicKey, const std::string&)> demand_peer;
  std::optional<Expression> Demand(const RecordView&, const std::string&);
  std::optional<Expression> DemandOperand(const std::string&, const Type&);
  std::map<std::pair<DynamicKey, std::string>, std::optional<Expression>> demand_cache;
  unsigned demand_depth = 0;
  bool Step();
  [[nodiscard]] bool Done() const { return cursor_ >= facts_.Size(lane_); }
  void Abandon() { Fail("unresolved_cross_thread_value", "Cross-thread operand has no matching extracted occurrence"); Invalidate(); shuffle_source_.reset(); shuffle_selector_.reset(); matrix_pending_ = false; warp_mma_pending_ = false; reduction_pending_ = false; ++cursor_; }


private:
  BranchScopeCapture* branch_capture_ = nullptr;
  std::shared_ptr<std::vector<std::uint8_t>> live_;
  v::NodeUse Add(std::unique_ptr<const v::Operation> op, std::vector<v::Value> operands = {}, v::SourceOrigin origin = {}) {
    origin.lowering_index = suboperation_++;
    if (demand_mode) return {graph_.add(std::move(op), std::move(operands), std::move(origin)), 0};
    if (const auto* coordinate = dynamic_cast<const v::CoordinateOp*>(op.get())) {
      const auto axis = static_cast<unsigned>(coordinate->axis());
      const auto cluster = frame_.cluster_dim();
      switch (coordinate->kind()) {
        case v::CoordinateKind::ThreadIndex: origin.coordinate_value = origin.thread_coordinates->at(axis); break;
        case v::CoordinateKind::BlockIndex: origin.coordinate_value = frame_.SelfCoordinates()[axis]; break;
        case v::CoordinateKind::ClusterBlockIndex: origin.coordinate_value = frame_.SelfCoordinates()[axis] % cluster[axis]; break;
        case v::CoordinateKind::ClusterRank:
          origin.coordinate_value = frame_.self().value; break;
      }
    }
    return graph_.addTemplate(std::move(op), std::move(operands), std::move(origin));
  }
  std::optional<Integer> Known(std::string text, bool uniform_only = false) const;
  std::optional<Expression> Special(std::string text);
  Expression LinearThread();
  bool Shuffle();
  // The value cross-thread readers select their sources from: the one published
  // Value when every source thread's publication is that producer at its own
  // thread and its node holds every source thread (fromThread re-indexes it), else
  // a ThreadForward node (Graph::addThreadUnion) of each source thread's own
  // publication. `sources` are (warp, lane) pairs, possibly repeated. Null when
  // a publication is missing.
  std::optional<v::Value> SourceValue(std::uint32_t instruction, std::uint64_t occurrence,
                                      const std::vector<std::pair<std::uint32_t, std::uint32_t>>& sources);
  bool MatrixMemory();
  bool WarpMma();
  // vote.sync and redux.sync: a combination of every membermask lane's operand.
  bool LaneReduction();
  // RunCollective's diagnostic code for the LaneReduction at instruction_.
  const char* LaneReductionCode() const { return instruction_->opcode == "redux" ? "unsupported_redux" : "unsupported_vote"; }
  // One Step of a collective: false while `run` waits for its peers; otherwise
  // a thrown exception is reported as `code`, `pending` clears and the cursor
  // advances.
  bool RunCollective(bool (ValueTemplateExtractor::*run)(), const char* code, bool& pending);
  // origin_ for `record` on this lane: CTA, thread, occurrence, source file and
  // function; the other fields are empty.
  void BeginOrigin(const RecordView& record);
  // The Partition rule of a whitelisted instruction (built on first use); null
  // where Partition returns at its whitelist or constant-copy exit.
  const PartitionSpec* PartitionRule(const Instruction&) const;

  Key Resolve(std::string_view name, const Scope &scope) const;
  std::optional<Expression> Operand(std::string text, const Type &type, bool defer_missing = false);
  // The value of declared register `key` before any instruction on this thread's path assigns it.
  Expression Unassigned(const Key& key);
  // `key`'s value on this thread's path: its binding, else its unspecified initial value while no
  // instruction on the path has assigned it (Unassigned); none for an invalidated or undeclared register.
  std::optional<Expression> CurrentValue(const Key& key);
  std::optional<Expression> Address(std::string text);
  std::optional<Expression> Bits(Expression expression);
  std::optional<Expression> Part(Expression expression, const Type& type, std::uint32_t bit);
  Expression Constant(const Type &type, const std::string &text);
  v::NodeUse Operation(std::string name, std::string attributes,
                           const std::vector<Type> &results,
                           const std::vector<Expression> &operands,
                           std::vector<std::string> labels = {});
  void InstructionValue();
  // The opaque compute region of record_ (instruction_ is its branch), if-converted.
  bool LiftRegion();
  // select(condition, value, other), the alternatives met in one type; `value` itself when both
  // are the same value; none when a multi-component alternative has no bit form.
  std::optional<Expression> Choose(const Expression& condition, const Expression& value, const Expression& other);
  // The merged one-armed diamond being read (execution.cpp, two-arm merge): every thread ran its
  // fall-through arm, which runs exactly when `runs` holds. A register the arm writes keeps its
  // value from before the arm (`before`) and becomes select(runs, with the arm, without it) at the
  // join. A tcgen05.st in the arm stores select(runs, data, the cells' current values); those are
  // the registers of a tcgen05.ld of the same cells in the arm since its last tcgen05.st (`loads`).
  struct MergedArmValues {
    std::uint32_t branch, join;
    Expression runs;
    std::map<Key, std::optional<Expression>> before;
    struct TmemLoad {
      std::string address;
      v::Value base;
      std::vector<Expression> registers;
    };
    std::vector<TmemLoad> loads;
  };
  std::optional<MergedArmValues> merged_;
  // At the first record outside the arm: the selects of the arm's registers.
  void FinishMergedArm();
  void Compute();
  void Memory();
  void AsyncCopy();
  std::optional<std::pair<Expression, bool>> WholeCopyCondition(std::string operand, bool predicate,
                                                                const Integer &bytes);
  void BulkCopy();
  void BulkReduce();
  std::optional<v::Value> LinearBulkEndpoint(unsigned operand, std::uint64_t size, const std::string& space,
                                             bool cluster_source);
  void Tma();
  void TensorMap();
  void MbarrierInvalidate();
  // A bracketed operand's address in `space` ("shared" or "global"), plus `extra` bytes.
  std::optional<v::Value> SpaceAddress(std::string_view operand, std::string_view space,
                                       std::optional<Expression> extra);
  bool ResolvedTma();
  // An abstract TMA's tile region (one per tile::gather4 row, else one): its first byte's
  // allocation-relative offset (s64), and a gather row's guard when the row is not proven
  // inside the tensor (the offset then holds the row clamped into the tensor).
  struct TmaRegionOrigin {
    Expression offset;
    std::optional<Expression> guard;
    bool exact = false;  // `offset` is the literal TmaAccesses::region_valued
  };
  // Each region an abstract pair of `transfer` names, from the resolved `descriptor`;
  // nothing when a region is not modeled (see ResolvedTma).
  std::optional<std::vector<std::optional<TmaRegionOrigin>>> TmaRegionOrigins(
      const TmaAccesses &transfer, const nlohmann::json &descriptor);
  void Tcgen();
  void TcgenMma();
  v::Value Native(v::NativeMatrixOpcode, v::ValueType, std::vector<v::Value>, v::NativeMatrixAttributes);
  v::Value ClusterRank();
  // Pointer constant of a resolved slot binding.
  v::Value SlotAddressValue(const algorithm::ConstantAddressBinding& binding);
  std::optional<Expression> Reinterpret(Expression expression, const Type &type);
  void Fail(std::string code, std::string message);
  void Invalidate();
  void Bind(const std::string &name, Expression value);
  v::Graph &graph_;
  std::vector<ValueIrDiagnostic> &diagnostics_;
  const FactSource &facts_;
  LaneRef lane_;
  const Module &module_;
  const Function &function_;
  // The value-IR CTA id (pointer owners, origins, bindThread): the value
  // view's CtaFacts::Linear(), or kAddressExtractorCta. Never read from frame_.
  std::uint32_t cta_;
  // Naming only: origin coordinates and the cluster rank.
  const UnitFrame& frame_;
  const std::map<Key, Type>& declarations_;
  const std::set<Key>& symbols_;
  const std::map<Key, std::string>& symbol_spaces_;
  std::map<Key, Expression> registers_, inputs_;
  std::set<Key> invalidated_;
  std::optional<Expression> carry_;
  const Instruction *instruction_ = nullptr;
  // ReadFull, or CopyMeta at dead positions: the record copy only (RecordCopy).
  // record_ views it, the expanded record, or a Demand producer.
  RecordCopy current_record_;
  std::optional<RecordView> record_;
  v::SourceOrigin origin_;
  std::size_t cursor_ = 0;
  std::uint64_t suboperation_ = 0;
  std::map<std::uint32_t, std::uint64_t> iterations_;
  const std::map<std::uint32_t, std::uint32_t>& loop_ends_;
  const algorithm::SharedString &source_file_, &function_name_;
  const std::vector<v::SourceOrigin>& instruction_origins_;
  const LocalRegionStores& local_stores_;
  std::optional<Expression> shuffle_source_, shuffle_selector_;
  std::uint32_t shuffle_from_ = 0;
  bool shuffle_valid_ = false;
  bool matrix_pending_ = false;
  bool warp_mma_pending_ = false;
  bool reduction_pending_ = false;
  // Inside LiftRegion: record_ is the region's branch, whose known registers predate the region.
  bool lifting_ = false;
public:
  using ShuffleKey = std::tuple<std::uint32_t, std::uint32_t, std::uint64_t, std::uint32_t,
      std::vector<std::pair<std::uint64_t, std::uint64_t>>>;
  struct WarpMmaState {
    std::array<std::vector<v::Value>, 32> inputs;
    std::array<v::SourceOrigin,32> origins;
    std::uint32_t arrived = 0, consumed = 0;
    std::optional<v::Value> result;
    std::array<v::Value,16> indices;
  };
  struct MatrixAddressState {
    std::array<std::optional<Expression>,32> addresses;
    std::uint32_t arrived = 0, consumed = 0;
  };
  std::map<ShuffleKey, Expression>* shuffle_values = nullptr;
  std::map<ShuffleKey, MatrixAddressState>* matrix_addresses = nullptr;
  std::map<ShuffleKey, WarpMmaState>* warp_mma_inputs = nullptr;
  // The builder's Partition rules (ValueIrExtractor::partition_specs_).
  PartitionSpecCache* partition_specs = nullptr;
  // The value view's slot resolution (bulk-copy and TMA endpoints, MMA
  // descriptors); null in the address-value extractor.
  Resources* resources = nullptr;
  const TmemProvenanceIndex* tmem_provenance = nullptr;

};

// Partition's whitelist: the instruction classes whose members can disagree
// on a fact or on a published value.
struct PartitionKinds {
  bool tensor=false, bulk=false, copy=false, mma=false, matrix=false, shuffle=false, election=false,
       allocation=false, tc_mma=false, parameter=false, reduction=false;
};
PartitionKinds PartitionKindsOf(const Instruction& i) {
  PartitionKinds kind;
  kind.tensor=i.opcode=="cp" && Has(i,"tensor") && Has(i,"bulk");
  kind.bulk=i.opcode=="cp" && Has(i,"bulk");
  kind.copy=i.opcode=="cp" && i.semantics.operation=="copy";
  kind.mma=i.opcode=="mma";
  kind.matrix=In(i.opcode,{"ldmatrix","stmatrix"});
  kind.shuffle=i.opcode=="shfl";
  kind.election=i.opcode=="elect";
  kind.allocation=i.opcode=="tcgen05" && Has(i,"alloc");
  kind.tc_mma=i.opcode=="tcgen05" && Has(i,"mma");
  kind.parameter=i.opcode=="ld" && Has(i,"param");
  kind.reduction=In(i.opcode,{"vote","redux"});
  return kind;
}

// The Partition rule of a whitelisted instruction: its key fragments in the
// order PartitionCursor::Next appends them; std::nullopt for a copy whose
// operands are all integers (Partition's constant-copy exit). Token fragments
// are the matches of `names` over each operand spelling, in order; an operand
// whose trimmed spelling is an integer contributes none.
std::optional<PartitionSpec> BuildPartitionSpec(const Module& module, const Instruction& i,
                                                const PartitionKinds& kind) {
  if (kind.copy && std::all_of(i.operands.begin()+2,i.operands.end(),[&](const auto& operand) {
        return ParseInteger(Trim(module.view(operand))).has_value();
      })) return std::nullopt;
  PartitionSpec spec;
  spec.instruction=i.id;
  auto& fragments=spec.fragments;
  if (kind.mma || kind.matrix) fragments.push_back({PartitionFragment::Warp,{},{},0});
  if (kind.allocation) fragments.push_back({PartitionFragment::Issuer,{},{},0});
  spec.record_at=static_cast<std::uint32_t>(fragments.size());
  // false: the operand is absent; its Missing fragment throws std::out_of_range
  // in PartitionCursor::Next.
  const auto tokens=[&](std::size_t operand) {
    if (operand>=i.operands.size()) {
      fragments.push_back({PartitionFragment::Missing,{},{},static_cast<std::uint32_t>(operand)});
      return false;
    }
    static const std::regex names(R"([%A-Za-z_$][A-Za-z0-9_.$]*)");
    const std::string spelling(module.view(i.operands[operand]));
    if (ParseInteger(Trim(spelling))) return true;
    for (std::sregex_iterator at(spelling.begin(),spelling.end(),names),end;at!=end;++at)
      fragments.push_back({PartitionFragment::Token,at->str(),{},0});
    return true;
  };
  if (kind.bulk || kind.parameter)
    for (std::size_t operand=i.destinations.empty()?0:1;operand<i.operands.size();++operand) tokens(operand);
  if (kind.tc_mma)
    for (std::size_t operand=Has(i,"sp")?4:3;operand<i.operands.size();++operand) tokens(operand);
  if (kind.tensor) fragments.push_back({PartitionFragment::Contract,{},{},0});
  if (kind.copy) for (std::size_t operand=2;operand<i.operands.size();++operand) tokens(operand);
  if (kind.election) {
    if (!tokens(1)) return spec;
    auto text=Trim(module.view(i.operands[1]));
    auto literal=ParseInteger(text);
    fragments.push_back({PartitionFragment::Elected,std::move(text),std::move(literal),1});
  }
  // shfl: the members share b, c and the membermask (the first phase reads the
  // leader's); source validity and producers are per-member parts of the read.
  if (kind.shuffle)
    for (unsigned operand=2;operand<5;++operand) if (!tokens(operand)) return spec;
  if (kind.reduction) tokens(2);
  return spec;
}

} // namespace
} // namespace sentryward::frontend::ptx

#include "value_ir_partitions.hpp"
#include "value_ir_operands.hpp"
#include "value_ir_collectives.hpp"
#include "value_ir_control.hpp"
#include "value_ir_compute.hpp"
#include "value_ir_memory.hpp"
#include "value_ir_tma.hpp"
#include "value_ir_tcgen.hpp"
#include "value_ir_demand.hpp"
