#include "sentryward/frontend/ptx/value_ir.h"
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

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
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
        instruction_origins_(context.instruction_origins) {
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
        // reads and guard are needed once any write is.
        std::uint8_t used = 0;
        for (const auto& name : record.opaque_compute_register_writes())
          if (auto found = needed.find(Resolve(name, instruction.scope)); found != needed.end()) used |= found->second;
        (*live_)[cursor] = used;
        if (!used) continue;
        auto reads = record.opaque_compute_register_reads();
        if (instruction.predicate) reads.push_back(*instruction.predicate);
        for (const auto& name : reads)
          if (const auto key = Resolve(name, instruction.scope); declarations_.contains(key)) needed[key] |= used;
        continue;
      }
      if (instruction.control != "none" && instruction.control != "call") continue;
      const bool pure = instruction.semantics.category == "compute" &&
          !instruction.semantics.memory_effect && instruction.control == "none" &&
          !In(instruction.opcode, {"shfl", "vote", "mma", "addc", "subc", "madc"}) && !Has(instruction, "cc");
      std::uint8_t used=0;
      for(const auto& name:instruction.destinations)
        if(auto found=needed.find(Resolve(name,instruction.scope));found!=needed.end())used|=found->second;
      if (pure && !used) { (*live_)[cursor] = false; continue; }
      const std::uint8_t demand=pure?used:std::uint8_t(used|1);
      (*live_)[cursor]=demand;
      if (instruction.opcode == "mbarrier") {
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
  void Abandon() { Fail("unresolved_cross_thread_value", "Cross-thread operand has no matching extracted occurrence"); Invalidate(); shuffle_source_.reset(); shuffle_selector_.reset(); matrix_pending_ = false; warp_mma_pending_ = false; vote_pending_ = false; ++cursor_; }


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
  bool Vote();
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
  void Compute();
  void Memory();
  void AsyncCopy();
  void BulkCopy();
  void Tma();
  void TensorMap();
  // A bracketed operand's address in `space` ("shared" or "global"), plus `extra` bytes.
  std::optional<v::Value> SpaceAddress(std::string_view operand, std::string_view space,
                                       std::optional<Expression> extra);
  bool ResolvedTma();
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
  std::optional<Expression> shuffle_source_, shuffle_selector_;
  std::uint32_t shuffle_from_ = 0;
  bool shuffle_valid_ = false;
  bool matrix_pending_ = false;
  bool warp_mma_pending_ = false;
  bool vote_pending_ = false;
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
       allocation=false, tc_mma=false, parameter=false, vote=false;
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
  kind.vote=i.opcode=="vote";
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
  if (kind.vote) tokens(2);
  return spec;
}

const PartitionSpec* ValueTemplateExtractor::PartitionRule(const Instruction& i) const {
  // Only an opcode some rule can whitelist gets an entry (the cache stays
  // sparse); its flags and rule are computed once per function.
  if (!In(i.opcode,{"cp","ld","mma","ldmatrix","stmatrix","shfl","elect","tcgen05","vote"})) return nullptr;
  if (!partition_specs) throw std::logic_error("Partition without the value view's rule cache");
  auto& rules=partition_specs->rules;
  auto found=rules.find(i.id);
  if (found==rules.end()) {
    std::unique_ptr<const PartitionSpec> rule;
    const auto kind=PartitionKindsOf(i);
    if (kind.bulk || kind.copy || kind.mma || kind.matrix || kind.shuffle || kind.election || kind.allocation ||
        kind.tc_mma || kind.parameter || kind.vote)
      if (auto spec=BuildPartitionSpec(module_,i,kind)) rule=std::make_unique<const PartitionSpec>(std::move(*spec));
    found=rules.emplace(i.id,std::move(rule)).first;
  }
  return found->second.get();
}

std::vector<std::vector<LaneRef>> ValueTemplateExtractor::Partition() const {
  if (Done() || members.size()<2) return {};
  const auto meta=facts_.Meta({lane_,cursor_});
  const auto& i=function_.instructions.at(meta.instruction_id());
  if (!(*live_)[cursor_] || meta.predicate_value()==false) return {};
  const auto* spec=PartitionRule(i);
  if (!spec) return {};
  // Per member: the fact-stage class, then the structural keys of its matrix
  // peers, so a veto or an exception arises at the member that causes it.
  // Parts are numbered by first appearance, so they are ordered by first
  // member and part 0 holds the leader.
  auto stage1=facts_.BeginPartition(members,cursor_,*spec);
  std::vector<std::vector<LaneRef>> parts;
  std::unordered_map<std::string,std::uint32_t> numbering; // "<class>:" ++ stage-2 key -> part
  std::string key;
  MemberFacts derived;
  try {
  while (!stage1.Done()) {
    const auto member=stage1.Member();
    // The member's stage-1 key enters the part key through its class.
    stage1.Next(derived);
    key.assign(std::to_string(stage1.Class(stage1.Evaluated()-1)));
    key+=':';
    if (matrix_pending_) {
      const auto found=matrix_addresses->find({facts_.Warp(member),i.id,facts_.Meta({member,cursor_}).occurrence(),0,
                                               origin_.unroll_context});
      if (found==matrix_addresses->end() || found->second.arrived!=UINT32_MAX) {
        stage1.Finish(query::PartitionEnd::Vetoed);
        return {};
      }
      const auto shape=DecodeMatrixShape(module_,i,i.opcode=="stmatrix");
      for (unsigned m=0;m<shape.count;++m) for (unsigned half=0;half<(shape.transpose?2u:1u);++half) {
        const unsigned peer=8*m+(shape.transpose?2*(facts_.LaneId(member)%4)+half:facts_.LaneId(member)/4);
        key+="source:";
        key+=graph_.structuralKey(found->second.addresses[peer]->value,origin_);
      }
    }
    const auto [part,added]=numbering.try_emplace(key,static_cast<std::uint32_t>(parts.size()));
    if (added) parts.emplace_back();
    parts[part->second].push_back(member);
  }
  } catch (...) {
    // An exception ends the partition stage (recorded as Threw), then propagates.
    stage1.Finish(query::PartitionEnd::Threw);
    throw;
  }
  stage1.Finish(query::PartitionEnd::Complete);
  if (parts.size()<2) return {};
  return parts;
}

FunctionContext::FunctionContext(const Module& module, const Function& function)
    : source_file(module.source_name), function_name(function.name) {
  instruction_origins.reserve(function.instructions.size());
  for (const auto& instruction : function.instructions) {
    auto& origin = instruction_origins.emplace_back();
    origin.source_text = std::string(module.view(instruction.source));
    origin.scope = ScopeText(instruction.scope);
    if (instruction.predicate)
      origin.predicate = (instruction.predicate_negated ? "!" : "") + *instruction.predicate;
  }
  static const std::regex type_regex("\\.(pred|bf[0-9]+(?:x[0-9]+)?|[busf][0-9]+(?:x[0-9]+)?)\\b");
  static const std::regex name_regex("(%?[A-Za-z_$][A-Za-z0-9_$]*)(?:<([0-9]+)>)?");
  const auto parse = [&](const Directive &directive) {
    if (!In(directive.name, {".reg", ".shared", ".global", ".local", ".const", ".param"})) return;
    const auto text = std::string(module.view(directive.arguments));
    std::smatch type_match;
    if (!std::regex_search(text, type_match, type_regex)) return;
    auto type = PtxType(type_match[1].str());
    if (!type) return;
    const auto names = text.substr(static_cast<std::size_t>(type_match.position() + type_match.length()));
    for (const auto &part : Split(names, ',')) {
      std::smatch name;
      if (!std::regex_search(part, name, name_regex)) continue;
      if (directive.name != ".reg") {
        symbols.emplace(directive.scope, name[1].str());
        symbol_spaces[{directive.scope, name[1].str()}] = directive.name.substr(1);
        continue;
      }
      if (name[2].matched) {
        const auto count = std::stoul(name[2]);
        if (count > 100000) throw std::invalid_argument("excessive PTX register declaration");
        for (std::size_t n = 0; n < count; ++n)
          declarations.emplace(Key{directive.scope, name[1].str() + std::to_string(n)}, *type);
      } else declarations.emplace(Key{directive.scope, name[1].str()}, *type);
    }
  };
  for (const auto &directive : module.directives) parse(directive);
  for (const auto &directive : function.directives) parse(directive);
  for (const auto &parameter : function.parameters)
    if (parameter.name) symbols.emplace(Scope{}, *parameter.name);
  for (const auto& instruction : function.instructions)
    if (instruction.branch_target && *instruction.branch_target <= instruction.id)
      loop_ends[*instruction.branch_target] = std::max(loop_ends[*instruction.branch_target], instruction.id);
}
Key ValueTemplateExtractor::Resolve(std::string_view name, const Scope &scope) const {
  auto search = scope;
  for (;;) {
    Key key{search, std::string(name)};
    if (declarations_.contains(key) || symbols_.contains(key)) return key;
    if (search.empty()) return key;
    search.pop_back();
  }
}
void ValueTemplateExtractor::Fail(std::string code, std::string message) {
  if(branch_capture_ && !demand_mode && cursor_<live_->size() && (*live_)[cursor_]==2) {
    branch_capture_->unavailable(code+": "+message);return;
  }
  diagnostics_.push_back({std::move(code), std::move(message), origin_});
}
void ValueTemplateExtractor::Bind(const std::string &name, Expression value) {
  if (name != "_") {
    const auto key = Resolve(name, instruction_->scope);
    const auto declared = declarations_.find(key);
    if (declared == declarations_.end()) {
      Fail("missing_register_declaration", "Destination '" + name + "' has no scoped register declaration");
      registers_.erase(key); invalidated_.insert(key); return;
    }
    if (declared->second.value.bitSize() != value.type.value.bitSize()) {
      if (declared->second.value.isScalar() && value.type.value.isScalar() &&
          declared->second.value.element().isBitLike() && value.type.value.element().isBitLike() &&
          value.type.bits < declared->second.bits && value.type.spelling != "pred" &&
          In(instruction_->opcode, {"ld", "ldu", "cvt"})) {
        const auto kind = value.type.value.element().kind() == v::ElementKind::SignedInteger ? "s" : "u";
        if (value.type.value.element().kind() == v::ElementKind::Bit) value = *Reinterpret(value, *PtxType("u" + std::to_string(value.type.bits)));
        const auto widened = *PtxType(std::string(kind) + std::to_string(declared->second.bits));
        value = {Add(std::make_unique<v::ConvertOp>(widened.value), {value.value}, origin_).result(), widened};
      } else {
        Fail("destination_width_mismatch", "Destination '" + name + "' declaration does not match the PTX result width");
        registers_.erase(key); invalidated_.insert(key); return;
      }
    }
    registers_.insert_or_assign(key, std::move(value));
    invalidated_.erase(key);
  }
}
void ValueTemplateExtractor::Invalidate() {
  for (const auto &name : instruction_->destinations) {
    const auto key = Resolve(name, instruction_->scope);
    registers_.erase(key);
    invalidated_.insert(key);
  }
  if (Has(*instruction_, "cc")) carry_.reset();
}
Expression ValueTemplateExtractor::Constant(const Type &type, const std::string &text) {
  return {Add(std::make_unique<v::ConstantOp>(type.value, text), {}, origin_).result(), type};
}
std::optional<Integer> ValueTemplateExtractor::Known(std::string text, bool uniform_only) const {
  text = Trim(text);
  if (auto literal = ParseInteger(text)) return literal;
  if (text == "true") return Integer(1);
  if (text == "false") return Integer(0);
  if (lifting_) return {};
  if (!uniform_only) {
    if (const auto* value = facts_.Value(*record_, text)) return ScalarInteger(*value);
    return {};
  }
  return facts_.LegacyUniformOnlyValue({lane_, cursor_}, members, text);
}
Expression ValueTemplateExtractor::LinearThread() {
  auto coordinate = [&](v::Axis axis) {
    return Expression{Add(std::make_unique<v::CoordinateOp>(v::CoordinateKind::ThreadIndex, axis), {}, origin_).result(), U32()};
  };
  auto result = coordinate(v::Axis::X);
  const auto& dim = facts_.launch().block_dim();
  for (unsigned axis = 1; axis < 3; ++axis) {
    if (dim[axis] == 1) continue;
    auto scale = Constant(U32(), std::to_string(axis == 1 ? dim[0] : dim[0] * dim[1]));
    auto part = Expression{Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::Multiply, U32().value),
        {coordinate(static_cast<v::Axis>(axis)).value, scale.value}, origin_).result(), U32()};
    result = {Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::Add, U32().value), {result.value, part.value}, origin_).result(), U32()};
  }
  return result;
}
std::optional<Expression> ValueTemplateExtractor::Special(std::string text) {
  for (const auto& [prefix, kind] : std::vector<std::pair<std::string, v::CoordinateKind>>{
      {"%tid.", v::CoordinateKind::ThreadIndex}, {"%ctaid.", v::CoordinateKind::BlockIndex},
      {"%cluster_ctaid.", v::CoordinateKind::ClusterBlockIndex}}) {
    if (text.starts_with(prefix) && text.size() == prefix.size()+1) {
      const auto axis = std::string("xyz").find(text.back());
      if (axis < 3) return Expression{Add(std::make_unique<v::CoordinateOp>(kind, static_cast<v::Axis>(axis)), {}, origin_).result(), U32()};
    }
  }
  if (text == "%cluster_ctarank")
    return Expression{Add(std::make_unique<v::CoordinateOp>(v::CoordinateKind::ClusterRank), {}, origin_).result(), U32()};
  if (text == "%laneid" || text == "%warpid") {
    auto thread = LinearThread(), width = Constant(U32(), std::to_string(facts_.launch().warp_size()));
    return Expression{Add(std::make_unique<v::BinaryOp>(text == "%laneid" ? v::BinaryOpcode::Remainder : v::BinaryOpcode::Divide, U32().value),
        {thread.value, width.value}, origin_).result(), U32()};
  }
  if (text.starts_with("%ntid.") && std::string("xyz").find(text.back()) < 3)
    return Constant(U32(), std::to_string(facts_.launch().block_dim()[std::string("xyz").find(text.back())]));
  if (text.starts_with("%clock") || text.starts_with("%globaltimer")) {
    Fail("unsupported_special_register", "Clock/timer values have no coordinate or parameter representation"); return {};
  }
  if (const auto known = facts_.Special(text))
    return Constant(U32(), known->convert_to<std::string>());
  return {};
}
std::optional<Expression> ValueTemplateExtractor::Bits(Expression expression) {
  if (expression.missing_definition) { Fail("missing_definition",*expression.missing_definition); return {}; }
  const auto bits = expression.type.value.bitSize();
  auto integer = PtxType("u" + std::to_string(bits));
  // Internal nibble storage, not a PTX register type. Packed e2m1 conversions
  // use it before widening to the eight-bit register holding both nibbles.
  if (bits == 4) integer = Type{v::ValueType::scalar(v::ElementType::unsignedInteger(4)), "u4", 4};
  if (!integer) { Fail("unsupported_register_width", "Register bit representation exceeds supported scalar width"); return {}; }
  if (expression.components.empty()) {
    if (expression.type.value == integer->value) return expression;
    if (!expression.type.value.isScalar()) { Fail("unresolved_register_components", "Matrix register components require an explicit mapping"); return {}; }
    return Expression{Add(std::make_unique<v::BitcastOp>(integer->value), {expression.value}, origin_).result(), *integer};
  }
  std::optional<Expression> result;
  std::uint32_t offset = 0;
  for (auto component : expression.components) {
    auto value = Bits(component);
    if (!value) return {};
    if (value->type.value != integer->value)
      value = Expression{Add(std::make_unique<v::ConvertOp>(integer->value), {value->value}, origin_).result(), *integer};
    if (offset) {
      auto shift = Constant(U32(), std::to_string(offset));
      value = Expression{Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::ShiftLeft, integer->value), {value->value, shift.value}, origin_).result(), *integer};
    }
    if (result) value = Expression{Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::BitOr, integer->value), {result->value, value->value}, origin_).result(), *integer};
    result = value;
    offset += static_cast<std::uint32_t>(component.type.value.bitSize());
  }
  if (offset != bits) { Fail("invalid_register_components", "Register components do not cover its bit width"); return {}; }
  return result;
}
std::optional<Expression> ValueTemplateExtractor::Part(Expression expression, const Type& type, std::uint32_t bit) {
  std::uint32_t offset = 0;
  for (auto component : expression.components) {
    if (offset == bit && component.type.value.bitSize() == type.value.bitSize()) return Reinterpret(component, type);
    offset += static_cast<std::uint32_t>(component.type.value.bitSize());
  }
  if (expression.components.empty() && expression.type.value.isMatrix() && expression.type.bits == type.bits && bit % type.bits == 0) {
    auto index = Constant(U32(), std::to_string(bit / type.bits));
    return Reinterpret({graph_.component(expression.value, {index.value}), Type{v::ValueType::scalar(expression.type.value.element()), expression.type.spelling, expression.type.bits}}, type);
  }
  auto bits = Bits(expression);
  if (!bits || bit + type.value.bitSize() > expression.type.value.bitSize()) return {};
  if (bit) {
    auto shift = Constant(U32(), std::to_string(bit));
    bits = Expression{Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::ShiftRight, bits->type.value), {bits->value, shift.value}, origin_).result(), bits->type};
  }
  const auto narrow = type.bits == 4
      ? Type{v::ValueType::scalar(v::ElementType::unsignedInteger(4)), "u4", 4}
      : *PtxType("u" + std::to_string(type.bits));
  if (narrow.value != bits->type.value)
    bits = Expression{Add(std::make_unique<v::ConvertOp>(narrow.value), {bits->value}, origin_).result(), narrow};
  return Reinterpret(*bits, type);
}
std::optional<Expression> ValueTemplateExtractor::Reinterpret(Expression expression, const Type &type) {
  if (expression.missing_definition) { Fail("missing_definition",*expression.missing_definition); return {}; }
  if (!expression.components.empty() && type.value.isScalar()) {
    auto bits = Bits(expression);
    if (!bits) return {};
    expression = *bits;
  }
  if (expression.type.value == type.value) return expression;
  if (expression.type.value.bitSize() != type.value.bitSize()) {
    Fail("operand_width_mismatch", "Operand width does not match PTX instruction type " + type.spelling);
    return {};
  }
  if (type.elements > 1) {
    const auto scalar = Type{v::ValueType::scalar(type.value.element()), type.spelling.substr(0, type.spelling.find('x')), type.bits};
    Expression result{{}, type};
    for (std::uint32_t n = 0; n < type.elements; ++n) {
      auto part = Part(expression, scalar, n * type.bits);
      if (!part) return {};
      result.components.push_back(*part);
    }
    return result;
  }
  if (!expression.components.empty()) {
    auto bits = Bits(expression);
    if (!bits) return {};
    return Reinterpret(*bits, type);
  }
  return Expression{Add(std::make_unique<v::BitcastOp>(type.value), {expression.value}, origin_).result(), type};
}
v::NodeUse ValueTemplateExtractor::Operation(std::string name, std::string attributes,
                                      const std::vector<Type> &results,
                                      const std::vector<Expression> &operands,
                                      std::vector<std::string> labels) {
  (void)attributes;
  std::vector<v::ValueType> result_types;
  std::vector<v::Value> values;
  for (const auto &type : results) result_types.push_back(type.value);
  for (const auto &operand : operands) values.push_back(operand.value);
  const bool helper = name == "ptx.address_add" || (name == "ptx.not" && labels.empty());
  const auto op = name.substr(4);
  const auto &i = *instruction_;
  v::ArithmeticAttributes arithmetic;
  if (Has(i, "rn") || Has(i, "rni")) arithmetic.rounding = v::RoundingMode::NearestEven;
  if (Has(i, "rz") || Has(i, "rzi")) arithmetic.rounding = v::RoundingMode::TowardZero;
  if (Has(i, "rm") || Has(i, "rmi")) arithmetic.rounding = v::RoundingMode::TowardNegative;
  if (Has(i, "rp") || Has(i, "rpi")) arithmetic.rounding = v::RoundingMode::TowardPositive;
  if (Has(i, "rna")) arithmetic.rounding = v::RoundingMode::TiesAway;
  arithmetic.saturate = Has(i, "sat");
  arithmetic.flush_to_zero = Has(i, "ftz");
  arithmetic.propagate_nan = Has(i, "NaN");
  if (Has(i, "wide")) arithmetic.multiply_mode = v::IntegerMultiplyMode::Wide;
  if (Has(i, "hi")) arithmetic.multiply_mode = v::IntegerMultiplyMode::High;
  std::unique_ptr<const v::Operation> operation;
  const auto check_modifiers = [&](std::initializer_list<std::string_view> allowed) {
    if (helper) return;
    for (const auto &modifier : i.modifiers)
      if (!PtxType(modifier) && !In(modifier, allowed))
        throw std::invalid_argument("PTX modifier '" + modifier + "' has no exact core semantics on '" + i.opcode + "'");
  };
  static const std::map<std::string, v::BinaryOpcode> binary{
      {"add", v::BinaryOpcode::Add}, {"sub", v::BinaryOpcode::Subtract},
      {"mul", v::BinaryOpcode::Multiply}, {"div", v::BinaryOpcode::Divide}, {"rem", v::BinaryOpcode::Remainder},
      {"and", v::BinaryOpcode::BitAnd}, {"or", v::BinaryOpcode::BitOr}, {"xor", v::BinaryOpcode::BitXor},
      {"shl", v::BinaryOpcode::ShiftLeft}, {"shr", v::BinaryOpcode::ShiftRight},
      {"min", v::BinaryOpcode::Minimum}, {"max", v::BinaryOpcode::Maximum}};
  if (name == "ptx.address_add") operation = std::make_unique<v::BinaryOp>(v::BinaryOpcode::Add, results[0].value);
  else if (auto found = binary.find(op); found != binary.end() && !Has(i, "cc")) {
    if (op == "mul") check_modifiers({"rn", "rz", "rm", "rp", "ftz", "sat", "lo", "hi", "wide"});
    else if (op == "div" && results[0].value.element().kind() == v::ElementKind::Floating)
      check_modifiers({"rn", "rz", "rm", "rp", "ftz", "sat", "approx", "full"});
    else if (In(op, {"add", "sub", "div"})) check_modifiers({"rn", "rz", "rm", "rp", "ftz", "sat"});
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-max
    else if (In(op, {"min", "max"})) check_modifiers({"ftz", "NaN"});
    else check_modifiers({});
    operation = std::make_unique<v::BinaryOp>(found->second, results[0].value, arithmetic);
  } else if (In(op, {"neg", "abs", "not"})) {
    if (op == "not") check_modifiers({});
    else check_modifiers({"ftz"});
    operation = std::make_unique<v::UnaryOp>(op == "neg" ? v::UnaryOpcode::Negate :
        op == "abs" ? v::UnaryOpcode::Absolute : v::UnaryOpcode::BitNot, results[0].value,
        helper ? v::ArithmeticAttributes{} : arithmetic);
  } else if (op == "mov") {
    check_modifiers({});
    const auto source_count = operands.size();
    if (results.size() > 1 || source_count > 1) {
      throw std::invalid_argument("register tuples must be resolved by the frontend");
    } else operation = std::make_unique<v::BitcastOp>(results[0].value);
  } else if (op == "cvt") {
    // PTX rni/rzi/rmi/rpi round a floating result to an integral value;
    // ordinary conversion rounding modes are not equivalent.
    check_modifiers({"rn", "rz", "rm", "rp", "rna", "ftz", "sat", "satfinite"});
    operation = std::make_unique<v::ConvertOp>(results[0].value,
        v::ConversionAttributes{arithmetic.rounding, arithmetic.saturate, arithmetic.flush_to_zero, Has(i, "satfinite")});
  } else if (op == "fma") {
    check_modifiers({"rn", "rz", "rm", "rp", "ftz", "sat"});
    operation = std::make_unique<v::FmaOp>(results[0].value, arithmetic);
  } else if (op == "selp") {
    check_modifiers({});
    // PTX syntax is true-value,false-value,predicate; the core Select primitive
    // uses predicate,true-value,false-value.
    values = {operands[2].value, operands[0].value, operands[1].value};
    operation = std::make_unique<v::SelectOp>(results[0].value);
  } else if (op == "setp" && results.size() == 1 && operands.size() == 2 &&
      !Has(i, "ftz") && (Has(i, "eq") || Has(i, "ne") || Has(i, "lt") || Has(i, "le") || Has(i, "gt") || Has(i, "ge"))) {
    auto comparison = Has(i, "eq") ? v::ComparePredicate::Equal : Has(i, "ne") ? v::ComparePredicate::NotEqual :
        Has(i, "lt") ? v::ComparePredicate::Less : Has(i, "le") ? v::ComparePredicate::LessEqual :
        Has(i, "gt") ? v::ComparePredicate::Greater : v::ComparePredicate::GreaterEqual;
    operation = std::make_unique<v::CompareOp>(comparison, results[0].value);
  } else {
    static const std::map<std::string, v::PtxScalarOpcode> extensions{
        {"add", v::PtxScalarOpcode::AddCarry}, {"addc", v::PtxScalarOpcode::AddCarry},
        {"sub", v::PtxScalarOpcode::SubtractCarry}, {"subc", v::PtxScalarOpcode::SubtractCarry},
        {"mad", v::PtxScalarOpcode::MultiplyAdd}, {"madc", v::PtxScalarOpcode::MultiplyAddCarry},
        {"setp", v::PtxScalarOpcode::SetPredicate}, {"set", v::PtxScalarOpcode::SetValue},
        {"slct", v::PtxScalarOpcode::SelectSign}, {"cvta", v::PtxScalarOpcode::AddressSpaceConvert},
        {"brev", v::PtxScalarOpcode::BitReverse}, {"bfind", v::PtxScalarOpcode::BitFind},
        {"clz", v::PtxScalarOpcode::CountLeadingZeros}, {"popc", v::PtxScalarOpcode::PopulationCount},
        {"bfe", v::PtxScalarOpcode::BitExtract}, {"bfi", v::PtxScalarOpcode::BitInsert},
        {"prmt", v::PtxScalarOpcode::BytePermute}, {"lop3", v::PtxScalarOpcode::Logic3},
        {"shf", v::PtxScalarOpcode::FunnelShift}, {"copysign", v::PtxScalarOpcode::CopySign},
        {"cnot", v::PtxScalarOpcode::LogicalNot}, {"testp", v::PtxScalarOpcode::TestPredicate},
        {"sin", v::PtxScalarOpcode::Sin}, {"cos", v::PtxScalarOpcode::Cos},
        {"lg2", v::PtxScalarOpcode::Log2}, {"ex2", v::PtxScalarOpcode::Exp2},
        {"rcp", v::PtxScalarOpcode::Reciprocal}, {"sqrt", v::PtxScalarOpcode::Sqrt},
        {"rsqrt", v::PtxScalarOpcode::ReciprocalSqrt}, {"tanh", v::PtxScalarOpcode::Tanh}};
    const auto found = extensions.find(op);
    if (found == extensions.end()) throw std::invalid_argument("No public value operation for PTX instruction");
    auto source_type = operands.front().type.value;
    if (op == "slct") source_type = operands[2].type.value;
    operation = std::make_unique<v::PtxScalarOp>(found->second, source_type, result_types,
        v::PtxScalarAttributes{i.modifiers, In(op, {"addc", "subc", "madc"}), Has(i, "cc")});
  }
  return Add(std::move(operation), std::move(values), origin_);
}

std::optional<Expression> ValueTemplateExtractor::Operand(std::string text, const Type &type, bool defer_missing) {
  text = Trim(text);
  if (text.empty()) { Fail("invalid_operand", "Empty PTX operand"); return {}; }
  if (text.front() == '!') {
    auto source = Operand(text.substr(1), Pred());
    if (!source) return {};
    return Expression{Operation("ptx.not", "not.pred", {Pred()}, {*source}).result(), Pred()};
  }
  if (auto literal=ParseInteger(text)) return Constant(type,literal->convert_to<std::string>());
  if (In(text, {"true", "false"})) return Constant(type, text);
  // Preserve PTX hexadecimal floating encodings verbatim; no host float roundtrip.
  static const std::regex floating_bits("0([fF][0-9a-fA-F]{8}|[dD][0-9a-fA-F]{16})");
  if (std::regex_match(text, floating_bits)) return Constant(type, text);
  const auto key = Resolve(text, instruction_->scope);
  if (demand_mode && declarations_.contains(key)) return DemandOperand(text, type);
  if (auto found = registers_.find(key); found != registers_.end()) return found->second;
  if (auto found = inputs_.find(key); found != inputs_.end() &&
      (!symbols_.contains(key) || found->second.type.value == type.value)) return found->second;
  if (symbols_.contains(key)) {
    const auto name = "symbol:" + ScopeText(key.first) + ':' + text;
    const auto space = symbol_spaces_.contains(key) ? symbol_spaces_.at(key) : "param";
    v::PointerProperties pointer{name, space, {}, {}};
    if (space == "shared" || space == "local") pointer.owner_cta = cta_;
    if (space == "local") pointer.owner_thread = origin_.thread;
    auto template_pointer = pointer;
    template_pointer.owner_thread.reset();
    const auto base = Add(std::make_unique<v::ConstantOp>(type.value, text, template_pointer), {}, origin_).result();
    Expression result{graph_.asPointer(base, pointer), type};
    // Symbol identity is independent of its address-register representation.
    inputs_.insert_or_assign(key, result);
    return result;
  }
  if (!declarations_.contains(key)) {
    if (auto parameter = facts_.launch().parameters().find(text); parameter != facts_.launch().parameters().end())
      return Constant(type, parameter->second.convert_to<std::string>());
  }
  if (!declarations_.contains(key) && text.starts_with('%')) {
    if (auto special = Special(text)) return special;
  }
  for (const auto &warp : facts_.launch().warps())
    if (warp.warp_id == facts_.Warp(lane_))
      if (auto value = warp.initial_registers.find(text); value != warp.initial_registers.end() && !invalidated_.contains(key)) {
        auto result = Constant(declarations_.contains(key) ? declarations_.at(key) : type,
                               value->second.convert_to<std::string>());
        registers_.emplace(key, result);
        return result;
      }
  const auto message="No recorded definition for operand '" + text + "' in scope " + ScopeText(key.first);
  if (defer_missing && declarations_.contains(key)) {
    Expression result{{},declarations_.at(key)};
    result.missing_definition=message;
    return result;
  }
  Fail("missing_definition", message);
  return {};
}
std::optional<Expression> ValueTemplateExtractor::Address(std::string text) {
  text = Trim(text);
  if (text.size() < 2 || text.front() != '[' || text.back() != ']') {
    Fail("unsupported_address", "Expected a bracketed scalar address: " + text); return {};
  }
  auto split = SplitAddress(std::string_view(text).substr(1, text.size() - 2));
  if (!split) { Fail("unsupported_address", "Unsupported address expression: " + text); return {}; }
  auto base = Operand(split->first, U64());
  if (!base) return {};
  if (base->type.elements != 1 || base->type.spelling == "pred" ||
      base->type.spelling.starts_with('f') || base->type.spelling.starts_with("bf")) {
    Fail("unsupported_address", "Address is not an integer/bit scalar: " + text); return {};
  }
  if (base->type.value.element().kind() == v::ElementKind::Bit)
    base = Reinterpret(*base, *PtxType("u" + std::to_string(base->type.bits)));
  if (split->second != 0) {
    auto offset = Constant(base->type, split->second.convert_to<std::string>());
    base = Expression{Operation("ptx.address_add", "byte offset", {base->type}, {*base, offset}).result(), base->type};
  }
  const auto space = instruction_->semantics.memory_space.value_or("generic");
  v::PointerProperties pointer{"", space, {}, {}};
  if (space.starts_with("shared") || space == "local" || space == "tmem") pointer.owner_cta = cta_;
  if (space == "local") pointer.owner_thread = origin_.thread;
  base->value = graph_.asPointer(base->value, pointer);
  return base;
}
bool ValueTemplateExtractor::Shuffle() {
  const auto& i = *instruction_;
  const auto key = [&](std::uint32_t lane) {
    return ShuffleKey{facts_.Warp(lane_), i.id, record_->occurrence(), lane, origin_.unroll_context};
  };
  const auto binary = [&](v::BinaryOpcode op, Expression x, Expression y) {
    return Expression{Add(std::make_unique<v::BinaryOp>(op, U32().value), {x.value, y.value}, origin_).result(), U32()};
  };
  if (!shuffle_source_) {
    if (i.operands.size() != 5 || !Has(i, "sync") || !Has(i, "b32") ||
        !(Has(i, "idx") || Has(i, "bfly") || Has(i, "up") || Has(i, "down")))
      throw std::invalid_argument("Unsupported shuffle form");
    auto b = Known(std::string(module_.view(i.operands[2])));
    auto c = Known(std::string(module_.view(i.operands[3])));
    auto mask = Known(std::string(module_.view(i.operands[4])));
    if (!b || !c || !mask) throw std::invalid_argument("Shuffle lane selector/clamp/membermask is dynamic");
    const auto lane = facts_.LaneId(lane_);
    const auto members = (*mask & 0xffffffff).convert_to<std::uint32_t>();
    const auto amount = (*b & 31).convert_to<std::uint32_t>();
    const auto segment = ((*c >> 8) & 31).convert_to<std::uint32_t>();
    const auto maximum = (lane & segment) | ((*c & 31).convert_to<std::uint32_t>() & ~segment);
    int source = (lane & segment) | (amount & ~segment);
    if (Has(i, "bfly")) source = lane ^ amount;
    if (Has(i, "up")) source = int(lane) - int(amount);
    if (Has(i, "down")) source = lane + amount;
    shuffle_valid_ = Has(i, "up") ? source >= int(maximum) : source <= int(maximum);
    if (!shuffle_valid_) source = lane;
    if (source < 0 || source >= 32 || !(members & (1u << lane)) || !(members & (1u << source)))
      throw std::invalid_argument("Shuffle issuer or source is outside the membermask");
    if (demand_mode) {
      const auto valid = shuffle_valid_;
      auto result = demand_peer(source, record_->Key(), std::string(module_.view(i.operands[1])));
      if (!result) { Invalidate(); return true; }
      result = Reinterpret(*result, *PtxType("b32"));
      if (!result) { Invalidate(); return true; }
      const auto destinations = Split(module_.view(i.operands[0]), '|');
      Bind(destinations[0], *result);
      if (destinations.size() == 2) Bind(destinations[1], Constant(Pred(), valid ? "true" : "false"));
      return true;
    }
    auto value = Operand(std::string(module_.view(i.operands[1])), *PtxType("b32"));
    if (value) value = Reinterpret(*value, *PtxType("b32"));
    if (!value) { Invalidate(); return true; }
    shuffle_source_ = *value;
    shuffle_from_ = source;
    if (this->members.empty()) shuffle_values->insert_or_assign(key(lane),*value);
    else for (const auto member:this->members)
      shuffle_values->insert_or_assign(ShuffleKey{facts_.Warp(member),i.id,record_->occurrence(),facts_.LaneId(member),origin_.unroll_context},*value);
    auto linear = LinearThread();
    if (!shuffle_valid_) shuffle_selector_ = linear;
    else if (Has(i, "bfly")) shuffle_selector_ = binary(v::BinaryOpcode::BitXor, linear, Constant(U32(), std::to_string(amount)));
    else if (Has(i, "up") || Has(i, "down"))
      shuffle_selector_ = binary(Has(i, "up") ? v::BinaryOpcode::Subtract : v::BinaryOpcode::Add, linear, Constant(U32(), std::to_string(amount)));
    else {
      auto retained = binary(v::BinaryOpcode::BitAnd, linear, Constant(U32(), std::to_string((~31u) | segment)));
      shuffle_selector_ = binary(v::BinaryOpcode::BitOr, retained, Constant(U32(), std::to_string(amount & ~segment)));
    }
    ++publications;
    return false;
  }
  // Each member reads its own source lane: ShuffleSource is its PTX source and
  // validity. Members whose validity differs keep one cohort; the selector then
  // falls back to the member's own thread where the source is invalid.
  shuffle_from_=ShuffleSource(facts_,{lane_,cursor_}).first;
  const std::vector<LaneRef> self{lane_};
  const auto& readers = members.empty() ? self : members;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> sources;
  sources.reserve(readers.size());
  bool mixed = false;
  for (const auto member:readers) {
    const auto [source, valid] = ShuffleSource(facts_,{member,cursor_});
    if (!shuffle_values->contains({facts_.Warp(member),i.id,record_->occurrence(),source,origin_.unroll_context})) return false;
    sources.emplace_back(facts_.Warp(member), source);
    mixed |= valid != shuffle_valid_;
  }
  const auto base = SourceValue(i.id, record_->occurrence(), sources);
  if (!base) return false;
  const auto destinations = Split(module_.view(i.operands[0]), '|');
  auto result = shuffle_values->find(key(shuffle_from_))->second;
  Expression selector = *shuffle_selector_, valid = Constant(Pred(), shuffle_valid_ ? "true" : "false");
  if (mixed) {
    auto b = Known(std::string(module_.view(i.operands[2])));
    auto c = Known(std::string(module_.view(i.operands[3])));
    if (!b || !c) throw std::invalid_argument("Shuffle lane selector/clamp is dynamic");
    const auto amount = (*b & 31).convert_to<std::uint32_t>();
    const auto segment = ((*c >> 8) & 31).convert_to<std::uint32_t>();
    const auto clamp = (*c & 31).convert_to<std::uint32_t>() & ~segment;
    const auto compare = [&](v::ComparePredicate op, Expression x, Expression y) {
      return Expression{Add(std::make_unique<v::CompareOp>(op, Pred().value), {x.value, y.value}, origin_).result(), Pred()};
    };
    const auto constant = [&](std::uint32_t n) { return Constant(U32(), std::to_string(n)); };
    // PTX shfl.sync: maxLane = (lane & segmask) | (c & ~segmask); up is valid when
    // lane - b >= maxLane, the other modes when their source lane <= maxLane.
    const auto linear = LinearThread();
    const auto lane = binary(v::BinaryOpcode::BitAnd, linear, constant(31));
    const auto maximum = binary(v::BinaryOpcode::BitOr, binary(v::BinaryOpcode::BitAnd, lane, constant(segment)), constant(clamp));
    if (Has(i, "up")) valid = compare(v::ComparePredicate::GreaterEqual, lane, binary(v::BinaryOpcode::Add, maximum, constant(amount)));
    else {
      const auto source_lane = Has(i, "down") ? binary(v::BinaryOpcode::Add, lane, constant(amount)) :
          Has(i, "bfly") ? binary(v::BinaryOpcode::BitXor, lane, constant(amount)) :
          binary(v::BinaryOpcode::BitOr, binary(v::BinaryOpcode::BitAnd, lane, constant(segment)), constant(amount & ~segment));
      valid = compare(v::ComparePredicate::LessEqual, source_lane, maximum);
    }
    // The selector a valid lane uses (as the first phase builds it for a valid leader).
    Expression source_thread = Has(i, "bfly") ? binary(v::BinaryOpcode::BitXor, linear, constant(amount)) :
        Has(i, "up") ? binary(v::BinaryOpcode::Subtract, linear, constant(amount)) :
        Has(i, "down") ? binary(v::BinaryOpcode::Add, linear, constant(amount)) :
        binary(v::BinaryOpcode::BitOr, binary(v::BinaryOpcode::BitAnd, linear, constant((~31u) | segment)), constant(amount & ~segment));
    selector = Expression{Add(std::make_unique<v::SelectOp>(U32().value), {valid.value, source_thread.value, linear.value}, origin_).result(), U32()};
  }
  result.value = graph_.fromThread(*base, selector.value);
  Bind(destinations[0], result);
  if (destinations.size() == 2) Bind(destinations[1], valid);
  return true;
}

std::optional<v::Value> ValueTemplateExtractor::SourceValue(
    std::uint32_t instruction, std::uint64_t occurrence, const std::vector<std::pair<std::uint32_t, std::uint32_t>>& sources) {
  const auto cta = *origin_.cta;
  const auto warp_size = facts_.launch().warp_size();
  std::vector<std::uint64_t> threads;
  std::vector<v::Value> values;
  std::set<std::uint64_t> seen;
  for (const auto [warp, lane] : sources) {
    const auto thread = std::uint64_t(warp) * warp_size + lane;
    if (!seen.insert(thread).second) continue;
    const auto found = shuffle_values->find({warp, instruction, occurrence, lane, origin_.unroll_context});
    if (found == shuffle_values->end()) return std::nullopt;
    threads.push_back(thread);
    values.push_back(found->second.value);
  }
  if (threads.empty()) return std::nullopt;
  // A publication is its cohort's prototype at that cohort's leader: rebinding it
  // from there to thread t gives t's own value (unchanged outside its domain).
  const auto own = [&](const v::Value& value) { return graph_.instanceOrigin(value).thread.value_or(0); };
  const auto at = [&](const v::Value& value, std::uint64_t thread) { return graph_.bindThread(value, cta, own(value), thread); };
  // One producer template: every publication is the first one rebound, and its node
  // (no selector or view in the way) holds each source thread.
  const auto& first = values.front();
  bool single = !first.threadSelector() && !first.conversion();
  for (std::size_t k = 1; single && k < values.size(); ++k) single = at(values[k], own(first)) == first;
  if (single) {
    const auto& node = graph_.definingNode(first);
    for (const auto thread : threads)
      if (!node.memberIndex(thread)) { single = false; break; }
  }
  if (single) return first;
  for (std::size_t k = 0; k < values.size(); ++k) values[k] = at(values[k], threads[k]);
  return graph_.addThreadUnion(origin_, threads, std::move(values));
}

bool ValueTemplateExtractor::MatrixMemory() {
  const auto& i = *instruction_;
  const bool store = i.opcode == "stmatrix";
  const auto shape = DecodeMatrixShape(module_, i, store);
  const ShuffleKey collective{facts_.Warp(lane_),i.id,record_->occurrence(),0,origin_.unroll_context};
  auto& state=(*matrix_addresses)[collective];
  if (!matrix_pending_) {
    auto address = Address(shape.address);
    if (!address) { Invalidate(); return true; }
    for (const auto member:members) {
      state.addresses[facts_.LaneId(member)]=*address;
      state.arrived |= std::uint32_t{1} << facts_.LaneId(member);
    }
    matrix_pending_ = true;
    ++publications;
    return false;
  }
  if (state.arrived != UINT32_MAX) return false;
  const auto binary=[&](v::BinaryOpcode op,v::Value a,v::Value b) {
    return Add(std::make_unique<v::BinaryOp>(op,U32().value),{a,b},origin_).result();
  };
  const auto thread=LinearThread().value;
  const auto lane=binary(v::BinaryOpcode::Remainder,thread,Constant(U32(),"32").value);
  const auto warp_base=binary(v::BinaryOpcode::Subtract,thread,lane);
  const auto four=Constant(U32(),"4").value;
  const auto lane_row=binary(v::BinaryOpcode::Divide,lane,four);
  const auto lane_col=binary(v::BinaryOpcode::Remainder,lane,four);
  const auto text = Trim(module_.view(i.operands[store ? 1 : 0]));
  const auto names = Split(std::string_view(text).substr(1, text.size()-2), ',');
  for (unsigned matrix = 0; matrix < shape.count; ++matrix) {
    std::optional<Expression> data;
    if (store) { data = Operand(names[matrix], *PtxType("b32")); if (!data) return true; }
    std::vector<Expression> parts;
    for (unsigned half = 0; half < (shape.transpose ? 2u : 1u); ++half) {
      const unsigned peer = 8*matrix + (shape.transpose ? 2*(facts_.LaneId(lane_)%4)+half : facts_.LaneId(lane_)/4);
      auto address = *state.addresses[peer];
      const auto relative=shape.transpose ? binary(v::BinaryOpcode::Multiply,lane_col,Constant(U32(),"2").value) : lane_row;
      const auto selector=binary(v::BinaryOpcode::Add,warp_base,
          binary(v::BinaryOpcode::Add,relative,Constant(U32(),std::to_string(8*matrix+half)).value));
      address.value = graph_.fromThread(address.value, selector);
      auto offset_value=binary(v::BinaryOpcode::Multiply,shape.transpose ? lane_row : lane_col,
          Constant(U32(),shape.transpose ? "2" : "4").value);
      if (address.type.value != U32().value) offset_value=graph_.convert(offset_value,address.type.value,{});
      const Expression offset{offset_value,address.type};
      address.value = Operation("ptx.address_add", "byte offset", {address.type}, {address, offset}).result();
      const auto type = *PtxType(shape.transpose ? "b16" : "b32");
      if (store) {
        auto part = shape.transpose ? Part(*data, type, 16*half) : Reinterpret(*data, type);
        if (!part) return true;
        Add(std::make_unique<v::StoreOp>(), {address.value, part->value}, origin_);
      } else parts.emplace_back(Add(std::make_unique<v::LoadOp>(type.value), {address.value}, origin_).result(), type);
    }
    if (!store) Bind(names[matrix], shape.transpose ? Expression{{}, *PtxType("b32"), std::move(parts)} : parts[0]);
  }
  for (const auto member:members) state.consumed |= std::uint32_t{1} << facts_.LaneId(member);
  if (state.consumed == UINT32_MAX) matrix_addresses->erase(collective);
  matrix_pending_ = false;
  return true;
}

bool ValueTemplateExtractor::WarpMma() {
  const auto& i=*instruction_;
  const auto element=i.modifiers.size()==9 ? i.modifiers[6] : std::string();
  const bool eight=element=="e4m3" || element=="e5m2";
  const std::vector<std::string> expected{"sync","aligned",eight ? "m16n8k32" : "m16n8k16","row","col","f32",
      element,element,"f32"};
  if (!In(element,{"f16","bf16","e4m3","e5m2"}) || i.modifiers!=expected || i.operands.size()!=4)
    throw std::invalid_argument("Unsupported warp MMA shape, layout or arithmetic type");
  const auto names=[&](std::size_t operand,std::size_t count) {
    const auto text=Trim(module_.view(i.operands.at(operand)));
    if (!text.starts_with('{') || !text.ends_with('}')) throw std::invalid_argument("Warp MMA needs register tuples");
    auto result=Split(std::string_view(text).substr(1,text.size()-2),',');
    if (result.size()!=count) throw std::invalid_argument("Warp MMA register tuple width mismatch");
    return result;
  };
  const auto destinations=names(0,4);
  const ShuffleKey collective{facts_.Warp(lane_),i.id,record_->occurrence(),0,origin_.unroll_context};
  auto& state=(*warp_mma_inputs)[collective];
  if (!warp_mma_pending_) {
    std::vector<v::Value> values;
    for (unsigned operand=1;operand<4;++operand)
      for (const auto& name:names(operand,operand==2 ? 2 : 4)) {
        const auto type=*PtxType(operand==3 ? "f32" : "b32");
        auto value=Operand(name,type);
        if (value) value=Reinterpret(*value,type);
        if (!value) { Invalidate(); return true; }
        values.push_back(value->value);
      }
    for (const auto member:members) {
      state.inputs[facts_.LaneId(member)]=values;
      auto origin=origin_;
      const auto thread=SetThread(origin,facts_,member);
      for (auto& value:state.inputs[facts_.LaneId(member)])
        value=graph_.bindThread(value,cta_,*origin_.thread,thread);
      state.origins[facts_.LaneId(member)]=origin;
      state.arrived |= std::uint32_t{1} << facts_.LaneId(member);
    }
    warp_mma_pending_=true;
    ++publications;
    return false;
  }
  if (state.arrived != UINT32_MAX) return false;
  if (!state.result) {
    // One numerical matrix per collective; memory occurrences remain per thread.
    std::array<v::Value,32> selectors;
    auto constant_origin=origin_;
    constant_origin.thread.reset(); constant_origin.warp.reset(); constant_origin.lane.reset();
    constant_origin.thread_coordinates.reset();
    for (unsigned lane=0;lane<32;++lane)
      selectors[lane]=graph_.add(std::make_unique<v::ConstantOp>(U32().value,
          std::to_string(facts_.Warp(lane_)*32+lane)),{},constant_origin).result();
    std::vector<v::Value> operands;
    operands.reserve(320);
    for (unsigned role=0;role<3;++role) {
      const unsigned begin=role==0 ? 0 : role==1 ? 4 : 6;
      const unsigned count=role==1 ? 2 : 4;
      for (unsigned lane=0;lane<32;++lane) {
        const auto& values=state.inputs[lane];
        for (unsigned reg=0;reg<count;++reg)
          operands.push_back(graph_.fromThread(values[begin+reg],selectors[lane]));
      }
    }
    state.result=graph_.addCollective(std::make_unique<v::WarpMmaOp>(element),std::move(operands),
        {state.origins.begin(),state.origins.end()}).result();
    for (unsigned index=0;index<state.indices.size();++index)
      state.indices[index]=facts_.Warp(lane_)==0 ? selectors[index] :
          graph_.add(std::make_unique<v::ConstantOp>(U32().value,std::to_string(index)),{},constant_origin).result();
    for (auto& values:state.inputs) std::vector<v::Value>().swap(values);
  }
  const auto binary=[&](v::BinaryOpcode op,v::Value a,unsigned b) {
    return Add(std::make_unique<v::BinaryOp>(op,U32().value),{a,Constant(U32(),std::to_string(b)).value},origin_).result();
  };
  const auto lane=binary(v::BinaryOpcode::Remainder,LinearThread().value,32);
  for (unsigned reg=0;reg<4;++reg) {
    const auto row=binary(v::BinaryOpcode::Add,binary(v::BinaryOpcode::Divide,lane,4),(reg/2)*8);
    const auto col=binary(v::BinaryOpcode::Add,binary(v::BinaryOpcode::Multiply,binary(v::BinaryOpcode::Remainder,lane,4),2),reg%2);
    const auto result=graph_.node(state.result->nodeId()).result(0,facts_.LaneId(lane_));
    Bind(destinations[reg],{graph_.component(result,{row,col}),*PtxType("f32")});
  }
  for (const auto member:members) state.consumed |= std::uint32_t{1} << facts_.LaneId(member);
  if (state.consumed == UINT32_MAX) warp_mma_inputs->erase(collective);
  warp_mma_pending_=false;
  return true;
}

// vote.sync.{ballot.b32,any.pred,all.pred,uni.pred} d, {!}a, membermask: each
// member lane publishes its predicate as for shfl; once every lane of the
// membermask has published, each member reads lane j's predicate as an idx
// shuffle from lane j (selector (thread & ~31) | j) and combines them.
// Partition keys the members by the membermask's value (BuildPartitionSpec); the
// sources, from whichever cohorts published them, are one SourceValue.
bool ValueTemplateExtractor::Vote() {
  const auto& i = *instruction_;
  const auto mode = i.modifiers.size() == 3 ? i.modifiers[1] : std::string();
  const bool ballot = mode == "ballot";
  if (i.operands.size() != 3 || i.modifiers.size() != 3 || i.modifiers[0] != "sync" ||
      !In(mode, {"ballot", "any", "all", "uni"}) || i.modifiers[2] != (ballot ? "b32" : "pred"))
    throw std::invalid_argument("Unsupported vote form");
  const auto raw = Known(std::string(module_.view(i.operands[2])));
  if (!raw || *raw < -(Integer(1) << 31) || *raw > UINT32_MAX || !(*raw & 0xffffffff))
    throw std::invalid_argument("Vote membermask is invalid or not statically determined");
  const auto mask = (*raw & 0xffffffff).convert_to<std::uint32_t>();
  if (!vote_pending_) {
    auto predicate = Operand(std::string(module_.view(i.operands[1])), Pred());
    if (!predicate) { Invalidate(); return true; }
    for (const auto member:members) {
      const auto lane = facts_.LaneId(member);
      if (lane >= 32 || !(mask & (std::uint32_t{1} << lane)))
        throw std::invalid_argument("Vote issuer is outside the membermask");
      shuffle_values->insert_or_assign(ShuffleKey{facts_.Warp(member),i.id,record_->occurrence(),lane,origin_.unroll_context},*predicate);
    }
    vote_pending_ = true;
    ++publications;
    return false;
  }
  std::vector<std::pair<std::uint32_t, std::uint32_t>> sources;
  for (const auto member:members)
    for (unsigned lane = 0; lane < 32; ++lane)
      if (mask & (std::uint32_t{1} << lane)) {
        if (!shuffle_values->contains({facts_.Warp(member),i.id,record_->occurrence(),lane,origin_.unroll_context})) return false;
        sources.emplace_back(facts_.Warp(member), lane);
      }
  const auto base = SourceValue(i.id, record_->occurrence(), sources);
  if (!base) return false;
  const auto binary = [&](v::BinaryOpcode op, const Type& type, v::Value a, v::Value b) {
    return Add(std::make_unique<v::BinaryOp>(op, type.value), {a, b}, origin_).result();
  };
  const auto warp_base = binary(v::BinaryOpcode::BitAnd, U32(), LinearThread().value, Constant(U32(), std::to_string(~31u)).value);
  std::optional<v::Value> any, all, none;
  for (unsigned lane = 0; lane < 32; ++lane) {
    if (!(mask & (std::uint32_t{1} << lane))) continue;
    const auto selector = binary(v::BinaryOpcode::BitOr, U32(), warp_base, Constant(U32(), std::to_string(lane)).value);
    const auto predicate = graph_.fromThread(*base, selector);
    if (ballot) {
      const auto bit = Add(std::make_unique<v::SelectOp>(U32().value), {predicate,
          Constant(U32(), std::to_string(std::uint32_t{1} << lane)).value, Constant(U32(), "0").value}, origin_).result();
      any = any ? binary(v::BinaryOpcode::BitOr, U32(), *any, bit) : bit;
      continue;
    }
    any = any ? binary(v::BinaryOpcode::BitOr, Pred(), *any, predicate) : predicate;
    all = all ? binary(v::BinaryOpcode::BitAnd, Pred(), *all, predicate) : predicate;
    if (mode == "uni") {
      const auto negated = Add(std::make_unique<v::UnaryOp>(v::UnaryOpcode::BitNot, Pred().value), {predicate}, origin_).result();
      none = none ? binary(v::BinaryOpcode::BitAnd, Pred(), *none, negated) : negated;
    }
  }
  const auto destination = Trim(module_.view(i.operands[0]));
  if (ballot) {
    auto result = Reinterpret(Expression{*any, U32()}, *PtxType("b32"));
    if (!result) { Invalidate(); return true; }
    Bind(destination, *result);
  } else {
    const auto result = mode == "any" ? *any : mode == "all" ? *all : binary(v::BinaryOpcode::BitOr, Pred(), *all, *none);
    Bind(destination, Expression{result, Pred()});
  }
  return true;
}

bool ValueTemplateExtractor::RunCollective(bool (ValueTemplateExtractor::*run)(), const char* code, bool& pending) {
  try { if (!(this->*run)()) return false; }
  catch (const std::exception& error) { Fail(code, error.what()); Invalidate(); }
  pending = false; ++cursor_; return true;
}

void ValueTemplateExtractor::BeginOrigin(const RecordView& record) {
  origin_ = {};
  origin_.cta = cta_;
  const auto coordinates = frame_.SelfCoordinates();
  origin_.cta_coordinates = std::array<std::uint64_t, 3>{coordinates[0], coordinates[1], coordinates[2]};
  SetThread(origin_, facts_, lane_);
  origin_.instruction_index = record.instruction_id();
  origin_.occurrence = record.occurrence();
  origin_.source_file = source_file_;
  origin_.function = function_name_;
}

bool ValueTemplateExtractor::Step() {
  if (Done()) return false;
  if (facts_.Expanded(lane_)) record_ = facts_.Meta({lane_, cursor_});
  else {
    const bool live = (*live_)[cursor_];
    current_record_ = facts_.Keep(live ? facts_.ReadFull({lane_, cursor_}) : facts_.CopyMeta({lane_, cursor_}));
    record_ = live ? facts_.ViewFull(current_record_, {lane_, cursor_})
                   : facts_.ViewPrototypeCopy(current_record_, {lane_, cursor_});
  }
  const auto record = *record_;
  if (warp_mma_pending_) return RunCollective(&ValueTemplateExtractor::WarpMma, "unsupported_warp_mma", warp_mma_pending_);
  if (vote_pending_) return RunCollective(&ValueTemplateExtractor::Vote, "unsupported_vote", vote_pending_);
  if (matrix_pending_) return RunCollective(&ValueTemplateExtractor::MatrixMemory, "unsupported_matrix_memory", matrix_pending_);
  if (shuffle_source_) {
    if (!Shuffle()) return false;
    shuffle_source_.reset(); shuffle_selector_.reset(); ++cursor_; return true;
  }
  suboperation_ = 0;
  {
    BeginOrigin(record);
    for (const auto& [header, end] : loop_ends_) {
      if (record.instruction_id() < header || record.instruction_id() > end) continue;
      if (cursor_) {
        const auto previous = facts_.Meta({lane_, cursor_ - 1});
        if (previous.instruction_id() < header || previous.instruction_id() > end) iterations_[header] = 0;
        else if (record.instruction_id() == header && previous.instruction_id() >= header &&
                 function_.instructions[previous.instruction_id()].branch_target == header) ++iterations_[header];
      }
      origin_.unroll_context.emplace_back(header, iterations_[header]);
    }
    if (record.instruction_id() >= function_.instructions.size()) {
      Fail("invalid_execution_record", "Recorded instruction index is out of range"); ++cursor_; return true;
    }
    instruction_ = &function_.instructions[record.instruction_id()];
    origin_.source_line = instruction_->line;
    const auto& static_origin = instruction_origins_[record.instruction_id()];
    origin_.source_text = static_origin.source_text;
    origin_.scope = static_origin.scope;
    origin_.predicate = static_origin.predicate;
    origin_.executed = record.predicate_value().value_or(true);
    const auto reached_group = graph_.observeOccurrence(origin_);
    if (!(*live_)[cursor_]) { ++cursor_; return true; }
    if (record.merged_arm_join())
      Fail("two_arm_merge_values", "Execution merged an unresolved one-armed diamond (every thread ran the fall-through arm); "
           "values after its join are not exported");
    if (!record.opaque_compute_diamond_indices().empty()) {
      const auto branch = instruction_;
      const auto origin = origin_;
      bool lifted = false;
      lifting_ = true;
      try { lifted = LiftRegion(); }
      catch (const std::invalid_argument& error) { Fail("invalid_value_operation", error.what()); }
      lifting_ = false; instruction_ = branch; origin_ = origin;
      if (lifted) { ++cursor_; return true; }
      for (const auto &name : record.opaque_compute_register_writes()) {
        const auto key = Resolve(name, instruction_->scope);
        registers_.erase(key);
        invalidated_.insert(key);
      }
      for (auto index : record.opaque_compute_diamond_indices())
        if (index < function_.instructions.size()) {
          const auto &skipped = function_.instructions[index];
          for (const auto &name : skipped.destinations) {
            const auto key = Resolve(name, skipped.scope);
            registers_.erase(key);
            invalidated_.insert(key);
          }
        }
      carry_.reset();
      ++cursor_; return true;
    }
    if (record.collapsed_wait_loop() || record.canonical_wait_probe()) {
      if (!record.normalized_wait_result()) {
        Fail("missing_wait_summary", "Normalized polling loop lacks its return-value summary");
        Invalidate();
      } else {
        for (const auto& destination : instruction_->destinations)
          Bind(destination, Constant(Pred(), *record.normalized_wait_result() ? "true" : "false"));
      }
      ++cursor_; return true;
    }
    if (record.predicate_value() == false) { ++cursor_; return true; }
    if (instruction_->predicate && !record.predicate_value()) {
      Fail("unknown_execution_predicate", "The active instruction path is not fixed"); Invalidate(); ++cursor_; return true;
    }
    if (branch_capture_ && record.branch_scope_decision()) {
      try {
        if (instruction_->control != "direct_branch" || !instruction_->predicate)
          throw std::invalid_argument("branch capture does not name a conditional direct branch");
        const auto name = *instruction_->predicate;
        const auto value = registers_.find(Resolve(name, instruction_->scope));
        const auto producer = facts_.Origin(record, name);
        if (value == registers_.end() || value->second.missing_definition ||
            !value->second.components.empty() || !producer)
          throw std::invalid_argument("pre-decision predicate definition is unavailable");
        branch_capture_->observe({value->second.value, reached_group, origin_, name,
            *producer, *record.branch_scope_decision(), instruction_->predicate_negated});
      } catch (const std::exception& error) {
        // Optional proof evidence must not change ordinary value semantics.
        // An incomplete capture is never published or used as an assumption.
        branch_capture_->unavailable(error.what());
      }
      ++cursor_; return true;
    }
    if (instruction_->opcode=="mma")
      return RunCollective(&ValueTemplateExtractor::WarpMma, "unsupported_warp_mma", warp_mma_pending_);
    if (In(instruction_->opcode, {"ldmatrix", "stmatrix"}))
      return RunCollective(&ValueTemplateExtractor::MatrixMemory, "unsupported_matrix_memory", matrix_pending_);
    if (instruction_->opcode == "shfl") {
      try {
        if (!Shuffle()) return false;
      } catch (const std::exception& error) { Fail("unsupported_shuffle", error.what()); Invalidate(); }
      shuffle_source_.reset(); shuffle_selector_.reset(); ++cursor_; return true;
    }
    if (instruction_->opcode == "vote" && shuffle_values && !demand_mode)
      return RunCollective(&ValueTemplateExtractor::Vote, "unsupported_vote", vote_pending_);
    try { InstructionValue(); }
    catch (const std::invalid_argument &error) {
      Fail("invalid_value_operation", error.what()); Invalidate();
    }
  }
  ++cursor_; return true;
}
// Execution skipped the region from a forward branch to its join on an unresolved predicate;
// every edge inside is forward. One program-order pass gives each instruction its path: the
// fall-through enters on the negated branch guard, a conditional edge adds its guard, a join
// takes the disjunction and a guarded instruction adds its own guard. Each instruction is
// lowered as usual on the values its sources have on its path. A register the region assigns
// has, on a path, its last assignment's value where that assignment covers the path, else
// select(assignment path, value, the earlier value on the rest of the path). A register
// without a value before the region is read only where its assignments cover the path, and
// keeps a value after the join only if they cover every path.
// Size limits of a lifted region: guard terms of one instruction's path, and path splits of
// all its register values.
constexpr std::size_t kLiftedPathTerms = 64;
constexpr std::size_t kLiftedValueSplits = 4096;
bool ValueTemplateExtractor::LiftRegion() {
  const auto& branch = *instruction_;
  const auto& indices = record_->opaque_compute_diamond_indices();
  const auto join = indices.back() + 1;
  const auto refuse = [&](const char* code, const std::string& message) { Fail(code, message); return false; };
  if (!branch.predicate || !branch.branch_target || indices.front() != branch.id + 1 ||
      indices.size() != join - indices.front() || *branch.branch_target > join || join > function_.instructions.size())
    return refuse("lifted_region_shape", "Opaque compute region is not the contiguous forward region of its branch");
  struct Assignment { GuardPath path; Expression value; };
  struct Lifted { std::optional<Expression> prior; std::vector<Assignment> assignments; };
  std::map<Key, Lifted> lifted;
  std::map<std::uint32_t, Expression> guards;
  std::map<std::tuple<Key, std::size_t, GuardPath>, std::optional<Expression>> values;
  std::size_t splits = 0;
  bool limited = false;
  const auto choose = [&](const Expression& condition, const Expression& value, const Expression& other) -> std::optional<Expression> {
    const bool scalar = value.components.empty() && other.components.empty() && !value.missing_definition &&
        !other.missing_definition;
    if (scalar && value.value == other.value) return value;
    auto updated = scalar ? std::optional<Expression>(value) : Bits(value);
    auto previous = scalar ? std::optional<Expression>(other) : Bits(other);
    // The alternatives meet in the typed one of the two (b32 and f32 meet in f32).
    if (updated && previous && previous->type.value != updated->type.value) {
      if (updated->type.value.element().kind() == v::ElementKind::Bit) updated = Reinterpret(*updated, previous->type);
      else previous = Reinterpret(*previous, updated->type);
    }
    if (!updated || !previous) return {};
    auto selected = Add(std::make_unique<v::SelectOp>(updated->type.value),
        {condition.value, updated->value, previous->value}, origin_).result();
    if (const auto& pointer = updated->value.pointer(); pointer && previous->value.pointer() && *pointer == *previous->value.pointer())
      selected = graph_.asPointer(selected, *pointer);
    return Expression{selected, updated->type};
  };
  // `key` on a nonempty `path` after its first `count` assignments: the last assignment that
  // covers the path, the earlier value where it is disjoint, else a select on the first guard
  // the path leaves open (the region's own branch structure, gated-SSA form).
  const auto value = [&](auto&& self, const Key& key, const Lifted& entry, std::size_t count,
                         const GuardPath& path) -> std::optional<Expression> {
    if (!count) return entry.prior;
    const auto& last = entry.assignments[count - 1];
    if (Implies(path, last.path)) return last.value;
    if (Disjoint(path, last.path)) return self(self, key, entry, count - 1, path);
    const auto cached = std::tuple{key, count, path};
    if (const auto found = values.find(cached); found != values.end()) return found->second;
    const auto open = Open(path, last.path);
    if (!open || ++splits > kLiftedValueSplits) { limited = true; return {}; }
    const auto taken = self(self, key, entry, count, Conjoin(path, {*open, true}));
    const auto other = taken ? self(self, key, entry, count, Conjoin(path, {*open, false})) : std::nullopt;
    const auto result = other ? choose(guards.at(*open), *taken, *other) : std::nullopt;
    return values.emplace(cached, result).first->second;
  };
  // A nonempty path's value of `key` (lifted), or a refusal.
  const auto read = [&](const Key& key, const GuardPath& path, const std::string& name) -> std::optional<Expression> {
    const auto diagnostics = diagnostics_.size();
    auto result = value(value, key, lifted.at(key), lifted.at(key).assignments.size(), path);
    if (!result && diagnostics_.size() == diagnostics)
      refuse(limited ? "lifted_region_path_limit" : "lifted_region_unassigned_register",
             limited ? "A path or value of the lifted region exceeds its size limit"
                     : "Register '" + name + "' may be unassigned on a path of the lifted region that reads it");
    return result;
  };
  // The guard of `instruction` (instruction_) on `path`: the value of literal (instruction, true).
  const auto guard = [&](const Instruction& instruction, const GuardPath& path) {
    const auto key = Resolve(*instruction.predicate, instruction.scope);
    auto truth = lifted.contains(key) ? read(key, path, *instruction.predicate) : Operand(*instruction.predicate, Pred());
    if (truth) truth = Reinterpret(*truth, Pred());
    if (!truth) return false;
    guards.insert_or_assign(instruction.id, *truth);
    return true;
  };
  std::map<std::uint32_t, GuardPath> reach;
  const auto flow = [&](std::uint32_t target, const GuardPath& path) {
    if (path.empty() || target >= join) return true;
    auto& entry = reach[target];
    entry = Disjoin(std::move(entry), path);
    return entry.size() <= kLiftedPathTerms ||
           refuse("lifted_region_path_limit", "A path of the lifted region exceeds " + std::to_string(kLiftedPathTerms) +
                  " guard terms");
  };
  if (!guard(branch, GuardPath{GuardTerm{}}) ||
      !flow(*branch.branch_target, GuardPath{GuardTerm{GuardLiteral{branch.id, !branch.predicate_negated}}}) ||
      !flow(branch.id + 1, GuardPath{GuardTerm{GuardLiteral{branch.id, branch.predicate_negated}}}))
    return false;
  for (auto index = branch.id + 1; index < join; ++index) {
    const auto at = reach.find(index);
    if (at == reach.end()) continue;
    const auto path = std::move(at->second);
    reach.erase(at);
    const auto& instruction = function_.instructions[index];
    instruction_ = &instruction;
    origin_.source_line = instruction.line;
    origin_.source_text = instruction_origins_[index].source_text;
    origin_.predicate = instruction_origins_[index].predicate;
    if (instruction.control == "direct_branch") {
      const auto target = instruction.branch_target;
      if (!target || *target <= index || *target > join)
        return refuse("lifted_region_control", "A branch inside the lifted region is not forward to at most its join");
      if (!instruction.predicate) { if (!flow(*target, path)) return false; continue; }
      if (!guard(instruction, path) || !flow(*target, Conjoin(path, {index, !instruction.predicate_negated})) ||
          !flow(index + 1, Conjoin(path, {index, instruction.predicate_negated})))
        return false;
      continue;
    }
    if (instruction.control != "none")
      return refuse("lifted_region_control", "Control instruction '" + instruction.opcode + "' inside the lifted region");
    const auto& semantics = instruction.semantics;
    const bool parameter = semantics.category == "memory" && In(instruction.opcode, {"ld", "ldu"}) &&
        semantics.memory_space == "param";
    if (semantics.category == "memory" && !parameter)
      return refuse("lifted_region_memory_access", "'" + instruction.opcode + "' inside the lifted region has no recorded "
                    "memory access; only parameter reads are lowered");
    // Warp-level operations read other lanes or the active mask, which the region's paths do not determine.
    if (!parameter && (semantics.category != "compute" || semantics.memory_effect || WarpLevelOpcode(instruction.opcode)))
      return refuse("lifted_region_unsupported_instruction", "'" + instruction.opcode + "' inside the lifted region is not "
                    "per-thread register computation");
    if (Has(instruction, "cc") || In(instruction.opcode, {"addc", "subc", "madc"}))
      return refuse("lifted_region_carry", "Carry-flag arithmetic inside the lifted region");
    auto executed = path;
    if (instruction.predicate) {
      if (!guard(instruction, path)) return false;
      executed = Conjoin(path, {index, !instruction.predicate_negated});
    }
    if (!flow(index + 1, path)) return false;
    if (executed.empty()) continue;
    // Region-assigned registers are absent from registers_ (and invalidated) between
    // instructions; the sources take their value on this path while the instruction lowers.
    std::vector<Key> present;
    for (std::size_t operand = instruction.destinations.empty() ? 0 : 1; operand < instruction.operands.size(); ++operand)
      for (const auto& name : Registers(module_.view(instruction.operands[operand]))) {
        const auto key = Resolve(name, instruction.scope);
        if (!lifted.contains(key) || registers_.contains(key)) continue;
        auto source = read(key, executed, name);
        if (!source) return false;
        registers_.insert_or_assign(key, *source);
        present.push_back(key);
      }
    std::vector<Key> written;
    for (const auto& name : instruction.destinations)
      if (name != "_") {
        const auto key = Resolve(name, instruction.scope);
        if (!lifted.contains(key)) {
          const auto prior = registers_.find(key);
          lifted[key].prior = prior == registers_.end() ? std::nullopt : std::optional<Expression>(prior->second);
        }
        written.push_back(key);
      }
    const auto diagnostics = diagnostics_.size();
    InstructionValue();
    std::vector<std::optional<Expression>> results;
    for (const auto& key : written) {
      const auto found = registers_.find(key);
      results.push_back(found == registers_.end() ? std::nullopt : std::optional<Expression>(found->second));
    }
    for (const auto* keys : {&present, &written})
      for (const auto& key : *keys) { registers_.erase(key); invalidated_.insert(key); }
    if (diagnostics_.size() != diagnostics) return false;
    for (std::size_t n = 0; n < written.size(); ++n) {
      if (!results[n]) return refuse("lifted_region_value", "Lowering defined no value for '" + written[n].second + "'");
      lifted.at(written[n]).assignments.push_back({executed, *results[n]});
    }
  }
  for (const auto& [key, entry] : lifted) {
    const auto result = value(value, key, entry, entry.assignments.size(), GuardPath{GuardTerm{}});
    if (limited) return refuse("lifted_region_path_limit", "A path or value of the lifted region exceeds its size limit");
    if (result) { registers_.insert_or_assign(key, *result); invalidated_.erase(key); }
  }
  return true;
}
void ValueTemplateExtractor::InstructionValue() {
  const auto &i = *instruction_;
  if (i.semantics.atomic_add) {
    Fail("atomic_add_value_semantics_unsupported", "Atomic add numerical semantics are unsupported");
    Invalidate(); return;
  }
  if (record_->normalized_clc_result().has_value()) {
    for (const auto& destination : i.destinations)
      Bind(destination, Constant(Pred(), *record_->normalized_clc_result() ? "true" : "false"));
    return;
  }
  if (i.opcode == "setmaxnreg") return;
  if (i.opcode == "clusterlaunchcontrol" && i.semantics.operation == "try_cancel_complete_tx") {
    if (facts_.launch().ClusterLaunchCancellation()) {
      Fail("cluster_launch_control_outcome", "CLC numerical response needs the no-cancellation scenario"); return;
    }
    if (i.operands.size() != 2) throw std::invalid_argument("CLC response requires destination and barrier");
    auto address = Address(Trim(module_.view(i.operands[0])));
    if (!address) return;
    const auto response = Add(std::make_unique<v::FailedClcResponseOp>(), {}, origin_).result();
    const auto shape = frame_.cluster_dim();
    const auto count = Has(i,"multicast::cluster::all") ? shape[0]*shape[1]*shape[2] : 1u;
    for (unsigned rank = 0; rank < count; ++rank) {
      auto destination = address->value;
      if (Has(i,"multicast::cluster::all")) {
        const auto target = Constant(U32(), std::to_string(rank));
        destination = Add(std::make_unique<v::PtxScalarOp>(v::PtxScalarOpcode::ClusterSharedAddress,
            address->type.value, std::vector{address->type.value}), {destination, target.value}, origin_).result();
      }
      Add(std::make_unique<v::StoreOp>(), {destination, response}, origin_);
    }
    return;
  }
  if (i.opcode == "mapa") {
    if (i.operands.size() != 3 || !Has(i, "shared::cluster") || !(Has(i, "u32") || Has(i, "u64"))) {
      Fail("unsupported_mapa", "Expected mapa.shared::cluster.u32/u64"); Invalidate(); return;
    }
    const auto type = Has(i, "u64") ? U64() : U32();
    auto base = Operand(std::string(module_.view(i.operands[1])), type);
    auto rank = Operand(std::string(module_.view(i.operands[2])), U32());
    if (base) base = Reinterpret(*base, type);
    if (rank) rank = Reinterpret(*rank, U32());
    if (!base || !rank) { Invalidate(); return; }
    const auto value = Add(std::make_unique<v::PtxScalarOp>(v::PtxScalarOpcode::ClusterSharedAddress,
        type.value, std::vector{type.value}), {base->value, rank->value}, origin_).result();
    Bind(std::string(module_.view(i.operands[0])), {value, type});
    return;
  }
  if (i.control != "none") {
    if (i.control == "call") {
      Fail("unsupported_call", "Opaque calls do not provide value definitions"); Invalidate();
    }
    return;
  }
  if (In(i.opcode, {"prefetch", "prefetchu"})) {
    std::string modifiers;
    for (const auto& modifier : i.modifiers) {
      if (!modifiers.empty()) modifiers += '.';
      modifiers += modifier;
    }
    const bool valid = i.opcode == "prefetchu" ? modifiers == "L1" :
        In(modifiers, {"L1", "L2", "global.L1", "global.L2", "local.L1", "local.L2",
                       "global.L2::evict_last", "global.L2::evict_normal",
                       "L1::32B.valid_addr", "global.L1::32B.valid_addr",
                       "tensormap", "const.tensormap", "param.tensormap"});
    if (!valid || i.operands.size() != 1) {
      Fail("unsupported_prefetch", "Unsupported cache-prefetch form"); return;
    }
    const auto address = Trim(module_.view(i.operands[0]));
    if (!address.starts_with('[') || !address.ends_with(']')) {
      Fail("unsupported_prefetch", "Cache prefetch requires a bracketed address"); return;
    }
    return;
  }
  if (i.opcode == "tcgen05") { Tcgen(); return; }
  if (i.opcode == "cp" && Has(i,"reduce")) {
    Fail("atomic_reduction_value_not_modeled", "Atomic bulk reduction needs a read-modify-write value effect and reduction-aware memory elimination");
    Invalidate(); return;
  }
  if (i.opcode == "cp" && Has(i, "async") && Has(i, "bulk") && Has(i, "tensor") && !Has(i, "prefetch")) { Tma(); return; }
  if (i.semantics.category == "memory" && In(i.opcode, {"ld", "ldu", "st"}) && !i.semantics.asynchronous) {
    Memory(); return;
  }
  if (i.opcode == "cp" && i.semantics.operation == "copy") { AsyncCopy(); return; }
  if (i.opcode == "cp" && i.semantics.operation == "bulk_complete_tx") { BulkCopy(); return; }
  if (i.opcode == "tensormap") { TensorMap(); return; }
  if (i.semantics.category == "compute") { Compute(); return; }
  if (i.semantics.memory_effect || !i.destinations.empty() || !i.semantics.modeled) {
    Fail("unsupported_operation", "No value-IR model for '" + i.opcode + "' (including its memory effects/results)");
    Invalidate();
  }
  // Synchronization-only events have no scalar value semantics. Their identity
  // remains in the execution records, and memory-node origins support a future join.
}

void ValueTemplateExtractor::Compute() {
  const auto &i = *instruction_;
  if (i.opcode == "elect") {
    if (i.modifiers != std::vector<std::string>{"sync"} || i.operands.size() != 2) {
      Fail("unsupported_election", "elect.sync requires lane/predicate destinations and a membermask"); Invalidate(); return;
    }
    const auto destinations = Split(module_.view(i.operands[0]), '|');
    if (destinations.size() != 2 || destinations[0].empty() || destinations[1].empty() || destinations[1] == "_") {
      Fail("unsupported_election", "elect.sync requires d|p destinations"); Invalidate(); return;
    }
    const auto raw = Known(std::string(module_.view(i.operands[1])));
    if (!raw || *raw < -(Integer(1) << 31) || *raw > UINT32_MAX) {
      Fail("unsupported_election", "Election membermask is invalid or not statically determined"); Invalidate(); return;
    }
    const auto mask = (*raw & 0xffffffff).convert_to<std::uint32_t>();
    if (facts_.LaneId(lane_) >= 32 || !mask || !(mask & (1u << facts_.LaneId(lane_)))) {
      Fail("unsupported_election", "Election issuer is outside membermask"); Invalidate(); return;
    }
    auto chosen = facts_.launch().elect_lane();
    if (!chosen) {
      if (std::popcount(mask) != 1) {
        Fail("unsupported_election", "Election policy is not configured"); Invalidate(); return;
      }
      chosen = std::countr_zero(mask);
    } else if (*chosen >= 32 || !(mask & (1u << *chosen))) {
      chosen = std::countr_zero(mask);
    }
    if (destinations[0] != "_") Bind(destinations[0], Constant(U32(), std::to_string(*chosen)));
    Bind(destinations[1], Constant(Pred(), facts_.LaneId(lane_) == *chosen ? "true" : "false"));
    return;
  }
  if (i.opcode == "isspacep") {
    // The answer follows from the address's provenance, as the executor decided it.
    const auto *address = i.operands.size() == 2 && i.modifiers.size() == 1
        ? facts_.Value(*record_, Trim(module_.view(i.operands[1]))) : nullptr;
    const auto inside = address ? GenericAddressInSpace(i.modifiers[0], *address, facts_.launch().resources()) : std::nullopt;
    if (!inside) { Fail("unsupported_isspacep", "isspacep needs an address of known state space"); Invalidate(); return; }
    Bind(Trim(module_.view(i.operands[0])), Constant(Pred(), *inside ? "true" : "false"));
    return;
  }
  static const std::map<std::string, std::pair<unsigned, unsigned>> arities{
      {"mov", {1, 1}}, {"cvt", {1, 2}}, {"cvta", {1, 1}},
      {"abs", {1, 1}}, {"neg", {1, 1}}, {"not", {1, 1}}, {"cnot", {1, 1}},
      {"clz", {1, 1}}, {"popc", {1, 1}}, {"brev", {1, 1}}, {"bfind", {1, 1}},
      {"sin", {1, 1}}, {"cos", {1, 1}}, {"lg2", {1, 1}}, {"ex2", {1, 1}},
      {"rcp", {1, 1}}, {"sqrt", {1, 1}}, {"rsqrt", {1, 1}}, {"tanh", {1, 1}}, {"testp", {1, 1}},
      {"add", {2, 2}}, {"addc", {2, 2}}, {"sub", {2, 2}}, {"subc", {2, 2}},
      {"mul", {2, 2}}, {"div", {2, 2}}, {"rem", {2, 2}}, {"min", {2, 3}}, {"max", {2, 3}},
      {"and", {2, 2}}, {"or", {2, 2}}, {"xor", {2, 2}}, {"shl", {2, 2}}, {"shr", {2, 2}},
      {"copysign", {2, 2}}, {"setp", {2, 3}}, {"set", {2, 3}},
      {"mad", {3, 3}}, {"madc", {3, 3}}, {"fma", {3, 3}}, {"selp", {3, 3}}, {"slct", {3, 3}},
      {"bfe", {3, 3}}, {"prmt", {3, 3}}, {"bfi", {4, 4}}, {"lop3", {4, 5}}, {"shf", {3, 3}}};
  auto arity = arities.find(i.opcode);
  if (arity == arities.end()) {
    Fail("unsupported_operation", "No exact per-thread value model for compute operation '" + i.opcode + "'"); Invalidate(); return;
  }
  if (i.operands.empty() || i.operands.size() - 1 < arity->second.first || i.operands.size() - 1 > arity->second.second) {
    Fail("unsupported_operands", "Unsupported operand count for '" + i.opcode + "'"); Invalidate(); return;
  }
  std::vector<Type> types;
  static const std::set<std::string> modifiers{
      "rn", "rz", "rm", "rp", "rna", "rni", "rzi", "rmi", "rpi", "approx", "full", "ftz", "sat", "satfinite", "relu",
      "lo", "hi", "wide", "cc", "to", "global", "shared", "local", "const", "param", "shared::cta", "shared::cluster",
      "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hs", "equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan",
      "and", "or", "xor", "shiftamt", "f4e", "b4e", "rc8", "ecl", "ecr", "rc16", "l", "r", "wrap", "clamp",
      "finite", "infinite", "number", "notanumber", "normal", "subnormal", "NaN", "abs", "oob"};
  for (const auto &modifier : i.modifiers) {
    if (auto type = PtxType(modifier)) types.push_back(*type);
    else if (!modifiers.contains(modifier)) {
      Fail("unsupported_modifier", "Unsupported modifier '" + modifier + "' on '" + i.opcode + "'"); Invalidate(); return;
    }
  }
  if (types.empty()) { Fail("missing_type", "No supported PTX type for '" + i.opcode + "'"); Invalidate(); return; }
  auto result_type = types.front();
  auto source_type = (In(i.opcode, {"cvt", "set", "slct"}) && types.size() > 1) ? types.back() : types.front();
  if (In(i.opcode, {"setp", "testp"})) result_type = Pred();
  if (In(i.opcode, {"clz", "popc", "bfind"})) result_type = U32();
  if (Has(i, "wide")) {
    auto wide = PtxType(std::string(1, source_type.spelling.front()) + std::to_string(source_type.bits * 2));
    if (!wide) { Fail("unsupported_type", "Unsupported wide result type"); Invalidate(); return; }
    result_type = *wide;
  }
  // Scalar mixed add/sub widen only a; c and d already have f32 type.
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#mixed-precision-floating-point-instructions-add
  const bool mixed_add_sub = In(i.opcode, {"add", "sub"}) && types.size() == 2 &&
      types[0].spelling == "f32" && In(types[1].spelling, {"f16", "bf16"});
  auto destination_text = Trim(module_.view(i.operands.front()));
  const bool destination_vector = destination_text.starts_with('{') && destination_text.ends_with('}');
  if (destination_vector) destination_text = destination_text.substr(1, destination_text.size() - 2);
  auto destinations = Split(destination_text, destination_vector ? ',' : '|');
  if (i.opcode == "selp") {
    if (auto predicate = Known(std::string(module_.view(i.operands[3])), cohort_template)) {
      auto value = Operand(std::string(module_.view(i.operands[*predicate != 0 ? 1 : 2])), result_type);
      if (value) value = Reinterpret(*value, result_type);
      if (value) Bind(destinations.front(), *value); else Invalidate();
      return;
    }
  }
  std::vector<Type> result_types;
  for (const auto &destination : destinations) {
    auto type = result_type;
    if (destination_vector) {
      auto declared = declarations_.find(Resolve(destination, i.scope));
      if (i.opcode == "mov" && destination == "_" &&
          (destinations.size() == 2 || destinations.size() == 4) &&
          result_type.bits % destinations.size() == 0) {
        const auto width = result_type.bits / destinations.size();
        result_types.push_back(*PtxType("b" + std::to_string(width)));
        continue;
      }
      if (i.opcode != "mov" || declared == declarations_.end()) {
        Fail("unsupported_vector_compute", "Only explicit mov register packing/unpacking is supported"); Invalidate(); return;
      }
      type = declared->second;
    } else if (destinations.size() > 1 && i.opcode != "setp") {
      Fail("unsupported_multi_result", "Unsupported multi-destination compute operation"); Invalidate(); return;
    }
    result_types.push_back(type);
  }
  std::vector<Expression> operands;
  std::vector<std::string> labels;
  bool valid = true;
  for (std::size_t n = 1; n < i.operands.size(); ++n) {
    auto text = Trim(module_.view(i.operands[n]));
    auto type = source_type;
    if (i.opcode == "fma" && types.size() == 2 && n <= 2) type = types[1];
    if (mixed_add_sub && n == 1) type = types[1];
    if ((i.opcode == "selp" && n == 3) || (In(i.opcode, {"setp", "set"}) && n == 3)) type = Pred();
    if ((In(i.opcode, {"shl", "shr"}) && n == 2) || (i.opcode == "bfe" && n >= 2) ||
        (i.opcode == "bfi" && n >= 3) || (i.opcode == "lop3" && n == 4)) type = U32();
    if (i.opcode == "mad" && Has(i, "wide") && n == 3) type = result_type;
    if (i.opcode == "slct" && n < 3) type = result_type;
    if (text.starts_with('{') && text.ends_with('}')) {
      if (i.opcode != "mov") { Fail("unsupported_vector_compute", "Non-mov register packs require explicit layout support"); valid = false; continue; }
      const auto parts = Split(std::string_view(text).substr(1, text.size() - 2), ',');
      if ((parts.size() != 2 && parts.size() != 4) || type.bits % parts.size() != 0) {
        Fail("invalid_register_pack", "mov pack needs two or four equal-width components"); Invalidate(); return;
      }
      const auto component_type = PtxType("b" + std::to_string(type.bits / parts.size()));
      if (!component_type) { Fail("invalid_register_pack", "Unsupported mov component width"); Invalidate(); return; }
      for (const auto &part : parts) {
        const auto declared = declarations_.find(Resolve(part, i.scope));
        auto expression = Operand(part, declared == declarations_.end() ? *component_type : declared->second, true);
        if (expression) { operands.push_back(*expression); labels.push_back(part); } else valid = false;
      }
    } else {
      auto expression = Operand(text, type);
      if (expression && i.opcode == "cvt" && expression->type.value.isScalar() &&
          type.value.isScalar() && expression->type.value.element().isBitLike() &&
          type.value.element().isBitLike() && type.spelling != "pred" &&
          expression->type.value.bitSize() > type.value.bitSize()) {
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#operand-size-exceeding-instruction-type-size
        expression = Part(*expression, type, 0);
      } else if (expression && i.opcode!="mov") expression = Reinterpret(*expression, type);
      if (expression && ((i.opcode == "fma" && types.size() == 2 && n <= 2) ||
                         (mixed_add_sub && n == 1)) && type.value != result_type.value) {
        expression = Expression{Add(std::make_unique<v::ConvertOp>(result_type.value), {expression->value}, origin_).result(), result_type};
      }
      if (expression) { operands.push_back(*expression); labels.push_back(text); } else valid = false;
    }
  }
  if (In(i.opcode, {"addc", "subc", "madc"})) {
    if (!carry_) { Fail("missing_carry_definition", "Carry-consuming operation has no recorded carry definition"); valid = false; }
    else { operands.push_back(*carry_); labels.push_back("CC.CF"); }
  }
  if (!valid) { Invalidate(); return; }
  if (i.opcode == "cvt" && result_type.spelling == "ue8m0x2") {
    // The reference conversion (cuda_fp8.hpp, __nv_cvt_float_to_e8m0 and __nv_cvt_bfloat16raw_to_e8m0):
    // .rz keeps the biased exponent field and ignores the sign, so zero and subnormals give 0 (2^-127)
    // and NaN or infinity give 0xFF (NaN); .satfinite turns an infinity into 0xFE. Source a fills the
    // upper byte. .rp (round the magnitude up to a power of two) has no model here.
    const bool f32 = source_type.spelling == "f32" && operands.size() == 2;
    if (!Has(i, "rz") || Has(i, "relu") || !(f32 || (source_type.spelling == "bf16x2" && operands.size() == 1))) {
      Fail("unsupported_ue8m0_conversion", "Only cvt.rz{.satfinite}.ue8m0x2 from .f32 or .bf16x2 has an exact value model");
      Invalidate(); return;
    }
    std::vector<Expression> sources;
    if (f32) sources = {operands[1], operands[0]};
    else for (std::uint32_t n = 0; n < 2; ++n) {
      auto half = Part(operands[0], *PtxType("bf16"), 16 * n);
      if (!half) { Invalidate(); return; }
      sources.push_back(*half);
    }
    const Type element{v::ValueType::scalar(result_type.value.element()), "ue8m0", 8};
    Expression packed{{}, result_type};
    for (const auto &source : sources) {
      auto bits = Bits(source);
      if (!bits) { Invalidate(); return; }
      const auto type = bits->type;
      const auto binary = [&](v::BinaryOpcode opcode, const Expression &a, const Expression &b) {
        return Expression{Add(std::make_unique<v::BinaryOp>(opcode, type.value), {a.value, b.value}, origin_).result(), type};
      };
      const bool wide = type.bits == 32;
      auto exponent = binary(v::BinaryOpcode::BitAnd, binary(v::BinaryOpcode::ShiftRight, *bits, Constant(U32(), wide ? "23" : "7")),
                             Constant(type, "255"));
      if (Has(i, "satfinite")) {
        const auto infinity = Expression{Add(std::make_unique<v::CompareOp>(v::ComparePredicate::Equal, Pred().value),
            {binary(v::BinaryOpcode::BitAnd, *bits, Constant(type, wide ? "2147483647" : "32767")).value,
             Constant(type, wide ? "2139095040" : "32640").value}, origin_).result(), Pred()};
        exponent = Expression{Add(std::make_unique<v::SelectOp>(type.value),
            {infinity.value, Constant(type, "254").value, exponent.value}, origin_).result(), type};
      }
      auto byte = Part(exponent, element, 0);
      if (!byte) { Invalidate(); return; }
      packed.components.push_back(*byte);
    }
    Bind(destinations[0], packed); return;
  }
  if (i.opcode == "mov" && (destination_vector ||
      Trim(module_.view(i.operands[1])).starts_with('{'))) {
    std::uint64_t result_bits = 0, source_bits = 0;
    for (const auto &item : result_types) result_bits += item.value.bitSize();
    for (std::size_t n = 0; n < operands.size(); ++n)
      source_bits += operands[n].type.value.bitSize();
    if (result_bits != source_bits || result_bits != types.front().value.bitSize()) {
      Fail("invalid_register_pack", "mov vector components do not exactly cover the PTX bit width"); Invalidate(); return;
    }
  }
  if (i.opcode == "mov") {
    Expression source = operands.front();
    if (operands.size() > 1) source = {{}, types.front(), operands};
    if (destinations.size() == 1) {
      if (source.type.value == result_types[0].value) Bind(destinations[0], source);
      else if (auto result = Reinterpret(source, result_types[0])) Bind(destinations[0], *result);
    } else {
      std::uint32_t bit = 0;
      for (std::size_t n = 0; n < destinations.size(); ++n) {
        if (destinations[n]!="_") {
          auto result = Part(source, result_types[n], bit);
          if (result) Bind(destinations[n], *result); else { Invalidate(); return; }
        }
        bit += static_cast<std::uint32_t>(result_types[n].value.bitSize());
      }
    }
    return;
  }
  if (result_type.elements > 1) {
    if (!In(i.opcode, {"add", "sub", "mul", "fma", "cvt", "min", "max"})) {
      Fail("unsupported_packed_operation", "Packed operation needs a defined scalar component mapping"); Invalidate(); return;
    }
    Expression packed{{}, result_type};
    const auto scalar_result = Type{v::ValueType::scalar(result_type.value.element()), result_type.spelling.substr(0, result_type.spelling.find('x')), result_type.bits};
    for (std::uint32_t n = 0; n < result_type.elements; ++n) {
      std::vector<Expression> components;
      if (i.opcode == "cvt" && source_type.elements == 1 && operands.size() == result_type.elements) {
        // Packed cvt.f16x2/e4m3x2/e5m2x2/e2m1x2.f32 places the
        // second source in the low component (bits [3:0] for e2m1x2).
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
        components.push_back(operands[result_type.elements - 1 - n]);
      } else for (auto operand : operands) {
        const auto scalar_source = Type{v::ValueType::scalar(operand.type.value.element()), operand.type.spelling.substr(0, operand.type.spelling.find('x')), operand.type.bits};
        auto part = Part(operand, scalar_source, n * operand.type.bits);
        if (!part) { Invalidate(); return; }
        components.push_back(*part);
      }
      packed.components.push_back({Operation("ptx." + i.opcode, "", {scalar_result}, components).result(), scalar_result});
    }
    Bind(destinations[0], packed); return;
  }
  if (In(i.opcode, {"min", "max"}) && operands.size() == 3) {
    auto partial = Expression{Operation("ptx." + i.opcode, "", {result_type}, {operands[0], operands[1]}).result(), result_type};
    operands = {partial, operands[2]};
  }
  if (i.opcode == "cvta") {
    // Explicit symbol/pointer origins already identify the addressed object.
    if (operands[0].value.pointer() && !operands[0].value.pointer()->identity.empty()) {
      Bind(destinations[0], operands[0]); return;
    }
  }
  if (i.opcode == "setp") {
    const std::vector<std::pair<std::string, v::ComparePredicate>> relations{
      {"eq",v::ComparePredicate::Equal},{"ne",v::ComparePredicate::NotEqual},
      {"lt",v::ComparePredicate::Less},{"le",v::ComparePredicate::LessEqual},
      {"gt",v::ComparePredicate::Greater},{"ge",v::ComparePredicate::GreaterEqual},
      {"lo",v::ComparePredicate::Less},{"ls",v::ComparePredicate::LessEqual},
      {"hi",v::ComparePredicate::Greater},{"hs",v::ComparePredicate::GreaterEqual},
      {"num",v::ComparePredicate::Ordered},{"nan",v::ComparePredicate::Unordered}};
    std::optional<v::ComparePredicate> relation;
    bool unordered = false;
    for (const auto& [name, value] : relations) {
      if (Has(i,name)) relation = value;
      if (In(name, {"eq","ne","lt","le","gt","ge"}) && Has(i,name+"u")) { relation = value; unordered = true; }
    }
    if (!relation) { Fail("unsupported_comparison", "Comparison relation is not modeled"); Invalidate(); return; }
    auto p = Expression{Add(std::make_unique<v::CompareOp>(*relation, Pred().value, Has(i,"ftz"), unordered),
        {operands[0].value,operands[1].value}, origin_).result(), Pred()};
    auto combine = [&](Expression condition) {
      if (operands.size() == 2) return condition;
      if (!(Has(i,"and") || Has(i,"or") || Has(i,"xor"))) throw std::invalid_argument("Missing predicate combination operator");
      auto op = Has(i,"and") ? v::BinaryOpcode::BitAnd : Has(i,"or") ? v::BinaryOpcode::BitOr : v::BinaryOpcode::BitXor;
      return Expression{Add(std::make_unique<v::BinaryOp>(op,Pred().value), {condition.value,operands[2].value},origin_).result(),Pred()};
    };
    Bind(destinations[0],combine(p));
    if (destinations.size() == 2) {
      auto q = Expression{Add(std::make_unique<v::UnaryOp>(v::UnaryOpcode::BitNot,Pred().value),{p.value},origin_).result(),Pred()};
      Bind(destinations[1],combine(q));
    }
    return;
  }
  if (Has(i, "cc")) result_types.push_back(Pred());
  std::string mnemonic = i.opcode;
  for (const auto &modifier : i.modifiers) mnemonic += '.' + modifier;
  const auto &node = Operation("ptx." + i.opcode, mnemonic, result_types, operands, std::move(labels));
  for (std::size_t n = 0; n < destinations.size(); ++n) Bind(destinations[n], {node.result(n), result_types[n]});
  if (Has(i, "cc")) carry_ = Expression{node.result(result_types.size() - 1), Pred()};
}

void ValueTemplateExtractor::Memory() {
  const auto &i = *instruction_;
  // The optional cache policy is a performance hint, not a data operand.
  const bool cache_hint = Has(i, "L2::cache_hint");
  if (i.operands.size() != (cache_hint ? 3u : 2u)) {
    Fail("unsupported_memory_operands", "Memory operand form is not a scalar address plus value/register vector"); Invalidate(); return;
  }
  std::optional<Type> type;
  std::uint32_t count = 1;
  for (const auto &modifier : i.modifiers) {
    if (auto parsed = PtxType(modifier)) type = *parsed;
    if (modifier == "v2") count = 2;
    if (modifier == "v4") count = 4;
    if (modifier == "v8") count = 8;
  }
  if (!type || type->elements != 1) {
    Fail("unsupported_memory_type", "Memory element type is unsupported"); Invalidate(); return;
  }
  const bool load = i.opcode != "st";
  std::optional<std::pair<std::string, Integer>> parameter;
  if (load && Has(i, "param")) {
    const auto text = Trim(module_.view(i.operands[1]));
    if (text.size() >= 2 && text.front() == '[' && text.back() == ']')
      parameter = SplitAddress(std::string_view(text).substr(1, text.size() - 2));
    if (!parameter) {
      Fail("unresolved_parameter_field", "Parameter field requires a static base and constant byte offset"); Invalidate(); return;
    }
    if (const auto* known = facts_.Value(*record_, parameter->first))
      if (const auto* pointer = std::get_if<ParameterPointer>(known)) {
        parameter->first = pointer->region;
        parameter->second += pointer->offset;
      }
    const auto declaration = std::find_if(function_.parameters.begin(), function_.parameters.end(),
        [&](const auto& declared) { return declared.name == parameter->first; });
    if (declaration == function_.parameters.end()) {
      Fail("unresolved_parameter_field", "Parameter field address has no statically resolved kernel-parameter identity"); Invalidate(); return;
    }
    std::optional<Integer> extent;
    for (const auto& spelling : declaration->types) {
      const auto element = PtxType(spelling.starts_with('.') ? std::string_view(spelling).substr(1) : spelling);
      if (element && element->bits % 8 == 0) extent = element->bits / 8 * element->elements;
    }
    if (extent && declaration->array) {
      const auto& array = *declaration->array;
      const auto elements = array.size() >= 2 && array.front() == '[' && array.back() == ']'
          ? ParseInteger(std::string_view(array).substr(1, array.size() - 2)) : std::optional<Integer>{};
      if (elements && *elements >= 0) *extent *= *elements;
      else extent.reset();
    }
    const Integer bytes = Integer(count) * type->bits / 8;
    if (!extent || parameter->second < 0 || parameter->second + bytes > *extent) {
      Fail("parameter_field_out_of_bounds", "Parameter field access exceeds its declared PTX parameter storage"); Invalidate(); return;
    }
  }
  auto address = Address(std::string(module_.view(i.operands[load ? 1 : 0])));
  if (!address) { Invalidate(); return; }
  auto text = Trim(module_.view(i.operands[load ? 0 : 1]));
  if (text.starts_with('{') && text.ends_with('}')) text = text.substr(1, text.size() - 2);
  const auto components = Split(text, ',');
  if (components.size() != count) {
    Fail("unsupported_memory_vector", "Memory vector width disagrees with explicit register components"); Invalidate(); return;
  }
  // Scalar memory components share the native instruction/occurrence origin.
  // The component label records their precise byte/register association; this
  // is not a source-language tensor or a cross-thread assembled value.
  for (std::uint32_t n = 0; n < count; ++n) {
    auto component_address = *address;
    if (n) {
      auto offset = Constant(address->type, std::to_string(n * type->bits / 8));
      component_address = {Operation("ptx.address_add", "vector component byte offset", {address->type}, {*address, offset}).result(), address->type};
    }
    std::string mnemonic = i.opcode;
    for (const auto &modifier : i.modifiers) mnemonic += '.' + modifier;
    v::MemoryAttributes attributes;
    if (load && Has(i, "param")) {
      const Integer offset = parameter->second + n * type->bits / 8;
      std::optional<v::PointerProperties> pointer;
      for (const auto& binding : facts_.launch().pointer_parameters())
        if ((binding.parameter == parameter->first || (binding.parameter_index &&
              *binding.parameter_index < function_.parameters.size() &&
              function_.parameters[*binding.parameter_index].name == parameter->first)) &&
            offset == binding.byte_offset && type->bits / 8 == binding.width_bytes) {
          pointer = v::PointerProperties{binding.resource, "global", {}, {}};
          // A resource handle (input.argN.offset) is not necessarily the
          // allocation identity used by the checked memory model. Preserve
          // the declared object identity for an actual allocation-base input.
          // A nonzero suballocation base is not an allocation-base pointer.
          for(const auto& resource:facts_.launch().resources())
            if(resource.value("id",std::string{})==binding.resource && resource.contains("allocation")) {
              const auto& allocation=resource.at("allocation");
              if(allocation.value("resolution",std::string{})=="concrete" &&
                  allocation.value("base_byte_offset",nlohmann::json(0))==nlohmann::json(0)) {
                pointer->identity=allocation.at("object").get<std::string>();
                pointer->addressing=resource.value("space",std::string("global"));
              }
              break;
            }
        }
      std::optional<std::string> specialized;
      // A floating field is specialized by its exact bits, read as b<width>.
      const bool floating = !type->value.element().isBitLike() && type->value.isScalar() &&
          type->value.element().kind() == v::ElementKind::Floating;
      const auto read_type = floating ? *PtxType("b" + std::to_string(type->bits)) : *type;
      if (!pointer && (type->value.element().isBitLike() || floating) && type->bits <= 64) {
        // Carry the existing public-input specialization across the value
        // boundary. Do not use simulated memory values or register contracts.
        if (auto region = facts_.launch().parameter_regions().find(parameter->first);
            region != facts_.launch().parameter_regions().end() && offset <= INT64_MAX) {
          if (auto value = region->second.ReadInteger(offset.convert_to<std::int64_t>(), type->bits / 8,
                  type->value.element().kind() == v::ElementKind::SignedInteger))
            specialized = value->convert_to<std::string>();
        } else if (count == 1 && !floating) {
          auto spelling = Trim(module_.view(i.operands[1]));
          if (spelling.starts_with('[') && spelling.ends_with(']'))
            spelling = Trim(std::string_view(spelling).substr(1, spelling.size() - 2));
          if (auto value = facts_.launch().parameters().find(spelling); value != facts_.launch().parameters().end())
            specialized = value->second.convert_to<std::string>();
        }
      }
      if (floating && specialized) {
        const auto &node = Add(std::make_unique<v::InputParameterOp>(read_type.value,
            parameter->first + "+" + offset.convert_to<std::string>(), pointer, specialized), {}, origin_);
        auto value = Reinterpret(Expression{node.result(), read_type}, *type);
        if (!value) { Invalidate(); return; }
        Bind(components[n], *value);
        continue;
      }
      const auto &node = Add(std::make_unique<v::InputParameterOp>(type->value,
          parameter->first + "+" + offset.convert_to<std::string>(), pointer, specialized), {}, origin_);
      Bind(components[n], {node.result(), *type});
      continue;
    }
    if (load) {
      std::vector<v::Value> operands{component_address.value};
      const auto &node = Add(std::make_unique<v::LoadOp>(type->value, attributes), std::move(operands), origin_);
      Bind(components[n], {node.result(), *type});
    } else {
      auto value = Operand(components[n], *type);
      if (value && type->value.isScalar() && type->value.element().isBitLike() &&
          value->type.value.bitSize() > type->value.bitSize()) {
        // Narrow stores use the low bits, including packed register values.
        value = Part(*value, *type, 0);
      }
      if (value) value = Reinterpret(*value, *type);
      if (!value) continue;
      std::vector<v::Value> operands{component_address.value, value->value};
      Add(std::make_unique<v::StoreOp>(attributes), std::move(operands), origin_);
    }
  }
}

void ValueTemplateExtractor::BulkCopy() {
  const auto& i = *instruction_;
  std::vector<std::string> spaces;
  std::set<std::string> seen;
  for (const auto& modifier : i.modifiers) {
    if (!In(modifier, {"async", "bulk", "weak", "shared::cta", "shared::cluster", "global",
                       "mbarrier::complete_tx::bytes", "L2::cache_hint"}) || !seen.insert(modifier).second) {
      Fail("unsupported_bulk_copy", "Unsupported linear bulk-copy modifier"); return;
    }
    if (In(modifier, {"shared::cta", "shared::cluster", "global"})) spaces.push_back(modifier);
  }
  if (spaces.size() != 2 ||
      !((In(spaces[0], {"shared::cta", "shared::cluster"}) && spaces[1] == "global") ||
        (spaces[0] == "shared::cluster" && spaces[1] == "shared::cta")) ||
      (Has(i, "L2::cache_hint") && spaces[1] != "global") ||
      i.operands.size() != (Has(i, "L2::cache_hint") ? 5u : 4u)) {
    Fail("unsupported_bulk_copy", "Unsupported linear bulk-copy operands or spaces"); return;
  }
  const auto bytes = Known(Trim(module_.view(i.operands[2])));
  if (!bytes || *bytes <= 0 || *bytes > UINT32_MAX || *bytes % 16 != 0) {
    Fail("unsupported_bulk_copy", "Bulk-copy byte count must be a positive constant multiple of 16"); return;
  }
  const auto size = bytes->convert_to<std::uint64_t>();
  const auto endpoint = [&](unsigned index) -> std::optional<v::Value> {
    auto text = Trim(module_.view(i.operands[index]));
    if (!text.starts_with('[') || !text.ends_with(']')) return {};
    if (resources) {
      resources->SetLane(lane_);
      const auto fragment = resources->Memory(*record_, MemoryRequest(text.substr(1, text.size()-2),
                                             nlohmann::json::object(), size, spaces[index]));
      const auto& binding = resources->Binding(fragment.first);
      if (fragment.second == size && binding.expression.empty() && binding.alias_scope.empty())
        return SlotAddressValue(binding);
    }
    if (spaces[1] == "shared::cta") return {};
    auto address = Address(text);
    if (!address) return {};
    return index == 0 ? graph_.asPointer(address->value, {"", "shared", cta_, {}}) :
                        graph_.asPointer(address->value, {"", "global", {}, {}});
  };
  const auto destination = endpoint(0), source = endpoint(1);
  if (!destination || !source) {
    Fail("unsupported_bulk_copy", "Bulk-copy endpoint requires a resolved allocation and byte range"); return;
  }
  const auto type = v::ValueType::matrix(v::ElementType::bit(8), {size});
  auto data = Add(std::make_unique<v::LoadOp>(type), {*source}, origin_).result();
  Add(std::make_unique<v::StoreOp>(), {*destination, data}, origin_);
}

std::optional<v::Value> ValueTemplateExtractor::SpaceAddress(std::string_view operand, std::string_view space,
                                                            std::optional<Expression> extra) {
  const std::string text = Trim(operand);
  if (text.size() < 2 || text.front() != '[' || text.back() != ']') return {};
  const auto split = SplitAddress(std::string_view(text).substr(1, text.size() - 2));
  if (!split) return {};
  auto base = Operand(split->first, U64());
  if (!base || base->type.elements != 1 || base->type.spelling == "pred" ||
      base->type.spelling.starts_with('f') || base->type.spelling.starts_with("bf")) return {};
  if (base->type.value.element().kind() == v::ElementKind::Bit)
    base = Reinterpret(*base, *PtxType("u" + std::to_string(base->type.bits)));
  if (!base) return {};
  const auto add = [&](const Expression& offset) {
    base = Expression{Operation("ptx.address_add", "byte offset", {base->type}, {*base, offset}).result(), base->type};
  };
  if (split->second != 0) add(Constant(base->type, split->second.convert_to<std::string>()));
  if (extra) {
    auto offset = *extra;
    if (offset.type.value != base->type.value)
      offset = Expression{Add(std::make_unique<v::ConvertOp>(base->type.value), {offset.value}, origin_).result(), base->type};
    add(offset);
  }
  v::PointerProperties pointer{"", std::string(space), {}, {}};
  if (space == "shared") pointer.owner_cta = cta_;
  return graph_.asPointer(base->value, pointer);
}

// Tensor-map edits, with the memory effects lowering gives them. TMA consumers
// take their descriptor from the executor's resolution (ResolvedTma); these
// effects keep the descriptor bytes themselves sound for any other reader.
// - tensormap.replace reads and rewrites the 128-byte object at its address.
//   CUtensorMap's layout is opaque, so the rewritten object is an OpaqueValue.
// - tensormap.cp_fenceproxy (warp-aligned) copies it to global memory, each lane
//   its own 4 bytes at 4 * lane.
// An edit this cannot place is unsupported, never a no-op.
void ValueTemplateExtractor::TensorMap() {
  const auto& i = *instruction_;
  constexpr std::uint64_t kBytes = 128;
  const auto text = [&](std::size_t n) { return Trim(module_.view(i.operands.at(n))); };
  const auto bytes = [](std::uint64_t n) { return v::ValueType::matrix(v::ElementType::bit(8), {n}); };
  if (Has(i, "replace")) {
    std::optional<std::string> space;
    if (Has(i, "shared::cta")) space = "shared";
    else if (Has(i, "global")) space = "global";
    if (!Has(i, "tile") || !space || i.operands.size() < 2 || i.operands.size() > 3) {
      Fail("unsupported_tensormap", "Unsupported tensormap.replace form"); Invalidate(); return;
    }
    const auto target = SpaceAddress(text(0), *space, std::nullopt);
    if (!target) {
      Fail("unsupported_tensormap", "tensormap.replace target is not an addressable object"); Invalidate(); return;
    }
    Add(std::make_unique<v::LoadOp>(bytes(kBytes)), {*target}, origin_);
    const auto opaque = Add(std::make_unique<v::OpaqueValueOp>(bytes(kBytes), "tensormap.replace"), {}, origin_).result();
    Add(std::make_unique<v::StoreOp>(), {*target, opaque}, origin_);
    return;
  }
  if (Has(i, "cp_fenceproxy")) {
    const auto size = i.operands.size() == 3 ? Known(text(2)) : std::nullopt;
    if (!Has(i, "global") || !Has(i, "shared::cta") || !Has(i, "aligned") || !size || *size != Integer(kBytes)) {
      Fail("unsupported_tensormap", "Unsupported tensormap.cp_fenceproxy form"); Invalidate(); return;
    }
    const auto binary = [&](v::BinaryOpcode op, v::Value a, v::Value b) {
      return Add(std::make_unique<v::BinaryOp>(op, U32().value), {a, b}, origin_).result();
    };
    const auto lane = binary(v::BinaryOpcode::BitAnd, LinearThread().value, Constant(U32(), "31").value);
    const Expression offset{binary(v::BinaryOpcode::Multiply, lane, Constant(U32(), "4").value), U32()};
    const auto destination = SpaceAddress(text(0), "global", offset), source = SpaceAddress(text(1), "shared", offset);
    if (!destination || !source) {
      Fail("unsupported_tensormap", "tensormap.cp_fenceproxy endpoints are not addressable"); Invalidate(); return;
    }
    const auto data = Add(std::make_unique<v::LoadOp>(bytes(4)), {*source}, origin_).result();
    v::MemoryAttributes descriptor_memory;
    descriptor_memory.interpretation = v::MemoryAttributes::Interpretation::TensorMap;
    Add(std::make_unique<v::StoreOp>(descriptor_memory), {*destination, data}, origin_);
    return;
  }
  Fail("unsupported_tensormap", "Unsupported tensormap operation"); Invalidate();
}

void ValueTemplateExtractor::AsyncCopy() {
  const auto &i = *instruction_;
  if (i.operands.size() < 3 || i.operands.size() > 4 ||
      !Has(i, "global") || !(Has(i, "shared") || Has(i, "shared::cta"))) {
    Fail("unsupported_async_copy", "Only direct global-to-shared cp.async with fixed byte counts is supported"); return;
  }
  const auto bytes = ParseInteger(Trim(module_.view(i.operands[2])));
  auto source_bytes = bytes;
  if (i.operands.size() == 4) {
    const auto operand = Trim(module_.view(i.operands[3]));
    source_bytes = Known(operand);
    if (!ParseInteger(operand)) {
      const auto declaration = declarations_.find(Resolve(operand, i.scope));
      if (declaration == declarations_.end() ||
          (declaration->second.bits != 1 && declaration->second.bits != 32)) {
        Fail("unsupported_async_copy", "cp.async source size requires a 32-bit register or predicate"); return;
      }
      if (source_bytes && declaration->second.bits == 1) {
        if (*source_bytes != 0 && *source_bytes != 1) {
          Fail("unsupported_async_copy", "cp.async ignore-source predicate is not Boolean"); return;
        }
        source_bytes = *source_bytes != 0 ? std::optional<Integer>(Integer(0)) : bytes;
      }
    }
  }
  if (!bytes || !source_bytes || (*bytes != 4 && *bytes != 8 && *bytes != 16) || *source_bytes < 0 || *source_bytes > *bytes) {
    Fail("unsupported_async_copy", "cp.async size/source-size must be statically determined legal byte counts"); return;
  }
  auto destination = Address(std::string(module_.view(i.operands[0])));
  auto source = *source_bytes ? Address(std::string(module_.view(i.operands[1]))) : std::optional<Expression>{};
  if (!destination || (*source_bytes && !source)) return;
  destination->value = graph_.asPointer(destination->value, {"", "shared", cta_, {}});
  if (source) source->value = graph_.asPointer(source->value, {"", "global", {}, {}});
  const auto count = bytes->convert_to<std::uint32_t>();
  const auto valid_count = source_bytes->convert_to<std::uint32_t>();
  const auto byte_type = *PtxType("b8");
  // A cp.async has one contiguous source payload, not unrelated byte reads.
  // Keep that payload identity so wider numerical consumers can select its
  // components even when the global address contains a dynamic page index.
  if (valid_count) {
    const auto type=v::ValueType::matrix(v::ElementType::bit(8),{valid_count});
    const auto data=Add(std::make_unique<v::LoadOp>(type),{source->value},origin_).result();
    Add(std::make_unique<v::StoreOp>(),{destination->value,data},origin_);
  }
  // The ISA source-size operand specifies an exact zero-filled suffix.
  // At most sixteen bytes; no source read is issued for these bytes.
  for (std::uint32_t n = valid_count; n < count; ++n) {
    auto address_at = [&](Expression base) {
      if (!n) return base;
      auto offset = Constant(base.type, std::to_string(n));
      return Expression{Operation("ptx.address_add", "", {base.type}, {base, offset}).result(), base.type};
    };
    auto value = Constant(byte_type, "0");
    Add(std::make_unique<v::StoreOp>(), {address_at(*destination).value, value.value}, origin_);
  }
}

v::Value ValueTemplateExtractor::Native(v::NativeMatrixOpcode opcode, v::ValueType type,
                              std::vector<v::Value> operands, v::NativeMatrixAttributes attributes) {
  return Add(std::make_unique<v::NativeMatrixOp>(opcode, std::move(type), std::move(attributes)),
             std::move(operands), origin_).result();
}
v::Value ValueTemplateExtractor::ClusterRank() {
  return Add(std::make_unique<v::CoordinateOp>(v::CoordinateKind::ClusterRank), {}, origin_).result();
}
v::Value ValueTemplateExtractor::SlotAddressValue(const algorithm::ConstantAddressBinding& binding) {
  auto offset = Constant(U64(), std::to_string(binding.offset));
  return graph_.asPointer(offset.value, {binding.allocation, binding.address_space, {}, {}});
}

void ValueTemplateExtractor::Tma() {
  if (ResolvedTma()) return;
  const auto& i = *instruction_;
  // Only constant byte copies model a gather; the descriptor-shaped native load
  // below would describe one map box, not the four gathered rows.
  if (Has(i, "tile::gather4")) {
    Fail("tma_gather4_dynamic_rows", "TMA tile::gather4 coordinates are not statically known "
         "(data-dependent rows); the gathered values are not modeled");
    Invalidate(); return;
  }
  const bool load = Has(i, "mbarrier::complete_tx::bytes");
  const bool multicast = Has(i, "multicast::cluster");
  unsigned rank = 0;
  for (unsigned n = 1; n <= 5; ++n) if (Has(i, std::to_string(n) + "d")) rank = n;
  for (const auto& mod : i.modifiers)
    if (!(In(mod, {"async", "bulk", "tensor", "global", "shared::cta", "shared::cluster", "tile", "tile::gather4",
          "mbarrier::complete_tx::bytes", "bulk_group", "multicast::cluster", "L2::cache_hint", "cta_group::1", "cta_group::2"}) ||
          mod == std::to_string(rank) + "d"))
      throw std::invalid_argument("Unsupported TMA value modifier: " + mod);
  if (!rank || !Has(i, "global") || (!load && !Has(i, "bulk_group")) || (!load && multicast))
    throw std::invalid_argument("Unsupported TMA direction/completion form");
  const auto count = (load ? 3u : 2u) + multicast + Has(i, "L2::cache_hint");
  if (i.operands.size() != count) throw std::invalid_argument("TMA operand count mismatch");
  const auto text = [&](unsigned n) { return Trim(module_.view(i.operands.at(n))); };
  const auto tensor = text(load ? 1 : 0);
  static const std::regex form(R"(^\[\s*([^,]+),\s*\{([^}]+)\}\s*\]$)");
  std::smatch match;
  if (!std::regex_match(tensor, match, form)) throw std::invalid_argument("Unsupported tensor-map operand form");
  const auto coordinates = Split(match[2].str(), ',');
  if (coordinates.size() != (Has(i,"tile::gather4") ? 5u : rank) || (Has(i,"tile::gather4") && (!load || rank != 2)))
    throw std::invalid_argument("TMA coordinate count mismatch");
  auto descriptor_pointer = Operand(Trim(match[1].str()), U64());
  auto shared = Address(text(load ? 0 : 1));
  if (!descriptor_pointer || !shared) return;
  shared->value = graph_.asPointer(shared->value, {"", Has(i,"shared::cluster") ? "cluster-shared-address" : "shared",
      Has(i,"shared::cluster") ? std::optional<std::uint64_t>{} : cta_, {}});
  std::vector<v::Value> coords;
  for (const auto& coordinate : coordinates) {
    auto value = Operand(coordinate, *PtxType("s32"));
    if (value) value = Reinterpret(*value, *PtxType("s32"));
    if (!value) return;
    coords.push_back(value->value);
  }
  v::MemoryAttributes descriptor_memory;
  descriptor_memory.interpretation = v::MemoryAttributes::Interpretation::TensorMap;
  auto descriptor = Add(std::make_unique<v::LoadOp>(v::ValueType::matrix(v::ElementType::bit(64), {16}), descriptor_memory),
      {descriptor_pointer->value}, origin_).result();
  std::string view = "tma.tensor";
  for (const auto& mod : i.modifiers) view += '.' + mod;
  const auto matrix = v::ValueType::parameterizedMatrix(v::ElementType::fromDescriptor(descriptor, view), descriptor, view);
  const auto address_type = v::ValueType::parameterizedMatrix(v::ElementType::unsignedInteger(64), descriptor, view);
  v::NativeMatrixAttributes attrs{i.modifiers}; attrs.coordinate_count = coordinates.size();
  std::vector<v::Value> global_args{descriptor}; global_args.insert(global_args.end(), coords.begin(), coords.end());
  auto global = Native(v::NativeMatrixOpcode::TmaGlobalAddress, address_type, global_args, attrs);
  global = graph_.asPointer(global, {"", "global", {}, {}});
  const auto issuer = ClusterRank();
  v::MemoryAttributes tensor_memory;
  tensor_memory.interpretation = v::MemoryAttributes::Interpretation::TmaTensor;
  tensor_memory.native_modifiers = i.modifiers;
  if (!load) {
    attrs.role = v::MatrixRole::Source;
    auto src = Native(v::NativeMatrixOpcode::TmaSharedAddress, address_type,
        {shared->value, descriptor, issuer, issuer}, attrs);
    src = graph_.asPointer(src, {"", "shared", cta_, {}});
    const auto data = Add(std::make_unique<v::LoadOp>(matrix), {src}, origin_).result();
    Add(std::make_unique<v::StoreOp>(tensor_memory), {global, data}, origin_);
    return;
  }
  const auto data = Add(std::make_unique<v::LoadOp>(matrix, tensor_memory), {global}, origin_).result();
  auto shape = frame_.cluster_dim();
  const auto cluster_size = shape[0] * shape[1] * shape[2];
  const auto current = frame_.self().value;
  std::optional<Expression> mask;
  if (multicast) {
    mask = Operand(text(3), U32());
    if (mask && mask->type.bits == 16) {
      mask = Reinterpret(*mask,*PtxType("u16"));
      if (mask) mask=Expression{Add(std::make_unique<v::ConvertOp>(U32().value),{mask->value},origin_).result(),U32()};
    }
    if (mask) mask = Reinterpret(*mask, U32());
    if (!mask) return;
  }
  if (multicast && cluster_size > 16) throw std::invalid_argument("TMA multicast mask supports at most 16 CTA ranks");
  const auto targets = multicast ? cluster_size : 1u;
  for (unsigned n = 0; n < targets; ++n) {
    const auto target_rank = multicast ? n : current;
    auto target = Constant(U32(), std::to_string(target_rank));
    attrs.role = v::MatrixRole::Destination;
    auto dst = Native(v::NativeMatrixOpcode::TmaSharedAddress, address_type,
        {shared->value, descriptor, issuer, target.value}, attrs);
    // Non-multicast ownership follows dstMem; cta_group only selects completion targets.
    dst = graph_.asPointer(dst, {"", "cluster-shared-view", {}, {}});
    v::MemoryAttributes store;
    std::vector<v::Value> args{dst, data};
    if (mask) {
      auto bit = Constant(U32(), std::to_string(std::uint64_t(1) << target_rank));
      auto enabled = Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::BitAnd, U32().value), {mask->value, bit.value}, origin_).result();
      auto zero = Constant(U32(), "0");
      auto predicate = Add(std::make_unique<v::CompareOp>(v::ComparePredicate::NotEqual, Pred().value), {enabled, zero.value}, origin_).result();
      args.push_back(predicate); store.guarded = true;
    }
    Add(std::make_unique<v::StoreOp>(store), std::move(args), origin_);
  }
}

bool ValueTemplateExtractor::ResolvedTma() {
  if (!resources) return false;
  resources->SetLane(lane_);
  TmaAccesses transfer;
  try {
    const auto& contract = facts_.launch().InstructionContract(instruction_->line, instruction_->id,
        facts_.Warp(lane_), record_->occurrence());
    transfer = resources->Tma(*record_, TmaRequest(contract, Has(*instruction_, "bulk_group")));
  } catch (const SemanticError&) { return false; }
  catch (const LayoutError& error) {
    // An unsupported gather layout stays a value diagnostic; the event view
    // reports the same layout error for the instruction.
    if (!Has(*instruction_, "tile::gather4")) throw;
    Fail(error.code, error.what()); Invalidate(); return true;
  }
  for (const auto& pair : transfer.pairs)
    for (const auto* endpoint : {&pair.source, &pair.destination})
      if (*endpoint) {
        const auto& [id, bytes] = **endpoint;
        const auto& binding = resources->Binding(id);
        if (!bytes || !*bytes || !binding.expression.empty() || !binding.alias_scope.empty()) return false;
      }
  const auto address = [&](const Fragment& fragment) {
    return SlotAddressValue(resources->Binding(fragment.first));
  };
  // The read of a runtime-written tensor map (the descriptor itself comes from its resolution).
  if (transfer.descriptor)
    Add(std::make_unique<v::LoadOp>(v::ValueType::matrix(v::ElementType::bit(8), {*transfer.descriptor->second})),
        {address(*transfer.descriptor)}, origin_);
  std::map<std::tuple<algorithm::AddressId, std::optional<std::uint64_t>, std::uint64_t>, v::Value> reads;
  for (const auto& pair : transfer.pairs) {
    std::optional<v::Value> data;
    if (pair.source) {
      const auto key = std::tuple{pair.source->first, pair.source->second, pair.logical_offset};
      if (const auto found = reads.find(key); found != reads.end()) data = found->second;
      else {
        const auto type = v::ValueType::matrix(v::ElementType::bit(8), {*pair.source->second});
        data = Add(std::make_unique<v::LoadOp>(type), {address(*pair.source)}, origin_).result();
        reads.emplace(key, *data);
      }
    }
    if (!pair.destination) continue;
    const auto type = v::ValueType::matrix(v::ElementType::bit(8), {*pair.destination->second});
    if (!data) data = Add(std::make_unique<v::ConstantOp>(type, "0"), {}, origin_).result();
    if (graph_.type(*data) != type) throw std::invalid_argument("Resolved TMA fragment byte extents differ");
    Add(std::make_unique<v::StoreOp>(), {address(*pair.destination), *data}, origin_);
  }
  return true;
}

void ValueTemplateExtractor::TcgenMma() {
  const auto& i = *instruction_;
  const bool sparse = Has(i, "sp"), scaled = Has(i, "block_scale");
  const unsigned descriptor_index = sparse ? 4 : 3;
  if (i.operands.size() <= descriptor_index + 1) throw std::invalid_argument("MMA operands missing");
  for (const auto& mod : i.modifiers)
    if (!In(mod, {"mma", "sp", "ws", "cta_group::1", "cta_group::2", "kind::f16", "kind::tf32", "kind::i8",
        "kind::f8f6f4", "kind::mxf8f6f4", "kind::mxf4", "kind::mxf4nvf4", "block_scale", "block16", "block32",
        "scale_vec::1X", "scale_vec::2X", "scale_vec::4X"}))
      throw std::invalid_argument("Unsupported MMA value modifier: " + mod);
  const bool ws = Has(i, "ws");
  if (ws && (sparse || scaled || Has(i, "cta_group::2")))
    throw std::invalid_argument("Unsupported .ws MMA value variant");
  const auto text = [&](unsigned n) { return Trim(module_.view(i.operands.at(n))); };
  auto d = Address(text(0)), a = text(1).starts_with('[') ? Address(text(1)) : Operand(text(1), U64());
  auto b = Operand(text(2), U64()), descriptor = Operand(text(descriptor_index), U32());
  if (a) a = Bits(*a);
  if (b) b = Bits(*b);
  if (descriptor) descriptor = Bits(*descriptor);
  if (!d || !a || !b || !descriptor) return;
  unsigned next = descriptor_index + 1;
  std::optional<Expression> sa, sb, meta;
  if (scaled) {
    if (i.operands.size() <= next + 2) throw std::invalid_argument("MMA scale operands missing");
    sa = Address(text(next++)); sb = Address(text(next++)); if (!sa || !sb) return;
  }
  if (sparse) { meta = Address(text(3)); if (!meta) return; }
  std::vector<v::Value> masks;
  bool masks_are_zero = true;
  if (text(next).starts_with('{')) {
    if (ws) throw std::invalid_argument(".ws MMA has no output lane mask operand");
    const auto mask_text = text(next++);
    if (mask_text.back() != '}') throw std::invalid_argument("MMA output mask is malformed");
    for (const auto& item : Split(std::string_view(mask_text).substr(1, mask_text.size()-2), ',')) {
      auto value = Operand(item, U32()); if (!value) return; masks.push_back(value->value);
      const auto known = Known(item);
      masks_are_zero = masks_are_zero && known && *known == 0;
    }
    if (masks.size() != (Has(i,"cta_group::2") ? 8u : 4u)) throw std::invalid_argument("MMA output mask width mismatch");
  }
  if (next >= i.operands.size()) throw std::invalid_argument("MMA enable-input-d missing");
  const auto enabled_text = text(next++);
  const auto enabled_known = Known(enabled_text);
  auto enabled = Operand(enabled_text, Pred()); if (!enabled) return;
  if (enabled->type.value.element().kind() != v::ElementKind::Predicate) throw std::invalid_argument("MMA enable-input-d must be a predicate");
  std::optional<Expression> exponent;
  if (ws && next < i.operands.size()) {
    // The .ws trailing operand is the zero-column mask descriptor, not scale-input-d.
    // Only a known zero (no masked or shifted B columns) keeps D = A*B(+D).
    const auto zero_column_mask = Known(text(next++));
    if (!zero_column_mask || *zero_column_mask != 0)
      throw std::invalid_argument("Unsupported .ws zero-column mask");
  } else if (next < i.operands.size()) { exponent = Operand(text(next++), U32()); if (!exponent) return; }
  if (next != i.operands.size()) throw std::invalid_argument("Unexpected MMA operands");
  const auto issuer = ClusterRank();
  std::string mode = "tcgen05";
  for (const auto& mod : i.modifiers) mode += '.' + mod;
  std::optional<MmaDescriptor> fixed_descriptor;
  auto known_descriptor=Known(text(descriptor_index));
  if (!known_descriptor && scaled && resources && tmem_provenance) {
    resources->SetLane(lane_);
    const auto resolver=[&](const RecordView& record,std::string_view token,unsigned)
        ->std::optional<Resources::KnownBits> {
      auto value=tmem_provenance->Resolve(facts_.Warp(lane_),record,token,facts_.LaneId(lane_));
      const auto* origin=facts_.Origin(record,token);
      const auto* current=facts_.Origin(*record_,token);
      if(std::holds_alternative<std::monostate>(value) && origin && current && *origin==*current)
        value=tmem_provenance->Resolve(facts_.Warp(lane_),*record_,token,facts_.LaneId(lane_));
      if(const auto* pointer=std::get_if<TmemPointer>(&value))
        return Resources::KnownBits{pointer->encoded_offset&0xc0000000U,0xc0000000U};
      return {};
    };
    const auto [value,mask]=resources->FixedBits(*record_,text(descriptor_index),32,resolver);
    if(mask==UINT32_MAX)known_descriptor=Integer(value);
  }
  if (!sparse)
    if (const auto known = known_descriptor; known && *known >= 0 && *known <= UINT32_MAX) {
      descriptor = Constant(U32(), known->convert_to<std::string>());
      for (const auto& modifier : i.modifiers)
        if (modifier.starts_with("kind::"))
          try { fixed_descriptor = scaled ? DecodeBlockScaledDescriptor(known->convert_to<std::uint64_t>(),modifier.substr(6)) :
              DecodeMmaDescriptor(modifier.substr(6),known->convert_to<std::uint64_t>()); }
          catch (const LayoutError&) {}
    }
  const auto type = [&](std::string role, bool address = false) {
    const auto view = mode + ':' + role;
    if (fixed_descriptor && (role == "A" || role == "B" || role == "D")) {
      const auto& d = *fixed_descriptor;
      auto bits = role == "A" ? d.a_element_bits : role == "B" ? d.b_element_bits : d.d_element_bits;
      const bool packed = scaled && bits==4 && role!="D";
      if (packed) bits=8;
      return v::ValueType::matrix(address ? v::ElementType::unsignedInteger(64) : v::ElementType::bit(bits),
          {role == "A" ? d.m : role == "B" ? d.n : d.m, role == "D" ? d.n : packed ? d.k/2 : d.k});
    }
    if (fixed_descriptor && scaled && (role=="scale_A" || role=="scale_B")) {
      const auto& d=*fixed_descriptor;
      const unsigned group=Has(i,"cta_group::2")?2:1;
      const unsigned vector=d.kind=="mxf4"?2:d.kind=="mxf4nvf4"?4:1;
      return v::ValueType::matrix(address ? v::ElementType::unsignedInteger(64) : v::ElementType::bit(8),
          {group,4,role=="scale_A"?d.m/group:d.n,vector});
    }
    return v::ValueType::parameterizedMatrix(address ? v::ElementType::unsignedInteger(64) :
        v::ElementType::fromDescriptor(descriptor->value, view), descriptor->value, view);
  };
  const auto address = [&](Expression base, v::MatrixRole role, const char* view, const char* space) {
    v::NativeMatrixAttributes attrs{i.modifiers}; attrs.role = role;
    base.value = graph_.asPointer(base.value, {"", space, cta_, {}});
    auto result = Native(v::NativeMatrixOpcode::TcgenMmaAddress, type(view, true), {base.value, descriptor->value, issuer}, attrs);
    return graph_.asPointer(result, {"", space, Has(i,"cta_group::2") ? std::optional<std::uint64_t>{} : cta_, {}});
  };
  auto read = [&](v::Value addr, const char* role) {
    return Add(std::make_unique<v::LoadOp>(type(role)), {addr}, origin_).result();
  };
  auto av = read(address(*a, v::MatrixRole::A, "A", text(1).starts_with('[') ? "tmem" : "shared-matrix-descriptor"), "A");
  auto bv = read(address(*b, v::MatrixRole::B, "B", "shared-matrix-descriptor"), "B");
  auto dst = address(*d, v::MatrixRole::D, "D", "tmem");
  v::TcgenMmaAttributes mma{i.modifiers, !enabled_known || *enabled_known != 0, scaled, sparse, exponent.has_value()};
  std::vector<v::Value> args{av, bv, descriptor->value};
  if (mma.accumulate) {
    v::MemoryAttributes memory; memory.guarded = !enabled_known;
    std::vector<v::Value> load_args{dst}; if (memory.guarded) load_args.push_back(enabled->value);
    args.push_back(Add(std::make_unique<v::LoadOp>(type("D"), memory), std::move(load_args), origin_).result());
  }
  if (scaled) {
    args.push_back(read(address(*sa,v::MatrixRole::ScaleA,"scale_A","tmem"),"scale_A"));
    args.push_back(read(address(*sb,v::MatrixRole::ScaleB,"scale_B","tmem"),"scale_B"));
  }
  if (sparse) args.push_back(read(address(*meta,v::MatrixRole::SparseMetadata,"metadata","tmem"),"metadata"));
  if (exponent) args.push_back(exponent->value);
  auto result = Add(std::make_unique<v::TcgenMmaOp>(type("D"), mma), std::move(args), origin_).result();
  std::vector<v::Value> store_args{dst, result}; v::MemoryAttributes store;
  if (!masks.empty() && !masks_are_zero) {
    std::vector<v::Value> mask_args{descriptor->value, d->value}; mask_args.insert(mask_args.end(), masks.begin(), masks.end());
    v::NativeMatrixAttributes attrs{i.modifiers}; attrs.mask_words = masks.size();
    auto mask_type = v::ValueType::parameterizedMatrix(v::ElementType::predicate(), descriptor->value, mode + ":D");
    store_args.push_back(Native(v::NativeMatrixOpcode::TcgenMmaOutputMask, mask_type, mask_args, attrs));
    store.guarded = true;
  }
  Add(std::make_unique<v::StoreOp>(store), std::move(store_args), origin_);
}

void ValueTemplateExtractor::Tcgen() {
  const auto& i = *instruction_;
  if (Has(i, "mma")) { TcgenMma(); return; }
  const auto text = [&](unsigned n) { return Trim(module_.view(i.operands.at(n))); };
  if (Has(i, "ld") || Has(i, "st")) {
    const bool load = Has(i, "ld");
    unsigned registers = 0, repeats = 0;
    for (const auto& mod : i.modifiers) {
      if (mod == "32x32b" || mod == "16x64b" || mod == "16x32bx2") registers = 1;
      if (mod == "16x128b") registers = 2;
      if (mod == "16x256b") registers = 4;
      if (std::regex_match(mod, std::regex("x(1|2|4|8|16|32|64|128)"))) repeats = std::stoul(mod.substr(1));
      if (!(In(mod,{"ld","st","sync","aligned","b32","32x32b","16x64b","16x128b","16x256b","16x32bx2","pack::16b","unpack::16b"}) ||
          std::regex_match(mod,std::regex("x(1|2|4|8|16|32|64|128)"))))
        throw std::invalid_argument("Unsupported TMEM register modifier: " + mod);
    }
    if (!registers || !repeats || !Has(i,"sync") || !Has(i,"aligned") || !Has(i,"b32"))
      throw std::invalid_argument("Incomplete TMEM register instruction");
    registers *= repeats;
    if (registers > 128 || (load && Has(i,"unpack::16b")) || (!load && Has(i,"pack::16b")))
      throw std::invalid_argument("Invalid TMEM register packing/width");
    if (i.operands.size() != (Has(i,"16x32bx2") ? 3u : 2u)) throw std::invalid_argument("TMEM register operand count mismatch");
    auto vector = text(load ? 0 : Has(i,"16x32bx2") ? 2 : 1);
    if (!vector.starts_with('{') || !vector.ends_with('}')) throw std::invalid_argument("TMEM register vector is missing");
    auto components = Split(std::string_view(vector).substr(1,vector.size()-2), ',');
    if (components.size() != registers) throw std::invalid_argument("TMEM register vector width mismatch");
    auto base = Address(text(load ? 1 : 0)); if (!base) return;
    base->value = graph_.asPointer(base->value,{"","tmem",cta_,{}});
    auto offset = Has(i,"16x32bx2") ? Operand(text(load ? 2 : 1),U32()) : std::optional<Expression>(Constant(U32(),"0"));
    if (!offset) return;
    auto thread = LinearThread();
    const auto data_type = *PtxType("b32");
    const bool packed=Has(i,"pack::16b") || Has(i,"unpack::16b");
    const auto element=packed ? *PtxType("b16") : data_type;
    std::optional<v::Value> loaded_components;
    if(load) {
      // One native instruction produces an indexed register payload, not a
      // collection of unrelated Load roots. Spatial members still have their
      // own instances; each physical fragment retains its source event binding.
      const auto count=registers*(packed?2u:1u);
      v::NativeMatrixAttributes attrs{i.modifiers};
      const auto address_type=v::ValueType::matrix(v::ElementType::unsignedInteger(64),{count});
      auto addr=Native(v::NativeMatrixOpcode::TcgenRegisterAddress,address_type,
          {base->value,thread.value,offset->value},attrs);
      addr=graph_.asPointer(addr,{"","tmem",cta_,{}});
      v::MemoryAttributes memory;
      memory.native_modifiers=i.modifiers;
      loaded_components=Add(std::make_unique<v::LoadOp>(
          v::ValueType::matrix(element.value.element(),{count}),memory),{addr},origin_).result();
    }
    for (unsigned n = 0; n < registers; ++n) {
      std::optional<Expression> source, combined;
      if (!load) { source=Operand(components[n],data_type); if (source) source=Reinterpret(*source,data_type); if (!source) continue; }
      for (unsigned half=0; half < (packed ? 2u : 1u); ++half) {
        v::NativeMatrixAttributes attrs{i.modifiers}; attrs.register_index = n;
        if (packed) attrs.packed_half=half;
        if (load) {
          const auto index=Constant(U32(),std::to_string(n*(packed?2u:1u)+half));
          Expression value{graph_.component(*loaded_components,{index.value}),element};
          if (packed) {
            auto integer=Reinterpret(value,*PtxType("u16")); if (!integer) return;
            value={Add(std::make_unique<v::ConvertOp>(U32().value),{integer->value},origin_).result(),U32()};
            if (half) {
              auto shift=Constant(U32(),"16");
              auto upper=Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::ShiftLeft,U32().value),{value.value,shift.value},origin_).result();
              value={Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::BitOr,U32().value),{combined->value,upper},origin_).result(),U32()};
            }
          }
          combined=value;
        } else {
          auto addr = Native(v::NativeMatrixOpcode::TcgenRegisterAddress,U64().value,{base->value,thread.value,offset->value},attrs);
          addr = graph_.asPointer(addr,{"","tmem",cta_,{}});
          auto value=packed ? Part(*source,element,16*half) : source;
          if (!value) return;
          Add(std::make_unique<v::StoreOp>(),{addr,value->value},origin_);
        }
      }
      if (load && combined) { auto value=Reinterpret(*combined,data_type); if (value) Bind(components[n],*value); }
    }
    return;
  }
  if (Has(i,"cp")) {
    unsigned rows = 0, bits = 0;
    for (const auto& mod : i.modifiers) {
      std::smatch m;
      if (std::regex_match(mod,m,std::regex("(128|64|32|4)x(128|256)b"))) { rows=std::stoul(m[1]);bits=std::stoul(m[2]); }
      else if (!In(mod,{"cp","cta_group::1","cta_group::2","warpx4","warpx2::02_13","warpx2::01_23"}))
        throw std::invalid_argument("Unsupported tcgen05.cp value modifier: " + mod);
    }
    if (!rows || !bits || i.operands.size()!=2 || !((rows==128 && (bits==128 || bits==256)) ||
        (rows==4 && bits==256) || ((rows==32 || rows==64) && bits==128)))
      throw std::invalid_argument("Invalid tcgen05.cp shape/operands");
    auto dst=Address(text(0)), src=Operand(text(1),U64());
    if (src) src=Reinterpret(*src,U64());
    if (!dst || !src) return;
    src->value=graph_.asPointer(src->value,{"","shared-matrix-descriptor",cta_,{}});
    dst->value=graph_.asPointer(dst->value,{"","tmem",cta_,{}});
    const auto issuer=ClusterRank();
    std::string view="tcgen05.copy"; for (const auto& mod : i.modifiers) view+='.'+mod;
    const auto type=[&](const char* role, bool address=false) {
      if (!address && (rows==128 || (rows==32 && Has(i,"warpx4"))))
        return v::ValueType::matrix(v::ElementType::bit(8),
            {std::uint64_t(rows)*bits/8*(Has(i,"cta_group::2")?2:1)*(Has(i,"warpx4")?4:1)});
      return v::ValueType::parameterizedMatrix(address ? v::ElementType::unsignedInteger(64) : v::ElementType::bit(8),
          src->value,view+':'+role);
    };
    v::NativeMatrixAttributes attrs{i.modifiers};
    const auto owner=Has(i,"cta_group::2") ? std::optional<std::uint64_t>{} : cta_;
    auto source=Native(v::NativeMatrixOpcode::TcgenCopyAddress,type("source",true),{src->value,issuer},attrs);
    source=graph_.asPointer(source,{"","shared-matrix-descriptor",owner,{}});
    auto value=Add(std::make_unique<v::LoadOp>(type("source")),{source},origin_).result();
    attrs.role=v::MatrixRole::Destination;
    value=Native(v::NativeMatrixOpcode::TcgenCopyValue,type("destination"),{value},attrs);
    auto destination=Native(v::NativeMatrixOpcode::TcgenCopyAddress,type("destination",true),{dst->value,issuer},attrs);
    destination=graph_.asPointer(destination,{"","tmem-copy-view",owner,{}});
    Add(std::make_unique<v::StoreOp>(),{destination,value},origin_);
    return;
  }
  if (Has(i,"alloc")) {
    if (i.operands.size()!=2) throw std::invalid_argument("Invalid TMEM allocation operands");
    if (facts_.LaneId(lane_) != 0) return;
    auto dst=Address(text(0)), columns=Operand(text(1),U32()); if (!dst || !columns) return;
    dst->value=graph_.asPointer(dst->value,{"","shared",cta_,{}});
    auto pointer=Native(v::NativeMatrixOpcode::TmemAllocation,U32().value,{columns->value},v::NativeMatrixAttributes{i.modifiers});
    pointer=graph_.asPointer(pointer,{"","tmem",cta_,{}});
    Add(std::make_unique<v::StoreOp>(),{dst->value,pointer},origin_);
    return;
  }
  if (In(i.semantics.operation,{"commit","wait_load","wait_store","fence","dealloc","relinquish_alloc_permit"})) return;
  Fail("unsupported_tcgen_value","No numerical model for this tcgen05 variant"); Invalidate();
}

std::optional<Expression> ValueTemplateExtractor::DemandOperand(const std::string& text, const Type& requested) {
  const auto key = Resolve(text, instruction_->scope);
  const auto type = declarations_.contains(key) ? declarations_.at(key) : requested;
  if (const auto* known = facts_.Value(*record_, text)) {
    if (const auto number = ScalarInteger(*known))
      return Constant(type, number->convert_to<std::string>());
    if (const auto* pointer = std::get_if<AllocationPointer>(known); pointer && type.bits == 64) {
      v::PointerProperties properties{pointer->resource, "global", {}, {}};
      return Expression{Add(std::make_unique<v::ConstantOp>(type.value,
          pointer->offset.convert_to<std::string>(), properties), {}, origin_).result(), type};
    }
  }
  const auto producer = demand_producer(*record_, key);
  if (!producer || !demand_definition || demand_depth >= 128 || graph_.nodes().size() >= 100000) return {};
  const auto cache_key = std::pair{*producer, text};
  if (const auto found = demand_cache.find(cache_key); found != demand_cache.end()) return found->second;
  demand_cache.emplace(cache_key, std::nullopt);
  const auto definition = demand_definition(lane_, *producer);
  if (!definition || definition->predicate_value() == false ||
      !definition->opaque_compute_diamond_indices().empty()) return {};
  const auto& instruction = function_.instructions.at(definition->instruction_id());
  if (Has(instruction, "cc") || In(instruction.opcode, {"addc", "subc", "madc"})) return {};
  auto result = Demand(*definition, text);
  demand_cache[cache_key] = result;
  return result;
}

std::optional<Expression> ValueTemplateExtractor::Demand(const RecordView& record, const std::string& text) {
  struct Restore {
    ValueTemplateExtractor& self;
    std::optional<RecordView> record;
    const Instruction* instruction;
    v::SourceOrigin origin;
    std::map<Key, Expression> registers;
    std::uint64_t suboperation;
    ~Restore() {
      self.record_ = record; self.instruction_ = instruction; self.origin_ = std::move(origin);
      self.registers_ = std::move(registers); self.suboperation_ = suboperation; --self.demand_depth;
    }
  } restore{*this, record_, instruction_, origin_, std::move(registers_), suboperation_};
  ++demand_depth;
  record_ = record;
  instruction_ = &function_.instructions.at(record.instruction_id());
  suboperation_ = 0;
  BeginOrigin(record);
  try {
    if (text.starts_with('[')) return Address(text);
    if (instruction_->opcode == "shfl") Shuffle();
    else if (instruction_->opcode == "ld" || instruction_->opcode == "ldu") Memory();
    else if (instruction_->semantics.category == "compute") Compute();
    else return {};
    const auto found = registers_.find(Resolve(text, instruction_->scope));
    if (found != registers_.end()) return found->second;
  } catch (const std::exception&) {
    // Preserve an opaque address when its dependency slice is unsupported.
  }
  return {};
}

// Explicit DAG serialization avoids duplicating shared subexpressions in JSON.
class AddressDagWriter {
  using Json = nlohmann::json;
  const v::Graph& graph;
  Json nodes = Json::array();
  using Key = std::tuple<std::size_t, std::size_t, std::size_t, const v::TypeConversion*>;
  std::map<Key, std::size_t> ids;
  std::vector<v::Value> retained_values;
  bool canonical_terms = false;
  std::unordered_map<std::string, std::size_t> canonical_ids;
  std::size_t slice_visits = 0;
  std::uint64_t pool_id = 0;
public:
  explicit AddressDagWriter(const v::Graph& graph, bool canonical = false)
      : graph(graph), canonical_terms(canonical) {
    static std::atomic<std::uint64_t> next_pool{1};
    if (canonical) pool_id = next_pool.fetch_add(1, std::memory_order_relaxed);
  }
  std::optional<std::uint64_t> Selector(v::Value value, unsigned depth = 0) const {
    if (depth > 64) return {};
    if (value.threadSelector()) return Selector(Selected(value),depth+1);
    if (!value.indices().empty() ||
        !value.type(graph).isScalar() || !value.type(graph).element().isInteger()) return {};
    const auto bits = value.type(graph).element().width();
    if (!bits || bits > 64) return {};
    const auto mask = bits == 64 ? UINT64_MAX : (std::uint64_t(1) << bits)-1;
    if (value.conversion()) {
      if (value.conversion()->spec->kind != v::TypeConversionKind::Bitcast) return {};
      auto result = Selector(value.conversion()->source, depth+1);
      return result ? std::optional<std::uint64_t>(*result & mask) : std::nullopt;
    }
    const auto& op = graph.definingNode(value).operation();
    if (const auto* literal = dynamic_cast<const v::ConstantOp*>(&op)) {
      auto integer = ParseInteger(literal->literal());
      if (!integer || literal->pointer()) return {};
      return (*integer & Integer(mask)).convert_to<std::uint64_t>();
    }
    const auto instance = graph.instance(value);
    if (const auto* coordinate = dynamic_cast<const v::CoordinateOp*>(&op)) {
      const auto axis = static_cast<unsigned>(coordinate->axis());
      if (coordinate->kind() == v::CoordinateKind::ThreadIndex && instance.origin.thread_coordinates)
        return instance.origin.thread_coordinates->at(axis);
      if (coordinate->kind() == v::CoordinateKind::BlockIndex && instance.origin.cta_coordinates)
        return instance.origin.cta_coordinates->at(axis);
      return {};
    }
    if (const auto* binary = dynamic_cast<const v::BinaryOp*>(&op)) {
      if (binary->attributes().saturate) return {};
      const auto a = Selector(instance.operands.at(0),depth+1);
      const auto b = Selector(instance.operands.at(1),depth+1);
      if (!a || !b) return {};
      switch (binary->opcode()) {
        case v::BinaryOpcode::Add: return (*a + *b) & mask;
        case v::BinaryOpcode::Subtract: return (*a - *b) & mask;
        case v::BinaryOpcode::BitAnd: return *a & *b & mask;
        case v::BinaryOpcode::BitOr: return (*a | *b) & mask;
        case v::BinaryOpcode::BitXor: return (*a ^ *b) & mask;
        default: return {};
      }
    }
    return {};
  }
  v::Value Selected(v::Value value) const {
    if (!value.threadSelector()) return value;
    const auto thread = Selector(*value.threadSelector());
    if (!thread) throw std::invalid_argument("address thread selector is not constant");
    if (!value.indices().empty()) throw std::invalid_argument("indexed address thread selection");
    v::Value result;
    if (value.conversion()) {
      const auto& conversion = *value.conversion();
      auto source = Selected(graph.fromThread(conversion.source,*value.threadSelector()));
      result = conversion.spec->kind == v::TypeConversionKind::Bitcast
          ? graph.bitcast(source,conversion.spec->target)
          : graph.convert(source,conversion.spec->target,conversion.spec->attributes);
    } else {
      const auto& node = graph.definingNode(value);
      const auto member = node.memberIndex(*thread);
      if (!member) throw std::invalid_argument("address selected thread is absent");
      result = node.result(value.resultIndex(),*member);
      // A ThreadForward instance is that source thread's own value.
      if (dynamic_cast<const v::ThreadForwardOp*>(&node.operation())) {
        if (value.pointer())
          throw std::invalid_argument("an address forwarded from another thread carries a pointer annotation");
        return Selected(graph.forwarded(result));
      }
    }
    if (value.pointer()) result = graph.asPointer(result,*value.pointer());
    return result;
  }
  v::Value Normalize(v::Value value) const {
    return NormalizeAddressExpression(graph,value,[&](v::Value x) { return Selected(x); });
  }
  std::optional<std::string> Allocation(v::Value value, unsigned depth = 0) const {
    value = Normalize(value);
    if (depth > 128 || value.type(graph).bitSize() != 64) return {};
    if (value.conversion()) {
      if (value.conversion()->spec->kind != v::TypeConversionKind::Bitcast) return {};
      return Allocation(value.conversion()->source, depth + 1);
    }
    const auto& op = graph.definingNode(value).operation();
    if (dynamic_cast<const v::BitcastOp*>(&op))
      return Allocation(graph.instance(value).operands.at(0), depth + 1);
    if (const auto* constant = dynamic_cast<const v::ConstantOp*>(&op)) {
      if (constant->pointer() && constant->pointer()->addressing == "global") return constant->pointer()->identity;
    }
    if (const auto* input = dynamic_cast<const v::InputParameterOp*>(&op)) {
      if (input->pointer() && input->pointer()->addressing == "global") return input->pointer()->identity;
    }
    // cvta changes the address representation, not its originating object.
    // Retain the original DAG for offset reasoning: do not erase the conversion
    // or infer an object for an untyped integer / loaded pointer value.
    if (const auto* conversion = dynamic_cast<const v::PtxScalarOp*>(&op);
        conversion && conversion->opcode() == v::PtxScalarOpcode::AddressSpaceConvert &&
        !conversion->attributes().carry_in && !conversion->attributes().carry_out &&
        (conversion->attributes().modifiers == std::vector<std::string>{"to", "global", "u64"} ||
         conversion->attributes().modifiers == std::vector<std::string>{"global", "u64"})) {
      const auto operands = graph.instance(value).operands;
      if (operands.size() == 1 && operands[0].type(graph).bitSize() == 64)
        return Allocation(operands[0], depth + 1);
    }
    if (const auto* binary = dynamic_cast<const v::BinaryOp*>(&op);
        binary && (binary->opcode() == v::BinaryOpcode::Add || binary->opcode() == v::BinaryOpcode::Subtract)) {
      const auto operands = graph.instance(value).operands;
      const auto a = Allocation(operands[0],depth+1), b = Allocation(operands[1],depth+1);
      if (a && !b) return a;
      if (b && !a && binary->opcode() == v::BinaryOpcode::Add) return b;
    }
    return {};
  }
  std::size_t Node(v::Value value) {
    value = Normalize(value);
    if (!value.indices().empty() || value.threadSelector() || !value.type(graph).isScalar())
      throw std::invalid_argument("non-scalar address value");
    if (value.conversion() && value.conversion()->spec->kind == v::TypeConversionKind::Bitcast)
      return Node(value.conversion()->source);
    if (!value.conversion() && dynamic_cast<const v::BitcastOp*>(&graph.definingNode(value).operation()))
      return Node(graph.instance(value).operands.at(0));
    const Key key{value.nodeId(), value.instanceIndex(), value.resultIndex(), value.conversion().get()};
    if (const auto found = ids.find(key); found != ids.end()) return found->second;
    if (++slice_visits > 4096) throw std::invalid_argument("address slice exceeds serialization budget");
    retained_values.push_back(value);
    const auto& definition = graph.definingNode(value);
    const auto instance = graph.instance(value);
    Json node{{"op", std::string(definition.operation().name())}, {"type", value.type(graph).toString()},
              {"result", value.resultIndex()}, {"operands", Json::array()}, {"operand_types", Json::array()}};
    if (value.conversion()) {
      node["op"] = "TypeConversion";
      node["attributes"] = v::typeConversionLabel(value);
      node["operands"].push_back(Node(value.conversion()->source));
      node["operand_types"].push_back(value.conversion()->source.type(graph).toString());
    } else {
      std::ostringstream attributes;
      definition.operation().printAttributes(attributes);
      node["attributes"] = attributes.str();
      for (const auto& operand : instance.operands) {
        node["operands"].push_back(Node(operand));
        node["operand_types"].push_back(operand.type(graph).toString());
      }
      if (dynamic_cast<const v::CoordinateOp*>(&definition.operation())) {
        const auto* coordinate = dynamic_cast<const v::CoordinateOp*>(&definition.operation());
        const auto axis = static_cast<unsigned>(coordinate->axis());
        if (coordinate->kind() == v::CoordinateKind::ThreadIndex)
          node["binding"] = instance.origin.thread_coordinates->at(axis);
        else if (coordinate->kind() == v::CoordinateKind::BlockIndex)
          node["binding"] = instance.origin.cta_coordinates->at(axis);
        else throw std::invalid_argument("unbound cluster coordinate");
      }
      if (definition.operation().name() == "Load")
        node["occurrence"] = {instance.origin.source_file->str(), instance.origin.function->str(),
            instance.origin.cta_coordinates.value(), instance.origin.thread.value(),
            instance.origin.instruction_index.value(), instance.origin.occurrence.value(), instance.origin.lowering_index};
    }
    if (canonical_terms) {
      const auto id = canonical_ids.try_emplace(node.dump(), canonical_ids.size()).first->second;
      ids.emplace(key, id);
      return id;
    }
    const auto id = nodes.size();
    nodes.push_back(std::move(node)); ids.emplace(key, id);
    return id;
  }
  Json Write(v::Value value) {
    slice_visits = 0;
    const auto root = Node(value);
    if (canonical_terms)
      return {{"offset", {{"operator", "opaque"},
          {"provenance", Json::array({"canonical_value_dag", value.graphIdentity(), pool_id, root})}}}};
    Json result{{"offset", {{"operator", "value_dag"}, {"nodes", std::move(nodes)}, {"root", root}}}};
    if (const auto allocation = Allocation(value)) result["allocation"] = *allocation;
    return result;
  }
};
} // namespace

struct ValueAddressIndex::Impl {
  const v::Graph& graph;
  const CtaFacts& facts;
  using Key = std::tuple<std::uint32_t, std::uint64_t, bool>;
  std::map<Key, std::vector<std::size_t>> nodes;
  std::size_t queries = 0, resolved = 0;
  bool defer_serialization = false;
  using AddressKey = std::tuple<std::size_t, std::size_t, std::size_t>;
  struct AddressReference { v::Value value; std::size_t memory_node, instance; };
  std::map<AddressKey, std::vector<AddressReference>> address_references;
  std::string first_unavailable;
  Impl(const v::Graph& g, const CtaFacts& f, std::uint32_t cta, std::size_t begin)
      : graph(g), facts(f) {
    for (std::size_t id = begin; id < graph.size(); ++id) {
      const auto& node = graph.node(id);
      const bool load = dynamic_cast<const v::LoadOp*>(&node.operation());
      if (!load && !dynamic_cast<const v::StoreOp*>(&node.operation())) continue;
      if (node.instanceCount() == 0) continue;
      const auto& origin = node.origin();
      if (origin.cta != cta || !origin.instruction_index || !origin.occurrence) continue;
      nodes[{*origin.instruction_index, *origin.occurrence, load}].push_back(id);
    }
  }
};
ValueAddressIndex::ValueAddressIndex(const v::Graph& graph, const CtaFacts& facts,
    std::size_t first_node, bool defer_serialization)
    : impl_(std::make_unique<Impl>(graph, facts, facts.Linear(), first_node)) {
  impl_->defer_serialization = defer_serialization;
}
ValueAddressIndex::~ValueAddressIndex() = default;
std::size_t ValueAddressIndex::queries() const { return impl_->queries; }
std::size_t ValueAddressIndex::resolved() const { return impl_->resolved; }
std::string ValueAddressIndex::firstUnavailableReason() const { return impl_->first_unavailable; }
std::optional<nlohmann::json> ValueAddressIndex::Extract(LaneRef lane,
    const RecordView& record, std::string_view text) {
  const FactSource::ConsumerScope consumer(impl_->facts, {Consumer::AddressIndex, 0});
  ++impl_->queries;
  const auto unavailable = [&](std::string reason) -> std::optional<nlohmann::json> {
    if (impl_->first_unavailable.empty()) impl_->first_unavailable = std::move(reason);
    return {};
  };
  const auto& instruction = impl_->facts.function().instructions.at(record.instruction_id());
  const auto canonical = [](std::string_view text) {
    auto result = Trim(text);
    if (result.starts_with('[') && result.ends_with(']'))
      result = Trim(std::string_view(result).substr(1, result.size()-2));
    return result;
  };
  const auto operand = [&](std::size_t i) {
    return i < instruction.operands.size() &&
        canonical(impl_->facts.module().view(instruction.operands[i])) == canonical(text);
  };
  std::optional<bool> load;
  if ((instruction.opcode == "ld" || instruction.opcode == "ldu") && operand(1)) load = true;
  else if (instruction.opcode == "st" && operand(0)) load = false;
  else if (instruction.opcode == "cp" && !Has(instruction, "bulk")) {
    if (operand(0) == operand(1)) return unavailable("ambiguous copy address operand");
    if (operand(0)) load = false;
    if (operand(1)) load = true;
  }
  if (!load) return unavailable("memory address operand not indexed");
  const auto found = impl_->nodes.find({record.instruction_id(), record.occurrence(), *load});
  if (found == impl_->nodes.end()) return unavailable("memory occurrence has no value node");
  const auto thread = std::uint64_t(impl_->facts.Warp(lane)) * impl_->facts.launch().warp_size() +
      impl_->facts.LaneId(lane);
  std::optional<v::Value> address;
  std::size_t memory_node = 0, memory_instance = 0;
  std::uint64_t first = UINT64_MAX;
  for (auto id : found->second) {
    const auto& node = impl_->graph.node(id);
    const auto member = node.instanceThreads().indexOf(thread);
    if (!member) continue;
    const auto instance = node.instances().at(*member);
    if (instance.operands.empty() || !instance.origin.lowering_index) continue;
    if (*instance.origin.lowering_index < first) {
      first = *instance.origin.lowering_index;
      address = instance.operands[0];
      memory_node = id;
      memory_instance = *member;
    } else if (*instance.origin.lowering_index == first) return unavailable("ambiguous memory component");
  }
  if (!address) return unavailable("memory occurrence has no matching thread instance");
  try {
    if (impl_->defer_serialization) {
      AddressDagWriter writer(impl_->graph);
      auto normalized = writer.Normalize(*address);
      while (true) {
        if (normalized.conversion() && normalized.conversion()->spec->kind == v::TypeConversionKind::Bitcast)
          normalized = writer.Normalize(normalized.conversion()->source);
        else if (!normalized.conversion() && dynamic_cast<const v::BitcastOp*>(&impl_->graph.definingNode(normalized).operation()))
          normalized = writer.Normalize(impl_->graph.instance(normalized).operands.at(0));
        else break;
      }
      auto& references = impl_->address_references[{normalized.nodeId(), normalized.instanceIndex(), normalized.resultIndex()}];
      const auto existing = std::find_if(references.begin(), references.end(),
          [&](const auto& entry) { return entry.value == normalized; });
      if (existing != references.end()) {
        memory_node = existing->memory_node;
        memory_instance = existing->instance;
      } else references.push_back({normalized, memory_node, memory_instance});
      nlohmann::json result{{"offset", {{"operator", "value_dag_ref"},
          {"memory_node", memory_node}, {"instance", memory_instance}}}};
      if (auto allocation = writer.Allocation(*address))
        result["allocation"] = *allocation;
      ++impl_->resolved;
      return result;
    }
    auto result = AddressDagWriter(impl_->graph).Write(*address);
    ++impl_->resolved;
    return result;
  } catch (const std::invalid_argument& error) { return unavailable(error.what()); }
}

void MaterializeValueAddresses(std::vector<algorithm::ConstantAddressBinding>& bindings,
                               const v::Graph& graph, bool canonical_terms) {
  using Json = nlohmann::json;
  AddressDagWriter canonical(graph, true);
  const auto resolve = [&](auto&& self, Json& expression) -> void {
    if (expression.is_object() && expression.value("operator", "") == "value_dag_ref") {
      try {
        const auto& node = graph.node(expression.at("memory_node").get<std::size_t>());
        if (!dynamic_cast<const v::LoadOp*>(&node.operation()) &&
            !dynamic_cast<const v::StoreOp*>(&node.operation()))
          throw std::invalid_argument("address reference is not a memory operation");
        auto instance = node.instances().at(expression.at("instance").get<std::size_t>());
        expression = (canonical_terms ? canonical.Write(instance.operands.at(0))
                                      : AddressDagWriter(graph).Write(instance.operands.at(0))).at("offset");
      } catch (const std::invalid_argument& error) {
        expression = {{"operator", "unsupported_value_address"}, {"reason", error.what()}};
      }
      return;
    }
    if (expression.is_object() || expression.is_array())
      for (auto& child : expression) self(self, child);
  };
  for (auto& binding : bindings) {
    if (binding.expression.find("value_dag_ref") == std::string::npos) continue;
    auto expression = Json::parse(binding.expression);
    if (expression.contains("value_dag_reference")) {
      auto original = std::move(expression.at("value_dag_reference"));
      expression = std::move(original);
    }
    if (canonical_terms) {
      auto reference = expression;
      resolve(resolve, expression);
      expression["value_dag_reference"] = std::move(reference);
    } else resolve(resolve, expression);
    binding.expression = expression.dump();
  }
}

// The address-value extractor's value-IR CTA id, the literal 0. It reaches
// pointer owners and origins, hence the address DAG JSON in artifact.addresses,
// so it is neither the real CTA id nor read from a frame.
constexpr std::uint32_t kAddressExtractorCta = 0;
struct AddressValueExtractor::Impl {
  const FactSource& facts;
  const UnitFrame& frame;
  FunctionContext context;
  v::Graph graph;
  std::vector<ValueIrDiagnostic> diagnostics;
  std::map<LaneRef, std::unique_ptr<ValueTemplateExtractor>> lanes;
  struct Definitions {
    std::map<DynamicKey, std::size_t> positions;
    std::map<Key, std::vector<std::pair<std::size_t, DynamicKey>>> writes;
  };
  std::map<PathKey, Definitions> paths;
  // The program's lanes, a snapshot taken at construction.
  std::map<std::pair<std::uint32_t,std::uint32_t>,LaneRef> peers;
  Definition definition;
  unsigned active_queries = 0;
  ValueTemplateExtractor& Get(LaneRef);
  Impl(const FactSource& f, const UnitFrame& u)
      : facts(f), frame(u), context(f.module(), f.function()) {
    for (const auto lane : facts.ProgramLanes()) peers.emplace(std::pair{facts.Warp(lane),facts.LaneId(lane)},lane);
  }
};
AddressValueExtractor::AddressValueExtractor(const FactSource& f, const UnitFrame& frame)
    : impl_(std::make_unique<Impl>(f,frame)) {}
AddressValueExtractor::~AddressValueExtractor() = default;
ValueTemplateExtractor& AddressValueExtractor::Impl::Get(LaneRef lane) {
  auto& extractor = lanes[lane];
  if (!extractor) {
    auto live = std::make_shared<std::vector<std::uint8_t>>(1, 1);
    extractor = std::make_unique<ValueTemplateExtractor>(graph, diagnostics, facts, lane,
        kAddressExtractorCta, frame, context, false, live);
    extractor->demand_mode = true;
    auto [path, inserted] = paths.try_emplace(facts.Path(lane));
    if (inserted) {
      const auto add = [&](const std::string& name, Scope scope, std::size_t index, DynamicKey key) {
        while (!scope.empty() && !context.declarations.contains({scope,name})) scope.pop_back();
        path->second.writes[{scope,name}].emplace_back(index,key);
      };
      for (std::size_t index = 0; index < facts.Size(lane); ++index) {
        const auto event = facts.Meta({lane, index});
        path->second.positions.emplace(event.Key(), index);
        if (event.predicate_value() == false) continue;
        const auto& instruction = facts.function().instructions.at(event.instruction_id());
        for (const auto& name : instruction.destinations) add(name,instruction.scope,index,event.Key());
        for (const auto& name : event.opaque_compute_register_writes()) add(name,instruction.scope,index,event.Key());
        for (const auto skipped : event.opaque_compute_diamond_indices()) {
          const auto& omitted = facts.function().instructions.at(skipped);
          for (const auto& name : omitted.destinations) add(name,omitted.scope,index,event.Key());
        }
      }
    }
    extractor->demand_producer = [definitions=&path->second](const RecordView& event, const Key& key)
        -> std::optional<DynamicKey> {
      const auto position = definitions->positions.find(event.Key());
      const auto writes = definitions->writes.find(key);
      if (position == definitions->positions.end() || writes == definitions->writes.end()) return {};
      const auto& entries = writes->second;
      const auto found = std::lower_bound(entries.begin(), entries.end(), position->second,
          [](const auto& entry, std::size_t position) { return entry.first < position; });
      return found == entries.begin() ? std::nullopt : std::optional{std::prev(found)->second};
    };
    extractor->demand_peer = [this,warp=facts.Warp(lane),token=facts.PeerPathToken(lane)](
        std::uint32_t source, DynamicKey key, const std::string& name)
        -> std::optional<Expression> {
      const auto peer = peers.find({warp,source});
      if (peer == peers.end() || active_queries >= 128) return {};
      // A shared recorded path establishes occurrence alignment. Divergent
      // peer paths need an explicit collective-occurrence mapping.
      if (!facts.LegacyPeerAligned(token, peer->second)) return {};
      const auto record = definition(peer->second,key);
      if (!record || record->predicate_value() == false) return {};
      ++active_queries;
      auto result = Get(peer->second).Demand(*record,"[" + name + "]");
      --active_queries;
      return result;
    };
  }
  extractor->demand_definition = definition;
  return *extractor;
}
std::optional<nlohmann::json> AddressValueExtractor::Extract(LaneRef lane,
    const RecordView& record, std::string_view text, const Definition& definition) {
  impl_->definition = definition;
  auto expression = impl_->Get(lane).Demand(record, std::string(text));
  impl_->diagnostics.clear();
  if (!expression) return {};
  try { return AddressDagWriter(impl_->graph).Write(expression->value); }
  catch (const std::exception&) { return {}; }
}

void ValueIrExtractor::AddCta(const CtaFacts &facts) {
  const FactSource::ConsumerScope consumer(facts, {Consumer::ValueView, 0});
  const auto cta = facts.Linear();
  if (stop_on_unsupported_ && !diagnostics_.empty()) return;
  if (!facts.Specialized()) {
    // The message names the storage type; its text is part of the output.
    diagnostics_.push_back({"invalid_execution_program", "ExecutionProgram has no PTX module/function", {}});
    return;
  }
  if (facts.LaneCount() == 0) return;
  std::map<ValueTemplateExtractor::ShuffleKey, Expression> shuffle;
  std::map<ValueTemplateExtractor::ShuffleKey, ValueTemplateExtractor::MatrixAddressState> matrix_addresses;
  std::map<ValueTemplateExtractor::ShuffleKey,ValueTemplateExtractor::WarpMmaState> warp_mma_inputs;
  const auto &function = facts.function();
  if (partition_specs_.function!=&function) partition_specs_={&function,{}};
  const auto &launch = facts.launch();
  const FunctionContext context(facts.module(),function);
  AddressTable addresses;
  BarrierTable barriers;
  Resources resources(facts,facts.frame(),addresses,barriers,SlotConfig{Consumer::ValueView,{}});
  std::optional<TmemLifetimeAnalysis> scale_provenance;
  if(std::any_of(function.instructions.begin(),function.instructions.end(),
      [](const auto& i){return i.opcode=="tcgen05" && Has(i,"block_scale");})) {
    TmemAnalysisOptions options;options.allow_cta_group2=true;
    scale_provenance=AnalyzeTmemLifetimes(facts,ExpansionSite::BlockScaleTmem,options);
  }
  const auto origins=[&](const std::vector<LaneRef>& members) {
    std::vector<v::SourceOrigin> result;
    for (const auto lane:members) {
      v::SourceOrigin origin;
      origin.cta=cta;
      SetThread(origin,facts,lane);
      result.push_back(std::move(origin));
    }
    return result;
  };
  struct Cohort {
    std::unique_ptr<ValueTemplateExtractor> extractor;
    v::ThreadTemplate state;
  };
  std::vector<Cohort> cohorts;
  std::map<PathKey,std::shared_ptr<std::vector<std::uint8_t>>> liveness;
  const auto template_groups=facts.TemplateGroups();
  std::vector<std::vector<std::size_t>> groups(template_groups.begin(),template_groups.end());
  if (groups.empty()) {
    for (const auto& cohort:facts.LanePathCohorts()) {
      auto& members=groups.emplace_back();
      members.reserve(cohort.size());
      for (const auto lane:cohort) members.push_back(lane.index);
    }
  }
  std::vector<bool> seen(facts.LaneCount());
  for (const auto& group:groups) {
    if (group.empty()) throw std::invalid_argument("empty frontend thread template");
    // Initial register values are part of a template's input environment.
    std::map<std::string,std::vector<LaneRef>> environments;
    for (auto index:group) {
      if (index>=seen.size() || seen[index]) throw std::invalid_argument("invalid frontend template partition");
      seen[index]=true;
      const auto lane=facts.Lane(index);
      std::ostringstream key;
      for (const auto& warp:launch.warps()) if (warp.warp_id==facts.Warp(lane))
        for (const auto& [name,value]:warp.initial_registers) key << name << '=' << value << ';';
      environments[key.str()].push_back(lane);
    }
    for (auto& [key,members]:environments) {
      const auto lane=members.front();
      auto& live=liveness[facts.Path(lane)];
      if (!live) live=std::make_shared<std::vector<std::uint8_t>>();
      auto extractor=std::make_unique<ValueTemplateExtractor>(graph_,diagnostics_,facts,lane,cta,facts.frame(),context,prune_unused_compute_,live,branch_capture_);
      extractor->cohort_template=true;
      extractor->members=members;
      extractor->shuffle_values=&shuffle;
      extractor->matrix_addresses=&matrix_addresses;
      extractor->warp_mma_inputs=&warp_mma_inputs;
      extractor->partition_specs=&partition_specs_;
      // TMA descriptors come from the executor's resolution, as in lowering: the
      // program's maps and those it published at run time (one version per location).
      extractor->resources=&resources;
      if(scale_provenance)extractor->tmem_provenance=&scale_provenance->provenance;
      cohorts.push_back({std::move(extractor),graph_.makeThreadTemplate(origins(members))});
    }
  }
  if (!std::all_of(seen.begin(),seen.end(),[](bool value){return value;}))
    throw std::invalid_argument("frontend template partition omits threads");
  std::uint64_t steps=0,splits=0;
  for (;;) {
    bool pending=false,progress=false;
    for (std::size_t index=0;index<cohorts.size();++index) {
      auto& cohort=cohorts[index];
      if (cohort.extractor->Done()) continue;
      pending=true;
      auto parts=cohort.extractor->Partition();
      if (parts.size()>1) {
        const auto parent=cohort.state.cursor;
        auto original=std::move(cohort.extractor);
        std::vector<Cohort> children;
        for (auto& members:parts)
          children.push_back({original->Fork(members),graph_.makeThreadTemplate(origins(members),parent)});
        cohorts[index]=std::move(children.front());
        for (std::size_t child=1;child<children.size();++child) cohorts.push_back(std::move(children[child]));
        ++splits;
        progress=true;
        continue;
      }
      auto& current=cohorts[index];
      graph_.activateThreadTemplate(current.state);
      const auto publications=current.extractor->publications;
      const auto diagnostic_begin=diagnostics_.size();
      try {
        progress |= current.extractor->Step();
      } catch (...) {
        graph_.suspendThreadTemplate(current.state);
        throw;
      }
      ++steps;
      progress |= current.extractor->publications!=publications;
      graph_.suspendThreadTemplate(current.state);
      if (!stop_on_unsupported_) {
        const auto diagnostic_end=diagnostics_.size();
        for (std::size_t k=diagnostic_begin;k<diagnostic_end;++k)
          for (std::size_t member=1;member<current.state.threads.size();++member) {
            auto diagnostic=diagnostics_[k];
            const auto& binding=current.state.threads[member];
            diagnostic.origin.thread=binding.thread;
            diagnostic.origin.warp=binding.warp; diagnostic.origin.lane=binding.lane;
            diagnostic.origin.thread_coordinates=binding.thread_coordinates;
            diagnostics_.push_back(std::move(diagnostic));
          }
      }
      if (stop_on_unsupported_ && !diagnostics_.empty()) return;
    }
    if (!pending) break;
    if (!progress) for (auto& cohort:cohorts) if (!cohort.extractor->Done()) {
      graph_.activateThreadTemplate(cohort.state);
      cohort.extractor->Abandon();
      graph_.suspendThreadTemplate(cohort.state);
      if (stop_on_unsupported_ && !diagnostics_.empty()) return;
    }
  }
  if (algorithm::MemoryProfileEnabled()) {
    algorithm::MemoryReport report("frontend.values.templates","value_graph",&graph_);
    report.Metric("cta",cta); report.Metric("lowering_steps",steps);
    report.Metric("splits",splits); report.Metric("cohorts",cohorts.size());
    report.Metric("nodes",graph_.size()); report.Emit();
  }
  for (const auto& cohort:cohorts) graph_.finishThreadTemplate(cohort.state);
  graph_.sealCta(cta);
}
} // namespace sentryward::frontend::ptx
