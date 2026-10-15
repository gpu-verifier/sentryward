#include "sentryward/frontend/ptx/readonly_origin.h"
#include "sentryward/frontend/ptx/address_expression.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/ptx_operation.h"
#include "sentryward/value_ir/type_conversion.h"
#include <algorithm>
#include <set>
#include <unordered_map>

namespace sentryward::frontend::ptx {
namespace {
namespace v = value_ir;
constexpr std::size_t entries = 4096, workLimit = 131072, patternLimit = 4096;
bool plain(const v::ArithmeticAttributes& a) {
  return a.rounding == v::RoundingMode::None && a.multiply_mode == v::IntegerMultiplyMode::Low &&
      !a.saturate && !a.flush_to_zero && !a.propagate_nan;
}
bool ordinary(const v::MemoryAttributes& a) {
  return !a.layout && !a.descriptor && a.interpretation == v::MemoryAttributes::Interpretation::Ordinary &&
      a.native_modifiers.empty() && !a.guarded;
}
bool global(const v::PointerProperties& p) {
  return p.addressing == "global" && !p.owner_cta && !p.owner_thread && p.identity.size() <= 4096;
}
bool integer(const v::Graph& g, v::Value x) {
  const auto& t = g.type(x);
  return t.isScalar() && t.element().isBitLike() && t.element().kind() != v::ElementKind::Predicate &&
      !t.element().isParameterized() && t.element().width() <= 64;
}
struct ValueHash {
  std::size_t operator()(const v::Value& x) const {
    // The admission path denies selection and conversions beyond exact
    // Bitcast. Full Value equality resolves any collision/annotation mismatch.
    auto h = std::size_t(x.graphIdentity());
    const auto mix = [&](std::size_t n) { h ^= n + 0x9e3779b9 + (h << 6) + (h >> 2); };
    mix(x.nodeId()); mix(x.instanceIndex()); mix(x.resultIndex());
    if (x.conversion()) {
      const auto& c = *x.conversion();
      mix(operator()(c.source)); mix(std::size_t(c.spec->kind));
      mix(std::size_t(c.spec->target.element().kind())); mix(c.spec->target.element().width());
      mix(std::size_t(c.spec->attributes.rounding)); mix(c.spec->attributes.saturate);
      mix(c.spec->attributes.flush_to_zero); mix(c.spec->attributes.saturate_finite);
    }
    if (x.pointer()) {
      mix(std::hash<std::string>{}(x.pointer()->identity));
      mix(std::hash<std::string>{}(x.pointer()->addressing));
      mix(x.pointer()->owner_cta.value_or(UINT64_MAX)); mix(x.pointer()->owner_thread.value_or(UINT64_MAX));
    }
    return h;
  }
};
struct Origin {
  enum class Kind { Unknown, Scalar, Object } kind = Kind::Unknown;
  std::string object;
};
} // namespace

struct AddressOriginAnalysis::Impl {
  const v::Graph& graph;
  Resolve resolve;
  std::unordered_map<v::Value, Origin, ValueHash> memo;
  std::size_t work = 0;
  bool exhausted = false;
  explicit Impl(const v::Graph& g, Resolve r) : graph(g), resolve(std::move(r)) {}
  bool charge(std::size_t n = 1) {
    if (exhausted || n > workLimit - work) { exhausted = true; return false; }
    work += n; return true;
  }
  Origin inspect(v::Value value, unsigned depth = 0) {
    if (resolve) value = resolve(value);
    if (!graph.contains(value) || depth >= 128 || !value.indices().empty() || value.threadSelector() ||
        !integer(graph, value)) return {};
    if (value.pointer() && !global(*value.pointer())) return {};
    auto edge = value;
    for (unsigned n = 0; edge.conversion(); ++n) {
      if (n == 128 || edge.conversion()->spec->kind != v::TypeConversionKind::Bitcast ||
          edge.conversion()->spec->attributes != v::ConversionAttributes{} ||
          !edge.conversion()->source.indices().empty() || edge.conversion()->source.threadSelector() ||
          (edge.conversion()->source.pointer() && !global(*edge.conversion()->source.pointer())) ||
          !integer(graph, edge.conversion()->source) ||
          graph.type(edge.conversion()->source).bitSize() != graph.type(edge).bitSize()) return {};
      edge = edge.conversion()->source;
    }
    value = NormalizeAddressExpression(graph, value, resolve);
    if (!graph.contains(value) || !value.indices().empty() || value.threadSelector() || !integer(graph, value)) return {};
    if (const auto found = memo.find(value); found != memo.end()) {
      if (!charge(found->second.object.size())) return {};
      return found->second;
    }
    if (memo.size() >= entries || !charge()) { exhausted = true; return {}; }
    Origin out;
    if (value.conversion()) {
      const auto& c = *value.conversion();
      if (c.spec->kind == v::TypeConversionKind::Bitcast && c.spec->attributes == v::ConversionAttributes{} &&
          graph.type(c.source).bitSize() == graph.type(value).bitSize()) out = inspect(c.source, depth + 1);
    } else {
      const auto& op = graph.definingNode(value).operation();
      if (const auto* p = dynamic_cast<const v::InputParameterOp*>(&op)) {
        if (!p->pointer()) out.kind = Origin::Kind::Scalar;
        else if (global(*p->pointer()) && !p->pointer()->identity.empty() && charge(p->pointer()->identity.size()))
          out = {Origin::Kind::Object, p->pointer()->identity};
      } else if (const auto* p = dynamic_cast<const v::ConstantOp*>(&op)) {
        if (!p->pointer()) out.kind = Origin::Kind::Scalar;
        else if (global(*p->pointer()) && !p->pointer()->identity.empty() && charge(p->pointer()->identity.size()))
          out = {Origin::Kind::Object, p->pointer()->identity};
      } else if (dynamic_cast<const v::CoordinateOp*>(&op) || dynamic_cast<const v::LoadOp*>(&op)) {
        // A typed integer data value is an offset, not a new allocation
        // provenance. A load used AS the whole address still has no Object.
        out.kind = Origin::Kind::Scalar;
      } else {
        // Deny unsupported producers before copying an instance record. The
        // operation's stored operand header is a borrowed O(1) arity query.
        const auto arity = graph.definingNode(value).operands().size();
        if (arity > 2 || !(dynamic_cast<const v::BitcastOp*>(&op) ||
            dynamic_cast<const v::PtxScalarOp*>(&op) || dynamic_cast<const v::BinaryOp*>(&op) ||
            dynamic_cast<const v::UnaryOp*>(&op) || dynamic_cast<const v::ConvertOp*>(&op))) return {};
        if (!charge(arity)) return {};
        const auto args = graph.instance(value).operands;
        if (dynamic_cast<const v::BitcastOp*>(&op) && args.size() == 1 &&
            graph.type(args[0]).bitSize() == graph.type(value).bitSize()) out = inspect(args[0], depth + 1);
        else if (const auto* p = dynamic_cast<const v::PtxScalarOp*>(&op);
            p && p->opcode() == v::PtxScalarOpcode::AddressSpaceConvert && args.size() == 1 &&
            !p->attributes().carry_in && !p->attributes().carry_out &&
            p->attributes().modifiers == std::vector<std::string>{"to", "global", "u64"} &&
            graph.type(value).bitSize() == 64 && graph.type(args[0]).bitSize() == 64) {
          auto a = inspect(args[0], depth + 1);
          if (a.kind == Origin::Kind::Object) out = std::move(a);
        } else if (const auto* p = dynamic_cast<const v::BinaryOp*>(&op); p && args.size() == 2) {
          const auto a = inspect(args[0], depth + 1), b = inspect(args[1], depth + 1);
          if (a.kind == Origin::Kind::Scalar && b.kind == Origin::Kind::Scalar) out.kind = Origin::Kind::Scalar;
          else if (plain(p->attributes()) && graph.type(value).bitSize() == 64 &&
              graph.type(args[0]).bitSize() == 64 && graph.type(args[1]).bitSize() == 64) {
            if ((p->opcode() == v::BinaryOpcode::Add || p->opcode() == v::BinaryOpcode::Subtract) &&
                a.kind == Origin::Kind::Object && b.kind == Origin::Kind::Scalar) out = a;
            else if (p->opcode() == v::BinaryOpcode::Add && a.kind == Origin::Kind::Scalar &&
                b.kind == Origin::Kind::Object) out = b;
          }
        } else if (dynamic_cast<const v::UnaryOp*>(&op) && args.size() == 1 &&
            inspect(args[0], depth + 1).kind == Origin::Kind::Scalar) out.kind = Origin::Kind::Scalar;
        else if (dynamic_cast<const v::ConvertOp*>(&op) && args.size() == 1 &&
            inspect(args[0], depth + 1).kind == Origin::Kind::Scalar) out.kind = Origin::Kind::Scalar;
      }
    }
    if (!charge(out.object.size())) return {};
    memo.emplace(value, out); return out;
  }
};
AddressOriginAnalysis::AddressOriginAnalysis(const v::Graph& g, Resolve r)
    : impl_(std::make_unique<Impl>(g, std::move(r))) {}
AddressOriginAnalysis::~AddressOriginAnalysis() = default;
std::optional<std::string> AddressOriginAnalysis::globalObject(v::Value value) {
  const auto o = impl_->inspect(value);
  return o.kind == Origin::Kind::Object && impl_->charge(o.object.size()) ? std::optional(o.object) : std::nullopt;
}
bool AddressOriginAnalysis::chargeWork(std::size_t n) { return impl_->charge(n); }
std::size_t AddressOriginAnalysis::work() const { return impl_->work; }
bool AddressOriginAnalysis::exhausted() const { return impl_->exhausted; }

struct ReadonlyOriginCensus::Impl {
  const v::Graph& graph;
  AddressOriginAnalysis origins;
  std::set<std::string> objects, written;
  std::map<std::string,std::string> resource_by_object;
  bool complete = true;
  std::size_t patterns = 0, originalSize = 0;
  Impl(const v::Graph& g, const FrontendConfig& c) : graph(g), origins(g), originalSize(g.size()) {
    const auto contract = c.fixed_input.value("kernel_pointer_contract", nlohmann::json::object());
    if (contract.value("aliasing", "") != "distinct_objects_noalias" ||
        c.fixed_input.value("launch_contract", nlohmann::json::object()).value("external_grids", "") != "no_overlap" ||
        c.analysis_thread_range || g.size() > 65536 || c.resources.size() > 4096 || c.pointer_parameters.size() > 4096)
      { complete = false; return; }
    std::map<std::string, const nlohmann::json*> resource_index;
    for (const auto& r : c.resources) {
      const auto id = r.value("id", "");
      if (id.empty() || id.size() > 4096 || !origins.chargeWork(1 + id.size()) ||
          !resource_index.emplace(id, &r).second) { complete = false; return; }
    }
    std::set<std::string> allocation_names;
    for (const auto& p : c.pointer_parameters) {
      if (p.resource.empty() || p.resource.size() > 4096 || p.width_bytes != 8 ||
          !origins.chargeWork(1 + p.resource.size())) { complete = false; return; }
      const auto resource = resource_index.find(p.resource);
      if (resource == resource_index.end() ||
          resource->second->value("space", "") != "global" ||
          !resource->second->contains("allocation") || resource->second->at("allocation").value("resolution", "") != "concrete")
        { complete = false; return; }
      const auto& allocation = resource->second->at("allocation");
      const auto found = allocation.find("object");
      if (found == allocation.end() || !found->is_string()) { complete = false; return; }
      const auto& object = found->get_ref<const std::string&>();
      if (object.empty() || object.size() > 4096 || !origins.chargeWork(1 + object.size()) ||
          !allocation_names.insert(object).second || !objects.insert(p.resource).second ||
          allocation.value("base_byte_offset", nlohmann::json(0)) != nlohmann::json(0))
        { complete = false; return; }
      resource_by_object.emplace(object,p.resource);
    }
    for (const auto& node : g.nodes()) {
      if (!origins.chargeWork(1)) { complete = false; break; }
      const auto* store = dynamic_cast<const v::StoreOp*>(&node.operation());
      if (!store) continue;
      if (!ordinary(store->attributes())) { complete = false; return; }
      // forEachInstancePattern has <=2*S+1 callbacks for S stored records.
      // Reserve before entering it; a denial must not scan an uncharged tail.
      const auto stored = node.storedInstanceCount();
      if (patterns >= patternLimit || stored > (patternLimit - patterns - 1) / 2 ||
          !origins.chargeWork(2 * stored + 1)) { complete = false; return; }
      node.forEachInstancePattern([&](const auto& instance, std::size_t, std::size_t, bool) {
        if (++patterns > patternLimit || instance.operands.size() != 2) { complete = false; return; }
        const auto address = instance.operands[0];
        if (!address.pointer()) { complete = false; return; }
        const auto& p = *address.pointer();
        if ((p.addressing == "shared" || p.addressing == "local" || p.addressing == "tmem") &&
            p.owner_cta) return; // Distinct actual private memory namespace.
        if (p.addressing != "global") { complete = false; return; }
        const auto object = origins.globalObject(address);
        if (!object || !resource_by_object.contains(*object)) { complete = false; return; }
        if (!origins.chargeWork(object->size())) { complete = false; return; }
        written.insert(resource_by_object.at(*object));
      });
      if (!complete) return;
    }
    if (origins.exhausted()) complete = false;
  }
};
ReadonlyOriginCensus::ReadonlyOriginCensus(const v::Graph& g, const FrontendConfig& c)
    : impl_(std::make_unique<Impl>(g, c)) {}
ReadonlyOriginCensus::~ReadonlyOriginCensus() = default;
bool ReadonlyOriginCensus::complete() const { return impl_->complete && impl_->graph.size() == impl_->originalSize; }
bool ReadonlyOriginCensus::readonly(const std::string& object) const {
  return complete() && impl_->objects.contains(object) && !impl_->written.contains(object);
}
std::optional<std::string> ReadonlyOriginCensus::initialLoadObject(v::Value value) {
  if (!complete() || !impl_->graph.contains(value) || value.conversion() || value.pointer() ||
      value.threadSelector() || !value.indices().empty()) return {};
  const auto* load = dynamic_cast<const v::LoadOp*>(&impl_->graph.definingNode(value).operation());
  if (!load || !ordinary(load->attributes()) || !impl_->graph.type(value).isScalar() ||
      impl_->graph.definingNode(value).operands().size() != 1 || !impl_->origins.chargeWork(1)) return {};
  const auto args = impl_->graph.instance(value).operands;
  if (args.size() != 1 || !args[0].pointer() || args[0].pointer()->addressing != "global") return {};
  const auto object = impl_->origins.globalObject(args[0]);
  if (!object) return {};
  const auto resource = impl_->resource_by_object.find(*object);
  return resource != impl_->resource_by_object.end() && readonly(resource->second)
      ? std::optional(resource->second) : std::nullopt;
}
std::size_t ReadonlyOriginCensus::storedWritePatterns() const { return impl_->patterns; }
} // namespace sentryward::frontend::ptx
