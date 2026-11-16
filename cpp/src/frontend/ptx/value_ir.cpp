#include "value_ir_impl.hpp"

namespace sentryward::frontend::ptx {
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
