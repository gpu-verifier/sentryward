#pragma once
#include "run_support.hpp"

int RunAttempt(int argc, char **argv, CallInputs &inputs, const std::optional<PerUnitRestart> &unit_restart,
               const std::optional<TableAtomRestart> &atom_restart,
               const std::optional<FounderOrderRestart> &order_restart,
               const std::optional<FounderPlanRestart> &plan_restart) {
  bool parsing_arguments = true;
  try {
    auto options = Arguments(argc, argv);
    parsing_arguments = false;
    alg::AnalysisScope attempt_scope;
    Json restart_record;
    if (unit_restart)
      restart_record = {{"reason", unit_restart->reason},
                        {"kept_synchronization_verdict", !unit_restart->verdict.empty()}};
    Monitor monitor(options, unit_restart ? std::optional(unit_restart->started)
                             : atom_restart ? std::optional(atom_restart->started)
                             : order_restart ? std::optional(order_restart->started)
                             : plan_restart ? std::optional(plan_restart->started) : std::nullopt,
                    std::move(restart_record));
    // A per-unit attempt that keeps the class-mode synchronization verdict reports it
    // from the start, also when the value side stops early.
    const bool kept_verdict = unit_restart && !unit_restart->verdict.empty();
    if (kept_verdict) monitor.MemoryVerified(unit_restart->verdict, unit_restart->verdict_elapsed_s);
    std::optional<rg::Document> reference;
    std::optional<sentryward::scalar_fold::Artifact> expression;
    // A required equivalence check without a reference expression is missing input.
    Json equivalence_mapping, reference_error={{"status","missing_input"},{"reason","reference_expression_not_supplied"}};
    // The bytes the binding compares: the scalar fold projects only those.
    std::map<std::string, sentryward::scalar_fold::ComparedOutput> compared_outputs;
    if(!options.reference_failure_json.empty()) {
      reference_error=Json::parse(options.reference_failure_json);
      if(!reference_error.is_object() || !reference_error.contains("result") ||
          !sentryward::results::IsResult(reference_error.value("result","")) ||
          reference_error.value("result","")=="verified" || reference_error.value("result","")=="race_free")
        throw std::invalid_argument("invalid reference failure report");
    }
    if (options.reference_stdin) {
      monitor.Stage("equivalence.reference_import");
      try {
        if (!std::isfinite(options.equivalence_timeout) || options.equivalence_timeout <= 0)
          throw std::invalid_argument("invalid equivalence timeout");
        std::istringstream in(inputs.ReferenceStdin());
        reference = rg::Document::load(in);
        const auto *mapping = inputs.File(options.equivalence_mapping);
        if (!mapping) throw std::invalid_argument("missing equivalence mapping");
        std::istringstream mapping_in(*mapping);
        mapping_in >> equivalence_mapping;
        compared_outputs = rg::comparedOutputs(equivalence_mapping);
      } catch (const std::exception& error) {
        reference.reset();
        compared_outputs.clear();
        reference_error = {{"status","unsupported"},{"reason","reference_import_failed"},{"message",error.what()}};
      }
    }
    alg::ProfileSession profiling(true);
    alg::ProfileScope total("pipeline.total");
    auto poll = [] { alg::ProfileSnapshot(); };
    Json metadata, input_metadata = Json::object();
    // Normalized tensor_properties records: trusted premises of the equivalence check too.
    Json tensor_property_inputs = Json::array();
    std::uint64_t records = 0;
    ptx::ThreadRecordStatistics record_compression;
    ptx::TemplateExecutionStatistics template_execution;
    std::uint64_t template_ctas=0;
    std::uint32_t ctas = 0;
    Json result;
    std::unique_ptr<sentryward::value_ir::Graph> values;
    std::unique_ptr<ptx::ValueIrExtractor> value_extractor;
    std::unique_ptr<ptx::BranchScopeCapture> branch_capture;
    std::unique_ptr<ptx::SemanticProgramBuilder> semantic_builder;
    std::optional<ptx::SemanticProgram> semantic_program;
    std::optional<sentryward::value_analysis::MemoryEliminationResult> eliminated;
    Json value_analysis = {{"status", "not_run"}, {"reason", "verification_not_completed"}};
    Json serialization_statistics;
    Json cache_provenance;
    bool replayed_correctness=false;
    // Cross-CTA class mode: the quotient tables; an attempt that cannot finish
    // exactly restarts per unit (PerUnitRestart). Value lifting (class
    // mode with values): the class tables. Whether units share their analysis
    // does not change the results; analysis statistics (program counts such as
    // threads, dependency-compression statistics) count only the units analysed.
    std::unique_ptr<alg::UnitQuotient> quotient;
    std::optional<sentryward::value_analysis::UnitLifting> unit_lifting;
    // Class mode with value lifting: the verified report of the class-mode synchronization
    // check and when it was reached, when a value_lifting restart may keep it (KeepsVerdict).
    std::string class_verdict;
    double class_verdict_elapsed_s = 0;
    // A piecewise class-mode quotient's value side is refused (run_values), so its value view is
    // released before the synchronization check, its addresses materialized first and these
    // statistics kept for the refusal's budget and the report.
    struct ReleasedValues {
      std::uint64_t nodes = 0, instances = 0, groups = 0;
    };
    std::optional<ReleasedValues> released_values;
    // The class loop took planned founders first (ptx::ContainmentPlan).
    bool founders_first = false;
    const auto restart_per_unit = [&](const alg::UnitClassRestart &restart) {
      if (unit_restart)
        throw std::logic_error("internal error: CTA class bookkeeping (restart requested by the run without classes)");
      // The class loop asked for founder planning (UnitClasses::PlanUseful).
      if (restart.reason == "founder_plan") {
        if (plan_restart || order_restart || founders_first)
          throw std::logic_error("internal error: CTA class bookkeeping (a second founder plan)");
        throw FounderPlanRestart{monitor.start};
      }
      // An attempt that took planned founders first is redone in unit order before anything else.
      if (founders_first) throw FounderOrderRestart{monitor.start};
      if (restart.reason == "table_atoms") {
        if (atom_restart) throw std::logic_error("internal error: CTA class bookkeeping (a second table-atom attempt)");
        throw TableAtomRestart{monitor.start};
      }
      PerUnitRestart next{restart.reason, monitor.start};
      if (restart.reason == "value_lifting") {
        next.verdict = class_verdict;
        next.verdict_elapsed_s = class_verdict_elapsed_s;
      }
      throw next;
    };
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/dot_options.inc"
#endif
    if ((!options.dump_value_ir.empty() || !options.dump_branch_scope.empty()) && !options.analyze_values) {
      values = std::make_unique<sentryward::value_ir::Graph>();
      if (!options.dump_branch_scope.empty()) branch_capture = std::make_unique<ptx::BranchScopeCapture>(*values);
      value_extractor = std::make_unique<ptx::ValueIrExtractor>(*values, false, false, branch_capture.get());
    }
    auto effective_assumptions = alg::DefaultPtxAssumptions();
    std::function<void(ptx::CheckedMemoryActions)> run_values;
    // Value lifting: a value stage of the class-mode quotient whose result
    // cannot stand for the members refuses lifting (RefuseLifting); the launch
    // re-runs per unit before the later (more expensive) stages, or, above the
    // re-run's budget, the value analysis ends with the budget refusal.
    const auto refuse_lifting = [&](const std::optional<LiftingStop> &stop = std::nullopt) {
      value_analysis = RefuseLifting(released_values ? released_values->instances : ValueInstances(semantic_program->values),
                                     *unit_lifting, "value analysis", stop);
      value_analysis["reason"] = value_analysis["diagnostic"]["code"];
      value_analysis["message"] = value_analysis["diagnostic"]["message"];
      value_analysis["stage"] = "value_lifting";
    };
    const auto class_lifting = [&] { return unit_lifting.has_value(); };
    run_values = [&](ptx::CheckedMemoryActions actions) {
      monitor.Stage("value_analysis.prepare");
      if (!semantic_program->valuesComplete()) {
        const auto& diagnostic = semantic_program->value_diagnostics.front();
        value_analysis = {{"status", "unsupported"}, {"reason", diagnostic.code},
            {"message", diagnostic.message}, {"stage", "value_lowering"},
            {"line", diagnostic.origin.source_line.value_or(0)}};
        return;
      }
      // A member whose exact addresses are not its representative's moved by one δ per allocation
      // (admitted on the table-lookup forms of its addresses) cannot have its values lifted; later
      // classes may then have admitted a piecewise δ, and later founders built no value view.
      if (class_lifting() && unit_lifting->valued_unliftable)
        return refuse_lifting(LiftingStop{"table_lookup_lifting",
            "a class member reads or writes, through fixed shape-array table elements, addresses that are not "
            "its representative's moved by one offset per allocation"});
      // Value lifting moves a member's cells, slot constants and output regions by one δ per
      // allocation; a class with a piecewise δ (UnitQuotient::PiecewiseDelta) cannot be lifted.
      // The value side is then redone per unit, or ends with the re-run's budget refusal
      // (RefuseLifting); the class-mode synchronization verdict stands either way (KeepsVerdict).
      if (class_lifting() && quotient->Piecewise())
        return refuse_lifting(LiftingStop{"piecewise_unit_delta",
            "A class's members move some global allocation by different amounts at different slot "
            "sites (a piecewise δ), which value lifting does not represent."});
      // A containment member skips some of its representative's plain global accesses and compute:
      // its values are not its representative's moved by δ.
      if (class_lifting() && quotient->Contained())
        return refuse_lifting(LiftingStop{"containment_member",
            "A class member executes only some of its representative's plain global accesses and "
            "computations (a containment member), so its values are not the representative's moved "
            "by one offset per allocation."});
      if (!actions.complete()) {
        const auto& diagnostic = actions.diagnostics.front();
        // A binding diagnostic (it names its occurrence, or is the domain
        // link's stop) heading a list in CTA order is the per-unit run's
        // first; any other first diagnostic of the quotient need not be.
        if (class_lifting() && !((diagnostic.occurrence || diagnostic.first_group) && quotient &&
                                 BindingEvidence(*semantic_program, *quotient).launch_order))
          return refuse_lifting(LiftingStop{diagnostic.code, diagnostic.message});
        value_analysis = {{"status", "unsupported"}, {"reason", diagnostic.code},
            {"message", diagnostic.message}, {"stage", "memory_effect_binding"}};
        if (diagnostic.occurrence) value_analysis["instruction_index"] = diagnostic.occurrence->instruction;
        if (diagnostic.value) {
          const auto& origin = semantic_program->values.node(diagnostic.value->node).instances().at(diagnostic.value->instance).origin;
          if (origin.source_line) value_analysis["line"] = *origin.source_line;
          value_analysis["instruction"] = origin.source_text;
        }
        return;
      }
      const auto sites_before = sentryward::value_analysis::UnitSiteCounts();
      auto pipeline = sentryward::value_analysis::runValuePasses(
          semantic_program->values, actions.actions, actions.contract, [&](std::string_view pass) {
            monitor.Stage("value_analysis." + std::string(pass));
          });
      if (class_lifting()) {
        // A BlockIndex read numerically by the value passes baked the
        // representative's coordinate into the values.
        namespace va = sentryward::value_analysis;
        const auto sites = va::UnitSiteCounts();
        const auto numeric = [&](va::UnitSite site) {
          return sites[std::size_t(site)] - sites_before[std::size_t(site)];
        };
        const auto read = numeric(va::UnitSite::ValueSpecialization) + numeric(va::UnitSite::ValueTemplateLeaf);
        if (read) return refuse_lifting();
        // A memory-elimination failure with the order-free identity is the
        // per-unit run's. Members fail on their addresses at the
        // representative's positions (admission compared their slot shapes,
        // and they write the same allocations), and the static key names no
        // unit. Any other failure restarts.
        const auto &stop = pipeline.memory.diagnostic;
        if (!pipeline.memory.complete && !(stop && stop->order_free))
          return refuse_lifting(LiftingStopOf(stop));
        if (RowModulusViolation(*unit_lifting)) return refuse_lifting();
      }
      const auto packing = pipeline.packing, exp2 = pipeline.ideal_exp2;
      eliminated = std::move(pipeline.memory);
      value_analysis = {{"status", eliminated->complete ? "expressions_ready" : "unsupported"},
          {"passes", Json::array({
              {{"name", "packing"}, {"semantics", "exact_fixed_width"},
               {"rewritten_nodes", packing.nodes}, {"rewritten_instances", packing.instances}},
              {{"name", "software_exp2"}, {"semantics", "ideal_exp2"},
               {"rewritten_nodes", exp2.nodes}, {"rewritten_instances", exp2.instances},
               {"abstraction", "matched approximation and clamp replaced by ideal Exp2"}},
              {{"name", "type_edges"}, {"removed_nodes", pipeline.type_edges.nodes},
               {"converted_instances", pipeline.type_edges.instances}},
              {{"name", "memory_elimination"}},
              {{"name", "view_normalization"}, {"identities", pipeline.views.identities},
               {"elements", pipeline.views.elements}, {"slices", pipeline.views.slices},
               {"layouts", pipeline.views.layouts}, {"segmented", pipeline.views.segmented},
               {"original_pieces", pipeline.views.original_pieces},
               {"mapping_blocks", pipeline.views.mapping_blocks},
               {"reused_initial_reads", pipeline.views.reused_initial_reads},
               {"reused_views", pipeline.views.reused_views},
               {"stored_mapping_blocks", pipeline.views.stored_mapping_blocks},
               {"typed_output_regions", pipeline.views.typed_output_regions},
               {"typed_output_pieces", pipeline.views.typed_output_pieces}}})},
          {"processed_memory_actions", eliminated->processed_actions},
          {"processed_memory_families", eliminated->processed_families},
          {"memory_action_storage", {{"logical_actions",actions.actions.size()},
              {"stored_runs",actions.actions.storedRunCount()}}},
          {"duplicate_write_value_checks", eliminated->duplicate_write_value_checks},
          {"read_substitutions", eliminated->reads.size()+eliminated->shared_read_instances},
          {"stored_read_substitutions", eliminated->reads.size()},
          {"shared_read_families", eliminated->shared_read_families.size()},
          {"shared_read_instances", eliminated->shared_read_instances},
          {"initial_read_families", eliminated->initial_read_families.size()},
          {"initial_memory_regions", eliminated->initial_memory.size()},
          {"output_regions", eliminated->outputCount()},
          {"output_region_families", eliminated->output_families.size()},
          {"equivalence_check", "not_run"}, {"equivalence_reason", "reference_expression_not_supplied"}};
      if (options.value_thread_range && eliminated->complete) {
        using RegionKey = std::pair<std::string, bool>;
        std::map<RegionKey, std::vector<std::pair<std::uint64_t,std::uint64_t>>> regions;
        for (const auto& action : sentryward::value_analysis::MemoryActionRange(actions.actions)) {
          if (!action.region) throw std::runtime_error("missing partition footprint");
          const auto& op = semantic_program->values.node(action.operation.node).operation();
          const bool write = dynamic_cast<const sentryward::value_ir::StoreOp*>(&op) != nullptr;
          if (!write && !dynamic_cast<const sentryward::value_ir::LoadOp*>(&op))
            throw std::runtime_error("unknown partition memory effect");
          const auto& r = *action.region;
          if (r.bytes) regions[{r.allocation,write}].push_back({r.offset,r.offset+r.bytes});
        }
        value_analysis["partition_footprints"] = Json::array();
        for (auto& [key,ranges] : regions) {
          std::sort(ranges.begin(),ranges.end());
          std::vector<std::pair<std::uint64_t,std::uint64_t>> merged;
          for (const auto& r : ranges) {
            if (!merged.empty() && r.first <= merged.back().second)
              merged.back().second = std::max(merged.back().second,r.second);
            else merged.push_back(r);
          }
          for (const auto& r : merged)
            value_analysis["partition_footprints"].push_back({{"allocation",key.first},
                {"write",key.second},{"begin",r.first},{"end",r.second}});
        }
      }
      if (eliminated->diagnostic) {
        value_analysis["reason"] = eliminated->diagnostic->code;
        value_analysis["message"] = eliminated->diagnostic->message;
        if (const auto& operation = eliminated->diagnostic->operation) {
          const auto& origin = semantic_program->values.node(operation->node).instances().at(operation->instance).origin;
          if (origin.source_line) value_analysis["line"] = *origin.source_line;
          value_analysis["instruction"] = origin.source_text;
        }
      }
    };
    try {
      monitor.Stage("ptx_input.read");
      auto text = alg::ProfileCall("ptx_input.read",
                                   [&] { return inputs.Read(options.ptx); });

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_identity.inc"
#endif
      monitor.Stage("ptx_parser");
      auto module = alg::ProfileCall("ptx_parser", [&] {
        return ptx::Parse(std::move(text), options.ptx);
      });
      monitor.Stage("input_configuration");
      auto config = alg::ProfileCall("input_configuration", [&] {
        auto supplied = ptx::LoadKernelInput(options.config, inputs.File(options.config));
        auto supplied_flags = effective_assumptions;
        const auto overrides = supplied.fields.value("assumptions", Json::object());
        alg::OverridePtxAssumptions(supplied_flags, overrides.get<alg::PtxAssumptions>());
        alg::OverridePtxAssumptions(supplied_flags, options.assumptions);
        supplied.fields["assumptions"] = supplied_flags;
        effective_assumptions = supplied_flags;
        ptx::InputOptions input_options;
        input_options.max_steps_per_thread = options.max_steps_per_thread;
        input_options.compress_thread_records = options.compress_thread_records;
        auto normalized = ptx::NormalizeKernelInput(supplied, module, input_options);
        const auto &entry = *std::find_if(module.functions.begin(), module.functions.end(),
            [&](const auto &f) { return f.name == normalized.entry; });
        auto inferred = ptx::InferPointerFields(module, entry, normalized);
        if (!inferred.empty()) normalized = ptx::NormalizeKernelInput(supplied, module, input_options, inferred);
        return normalized;
      });

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/value_resource_bindings.inc"
#endif
      Json frontend_assumptions = {
          {"trap_only_equality_assertions_hold", config.fixed_input.value("assume_trap_assertions", false)},
          {"assertion_failure_forms", {"trap", "__assertfail"}},
          {"fixed_election", config.elect_lane.has_value()},
          {"election_policy", config.elect_lane ? Json{{"strategy","preferred_member"},
              {"preferred_lane",*config.elect_lane}, {"fallback","lowest_member"}} : Json(nullptr)},
          {"noalias_unbounded", false},
          {"pointer_contract", config.fixed_input.value("kernel_pointer_contract",Json::object())},
          {"isolated_launch", config.fixed_input.value("launch_contract",Json::object()).value("external_grids","") == "no_overlap"},
          {"clc_failure", config.fixed_input.contains("cluster_launch_cancellation") &&
              config.fixed_input["cluster_launch_cancellation"] == false},
          {"assertion_uses",0}, {"election_uses",0}, {"clc_uses",0},
          {"representative_branch_uses",0}, {"bounded_loop_uses",0}, {"atomic_uses",0},
          {"data_branch_decisions",Json::array()},
          {"opaque_compute_regions",Json::array()}};
      alg::OverridePtxAssumptions(effective_assumptions,
          config.raw.value("semantic_assumptions", Json::object()).get<alg::PtxAssumptions>());
      alg::OverridePtxAssumptions(effective_assumptions, options.assumptions);
      config.raw["semantic_assumptions"] = effective_assumptions;
      frontend_assumptions["fixed_election"] = effective_assumptions.at("representative_election");
      frontend_assumptions["clc_failure"] = effective_assumptions.at("clc_always_fails");
      input_metadata = {{"frontend_assumptions", frontend_assumptions},
          {"input_objects",config.raw.value("input_objects",Json::array())},
          {"user_branch_decisions",config.raw.value("user_branch_decisions",Json::array())}};
      if (config.fixed_input.contains("shape_input_contract"))
        input_metadata["frontend_assumptions"]["shape_input_contract"] =
            config.fixed_input["shape_input_contract"];
      if (config.fixed_input.contains("pointer_array_contract"))
        input_metadata["frontend_assumptions"]["pointer_array_contract"] =
            config.fixed_input["pointer_array_contract"];
      if (config.fixed_input.contains("tensor_property_contract")) {
        input_metadata["frontend_assumptions"]["tensor_properties"] = config.fixed_input["tensor_property_contract"];
        input_metadata["frontend_assumptions"]["tensor_property_decisions"] = Json::array();
      }
      const ptx::Function *function = nullptr;
      const bool explicit_prefix = config.entry.starts_with("prefix:");
      const auto selector =
          explicit_prefix ? config.entry.substr(7) : config.entry;
      if (!explicit_prefix)
        for (const auto &f : module.functions)
          if (f.name == selector) {
            function = &f;
            break;
          }
      if (!function)
        for (const auto &f : module.functions)
          if (f.name.starts_with(selector)) {
            if (function)
              throw std::invalid_argument(
                  "entry selector matches multiple functions");
            function = &f;
          }
      if (!function)
        throw std::invalid_argument("configured entry missing from PTX");
      monitor.Stage("launch_binding");
      config = alg::ProfileCall("launch_binding", [&] {
        return ptx::BindLaunchObjects(module, *function, config);
      });

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_provenance.inc"
#else
      {
#endif
      auto [grid, block] = ptx::LaunchDimensions(config);
      (void)block;
      if (options.trusted_reference) {
        const std::set<std::string> forbidden = {"bar", "barrier", "mbarrier", "shfl",
            "vote", "match", "elect", "activemask", "redux", "mma", "wmma", "wgmma",
            "tcgen05", "cp", "clusterlaunchcontrol", "atom", "red", "call"};
        for (const auto& inst : function->instructions)
          if (forbidden.contains(inst.opcode) || inst.semantics.asynchronous)
            throw ptx::ExecutionError("unsupported_reference_order", inst.line,
                "Trusted-reference mode currently requires synchronous, noncollective instructions");
      }
      if (options.value_thread_range) {
        const auto total = std::uint64_t(grid[0])*grid[1]*grid[2]*config.Threads();
        if (options.value_thread_range->second > total ||
            (config.cluster_dim && *config.cluster_dim != ptx::Coordinates{1,1,1}))
          throw std::invalid_argument("thread subset exceeds launch or uses a cluster");
        const std::set<std::string> forbidden = {"bar", "barrier", "mbarrier", "shfl",
            "vote", "match", "elect", "activemask", "redux", "mma", "wmma", "wgmma",
            "tcgen05", "cp", "clusterlaunchcontrol", "atom", "red", "call"};
        for (const auto& inst : function->instructions)
          if (forbidden.contains(inst.opcode) || inst.semantics.asynchronous)
            throw std::invalid_argument("thread subset requires independent synchronous instructions: "+inst.opcode);
      }
      if (ptx::Integer(grid[0]) * grid[1] * grid[2] * config.Threads() >
          UINT32_MAX)
        throw std::invalid_argument("launch exceeds native actor ID domain");
      try {
      std::unique_ptr<ptx::ModelLowerer> lowerer;
      // Cross-CTA: launch units, the unit-independence precondition and, in
      // class mode, the unit-major loop below. With class mode off (a gate or
      // the per-unit re-run) nothing below observes the run.
      ptx::UnitGates unit_gates;
      unit_gates.value_analysis = options.analyze_values;
      unit_gates.scalar_fold = options.analyze_scalar_fold;
      unit_gates.value_thread_range = options.value_thread_range.has_value();
      unit_gates.trusted_reference = options.trusted_reference;
      unit_gates.value_dump = values != nullptr;
      unit_gates.branch_scope = !options.dump_branch_scope.empty();
      unit_gates.scalar_fold_dump = !options.dump_scalar_fold.empty();
      unit_gates.exhaustive_search = options.exhaustive_memory_search;
      unit_gates.continue_search = options.continue_memory_search;
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
      unit_gates.development_exports = true;
#endif
      auto unit_plan = ptx::PlanUnits(config, *function, unit_gates);
      if (unit_restart) unit_plan.class_mode = false; // the per-unit attempt
      // Class mode reads the fixed shape-array tables as table atoms (unit_config below): their
      // lookups are valued (TableValuation) by allocation object, element width and raw bits.
      // Table atoms: in class mode over fixed shape-array tables, in the table-atom attempt, or from the
      // start when no value stage lifts (verification only). The first attempt of a run that lifts values
      // reads the tables' elements exactly and restarts with them once its per-unit value analysis is over
      // budget (TableAtomRestart).
      const bool has_tables = config.fixed_input.contains("global_memory_inputs");
      const bool table_atoms = unit_plan.class_mode && has_tables && (atom_restart || !unit_plan.lift_values);
      const bool atom_trigger = unit_plan.class_mode && has_tables && !table_atoms;
      if (table_atoms) input_metadata["class_table_atoms"] = {{"attempt", atom_restart ? "restart" : "verification_only"}};
      const std::string table_atoms_json = table_atoms ? TableAtomsJson(config) : std::string("[]");
      // Class mode's log budget, min(2 GiB, 10% of --rss-limit-mib).
      const std::uint64_t class_budget = std::min<std::uint64_t>(
          std::uint64_t(2) << 30, options.rss_mib ? *options.rss_mib * 1048576 / 10 : UINT64_MAX);
      if (options.analyze_values || (!options.extract && !options.semantic && !values)) {
        semantic_builder = std::make_unique<ptx::SemanticProgramBuilder>(module, *function, poll,
            ptx::SemanticProgramOptions{
                .collect_synchronization_groups = options.analyze_values && !options.trusted_reference,
                .compact_memory_linking = true,
                .link_numerical_memory = options.analyze_values,
                .defer_address_serialization = !options.trusted_reference,
                .capture_branch_scope = !options.dump_branch_scope.empty(),
                .atomics = effective_assumptions.at("order_independent_atomics"),
                .values_on_demand = !options.analyze_values,
                .threads = options.threads,
                .table_atoms = table_atoms_json});
      } else if (!options.semantic && !values)
        lowerer = alg::ProfileCall("frontend.static_lowering_prepare", [&] {
          return std::make_unique<ptx::ModelLowerer>(module, *function, poll, false,
              effective_assumptions.at("order_independent_atomics"), options.threads);
        });
      // The classes of this launch (class mode only).
      std::unique_ptr<ptx::UnitClasses> classes;
      bool values_suspended = false; // SemanticProgramBuilder::SuspendValues was called
      // The founder plan holds a containment member; SuspendValues was called on the plan's word alone.
      bool plan_contains = false, suspended_by_plan = false;
      if (unit_plan.class_mode) {
        classes = std::make_unique<ptx::UnitClasses>(unit_plan, class_budget);
        // Without value lifting, classes admit a piecewise δ and a valued-unliftable unit from the start;
        // in the table-atom attempt (no value stage could complete per unit), such a unit.
        // Classes admit a piecewise δ from the start. Value lifting does not represent one: a
        // value stage that would lift a piecewise quotient refuses lifting (run_values), as any
        // other unliftable value stage does.
        classes->AdmitPiecewise();
        // Containment members (query_log.h UnitDelta::contained) are never lifted: classes admit
        // them where they admit a valued-unliftable unit.
        if (table_atoms) {
          classes->AdmitFormOnly();
          classes->AdmitContainment();
        }
        // The first attempt over fixed tables: once the per-unit value analysis, estimated from the value
        // views built so far, is over the per-unit re-run budget, restart with table atoms, right after
        // the founder's value view, before its lowering (a founder with a value diagnostic lifts nothing).
        if (semantic_builder && atom_trigger && unit_plan.lift_values)
          semantic_builder->SetValueViewHook([&] {
            if (semantic_builder->valueDiagnostics().empty() &&
                PerUnitValuesOverBudget(semantic_builder->graph(), unit_plan.units.size(),
                                        semantic_builder->valueViewCtas() / unit_plan.UnitSize()))
              throw alg::UnitClassRestart{"table_atoms"};
          });
        // Lowering weights by the class tables, complete before its Finish.
        quotient = std::make_unique<alg::UnitQuotient>();
        if (semantic_builder) semantic_builder->TrackUnits(quotient.get());
        if (lowerer) lowerer->TrackUnits(quotient.get());
      }
      {
        alg::ProfileScope frontend("frontend.dynamic_pipeline");
        const auto purpose = branch_capture ? ptx::SpecializationPurpose::BranchScopeCapture
                            : !options.dump_branch_scope.empty() ? ptx::SpecializationPurpose::VerificationWithBranchScope
                            : values ? ptx::SpecializationPurpose::ValueExtraction
                                    : ptx::SpecializationPurpose::Verification;
        std::set<ptx::Coordinates> emitted;
        std::map<std::uint64_t, ptx::DataBranchDecision> data_branch_decisions;
        std::map<std::uint64_t, std::uint64_t> opaque_regions;
        std::map<std::string, std::set<std::uint64_t>> property_lines;
        std::set<std::uint64_t> static_choice_lines;
        // The driver's accounting of one executed CTA (every CTA, members included), applied in
        // launch order: its SummarizeCta.
        const auto account = [&](const CtaAccount &cta_account) {
              const auto &execution=cta_account.execution;
              template_ctas+=execution.used;
              template_execution.steps+=execution.steps;
              template_execution.logical_steps+=execution.logical_steps;
              template_execution.splits+=execution.splits;
              template_execution.value_nodes+=execution.value_nodes;
              template_execution.record_nodes+=execution.record_nodes;
              template_execution.record_bindings+=execution.record_bindings;
              template_execution.coordinate_bindings+=execution.coordinate_bindings;
              template_execution.path_entries+=execution.path_entries;
              template_execution.thread_memberships+=execution.thread_memberships;
              alg::ReportIrSize("frontend.execution", {{"completed_ctas",template_ctas},
                  {"logical_steps",template_execution.logical_steps},
                  {"record_nodes",template_execution.record_nodes},
                  {"value_nodes",template_execution.value_nodes},
                  {"record_bindings",template_execution.record_bindings},
                  {"coordinate_bindings",template_execution.coordinate_bindings},
                  {"path_entries",template_execution.path_entries}});
              const auto &storage = cta_account.storage;
              record_compression.record_count += storage.record_count;
              record_compression.template_count += storage.template_count;
              record_compression.unique_binding_count += storage.unique_binding_count;
              record_compression.estimated_owned_bytes += storage.estimated_owned_bytes;
              records += cta_account.records;
              auto &facts = input_metadata["frontend_assumptions"];
              const auto add = [&](const char *counter, std::uint64_t count) {
                facts[counter] = facts[counter].get<std::uint64_t>() + count;
              };
              add("assertion_uses", cta_account.assumed_assertions);
              for (const auto &[line, count] : cta_account.opaque_regions) opaque_regions[line] += count;
              for (const auto &[line, decision] : cta_account.data_branch_decisions)
                AddDecision(data_branch_decisions[line], decision);
              add("representative_branch_uses", cta_account.representative_branch_uses);
              add("bounded_loop_uses", cta_account.bounded_loop_uses);
              static_choice_lines.insert(cta_account.static_choice_lines.begin(),
                                         cta_account.static_choice_lines.end());
              add("election_uses", cta_account.election_uses);
              add("clc_uses", cta_account.clc_uses);
              add("atomic_uses", cta_account.atomic_uses);
              // Unit independence forbids an executed global atomic: restart if this CTA executes one.
              if (unit_plan.class_mode && cta_account.executes_atomic_site)
                throw alg::UnitClassRestart{"executed_global_atomic"};
        };
        const auto summarize = [&](const ptx::CtaFacts &cta_facts) {
          return SummarizeCta(cta_facts, *function, unit_plan.precondition.atomic_sites);
        };
        // A CTA's tensor_properties decisions, read once every consumer is done with it: template
        // values evaluated after its run (lowering, class admission) are decided by bounds too.
        const auto account_properties = [&](const std::map<std::string, std::set<std::uint32_t>> &decisions) {
          for (const auto &[object, sites] : decisions)
            for (const auto id : sites) property_lines[object].insert(function->instructions[id].line);
        };
        const auto linear_id = [&](const ptx::Coordinates &c) {
          return static_cast<std::uint32_t>((std::uint64_t(c[2]) * grid[1] + c[1]) * grid[0] + c[0]);
        };
        if (unit_plan.class_mode) {
          // Class loop: units in unit order, every CTA executed and accounted for;
          // a unit joins a class before any of its CTAs is consumed, and only
          // the founding (representative) units are consumed, in rank order.
          std::uint32_t consumed = 0;
          // Up to --sync-threads look-ahead workers (below).
          const std::size_t workers = options.threads > 1 ? options.threads : 0;
          // One unit's execution: its ranks' programs, their facts, structure keys and driver
          // accounting (pure functions of each CTA's own execution). Facts point into the
          // programs, so they are destroyed first.
          struct ExecutedUnit {
            std::vector<ptx::ExecutionProgram> programs;
            std::vector<std::unique_ptr<ptx::UnitFrame>> frames;
            std::vector<std::unique_ptr<ptx::CtaFacts>> facts;
            std::vector<ptx::StructureKey> keys;
            // Every rank's containment profile when classes admit containment members (else empty).
            std::vector<ptx::ContainmentProfile> profiles;
            std::vector<CtaAccount> accounts;
            // Per rank: the CTA's tensor_properties decisions, read when its program is released.
            std::vector<std::map<std::string, std::set<std::uint32_t>>> property_sites;
            // Admission against the classes founded before this unit ran (look-ahead).
            ptx::UnitClasses::PreAdmission pre;
            // Every rank's program released: only the accounting, the decisions and the admission remain.
            bool released = false;
          };
          // Releases a rank's program after its last consumer (admission, or the founder's
          // consumption), on whichever thread finished with it. The executions were allocated in
          // the workers' arenas, which keep the freed pages: return them every 8 released units.
          [[maybe_unused]] std::atomic<std::size_t> released_units = 0;
          const auto release_rank = [&](ExecutedUnit &executed, std::size_t rank) {
            executed.property_sites[rank] = executed.facts[rank]->PropertyDecisionsSnapshot();
            executed.facts[rank].reset();
            alg::ProfileCall("frontend.release_execution", [&] { executed.programs[rank] = ptx::ExecutionProgram{}; });
            if (rank + 1 == executed.programs.size()) executed.released = true;
#if defined(__GLIBC__)
            if (workers && executed.released && ++released_units % 8 == 0) malloc_trim(0);
#endif
          };
          const auto release = [&](ExecutedUnit &executed) {
            for (std::size_t rank = 0; rank < executed.programs.size(); ++rank) release_rank(executed, rank);
          };
          // A unit the look-ahead admits is released by the thread that admitted it; the loop
          // then applies its accounting and joins it without its facts.
          const auto pre_admit = [&](ExecutedUnit &executed) {
            std::vector<const ptx::CtaFacts *> view;
            for (const auto &facts : executed.facts) view.push_back(facts.get());
            executed.pre = alg::ProfileCall("class.pre_admit", [&] {
              return classes->PreAdmit(view, executed.keys, std::move(executed.pre), executed.profiles);
            });
            if (executed.pre.joined) {
              release(executed);
              return;
            }
            // The replay's reads filled the template value caches; drop them before the
            // unit waits in the window (a later read, e.g. a founder's lowering, recomputes).
            for (const auto &facts : executed.facts) facts->ReleaseValues();
          };
          // Class mode reads whole elements of fixed shape-array tables as table atoms (exact values
          // that keep their element form for the slot forms admission compares up to δ).
          const auto unit_config = [&](const ptx::Coordinates &coordinates) {
            auto local = ptx::ConfigForCta(config, coordinates);
            if (table_atoms) local.fixed_input["table_atoms"] = true;
            return local;
          };
          // One unit's programs, rank order.
          const auto specialize = [&](const ptx::LaunchUnit &unit, const std::function<void()> &unit_poll) {
            const auto size = unit.ctas.size();
            std::vector<ptx::ExecutionProgram> programs(size);
            // Set when a routed cluster run (ExecutionError::shared_routed) failed: the remaining
            // ranks run without shared-memory routing, as before routing existed.
            bool unrouted = false;
            for (std::size_t rank = 0; rank < size; ++rank) {
              const auto &coordinates = unit.ctas[rank];
              try {
                programs[rank] = alg::ProfileCall("ptx_execution", [&] {
                  return unrouted ? ptx::SpecializeFunctionUnrouted(module, *function, unit_config(coordinates), unit_poll, purpose)
                                  : ptx::SpecializeFunction(module, *function, unit_config(coordinates), unit_poll, purpose);
                });
              } catch (const ptx::ExecutionError &error) {
                if (!error.shared_control) throw;
                // Shared control (as without classes, but per unit): run the whole
                // cluster; ranks executed individually before keep their programs.
                std::vector<ptx::FrontendConfig> configs;
                for (const auto &member : unit.ctas) configs.push_back(unit_config(member));
                std::optional<std::vector<ptx::ExecutionProgram>> cluster;
                try { ptx::SharedRoutingAttempt attempt(error); cluster = ptx::SpecializeCluster(module, *function, configs, unit_poll, purpose); }
                catch (const ptx::ExecutionError &) { if (!error.shared_routed) throw; }
                if (!cluster) {
                  unrouted = true;
                  try {
                    programs[rank] = alg::ProfileCall("ptx_execution", [&] {
                      return ptx::SpecializeFunctionUnrouted(module, *function, unit_config(coordinates), unit_poll, purpose);
                    });
                    continue;
                  } catch (const ptx::ExecutionError &unrouted_error) {
                    // Shared control without routing: the whole cluster's run, as before routing existed.
                    if (!unrouted_error.shared_control) throw;
                    cluster = ptx::SpecializeCluster(module, *function, configs, unit_poll, purpose);
                  }
                }
                for (std::size_t r = rank; r < size; ++r) programs[r] = std::move((*cluster)[r]);
                break;
              }
            }
            return programs;
          };
          const auto execute = [&](std::size_t index, const std::function<void()> &unit_poll) {
            const auto &unit = unit_plan.units[index];
            const auto size = unit.ctas.size();
            auto result = std::make_unique<ExecutedUnit>();
            result->programs = specialize(unit, unit_poll);
            auto &programs = result->programs;
            for (std::size_t rank = 0; rank < size; ++rank) {
              auto &program = programs[rank];
              result->frames.push_back(std::make_unique<ptx::UnitFrame>(unit.ctas[rank], program.config.cluster_dim));
              result->facts.push_back(
                  std::make_unique<ptx::CtaFacts>(program, *result->frames.back(), linear_id(unit.ctas[rank])));
              result->accounts.push_back(summarize(*result->facts.back()));
              result->keys.push_back(alg::ProfileCall("class.structure_key",
                                                      [&] { return ptx::ComputeStructureKey(*result->facts.back()); }));
            }
            if (classes->ContainmentOn())
              for (const auto &facts : result->facts)
                result->profiles.push_back(alg::ProfileCall("class.containment_profile",
                                                            [&] { return ptx::ComputeContainmentProfile(*facts); }));
            result->property_sites.resize(size);
            pre_admit(*result);
            return result;
          };
          // Look-ahead: the workers execute the next units while
          // this loop consumes in unit order; a unit's execution error is rethrown here,
          // at that unit, and leaving the loop (a restart included) joins the workers.
          // The first unit runs inline: the bytes it allocates bound the workers (each
          // keeps its own allocator arena at about one execution) and the executed units
          // in flight to LookAheadBudget (a quarter of the memory limit).
          // The order in which the loop takes the units: unit order, or, when classes admit containment
          // members from the start, the planned founders first (ptx::ContainmentPlan). The plan pass
          // executes every unit once more for its profiles alone (the workers' window of programs at a
          // time): O(units x execution), about the cost of the loop's own executions.
          std::vector<std::uint32_t> order(unit_plan.units.size());
          std::iota(order.begin(), order.end(), 0u);
          // The plan pass runs only in the attempt that a unit-order attempt's class loop asked for
          // (FounderPlanRestart); `plan_probe`: this attempt is that unit-order attempt.
          // Founder planning serves only a clean pass through containment members (a planned attempt
          // with any other end is redone in unit order): while containment is race-only
          // (alg::kContainmentReportsClean off) no attempt plans.
          const bool plan_possible =
              alg::kContainmentReportsClean && !order_restart && workers && unit_plan.units.size() > 2;
          const bool plan_probe = plan_possible && !plan_restart;
          if (classes->ContainmentOn() && plan_possible && plan_restart) {
            monitor.Stage("class_plan");
            ptx::ContainmentPlan plan;
            {
              ptx::LookAhead<std::vector<ptx::ContainmentProfile>> profiles(
                  unit_plan.units.size(), workers, workers,
                  [&](std::size_t index, const std::function<void()> &unit_poll) {
                    const auto &unit = unit_plan.units[index];
                    auto programs = specialize(unit, unit_poll);
                    std::vector<ptx::ContainmentProfile> result;
                    for (std::size_t rank = 0; rank < programs.size(); ++rank) {
                      const ptx::UnitFrame frame(unit.ctas[rank], programs[rank].config.cluster_dim);
                      const ptx::CtaFacts facts(programs[rank], frame, linear_id(unit.ctas[rank]));
                      result.push_back(alg::ProfileCall("class.containment_profile",
                                                        [&] { return ptx::ComputeContainmentProfile(facts); }));
                    }
                    return result;
                  },
                  poll);
              for (std::uint32_t unit = 0; unit < unit_plan.units.size(); ++unit) plan.Add(unit, profiles.Take());
            }
            auto planned = alg::ProfileCall("class.plan", [&] {
              return plan.Plan(static_cast<std::uint32_t>(unit_plan.units.size()));
            });
            if (!std::is_sorted(planned.units.begin(), planned.units.end())) {
              order = std::move(planned.units);
              founders_first = true;
              plan_contains = planned.contains;
              classes->ExpectFounders(planned.founders);
              input_metadata["class_founder_plan"] = {{"founders", planned.founders}, {"units", order.size()}};
            }
#if defined(__GLIBC__)
            malloc_trim(0);
#endif
          }
          std::unique_ptr<ExecutedUnit> first;
          std::size_t window = workers;
          if (workers && !unit_plan.units.empty()) {
            monitor.Stage("ptx_execution.cta" + std::to_string(ctas));
            const auto before = AllocatedBytes();
            first = execute(order[0], poll);
            const auto after = AllocatedBytes();
            if (before && after && *after > *before)
              window = static_cast<std::size_t>(std::clamp<std::uint64_t>(
                  LookAheadBudget(options.rss_mib) / (*after - *before), 1, workers));
          }
          const std::size_t skip = first ? 1 : 0;
          std::optional<ptx::LookAhead<std::unique_ptr<ExecutedUnit>>> executions;
          executions.emplace(unit_plan.units.size() - skip, window, window,
              [&](std::size_t index, const std::function<void()> &unit_poll) { return execute(order[index + skip], unit_poll); },
              poll);
          for (const auto unit_index : order) {
            const auto &unit = unit_plan.units[unit_index];
            const auto size = unit.ctas.size();
            monitor.Stage("ptx_execution.cta" + std::to_string(ctas));
            const auto executed_unit = first ? std::move(first) : executions->Take();
            for (const auto &cta_account : executed_unit->accounts) {
              account(cta_account);
              ++ctas;
            }
            const auto account_released = [&] {
              for (const auto &sites : executed_unit->property_sites) account_properties(sites);
            };
            // After a unit joined or founded a class: the two founder-order restarts.
            const auto order_checks = [&] {
              // A plan could now change the classes: restart this attempt with the plan pass.
              if (plan_probe && classes->ContainmentOn() && classes->PlanUseful())
                throw alg::UnitClassRestart{"founder_plan"};
              // A planned-founder attempt stands only when it ends with a clean pass, and a class with
              // a marked member never reports one (the memory check reports a race or restarts per
              // unit): the attempt is redone in unit order now instead of after its checks.
              if (founders_first && classes->MarkedAdmitted()) throw alg::UnitClassRestart{"founder_order"};
            };
            if (executed_unit->released) {
              account_released();
              classes->Join(unit.index, std::move(executed_unit->pre));
              order_checks();
              continue;
            }
            auto &unit_facts = executed_unit->facts;
            std::vector<const ptx::CtaFacts *> view;
            for (const auto &facts : unit_facts) view.push_back(facts.get());
            if (classes->Admit(unit.index, view, std::move(executed_unit->keys), std::move(executed_unit->pre),
                               std::move(executed_unit->profiles))) {
              release(*executed_unit);
              account_released();
              order_checks();
              continue;
            }
            for (std::size_t rank = 0; rank < size; ++rank) {
              auto &cta_facts = *unit_facts[rank];
              ptx::QuerySink *const sink = classes->Record(unit.index, static_cast<std::uint32_t>(rank), cta_facts);
              {
                // The recorder is attached while the consumers read this CTA.
                struct Detach {
                  ptx::CtaFacts &facts;
                  bool attached;
                  ~Detach() { if (attached) facts.Attach(nullptr); }
                } detach{cta_facts, sink != nullptr};
                if (sink) cta_facts.Attach(sink);
                if (lowerer) {
                  monitor.Stage("source_ir.cta" + std::to_string(consumed));
                  lowerer->AddCta(cta_facts);
                }
                if (semantic_builder) {
                  monitor.Stage("semantic_ir.cta" + std::to_string(consumed));
                  // Once an admitted member makes value lifting impossible (its exact offsets are not
                  // one δ per allocation from its representative's, or it has a piecewise δ), the value
                  // stage will refuse lifting before reading any view: later founders build none (only for
                  // an address-bridge query), and, as for a value diagnostic, later classes may admit a
                  // piecewise δ.
                  // With planned founders first, the members that make lifting impossible come after
                  // every founder: when the plan holds a containment member, only the first founder
                  // builds a value view (the per-unit budget's estimate); the loop's end checks that such
                  // a member was admitted (else the attempt is redone in unit order).
                  if (unit_plan.lift_values && !values_suspended &&
                      (classes->ValuedUnliftable() || classes->PiecewiseAdmitted() || classes->ContainmentAdmitted() ||
                       (plan_contains && consumed >= unit_plan.UnitSize()))) {
                    semantic_builder->SuspendValues();
                    classes->AdmitPiecewise();
                    values_suspended = true;
                    suspended_by_plan = !(classes->ValuedUnliftable() || classes->PiecewiseAdmitted() ||
                                          classes->ContainmentAdmitted());
                  }
                  semantic_builder->AddCta(cta_facts);
                  // A value diagnostic leaves the program's values incomplete: no value stage lifts,
                  // so this class and later ones may admit a piecewise δ and a valued-unliftable unit.
                  if (!semantic_builder->valueDiagnostics().empty()) {
                    classes->AdmitPiecewise();
                    classes->AdmitFormOnly();
                    classes->AdmitContainment();
                  }
                }
                ++consumed;
              }
              classes->Recorded(unit.index, static_cast<std::uint32_t>(rank));
              release_rank(*executed_unit, rank);
              account_properties(executed_unit->property_sites[rank]);
            }
            // The units executed while this one was consumed were pre-admitted without its class:
            // try it on them now, in parallel (Admit would replay it here, one unit at a time).
            if (workers)
              executions->RefreshReady(workers, [&](std::unique_ptr<ExecutedUnit> &waiting) {
                if (!waiting->pre.joined) pre_admit(*waiting);
              });
            order_checks();
          }
          executions.reset();
          // Value views were suspended on the plan's word, and no admitted member makes lifting
          // impossible: the attempt is redone in unit order.
          if (suspended_by_plan &&
              !(classes->ValuedUnliftable() || classes->PiecewiseAdmitted() || classes->ContainmentAdmitted()))
            throw alg::UnitClassRestart{"founder_order"};
#if defined(__GLIBC__)
          // The workers' arenas keep the pages of the programs the loop freed;
          // return them before the analysis stages (gigabytes on large launches).
          if (workers) malloc_trim(0);
#endif
        } else
        for (std::uint32_t z = 0; z < grid[2]; ++z)
          for (std::uint32_t y = 0; y < grid[1]; ++y)
            for (std::uint32_t x = 0; x < grid[0]; ++x) {
              ptx::Coordinates coordinates{x, y, z};
              if (emitted.contains(coordinates)) continue;
              auto local = ptx::ConfigForCta(config, coordinates);
              if (options.value_thread_range) {
                const auto base = (std::uint64_t(z)*grid[1]*grid[0]+std::uint64_t(y)*grid[0]+x)*config.Threads();
                const auto begin = std::max(base, options.value_thread_range->first);
                const auto end = std::min(base+config.Threads(), options.value_thread_range->second);
                if (begin >= end) continue;
                local.analysis_thread_range = {{static_cast<std::uint32_t>(begin-base),
                                               static_cast<std::uint32_t>(end-base)}};
              }
              monitor.Stage("ptx_execution.cta" + std::to_string(ctas));
              std::vector<std::pair<ptx::Coordinates,ptx::ExecutionProgram>> programs;
              try {
                programs.emplace_back(coordinates,alg::ProfileCall("ptx_execution", [&] {
                  return ptx::SpecializeFunction(module, *function, local, poll, purpose);
                }));
              } catch (const ptx::ExecutionError &error) {
                if (!error.shared_control) throw;
                const auto shape=config.cluster_dim.value_or(ptx::Coordinates{1,1,1});
                const ptx::Coordinates base{x/shape[0]*shape[0],y/shape[1]*shape[1],z/shape[2]*shape[2]};
                std::vector<ptx::FrontendConfig> configs;
                std::vector<ptx::Coordinates> members;
                bool complete=true;
                for (unsigned dz=0;dz<shape[2];++dz)
                  for (unsigned dy=0;dy<shape[1];++dy)
                    for (unsigned dx=0;dx<shape[0];++dx) {
                      ptx::Coordinates member{base[0]+dx,base[1]+dy,base[2]+dz};
                      for (unsigned axis=0;axis<3;++axis) if (member[axis]>=grid[axis]) {
                        if (!error.shared_routed)
                          throw ptx::ExecutionError("shared_control_cluster",error.line,"incomplete CTA cluster in fixed launch");
                        complete=false;
                      }
                      members.push_back(member); configs.push_back(ptx::ConfigForCta(config,member));
                    }
                // Shared-memory routing (ExecutionError::shared_routed): when the routed cluster run
                // cannot run or fails, this CTA runs without routing, as before routing existed (its
                // own run, or on shared control the whole cluster's).
                std::optional<std::vector<ptx::ExecutionProgram>> cluster;
                if (complete) {
                  try { ptx::SharedRoutingAttempt attempt(error); cluster=ptx::SpecializeCluster(module,*function,configs,poll,purpose); }
                  catch (const ptx::ExecutionError &) { if (!error.shared_routed) throw; }
                }
                if (!cluster) {
                  try {
                    programs.emplace_back(coordinates,alg::ProfileCall("ptx_execution", [&] {
                      return ptx::SpecializeFunctionUnrouted(module, *function, local, poll, purpose);
                    }));
                  } catch (const ptx::ExecutionError &unrouted_error) {
                    if (!unrouted_error.shared_control) throw;
                    if (!complete)
                      throw ptx::ExecutionError("shared_control_cluster",unrouted_error.line,"incomplete CTA cluster in fixed launch");
                    cluster=ptx::SpecializeCluster(module,*function,configs,poll,purpose);
                  }
                }
                if (cluster)
                  for (std::size_t rank=0;rank<cluster->size();++rank)
                    if (!emitted.contains(members[rank])) programs.emplace_back(members[rank],std::move((*cluster)[rank]));
              }
              for (auto &[member,program] : programs) {
              {
              // The only consumer view of this program; destroyed before the program is released.
              const ptx::UnitFrame frame(member, program.config.cluster_dim);
              ptx::CtaFacts cta_facts(program, frame, linear_id(member));
              account(summarize(cta_facts));
              if (lowerer) {
                monitor.Stage("source_ir.cta" + std::to_string(ctas));
                lowerer->AddCta(cta_facts);
              }
              if (value_extractor) {
                monitor.Stage("value_ir.cta" + std::to_string(ctas));
                value_extractor->AddCta(cta_facts);
              }
              if (semantic_builder) {
                monitor.Stage("semantic_ir.cta" + std::to_string(ctas));
                semantic_builder->AddCta(cta_facts);
              }
              account_properties(cta_facts.PropertyDecisionsSnapshot());
              }
              alg::ProfileCall("frontend.release_execution",
                               [&] { program = ptx::ExecutionProgram{}; });
              ++ctas;
              emitted.insert(member);
              }
            }
        for (const auto &[line, count] : opaque_regions)
          input_metadata["frontend_assumptions"]["opaque_compute_regions"].push_back({{"line", line}, {"lanes", count}});
        for (const auto &[object, lines] : property_lines)
          input_metadata["frontend_assumptions"]["tensor_property_decisions"].push_back(
              {{"object", object}, {"lines", lines}});
        // Choices that a statically known predicate overrode on some visit (listed only when any).
        if (!static_choice_lines.empty())
          input_metadata["frontend_assumptions"]["branch_choices_with_static_visits"] = static_choice_lines;
        auto &decisions = input_metadata["frontend_assumptions"]["data_branch_decisions"];
        for (const auto &[line, decision] : data_branch_decisions)
          decisions.push_back({{"line", line}, {"assumption", decision.assumption}, {"rule", decision.rule},
                               {"taken", decision.taken}, {"not_taken", decision.not_taken}});
        if (unit_plan.class_mode) {
          // The quotient tables (lowering weights by them at Finish); the tensormap
          // locality rule needs every unit's slots, so it is decided only now. An
          // input map overwrite is found again by the per-unit run.
          *quotient = classes->Quotient();
          quotient->reordered = founders_first;
          if (!classes->TensormapLocality()) throw alg::UnitClassRestart{"tensormap_locality"};
          if (quotient->Contained()) {
            std::uint64_t members = 0, holding = 0, marked = 0;
            for (const auto &c : quotient->classes) {
              const auto n = std::count(c.contained.begin(), c.contained.end(), true);
              members += n;
              holding += n != 0;
              marked += c.hazard;
            }
            // `marked_classes`: the classes with a marked member (UnitQuotient::Class::hazard).
            input_metadata["class_containment"] = {{"members", members}, {"classes", holding},
                                                   {"class_count", quotient->classes.size()},
                                                   {"marked_classes", marked}};
          }
          // Values are lifted only when some class has members (fewer
          // classes than units); otherwise every unit was analysed (the per-unit run's fold).
          if (unit_plan.lift_values && quotient->Shared()) {
            unit_lifting = ClassLifting(*quotient);
            // A piecewise member admitted after the last founder: Finish links no memory either.
            if (!values_suspended && (quotient->Piecewise() || quotient->Contained()) && semantic_builder) {
              semantic_builder->SuspendValues();
              values_suspended = true;
            }
            if (values_suspended && semantic_builder) unit_lifting->value_view_ctas = semantic_builder->valueViewCtas();
          }
        }
        classes.reset();
      }
      if (values) {
        result = {{"status", value_extractor->complete() ? "value_view_complete" : "value_view_incomplete"},
                  {"ctas", ctas}, {"executed_records", records},
                  {"verification_performed", false},
                  {"memory_analysis_complete", false},
                  {"diagnostics", Json::array()}};
        for (const auto &diagnostic : value_extractor->diagnostics())
          result["diagnostics"].push_back({{"code", diagnostic.code}, {"message", diagnostic.message},
              {"location", {{"file", diagnostic.origin.source_file.value_or(options.ptx)},
                            {"line", diagnostic.origin.source_line.value_or(0)}}},
              {"thread", diagnostic.origin.thread.value_or(0)},
              {"instruction_index", diagnostic.origin.instruction_index.value_or(0)},
              {"occurrence", diagnostic.origin.occurrence.value_or(0)}});
        result["frontend"] = input_metadata;
      } else if (options.semantic) {
        result = {{"status", "semantic_complete"},
                  {"ctas", ctas},
                  {"executed_records", records},
                  {"static_instructions", function->instructions.size()},
                  {"verification_performed", false},
                  {"memory_analysis_complete", false}};
        result["frontend"] = input_metadata;
      } else {
        monitor.Stage("source_ir.finalize");

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/branch_failure_export.inc"
#endif
        if (semantic_builder) semantic_program.emplace(semantic_builder->Finish(
            [&](std::string_view stage) { monitor.Stage(std::string(stage)); }));
        // Binding diagnostics are weighted per CTA (BindingEvidence); one that
        // names no CTA, or a list whose order cannot be expanded in launch
        // order, cannot stand for the members: per-unit run.
        if (semantic_program && quotient && quotient->Shared() &&
            !BindingEvidence(*semantic_program, *quotient).expandable)
          throw alg::UnitClassRestart{"binding_evidence"};
        auto lowered = semantic_program ? std::move(semantic_program->lowering) : lowerer->Finish();
        // The tensormap input-map overlap, on every path: class mode restarted on
        // it, so this is the per-unit program's (every CTA lowered).
        if (ptx::TensormapInputOverlap(lowered.tensormap_destinations, unit_plan.input_maps))
          lowered.artifact.verification_limitations.push_back({"tensormap_input_overwrite",
              "A tensormap.cp_fenceproxy destination overlaps an input tensor map's location; descriptor "
              "fetches are not modeled as reads."});
        ptx::ValidateConstantInputs(config, lowered.artifact);
        alg::ProfileCall("frontend.release_builder", [&] { lowerer.reset(); });
        alg::ReportRecordGraphSize("source.frontend_ready", lowered.artifact.program.instructions, lowered.artifact.program.edges);
        alg::ReportSourceMemory(lowered.artifact, "frontend.ready");
        metadata = std::move(lowered.metadata);
        metadata.update(input_metadata);
        lowered.artifact.frontend_assumptions_json = input_metadata["frontend_assumptions"].dump();
        lowered.artifact.address_invariants_json =
            config.raw.value("address_invariants", Json::array()).dump();
        tensor_property_inputs = config.fixed_input.value("tensor_property_inputs", Json::array());
        lowered.artifact.tensor_properties_json = tensor_property_inputs.dump();
        // Class mode read the fixed shape-array tables as table atoms (unit_config): the address model
        // values their lookups (TableValuation).
        lowered.artifact.table_atoms_json = table_atoms_json;
        metadata["executed_records"] = records;
        const auto assumptions =
            config.raw.value("semantic_assumptions", Json::object());
        for (auto it = assumptions.begin(); it != assumptions.end(); ++it)
          lowered.artifact.semantic_assumptions[it.key()] =
              it.value().get<bool>();
        alg::OverridePtxAssumptions(lowered.artifact.semantic_assumptions,
                                    options.assumptions);
        // A piecewise or valued-unliftable quotient's complete values are refused lifting before the
        // value side reads the value view (run_values); the synchronization check's memory model reads it only to
        // materialize value-derived addresses. So those are materialized now and the view is
        // released, with the statistics the refusal and the report read. O(value nodes + address
        // bindings), done once instead of after the check.
        if (unit_lifting && (quotient->Piecewise() || quotient->Contained() || unit_lifting->valued_unliftable) &&
            semantic_program &&
            semantic_program->valuesComplete()) {
          const bool canonical = Json::parse(lowered.artifact.address_invariants_json).empty();
          alg::ProfileCall("memory.value_addresses", [&] {
            ptx::MaterializeValueAddresses(lowered.artifact.addresses, semantic_program->values, canonical);
          });
          const auto& values = semantic_program->values;
          released_values = ReleasedValues{values.size(), ValueInstances(values), values.groups().size()};
          alg::ProfileCall("frontend.release_value_view", [&] { semantic_program->values = sentryward::value_ir::Graph{}; });
#if defined(__GLIBC__)
          malloc_trim(0);
#endif
        }

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/source_export.inc"
#endif
        if (!lowered.diagnostics.empty()) {
          // A reachable undefined behavior the lowering decided from concrete values decides the
          // result wherever it is among the diagnostics: an unsupported instruction lowered
          // earlier cannot hide it. Otherwise the first diagnostic does. The deciding one leads
          // the list (readers take the first diagnostic as the cause); the rest keep their order.
          if (const auto deciding = ptx::DecidingLoweringDiagnostic(lowered.diagnostics); deciding) {
            Json first = std::move(lowered.diagnostics[deciding]);
            lowered.diagnostics.erase(deciding);
            lowered.diagnostics.insert(lowered.diagnostics.begin(), std::move(first));
          }
          const auto reason = lowered.diagnostics.front().value("code", std::string("event_view_construction_incomplete"));
          const bool reachable = ptx::LoweringUndefinedBehavior(reason);
          result = {{"status", reachable ? "undefined_behavior" : "event_view_incomplete"},
                    {"stage","event_view_construction"}, {"reason", reason},
                    {"diagnostics", std::move(lowered.diagnostics)},
                    {"verification_performed", false},
                    {"memory_analysis_complete", false}};
          for (const auto &violation : lowered.artifact.source_violations)
            if (violation.code == "cluster_arrive_without_wait") {
              result["status"] = "program_constraint_violation";
              result["stage"] = "cluster_barrier_check";
              result["reason"] = violation.code;
              break;
            }
        } else if (options.extract) {
          result = {{"status", "extraction_complete"},
                    {"verification_performed", false},
                    {"memory_analysis_complete", false},
                    {"verification_limitations", Json::array()},
                    {"source_violations", Json::array()}};
          for (const auto &l : lowered.artifact.verification_limitations)
            result["verification_limitations"].push_back(
                {{"code", l.code}, {"message", l.message}});
          for (const auto &v : lowered.artifact.source_violations)
            result["source_violations"].push_back(
                {{"code", v.code}, {"message", v.message}});
        } else {
          monitor.Frontend(metadata, lowered.artifact);
          monitor.Stage("verification");
          alg::ReachabilityOptions reachability;
          reachability.worker_threads = options.threads;
          alg::MemoryRaceOptions memory;
          memory.collect_duplicate_write_groups=semantic_program.has_value() && options.analyze_values;
          memory.exhaustive = options.exhaustive_memory_search;
          memory.continue_after_race = options.continue_memory_search;
          memory.units = quotient.get();
          // The per-unit path's units: repeated entries are reported per unit.
          const alg::LaunchUnits launch_units{unit_plan.grid, unit_plan.unit_shape};
          if (!quotient && unit_plan.active) memory.launch_units = &launch_units;
          memory.on_search_strategy = [&](const std::string &strategy, bool fallback) {
            monitor.MemoryStrategy(strategy, fallback);
          };
          alg::VerifiedProgramObserver on_verified;
          if (semantic_program && options.analyze_values) {
            on_verified = [&](const alg::SourceArtifact& checked, const alg::PassCertificate& certificate,
                              const alg::MemoryRaceResult& memory_result,
                              const alg::MemoryAccessRecords& accesses, std::string_view verified_report)
                -> std::optional<alg::SameValueFailure> {
              monitor.MemoryVerified(verified_report,
                  kept_verdict ? std::optional(unit_restart->verdict_elapsed_s) : std::nullopt);
              if (unit_lifting && KeepsVerdict(memory_result)) {
                class_verdict.assign(verified_report);
                class_verdict_elapsed_s = monitor.Elapsed();
              }
              // A class member whose values cannot be lifted (UnitLifting::valued_unliftable): the value
              // stage refuses lifting before it reads any memory effect, so none is bound.
              if (unit_lifting && unit_lifting->valued_unliftable) {
                run_values(ptx::CheckedMemoryActions{});
                if (memory_result.same_value_write_groups.empty()) return std::nullopt;
                return SameValueOutcome(eliminated, value_analysis, {});
              }
              // A piecewise quotient's complete values are refused lifting (run_values) before
              // the memory actions are read, so they are not serialized.
              if (class_lifting() && (quotient->Piecewise() || quotient->Contained()) &&
                  semantic_program->valuesComplete()) {
                run_values(ptx::CheckedMemoryActions{});
                if (memory_result.same_value_write_groups.empty()) return std::nullopt;
                return SameValueOutcome(eliminated, value_analysis, {});
              }
              const auto serialization_start=Clock::now();
              auto actions = alg::ProfileCall("value_analysis.memory_effect_binding", [&] {
                alg::ProfileSnapshot();
                return ptx::SerializeCheckedMemory(*semantic_program, checked, certificate, memory_result, accesses, true);
              });
              const auto& s=actions.statistics;
              serialization_statistics={{"event_families",s.event_families},{"value_families",s.value_families},
                  {"relations",s.relations},{"scheduled_batches",s.scheduled_batches},
                  {"logical_events",s.logical_events},{"binding_patterns",s.binding_patterns},
                  {"action_patterns",s.action_patterns},{"logical_actions",s.logical_actions},
                  {"expanded_event_records",s.expanded_event_records},{"reachability_queries",s.reachability_queries}};
              serialization_statistics["domain_refinements"]=s.domain_refinements;
              serialization_statistics["singleton_batches"]=s.singleton_batches;
              serialization_statistics["largest_batch_members"]=s.largest_batch_members;
              serialization_statistics["elapsed_seconds"]=std::chrono::duration<double>(Clock::now()-serialization_start).count();
              serialization_statistics["rss_mib_at_completion"]=Rss()/1048576.0;
              serialization_statistics["peak_rss_mib_at_completion"]=PeakRss()/1048576.0;
              monitor.Serialized(serialization_statistics,actions.complete());

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_save.inc"
#endif

              const auto same_value_writes = std::move(actions.same_value_writes);
              run_values(std::move(actions));
              if (memory_result.same_value_write_groups.empty()) return std::nullopt;
              auto failure = SameValueOutcome(eliminated, value_analysis, same_value_writes);
              if (!failure) monitor.SameValueWritesProven();
              return failure;
            };
          }
          if (options.trusted_reference) {
            monitor.Stage("value_analysis.trusted_reference_order");
            run_values(ptx::BuildTrustedReferenceMemoryActions(*semantic_program, lowered.artifact));
            result = {{"status", value_analysis.value("status", "unsupported") == "expressions_ready"
                                   ? "reference_expressions_ready" : "reference_unsupported"},
                      {"verification_performed", false}, {"memory_analysis_complete", false},
                      {"warnings", Json::array()}, {"warnings_complete", false},
                      {"execution_order_basis", "trusted_reference"}};
          } else {
            const bool canonical_addresses = Json::parse(lowered.artifact.address_invariants_json).empty();
            auto memory_addresses = std::move(lowered.artifact.addresses);
            result = Json::parse(alg::AnalyzeSourceArtifact(
                std::move(lowered.artifact), true, memory, {}, reachability,
                options.sync_timeout, on_verified,
                [addresses = std::move(memory_addresses), &semantic_program, &released_values, canonical_addresses]() mutable {
                  if (semantic_program && !released_values)
                    alg::ProfileCall("memory.value_addresses", [&] {
                      ptx::MaterializeValueAddresses(addresses, semantic_program->values, canonical_addresses);
                    });
                  return std::move(addresses);
                }, kept_verdict ? std::string_view(unit_restart->verdict) : std::string_view()));
            if (result.contains("same_value_writes")) monitor.SameValueWritesSettled(result);
            // Planned founders first: only a clean pass stands (FounderOrderRestart).
            if (founders_first && result.value("status", "") != "race_free") throw alg::UnitClassRestart{"founder_order"};
          }
        }
        result["frontend"] = std::move(metadata);
      }
      } catch (const alg::UnitClassRestart &restart) {
        restart_per_unit(restart);
      }
      } // fresh PTX path; checkpoint replay already supplied the checked value inputs
    } catch (const FileError &error) {
      result = FileFailure(error);
    } catch (const ptx::LayoutError &error) {
      result = {{"status","memory_layout_unsupported"}, {"reason",error.code}, {"details",error.details},
          {"stage","memory_layout"}, {"failure_stage",monitor.stage}, {"message",error.what()},
          {"memory_analysis_complete",false}, {"warnings_complete",false}};
    } catch (const ptx::ParserError &error) {
      result = {{"status","parser_error"}, {"reason","invalid_ptx_syntax"}, {"stage","ptx_parser"},
          {"message",error.what()}, {"location",{{"file",error.file},{"line",error.line},{"column",error.column}}},
          {"memory_analysis_complete",false}, {"warnings_complete",false}};
    } catch (const std::bad_alloc &error) {
      result = {{"status","allocation_failed"}, {"reason","memory_allocation_failed"},
          {"stage",monitor.stage}, {"resource",{{"kind","memory"},{"origin","allocator"},{"cause","unknown"}}},
          {"message",error.what()}, {"memory_analysis_complete",false}, {"warnings_complete",false}};
    } catch (const ptx::ExecutionError &error) {
      input_metadata["required_parameters"] = error.required_parameters;
      result = {{"status","ptx_execution_unsupported"}, {"stage","ptx_execution"}, {"reason",error.code}, {"message",error.what()},
          {"location",{{"file",options.ptx},{"line",error.line},{"column",0}}},
          {"details",{{"required_parameters",error.required_parameters}}},
          {"frontend",input_metadata}, {"warnings",Json::array()}, {"memory_analysis_complete",false}};
      if (error.code == "execution_step_limit_exceeded") {
        result["status"]="analysis_budget_exceeded";
        result["resource"]={{"origin","algorithm_budget"},{"kind","steps_per_thread"},
            {"limit",options.max_steps_per_thread},{"unit","steps"}};
      }
    } catch (const ptx::InputError &error) {
      result = {{"status", error.status}, {"stage","input_configuration"}, {"reason", error.code}, {"path", error.path},
                {"message",error.what()}, {"frontend",input_metadata},
                {"failure_stage", monitor.stage}, {"warnings", Json::array()},
                {"completed_memory_check", false}, {"memory_analysis_complete", false}};
    } catch (const std::exception &error) {
      result = {{"status", "internal_error"}, {"reason","unexpected_exception"},
                {"stage", monitor.stage}, {"failure_stage", monitor.stage},
                {"message", error.what()},
                {"memory_analysis_complete", false},
                {"warnings_complete", false}};
    }
    if (options.analyze_values) {
      if (value_analysis["status"] == "not_run") {
        value_analysis["reason"] = "verification_prerequisite_not_satisfied";
        value_analysis["prerequisite_status"] = result.value("status", "incomplete");
      }
      result["value_analysis"] = value_analysis;
      if(!serialization_statistics.is_null())result["value_analysis"]["serialization"]=serialization_statistics;
      if (options.analyze_scalar_fold) {
        result["value_analysis"]["scalar_fold"] = {
            {"status", "not_run"}, {"reason", "memory_elimination_not_completed"}};
      }
      const auto* graph = semantic_program ? &semantic_program->values :
          semantic_builder ? &semantic_builder->graph() : nullptr;
      if (graph) {
        if (options.analyze_scalar_fold && eliminated && eliminated->complete) {
          try {
            result["value_analysis"]["scalar_fold"] = AnalyzeScalarFold(*graph, *eliminated, options, monitor, expression,
                unit_lifting ? &*unit_lifting : nullptr, reference ? &compared_outputs : nullptr);
          } catch (const alg::UnitClassRestart &restart) {
            // A lifting refusal re-runs per unit from execution.
            restart_per_unit(restart);
          }
        }
        result["value_ir_node_count"] = released_values ? released_values->nodes : graph->size();
        result["value_ir_instance_count"] = released_values ? released_values->instances : ValueInstances(*graph);
        result["value_view_complete"] = semantic_program && semantic_program->valuesComplete();
        result["semantic_program"] = {{"thread_prefix_groups", released_values ? released_values->groups : graph->groups().size()},
            {"execution_cohorts", semantic_program ? semantic_program->execution_cohorts.size() : 0},
            {"synchronization_groups", semantic_program ? semantic_program->synchronization_groups.size() : 0},
            {"synthetic_synchronization_nodes", semantic_program ? semantic_program->synthetic_synchronization_nodes.size() : 0},
            {"memory_bindings", semantic_program ? semantic_program->memory_bindings.size() : 0},
            {"memory_binding_stored_rows", semantic_program ? semantic_program->memory_bindings.storedBindingCount() : 0},
            {"memory_binding_templates", semantic_program ? semantic_program->memory_bindings.bindingTemplateCount() : 0},
            {"memory_linking_templates_used", semantic_program && semantic_program->memory_linking_templates_used},
            {"mapping_complete", semantic_program && semantic_program->mappingComplete()},
            {"synchronization_groups_materialized", semantic_program && semantic_program->synchronization_groups_materialized}};
        if(replayed_correctness)result["semantic_program"]={{"thread_prefix_groups",graph->groups().size()},
            {"status","restored_value_graph_only"},{"source_bindings","not_rebuilt"}};
        if (semantic_program) {
          auto& info = result["semantic_program"];
          if (released_values) info["value_view_released_before_verification"] = true;
          info["value_diagnostic_count"] = semantic_program->value_diagnostics.size();
          info["binding_diagnostic_count"] = semantic_program->diagnostics.size();
          info["value_diagnostics"] = Json::array();
          info["binding_diagnostics"] = Json::array();
          for (const auto& diagnostic : semantic_program->value_diagnostics) {
            info["value_diagnostics"].push_back({{"code", diagnostic.code}, {"message", diagnostic.message},
                {"line", diagnostic.origin.source_line.value_or(0)}});
          }
          if (quotient && quotient->Shared()) {
            // Weighted and expanded in launch order (checked expandable after Finish).
            const auto expansion = BindingEvidence(*semantic_program, *quotient, &info["binding_diagnostics"]);
            info["binding_diagnostic_count"] = expansion.count;
          } else
          for (const auto& diagnostic : semantic_program->diagnostics) {
            info["binding_diagnostics"].push_back({{"code", diagnostic.code}, {"message", diagnostic.message}});
          }
        }

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/value_export.inc"
#endif
      }
    }
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/branch_capture_result.inc"
#endif
    else if (values) {
      const auto edges = values->lowerTypeConversions();
      result["type_edges"] = {{"removed_nodes", edges.nodes}, {"converted_instances", edges.instances}};
      result["value_ir_node_count"] = values->size();

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/dot_export.inc"
#endif
      result["value_ir_node_count"] = values->nodes().size();
      result["value_view_complete"] = result.value("status", "") == "value_view_complete";
      result["verification_performed"] = false;
    }
    const auto verified = result.value("verification_status", result.value("status", ""));
    // Every kernel verification reports its equivalence check. Without a reference expression or a
    // reported reference failure, and without --require-equivalence, the check is skipped and a
    // kernel whose synchronization check passed keeps that verdict, unless a value stage that ran
    // found reachable undefined behavior (memory elimination runs on request, --analyze-values; the
    // scalar fold, which checks sparse-metadata validity, only with --analyze-scalar-fold).
    const bool skip_equivalence = !options.reference_stdin && !options.require_equivalence;
    // The first value-stage failure, if any. A not-run scalar projection must
    // not hide a precise lowering/elimination diagnostic (e.g. atomic add), nor
    // relabel that failure as a generic scalar/Fold blockage. Without a
    // reference a complete fold keeps no artifact, so "ready" is no failure.
    const auto value_failure = [&]() -> std::optional<Json> {
      if (expression || !result.contains("value_analysis")) return std::nullopt;
      const auto& analysis = result["value_analysis"];
      auto folded = analysis.value("scalar_fold", Json::object());
      auto stage = std::string("value_analysis.scalar_fold");
      if (folded.value("status", "not_run") == "not_run" &&
          analysis.value("status", "not_run") != "expressions_ready") {
        folded = analysis;
        folded.erase("scalar_fold");
        stage = analysis.value("stage", "value_analysis.memory_elimination");
      }
      const auto state = folded.value("status", "not_run");
      if (!reference && state == "ready") return std::nullopt;
      const auto diagnostic = folded.value("diagnostic", Json::object());
      const auto reason = diagnostic.value("code", folded.value("reason", "expression_not_ready"));
      // The public diagnostic's text: the failure's own message, else that of its
      // entry in the stage's (normalized) diagnostics.
      std::string message;
      const auto text = [](const Json& object) {
        return object.is_object() && object.contains("message") && object["message"].is_string()
            ? object["message"].get<std::string>() : std::string();
      };
      message = text(diagnostic);
      if (message.empty()) message = text(folded);
      if (message.empty() && folded.contains("diagnostics") && folded["diagnostics"].is_array())
        for (const auto& entry : folded["diagnostics"])
          if (entry.is_object() && entry.value("code", "") == reason && !(message = text(entry)).empty()) break;
      Json failure{
          {"status", state == "analysis_budget_exceeded" ? "analysis_budget_exceeded" : state == "representation_limit_exceeded" ? "unsupported" :
                     state == "unsupported" ? "unsupported" : state == "undefined_behavior" ? "undefined_behavior" : "blocked"},
          {"reason", reason}, {"stage", stage}, {"expression_diagnostic", folded}};
      if (!message.empty()) failure["message"] = std::move(message);
      if (state == "undefined_behavior" && diagnostic.contains("undefined_behavior"))
        failure["details"] = diagnostic["undefined_behavior"];
      return failure;
    };
    if (verified != "race_free" || !result.value("memory_analysis_complete", false))
      result["equivalence"] = {{"status","blocked"},{"reason","verification_prerequisite_not_satisfied"}};
    else if (skip_equivalence) {
      Json skipped{{"status","skipped"},{"reason","reference_expression_not_supplied"},
                   {"message", std::string("Equivalence checking was skipped: no reference expression was supplied.") +
                       (options.analyze_scalar_fold ? "" : " The scalar fold did not run, so its checks (sparse-metadata "
                                                         "validity among them) were not run.")}};
      auto failure = value_failure();
      if (failure && failure->value("status", "") == "undefined_behavior") {
        result["equivalence"] = std::move(*failure);
        result["equivalence"]["diagnostics"] = sentryward::results::Report(skipped, "equivalence")["diagnostics"];
      } else result["equivalence"] = std::move(skipped);
    } else {
      // A failed value stage is reported before a missing or failed reference:
      // supplying the reference would not change the result. Both are diagnosed.
      if (auto failure = value_failure()) {
        result["equivalence"] = std::move(*failure);
        if (!reference)
          result["equivalence"]["diagnostics"] = sentryward::results::Report(reference_error, "equivalence")["diagnostics"];
      }
      else if (!reference) result["equivalence"] = reference_error;
      else {
        // All diagnostic exports are complete. The checker needs only scalar/Fold
        // objects, not a second live copy of the memory-elimination pipeline.
        eliminated.reset();
        semantic_builder.reset();
        semantic_program.reset();
        monitor.Stage("equivalence.object_comparison");
        const auto started = Clock::now();
        rg::ContextBuilder context;
        context = [&](rg::Arena&, const Json& mapping,
            rg::EquivalenceContext& scope, Json& receipt) {
          std::string input_hash, provenance_hash;
          const auto input = rg::fixed_input::metadata(options.config, inputs.File(options.config), input_hash);
          if(input.value("shape_arrays",Json::array()).empty())return;
          if (!mapping.contains("provenance") && !options.reference_provenance.empty()) {
            const auto provenance = rg::fixed_input::metadata(
                options.reference_provenance, inputs.File(options.reference_provenance), provenance_hash);
            scope.fixed_inputs = rg::fixed_input::decode(input, result, provenance, mapping,
                input_hash, provenance_hash);
          } else scope.fixed_inputs = rg::fixed_input::decode_binding(input,result,mapping,input_hash,provenance_hash);
          for (auto& fixed : scope.fixed_inputs)
            fixed.reason = "current verified fixed-read-only input; input sha256=" + input_hash +
                           "; provenance sha256=" + provenance_hash;
          receipt = {{"kind","verified_current_fixed_input"},{"input_sha256",input_hash},
                     {"provenance_sha256",provenance_hash},{"elements",scope.fixed_inputs.size()}};
        };
        AppendTensorPropertyContract(equivalence_mapping, tensor_property_inputs);
        if (const auto* contents = inputs.File(options.config)) {
          std::string ignored_hash;
          const auto input = rg::fixed_input::metadata(options.config, contents, ignored_hash);
          AppendKernelOnlyFixedInputs(equivalence_mapping, input, result);
        }
        result["equivalence"] = alg::ProfileCall("equivalence.object_comparison", [&] {
          return rg::compareArtifact(std::move(*reference), *expression,
              std::move(equivalence_mapping),
              {.seconds = options.equivalence_timeout,
               .floating_range_premise = effective_assumptions.at("floating_values_in_range"),
               .kernel_hardware_values = true},
              context).report;
        });
        result["equivalence"]["comparison_elapsed_seconds"] = std::chrono::duration<double>(Clock::now()-started).count();
        result["equivalence"]["transport"] = "in_memory_cpp_objects";
        // The software exp2 abstraction of the value passes (value_analysis::abstractSoftwareExp2,
        // pass "software_exp2") changes the kernel side the comparison sees, so it is listed with the
        // comparison's own idealizations, only when the pass rewrote at least one node. O(passes).
        if (result.contains("value_analysis") && result["value_analysis"].contains("passes") &&
            result["equivalence"].contains("idealizations_applied") &&
            result["equivalence"]["idealizations_applied"].is_array())
          for (const auto &pass : result["value_analysis"]["passes"])
            if (pass.value("name", std::string()) == "software_exp2" &&
                pass.value("rewritten_nodes", std::size_t{0}) > 0)
              result["equivalence"]["idealizations_applied"].push_back("complete binary32 software exp2 routine of the kernel (argument clamp max(x, -127), round-down integer split, cubic RN-FMA polynomial, exponent-bit assembly) is ideal Exp2 of its argument; the polynomial error and the clamp region x < -127 are counted as approximation");
        result["equivalence"]["numerical_status"] = result["equivalence"]["status"];
        if (result["equivalence"].value("status", "") == "equivalent" &&
            result["equivalence"].value("access_validity", "") == "not_proven") {
          result["equivalence"]["status"] = "not_proven";
          result["equivalence"]["reason"] = "input_access_validity_not_proven";
        }
      }
    }
    if (options.analyze_values || result.contains("value_analysis")) {
      result["value_analysis"]["equivalence_check"] = result["equivalence"].value("result", result["equivalence"].value("status", std::string("not_run")));
      result["value_analysis"].erase("equivalence_reason");
    }
    total.Stop();
    monitor.Stop();
    // Normal completion and deadline reports expose the same completed
    // verification checkpoint independently of the numerical outcome.
    if (monitor.frontend_checkpoint.contains("verification_status")) {
      result["verification_status"] = monitor.frontend_checkpoint["verification_status"];
      result["verification_elapsed_seconds"] = monitor.frontend_checkpoint["verification_elapsed_seconds"];
    }
    if (monitor.memory_limit_mib &&
        std::max(monitor.peak.load(), monitor.peak_footprint.load()) > *monitor.memory_limit_mib * 1024 * 1024) {
      Json partial = std::move(result);
      result = {{"schema", "spverify.native-ptx-run/v1"},
                {"status", "memory_limit_exceeded"},
                {"resource", monitor.MemoryResource(*monitor.memory_limit_mib)},
                {"message", "Memory budget exceeded (RSS or physical footprint) before final monitor sample"},
                {"partial_report", std::move(partial)},
                {"memory_analysis_complete", false},
                {"warnings_complete", false}};
      result.update(monitor.frontend_checkpoint);
      if(options.require_equivalence || options.reference_stdin) {
        if(result["partial_report"].contains("equivalence")) {
          result["equivalence"]=std::move(result["partial_report"]["equivalence"]);
          result["partial_report"].erase("equivalence");
        } else result["equivalence"]={{"status","blocked"},{"reason","verification_prerequisite_not_satisfied"},
          {"prerequisite_status","memory_limit_exceeded"},{"stage",monitor.stage}};
      }
    }
    if (options.value_thread_range) {
      result["analysis_scope"] = {{"kind","thread_subset"},
          {"first",options.value_thread_range->first},
          {"count",options.value_thread_range->second-options.value_thread_range->first},
          {"whole_launch_verified",false}};
    }
    result["template_execution"]={{"completed_ctas",template_ctas},
        {"template_steps",template_execution.steps},{"logical_thread_steps",template_execution.logical_steps},
        {"splits",template_execution.splits},{"value_nodes",template_execution.value_nodes},
        {"record_nodes",template_execution.record_nodes},
        {"record_bindings",template_execution.record_bindings},
        {"coordinate_bindings",template_execution.coordinate_bindings},
        {"path_entries",template_execution.path_entries},
        {"thread_memberships",template_execution.thread_memberships}};
    result["thread_record_compression"] = {
        {"enabled", options.compress_thread_records}, {"completed_ctas", ctas},
        {"logical_records", records}, {"intern_calls", record_compression.record_count},
        {"templates", record_compression.template_count},
        {"unique_bindings", record_compression.unique_binding_count},
        {"record_handle_bytes", sizeof(ptx::ExecutedInstruction)},
        {"estimated_pool_bytes_sum", record_compression.estimated_owned_bytes}};
    result["pipeline_elapsed_seconds"] = monitor.Elapsed();
    result["semantic_assumptions"] = effective_assumptions;
    // Floating values stay in range: the equivalence import uses the premise where it drops a range
    // bound (counter range_premise_bounds_dropped). Unused only when the comparison imported every
    // output family without a drop; unmeasured (null) when it did not run or stopped early.
    std::optional<Json> floating_range_used;
    if (effective_assumptions.at("floating_values_in_range")) {
      floating_range_used = Json(nullptr);
      if (const auto equivalence = result.find("equivalence");
          equivalence != result.end() && equivalence->is_object() && equivalence->contains("counters")) {
        if (equivalence->at("counters").value("range_premise_bounds_dropped", std::uint64_t{0}) > 0)
          floating_range_used = Json(true);
        else if (equivalence->contains("total_output_families") &&
                 equivalence->value("finished_families", Json()) == equivalence->at("total_output_families"))
          floating_range_used = Json(false);
      }
    }
    if (!result.contains("assumptions")) {
      result["assumptions"] = Json::array();
      for (const auto& entry : alg::AssumptionPool()) {
        Json triggered = effective_assumptions.at(entry.key) ? Json(nullptr) : Json(false);
        if (effective_assumptions.at(entry.key) && result.value("status", "") == "extraction_complete") {
          const auto facts = input_metadata.value("frontend_assumptions", Json::object());
          if (entry.key == "representative_election") triggered = facts.value("election_uses", 0) > 0;
          if (entry.key == "clc_always_fails") triggered = facts.value("clc_uses", 0) > 0;
          if (entry.key == "representative_branches") triggered = facts.value("representative_branch_uses", 0) > 0;
          if (entry.key == "bounded_computation_loops") triggered = facts.value("bounded_loop_uses", 0) > 0;
          if (entry.key == "order_independent_atomics") triggered = facts.value("atomic_uses", 0) > 0;
        }
        if (entry.key == "floating_values_in_range" && floating_range_used) triggered = *floating_range_used;
        result["assumptions"].push_back({{"label",entry.label},{"key",entry.key},{"text",entry.text},
            {"enabled",effective_assumptions.at(entry.key)},
            {"triggered",triggered}});
      }
    } else if (floating_range_used && result["assumptions"].is_array()) {
      // The verification report's list (source_json.cpp RenderReport) saw no comparison.
      for (auto& entry : result["assumptions"])
        if (entry.value("key", "") == "floating_values_in_range") entry["triggered"] = *floating_range_used;
    }
    const auto reason = result.value("reason", "");
    if (reason == "branch_not_static" || reason == "missing_parameter_value") {
      result["assumption_hints"] = Json::array();
      for (const auto& entry : alg::AssumptionPool())
        if ((entry.key == "representative_branches" || entry.key == "bounded_computation_loops") &&
            !effective_assumptions.at(entry.key))
          result["assumption_hints"].push_back({{"label",entry.label},{"key",entry.key},
            {"suggested_value",true},{"message","Supply the missing control input for exact analysis. This assumption applies only if CFG classification identifies the corresponding structured control; polling and unclassified loops remain unsupported."}});
    }
    for (const auto& entry : alg::AssumptionPool())
      if (reason == "assumption_required_" + entry.key)
        result["assumption_hints"] = Json::array({{{"label",entry.label},{"key",entry.key},
            {"suggested_value",true},{"message",result.value("message", "")}}});
    result["rss_limit_mib"] = options.rss_mib ? Json(*options.rss_mib) : Json(nullptr);
    result["peak_rss_mib"] = monitor.peak.load() / 1048576.0;
    result["peak_footprint_mib"] = monitor.peak_footprint.load() / 1048576.0;
    result["implementation"] = "cpp";
    result["wellsync_timeout_seconds"] = options.sync_timeout;
    result["whole_pipeline_timeout_seconds"] = options.timeout;
    monitor.RecordUnitRestart(result);
    const auto exit_status = result.value("status", "");
    result = sentryward::results::PublicReport(
        std::move(result), options.extract ? "extraction" : options.semantic ? "semantic_analysis" : "kernel_verification");
    Write(options.report, result);

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/branch_capture_export.inc"
#endif

    std::cout << result.dump(2) << '\n';
    auto status = exit_status;
    if (status == "memory_limit_exceeded")
      return 137;
    if (status == "invalid_configuration" || status == "parser_error" || status == "input_file_error" || status == "output_file_error") return 1;
    return status == "internal_error" ||
                   status == "value_view_incomplete" ||
                   status == "branch_scope_incomplete" ||
                   status == "event_view_incomplete"
               ? 2
               : 0;
  } catch (const FileError &error) {
    std::cout << sentryward::results::PublicReport(FileFailure(error), "kernel_verification").dump() << '\n';
    return 1;
  } catch (const std::bad_alloc &error) {
    std::cout << sentryward::results::PublicReport(Json{{"schema","spverify.native-ptx-run/v1"},{"status","allocation_failed"},
        {"resource",{{"kind","memory"},{"origin","allocator"},{"cause","unknown"}}},
        {"message",error.what()},{"memory_analysis_complete",false},{"warnings_complete",false}}, "kernel_verification").dump() << '\n';
    return 2;
  } catch (const std::exception &error) {
    const auto status = parsing_arguments ? "invalid_configuration" : "internal_error";
    std::cout << sentryward::results::PublicReport(Json{{"schema","spverify.native-ptx-run/v1"},{"status",status},
        {"stage",parsing_arguments ? "cli_arguments" : "pipeline"},
        {"reason",parsing_arguments ? "invalid_cli_arguments" : "unexpected_exception"},
        {"message",error.what()},{"memory_analysis_complete",false},{"warnings_complete",false}}, "kernel_verification").dump() << '\n';
    std::cerr << "sentryward-ptx: " << error.what() << '\n';
    return parsing_arguments ? 1 : 2;
  }
}
