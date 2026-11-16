#pragma once
#include "run_support.hpp"

namespace {
Json AnalyzeScalarFold(const sentryward::value_ir::Graph& values,
                       const sentryward::value_analysis::MemoryEliminationResult& eliminated,
                       const Options& options, Monitor& monitor,
                       std::optional<sentryward::scalar_fold::Artifact>& artifact,
                       const sentryward::value_analysis::UnitLifting* lifting,
                       const std::map<std::string, sentryward::scalar_fold::ComparedOutput>* compared) {
  namespace sf = sentryward::scalar_fold;
  Json report;
  try {
    monitor.Stage("value_analysis.scalar_fold");
    auto folded = alg::ProfileCall("value_analysis.scalar_fold", [&] {
      sentryward::value_analysis::ScalarFoldOptions fold_options;
      fold_options.compact_output_families=true;
      fold_options.unit_lifting=lifting;
      fold_options.compared_outputs=compared;
      return sentryward::value_analysis::lowerScalarFold(values, eliminated,fold_options);
    });
    // Reachable undefined behavior: its witness is an instance of an analysed unit, so it
    // is one of the launch whether or not the representatives stand for their members.
    const bool undefined = folded.diagnostic &&
        sentryward::value_analysis::FoldFailureUndefinedBehavior(folded.diagnostic->code) &&
        folded.diagnostic->undefined_behavior;
    if (lifting && !undefined) {
      if (folded.lifting.refused) {
        // The first-item rule: a class-mode fold failure that is provably the per-unit
        // run's first failure is reported as computed; any other refusal restarts.
        bool matches = false;
        if (folded.diagnostic) matches = FirstFailureMatchesPerUnit(values, eliminated, folded, *lifting);
        if (!matches) return RefuseLifting(ValueInstances(values), *lifting, "lowering", LiftingStopOf(folded.diagnostic));
      }
      // The fold's own row maps (audited row widths) after the value passes' ones.
      if (RowModulusViolation(*lifting)) return RefuseLifting(ValueInstances(values), *lifting, "lowering");
    }
    report = {{"status", folded.complete ? "ready" : "unsupported"},
              {"source_outputs", folded.source_outputs},
              {"construction_collections", folded.construction_collections},
              {"peak_construction_nodes", folded.peak_construction_nodes},
              {"shared_view_projection_hits", folded.shared_view_projection_hits}};
    if (folded.narrowing_slice_projections)
      report["narrowing_slice_projections"] = folded.narrowing_slice_projections;
    if (folded.diagnostic) {
      const auto& diagnostic = *folded.diagnostic;
      report["diagnostic"] = {{"code", diagnostic.code}, {"message", diagnostic.message}};
      if (diagnostic.code.ends_with("_limit_exceeded")) {
        report["status"] = "analysis_budget_exceeded";
        report["resource"] = {{"origin","algorithm_budget"},{"kind",diagnostic.code}};
      } else if (diagnostic.code == "index_representation_overflow") {
        report["status"] = "representation_limit_exceeded";
        report["resource"] = {{"origin","representation"},{"kind","integer_range"}};
      } else if (diagnostic.code == "invalid_analysis_budget") report["status"] = "invalid_configuration";
      else if (diagnostic.code == "source_dependency_cycle") report["status"] = "internal_error";
      else if (undefined) report["status"] = "undefined_behavior";
      if (diagnostic.operation) {
        const auto& operation = *diagnostic.operation;
        report["diagnostic"]["value_node"] = operation.node;
        report["diagnostic"]["instance"] = operation.instance;
        const auto& origin = values.node(operation.node).instances().at(operation.instance).origin;
        if (origin.source_line) report["diagnostic"]["line"] = *origin.source_line;
        report["diagnostic"]["instruction"] = origin.source_text;
      }
      if (undefined) {
        const auto& w = *diagnostic.undefined_behavior;
        Json witness = {{"condition", w.condition}, {"isa_reference", w.isa_reference},
            {"isa_section", w.isa_section}, {"isa_wording", w.isa_wording},
            {"source", w.source}, {"operand_byte", w.operand_byte},
            {"field", w.field == 0 ? "low" : "high"}, {"witness_value", w.value},
            {"witness_bits", "0b" + std::bitset<4>(w.value).to_string()}, {"defined_values", w.defined_values}};
        if (w.source == "input_object") {
          witness["object"] = w.object;
          witness["property"] = w.property;
          if (w.property == "field_values") witness["declared_values"] = w.declared_values;
          if (w.object_byte_offset) witness["object_byte_offset"] = *w.object_byte_offset;
        } else if (w.source == "unwritten_tensor_memory")
          witness["note"] = "no store covers the metadata bytes before this MMA reads them, so their contents are unspecified";
        if (report["diagnostic"].contains("line")) witness["line"] = report["diagnostic"]["line"];
        if (report["diagnostic"].contains("instruction")) witness["instruction"] = report["diagnostic"]["instruction"];
        report["diagnostic"]["undefined_behavior"] = std::move(witness);
      }
    }
    if (!folded.complete) {
      report["partial_node_count"] = folded.graph.size();
      report["partial_output_count"] = folded.outputs.size();
      return report;
    }
    std::vector<sf::Ref> roots;
    std::vector<std::uint32_t> root_binders;
    roots.reserve(folded.outputs.size());
    for (const auto& output : folded.outputs) { roots.push_back(output.value);root_binders.push_back(0); }
    std::uint64_t logical_outputs=folded.outputs.size();
    for (const auto& family:folded.output_families) {
      roots.push_back(family.body);root_binders.push_back(family.shape.size());
      std::uint64_t count=1;
      for(auto extent:family.shape) {
        if(extent && count>UINT64_MAX/extent)throw std::overflow_error("output count overflow");
        count*=extent;
      }
      if(count>UINT64_MAX-logical_outputs)throw std::overflow_error("output count overflow");
      logical_outputs+=count;
    }
    const auto stats = sf::statistics(folded.graph, roots,0,root_binders);
    report["output_count"] = logical_outputs;
    report["output_root_count"] = roots.size();
    report["output_family_count"] = folded.output_families.size();
    report["output_families_merged"] = folded.output_families_merged;
    // Reported only when it happened, so that reports of other lowerings are unchanged.
    if (folded.pending_output_merges) report["pending_output_merges"] = folded.pending_output_merges;
    report["spatial_families_projected"] = folded.spatial_families_projected;
    report["spatial_outputs_projected"] = folded.spatial_outputs_projected;
    report["spatial_template_operations"] = folded.spatial_template_operations;
    report["statistics"] = {
        {"nodes", stats.nodes}, {"storage_nodes", folded.graph.size()},
        {"nodes_before_final_collection", folded.nodes_before_final_collection},
        {"nodes_reclaimed", folded.nodes_reclaimed},
        {"edges", stats.edges}, {"constants", stats.constants},
        {"parameters", stats.parameters}, {"initial_cells", stats.initial_cells},
        {"indices", stats.indices}, {"operations", stats.operations}, {"folds", stats.folds},
        {"unique_fold_bodies", stats.unique_fold_bodies},
        {"max_fold_nesting", stats.max_fold_nesting},
        {"range_descriptions", stats.range_descriptions},
        {"logical_iterations", stats.logical_iterations},
        {"logical_iterations_overflow", stats.logical_iterations_overflow}};

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/fold_export.inc"
#endif
    if (options.reference_stdin) {
      artifact.emplace();
      artifact->graph = std::move(folded.graph);
      artifact->output_families = std::move(folded.output_families);
      for (const auto& output : folded.outputs)
        artifact->outputs.push_back({output.region.allocation, output.region.offset,
                                    output.region.bytes, output.value});
    }
  } catch (const std::bad_alloc& error) {
    artifact.reset();
    report = {{"status","allocation_failed"}, {"stage","scalar_expression_construction"},
        {"resource",{{"kind","memory"},{"origin","allocator"},{"cause","unknown"}}}, {"message",error.what()}};
  } catch (const std::exception& error) {
    artifact.reset();
    report = {{"status", "internal_error"}, {"stage","scalar_expression_construction"}, {"reason", "unexpected_exception"}, {"message", error.what()}};
  }
  return report;
}
// The bounds of tensor_properties are equivalence premises too, as input-contract ranges of
// the mapped objects on both sides (the same trusted premises the race check and memory
// elimination use; injectivity is not one). Records without a bound beyond their type's
// range, objects the mapping does not name on both sides, and objects/types the mapping
// already declares are skipped.
// Shape arrays (verified fixed read-only inputs) that neither side of the mapping names: the
// reference reads them as no tensor and receives their values only as scalars bound to these same
// shape_arrays records. Each is named on the kernel side only, with a dense one-dimensional layout
// over its verified read-only region and a kernel-only shared identity, and listed in
// right_inputs.fixed_only, so the comparison substitutes its verified values in kernel reads
// (fixed_input::decode_verified). A read it cannot substitute stays a kernel-only input that no
// reference input equals. O(arrays + regions).
void AppendKernelOnlyFixedInputs(Json& mapping, const Json& input, const Json& report) {
  const auto arrays = input.value("shape_arrays", Json::array());
  if (!arrays.is_array() || arrays.empty() || !mapping.is_object() ||
      !mapping.contains("left_inputs") || !mapping.contains("right_inputs")) return;
  if (!report.contains("frontend") || !report["frontend"].contains("frontend_assumptions")) return;
  const auto& assumptions = report["frontend"]["frontend_assumptions"];
  if (!assumptions.contains("shape_input_contract") ||
      !assumptions["shape_input_contract"].contains("regions")) return;
  std::map<std::string, const Json*> regions;
  for (const auto& region : assumptions["shape_input_contract"]["regions"])
    if (region.is_object() && region.contains("resource") && region["resource"].is_string())
      regions.emplace(region["resource"].get<std::string>(), &region);
  auto& right = mapping["right_inputs"];
  const auto& left = mapping["left_inputs"];
  const auto named = [](const Json& side, const std::string& raw) {
    return side.contains("allocations") && side["allocations"].is_object() && side["allocations"].contains(raw);
  };
  std::set<std::string> shared;
  for (const Json* side : std::array<const Json*, 2>{&left, &right})
    if (side->contains("allocations") && side->at("allocations").is_object())
      for (const auto& item : side->at("allocations").items())
        if (item.value().is_string()) shared.insert(item.value().get<std::string>());
  const auto natural = [](const Json& value) -> std::optional<std::uint64_t> {
    if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<std::int64_t>() < 0)) return std::nullopt;
    return value.get<std::uint64_t>();
  };
  for (const auto& array : arrays) {
    if (!array.is_object() || !array.contains("parameter")) continue;
    const auto parameter = natural(array["parameter"]), offset = natural(array.value("offset", Json(0)));
    if (!parameter || !offset) continue;
    const auto found = regions.find("input.arg" + std::to_string(*parameter) + "." + std::to_string(*offset));
    if (found == regions.end()) continue;
    const auto& region = *found->second;
    const auto object = region.value("allocation", std::string());
    const auto type = array.value("element_type", std::string("s32"));
    const std::uint64_t bytes = type.ends_with("64") ? 8 : 4;
    const auto first = natural(region.value("byte_offset", Json(0))), extent = natural(region.value("bytes", Json(0)));
    if (!first || !extent) continue;
    const auto begin = *first, size = *extent;
    const auto raw = sentryward::value_analysis::AllocationIdentity("global", object);
    if (object.empty() || !size || begin % bytes || size % bytes || named(left, raw) || named(right, raw)) continue;
    const auto identity = "kernel_fixed_input:" + object;
    if (!shared.insert(identity).second) continue;
    const auto count = (begin + size) / bytes;
    right["allocations"][raw] = identity;
    right["layouts"][raw] = {{"source_offset", 0}, {"target_offset", 0}, {"element_bytes", bytes},
        {"shape", Json::array({count})}, {"source_strides_bytes", Json::array({bytes})},
        {"target_strides_bytes", Json::array({bytes})}};
    right["fixed_only"].push_back(raw);
  }
}
void AppendTensorPropertyContract(Json& mapping, const Json& properties) {
  if (!properties.is_array() || properties.empty() || !mapping.is_object()) return;
  auto& context = mapping["checker_context"];
  if (!context.is_object()) context = Json::object();
  auto& contract = context["input_contract"];
  if (!contract.is_array()) contract = Json::array();
  std::set<std::pair<std::string, std::string>> declared;
  for (const auto& entry : contract)
    if (entry.is_object() && entry.contains("object") && entry.contains("element_type"))
      declared.emplace(entry["object"].get<std::string>(), entry["element_type"].get<std::string>());
  const auto mapped = [&](const char* side, const std::string& identity) -> std::optional<std::string> {
    if (!mapping.contains(side) || !mapping[side].contains("allocations")) return std::nullopt;
    const auto& table = mapping[side]["allocations"];
    const auto found = table.find(identity);
    if (found == table.end() || !found->is_string()) return std::nullopt;
    return found->get<std::string>();
  };
  for (const auto& record : properties) {
    const auto object = record.at("object").get<std::string>(), type = record.at("element_type").get<std::string>();
    const auto bits = record.at("element_bits").get<unsigned>();
    const bool sign = record.at("signed").get<bool>();
    const auto lower = ptx::JsonInteger(record.at("lower_bound")), upper = ptx::JsonInteger(record.at("upper_bound"));
    const ptx::Integer type_lower = sign ? ptx::Integer(-(ptx::Integer(1) << (bits - 1))) : ptx::Integer(0);
    const ptx::Integer type_upper = (ptx::Integer(1) << (bits - unsigned(sign))) - 1;
    if (lower <= type_lower && upper >= type_upper) continue;
    const auto identity = sentryward::value_analysis::AllocationIdentity("global", object);
    const auto left = mapped("left_inputs", identity), right = mapped("right_inputs", identity);
    if (!left || !right || *left != *right || !declared.emplace(*left, type).second) continue;
    contract.push_back({{"object", *left}, {"element_type", type}, {"source", "tensor_properties"},
        {"region", "whole_object"}, {"min_inclusive", record.at("lower_bound")}, {"max_inclusive", record.at("upper_bound")}});
  }
  if (contract.empty()) context.erase("input_contract");
  if (context.empty()) mapping.erase("checker_context");
}
} // namespace
