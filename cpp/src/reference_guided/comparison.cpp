#include "checker.hpp"
#include "importer.hpp"
#include "fixed_input_context.hpp"
#include "welford_pattern.hpp"
#include "online_softmax_pattern.hpp"
#include "linear_reduction_pattern.hpp"
#include "exponential_equivalence.hpp"
#include "exact_proof_reuse.hpp"
#include "finite_state_reuse.hpp"
#include "unary_reduction_composition.hpp"
#include "comparison.hpp"
#include "context_report.hpp"
#include "caller_context.hpp"
#include "affine_output_dispatch.hpp"
#include "finite_bilinear_projection.hpp"
#include <filesystem>
#include <iostream>
#include <sys/resource.h>
using namespace rg;
ComparisonResult rg::compare(Document left, Document right, J mapping, const ComparisonOptions& options,
    EquivalenceContext context, InputContract contract, const ContextBuilder& build_context) {
  auto begin = std::chrono::steady_clock::now();
  auto elapsed = [&]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         begin)
        .count();
  };
  std::optional<PreparedContext> prepared;
  Arena a(options.seconds);
  Checker checker(a, true);
  checker.scalar.profiling=options.ep_profile;
  checker.scalar.reuse_cache=options.ep_reuse_cache;
  // This is a memoized DAG traversal, not a combinatorial search. Production
  // uses the Arena deadline and process memory limit; an arbitrary visit cap
  // must not truncate a valid large certificate. Explicit diagnostic budgets
  // remain supported by ExactProofReuse and its negative tests.
  ExactProofReuse proved_equalities(checker,std::numeric_limits<size_t>::max());
  J r = {{"transport", "in_memory_objects"},
         {"status", "not_proven"},
         {"scope", "mapped_expression_equivalence"},
         {"engine", "reference_guided_cpp"},
         {"semantics", "ideal_real numerical contract; exact word/index "
                       "import; no added assumptions"},
         {"context_policy",
          "sidecar contexts not consumed; mapping premises retained"},
         {"rows", J::array()}};
  std::string phase = "load";
  size_t passed = 0, finished = 0;
  std::vector<Id> roots;
  try {
    if (left.meta.value("numerical_semantics", "") != "ideal_real" ||
        right.meta.value("numerical_semantics", "") != "ideal_real")
      throw Unsupported("this checker requires explicit ideal_real semantics "
                        "on both cached expressions");
    if (!mapping.contains("outputs") ||
        !mapping["outputs"].contains("tensors") ||
        mapping["outputs"]["tensors"].empty())
      throw Unsupported("no mapped output families; an empty comparison is not "
                        "a completed proof");
    r["raw_nodes"] = {left.nodes.size(), right.nodes.size()};
    if(right.meta.contains("source_semantics_recovery"))
      r["source_semantics_recovery"]=right.meta["source_semantics_recovery"];
    if(mapping.contains("checker_context") && mapping.at("checker_context").contains("input_contract")) {
      if(!contract.rules.empty())throw Unsupported("provide one input contract, not both mapping and API contracts");
      contract=InputContract::parse(mapping.at("checker_context").at("input_contract"));
    }
    contract.validate_mapping(mapping);
    if (!contract.rules.empty()) r["input_contract"]=contract.declaration;
    for(const auto &[key,rule]:contract.rules) {
      if(rule.range) context.assumptions.emplace_back(std::in_place_type<InputElementAssumption>,InputElementAssumption{
          rule.object,rule.element_type,Interval{rule.range->first,rule.range->second,true,true},rule.source});
      else context.assumptions.emplace_back(std::in_place_type<InputElementAssumption>,InputElementAssumption{rule.object,rule.element_type,Finite{},rule.source});
    }
    appendCallerPredicates(a,mapping,context);
    if (build_context) {
      phase = "native_context";
      build_context(a, mapping, context, r["native_context"]);
      r["context_policy"] = "explicit native caller premises; no implied provenance certification";
      r["semantics"] = "ideal_real numerical contract; exact word/index import; conditional on reported native premises, fixed input values and any input_contract";
    }
    bind_context_inputs(mapping,context);
    prepared.emplace(a,context);
    r["context_declaration"]=contextDeclaration(a,context);
    r["mapping_input_memory_constraints"]=mapping.value("input_memory_constraints",J::object());
    phase = "scope_analysis";
    Importer l(a, std::move(left), mapping, "left", *prepared),
        o(a, std::move(right), mapping, "right", *prepared);
    FiniteStateReuse piece_equalities(checker, mapping);
    UnaryReductionComposition unary_reductions(checker);
    AffineStatePattern affine_states(checker);
    AffineOutputDispatch affine_outputs(checker, affine_states);
    FiniteBilinearProjection bilinear(checker);
    r["timings"]["load_and_scope_analysis"] = elapsed();
    if (!l.doc.value("outputs", J::array()).empty() ||
        !o.doc.value("outputs", J::array()).empty())
      throw Unsupported("point output mapping adapter is not implemented");
    std::map<std::string, const J *> fl, fr;
    for (auto &f : l.doc["output_families"])
      fl[f["allocation"]] = &f;
    for (auto &f : o.doc["output_families"])
      fr[f["allocation"]] = &f;
    const auto &outputs = mapping["outputs"]["tensors"];
    r["total_output_families"] = outputs.size();
    size_t count = options.family_limit ? std::min(options.family_limit, outputs.size()) : outputs.size();
    for (size_t j = 0; j < count; ++j) {
      if (elapsed() > options.seconds)
        throw Timeout();
      const auto &out = outputs[j];
      const auto &lf = *fl.at(out["left_allocation"]),
                 &rf = *fr.at(out["right_allocation"]);
      for (auto pair : {std::pair("left", &lf), std::pair("right", &rf)}) {
        auto &f = *pair.second;
        std::string side = pair.first;
        if (f["shape"] != out["shape"] ||
            f["strides_bytes"] != out[side + "_strides_bytes"] ||
            f["offset"] != out[side + "_offset"] ||
            f["element_bytes"] != out["element_bytes"])
          throw Unsupported("nonidentity output family mapping");
      }
      r["rows"].push_back({{"family", j}});
      auto &row = r["rows"].back();
      auto stage = [&](const char *name, auto fn) {
        phase = name;
        double start = elapsed();
        Id n = fn();
        row[name] = elapsed() - start;
        return n;
      };
      bool state_abstracted = false;
      auto use_sealed_state = [&](Id body, bool reference) {
        if (!piece_equalities.has_sealed_schema()) return body;
        double start = elapsed();
        auto rewritten = reference ? piece_equalities.abstract_reference(body)
                                   : piece_equalities.abstract_optimized(body);
        row["piece_union_reuse_s"] = row.value("piece_union_reuse_s", 0.0) + elapsed() - start;
        state_abstracted |= rewritten && *rewritten != body;
        return rewritten.value_or(body);
      };
      Id lr = stage("left_import_s", [&] { return l.family(lf); });
      Id rr = stage("right_import_s", [&] { return o.family(rf,l.family_type(lf)); });
      double template_start = elapsed();
      const auto template_attempts = a.stats["parametric_state_attempts"];
      bool template_equal = piece_equalities.prove_instance(lr, rr);
      // Preserve the theorem decline even if a later family times out. At
      // most one attempt per group; do not rebuild this receipt per output.
      if (a.stats["parametric_state_attempts"] != template_attempts)
        r["piece_union"] = piece_equalities.report();
      row["parametric_state_s"] = elapsed() - template_start;
      if (template_equal) row["parametric_state_proof"] =
          "formal-domain theorem; both complete bodies checked as capture-free instances";
      Id left_body = lr, right_body = rr;
      {
        phase = "affine_state_pattern";
        double start = elapsed();
        try {
          Id projection_read=invalid,projection_axis=invalid;
          bool finite_sum_scope=false;
          if (auto proof = affine_outputs.rewrite(lr)) {
            left_body = proof->replacement;
            row["affine_state_proof"] = proof->receipt;
            projection_read=proof->projection_read;projection_axis=proof->projection_axis;
            finite_sum_scope=true;
          } else if (!affine_outputs.last_failure.empty()) row["affine_state_decline"] = affine_outputs.last_failure;
          // Finite-sum identities do not need a recurrence or an orientation
          // descriptor. A unique verified descriptor enables Fubini only; no
          // descriptor means ordinary exact Sum simplification, not guessing.
          // Reuse the dispatcher's bounded root classification; do not send
          // unrelated scalar-output DAGs through a fresh optional traversal.
          finite_sum_scope |= affine_outputs.has_projection_sum();
          if(finite_sum_scope) {
            if (auto projected = bilinear.rewrite(left_body, projection_read, projection_axis)) {
              left_body = projected->replacement;
              row["reference_bilinear_proof"] = projected->receipt;
            } else if (!bilinear.last_failure.empty()) row["reference_bilinear_decline"] = bilinear.last_failure;
            if (auto projected = bilinear.rewrite(right_body, projection_read, projection_axis)) {
              right_body = projected->replacement;
              row["optimized_bilinear_proof"] = projected->receipt;
            } else if (!bilinear.last_failure.empty()) row["optimized_bilinear_decline"] = bilinear.last_failure;
          }
        } catch (const Unsupported &e) {
          // Optional theorem discovery cannot turn a previously supported pair
          // into a translation blocker. Completed local rewrites are identities.
          row["affine_state_decline"] = e.what();
          row.erase("affine_state_proof");
          row.erase("reference_bilinear_proof"); row.erase("optimized_bilinear_proof");
          left_body = lr; right_body = rr;
        }
        row["affine_state_pattern_s"] = elapsed() - start;
      }
      {
        double start = elapsed();
        piece_equalities.seal_ready();
        row["piece_union_seal_s"] = elapsed() - start;
      }
      left_body = use_sealed_state(left_body, true);
      right_body = use_sealed_state(right_body, false);
      Id ln = stage("left_normalize_s", [&] { return template_equal ? left_body : checker.normalize(left_body, true); });
      Id rn = stage("right_normalize_s", [&] { return template_equal ? right_body : checker.normalize(right_body); });
      auto compose_state_output = [&](Id expected, Id optimized) {
        if (checker.scalar.equal(expected, optimized)) return true;
        if (auto reduced = unary_reductions.rewrite(expected, optimized);
            reduced && *reduced != optimized) {
          row["piece_union_boundary_proof"] = unary_reductions.last_receipt;
          optimized = checker.normalize(*reduced);
          if (checker.scalar.equal(expected, optimized)) return true;
        }
        LinearReductionPattern linear(checker);
        if (auto proof = linear.prove(expected, optimized)) {
          row["piece_union_reduction_proof"] = proof->receipt;
          return true;
        }
        row["piece_union_decline"] = linear.last_failure;
        return false;
      };
      row["scalar_engine"] = "mandatory_exponential_polynomial";
      bool equal = template_equal || checker.scalar.equal(ln, rn);
      if (!equal && state_abstracted) {
        phase = "piece_union_reuse";
        double start = elapsed();
        try { equal = compose_state_output(ln, rn); }
        catch (const Unsupported &) { ++a.stats["piece_union_normalization_declines"]; }
        row["piece_union_reuse_s"] = row.value("piece_union_reuse_s", 0.0) + elapsed() - start;
      }
      // Unary congruence needs a complete argument reduction proof, not a
      // prior state-output abstraction. Apply the existing general theorem
      // equally to ordinary projected numerical outputs.
      if(!equal && !state_abstracted) {
        phase="unary_reduction_composition";
        const auto start=elapsed();
        if(auto reduced=unary_reductions.rewrite(ln,rn);reduced && *reduced!=rn) {
          rn=checker.normalize(*reduced);
          equal=checker.scalar.equal(ln,rn);
          row["unary_reduction_proof"]=unary_reductions.last_receipt;
        } else if(!unary_reductions.last_failure.empty()) {
          row["unary_reduction_decline"]=unary_reductions.last_failure;
        }
        row["unary_reduction_s"]=elapsed()-start;
      }
      if (!equal) {
        phase = "state_pattern";
        double start = elapsed();
        WelfordPatternPass welford(l, o, &checker.scalar);
        if (welford.prove(lf, rf, lr, rr)) {
          equal = true;
          row["state_pattern_proof"] = welford.receipt;
        } else {
          OnlineSoftmaxPattern online(checker);
          if (auto proof = online.prove(ln, rn)) {
            equal = true;
            if (proof->logarithmic) row["state_pattern_proof"] = {
                {"method", "online_softmax_log_mass_certificate"},
                {"denominator_spec", proof->denominator_spec},
                {"common_scale", a.render(proof->common_scale, 6)},
                {"pivot", a.render(proof->pivot, 6)},
                {"pivot_basis", proof->base2_pivot ? "base2" : "natural"},
                {"coverage", "exact_uniform_weighted_mass_with_certified_common_scale"},
                {"visited_nodes", proof->visited_nodes},
                {"contribution_pieces", proof->contribution_pieces}};
            else row["state_pattern_proof"] = {
                {"method", "online_softmax_common_scale_certificate"},
                {"numerator_spec", proof->numerator_spec},
                {"denominator_spec", proof->denominator_spec},
                {"common_scale", a.render(proof->common_scale, 6)},
                {"coverage", "exact_uniform_weighted_coverage_and_equal_total_scales"},
                {"visited_nodes", proof->visited_nodes},
                {"contribution_pieces", proof->contribution_pieces}};
          } else if (!online.last_failure.empty()) {
            row["online_softmax_decline"] = online.last_failure;
          }
        }
        if (!equal) {
          LinearReductionPattern linear(checker);
          if (auto proof = linear.prove(ln, rn)) {
            equal = true;
            row["state_pattern_proof"] = proof->receipt;
          } else if (!linear.last_failure.empty()) {
            row["linear_reduction_decline"] = linear.last_failure;
          }
        }
        row["state_pattern_s"] = elapsed() - start;
      }
      if (!equal && !state_abstracted) {
        phase = "piece_union_reuse";
        double start = elapsed();
        auto abstract_left = piece_equalities.abstract_reference(lr);
        auto abstract_right = piece_equalities.abstract_optimized(rr);
        if (abstract_left && abstract_right && (*abstract_left != lr || *abstract_right != rr)) {
          try {
            Id expected = checker.normalize(*abstract_left, true);
            Id composed = checker.normalize(*abstract_right);
            equal = compose_state_output(expected, composed);
            if (equal)
              row["piece_union_proof"] = "complete proved pieces; exact finite-domain coverage; shared defined state; complete outer equality";
          } catch (const Unsupported &) {
            ++a.stats["piece_union_normalization_declines"];
          }
          // Keep later fallback recognition in the original reference vocabulary.
          if (!equal) checker.normalize(lr, true);
        }
        row["piece_union_reuse_s"] = elapsed() - start;
      }
      for(const auto& [canonical,alternative]:checker.complete_reduction_equalities)
        proved_equalities.record(canonical,alternative);
      checker.complete_reduction_equalities.clear();
      if (!equal && !proved_equalities.empty()) {
        phase = "proof_reuse";
        double start = elapsed();
        auto reduced = proved_equalities.rewrite(rn);
        auto expected = proved_equalities.rewrite(ln);
        if (reduced && expected && (*reduced != rn || *expected != ln)) {
          try {
          Id composed = checker.normalize(*reduced);
          Id wanted = checker.normalize(*expected);
          equal = checker.scalar.equal(composed, wanted);
          if (!equal) {
            LinearReductionPattern linear(checker);
            if (auto proof = linear.prove(wanted, composed)) {
              equal = true;
              row["composed_reduction_proof"] = proof->receipt;
            }
          }
          if (equal) row["exact_proof_reuse"] = "complete output or reduction coverage equalities; exact Arena identities; no parameter generalization";
          } catch (const Unsupported &) {
            row["exact_proof_reuse_decline"] = "unsupported_normalized_replacement";
          }
        }
        row["proof_reuse_s"] = elapsed() - start;
      }
      bool partial_equal = false;
      if (!equal) {
        // Retain pre-normalization domains: cancellation must not erase holes.
        ExponentialPolynomial partial(a);
        auto raw_left = partial.domain(lr), raw_right = partial.domain(rr);
        auto norm_left = partial.domain(ln), norm_right = partial.domain(rn);
        if (raw_left.supported && raw_right.supported && norm_left.supported && norm_right.supported) {
          auto canonical_conditions = [&](const std::set<Id> &conditions) {
            std::set<Id> result;
            for (Id condition : conditions) result.insert(checker.normalize(condition));
            return result;
          };
          auto dl = canonical_conditions(raw_left.nonzero);
          auto dr = canonical_conditions(raw_right.nonzero);
          // Require both normalization steps to preserve the same domain.
          if (dl == dr && dl == norm_left.nonzero && dr == norm_right.nonzero &&
              partial.equal_partial(ln, rn)) {
            equal = partial_equal = true;
            row["partial_function_proof"] = {{"method", "equal_nonzero_domains_then_ep_values"},
                {"nonzero_conditions", dl}, {"extra_input_assumptions", false}};
          }
        }
      }
      row["total_definedness"] = a.nodes[lr].facts.finite && a.nodes[rr].facts.finite
          ? "proved" : "not_proven";
      row["equivalence_contract"] = partial_equal ? "same_domain_and_equal_values" : "existing_total_value_proof";
      if (equal && !partial_equal) {
        if (state_abstracted)
          row["piece_union_proof"] = "complete proved pieces; exact finite-domain coverage; shared defined state; complete outer equality";
        double start = elapsed();
        // Preserve the complete imported expressions and their original local
        // domains. A successful normalization equality is also a full proof.
        piece_equalities.record(lr, rr, lf, rf);
        row["piece_union_record_s"] = elapsed() - start;
        if (!template_equal && ln != rn) proved_equalities.record(ln, rn);
      }
      row["cumulative_elapsed_s"]=elapsed();
      row["status"] = equal ? "equivalent" : "not_proven";
      if (!equal) {
        row["left"] = a.render(ln, 5);
        row["right"] = a.render(rn, 5);
      }
      ++finished;
      if (equal)
        ++passed;
      roots.push_back(ln);
      roots.push_back(rn);
    }
    r["piece_union"] = piece_equalities.report();
    if (passed == outputs.size())
      r["status"] = "equivalent";
    else if (passed == count && count)
      r["status"] = "partial_scope_equivalent";
  } catch (const Unsupported &e) {
    r["status"] = "translation_or_supported_fragment_blocked";
    r["reason"] = e.code;
    r["diagnostics"] = J::array({{{"code",e.code},{"message",e.what()},
      {"stage",(e.code=="input_alignment_not_proven" || e.code.starts_with("saturation_")) ? "equivalence_import" : phase},{"details",e.details}}});
    r["blocked_stage"] = phase;
  } catch (const Timeout &e) {
    r["status"] = "timeout";
    r["reason"] = "time_budget_exceeded";
    r["message"] = e.what();
    r["blocked_stage"] = phase;
  } catch (const std::bad_alloc &e) {
    r["status"] = "allocation_failed";
    r["reason"] = "memory_allocation_failed";
  } catch (const std::exception &e) {
    r["status"] = "internal_error";
    r["reason"] = "unexpected_exception";
    r["message"] = e.what();
    r["blocked_stage"] = phase;
  }
  r["idealizations_applied"] = J::array();
  if (a.stats["ideal_compiler_exp_sequences"])
    r["idealizations_applied"].push_back("recognized fma.rm exponent encoding with split log2(e) is ideal Exp");
  if (a.stats["ideal_single_constant_exp_sequences"])
    r["idealizations_applied"].push_back("compiler Exp2 with binary32 log2(e) multiplier is ideal Exp");
  if (a.stats["ideal_additive_compiler_exp_sequences"])
    r["idealizations_applied"].push_back("typed additive compiler Exp2 idealizes only binary32 log2(e)-scaled terms as Exp; unscaled residual terms retain base two");
  if (a.stats["ideal_compiler_lse_sequences"])
    r["idealizations_applied"].push_back("complete binary32 PTX lg2 LSE tail is ln(mass)+ln(2)*base2_pivot; mass and pivot coefficients are retained");
  if (a.stats["ideal_compiler_log2_polynomials"])
    r["idealizations_applied"].push_back("complete typed RN-FMA binary32 log2 polynomial with subnormal and exceptional guards is ideal Log2 on proved finite positive input");
  r["input_contract_applied_reads"] = J::array();
  for(const auto &[side,object,type]:(prepared ? prepared->contract.applied : contract.applied))
    r["input_contract_applied_reads"].push_back(J{{"side",side},{"object",object},{"element_type",type}});
  r["access_validity"] = a.stats["symbolic_byte_reads_without_extent_proof"] ? "not_proven" : "no_unresolved_symbolic_byte_reads";
  r["equivalence_context"]={{"element_assumptions",std::count_if(context.assumptions.begin(),context.assumptions.end(),
        [](const Assumption &value){return std::holds_alternative<InputElementAssumption>(value);})},
      {"fixed_input_elements",context.fixed_inputs.size()},
      {"applied_predicates",prepared ? prepared->applied_predicates.size() : 0},
      {"scoped_predicates",prepared ? prepared->scoped_predicates.configured_count() : 0}};
  if (prepared && prepared->scoped_predicates.configured_count())
    r["scoped_predicate_context"] = prepared->scoped_predicates.report();
  if (prepared && !context.fixed_inputs.empty())
    r["fixed_input_context"] = fixed_input_context_report(*prepared);
  size_t defined_families = 0, partial_families = 0;
  for (const auto &row : r["rows"]) {
    defined_families += row.value("total_definedness", "not_proven") == "proved";
    partial_families += row.contains("partial_function_proof");
  }
  r["definedness"] = {{"status", finished && finished == r.value("total_output_families", size_t(0)) && defined_families == finished ? "proved" : "not_proven"},
      {"proved_families", defined_families}, {"partial_equivalence_families", partial_families}};
  r["equivalence_contract"] = "same_domain_and_equal_values; total definedness reported separately";
  r["equivalent_families"] = passed;
  r["finished_families"] = finished;
  r["elapsed_s"] = elapsed();
  struct rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  r["peak_rss_kib"] = ru.ru_maxrss;
#ifdef __APPLE__
  r["peak_rss_kib"] = ru.ru_maxrss / 1024;
#endif
  r["peak_rss_scope"] = "process_lifetime_not_comparison_delta";
  r["arena_nodes"] = a.nodes.size();
  r["pattern_nodes"] = checker.patterns.nodes.size();
  r["reference_term_patterns"] = checker.specs.size();
  r["piece_nodes"] = checker.pieces.size();
  r["scalar_engine"] = "mandatory_exponential_polynomial";
  r["legacy_scalar_fallback"] = false;
  r["ep_profile"] = checker.scalar.profile();
  r["counters"] = a.stats;
  r["unmatched_reasons"] = checker.failures;
  r["prerequisites"] = "caller must combine numerical proof with synchronization, memory and scope results";
  for (const auto *stage : {"left_import_s", "left_normalize_s",
                            "right_import_s", "right_normalize_s", "state_pattern_s", "proof_reuse_s",
                            "piece_union_record_s", "piece_union_reuse_s", "piece_union_seal_s", "parametric_state_s", "affine_state_pattern_s", "exponential_polynomial_s"}) {
    double sum = 0;
    for (auto &row : r["rows"])
      sum += row.value(stage, 0.0);
    r["stages_s"][stage] = sum;
  }
  std::unordered_set<Id> reachable;
  while (!roots.empty()) {
    Id n = roots.back();
    roots.pop_back();
    if (!reachable.insert(n).second)
      continue;
    for (Id x : a.nodes[n].args)
      roots.push_back(x);
  }
  r["normalized_reachable_nodes"] = reachable.size();
  return ComparisonResult::fromReport(std::move(r));
}
