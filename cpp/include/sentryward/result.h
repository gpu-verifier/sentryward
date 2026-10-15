#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
namespace sentryward::results {
using Json=nlohmann::json;
inline const Json& Rules() { static const Json rules=Json::parse(R"RESULT({"results":["verified","race_free","not_wellsynced","potential_race","invalid_lifecycle","equivalence_not_proven","unsupported","invalid_input","missing_input","timeout","memory_budget_exceeded","analysis_budget_exceeded","allocation_failed","tool_failure"],"legacy":{"race_free":"race_free","equivalent":"verified","not_proven":"equivalence_not_proven","partial_scope_equivalent":"analysis_budget_exceeded","translation_or_supported_fragment_blocked":"unsupported","blocked":"unsupported","missing_reference":"missing_input","not_run":"unsupported","resource_limit":"tool_failure","memory_limit_exceeded":"memory_budget_exceeded","native_resource_limit":"memory_budget_exceeded","internal_error":"tool_failure","implementation_error":"tool_failure","process_failed":"tool_failure","native_process_error":"tool_failure","launch_error":"tool_failure","invalid_process_output":"tool_failure","invalid_or_missing_report":"tool_failure","interrupted":"tool_failure","parser_error":"invalid_input","invalid_configuration":"invalid_input","invalid_input":"invalid_input","input_file_error":"missing_input","output_file_error":"tool_failure","unsafe_lifecycle":"invalid_lifecycle","source_lifecycle_violation":"invalid_lifecycle","program_constraint_violation":"invalid_lifecycle","invalid_event_view":"tool_failure","invalid_source_ir":"tool_failure","event_view_incomplete":"unsupported","source_ir_incomplete":"unsupported","event_view_unsupported":"unsupported","source_ir_unsupported":"unsupported","ptx_execution_unsupported":"unsupported","memory_layout_unsupported":"unsupported","memory_analysis_inconclusive":"unsupported","shared_memory_value_unproven":"unsupported","representation_limit_exceeded":"unsupported","analysis_limit_exceeded":"analysis_budget_exceeded","expression_export_failed":"tool_failure","valid":"verified","extraction_complete":"verified","semantic_complete":"verified","value_view_complete":"verified","expressions_ready":"verified","reference_expressions_ready":"verified","ready":"verified","complete":"verified","value_view_incomplete":"unsupported","branch_scope_incomplete":"unsupported","native_frontend_error":"tool_failure","native_crash":"tool_failure","invalid_native_report":"tool_failure","frontend_unsupported":"unsupported","unresolved_translation":"unsupported","invalid_source":"tool_failure","memory_model_unsupported":"unsupported","synchronization_candidate":"not_wellsynced","race_candidate":"potential_race","resource_exhausted":"allocation_failed","process_resource_limit":"allocation_failed","native_timeout":"timeout","runner_timeout":"timeout","wellsync_timeout":"timeout","validated":"verified","unresolved":"unsupported","not_reached":"unsupported","reference_unsupported":"unsupported"},"diagnostic_codes":{"duplicate-barrier-initialization":"barrier_initialized_multiple_times","cluster_arrive_without_wait":"cluster_barrier_arrive_repeated_without_wait","branch_not_static":"control_predicate_not_static","missing_parameter_value":"control_parameter_value_missing","allocation_identity_unresolved":"allocation_alias_relation_unknown","tma_coordinate_rank_mismatch":"tma_rank_mismatch","tma_element_stride_unsupported":"tma_nonunit_element_stride_unsupported","execution_step_limit_exceeded":"execution_step_budget_exceeded","verification_prerequisite_not_satisfied":"prerequisite_check_not_passed","memory_check":"conflicting_accesses_unordered","unresolved_lifetime_obligation":"lifecycle_order_endpoint_missing"}})RESULT");return rules; }
inline bool IsResult(const std::string& value) { for(const auto& x:Rules()["results"])if(x==value)return true;return false; }
inline std::string Category(const std::string& status,const std::string& reason="",const Json& resource=Json::object()) {
  if(reason=="time_budget" || reason=="reference_capture_timeout" || status=="native_timeout" || status=="runner_timeout" || status=="wellsync_timeout")return "timeout";
  if(reason=="rss_limit" || reason=="native_pipeline_memory_limit_exceeded")return "memory_budget_exceeded";
  if(status=="representation_limit_exceeded" || reason=="representation_limit_exceeded")return "unsupported";
  if(reason=="allocation_failed")return "allocation_failed";
  if(reason=="node_budget" || reason=="iteration_budget" || reason=="work_budget" || reason=="alpha_normalization_context_budget" || reason=="alpha_normalization_node_budget" || reason=="projection_memo_budget" || reason.ends_with("_limit_exceeded"))return "analysis_budget_exceeded";
  if(reason=="inputs_changed_during_run" || reason=="extraction_inputs_changed_during_run" || reason=="unexpected_process_exit" || reason=="invalid_native_result" || reason.starts_with("process_error:"))return "tool_failure";
  if(IsResult(status))return status;
  if(status=="resource_limit") { auto origin=resource.value("origin","");return origin=="configured_budget"?"memory_budget_exceeded":origin=="algorithm_budget"?"analysis_budget_exceeded":origin=="representation"?"unsupported":origin=="allocator"?"allocation_failed":"tool_failure"; }
  return Rules()["legacy"].value(status,"tool_failure");
}
inline Json Failure(const std::string& result,const std::string& code,const std::string& message) {
 return {{"result",result},{"diagnostics",Json::array({{{"code",code},{"message",message}}})},
   {"checks",{{"synchronization","not_run"},{"memory","not_run"},{"equivalence","not_run"}}}};
}

inline void Diagnostics(Json& out) {
 if(!out.contains("diagnostics") || !out["diagnostics"].is_array())return;
 for(auto& d:out["diagnostics"]) {
  if(!d.is_object())continue;
  auto code=d.value("code","diagnostic_code_missing");
  if(Rules()["diagnostic_codes"].contains(code))code=Rules()["diagnostic_codes"][code];
  auto stage=d.contains("stage") && d["stage"].is_string()?d["stage"].get<std::string>():
    out.contains("stage") && out["stage"].is_string()?out["stage"].get<std::string>():std::string("pipeline");
  Json details=d.value("details",Json::object());
  if(!details.is_object())details=Json{{"evidence",std::move(details)}};
  for(auto it=d.begin();it!=d.end();) {
    if(it.key()!="code" && it.key()!="stage" && it.key()!="message" && it.key()!="details") {
      details[it.key()]=std::move(it.value());it=d.erase(it);
    } else ++it;
  }
  if(out.value("result","")=="memory_budget_exceeded" &&
     (code==stage || code=="memory_budget_exceeded" || code=="memory_limit_exceeded" || code.starts_with("semantic_ir.cta") || code.starts_with("source_ir.cta") ||
      code.starts_with("value_ir.cta") || code=="verification.memory")) {
    if(code.starts_with("semantic_ir.cta") || code.starts_with("source_ir.cta") || code.starts_with("value_ir.cta") || code=="verification.memory")stage=code;
    code="process_memory_budget_exceeded";
  }
  if(code=="rss_limit" || code=="native_pipeline_memory_limit_exceeded" || code=="memory_limit_exceeded")
    code="process_memory_budget_exceeded";
  for(auto prefix:{"semantic_ir.cta","source_ir.cta","value_ir.cta"}) {
    if(stage.starts_with(prefix)) {
      auto cta=stage.substr(std::string(prefix).size());
      if(!cta.empty() && cta.find_first_not_of("0123456789")==std::string::npos)details["cta_progress"]=std::stoull(cta);
      else details["cta_progress"]=cta;
      stage=std::string(prefix)=="source_ir.cta"?"event_view_construction":"value_view_construction";
      break;
    }
  }
  if(stage=="verification.memory")stage="memory_check";
  if(code=="process_memory_budget_exceeded")
    for(auto key:{"resource","peak_rss_mib","rss_limit_mib","memory_limit_mib"})if(out.contains(key))details[key]=out[key];
  if(code=="total_definedness_not_proven" || code=="input_access_validity_not_proven")
    for(auto key:{"definedness","access_validity","equivalent_families"})if(out.contains(key))details[key]=out[key];
  if(code=="prerequisite_check_not_passed")
    for(auto key:{"prerequisite_status","prerequisite_result","prerequisite_diagnostics","blocked_stage"})if(out.contains(key))details[key]=out[key];
  d["code"]=code;d["stage"]=stage;
  if(!details.empty())d["details"]=std::move(details);
 }
 for(auto key:{"equivalence","value_analysis"}) {
   if(!out.contains(key) || !out[key].is_object() || !out[key].contains("diagnostics"))continue;
   for(auto& d:out[key]["diagnostics"])if(d.is_object() && d.value("code","")=="prerequisite_check_not_passed") {
     d["details"]["prerequisite_result"]=out.value("result","unsupported");
     d["details"]["prerequisite_checks"]=out.value("checks",Json::object());
     d["details"]["prerequisite_diagnostics"]=out["diagnostics"];
   }
 }
}
// Mutates only report metadata and bounded child reports. Witnesses/expression DAGs
// are not traversed or copied; work is linear in report/diagnostic metadata.
inline void Normalize(Json& out,const std::string& operation="") {
 if(!out.is_object()){out=Failure("tool_failure","invalid_result_object","Expected a JSON object");return;}
 if(out.contains("result") && IsResult(out.value("result","")) && !out.contains("status") && out.contains("diagnostics") && out.contains("checks") && out.contains("scope") && !out.contains("reason") && !out.contains("message")) {
   if(!out["diagnostics"].is_array())out=Failure("tool_failure","invalid_result_contract","Unknown result or missing diagnostics");
   for(auto key:{"equivalence","value_analysis","scalar_fold","expression_export","partial_report","native_result","partial_native_result","expression_diagnostic"})
     if(out.contains(key) && out[key].is_object() && (out[key].contains("status") || out[key].contains("result")))Normalize(out[key],key);
   Diagnostics(out);
   return;
 }
 for(auto key:{"stage","blocked_stage","active_stage"})if(out.contains(key) && out[key].is_null())out.erase(key);
 const auto status=out.value("status",out.value("result",std::string("internal_error")));
 auto reason=out.value("reason",std::string{});
 if(reason.empty())for(auto key:{"rejection","error","diagnostic"})
   if(out.contains(key) && out[key].is_object() && !out[key].value("code",std::string{}).empty()){reason=out[key]["code"];break;}
 const auto stage=out.value("stage",out.value("blocked_stage",out.value("active_stage",std::string("pipeline"))));
 auto result=Category(status,reason,out.value("resource",Json::object()));
 out.erase("status");out.erase("result");out.erase("reason");
 if(out.contains("numerical_status")){out["numerical_result"]=Category(out.value("numerical_status",""));out.erase("numerical_status");}
 Json diagnostics=out.contains("diagnostics") && out["diagnostics"].is_array()?std::move(out["diagnostics"]):Json::array();
 if(status=="invalid_source_ir" || status=="invalid_event_view" || status=="invalid_source")
   for(const auto& d:diagnostics)if(d.is_object() && (d.value("code","")=="duplicate-barrier-initialization" || d.value("code","")=="barrier_initialized_multiple_times"))result="invalid_lifecycle";
 if(status=="blocked" && reason=="verification_prerequisite_not_satisfied")result=Category(out.value("prerequisite_status","unsupported"));
 if(status=="partial_scope_equivalent"){result="analysis_budget_exceeded";reason="output_family_budget_exceeded";}
 if(status=="resource_limit" && result=="tool_failure" && reason.empty())reason="unclassified_resource_limit";
 Json checks=out.value("checks",Json::object());
 const bool kernel=operation=="kernel_verification" || operation=="verification" || out.value("mode","")=="verify";
 for(auto key:{"synchronization","memory","equivalence"})if(!checks.contains(key))checks[key]=kernel && std::string(key)!="equivalence"?"not_run":"not_requested";
 const auto verified=out.value("verification_status",std::string{});
 out.erase("verification_status");
 if(verified=="race_free" || status=="race_free" || status=="potential_race" || status=="memory_analysis_inconclusive") {
   checks["synchronization"]="passed";
   checks["memory"]=status=="potential_race"?"failed":out.value("memory_analysis_complete",false)?"passed":"incomplete";
 } else if(status=="not_wellsynced") {checks["synchronization"]="failed";checks["memory"]="not_run";checks["equivalence"]="not_run";}
 else if(out.value("mode","")=="verify" || out.contains("verification_performed")){checks["synchronization"]="incomplete";checks["memory"]="not_run";}
 if(!verified.empty())out["verification_result"]=Category(verified);
 for(auto key:{"equivalence","value_analysis","scalar_fold","expression_export","partial_report","native_result","partial_native_result","expression_diagnostic"})
   if(out.contains(key) && out[key].is_object() && (out[key].contains("status") || out[key].contains("result")))
     Normalize(out[key],std::string(key)=="equivalence"?"equivalence":key);
 if(operation=="equivalence" || status=="equivalent" || status=="not_proven" || status=="partial_scope_equivalent" || status=="translation_or_supported_fragment_blocked") {
   checks["equivalence"]=result=="verified"?"passed":status=="blocked" || status=="missing_reference" || reason=="verification_prerequisite_not_satisfied"?"not_run":"incomplete";
   // Equal domains and equal values suffice; total definedness is separate.
   if(result=="verified" && out.value("definedness",Json::object()).value("status","")=="not_proven")
     diagnostics.push_back({{"code","total_definedness_not_proven"},{"stage","equivalence_check"}});
   if(result=="verified" && out.value("access_validity","")=="not_proven") {
     result="equivalence_not_proven";reason="input_access_validity_not_proven";checks["equivalence"]="incomplete";
   }
 }
 if(out.contains("equivalence") && out["equivalence"].is_object() && out["equivalence"].contains("result")) {
   auto& eq=out["equivalence"];
   checks["equivalence"]=eq.value("checks",Json::object()).value("equivalence","not_run");
   if(checks["synchronization"]=="passed" && checks["memory"]=="passed" && (result=="race_free" || result=="verified")) {
     result=eq["result"];
     for(const auto& d:eq["diagnostics"])diagnostics.push_back(d);
   }
 }
 bool success=false;
 for(auto s:{"race_free","equivalent","verified","valid","extraction_complete","semantic_complete","value_view_complete","expressions_ready","reference_expressions_ready","ready","complete"})if(status==s)success=true;
 if(!reason.empty() || out.contains("message") || (!success && diagnostics.empty())) {
   Json d={{"code",reason.empty()?(result=="memory_budget_exceeded"?std::string("process_memory_budget_exceeded"):status):reason},{"stage",stage}};
   for(auto key:{"rejection","error","diagnostic"})
     if(out.contains(key) && out[key].is_object() && out[key].value("code","")==d.value("code",""))
       for(auto it=out[key].begin();it!=out[key].end();++it)d[it.key()]=it.value();
   if(out.contains("message"))d["message"]=std::move(out["message"]);
   for(auto key:{"location","path","resource","details"})if(out.contains(key))d[key]=out[key];
   bool same=false;
   for(const auto& old:diagnostics)if(old.is_object() && old.value("code","")==d.value("code",""))same=true;
   if(diagnostics.empty() || (!reason.empty() && reason!="unsafe-lifecycle" && !same))diagnostics.insert(diagnostics.begin(),std::move(d));
 }
 out.erase("message");
 for(auto& d:diagnostics)if(d.is_object() && !d.contains("stage"))d["stage"]=stage;
 for(const auto& d:diagnostics)if(d.is_object() && (d.value("code","")=="unexpected_exception" || d.value("code","")=="active_thread_context_missing"))result="tool_failure";
 out["result"]=result;out["diagnostics"]=std::move(diagnostics);out["checks"]=std::move(checks);
 Diagnostics(out);
 if(out.contains("scope") && out["scope"].is_string())out["scope"]=Json{{"kind",out["scope"]}};
 if(!out.contains("scope") || !out["scope"].is_object())out["scope"]=Json::object();
 if(!out["scope"].contains("operation"))out["scope"]["operation"]=operation.empty()?(out["checks"]["synchronization"]!="not_requested"?"verification":"analysis"):operation;
}
inline Json Report(Json report,const std::string& operation=""){Normalize(report,operation);return report;}
}
