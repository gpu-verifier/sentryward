#include "sentryward/frontend/ptx/module.h"

#include <algorithm>
#include <initializer_list>
#include <string_view>

namespace sentryward::frontend::ptx {
namespace {
bool In(std::string_view value,
        std::initializer_list<std::string_view> values) {
  return std::find(values.begin(), values.end(), value) != values.end();
}
std::string Join(const std::vector<std::string> &parts) {
  std::string result;
  for (const auto &part : parts) {
    if (!result.empty())
      result += '.';
    result += part;
  }
  return result;
}
std::string Collective(const Instruction &i) {
  auto tail = Join(i.modifiers);
  if (i.opcode == "bar") {
    if (tail == "warp.sync")
      return "warp_sync";
    if (tail.starts_with("cta."))
      tail.erase(0, 4);
    if (tail == "sync")
      return "cta_sync";
    if (tail == "arrive")
      return "named_arrive";
    if (In(tail, {"red.popc.u32", "red.and.pred", "red.or.pred"}))
      return "named_reduce";
  } else {
    if (In(tail, {"cluster.arrive", "cluster.arrive.aligned",
                  "cluster.arrive.release", "cluster.arrive.release.aligned",
                  "cluster.arrive.relaxed", "cluster.arrive.relaxed.aligned"}))
      return "cluster_arrive";
    if (In(tail, {"cluster.wait", "cluster.wait.aligned",
                  "cluster.wait.acquire", "cluster.wait.acquire.aligned"}))
      return "cluster_wait";
    if (tail.starts_with("cta."))
      tail.erase(0, 4);
    if (In(tail, {"sync", "sync.aligned"}))
      return "cta_sync";
    if (In(tail, {"arrive", "arrive.aligned"}))
      return "named_arrive";
    if (In(tail,
           {"red.popc.u32", "red.popc.aligned.u32", "red.and.pred",
            "red.and.aligned.pred", "red.or.pred", "red.or.aligned.pred"}))
      return "named_reduce";
  }
  return "unsupported";
}
} // namespace
Semantics Classify(const Instruction &i) {
  const auto &base = i.opcode;
  const auto &p = i.modifiers;
  const auto has = [&](std::string_view v) {
    return std::find(p.begin(), p.end(), v) != p.end();
  };
  const auto contains = [&](std::string_view v) {
    return std::any_of(p.begin(), p.end(), [&](const auto &x) {
      return x.find(v) != std::string::npos;
    });
  };
  const auto prefix = [&](std::string_view v) {
    return std::any_of(p.begin(), p.end(),
                       [&](const auto &x) { return x.starts_with(v); });
  };
  const auto first = p.empty() ? std::string{} : p.front();
  const auto tail = Join(p);
  const auto text = base + (tail.empty() ? "" : "." + tail);
  Semantics s;
  s.atomic_add = has("add") &&
      (base == "atom" || (base == "red" && !has("sync")) ||
       base == "sured" || (base == "cp" && has("reduce")) ||
       (base == "multimem" &&
        (has("red") || (has("cp") && has("reduce")))));
  const auto finish = [&](std::string category, std::string operation) {
    s.category = std::move(category);
    s.operation = std::move(operation);
    s.modeled = s.operation != "unsupported";
    return s;
  };
  if (i.control != "none") {
    s.control_relevant = true;
    return finish("control", i.control);
  }
  if (base == "mbarrier") {
    std::string op = "unsupported";
    if (In(first, {"init", "inval", "expect_tx", "complete_tx", "arrive_drop",
                   "pending_count", "report"}))
      op = first;
    else if (first == "arrive")
      op = has("expect_tx") ? "arrive_expect_tx" : "arrive";
    else if (In(first, {"test_wait", "try_wait"}))
      op = first + (has("parity") ? "_parity" : "_state");
    else if (tail.find("arrive") != std::string::npos)
      op = "arrive";
    return finish("mbarrier", op);
  }
  if (base == "cp" && has("async")) {
    std::string op;
    if (has("prefetch"))
      op = "prefetch";
    else if (has("bulk")) {
      if (contains("complete_tx"))
        op = has("tensor") ? "tma_load_complete_tx" : "bulk_complete_tx";
      else if (has("tensor"))
        op = "tma_bulk_tensor";
      else if (has("commit_group"))
        op = "bulk_commit_group";
      else if (prefix("wait_group"))
        op = "bulk_wait_group";
      else if (has("wait_all"))
        op = "unsupported";
      else
        op = has("reduce") ? "bulk_reduce" : "bulk_copy";
    } else if (contains("mbarrier"))
      op = "cp_async_mbarrier";
    else if (has("commit_group"))
      op = "commit_group";
    else if (prefix("wait_group"))
      op = "wait_group";
    else if (has("wait_all"))
      op = "wait_all";
    else
      op = "copy";
    s.asynchronous = true;
    s.memory_effect =
        !In(op, {"prefetch", "commit_group", "wait_group", "wait_all",
                 "bulk_commit_group", "bulk_wait_group", "cp_async_mbarrier"});
    return finish("async", op);
  }
  if (base == "griddepcontrol") {
    s.control_relevant = true;
    return finish("external_control", tail == "wait" ? "grid_dependency_wait"
                                      : tail == "launch_dependents"
                                          ? "grid_dependency_launch_dependents"
                                          : "unsupported");
  }
  if (base == "clusterlaunchcontrol") {
    std::string op = "unsupported";
    if (first == "query_cancel") {
      if (has("is_canceled"))
        op = "query_cancel_is_canceled";
      else if (prefix("get_first_ctaid"))
        op = "query_cancel_get_first_ctaid";
    } else if (first == "try_cancel" && has("async") &&
               contains("mbarrier::complete_tx::bytes"))
      op = "try_cancel_complete_tx";
    if (op.starts_with("query_cancel_")) {
      s.control_relevant = true;
      return finish("compute", op);
    }
    s.asynchronous = op == "try_cancel_complete_tx";
    s.memory_effect = true;
    if (s.asynchronous) {
      s.memory_space = "shared::cta";
      s.memory_mode = "write";
    }
    return finish("async", op);
  }
  if (base == "tcgen05") {
    std::string op = "unsupported";
    if (In(first, {"mma", "alloc", "dealloc", "relinquish_alloc_permit",
                   "fence", "shift", "cp"}))
      op = first;
    else if (first.starts_with("fence::"))
      op = "fence";
    else if (first == "commit" && contains("mbarrier"))
      op = "commit";
    else if (first == "ld")
      op = "load";
    else if (first == "st")
      op = "store";
    else if (first.starts_with("wait::ld"))
      op = "wait_load";
    else if (first.starts_with("wait::st"))
      op = "wait_store";
    s.asynchronous = In(op, {"mma", "cp", "shift", "load", "store"});
    s.memory_effect = s.asynchronous || op == "unsupported";
    return finish(In(op, {"alloc", "dealloc", "relinquish_alloc_permit"})
                      ? "tmem_lifetime"
                  : op == "fence" ? "fence"
                                  : "async",
                  op);
  }
  if (In(base, {"bar", "barrier"}))
    return finish("collective", Collective(i));
  if (base == "fence" || text.starts_with("membar"))
    return finish("fence", text);
  if (base == "st" && has("async")) {
    s.memory_space = "shared::cluster";
    s.memory_mode = "write";
    s.asynchronous = s.memory_effect = true;
    return finish("async", has("shared::cluster") && contains("mbarrier") &&
                                   contains("complete_tx") && contains("bytes")
                               ? "st_async_complete_tx"
                               : "unsupported");
  }
  if (In(base, {"ld", "ldu", "st", "atom", "red", "suld", "sust"})) {
    for (auto x : p)
      if (In(x, {"global", "shared", "shared::cta", "shared::cluster", "local",
                 "param", "const"})) {
        if (auto at = x.find("::"); at != std::string::npos)
          x.replace(at, 2, ".");
        s.memory_space = x;
        break;
      }
    if (!s.memory_space && In(base, {"ld", "ldu", "st", "atom", "red"}))
      s.memory_space = "generic";
    s.memory_mode = In(base, {"ld", "ldu", "suld"}) ? "read"
                    : In(base, {"st", "sust"})      ? "write"
                                                    : "atomic";
    s.memory_effect = true;
    s = finish("memory", base);
    s.modeled = s.memory_space.has_value();
    return s;
  }
  if (In(base, {"ldmatrix", "stmatrix"})) {
    s.memory_space = "shared";
    s.memory_mode = base == "ldmatrix" ? "read" : "write";
    s.memory_effect = true;
    return finish("memory", base);
  }
  if (base == "wmma" && (has("load") || has("store"))) {
    s.memory_mode = has("load") ? "read" : "write";
    s.memory_space = "generic";
    s.memory_effect = true;
    for (auto x : p)
      if (In(x, {"global", "shared", "shared::cta", "shared::cluster"})) {
        if (auto at = x.find("::"); at != std::string::npos)
          x.replace(at, 2, ".");
        s.memory_space = x;
        break;
      }
    return finish("memory", "wmma_" + *s.memory_mode);
  }
  if (base == "multimem") {
    std::string op = "unsupported";
    for (auto x : {"ld_reduce", "ld", "st", "red"})
      if (has(x)) {
        op = x;
        break;
      }
    s.memory_space = "global";
    s.memory_effect = true;
    if (op != "unsupported")
      s.memory_mode = In(op, {"ld", "ld_reduce"}) ? "read"
                      : op == "st"                ? "write"
                                                  : "atomic";
    return finish("memory", op);
  }
  if (In(base, {"tex", "tld4", "txq", "sured", "suq"})) {
    s.memory_space = "texture_or_surface";
    return finish("memory", "unsupported");
  }
  if (base=="tensormap") {
    const auto &m=i.modifiers;
    const bool replace=m.size()==6 && m[0]=="replace" && m[1]=="tile" && m[3]=="shared::cta" && m[4]=="b1024";
    const bool copy=m.size()==8 && m[0]=="cp_fenceproxy" && m[1]=="global" && m[2]=="shared::cta" &&
        m[3]=="tensormap::generic" && m[4]=="release" && In(m[5],{"cta","cluster","gpu","sys"}) && m[6]=="sync" && m[7]=="aligned";
    s.memory_effect=true; s=finish("async",replace?"tensor_map_replace":"tensor_map_copy"); s.modeled=replace||copy; return s;
  }
  if (In(base, {"wgmma", "cp"})) {
    s.asynchronous = true;
    s = finish("async", text);
    s.modeled = false;
    return s;
  }
  for (auto x : {"ld", "st", "atom", "mbarrier", "barrier", "tcgen", "wgmma",
                 "multimem", "tensormap", "griddepcontrol",
                 "clusterlaunchcontrol", "tex", "tld", "txq"})
    if (base.starts_with(x)) {
      s.asynchronous = true;
      return finish("async", "unsupported");
    }
  s.control_relevant =
      In(base, {"mov", "cvt", "cvta", "add",  "sub",  "mul",   "mad",  "shl",
                "shr", "and", "or",   "xor",  "not",  "setp",  "selp", "min",
                "max", "rem", "div",  "shfl", "vote", "elect", "mapa", "lop3"});
  return finish("compute", text);
}
} // namespace sentryward::frontend::ptx
