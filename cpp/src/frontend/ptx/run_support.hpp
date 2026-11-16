#pragma once
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/branch_headers.inc"
#endif
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/graph_headers.inc"
#endif
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/export_headers.inc"
#endif
#include "sentryward/result.h"
#include "sentryward/algorithm/base/analysis_scope.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/diagnostics/ir_size_profile.h"
#include "sentryward/algorithm/diagnostics/memory_profile.h"
#include "sentryward/frontend/ptx/config.h"
#include "sentryward/frontend/ptx/public_input.h"
#include "sentryward/frontend/ptx/execution.h"
#include "sentryward/frontend/ptx/lowering.h"
#include "sentryward/frontend/ptx/layouts.h"
#include "sentryward/frontend/ptx/value_ir.h"
#include "sentryward/frontend/ptx/branch_scope_capture.h"
#include "sentryward/frontend/ptx/semantic_program.h"
#include "sentryward/frontend/ptx/serialization.h"
#include "sentryward/frontend/ptx/unit_facts.h"
#include "sentryward/frontend/ptx/unit_classes.h"
#include "sentryward/algorithm/checker/unit_quotient.h"
#include "sentryward/frontend/ptx/query_log.h"
#include "sentryward/frontend/ptx/look_ahead.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_analysis/passes.h"
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_headers.inc"
#endif
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_version.inc"
#endif
#include <openssl/evp.h>
#if defined(__GLIBC__)
#include <malloc.h>
#include <array>
#endif
#include <bitset>
#include <filesystem>
#include "sentryward/value_analysis/scalar_fold_lowering.h"
#include "sentryward/value_analysis/scalar_fold_failure_codes.h"
#include "sentryward/scalar_fold/graph.h"
#include "sentryward/scalar_fold/artifact.h"
#include "production.hpp"
#include "fixed_input_context.hpp"
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif
namespace ptx = sentryward::frontend::ptx;
namespace alg = sentryward::algorithm;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

#include "run_options.hpp"
#include "run_resources.hpp"
#include "run_lifting.hpp"
#include "run_scalar_analysis.hpp"
#include "run_restarts.hpp"
