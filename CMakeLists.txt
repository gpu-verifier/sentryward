cmake_minimum_required(VERSION 3.27)

project(sentryward LANGUAGES CXX)
find_package(Threads REQUIRED)

set(SENTRYWARD_ASSUMPTION_DEFAULTS "${CMAKE_CURRENT_SOURCE_DIR}/config/default-assumptions.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SENTRYWARD_ASSUMPTION_DEFAULTS}")
file(READ "${SENTRYWARD_ASSUMPTION_DEFAULTS}" SENTRYWARD_ASSUMPTION_DEFAULTS_JSON)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/config/assumption-pool.json")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/config/assumption-pool.json" SENTRYWARD_ASSUMPTION_POOL_JSON)
configure_file(cpp/include/sentryward/algorithm/source_ir/default_assumptions.h.in
  generated/sentryward/algorithm/source_ir/default_assumptions.h @ONLY)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build configuration" FORCE)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/checks.cmake")
  option(SENTRYWARD_BUILD_TESTS "Build private C++ correctness tests" OFF)
else()
  if(SENTRYWARD_BUILD_TESTS)
    message(WARNING "Private tests are absent; ignoring SENTRYWARD_BUILD_TESTS=ON.")
  endif()
  unset(SENTRYWARD_BUILD_TESTS CACHE)
  set(SENTRYWARD_BUILD_TESTS OFF)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/support/cpp/tools/CMakeLists.txt")
  option(SENTRYWARD_BUILD_INTERNAL_TOOLS "Build internal diagnostic executables" OFF)
else()
  if(SENTRYWARD_BUILD_INTERNAL_TOOLS)
    message(WARNING "Private tools are absent; ignoring SENTRYWARD_BUILD_INTERNAL_TOOLS=ON.")
  endif()
  unset(SENTRYWARD_BUILD_INTERNAL_TOOLS CACHE)
  set(SENTRYWARD_BUILD_INTERNAL_TOOLS OFF)
endif()
option(SENTRYWARD_WARNINGS_AS_ERRORS "Treat C++ warnings as errors" OFF)
option(SENTRYWARD_ENABLE_SANITIZERS "Enable address and undefined sanitizers" OFF)
option(SENTRYWARD_WITH_Z3 "Link Z3 for the symbolic address model's SMT layer" ON)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/support/cpp/diagnostics/CMakeLists.txt")
  option(SENTRYWARD_ENABLE_DEVELOPMENT_DIAGNOSTICS "Enable private development tracing" OFF)
elseif(SENTRYWARD_ENABLE_DEVELOPMENT_DIAGNOSTICS)
  message(FATAL_ERROR "Development tracing requires private diagnostics sources.")
endif()
if(SENTRYWARD_ENABLE_DEVELOPMENT_DIAGNOSTICS OR SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/tests/support/cpp/diagnostics/include")
endif()
if(SENTRYWARD_ENABLE_DEVELOPMENT_DIAGNOSTICS)
  add_compile_definitions(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS)
endif()
option(SENTRYWARD_ENABLE_COMPRESSION_AUDIT "Compile opt-in developer compression diagnostics" OFF)
if(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
  add_compile_definitions(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
endif()

add_library(sentryward_algorithm STATIC
  cpp/src/analysis/family_relation_query.cpp
  cpp/src/memory/family_access.cpp
  cpp/src/checker/family_memory_checker.cpp
  cpp/src/checker/cross_unit_races.cpp
  cpp/src/memory/family_conflict_order.cpp
  cpp/src/analysis/family_hb_reachability.cpp
  cpp/src/diagnostics/scoped_deadline.cpp
  cpp/src/source_ir/diagnostic.cpp
  cpp/src/source_ir/builder.cpp
  cpp/src/source_ir/initialization_recovery.cpp
  cpp/src/source_ir/ptx_assumptions.cpp
  cpp/src/source_ir/unobserved_synchronization.cpp
  cpp/src/source_ir/validator.cpp
  cpp/src/event_ir/dag.cpp
  cpp/src/event_ir/lowering.cpp
  cpp/src/event_ir/origin_map.cpp
  cpp/src/event_ir/program.cpp
  cpp/src/semantics/barrier_state.cpp
  cpp/src/semantics/collective_state.cpp
  cpp/src/semantics/reference_trace.cpp
  cpp/src/semantics/simulator.cpp
  cpp/src/solver/dinic.cpp
  cpp/src/solver/minimum_weight_ideal.cpp
  cpp/src/analysis/phase_database.cpp
  cpp/src/analysis/phase_algebra.cpp
  cpp/src/analysis/phase_boundary.cpp
  cpp/src/analysis/happens_before_graph.cpp
  cpp/src/analysis/trace_order.cpp
  cpp/src/analysis/reachability.cpp
  cpp/src/analysis/parallel_reachability.cpp
  cpp/src/analysis/shared_reachability_index.cpp
  cpp/src/analysis/hb_reachability.cpp
  cpp/src/analysis/compact_hb_reachability.cpp
  cpp/src/analysis/wait_checker.cpp
  cpp/src/analysis/wait_reachability_batch.cpp
  cpp/src/analysis/lifecycle_checker.cpp
  cpp/src/analysis/unordered_access_pairs.cpp
  cpp/src/memory/access.cpp
  cpp/src/memory/symbolic_address_model.cpp
  cpp/src/memory/constant_address_model.cpp
  cpp/src/memory/constant_address_candidates.cpp
  cpp/src/memory/conflict_order.cpp
  cpp/src/checker/wellsync_checker.cpp
  cpp/src/checker/memory_race_checker.cpp
  cpp/src/checker/duplicate_tma_stores.cpp
  cpp/src/checker/verifier.cpp
)
add_library(sentryward::algorithm ALIAS sentryward_algorithm)
target_link_libraries(sentryward_algorithm PUBLIC Threads::Threads)

# Layer 2 of the symbolic address model uses the Z3 C++ API. A matching-
# architecture z3-solver Python wheel also ships z3/include and z3/lib.
if(SENTRYWARD_WITH_Z3)
  find_path(SENTRYWARD_Z3_INCLUDE_DIR z3++.h
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/build/native-deps/z3/include)
  find_library(SENTRYWARD_Z3_LIBRARY NAMES z3 libz3
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/build/native-deps/z3/lib)
  if(NOT SENTRYWARD_Z3_INCLUDE_DIR OR NOT SENTRYWARD_Z3_LIBRARY)
    message(FATAL_ERROR "Z3 not found. Set SENTRYWARD_Z3_INCLUDE_DIR and SENTRYWARD_Z3_LIBRARY, "
      "or configure with -DSENTRYWARD_WITH_Z3=OFF (symbolic offsets then stay unresolved).")
  endif()
  target_sources(sentryward_algorithm PRIVATE cpp/src/memory/address_solver_z3.cpp)
  target_include_directories(sentryward_algorithm SYSTEM PRIVATE ${SENTRYWARD_Z3_INCLUDE_DIR})
  target_link_libraries(sentryward_algorithm PRIVATE ${SENTRYWARD_Z3_LIBRARY})
else()
  target_sources(sentryward_algorithm PRIVATE cpp/src/memory/address_solver_none.cpp)
endif()

target_include_directories(sentryward_algorithm
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)
target_compile_features(sentryward_algorithm PUBLIC cxx_std_20)
target_include_directories(sentryward_algorithm SYSTEM PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/nlohmann_json/single_include)

if(MSVC)
  target_compile_options(sentryward_algorithm PRIVATE /W4)
  if(SENTRYWARD_WARNINGS_AS_ERRORS)
    target_compile_options(sentryward_algorithm PRIVATE /WX)
  endif()
else()
  target_compile_options(sentryward_algorithm PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  if(SENTRYWARD_WARNINGS_AS_ERRORS)
    target_compile_options(sentryward_algorithm PRIVATE -Werror)
  endif()
endif()

if(SENTRYWARD_ENABLE_SANITIZERS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(sentryward_algorithm PUBLIC
      -fsanitize=address,undefined
      -fno-omit-frame-pointer
    )
    target_link_options(sentryward_algorithm PUBLIC
      -fsanitize=address,undefined
      -fno-omit-frame-pointer
    )
  else()
    message(FATAL_ERROR "SENTRYWARD_ENABLE_SANITIZERS requires Clang or GCC")
  endif()
endif()

add_library(sentryward_io STATIC cpp/src/io/source_json.cpp
  cpp/src/io/source_storage.cpp
)
target_link_libraries(sentryward_io PUBLIC sentryward::algorithm)
target_compile_options(sentryward_io PRIVATE $<TARGET_PROPERTY:sentryward_algorithm,COMPILE_OPTIONS>)
target_include_directories(sentryward_io SYSTEM PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/nlohmann_json/single_include)

add_library(sentryward_value_ir STATIC
  cpp/src/value_ir/type.cpp
  cpp/src/value_ir/operations.cpp
  cpp/src/value_ir/native_matrix.cpp
  cpp/src/value_ir/ptx_operation.cpp
  cpp/src/value_ir/graph.cpp
 )

add_library(sentryward::value_ir ALIAS sentryward_value_ir)
target_include_directories(sentryward_value_ir PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
target_include_directories(sentryward_value_ir SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vendor/nlohmann_json/single_include)
target_compile_features(sentryward_value_ir PUBLIC cxx_std_20)
target_compile_options(sentryward_value_ir PRIVATE $<TARGET_PROPERTY:sentryward_algorithm,COMPILE_OPTIONS>)
if(SENTRYWARD_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(sentryward_value_ir PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(sentryward_value_ir PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
endif()

# PTX syntax/semantic decoding is independent of Python and graph verification.
add_library(sentryward_value_analysis STATIC
  cpp/src/scalar_fold/graph.cpp
  cpp/src/scalar_fold/consolidate_outputs.cpp
  cpp/src/scalar_fold/reduction_builder.cpp
  cpp/src/scalar_fold/real_expression_builder.cpp
  cpp/src/equivalence/egraph.cpp
  cpp/src/equivalence/constant_values.cpp
  cpp/src/equivalence/conversion_uses.cpp
  cpp/src/equivalence/output_mapping.cpp
  cpp/src/equivalence/input_mapping.cpp
  cpp/src/equivalence/packed_values.cpp
  cpp/src/equivalence/bit_origins.cpp
  cpp/src/equivalence/representation_rewriting.cpp
  cpp/src/equivalence/numerical_abstraction.cpp
  cpp/src/equivalence/numeric_semantics.cpp
  cpp/src/equivalence/math_intrinsics.cpp
  cpp/src/equivalence/polynomial_certificate.cpp
  cpp/src/equivalence/integer_interval.cpp
  cpp/src/equivalence/bounded_linear_reduction.cpp
  cpp/src/equivalence/finite_prefix_certificate.cpp
  cpp/src/equivalence/epoch_work_allowance.cpp
  cpp/src/equivalence/causal_region_certificate.cpp
  cpp/src/equivalence/bounded_index_translation.cpp
  cpp/src/equivalence/moment_certificate.cpp
  cpp/src/equivalence/comparison_context.cpp
  cpp/src/equivalence/lexical_scope.cpp
  cpp/src/equivalence/fold_rewriting.cpp
  cpp/src/equivalence/output_family.cpp
  cpp/src/equivalence/partitioned_outputs.cpp
  cpp/src/value_analysis/scalar_fold_lowering.cpp
  cpp/src/value_analysis/passes.cpp
  cpp/src/value_analysis/view_normalization.cpp
  cpp/src/value_analysis/indexed_source.cpp
  cpp/src/value_analysis/memory_elimination.cpp
 )
add_library(sentryward::value_analysis ALIAS sentryward_value_analysis)
add_library(sentryward_reference_guided STATIC
  cpp/src/reference_guided/comparison.cpp cpp/src/reference_guided/production.cpp)
target_include_directories(sentryward_reference_guided PUBLIC cpp/src/reference_guided cpp/include)
target_include_directories(sentryward_reference_guided SYSTEM PUBLIC vendor/nlohmann_json/single_include)
target_link_libraries(sentryward_reference_guided PUBLIC sentryward::value_analysis)

target_link_libraries(sentryward_value_analysis PUBLIC sentryward::value_ir)
target_compile_options(sentryward_value_analysis PRIVATE $<TARGET_PROPERTY:sentryward_algorithm,COMPILE_OPTIONS>)

add_library(sentryward_ptx STATIC
  cpp/src/frontend/ptx/parser.cpp
  cpp/src/frontend/ptx/semantics.cpp
  cpp/src/frontend/ptx/operands.cpp
  cpp/src/frontend/ptx/config.cpp
  cpp/src/frontend/ptx/public_input_parse.cpp
  cpp/src/frontend/ptx/public_input_normalize.cpp
  cpp/src/frontend/ptx/public_input_inference.cpp
  cpp/src/frontend/ptx/public_input_memory.cpp
  cpp/src/frontend/ptx/execution.cpp
  cpp/src/frontend/ptx/execution_records.cpp
  cpp/src/frontend/ptx/shared_execution.cpp
  cpp/src/frontend/ptx/unit_facts.cpp
  cpp/src/frontend/ptx/unit_frame.cpp
  cpp/src/frontend/ptx/unit_classes.cpp
  cpp/src/frontend/ptx/query_log.cpp
  cpp/src/frontend/ptx/tensor_maps.cpp
  cpp/src/frontend/ptx/layouts.cpp
  cpp/src/frontend/ptx/known_bits.cpp
  cpp/src/frontend/ptx/tmem_lifetime.cpp
  cpp/src/frontend/ptx/resources.cpp
  cpp/src/frontend/ptx/slot_resolver.cpp
  cpp/src/frontend/ptx/matrix.cpp
  cpp/src/frontend/ptx/refinement.cpp
  cpp/src/frontend/ptx/lowering.cpp
  cpp/src/frontend/ptx/value_ir.cpp
  cpp/src/frontend/ptx/branch_scope_capture.cpp
  cpp/src/frontend/ptx/readonly_origin.cpp
  cpp/src/frontend/ptx/rounded_sign.cpp
  cpp/src/frontend/ptx/address_expression.cpp
  cpp/src/frontend/ptx/semantic_program.cpp
  cpp/src/frontend/ptx/serialization.cpp
  cpp/src/frontend/ptx/selector_patterns.cpp
  cpp/src/frontend/ptx/group2_lifetime.cpp)
target_include_directories(sentryward_ptx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
target_compile_features(sentryward_ptx PUBLIC cxx_std_20)
target_compile_options(sentryward_ptx PRIVATE $<TARGET_PROPERTY:sentryward_algorithm,COMPILE_OPTIONS>)

find_package(OpenSSL REQUIRED COMPONENTS Crypto)
find_path(SENTRYWARD_BOOST_INCLUDE_DIR boost/multiprecision/cpp_int.hpp
  HINTS ${CMAKE_CURRENT_SOURCE_DIR}/build/native-deps/boost/usr/include REQUIRED)
target_include_directories(sentryward_ptx SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/nlohmann_json/single_include
  ${SENTRYWARD_BOOST_INCLUDE_DIR})
target_link_libraries(sentryward_ptx PUBLIC sentryward_io sentryward::value_ir OpenSSL::Crypto)
target_link_libraries(sentryward_value_analysis PUBLIC sentryward_ptx)
add_executable(sentryward-ptx cpp/src/frontend/ptx/run.cpp)
target_include_directories(sentryward-ptx PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(sentryward-ptx PRIVATE sentryward_ptx sentryward::value_analysis sentryward_reference_guided)

# Python wheels (scikit-build-core) carry the verifier and configuration
# inside the package; sentryward/_paths.py resolves them there.
if(SKBUILD)
  install(TARGETS sentryward-ptx
    RUNTIME DESTINATION sentryward/_native)
  install(FILES config/default-assumptions.json config/assumption-pool.json config/result-categories.json
    DESTINATION sentryward/_config)
endif()

if(SENTRYWARD_BUILD_INTERNAL_TOOLS)
  include(tests/support/cpp/tools/CMakeLists.txt)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/support/cpp/exports/CMakeLists.txt" AND
   (SENTRYWARD_BUILD_INTERNAL_TOOLS OR SENTRYWARD_BUILD_TESTS OR SENTRYWARD_ENABLE_COMPRESSION_AUDIT))
  include(tests/support/cpp/exports/CMakeLists.txt)
elseif(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
  message(FATAL_ERROR "Compression audit requires private development exports.")
endif()

if(SENTRYWARD_BUILD_TESTS)
  include(tests/cmake/checks.cmake)
endif()

if(SENTRYWARD_ENABLE_DEVELOPMENT_DIAGNOSTICS)
  include(tests/support/cpp/diagnostics/CMakeLists.txt)
endif()
