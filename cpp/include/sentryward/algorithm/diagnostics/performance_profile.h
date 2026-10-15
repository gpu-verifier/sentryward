#pragma once
#ifdef SENTRYWARD_DEVELOPMENT_DIAGNOSTICS
#include "sentryward/development/performance_profile.h"
#else
#include <cstddef>
#include <cstdint>
#include <utility>
namespace sentryward::algorithm {
struct PerformanceProfile;
inline PerformanceProfile* CurrentPerformanceProfile() noexcept { return nullptr; }
inline void ProfileSnapshot() {}
struct ProfileWorkerSession { explicit ProfileWorkerSession(PerformanceProfile*) {} };
struct ProfileSession { explicit ProfileSession(bool) {} };
struct ProfileScope {
 explicit ProfileScope(const char*) {}
 void Stop() {}
 void Phase(std::uint32_t, std::uint32_t, std::size_t) {}
 std::uint64_t nodes=0,edges=0,pruned=0;
};
template<class Function> decltype(auto) ProfileCall(const char*,Function&& function) {
 return std::forward<Function>(function)();
}
}
#endif
