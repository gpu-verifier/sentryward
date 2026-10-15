#pragma once
#ifdef SENTRYWARD_DEVELOPMENT_DIAGNOSTICS
#include "sentryward/development/hb_index_census.h"
#else
#include <cstddef>
#include <vector>
namespace sentryward::algorithm {
inline bool HbIndexCensusEnabled() { return false; }
inline void HbIndexCensus(const char*,const void*,std::size_t=0,std::size_t=0) {}
template<class T> void HbIndexVector(const char*,const void*,const std::vector<T>&) {}
template<class T> void HbIndexRows(const char*,const void*,const std::vector<std::vector<T>>&) {}
}
#endif
