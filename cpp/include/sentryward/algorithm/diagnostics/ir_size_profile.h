#pragma once
#ifdef SENTRYWARD_DEVELOPMENT_DIAGNOSTICS
#include "sentryward/development/ir_size_profile.h"
#else
#include <cstdint>
#include <initializer_list>
#include <string_view>
#include <utility>
namespace sentryward::algorithm {
inline void ReportIrSize(std::string_view,std::initializer_list<std::pair<std::string_view,std::uint64_t>>) {}
template<class N,class E> void ReportRecordGraphSize(std::string_view,const N&,const E&) {}
}
#endif
