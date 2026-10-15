#pragma once
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/trace.h"
#else
namespace sentryward::debug { inline const char* environment(const char*) {return nullptr;} }
#define SENTRYWARD_DEBUG_PRINT(id) ((void)0)
#endif
