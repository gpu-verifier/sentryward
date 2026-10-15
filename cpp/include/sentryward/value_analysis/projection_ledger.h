#pragma once
#ifdef SENTRYWARD_ENABLE_COMPRESSION_AUDIT
#include "sentryward/development/projection_ledger.h"
#else
#define SENTRYWARD_LEDGER(...) ((void)0)
#endif
