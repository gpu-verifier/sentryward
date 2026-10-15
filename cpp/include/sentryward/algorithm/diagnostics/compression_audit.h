#pragma once
#ifdef SENTRYWARD_ENABLE_COMPRESSION_AUDIT
#include "sentryward/development/compression_audit.h"
#else
#define SENTRYWARD_AUDIT_STAGE(name) ((void)0)
#define SENTRYWARD_AUDIT(...) ((void)0)
#endif
