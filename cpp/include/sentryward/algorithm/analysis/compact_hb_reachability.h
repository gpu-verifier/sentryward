#pragma once
#include "sentryward/algorithm/analysis/hb_reachability.h"

namespace sentryward::algorithm {
// Returns null when persistent labels exceed the construction/storage budget.
std::unique_ptr<HbReachability> TryCompactHbReachability(
    const HappensBeforeGraph&, const HbReachabilityOptions&);
}
