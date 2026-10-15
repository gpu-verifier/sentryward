#pragma once

#include <algorithm>
#include <cstddef>

namespace sentryward::equivalence::scoped_work_budget {
// Resource guards only, never a value/domain/body certificate. The aggregate
// is the ONE existing physical-epoch allowance borrowed by a coverage goal;
// the local counter belongs to the currently active full mapper body. A
// suspended aggregation caller has local==nullptr; a child installs its own
// local counter without changing aggregate. No allocation or ancestor list.
struct Guards {
  std::size_t* aggregate=nullptr;
  std::size_t* local=nullptr;
  bool exhausted() const noexcept {
    return (aggregate && !*aggregate) || (local && !*local);
  }
  std::size_t available(std::size_t upper) const noexcept {
    if(aggregate)upper=std::min(upper,*aggregate);
    if(local)upper=std::min(upper,*local);
    return upper;
  }
  // Precheck BOTH counters before any debit. Failure marks only exhausted
  // guards; it does not spend work that was never performed. Aliased guards
  // are one counter and are charged once. Overflow is avoided by comparison.
  bool charge(std::size_t amount) noexcept {
    bool denied=false;
    if(aggregate && amount>*aggregate){*aggregate=0;denied=true;}
    if(local && local!=aggregate && amount>*local){*local=0;denied=true;}
    if(denied)return false;
    if(aggregate)*aggregate-=amount;
    if(local && local!=aggregate)*local-=amount;
    return true;
  }
};
} // namespace sentryward::equivalence::scoped_work_budget
