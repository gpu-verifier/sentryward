#include "lowering_impl.hpp"

namespace sentryward::frontend::ptx {
void ModelLowerer::AddCta(const CtaFacts &facts, ValueAddresses value_addresses, bool lane_workers) {
  impl_->Add(facts, std::move(value_addresses), lane_workers);
}
const algorithm::SourceProgram &ModelLowerer::currentProgram() const noexcept {
  return impl_->source.program;
}
void ModelLowerer::TrackUnits(algorithm::UnitQuotient *quotient) noexcept { impl_->units = quotient; }
LoweringResult ModelLowerer::Finish() { return impl_->Finish(); }
} // namespace sentryward::frontend::ptx
