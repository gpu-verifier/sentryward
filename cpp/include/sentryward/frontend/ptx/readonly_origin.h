#pragma once

#include "sentryward/frontend/ptx/config.h"
#include "sentryward/value_ir/graph.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace sentryward::frontend::ptx {
// Object provenance only. It never returns a numerical address or drops an
// integer offset. The caller's existing distinct-object address model still
// interprets pointer+integer as an object-relative byte address.
class AddressOriginAnalysis {
 public:
  using Resolve = std::function<value_ir::Value(value_ir::Value)>;
  explicit AddressOriginAnalysis(const value_ir::Graph&, Resolve = {});
  ~AddressOriginAnalysis();
  AddressOriginAnalysis(const AddressOriginAnalysis&) = delete;
  AddressOriginAnalysis& operator=(const AddressOriginAnalysis&) = delete;
  std::optional<std::string> globalObject(value_ir::Value);
  // Shared fixed ledger; used by the Store census, never reset per address.
  bool chargeWork(std::size_t);
  std::size_t work() const;
  bool exhausted() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Use ONLY a complete original raw capture (before value passes), paired with
// its hash-bound normalized original configuration. All modeled writes are
// retained as Store patterns by ValueIrExtractor; unsupported/opaque effects
// make capture incomplete before this API is called. No HB premise is needed
// for an object to which the complete executed scope contains no writes.
class ReadonlyOriginCensus {
 public:
  ReadonlyOriginCensus(const value_ir::Graph&, const FrontendConfig&);
  ~ReadonlyOriginCensus();
  bool complete() const;
  bool readonly(const std::string& resource) const;
  // The complete typed Load and original address remain part of the witness.
  // An origin is not permission to change its offset, width, tail or scope.
  std::optional<std::string> initialLoadObject(value_ir::Value);
  std::size_t storedWritePatterns() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace sentryward::frontend::ptx
