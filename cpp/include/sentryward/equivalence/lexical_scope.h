#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>

namespace sentryward::equivalence::lexical_scope {

using Id = std::size_t;
using Count = std::optional<std::uint64_t>;
// Id zero denotes Unknown (including optional certificate exhaustion), never
// an empty or global scope. It carries no admissible binder/range facts.
inline constexpr Id unknown = 0;

struct Limits {
  std::size_t max_scopes = 128;
  std::size_t max_bindings = 2048;
  std::size_t max_metadata_slots = 16384;
  std::size_t max_work = 131072;
  std::uint32_t max_levels = 64;
};
struct Statistics {
  std::size_t scopes = 0, bindings = 0, metadata_slots = 0, work = 0;
  bool exhausted = false;
};

// One bounded comparison-local catalog. Complete structural keys distinguish
// unknown bounds, zero-length Fold bodies and different counts at the same
// lexical depth. No extent is inferred from an Index or from another use.
// Source graph identity/context remain part of the caller's projection memo.
class Catalog {
 public:
  using Charge = std::function<bool(std::size_t)>;
  explicit Catalog(Limits = {}, Charge = {});
  ~Catalog();
  Catalog(Catalog&&) noexcept;
  Catalog& operator=(Catalog&&) noexcept;
  Catalog(const Catalog&) = delete;
  Catalog& operator=(const Catalog&) = delete;

  // Only the caller's validated output domain may supply this known prefix.
  // Missing outer bounds stay Unknown; a supplied zero extent is malformed.
  // Create/retain the comparison root Id once: a root re-request is charged
  // and may return Unknown after work exhaustion, unlike cached bindBody().
  std::optional<Id> root(std::uint32_t outer_binders,
                         std::span<const std::uint64_t> known_prefix = {});
  // A Fold binds its BODY, not its seed. The caller retains parent for the
  // seed and passes this returned scope only to the body. Count zero records
  // an empty domain, not a singleton. Skips/rebinding and Unknown parent fail.
  std::optional<Id> bindBody(Id parent, std::uint32_t binder,
                            std::uint64_t count);
  // Derive an enclosing prefix ONLY from an actual validated use scope.
  // No absent/global extent is inferred. Cached prefixes survive exhaustion.
  std::optional<Id> prefix(Id actual, std::uint32_t rank);

  // O(1), allocation-free scalar access. Empty means Unknown/invalid.
  std::optional<std::uint32_t> rank(Id) const noexcept;
  Count count(Id, std::uint32_t level) const noexcept;
  // Borrowed immutable slots remain valid until destruction/move-assignment;
  // catalog insertion does not invalidate them. No record reconstruction.
  std::optional<std::span<const Count>> counts(Id) const noexcept;
  Statistics statistics() const noexcept;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace sentryward::equivalence::lexical_scope
