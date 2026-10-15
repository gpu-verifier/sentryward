#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace sentryward::equivalence::bounded_index_translation {

using NodeId = std::size_t;
enum class Kind { Literal, Index, Add, Multiply, Divide, Remainder };
struct Node {
  Kind kind = Kind::Literal;
  std::array<NodeId,2> children{};
  std::size_t child_count = 0;
  std::uint32_t index_level = 0;
  std::string_view type, literal;
  // Default deny. The resolver must verify the exact operation/arity, empty
  // attributes and provenance, and absence of conversion/opaque contexts.
  bool admitted = false;
};
struct IndexExtent { std::uint32_t level = 0; std::uint64_t count = 0; };
struct Limits {
  std::size_t max_nodes = 2048; // Combined two scoped node caches.
  std::size_t max_terms = 32;  // Per form, including linear index terms.
  std::size_t max_cached_terms = 8192;
  std::size_t max_atoms = 128;
  std::size_t max_indices = 64;
  std::size_t max_work = 131072;
};
// Hard admissible maxima are respectively 4096,64,32768,512,64,1048576;
// a larger configuration is rejected, never silently treated as unbounded.
struct Statistics { std::size_t nodes=0,atoms=0,cached_terms=0,work=0; bool exhausted=false; };

// Compare left(j+shift) with right(j) for 0 <= j < count, under the explicit
// lexical outer extents. The translated binder must not also occur in extents.
// Only [exact:]u64, nonnegative nonwrapping Add/constant-Multiply, and positive
// literal Divide/Remainder of affine numerators are recognized. Every source
// intermediate has an integer_interval certificate in its PARTICULAR scope.
// Constant multiples of the divisor may be extracted from a floor numerator;
// only multiples of the divisor may be removed from a remainder numerator.
// No rule asserts that a quotient itself is affine.
//
// A session resolves each stored node at most once PER SIDE. Unknown/cyclic
// nodes and exhausted caps are cached. Summaries combine range + normal form
// in one iterative DAG pass; no second range/ancestor walk, Fold expansion,
// logical element loop, or e-class-member scan occurs. The resolver's IDs and
// borrowed strings must remain stable for the session (append-only is fine;
// collection/remapping or congruence changes require a new session).
class Session {
 public:
  using Resolve = std::function<std::optional<Node>(NodeId)>;
  using Charge = std::function<bool(std::size_t)>;
  Session(Resolve,std::span<const IndexExtent>,std::uint32_t binder,
          std::uint64_t shift,std::uint64_t count,Limits = {},Charge = {});
  ~Session();
  Session(Session&&) noexcept;
  Session& operator=(Session&&) noexcept;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  // true is an exact certificate; false is Unknown/mismatch, NOT inequivalence.
  bool equivalent(NodeId left,NodeId right);
  Statistics statistics() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace sentryward::equivalence::bounded_index_translation
