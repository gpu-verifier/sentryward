#pragma once
#include "sentryward/scalar_fold/graph.h"
#include "sentryward/equivalence/integer_interval.h"
#include <optional>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <utility>

namespace sentryward::equivalence {
// Absent cell denotes a constant bit; inverted is then its Boolean value.
struct BitOrigin {
  std::optional<scalar_fold::Id> cell;
  unsigned bit=0;
  bool inverted=false;
  bool operator==(const BitOrigin&) const = default;
};
using BitOrigins=std::vector<BitOrigin>;
unsigned bitWidth(const std::string& type);
std::optional<BitOrigins> traceBitOrigins(const scalar_fold::Graph&,scalar_fold::Ref);

// Bounds belong to this use's validated lexical scope. No binder extent is
// inferred from the graph, and no Fold member or output element is enumerated.
using BitDomains=std::map<std::uint32_t,std::uint64_t>;
struct SymbolicBitOffset {
  std::optional<std::size_t> index;
  std::uint64_t coefficient=0,constant=0;
  bool operator==(const SymbolicBitOffset&) const = default;
};
struct SymbolicBitOrigin {
  enum class Kind { Unknown, Constant, Source };
  Kind kind=Kind::Unknown;
  std::size_t source=0;
  SymbolicBitOffset offset;
  bool inverted=false;
  bool operator==(const SymbolicBitOrigin&) const = default;
};
using SymbolicBitOrigins=std::vector<SymbolicBitOrigin>;
struct BitSource {
  enum class Kind { Memory, OpaqueValue };
  Kind kind=Kind::OpaqueValue;
  scalar_fold::Ref value;
  unsigned width=0;
};
struct SymbolicInteger {
  std::optional<std::size_t> index;
  std::uint64_t coefficient=0,constant=0,lower=0,upper=0;
};
struct BitOriginStatistics {
  std::size_t cached_nodes=0,scopes=0,sources=0,indices=0,work=0;
  bool exhausted=false;
  std::size_t cached_bit_nodes=0,cached_integer_nodes=0;
  std::uint32_t exhausted_limits=0;
};
enum class BitOriginLimit : std::uint32_t {
  Work=1,BitNodes=2,IntegerNodes=4,Scopes=8,Sources=16,Indices=32,Depth=64
};
// A graph-append-safe session. Do not collect/remap the graph while it lives.
// Hardware width <=64; depth <=256; <=4096 node/scope entries per cache;
// <=128 scopes, <=4096 sources, <=512 indices, <=64 domain levels, metadata
// <=128 bytes, and <=2^20 charged work units. Exhaustion yields Unknown.
class BitOriginAnalysis {
 public:
  explicit BitOriginAnalysis(const scalar_fold::Graph&);
  ~BitOriginAnalysis();
  BitOriginAnalysis(const BitOriginAnalysis&)=delete;
  BitOriginAnalysis& operator=(const BitOriginAnalysis&)=delete;
  std::optional<SymbolicBitOrigins> trace(scalar_fold::Ref,const BitDomains&);
  std::optional<SymbolicInteger> integer(scalar_fold::Ref,const BitDomains&);
  // Range-only facts reuse this same scoped/depth/work-capped integer cache.
  // A multi-index or bitwise range does NOT create a symbolic bit-origin alias.
  integer_interval::Result integerRange(scalar_fold::Ref,const BitDomains&);
  const BitSource* source(std::size_t) const;
  std::optional<std::pair<std::uint64_t,std::uint64_t>> bounds(const SymbolicBitOffset&) const;
  // Consumers charge their bounded certificates/field copies to this session.
  bool chargeWork(std::size_t);
  // Emit exactly the certified expression, optionally divided by a common
  // factor. A nondivisible coefficient/constant or invalid index fails.
  std::optional<scalar_fold::Ref> emit(scalar_fold::Graph&,const SymbolicBitOffset&,
      std::uint64_t divisor=1) const;
  BitOriginStatistics statistics() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
