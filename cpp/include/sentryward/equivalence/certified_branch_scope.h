#pragma once
#include "sentryward/equivalence/comparison_context.h"
#include "sentryward/scalar_fold/graph.h"
#include <array>
#include <map>
#include <memory>
#include <span>

namespace sentryward::equivalence {
struct InputIdentityMapping;
// A separately hash-bound, original reached-scope hook. ComparisonContext v1
// remains fixed-readonly-only. This is not a source of unconditional truths.
class CertifiedBranchScope {
 public:
  struct Piece {
    scalar_fold::Ref expression;
    std::array<scalar_fold::Ref,3> encoded_operands;
    std::string format;
    std::uint64_t begin=0,count=0; // Original global thread.x = local i0+begin.
    std::size_t observation=0,reached_group=0;
    std::uint64_t source_line=0,instruction=0,occurrence=0;
    bool taken=false,predicate_negated=false;
  };

  const scalar_fold::Graph& graph() const {return graph_;}
  std::span<const Piece> pieces() const {return pieces_;}
  const std::string& digest() const {return digest_;}
  const std::map<std::string,std::string>& sourceHashes() const {return hashes_;}
 private:
  friend class CertifiedBranchScopeReader;
  CertifiedBranchScope()=default;
  scalar_fold::Graph graph_;
  std::vector<Piece> pieces_;
  std::map<std::string,std::string> hashes_;
  std::string digest_;
};
// A Piece is a CONDITIONAL implication only: a same-format, full scoped body
// relation must map EVERY exact encoded leaf to its finite Numeric counterpart
// and prove the consumer's actual/masked domain is covered by reached Pieces.
// No graph Ref or ideal sign is globally aliased by this factory. Exact,
// predicate, bit and address consumers do not consume this hook.
} // namespace sentryward::equivalence
