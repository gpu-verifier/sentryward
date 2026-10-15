#pragma once
#include "sentryward/scalar_fold/graph.h"
#include "sentryward/equivalence/bit_origins.h"
#include <span>
#include <string_view>
#include <vector>
#include <memory>

namespace sentryward::equivalence {
enum class RewriteSemantics { ExactBits, ExactNumeric };
struct RepresentationRule {
  std::string_view id;
  RewriteSemantics semantics;
  std::string_view condition;
  scalar_fold::Ref (*apply)(scalar_fold::Graph&,scalar_fold::Ref);
  scalar_fold::Ref (*apply_scoped)(scalar_fold::Graph&,scalar_fold::Ref,
      BitOriginAnalysis&,const BitDomains&)=nullptr;
  // Conservative necessary header predicate; nullptr preserves old admission.
  // Shared by rule dispatch and session memo scheduling, never a proof itself.
  bool (*candidate)(const scalar_fold::Graph&,const scalar_fold::Ref&)=nullptr;
};
std::span<const RepresentationRule> representationRules();
scalar_fold::Ref rewriteRepresentation(scalar_fold::Graph&,scalar_fold::Ref,
    std::vector<std::string_view>* applied_rules=nullptr);
scalar_fold::Ref rewriteRepresentation(scalar_fold::Graph&,scalar_fold::Ref,
    BitOriginAnalysis&,const BitDomains&,
    std::vector<std::string_view>* applied_rules=nullptr);
struct RepresentationRewriteStatistics {
  std::size_t cached_uses=0,scopes=0;
  std::uint64_t cache_hits=0,cache_misses=0,lookup_work=0;
  BitOriginStatistics bit_origins;
  std::uint64_t candidate_uses=0,candidate_bypasses=0;
};
// Full-use memoization is distinct from producer bit tracing: the Ref includes
// all conversions and provenance, and the key includes its explicit lexical
// scope. Early certified results remain available after analysis exhaustion.
// Graph appends are safe; graph collection/remapping is not. <=4096 cached
// uses, <=128 scopes/64 levels, <=128-byte Ref metadata, <=80 rule IDs per use.
class RepresentationRewriter {
 public:
  explicit RepresentationRewriter(scalar_fold::Graph&);
  ~RepresentationRewriter();
  RepresentationRewriter(const RepresentationRewriter&)=delete;
  RepresentationRewriter& operator=(const RepresentationRewriter&)=delete;
  scalar_fold::Ref rewrite(scalar_fold::Ref,const BitDomains& = {},
      std::vector<std::string_view>* applied_rules=nullptr);
  RepresentationRewriteStatistics statistics() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
