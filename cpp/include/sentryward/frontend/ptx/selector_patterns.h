#pragma once
#include <map>
#include <unordered_map>
#include "sentryward/algorithm/base/affine_index.h"
#include "sentryward/value_ir/graph.h"

namespace sentryward::frontend::ptx {
// Finite, exact thread selector interpretation. Only coordinate/index
// arithmetic is evaluated here, not tensor values or memory read-from.
class SelectorPatterns {
 public:
  struct Piece {algorithm::AffineDomain threads;__int128 first=0,step=0;};
  explicit SelectorPatterns(const value_ir::Graph&);
  std::vector<Piece> evaluate(value_ir::Value,std::uint64_t cta,std::uint64_t base_thread,
      algorithm::AffineDomain threads,bool rebind=true) const;
 private:
  const value_ir::Graph& graph_;
  std::map<std::pair<std::uint64_t,unsigned>,std::vector<Piece>> coordinates_;
  struct CachedEvaluation {
    value_ir::Value value;
    std::uint64_t cta,base;
    algorithm::AffineDomain threads;
    bool rebind;
    std::vector<Piece> result;
  };
  using Cache=std::unordered_map<std::size_t,std::vector<CachedEvaluation>>;
  std::vector<Piece> evaluate(value_ir::Value,std::uint64_t,std::uint64_t,
      algorithm::AffineDomain,bool,unsigned,Cache&) const;
  std::vector<Piece> evaluateImpl(value_ir::Value,std::uint64_t,std::uint64_t,
      algorithm::AffineDomain,bool,unsigned,Cache&) const;
};
}
