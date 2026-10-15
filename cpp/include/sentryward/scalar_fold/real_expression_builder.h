#pragma once

#include <functional>
#include <memory>
#include "sentryward/scalar_fold/reduction_builder.h"

namespace sentryward::scalar_fold {

// Constructs ideal-real expressions and extends structurally matched finite folds.
class RealExpressionBuilder {
 public:
  RealExpressionBuilder(Graph&, ReductionBuilder&, std::function<void()> poll = {});
  ~RealExpressionBuilder();
  RealExpressionBuilder(const RealExpressionBuilder&) = delete;
  RealExpressionBuilder& operator=(const RealExpressionBuilder&) = delete;

  // Exact PTX literal decoding precedes ideal-real update construction.
  Ref constant(std::string type, std::string literal);
  Ref operation(std::string name, std::string type, std::vector<Ref> operands,
                std::string attributes = {}, std::uint32_t binder_level = 0);
  std::size_t collect(std::span<Ref> roots);
  std::size_t collect(std::span<Ref> roots,std::span<const Ref> extendable,
                      std::span<std::optional<Id>> weak_nodes = {});
  struct Statistics {
    std::size_t templates=0,compiles=0,cache_hits=0,cache_misses=0,
      cache_capacity_rejections=0,query_work=0,compile_nodes=0,
      budget_rejections=0,summary_nodes=0,summary_remaps=0;
  };
  const Statistics& statistics() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace sentryward::scalar_fold
