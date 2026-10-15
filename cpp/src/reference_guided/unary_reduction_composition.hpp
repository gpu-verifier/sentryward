#pragma once
#include "linear_reduction_pattern.hpp"

namespace rg {
struct UnaryReductionCompositionLimits {
  size_t work = 40000, boundaries = 8, pairs = 32;
};

// Optional composition after certified state abstraction. This does not add an
// algebra rule: each replacement has a complete LinearReductionPattern proof
// of the entire argument, followed by ordinary unary congruence. The caller
// must still compare the COMPLETE resulting output with the reference.
class UnaryReductionComposition {
  Checker &checker;
  Arena &a;
  UnaryReductionCompositionLimits limits;
  size_t work = 0, attempted_pairs = 0;
  struct Miss { const char *why; };
  struct PairResult { bool equal = false; J proof; };
  std::map<std::pair<Id,Id>,PairResult> pairs;
  std::set<Id> active_snapshot;
  struct Skeleton { std::vector<Id> order, boundaries; };
  void charge(size_t amount = 1) {
    a.check();
    if (amount > limits.work - std::min(work,limits.work)) throw Miss{"work_budget"};
    work += amount;
  }
  bool unary(Id id) const {
    const auto &n=a.nodes[id];
    return (n.op=="reciprocalsqrt" || n.op=="sqrt" || n.op=="recip") &&
           n.args.size()==1 && n.facts.finite && a.nodes[n.args[0]].facts.finite;
  }
  Skeleton scan(Id root,const std::unordered_set<std::string>* operators=nullptr,
                const std::unordered_set<Id>* common=nullptr) {
    Skeleton result;
    std::unordered_set<Id> seen;
    std::vector<std::pair<Id,bool>> todo{{root,false}};
    while (!todo.empty()) {
      charge(); auto [id,ready]=todo.back(); todo.pop_back();
      if (ready) { result.order.push_back(id); continue; }
      if (!seen.insert(id).second) continue;
      const auto &n=a.nodes[id];
      if (n.op=="add" || n.op=="mul") {
        charge(n.args.size());
        todo.emplace_back(id,true);
        for (auto it=n.args.rbegin();it!=n.args.rend();++it) todo.emplace_back(*it,false);
      } else {
        result.order.push_back(id);
        if (unary(id) && (!operators || operators->contains(n.op)) &&
            (!common || !common->contains(id))) {
          if (result.boundaries.size()==limits.boundaries) throw Miss{"boundary_budget"};
          result.boundaries.push_back(id);
        }
      }
    }
    return result;
  }
  bool eligible_argument(Id root) {
    std::unordered_set<Id> seen; std::vector<Id> todo{root};
    while (!todo.empty()) {
      charge(); Id id=todo.back();todo.pop_back();if(!seen.insert(id).second)continue;
      const auto &n=a.nodes[id];
      if(n.op=="call" && n.data.size()==1 && n.data[0].denominator()==1 &&
         n.data[0]>=0 && n.data[0]<Q(checker.specs.size())) {
        Id sid=n.data[0].numerator().convert_to<Id>();const auto &s=checker.specs[sid];
        if(checker.active.contains(sid) && !s.maximum && s.facts.finite && s.count>0) return true;
      }
      if(n.op=="add" || n.op=="mul") {
        charge(n.args.size());todo.insert(todo.end(),n.args.begin(),n.args.end());
      }
    }
    return false;
  }
  const PairResult *prove_pair(Id reference,Id optimized) {
    charge(); const auto key=std::pair(reference,optimized);
    if(auto found=pairs.find(key);found!=pairs.end()) {
      ++a.stats["unary_reduction_composition_pair_cache_hits"];return &found->second;
    }
    if(attempted_pairs==limits.pairs) {
      ++a.stats["unary_reduction_composition_pair_budget_declines"];return nullptr;
    }
    ++attempted_pairs;++a.stats["unary_reduction_composition_argument_attempts"];
    LinearReductionPattern linear(checker);
    auto proof=linear.prove(reference,optimized);
    PairResult result;
    if(proof && proof->replacement==reference) {
      result.equal=true;result.proof=std::move(proof->receipt);
      ++a.stats["unary_reduction_composition_argument_proofs"];
    } else {
      ++a.stats["unary_reduction_composition_argument_decline_"+
                (linear.last_failure.empty()?std::string("no_target"):linear.last_failure)];
    }
    return &pairs.emplace(key,std::move(result)).first->second;
  }
public:
  std::string last_failure;
  J last_receipt=J::object();
  explicit UnaryReductionComposition(Checker &owner,UnaryReductionCompositionLimits budget={})
      :checker(owner),a(owner.a),limits(budget) {
    limits.work=std::min<size_t>(limits.work,40000);
    limits.boundaries=std::min<size_t>(limits.boundaries,8);
    limits.pairs=std::min<size_t>(limits.pairs,32);
  }
  std::optional<Id> rewrite(Id reference,Id optimized) {
    work=0;last_failure.clear();last_receipt=J::object();
    ++a.stats["unary_reduction_composition_attempts"];
    try {
      if(reference==optimized)return optimized;
      if(active_snapshot!=checker.active) {
        // Exact successful equalities remain valid within this Arena/premise
        // context. A failed search depends on the current reference vocabulary.
        for(auto it=pairs.begin();it!=pairs.end();) {
          if(!it->second.equal)it=pairs.erase(it);else ++it;
        }
        active_snapshot=checker.active;
        ++a.stats["unary_reduction_composition_context_changes"];
      }
      Skeleton left=scan(reference);
      std::unordered_set<Id> common(left.boundaries.begin(),left.boundaries.end());
      std::vector<Id> candidates;
      std::unordered_set<std::string> operators;
      for(Id id:left.boundaries) {
        charge();if(eligible_argument(a.nodes[id].args[0])) {
          candidates.push_back(id);operators.insert(a.nodes[id].op);
        }
      }
      if(candidates.empty())return optimized;
      // Collect only boundaries for which the reference has a reduction
      // target. Unrelated reciprocal factors and already-identical boundaries
      // cannot exhaust the bounded candidate list before matching begins.
      Skeleton right=scan(optimized,&operators,&common);
      std::unordered_map<Id,Id> replacements;
      J receipts=J::array();
      for(Id target:right.boundaries) {
        charge();if(common.contains(target))continue;
        const auto &n=a.nodes[target];
        for(Id source:candidates) {
          charge();const auto &r=a.nodes[source];
          if(n.op!=r.op || n.tag!=r.tag || n.data!=r.data)continue;
          const PairResult *result=prove_pair(r.args[0],n.args[0]);
          if(result && result->equal) {
            replacements.emplace(target,source);
            receipts.push_back({{"operator",n.op},{"reference_argument",r.args[0]},
                                {"optimized_argument",n.args[0]},{"argument_proof",result->proof}});
            break;
          }
        }
      }
      if(replacements.empty())return optimized;
      std::unordered_map<Id,Id> memo;
      for(Id id:right.order) {
        charge();
        if(auto replacement=replacements.find(id);replacement!=replacements.end()) {
          memo.emplace(id,replacement->second);continue;
        }
        const auto &n=a.nodes[id];Id value=id;
        if(n.op=="add" || n.op=="mul") {
          charge(n.args.size());std::vector<Id> args;args.reserve(n.args.size());bool changed=false;
          for(Id child:n.args) { Id next=memo.at(child);args.push_back(next);changed|=next!=child; }
          if(changed)value=checker.reapply(id,args);
        }
        memo.emplace(id,value);
      }
      Id result=memo.at(optimized);
      a.stats["unary_reduction_composition_replacements"]+=replacements.size();
      a.stats["unary_reduction_composition_stored_work"]+=work;
      last_receipt={{"method","complete_linear_argument_proofs_then_unary_congruence"},
                    {"boundaries",std::move(receipts)},{"stored_work",work},
                    {"whole_output_check","required_by_caller"}};
      return result;
    } catch(const Miss &miss) { last_failure=miss.why; }
      catch(const Unsupported &) { last_failure="unsupported_reapply"; }
    ++a.stats["unary_reduction_composition_decline_"+last_failure];
    a.stats["unary_reduction_composition_stored_work"]+=work;
    return {};
  }
};
} // namespace rg
