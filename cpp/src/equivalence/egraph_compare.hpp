#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::compare(std::span<const Class> a, std::span<const Class> b,bool compile_only,
               std::span<const unsigned char> storage_a,std::span<const unsigned char> storage_b) -> Result {
  if(compile_only && options_.certified_branch_scope)
    return result(Status::Unsupported,"conditional_scope_requires_output_comparison");
  if(options_.ideal_numeric_outputs && options_.certified_branch_scope)prepareBranchFacts();
  std::vector<Class> numeric_a,numeric_b;
  if(options_.ideal_numeric_outputs) {
    numeric_a.reserve(a.size());numeric_b.reserve(b.size());
    for(std::size_t i=0;i<a.size();++i) {
      poll();
      if(numericOutput(a[i]) && numericOutput(b[i])) {
        numeric_a.push_back(project(numericOutputProducer(a[i]),true));
        numeric_b.push_back(project(numericOutputProducer(b[i]),true));
      } else {
        numeric_a.push_back(project(a[i],false));
        numeric_b.push_back(project(b[i],false));
      }
    }
    a=numeric_a;b=numeric_b;
  } else {
    numeric_a.assign(a.begin(),a.end());numeric_b.assign(b.begin(),b.end());
  }
  for(auto& root:numeric_a)for(const auto* base:{"Exp2","Exp"})root=normalizeExponentialRatio(root,base);
  for(auto& root:numeric_b)for(const auto* base:{"Exp2","Exp"})root=normalizeExponentialRatio(root,base);
  compactImported(numeric_a,numeric_b);
  imported_node_count_=nodes_.size();
  // Normalize affine nested reductions before polynomial certificates add
  // cyclic alternatives such as zero=x-x. Reuse the same bounded rule;
  // there is no element expansion or second normalization implementation.
  {
    rebuild();const auto snapshot=nodes_;
    const auto admitted=nonrecursiveHeads(snapshot);
    std::vector<std::vector<Class>> members(nodes_.size());
    for(Class id=0;id<snapshot.size();++id)if(admitted[id])members[find(id)].push_back(id);
    std::unordered_map<Class,Class> replacements;
    for(Class id=0;id<snapshot.size();++id)if(admitted[id])
      flattenAffineFold(id,snapshot[id],snapshot,members,&replacements);
    // Publish the normalized acyclic import witness, not only an e-class
    // alias that the minimum-ID linear catalog cannot select. One traversal
    // of stored nodes/edges; neither Fold members nor output elements expand.
    if(!replacements.empty()) {
      const auto normalization_size=nodes_.size();std::unordered_map<Class,Class> mapped;
      std::unordered_set<Class> active;
      const auto normalize=[&](Class root) {
        std::vector<std::pair<Class,bool>> todo{{root,false}};
        while(!todo.empty()) {
          poll();auto [id,ready]=todo.back();todo.pop_back();
          if(mapped.contains(id))continue;
          const auto source=replacements.contains(id)?replacements.at(id):id;
          if(source>=normalization_size)throw std::logic_error("invalid import Fold replacement");
          if(!ready) {
            if(!active.insert(id).second)throw std::logic_error("cyclic import Fold normalization");
            todo.emplace_back(id,true);
            for(auto child:nodes_[source].children)if(!mapped.contains(child))todo.emplace_back(child,false);
          } else {
            auto key=nodes_[source];for(auto& child:key.children)child=mapped.at(child);
            mapped.emplace(id,intern(std::move(key)));active.erase(id);
          }
        }
        return mapped.at(root);
      };
      for(auto& root:numeric_a)root=normalize(root);
      for(auto& root:numeric_b)root=normalize(root);
      compactImported(numeric_a,numeric_b);
      imported_node_count_=nodes_.size();
    }
  }
  mergeAlphaEquivalentFolds();
  a=numeric_a;b=numeric_b;
  if(sentryward::debug::environment("SENTRYWARD_EGRAPH_PREFIX_TRACE"))diagnoseFinitePrefixSourceHeads(a,b);
  if(!compile_only)proveFirstFinitePrefixOutput(a,b);
  bool storage_equality_used=false;
  const auto equal = [&] {
    storage_equality_used=false;
    for (std::size_t i=0; i<a.size(); ++i) {
      poll();
      if (find(a[i]) != find(b[i])) {
        if(i>=storage_a.size() || i>=storage_b.size() || !storage_a[i] || !storage_b[i] ||
            !sameStoredOutput(a[i],b[i]))return false;
        storage_equality_used=true;
      }
    }
    return true;
  };
  for (std::size_t iteration = 0;; ++iteration) {
    rebuild();
    if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE")) {
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/blocks/egraph_trace_0.inc"
#endif
}
    if (!compile_only && equal()) return result(Status::Equivalent,
        storage_equality_used?"same_stored_bytes":"same_eclass");
    if (iteration == options_.max_iterations) {
      diagnose(a,b);
      return result(Status::NotProven, "iteration_budget");
    }
    statistics_.iterations = iteration + 1;
    const auto before = unions_;
    shifted_body_matches_.clear();
    const auto snapshot = nodes_;
    auto nonrecursive=nonrecursiveHeads(snapshot);
    mergeAssociativeNormalForms(snapshot,nonrecursive);
    mergePolynomialNormalForms(snapshot,nonrecursive);
    // Unions above merge each class's facts immediately, but an ancestor's
    // conservative dependency union is updated through the incremental use
    // queue. Drain it before interpreting an empty dependency set as closed.
    propagateProperties();
    // Algebraic equality may introduce recursive class alternatives. Keep
    // certificates, but do not unfold those alternatives in later matching.
    nonrecursive=nonrecursiveHeads(snapshot);
    mergeLinearReductionNormalForms(snapshot,nonrecursive,a,b);
    nonrecursive=nonrecursiveHeads(snapshot);
    factorAdditiveForests(snapshot,nonrecursive);
    propagateProperties(); // Newly emitted residues retain their exact dependencies.
    nonrecursive=nonrecursiveHeads(snapshot);
    std::vector<std::vector<std::size_t>> members(nodes_.size());
    for (std::size_t i = 0; i < snapshot.size(); ++i)if(nonrecursive[i])members[find(i)].push_back(i);
    rewriteScopedBranchPredicates(snapshot,members);
    const auto selects=indexSelectPatterns(snapshot,nonrecursive);
    std::map<std::tuple<std::string,std::uint32_t,std::uint64_t>,std::vector<Class>> sum_targets;
    for(Class id=0;id<snapshot.size();++id) {
      const auto& k=snapshot[id];
      if(nonrecursive[id] && k.kind==static_cast<int>(s::Kind::Fold) && k.fold &&
          k.fold->reducer==s::Reducer::Sum && k.fold->count &&
          k.attributes.empty() && k.provenance.empty() &&
          k.children.size()==2 && zero_[find(k.children[0])] && finite_[find(id)])
        sum_targets[{k.type,k.fold->binder_level,k.fold->count}].push_back(id);
    }
    for (std::size_t id = 0; id < snapshot.size(); ++id) {
      poll();
      // This fixed-head identity neither unfolds an eclass alternative nor
      // traverses children. Proven cyclic aliases must not hide the stored
      // compare-select witness; expansive patterns remain SCC-guarded below.
      if(rewriteKnownPredicateSelect(id,snapshot[id],snapshot))continue;
      if(rewriteFiniteCompareSelectExtremum(id,snapshot[id],snapshot))continue;
      // These inspect fixed local heads and cached interval facts only.
      // A harmless x%N=x alias must not suppress their exact conversion proof.
      rewriteUnsignedShift(id,snapshot[id]);
      rewriteExactIntegerNumeric(id,snapshot[id],members);
      if(!nonrecursive[id])continue;
      Key k = snapshot[id];
      const auto original_children = k.children;
      for (auto& c : k.children) c = find(c);
      if(auto singleton=singletonConstant(k)) {
        rewrite(id,intern(std::move(*singleton)));continue;
      }
      flattenAffineFold(id,k,snapshot,members);
      if(arithmetic(k) && k.name=="Add" && !coalesced_forests_.contains(id)) {
        coalesceInterleavedFolds(id,k,snapshot,members,sum_targets);
        if(!coalesced_forests_.contains(id) && finite_[find(id)] && has_fold_[find(id)])
          coalesceRectangularFolds(id,k,snapshot,members,sum_targets);
      }
      if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold->reducer==s::Reducer::Sum &&
          k.children.size()==2 && realType(k.type) && finite_[find(id)] &&
          has_fold_[find(k.children[1])] && !coalesced_forests_.contains(id))
        coalesceRectangularFolds(id,k,snapshot,members,sum_targets);
      rewriteSelectPatterns(id,k,original_children,snapshot,selects);
      rewriteExactU64Add(id,k,snapshot,selects);
      if(k.kind==static_cast<int>(s::Kind::Operation) && k.children.size()==2 &&
          k.provenance.empty() && k.attributes.empty() && integer_interval::parseType(k.type)) {
        const auto a=integer_ranges_[find(k.children[0])],b=integer_ranges_[find(k.children[1])];
        if(a && b && a->type==b->type && snapshot[k.children[0]].type==k.type &&
            snapshot[k.children[1]].type==k.type && a->lower>=0 &&
            b->lower==b->upper && b->lower>0 && a->lower/b->lower==a->upper/b->lower) {
          const auto quotient=a->lower/b->lower;
          if(k.name=="Remainder") {
            if(!quotient)rewrite(id,find(k.children[0]));
            else {
              Key offset;offset.kind=static_cast<int>(s::Kind::Constant);
              offset.type=k.type;offset.name=std::to_string(quotient*b->lower);
              auto subtract=k;subtract.name="Subtract";
              subtract.children={find(k.children[0]),intern(std::move(offset))};
              rewrite(id,intern(std::move(subtract)));
            }
          }
          if(k.name=="Divide") {
            Key zero;zero.kind=static_cast<int>(s::Kind::Constant);zero.type=k.type;zero.name=std::to_string(quotient);
            rewrite(id,intern(std::move(zero)));
          }
        }
      }
      if (k.kind == static_cast<int>(s::Kind::Fold) && k.fold->count == 0) {
        rewrite(id, k.children.at(0));
        continue;
      }
      if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold->reducer==s::Reducer::Sum &&
          zero_[find(k.children[1])] && finite_[find(k.children[0])]) {
        rewrite(id,k.children[0]);continue;
      }
      if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold->reducer==s::Reducer::Sum &&
          k.type=="real" && k.attributes.empty() && k.provenance.empty() && k.children.size()==2 &&
          !factored_folds_.contains(id) && k.children[0]<snapshot.size() && k.children[1]<snapshot.size() &&
          snapshot[k.children[0]].type==k.type &&
          snapshot[k.children[1]].type==k.type && snapshot[k.children[0]].provenance.empty() &&
          snapshot[k.children[1]].provenance.empty() && finite_[find(k.children[0])] &&
          finite_[find(k.children[1])] && !dependsOn(k.children[1],k.fold->binder_level)) {
        // ONE ideal-real invariant body, not count logical samples. The
        // conservative cached dependency union certifies absence of this
        // binder; other captured axes and the seed are retained unchanged.
        // Count0 and zero-body semantics were handled above. Reuse the
        // existing once-per-Fold factor marker and exact count literal.
        Key count;count.kind=static_cast<int>(s::Kind::Constant);count.type=k.type;
        count.name=std::to_string(k.fold->count);
        Key product;product.kind=static_cast<int>(s::Kind::Operation);product.type=k.type;
        product.name="Multiply";product.attributes="ideal_arithmetic";
        product.children={intern(std::move(count)),k.children[1]};
        auto total=intern(std::move(product));
        if(!zero_[find(k.children[0])]) {
          Key add;add.kind=static_cast<int>(s::Kind::Operation);add.type=k.type;
          add.name="Add";add.attributes="ideal_arithmetic";add.children={k.children[0],total};
          total=intern(std::move(add));
        }
        rewrite(id,total);factored_folds_.insert(id);continue;
      }
      if(idealOperation(k) && (k.name=="Maximum" || k.name=="Minimum")) {
        for(unsigned side=0;side<2;++side)
          if((k.name=="Maximum"?negative_infinity_[find(k.children[side])]:
              positive_infinity_[find(k.children[side])]) &&
              finite_[find(k.children[1-side])])rewrite(id,k.children[1-side]);
        for(auto a:members[original_children[0]])for(auto b:members[original_children[1]]) {
          const auto& left=snapshot[a];const auto& right=snapshot[b];
          if(left.kind!=static_cast<int>(s::Kind::Constant) ||
              right.kind!=static_cast<int>(s::Kind::Constant) || !left.provenance.empty() ||
              !right.provenance.empty() || !left.attributes.empty() || !right.attributes.empty())continue;
          double x=0,y=0;
          const auto xp=floatFromChars(left.name.data(),left.name.data()+left.name.size(),x);
          const auto yp=floatFromChars(right.name.data(),right.name.data()+right.name.size(),y);
          if(xp.ec!=std::errc{} || yp.ec!=std::errc{} || xp.ptr!=left.name.data()+left.name.size() ||
              yp.ptr!=right.name.data()+right.name.size() || !std::isfinite(x) || !std::isfinite(y))continue;
          // Parsing is monotone: strict inequality certifies the ordering.
          // Equal rounded values with distinct exact literals stay unknown.
          if(x<y)rewrite(id,k.name=="Maximum"?b:a);
          if(x>y)rewrite(id,k.name=="Maximum"?a:b);
        }
      }
      if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold->reducer==s::Reducer::Sum &&
          realType(k.type) && finite_[find(id)] && !split_seed_folds_.contains(id)) {
        // Initial is outside the binder: expose it once, not once per term.
        bool zero_seed=false;
        for(auto seed:members[original_children[0]]) {
          const auto& n=snapshot[seed];
          zero_seed |= n.kind==static_cast<int>(s::Kind::Constant) && n.type==k.type && n.name=="0";
        }
        if(!zero_seed) {
          Key zero;zero.kind=static_cast<int>(s::Kind::Constant);zero.type=k.type;zero.name="0";
          auto sum=k;sum.children[0]=intern(std::move(zero));
          Key add;add.kind=static_cast<int>(s::Kind::Operation);add.type=k.type;
          add.name="Add";add.attributes="ideal_arithmetic";
          add.children={k.children[0],intern(std::move(sum))};
          rewrite(id,intern(std::move(add)));split_seed_folds_.insert(id);
        }
      }
      if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold->reducer==s::Reducer::Sum &&
          realType(k.type) && finite_[find(id)] &&
          !factored_folds_.contains(id)) {
        const auto extract=[&](Class factor,Class body,bool division=false) {
          factor=find(factor);
          if(!finite_[factor] || dependsOn(factor,k.fold->binder_level) ||
              (division && !nonzero_[factor]))return;
          const auto emit=[&](Class initial) {
            auto sum=k;sum.children={find(initial),find(body)};
            const auto reduced=intern(std::move(sum));
            Key outer;outer.kind=static_cast<int>(s::Kind::Operation);outer.type=k.type;
            outer.name=division?"Divide":"Multiply";outer.attributes="ideal_arithmetic";
            outer.children=division?std::vector<Class>{reduced,factor}:std::vector<Class>{factor,reduced};
            rewrite(id,intern(std::move(outer)));factored_folds_.insert(id);
          };
          if(division) {
            // The initial value remains OUTSIDE division. Only a proven
            // zero seed may use this form; the existing seed-split rule
            // retains every nonzero seed separately before factoring.
            if(zero_[find(k.children[0])] && finite_[find(k.children[0])])emit(k.children[0]);
            return;
          }
          for(auto initial:members[original_children[0]]) {
            poll();const auto& init=snapshot[initial];
            if(init.kind==static_cast<int>(s::Kind::Constant) && init.type==k.type && init.name=="0")
              emit(initial);
            else if(arithmetic(init) && init.name=="Multiply" && init.type==k.type)
              for(unsigned j=0;j<2;++j)
                if(find(init.children[j])==factor)emit(init.children[1-j]);
          }
        };
        // ONE already-indexed complete local Divide witness, not a search
        // through body aliases or a parallel indicator arithmetic walk.
        // This new identity is confined to the explicit ideal-real view.
        if(k.type=="real" && k.attributes.empty() && k.provenance.empty() && k.children.size()==2) {
          if(auto quotient=selects.get(original_children[1]).quotient) {
            const auto numerator=find(quotient->first),denominator=find(quotient->second);
            if(numerator<snapshot.size() && denominator<snapshot.size() &&
                snapshot[numerator].type==k.type && snapshot[denominator].type==k.type &&
                finite_[numerator] && finite_[denominator] && nonzero_[denominator])
              extract(denominator,numerator,true);
          }
        }
        if(!factored_folds_.contains(id))for(auto term:members[original_children[1]]) {
          poll();const auto& product=snapshot[term];
          if(!arithmetic(product) || product.name!="Multiply" || product.type!=k.type)continue;
          for(unsigned side=0;side<2;++side) {
            extract(product.children[side],product.children[1-side]);
            if(factored_folds_.contains(id))break;
            // The equally local associative pattern u*(c*v). Do not search
            // arbitrary ancestors to find c; each successful match removes
            // one invariant factor and leaves the remaining Fold compact.
            for(auto member:members[product.children[side]]) {
              const auto& nested=snapshot[member];
              if(!arithmetic(nested) || nested.name!="Multiply" || nested.type!=k.type)continue;
              for(unsigned inner=0;inner<2;++inner) {
                const auto factor=find(nested.children[inner]);
                if(!finite_[factor] || dependsOn(factor,k.fold->binder_level))continue;
                auto body=product;body.children={find(product.children[1-side]),find(nested.children[1-inner])};
                extract(factor,intern(std::move(body)));
                if(factored_folds_.contains(id))break;
              }
              if(factored_folds_.contains(id))break;
            }
            if(factored_folds_.contains(id))break;
          }
          if(factored_folds_.contains(id))break;
        }
      }
      if(idealOperation(k)) {
        const auto make=[&](std::string name,std::vector<Class> children) {
          Key operation;operation.kind=static_cast<int>(s::Kind::Operation);
          operation.type=k.type;operation.name=std::move(name);
          operation.attributes="ideal_arithmetic";operation.children=std::move(children);
          return intern(std::move(operation));
        };
        const auto zero=[&] {
          Key constant;constant.kind=static_cast<int>(s::Kind::Constant);
          constant.type=k.type;constant.name="0";return intern(std::move(constant));
        };
        const auto one=[&] {
          Key constant;constant.kind=static_cast<int>(s::Kind::Constant);
          constant.type=k.type;constant.name="1";return intern(std::move(constant));
        };
        // Canonical two-sided clamp in the NUMERICAL projection only.
        // min(U,max(L,x)) == max(L,min(U,x)) for finite ideal-real x and
        // finite L<=U; the exact min_num/max_num orders differ on NaN.
        // Inspect only two immediate heads and cached finite facts. Emit
        // one canonical form, never a min/max distribution/member search.
        if(k.type=="real" && k.name=="Maximum" && k.children.size()==2 &&
            k.provenance.empty()) {
          const auto bound=[&](Class cls)->std::optional<double> {
            cls=find(cls);if(cls>=snapshot.size())return {};
            const auto& n=snapshot[cls];
            if(n.kind!=static_cast<int>(s::Kind::Constant) || n.type!="real" ||
                !n.attributes.empty() || !n.provenance.empty() || n.name.size()>128 ||
                !finite_[cls])return {};
            double result=0;
            const auto parsed=floatFromChars(n.name.data(),n.name.data()+n.name.size(),result);
            if(parsed.ec!=std::errc{} || parsed.ptr!=n.name.data()+n.name.size() ||
                !std::isfinite(result))return {};
            return result;
          };
          bool clamp_emitted=false;
          for(unsigned outside=0;outside<2 && !clamp_emitted;++outside) {
            const auto lower=find(k.children[outside]);
            const auto lo=bound(lower);if(!lo)continue;
            const auto inner_id=find(k.children[1-outside]);
            if(inner_id>=snapshot.size())continue;
            const auto& inner=snapshot[inner_id];
            if(!idealOperation(inner) || inner.type!="real" || inner.name!="Minimum" ||
                !inner.provenance.empty() || inner.children.size()!=2)continue;
            for(unsigned endpoint=0;endpoint<2;++endpoint) {
              const auto upper=find(inner.children[endpoint]),value=find(inner.children[1-endpoint]);
              const auto hi=bound(upper);
              if(!hi || value>=snapshot.size() || snapshot[value].type!="real" ||
                  !finite_[value] || !(lower==upper || *lo<*hi))continue;
              // Strict ordering of monotone rounded parses proves ordering
              // of exact real literals. Distinct equal parses stay Unknown.
              auto inside=make("Maximum",{lower,value});
              rewrite(id,make("Minimum",{upper,inside}));
              clamp_emitted=true;break; // One witness emits at most two enodes.
            }
          }
        }
        const auto literal=[&](Class id,const char* text) {
          const auto& n=snapshot[id];
          return n.kind==static_cast<int>(s::Kind::Constant) && n.type==k.type &&
            n.name==text && n.attributes.empty() && n.provenance.empty();
        };
        if(k.name=="Fma" && std::all_of(k.children.begin(),k.children.end(),[&](auto c){return finite_[find(c)];}))
          rewrite(id,make("Add",{make("Multiply",{k.children[0],k.children[1]}),k.children[2]}));
        if(k.name=="Subtract" && finite_[k.children[0]] && finite_[k.children[1]])
          rewrite(id,make("Add",{k.children[0],make("Negate",{k.children[1]})}));
        if(k.name=="Subtract" && negative_infinity_[find(k.children[0])] && finite_[find(k.children[1])])
          rewrite(id,k.children[0]);
        if((k.name=="Exp2" || k.name=="Exp") && negative_infinity_[find(k.children[0])])rewrite(id,zero());
        if(k.name=="Negate" && (negative_infinity_[find(k.children[0])] ||
            positive_infinity_[find(k.children[0])])) {
          // Exact sign change of a recognized extended-real sentinel. Do not
          // require finiteness or confer it on either infinity; unknown or
          // opaque values cannot enter this rule through decimal spelling.
          Key sentinel;sentinel.kind=static_cast<int>(s::Kind::Constant);
          sentinel.type=k.type;sentinel.name=positive_infinity_[find(k.children[0])]?"-inf":"inf";
          rewrite(id,intern(std::move(sentinel)));
        }
        if(k.name=="Divide" && finite_[find(k.children[0])])
          for(auto divisor:members[original_children[1]])if(literal(divisor,"1"))rewrite(id,k.children[0]);
        // One immediate exact unit witness, not Mul/Divide expansion or
        // denominator cancellation. Both sides retain the same finite,
        // nonzero real domain. The existing indexed unit set avoids scanning
        // numerator alternatives and emits at most one unary head per Divide.
        // Only the explicitly mathematical real view is admitted. Raw
        // unprojected f32/f64 producers may also have exact bit consumers;
        // their existing import path does not mint a separate Numeric view.
        if(k.type=="real" && k.name=="Divide" && k.provenance.empty() && k.children.size()==2 &&
            k.children[0]<snapshot.size() && k.children[1]<snapshot.size() &&
            snapshot[k.children[0]].type==k.type && snapshot[k.children[1]].type==k.type &&
            selects.ones.contains(k.children[0]) && finite_[k.children[1]] && nonzero_[k.children[1]])
          rewrite(id,make("Reciprocal",{k.children[1]}));
        if(k.name=="Divide" && find(k.children[0])==find(k.children[1]) &&
            finite_[find(k.children[0])] && nonzero_[find(k.children[0])])rewrite(id,one());
        // Primitive identities retain their real-domain guards. In particular,
        // sqrt(x*x) is not x for arbitrary x and rsqrt(0) is not finite.
        if(k.name=="Exp" && finite_[find(k.children[0])])
          for(auto operand:members[original_children[0]])
            if(literal(operand,"0"))rewrite(id,one());
        if(k.name=="Power" && positive_[find(k.children[0])] &&
            finite_[find(k.children[0])] && zero_[find(k.children[1])])
          rewrite(id,one());
        if(k.name=="Exp" || k.name=="Exp2" || k.name=="Log" || k.name=="Log2") {
          const auto inverse=k.name=="Exp"?"Log":k.name=="Exp2"?"Log2":
              k.name=="Log"?"Exp":"Exp2";
          const auto& heads=selects.get(original_children[0]).unary;
          if(auto found=heads.find(inverse);found!=heads.end()) {
            const auto argument=find(snapshot[found->second].children[0]);
            const bool exponential=k.name=="Exp" || k.name=="Exp2";
            if(finite_[argument] && (!exponential || positive_[argument]))
              rewrite(id,argument);
          }
          if(k.name=="Exp" || k.name=="Exp2") {
            // One more immediate head, not a search through the argument:
            // exp(-log(a))=1/a only inside the Log domain. A merely nonzero
            // a would incorrectly admit negative real logarithm arguments.
            if(auto negated=heads.find("Negate");negated!=heads.end()) {
              const auto negated_argument=find(snapshot[negated->second].children[0]);
              const auto& inner=selects.get(negated_argument).unary;
              if(auto logarithm=inner.find(inverse);logarithm!=inner.end()) {
                const auto argument=find(snapshot[logarithm->second].children[0]);
                if(finite_[argument] && positive_[argument])
                  rewrite(id,make("Reciprocal",{argument}));
              }
            }
          }
        }
        if(k.name=="Sqrt" && finite_[find(k.children[0])] &&
            nonnegative_[find(k.children[0])])
          for(auto operand:members[original_children[0]]) {
            if(literal(operand,"0"))rewrite(id,zero());
            if(literal(operand,"1"))rewrite(id,one());
          }
        if((k.name=="Rsqrt" || k.name=="ReciprocalSqrt") &&
            finite_[find(k.children[0])] && positive_[find(k.children[0])])
          rewrite(id,make("Reciprocal",{make("Sqrt",{k.children[0]})}));
        if(k.name=="Absolute" && finite_[find(k.children[0])]) {
          if(nonnegative_[find(k.children[0])])rewrite(id,k.children[0]);
          // One certified unary head from the shared epoch index; no
          // operand-ancestor search or logical Fold-element enumeration.
          const auto& unary=selects.get(original_children[0]).unary;
          if(auto negation=unary.find("Negate");negation!=unary.end()) {
            const auto value=find(snapshot[negation->second].children[0]);
            if(finite_[value])rewrite(id,make("Absolute",{value}));
          }
        }
        // A fixed-depth ideal-real vocabulary identity, not a global
        // coefficient rewrite or floating-point accuracy claim. Reuse the
        // epoch-local head index: inspect one product and its two unary
        // heads, never ancestors, alternative pairs, or Fold members.
        if(options_.ideal_numeric_outputs && k.type=="real" && k.name=="Exp2") {
          const auto& product=selects.get(original_children[0]).product;
          if(product)for(unsigned side=0;side<2;++side) {
            poll();const auto factor=find(side?product->second:product->first);
            const auto exponent=find(side?product->first:product->second);
            const auto& unary=selects.get(factor).unary;
            const auto logarithm=unary.find("Log2");
            if(logarithm==unary.end())continue;
            const auto base=find(snapshot[logarithm->second].children[0]);
            if(positive_[base] && finite_[base] && finite_[exponent])
              rewrite(id,make("Power",{base,exponent}));
          }
        }
        if(k.name=="Exp2" || k.name=="Exp")for(auto operand:members[original_children[0]]) {
          if(literal(operand,"0"))rewrite(id,one());
          const auto& difference=snapshot[operand];
          if(id<imported_node_count_ && !expanded_exponentials_.contains(id) && idealOperation(difference) &&
              difference.name=="Subtract" && finite_[find(difference.children[0])] &&
              finite_[find(difference.children[1])]) {
            auto left=make(k.name,{find(difference.children[0])});
            auto right=make(k.name,{make("Negate",{find(difference.children[1])})});
            rewrite(id,make("Multiply",{left,right}));
            expanded_exponentials_.insert(id);
          }
          if(id<imported_node_count_ && !expanded_exponentials_.contains(id) && arithmetic(difference) &&
              difference.name=="Add" && finite_[find(difference.children[0])] &&
              finite_[find(difference.children[1])]) {
            // Separate a finite shift independent of at least one coordinate
            // of the other operand. This is the same exp(a+b) identity,
            // including a negative literal times a maximum; no syntactic
            // Negate wrapper is required. Cached coordinate sets suffice.
            const auto& left_indices=free_indices_[find(difference.children[0])];
            const auto& right_indices=free_indices_[find(difference.children[1])];
            const bool independent_shift=
                (left_indices.size()<right_indices.size() &&
                 std::includes(right_indices.begin(),right_indices.end(),left_indices.begin(),left_indices.end())) ||
                (right_indices.size()<left_indices.size() &&
                 std::includes(left_indices.begin(),left_indices.end(),right_indices.begin(),right_indices.end()));
            if(k.type=="real" && independent_shift) {
              rewrite(id,make("Multiply",{make(k.name,{find(difference.children[0])}),
                                           make(k.name,{find(difference.children[1])})}));
              expanded_exponentials_.insert(id);continue;
            }
            for(unsigned side=0;side<2;++side)for(auto candidate:members[difference.children[side]]) {
              const auto& negative=snapshot[candidate];
              // Also retain a complete ideal-real subtractive offset as
              // ONE exponential factor: exp(x+(p-y))=exp(x)*exp(p-y).
              // This is a local finite identity, not distribution into a
              // Fold or a global rounded-coefficient rewrite. Exact/typed
              // views do not gain this new admission.
              const bool subtractive_offset=k.type=="real" && negative.type==k.type &&
                  negative.name=="Subtract" && negative.children.size()==2 &&
                  negative.provenance.empty() && finite_[find(negative.children[0])] &&
                  finite_[find(negative.children[1])];
              // Ratio recentering can leave a literal offset instead of a
              // subtractive one. Keep that exact real literal local too:
              // exp(x+p)=exp(x)*exp(p), without rewriting p itself.
              const bool literal_offset=k.type=="real" && negative.type==k.type &&
                  negative.kind==static_cast<int>(s::Kind::Constant) &&
                  negative.children.empty() && negative.attributes.empty() &&
                  negative.provenance.empty() && finite_[find(candidate)];
              if(!literal_offset && (!idealOperation(negative) ||
                  (negative.name!="Negate" && !subtractive_offset)))continue;
              rewrite(id,make("Multiply",{make(k.name,{find(difference.children[0])}),
                                           make(k.name,{find(difference.children[1])})}));
              expanded_exponentials_.insert(id);break;
            }
          }
        }
        if(k.name=="Negate" && finite_[k.children[0]]) {
          if(!negated_products_.contains(id)) {
            const auto& product=selects.get(original_children[0]).product;
            if(product && finite_[find(product->first)] && finite_[find(product->second)]) {
              rewrite(id,make("Multiply",{find(product->first),make("Negate",{find(product->second)})}));
              negated_products_.insert(id);
            }
          }
          for(auto operand:members[original_children[0]]) {
            const auto& n=snapshot[operand];
            if(literal(operand,"0"))rewrite(id,zero());
            if(!idealOperation(n))continue;
            if(n.name=="Negate")rewrite(id,find(n.children[0]));
            if(n.name=="Add" && find(n.children[0])!=find(operand) && find(n.children[1])!=find(operand))
              rewrite(id,make("Add",{make("Negate",{find(n.children[0])}),make("Negate",{find(n.children[1])})}));
          }
        }
        if(k.name=="Reciprocal" && finite_[k.children[0]] && nonzero_[k.children[0]]) {
          for(auto operand:members[original_children[0]]) {
            const auto& n=snapshot[operand];
            if(literal(operand,"1"))rewrite(id,one());
            if(idealOperation(n) && (n.name=="Exp2" || n.name=="Exp"))
              rewrite(id,make(n.name,{make("Negate",{find(n.children[0])})}));
            if(idealOperation(n) && n.name=="Multiply" &&
                positive_[find(n.children[0])] && positive_[find(n.children[1])])
              rewrite(id,make("Multiply",{make("Reciprocal",{find(n.children[0])}),make("Reciprocal",{find(n.children[1])})}));
          }
        }
        // Divide is canonical. The opposite Mul/Reciprocal -> Divide rule
        // below supplies equality without endlessly cycling between forms.
        if(k.name=="Multiply")for(unsigned side=0;side<2;++side) {
          const auto& quotient=selects.get(original_children[side]).quotient;
          auto other=find(k.children[1-side]);
          if(quotient && finite_[other] && finite_[find(quotient->first)] &&
              finite_[find(quotient->second)])
            rewrite(id,make("Divide",{make("Multiply",{other,find(quotient->first)}),find(quotient->second)}));
        }
        if(k.name=="Multiply" && !combined_exponentials_.contains(id))for(const auto* base:{"Exp2","Exp"}) {
          // Combine exponentials through one local product layer without
          // pushing anything into Fold. Work is over e-class alternatives
          // of this fixed-size pattern, not ancestors or Fold elements.
          struct ExponentialFactor {Class exponent;std::optional<Class> rest;};
          const auto factors=[&](Class child)->std::optional<ExponentialFactor> {
            const auto root=find(child);const auto& heads=selects.get(root);
            const auto direct=heads.unary.find(base);
            if(direct!=heads.unary.end() && finite_[root])
              return ExponentialFactor{find(snapshot[direct->second].children[0]),{}};
            if(!heads.product)return {};
            auto [a,b]=*heads.product;
            if(a==root || b==root)return {};
            for(unsigned side=0;side<2;++side) {
              auto exp_class=side?b:a,rest=side?a:b;
              const auto& unary=selects.get(exp_class).unary;
              const auto exp=unary.find(base);
              if(exp!=unary.end() && finite_[exp_class] && finite_[rest])
                return ExponentialFactor{find(snapshot[exp->second].children[0]),rest};
            }
            return {};
          };
          auto left=factors(original_children[0]),right=factors(original_children[1]);
          if(left && right) {
            const auto& l=*left;const auto& r=*right;
            auto combined=make(base,{make("Add",{l.exponent,r.exponent})});
            std::optional<Class> rest=l.rest;
            if(r.rest)rest=rest?make("Multiply",{*rest,*r.rest}):r.rest;
            rewrite(id,rest?make("Multiply",{combined,*rest}):combined);
            combined_exponentials_.insert(id);
          }
        }
        if(k.name=="Add" || k.name=="Multiply") {
          for(unsigned side=0;side<2;++side)for(auto operand:members[original_children[side]]) {
            poll();const auto& n=snapshot[operand];
            const auto other=k.children[1-side];
            if(!idealOperation(n))continue;
            if(k.name=="Add" && n.name=="Negate" && find(n.children[0])==other && finite_[other])
              rewrite(id,zero());
            if(k.name=="Multiply" && n.name=="Reciprocal" &&
                finite_[find(n.children[0])] && finite_[other]) {
              // This definitional form preserves the SAME possible zero
              // denominator on both sides; it asserts no nonzero/finite
              // result. Cancelling a denominator still needs nonzero below.
              rewrite(id,make("Divide",{other,find(n.children[0])}));
              if(find(n.children[0])==other && nonzero_[other])rewrite(id,one());
            }
          }
        }
        if(k.name=="Add") {
          // Factoring, not the reverse distributive expansion. Inspect the
          // two immediate product/Fold e-classes; never enumerate a domain.
          for(auto left:members[original_children[0]])for(auto right:members[original_children[1]]) {
            poll();const auto& l=snapshot[left];const auto& r=snapshot[right];
            if(arithmetic(l) && arithmetic(r) && l.name=="Multiply" && r.name=="Multiply" &&
                l.type==k.type && r.type==k.type && finite_[find(left)] && finite_[find(right)] &&
                !factored_sums_.contains(id)) {
              for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j)
                if(find(l.children[i])==find(r.children[j])) {
                  emitCommonAddFactor(id,k,find(l.children[i]),
                                      find(l.children[1-i]),find(r.children[1-j]));
                }
            }
            if(joined_sums_.contains(id))continue;
            const auto sumFold=[&](const Key& f,Class root) {
              return f.kind==static_cast<int>(s::Kind::Fold) && f.type==k.type &&
                  f.attributes.empty() && f.provenance.empty() && f.children.size()==2 &&
                  f.fold->reducer==s::Reducer::Sum && f.fold->count && finite_[find(root)] &&
                  zero_[find(f.children[0])];
            };
            if(!sumFold(l,left) || !sumFold(r,right) ||
                l.fold->binder_level!=r.fold->binder_level || l.fold->count>UINT64_MAX-r.fold->count)continue;
            for(unsigned order=0;order<2;++order) {
              const auto& a=order?r:l;const auto& b=order?l:r;
              if(!shiftedProjectedBody(a.children[1],b.children[1],a.fold->binder_level,
                                       a.fold->count,b.fold->count))continue;
              auto joined=a;joined.fold->count+=b.fold->count;
              rewrite(id,intern(std::move(joined)));joined_sums_.insert(id);break;
            }
          }
        }
        if(k.name=="Divide" && finite_[find(k.children[0])] && finite_[find(k.children[1])] &&
            nonzero_[find(k.children[1])]) {
          for(auto numerator:members[original_children[0]]) {
            const auto& n=snapshot[numerator];
            if(!arithmetic(n) || n.name!="Multiply")continue;
            for(auto denominator:members[original_children[1]]) {
              poll();const auto& d=snapshot[denominator];
              if(!arithmetic(d) || d.name!="Multiply")continue;
              for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j) {
                auto factor=find(n.children[i]), divisor=find(d.children[1-j]);
                if(factor==find(d.children[j]) && nonzero_[factor] &&
                    finite_[factor] && nonzero_[divisor] && finite_[divisor])
                  rewrite(id,make("Divide",{find(n.children[1-i]),divisor}));
              }
            }
          }
        }
        if(k.name=="Subtract" && k.children[0]==k.children[1] && finite_[find(k.children[0])])
          rewrite(id,zero());
        if(k.name=="Add" || k.name=="Multiply") {
          for(unsigned side=0;side<2;++side) {
            for(auto left:members[original_children[side]]) {
              poll();
              auto l=snapshot[left];
              for(auto& child:l.children)child=find(child);
              auto other=find(k.children[1-side]);
              if(k.name=="Add" && arithmetic(l) && l.name=="Add" &&
                  finite_[find(left)] && finite_[other]) {
                for(unsigned inner=0;inner<2;++inner) {
                  for(auto candidate:members[snapshot[left].children[inner]]) {
                    const auto& neg=snapshot[candidate];
                    if(idealOperation(neg) && neg.name=="Negate" && find(neg.children[0])==other)
                      rewrite(id,l.children[1-inner]);
                  }
                  for(auto candidate:members[original_children[1-side]]) {
                    const auto& neg=snapshot[candidate];
                    if(idealOperation(neg) && neg.name=="Negate" && find(neg.children[0])==l.children[inner])
                      rewrite(id,l.children[1-inner]);
                  }
                }
              }
              if(k.name=="Add" && idealOperation(l) && l.name=="Subtract" &&
                  finite_[find(left)] && finite_[other]) {
                if(l.children[1]==other)rewrite(id,l.children[0]);
                for(auto right:members[original_children[1-side]]) {
                  poll();auto r=snapshot[right];
                  for(auto& child:r.children)child=find(child);
                  if(idealOperation(r) && r.name=="Subtract" &&
                      l.children[1]==r.children[0] && finite_[find(right)])
                    rewrite(id,make("Subtract",{l.children[0],r.children[1]}));
                }
              }
            }
          }
        }
      }
      if (!arithmetic(k) || !finite_[find(k.children[0])] || !finite_[find(k.children[1])]) continue;
      auto x = k.children[0], y = k.children[1];
      Key swapped = k; std::swap(swapped.children[0], swapped.children[1]);
      rewrite(id, intern(std::move(swapped)));
      for (unsigned side = 0; side < 2; ++side) {
        auto other = side ? x : y;
        // Snapshot membership remains valid even when a rewrite joins classes.
        for (auto member : members[original_children[side]]) {
          poll();
          auto inner = snapshot[member];
          for (auto& c : inner.children) c = find(c);
          if (inner.kind == static_cast<int>(s::Kind::Constant) && inner.type == k.type) {
            if ((k.name == "Add" && inner.name == "0") ||
                (k.name == "Multiply" && inner.name == "1")) rewrite(id, other);
            if(k.name=="Multiply" && inner.name=="0" && finite_[find(other)])rewrite(id,find(member));
          }
          if (!arithmetic(inner) || inner.type != k.type) continue;
          // AC equality is certified by frozen multiset signatures below,
          // not by generating alternative binary bracketings.
          // Never emit the reverse distributive expansion: the common-
          // factor and invariant-Fold rules already canonicalize compactly.
        }
      }
      if ((k.name == "Maximum" || k.name == "Minimum") && find(x) == find(y))
        rewrite(id, x);
    }
    rebuild();
    if (!compile_only && equal()) return result(Status::Equivalent,
        storage_equality_used?"same_stored_bytes":"same_eclass");
    if (before == unions_) {
      diagnose(a,b);
      return result(Status::NotProven, "saturated");
    }
  }
}

} // namespace
} // namespace sentryward::equivalence
