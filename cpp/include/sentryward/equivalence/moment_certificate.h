#pragma once

#include "sentryward/equivalence/bounded_linear_reduction.h"
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string_view>
#include <tuple>
#include <vector>

namespace sentryward::equivalence::moment_certificate {
using Coefficient=bounded_linear_reduction::Coefficient;
using Interval=bounded_linear_reduction::WeightedInterval;
using MeanId=std::size_t;
struct Limits {
  std::size_t max_means=512,max_support_entries=32768,max_intervals=1024;
  std::size_t max_square_terms=512,max_work=1000000;
  polynomial_certificate::Limits coefficients;
};
struct MeanRequest {
  // Caller supplies exact same-body affine coverage of the ACTUAL linear
  // expression, not an assumed mean. Tokens denote validated type/scope/body
  // keys local to this proof epoch. Finiteness defaults to deny.
  std::size_t group=0,context=0;
  Coefficient constant;
  std::span<const Interval> weighted_samples;
  bool finite=false;
};
struct SquareTerm {MeanId mean;Coefficient weight;bool operator==(const SquareTerm&) const=default;};
struct Signature {
  const std::vector<Interval>& secondMoment() const {return second_moment;}
  const std::vector<SquareTerm>& meanSquares() const {return mean_squares;}
 private:
  std::size_t group=0,context=0;
  std::vector<Interval> second_moment;
  std::vector<SquareTerm> mean_squares;
  friend class Builder;
  friend bool equal(const Signature&,const Signature&);
  std::uint64_t owner=0;
};
bool equal(const Signature&,const Signature&);
class Builder {
 private:
  struct FamilyTerm {
    std::uint64_t begin=0,step=1,count=0;
    Coefficient weight;
  };
 public:
  // Translation of a validated uniform mean's entire support, not an
  // assumed numeric range. The caller certifies the same body/captured scope
  // for every index and the exact common affine translation coefficient.
  struct Translation {std::uint64_t begin=0,step=1,count=1;};
  struct Correction {
   private:
    friend class Builder;
    std::array<MeanId,3> means;std::array<Coefficient,3> weights;
    std::uint64_t owner=0;
  };
  struct Form {
   private:
    friend class Builder;
    std::optional<std::pair<std::size_t,std::size_t>> scope;
    std::vector<Interval> second_moment;
    std::map<MeanId,Coefficient> mean_squares;
    std::uint64_t owner=0;
    bool valid=true;
  };
  struct FamilyForm {
   private:
    friend class Builder;
    std::optional<std::pair<std::size_t,std::size_t>> scope;
    std::map<std::size_t,std::vector<FamilyTerm>> terms;
    std::size_t term_count=0;
    std::uint64_t owner=0;
    bool valid=true;
  };
  struct FamilySignature {
    std::size_t storedTerms() const {return term_count;}
   private:
    friend class Builder;
    std::pair<std::size_t,std::size_t> scope;
    std::map<std::size_t,std::vector<FamilyTerm>> terms;
    std::size_t term_count=0;
    std::uint64_t owner=0;
  };
  explicit Builder(Limits={});
  Builder(const Builder&)=delete;
  Builder& operator=(const Builder&)=delete;
  Builder(Builder&&)=delete;
  Builder& operator=(Builder&&)=delete;
  std::optional<MeanId> mean(const MeanRequest&);
  // Exact local identity only: coefficient MUST be nA*nB/(nA+nB).
  // Disjoint source supports and matching body/scope are mandatory.
  std::optional<Correction> correction(MeanId,MeanId,const Coefficient&);
  bool add(Form&,const Correction&,const Coefficient& scale);
  bool addCentered(Form&,MeanId,std::span<const Interval>,bool finite=false);
  std::optional<Signature> finish(const Form&);
  // The complete centered-square region must have uniform weight and exactly
  // the same sample support as its independently certified mean. No Fold
  // element is opened, and no partial centered Fold is treated as a moment.
  std::optional<Signature> centered(MeanId,std::span<const Interval>,bool finite=false);
  // Pointwise exact corrections summed over compact affine translations.
  // No logical index is visited; unknown finiteness defaults to deny.
  bool addFamily(FamilyForm&,const Correction&,Translation,const Coefficient& scale,bool finite=false);
  bool addFixed(FamilyForm&,const Correction&,const Coefficient& scale);
  bool addCentered(FamilyForm&,MeanId,std::span<const Interval>,bool finite=false);
  std::optional<FamilySignature> finishFamily(const FamilyForm&);
  // true is an exact whole certificate; false is a signature mismatch (not
  // kernel inequivalence); empty is Unknown/cap. A shared residue lattice is
  // chosen from BOTH signatures so stride1 and interleaved stride2 can match.
  std::optional<bool> equalFamilies(const FamilySignature&,const FamilySignature&);
  std::optional<Coefficient> coefficient(std::string_view);
  std::size_t work() const;
  bool exhausted() const;
 private:
  struct Mean {
    std::size_t group,context;
    std::uint64_t count;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> support;
  };
  bool charge(std::size_t=1);
  bool identity(std::uint64_t,std::uint64_t,const Coefficient&);
  bool addSquare(Form&,MeanId,const Coefficient&);
  std::optional<std::size_t> familyTemplate(const Mean&);
  bool appendFamily(FamilyForm&,MeanId,Translation,const Coefficient&);
  bool appendFamilyTerm(FamilyForm&,std::size_t,FamilyTerm);
  bool constant(const Coefficient&);
  std::optional<Coefficient> apply(polynomial_certificate::Operation,const Coefficient&,const Coefficient&);
  Limits limits_;std::size_t local_work_=0,support_entries_=0;bool exhausted_=false;
  polynomial_certificate::Builder coefficients_;
  bounded_linear_reduction::Builder intervals_;
  std::vector<Mean> means_;
  using MeanKey=std::tuple<std::size_t,std::size_t,std::uint64_t,
      std::vector<std::pair<std::uint64_t,std::uint64_t>>>;
  std::map<MeanKey,MeanId> mean_index_;
  std::map<MeanKey,std::size_t> family_templates_;
  std::map<std::pair<std::uint64_t,std::uint64_t>,bool> identities_;
  std::uint64_t owner_=0;
};
} // namespace sentryward::equivalence::moment_certificate
