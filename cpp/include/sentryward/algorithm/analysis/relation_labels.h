#pragma once
#include <array>
#include <bit>
#include <compare>
#include <cstddef>
#include <cstdint>

namespace sentryward::algorithm {
// The source labels of one relation batch (FamilyRelationQuery::checkOrderBatch): one bit per
// source of the batch, kRelationLabelBits in all, so that one traversal answers that many source
// families. Plain bit-set algebra over kRelationLabelWords words; implicit from one 64-bit word
// (its low bits), so single-source callers keep writing 1. Trivially default-constructible (an
// uninitialized value is only ever assigned before it is read); `RelationLabels{}` is empty.
inline constexpr std::size_t kRelationLabelWords=4;
inline constexpr std::size_t kRelationLabelBits=64*kRelationLabelWords;
struct RelationLabels {
  std::array<std::uint64_t,kRelationLabelWords> words;
  RelationLabels()=default;
  constexpr RelationLabels(std::uint64_t low):words{low} {}
  static constexpr RelationLabels Bit(std::size_t i) {
    RelationLabels r{};r.words[i/64]=std::uint64_t(1)<<(i%64);return r;
  }
  static constexpr RelationLabels All() {
    RelationLabels r{};for(auto& w:r.words)w=~std::uint64_t(0);return r;
  }
  constexpr explicit operator bool() const {for(auto w:words)if(w)return true;return false;}
  constexpr bool operator!() const {return !bool(*this);}
  constexpr bool test(std::size_t i) const {return (words[i/64]>>(i%64))&1;}
  constexpr int count() const {int n=0;for(auto w:words)n+=std::popcount(w);return n;}
  constexpr RelationLabels& operator&=(const RelationLabels& b) {
    for(std::size_t i=0;i<kRelationLabelWords;++i)words[i]&=b.words[i];return *this;
  }
  constexpr RelationLabels& operator|=(const RelationLabels& b) {
    for(std::size_t i=0;i<kRelationLabelWords;++i)words[i]|=b.words[i];return *this;
  }
  friend constexpr RelationLabels operator&(RelationLabels a,const RelationLabels& b){return a&=b;}
  friend constexpr RelationLabels operator|(RelationLabels a,const RelationLabels& b){return a|=b;}
  friend constexpr RelationLabels operator~(RelationLabels a){for(auto& w:a.words)w=~w;return a;}
  friend constexpr bool operator==(const RelationLabels&,const RelationLabels&)=default;
  friend constexpr auto operator<=>(const RelationLabels&,const RelationLabels&)=default;
};
}  // namespace sentryward::algorithm
