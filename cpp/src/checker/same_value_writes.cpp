#include "sentryward/algorithm/checker/same_value_writes.h"

#include <algorithm>
#include <stdexcept>
#include <string_view>

#include "sentryward/algorithm/checker/unit_quotient.h"

namespace sentryward::algorithm {
namespace {

// PTX ISA 8.6 (Proxies): only .async.bulk operations and tensor-map accesses go through a
// non-generic proxy, so st, stmatrix and non-bulk cp.async write through the generic proxy.
// st.async (remote completion) and st.bulk are left out, and st.mmio, whose writes may have
// device side effects. Atomics carry an atomic scope and never reach this test.
bool GenericPlainWrite(std::string_view opcode) {
  // Splits off the next dot-separated token; empty once none is left.
  const auto next = [&opcode] {
    const auto dot = opcode.find('.');
    const auto token = opcode.substr(0, dot);
    opcode.remove_prefix(dot == std::string_view::npos ? opcode.size() : dot + 1);
    return token;
  };
  const auto base = next();
  if (base == "st") {
    while (!opcode.empty()) {
      const auto token = next();
      if (token == "async" || token == "bulk" || token == "mmio") return false;
    }
    return true;
  }
  if (base == "stmatrix") return true;
  if (base != "cp" || next() != "async") return false;
  const auto kind = next();
  return kind == "ca" || kind == "cg";
}

}  // namespace

bool SameValueWrites::eligible(const MemoryAccess& access) const {
  if (access.mode != AccessMode::kWrite || access.atomic_scope || !access.bytes || *access.bytes == 0 ||
      !std::holds_alternative<AddressId>(access.address) ||
      access.instruction.value() >= source_->instructions.size())
    return false;
  return GenericPlainWrite(source_->instructions.opcode(access.instruction.value()).view());
}

bool SameValueWrites::eligible(const MemoryAccessFamily& family) const {
  // Members share the template's mode, scope, width and endpoint kind; of the fields this test
  // reads, only the instruction varies.
  if (!family.count || !eligible(family.first)) return false;
  const auto step = family.delta[MemoryAccessFamily::kInstructionField];
  const auto first = std::int64_t(family.first.instruction.value());
  for (std::size_t k = 1; step && k < family.count; ++k) {
    const auto instruction = first + step * std::int64_t(k);
    if (instruction < 0 || std::uint64_t(instruction) >= source_->instructions.size() ||
        !GenericPlainWrite(source_->instructions.opcode(std::size_t(instruction)).view()))
      return false;
  }
  return true;
}

bool SameValueWrites::sameBytes(const MemoryAccess& first, const MemoryAccess& second,
                                std::uint64_t overlap_bytes) const {
  // The overlap is the intersection of both byte ranges; its size equals each width only when the
  // two ranges coincide.
  return first.bytes && second.bytes && *first.bytes == *second.bytes && overlap_bytes == *first.bytes &&
         eligible(first) && eligible(second);
}

bool SameValueWrites::pair(const MemoryAccess& first, const MemoryAccess& second,
                           const OverlapEvidence& overlap) const {
  const auto& region = overlap.constant_region;
  return first.event != second.event && region && sameBytes(first, second, region->bytes);
}

bool SameValueWrites::count(const SourceLocation& first, ActorId first_actor, const SourceLocation& second,
                            ActorId second_actor, MemoryRaceResult& result) {
  if (!ctas_) throw std::logic_error("same-value warning samples need the actors' CTAs");
  auto a = std::tuple(first.file.view(), first.line, first.column, (*ctas_)(first_actor));
  auto b = std::tuple(second.file.view(), second.line, second.column, (*ctas_)(second_actor));
  auto actor = first_actor;
  if (b < a) {
    std::swap(a, b);
    actor = second_actor;
  }
  const auto& [a_file, a_line, a_column, a_cta] = a;
  const auto& [b_file, b_line, b_column, b_cta] = b;
  (void)b_cta;
  if (!sites_.emplace(std::string(a_file), a_line, a_column, std::string(b_file), b_line, b_column, a_cta).second)
    return false;
  const std::uint64_t weight = weights_ ? (*weights_)(actor) : 1;
  if (weight > SIZE_MAX - result.same_value_write_count) throw std::overflow_error("warning count overflow");
  result.same_value_write_count += weight;
  return true;
}

bool SameValueWrites::keeps(const MemoryRaceResult& result) const {
  // The first sample is also the witness when the obligations are not decided.
  return result.same_value_write_samples.empty() || result.same_value_write_samples.size() < max_samples_;
}

void SameValueWrites::sample(const MemoryAccess& first, const MemoryAccess& second,
                             const OverlapEvidence& overlap, MemoryRaceResult& result) {
  if (count(first.location, first.actor, second.location, second.actor, result) && keeps(result))
    result.same_value_write_samples.push_back({first, second, overlap});
}

bool SameValueWrites::sampleRelation(const MemoryAccessFamily& first, const MemoryAccessFamily& second,
                                     const MemberRelation& relation,
                                     const std::function<OverlapEvidence(std::size_t, std::size_t)>& evidence,
                                     MemoryRaceResult& result) {
  if (!ctas_) throw std::logic_error("same-value warning samples need the actors' CTAs");
  if (!ctas_->contiguous()) return false;
  const auto witness = relation.witness();
  if (!witness) return true;
  const auto visit = [&](std::size_t x, std::size_t y) {
    if (count(first.first.location, first.actor(x), second.first.location, second.actor(y), result) &&
        keeps(result))
      result.same_value_write_samples.push_back({first.at(x), second.at(y), evidence(x, y)});
  };
  visit(witness->first, witness->second);
  const auto first_cta = [&](std::size_t x) { return (*ctas_)(first.actor(x)); };
  const auto second_cta = [&](std::size_t y) { return (*ctas_)(second.actor(y)); };
  if (relation.diagonal) {
    // Pairs (x, x + shift): one per range where both CTAs are constant.
    const auto partner = [&](std::size_t x) { return std::size_t(std::int64_t(x) + relation.shift); };
    const auto partner_cta = [&](std::size_t x) { return second_cta(partner(x)); };
    relation.left.forEachRun([&](std::size_t begin, std::size_t end) {
      ForEachCtaRange(begin, end, first_cta, [&](std::size_t lo, std::size_t hi) {
        ForEachCtaRange(lo, hi, partner_cta, [&](std::size_t x, std::size_t) { visit(x, partner(x)); });
      });
    });
    return true;
  }
  // One member per CTA of a side, by CTA.
  const auto ctas = [&](const MemberSet& members, const auto& cta_of) {
    std::vector<std::pair<std::uint32_t, std::size_t>> found;
    members.forEachRun([&](std::size_t begin, std::size_t end) {
      ForEachCtaRange(begin, end, cta_of, [&](std::size_t lo, std::size_t) { found.emplace_back(cta_of(lo), lo); });
    });
    std::sort(found.begin(), found.end());
    found.erase(std::unique(found.begin(), found.end(),
                            [](const auto& a, const auto& b) { return a.first == b.first; }),
                found.end());
    return found;
  };
  const auto left = ctas(relation.left, first_cta), right = ctas(relation.right, second_cta);
  // A count names the CTA of its lesser (site, CTA): one side's CTA when the sites differ, the
  // lesser CTA when they are equal. Each CTA of one side paired with the greatest of the other
  // yields every such CTA (both sides are nonempty: the relation has a witness).
  for (const auto& group : left) visit(group.second, right.back().second);
  for (const auto& group : right) visit(left.back().second, group.second);
  return true;
}

}  // namespace sentryward::algorithm
