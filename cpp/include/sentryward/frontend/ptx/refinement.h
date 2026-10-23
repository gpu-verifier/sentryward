#pragma once
#include "sentryward/algorithm/io/source_json.h"
#include <map>
#include <nlohmann/json.hpp>
#include <vector>
namespace sentryward::frontend::ptx {
using namespace sentryward::algorithm;
struct MmaPlan {
  std::vector<AsyncEffect> effects;
  std::vector<std::uint32_t> writes, accumulators;
  std::vector<std::pair<std::uint32_t, MilestoneRef>> links;
  std::vector<std::pair<std::uint32_t, MilestoneRef>> operand_links;
  std::vector<std::pair<std::uint32_t,std::uint32_t>> split_accumulators;
  std::vector<std::uint32_t> a_tmem_reads;
  unsigned cta_group = 0;
  std::map<std::uint32_t, std::pair<std::string, std::uint64_t>> operand_addresses;
  std::vector<std::pair<std::uint32_t, MilestoneRef>> copy_links;
  // The subset of copy_links accepted because the operand read lies inside the
  // copy's written TMEM bytes (see CopyFeedsOperandBase).
  std::vector<std::pair<std::uint32_t, MilestoneRef>> contained_copy_links;
};
struct TcgenCopyPlan {
  std::vector<AsyncEffect> effects;
  unsigned cta_group = 0;
  std::string allocation;
  std::uint64_t address = 0;
  bool any_operand_address = false;
  std::vector<std::pair<std::uint32_t, MilestoneRef>> operand_links;
};
// PTX pairs a non-4x256b tcgen05.cp with an MMA reading the "same address".
// An operand wider than one copy is filled by several copies at and after its
// base, so accept copies in the operand's lanes starting at or after its base
// column. A copy starting before the base also feeds the operand when the
// operand's read bytes lie inside the copy's written bytes (user decision
// 2026-10-21: "same address" = the MMA reads what the copy wrote; e.g. CUTLASS
// copies a two-tile scale-factor atom and each N tile reads its half).
[[nodiscard]] inline bool CopyFeedsOperandBase(std::uint64_t copy,
                                               std::uint64_t base) {
  return copy >> 16 == base >> 16 && (copy & 0xffff) >= (base & 0xffff);
}
struct TcgenRecord {
  std::uint32_t id;
  int group;
  std::string operation;
};
struct TcgenHistory {
  std::vector<TcgenRecord> records;
  std::map<int, std::vector<std::uint32_t>> work, commits;
};
[[nodiscard]] nlohmann::json Refine(const SourceProgram &,
                                    const std::vector<RequiredOrder> &,
                                    const std::map<std::uint32_t, MmaPlan> &,
                                    const std::vector<TcgenHistory> &,
                                    const std::map<std::uint32_t,TcgenCopyPlan> &copies = {});
} // namespace sentryward::frontend::ptx
