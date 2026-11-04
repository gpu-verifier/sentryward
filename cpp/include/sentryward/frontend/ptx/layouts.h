#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "sentryward/frontend/ptx/module.h"
#include <nlohmann/json.hpp>

namespace sentryward::frontend::ptx {
class LayoutError : public std::runtime_error {
public:
  std::string code;
  nlohmann::json details = nlohmann::json::object();
  explicit LayoutError(std::string message, std::string reason = "unsupported_memory_layout", nlohmann::json evidence = nlohmann::json::object())
      : std::runtime_error(std::move(message)), code(std::move(reason)), details(std::move(evidence)) {}
};
struct ByteInterval {
  std::uint64_t offset{}, size{};
  bool operator==(const ByteInterval &) const = default;
};
struct ExactTmemFootprint {
  std::string resource;
  std::vector<ByteInterval> reads, writes;
};
struct MmaDescriptor {
  std::string kind;
  std::uint32_t m{}, n{}, d_element_bits{}, a_element_bits{}, b_element_bits{};
  std::string a_major, b_major;
  std::uint32_t k{};
  bool sparse{};
};
struct SharedMemoryDescriptor {
  std::uint32_t start_address{}, leading_byte_offset{}, stride_byte_offset{},
      base_offset{}, layout{};
  bool leading_absolute{};
};
[[nodiscard]] std::pair<std::uint32_t, std::uint32_t>
DecodeTmemAddress(std::uint64_t address);
[[nodiscard]] std::uint64_t EncodeCellByteOffset(std::uint32_t lane,
                                                 std::uint32_t column);
[[nodiscard]] std::vector<ByteInterval>
MergeIntervals(std::vector<ByteInterval> intervals);
[[nodiscard]] ExactTmemFootprint
DecodeLdstFootprint(const Instruction &, std::uint64_t taddr,
                    std::string resource,
                    std::optional<std::uint64_t> half_split_offset = {},
                    std::optional<std::uint32_t> allocated_columns = {},
                    std::optional<std::uint32_t> warp_in_warpgroup = {});
// Register order is significant; intervals are never merged here.
[[nodiscard]] std::vector<ByteInterval>
DecodeLdstLaneRegisters(const Instruction &, std::uint64_t taddr,
                        std::uint32_t lane_id, std::uint32_t warp_in_warpgroup,
                        std::optional<std::uint32_t> allocated_columns = {},
                        std::optional<std::uint64_t> half_split_offset = {});
[[nodiscard]] MmaDescriptor DecodeMmaDescriptor(std::string_view kind,
                                                std::uint64_t value);
// PTX 9.2 tcgen05.mma.ws shapes: dense M in {32,64,128}, N in {64,128,256}.
[[nodiscard]] constexpr bool WsMmaShape(std::uint32_t m, std::uint32_t n) {
  return (m == 32 || m == 64 || m == 128) && (n == 64 || n == 128 || n == 256);
}
// The tcgen05.mma data-path organization (PTX ISA, "Data Path Layout Organization",
// layouts A-G). Each CTA holds `rows` = M/cta_group rows of D. Its TMEM lanes form
// `slices` lane groups: lane group s holds columns [s*N/slices, (s+1)*N/slices) of
// every row. Layouts C and F use 16 lanes of each 32-lane quarter. The ISA does not
// say where a lane group reads a Tensor Memory A operand; as in CUTLASS's duplicated
// TMEM-A allocation (UMMA::TmemAllocMode::Duplicated), lane group s reads every A row
// from its own lanes, so the operand holds one copy of A per lane group.
struct MmaDataPath {
  std::uint32_t rows{}, slices{1};
  bool half_quarters{};
  // The lane of `row` in lane group `slice`, relative to the lane alignment.
  [[nodiscard]] std::uint32_t Lane(std::uint32_t row, std::uint32_t slice) const {
    return half_quarters ? 32 * (row / 16) + row % 16 : row + rows * slice;
  }
  // Whether `lane` is an ISA Tensor Memory datapath lane alignment: 0, or 16 for layouts C and F.
  [[nodiscard]] bool LaneAlignment(std::uint32_t lane) const {
    return lane == 0 || (half_quarters && lane == 16);
  }
};
// nullopt for an M, .cta_group and .ws combination outside the ISA's layout table.
[[nodiscard]] std::optional<MmaDataPath> DecodeMmaDataPath(std::uint32_t m, std::uint32_t cta_group,
                                                           bool ws, bool sparse);
[[nodiscard]] MmaDescriptor DecodeBlockScaledDescriptor(std::uint64_t value, std::string_view kind);
[[nodiscard]] std::vector<ByteInterval> DecodeScaleFootprint(
    std::uint64_t taddr, unsigned selector, unsigned rows, unsigned vector_bytes,
    bool operand_b, std::optional<std::uint32_t> columns = {});
[[nodiscard]] std::vector<ByteInterval> DecodeSparseMetadataFootprint(
    std::uint64_t idesc, std::uint64_t taddr, unsigned group, std::string_view kind,
    std::optional<std::uint32_t> columns = {});
[[nodiscard]] SharedMemoryDescriptor
DecodeSharedMemoryDescriptor(std::uint64_t value);
[[nodiscard]] std::vector<ByteInterval>
DecodeSharedMatrixFootprint(std::uint64_t descriptor, std::uint32_t mn,
                            std::uint32_t k, std::uint32_t element_bits,
                            std::string_view major);
// Entries are in logical (MN, K) row-major order, independent of storage major.
// One physical contiguous run, with an affine logical payload coordinate.
// Members are elements, not a materialized list of their addresses.
struct MatrixByteMapping {
  std::uint64_t physical_offset, payload_offset;
  std::uint64_t element_bytes, count, payload_stride;
};
void ForEachSharedMatrixMapping(std::uint64_t descriptor, std::uint32_t mn,
    std::uint32_t k, std::uint32_t element_bits, std::string_view major,
    const std::function<void(const MatrixByteMapping&)>& consume);
[[nodiscard]] std::vector<ByteInterval>
DecodeSharedMatrixElements(std::uint64_t descriptor, std::uint32_t mn,
                           std::uint32_t k, std::uint32_t element_bits,
                           std::string_view major);
// Lanes (rows) of a CTA's Tensor Memory (PTX ISA, Tensor Memory Addressing).
inline constexpr std::uint32_t kTmemLanes = 128;
// A tcgen05.cp data-movement shape (PTX ISA, tcgen05 "Data Movement Shape" and
// tcgen05.cp): `rows` source rows of `row_bits` bits each. A .multicast qualifier
// replicates the rows into the lane quarters (warps) of the TMEM lanes.
struct CopyShape {
  enum class Multicast { kNone, kWarpx4, kWarpx2_02_13, kWarpx2_01_23 };
  std::uint32_t rows{}, row_bits{};
  Multicast multicast = Multicast::kNone;
  std::string name; // the shape and multicast qualifiers, e.g. "32x128b.warpx4"
  [[nodiscard]] std::uint64_t SourceBytes() const { return std::uint64_t{rows} * row_bits / 8; }
  // Whether every one of the 128 TMEM lanes receives a known source row: the 128-row
  // shapes and the multicast shapes (a .4x256b copy has no lane mapping here).
  [[nodiscard]] bool LaneMapped() const { return rows == kTmemLanes || multicast != Multicast::kNone; }
  // Bytes written to TMEM: all lanes when LaneMapped, else the source bytes.
  [[nodiscard]] std::uint64_t TensorBytes() const {
    return LaneMapped() ? std::uint64_t{kTmemLanes} * row_bits / 8 : SourceBytes();
  }
  // The source row that TMEM lane `lane` (0-127, from the copy's base lane) receives; LaneMapped only.
  [[nodiscard]] std::uint32_t SourceRow(std::uint32_t lane) const;
};
// The shape of a tcgen05.cp's modifiers: exactly one shape with the multicast qualifier
// that the ISA requires for it (.64x128b: .warpx2::02_13 or .warpx2::01_23; .32x128b:
// .warpx4; the other shapes none). nullopt otherwise.
[[nodiscard]] std::optional<CopyShape> DecodeCopyShape(const std::vector<std::string> &modifiers);
[[nodiscard]] std::vector<std::pair<ByteInterval, ByteInterval>>
DecodeCopyFragments(std::uint64_t descriptor, std::uint64_t taddr,
                    const CopyShape &shape,
                    std::optional<std::uint32_t> allocated_columns = {});
[[nodiscard]] ExactTmemFootprint DecodeMmaDFootprint(
    const Instruction &, std::uint64_t d_taddr, std::uint64_t idesc,
    bool enable_input_d, std::span<const std::uint32_t> disable_output_lanes,
    std::string resource, std::optional<std::uint32_t> allocated_columns = {},
    std::optional<std::uint32_t> cta_group_part = {});
[[nodiscard]] ExactTmemFootprint
DecodeMmaATmemFootprint(const Instruction &, std::uint64_t a_taddr,
                        std::uint64_t idesc, std::string resource,
                        std::optional<std::uint32_t> allocated_columns = {},
                        std::optional<std::uint32_t> cta_group_part = {},
                        bool allow_layout_f = false);
[[nodiscard]] std::uint64_t SwizzleByteAddress(std::uint64_t address,
                                               unsigned bits);

struct TmaTransferSegment {
  std::optional<std::uint64_t> global_offset;
  std::uint64_t shared_offset{}, bytes{}, logical_offset{};
  std::uint32_t target_rank{};
  bool abstract_global{};
  // The tile::gather4 row (0..3) the segment belongs to; 0 for every other mode.
  std::uint32_t gather_row{};
};
struct TmaCompletion {
  std::uint32_t rank{};
  std::uint64_t bytes{};
};
struct TmaTransferMapping {
  std::vector<TmaTransferSegment> segments;
  std::uint64_t transfer_bytes{};
  std::vector<TmaCompletion> completions;
  bool store{};
};
struct TmaTransferOptions {
  // Unknown origins affect global placement, not shared layout/completion bytes.
  std::vector<bool> symbolic_coordinates;
  // tile::gather4: the four row coordinates (origins are {column, 0}). The
  // destination is the dense {box_shape[0], 4} box; row k reads tensor row k.
  std::optional<std::array<std::int64_t, 4>> gather_rows;
  std::array<bool, 4> symbolic_rows{};
  std::uint64_t shared_base{};
  std::uint32_t destination_rank{}, cluster_size{1}, cta_group{1};
  std::optional<std::uint32_t> completion_rank, multicast_mask;
  bool store{};
  std::optional<std::uint64_t> transfer_bytes;
  std::uint64_t max_elements{1 << 20};
};
// Descriptor JSON is small fixed-input metadata. The transfer itself is typed.
[[nodiscard]] TmaTransferMapping
DecodeTmaTransfer(const nlohmann::json &descriptor,
                  std::span<const std::int64_t> origins,
                  const TmaTransferOptions &options);
// The caller proves all 32 lanes rendezvous and resolves the 8*count CTA-local
// row addresses once per dynamic warp occurrence, then reuses them for all
// lanes.
[[nodiscard]] std::vector<ByteInterval>
MatrixLaneFootprint(unsigned count, bool transpose, unsigned lane,
                    std::span<const std::uint64_t> rows);
struct MatrixShape {
  unsigned count{};
  bool transpose{};
  std::string address;
};
[[nodiscard]] MatrixShape DecodeMatrixShape(const Module &, const Instruction &,
                                            bool store);

// Resolves static lexical scopes once. Repeated runtime register uses reuse
// IDs.
class RegisterTable {
public:
  RegisterTable(const Module &, const Function &);
  [[nodiscard]] std::string Resolve(std::span<const std::uint32_t> scope,
                                    std::string_view name) const;
  [[nodiscard]] std::uint32_t Width(std::string_view resolved_name) const;
  [[nodiscard]] std::optional<std::uint32_t> DeclaredWidth(std::string_view resolved_name) const {
    const auto it = widths_.find(std::string(resolved_name));
    if (it == widths_.end()) return {};
    return it->second;
  }

private:
  std::unordered_map<std::string, std::string> declarations_;
  std::unordered_map<std::string, std::uint32_t> widths_;
};
} // namespace sentryward::frontend::ptx
