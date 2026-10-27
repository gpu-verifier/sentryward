#include "sentryward/frontend/ptx/layouts.h"

#include "sentryward/frontend/ptx/operands.h"
#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>
#include <tuple>

namespace sentryward::frontend::ptx {
namespace {
[[noreturn]] void Fail(std::string message) {
  throw LayoutError(std::move(message));
}
bool Has(const Instruction &instruction, std::string_view modifier) {
  return std::find(instruction.modifiers.begin(), instruction.modifiers.end(),
                   modifier) != instruction.modifiers.end();
}
struct LdstShape {
  std::string_view shape;
  std::uint32_t repeat{};
};
LdstShape Shape(const Instruction &instruction) {
  LdstShape result;
  unsigned shapes = 0, repeats = 0;
  for (const auto &modifier : instruction.modifiers) {
    if (modifier == "16x64b" || modifier == "16x128b" ||
        modifier == "16x256b" || modifier == "16x32bx2" ||
        modifier == "32x32b") {
      result.shape = modifier;
      ++shapes;
    }
    if (modifier == "x1" || modifier == "x2" || modifier == "x4" ||
        modifier == "x8" || modifier == "x16" || modifier == "x32" ||
        modifier == "x64" || modifier == "x128") {
      result.repeat =
          static_cast<std::uint32_t>(std::stoul(modifier.substr(1)));
      ++repeats;
    }
  }
  if (shapes != 1 || repeats != 1)
    Fail("tcgen05 ld/st requires exactly one shape and repeat factor");
  return result;
}
void Rectangle(std::vector<ByteInterval> &result, std::uint32_t lane,
               std::uint32_t column, std::uint32_t lanes, std::uint32_t columns,
               std::uint32_t limit) {
  if (!lanes || !columns || lane >= 128 || lanes > 128 - lane)
    Fail("TMEM rectangle exceeds its lane domain");
  if (!limit || limit > 512 || column > limit || columns > limit - column)
    Fail("TMEM rectangle exceeds its allocation columns");
  result.reserve(result.size() + lanes);
  for (auto row = lane; row < lane + lanes; ++row)
    result.push_back(
        {EncodeCellByteOffset(row, column), std::uint64_t{columns} * 4});
}
MmaDescriptor StandardMma(const Instruction &instruction, std::uint64_t idesc,
                          std::optional<std::uint32_t> part, bool output = false) {
  if (instruction.opcode != "tcgen05" || !Has(instruction, "mma"))
    Fail("instruction is not tcgen05.mma");
  if (Has(instruction, "cta_group::2") && (!part || *part > 1))
    Fail("cta_group::2 MMA requires local group part 0 or 1");
  const bool ws = Has(instruction, "ws");
  if (ws && Has(instruction, "cta_group::2"))
    Fail(".ws MMA supports only cta_group::1");
  std::string_view kind;
  unsigned kinds = 0;
  for (const auto &modifier : instruction.modifiers) {
    if (modifier == "block_scale" || (modifier == "sp" && (!output || ws)) ||
        modifier == "ashift" || modifier.starts_with("collector::"))
      Fail("MMA variant requires a different exact layout");
    if (modifier.starts_with("kind::")) {
      kind = std::string_view(modifier).substr(6);
      ++kinds;
    }
  }
  if (kinds != 1)
    Fail("tcgen05.mma requires exactly one kind qualifier");
  auto descriptor = DecodeMmaDescriptor(kind, idesc);
  if (descriptor.sparse && (!output || ws))
    Fail("sparse MMA requires additional metadata layout");
  if (ws && !WsMmaShape(descriptor.m, descriptor.n))
    Fail(".ws MMA shape is outside the PTX shape list");
  return descriptor;
}
std::uint64_t Integer(const nlohmann::json &value, std::string_view name,
                      std::uint64_t minimum = 0) {
  if (!value.is_number_integer() ||
      (value.is_number_integer() && !value.is_number_unsigned() &&
       value.get<std::int64_t>() < 0))
    Fail(std::string(name) + " must be a nonnegative integer");
  const auto number = value.get<std::uint64_t>();
  if (number < minimum)
    Fail(std::string(name) + " is below its minimum");
  return number;
}
std::vector<std::uint64_t> Vector(const nlohmann::json &descriptor,
                                  std::string_view key, std::uint64_t minimum = 1) {
  auto found = descriptor.find(std::string(key));
  if (found == descriptor.end() || !found->is_array())
    Fail(std::string(key) + " must be an explicit integer vector");
  std::vector<std::uint64_t> values;
  values.reserve(found->size());
  for (const auto &value : *found)
    values.push_back(Integer(value, key, minimum));
  return values;
}
void AppendSegment(std::vector<TmaTransferSegment> &segments,
                   TmaTransferSegment segment) {
  if (!segments.empty()) {
    auto &previous = segments.back();
    const bool global_continues =
        (!previous.global_offset && !segment.global_offset) ||
        (previous.global_offset && segment.global_offset &&
         *previous.global_offset <=
             std::numeric_limits<std::uint64_t>::max() - previous.bytes &&
         *previous.global_offset + previous.bytes == *segment.global_offset);
    // Gather rows never merge: each abstract row carries its own coordinate.
    if (previous.abstract_global == segment.abstract_global &&
        previous.gather_row == segment.gather_row &&
        previous.target_rank == segment.target_rank &&
        previous.logical_offset + previous.bytes == segment.logical_offset &&
        previous.shared_offset + previous.bytes == segment.shared_offset &&
        global_continues) {
      previous.bytes += segment.bytes;
      return;
    }
  }
  segments.push_back(std::move(segment));
}
std::string ScopedKey(std::span<const std::uint32_t> scope,
                      std::string_view name) {
  std::string key;
  for (const auto part : scope) {
    key += std::to_string(part);
    key += '.';
  }
  key += ':';
  key += name;
  return key;
}
std::string Trim(std::string_view value) {
  const auto begin = value.find_first_not_of(" \t\n\r");
  if (begin == value.npos)
    return {};
  return std::string(
      value.substr(begin, value.find_last_not_of(" \t\n\r") - begin + 1));
}
} // namespace

std::pair<std::uint32_t, std::uint32_t>
DecodeTmemAddress(std::uint64_t address) {
  if (address > 0xffffffffULL || (address & ~std::uint64_t{0x007f01ff}))
    Fail("TMEM address has non-address bits set");
  return {static_cast<std::uint32_t>((address >> 16) & 0x7f),
          static_cast<std::uint32_t>(address & 0x1ff)};
}
std::uint64_t EncodeCellByteOffset(std::uint32_t lane, std::uint32_t column) {
  if (lane >= 128 || column >= 512)
    Fail("TMEM cell outside its lane/column domain");
  return 4ULL * (lane * 512 + column);
}
std::uint64_t SwizzleByteAddress(std::uint64_t address, unsigned bits) {
  if (bits > 3)
    Fail("unsupported shared swizzle bit count");
  return address ^ (((address >> 7) & ((1U << bits) - 1)) << 4);
}
std::vector<ByteInterval> MergeIntervals(std::vector<ByteInterval> intervals) {
  std::sort(intervals.begin(), intervals.end(),
            [](const auto &a, const auto &b) {
              return std::tie(a.offset, a.size) < std::tie(b.offset, b.size);
            });
  std::size_t output = 0;
  for (const auto interval : intervals) {
    if (!interval.size ||
        interval.offset >
            std::numeric_limits<std::uint64_t>::max() - interval.size)
      Fail("invalid byte interval");
    if (output && interval.offset <= intervals[output - 1].offset +
                                         intervals[output - 1].size) {
      auto &previous = intervals[output - 1];
      previous.size = std::max(previous.offset + previous.size,
                               interval.offset + interval.size) -
                      previous.offset;
    } else {
      intervals[output++] = interval;
    }
  }
  intervals.resize(output);
  return intervals;
}
ExactTmemFootprint DecodeLdstFootprint(const Instruction &instruction,
                                       std::uint64_t taddr,
                                       std::string resource,
                                       std::optional<std::uint64_t> split,
                                       std::optional<std::uint32_t> columns,
                                       std::optional<std::uint32_t> warp) {
  if (instruction.opcode != "tcgen05" || instruction.modifiers.empty() ||
      (instruction.modifiers[0] != "ld" && instruction.modifiers[0] != "st"))
    Fail("instruction is not tcgen05 ld/st");
  if (resource.empty())
    Fail("TMEM resource identity cannot be empty");
  const auto shape = Shape(instruction);
  const auto [lane, column] = DecodeTmemAddress(taddr);
  std::array<std::array<std::uint32_t, 4>, 2> rectangles{};
  unsigned count = 1;
  if (shape.shape == "32x32b")
    rectangles[0] = {lane, column, 32, shape.repeat};
  else if (shape.shape == "16x64b")
    rectangles[0] = {lane, column, 16, 2 * shape.repeat};
  else if (shape.shape == "16x128b")
    rectangles[0] = {lane, column, 16, 4 * shape.repeat};
  else if (shape.shape == "16x256b")
    rectangles[0] = {lane, column, 16, 8 * shape.repeat};
  else {
    if (!split || *split > std::numeric_limits<std::uint32_t>::max() - taddr)
      Fail("16x32bx2 requires a valid nonnegative half-split offset");
    const auto [second_lane, second_column] = DecodeTmemAddress(taddr + *split);
    rectangles[0] = {lane, column, 16, shape.repeat};
    rectangles[1] = {second_lane, second_column, 16, shape.repeat};
    count = 2;
  }
  std::vector<ByteInterval> intervals;
  for (unsigned index = 0; index < count; ++index) {
    auto rectangle = rectangles[index];
    if (warp) {
      if (*warp >= 4)
        Fail("warp index within warpgroup is outside [0,4)");
      const auto local = rectangle[0] % 32;
      if (local + rectangle[2] > 32)
        Fail("TMEM ld/st crosses the issuing warp chunk");
      rectangle[0] = 32 * *warp + local;
    }
    Rectangle(intervals, rectangle[0], rectangle[1], rectangle[2], rectangle[3],
              columns.value_or(512));
  }
  ExactTmemFootprint result{std::move(resource), {}, {}};
  (instruction.modifiers[0] == "ld" ? result.reads : result.writes) =
      MergeIntervals(std::move(intervals));
  return result;
}
std::vector<ByteInterval>
DecodeLdstLaneRegisters(const Instruction &instruction, std::uint64_t taddr,
                        std::uint32_t lane_id, std::uint32_t warp,
                        std::optional<std::uint32_t> columns,
                        std::optional<std::uint64_t> split) {
  if (instruction.opcode != "tcgen05" || instruction.modifiers.empty() ||
      (instruction.modifiers[0] != "ld" && instruction.modifiers[0] != "st"))
    Fail("instruction is not tcgen05 ld/st");
  if (lane_id >= 32 || warp >= 4)
    Fail("thread lane or warp index is outside its domain");
  for (const auto &modifier : instruction.modifiers)
    if (modifier == "red")
      Fail("packed/reduced TMEM registers need a separate map");
  const bool packed = std::find(instruction.modifiers.begin(), instruction.modifiers.end(), "pack::16b") != instruction.modifiers.end() ||
      std::find(instruction.modifiers.begin(), instruction.modifiers.end(), "unpack::16b") != instruction.modifiers.end();
  const auto shape = Shape(instruction);
  if (shape.shape == "16x32bx2") {
    // PTX matrix fragment: lanes 0..15 use taddr; lanes 16..31 use
    // taddr + immHalfSplitoff. Register r advances one column (two
    // columns for pack/unpack). The offset is an encoded address offset,
    // not necessarily just a column offset.
    if (!split || taddr > UINT32_MAX || *split > UINT32_MAX - taddr)
      Fail("16x32bx2 requires a valid nonnegative half-split offset");
    const auto first = DecodeTmemAddress(taddr);
    const auto second = DecodeTmemAddress(taddr + *split);
    const unsigned halves = packed ? 2U : 1U;
    const auto limit = columns.value_or(512);
    const auto width = halves * shape.repeat;
    for (const auto base : {first, second})
      if (base.first % 32 + 16 > 32 || !limit || limit > 512 ||
          base.second > limit || width > limit - base.second)
        Fail("TMEM split lane-register footprint exceeds allocation or warp chunk");
    const auto base = lane_id < 16 ? first : second;
    const auto row = 32 * warp + base.first % 32 + lane_id % 16;
    std::vector<ByteInterval> result;
    result.reserve(halves * shape.repeat);
    for (unsigned reg = 0; reg < shape.repeat; ++reg)
      for (unsigned half = 0; half < halves; ++half)
        result.push_back({EncodeCellByteOffset(row, base.second + halves * reg + half),
                          packed ? 2U : 4U});
    return result;
  }
  const unsigned per_repeat = shape.shape == "32x32b"    ? 1
                              : shape.shape == "16x128b" ? 2
                              : shape.shape == "16x256b" ? 4
                                                         : 0;
  if (!per_repeat || per_repeat * shape.repeat > 128)
    throw LayoutError("TMEM register vector layout is unsupported", "tmem_register_layout_unsupported",
      {{"shape",shape.shape},{"repeat",shape.repeat},{"register_count",per_repeat?nlohmann::json(per_repeat*shape.repeat):nlohmann::json(nullptr)}});
  // Validate dimensions without allocating a collective footprint per lane.
  const auto [base_lane, base_column] = DecodeTmemAddress(taddr);
  const auto rows = shape.shape == "32x32b" ? 32U : 16U;
  const auto width = (packed ? 2U : 1U) * shape.repeat * (shape.shape == "32x32b"    ? 1U
                                     : shape.shape == "16x128b" ? 4U
                                                                : 8U);
  const auto limit = columns.value_or(512);
  if (base_lane % 32 + rows > 32 || !limit || limit > 512 ||
      base_column > limit || width > limit - base_column)
    Fail("TMEM lane-register footprint exceeds allocation or warp chunk");
  std::vector<ByteInterval> result;
  result.reserve(per_repeat * shape.repeat);
  for (unsigned reg = 0; reg < per_repeat * shape.repeat; ++reg) {
    unsigned row, col;
    if (shape.shape == "32x32b") {
      row = lane_id;
      col = reg;
    } else if (shape.shape == "16x128b") {
      row = lane_id / 4 + 8 * (reg % 2);
      col = lane_id % 4 + 4 * (reg / 2);
    } else {
      row = lane_id / 4 + 8 * ((reg % 4) / 2);
      col = 2 * (lane_id % 4) + reg % 2 + 8 * (reg / 4);
    }
    for (unsigned half = 0; half < (packed ? 2U : 1U); ++half)
      result.push_back({EncodeCellByteOffset(32 * warp + base_lane % 32 + row,
                       base_column + (packed ? 2 * col + half : col)),
                       packed ? 2U : 4U});
  }
  return result;
}
MmaDescriptor DecodeMmaDescriptor(std::string_view kind, std::uint64_t value) {
  if (kind != "tf32" && kind != "f16" && kind != "f8f6f4" && kind != "i8")
    Fail("unsupported MMA kind");
  if (value > 0xffffffffULL ||
      (value & ((1ULL << 6) | (1ULL << 23) | (7ULL << 29))))
    Fail("MMA descriptor has reserved bits set or exceeds u32");
  const auto dtype = (value >> 4) & 3;
  unsigned d_bits = 0;
  if (kind == "tf32" && dtype == 1)
    d_bits = 32;
  if ((kind == "f16" || kind == "f8f6f4") && dtype <= 1)
    d_bits = dtype ? 32 : 16;
  if (kind == "i8" && dtype == 2)
    d_bits = 32;
  if (!d_bits)
    Fail("invalid MMA destination type");
  auto input_bits = [&](unsigned code) -> unsigned {
    if (kind == "tf32" && code == 2)
      return 32;
    if (kind == "f16" && code <= 1)
      return 16;
    if (kind == "i8" && code <= 1)
      return 8;
    if (kind == "f8f6f4") {
      if (code <= 1)
        return 8;
      if (code == 3 || code == 4)
        return 6;
      if (code == 5)
        return 4;
    }
    Fail("invalid MMA input type");
  };
  const auto m = static_cast<unsigned>((value >> 24) & 31) << 4;
  const auto n = static_cast<unsigned>((value >> 17) & 63) << 3;
  if (!m || !n)
    Fail("MMA descriptor has a zero dimension");
  return {std::string(kind),
          m,
          n,
          d_bits,
          input_bits((value >> 7) & 7),
          input_bits((value >> 10) & 7),
          value & (1ULL << 15) ? "mn" : "k",
          value & (1ULL << 16) ? "mn" : "k",
          (kind == "tf32"  ? 8U
           : kind == "f16" ? 16U
                           : 32U) *
              (value & 4 ? 2U : 1U),
          bool(value & 4)};
}
MmaDescriptor DecodeBlockScaledDescriptor(std::uint64_t value, std::string_view kind) {
  if (value > UINT32_MAX) Fail("block-scaled descriptor must be u32");
  if (kind == "mxf4" || kind == "mxf4nvf4") {
    if (value & (0xbULL | (1ULL << 6) | (3ULL << 15) | (1ULL << 31)))
      Fail("packed FP4 requires K-major operands and zero reserved bits");
    if (((value >> 7) & 7) != 1 || ((value >> 10) & 7) != 1)
      Fail("packed FP4 operands must use E2M1");
    if (kind == "mxf4") {
      if (!(value & (1ULL << 23))) Fail("MXF4 scale type must be UE8M0");
      if (value & ((1ULL << 4) | (1ULL << 29))) Fail("block32 scale IDs must be 0 or 2");
    } else if (value & ((3ULL << 4) | (3ULL << 29))) Fail("block16 scale IDs must be zero");
    const auto standard = (value & ~((1ULL << 23) | (7ULL << 7) | (7ULL << 10) |
                                    (3ULL << 4) | (3ULL << 29))) |
        (1ULL << 4) | (5ULL << 7) | (5ULL << 10);
    auto result = DecodeMmaDescriptor("f8f6f4", standard);
    result.kind = kind; result.k = (value & 4) ? 128 : 64;
    return result;
  }
  if (kind != "mxf8f6f4") Fail("unsupported block-scaled kind");
  if (value & (8ULL | (1ULL << 6) | (7ULL << 24) | (1ULL << 31)) ||
      ((value & 3) && !(value & 4))) Fail("unsupported block-scaled descriptor layout");
  if (!(value & (1ULL << 23))) Fail("MXF8 scale type must be UE8M0");
  const auto standard = (value & ~((3ULL << 4) | (1ULL << 23) | (3ULL << 29))) | (1ULL << 4);
  auto result = DecodeMmaDescriptor("f8f6f4", standard);
  result.kind = "mxf8f6f4";
  return result;
}
std::vector<ByteInterval> DecodeScaleFootprint(std::uint64_t taddr,
    unsigned selector, unsigned rows, unsigned vector_bytes, bool operand_b,
    std::optional<std::uint32_t> columns) {
  if (taddr > UINT32_MAX || selector >= 4 || (vector_bytes != 1 && vector_bytes != 2 && vector_bytes != 4) ||
      selector % vector_bytes || (rows != 128 && rows != 256 && !(operand_b && rows == 64)))
    Fail("invalid scale footprint operands");
  const auto [lane, column] = DecodeTmemAddress(taddr & 0x3fffffff);
  if (lane) Fail("replicated scale base must select lane zero");
  std::vector<ByteInterval> result, bounds;
  Rectangle(bounds, 0, column, 128, rows / 32, columns.value_or(512));
  for (unsigned row = 0; row < 128; ++row)
    for (unsigned c = 0; c < rows / 32; ++c)
      result.push_back({EncodeCellByteOffset(row, column + c) + selector, vector_bytes});
  return result;
}
std::vector<ByteInterval> DecodeSparseMetadataFootprint(std::uint64_t idesc,
    std::uint64_t taddr, unsigned group, std::string_view kind,
    std::optional<std::uint32_t> columns) {
  if (kind != "f16" && kind != "f8f6f4") Fail("unsupported sparse metadata kind");
  const auto desc = DecodeMmaDescriptor(kind, idesc);
  if (!desc.sparse || (group != 1 && group != 2) || desc.m != 128 * group ||
      (idesc & 3) > (kind == "f16" ? 1U : 0U)) Fail("unsupported sparse metadata layout");
  const auto [lane, column] = DecodeTmemAddress(taddr);
  std::vector<ByteInterval> result;
  Rectangle(result, lane, column + (idesc & 3), 128, kind == "f16" ? 1 : 2, columns.value_or(512));
  return result;
}
SharedMemoryDescriptor DecodeSharedMemoryDescriptor(std::uint64_t value) {
  if (value & ((3ULL << 14) | (3ULL << 30) | (1ULL << 48) | (0xffULL << 53)))
    Fail("shared descriptor has reserved bits set");
  if (((value >> 46) & 3) != 1)
    Fail("shared descriptor is not Blackwell version 1");
  const auto layout = static_cast<unsigned>(value >> 61);
  if (layout != 0 && layout != 1 && layout != 2 && layout != 4 && layout != 6)
    Fail("invalid shared descriptor layout");
  return {static_cast<unsigned>(value & 0x3fff) << 4,
          static_cast<unsigned>((value >> 16) & 0x3fff) << 4,
          static_cast<unsigned>((value >> 32) & 0x3fff) << 4,
          static_cast<unsigned>((value >> 49) & 7),
          layout,
          bool(value & (1ULL << 52))};
}
std::vector<ByteInterval>
DecodeSharedMatrixFootprint(std::uint64_t value, std::uint32_t mn,
                            std::uint32_t k, std::uint32_t element_bits,
                            std::string_view major) {
  const auto d = DecodeSharedMemoryDescriptor(value);
  if (element_bits != 8 && element_bits != 16 && element_bits != 32)
    Fail("packed shared MMA input is not decoded");
  if ((major != "mn" && major != "k") || !mn || !k)
    Fail("invalid shared matrix dimensions or major mode");
  if (d.leading_absolute || d.layout == 1)
    Fail("shared MMA descriptor requires unsupported canonical layout");
  const unsigned bits = d.layout == 0   ? 0
                        : d.layout == 2 ? 3
                        : d.layout == 4 ? 2
                                        : 1;
  const unsigned atom = 1U << bits;
  if (d.base_offset)
    Fail("nonzero swizzle base offset requires a relative-origin mapping");
  const auto normalized = 128 / element_bits;
  if (mn % normalized || k % normalized)
    Fail("matrix does not tile normalized 128-bit elements");
  const auto leading = d.leading_byte_offset / 16,
             stride = d.stride_byte_offset / 16;
  // Physical descriptors have an 18-bit address domain, so a fixed-size bitset
  // avoids tree nodes and repeated sort/deduplicate work per matrix element.
  std::array<std::uint64_t, (1 << 14) / 64> seen{};
  std::vector<ByteInterval> intervals;
  auto add = [&](std::uint64_t chunk) {
    const auto raw = d.start_address + 16 * chunk;
    if (raw > (1U << 18) - 16)
      Fail("shared MMA footprint exceeds 18-bit address domain");
    const auto address = SwizzleByteAddress(raw, bits);
    const auto index = address / 16;
    const auto mask = std::uint64_t{1} << (index % 64);
    // Python's chunk set coalesces repeated logical chunks. XOR swizzling is
    // bijective, hence the same physical chunk can safely be deduplicated here.
    if (!(seen[index / 64] & mask)) {
      seen[index / 64] |= mask;
      intervals.push_back({address, 16});
    }
  };
  if (major == "mn") {
    const auto chunks = mn / normalized;
    if (chunks % atom || k % 8)
      Fail("MN-major matrix does not tile swizzle atom");
    for (unsigned outer = 0; outer < chunks / atom; ++outer)
      for (unsigned inner = 0; inner < atom; ++inner)
        for (unsigned ko = 0; ko < k / 8; ++ko)
          for (unsigned ki = 0; ki < 8; ++ki)
            add(!bits ? std::uint64_t{outer} * stride + ki +
                            std::uint64_t{ko} * leading
                      : inner + std::uint64_t{outer} * leading + ki * atom +
                            std::uint64_t{ko} * stride);
  } else {
    if (mn % 8)
      Fail("K-major matrix MN dimension does not tile groups of eight");
    for (unsigned outer = 0; outer < mn / 8; ++outer)
      for (unsigned inner = 0; inner < 8; ++inner)
        for (unsigned chunk = 0; chunk < k / normalized; ++chunk)
          add(!bits ? inner + std::uint64_t{outer} * stride +
                          std::uint64_t{chunk} * leading
                    : inner * atom + std::uint64_t{outer} * stride + chunk);
  }
  return MergeIntervals(std::move(intervals));
}
std::vector<ByteInterval>
DecodeSharedMatrixElements(std::uint64_t descriptor, std::uint32_t mn,
                           std::uint32_t k, std::uint32_t element_bits,
                           std::string_view major) {
  if (std::uint64_t{mn} * k > (1u << 22))
    Fail("shared matrix element enumeration exceeds resource limit");
  // The footprint decoder supplies the canonical-layout validity checks.
  (void)DecodeSharedMatrixFootprint(descriptor, mn, k, element_bits, major);
  const auto d = DecodeSharedMemoryDescriptor(descriptor);
  const unsigned swizzle = d.layout == 0 ? 0 : d.layout == 2 ? 3 : d.layout == 4 ? 2 : 1;
  const unsigned atom = 1u << swizzle, normalized = 128 / element_bits;
  const auto leading = d.leading_byte_offset / 16, stride = d.stride_byte_offset / 16;
  const auto bytes = element_bits / 8;
  std::vector<ByteInterval> result;
  result.reserve(std::uint64_t{mn} * k);
  // PTX canonical SMEM layouts: logical coordinates remain (MN,K).
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-layout
  for (unsigned row = 0; row < mn; ++row)
    for (unsigned column = 0; column < k; ++column) {
      std::uint64_t chunk;
      unsigned element;
      if (major == "mn") {
        const unsigned outer = row / (normalized * atom), inner = row / normalized % atom;
        const unsigned ko = column / 8, ki = column % 8;
        chunk = !swizzle ? std::uint64_t{outer} * stride + ki + std::uint64_t{ko} * leading
                         : inner + std::uint64_t{outer} * leading + ki * atom + std::uint64_t{ko} * stride;
        element = row % normalized;
      } else {
        const unsigned outer = row / 8, inner = row % 8, part = column / normalized;
        chunk = !swizzle ? inner + std::uint64_t{outer} * stride + std::uint64_t{part} * leading
                         : inner * atom + std::uint64_t{outer} * stride + part;
        element = column % normalized;
      }
      result.push_back({SwizzleByteAddress(d.start_address + 16 * chunk, swizzle) + bytes * element, bytes});
    }
  return result;
}

void ForEachSharedMatrixMapping(std::uint64_t descriptor, std::uint32_t mn,
    std::uint32_t k, std::uint32_t element_bits, std::string_view major,
    const std::function<void(const MatrixByteMapping&)>& consume) {
  // Reuse canonical validity and physical-bound checks. This visits 16-byte
  // physical chunks, not individual numerical elements.
  (void)DecodeSharedMatrixFootprint(descriptor,mn,k,element_bits,major);
  const auto d=DecodeSharedMemoryDescriptor(descriptor);
  const unsigned swizzle=d.layout==0?0:d.layout==2?3:d.layout==4?2:1;
  const unsigned atom=1u<<swizzle, normalized=128/element_bits, bytes=element_bits/8;
  const auto leading=d.leading_byte_offset/16,stride=d.stride_byte_offset/16;
  if(major=="mn") {
    for(unsigned row=0;row<mn;row+=normalized)
      for(unsigned column=0;column<k;++column) {
        const unsigned outer=row/(normalized*atom),inner=row/normalized%atom;
        const unsigned ko=column/8,ki=column%8;
        const auto chunk=!swizzle?std::uint64_t{outer}*stride+ki+std::uint64_t{ko}*leading:
            inner+std::uint64_t{outer}*leading+ki*atom+std::uint64_t{ko}*stride;
        consume({SwizzleByteAddress(d.start_address+16*chunk,swizzle),
            (std::uint64_t{row}*k+column)*bytes,bytes,normalized,std::uint64_t{k}*bytes});
      }
  } else {
    for(unsigned row=0;row<mn;++row)
      for(unsigned column=0;column<k;column+=normalized) {
        const unsigned outer=row/8,inner=row%8,part=column/normalized;
        const auto chunk=!swizzle?inner+std::uint64_t{outer}*stride+std::uint64_t{part}*leading:
            inner*atom+std::uint64_t{outer}*stride+part;
        consume({SwizzleByteAddress(d.start_address+16*chunk,swizzle),
            (std::uint64_t{row}*k+column)*bytes,bytes,normalized,bytes});
      }
  }
}

std::vector<std::pair<ByteInterval, ByteInterval>>
DecodeCopyFragments(std::uint64_t value, std::uint64_t taddr,
                    std::string_view shape,
                    std::optional<std::uint32_t> columns) {
  if (shape != "128x128b" && shape != "128x256b" && shape != "32x128b.warpx4")
    Fail("copy fragment mapping is unavailable for this shape");
  const unsigned width = shape == "128x256b" ? 32 : 16;
  const unsigned rows = shape == "32x128b.warpx4" ? 32 : 128;
  const auto d = DecodeSharedMemoryDescriptor(value);
  (void)DecodeSharedMatrixFootprint(value, rows, width, 8, "k");
  const auto [lane, column] = DecodeTmemAddress(taddr);
  std::vector<ByteInterval> bounds;
  Rectangle(bounds, lane, column, 128, width / 4, columns.value_or(512));
  const unsigned bits = d.layout == 0 ? 0 : d.layout == 2 ? 3 : d.layout == 4 ? 2 : 1;
  std::vector<std::pair<ByteInterval, ByteInterval>> result;
  for (unsigned row = 0; row < 128; ++row)
    for (unsigned chunk = 0; chunk < width / 16; ++chunk) {
      const auto src_row = row % rows;
      auto offset = std::uint64_t{src_row / 8} * d.stride_byte_offset;
      offset += !bits ? (src_row % 8) * 16 + chunk * d.leading_byte_offset
                      : (src_row % 8) * (1U << bits) * 16 + chunk * 16;
      result.push_back({{SwizzleByteAddress(d.start_address + offset, bits), 16},
                        {EncodeCellByteOffset(lane + row, column + 4 * chunk), 16}});
    }
  return result;
}
ExactTmemFootprint DecodeMmaDFootprint(
    const Instruction &instruction, std::uint64_t taddr, std::uint64_t idesc,
    bool input_d, std::span<const std::uint32_t> masks, std::string resource,
    std::optional<std::uint32_t> columns, std::optional<std::uint32_t> part) {
  const auto d = StandardMma(instruction, idesc, part, true);
  const bool group2 = Has(instruction, "cta_group::2");
  if (resource.empty())
    Fail("TMEM resource identity cannot be empty");
  if (masks.size() != (group2 ? 8U : 4U) ||
      std::any_of(masks.begin(), masks.end(),
                  [](auto mask) { return mask != 0; }))
    Fail("MMA output lane masks need the exact supported all-zero shape");
  const auto [base_lane, base_column] = DecodeTmemAddress(taddr);
  auto width = d.n;
  bool layout_f = false;
  if (Has(instruction, "ws")) {
    // PTX 9.2 layouts D/E/G: 128/M lane groups of M lanes each hold N*M/128
    // columns at the same base column; M=128 is layout D.
    if (base_lane != 0)
      Fail(".ws MMA requires lane alignment zero");
    width = d.n * d.m / 128;
  } else if (group2) {
    if ((d.m != 128 && d.m != 256) || base_lane != 0)
      Fail("group2 MMA requires M=128/256 and lane alignment zero");
    if (d.n % 2)
      Fail("group2 N must split equally");
    if (d.m == 128)
      width /= 2;
  } else if (d.m == 128) {
    if (base_lane)
      Fail("M=128 MMA requires lane alignment zero");
  } else if (d.m == 64) {
    if (base_lane != 0 && base_lane != 16)
      Fail("M=64 MMA requires lane alignment zero or sixteen");
    layout_f = true;
  } else
    Fail("unsupported MMA D layout");
  const auto limit = columns.value_or(512);
  if (!limit || limit > 512 || base_column > limit ||
      width > limit - base_column)
    Fail("MMA D exceeds allocation columns");
  std::vector<ByteInterval> intervals;
  intervals.reserve((layout_f ? 64 : 128) *
                    (d.d_element_bits == 32 ? 1 : width));
  for (unsigned lane = 0; lane < 128; ++lane) {
    if (layout_f && (lane % 32 < base_lane || lane % 32 >= base_lane + 16))
      continue;
    if (d.d_element_bits == 32)
      intervals.push_back(
          {EncodeCellByteOffset(lane, base_column), std::uint64_t{width} * 4});
    else
      for (unsigned col = base_column; col < base_column + width; ++col)
        intervals.push_back(
            {EncodeCellByteOffset(lane, col), d.d_element_bits / 8});
  }
  intervals = MergeIntervals(std::move(intervals));
  ExactTmemFootprint result{std::move(resource), {}, std::move(intervals)};
  if (input_d)
    result.reads = result.writes;
  return result;
}
ExactTmemFootprint DecodeMmaATmemFootprint(
    const Instruction &instruction, std::uint64_t taddr, std::uint64_t idesc,
    std::string resource, std::optional<std::uint32_t> columns,
    std::optional<std::uint32_t> part, bool allow_layout_f) {
  const auto d = StandardMma(instruction, idesc, part);
  const bool group2 = Has(instruction, "cta_group::2");
  const auto [lane, column] = DecodeTmemAddress(taddr);
  // .ws lane groups each read A rows from their own lanes: all 128 lanes.
  const bool ws = Has(instruction, "ws");
  const bool layout_f = !ws && !group2 && d.m == 64 && allow_layout_f;
  if (resource.empty())
    Fail("TMEM resource identity cannot be empty");
  if ((!layout_f && ((d.m != 128 && !(group2 && d.m == 256) && !ws) || lane)) ||
      (layout_f && lane != 0 && lane != 16))
    Fail("unsupported MMA A layout or alignment");
  if (d.a_element_bits != 8 && d.a_element_bits != 16 && d.a_element_bits != 32)
    Fail("packed TMEM A input is not decoded");
  const auto row_bits = d.k * d.a_element_bits;
  if (row_bits % 32)
    Fail("MMA A row occupies a fractional TMEM cell");
  ExactTmemFootprint result{std::move(resource), {}, {}};
  if (layout_f)
    for (unsigned partition = 0; partition < 4; ++partition)
      Rectangle(result.reads, lane + 32 * partition, column, 16, row_bits / 32,
                columns.value_or(512));
  else
    Rectangle(result.reads, lane, column, 128, row_bits / 32,
              columns.value_or(512));
  return result;
}
TmaTransferMapping DecodeTmaTransfer(const nlohmann::json &descriptor,
                                     std::span<const std::int64_t> origins,
                                     const TmaTransferOptions &o) {
  if (!descriptor.is_object() ||
      descriptor.value("interleave", std::string{}) != "none" ||
      descriptor.value("load_mode", std::string{"tile"}) != "tile")
    Fail("TMA requires explicit non-interleaved tiled descriptor");
  const auto swizzle = descriptor.value("swizzle", std::string{});
  const unsigned bits = swizzle == "none"   ? 0
                        : swizzle == "32B"  ? 1
                        : swizzle == "64B"  ? 2
                        : swizzle == "128B" ? 3
                                            : 4;
  if (bits == 4 || descriptor.value("swizzle_atomicity", 16) != 16)
    Fail("unsupported TMA swizzle or atomicity");
  auto optional_vector=[&](const char *key, unsigned minimum) {
    std::vector<std::optional<std::uint64_t>> result;
    if (!descriptor.contains(key) || !descriptor[key].is_array()) Fail(std::string("missing TMA ")+key);
    for (const auto &v : descriptor[key])
      result.push_back(v.is_null()?std::nullopt:std::optional<std::uint64_t>(Integer(v,key,minimum)));
    return result;
  };
  const auto shape = Vector(descriptor, "box_shape"), steps = Vector(descriptor, "element_strides");
  const auto bounds=optional_vector("global_shape",1), strides=optional_vector("coordinate_strides_bytes",0);
  const auto rank = shape.size();
  if (!rank || rank > 5 || bounds.size() != rank || strides.size() != rank ||
      steps.size() != rank || origins.size() != rank)
    throw LayoutError("TMA rank and coordinate vectors disagree", "tma_coordinate_rank_mismatch",
      {{"rank",rank},{"supported_rank_min",1},{"supported_rank_max",5},
       {"box_shape",shape},{"global_shape",descriptor.at("global_shape")},
       {"coordinate_strides_bytes",descriptor.at("coordinate_strides_bytes")},
       {"element_strides",steps},{"coordinates",std::vector<std::int64_t>(origins.begin(),origins.end())},
       {"vector_lengths",{{"box_shape",rank},{"global_shape",bounds.size()},
         {"coordinate_strides_bytes",strides.size()},{"element_strides",steps.size()},{"coordinates",origins.size()}}}});
  if (!o.symbolic_coordinates.empty() && o.symbolic_coordinates.size() != rank)
    Fail("symbolic TMA coordinate rank mismatch");
  const auto symbolic = [&](std::size_t dim) {
    return !o.symbolic_coordinates.empty() && o.symbolic_coordinates[dim];
  };
  std::vector<std::int64_t> coordinates(origins.begin(), origins.end());
  const bool packed = descriptor.contains("packed_element_bits") &&
                      !descriptor["packed_element_bits"].is_null();
  const auto packed_alignment = descriptor.value("packed_shared_alignment", 8U);
  // tile::gather4 (PTX ISA 8.6+): four rows of a 2D map, each box_shape[0]
  // elements from the column origin, land as the dense {box_shape[0], 4} box.
  const bool gather = o.gather_rows.has_value();
  if (gather && (rank != 2 || shape[1] != 1 || origins[1] != 0 || o.store ||
                 packed || o.multicast_mask))
    Fail("TMA gather4 requires an unpacked rank-2 load map with a one-row box and no multicast");
  if (gather)
    for (const auto row : *o.gather_rows)
      if (row < std::numeric_limits<std::int32_t>::min() ||
          row > std::numeric_limits<std::int32_t>::max())
        Fail("TMA coordinates must fit signed 32 bits");
  if (packed) {
    if (symbolic(0))
      throw LayoutError("Packed TMA requires proven inner-coordinate alignment", "tma_packed_coordinate_alignment_not_proven");
    if (descriptor["packed_element_bits"] != 4 || o.store ||
        descriptor.value("element_bytes", 0) != 1 ||
        (packed_alignment != 8 && packed_alignment != 16) ||
        shape[0] % 8 || !bounds[0] || *bounds[0] % 8 || coordinates[0] % 16)
      Fail("packed TMA requires a supported whole-group U4 load layout");
    if (packed_alignment == 16 && (shape[0] != 64 || *bounds[0] % 64))
      Fail("U4 ALIGN16B requires a 128-element inner tile");
    coordinates[0] /= 2;
  }
  if (o.store && std::any_of(origins.begin(), origins.end(),
                           [](auto coordinate) { return coordinate < 0; }))
    Fail("TMA store origins must be nonnegative");
  if (descriptor.contains("rank") &&
      Integer(descriptor["rank"], "rank") != rank)
    Fail("TMA explicit rank disagrees with shape");
  if (std::any_of(steps.begin(), steps.end(),
                  [](auto step) { return step != 1; }))
    throw LayoutError("Nonunit TMA element strides are unsupported", "tma_element_stride_unsupported",
      {{"element_strides",steps},{"supported_element_stride",1}});
  for (const auto origin : origins)
    if (origin < std::numeric_limits<std::int32_t>::min() ||
        origin > std::numeric_limits<std::int32_t>::max())
      Fail("TMA coordinates must fit signed 32 bits");
  if (!descriptor.contains("element_bytes"))
    Fail("missing TMA element_bytes");
  const auto element = Integer(descriptor["element_bytes"], "element_bytes", 1);
  if ((element != 1 && element != 2 && element != 4 && element != 8 &&
       element != 16) ||
      strides[0] != element)
    Fail("TMA dimension zero must have contiguous byte elements");
  // The enumerated destination box; a gather has four rows of the one-row map box.
  auto box = shape;
  if (gather) box[1] = 4;
  std::uint64_t count = 1;
  for (const auto extent : box) {
    if (!o.max_elements || extent > o.max_elements / count)
      Fail("TMA enumeration exceeds explicit bound");
    count *= extent;
  }
  if (count > std::numeric_limits<std::uint64_t>::max() / element)
    Fail("TMA tile byte count overflows u64");
  const auto size = count * element;
  // A map's own payload is one box: one gather row.
  const auto box_bytes = gather ? size / 4 : size;
  if ((o.transfer_bytes && *o.transfer_bytes != size) ||
      (descriptor.contains("transfer_bytes") &&
       !descriptor["transfer_bytes"].is_null() &&
       Integer(descriptor["transfer_bytes"], "transfer_bytes") != box_bytes))
    Fail("TMA transfer count disagrees with logical tile");
  const auto padding = packed ? packed_alignment / 8 : 1U;
  if (size > (1ULL << 32) / padding)
    Fail("TMA padded tile exceeds shared address space");
  const auto shared_size = size * padding;
  if (o.shared_base % 128 || o.shared_base > (1ULL << 32) - shared_size)
    Fail("TMA shared base must be aligned and tile fit u32");
  if (bits && shape[0] * element * padding != (16U << bits))
    Fail("TMA swizzled inner dimension must equal swizzle span");
  const bool abstract_address=descriptor.value("symbolic_global",false) ||
      (descriptor.contains("base_byte_offset") && descriptor["base_byte_offset"].is_null()) ||
      std::any_of(strides.begin(),strides.end(),[](const auto &v){return !v;});
  const auto global_base =
      descriptor.contains("base_byte_offset") && !descriptor["base_byte_offset"].is_null()
          ? Integer(descriptor["base_byte_offset"], "base_byte_offset")
          : 0;
  if (!o.cluster_size || o.cluster_size > 16 ||
      o.destination_rank >= o.cluster_size)
    Fail("TMA destination is outside <=16 CTA cluster");
  if (o.cta_group != 1 && o.cta_group != 2)
    Fail("TMA group must be one or two");
  if (o.cta_group == 2 && (o.destination_rank ^ 1) >= o.cluster_size)
    Fail("group2 TMA destination peer is absent");
  if (o.store && (o.cta_group != 1 || o.completion_rank || o.multicast_mask))
    Fail("TMA stores cannot use barrier/group2/multicast routing");
  std::vector<std::uint32_t> destinations;
  if (o.multicast_mask) {
    if (!*o.multicast_mask || *o.multicast_mask >= (1U << o.cluster_size))
      Fail("TMA multicast mask is empty or includes absent CTA");
    for (unsigned destination = 0; destination < o.cluster_size; ++destination)
      if (*o.multicast_mask & (1U << destination))
        destinations.push_back(destination);
  } else
    destinations.push_back(o.destination_rank);
  TmaTransferMapping result{{}, size, {}, o.store};
  if (o.completion_rank) {
    const auto completion = *o.completion_rank;
    if (completion >= o.cluster_size ||
        (o.cta_group == 1 && completion != o.destination_rank) ||
        (o.cta_group == 2 && completion / 2 != o.destination_rank / 2))
      Fail("TMA completion barrier is outside permitted destination/peer CTA");
    result.completions.reserve(destinations.size());
    for (const auto destination : destinations) {
      const auto target = o.cta_group == 1
                              ? destination
                              : (destination & ~1U) | (completion & 1);
      if (target >= o.cluster_size)
        Fail("TMA multicast completion peer is absent");
      result.completions.push_back({target, size});
    }
  }
  // Compute address correspondence once per payload, then replicate typed runs
  // for multicast targets. This avoids recomputing coordinates for each peer.
  std::vector<TmaTransferSegment> payload;
  payload.reserve(
      static_cast<std::size_t>(std::min<std::uint64_t>(count, 4096)));
  std::array<std::uint64_t, 5> delta{};
  for (std::uint64_t linear = 0; linear < count; ++linear) {
    bool oob = false;
    bool unknown_bounds=false;
    std::array<std::uint64_t, 5> coordinate{};
    for (std::size_t dim = 0; dim < rank; ++dim) {
      // A gather row's tensor row is its own coordinate (no box offset).
      const bool row = gather && dim == 1;
      if (row ? o.symbolic_rows[delta[1]] : symbolic(dim)) { unknown_bounds = true; continue; }
      const auto origin_value = row ? (*o.gather_rows)[delta[1]] : coordinates[dim];
      const auto step = row ? std::uint64_t{0} : delta[dim];
      // Compute signed origin + unsigned delta without overflow. An
      // unrepresentable coordinate is out of bounds for every representable
      // global shape.
      if (origin_value < 0) {
        const auto magnitude = static_cast<std::uint64_t>(-origin_value);
        if (step < magnitude) {
          oob = true;
          continue;
        }
        coordinate[dim] = step - magnitude;
      } else {
        const auto origin = static_cast<std::uint64_t>(origin_value);
        if (step > std::numeric_limits<std::uint64_t>::max() - origin) {
          oob = true;
          continue;
        }
        coordinate[dim] = step + origin;
      }
      if (!bounds[dim]) unknown_bounds=true;
      else if (coordinate[dim] >= *bounds[dim])
        oob = true;
    }
    if (oob && !o.store && descriptor.value("oob_fill", std::string{}) != "zero")
      Fail("out-of-bounds TMA tile requires explicit zero-fill load");
    std::optional<std::uint64_t> global;
    if (!oob && !abstract_address && !unknown_bounds) {
      // A tile ending exactly at 2^64 is valid. Subtract element-1 from
      // UINT64_MAX to express that exclusive-end bound without overflow.
      std::uint64_t offset = global_base;
      const auto maximum_offset =
          std::numeric_limits<std::uint64_t>::max() - (element - 1);
      if (offset > maximum_offset)
        Fail("TMA global byte interval exceeds u64");
      for (std::size_t dim = 0; dim < rank; ++dim) {
        if (*strides[dim] != 0 &&
            coordinate[dim] > (maximum_offset - offset) / *strides[dim])
          Fail("TMA global byte interval exceeds u64");
        offset += coordinate[dim] * *strides[dim];
      }
      global = offset;
    }
    const auto logical = linear * element;
    const auto physical = packed ? logical / 8 * packed_alignment + logical % 8
                                 : logical;
    AppendSegment(payload,
                  {global, SwizzleByteAddress(o.shared_base + physical, bits),
                   element, logical, destinations.front(), !oob && (abstract_address || unknown_bounds),
                   static_cast<std::uint32_t>(gather ? delta[1] : 0U)});
    for (std::size_t dim = 0; dim < rank; ++dim) {
      if (++delta[dim] != box[dim])
        break;
      delta[dim] = 0;
    }
  }
  result.segments.reserve(payload.size() * destinations.size());
  for (const auto destination : destinations)
    for (auto segment : payload) {
      segment.target_rank = destination;
      result.segments.push_back(std::move(segment));
    }
  return result;
}
std::vector<ByteInterval>
MatrixLaneFootprint(unsigned count, bool transpose, unsigned lane,
                    std::span<const std::uint64_t> rows) {
  if ((count != 1 && count != 2 && count != 4) || lane >= 32 ||
      rows.size() != 8 * count)
    Fail("invalid m8n8.b16 matrix register/lane/row dimensions");
  for (const auto row : rows)
    if (row % 16 || row > (1ULL << 32) - 16)
      Fail("matrix row address requires aligned in-range shared u32");
  std::vector<ByteInterval> result;
  result.reserve(count * (transpose ? 2 : 1));
  for (unsigned matrix = 0; matrix < count; ++matrix) {
    if (transpose)
      for (unsigned half = 0; half < 2; ++half)
        result.push_back(
            {rows[8 * matrix + 2 * (lane % 4) + half] + 2 * (lane / 4), 2});
    else
      result.push_back({rows[8 * matrix + lane / 4] + 4 * (lane % 4), 4});
  }
  return result;
}
MatrixShape DecodeMatrixShape(const Module &module,
                              const Instruction &instruction, bool store) {
  if (instruction.opcode != (store ? "stmatrix" : "ldmatrix") ||
      instruction.operands.size() != 2)
    Fail("matrix operation requires one vector and one shared address");
  unsigned count = 0, counts = 0, spaces = 0;
  std::set<std::string> seen;
  for (const auto &modifier : instruction.modifiers) {
    if (!seen.insert(modifier).second)
      Fail("duplicate matrix opcode modifier");
    if (modifier == "x1" || modifier == "x2" || modifier == "x4") {
      count = static_cast<unsigned>(modifier[1] - '0');
      ++counts;
    } else if (modifier == "shared" || modifier == "shared::cta")
      ++spaces;
    else if (modifier != "sync" && modifier != "aligned" &&
             modifier != "m8n8" && modifier != "b16" && modifier != "trans")
      Fail("unsupported matrix opcode modifier");
  }
  if (counts != 1 || spaces != 1 || !Has(instruction, "sync") ||
      !Has(instruction, "aligned") || !Has(instruction, "m8n8") ||
      !Has(instruction, "b16"))
    Fail("only aligned explicit-shared m8n8.b16 matrix memory is decoded");
  const auto vector = Trim(module.view(instruction.operands[store ? 1 : 0]));
  const auto address = Trim(module.view(instruction.operands[store ? 0 : 1]));
  if (vector.size() < 2 || vector.front() != '{' || vector.back() != '}' ||
      address.size() < 2 || address.front() != '[' || address.back() != ']')
    Fail("matrix operand syntax disagrees with direction");
  unsigned slots = 0;
  auto text = std::string_view(vector).substr(1, vector.size() - 2);
  while (true) {
    const auto comma = text.find(',');
    const auto name = Trim(text.substr(0, comma));
    const auto registers = Registers(name);
    if (registers.size() != 1 || registers[0] != name)
      Fail("matrix register vector contains non-register operand");
    ++slots;
    if (comma == text.npos)
      break;
    text.remove_prefix(comma + 1);
  }
  if (slots != count)
    Fail("m8n8.b16 requires one 32-bit register per matrix");
  return {count, Has(instruction, "trans"), address};
}
RegisterTable::RegisterTable(const Module &module, const Function &function) {
  static const std::regex declaration(
      R"(^\s*\.reg\s+\.(pred|(?:[busf]|bf)[0-9]+)\s+([^;]+);?\s*$)");
  static const std::regex names(
      R"((%?[A-Za-z_$][A-Za-z0-9_$]*)(?:<([0-9]+)>)?)");
  for (const auto &directive : function.directives) {
    const auto text = std::string(module.view(directive.source));
    std::smatch match;
    if (!std::regex_match(text, match, declaration))
      continue;
    const auto type = match[1].str(), list = match[2].str();
    const auto digit = type.find_first_of("0123456789");
    const auto width =
        type == "pred"
            ? 1U
            : static_cast<unsigned>(std::stoul(type.substr(digit))) / 8;
    for (std::sregex_iterator it(list.begin(), list.end(), names), end;
         it != end; ++it) {
      const auto name = (*it)[1].str();
      const bool array = (*it)[2].matched;
      const auto extent = array ? std::stoull((*it)[2].str()) : 1;
      if (extent > 1'000'000)
        Fail("register declaration expansion exceeds supported bound");
      for (std::uint64_t index = 0; index < extent; ++index) {
        const auto reg = name + (array ? std::to_string(index) : std::string{});
        const auto key = ScopedKey(directive.scope, reg);
        std::string identity = reg;
        if (!directive.scope.empty()) {
          identity += "@scope:";
          for (std::size_t i = 0; i < directive.scope.size(); ++i) {
            if (i)
              identity += '.';
            identity += std::to_string(directive.scope[i]);
          }
        }
        if (!declarations_.emplace(key, identity).second)
          Fail("duplicate register declaration in one lexical scope");
        widths_[identity] = width;
      }
    }
  }
}
std::string RegisterTable::Resolve(std::span<const std::uint32_t> scope,
                                   std::string_view name) const {
  if (name.find("@scope:") != name.npos && widths_.contains(std::string(name)))
    return std::string(name);
  for (std::size_t length = scope.size() + 1; length > 0; --length) {
    const auto found =
        declarations_.find(ScopedKey(scope.first(length - 1), name));
    if (found != declarations_.end())
      return found->second;
  }
  return std::string(name);
}
std::uint32_t RegisterTable::Width(std::string_view name) const {
  if (const auto found = widths_.find(std::string(name));
      found != widths_.end())
    return found->second;
  const auto base = name.substr(0, name.find('.'));
  if (base == "%tid" || base == "%ntid" || base == "%ctaid" ||
      base == "%nctaid" || base == "%laneid" || base == "%warpid" ||
      base == "%clock" || base == "%smid")
    return 4;
  Fail("unknown declared scalar register width: " + std::string(name));
}
} // namespace sentryward::frontend::ptx
