#pragma once
#include "importer.hpp"

namespace rg {

// The layout mapping below reads each source coordinate as (offset / stride_j) mod extent_j, the
// digits of a dense source layout only. A strided source layout (nested_layouts, which holds the
// dominant ones too) is refused: with padding, e.g. strides [1536, 256, 2] for shape [7, 4, 128],
// that formula maps offset 1536 to (1, 2, 0) instead of (1, 0, 0) and padding bytes to elements.
// An identity layout (target strides = source strides, dense) is never strided.
inline auto Importer::canonical_packed_slice(Id source,unsigned lo,unsigned count) -> std::optional<PackedDecodePass::Bits> {
  const auto &read=a.nodes[source];
  if(read.op!="readbits" || read.args.size()!=1 || !count || count>4 ||
     lo%8+count>8 || Q(lo+count)>read.data[0] ||
     !m.contains("layouts") || !m["layouts"].contains(read.tag) ||
     !packed_dense_layouts.contains(read.tag) || nested_layouts.contains(read.tag))return {};
  const auto &layout=m["layouts"][read.tag];
  unsigned bytes=layout["element_bytes"].get<unsigned>();
  if(!bytes)return {};
  Id off=a.sub(read.args[0],a.constant(jq(layout.value("source_offset",J(0)))));
  Z extent=bytes;
  for(size_t j=0;j<layout["shape"].size();++j)
    extent+=(jq(layout["shape"][j]).numerator()-1)*jq(layout["source_strides_bytes"][j]).numerator();
  auto bounds=a.nodes[off].bounds;
  // Establish coverage of the original load, not merely its selected nibble.
  // Unknown accesses retain the original expression and no new assumptions.
  if(!bounds || bounds->first<0 || bounds->second+Q((read.data[0].numerator()+7)/8)>Q(extent))return {};
  Id selected=a.add({off,a.constant(lo/8)});
  if(packed_identity_layouts.contains(read.tag)) {
    // Same dense coordinates: no need to decompose and reassemble each axis.
    Id byte_address=a.add({selected,a.constant(jq(layout.value("target_offset",J(0))))});
    Id bit_address=a.add({a.mul({a.constant(8),byte_address}),a.constant(lo%8)});
    ++a.stats["packed_slice_identity_mappings"];
    return PackedDecodePass::Bits{mapped_name(m,"allocations",read.tag),bit_address,count};
  }
  const auto &target=layout.contains("target_strides_bytes")?layout["target_strides_bytes"]:layout["source_strides_bytes"];
  std::vector<Id> terms{a.constant(jq(layout.value("target_offset",J(0)))),a.imod(selected,Z(bytes))};
  for(size_t j=0;j<layout["shape"].size();++j) {
    Id coordinate=a.imod(a.idiv(selected,jq(layout["source_strides_bytes"][j]).numerator()),jq(layout["shape"][j]).numerator());
    terms.push_back(a.mul({a.constant(jq(target[j])),coordinate}));
  }
  Id bit_address=a.add({a.mul({a.constant(8),a.add(terms)}),a.constant(lo%8)});
  ++a.stats["packed_slice_layout_mappings"];
  return PackedDecodePass::Bits{mapped_name(m,"allocations",read.tag),bit_address,count};
}

inline auto Importer::typed_integer_fact(Id value, const std::string &type) -> Id {
  // Preserve an existing declared integer type at its producer; this is
  // not inferred from a numeric range and adds no input assumption.
  if (!floating(type) && type != "pred" &&
      (type.starts_with("s") || type.starts_with("u") || type.starts_with("b"))) {
    packed_decoders.known_integers.insert(value);
    fp4_boundaries.known_integers.insert(value);
  }
  return value;
}

inline auto Importer::typed_read(const std::string &raw, Id offset, const std::string &t, bool require_valid) -> Id {
  if(t=="e8m0" || t=="ue8m0")return typed_e8m0_input(raw,offset);
  if (t == "ue4m3") return typed_ue4m3_input(raw, offset);
  std::string alloc = mapped_name(m, "allocations", raw);
  if (!m.contains("layouts") || !m["layouts"].contains(raw))
    throw Unsupported("missing input layout " + raw);
  const auto &l = m["layouts"][raw];
  Id off = a.sub(offset, a.constant(jq(l.value("source_offset", J(0)))));
  unsigned bytes = l["element_bytes"];
  if (width(t) != 8 * bytes)
    throw Unsupported(
        "typed input interpretation crosses logical storage elements: " +
        alloc + " " + t);
  // An exact digit certificate is applied before extent and typed-coordinate
  // construction. All enclosing strides must respect the last-axis span;
  // packed lanes arrive here separately with their exact byte displacement.
  if (floating(t) && bytes == 2 && xor_coordinates.eligible(off) && !l["shape"].empty() &&
      jq(l["source_strides_bytes"].back()) == Q(bytes)) {
    Q last = jq(l["shape"].back());
    if (last == 64 || last == 128) {
      Z span = Z(bytes) * last.numerator();
      bool aligned = true;
      for (size_t j = 0; j + 1 < l["shape"].size(); ++j)
        if (modz(jq(l["source_strides_bytes"][j]).numerator(), span) != 0) aligned = false;
      if (aligned) off = xor_coordinates.address(off, bytes, last == 64 ? 6 : 7);
    }
  }
  Id remainder = a.imod(off, bytes);
  if (remainder != a.zero)
    throw Unsupported("Could not prove that the input read is element-aligned",
                      "input_alignment_not_proven",
                      {{"allocation",alloc},{"element_bytes",bytes},
                       {"offset_expression",a.render(off,3)},{"remainder_expression",a.render(remainder,3)}});
  Z extent = bytes;
  for (size_t j = 0; j < l["shape"].size(); ++j)
    extent += (jq(l["shape"][j]).numerator() - 1) *
              jq(l["source_strides_bytes"][j]).numerator();
  auto b = a.nodes[off].bounds;
  bool valid = b && b->first >= 0 &&
      b->second + Q(std::max(1u, width(t) / 8)) <= Q(extent);
  // A nested strided layout: the offset must be a multiple of the innermost stride and each
  // inner digit below its axis's extent x stride (the outermost digit is bounded by the
  // extent above); then its digits are the element's coordinates. O(rank) Arena terms.
  std::vector<Id> nested_coordinates;
  if (const auto nested = nested_layouts.find(raw); valid && nested != nested_layouts.end()) {
    const auto &axes = nested->second;
    nested_coordinates.assign(l["shape"].size(), a.zero);
    if (dominant_layouts.contains(raw)) {
      // Greedy digits from the largest stride: each remainder must lie below its axis's
      // extent x stride, and the last remainder must be zero. O(rank) Arena terms.
      Id rest = off;
      for (size_t j = axes.size(); valid && j-- > 0;) {
        const auto &bound = a.nodes[rest].bounds;
        if (j + 1 < axes.size() &&
            (!bound || bound->first < 0 || bound->second >= Q(axes[j].stride * axes[j].count))) valid = false;
        nested_coordinates[axes[j].dim] = a.idiv(rest, axes[j].stride);
        rest = a.imod(rest, axes[j].stride);
      }
      if (rest != a.zero) valid = false;
    } else {
      if (!axes.empty() && a.imod(off, axes.front().stride) != a.zero) valid = false;
      for (size_t j = 0; valid && j < axes.size(); ++j) {
        Id digit = off;
        if (j + 1 < axes.size()) {
          digit = a.imod(off, axes[j + 1].stride);
          const auto &bound = a.nodes[digit].bounds;
          if (!bound || bound->first < 0 || bound->second >= Q(axes[j].stride * axes[j].count)) valid = false;
        }
        nested_coordinates[axes[j].dim] = a.idiv(digit, axes[j].stride);
      }
    }
  }
  if (!valid) {
    if(require_valid || l.value("target_strides_bytes",l["source_strides_bytes"]) != l["source_strides_bytes"])
      throw Unsupported("input offset range not established for layout/constraint: " + alloc);
    unresolved_accesses.emplace(alloc,offset);
    ++a.stats["symbolic_byte_reads_without_extent_proof"];
    Id canonical=a.add({off,a.constant(jq(l.value("target_offset",J(0))))});
    Bounds storage;
    if(t.starts_with("s") || t.starts_with("u")) {
      unsigned bits=width(t);
      storage=t.starts_with("s")?Bounds(std::pair(Q(-pow2(bits-1)),Q(pow2(bits-1)-1))):Bounds(std::pair(Q(0),Q(pow2(bits)-1)));
    }
    // A separate interpretation tag prevents collisions with coordinate reads.
    // Object contracts deliberately do not apply outside established coverage.
    return typed_integer_fact(a.read(alloc,{canonical},(floating(t)?"real:byte-address:":"byte-address:")+t,storage),t);
  }
  std::vector<Id> coords=std::move(nested_coordinates);
  if (coords.empty())
    for (size_t j = 0; j < l["shape"].size(); ++j)
      coords.push_back(
          a.imod(a.idiv(off, jq(l["source_strides_bytes"][j]).numerator()),
                 jq(l["shape"][j]).numerator()));
  if (floating(t) && !ideal)
    throw Unsupported("non-ideal numeric input");
  Bounds bound = ranges.contains(alloc) ? ranges[alloc] : Bounds{};
  if (!bound && !floating(t) && (t.starts_with("s") || t.starts_with("u"))) {
    unsigned bits = width(t);
    bound = t.starts_with("s") ? Bounds(std::pair(Q(-pow2(bits-1)), Q(pow2(bits-1)-1)))
                                : Bounds(std::pair(Q(0), Q(pow2(bits)-1)));
  }
  if (contract) if (const auto *rule = contract->find(alloc,t); rule && region_covers(rule->region,raw,coords,width(t)/8)) {
    Bounds effective=rule->range;
    if(!effective && bound && (rule->lower || rule->upper))
      effective=std::pair(rule->lower.value_or(bound->first),rule->upper.value_or(bound->second));
    if (effective) {
      auto incoming=*effective;
      if (bound) incoming={std::max(bound->first,incoming.first),std::min(bound->second,incoming.second)};
      if(incoming.first>incoming.second) throw Unsupported("conflicting input ranges: "+alloc);
      bound=incoming;
    }
    consumed_contracts.emplace(alloc,t);
    contract->applied.emplace(side,alloc,t);
  }
  // Bind fixed read-only inputs on both sides (see fixed_bits) only after the
  // ordinary identity, storage type, bounds and coordinate checks above;
  // never specialize by expression node ID.
  if(context)
    if(auto fixed=context->fixed_read(a,alloc,t,coords,side!="left")) {
      require_fixed_in_range(alloc,*fixed,bound);
      return *fixed;
    }
  return typed_integer_fact(a.read(alloc, coords, floating(t) ? "real:" + t : t, bound),t);
}

inline auto Importer::reconstruct(const std::vector<Id> &xs) -> std::optional<Id> {
  if (xs.size() != 2)
    return {};
  for (unsigned j = 0; j < 2; ++j) {
    const auto &lo = a.nodes[xs[j]];
    const auto &hi = a.nodes[xs[1 - j]];
    if (hi.op != "mul" || hi.data != std::vector<Q>{Q(pow2(32))} ||
        hi.args.size() != 1 || lo.args.size() != 1)
      continue;
    const auto &h = a.nodes[hi.args[0]];
    if (h.args != lo.args)
      continue;
    const Id source = lo.args[0];
    if (lo.op == "extractbits" && lo.data == std::vector<Q>{Q(0), Q(32)} &&
        h.op == "extractbits" && h.data == std::vector<Q>{Q(32), Q(32)}) {
      ++a.stats["exact_word_reconstruction"];
      return unwrap(word(source, 64), 64);
    }
    // The same split in arithmetic form (a u64 shifted right by 32 and
    // truncated to u32): x mod 2^32 + 2^32*floor(x/2^32) = x for every
    // integer x, so the sum is congruent to x modulo 2^64.
    if (lo.op == "word" && lo.data == std::vector<Q>{Q(32)} &&
        h.op == "idiv" && h.data == std::vector<Q>{Q(pow2(32))} &&
        a.known_bits(source).integer) {
      ++a.stats["exact_word_arithmetic_reconstruction"];
      return unwrap(word(source, 64), 64);
    }
  }
  return {};
}

inline auto Importer::mask32(Id x, uint32_t mask) -> Id {
  std::vector<Id> pieces;
  for_each_bit_run(mask, [&](unsigned lo, unsigned count) {
    pieces.push_back(a.mul({a.constant(Q(pow2(lo))), a.imod(a.idiv(x, pow2(lo)), pow2(count))}));
  });
  return a.add(pieces);
}

inline auto Importer::byte_permute(Id x, Id y, unsigned control) -> Id {
  std::vector<Id> bytes;
  for (unsigned i = 0; i < 4; ++i) {
    unsigned selector = (control >> (4 * i)) & 15;
    unsigned lane = selector & 7;
    Id source = lane < 4 ? x : y;
    Id byte = a.imod(a.idiv(source, pow2(8 * (lane % 4))), Z(256));
    if (selector & 8) byte = a.mul({a.constant(255), a.idiv(byte, Z(128))});
    bytes.push_back(a.mul({a.constant(Q(pow2(8 * i))), byte}));
  }
  return a.add(bytes);
}

inline auto Importer::logic3_constants(Id x, uint32_t b, uint32_t c, unsigned lut) -> Id {
  uint32_t positive = 0, negative = 0, ones = 0;
  for (unsigned i = 0; i < 32; ++i) {
    unsigned index = (((b >> i) & 1u) << 1) | ((c >> i) & 1u);
    bool low = (lut >> index) & 1u, high = (lut >> (index + 4)) & 1u;
    if (low && high) ones |= uint32_t(1) << i;
    else if (high) positive |= uint32_t(1) << i;
    else if (low) negative |= uint32_t(1) << i;
  }
  // (x & positive) | ones with disjoint masks (e.g. lop3 0xEA, the dequantization idiom
  // (w & 0x00F000F0) | 0x64006400): keep each selected run a bounded bit field of x, so
  // the fields stay visible to typed slices and packed decoders. O(runs).
  if (negative == 0) {
    std::vector<Id> pieces{a.constant(Q(Z(ones)))};
    for_each_bit_run(positive, [&](unsigned lo, unsigned count) {
      pieces.push_back(a.mul({a.constant(Q(pow2(lo))), a.bit_slice(x, lo, count)}));
    });
    return a.add(std::move(pieces));
  }
  return a.add({a.constant(Q(Z(ones) + Z(negative))), mask32(x, positive),
                a.neg(mask32(x, negative))});
}

// A demanded integer quotient may discard a bounded low radix digit.
// Keep the proof at the typed import boundary: general Arena arithmetic and
// numerical real division do not acquire another address search.
inline auto Importer::integer_index_quotient(Id dividend, const Z &divisor,
                          unsigned work_limit) -> Id {
  unsigned budget = std::min(work_limit, PackedDecodePass::max_nodes);
  auto reduce_once = [&](Id operand, const Z &denominator) -> std::optional<Id> {
    const auto &node = a.nodes[operand];
    if (!budget || node.op != "add" || node.args.size() > 8 || denominator <= 1 ||
        (denominator & (denominator - 1)) != 0) return {};
    auto result = packed_decoders.mixed_radix_quotient(operand, denominator, budget);
    if (result) ++a.stats["input_index_radix_quotients"];
    return result;
  };
  auto first = reduce_once(dividend, divisor);
  Id result = first ? *first : a.idiv(dividend, divisor);
  // Discarding a lane digit can expose another short quotient, such as
  // floor((8*k+head)/1024) after the byte lane was removed. Follow only the
  // returned root or direct quotient terms of a small Add, for four rounds
  // total. Every probe shares the original 32-unit transport budget.
  for (unsigned round = 1; round < 4 && budget; ++round) {
    const auto &node = a.nodes[result];
    auto follow = [&](Id id) -> Id {
      const auto &part = a.nodes[id];
      if (part.op != "idiv" || part.args.size() != 1 || part.data.size() != 1 ||
          part.data[0].denominator() != 1) return id;
      Z denominator = part.data[0].numerator();
      // Do not repeat an initial miss on an unchanged quotient.
      if (!first && part.args[0] == dividend && denominator == divisor) return id;
      auto next = reduce_once(part.args[0], denominator);
      if (!next || *next == id) return id;
      ++a.stats["input_index_radix_followups"];
      return *next;
    };
    Id next = result;
    if (node.op == "idiv") next = follow(result);
    else if (node.op == "add" && node.args.size() <= 8 && node.args.size() == node.data.size()) {
      auto args = node.args; auto coefficients = node.data;
      bool changed = false;
      for (Id &argument : args) {
        Id rewritten = follow(argument);
        changed |= rewritten != argument;
        argument = rewritten;
      }
      if (changed) {
        std::vector<Id> terms;
        for (size_t j = 0; j < args.size(); ++j)
          terms.push_back(a.mul({a.constant(coefficients[j]), args[j]}));
        next = a.add(std::move(terms));
      }
    }
    if (next == result) break;
    result = next;
  }
  return result;
}

} // namespace rg
