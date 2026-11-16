#pragma once
#include "execution_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
void Executor::Data(const Decoded &d, State &s, std::uint32_t warp,
                    std::uint32_t lane, std::uint64_t occurrence,
                    const Json &contract, WarpContext &context,
                    bool probe) const {
  const auto &i = *d.instruction;
  const auto &op = i.opcode;
  auto n = d.operands.size();
  auto val = [&](std::size_t index) -> Scalar {
    return index < n ? Value(d.operands[index], s) : Scalar{};
  };
  if (IsWait(i)) {
    if (contract.contains("register_results"))
      Error(i, warp, lane,
            "mbarrier wait results cannot come from fixed-input "
            "register_results");
    if (probe)
      Set(d, s, {false});
    else
      Invalidate(d, s);
    return;
  }
  // .local memory is private to the thread: an exact store fills slots that a load of the same
  // bytes reads back in program order; any other write that may reach .local memory forgets
  // every slot. A load without a filled slot keeps the behaviour below.
  if (local_memory_ && (LocalSpace(i) || MayWriteLocal(i))) {
    const auto target = LocalSpace(i) ? Local(d, s) : LocalTarget{{}, true};
    const auto width = i.memory_width.value_or(0);
    if (op == "ld" && target.exact && !contract.contains("register_results")) {
      const auto count = d.destination_slots.size();
      if (count && width % count == 0) {
        const auto bytes = static_cast<unsigned>(width / count);
        const bool sign = Has(i, "s" + std::to_string(bytes * 8));
        std::vector<Scalar> values;
        bool known = false;
        for (std::size_t k = 0; k < count; ++k) {
          const auto slot = s.local.find(LocalAddress{target.exact->first, target.exact->second + std::int64_t(k * bytes)});
          values.push_back(slot != s.local.end() && slot->second.bytes == bytes
                               ? LocalLoaded(slot->second.value, bytes, sign) : Scalar{});
          known |= !Unknown(values.back());
        }
        if (known) {
          Set(d, s, std::move(values));
          return;
        }
      }
    } else if (MayWriteLocal(i) && target.may_alias) {
      const auto parts = op == "st" && n >= 2 ? VectorElements(d.operands[1].text) : std::vector<std::string_view>{};
      if (target.exact && width && !parts.empty() && width % parts.size() == 0) {
        const auto bytes = static_cast<unsigned>(width / parts.size());
        ForgetLocal(s.local, *target.exact, width);
        for (std::size_t k = 0; k < parts.size(); ++k)
          if (auto value = ValueText(parts[k], s); !Unknown(value))
            s.local.insert_or_assign(LocalAddress{target.exact->first, target.exact->second + std::int64_t(k * bytes)},
                                     LocalSlot<Scalar>{bytes, std::move(value)});
      } else
        s.local.clear();
    }
  }
  if (op == "ld" && Has(i, "param")) {
    if (n < 2)
      Error(i, warp, lane, "malformed ld.param");
    if (!i.memory_width || d.destinations.empty()) {
      Invalidate(d, s);
      return;
    }
    if (*i.memory_width % d.destinations.size())
      Error(i, warp, lane, "parameter load cannot split across destinations");
    auto width = *i.memory_width / d.destinations.size();
    auto address = d.operands[1].text;
    auto view = TrimView(address);
    if (view.starts_with('[') && view.ends_with(']'))
      view = TrimView(view.substr(1, view.size() - 2));
    auto split = SplitAddress(view);
    std::string name = split ? split->first : std::string(view);
    Integer offset = split ? split->second : Integer(0);
    auto base = ValueText(name, s);
    if (auto p = std::get_if<ParameterPointer>(&base)) {
      name = p->region;
      offset += p->offset;
    }
    const bool sign = SignedModifier(i);
    std::vector<Scalar> values;
    for (std::size_t index = 0; index < d.destinations.size(); ++index) {
      auto field = Int64(offset + index * width, "parameter offset");
      if (field >= 0) {
        auto pointer = pointers_.find({name, std::uint64_t(field), width});
        if (pointer != pointers_.end()) {
          const auto &p = *pointer->second;
          values.emplace_back(AllocationPointer{p.resource, 0, p.extent_bytes});
          continue;
        }
      }
      const auto &reg = names_[d.destinations[index]];
      std::optional<DescriptorFragment> descriptor_fragment;
      for (auto it=c_.tma_descriptors.begin();it!=c_.tma_descriptors.end();++it) {
        const auto &map=it.value();
        if (!map.contains("parameter_location")) continue;
        const auto &location=map.at("parameter_location");
        const auto begin=location.at(1).get<std::int64_t>();
        if (location.at(0)==name && field>=begin && Integer(field)+width<=Integer(begin)+128) {
          descriptor_fragment=DescriptorFragment{it.key(),static_cast<unsigned>(field-begin),static_cast<unsigned>(width)};
          break;
        }
      }
      if (descriptor_fragment) { values.emplace_back(*descriptor_fragment); continue; }
      if (contract.contains("register_results") &&
          contract["register_results"].is_object() &&
          contract["register_results"].contains(reg) &&
          (contract["register_results"][reg].is_number_integer() ||
           contract["register_results"][reg].is_boolean())) {
        values.emplace_back(JsonInteger(contract["register_results"][reg]));
        continue;
      }
      auto region = c_.parameter_regions.find(name);
      if (region != c_.parameter_regions.end()) {
        auto v = (field < 0 || std::uint64_t(field) < region->second.size_bytes)
                     ? region->second.ReadInteger(field, width, sign)
                     : std::optional<Integer>{};
        values.push_back(v ? Scalar(*v) : Scalar{});
        continue;
      }
      auto exact = c_.parameters.find(std::string(view));
      values.push_back(d.destinations.size() == 1 &&
                               exact != c_.parameters.end()
                           ? Scalar(exact->second)
                           : Scalar{});
    }
    Set(d, s, std::move(values));
    return;
  }
  // Generic loads qualify too: the pointer must still resolve to a global
  // fixed-input allocation below.
  const bool global_or_generic =
      Has(i, "global") ||
      std::none_of(i.modifiers.begin(), i.modifiers.end(), [](const auto &m) {
        return m.starts_with("shared") || In(m, {"local", "param", "const"});
      });
  if (op == "ld" && global_or_generic && !Floating(i) && n == 2u + Has(i, "L2::cache_hint") &&
      d.operands[1].address && i.memory_width && !d.destination_slots.empty()) {
    const auto &[base, displacement] = *d.operands[1].address;
    // A table-atom pointer reads at its exact offset.
    const auto pointer_value = PlainValue(ValueText(base, s));
    // A whole 8-byte slot of a pointer table yields its entry; any other access is unknown.
    if (auto pointer = std::get_if<AllocationPointer>(&pointer_value);
        pointer && pointer_tables_.contains(pointer->resource)) {
      const auto &table = pointer_tables_.at(pointer->resource);
      const auto count = d.destination_slots.size();
      std::vector<Scalar> values;
      for (std::size_t index = 0; index < count; ++index) {
        const auto offset = Int64(pointer->offset + displacement + index * 8, "pointer table offset");
        const auto found = offset < 0 ? table.end() : table.find(std::uint64_t(offset));
        values.push_back(*i.memory_width == 8 * count && found != table.end() ? found->second : Scalar{});
      }
      Set(d, s, std::move(values));
      return;
    }
    if (auto bounded = std::get_if<BoundedAllocationPointer>(&pointer_value);
        bounded && pointer_tables_.contains((*bounded)->resource) && *i.memory_width == 8 * d.destination_slots.size()) {
      std::vector<Scalar> values;
      bool known = false;
      for (std::size_t index = 0; index < d.destination_slots.size(); ++index) {
        values.push_back(BoundedTableEntry(pointer_tables_.at((*bounded)->resource), *bounded->value,
                                           displacement + Integer(index * 8)));
        known |= !Unknown(values.back());
      }
      if (known) {
        Set(d, s, std::move(values));
        return;
      }
    }
    // A pointer with a non-affine offset reads every candidate: a constant input gives
    // the loaded values over the offset's key; an object with tensor_properties gives its
    // bounds, the element named by the offset (read-only: one offset, one value).
    if (auto bounded = std::get_if<BoundedAllocationPointer>(&pointer_value);
        bounded && (*bounded)->table &&
        (constant_inputs_.contains((*bounded)->resource) || tensor_properties_.contains((*bounded)->resource))) {
      const auto &pointer = *bounded->value;
      const auto &offsets = *pointer.table;
      const auto count = d.destination_slots.size();
      std::vector<Scalar> values;
      if (const auto region = constant_inputs_.find(pointer.resource); region != constant_inputs_.end()) {
        if (*i.memory_width % count != 0) { Invalidate(d, s); return; }
        const auto width = *i.memory_width / count;
        const bool sign = SignedModifier(i);
        for (std::size_t index = 0; index < count; ++index) {
          BoundedTable table{offsets.key, offsets.key_lower, {}};
          for (const auto &candidate : offsets.values) {
            const Integer offset = candidate + displacement + Integer(index * width);
            const auto read = offset < 0 || offset > Integer(region->second.size_bytes) ||
                    Integer(width) > Integer(region->second.size_bytes) - offset
                ? std::optional<Integer>{} : region->second.ReadInteger(offset.convert_to<std::int64_t>(), width, sign);
            if (!read) { table.values.clear(); break; }
            table.values.push_back(*read);
          }
          values.push_back(table.values.empty() ? Scalar{} : TableValue(std::move(table),
              "load:" + pointer.resource + "[" + pointer.offset_identity + "+" +
                  (displacement + Integer(index * width)).convert_to<std::string>() + "]:" + std::to_string(width),
              pointer.sources));
        }
      } else {
        const auto &property = tensor_properties_.at(pointer.resource);
        const auto bytes = property.bits / 8;
        if (*i.memory_width != count * bytes) { Invalidate(d, s); return; }
        for (std::size_t index = 0; index < count; ++index) {
          const Integer shift = displacement + Integer(index * bytes);
          const bool aligned = std::all_of(offsets.values.begin(), offsets.values.end(), [&](const Integer &v) {
            return v + shift >= 0 && (v + shift) % bytes == 0;
          });
          if (!aligned) { values.push_back(Scalar{}); continue; }
          if (ExactElement(property, i.id, values)) continue;
          const auto type = property.Type();
          auto sources = pointer.sources;
          if (std::find(sources.begin(), sources.end(), property.object) == sources.end()) sources.push_back(property.object);
          values.push_back(MakeBounded(property.lower, property.upper,
              "element:" + property.object + "+[" + pointer.offset_identity + "+" + shift.convert_to<std::string>() +
                  "]:" + type, std::move(sources)));
        }
      }
      Set(d, s, std::move(values));
      return;
    }
    if (auto pointer = std::get_if<AllocationPointer>(&pointer_value);
        pointer && constant_inputs_.contains(pointer->resource)) {
      const auto count = d.destination_slots.size();
      if (*i.memory_width % count != 0) { Invalidate(d, s); return; }
      const auto width = *i.memory_width / count;
      const auto &region = constant_inputs_.at(pointer->resource);
      const bool sign = SignedModifier(i);
      std::vector<Scalar> values;
      for (std::size_t index = 0; index < count; ++index) {
        const auto offset = Int64(pointer->offset + displacement + index * width, "constant input offset");
        const auto value = offset < 0 || std::uint64_t(offset) > region.size_bytes ||
                width > region.size_bytes - std::uint64_t(offset)
            ? std::optional<Integer>{} : region.ReadInteger(offset, width, sign);
        values.push_back(value ? TableElement(pointer->resource, region, std::uint64_t(offset),
                                              static_cast<unsigned>(width), sign, *value)
                               : Scalar{});
      }
      Set(d, s, std::move(values));
      return;
    }
    // A whole aligned element of an object with tensor_properties is bounded; its
    // identity is the element, which the read-only contract keeps unchanged. A scalar
    // load of an element's low-order bytes (at the element's address, little-endian) is
    // that element when every value within the bounds fits the load's type.
    if (auto pointer = std::get_if<AllocationPointer>(&pointer_value);
        pointer && tensor_properties_.contains(pointer->resource)) {
      const auto &property = tensor_properties_.at(pointer->resource);
      const auto bytes = property.bits / 8;
      const auto count = d.destination_slots.size();
      const bool narrow = count == 1 && *i.memory_width < bytes &&
                          NarrowElementFits(property, *i.memory_width, SignedModifier(i));
      if (*i.memory_width != count * bytes && !narrow) { Invalidate(d, s); return; }
      std::vector<Scalar> values;
      for (std::size_t index = 0; index < count; ++index) {
        const Integer offset = pointer->offset + displacement + Integer(index * bytes);
        if (offset < 0 || offset % bytes != 0) { values.push_back(Scalar{}); continue; }
        if (ExactElement(property, i.id, values)) continue;
        if (narrow) property_decisions_->Insert(property.object, i.id);
        const auto type = property.Type();
        values.push_back(MakeBounded(property.lower, property.upper,
            "element:" + property.object + "+" + offset.convert_to<std::string>() + ":" + type,
            {property.object}, AffineForm{0, {{ElementAtom{property.object, type, offset}, Integer(1)}}}));
      }
      Set(d, s, std::move(values));
      return;
    }
  }
  if (op == "clusterlaunchcontrol" && Has(i, "query_cancel") && Has(i, "is_canceled") &&
      c_.fixed_input.contains("cluster_launch_cancellation") &&
      c_.fixed_input["cluster_launch_cancellation"] == false) {
    Set(d, s, {false});
    return;
  }
  if (op == "mov" && n > 1) {
    if (d.destinations.empty())
      return;
    auto width = Width(i);
    auto value = val(1);
    auto integer = ScalarInteger(value);
    if (Has(i, "pred")) {
      Set(d, s, {DataPredicate(value) ? value : integer ? Scalar(*integer != 0) : Scalar{}});
      return;
    }
    auto destination = TrimView(d.operands[0].text);
    if (destination.starts_with('{') && destination.ends_with('}')) {
      std::vector<std::string> slots;
      auto text = destination.substr(1, destination.size() - 2);
      std::size_t begin = 0;
      while (true) {
        auto end = text.find(',', begin);
        slots.emplace_back(TrimView(text.substr(begin, end == std::string_view::npos
                                                       ? text.size() - begin
                                                       : end - begin)));
        if (end == std::string_view::npos)
          break;
        begin = end + 1;
      }
      if (slots.size() > 1 && width && *width % slots.size() == 0) {
        std::vector<Scalar> result;
        auto part = *width / slots.size();
        Integer mask = (Integer(1) << part) - 1;
        for (std::size_t j = 0; j < slots.size(); ++j)
          if (slots[j] != "_")
            result.push_back(
                integer ? Scalar(Integer((*integer >> (j * part)) & mask))
                        : Scalar{});
        Set(d, s, std::move(result));
        return;
      }
    }
    if (d.destinations.size() > 1) {
      if (!width || *width % d.destinations.size() || !integer) {
        Invalidate(d, s);
        return;
      }
      auto part = *width / d.destinations.size();
      Integer mask = (Integer(1) << part) - 1;
      std::vector<Scalar> result;
      for (std::size_t j = 0; j < d.destinations.size(); ++j)
        result.emplace_back(Integer((*integer >> (j * part)) & mask));
      Set(d, s, std::move(result));
      return;
    }
    if (d.operands[1].text.starts_with('{') && d.operands[1].text.ends_with('}')) {
      std::vector<std::string> parts;
      auto source = std::string_view(d.operands[1].text);
      source.remove_prefix(1); source.remove_suffix(1);
      for (std::size_t begin = 0;;) {
        auto end = source.find(',', begin);
        parts.emplace_back(TrimView(source.substr(begin, end == source.npos ? source.size() - begin : end - begin)));
        if (end == source.npos) break;
        begin = end + 1;
      }
      auto count = parts.size();
      if (!width || *width % count) {
        Invalidate(d, s);
        return;
      }
      auto part = *width / count;
      Integer packed = 0, mask = (Integer(1) << part) - 1;
      for (std::size_t j = 0; j < count; ++j) {
        auto v = ScalarInteger(ValueText(parts[j], s));
        if (!v) {
          Invalidate(d, s);
          return;
        }
        packed |= (*v & mask) << (j * part);
      }
      Set(d, s, {packed});
      return;
    }
    if (std::holds_alternative<ClusterSharedPointer>(value) &&
        (!width || *width < 32))
      value = Scalar{};
    Set(d, s, {value});
    return;
  }
  if (In(op, {"brev", "bfind", "clz", "popc"}) && n == 2) {
    auto value = ScalarInteger(val(1));
    auto width = Width(i);
    if (!value || !width || (*width != 32 && *width != 64)) {
      Invalidate(d, s); return;
    }
    const Integer mask = (Integer(1) << *width) - 1;
    Integer bits = *value & mask, result = 0;
    if (op == "bfind" && Has(i, "s" + std::to_string(*width)) &&
        (bits & (Integer(1) << (*width - 1))) != 0) bits ^= mask;
    if (op == "brev") {
      for (unsigned bit = 0; bit < *width; ++bit)
        if ((bits & (Integer(1) << bit)) != 0) result |= Integer(1) << (*width - 1 - bit);
    } else if (op == "popc") {
      for (unsigned bit = 0; bit < *width; ++bit)
        if ((bits & (Integer(1) << bit)) != 0) ++result;
    } else {
      int top = int(*width) - 1;
      while (top >= 0 && (bits & (Integer(1) << top)) == 0) --top;
      if (op == "clz") result = int(*width) - 1 - top;
      else if (top < 0) result = UINT32_MAX;
      else result = Has(i, "shiftamt") ? int(*width) - 1 - top : top;
    }
    Set(d, s, {result}); return;
  }
  if (op == "isspacep" && n == 2 && i.modifiers.size() == 1) {
    const auto inside = GenericAddressInSpace(i.modifiers[0], val(1), c_.resources);
    Set(d, s, {inside ? Scalar(*inside) : Scalar{}});
    return;
  }
  if (op == "bfe" && n == 4) {
    auto value = ScalarInteger(val(1)), position = ScalarInteger(val(2)), length = ScalarInteger(val(3));
    auto width = Width(i);
    if (!value || !position || !length || !width || (*width != 32 && *width != 64)) {
      Invalidate(d, s);
      return;
    }
    const auto start = (*position & 255).convert_to<unsigned>();
    const auto count = (*length & 255).convert_to<unsigned>();
    const Integer mask = (Integer(1) << *width) - 1;
    Integer bits = *value & mask;
    const bool sign = Has(i, "s" + std::to_string(*width));
    if (sign && (bits & (Integer(1) << (*width - 1))) != 0)
      bits -= Integer(1) << *width;
    Integer result = 0;
    if (count) {
      result = (bits >> start) & ((Integer(1) << count) - 1);
      if (sign && ((result >> (count - 1)) & 1) != 0)
        result -= Integer(1) << count;
    }
    Set(d, s, {Integer(result & mask)});
    return;
  }
  // shf.{l,r}.{clamp,wrap}.b32 d, a, b, c (PTX ISA, logic and shift instructions): d is the
  // most- (l) or least- (r) significant word of the 64-bit value [b, a] shifted by
  // n = min(c, 32) (.clamp) or c & 0x1f (.wrap), with c read as u32.
  if (op == "shf" && n == 4) {
    constexpr unsigned word_bits = 32;
    const bool left = Has(i, "l"), clamp = Has(i, "clamp");
    const auto low = ScalarInteger(val(1)), high = ScalarInteger(val(2)), amount = ScalarInteger(val(3));
    if (left == Has(i, "r") || clamp == Has(i, "wrap") || !Has(i, "b32") || !low || !high || !amount) {
      Invalidate(d, s);
      return;
    }
    const Integer mask = (Integer(1) << word_bits) - 1, count = *amount & mask;
    const auto shift = (clamp ? std::min<Integer>(count, word_bits) : Integer(count & (word_bits - 1)))
                           .convert_to<unsigned>();
    const Integer source = ((*high & mask) << word_bits) | (*low & mask);
    const Integer shifted = left ? Integer((source << shift) >> word_bits) : Integer(source >> shift);
    Set(d, s, {Integer(shifted & mask)});
    return;
  }
  if (In(op, {"cvt", "cvta"}) && n > 1) {
    if (op == "cvt" && Floating(i)) {
      Set(d,s,{ExactIntegerToFloat(val(1),i)});
      return;
    }
    auto value = val(1);
    auto width = Width(i);
    if (std::holds_alternative<ClusterSharedPointer>(value) &&
        (!width || *width < 32))
      value = Scalar{};
    // A table atom keeps its form through cvt and cvta only where the bounded rule keeps the
    // value unchanged (it fits both types; a 64-bit pointer); otherwise it converts as its plain value.
    if (const auto *pinned = PinnedInteger(value)) {
      std::vector<std::string> kinds;
      for (const auto &m : i.modifiers)
        if (m.size() > 1 && In(m.substr(0, 1), {"b", "u", "s"}) &&
            std::all_of(m.begin() + 1, m.end(), [](unsigned char c) { return std::isdigit(c); }))
          kinds.push_back(m);
      const auto fits = [&](const std::string &type) {
        const unsigned bits = std::stoul(type.substr(1));
        const bool sign = type[0] == 's';
        const Integer low = sign ? -(Integer(1) << (bits - 1)) : Integer(0);
        const Integer high = (Integer(1) << (bits - unsigned(sign))) - 1;
        return low <= pinned->lower && pinned->upper <= high;
      };
      if (op != "cvt" || kinds.size() != 2 || !fits(kinds[0]) || !fits(kinds[1])) value = PlainValue(value);
    } else if (const auto *pointer = PinnedPointer(value); pointer && op == "cvt") {
      value = PlainValue(value);
    }
    // Integer cvt: read the source at its own width and signedness (sign- or zero-
    // extending), then wrap, or clamp under .sat, to the destination type.
    std::vector<std::string> types;
    for (const auto &m : i.modifiers)
      if (m.size() > 1 && In(m.substr(0, 1), {"b", "u", "s"}) &&
          std::all_of(m.begin() + 1, m.end(), [](unsigned char c) { return std::isdigit(c); }))
        types.push_back(m);
    if (const auto *bounded = std::get_if<BoundedInteger>(&value); bounded && op == "cvt" && types.size() == 2) {
      // Exact when the value fits both the source and the destination type.
      if (!FitsType(*bounded, std::stoul(types[1].substr(1)), types[1][0] == 's') ||
          !FitsType(*bounded, std::stoul(types[0].substr(1)), types[0][0] == 's'))
        value = Scalar{};
    } else if (std::holds_alternative<BoundedAllocationPointer>(value) && op == "cvt" &&
               (types.size() != 2 || types[0].substr(1) != "64" || types[1].substr(1) != "64")) {
      value = Scalar{};
    } else if (auto n = ScalarInteger(value); op == "cvt" && n && types.size() == 2) {
      const auto dest = std::stoul(types[0].substr(1)), source = std::stoul(types[1].substr(1));
      const bool dest_sign = types[0][0] == 's';
      Integer x = Reduce(*n, source, types[1][0] == 's');
      if (Has(i, "sat")) {
        const Integer low = dest_sign ? -(Integer(1) << (dest - 1)) : Integer(0);
        const Integer high = (Integer(1) << (dest - dest_sign)) - 1;
        x = x < low ? low : x > high ? high : x;
      } else
        x = Reduce(x, dest, dest_sign);
      value = x;
    }
    Set(d, s, {value});
    return;
  }
  if (In(op, {"and", "or", "xor"}) && Has(i, "pred") && n > 2) {
    Set(d, s, {PredicateBinary(op, val(1), val(2))});
    return;
  }
  if (op == "mul" && n > 2) {
    if (Floating(i))
      Invalidate(d, s);
    else
      Set(d, s, {IntegerProduct(i, val(1), val(2))});
    return;
  }
  if (In(op, {"add", "sub", "shl", "shr", "and", "or", "xor", "min", "max",
              "rem", "div"}) &&
      n > 2) {
    if (Floating(i)) {
      Invalidate(d, s);
      return;
    }
    auto left = val(1), right = val(2);
    auto pointer = PointerArithmetic(op, left, right, c_);
    Set(d, s, {pointer ? *pointer : IntegerBinary(op, left, right, &i)});
    return;
  }
  // On a bounded operand, neg x = 0 - x and not x = -1 - x (exact, like Integer).
  if (op == "neg" && n > 1) {
    auto value = val(1);
    auto x = ScalarInteger(value);
    Set(d, s, {Floating(i) ? Scalar{}
               : std::holds_alternative<BoundedInteger>(value) ? IntegerBinary("sub", Scalar(Integer(0)), value, &i)
               : x ? Scalar(Integer(-*x)) : Scalar{}});
    return;
  }
  if (op == "not" && n > 1) {
    auto value = val(1);
    auto x = ScalarInteger(value);
    Set(d, s,
        {Has(i, "pred") ? NegatePredicate(value)
         : std::holds_alternative<BoundedInteger>(value) ? IntegerBinary("sub", Scalar(Integer(-1)), value, &i)
         : x            ? Scalar(Integer(~*x))
                        : Scalar{}});
    return;
  }
  if (op == "mad" && n > 3) {
    if (Floating(i)) {
      Invalidate(d, s);
      return;
    }
    auto left = IntegerBinary("mul", val(1), val(2)), right = val(3);
    auto pointer = PointerArithmetic("add", left, right, c_);
    Set(d, s, {pointer ? *pointer : IntegerBinary("add", left, right)});
    return;
  }
  if (op == "setp" && n > 2) {
    std::string relation;
    for (const auto &modifier : i.modifiers)
      if (In(modifier, {"eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu",
                        "leu", "gtu", "geu", "num", "nan"})) {
        relation = modifier;
        break;
      }
    const auto left = val(1), right = val(2);
    auto comparison = Compare(i, relation, left, right, c_);
    if (!Unknown(comparison)) NoteBoundedDecision(i.id, {&left, &right});
    std::string boolean;
    for (const auto &modifier : i.modifiers)
      if (In(modifier, {"and", "or", "xor"})) {
        boolean = modifier;
        break;
      }
    if (!boolean.empty() && n > 3)
      Set(d, s,
          {PredicateBinary(boolean, comparison, val(3)),
           PredicateBinary(boolean, NegatePredicate(comparison), val(3))});
    else
      Set(d, s, {comparison, NegatePredicate(comparison)});
    return;
  }
  if (op == "selp" && n > 3) {
    auto cond = Truth(val(3));
    Set(d, s, {cond ? val(*cond ? 1 : 2) : Scalar{}});
    return;
  }
  if (op == "mapa" && n > 2) {
    auto address = ScalarInteger(val(1)), rank = ScalarInteger(val(2));
    if (address && *address >= 0 && *address < (Integer(1) << 32) && rank &&
        *rank >= 0 && c_.cluster_dim &&
        *rank < Integer((*c_.cluster_dim)[0]) * (*c_.cluster_dim)[1] *
                    (*c_.cluster_dim)[2])
      Set(d, s,
          {ClusterSharedPointer{*address, rank->convert_to<std::uint32_t>()}});
    else
      Invalidate(d, s);
    return;
  }
  if (op == "elect" && !i.modifiers.empty() && i.modifiers[0] == "sync") {
    if (n != 2) Error(i, warp, lane, "elect.sync requires destination and membermask");
    auto raw = ScalarInteger(val(1));
    if (!raw || *raw < -(Integer(1) << 31) || *raw > UINT32_MAX)
      Error(i, warp, lane, "invalid or unresolved election mask");
    const auto mask = (*raw & 0xffffffff).convert_to<std::uint32_t>();
    if (!mask || !(mask & (1u << lane))) Error(i, warp, lane, "elect issuer is outside membermask");
    auto chosen = c_.elect_lane;
    if (!chosen) {
      if (std::popcount(mask) != 1) Error(i, warp, lane, "election policy not configured");
      chosen = std::countr_zero(mask);
    } else if (*chosen >= 32 || !(mask & (1u << *chosen))) chosen = std::countr_zero(mask);
    Set(d, s, {Integer(*chosen), lane == *chosen});
    return;
  }
  if (op == "shfl") {
    if (i.modifiers.size() >= 2 && i.modifiers[0] == "sync" &&
        In(i.modifiers[1], {"idx", "up", "down", "bfly"}) && n == 5) {
      context.shuffle[{i.id, occurrence, lane}] = val(1);
      const auto selected=SelectShuffle(d,s,warp,lane);
      if (!selected) {
        Invalidate(d, s);
        return;
      }
      const auto from=selected->lane;
      auto it = context.shuffle.find({i.id, occurrence, from});
      if (it == context.shuffle.end())
        throw Deferred{static_cast<std::uint32_t>(from), i.id, occurrence};
      Set(d, s, {it->second, selected->valid});
      return;
    }
    if (contract.value("source_is_warp_uniform", false) && n > 1) {
      auto v = val(1);
      Set(d, s, {v, Unknown(v) ? Scalar{} : Scalar(true)});
      return;
    }
    if (contract.contains("result") &&
        (contract["result"].is_number_integer() ||
         contract["result"].is_boolean())) {
      Set(d, s, {JsonInteger(contract["result"]), true});
      return;
    }
    Invalidate(d, s);
    return;
  }
  if (op == "vote" && i.modifiers.size() == 3 && i.modifiers[0] == "sync" &&
      In(i.modifiers[1], {"ballot", "all", "any", "uni"}) && n == 3) {
    auto mask = ScalarInteger(val(2));
    if (!mask) { Invalidate(d, s); return; }
    const auto members = (*mask & 0xffffffff).convert_to<std::uint32_t>();
    if (!(members & (1u << lane)))
      Error(i, warp, lane, "executing lane absent from vote membermask");
    auto predicate = val(1);
    auto text = TrimView(d.operands[1].text);
    if (text.starts_with('!')) {
      auto value = ScalarInteger(ValueText(text.substr(1), s));
      predicate = value ? Scalar(*value == 0) : Scalar{};
    }
    context.shuffle[{i.id, occurrence, lane}] = predicate;
    context.vote_masks[{i.id, occurrence, lane}] = members;
    std::uint32_t bits = 0;
    bool unknown = false;
    for (unsigned peer = 0; peer < 32; ++peer) {
      if (!(members & (1u << peer))) continue;
      auto it = context.shuffle.find({i.id, occurrence, peer});
      if (it == context.shuffle.end()) throw Deferred{peer, i.id, occurrence};
      if (context.vote_masks.at({i.id, occurrence, peer}) != members)
        Error(i, warp, lane, "vote participants have different membermasks");
      auto value = ScalarInteger(it->second);
      if (!value) unknown = true;
      else if (*value != 0) bits |= 1u << peer;
    }
    if (unknown) { Invalidate(d, s); return; }
    const auto &mode = i.modifiers[1];
    if (mode == "ballot") Set(d, s, {Integer(bits)});
    else Set(d, s, {mode == "all" ? bits == members :
                   mode == "any" ? bits != 0 : bits == 0 || bits == members});
    return;
  }
  // match.{any,all}.sync: the participants whose value equals this lane's (Compare's equality).
  if (op == "match" && i.modifiers.size() == 3 && In(i.modifiers[0], {"any", "all"}) && i.modifiers[1] == "sync" &&
      n == 3) {
    auto mask = ScalarInteger(val(2));
    if (!mask) { Invalidate(d, s); return; }
    const auto members = (*mask & 0xffffffff).convert_to<std::uint32_t>();
    if (!(members & (1u << lane)))
      Error(i, warp, lane, "executing lane absent from match membermask");
    const auto own = val(1);
    context.shuffle[{i.id, occurrence, lane}] = own;
    context.vote_masks[{i.id, occurrence, lane}] = members;
    std::uint32_t bits = 0;
    bool unknown = false, unequal = false;
    for (unsigned peer = 0; peer < 32; ++peer) {
      if (!(members & (1u << peer))) continue;
      auto it = context.shuffle.find({i.id, occurrence, peer});
      if (it == context.shuffle.end()) throw Deferred{peer, i.id, occurrence};
      if (context.vote_masks.at({i.id, occurrence, peer}) != members)
        Error(i, warp, lane, "match participants have different membermasks");
      const auto equal = Truth(Compare(i, "eq", own, it->second, c_));
      if (!equal) unknown = true;
      else if (*equal) bits |= 1u << peer;
      else unequal = true;
    }
    if (i.modifiers[0] == "any") Set(d, s, {unknown ? Scalar{} : Scalar(Integer(bits))});
    else if (unequal) Set(d, s, {Integer(0), false});
    else if (unknown) Invalidate(d, s);
    else Set(d, s, {Integer(members), true});
    return;
  }
  // activemask: lanes that follow the warp's common control path since launch execute convergently
  // (see WarpContext::Decide), so the active set is the whole warp when every lane is on it here.
  if (op == "activemask" && n == 1) {
    const auto &own = context.control[lane];
    const Scalar position = own.diverged ? Scalar{} : Scalar(Integer(own.decisions));
    context.shuffle[{i.id, occurrence, lane}] = position;
    context.queries.emplace(i.id, occurrence);
    const auto first = warp * c_.warp_size;
    bool known = convergence_ && !Unknown(position) && (!c_.analysis_thread_range ||
        (c_.analysis_thread_range->first <= first && first + c_.warp_size <= c_.analysis_thread_range->second));
    for (std::uint32_t peer = 0; known && peer < c_.warp_size; ++peer) {
      const auto found = context.shuffle.find({i.id, occurrence, peer});
      if (found != context.shuffle.end()) known = found->second == position;
      else if (context.control[peer].diverged) known = false;
      else throw Deferred{peer, i.id, occurrence};
    }
    Set(d, s, {known ? Scalar(Integer(0xffffffffu)) : Scalar{}});
    return;
  }
  if (contract.contains("register_results") &&
      contract["register_results"].is_object()) {
    const auto &results = contract["register_results"];
    for (auto id : d.destinations) {
      auto found = results.find(names_[id]);
      s.registers[id] = found != results.end() && (found->is_number_integer() ||
                                                   found->is_boolean())
                            ? Scalar(JsonInteger(*found))
                            : Scalar{};
    }
    return;
  }
  Invalidate(d, s);
}
} // namespace
} // namespace sentryward::frontend::ptx
