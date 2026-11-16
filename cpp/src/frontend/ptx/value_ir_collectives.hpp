#pragma once
#include "value_ir_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
bool ValueTemplateExtractor::Shuffle() {
  const auto& i = *instruction_;
  const auto key = [&](std::uint32_t lane) {
    return ShuffleKey{facts_.Warp(lane_), i.id, record_->occurrence(), lane, origin_.unroll_context};
  };
  const auto binary = [&](v::BinaryOpcode op, Expression x, Expression y) {
    return Expression{Add(std::make_unique<v::BinaryOp>(op, U32().value), {x.value, y.value}, origin_).result(), U32()};
  };
  if (!shuffle_source_) {
    if (i.operands.size() != 5 || !Has(i, "sync") || !Has(i, "b32") ||
        !(Has(i, "idx") || Has(i, "bfly") || Has(i, "up") || Has(i, "down")))
      throw std::invalid_argument("Unsupported shuffle form");
    auto b = Known(std::string(module_.view(i.operands[2])));
    auto c = Known(std::string(module_.view(i.operands[3])));
    auto mask = Known(std::string(module_.view(i.operands[4])));
    if (!b || !c || !mask) throw std::invalid_argument("Shuffle lane selector/clamp/membermask is dynamic");
    const auto lane = facts_.LaneId(lane_);
    const auto members = (*mask & 0xffffffff).convert_to<std::uint32_t>();
    const auto amount = (*b & 31).convert_to<std::uint32_t>();
    const auto segment = ((*c >> 8) & 31).convert_to<std::uint32_t>();
    const auto maximum = (lane & segment) | ((*c & 31).convert_to<std::uint32_t>() & ~segment);
    int source = (lane & segment) | (amount & ~segment);
    if (Has(i, "bfly")) source = lane ^ amount;
    if (Has(i, "up")) source = int(lane) - int(amount);
    if (Has(i, "down")) source = lane + amount;
    shuffle_valid_ = Has(i, "up") ? source >= int(maximum) : source <= int(maximum);
    if (!shuffle_valid_) source = lane;
    if (source < 0 || source >= 32 || !(members & (1u << lane)) || !(members & (1u << source)))
      throw std::invalid_argument("Shuffle issuer or source is outside the membermask");
    if (demand_mode) {
      const auto valid = shuffle_valid_;
      auto result = demand_peer(source, record_->Key(), std::string(module_.view(i.operands[1])));
      if (!result) { Invalidate(); return true; }
      result = Reinterpret(*result, *PtxType("b32"));
      if (!result) { Invalidate(); return true; }
      const auto destinations = Split(module_.view(i.operands[0]), '|');
      Bind(destinations[0], *result);
      if (destinations.size() == 2) Bind(destinations[1], Constant(Pred(), valid ? "true" : "false"));
      return true;
    }
    auto value = Operand(std::string(module_.view(i.operands[1])), *PtxType("b32"));
    if (value) value = Reinterpret(*value, *PtxType("b32"));
    if (!value) { Invalidate(); return true; }
    shuffle_source_ = *value;
    shuffle_from_ = source;
    if (this->members.empty()) shuffle_values->insert_or_assign(key(lane),*value);
    else for (const auto member:this->members)
      shuffle_values->insert_or_assign(ShuffleKey{facts_.Warp(member),i.id,record_->occurrence(),facts_.LaneId(member),origin_.unroll_context},*value);
    auto linear = LinearThread();
    if (!shuffle_valid_) shuffle_selector_ = linear;
    else if (Has(i, "bfly")) shuffle_selector_ = binary(v::BinaryOpcode::BitXor, linear, Constant(U32(), std::to_string(amount)));
    else if (Has(i, "up") || Has(i, "down"))
      shuffle_selector_ = binary(Has(i, "up") ? v::BinaryOpcode::Subtract : v::BinaryOpcode::Add, linear, Constant(U32(), std::to_string(amount)));
    else {
      auto retained = binary(v::BinaryOpcode::BitAnd, linear, Constant(U32(), std::to_string((~31u) | segment)));
      shuffle_selector_ = binary(v::BinaryOpcode::BitOr, retained, Constant(U32(), std::to_string(amount & ~segment)));
    }
    ++publications;
    return false;
  }
  // Each member reads its own source lane: ShuffleSource is its PTX source and
  // validity. Members whose validity differs keep one cohort; the selector then
  // falls back to the member's own thread where the source is invalid.
  shuffle_from_=ShuffleSource(facts_,{lane_,cursor_}).first;
  const std::vector<LaneRef> self{lane_};
  const auto& readers = members.empty() ? self : members;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> sources;
  sources.reserve(readers.size());
  bool mixed = false;
  for (const auto member:readers) {
    const auto [source, valid] = ShuffleSource(facts_,{member,cursor_});
    if (!shuffle_values->contains({facts_.Warp(member),i.id,record_->occurrence(),source,origin_.unroll_context})) return false;
    sources.emplace_back(facts_.Warp(member), source);
    mixed |= valid != shuffle_valid_;
  }
  const auto base = SourceValue(i.id, record_->occurrence(), sources);
  if (!base) return false;
  const auto destinations = Split(module_.view(i.operands[0]), '|');
  auto result = shuffle_values->find(key(shuffle_from_))->second;
  Expression selector = *shuffle_selector_, valid = Constant(Pred(), shuffle_valid_ ? "true" : "false");
  if (mixed) {
    auto b = Known(std::string(module_.view(i.operands[2])));
    auto c = Known(std::string(module_.view(i.operands[3])));
    if (!b || !c) throw std::invalid_argument("Shuffle lane selector/clamp is dynamic");
    const auto amount = (*b & 31).convert_to<std::uint32_t>();
    const auto segment = ((*c >> 8) & 31).convert_to<std::uint32_t>();
    const auto clamp = (*c & 31).convert_to<std::uint32_t>() & ~segment;
    const auto compare = [&](v::ComparePredicate op, Expression x, Expression y) {
      return Expression{Add(std::make_unique<v::CompareOp>(op, Pred().value), {x.value, y.value}, origin_).result(), Pred()};
    };
    const auto constant = [&](std::uint32_t n) { return Constant(U32(), std::to_string(n)); };
    // PTX shfl.sync: maxLane = (lane & segmask) | (c & ~segmask); up is valid when
    // lane - b >= maxLane, the other modes when their source lane <= maxLane.
    const auto linear = LinearThread();
    const auto lane = binary(v::BinaryOpcode::BitAnd, linear, constant(31));
    const auto maximum = binary(v::BinaryOpcode::BitOr, binary(v::BinaryOpcode::BitAnd, lane, constant(segment)), constant(clamp));
    if (Has(i, "up")) valid = compare(v::ComparePredicate::GreaterEqual, lane, binary(v::BinaryOpcode::Add, maximum, constant(amount)));
    else {
      const auto source_lane = Has(i, "down") ? binary(v::BinaryOpcode::Add, lane, constant(amount)) :
          Has(i, "bfly") ? binary(v::BinaryOpcode::BitXor, lane, constant(amount)) :
          binary(v::BinaryOpcode::BitOr, binary(v::BinaryOpcode::BitAnd, lane, constant(segment)), constant(amount & ~segment));
      valid = compare(v::ComparePredicate::LessEqual, source_lane, maximum);
    }
    // The selector a valid lane uses (as the first phase builds it for a valid leader).
    Expression source_thread = Has(i, "bfly") ? binary(v::BinaryOpcode::BitXor, linear, constant(amount)) :
        Has(i, "up") ? binary(v::BinaryOpcode::Subtract, linear, constant(amount)) :
        Has(i, "down") ? binary(v::BinaryOpcode::Add, linear, constant(amount)) :
        binary(v::BinaryOpcode::BitOr, binary(v::BinaryOpcode::BitAnd, linear, constant((~31u) | segment)), constant(amount & ~segment));
    selector = Expression{Add(std::make_unique<v::SelectOp>(U32().value), {valid.value, source_thread.value, linear.value}, origin_).result(), U32()};
  }
  result.value = graph_.fromThread(*base, selector.value);
  Bind(destinations[0], result);
  if (destinations.size() == 2) Bind(destinations[1], valid);
  return true;
}

std::optional<v::Value> ValueTemplateExtractor::SourceValue(
    std::uint32_t instruction, std::uint64_t occurrence, const std::vector<std::pair<std::uint32_t, std::uint32_t>>& sources) {
  const auto cta = *origin_.cta;
  const auto warp_size = facts_.launch().warp_size();
  std::vector<std::uint64_t> threads;
  std::vector<v::Value> values;
  std::set<std::uint64_t> seen;
  for (const auto [warp, lane] : sources) {
    const auto thread = std::uint64_t(warp) * warp_size + lane;
    if (!seen.insert(thread).second) continue;
    const auto found = shuffle_values->find({warp, instruction, occurrence, lane, origin_.unroll_context});
    if (found == shuffle_values->end()) return std::nullopt;
    threads.push_back(thread);
    values.push_back(found->second.value);
  }
  if (threads.empty()) return std::nullopt;
  // A publication is its cohort's prototype at that cohort's leader: rebinding it
  // from there to thread t gives t's own value (unchanged outside its domain).
  const auto own = [&](const v::Value& value) { return graph_.instanceOrigin(value).thread.value_or(0); };
  const auto at = [&](const v::Value& value, std::uint64_t thread) { return graph_.bindThread(value, cta, own(value), thread); };
  // One producer template: every publication is the first one rebound, and its node
  // (no selector or view in the way) holds each source thread.
  const auto& first = values.front();
  bool single = !first.threadSelector() && !first.conversion();
  for (std::size_t k = 1; single && k < values.size(); ++k) single = at(values[k], own(first)) == first;
  if (single) {
    const auto& node = graph_.definingNode(first);
    for (const auto thread : threads)
      if (!node.memberIndex(thread)) { single = false; break; }
  }
  if (single) return first;
  for (std::size_t k = 0; k < values.size(); ++k) values[k] = at(values[k], threads[k]);
  return graph_.addThreadUnion(origin_, threads, std::move(values));
}

bool ValueTemplateExtractor::MatrixMemory() {
  const auto& i = *instruction_;
  const bool store = i.opcode == "stmatrix";
  const auto shape = DecodeMatrixShape(module_, i, store);
  const ShuffleKey collective{facts_.Warp(lane_),i.id,record_->occurrence(),0,origin_.unroll_context};
  auto& state=(*matrix_addresses)[collective];
  if (!matrix_pending_) {
    auto address = Address(shape.address);
    if (!address) { Invalidate(); return true; }
    for (const auto member:members) {
      state.addresses[facts_.LaneId(member)]=*address;
      state.arrived |= std::uint32_t{1} << facts_.LaneId(member);
    }
    matrix_pending_ = true;
    ++publications;
    return false;
  }
  if (state.arrived != UINT32_MAX) return false;
  const auto binary=[&](v::BinaryOpcode op,v::Value a,v::Value b) {
    return Add(std::make_unique<v::BinaryOp>(op,U32().value),{a,b},origin_).result();
  };
  const auto thread=LinearThread().value;
  const auto lane=binary(v::BinaryOpcode::Remainder,thread,Constant(U32(),"32").value);
  const auto warp_base=binary(v::BinaryOpcode::Subtract,thread,lane);
  const auto four=Constant(U32(),"4").value;
  const auto lane_row=binary(v::BinaryOpcode::Divide,lane,four);
  const auto lane_col=binary(v::BinaryOpcode::Remainder,lane,four);
  const auto text = Trim(module_.view(i.operands[store ? 1 : 0]));
  const auto names = Split(std::string_view(text).substr(1, text.size()-2), ',');
  for (unsigned matrix = 0; matrix < shape.count; ++matrix) {
    std::optional<Expression> data;
    if (store) { data = Operand(names[matrix], *PtxType("b32")); if (!data) return true; }
    std::vector<Expression> parts;
    for (unsigned half = 0; half < (shape.transpose ? 2u : 1u); ++half) {
      const unsigned peer = 8*matrix + (shape.transpose ? 2*(facts_.LaneId(lane_)%4)+half : facts_.LaneId(lane_)/4);
      auto address = *state.addresses[peer];
      const auto relative=shape.transpose ? binary(v::BinaryOpcode::Multiply,lane_col,Constant(U32(),"2").value) : lane_row;
      const auto selector=binary(v::BinaryOpcode::Add,warp_base,
          binary(v::BinaryOpcode::Add,relative,Constant(U32(),std::to_string(8*matrix+half)).value));
      address.value = graph_.fromThread(address.value, selector);
      auto offset_value=binary(v::BinaryOpcode::Multiply,shape.transpose ? lane_row : lane_col,
          Constant(U32(),shape.transpose ? "2" : "4").value);
      if (address.type.value != U32().value) offset_value=graph_.convert(offset_value,address.type.value,{});
      const Expression offset{offset_value,address.type};
      address.value = Operation("ptx.address_add", "byte offset", {address.type}, {address, offset}).result();
      const auto type = *PtxType(shape.transpose ? "b16" : "b32");
      if (store) {
        auto part = shape.transpose ? Part(*data, type, 16*half) : Reinterpret(*data, type);
        if (!part) return true;
        Add(std::make_unique<v::StoreOp>(), {address.value, part->value}, origin_);
      } else parts.emplace_back(Add(std::make_unique<v::LoadOp>(type.value), {address.value}, origin_).result(), type);
    }
    if (!store) Bind(names[matrix], shape.transpose ? Expression{{}, *PtxType("b32"), std::move(parts)} : parts[0]);
  }
  for (const auto member:members) state.consumed |= std::uint32_t{1} << facts_.LaneId(member);
  if (state.consumed == UINT32_MAX) matrix_addresses->erase(collective);
  matrix_pending_ = false;
  return true;
}

bool ValueTemplateExtractor::WarpMma() {
  const auto& i=*instruction_;
  const auto element=i.modifiers.size()==9 ? i.modifiers[6] : std::string();
  const bool eight=element=="e4m3" || element=="e5m2";
  const std::vector<std::string> expected{"sync","aligned",eight ? "m16n8k32" : "m16n8k16","row","col","f32",
      element,element,"f32"};
  if (!In(element,{"f16","bf16","e4m3","e5m2"}) || i.modifiers!=expected || i.operands.size()!=4)
    throw std::invalid_argument("Unsupported warp MMA shape, layout or arithmetic type");
  const auto names=[&](std::size_t operand,std::size_t count) {
    const auto text=Trim(module_.view(i.operands.at(operand)));
    if (!text.starts_with('{') || !text.ends_with('}')) throw std::invalid_argument("Warp MMA needs register tuples");
    auto result=Split(std::string_view(text).substr(1,text.size()-2),',');
    if (result.size()!=count) throw std::invalid_argument("Warp MMA register tuple width mismatch");
    return result;
  };
  const auto destinations=names(0,4);
  const ShuffleKey collective{facts_.Warp(lane_),i.id,record_->occurrence(),0,origin_.unroll_context};
  auto& state=(*warp_mma_inputs)[collective];
  if (!warp_mma_pending_) {
    std::vector<v::Value> values;
    for (unsigned operand=1;operand<4;++operand)
      for (const auto& name:names(operand,operand==2 ? 2 : 4)) {
        const auto type=*PtxType(operand==3 ? "f32" : "b32");
        auto value=Operand(name,type);
        if (value) value=Reinterpret(*value,type);
        if (!value) { Invalidate(); return true; }
        values.push_back(value->value);
      }
    for (const auto member:members) {
      state.inputs[facts_.LaneId(member)]=values;
      auto origin=origin_;
      const auto thread=SetThread(origin,facts_,member);
      for (auto& value:state.inputs[facts_.LaneId(member)])
        value=graph_.bindThread(value,cta_,*origin_.thread,thread);
      state.origins[facts_.LaneId(member)]=origin;
      state.arrived |= std::uint32_t{1} << facts_.LaneId(member);
    }
    warp_mma_pending_=true;
    ++publications;
    return false;
  }
  if (state.arrived != UINT32_MAX) return false;
  if (!state.result) {
    // One numerical matrix per collective; memory occurrences remain per thread.
    std::array<v::Value,32> selectors;
    auto constant_origin=origin_;
    constant_origin.thread.reset(); constant_origin.warp.reset(); constant_origin.lane.reset();
    constant_origin.thread_coordinates.reset();
    for (unsigned lane=0;lane<32;++lane)
      selectors[lane]=graph_.add(std::make_unique<v::ConstantOp>(U32().value,
          std::to_string(facts_.Warp(lane_)*32+lane)),{},constant_origin).result();
    std::vector<v::Value> operands;
    operands.reserve(320);
    for (unsigned role=0;role<3;++role) {
      const unsigned begin=role==0 ? 0 : role==1 ? 4 : 6;
      const unsigned count=role==1 ? 2 : 4;
      for (unsigned lane=0;lane<32;++lane) {
        const auto& values=state.inputs[lane];
        for (unsigned reg=0;reg<count;++reg)
          operands.push_back(graph_.fromThread(values[begin+reg],selectors[lane]));
      }
    }
    state.result=graph_.addCollective(std::make_unique<v::WarpMmaOp>(element),std::move(operands),
        {state.origins.begin(),state.origins.end()}).result();
    for (unsigned index=0;index<state.indices.size();++index)
      state.indices[index]=facts_.Warp(lane_)==0 ? selectors[index] :
          graph_.add(std::make_unique<v::ConstantOp>(U32().value,std::to_string(index)),{},constant_origin).result();
    for (auto& values:state.inputs) std::vector<v::Value>().swap(values);
  }
  const auto binary=[&](v::BinaryOpcode op,v::Value a,unsigned b) {
    return Add(std::make_unique<v::BinaryOp>(op,U32().value),{a,Constant(U32(),std::to_string(b)).value},origin_).result();
  };
  const auto lane=binary(v::BinaryOpcode::Remainder,LinearThread().value,32);
  for (unsigned reg=0;reg<4;++reg) {
    const auto row=binary(v::BinaryOpcode::Add,binary(v::BinaryOpcode::Divide,lane,4),(reg/2)*8);
    const auto col=binary(v::BinaryOpcode::Add,binary(v::BinaryOpcode::Multiply,binary(v::BinaryOpcode::Remainder,lane,4),2),reg%2);
    const auto result=graph_.node(state.result->nodeId()).result(0,facts_.LaneId(lane_));
    Bind(destinations[reg],{graph_.component(result,{row,col}),*PtxType("f32")});
  }
  for (const auto member:members) state.consumed |= std::uint32_t{1} << facts_.LaneId(member);
  if (state.consumed == UINT32_MAX) warp_mma_inputs->erase(collective);
  warp_mma_pending_=false;
  return true;
}

// vote.sync.{ballot.b32,any.pred,all.pred,uni.pred} d, {!}a, membermask and
// redux.sync.op.type d, a, membermask: each member lane publishes its operand as
// for shfl; once every lane of the membermask has published, each member reads
// lane j's operand as an idx shuffle from lane j (selector (thread & ~31) | j) and
// combines them. A lane that never publishes leaves its readers unresolved.
// Partition keys the members by the membermask's value (BuildPartitionSpec); the
// sources, from whichever cohorts published them, are one SourceValue.
// redux.sync (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-redux-sync)
// "performs a reduction operation .op of the 32 bit source register src across all
// non-exited threads in the membermask"; "For the .add operation result is truncated
// to 32 bits" (the type's wrapping Add). For .f32, "+0.0 > -0.0", .abs reduces the
// inputs' absolute values, and with .NaN "the result of the reduction operation is
// canonical NaN if the input to the reduction operation from any participating thread
// is NaN": exactly min.NaN/max.NaN (propagate_nan), associative and commutative on that
// order, so a balanced tree over the lanes is exact; a single lane is reduced with
// itself, which makes its NaN canonical. Without .NaN "the result will be canonical NaN
// when all inputs are NaNs", which plain Minimum/Maximum do not state: refused.
bool ValueTemplateExtractor::LaneReduction() {
  const auto& i = *instruction_;
  const bool redux = i.opcode == "redux";
  const auto mode = i.modifiers.size() >= 3 ? i.modifiers[1] : std::string();
  const bool ballot = !redux && mode == "ballot";
  std::optional<Type> type;
  bool absolute = false, propagate_nan = false;
  if (redux) {
    std::size_t at = 2;
    if (at < i.modifiers.size() && i.modifiers[at] == "abs") { absolute = true; ++at; }
    if (at < i.modifiers.size() && i.modifiers[at] == "NaN") { propagate_nan = true; ++at; }
    const auto spelling = at + 1 == i.modifiers.size() ? i.modifiers[at] : std::string();
    const bool floating = spelling == "f32";
    if (i.operands.size() != 3 || i.modifiers.empty() || i.modifiers[0] != "sync" ||
        !((In(mode, {"add", "min", "max"}) && In(spelling, {"u32", "s32"})) ||
          (In(mode, {"and", "or", "xor"}) && spelling == "b32") || (In(mode, {"min", "max"}) && floating)) ||
        (!floating && (absolute || propagate_nan)))
      throw std::invalid_argument("Unsupported redux form");
    if (floating && !propagate_nan)
      throw std::invalid_argument("redux .f32 without .NaN has no exact value model (canonical NaN when every input is NaN)");
    type = PtxType(spelling);
  } else if (i.operands.size() != 3 || i.modifiers.size() != 3 || i.modifiers[0] != "sync" ||
             !In(mode, {"ballot", "any", "all", "uni"}) || i.modifiers[2] != (ballot ? "b32" : "pred"))
    throw std::invalid_argument("Unsupported vote form");
  const auto raw = Known(std::string(module_.view(i.operands[2])));
  if (!raw || *raw < -(Integer(1) << 31) || *raw > UINT32_MAX || !(*raw & 0xffffffff))
    throw std::invalid_argument(std::string(redux ? "Redux" : "Vote") + " membermask is invalid or not statically determined");
  const auto mask = (*raw & 0xffffffff).convert_to<std::uint32_t>();
  if (!reduction_pending_) {
    auto operand = Operand(std::string(module_.view(i.operands[1])), type ? *type : Pred());
    if (operand && type) operand = Reinterpret(*operand, *type);
    if (!operand) { Invalidate(); return true; }
    for (const auto member:members) {
      const auto lane = facts_.LaneId(member);
      if (lane >= 32 || !(mask & (std::uint32_t{1} << lane)))
        throw std::invalid_argument(std::string(redux ? "Redux" : "Vote") + " issuer is outside the membermask");
      shuffle_values->insert_or_assign(ShuffleKey{facts_.Warp(member),i.id,record_->occurrence(),lane,origin_.unroll_context},*operand);
    }
    reduction_pending_ = true;
    ++publications;
    return false;
  }
  std::vector<std::pair<std::uint32_t, std::uint32_t>> sources;
  for (const auto member:members)
    for (unsigned lane = 0; lane < 32; ++lane)
      if (mask & (std::uint32_t{1} << lane)) {
        if (!shuffle_values->contains({facts_.Warp(member),i.id,record_->occurrence(),lane,origin_.unroll_context})) return false;
        sources.emplace_back(facts_.Warp(member), lane);
      }
  const auto base = SourceValue(i.id, record_->occurrence(), sources);
  if (!base) return false;
  const auto binary = [&](v::BinaryOpcode op, const Type& type, v::Value a, v::Value b) {
    return Add(std::make_unique<v::BinaryOp>(op, type.value), {a, b}, origin_).result();
  };
  const auto warp_base = binary(v::BinaryOpcode::BitAnd, U32(), LinearThread().value, Constant(U32(), std::to_string(~31u)).value);
  std::optional<v::Value> any, all, none;
  std::vector<v::Value> lanes;
  for (unsigned lane = 0; lane < 32; ++lane) {
    if (!(mask & (std::uint32_t{1} << lane))) continue;
    const auto selector = binary(v::BinaryOpcode::BitOr, U32(), warp_base, Constant(U32(), std::to_string(lane)).value);
    const auto value = graph_.fromThread(*base, selector);
    if (redux) { lanes.push_back(value); continue; }
    if (ballot) {
      const auto bit = Add(std::make_unique<v::SelectOp>(U32().value), {value,
          Constant(U32(), std::to_string(std::uint32_t{1} << lane)).value, Constant(U32(), "0").value}, origin_).result();
      any = any ? binary(v::BinaryOpcode::BitOr, U32(), *any, bit) : bit;
      continue;
    }
    any = any ? binary(v::BinaryOpcode::BitOr, Pred(), *any, value) : value;
    all = all ? binary(v::BinaryOpcode::BitAnd, Pred(), *all, value) : value;
    if (mode == "uni") {
      const auto negated = Add(std::make_unique<v::UnaryOp>(v::UnaryOpcode::BitNot, Pred().value), {value}, origin_).result();
      none = none ? binary(v::BinaryOpcode::BitAnd, Pred(), *none, negated) : negated;
    }
  }
  const auto destination = Trim(module_.view(i.operands[0]));
  if (redux) {
    const auto opcode = mode == "add" ? v::BinaryOpcode::Add : mode == "min" ? v::BinaryOpcode::Minimum :
        mode == "max" ? v::BinaryOpcode::Maximum : mode == "and" ? v::BinaryOpcode::BitAnd :
        mode == "or" ? v::BinaryOpcode::BitOr : v::BinaryOpcode::BitXor;
    v::ArithmeticAttributes attributes;
    attributes.propagate_nan = propagate_nan;
    if (absolute)
      for (auto& value : lanes)
        value = Add(std::make_unique<v::UnaryOp>(v::UnaryOpcode::Absolute, type->value), {value}, origin_).result();
    if (lanes.size() == 1 && propagate_nan) lanes.push_back(lanes.front());
    // Pairs of adjacent lanes, then pairs of those results: log2 of the lane count levels.
    while (lanes.size() > 1) {
      std::vector<v::Value> level;
      for (std::size_t k = 0; k + 1 < lanes.size(); k += 2)
        level.push_back(Add(std::make_unique<v::BinaryOp>(opcode, type->value, attributes),
                            {lanes[k], lanes[k + 1]}, origin_).result());
      if (lanes.size() % 2) level.push_back(lanes.back());
      lanes = std::move(level);
    }
    Bind(destination, Expression{lanes.front(), *type});
  } else if (ballot) {
    auto result = Reinterpret(Expression{*any, U32()}, *PtxType("b32"));
    if (!result) { Invalidate(); return true; }
    Bind(destination, *result);
  } else {
    // The issuing lane's own term in front of each chain (ArithmeticAttributes::lane_own): it is one
    // of the chain's terms, so the value is the chain's; the scalar fold takes it as the initial of
    // one reduction over the lanes. O(1) nodes per vote.
    v::ArithmeticAttributes own_term;
    own_term.lane_own = true;
    const auto own = graph_.fromThread(*base, LinearThread().value);
    const auto with_own = [&](v::BinaryOpcode op, v::Value mine, v::Value chain) {
      return Add(std::make_unique<v::BinaryOp>(op, Pred().value, own_term), {mine, chain}, origin_).result();
    };
    v::Value result = *any;
    if (mode == "any") result = with_own(v::BinaryOpcode::BitOr, own, *any);
    else if (mode == "all") result = with_own(v::BinaryOpcode::BitAnd, own, *all);
    else {
      const auto not_own = Add(std::make_unique<v::UnaryOp>(v::UnaryOpcode::BitNot, Pred().value), {own}, origin_).result();
      result = binary(v::BinaryOpcode::BitOr, Pred(), with_own(v::BinaryOpcode::BitAnd, own, *all),
                      with_own(v::BinaryOpcode::BitAnd, not_own, *none));
    }
    Bind(destination, Expression{result, Pred()});
  }
  return true;
}

} // namespace
} // namespace sentryward::frontend::ptx
