#include "sentryward/frontend/ptx/tensor_maps.h"
#include "sentryward/frontend/ptx/resources.h"

namespace sentryward::frontend::ptx {
namespace {
[[noreturn]] void Fail(const std::string &message) {
  throw ExecutionError("nonconstant_tma_descriptor",0,message);
}
Scalar Operand(const ExecutionProgram &program,const ExecutedInstruction &event,std::string text) {
  text=Trim(text); if (text.starts_with('[') && text.ends_with(']')) text=Trim(std::string_view(text).substr(1,text.size()-2));
  auto split=SplitAddress(text); if (!split) return {};
  Scalar value;
  if (auto known=program.KnownRegister(event,split->first)) value=*known;
  else if (program.config.parameters.contains(split->first)) value=program.config.parameters.at(split->first);
  else if (auto number=ParseInteger(split->first)) value=*number;
  if (auto pointer=std::get_if<AllocationPointer>(&value)) { pointer->offset+=split->second; return value; }
  if (auto number=ScalarInteger(value)) return Integer(*number+split->second);
  return {};
}
}
nlohmann::json DecodeTensorMap(const TensorMapFields &fields,const AllocationPointer &destination) {
  auto value=[&](std::string field,std::optional<std::int64_t> index={}) -> Scalar {
    auto found=fields.find({field,index}); return found==fields.end()?Scalar{}:found->second;
  };
  auto number=[&](std::string field,std::optional<std::int64_t> index={}) -> std::int64_t {
    auto n=ScalarInteger(value(field,index));
    if (!n || *n<INT64_MIN || *n>INT64_MAX) Fail("runtime tensor map has unresolved "+field);
    return n->convert_to<std::int64_t>();
  };
  // A partial update need not supply the descriptor's structural fields.
  const auto rank_value=ScalarInteger(value("rank"));
  bool structural=rank_value && *rank_value>=0 && *rank_value<5;
  for (const auto *field : {"elemtype","interleave_layout","fill_mode","swizzle_mode"})
    structural=structural && ScalarInteger(value(field)).has_value();
  if (structural)
    for (std::int64_t axis=0;axis<=rank_value->convert_to<std::int64_t>();++axis)
      structural=structural && ScalarInteger(value("box_dim",axis)).has_value() &&
                 ScalarInteger(value("element_stride",axis)).has_value();
  if (!structural) {
    nlohmann::json updates=nlohmann::json::array();
    for (const auto &[key,v] : fields) {
      nlohmann::json encoded=nullptr;
      if (auto n=ScalarInteger(v)) encoded=n->convert_to<std::string>();
      else if (auto p=std::get_if<AllocationPointer>(&v))
        encoded={{"resource",p->resource},{"offset",p->offset.convert_to<std::string>()}};
      updates.push_back({{"field",key.first},{"index",key.second?nlohmann::json(*key.second):nlohmann::json(nullptr)},
                         {"value",encoded}});
    }
    return {{"runtime_updates",updates},{"partial_descriptor",true},
            {"allocation_location",{destination.resource,Unsigned(destination.offset,"tensor map location")}}};
  }
  const auto rank=number("rank")+1;
  const auto global=value("global_address");
  const auto *pointer=std::get_if<AllocationPointer>(&global);
  if (rank<1 || rank>5) Fail("runtime tensor map rank is invalid");
  if (number("elemtype")!=6 || number("interleave_layout")!=0 || number("fill_mode")!=0)
    Fail("runtime tensor map requires f16, no interleave and zero fill");
  const auto swizzle=number("swizzle_mode");
  if (swizzle<0 || swizzle>3) Fail("unsupported runtime tensor map swizzle");
  auto vector=[&](const char *name,std::int64_t count,bool abstract=false) {
    nlohmann::json result=nlohmann::json::array();
    for (std::int64_t k=0;k<count;++k) {
      if (abstract && !ScalarInteger(value(name,k))) { result.push_back(nullptr); continue; }
      const auto n=number(name,k); if (n<=0) Fail("invalid tensor map dimension"); result.push_back(n);
    }
    return result;
  };
  const auto box=vector("box_dim",rank);
  Integer bytes=2; for (const auto &n : box) bytes*=n.get<std::int64_t>();
  const auto offset=pointer?nlohmann::json(Unsigned(pointer->offset,"tensor map base")):nlohmann::json(nullptr);
  auto strides=vector("global_stride",rank-1,true); strides.insert(strides.begin(),2);
  return {{"resource",pointer?pointer->resource:"unresolved"},{"base_byte_offset",offset},{"rank",rank},{"element_bytes",2},
      {"symbolic_global",!pointer || std::any_of(strides.begin(),strides.end(),[](const auto &v){return v.is_null();})},
      {"global_shape",vector("global_dim",rank,true)},{"coordinate_strides_bytes",strides},
      {"box_shape",box},{"element_strides",vector("element_stride",rank)},
      {"transfer_bytes",Unsigned(bytes,"tensor map payload")},{"swizzle",std::array{"none","32B","64B","128B"}[swizzle]},
      {"interleave","none"},{"oob_fill","zero"},
      {"allocation_location",{destination.resource,Unsigned(destination.offset,"tensor map location")}}};
}
 nlohmann::json ResolveTensorMap(const nlohmann::json &descriptor,unsigned rank) {
  if (!descriptor.value("partial_descriptor",false)) return descriptor;
  TensorMapFields fields;
  for (const auto &update : descriptor.value("runtime_updates",nlohmann::json::array())) {
    std::optional<std::int64_t> axis;
    if (!update.at("index").is_null()) axis=update.at("index").get<std::int64_t>();
    const auto &v=update.at("value"); Scalar value;
    if (v.is_string()) value=Integer(v.get<std::string>());
    else if (v.is_object()) value=AllocationPointer{v.at("resource").get<std::string>(),Integer(v.at("offset").get<std::string>()),{}};
    fields[{update.at("field").get<std::string>(),axis}]=value;
  }
  if (!fields.contains({"rank",{}})) fields[{"rank",{}}]=Integer(rank-1);
  if (descriptor.contains("initial_descriptor")) {
    auto result=descriptor.at("initial_descriptor");
    result.erase("parameter_location");
    result["allocation_location"]=descriptor.at("allocation_location");
    for (const auto &update : descriptor.at("runtime_updates")) {
      const auto name=update.at("field").get<std::string>();
      const auto &v=update.at("value");
      if (name=="global_address") {
        result["resource"]=v.is_object()?v.at("resource"):nlohmann::json("unresolved");
        result["base_byte_offset"]=v.is_object()?nlohmann::json(Unsigned(Integer(v.at("offset").get<std::string>()),"tensor map base")):nlohmann::json(nullptr);
        result["symbolic_global"]=!v.is_object();
      } else if (name=="global_dim" || name=="global_stride") {
        const auto axis=update.at("index").get<unsigned>();
        if (axis+(name=="global_stride")>=rank) continue;
        auto number=v.is_null()?nlohmann::json(nullptr):nlohmann::json(Unsigned(Integer(v.get<std::string>()),"tensor map dimension"));
        if (name=="global_dim" && axis==0 && result.value("packed_element_bits",0)==4 && !number.is_null()) {
          const auto n=number.get<std::uint64_t>();
          if (n%2) throw SemanticError("nonconstant_tma_descriptor","packed tensor map dimension must contain whole bytes");
          number=n/2;
        }
        result[name=="global_dim"?"global_shape":"coordinate_strides_bytes"][axis+(name=="global_stride")]=number;
      } else {
        throw SemanticError("unsupported_tma_descriptor_update","initial descriptor update not implemented: "+name);
      }
    }
    return result;
  }
  std::string missing;
  auto require=[&](const char *name,std::optional<std::int64_t> axis={}) {
    if (!ScalarInteger(fields[{name,axis}])) {
      if (!missing.empty()) missing+=", ";
      missing+=name; if (axis) missing+="["+std::to_string(*axis)+"]";
    }
  };
  for (const auto *name : {"rank","elemtype","interleave_layout","fill_mode","swizzle_mode"}) require(name);
  for (unsigned axis=0;axis<rank;++axis) { require("box_dim",axis); require("element_stride",axis); }
  if (!missing.empty()) throw SemanticError("missing_tma_descriptor","TMA descriptor lacks structural fields: "+missing);
  if (ScalarInteger(fields.at({"rank",{}}))!=Integer(rank-1))
    throw SemanticError("tma_opcode_rank","descriptor rank disagrees with opcode");
  const auto &location=descriptor.at("allocation_location");
  return DecodeTensorMap(fields,AllocationPointer{location.at(0).get<std::string>(),Integer(location.at(1).get<std::uint64_t>()),{}});
}
void PublishTensorMap(nlohmann::json &maps,const nlohmann::json &decoded) {
  for (auto it=maps.begin();it!=maps.end();++it)
    if (it.value().value("allocation_location",nlohmann::json())==decoded.at("allocation_location")) {
      if (it.value()!=decoded) Fail("runtime tensor map changes at a reused global address");
      return;
    }
  maps["runtime."+std::to_string(maps.size())]=decoded;
}
void RecoverTensorMaps(ExecutionProgram &program) {
  if (std::none_of(program.function->instructions.begin(),program.function->instructions.end(),
                   [](const auto &i){return i.opcode=="tensormap";})) return;
  using Copy=std::tuple<Scalar,Scalar,Scalar>;
  std::map<std::pair<std::uint32_t,DynamicKey>,std::map<std::uint32_t,Copy>> copies;
  for (const auto &lane : program.lane_traces) {
    std::map<Integer,TensorMapFields> fields;
    for (std::size_t index=0;index<lane.instructions.size();++index) {
      const auto &metadata=lane.instructions.metadata(index);
      const auto &i=program.function->instructions[metadata.instruction_id()];
      if (i.opcode!="tensormap") continue;
      const auto event=lane.instructions.read(index);
      const auto ops=OperandTexts(*program.module,i);
      auto value=[&](std::size_t k){return Operand(program,event,ops.at(k));};
      if (Has(i,"replace") && Has(i,"tile")) {
        const auto address=ScalarInteger(value(0));
        if (!address || i.modifiers.size()<3) Fail("runtime tensor map shared address unresolved");
        std::optional<std::int64_t> ordinal;
        if (ops.size()==3) ordinal=Unsigned(ScalarInteger(value(1)).value_or(Integer(-1)),"tensor map ordinal",4);
        fields[*address][{i.modifiers[2],ordinal}]=value(ops.size()-1);
      } else if (Has(i,"cp_fenceproxy")) {
        const auto destination=value(0), source=value(1), size=value(2);
        copies[{lane.warp_id,event.Key()}][lane.lane_id]={source,destination,size};
        const auto address=ScalarInteger(source);
        if (!address || !fields.contains(*address)) continue;
        const auto *pointer=std::get_if<AllocationPointer>(&destination);
        if (!pointer) Fail("runtime tensor map destination has no allocation");
        auto decoded=DecodeTensorMap(fields.at(*address),*pointer);
        for (const auto &existing : program.config.tma_descriptors)
          if (existing.value("allocation_location",nlohmann::json())==decoded.at("allocation_location") && existing.contains("initial_descriptor"))
            decoded["initial_descriptor"]=existing.at("initial_descriptor");
        PublishTensorMap(program.config.tma_descriptors,decoded);
      }
    }
  }
  for (const auto &[key,lanes] : copies) {
    (void)key;
    if (lanes.size()!=32) Fail("tensor map copy requires all 32 lanes");
    const auto &first=lanes.begin()->second;
    for (unsigned lane=0;lane<32;++lane)
      if (!lanes.contains(lane) || lanes.at(lane)!=first) Fail("tensor map copy requires identical operands");
    if (!ScalarInteger(std::get<0>(first)) || !std::holds_alternative<AllocationPointer>(std::get<1>(first)) ||
        ScalarInteger(std::get<2>(first))!=Integer(128)) Fail("tensor map copy has invalid operands or size");
  }
}
}
