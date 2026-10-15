#include "sentryward/equivalence/comparison_context.h"
#include "sentryward/equivalence/input_mapping.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace sentryward::equivalence {
namespace {
using Json=nlohmann::json;
[[noreturn]] void invalid(const char* reason) {
  throw std::invalid_argument(std::string("comparison_context_")+reason);
}
class Sha256 {
 public:
  void update(const char* bytes,std::size_t count) {
    if(count>UINT64_MAX/8-total_)invalid("hash_input_too_large");
    total_+=count;
    while(count) {
      const auto n=std::min(count,block_.size()-used_);
      std::copy_n(reinterpret_cast<const unsigned char*>(bytes),n,block_.begin()+used_);
      bytes+=n;count-=n;used_+=n;
      if(used_==block_.size()){compress();used_=0;}
    }
  }
  std::string finish() {
    const auto bits=total_*8;
    block_[used_++]=0x80;
    if(used_>56){std::fill(block_.begin()+used_,block_.end(),0);compress();used_=0;}
    std::fill(block_.begin()+used_,block_.begin()+56,0);
    for(unsigned i=0;i<8;++i)block_[63-i]=static_cast<unsigned char>(bits>>(8*i));
    compress();
    constexpr char hex[]="0123456789abcdef";
    std::string result;result.reserve(64);
    for(auto word:state_)for(int shift=28;shift>=0;shift-=4)result+=hex[(word>>shift)&15];
    return result;
  }
 private:
  static std::uint32_t rotate(std::uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));}
  void compress() {
    static constexpr std::array<std::uint32_t,64> k{
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<std::uint32_t,64> w{};
    for(unsigned i=0;i<16;++i)for(unsigned j=0;j<4;++j)w[i]=(w[i]<<8)|block_[4*i+j];
    for(unsigned i=16;i<64;++i) {
      const auto a=w[i-15],b=w[i-2];
      w[i]=w[i-16]+(rotate(a,7)^rotate(a,18)^(a>>3))+w[i-7]+(rotate(b,17)^rotate(b,19)^(b>>10));
    }
    auto [a,b,c,d,e,f,g,h]=state_;
    for(unsigned i=0;i<64;++i) {
      const auto first=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+k[i]+w[i];
      const auto second=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
      h=g;g=f;f=e;e=d+first;d=c;c=b;b=a;a=first+second;
    }
    const std::array<std::uint32_t,8> added{a,b,c,d,e,f,g,h};
    for(unsigned i=0;i<8;++i)state_[i]+=added[i];
  }
  std::array<std::uint32_t,8> state_{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
      0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  std::array<unsigned char,64> block_{};
  std::size_t used_=0;
  std::uint64_t total_=0;
};
std::ifstream open(const std::string& path) {
  std::ifstream file(path,std::ios::binary);if(!file)invalid("source_unreadable");return file;
}
void object(const Json& j,const std::set<std::string>& keys) {
  if(!j.is_object())invalid("invalid_object");
  for(auto i=j.begin();i!=j.end();++i)if(!keys.contains(i.key()))invalid("unknown_field");
}
std::uint64_t natural(const Json& j) {
  if(!j.is_number_integer() || (!j.is_number_unsigned() && j.get<std::int64_t>()<0))invalid("invalid_integer");
  return j.get<std::uint64_t>();
}
std::string hash(const Json& j) {
  if(!j.is_string())invalid("invalid_hash");
  auto text=j.get<std::string>();
  if(text.size()!=64 || text.find_first_not_of("0123456789abcdef")!=std::string::npos)invalid("invalid_hash");
  return text;
}
std::uint64_t add(std::uint64_t a,std::uint64_t b) {
  if(b>UINT64_MAX-a)invalid("offset_overflow");return a+b;
}
Json boundedJson(const std::string& path,std::string& digest) {
  constexpr std::size_t maximum=8*1024*1024;
  auto file=open(path);std::string bytes;std::array<char,65536> buffer{};Sha256 sha;
  while(file) {
    file.read(buffer.data(),buffer.size());const auto count=static_cast<std::size_t>(file.gcount());
    if(count>maximum-bytes.size())invalid("metadata_size_cap");
    bytes.append(buffer.data(),count);sha.update(buffer.data(),count);
  }
  if(file.bad() || !file.eof())invalid("source_read_failed");digest=sha.finish();return Json::parse(bytes);
}
std::string integerLiteral(const Json& j,const std::string& type) {
  if(!j.is_number_integer())invalid("invalid_cell_value");
  if(type=="s32" || type=="s64") {
    if(j.is_number_unsigned() && j.get<std::uint64_t>()>std::uint64_t(INT64_MAX))invalid("cell_value_out_of_range");
    const auto value=j.get<std::int64_t>();
    if(type=="s32" && (value<INT32_MIN || value>INT32_MAX))invalid("cell_value_out_of_range");
    return std::to_string(value);
  }
  if(type!="u32" && type!="u64")invalid("unsupported_cell_type");
  const auto value=natural(j);
  if(type=="u32" && value>UINT32_MAX)invalid("cell_value_out_of_range");
  return std::to_string(value);
}
std::string globalIdentity(const std::string& object) {
  return "6:global"+std::to_string(object.size())+":"+object;
}
} // namespace

std::string comparisonSha256(std::istream& file) {
  Sha256 sha;std::array<char,65536> buffer{};
  while(file){file.read(buffer.data(),buffer.size());sha.update(buffer.data(),static_cast<std::size_t>(file.gcount()));}
  if(file.bad() || !file.eof())invalid("source_read_failed");return sha.finish();
}
std::string comparisonFileSha256(const std::string& path) {auto file=open(path);return comparisonSha256(file);}

std::shared_ptr<const ComparisonContext> ComparisonContext::load(
    const std::string& path,const ComparisonContextFiles& files,const InputIdentityMapping& right_mapping) {
  auto result=std::shared_ptr<ComparisonContext>(new ComparisonContext);
  const auto spec=boundedJson(path,result->digest_);
  object(spec,{"schema","sources"});
  if(spec.at("schema")!="spverify.comparison-context.v1")invalid("unsupported_schema");
  const auto& sources=spec.at("sources");
  object(sources,{"input","report","provenance","left","right","mapping"});
  for(const auto& [name,source]:std::array<std::pair<const char*,std::string>,3>{
        {{"left",files.left},{"right",files.right},{"mapping",files.mapping}}}) {
    const auto& entry=sources.at(name);object(entry,{"sha256"});
    const auto expected=hash(entry.at("sha256"));
    if(comparisonFileSha256(source)!=expected)invalid("artifact_hash_mismatch");
    result->source_hashes_.emplace(name,expected);
  }
  std::map<std::string,Json> metadata;
  for(const auto* name:{"input","report","provenance"}) {
    const auto& entry=sources.at(name);object(entry,{"path","sha256"});
    const auto expected=hash(entry.at("sha256"));
    auto source=entry.at("path").get<std::string>();if(source.empty())invalid("empty_source_path");
    auto resolved=std::filesystem::path(source);
    if(resolved.is_relative())resolved=std::filesystem::path(path).parent_path()/resolved;
    std::string actual;auto j=boundedJson(resolved.string(),actual);
    if(actual!=expected)invalid("scope_hash_mismatch");
    result->source_hashes_.emplace(name,actual);metadata.emplace(name,std::move(j));
  }
  const auto& input=metadata.at("input");const auto& report=metadata.at("report");const auto& provenance=metadata.at("provenance");
  if(provenance.at("optimized_input_sha256")!=result->source_hashes_.at("input") ||
      provenance.at("scope_shape_arrays")!=input.at("shape_arrays"))invalid("provenance_scope_mismatch");
  if(report.value("status","")!="race_free" || !report.value("memory_analysis_complete",false))invalid("readonly_validation_missing");
  const auto& front=report.at("frontend");
  const auto& contract=front.at("frontend_assumptions").at("shape_input_contract");
  if(report.contains("frontend_assumptions") && report.at("frontend_assumptions").contains("shape_input_contract") &&
      report.at("frontend_assumptions").at("shape_input_contract")!=contract)invalid("conflicting_readonly_contracts");
  if(contract.value("kind","")!="fixed_read_only" || contract.value("origin","")!="shape_arrays")invalid("readonly_validation_missing");
  const auto& regions=contract.at("regions");const auto& arrays=input.at("shape_arrays");
  const auto& declarations=input.at("declarations");const auto& objects=front.at("input_objects");
  if(!regions.is_array() || !arrays.is_array() || arrays.empty() || !declarations.is_array() || !objects.is_array())invalid("invalid_scope");
  constexpr std::size_t maximum_cells=4096,maximum_records=4096;
  if(arrays.size()>maximum_records || declarations.size()>maximum_records || objects.size()>maximum_records || regions.size()>maximum_records)invalid("record_cap");
  if(right_mapping.allocations.size()>maximum_records || right_mapping.parameters.size()>maximum_records ||
      right_mapping.layouts.size()>maximum_records || right_mapping.parameter_constants.size()>maximum_records)invalid("mapping_record_cap");
  for(const auto& [source,target]:right_mapping.allocations)
    if(source.size()>8192 || target.size()>4096)invalid("identity_size_cap");
  for(const auto& [name,layout]:right_mapping.layouts) {
    (void)name;if(layout.shape.size()>64 || layout.source_strides_bytes.size()>64 || layout.target_strides_bytes.size()>64)invalid("layout_rank_cap");
  }
  std::map<std::pair<std::uint64_t,std::uint64_t>,const Json*> pointer_declarations;
  for(const auto& declaration:declarations)if(declaration.at("parameter").is_number_integer()) {
    const auto key=std::pair{natural(declaration.at("parameter")),natural(declaration.value("offset",Json(0)))};
    if(!pointer_declarations.emplace(key,&declaration).second)invalid("duplicate_pointer_declaration");
  }
  std::set<std::pair<std::string,std::uint64_t>> recorded_objects;
  for(const auto& object:objects)if(natural(object.at("bytes"))==8 && object.value("source","")=="declaration")
    recorded_objects.emplace(object.at("object").get<std::string>(),natural(object.value("offset",Json(0))));
  std::map<std::string,const Json*> by_resource;
  for(const auto& region:regions)
    if(!by_resource.emplace(region.at("resource").get<std::string>(),&region).second)invalid("duplicate_readonly_region");
  scalar_fold::Artifact scope;
  std::vector<std::pair<std::string,std::string>> values;
  std::set<std::string> supplied_resources;
  for(const auto& array:arrays) {
    // Named selectors require the original parsed parameter table; do not
    // guess their parameter number or object from allocation spelling.
    const auto parameter=natural(array.at("parameter"));
    const auto field_offset=natural(array.value("offset",Json(0)));
    const auto resource="input.arg"+std::to_string(parameter)+"."+std::to_string(field_offset);
    if(!supplied_resources.insert(resource).second)invalid("duplicate_shape_resource");
    const auto region=by_resource.find(resource);if(region==by_resource.end())invalid("readonly_region_missing");
    const auto allocation=region->second->at("allocation").get<std::string>();
    if(allocation.empty())invalid("empty_allocation");
    if(allocation.size()>4096)invalid("identity_size_cap");
    const auto declaration=pointer_declarations.find({parameter,field_offset});
    if(declaration==pointer_declarations.end() || declaration->second->value("type","")!="pointer" || declaration->second->value("object","")!=allocation ||
        !recorded_objects.contains({allocation,field_offset}))invalid("pointer_identity_missing");
    const auto type=array.value("element_type",std::string("s32"));
    if(type!="s32" && type!="u32" && type!="s64" && type!="u64")invalid("unsupported_cell_type");
    const std::uint64_t bytes=type.ends_with("32")?4:8;
    const auto& shapes=array.at("shapes");if(!shapes.is_array() || shapes.empty())invalid("invalid_shape_values");
    const auto first=natural(region->second->at("byte_offset"));auto offset=first;
    std::size_t rank=0;
    for(const auto& shape:shapes) {
      if(!shape.is_array() || shape.empty() || (rank && shape.size()!=rank))invalid("invalid_shape_values");rank=shape.size();
      for(const auto& value:shape) {
        if(values.size()>=maximum_cells)invalid("cell_cap");
        // Public shape values are nonnegative and type checked; do not admit
        // values the original NormalizeKernelInput would have rejected.
        (void)natural(value);const auto literal=integerLiteral(value,type);
        auto cell=scope.graph.initialCell(globalIdentity(allocation),scope.graph.constant("u64",std::to_string(offset)),type);
        scope.outputs.push_back({"context",static_cast<std::uint64_t>(values.size())*8,bytes,cell});
        values.emplace_back(type,literal);offset=add(offset,bytes);
      }
    }
    if(offset-first!=natural(region->second->at("bytes")))invalid("readonly_region_width_mismatch");
  }
  if(by_resource.size()!=supplied_resources.size())invalid("extra_readonly_region");
  // One ordinary InputMapping pass validates injective identities and dense
  // layouts for every supplied cell; no new address/layout analyzer is used.
  auto unbound=right_mapping;unbound.comparison_context.reset();
  auto aligned=alignInputs(scope,unbound,false);
  if(!aligned.complete)throw std::invalid_argument("comparison_context_"+aligned.reason);
  if(aligned.artifact.outputs.size()!=values.size())invalid("mapped_cell_count_mismatch");
  for(std::size_t i=0;i<aligned.artifact.outputs.size();++i) {
    const auto& ref=aligned.artifact.outputs[i].value;const auto& node=aligned.artifact.graph.node(ref);
    if(node.kind!=scalar_fold::Kind::InitialCell || node.type!=values[i].first ||
        !ref.conversion.empty() || !ref.pointer_provenance.empty())invalid("invalid_mapped_cell");
    const auto& address=node.operands.at(0);const auto& offset=aligned.artifact.graph.node(address);
    if(offset.kind!=scalar_fold::Kind::Constant || offset.type!="u64" || !address.conversion.empty() || !address.pointer_provenance.empty())invalid("nonconstant_mapped_cell");
    std::uint64_t start=0;const auto parsed=std::from_chars(offset.name.data(),offset.name.data()+offset.name.size(),start);
    if(parsed.ec!=std::errc{} || parsed.ptr!=offset.name.data()+offset.name.size())invalid("invalid_mapped_offset");
    const auto bytes=values[i].first.ends_with("32")?4u:8u;const auto end=add(start,bytes);
    auto& cells=result->cells_[node.name];const auto next=cells.lower_bound(start);
    if((next!=cells.end() && next->first<end) || (next!=cells.begin() && add(std::prev(next)->first,std::prev(next)->second.bytes)>start))invalid("overlapping_cells");
    cells.emplace(start,Cell{values[i].first,values[i].second,bytes});++result->cell_count_;
  }
  return result;
}
std::optional<std::string> ComparisonContext::literal(std::string_view allocation,
    std::uint64_t byte_offset,const std::string& type) const {
  const auto region=cells_.find(allocation);if(region==cells_.end())return {};
  const auto cell=region->second.find(byte_offset);
  if(cell==region->second.end() || cell->second.type!=type)return {};
  return cell->second.literal;
}
} // namespace sentryward::equivalence
