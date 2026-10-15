#pragma once
#include "equivalence_context.hpp"
#include <array>
#include <filesystem>
#include <iterator>

namespace rg {
namespace fixed_input {
// Streaming SHA-256 from the existing comparison-context adapter. No shell or
// platform-specific crypto dependency; hashes bind the exact saved bytes.
class Sha256 {
 public:
  void update(const char* bytes,std::size_t count) {
    if(count>UINT64_MAX/8-total_)throw Unsupported("comparison context hash input too large");
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

inline void reject(const std::string &reason) {
  throw Unsupported("comparison context: "+reason);
}
inline J metadata(const std::filesystem::path &path,std::string &hash) {
  std::ifstream stream(path,std::ios::binary);
  if(!stream)reject("cannot open "+path.string());
  constexpr size_t maximum=32*1024*1024;
  Sha256 sha;std::array<char,65536> block{};std::string bytes;
  while(stream) {
    stream.read(block.data(),block.size());const auto size=static_cast<size_t>(stream.gcount());
    if(size>maximum-bytes.size())reject("metadata size cap");
    bytes.append(block.data(),size);sha.update(block.data(),size);
  }
  if(stream.bad() || !stream.eof())reject("source read failed");
  hash=sha.finish();return J::parse(bytes);
}
inline void fields(const J &value,std::initializer_list<const char*> allowed) {
  if(!value.is_object())reject("expected object");
  std::set<std::string> names;for(const char *name:allowed)names.insert(name);
  for(auto it=value.begin();it!=value.end();++it)
    if(!names.contains(it.key()))reject("unknown field "+it.key());
}
inline uint64_t natural(const J &value) {
  if(!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>()<0))
    reject("expected nonnegative integer");
  return value.get<uint64_t>();
}
struct Layout {
  Z offset, target_offset;
  unsigned bytes;
  std::vector<Z> shape, strides, target_strides;
};
inline Layout layout(const J &value) {
  fields(value,{"source_offset","target_offset","element_bytes","shape","source_strides_bytes","target_strides_bytes"});
  Layout result;
  const auto bytes=natural(value.at("element_bytes"));
  if(bytes!=4 && bytes!=8)reject("fixed integer storage width must be 4 or 8 bytes");
  result.bytes=static_cast<unsigned>(bytes);
  result.offset=natural(value.value("source_offset",J(0)));
  result.target_offset=natural(value.value("target_offset",J(0)));
  const auto &shape=value.at("shape"),&strides=value.at("source_strides_bytes");
  const auto &target=value.contains("target_strides_bytes")?value.at("target_strides_bytes"):strides;
  if(!shape.is_array() || shape.empty() || shape.size()>64 || !strides.is_array() ||
     !target.is_array() || shape.size()!=strides.size() || shape.size()!=target.size())
    reject("invalid fixed input layout rank");
  std::vector<std::pair<Z,Z>> dimensions;
  for(size_t i=0;i<shape.size();++i) {
    Z count=natural(shape[i]),stride=natural(strides[i]),to=natural(target[i]);
    if(count==0 || stride==0 || to==0)reject("nonpositive fixed input layout");
    result.shape.push_back(count);result.strides.push_back(stride);result.target_strides.push_back(to);
    if(count>1)dimensions.emplace_back(stride,count);
  }
  std::sort(dimensions.begin(),dimensions.end());Z span=bytes;
  for(const auto &[stride,count]:dimensions) {
    if(stride!=span)reject("fixed input source layout is not dense permutation");
    span*=count;
  }
  return result;
}
inline std::vector<Z> coordinates(const Layout &view,const Z &byte_offset) {
  const Z relative=byte_offset-view.offset;
  Z extent=view.bytes;for(const Z &count:view.shape)extent*=count;
  if(relative<0 || relative+view.bytes>extent || relative%view.bytes!=0)
    reject("fixed input element outside layout or unaligned");
  std::vector<Z> result;Z reconstructed=0;
  for(size_t i=0;i<view.shape.size();++i) {
    Z coordinate=(relative/view.strides[i])%view.shape[i];
    result.push_back(coordinate);reconstructed+=coordinate*view.strides[i];
  }
  if(reconstructed!=relative)reject("fixed input layout coordinate mismatch");
  return result;
}
inline std::map<std::string,std::string> identities(const J &mapping,const char *side) {
  const auto &table=mapping.at(side).at("allocations");
  if(!table.is_object() || table.size()>4096)reject("invalid input identities");
  std::map<std::string,std::string> result;std::set<std::string> shared;
  for(auto it=table.begin();it!=table.end();++it) {
    if(!it.value().is_string())reject("invalid shared identity");
    auto name=it.value().get<std::string>();
    if(it.key().empty() || it.key().size()>8192 || name.empty() || name.size()>4096 || !shared.insert(name).second)
      reject("noninjective or invalid input identity");
    result.emplace(it.key(),std::move(name));
  }
  return result;
}
// Pure JSON-to-native conversion, separately testable from file transport.
inline std::vector<FixedInputElement> decode_verified(const J &input,const J &report,
                                            const J &mapping,const std::string &input_hash,
                                            const std::string &context_hash) {
  if(input.value("schema","")!="spverify.kernel-input/v1")reject("unsupported kernel input schema");
  const auto &arrays=input.at("shape_arrays");
  if(report.value("status","")!="race_free" || !report.value("memory_analysis_complete",false))
    reject("read-only validation missing");
  const auto &front=report.at("frontend");
  const auto &contract=front.at("frontend_assumptions").at("shape_input_contract");
  if(report.contains("frontend_assumptions") && report.at("frontend_assumptions").contains("shape_input_contract") &&
     report.at("frontend_assumptions").at("shape_input_contract")!=contract)
    reject("conflicting read-only contracts");
  if(contract.value("kind","")!="fixed_read_only" || contract.value("origin","")!="shape_arrays")
    reject("read-only scope missing");
  const auto &declarations=input.at("declarations"),&regions=contract.at("regions"),&objects=front.at("input_objects");
  for(const J *records:{&arrays,&declarations,&regions,&objects})
    if(!records->is_array() || records->size()>4096)reject("invalid scope records");
  if(arrays.empty())reject("empty fixed input scope");
  std::map<std::pair<uint64_t,uint64_t>,const J*> pointers;
  for(const auto &entry:declarations)if(entry.value("type","")=="pointer") {
    auto key=std::pair(natural(entry.at("parameter")),natural(entry.value("offset",J(0))));
    if(!pointers.emplace(key,&entry).second)reject("duplicate pointer declaration");
  }
  std::set<std::pair<std::string,uint64_t>> recorded;
  for(const auto &entry:objects)
    if(natural(entry.at("bytes"))==8 && entry.value("source","")=="declaration")
      recorded.emplace(entry.at("object").get<std::string>(),natural(entry.value("offset",J(0))));
  std::map<std::string,const J*> by_resource;
  for(const auto &entry:regions)
    if(!by_resource.emplace(entry.at("resource").get<std::string>(),&entry).second)
      reject("duplicate read-only resource");
  const auto left=identities(mapping,"left_inputs"),right=identities(mapping,"right_inputs");
  std::map<std::string,std::string> reference;
  for(const auto &[raw,shared]:left)reference.emplace(shared,raw);
  std::set<std::string> supplied;
  std::vector<FixedInputElement> result;
  for(const auto &array:arrays) {
    fields(array,{"parameter","offset","element_type","shapes"});
    const auto parameter=natural(array.at("parameter")),offset=natural(array.value("offset",J(0)));
    const std::string resource="input.arg"+std::to_string(parameter)+"."+std::to_string(offset);
    if(!supplied.insert(resource).second || !by_resource.contains(resource))reject("missing or duplicate fixed resource");
    const auto &region=*by_resource.at(resource);
    const auto object=region.at("allocation").get<std::string>();
    const auto pointer=pointers.find({parameter,offset});
    if(object.empty() || object.size()>4096 || pointer==pointers.end() ||
       pointer->second->value("object","")!=object || !recorded.contains({object,offset}))
      reject("pointer identity missing");
    const auto raw="6:global"+std::to_string(object.size())+":"+object;
    if(!right.contains(raw) || !reference.contains(right.at(raw)))reject("fixed input lacks common identity");
    const auto &shared=right.at(raw),&left_raw=reference.at(shared);
    const auto optimized=layout(mapping.at("right_inputs").at("layouts").at(raw));
    const auto ref=layout(mapping.at("left_inputs").at("layouts").at(left_raw));
    if(ref.shape!=optimized.shape || ref.bytes!=optimized.bytes ||
       ref.target_strides!=optimized.target_strides || ref.target_offset!=optimized.target_offset)
      reject("fixed input canonical layout mismatch");
    const auto type=array.value("element_type",std::string("s32"));
    if(type!="s32" && type!="u32" && type!="s64" && type!="u64")reject("unsupported fixed element type");
    const unsigned bits=type.ends_with("32")?32:64,bytes=bits/8;
    if(bytes!=ref.bytes)reject("fixed input type/storage mismatch");
    const auto &rows=array.at("shapes");
    if(!rows.is_array() || rows.empty())reject("invalid fixed element rows");
    Z position=natural(region.at("byte_offset")),first=position;size_t rank=0;
    for(const auto &row:rows) {
      if(!row.is_array() || row.empty() || (rank && rank!=row.size()))reject("invalid fixed element row rank");
      rank=row.size();
      for(const auto &value:row) {
        if(result.size()>=4096)reject("fixed element count cap");
        Z literal=natural(value);
        if(literal>(type[0]=='s'?Z(pow2(bits-1)-1):Z(pow2(bits)-1)))reject("fixed literal outside integer type");
        auto coordinate=coordinates(optimized,position);
        result.push_back({shared,type,std::move(coordinate),Q(literal),
          "legacy comparison context sha256="+context_hash+"; input sha256="+input_hash+"; "+resource});
        position+=bytes;
      }
    }
    if(position-first!=Z(natural(region.at("bytes"))))reject("fixed region width mismatch");
  }
  if(supplied.size()!=by_resource.size())reject("extra read-only regions");
  return result;
}
inline std::vector<FixedInputElement> decode(const J &input,const J &report,const J &provenance,
                                            const J &mapping,const std::string &input_hash,
                                            const std::string &context_hash) {
  if(provenance.at("optimized_input_sha256")!=input_hash ||
     provenance.at("scope_shape_arrays")!=input.at("shape_arrays"))
    reject("provenance scope mismatch");
  return decode_verified(input,report,mapping,input_hash,context_hash);
}

// Binding provenance is metadata for the same fixed-input checks, not new premises.
// Hash only its serialized object; relocating the containing file cannot change scope.
inline std::vector<FixedInputElement> decode_binding(
    const J &input, const J &report, const J &mapping,
    const std::string &input_hash, std::string &provenance_hash) {
  if(!mapping.contains("provenance")) {
    provenance_hash.clear();
    return decode_verified(input,report,mapping,input_hash,"current-native-run");
  }
  const auto &provenance=mapping.at("provenance");
  if(!provenance.is_object())reject("binding provenance must be an object");
  const auto bytes=provenance.dump();
  Sha256 hash;hash.update(bytes.data(),bytes.size());
  provenance_hash=hash.finish();
  return decode(input,report,provenance,mapping,input_hash,provenance_hash);
}

} // namespace fixed_input

inline J fixed_input_context_report(const PreparedContext &context) {
  J result={{"configured_elements",context.fixed_inputs.size()},
            {"applied_elements",context.applied_fixed_inputs.size()},
            {"configured",J::array()},{"applied",J::array()}};
  auto describe=[](const FixedInputElement &entry) {
    J coordinates=J::array();
    for(const Z &coordinate:entry.coordinates)coordinates.push_back(coordinate.str());
    return J{{"input",entry.input},{"interpretation",entry.interpretation},
      {"coordinates",coordinates},{"value",str(entry.value)},{"reason",entry.reason}};
  };
  for(const auto &entry:context.fixed_inputs)result["configured"].push_back(describe(entry));
  for(size_t index:context.applied_fixed_inputs) {
    J entry=describe(context.fixed_inputs.at(index));entry["side"]="left";
    result["applied"].push_back(std::move(entry));
  }
  return result;
}
} // namespace rg
