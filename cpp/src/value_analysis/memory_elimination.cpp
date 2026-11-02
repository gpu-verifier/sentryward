#include "sentryward/value_analysis/memory_instance_set.h"
#include "sentryward/value_analysis/memory_elimination.h"
#include "sentryward/value_analysis/scalar_specialization.h"
#include "sentryward/algorithm/diagnostics/compression_audit.h"

#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <typeinfo>
#include <utility>

#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/native_matrix.h"
#include "sentryward/value_ir/ptx_operation.h"

namespace sentryward::value_analysis {
namespace {
namespace v = value_ir;

struct Cell final {
  ConstantRegion region;
  v::Value value;
  MemoryInstance writer;
  std::uint64_t value_byte_offset = 0;
  // Written as a tensor map (tensormap.cp_fenceproxy): descriptor bytes, not tensor data.
  bool tensor_map = false;
  // Set when reduction stores combined into these bytes (one element); `value` is then the
  // last contribution. Shared only transiently: an append copies a shared record first.
  std::shared_ptr<ReducedContents> reduced = {};
};

using Cells = std::map<std::uint64_t, Cell>;

// Record the storage coordinate at read resolution, when the exact writer is
// already known. Component syntax is not a distinct writer: a fixed component
// is a byte slice of the SAME matrix occurrence. Numerical conversions and
// selections are deliberately not stripped. The immutable index-fact cache
// visits each encountered index DAG once, never the matrix's math ancestors.
void recordReadSlice(MemoryEliminationResult& result,const v::Graph& graph,
    ScalarSpecialization& indices,MemoryInstance reader,const ReadReplacement& replacement,
    std::uint64_t result_offset=0) {
  if(const auto* assembly=std::get_if<AssembledRead>(&replacement)) {
    for(const auto& piece:assembly->pieces) {
      const auto bits=piece.type.bitSize();
      if(bits%8 || bits/8>UINT64_MAX-result_offset)
        throw std::length_error("read slice fragment extent overflow");
      recordReadSlice(result,graph,indices,reader,ReadReplacement{piece},result_offset);
      result_offset+=bits/8;
    }
    return;
  }
  const auto* slice=std::get_if<ReinterpretedRead>(&replacement);
  const auto* source=slice?std::get_if<v::Value>(&slice->source):nullptr;
  if(source && source->indices().empty() && !source->conversion() &&
      !source->threadSelector() && !source->pointer() && !result_offset &&
      slice->type.bitSize()==graph.type(graph.node(reader.node).result(0,reader.instance)).bitSize()) {
    // A copy of an already resolved initial-memory read still denotes those
    // immutable initial bytes, even if memory was subsequently overwritten.
    // Lookup follows the retained read result, never the current memory state
    // or mathematical operands. Raw read-from/writer provenance stays intact.
    const auto* prior=result.readReplacement({source->nodeId(),source->instanceIndex()});
    const auto* initial=prior?std::get_if<InitialMemorySymbol>(prior):nullptr;
    if(initial && slice->type.bitSize()%8==0 && slice->byte_offset<=initial->region.bytes &&
        slice->type.bitSize()/8<=initial->region.bytes-slice->byte_offset &&
        slice->byte_offset<=UINT64_MAX-initial->region.offset) {
      auto mapped=*initial;mapped.region.offset+=slice->byte_offset;
      mapped.region.bytes=slice->type.bitSize()/8;mapped.type=slice->type;
      result.recordInitialRead(reader,mapped);
      return;
    }
  }
  if(source && !source->indices().empty() && !source->conversion() && !source->threadSelector() && !source->pointer()) {
    const auto matrix=source->withoutSelection();const auto& type=graph.type(matrix);
    const auto& shape=type.shape();bool valid=type.hasStaticShape() &&
        shape.size()==source->indices().size() && type.element().width() && type.element().width()%8==0;
    unsigned __int128 element=0;
    for(std::size_t axis=0;valid && axis<shape.size();++axis) {
      const auto known=indices.known(source->indices()[axis]);
      if(!known){valid=false;break;}
      std::size_t used=0;std::uint64_t index=0;
      try{index=std::stoull(known->literal,&used,0);}catch(const std::exception&){valid=false;break;}
      if(used!=known->literal.size() || known->literal.starts_with('-') || index>=shape[axis]){valid=false;break;}
      element=element*shape[axis]+index;
      if(element>UINT64_MAX){valid=false;break;}
    }
    const auto offset=element*(type.element().width()/8)+slice->byte_offset;
    if(valid && offset<=UINT64_MAX) {
      result.recordReadSlice(reader,ReinterpretedRead{matrix,slice->type,static_cast<std::uint64_t>(offset)},result_offset);
      return;
    }
  }
  result.recordReadSlice(reader,replacement,result_offset);
}

// Range assignment and maximum query over allocation-relative byte offsets.
class MemoryVersions {
  struct Node {std::uint64_t version=0;std::size_t left=0,right=0;bool uniform=true;};
  std::vector<Node> nodes{{}};
  static std::uint64_t end(std::uint64_t begin,unsigned bits) {
    return bits==64?UINT64_MAX:begin+((std::uint64_t{1}<<bits)-1);
  }
  void assign(std::size_t id,std::uint64_t begin,unsigned bits,
              std::uint64_t lo,std::uint64_t hi,std::uint64_t version) {
    if(lo<=begin && end(begin,bits)<=hi) {
      nodes[id].version=version;nodes[id].uniform=true;return;
    }
    const auto middle=begin+(std::uint64_t{1}<<(bits-1));
    if(!nodes[id].left) {
      const auto old=nodes[id].version;
      const auto left=nodes.size();nodes.push_back({old});nodes.push_back({old});
      nodes[id].left=left;nodes[id].right=left+1;
    }
    const auto left=nodes[id].left,right=nodes[id].right;
    if(nodes[id].uniform) {
      nodes[left].version=nodes[right].version=nodes[id].version;
      nodes[left].uniform=nodes[right].uniform=true;nodes[id].uniform=false;
    }
    if(lo<middle)assign(left,begin,bits-1,lo,hi,version);
    if(hi>=middle)assign(right,middle,bits-1,lo,hi,version);
    nodes[id].version=std::max(nodes[left].version,nodes[right].version);
  }
  std::uint64_t query(std::size_t id,std::uint64_t begin,unsigned bits,
                      std::uint64_t lo,std::uint64_t hi) const {
    if(nodes[id].uniform || (lo<=begin && end(begin,bits)<=hi))return nodes[id].version;
    const auto middle=begin+(std::uint64_t{1}<<(bits-1));
    std::uint64_t version=0;
    if(lo<middle)version=query(nodes[id].left,begin,bits-1,lo,hi);
    if(hi>=middle)version=std::max(version,query(nodes[id].right,middle,bits-1,lo,hi));
    return version;
  }
 public:
  void write(const ConstantRegion& region,std::uint64_t version) {
    assign(0,0,64,region.offset,region.offset+region.bytes-1,version);
  }
  std::uint64_t read(const ConstantRegion& region) const {
    return query(0,0,64,region.offset,region.offset+region.bytes-1);
  }
};

struct ValuePiece {
  std::variant<v::Value,std::size_t> source;
  std::uint64_t offset,bytes;
  friend bool operator==(const ValuePiece&,const ValuePiece&)=default;
};

// Pieces one signature visits at most, so that one write's cost is bounded whatever its copy
// chain; a longer chain has no signature (`exhausted` is then set).
constexpr std::size_t kSignatureSteps=1000000;

std::optional<std::vector<ValuePiece>> copyValueSignature(const v::Graph& graph,
    v::Value value,std::uint64_t offset,std::uint64_t bytes,
    const MemoryEliminationResult& memory,bool* exhausted=nullptr) {
  struct Work { std::variant<v::Value,InitialMemorySymbol> source; std::uint64_t offset,bytes; };
  std::vector<Work> pending{{value,offset,bytes}};
  std::vector<ValuePiece> result;
  std::size_t steps=0;
  while (!pending.empty()) {
    if (++steps>kSignatureSteps) {
      if (exhausted) *exhausted=true;
      return {};
    }
    auto item=std::move(pending.back()); pending.pop_back();
    if (auto* value=std::get_if<v::Value>(&item.source)) {
      *value=graph.resolve(*value);
      if (value->conversion() && value->indices().empty() && !value->threadSelector() &&
          value->conversion()->spec->kind==v::TypeConversionKind::Bitcast) {
        pending.push_back({value->conversion()->source,item.offset,item.bytes});
        continue;
      }
    }
    if (auto* read=std::get_if<v::Value>(&item.source); read && !read->conversion() && read->indices().empty() && !read->threadSelector() &&
        dynamic_cast<const v::LoadOp*>(&graph.definingNode(*read).operation()) &&
        !memory.symbolic_initial_reads.contains({read->nodeId(),read->instanceIndex()})) {
      const auto* found=memory.readReplacement({read->nodeId(),read->instanceIndex()});
      if (!found) return {};
      if (const auto* source=std::get_if<v::Value>(found)) pending.push_back({*source,item.offset,item.bytes});
      else if (const auto* source=std::get_if<InitialMemorySymbol>(found)) pending.push_back({*source,item.offset,item.bytes});
      else if (const auto* source=std::get_if<ReinterpretedRead>(found))
        pending.push_back({source->source,source->byte_offset+item.offset,item.bytes});
      else {
        const auto& assembled=std::get<AssembledRead>(*found);
        std::uint64_t position=0,covered=0;
        std::vector<Work> pieces;
        for (const auto& piece:assembled.pieces) {
          const auto size=piece.type.bitSize()/8;
          const auto begin=std::max(position,item.offset),end=std::min(position+size,item.offset+item.bytes);
          if (begin<end) {
            pieces.push_back({piece.source,piece.byte_offset+begin-position,end-begin});
            covered+=end-begin;
          }
          position+=size;
        }
        if (covered!=item.bytes) return {};
        pending.insert(pending.end(),pieces.rbegin(),pieces.rend());
      }
      continue;
    }
    ValuePiece piece;
    if (const auto* source=std::get_if<v::Value>(&item.source)) piece.source=*source;
    else piece.source=std::get<InitialMemorySymbol>(item.source).id;
    piece.offset=item.offset; piece.bytes=item.bytes;
    if (!result.empty() && result.back().source==piece.source && result.back().offset+result.back().bytes==piece.offset)
      result.back().bytes+=piece.bytes;
    else result.push_back(std::move(piece));
  }
  return result;
}

// Bitwise equality of two stored byte ranges, for write obligations (docs/assumptions.md,
// "Same-value writes"): equal byte-value signatures (copies of the same bytes), or pieces at
// equal offsets whose values are equal. Values are equal when they are the same instance;
// evaluate to the same literal of the same type (ScalarSpecialization: constants, parameters,
// coordinates); are constants or parameters with the same literal or symbol and type; are
// loads of the same initial memory or of equal stored values, or loads of unwritten storage at
// equal addresses; or are the same deterministic operation (kind, attributes, types) of equal
// operands. A single add, mul, min or max of integers or of f16/bf16/f32, and and/or/xor, may
// swap its two operands: the exact result is symmetric and PTX leaves the NaN of a NaN result
// of those floating-point forms unspecified for either order (PTX ISA 9.7.3, 9.7.4). No other
// algebra. Thread selections (shfl) are resolved per instance. Memoized over pairs of plain
// instances; each query is bounded by kWork pair visits and kDepth nesting (and its
// signatures and selection chains by their own bounds), beyond which it is neither proven
// nor refuted: truncated().
class StoredBytesEquality final {
 public:
  StoredBytesEquality(const v::Graph& graph,const MemoryEliminationResult& memory)
      :graph_(graph),memory_(memory),known_(graph) {}
  bool equal(v::Value a,std::uint64_t a_offset,v::Value b,std::uint64_t b_offset,std::uint64_t bytes);
  // The last equal() that returned false hit a bound: its bytes are undecided, not unequal.
  [[nodiscard]] bool truncated() const noexcept {return truncated_;}
 private:
  // Value pairs one query visits: a bound per write comparison, independent of the
  // expressions' size.
  static constexpr std::size_t kWork=std::size_t(1)<<20;
  // Operand nesting one query follows: value() and node() recurse once per level, so this
  // bounds their stack depth.
  static constexpr unsigned kDepth=2048;
  // Selections behind bitcast views one normalization resolves (one per shfl of a view, as
  // in a butterfly's steps); a longer chain truncates the query.
  static constexpr unsigned kSelectionSteps=16;
  using Key=std::array<std::size_t,3>;  // node, result, instance
  v::Value normal(v::Value value);
  bool value(v::Value a,v::Value b,unsigned depth);
  bool node(v::Value a,v::Value b,unsigned depth);
  bool replacement(const ReadReplacement& a,const ReadReplacement& b,unsigned depth);
  bool source(const std::variant<v::Value,InitialMemorySymbol>& a,const std::variant<v::Value,InitialMemorySymbol>& b,unsigned depth);
  const v::LoadOp* load(v::Value value) const {
    if(value.conversion() || !value.indices().empty() || value.threadSelector())return nullptr;
    return dynamic_cast<const v::LoadOp*>(&graph_.definingNode(value).operation());
  }
  static bool swappable(const v::Operation& operation,const v::ValueType& type);
  const v::Graph& graph_;
  const MemoryEliminationResult& memory_;
  ScalarSpecialization known_;
  std::set<std::pair<Key,Key>> proven_,refuted_;
  std::size_t work_=0;
  // A query that hit a bound refutes nothing permanently.
  bool truncated_=false;
};

bool StoredBytesEquality::equal(v::Value a,std::uint64_t a_offset,v::Value b,std::uint64_t b_offset,
    std::uint64_t bytes) {
  work_=0;truncated_=false;
  const auto x=copyValueSignature(graph_,a,a_offset,bytes,memory_,&truncated_),
      y=copyValueSignature(graph_,b,b_offset,bytes,memory_,&truncated_);
  if(!x || !y || x->size()!=y->size())return false;
  for(std::size_t i=0;i<x->size();++i) {
    const auto& p=(*x)[i];const auto& q=(*y)[i];
    if(p.bytes!=q.bytes || p.offset!=q.offset || p.source.index()!=q.source.index())return false;
    if(const auto* symbol=std::get_if<std::size_t>(&p.source)) {
      if(*symbol!=std::get<std::size_t>(q.source))return false;
    } else if(!value(std::get<v::Value>(p.source),std::get<v::Value>(q.source),0))return false;
  }
  return true;
}

bool StoredBytesEquality::swappable(const v::Operation& operation,const v::ValueType& type) {
  const auto* binary=dynamic_cast<const v::BinaryOp*>(&operation);
  if(!binary)return false;
  const auto& element=type.element();
  switch(binary->opcode()) {
    case v::BinaryOpcode::BitAnd: case v::BinaryOpcode::BitOr: case v::BinaryOpcode::BitXor:
      return true;
    case v::BinaryOpcode::Add: case v::BinaryOpcode::Multiply:
    case v::BinaryOpcode::Minimum: case v::BinaryOpcode::Maximum:
      return element.isInteger() ||
          (element.kind()==v::ElementKind::Floating && !element.isParameterized() && element.width()<=32);
    default: return false;
  }
}

// The value a thread selection resolves to. A selection can sit inside a bitcast view (shfl
// moves the 32-bit view of a register): select the view's source, then drop bitcast chains
// that return to the same type. Each step selects a strictly inner source.
v::Value StoredBytesEquality::normal(v::Value value) {
  value=known_.selected(graph_.resolve(value));
  for(unsigned step=0;value.conversion() && value.indices().empty() && !value.threadSelector() &&
      !value.pointer() && value.conversion()->spec->kind==v::TypeConversionKind::Bitcast;++step) {
    const auto source=graph_.resolve(value.conversion()->source);
    const auto selected=known_.selected(source);
    if(selected==source)break;
    if(step==kSelectionSteps){truncated_=true;break;}
    value=known_.selected(graph_.bitcast(selected,value.conversion()->spec->target));
  }
  return value;
}

bool StoredBytesEquality::value(v::Value a,v::Value b,unsigned depth) {
  if(++work_>kWork || depth>kDepth) {truncated_=true;return false;}
  a=normal(a);b=normal(b);
  if(a==b)return true;
  if(graph_.type(a)!=graph_.type(b) || a.threadSelector() || b.threadSelector())return false;
  // A load of a stored value is that value.
  for(auto* side:{&a,&b})
    if(load(*side) && !memory_.symbolic_initial_reads.contains({side->nodeId(),side->instanceIndex()}))
      if(const auto* found=memory_.readReplacement({side->nodeId(),side->instanceIndex()}))
        if(const auto* stored=std::get_if<v::Value>(found))*side=normal(*stored);
  if(a==b)return true;
  if(const auto x=known_.known(a),y=known_.known(b);x && y)return x->type==y->type && x->literal==y->literal;
  if(a.indices().size()!=b.indices().size() || bool(a.conversion())!=bool(b.conversion()) ||
      bool(a.pointer())!=bool(b.pointer()) || (a.pointer() && !(*a.pointer()==*b.pointer())))return false;
  if(!a.indices().empty()) {
    for(std::size_t i=0;i<a.indices().size();++i)if(!value(a.indices()[i],b.indices()[i],depth+1))return false;
    return value(a.withoutSelection(),b.withoutSelection(),depth+1);
  }
  if(a.conversion()) {
    const auto& x=*a.conversion();const auto& y=*b.conversion();
    return (x.spec==y.spec || *x.spec==*y.spec) && value(x.source,y.source,depth+1);
  }
  return node(a,b,depth);
}

bool StoredBytesEquality::node(v::Value a,v::Value b,unsigned depth) {
  const auto& x=graph_.definingNode(a);const auto& y=graph_.definingNode(b);
  if(a.resultIndex()!=b.resultIndex())return false;
  const Key ka{x.id(),a.resultIndex(),a.instanceIndex()},kb{y.id(),b.resultIndex(),b.instanceIndex()};
  const auto pair=std::minmax(ka,kb);
  if(proven_.contains(pair))return true;
  if(refuted_.contains(pair))return false;
  const auto remember=[&](bool equal) {
    if(equal)proven_.insert(pair);else if(!truncated_)refuted_.insert(pair);
    return equal;
  };
  const auto& op=x.operation();
  if(typeid(op)!=typeid(y.operation()))return remember(false);
  if(dynamic_cast<const v::LoadOp*>(&op)) {
    const MemoryInstance first{x.id(),a.instanceIndex()},second{y.id(),b.instanceIndex()};
    const auto symbolic=memory_.symbolic_initial_reads.find(first);
    if(symbolic!=memory_.symbolic_initial_reads.end()) {
      // Storage that no action writes: equal addresses read equal initial bytes.
      const auto other=memory_.symbolic_initial_reads.find(second);
      if(other==memory_.symbolic_initial_reads.end() || other->second!=symbolic->second)return remember(false);
      std::ostringstream p,q;op.printAttributes(p);y.operation().printAttributes(q);
      const auto u=graph_.instanceOperands(a),w=graph_.instanceOperands(b);
      if(p.str()!=q.str() || u.size()!=w.size())return remember(false);
      for(std::size_t i=0;i<u.size();++i)if(!value(u[i],w[i],depth+1))return remember(false);
      return remember(true);
    }
    const auto* ra=memory_.readReplacement(first);const auto* rb=memory_.readReplacement(second);
    return remember(ra && rb && replacement(*ra,*rb,depth+1));
  }
  const bool pure=dynamic_cast<const v::UnaryOp*>(&op) || dynamic_cast<const v::BinaryOp*>(&op) ||
      dynamic_cast<const v::FmaOp*>(&op) || dynamic_cast<const v::ConvertOp*>(&op) ||
      dynamic_cast<const v::BitcastOp*>(&op) || dynamic_cast<const v::CompareOp*>(&op) ||
      dynamic_cast<const v::SelectOp*>(&op) || dynamic_cast<const v::PtxScalarOp*>(&op) ||
      dynamic_cast<const v::ConstantOp*>(&op) || dynamic_cast<const v::InputParameterOp*>(&op) ||
      // An abstraction of matched code: only members of one node are known to share it.
      (dynamic_cast<const v::IdealExp2Op*>(&op) && x.id()==y.id());
  if(!pure)return remember(false);
  std::ostringstream p,q;op.printAttributes(p);y.operation().printAttributes(q);
  if(op.name()!=y.operation().name() || p.str()!=q.str() ||
      !std::ranges::equal(op.resultTypes(),y.operation().resultTypes()))return remember(false);
  if(const auto* scalar=dynamic_cast<const v::PtxScalarOp*>(&op))
    if(!(scalar->sourceType()==dynamic_cast<const v::PtxScalarOp&>(y.operation()).sourceType()))return remember(false);
  const auto u=graph_.instanceOperands(a),w=graph_.instanceOperands(b);
  if(u.size()!=w.size())return remember(false);
  // A constant (its literal) or a launch parameter (its symbol) is fixed by its attributes.
  if(u.empty())return remember(true);
  bool straight=true;
  for(std::size_t i=0;straight && i<u.size();++i)straight=value(u[i],w[i],depth+1);
  if(straight)return remember(true);
  return remember(u.size()==2 && swappable(op,graph_.type(a)) &&
      value(u[0],w[1],depth+1) && value(u[1],w[0],depth+1));
}

bool StoredBytesEquality::source(const std::variant<v::Value,InitialMemorySymbol>& a,
    const std::variant<v::Value,InitialMemorySymbol>& b,unsigned depth) {
  if(a.index()!=b.index())return false;
  if(const auto* symbol=std::get_if<InitialMemorySymbol>(&a))return symbol->id==std::get<InitialMemorySymbol>(b).id;
  return value(std::get<v::Value>(a),std::get<v::Value>(b),depth);
}

bool StoredBytesEquality::replacement(const ReadReplacement& a,const ReadReplacement& b,unsigned depth) {
  if(a.index()!=b.index())return false;
  if(const auto* stored=std::get_if<v::Value>(&a))return value(*stored,std::get<v::Value>(b),depth);
  if(const auto* symbol=std::get_if<InitialMemorySymbol>(&a))return symbol->id==std::get<InitialMemorySymbol>(b).id;
  const auto piece=[&](const ReinterpretedRead& x,const ReinterpretedRead& y) {
    return x.type==y.type && x.byte_offset==y.byte_offset && source(x.source,y.source,depth);
  };
  if(const auto* read=std::get_if<ReinterpretedRead>(&a))return piece(*read,std::get<ReinterpretedRead>(b));
  const auto& x=std::get<AssembledRead>(a);const auto& y=std::get<AssembledRead>(b);
  if(!(x.type==y.type) || x.pieces.size()!=y.pieces.size())return false;
  for(std::size_t i=0;i<x.pieces.size();++i)if(!piece(x.pieces[i],y.pieces[i]))return false;
  return true;
}

bool overlaps(const ConstantRegion& left, const ConstantRegion& right) {
  return left.offset < right.offset + right.bytes && right.offset < left.offset + left.bytes;
}

std::optional<std::string> footprintError(const v::ValueType& type,
                                         const ConstantRegion& region) {
  if (type.element().isParameterized() || !type.hasStaticShape()) {
    return "The memory value's element type or shape is not constant.";
  }
  try {
    const auto bits = type.bitSize();
    if (bits % 8 != 0 || bits / 8 != region.bytes) {
      return "The memory value does not match the complete byte region.";
    }
  } catch (const std::exception& error) {
    return std::string("The memory value's byte size cannot be represented: ") + error.what();
  }
  return {};
}

using Slice=std::optional<std::pair<std::uint64_t,std::uint64_t>>;

// The one owner of dependency-walk membership in one eliminateMemory call. A
// touched node has 2 bits per logical instance (available: proved by a
// committed walk; visiting: reached by the current walk) and a flag for its
// shared collective operand list (pushed by this walk or proved). The current
// walk's marks are logged: commit equals inserting the walk's visited set into
// the available set, and rollback leaves the available set untouched.
class DependencyMarks final {
 public:
  explicit DependencyMarks(const v::Graph& graph):graph_(graph) {}
  [[nodiscard]] bool available(MemoryInstance id) const noexcept {
    if(id.node>=nodes_.size() || !nodes_[id.node].block)return false;
    return word(id)[0]>>(id.instance%64)&1;
  }
  // Requires !available(id). False iff this walk already visited id.
  bool visit(MemoryInstance id) {
    if(id.node>=nodes_.size())nodes_.resize(id.node+1);
    if(!nodes_[id.node].block) {
      // Once per node: a valid handle's instance is below the node's count.
      const auto count=graph_.node(id.node).instanceCount();
      const auto first=words_.size(),size=2*(count/64+(count%64!=0));
      if(first>=std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("dependency mark storage overflow");
      words_.resize(first+size);
      nodes_[id.node].block=static_cast<std::uint32_t>(first+1);
    }
    auto* bits=word(id);
    const auto bit=std::uint64_t{1}<<(id.instance%64);
    if(bits[1]&bit)return false;
    visited_.push_back(id);
    bits[1]|=bit;
    return true;
  }
  [[nodiscard]] bool sharedSeen(std::size_t node) const noexcept {
    return node<nodes_.size() && nodes_[node].shared;
  }
  void markShared(std::size_t node) {
    if(node>=nodes_.size())nodes_.resize(node+1);
    if(nodes_[node].shared)return;
    shared_.push_back(node);
    nodes_[node].shared=true;
  }
  void commit() noexcept {finish(true);}
  void rollback() noexcept {finish(false);}
 private:
  struct NodeMarks {std::uint32_t block=0;bool shared=false;};  // block: 1 + first word, 0 if none
  std::uint64_t* word(MemoryInstance id) noexcept {
    return words_.data()+(nodes_[id.node].block-1)+2*(id.instance/64);
  }
  const std::uint64_t* word(MemoryInstance id) const noexcept {
    return words_.data()+(nodes_[id.node].block-1)+2*(id.instance/64);
  }
  void finish(bool proved) noexcept {
    for(const auto id:visited_) {
      auto* bits=word(id);
      const auto bit=std::uint64_t{1}<<(id.instance%64);
      if(proved)bits[0]|=bit;
      bits[1]&=~bit;
    }
    if(!proved)for(const auto node:shared_)nodes_[node].shared=false;
    visited_.clear();shared_.clear();
  }
  const v::Graph& graph_;
  std::vector<NodeMarks> nodes_;
  std::vector<std::uint64_t> words_;  // per 64 instances: available word, visiting word
  std::vector<MemoryInstance> visited_;
  std::vector<std::size_t> shared_;
};

// The text reported for an unread Load, a pure function of the const graph.
std::string missingRead(const v::Graph& graph,const v::Value& value,const Slice& required) {
  const auto origin=graph.instanceOrigin(value);
  auto missing=" Missing read at PTX line " + std::to_string(origin.source_line.value_or(0)) +
      ", node " + std::to_string(value.nodeId()) + ", instance " + std::to_string(value.instanceIndex()) +
      (required ? ", logical bytes [" + std::to_string(required->first) + ", " +
          std::to_string(required->first+required->second) + ")." : ", complete value required.");
  if (const auto operands=graph.instanceOperands(value); !operands.empty()) {
    const auto& address=graph.definingNode(operands.front()).operation();
    std::ostringstream detail; address.printAttributes(detail);
    missing += " Address operation: " + std::string(address.name()) + " " + detail.str();
  }
  return missing;
}

// Read-dependency walk of one eliminateMemory call over the live read and
// fragment records. Its stack and marks are reused by every call. Per pop: one
// move, contains/resolve and an O(1) mark test; per new visit, one operand
// fetch (borrowed unless a rebound template member) and one push per edge.
class DependencyWalk final {
 public:
  DependencyWalk(const v::Graph& graph,const MemoryInstanceSet& reads,
                 const std::map<MemoryInstance,std::map<std::uint64_t,std::uint64_t>>& fragments)
      :graph_(graph),reads_(reads),fragments_(fragments),marks_(graph) {}
  bool ready(std::span<const v::Value> roots,Slice slice={},std::string* missing=nullptr);
 private:
  const v::Graph& graph_;
  const MemoryInstanceSet& reads_;
  const std::map<MemoryInstance,std::map<std::uint64_t,std::uint64_t>>& fragments_;
  DependencyMarks marks_;
  std::vector<std::pair<v::Value,Slice>> pending_;
};

bool DependencyWalk::ready(std::span<const v::Value> roots,Slice slice,std::string* missing) {
  for (const auto& root : roots) pending_.emplace_back(root,slice);
  // The last fragment-covered unread Load of this call; it is reported when
  // the walk later fails on an invalid handle.
  std::optional<std::pair<v::Value,Slice>> covered;
  const auto reject=[&](const v::Value* unread,const Slice& required) {
    pending_.clear();marks_.rollback();
    if (missing && unread) *missing=missingRead(graph_,*unread,required);
    else if (missing && covered) *missing=missingRead(graph_,covered->first,covered->second);
    return false;
  };
  while (!pending_.empty()) {
    auto item = std::move(pending_.back());
    pending_.pop_back();
    const auto& required = item.second;
    if (!graph_.contains(item.first)) return reject(nullptr,{});
    const auto value = graph_.resolve(std::move(item.first));
    for (const auto& index : value.indices()) pending_.emplace_back(index,std::nullopt);
    if (const auto* selector=value.threadSelector()) pending_.emplace_back(*selector,std::nullopt);
    if (const auto& conversion=value.conversion()) {
      const auto slice=conversion->spec->kind==v::TypeConversionKind::Bitcast && value.indices().empty()
          ? required : Slice{};
      pending_.emplace_back(conversion->source,slice);
      continue;
    }
    const MemoryInstance id{value.nodeId(), value.instanceIndex()};
    if (marks_.available(id) || (!required && !marks_.visit(id))) continue;
    // A resolved handle without a conversion has a live slot: definingNode.
    const auto& node = graph_.node(id.node);
    if (dynamic_cast<const v::LoadOp*>(&node.operation())) {
      if (!reads_.contains(id)) {
        if (!required || !value.indices().empty()) return reject(&value,required);
        const auto found=fragments_.find(id);
        if (found==fragments_.end()) return reject(&value,required);
        auto cursor=required->first;
        const auto end=cursor+required->second;
        auto piece=found->second.upper_bound(cursor);
        if (piece!=found->second.begin()) --piece;
        while (piece!=found->second.end() && piece->first<=cursor) {
          cursor=std::max(cursor,piece->second);
          if (cursor>=end) break;
          ++piece;
        }
        if (cursor<end) return reject(&value,required);
        if (missing) covered.emplace(value,required);
      }
      continue;
    }
    auto operands = node.instanceOperands(id.instance);
    const auto* mma=dynamic_cast<const v::TcgenMmaOp*>(&node.operation());
    const auto* native=dynamic_cast<const v::NativeMatrixOp*>(&node.operation());
    const bool copy=native && native->opcode()==v::NativeMatrixOpcode::TcgenCopyValue &&
        operands.size()==1 && graph_.type(value).hasStaticShape() &&
        graph_.type(value)==graph_.type(operands[0]);
    const bool accumulate=mma && mma->attributes().accumulate;
    // D contributes pointwise to D'=A*B+D; A and B still require all reads.
    const bool sliced=required && value.indices().empty() && (copy || (accumulate && operands.size()>3));
    // All collective members borrow the one prototype list; the identity test
    // falls back to pushing. An unsliced push of it changes nothing once this
    // walk pushed it (LIFO on an acyclic graph: fully processed before another
    // member pops) or a committed walk proved it: re-pops stop at marked ids.
    const bool shared=!sliced && node.isCollectiveOccurrence() && operands.borrowed() &&
        operands.span().data()==node.operands().data();
    if (shared && marks_.sharedSeen(id.node)) continue;
    const auto push=[&](auto&& operand,std::size_t index) {
      pending_.emplace_back(std::forward<decltype(operand)>(operand),
          sliced && (copy || index==3) ? required : Slice{});
    };
    if (operands.borrowed()) for (std::size_t index=0;index<operands.size();++index) push(operands[index],index);
    else {
      // A rebound template member owns its list; move it onto the stack.
      auto rebound=std::move(operands).toVector();
      for (std::size_t index=0;index<rebound.size();++index) push(std::move(rebound[index]),index);
    }
    if (shared) marks_.markShared(id.node);
  }
  marks_.commit();
  return true;
}

// Immutable input allocations and disjoint output regions admit family-level elimination.
std::optional<MemoryEliminationResult> eliminateIndependentFamilies(
    const v::Graph& graph,const MemoryActions& actions,const CheckedOrderContract& contract) {
  if(contract.basis!=MemoryOrderBasis::TrustedReference)return {};
  struct Column {OrderedMemoryAction base;MemoryActionCodec::Delta delta;std::size_t count;};
  std::vector<Column> columns;
  std::map<std::string,AllocationInfo> allocations;
  for(const auto& allocation:contract.allocations)
    if(allocation.identity.empty() || !allocations.emplace(allocation.identity,allocation).second)return {};
  bool valid=true;
  std::set<std::size_t> nodes;
  for(const auto& node:graph.nodes())nodes.insert(node.id());
  std::map<std::size_t,std::vector<std::pair<std::size_t,std::size_t>>> coverage;
  std::set<std::string> loaded,stored;
  std::map<std::string,std::vector<std::pair<std::uint64_t,std::uint64_t>>> stores;
  MemoryEliminationResult result;
  std::size_t symbol_count=0;
  actions.forEachAffineColumn([&](const auto& action,const auto& delta,auto,auto,std::size_t count){
    if(!valid)return;
    if(!count || action.value_byte_offset || action.duplicate_write_group || !action.region ||
       (count>1 && delta[0]!=1) || action.operation.instance>SIZE_MAX-count) {valid=false;return;}
    if(!nodes.contains(action.operation.node)){valid=false;return;}
    const auto& node=graph.node(action.operation.node);
    if(node.storedInstanceCount()!=1 || action.operation.instance+count>node.instanceCount()) {valid=false;return;}
    const auto* load=dynamic_cast<const v::LoadOp*>(&node.operation());
    const auto* store=dynamic_cast<const v::StoreOp*>(&node.operation());
    if(!load && !store){valid=false;return;}
    const auto& attrs=load?load->attributes():store->attributes();
    const auto allocation=allocations.find(action.region->allocation);
    // A reduction combines into the prior contents, which only the serial pass models.
    if(attrs.guarded || attrs.layout || attrs.descriptor || attrs.reduction ||
       attrs.interpretation!=v::MemoryAttributes::Interpretation::Ordinary ||
       allocation==allocations.end() || !action.region->bytes) {valid=false;return;}
    const auto last=MemoryActionCodec::translate(MemoryActionCodec::fields(action),delta,count-1);
    if(!last || action.region->offset>UINT64_MAX-action.region->bytes ||
       (*last)[1]>UINT64_MAX-action.region->bytes){valid=false;return;}
    coverage[node.id()].push_back({action.operation.instance,action.operation.instance+count});
    columns.push_back({action,delta,count});
    if(load) {
      const auto type=graph.type(node.result(0,action.operation.instance));
      if(!type.isScalar() || footprintError(type,*action.region) || !allocation->second.has_initial_contents ||
         count>SIZE_MAX-symbol_count){valid=false;return;}
      result.initial_read_families.emplace(action.operation,InitialReadFamily{
          action.operation,count,delta[1],InitialMemorySymbol{symbol_count,*action.region,type}});
      symbol_count+=count;loaded.insert(action.region->allocation);
    } else {
      const auto stride=delta[1]<0?static_cast<std::uint64_t>(-(delta[1]+1))+1:static_cast<std::uint64_t>(delta[1]);
      if(count>1 && stride<action.region->bytes){valid=false;return;}
      stored.insert(action.region->allocation);
      stores[action.region->allocation].push_back({std::min(action.region->offset,(*last)[1]),
          std::max(action.region->offset,(*last)[1])+action.region->bytes});
    }
  });
  if(!valid)return {};
  for(const auto& name:loaded)if(stored.contains(name))return {};
  for(auto& [name,regions]:stores) {
    std::sort(regions.begin(),regions.end());
    for(std::size_t i=1;i<regions.size();++i)if(regions[i].first<regions[i-1].second)return {};
  }
  for(const auto& node:graph.nodes()) {
    if(!dynamic_cast<const v::LoadOp*>(&node.operation()) && !dynamic_cast<const v::StoreOp*>(&node.operation()))continue;
    auto found=coverage.find(node.id());if(found==coverage.end())return {};
    auto& intervals=found->second;std::sort(intervals.begin(),intervals.end());
    std::size_t end=0;
    for(auto [begin,next]:intervals){if(begin!=end)return {};end=next;}
    if(end!=node.instanceCount())return {};
  }
  for(const auto& column:columns) {
    const auto& node=graph.node(column.base.operation.node);
    if(!dynamic_cast<const v::StoreOp*>(&node.operation()))continue;
    if(column.count>1 && node.hasUniformThreadTemplate()) {
      const auto value=node.instanceOperands(column.base.operation.instance).at(1);
      if(!graph.type(value).isScalar() || footprintError(graph.type(value),*column.base.region))return {};
      if(allocations.at(column.base.region->allocation).output)
        result.output_families.push_back({column.base.operation,column.count,*column.base.region,column.delta[1]});
      continue;
    }
    for(std::size_t i=0;i<column.count;++i) {
      auto action=column.base;
      MemoryActionCodec::bind(action,*MemoryActionCodec::translate(MemoryActionCodec::fields(action),column.delta,i));
      const auto value=node.instanceOperands(action.operation.instance).at(1);
      if(!graph.type(value).isScalar() || footprintError(graph.type(value),*action.region))return {};
      if(allocations.at(action.region->allocation).output)
        result.outputs.push_back({*action.region,value,action.operation,0});
    }
  }
  std::sort(result.outputs.begin(),result.outputs.end(),[](const auto& a,const auto& b){return a.region<b.region;});
  result.processed_actions=actions.size();result.processed_families=columns.size();result.complete=true;
  return result;
}

const v::MemoryAttributes& memoryAttributes(const v::Node& node) {
  if(const auto* load=dynamic_cast<const v::LoadOp*>(&node.operation()))return load->attributes();
  return dynamic_cast<const v::StoreOp&>(node.operation()).attributes();
}

// A Load without a constant region that reads initial contents of an
// allocation no action writes: kept as an original Load leaf.
bool symbolicInitialRead(const v::Node& node,const OrderedMemoryAction& action,
    const std::map<std::string,AllocationInfo>& allocations,
    const std::set<std::string>& written_allocations,bool unknown_write_allocation) {
  if(!dynamic_cast<const v::LoadOp*>(&node.operation()) || !action.symbolic_region || action.value_byte_offset ||
     unknown_write_allocation || written_allocations.contains(action.symbolic_region->allocation))return false;
  const auto allocation=allocations.find(action.symbolic_region->allocation);
  const auto& attributes=memoryAttributes(node);
  return allocation!=allocations.end() && allocation->second.has_initial_contents &&
      attributes.interpretation==v::MemoryAttributes::Interpretation::Ordinary &&
      !attributes.layout && !attributes.descriptor;
}

// Index-function memory: accesses at a symbolic position constant + sum(coefficient x
// element of a tensor_properties input).
// A segment is (allocation, base): within it the constant offsets are exact and the
// ordinary cell model applies, since equal bases denote equal atom values (the atoms'
// objects are read-only inputs). Segments of one allocation are related by a static
// census over the whole action list, which never depends on the processing order:
//  - disjoint: two single-term bases on one injective object at different elements with
//    one coefficient K whose constant ranges lie in [0, |K|), or disjoint byte hulls by
//    the atoms' bounds;
//  - slotted: single-term bases with one coefficient K whose constant ranges lie in
//    [0, |K|). Their accesses overlap exactly when the atoms are equal and the constant
//    ranges overlap, so a range that meets no cell of the other segment is disjoint
//    for every input; meeting cells are decided per access in the serial pass;
//  - otherwise the allocation is refused (no case split is modelled).
// A positioned segment and the allocation's constant-region accesses must have disjoint
// hulls when either writes. Census: O(A log A) for the A positioned actions, O(S^2) pair
// decisions per allocation for its S segments, O(1) per constant-region action.
using SegmentKey=std::pair<std::string,std::vector<SymbolicAddressTerm>>;
class SegmentCensus final {
 public:
  enum class Relation {Disjoint,Slotted};
  explicit SegmentCensus(std::span<const TensorPropertyFact> facts):facts_(facts) {}
  void constantAccess(const ConstantRegion& region,bool store) {
    auto& hull=constants_[region.allocation];
    const auto begin=static_cast<__int128>(region.offset),end=begin+region.bytes;
    hull.begin=hull.any?std::min(hull.begin,begin):begin;hull.end=hull.any?std::max(hull.end,end):end;
    hull.any=true;hull.store|=store;
  }
  void positionedAccess(const SymbolicPosition& position,bool store,
      const std::tuple<std::uint64_t,std::uint64_t,std::uint64_t>& key,MemoryInstance operation) {
    auto& segment=segments_[{position.constant.allocation,position.base}];
    segment.ranges.push_back({position.constant.offset,position.constant.offset+position.constant.bytes});
    segment.store|=store;
    if(!segment.least || key<segment.least->first)segment.least.emplace(key,operation);
  }
  [[nodiscard]] bool positioned() const noexcept {return !segments_.empty();}
  // The first refusal in (allocation, base) order; nullopt when every pair is decided.
  [[nodiscard]] std::optional<MemoryEliminationDiagnostic> decide() {
    for(auto& [key,segment]:segments_) {
      std::sort(segment.ranges.begin(),segment.ranges.end());
      segment.ranges.erase(std::unique(segment.ranges.begin(),segment.ranges.end()),segment.ranges.end());
      segment.hull=hull(key.second,segment);
    }
    for(auto first=segments_.begin();first!=segments_.end();++first) {
      const auto& [key,segment]=*first;
      const auto constant=constants_.find(key.first);
      if(constant!=constants_.end() && (segment.store || constant->second.store) &&
         !(segment.hull && (segment.hull->second<=constant->second.begin || constant->second.end<=segment.hull->first)))
        return refusal(segment,"A symbolic position may overlap a constant-region access of allocation "+key.first+
            " (the atoms' bounds do not separate them).");
      for(auto second=std::next(first);second!=segments_.end() && second->first.first==key.first;++second) {
        const auto& other=second->second;
        if(!segment.store && !other.store)continue;
        const auto relation=relate(key.second,segment,second->first.second,other);
        if(!relation)return refusal(segment,"Two symbolic positions of allocation "+key.first+
            " may partially overlap: their bases are not slotted and no property separates them.");
        if(*relation==Relation::Slotted) {
          partners_[first->first].push_back(&second->first);partners_[second->first].push_back(&first->first);
        }
      }
    }
    return std::nullopt;
  }
  // Slotted partner segments of `key` (decided per access).
  [[nodiscard]] std::span<const SegmentKey* const> partners(const SegmentKey& key) const {
    const auto found=partners_.find(key);
    return found==partners_.end()?std::span<const SegmentKey* const>{}:std::span(found->second);
  }
 private:
  struct Hull {__int128 begin=0,end=0;bool any=false,store=false;};
  struct Segment {
    std::vector<std::pair<std::uint64_t,std::uint64_t>> ranges;
    bool store=false;
    std::optional<std::pair<__int128,__int128>> hull;
    std::optional<std::pair<std::tuple<std::uint64_t,std::uint64_t,std::uint64_t>,MemoryInstance>> least;
  };
  const TensorPropertyFact* fact(const SymbolicAddressTerm& term) const {
    return findTensorProperty(facts_,term);
  }
  // The byte hull [begin, end) over every valuation within the atoms' bounds.
  std::optional<std::pair<__int128,__int128>> hull(const std::vector<SymbolicAddressTerm>& base,const Segment& segment) const {
    if(segment.ranges.empty())return std::nullopt;
    __int128 low=0,high=0;
    for(const auto& term:base) {
      const auto* bounds=fact(term);
      if(!bounds || !bounds->lower || !bounds->upper)return std::nullopt;
      const auto a=static_cast<__int128>(term.coefficient)*(*bounds->lower),b=static_cast<__int128>(term.coefficient)*(*bounds->upper);
      low+=std::min(a,b);high+=std::max(a,b);
    }
    __int128 end=0;
    for(const auto& [begin,last]:segment.ranges)end=std::max<__int128>(end,last);
    return std::pair{low+segment.ranges.front().first,high+end};
  }
  static bool within(const Segment& segment,std::uint64_t width) {
    for(const auto& [begin,end]:segment.ranges)if(end>width)return false;
    return true;
  }
  std::optional<Relation> relate(const std::vector<SymbolicAddressTerm>& a,const Segment& x,
      const std::vector<SymbolicAddressTerm>& b,const Segment& y) const {
    if(x.hull && y.hull && (x.hull->second<=y.hull->first || y.hull->second<=x.hull->first))return Relation::Disjoint;
    if(a.size()!=1 || b.size()!=1 || a[0].coefficient!=b[0].coefficient || a[0].coefficient==INT64_MIN)return std::nullopt;
    const auto width=static_cast<std::uint64_t>(a[0].coefficient<0?-a[0].coefficient:a[0].coefficient);
    if(!within(x,width) || !within(y,width))return std::nullopt;
    const auto* property=fact(a[0]);
    if(a[0].object==b[0].object && a[0].element_type==b[0].element_type && a[0].element_offset!=b[0].element_offset &&
       property && property->injective)return Relation::Disjoint;
    return Relation::Slotted;
  }
  static MemoryEliminationDiagnostic refusal(const Segment& segment,std::string message) {
    return {"symbolic_partial_alias_unsupported",std::move(message),
        segment.least?std::optional(segment.least->second):std::nullopt,true};
  }
  std::span<const TensorPropertyFact> facts_;
  std::map<std::string,Hull> constants_;
  std::map<SegmentKey,Segment> segments_;
  std::map<SegmentKey,std::vector<const SegmentKey*>> partners_;
};

// Whether a Load or Store without a constant region is handled at its symbolic position.
bool positionedAccess(const v::Node& node,const OrderedMemoryAction& action) {
  if(action.region || !action.symbolic_position || action.value_byte_offset || action.duplicate_write_group)return false;
  const auto& attributes=memoryAttributes(node);
  return !attributes.guarded && attributes.interpretation==v::MemoryAttributes::Interpretation::Ordinary &&
      !attributes.layout && !attributes.descriptor;
}

// A positioned access against its slotted partner segments' current cells (census): it
// is disjoint from every cell its constant range does not meet. A meeting cell aliases
// when the atoms are equal: a write over it or a read of it would need a case split on
// the atoms, which is not modelled. O(partners x log cells).
std::optional<std::pair<std::string,std::string>> slottedConflict(const std::map<SegmentKey,Cells>& segments,
    std::span<const SegmentKey* const> partners,const ConstantRegion& region,bool store) {
  for(const auto* partner:partners) {
    const auto found=segments.find(*partner);
    if(found==segments.end())continue;
    const auto& cells=found->second;
    auto cell=cells.lower_bound(region.offset);
    if(cell!=cells.begin() && overlaps(std::prev(cell)->second.region,region))--cell;
    if(cell==cells.end() || !overlaps(cell->second.region,region))continue;
    if(store)return std::pair{std::string("symbolic_write_family_not_injective"),
        "A symbolic store may overwrite another segment's stored bytes of allocation "+region.allocation+
        " (equal atoms); the final contents need a case split, which is not modelled."};
    if(cell->second.region!=region)return std::pair{std::string("symbolic_partial_alias_unsupported"),
        "A symbolic load partially meets another segment's stored bytes of allocation "+region.allocation+"."};
    return std::pair{std::string("symbolic_alias_case_split_unsupported"),
        "A symbolic load may read another segment's stored bytes of allocation "+region.allocation+
        " (equal atoms); a case split on the atoms is not modelled."};
  }
  return std::nullopt;
}

// Order-free failure identity. Whether an action fails with
// dynamic_address_unsupported when it is processed is decided by the action
// alone (no constant region; fragmented, or neither guarded nor a symbolic
// initial read), the allocation contract and the set of written allocations
// of the whole action list. It never depends on the actions processed before
// it. When such actions exist, elimination reports the one with the least
// static key (instruction index, occurrence, lowering index, message) rather
// than whichever failure the serial pass reaches first; the serial pass
// cannot complete then. The key carries no thread or CTA, so the choice does
// not depend on the serializer's order, nor on which units' actions are
// listed when every unit has the same failing positions. nullopt when no
// action fails so, or some action names no Load or Store instance (`memory`:
// the instance count of each Load and Store node; the serial failure stands).
// O(actions) decoding plus one origin per failing action; only on failure.
std::optional<MemoryEliminationDiagnostic> leastDynamicAddressFailure(const v::Graph& graph,
    MemoryActionRange actions,const std::map<std::string,AllocationInfo>& allocations,
    const std::map<std::size_t,std::size_t>& memory) {
  std::set<std::string> written_allocations;
  bool unknown_write_allocation=false;
  for(const auto& action:actions) {
    const auto count=memory.find(action.operation.node);
    if(count==memory.end() || action.operation.instance>=count->second)return std::nullopt;
    if(!dynamic_cast<const v::StoreOp*>(&graph.node(action.operation.node).operation()))continue;
    if(action.region)written_allocations.insert(action.region->allocation);
    else if(action.symbolic_region)written_allocations.insert(action.symbolic_region->allocation);
    else unknown_write_allocation=true;
  }
  using Key=std::tuple<std::optional<std::uint64_t>,std::optional<std::uint64_t>,std::optional<std::uint64_t>,std::string>;
  std::optional<std::pair<Key,MemoryInstance>> least;
  bool complete_keys=true;
  for(const auto& action:actions) {
    if(action.region)continue;
    const auto& node=graph.node(action.operation.node);
    if(!action.value_byte_offset && (memoryAttributes(node).guarded || positionedAccess(node,action) ||
        symbolicInitialRead(node,action,allocations,written_allocations,unknown_write_allocation)))continue;
    const auto origin=node.instanceOrigin(action.operation.instance);
    complete_keys=complete_keys && origin.instruction_index && origin.occurrence && origin.lowering_index;
    Key key{origin.instruction_index,origin.occurrence,origin.lowering_index,
        action.value_byte_offset || !action.unsupported_reason.empty() ? action.unsupported_reason
            : "The memory access has no constant allocation-relative byte region."};
    if(!least || key<least->first)least.emplace(std::move(key),action.operation);
  }
  if(!least)return std::nullopt;
  return MemoryEliminationDiagnostic{"dynamic_address_unsupported",std::get<3>(least->first),least->second,complete_keys};
}

}  // namespace

MemoryEliminationResult eliminateMemory(const v::Graph& graph,
                                        MemoryActionRange actions,
                                        const CheckedOrderContract& contract) {
  SENTRYWARD_AUDIT_STAGE("memory_elimination");
  MemoryEliminationResult result;
  if (contract.same_value_groups_begin != SIZE_MAX) result.pending_same_value_writes = SIZE_MAX;
  ScalarSpecialization read_indices(graph);
  std::map<std::string, AllocationInfo> allocations;
  std::map<std::size_t,std::size_t> expected;  // instance count of each Load and Store node
  bool censused = false;  // the allocation contract is valid and `expected` complete
  const auto fail = [&](std::string code, std::string message,
                        std::optional<MemoryInstance> operation = {}) {
    result.diagnostic = MemoryEliminationDiagnostic{std::move(code), std::move(message), operation};
    // From then on, an action failing on its address takes the order-free
    // identity, whichever failure the serial pass reached first.
    if (censused)
      if (auto first = leastDynamicAddressFailure(graph, actions, allocations, expected))
        result.diagnostic = std::move(*first);
    return result;
  };
  if (!contract.permitsElimination()) {
    return fail("unchecked_execution_order",
                "Memory elimination requires checked execution order or explicit trusted-reference order.");
  }
  if(const auto* compact=actions.compact())
    if(auto bulk=eliminateIndependentFamilies(graph,*compact,contract))return std::move(*bulk);

  for (const auto& allocation : contract.allocations) {
    if (allocation.identity.empty() || !allocations.emplace(allocation.identity, allocation).second) {
      return fail("invalid_allocation_contract", "Allocation identities must be nonempty and unique.");
    }
  }

  std::size_t expected_count=0;
  for (const auto& node : graph.nodes()) {
    if (dynamic_cast<const v::LoadOp*>(&node.operation()) ||
        dynamic_cast<const v::StoreOp*>(&node.operation())) {
      const auto count=node.instanceCount();
      if(count>std::numeric_limits<std::size_t>::max()-expected_count)
        return fail("memory_instance_count_overflow", "The logical memory instance count cannot be represented.");
      expected.emplace(node.id(),count);
      expected_count+=count;
    }
  }
  censused = true;
  MemoryInstanceSet supplied;
  std::size_t same_value_writes=0;
  std::set<std::string> written_allocations;
  bool unknown_write_allocation=false;
  MemoryInstanceSet whole_actions;
  std::map<MemoryInstance, std::vector<std::pair<std::uint64_t,std::uint64_t>>> fragments;
  SegmentCensus census(contract.tensor_properties);
  SENTRYWARD_AUDIT(if(const auto* compact=actions.compact())
    audit->event("INSTANCE_ENUMERATION","validate_logical_memory_actions",
        compact->size(),compact->storedRunCount(),0,compact->size()>compact->storedRunCount()));
  for (const auto& action : actions) {
    SENTRYWARD_AUDIT(audit->add(AuditCounter::logical_actions_validated));
    const auto node_count=expected.find(action.operation.node);
    if (node_count==expected.end() || action.operation.instance>=node_count->second) {
      return fail("invalid_memory_action", "The action does not identify a Load or Store instance.", action.operation);
    }
    const auto& action_node=graph.node(action.operation.node);
    const bool action_store=dynamic_cast<const v::StoreOp*>(&action_node.operation());
    if(action_store) {
      if(action.region)written_allocations.insert(action.region->allocation);
      else if(action.symbolic_region)written_allocations.insert(action.symbolic_region->allocation);
      else unknown_write_allocation=true;
      same_value_writes+=action.duplicate_write_group && *action.duplicate_write_group>=contract.same_value_groups_begin;
    }
    if(positionedAccess(action_node,action)) {
      const auto origin=action_node.instanceOrigin(action.operation.instance);
      census.positionedAccess(*action.symbolic_position,action_store,{origin.instruction_index.value_or(UINT64_MAX),
          origin.occurrence.value_or(UINT64_MAX),origin.lowering_index.value_or(UINT64_MAX)},action.operation);
    }
    if (action.value_byte_offset) {
      if (whole_actions.contains(action.operation))
        return fail("duplicate_memory_action", "Whole and fragmented actions refer to the same memory instance.", action.operation);
      if (!action.region)
        return fail("dynamic_address_unsupported", action.unsupported_reason, action.operation);
      if (!action.region->bytes || *action.value_byte_offset > UINT64_MAX-action.region->bytes)
        return fail("invalid_memory_fragment", "The logical byte fragment is empty or overflows.", action.operation);
      fragments[action.operation].push_back({*action.value_byte_offset, action.region->bytes});
      supplied.insert(action.operation);
    } else if (!supplied.insert(action.operation).second) {
      return fail("duplicate_memory_action", "A memory instance occurs more than once in the supplied order.", action.operation);
    } else whole_actions.insert(action.operation);
  }
  if (supplied.size() != expected_count) {
    return fail("incomplete_memory_order", "The supplied order omits memory operation instances.");
  }
  result.pending_same_value_writes=same_value_writes;
  if(census.positioned()) {
    // Only with positioned actions: one more decoding pass for the constant-region hulls. A
    // positioned load is decided too: it may read bytes a constant-region store wrote.
    for(const auto& action:actions)
      if(action.region)census.constantAccess(*action.region,dynamic_cast<const v::StoreOp*>(&graph.node(action.operation.node).operation()));
    if(auto refused=census.decide()) {
      result.diagnostic=std::move(*refused);
      if(auto first=leastDynamicAddressFailure(graph,actions,allocations,expected))result.diagnostic=std::move(*first);
      return result;
    }
  }
  for (auto& [operation, pieces] : fragments) {
    std::sort(pieces.begin(), pieces.end());
    std::uint64_t end=0;
    for (const auto& [offset, bytes] : pieces) {
      if (offset != end)
        return fail("invalid_memory_fragment", "Logical fragments overlap or leave a gap.", operation);
      end += bytes;
    }
    const auto& node = graph.node(operation.node);
    const auto value = dynamic_cast<const v::LoadOp*>(&node.operation()) ? node.result(0,operation.instance)
        : node.instanceOperands(operation.instance).at(1);
    if (const auto error = footprintError(graph.type(value), {"logical value",0,end}))
      return fail("memory_type_unsupported", *error, operation);
  }

  std::map<std::string, Cells> memory;
  std::map<SegmentKey, Cells> segments;
  std::set<std::string> reduced_allocations;  // allocations holding a reduced cell
  struct ReadMapping {std::uint64_t epoch,version;MemoryInstance prototype;v::ValueType type;ConstantRegion region;};
  std::map<std::string,std::uint64_t> epochs;
  std::map<std::string,MemoryVersions> versions;
  std::map<std::size_t,ReadMapping> read_mappings;
  std::map<ConstantRegion, std::size_t> initial_symbols;
  MemoryInstanceSet completed_reads;
  std::map<MemoryInstance,std::map<std::uint64_t,std::uint64_t>> read_fragments;
  DependencyWalk dependencies(graph,completed_reads,read_fragments);
  std::map<MemoryInstance,std::map<std::uint64_t,ReadReplacement>> pending_reads;
  // The first write of each obligation group and byte range.
  struct DuplicateValue {
    std::optional<std::vector<ValuePiece>> signature;
    v::Value value;
    std::uint64_t offset;
  };
  std::map<std::tuple<std::size_t,std::string,std::uint64_t,std::uint64_t>,DuplicateValue> duplicate_values;
  StoredBytesEquality equality(graph,result);
  SENTRYWARD_AUDIT(if(const auto* compact=actions.compact())
    audit->event("INSTANCE_ENUMERATION","execute_logical_memory_actions",
        compact->size(),compact->storedRunCount(),0,compact->size()>compact->storedRunCount()));
  for (const auto& action : actions) {
    SENTRYWARD_AUDIT(audit->add(AuditCounter::logical_actions_processed));
    const auto& node = graph.node(action.operation.node);
    const auto operands = node.instanceOperands(action.operation.instance);
    const auto* load = dynamic_cast<const v::LoadOp*>(&node.operation());
    const auto* store = dynamic_cast<const v::StoreOp*>(&node.operation());
    const auto& attributes = load ? load->attributes() : store->attributes();
    if (attributes.guarded) {
      return fail("guarded_memory_unsupported", "A guarded memory effect must have its guard resolved first.", action.operation);
    }
    const bool initial_read=!action.region &&
        symbolicInitialRead(node,action,allocations,written_allocations,unknown_write_allocation);
    const bool positioned=!action.region && !initial_read && positionedAccess(node,action);
    if (!action.region && !positioned) {
      if(initial_read) {
        const auto& region=*action.symbolic_region;
        const auto value=node.result(0,action.operation.instance);
        if(const auto error=footprintError(graph.type(value),region))
          return fail("memory_type_unsupported",*error,action.operation);
        std::string missing;
        if(!dependencies.ready(operands.span(),{},&missing))
          return fail("invalid_value_order","An initial-memory address depends on an unexecuted read."+missing,action.operation);
        result.symbolic_initial_reads.emplace(action.operation,region);
        result.reads.emplace(action.operation,value);
        completed_reads.insert(action.operation);
        ++result.processed_actions;
        continue;
      }
      return fail("dynamic_address_unsupported", action.unsupported_reason.empty()
          ? "The memory access has no constant allocation-relative byte region."
          : action.unsupported_reason, action.operation);
    }
    // A positioned access works on its segment's cells at the position's constant part.
    const auto region = positioned ? action.symbolic_position->constant : *action.region;
    if (region.allocation.empty() || region.bytes == 0 ||
        region.offset > std::numeric_limits<std::uint64_t>::max() - region.bytes) {
      return fail("invalid_constant_region", "The byte region is empty, unnamed, or overflows its address range.", action.operation);
    }
    const auto allocation = allocations.find(region.allocation);
    if (allocation == allocations.end()) {
      return fail("unknown_allocation", "The region has no allocation contract.", action.operation);
    }
    const bool tensor_map = store && attributes.interpretation == v::MemoryAttributes::Interpretation::TensorMap;
    if ((attributes.interpretation != v::MemoryAttributes::Interpretation::Ordinary && !tensor_map) ||
        attributes.layout || attributes.descriptor) {
      return fail("memory_layout_unsupported", "Memory elimination requires a resolved ordinary contiguous layout.", action.operation);
    }
    bool operands_ready;
    std::string missing;
    if (store && action.value_byte_offset) {
      operands_ready=dependencies.ready(operands.span().first(1),{},&missing) &&
          dependencies.ready(operands.span().subspan(1,1),std::pair{*action.value_byte_offset,region.bytes},&missing);
    } else operands_ready=dependencies.ready(operands.span(),{},&missing);
    if (!operands_ready) {
      return fail("invalid_value_order", "A "+std::string(node.operation().name())+
          " operand depends on a read that has not executed."+missing, action.operation);
    }
    const auto value = load ? node.result(0, action.operation.instance) : operands.at(1);
    const auto value_type = action.value_byte_offset
        ? v::ValueType::matrix(v::ElementType::bit(8),{region.bytes}) : graph.type(value);
    if (const auto error = footprintError(value_type, region)) {
      return fail("memory_type_unsupported", *error, action.operation);
    }
    const auto finishRead = [&](ReadReplacement replacement) {
      if(!action.value_byte_offset)
        if(const auto* initial=std::get_if<InitialMemorySymbol>(&replacement))
          result.recordInitialRead(action.operation,*initial);
      if(node.instanceCount()>1) {
        if(const auto* source=std::get_if<v::Value>(&replacement))
          recordReadSlice(result,graph,read_indices,action.operation,ReinterpretedRead{*source,
              v::ValueType::matrix(v::ElementType::bit(8),{region.bytes}),0},action.value_byte_offset.value_or(0));
        else recordReadSlice(result,graph,read_indices,action.operation,replacement,action.value_byte_offset.value_or(0));
      }
      if (!action.value_byte_offset) {
        result.reads.emplace(action.operation,std::move(replacement));
        // A positioned read's region is relative to its base: never a constant mapping.
        if(node.instanceCount()>1 && !positioned)
          read_mappings.insert_or_assign(action.operation.node,
              ReadMapping{epochs[region.allocation],versions[region.allocation].read(region),
                  action.operation,value_type,region});
        completed_reads.insert(action.operation);
        return;
      }
      auto& pieces = pending_reads[action.operation];
      read_fragments[action.operation].emplace(*action.value_byte_offset,*action.value_byte_offset+region.bytes);
      pieces.emplace(*action.value_byte_offset,std::move(replacement));
      if (pieces.size()!=fragments.at(action.operation).size()) return;
      AssembledRead assembled{graph.type(value),{}};
      std::size_t index=0;
      for (const auto& [offset,piece] : pieces) {
        const auto bytes = fragments.at(action.operation).at(index++).second;
        const auto type = v::ValueType::matrix(v::ElementType::bit(8),{bytes});
        if (const auto* source = std::get_if<v::Value>(&piece)) assembled.pieces.push_back({*source,type,0});
        else if (const auto* initial = std::get_if<InitialMemorySymbol>(&piece)) assembled.pieces.push_back({*initial,type,0});
        else if (const auto* reinterpreted = std::get_if<ReinterpretedRead>(&piece)) assembled.pieces.push_back(*reinterpreted);
        else {
          const auto& nested = std::get<AssembledRead>(piece).pieces;
          assembled.pieces.insert(assembled.pieces.end(),nested.begin(),nested.end());
        }
      }
      result.reads.emplace(action.operation,std::move(assembled));
      completed_reads.insert(action.operation);
      pending_reads.erase(action.operation);
    };

    if(load && !action.value_byte_offset && !positioned) {
      const auto cached=read_mappings.find(action.operation.node);
      if(cached!=read_mappings.end() && cached->second.region==region && cached->second.type==value_type &&
         (cached->second.epoch==epochs[region.allocation] ||
          cached->second.version==versions[region.allocation].read(region))) {
        cached->second.epoch=epochs[region.allocation];
        result.shareRead(action.operation,cached->second.prototype);
        // A shared mapping is exact for this read's current byte version. Feed
        // the same producer proof to spatial consumers without another search.
        const auto* replacement=result.readReplacement(cached->second.prototype);
        if(const auto* initial=std::get_if<InitialMemorySymbol>(replacement))
          result.recordInitialRead(action.operation,*initial);
        if(const auto* source=std::get_if<v::Value>(replacement))
          recordReadSlice(result,graph,read_indices,action.operation,ReinterpretedRead{*source,value_type,0});
        else recordReadSlice(result,graph,read_indices,action.operation,*replacement);
        completed_reads.insert(action.operation);
        ++result.processed_actions;
        continue;
      }
    }

    const SegmentKey* segment_key=nullptr;
    if (positioned) {
      segment_key=&segments.try_emplace({region.allocation,action.symbolic_position->base}).first->first;
      if (auto refused = slottedConflict(segments, census.partners(*segment_key), region, store != nullptr))
        return fail(refused->first, std::move(refused->second), action.operation);
    }
    auto& cells = positioned ? segments.at(*segment_key) : memory[region.allocation];
    auto existing = cells.lower_bound(region.offset);
    if (existing != cells.begin()) {
      const auto previous = std::prev(existing);
      if (overlaps(previous->second.region, region)) existing = previous;
    }
    if (store) {
      // Positioned writes never meet a constant-region read (census), whose cache this versions.
      if (!positioned) {
        if(epochs[region.allocation]==UINT64_MAX)
          return fail("memory_version_overflow","The allocation version cannot be represented.",action.operation);
        ++epochs[region.allocation];
        if(auto index=versions.find(region.allocation);index!=versions.end())
          index->second.write(region,epochs[region.allocation]);
      }
      if (action.duplicate_write_group) {
        const bool same_value = *action.duplicate_write_group >= contract.same_value_groups_begin;
        const auto offset = action.value_byte_offset.value_or(0);
        bool exhausted=false;
        const auto signature=copyValueSignature(graph,value,offset,region.bytes,result,&exhausted);
        if (!signature && !same_value) {
          // A bound, not an unresolved read: undecided, as when the equality proof stops.
          if (exhausted) return fail("stored_value_equality_limit_exceeded",
              "The byte-value signature of an accepted duplicate store exceeded its step bound.",action.operation);
          return fail("duplicate_write_value_equality_unproved",
              "An accepted duplicate store has no resolved byte-value signature.",action.operation);
        }
        const auto key=std::tuple{*action.duplicate_write_group,region.allocation,region.offset,region.bytes};
        auto [previous,inserted]=duplicate_values.try_emplace(key,DuplicateValue{signature,value,offset});
        auto& first = previous->second;
        if (!inserted) {
          ++result.duplicate_write_value_checks;
          if (!(signature && first.signature && *signature == *first.signature) &&
              !equality.equal(first.value,first.offset,value,offset,region.bytes)) {
            // A bound stopped the comparison: the bytes are undecided, not unequal. A same-value
            // obligation stays pending, so it is reported unchecked with this reason.
            if (equality.truncated()) return fail("stored_value_equality_limit_exceeded",
                "Comparing the values of writes of the same bytes exceeded the equality proof's bounds.",
                action.operation);
            if (!same_value) return fail("duplicate_write_value_equality_unproved",
                "Accepted duplicate stores read different symbolic byte values; a unique output is not established.",action.operation);
            result.unproven_same_value_write=action.operation;
            return fail("same_value_write_unproven",
                "Unordered writes of the same bytes could not be proven to store equal values.",action.operation);
          }
        }
        if (same_value) --result.pending_same_value_writes;
      }
      if (const auto& reduction = store->attributes().reduction) {
        // Reduction: the element's cell accumulates this payload element on top of its prior
        // contents. Elementwise only (the memory link binds one fragment per element); one
        // operator per cell.
        if (positioned) return fail("symbolic_reduction_unsupported",
            "A reduction at a symbolic position is not modelled.", action.operation);
        if (reduction->element.width() % 8 || region.bytes != reduction->element.width() / 8)
          return fail("reduction_partial_overlap_unsupported",
              "A reduction store must combine one whole element.", action.operation);
        const std::pair contribution{value, action.value_byte_offset.value_or(0)};
        if (existing != cells.end() && overlaps(existing->second.region, region)) {
          auto& cell = existing->second;
          if (cell.region != region || cell.tensor_map) return fail("reduction_partial_overlap_unsupported",
              "A reduction store partially overlaps stored bytes.", action.operation);
          if (cell.reduced) {
            if (cell.reduced->operation != reduction->operation || !(cell.reduced->element == reduction->element))
              return fail("mixed_reduction_operations",
                  "One element is combined by different reduction operators or types.", action.operation);
            if (cell.reduced.use_count() != 1) cell.reduced = std::make_shared<ReducedContents>(*cell.reduced);
            cell.reduced->contributions.push_back(contribution);
          } else {
            cell.reduced = std::make_shared<ReducedContents>(ReducedContents{reduction->operation, reduction->element,
                std::pair{cell.value, cell.value_byte_offset}, {contribution}});
          }
          cell.value = value; cell.value_byte_offset = contribution.second;
          cell.writer = action.operation;
        } else {
          if (!allocation->second.has_initial_contents) return fail("uninitialized_memory_read",
              "A reduction combines into an intermediate allocation before a covering store.", action.operation);
          Cell cell{region, value, action.operation, contribution.second};
          cell.reduced = std::make_shared<ReducedContents>(ReducedContents{reduction->operation, reduction->element,
              std::nullopt, {contribution}});
          cells.emplace(region.offset, std::move(cell));
        }
        reduced_allocations.insert(region.allocation);
        ++result.processed_actions;
        continue;
      }
      std::vector<Cell> retained;
      const auto end = region.offset + region.bytes;
      while (existing != cells.end() && overlaps(existing->second.region, region)) {
        const auto previous = existing->second;
        const auto previous_end = previous.region.offset + previous.region.bytes;
        if (previous.reduced && (previous.region.offset < region.offset || previous_end > end))
          return fail("reduction_partial_overlap_unsupported",
              "A store partially overwrites a reduced element.", action.operation);
        if (previous.region.offset < region.offset) {
          auto prefix = previous;
          prefix.region.bytes = region.offset - previous.region.offset;
          retained.push_back(std::move(prefix));
        }
        if (previous_end > end) {
          auto suffix = previous;
          suffix.region.offset = end;
          suffix.region.bytes = previous_end - end;
          suffix.value_byte_offset += end - previous.region.offset;
          retained.push_back(std::move(suffix));
        }
        existing = cells.erase(existing);
      }
      for (auto& cell : retained) cells.emplace(cell.region.offset, std::move(cell));
      cells.insert_or_assign(region.offset,
          Cell{region, value, action.operation, action.value_byte_offset.value_or(0), tensor_map});
      ++result.processed_actions;
      continue;
    }
    if (load && reduced_allocations.contains(region.allocation))
      for (auto cell = existing; cell != cells.end() && overlaps(cell->second.region, region); ++cell)
        if (cell->second.reduced) return fail("reduced_read_unsupported",
            "A load reads an element that reduction stores combined into; reduced reads are not modelled.",
            action.operation);
    if (existing != cells.end() && overlaps(existing->second.region, region) &&
        existing->second.region != region) {
      const auto& previous = existing->second;
      if (load && previous.region.offset <= region.offset &&
          previous.region.offset + previous.region.bytes >= region.offset + region.bytes) {
        finishRead(ReinterpretedRead{
            previous.value, value_type, previous.value_byte_offset + region.offset - previous.region.offset});
        ++result.processed_actions;
        continue;
      }
      if (load) {
        AssembledRead assembled{value_type, {}};
        auto cursor = region.offset;
        auto cell = existing;
        const auto end = region.offset + region.bytes;
        while (cursor < end && cell != cells.end() && cell->second.region.offset <= cursor) {
          const auto& piece = cell->second;
          const auto piece_end = std::min(end, piece.region.offset + piece.region.bytes);
          if (piece_end <= cursor) break;
          assembled.pieces.push_back({piece.value,
              v::ValueType::matrix(v::ElementType::bit(8), {piece_end-cursor}), piece.value_byte_offset + cursor-piece.region.offset});
          cursor = piece_end;
          ++cell;
        }
        if (cursor == end) {
          finishRead(std::move(assembled));
          ++result.processed_actions;
          continue;
        }
      }
      return fail("partial_memory_overlap_unsupported", std::string(store ? "Store" : "Load") +
          " region [" + std::to_string(region.offset) + "," + std::to_string(region.offset + region.bytes) +
          ") partially overlaps stored region [" + std::to_string(previous.region.offset) + "," +
          std::to_string(previous.region.offset + previous.region.bytes) + "); fragmented-byte assembly is not implemented.", action.operation);
    }
    if (existing != cells.end() && existing->second.region == region) {
      if (existing->second.value_byte_offset || graph.type(existing->second.value) != value_type) {
        finishRead(ReinterpretedRead{existing->second.value, value_type, existing->second.value_byte_offset});
      } else finishRead(existing->second.value);
    } else {
      if (!allocation->second.has_initial_contents) {
        return fail("uninitialized_memory_read", "The intermediate allocation is read before a covering store.", action.operation);
      }
      if (positioned) {
        // No write reached these bytes (same segment: no cell; others: disjoint or no
        // meeting cell): the initial contents at the symbolic address, an original Load leaf.
        result.symbolic_initial_reads.emplace(action.operation,*action.symbolic_region);
        result.symbolic_read_positions.emplace(action.operation,*action.symbolic_position);
        result.reads.emplace(action.operation,value);
        completed_reads.insert(action.operation);
        ++result.processed_actions;
        continue;
      }
      auto initial = initial_symbols.find(region);
      if (initial == initial_symbols.end()) {
        const auto id = result.initial_memory.size();
        result.initial_memory.push_back({id, region, value_type});
        initial = initial_symbols.emplace(region, id).first;
      }
      const auto& symbol = result.initial_memory.at(initial->second);
      if (symbol.type != value_type) {
        finishRead(ReinterpretedRead{symbol, value_type});
      } else finishRead(symbol);
    }
    ++result.processed_actions;
  }

  for (const auto& [allocation, cells] : memory) {
    if (!allocations.at(allocation).output) continue;
    for (const auto& [offset, cell] : cells) {
      static_cast<void>(offset);
      // Tensor maps configure TMA; the reference compares tensor data. A tensor
      // map left where the reference expects data fails output coverage.
      if (cell.tensor_map) continue;
      result.outputs.push_back({cell.region, cell.value, cell.writer, cell.value_byte_offset, {}, {}, cell.reduced});
    }
  }
  // Final cells of each segment, at their symbolic positions (census: segments are
  // disjoint, or slotted and never both written at meeting ranges).
  if (!segments.empty()) result.tensor_properties = contract.tensor_properties;
  for (const auto& [key, cells] : segments) {
    if (!allocations.at(key.first).output) continue;
    for (const auto& [offset, cell] : cells) {
      static_cast<void>(offset);
      result.outputs.push_back({cell.region, cell.value, cell.writer, cell.value_byte_offset, {}, key.second});
    }
  }
  result.complete = true;
  SENTRYWARD_AUDIT(audit->add(AuditCounter::output_records,result.outputs.size());
    audit->add(AuditCounter::output_families,result.output_families.size());
    audit->event("CONCRETE_OUTPUTS","final_memory_cells_exported_as_records",
        result.outputs.size(),result.output_families.size()));
  return result;
}

}  // namespace sentryward::value_analysis
