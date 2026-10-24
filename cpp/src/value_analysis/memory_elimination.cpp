#include "sentryward/value_analysis/memory_instance_set.h"
#include "sentryward/value_analysis/memory_elimination.h"
#include "sentryward/value_analysis/scalar_specialization.h"
#include "sentryward/algorithm/diagnostics/compression_audit.h"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/native_matrix.h"

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

std::optional<std::vector<ValuePiece>> copyValueSignature(const v::Graph& graph,
    v::Value value,std::uint64_t offset,std::uint64_t bytes,
    const MemoryEliminationResult& memory) {
  struct Work { std::variant<v::Value,InitialMemorySymbol> source; std::uint64_t offset,bytes; };
  std::vector<Work> pending{{value,offset,bytes}};
  std::vector<ValuePiece> result;
  std::size_t steps=0;
  while (!pending.empty()) {
    if (++steps>1000000) return {};
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
    if(attrs.guarded || attrs.layout || attrs.descriptor ||
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

}  // namespace

MemoryEliminationResult eliminateMemory(const v::Graph& graph,
                                        MemoryActionRange actions,
                                        const CheckedOrderContract& contract) {
  SENTRYWARD_AUDIT_STAGE("memory_elimination");
  MemoryEliminationResult result;
  ScalarSpecialization read_indices(graph);
  const auto fail = [&](std::string code, std::string message,
                        std::optional<MemoryInstance> operation = {}) {
    result.diagnostic = MemoryEliminationDiagnostic{std::move(code), std::move(message), operation};
    return result;
  };
  if (!contract.permitsElimination()) {
    return fail("unchecked_execution_order",
                "Memory elimination requires checked execution order or explicit trusted-reference order.");
  }
  if(const auto* compact=actions.compact())
    if(auto bulk=eliminateIndependentFamilies(graph,*compact,contract))return std::move(*bulk);

  std::map<std::string, AllocationInfo> allocations;
  for (const auto& allocation : contract.allocations) {
    if (allocation.identity.empty() || !allocations.emplace(allocation.identity, allocation).second) {
      return fail("invalid_allocation_contract", "Allocation identities must be nonempty and unique.");
    }
  }

  std::map<std::size_t,std::size_t> expected;
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
  MemoryInstanceSet supplied;
  std::set<std::string> written_allocations;
  bool unknown_write_allocation=false;
  MemoryInstanceSet whole_actions;
  std::map<MemoryInstance, std::vector<std::pair<std::uint64_t,std::uint64_t>>> fragments;
  SENTRYWARD_AUDIT(if(const auto* compact=actions.compact())
    audit->event("INSTANCE_ENUMERATION","validate_logical_memory_actions",
        compact->size(),compact->storedRunCount(),0,compact->size()>compact->storedRunCount()));
  for (const auto& action : actions) {
    SENTRYWARD_AUDIT(audit->add(AuditCounter::logical_actions_validated));
    const auto node_count=expected.find(action.operation.node);
    if (node_count==expected.end() || action.operation.instance>=node_count->second) {
      return fail("invalid_memory_action", "The action does not identify a Load or Store instance.", action.operation);
    }
    if(dynamic_cast<const v::StoreOp*>(&graph.node(action.operation.node).operation())) {
      if(action.region)written_allocations.insert(action.region->allocation);
      else if(action.symbolic_region)written_allocations.insert(action.symbolic_region->allocation);
      else unknown_write_allocation=true;
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
  struct ReadMapping {std::uint64_t epoch,version;MemoryInstance prototype;v::ValueType type;ConstantRegion region;};
  std::map<std::string,std::uint64_t> epochs;
  std::map<std::string,MemoryVersions> versions;
  std::map<std::size_t,ReadMapping> read_mappings;
  std::map<ConstantRegion, std::size_t> initial_symbols;
  MemoryInstanceSet completed_reads;
  std::map<MemoryInstance,std::map<std::uint64_t,std::uint64_t>> read_fragments;
  DependencyWalk dependencies(graph,completed_reads,read_fragments);
  std::map<MemoryInstance,std::map<std::uint64_t,ReadReplacement>> pending_reads;
  std::map<std::tuple<std::size_t,std::string,std::uint64_t,std::uint64_t>,std::vector<ValuePiece>> duplicate_values;
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
    if (!action.region) {
      if(load && action.symbolic_region && !action.value_byte_offset &&
          !unknown_write_allocation && !written_allocations.contains(action.symbolic_region->allocation)) {
        const auto& region=*action.symbolic_region;
        const auto allocation=allocations.find(region.allocation);
        if(allocation!=allocations.end() && allocation->second.has_initial_contents &&
            attributes.interpretation==v::MemoryAttributes::Interpretation::Ordinary &&
            !attributes.layout && !attributes.descriptor) {
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
      }
      return fail("dynamic_address_unsupported", action.unsupported_reason.empty()
          ? "The memory access has no constant allocation-relative byte region."
          : action.unsupported_reason, action.operation);
    }
    const auto& region = *action.region;
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
        if(node.instanceCount()>1)
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

    if(load && !action.value_byte_offset) {
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

    auto& cells = memory[region.allocation];
    auto existing = cells.lower_bound(region.offset);
    if (existing != cells.begin()) {
      const auto previous = std::prev(existing);
      if (overlaps(previous->second.region, region)) existing = previous;
    }
    if (store) {
      if(epochs[region.allocation]==UINT64_MAX)
        return fail("memory_version_overflow","The allocation version cannot be represented.",action.operation);
      ++epochs[region.allocation];
      if(auto index=versions.find(region.allocation);index!=versions.end())
        index->second.write(region,epochs[region.allocation]);
      if (action.duplicate_write_group) {
        const auto signature=copyValueSignature(graph,value,action.value_byte_offset.value_or(0),region.bytes,result);
        if (!signature) return fail("duplicate_write_value_equality_unproved",
            "An accepted duplicate store has no resolved byte-value signature.",action.operation);
        const auto key=std::tuple{*action.duplicate_write_group,region.allocation,region.offset,region.bytes};
        const auto [previous,inserted]=duplicate_values.emplace(key,*signature);
        if (!inserted) ++result.duplicate_write_value_checks;
        if (!inserted && previous->second!=*signature) return fail("duplicate_write_value_equality_unproved",
            "Accepted duplicate stores read different symbolic byte values; a unique output is not established.",action.operation);
      }
      std::vector<Cell> retained;
      const auto end = region.offset + region.bytes;
      while (existing != cells.end() && overlaps(existing->second.region, region)) {
        const auto previous = existing->second;
        const auto previous_end = previous.region.offset + previous.region.bytes;
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
      result.outputs.push_back({cell.region, cell.value, cell.writer, cell.value_byte_offset});
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
