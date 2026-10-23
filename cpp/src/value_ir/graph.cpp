#include "sentryward/value_ir/graph.h"

#include <atomic>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <map>
#include <set>
#include <sstream>
#include <typeinfo>
#include <cctype>
#include <iomanip>
#include <tuple>
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/ptx_operation.h"
#include "graph_storage.h"

namespace sentryward::value_ir {
using namespace graph_detail;


bool operator==(const Value& a, const Value& b) {
  return a.owner_ == b.owner_ && a.node_id_ == b.node_id_ && a.result_index_ == b.result_index_ &&
      a.instance_index_ == b.instance_index_ && a.indices_ == b.indices_ &&
      (a.conversion_ == b.conversion_ || (a.conversion_ && b.conversion_ && *a.conversion_ == *b.conversion_)) &&
      (a.pointer_ == b.pointer_ || (a.pointer_ && b.pointer_ && *a.pointer_ == *b.pointer_)) &&
      ((!a.thread_selector_ && !b.thread_selector_) ||
       (a.thread_selector_ && b.thread_selector_ && *a.thread_selector_ == *b.thread_selector_));
}

std::string Operation::operandLabel(std::size_t index) const {
  return "operand[" + std::to_string(index) + "]";
}
const ValueType& Value::type(const Graph& owner) const { return owner.type(*this); }
Node::Node(std::uint64_t owner, std::size_t id, std::unique_ptr<const Operation> operation,
           std::vector<Value> operands, SourceOrigin origin, const InstanceTypes& types)
    : owner_(owner), id_(id), operation_(std::move(operation)),
      instances_{} {
  instances_.emplace(0,std::make_shared<Instance>(makeInstance(std::move(origin), std::move(operands), types)));
}
const Instance* Node::storedInstance(std::size_t index) const {
  if(index>=instance_count_)throw std::out_of_range("value instance index out of range");
  const auto found=instances_.find(index);
  return found==instances_.end()?nullptr:found->second.get();
}
Instance Node::readInstance(std::size_t index) const {
  if (const auto* stored=storedInstance(index)) return *stored;
  return materialize(index);
}
Instance Node::materialize(std::size_t index) const {
  const auto& base=*instances_.at(0);
  Instance result{origin_template_(index),operand_template_?operand_template_(index):base.operands,
                  base.result_types,base.element_types};
  if (types_template_) {
    const auto& types=types_template_(index);
    result.result_types=types.results; result.element_types=types.elements;
  }
  return result;
}
OperandsRef Node::instanceOperands(std::size_t index) const {
  if (const auto* stored=storedInstance(index)) return OperandsRef(stored->operands);
  // Fail as materialize would; only a template has non-stored members.
  if (!origin_template_) throw std::bad_function_call();
  if (!operand_template_) return OperandsRef(instances_.at(0)->operands);
  return OperandsRef(operand_template_(index));
}
SourceOrigin Node::readOrigin(std::size_t index) const {
  if (const auto* stored=storedInstance(index)) return stored->origin;
  return origin_template_(index);
}
std::size_t Node::storedInstanceCount() const noexcept {
  return instances_.size();
}
void Node::forEachInstancePattern(const std::function<void(
    const Instance&,std::size_t,std::size_t,bool)>& visit) const {
  std::size_t cursor=0;
  for(const auto& [index,instance]:instances_) {
    if(index>cursor) visit(*instances_.at(0),cursor,index-cursor,rebind_operands_);
    // The first stored record is the prototype, not an exceptional member.
    if(index==0 && origin_template_) continue;
    visit(*instance,index,1,false);cursor=index+1;
  }
  if(cursor<instance_count_)visit(*instances_.at(0),cursor,instance_count_-cursor,rebind_operands_);
}
std::size_t Node::resultCount() const noexcept { return operation_->resultTypes().size(); }
Value Node::result(std::size_t index, std::size_t instance) const {
  if (index >= resultCount()) throw std::out_of_range("operation result index out of range");
  if (instance >= instance_count_) throw std::out_of_range("operation instance index out of range");
  return Value(owner_, id_, index, instance);
}

Graph::Graph() : storage_(std::make_unique<Storage>()) {}
Graph::~Graph() = default;
Graph::Graph(Graph&&) noexcept = default;
Graph& Graph::operator=(Graph&&) noexcept = default;


const Node& Graph::add(std::unique_ptr<const Operation> operation,
                       std::vector<Value> operands, SourceOrigin origin) {
  if (!operation) throw std::invalid_argument("null value operation");
  if (operation->name().empty()) throw std::invalid_argument("operation name is empty");
  if (!storage_) storage_ = std::make_unique<Storage>();
  storage_->requireOpen(origin);
  if (storage_->rewriting && (hasOccurrence(origin) ||
      !dynamic_cast<const ConstantOp*>(operation.get())))
    throw std::logic_error("Cannot resume extraction after value rewriting");
  std::vector<ValueType> operand_types;
  operand_types.reserve(operands.size());
  for (const auto& operand : operands) operand_types.push_back(type(operand));
  operation->validateOperands(operand_types);
  for (const auto& type : operation->resultTypes()) validateType(type);
  const auto id = storage_->nodes.size();
  const auto& types = storage_->typesFor(operation->resultTypes());
  storage_->nodes.push_back(std::unique_ptr<Node>(new Node(storage_->identity, id, std::move(operation),
                                std::move(operands), std::move(origin), types)));
  ++storage_->live_nodes;
  return *storage_->nodes.back();
}
const Node& Graph::addCollective(std::unique_ptr<const Operation> operation,
                                std::vector<Value> operands, std::vector<SourceOrigin> members) {
  if (members.empty() || !members.front().cta) throw std::invalid_argument("collective has no CTA members");
  std::set<std::uint64_t> seen;
  std::vector<std::uint64_t> threads;
  for (const auto& member:members) {
    if (member.cta!=members.front().cta || !member.thread || !seen.insert(*member.thread).second ||
        member.instruction_index!=members.front().instruction_index ||
        member.occurrence!=members.front().occurrence || member.unroll_context!=members.front().unroll_context)
      throw std::invalid_argument("collective members must share one occurrence and have distinct threads");
    threads.push_back(*member.thread);
  }
  const auto& result=add(std::move(operation),std::move(operands),members.front());
  auto& node=*storage_->nodes[result.id()];
  auto origins=std::make_shared<const std::vector<SourceOrigin>>(std::move(members));
  node.collective_origins_=origins;
  node.instance_threads_=ThreadMembers(std::move(threads));
  node.instance_count_=origins->size();
  // Members share the prototype operands; no operand_template_.
  node.origin_template_=[origins](std::size_t index) { return origins->at(index); };
  return result;
}
namespace {
struct LocalRewriteBoundary {};
}

LocalRewriteMatch::LocalRewriteMatch(const Graph& graph, Value root,
    const Instance& pattern, std::size_t count, bool rebind,
    LocalRewriteStatistics& statistics)
    :graph_(graph),root_(root),count_(count),rebind_(rebind),statistics_(statistics),
     inspected_{{root,pattern}},exposed_(pattern.operands) {
  exposed_.push_back(root);
}

const Instance& LocalRewriteMatch::instance(Value value) {
  if(std::ranges::find(exposed_,value)==exposed_.end())
    throw std::invalid_argument("Local rewrite can inspect only matched operand edges");
  for(const auto& [known,record]:inspected_)if(known==value)return record;
  // Fixed rule-work limit, not graph depth or logical member count.
  if(inspected_.size()>=128) {
    ++statistics_.budget_rejections;
    throw LocalRewriteBoundary{};
  }
  const auto& source=graph_.definingNode(value);
  const auto& root_node=graph_.definingNode(root_);
  bool source_varies=false;
  if(count_>1) {
    const bool uniform=source.instanceCount()==1 || source.hasUniformThreadTemplate();
    const auto& source_threads=source.instanceThreads();
    const auto& root_threads=root_node.instanceThreads();
    bool same_member_map=source_threads.storageIdentity()==root_threads.storageIdentity();
    // Reuse the extraction-prefix domain certificate. The index is built once
    // over stored groups, not searched from each operand or logical member.
    if(!same_member_map && source.groupId() && root_node.groupId())
      same_member_map=graph_.groupContains(*source.groupId(),*root_node.groupId());
    // Constants have the same literal and static type throughout a node.
    const bool constant=dynamic_cast<const ConstantOp*>(&source.operation());
    const bool fixed=!rebind_ || source.instanceCount()==1 ||
        !root_node.origin().cta || source.origin().cta!=root_node.origin().cta ||
        !root_node.origin().thread || source_threads.empty() ||
        source_threads.at(value.instanceIndex())!=root_node.origin().thread;
    if(!uniform || (!constant && !fixed && !same_member_map)) {
      ++statistics_.domain_rejections;
      throw LocalRewriteBoundary{};
    }
    source_varies=!fixed && same_member_map;
  }
  auto record=graph_.instance(value);
  // Substitution must commute with following a matched edge. A fixed-thread
  // or collective value must not turn into a same-thread operand just because
  // the rewrite bypasses it. Keep that boundary unless the exposed operands
  // are also invariant under the root's thread substitution.
  if(count_>1 && rebind_ && (!source_varies || !source.rebind_operands_))
    for(auto operand:record.operands) {
      const auto& child=graph_.definingNode(operand);
      const bool annotated=operand.conversion() || !operand.indices().empty() ||
          operand.threadSelector() || operand.pointer();
      const bool same_thread=child.origin().cta==root_node.origin().cta &&
          !child.instanceThreads().empty() &&
          child.instanceThreads().at(operand.instanceIndex())==root_node.origin().thread;
      if(annotated || (same_thread && child.instanceCount()>1)) {
        ++statistics_.domain_rejections;throw LocalRewriteBoundary{};
      }
    }
  // Local rules may not hide a large collective operand list.
  if(record.operands.size()>128 || exposed_.size()+record.operands.size()>256) {
    ++statistics_.budget_rejections;
    throw LocalRewriteBoundary{};
  }
  ++statistics_.operand_reads;
  exposed_.insert(exposed_.end(),record.operands.begin(),record.operands.end());
  inspected_.emplace_back(value,std::move(record));
  return inspected_.back().second;
}

bool Graph::rewritePure(std::size_t id, std::unique_ptr<const Operation> operation,
                        const LocalRewrite& rewrite, LocalRewriteStatistics* statistics) {
  static_cast<void>(node(id));
  auto& target = *storage_->nodes.at(id);
  const auto eligible = [](const Operation& op) {
    return dynamic_cast<const BinaryOp*>(&op) || dynamic_cast<const BitcastOp*>(&op) ||
           dynamic_cast<const PtxScalarOp*>(&op) || dynamic_cast<const IdealExp2Op*>(&op);
  };
  if (!operation || !eligible(target.operation()) || !eligible(*operation) ||
      !rewrite)
    throw std::invalid_argument("Invalid pure value rewrite");
  LocalRewriteStatistics local;
  auto& counters=statistics?*statistics:local;
  std::map<std::size_t,std::shared_ptr<Instance>> replacements;
  for(const auto& [index,stored]:target.instances_) {
    const auto& instance=*stored;
    ++counters.patterns;
    const auto count=index==0 && target.origin_template_?target.instance_count_:1;
    LocalRewriteMatch match(*this,target.result(0,index),instance,count,
                            target.rebind_operands_,counters);
    std::optional<std::vector<Value>> operands;
    try {operands=rewrite(match);} catch(const LocalRewriteBoundary&) {return false;}
    if(!operands)return false;
    if (!std::ranges::equal(instance.result_types, operation->resultTypes()) ||
        !std::ranges::all_of(instance.result_types, [](const auto& t) {
          return t.isScalar() && !t.element().isParameterized();
        }))
      throw std::invalid_argument("Value rewrite must preserve scalar result types");
    if(operands->size()>128)throw std::invalid_argument("Local rewrite operand budget exceeded");
    auto allowed = [&](Value value) {
      if(value.nodeId()==id)return false; // Never introduce a self dependency.
      if (std::ranges::find(match.exposed_,value)!=match.exposed_.end()) return true;
      const auto& source = definingNode(value);
      if (dynamic_cast<const ConstantOp*>(&source.operation()) &&
          value.indices().empty() && !value.threadSelector()) return true;
      return false;
    };
    std::vector<ValueType> types;
    for (auto value : *operands) {
      if (!allowed(value)) throw std::invalid_argument("Value rewrite operands must be local match bindings or constants");
      types.push_back(type(value));
    }
    operation->validateOperands(types);
    auto replacement=std::make_shared<Instance>(instance);
    replacement->operands=std::move(*operands);
    replacements.emplace(index,std::move(replacement));
  }
  storage_->rewriting = true;
  target.operation_ = std::move(operation);
  target.instances_=std::move(replacements);
  // Members follow the new prototype; collective members share its operands.
  // Its result types are scalar and unparameterized (checked above), so the
  // node has no types_template_.
  if (target.origin_template_ && target.rebind_operands_)
    target.operand_template_=memberOperands(*storage_,target.instances_.at(0),target);
  return true;
}

void Graph::validateType(const ValueType& type) const {
  for (const auto* query : {&type.element().query(), &type.shapeQuery()}) {
    if (!*query) continue;
    const auto& descriptor_type = this->type((*query)->descriptor);
    if (!descriptor_type.element().isBitLike() || descriptor_type.element().kind() == ElementKind::Predicate)
      throw std::invalid_argument("descriptor type parameter must contain integer or bit elements");
  }
}
std::string Graph::structuralTypeKey(const ValueType& type, const SourceOrigin& consumer) const {
  std::ostringstream out;
  if (type.element().query()) {
    const auto& query = *type.element().query();
    out << "ElementTypeOf(" << structuralKey(query.descriptor, consumer) << ',' << std::quoted(query.interpretation) << ')';
  } else out << type.element().toString();
  if (type.shapeQuery()) {
    const auto& query = *type.shapeQuery();
    out << " ShapeOf(" << structuralKey(query.descriptor, consumer) << ',' << std::quoted(query.interpretation) << ')';
  } else {
    out << (type.isScalar() ? " Scalar" : " Matrix");
    for (auto extent : type.shape()) out << '[' << extent << ']';
  }
  return out.str();
}
std::string Graph::structuralKey(Value value, const SourceOrigin& consumer) const {
  value = resolve(value);
  const auto source = definingNode(value).readOrigin(value.instanceIndex());
  std::ostringstream out;
  if (value.conversion()) {
    const auto& c = *value.conversion();
    out << "view(" << structuralKey(c.source, consumer) << "," << typeConversionLabel(value) << ")";
  }
  out << value.nodeId() << ':' << value.resultIndex();
  if (value.threadSelector()) out << "@(" << structuralKey(*value.threadSelector(), consumer) << ')';
  else if (source.thread == consumer.thread && source.cta == consumer.cta) out << "@self";
  else out << "@instance" << value.instanceIndex();
  for (auto index : value.indices()) out << '[' << structuralKey(index, consumer) << ']';
  if (value.pointer()) {
    const auto& p = *value.pointer();
    out << ":pointer=" << p.identity << ':' << p.addressing;
    if (p.owner_cta) out << ":cta=" << *p.owner_cta;
    if (p.owner_thread) {
      if (p.owner_thread == consumer.thread && p.owner_cta == consumer.cta) out << ":thread=self";
      else out << ":thread=" << *p.owner_thread;
    }
  }
  return out.str();
}
void Graph::beginThreadTemplate(std::vector<SourceOrigin> threads) {
  if (!storage_) storage_ = std::make_unique<Storage>();
  if (!storage_->template_threads.empty() || threads.empty()) throw std::invalid_argument("invalid thread template scope");
  std::set<std::uint64_t> seen;
  const auto cta=threads.front().cta;
  for (const auto &thread:threads) {
    if (!cta || thread.cta!=cta || !thread.thread || !seen.insert(*thread.thread).second)
      throw std::invalid_argument("thread template requires distinct threads in one CTA");
    storage_->requireOpen(thread);
    if (storage_->cursors.contains({*cta,*thread.thread})) throw std::invalid_argument("thread template must start at CTA entry");
  }
  for (const auto &thread:threads) storage_->thread_bindings.try_emplace(std::pair{*cta,*thread.thread},thread);
  storage_->template_members=ThreadMembers(std::vector<std::uint64_t>(seen.begin(),seen.end()));
  auto [root,inserted]=storage_->roots.try_emplace(*cta,storage_->groups.size());
  if(inserted)storage_->groups.push_back({root->second,std::nullopt,*cta,storage_->template_members});
  else {
    const auto& old=storage_->groups[root->second].threads;
    std::vector<std::uint64_t> members;
    std::set_union(old.begin(),old.end(),seen.begin(),seen.end(),std::back_inserter(members));
    storage_->groups[root->second].threads=ThreadMembers(std::move(members));
  }
  storage_->template_cursor=root->second;
  for(const auto& thread:threads)storage_->cursors.emplace(std::pair{*cta,*thread.thread},root->second);
  std::vector<std::uint64_t> ordered;
  ordered.reserve(threads.size());
  for(const auto& thread:threads)ordered.push_back(*thread.thread);
  storage_->template_instance_members=ThreadMembers(std::move(ordered));
  storage_->template_threads=std::move(threads);
}
void Graph::endThreadTemplate() {
  if(!storage_ || storage_->template_threads.empty())return;
  for(const auto& thread:storage_->template_threads)
    storage_->cursors.at({*thread.cta,*thread.thread})=*storage_->template_cursor;
  storage_->template_threads.clear();storage_->template_members={};storage_->template_instance_members={};storage_->template_cursor.reset();
}

ThreadTemplate Graph::makeThreadTemplate(std::vector<SourceOrigin> threads,
                                         std::optional<ThreadGroupId> parent) {
  if (!storage_) storage_=std::make_unique<Storage>();
  if (threads.empty() || !storage_->template_threads.empty())
    throw std::invalid_argument("template creation requires nonempty inactive members");
  ThreadTemplate result;
  result.threads=std::move(threads);
  const auto cta=result.threads.front().cta;
  std::vector<std::uint64_t> members;
  for (const auto& binding:result.threads) {
    storage_->requireOpen(binding);
    if (!cta || binding.cta!=cta || !binding.thread)
      throw std::invalid_argument("template members require one CTA and thread identities");
    members.push_back(*binding.thread);
    storage_->thread_bindings.try_emplace(std::pair{*cta,*binding.thread},binding);
  }
  result.instances=ThreadMembers(members);
  std::sort(members.begin(),members.end());
  if (std::adjacent_find(members.begin(),members.end())!=members.end())
    throw std::invalid_argument("duplicate template member");
  result.members=ThreadMembers(std::move(members));
  if (parent && storage_->groups.at(*parent).cta!=cta)
    throw std::invalid_argument("template parent belongs to another CTA");
  if (parent) for (auto member:result.members)
    if (!storage_->groups.at(*parent).threads.indexOf(member))
      throw std::invalid_argument("template split adds a member outside its parent");
  result.cursor=storage_->groups.size();
  storage_->groups.push_back({*result.cursor,parent,*cta,result.members});
  return result;
}
void Graph::activateThreadTemplate(ThreadTemplate& state) {
  if (!storage_ || !storage_->template_threads.empty() || state.threads.empty())
    throw std::logic_error("invalid template activation");
  storage_->template_threads.swap(state.threads);
  std::swap(storage_->template_instance_members,state.instances);
  std::swap(storage_->template_members,state.members);
  std::swap(storage_->template_cursor,state.cursor);
}
void Graph::suspendThreadTemplate(ThreadTemplate& state) {
  if (!storage_ || storage_->template_threads.empty() || !state.threads.empty())
    throw std::logic_error("invalid template suspension");
  storage_->template_threads.swap(state.threads);
  std::swap(storage_->template_instance_members,state.instances);
  std::swap(storage_->template_members,state.members);
  std::swap(storage_->template_cursor,state.cursor);
}
void Graph::finishThreadTemplate(const ThreadTemplate& state) {
  if (!storage_ || !state.cursor || state.threads.empty())
    throw std::logic_error("cannot finish active or empty template");
  for (const auto& thread:state.threads)
    storage_->cursors.insert_or_assign(std::pair{*thread.cta,*thread.thread},*state.cursor);
}
NodeUse Graph::addTemplate(std::unique_ptr<const Operation> operation,
                          std::vector<Value> operands, SourceOrigin origin) {
  if (!storage_ || storage_->template_threads.empty())
    throw std::logic_error("template lowering requires an active member domain");
  if (storage_->rewriting || !hasOccurrence(origin) || !origin.lowering_index)
    throw std::logic_error("template node requires an extraction occurrence");
  storage_->requireOpen(origin);
  const auto group=storage_->advance(origin,{});
  const auto& node=add(std::move(operation),std::move(operands),std::move(origin));
  setInstanceTemplate(node,group,storage_->template_instance_members);
  return {node,0};
}
Value Graph::bindThread(Value value, std::uint64_t cta, std::uint64_t from, std::uint64_t to) {
  return rebindThread(*storage_,std::move(value),cta,from,to);
}
void Graph::forEachThreadBinding(const std::function<void(const SourceOrigin&)>& visit) const {
  for(const auto& [key,origin]:storage_->thread_bindings){(void)key;visit(origin);}
}

NodeUse Graph::addInstance(std::unique_ptr<const Operation> operation,
                           std::vector<Value> operands, SourceOrigin origin) {
  if (storage_ && !storage_->template_threads.empty())
    return addTemplate(std::move(operation),std::move(operands),std::move(origin));
  if (!operation) throw std::invalid_argument("null value operation");
  if (operation->name().empty()) throw std::invalid_argument("operation name is empty");
  if (storage_ && storage_->rewriting) throw std::logic_error("Cannot resume extraction after value rewriting");
  if (storage_) storage_->requireOpen(origin);
  if (!hasOccurrence(origin) || !origin.lowering_index) {
    const auto cta = origin.cta, thread = origin.thread;
    const auto& node = add(std::move(operation), std::move(operands), std::move(origin));
    if (cta && thread) {
      SourceOrigin member;
      member.cta = cta; member.thread = thread;
      storage_->nodes.back()->group_id_ = storage_->advance(member, "private:" + std::to_string(node.id()));
    }
    return {node, 0};
  }
  if (!storage_) storage_ = std::make_unique<Storage>();
  storage_->thread_bindings.try_emplace(std::pair{*origin.cta,*origin.thread}, origin);
  std::vector<ValueType> types;
  for (auto value : operands) types.push_back(type(value));
  operation->validateOperands(types);
  for (const auto& type : operation->resultTypes()) validateType(type);
  std::ostringstream key;
  key << "operation:" << storage_->occurrenceKey(origin) << ':' << *origin.lowering_index;
  const auto& concrete_operation = *operation;
  key << '|' << typeid(concrete_operation).name() << ':' << operation->name() << '|';
  operation->printAttributes(key);
  for (const auto& type : operation->resultTypes()) key << '|' << structuralTypeKey(type, origin);
  for (auto operand : operands) key << "|(" << structuralKey(operand, origin) << ')';
  const auto operation_key=key.str();
  const auto group_id=storage_->advance(origin,operation_key);
  const auto shared = storage_->shared.find(group_id);
  if (shared != storage_->shared.end()) {
    auto& node = *storage_->nodes[shared->second];
    const auto index = node.instance_count_;
    auto actual = makeInstance(std::move(origin), std::move(operands), storage_->typesFor(operation->resultTypes()));
    const auto append=[&](Instance instance) {
      auto slot=node.instance_count_;
      node.instance_threads_.push_back(*instance.origin.thread);
      bool compressed=false;
      if(node.origin_template_) {
        const auto expected=node.materialize(slot);
        compressed=expected.origin==instance.origin && expected.operands==instance.operands &&
            std::ranges::equal(expected.result_types,instance.result_types) &&
            std::ranges::equal(expected.element_types,instance.element_types);
      }
      if(!compressed)node.instances_.emplace(slot,std::make_shared<Instance>(std::move(instance)));
      ++node.instance_count_;
    };
    append(std::move(actual));
    return {node, index};
  }
  const auto& node = add(std::move(operation), std::move(operands), std::move(origin));
  setInstanceTemplate(node,group_id,ThreadMembers{*node.origin().thread});
  storage_->shared.emplace(group_id, node.id());
  return {node, 0};
}
void Graph::setInstanceTemplate(const Node& node,ThreadGroupId group_id,ThreadMembers members) {
  auto& target=*storage_->nodes.at(node.id());
  target.rebind_operands_=true;
  target.group_id_=group_id;
  const std::shared_ptr<const Instance> base=target.instances_.at(0);
  auto* storage = storage_.get();
  const auto* node_ptr = &node;
  target.instance_threads_=std::move(members);
  target.instance_count_=target.instance_threads_.size();
  // Capture the stable storage, not the movable Graph facade.
  target.origin_template_ = [storage,base,node_ptr](std::size_t index) {
    const auto thread = node_ptr->instance_threads_.at(index);
    const auto& binding = storage->thread_bindings.at({*base->origin.cta,thread});
    auto origin = base->origin;
    origin.thread = binding.thread;
    origin.warp = binding.warp;
    origin.lane = binding.lane;
    origin.thread_coordinates = binding.thread_coordinates;
    if (const auto* coordinate = dynamic_cast<const CoordinateOp*>(&node_ptr->operation());
        coordinate && coordinate->kind()==CoordinateKind::ThreadIndex && origin.thread_coordinates)
      origin.coordinate_value = origin.thread_coordinates->at(static_cast<unsigned>(coordinate->axis()));
    return origin;
  };
  const bool parameterized=std::ranges::any_of(base->result_types,[](const auto& type) {
    return !type.hasStaticShape() || type.element().isParameterized();
  });
  if (parameterized) target.types_template_=[storage,base,node_ptr](std::size_t index)->const InstanceTypes& {
    const auto key=std::pair{node_ptr->id(),index};
    if (const auto found=storage->bound_types.find(key);found!=storage->bound_types.end()) return *found->second;
    const auto thread=node_ptr->instance_threads_.at(index);
    const auto rebind=[&](Value value) { return Graph::rebindThread(*storage,value,*base->origin.cta,*base->origin.thread,thread); };
    std::vector<ValueType> types;
    for (const auto& type:base->result_types) {
      auto element=type.element();
      if (const auto& query=element.query()) element=ElementType::fromDescriptor(rebind(query->descriptor),query->interpretation);
      if (const auto& query=type.shapeQuery())
        types.push_back(ValueType::parameterizedMatrix(element,rebind(query->descriptor),query->interpretation));
      else if (type.isScalar()) types.push_back(ValueType::scalar(element));
      else types.push_back(ValueType::matrix(element,{type.shape().begin(),type.shape().end()}));
    }
    const auto& result=storage->typesFor(types);
    storage->bound_types.emplace(key,&result);
    return result;
  };
  target.operand_template_=memberOperands(*storage,base,target);
}
// The one owner of rebound member operands. The member thread is the binding
// that origin_template_ assigns to origin.thread, so values and lookup
// failures equal a rebinding of the materialized record, without its origin.
std::function<std::vector<Value>(std::size_t)> Graph::memberOperands(
    Storage& owner, std::shared_ptr<const Instance> base, const Node& member) {
  return [storage=&owner,base=std::move(base),node=&member](std::size_t index) {
    const auto cta=*base->origin.cta,from=*base->origin.thread;
    const auto to=*storage->thread_bindings.at({cta,node->instance_threads_.at(index)}).thread;
    std::vector<Value> operands;
    operands.reserve(base->operands.size());
    for (const auto& operand:base->operands) operands.push_back(Graph::rebindThread(*storage,operand,cta,from,to));
    return operands;
  };
}
ThreadGroupId Graph::observeOccurrence(const SourceOrigin& origin) {
  if (storage_ && storage_->rewriting) throw std::logic_error("Cannot resume extraction after value rewriting");
  if (!hasOccurrence(origin)) throw std::invalid_argument("occurrence requires CTA, thread, instruction and occurrence IDs");
  if (storage_) storage_->requireOpen(origin);
  if (groupAt(origin)) throw std::invalid_argument("PTX occurrence already observed for this thread");
  if (!storage_) storage_ = std::make_unique<Storage>();
  const auto key = storage_->occurrenceKey(origin);
  const auto id = storage_->advance(origin, "occurrence:" + key);
  auto& candidates = storage_->occurrences[key];
  if (std::find(candidates.begin(), candidates.end(), id) == candidates.end()) candidates.push_back(id);
  auto& execution = storage_->execution_occurrences[{*origin.cta, *origin.instruction_index, *origin.occurrence}];
  if (std::find(execution.begin(), execution.end(), id) == execution.end()) execution.push_back(id);
  return id;
}
void Graph::sealCta(std::uint64_t cta) {
  if (!storage_) storage_ = std::make_unique<Storage>();
  if (!storage_->sealed_ctas.insert(cta).second) return;
  std::erase_if(storage_->transitions, [&](const auto& item) {
    return storage_->groups[item.first.first].cta == cta;
  });
  std::erase_if(storage_->shared, [&](const auto& item) {
    return storage_->groups[item.first].cta == cta;
  });
}
std::optional<ThreadGroupId> Graph::groupAt(const SourceOrigin& origin) const {
  if (!storage_ || !hasOccurrence(origin)) return std::nullopt;
  const auto found = storage_->occurrences.find(storage_->occurrenceKey(origin));
  if (found == storage_->occurrences.end()) return std::nullopt;
  for (auto id : found->second) {
    const auto& members = storage_->groups[id].threads;
    if (std::binary_search(members.begin(), members.end(), *origin.thread)) return id;
  }
  return std::nullopt;
}
std::optional<ThreadGroupId> Graph::groupAtExecution(std::uint64_t cta, std::uint64_t thread,
                                                   std::uint64_t instruction, std::uint64_t occurrence) const {
  if (!storage_) return std::nullopt;
  const auto found = storage_->execution_occurrences.find({cta, instruction, occurrence});
  if (found == storage_->execution_occurrences.end()) return std::nullopt;
  std::optional<ThreadGroupId> match;
  for (auto id : found->second) {
    const auto& members = storage_->groups[id].threads;
    if (!std::binary_search(members.begin(), members.end(), thread)) continue;
    if (match) return std::nullopt;
    match = id;
  }
  return match;
}
std::optional<ThreadGroupId> Graph::currentGroup(std::uint64_t cta, std::uint64_t thread) const {
  if (!storage_) return std::nullopt;
  if(!storage_->template_threads.empty() && storage_->template_threads.front().cta==cta &&
      std::binary_search(storage_->template_members.begin(),storage_->template_members.end(),thread))
    return storage_->template_cursor;
  const auto found = storage_->cursors.find({cta, thread});
  return found == storage_->cursors.end() ? std::nullopt : std::optional(found->second);
}
const ThreadGroup& Graph::group(ThreadGroupId id) const {
  if (!storage_) throw std::out_of_range("thread group index out of range");
  return storage_->groups.at(id);
}
bool Graph::groupContains(ThreadGroupId ancestor,ThreadGroupId descendant) const {
  (void)group(ancestor);(void)group(descendant);
  storage_->indexGroups();
  const auto [begin,end]=storage_->group_intervals[ancestor];
  const auto position=storage_->group_intervals[descendant].first;
  return begin<=position && position<end;
}
const std::deque<ThreadGroup>& Graph::groups() const noexcept {
  static const std::deque<ThreadGroup> empty;
  return storage_ ? storage_->groups : empty;
}
Value Graph::component(Value matrix, std::vector<Value> indices) const {
  matrix = resolve(matrix);
  const auto& matrix_type = type(matrix);
  if (!matrix_type.isMatrix() || !matrix.indices().empty() || indices.empty() ||
      (matrix_type.hasStaticShape() && indices.size() != matrix_type.shape().size()))
    throw std::invalid_argument("component indices must match matrix rank");
  for (std::size_t axis = 0; axis < indices.size(); ++axis) {
    const auto& index_type = type(indices[axis]);
    if (!index_type.isScalar() || !(index_type.element().isInteger() || index_type.element().kind() == ElementKind::Bit))
      throw std::invalid_argument("component index must be an integer scalar");
    if (const auto* literal = dynamic_cast<const ConstantOp*>(&definingNode(indices[axis]).operation())) {
      const auto& text = literal->literal();
      if (!text.empty() && (std::isdigit(static_cast<unsigned char>(text[0])) || text[0] == '-')) {
        try {
          std::size_t consumed = 0;
          auto number = std::stoull(text, &consumed, 0);
          if (consumed == text.size() && (text[0] == '-' ||
              (matrix_type.hasStaticShape() && number >= matrix_type.shape()[axis])))
            throw std::out_of_range("constant component index out of bounds");
        } catch (const std::invalid_argument&) {}
      }
    }
  }
  matrix.indices_ = std::move(indices);
  return matrix;
}
Value Graph::asPointer(Value value, PointerProperties pointer) const {
  value = resolve(value);
  if (!type(value).element().isBitLike() || type(value).element().kind() == ElementKind::Predicate)
    throw std::invalid_argument("pointer value requires integer/bit elements");
  value.pointer_ = std::make_shared<const PointerProperties>(std::move(pointer));
  return value;
}
Value Graph::fromThread(Value value, Value selector) const {
  value = resolve(value);
  static_cast<void>(type(value));
  if (!type(selector).isScalar() || !type(selector).element().isInteger())
    throw std::invalid_argument("thread selector must be an integer scalar");
  if (value.thread_selector_) selector=fromThread(*value.thread_selector_,selector);
  value.thread_selector_ = std::make_shared<const Value>(selector);
  return value;
}
Value Graph::addThreadUnion(const SourceOrigin& base, const std::vector<std::uint64_t>& threads,
                            std::vector<Value> values) {
  if (!storage_) storage_ = std::make_unique<Storage>();
  if (threads.empty() || threads.size() != values.size() || !base.cta)
    throw std::invalid_argument("thread union needs one value per thread of one CTA");
  if (storage_->rewriting) throw std::logic_error("Cannot resume extraction after value rewriting");
  storage_->requireOpen(base);
  const auto value_type = type(values.front());
  for (const auto& value : values)
    if (type(value) != value_type) throw std::invalid_argument("thread union values differ in type");
  std::vector<std::uint64_t> sorted(threads);
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    throw std::invalid_argument("thread union repeats a thread");
  const auto origin = [&](std::uint64_t thread) {
    const auto& binding = storage_->thread_bindings.at({*base.cta, thread});
    auto result = base;
    result.thread = binding.thread; result.warp = binding.warp; result.lane = binding.lane;
    result.thread_coordinates = binding.thread_coordinates; result.coordinate_value.reset();
    return result;
  };
  const auto& node = add(std::make_unique<ThreadForwardOp>(value_type), {values.front()}, origin(threads.front()));
  auto& target = *storage_->nodes.at(node.id());
  const auto& types = storage_->typesFor(target.operation().resultTypes());
  for (std::size_t k = 1; k < threads.size(); ++k)
    target.instances_.emplace(k, std::make_shared<Instance>(makeInstance(origin(threads[k]), {values[k]}, types)));
  target.instance_threads_ = ThreadMembers(threads);
  target.instance_count_ = threads.size();
  return node.result(0, 0);
}
Value Graph::forwarded(Value value) const {
  const auto& node = definingNode(value);
  if (!dynamic_cast<const ThreadForwardOp*>(&node.operation())) return value;
  return instanceOperands(value).at(0);
}
Graph::ThreadBindingRelation Graph::threadBindingRelation(Value value,Value context) const {
  using Relation=ThreadBindingRelation;
  context=resolve(context);
  const auto& parent=definingNode(context);
  if(!parent.hasUniformThreadTemplate() || parent.instanceThreads().empty() ||
     context.instanceIndex()>=parent.instanceThreads().size() || !parent.origin().cta)
    return Relation::Unknown;
  const auto cta=*parent.origin().cta;
  const auto thread=parent.instanceThreads().at(context.instanceIndex());
  unsigned work=0;
  const auto merge=[](Relation a,Relation b) {
    return a==Relation::Unknown || b==Relation::Unknown?Relation::Unknown:
        a==Relation::SameDomain || b==Relation::SameDomain?Relation::SameDomain:Relation::Fixed;
  };
  const auto inspect=[&](auto&& self,Value x)->Relation {
    if(++work>128)return Relation::Unknown;
    x=resolve(x);const auto& node=definingNode(x);
    Relation result=Relation::Fixed;
    const auto& members=node.instanceThreads();
    // Mirror rebindThread's own-thread predicate, without materializing the
    // instance. A singleton cannot switch to another context member.
    // Rewritten/aliased values can retain a historical thread-members table;
    // it is not the cardinality of the retained numerical value domain.
    if(node.instanceCount()>1 && members.size()>1 && node.origin().cta==cta && x.instanceIndex()<members.size() &&
       members.at(x.instanceIndex())==thread) {
      if(members.storageIdentity()!=parent.instanceThreads().storageIdentity() ||
         x.instanceIndex()!=context.instanceIndex())return Relation::Unknown;
      result=Relation::SameDomain;
    }
    if(x.pointer() && x.pointer()->owner_cta==cta && x.pointer()->owner_thread==thread)
      result=merge(result,Relation::SameDomain);
    for(const auto& index:x.indices())result=merge(result,self(self,index));
    if(x.threadSelector())result=merge(result,self(self,*x.threadSelector()));
    if(x.conversion())result=merge(result,self(self,x.conversion()->source));
    return result;
  };
  return inspect(inspect,value);
}
std::optional<Graph::ThreadBindingMap> Graph::threadBindingMap(Value value,Value context) const {
  const auto relation=threadBindingRelation(value,context);
  if(relation==ThreadBindingRelation::Fixed)return ThreadBindingMap{true,0,0};
  if(relation==ThreadBindingRelation::SameDomain)return ThreadBindingMap{};
  value=resolve(value);context=resolve(context);
  // Type-edge views retain their producer/member identity. Peel only such
  // identity-preserving annotations, not numerical operand ancestry.
  for(unsigned depth=0;;++depth) {
    if(depth==128 || value.pointer() || value.threadSelector())return {};
    for(const auto& index:value.indices())
      if(threadBindingRelation(index,context)!=ThreadBindingRelation::Fixed)return {};
    if(!value.indices().empty())value=value.withoutSelection();
    if(!value.conversion())break;
    auto source=resolve(value.conversion()->source);
    if(source.nodeId()!=value.nodeId() || source.instanceIndex()!=value.instanceIndex())return {};
    value=std::move(source);
  }
  // Annotated handles can carry several independent domains. Do not assign
  // one producer map to their selectors, indices or conversion sources.
  if(value.pointer() || !value.indices().empty() || value.threadSelector() || value.conversion())return {};
  const auto& producer=definingNode(value);const auto& consumer=definingNode(context);
  const auto& p=producer.instanceThreads();const auto& c=consumer.instanceThreads();
  if(!consumer.hasUniformThreadTemplate() || !producer.hasUniformThreadTemplate() ||
      !consumer.origin().cta || producer.origin().cta!=consumer.origin().cta ||
      p.empty() || c.empty() || value.instanceIndex()>=p.size() || context.instanceIndex()>=c.size() ||
      p.at(value.instanceIndex())!=c.at(context.instanceIndex()))return {};
  const auto cs=c.affineStep();
  const auto start=p.indexOf(c.front());
  if(!cs || !start)return {};
  const auto finish=p.indexOf(c.at(c.size()-1));
  if(!finish)return {};
  const auto difference=__int128(*finish)-*start;
  if(c.size()>1 && difference%(c.size()-1))return {};
  const auto base=__int128(*start),stride=c.size()>1?difference/(c.size()-1):0;
  const auto last=base+stride*(c.size()-1);
  if(base+stride*context.instanceIndex()!=value.instanceIndex())return {};
  if(base<0 || last<0 || base>=producer.instanceCount() || last>=producer.instanceCount() ||
      base>INT64_MAX || stride<INT64_MIN || stride>INT64_MAX)return {};
  if(!stride) {
    if(*cs)return {};
    return ThreadBindingMap{false,static_cast<std::int64_t>(base),0};
  }
  // Endpoints propose a map; certify it over every intersected stored segment.
  // This visits segments, never the consumer's logical members.
  const auto low=std::min(base,last),high=std::max(base,last),step=stride<0?-stride:stride;
  std::size_t covered=0;
  for(std::size_t cursor=static_cast<std::size_t>(low);cursor<=high;) {
    const auto run=p.affineSegment(cursor);if(!run)return {};
    const auto begin=std::max(low,__int128(run->begin));
    const auto end=std::min(high,__int128(run->begin)+run->count-1);
    const auto q0=(begin-low+step-1)/step,q1=(end-low)/step;
    if(q0<=q1) {
      const auto member=low+step*q0;
      const auto k=stride<0?__int128(c.size()-1)-q0:q0;
      if(__int128(p.at(run->begin))+__int128(run->stride)*(member-run->begin)!=
          __int128(c.front())+__int128(*cs)*k)return {};
      if(q1>q0 && __int128(run->stride)*step!=__int128(*cs)*(stride<0?-1:1))return {};
      covered+=static_cast<std::size_t>(q1-q0+1);
    }
    cursor=run->begin+run->count;
  }
  if(covered!=c.size())return {};
  return ThreadBindingMap{false,static_cast<std::int64_t>(base),static_cast<std::int64_t>(stride)};
}
Value Graph::rebindThread(Storage& storage, Value value, std::uint64_t cta,
    std::uint64_t from, std::uint64_t to) {
  if (from == to) return value;
  const auto& node = storage.nodes.at(value.node_id_);
  const ThreadMembers* threads = nullptr;
  std::optional<std::uint64_t> owner_cta;
  if (node && !node->instance_threads_.empty()) {
    threads = &node->instance_threads_; owner_cta = node->origin().cta;
  } else if (!node) {
    const auto found = storage.aliases.find(value.node_id_);
    if (found != storage.aliases.end()) {threads = &found->second.threads; owner_cta = found->second.cta;}
  }
  if (threads && owner_cta == cta && value.instance_index_ < threads->size() &&
      threads->at(value.instance_index_) == from) {
    if(const auto found=threads->indexOf(to))value.instance_index_=*found;
  }
  for (auto& index : value.indices_) index = rebindThread(storage,index,cta,from,to);
  if (value.thread_selector_) value.thread_selector_ = std::make_shared<const Value>(rebindThread(storage,*value.thread_selector_,cta,from,to));
  if (value.conversion_) value.conversion_ = std::make_shared<const TypeConversion>(TypeConversion{
      rebindThread(storage,value.conversion_->source,cta,from,to),value.conversion_->spec});
  if (value.pointer_ && value.pointer_->owner_cta == cta && value.pointer_->owner_thread == from) {
    auto pointer = *value.pointer_; pointer.owner_thread = to;
    value.pointer_ = std::make_shared<const PointerProperties>(std::move(pointer));
  }
  return value;
}
// resolveStorage returns a handle with a live slot unchanged, so these
// accessors use such a handle in place. Only an alias of a lowered conversion
// is resolved, and it then takes the former resolve/definingNode path.
std::pair<const Node*,std::size_t> Graph::locate(const Value& value) const {
  if (!contains(value)) throw std::invalid_argument("Value is invalid, stale, or belongs to another graph");
  if (const auto& slot = storage_->nodes[value.node_id_]) return {slot.get(),value.instance_index_};
  const auto resolved = resolveStorage(*storage_, value);
  return {&definingNode(resolved),resolved.instance_index_};
}
Instance Graph::instance(const Value& value) const {
  const auto [node,index] = locate(value);
  return node->readInstance(index);
}
OperandsRef Graph::instanceOperands(const Value& value) const {
  const auto [node,index] = locate(value);
  return node->instanceOperands(index);
}
SourceOrigin Graph::instanceOrigin(const Value& value) const {
  const auto [node,index] = locate(value);
  return node->instanceOrigin(index);
}
bool Graph::contains(const Value& value) const noexcept {
  if (!storage_ || value.owner_ != storage_->identity || value.node_id_ >= storage_->nodes.size()) return false;
  const auto& node = storage_->nodes[value.node_id_];
  if (node) return value.result_index_ < node->resultCount() && value.instance_index_ < node->instance_count_;
  auto alias = storage_->aliases.find(value.node_id_);
  return alias != storage_->aliases.end() && value.result_index_ == 0 &&
      value.instance_index_ < alias->second.count;
}
const ValueType& Graph::type(const Value& value) const {
  if (!contains(value)) throw std::invalid_argument("Value is invalid, stale, or belongs to another graph");
  if (const auto& slot = storage_->nodes[value.node_id_]) return resolvedType(value, slot.get());
  // The returned type lives in its node or conversion spec, not in this handle.
  const auto resolved = resolveStorage(*storage_, value);
  return resolvedType(resolved, resolved.conversion() ? nullptr : &definingNode(resolved));
}
const ValueType& Graph::resolvedType(const Value& value, const Node* producer) const {
  if (value.conversion())
    return value.indices().empty() ? value.conversion()->spec->target : value.conversion()->spec->element;
  const auto& node = *producer;
  const auto* stored = node.storedInstance(value.instanceIndex());
  if (!stored && node.types_template_) {
    const auto& types=node.types_template_(value.instanceIndex());
    return value.indices_.empty()?types.results[value.result_index_]:types.elements[value.result_index_];
  }
  const auto& instance = stored ? *stored : *node.instances_.at(0);
  return value.indices_.empty() ? instance.result_types[value.result_index_]
                               : instance.element_types[value.result_index_];
}
const Node& Graph::definingNode(const Value& value) const {
  if (!contains(value)) throw std::invalid_argument("Value is invalid, stale, or belongs to another graph");
  if (const auto& slot = storage_->nodes[value.node_id_]) return *slot;
  return *storage_->nodes[resolveStorage(*storage_, value).node_id_];
}
const Node& Graph::node(std::size_t id) const {
  if (!storage_ || id >= storage_->nodes.size() || !storage_->nodes[id])
    throw std::out_of_range("node ID is absent or was lowered to an edge conversion");
  return *storage_->nodes[id];
}
NodeView Graph::nodes() const noexcept {
  static const NodeView::Storage empty;
  return {storage_ ? storage_->nodes : empty, size()};
}
std::size_t Graph::size() const noexcept { return storage_ ? storage_->live_nodes : 0; }

Value Graph::bitcast(Value source, ValueType target) const {
  source = resolve(source);
  BitcastOp op(target);
  const std::vector types{type(source)};
  op.validateOperands(types);
  auto result = source;
  auto spec = std::make_shared<const TypeConversionSpec>(TypeConversionSpec{
      TypeConversionKind::Bitcast, target, ValueType::scalar(target.element()), {}});
  result.conversion_ = std::make_shared<const TypeConversion>(TypeConversion{source, std::move(spec)});
  result.indices_.clear();
  result.thread_selector_.reset();
  return result;
}
Value Graph::convert(Value source, ValueType target, ConversionAttributes attributes) const {
  source = resolve(source);
  ConvertOp op(target, attributes);
  const std::vector types{type(source)};
  op.validateOperands(types);
  auto result = source;
  auto spec = std::make_shared<const TypeConversionSpec>(TypeConversionSpec{
      TypeConversionKind::Convert, target, ValueType::scalar(target.element()), attributes});
  result.conversion_ = std::make_shared<const TypeConversion>(TypeConversion{source, std::move(spec)});
  result.indices_.clear();
  result.thread_selector_.reset();
  return result;
}
Value Graph::resolve(Value value) const {
  if (!contains(value)) throw std::invalid_argument("Value is invalid, stale, or belongs to another graph");
  return resolveStorage(*storage_, std::move(value));
}
Value Graph::canonicalInterpretation(Value value) const {
  value=resolve(std::move(value));
  if(!value.indices().empty() || value.threadSelector() || !value.conversion() ||
     value.conversion()->spec->kind!=TypeConversionKind::Bitcast)return value;
  const auto interpretation=value.conversion(); // owns the target type below
  const auto& target=interpretation->spec->target;
  auto cursor=value;
  // This examines type-use edges only, never traverses the expression DAG.
  while(cursor.indices().empty() && !cursor.threadSelector() && cursor.conversion() &&
        cursor.conversion()->spec->kind==TypeConversionKind::Bitcast) {
    cursor=resolve(cursor.conversion()->source);
    if(type(cursor)==target) {
      auto canonical=cursor;
      canonical.pointer_=value.pointer_;
      value=std::move(canonical);
    }
  }
  return value;
}
Value Graph::resolveStorage(Storage& storage, Value value) {
  // Canonical views point directly to a retained producer.
  if (value.conversion()) {
    if (storage.nodes[value.node_id_]) return value;
    auto source = resolveStorage(storage, value.conversion()->source);
    auto result = source;
    result.conversion_ = std::make_shared<const TypeConversion>(TypeConversion{source, value.conversion()->spec});
    result.indices_ = value.indices_;
    result.thread_selector_ = value.thread_selector_;
    if (value.pointer_) result.pointer_ = value.pointer_;
    return result;
  }
  auto& slot = storage.nodes[value.node_id_];
  if (slot) return value;
  auto& alias = storage.aliases.at(value.node_id_);
  const auto i = value.instance_index_;
  auto found = alias.entries.find(i);
  Value result;
  if (found == alias.entries.end()) {
    result = resolveStorage(storage, Value(storage.identity, value.node_id_, 0, 0));
    result = rebindThread(storage, result, *alias.cta, alias.threads.front(), alias.threads.at(i));
  } else {
    auto& entry = found->second;
    if (!entry.resolved) {
      auto source = resolveStorage(storage, entry.source);
      result = source;
      result.indices_.clear();
      result.thread_selector_.reset();
      result.conversion_ = std::make_shared<const TypeConversion>(TypeConversion{source, entry.spec});
      entry.source = std::move(result);
      entry.resolved = true;
    }
    result = entry.source;
  }
  result.indices_ = value.indices_;
  result.thread_selector_ = value.thread_selector_;
  if (value.pointer_) result.pointer_ = value.pointer_;
  return result;
}
TypeEdgeStatistics Graph::lowerTypeConversions() {
  TypeEdgeStatistics stats;
  for (auto& slot : storage_->nodes) {
    if (!slot) continue;
    const auto* cast = dynamic_cast<const BitcastOp*>(&slot->operation());
    const auto* convert = dynamic_cast<const ConvertOp*>(&slot->operation());
    if (!cast && !convert) continue;
    auto& alias = storage_->aliases[slot->id()];
    alias.threads = slot->instance_threads_;
    alias.cta = slot->origin().cta;
    std::shared_ptr<const TypeConversionSpec> last;
    alias.count = slot->instance_count_;
    for (const auto& [i,stored]:slot->instances_) {
      const auto& instance = *stored;
      const auto& target = instance.result_types[0];
      if (!last || last->target != target) {
        last = std::make_shared<const TypeConversionSpec>(TypeConversionSpec{
            cast ? TypeConversionKind::Bitcast : TypeConversionKind::Convert,
            target, ValueType::scalar(target.element()),
            convert ? convert->attributes() : ConversionAttributes{}});
      }
      alias.entries.emplace(i, Storage::TypeAlias::Entry{instance.operands[0],last,false});
    }
    ++stats.nodes; stats.instances += alias.count;
    slot.reset();
    --storage_->live_nodes;
  }
  if (!stats.nodes) return stats;
  storage_->rewriting = true;
  // Template members need no wrapper: their operands are the prototype's
  // (instances_.at(0), resolved here) or its rebinding, which keeps node IDs.
  // Every resolved handle has a live slot, so resolveStorage returns each
  // member operand unchanged. A later lowering resolves the prototype again.
  for (auto& slot : storage_->nodes) {
    if (!slot) continue;
    for (auto& [index,instance] : slot->instances_)
      if (instance) for (auto& operand : instance->operands) operand = resolve(operand);
  }
  return stats;
}
std::string typeConversionLabel(const Value& value) {
  std::vector<const TypeConversion*> chain;
  for (auto current = &value; current->conversion(); current = &current->conversion()->source)
    chain.push_back(current->conversion().get());
  std::ostringstream out;
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    if (it != chain.rbegin()) out << "; ";
    const auto& spec = *(*it)->spec;
    out << (spec.kind == TypeConversionKind::Bitcast ? "Bitcast -> " : "Convert -> ") << spec.target.toString();
    if (spec.kind == TypeConversionKind::Convert) {
      std::ostringstream attrs;
      ConvertOp(spec.target, spec.attributes).printAttributes(attrs);
      if (!attrs.str().empty()) out << " (" << attrs.str() << ')';
    }
  }
  return out.str();
}

}  // namespace sentryward::value_ir
