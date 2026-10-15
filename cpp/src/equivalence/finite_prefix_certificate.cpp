#include "sentryward/equivalence/finite_prefix_certificate.h"
#include "sentryward/equivalence/epoch_work_allowance.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <limits>
#include <stdexcept>

namespace sentryward::equivalence::finite_prefix_certificate {
namespace {
Limits bounded(Limits v) {
  v.max_frames=std::min<std::size_t>(v.max_frames,256);
  v.max_states=std::min(v.max_states,epoch_work_allowance::maximum_retained_headers);
  v.max_cached_nodes=std::min<std::size_t>(v.max_cached_nodes,8192);
  v.max_relations=std::min<std::size_t>(v.max_relations,4096);
  v.max_work=std::min(v.max_work,epoch_work_allowance::maximum_work);return v;
}
unsigned arity(Kind k) {
  if(k==Kind::Add || k==Kind::Subtract || k==Kind::Multiply)return 2;
  if(k==Kind::Negate || k==Kind::NaturalExp)return 1;
  return 0;
}
template<class T> void hashPart(std::size_t& hash,const T& value) {
  const auto part=std::hash<T>{}(value);hash^=part+0x9e3779b9+(hash<<6)+(hash>>2);
}
template<class T> void hashPart(std::size_t& hash,const std::optional<T>& value) {
  hashPart(hash,bool(value));if(value)hashPart(hash,*value);
}
std::optional<std::uint64_t> point(std::uint64_t first,std::uint64_t stride,std::uint64_t ordinal) {
  std::uint64_t offset,result;
  if(__builtin_mul_overflow(stride,ordinal,&offset) || __builtin_add_overflow(first,offset,&result))return {};
  return result;
}
}
std::size_t Builder::NodeHash::operator()(const NodeKey& key) const {
  std::size_t hash=0;hashPart(hash,key.expr);hashPart(hash,key.scope);return hash;
}
std::size_t Builder::RelationHash::operator()(const RelationKey& key) const {
  std::size_t hash=0;std::apply([&](const auto&... item){(hashPart(hash,item),...);},key);return hash;
}
Builder::Builder(Resolve resolve,Relate relate,Limits v,WorkCharge shared):resolve_(std::move(resolve)),
    relate_(std::move(relate)),limits_(bounded(v)),charge_shared_(std::move(shared)) {
  static std::atomic<std::uint64_t> next{1};auto id=next.load(std::memory_order_relaxed);
  for(;;) {
    if(!id)throw std::overflow_error("finite-prefix certificate owner overflow");
    if(next.compare_exchange_weak(id,id+1,std::memory_order_relaxed)){owner_=id;break;}
  }
}
bool Builder::charge(std::size_t work,bool shared) {
  if(exhausted_ || statistics_.work>limits_.max_work || work>limits_.max_work-statistics_.work ||
      (shared && charge_shared_ && !charge_shared_(work))) {
    exhausted_=true;return false;
  }
  statistics_.work+=work;return true;
}
std::optional<Node> Builder::read(ExprId expr,ScopeId scope) {
  if(!charge(1))return {};
  const NodeKey key{expr,scope};
  if(auto found=nodes_.find(key);found!=nodes_.end()) {
    node_age_.splice(node_age_.end(),node_age_,found->second.age);return found->second.node;
  }
  if(!limits_.max_cached_nodes){exhausted_=true;return {};}
  ++statistics_.resolves;auto node=resolve_?resolve_(expr,scope):std::nullopt;
  if(node) {
    const bool atom=node->kind==Kind::Opaque;
    if(!node->finite_ideal || node->type!="real" || !node->provenance.empty() ||
        node->arity!=arity(node->kind) || node->free_count>node->free_indices.size() ||
        (!atom && !node->attributes.empty() && node->attributes!="ideal_arithmetic"))node.reset();
    if(node && node->exact_dependencies &&
        !std::is_sorted(node->free_indices.begin(),node->free_indices.begin()+node->free_count))node.reset();
  }
  if(nodes_.size()>=limits_.max_cached_nodes) {
    if(!charge())return {};nodes_.erase(node_age_.front());node_age_.pop_front();
  }
  node_age_.push_back(key);nodes_.emplace(key,CachedNode{node,std::prev(node_age_.end())});
  statistics_.peak_cached_nodes=std::max(statistics_.peak_cached_nodes,nodes_.size());return node;
}
bool Builder::dependencies(const Node& node,std::optional<std::uint32_t> axis) const {
  if(!axis)return true;
  return node.exact_dependencies && !std::binary_search(node.free_indices.begin(),
      node.free_indices.begin()+node.free_count,*axis);
}
bool Builder::related(const RelationRequest& request) {
  if(request.axis_count>request.axes.size() || !charge(1+request.axis_count))return false;
  for(unsigned i=0;i<request.axis_count;++i) {
    const auto& a=request.axes[i];
    if(!a.extent || a.extent-1>UINT64_MAX-a.offset)return false;
    for(unsigned j=0;j<i;++j) {
      const auto& b=request.axes[j];
      if(a.reference_axis==b.reference_axis ||
          (a.actual_axis && b.actual_axis && a.actual_axis==b.actual_axis))return false;
    }
  }
  const auto& a=request.axes[0];const auto& b=request.axes[1];
  const RelationKey key{request.reference,request.actual,request.reference_scope,request.actual_scope,
      a.reference_axis,a.actual_axis,a.offset,a.extent,
      b.reference_axis,b.actual_axis,b.offset,b.extent,request.axis_count};
  if(!charge())return false;
  if(auto found=relations_.find(key);found!=relations_.end()) {
    relation_age_.splice(relation_age_.end(),relation_age_,found->second.age);return found->second.proved;
  }
  if(!limits_.max_relations){exhausted_=true;return false;}
  ++statistics_.relations;auto bounded_request=request;
  bounded_request.max_work=limits_.max_work-statistics_.work;
  const auto result=relate_?relate_(bounded_request):RelationResult{};
  if(result.work>bounded_request.max_work){exhausted_=true;return false;}
  if(!charge(result.work,!charge_shared_))return false;
  const bool proved=result.status==RelationStatus::Proved;
  if(relations_.size()>=limits_.max_relations) {
    if(!charge())return false;relations_.erase(relation_age_.front());relation_age_.pop_front();
  }
  relation_age_.push_back(key);relations_.emplace(key,CachedRelation{proved,std::prev(relation_age_.end())});
  statistics_.peak_cached_relations=std::max(statistics_.peak_cached_relations,relations_.size());return proved;
}
std::optional<Builder::FrameId> Builder::begin(const FrameRequest& request) {
  if(!charge() || request.time.end<=request.time.begin)return {};
  if(frames_.size()>=limits_.max_frames){exhausted_=true;return {};}
  if(request.template_state_axis.has_value()!=request.actual_state_axis.has_value() ||
      (request.template_state_axis && (!request.state_extent ||
          *request.template_state_axis==request.template_time_axis)) ||
      (request.output_time_axis &&
          (*request.output_time_axis==request.template_time_axis ||
           request.template_state_axis==request.output_time_axis ||
           request.actual_state_axis==request.output_time_axis)))return {};
  const auto p=read(request.potential_template,request.template_scope);
  const auto q=read(request.injection_template,request.template_scope);
  const auto entry=read(request.entry_state,request.state_scope);
  const auto e=read(request.entry_potential,request.state_scope);
  if(!p || !q || !entry || !e || !dependencies(*p,request.template_state_axis) ||
      !dependencies(*p,request.output_time_axis) || !dependencies(*q,request.output_time_axis) ||
      !dependencies(*e,request.actual_state_axis) || !dependencies(*e,request.output_time_axis))return {};
  if(request.carry) {
    const auto carried=state(*request.carry);if(!carried)return {};
    const auto& previous=frames_[carried->frame].request;
    if(carried->endpoint!=previous.time.end-1 ||
        previous.actual_state_axis.has_value()!=request.actual_state_axis.has_value() ||
        previous.state_extent!=request.state_extent)return {};
    // A previously completed descriptor is the proof boundary, not a request
    // to traverse its whole predecessor history again. Different lexical
    // contexts still require the full scoped body relation below.
    if(carried->expression!=request.entry_state || previous.state_scope!=request.state_scope ||
        previous.actual_state_axis!=request.actual_state_axis) {
      RelationRequest relation{carried->expression,request.entry_state,previous.state_scope,request.state_scope};
      if(previous.actual_state_axis) {
        relation.axes[0]={*previous.actual_state_axis,request.actual_state_axis,0,request.state_extent};
        relation.axis_count=1;
      }
      if(!related(relation))return {};
    }
    if(previous.output_time_axis!=request.output_time_axis &&
        !dependencies(*entry,request.output_time_axis))return {};
  } else if(!dependencies(*entry,request.output_time_axis))return {};
  const auto id=frames_.size();frames_.push_back({request,{}});++statistics_.frames;
  FrameId result;result.owner=owner_;result.id=id;return result;
}
bool Builder::weight(ExprId expr,ExprId current,ExprId previous,ScopeId scope) {
  const auto node=read(expr,scope);if(!node)return false;
  if(node->kind==Kind::One)return current==previous;
  if(node->kind!=Kind::NaturalExp || node->children[0]==expr)return false;
  const auto argument=node->children[0];const auto difference=read(argument,scope);
  if(!difference)return false;
  // An existing ideal zero simplification may remove the explicit P - 0
  // argument even when the carried state is nonzero. Certify the ACTUAL
  // previous-potential node; the entry state is a separate update operand.
  if(argument==current) {
    const auto baseline=read(previous,scope);
    return baseline && baseline->kind==Kind::Zero;
  }
  if(difference->kind==Kind::Zero)return current==previous;
  if(difference->kind==Kind::Subtract)
    return difference->children[0]==current && difference->children[1]==previous;
  if(difference->kind!=Kind::Add)return false;
  const bool left=difference->children[0]==current,right=difference->children[1]==current;
  if(left==right)return false;
  const auto negative=difference->children[left?1:0];
  if(negative==argument || negative==expr)return false;
  const auto neg=read(negative,scope);
  return neg && neg->kind==Kind::Negate && neg->children[0]==previous;
}
bool Builder::scaled(ExprId expr,ExprId previous,ExprId current_p,ExprId previous_p,ScopeId scope) {
  if(expr==previous && current_p==previous_p)return true;
  const auto product=read(expr,scope);if(!product || product->kind!=Kind::Multiply)return false;
  const bool a=product->children[0]==previous,b=product->children[1]==previous;
  if(a==b)return false;
  const auto factor=product->children[a?1:0];
  if(factor==expr)return false;
  return weight(factor,current_p,previous_p,scope);
}
bool Builder::update(ExprId expr,ExprId previous,ExprId injection,ExprId current_p,
    ExprId previous_p,ScopeId scope,bool zero_entry) {
  // Existing ideal zero simplification can erase the entire first scaled arm.
  if(zero_entry && expr==injection)return true;
  const auto sum=read(expr,scope);if(!sum || sum->kind!=Kind::Add)return false;
  const bool a=sum->children[0]==injection,b=sum->children[1]==injection;
  if(a==b)return false;
  const auto scaled_state=sum->children[a?1:0];
  if(scaled_state==expr && scaled_state!=previous)return false;
  return scaled(scaled_state,previous,current_p,previous_p,scope);
}
std::optional<Builder::StateId> Builder::append(std::size_t frame,const UpdateRequest& request) {
  if(!charge(4))return {};
  auto& f=frames_[frame];std::optional<std::size_t> selected;
  if(f.tail_run) {
    auto& run=runs_[*f.tail_run];
    const auto next=point(run.first_endpoint,1,run.count);
    if(next && *next==request.endpoint) {
      if(run.count==1 && request.state>=run.first_expression && request.potential>=run.first_potential) {
        run.expression_stride=request.state-run.first_expression;
        run.potential_stride=request.potential-run.first_potential;selected=f.tail_run;
      } else if(run.count>1 && point(run.first_expression,run.expression_stride,run.count)==request.state &&
          point(run.first_potential,run.potential_stride,run.count)==request.potential)selected=f.tail_run;
    }
  }
  StateId result;result.owner=owner_;
  if(selected) {auto& run=runs_[*selected];result.run=*selected;result.ordinal=run.count++;}
  else {
    if(runs_.size()>=limits_.max_states){exhausted_=true;return {};}
    result.run=runs_.size();runs_.push_back({frame,request.state,request.potential,0,0,request.endpoint,1});
    f.tail_run=result.run;++statistics_.membership_runs;
  }
  ++statistics_.states;return result;
}
std::optional<Builder::State> Builder::state(StateId id) const {
  if(id.owner!=owner_ || id.run>=runs_.size())return {};
  const auto& run=runs_[id.run];if(id.ordinal>=run.count || run.frame>=frames_.size())return {};
  const auto expr=point(run.first_expression,run.expression_stride,id.ordinal);
  const auto potential=point(run.first_potential,run.potential_stride,id.ordinal);
  const auto endpoint=point(run.first_endpoint,1,id.ordinal);
  if(!expr || !potential || !endpoint)return {};
  return State{run.frame,*expr,*potential,*endpoint};
}
std::optional<Builder::StateId> Builder::advance(FrameId frame,std::optional<StateId> predecessor,
    const UpdateRequest& request) {
  if(frame.owner!=owner_ || frame.id>=frames_.size() || !charge())return {};
  const auto& f=frames_[frame.id].request;
  if(request.endpoint<f.time.begin || request.endpoint>=f.time.end)return {};
  ExprId previous=f.entry_state,previous_p=f.entry_potential;
  if(predecessor) {
    const auto prior=state(*predecessor);if(!prior)return {};
    if(prior->frame!=frame.id || prior->endpoint==UINT64_MAX || prior->endpoint+1!=request.endpoint)return {};
    previous=prior->expression;previous_p=prior->potential;
  } else if(request.endpoint!=f.time.begin)return {};
  const auto p=read(request.potential,f.state_scope);
  const auto q=read(request.injection,f.state_scope);
  const auto selected_state=read(request.state,f.state_scope);
  if(!p || !q || !selected_state || !dependencies(*p,f.actual_state_axis) ||
      !dependencies(*p,f.output_time_axis) || !dependencies(*q,f.output_time_axis))return {};
  RelationRequest potential{f.potential_template,request.potential,f.template_scope,f.state_scope};
  potential.axes[0]={f.template_time_axis,std::nullopt,request.endpoint,1};potential.axis_count=1;
  if(f.template_state_axis) {
    potential.axes[1]={*f.template_state_axis,f.actual_state_axis,0,f.state_extent};potential.axis_count=2;
  }
  auto injection=potential;injection.reference=f.injection_template;injection.actual=request.injection;
  if(!related(potential) || !related(injection))return {};
  const auto entry=read(previous,f.state_scope);if(!entry)return {};
  const bool zero_entry=!predecessor && entry->kind==Kind::Zero;
  if(!update(request.state,previous,request.injection,request.potential,previous_p,f.state_scope,zero_entry))return {};
  ++statistics_.updates;return append(frame.id,request);
}
std::optional<Builder::Signature> Builder::signature(StateId state) {
  if(!charge())return {};const auto value=this->state(state);if(!value)return {};
  Signature result;result.owner=owner_;result.frame=value->frame;result.endpoint=value->endpoint;return result;
}
std::optional<bool> Builder::equal(const Signature& a,const Signature& b) {
  if(a.owner!=owner_ || b.owner!=owner_ || !charge())return {};
  return a.frame==b.frame && a.endpoint==b.endpoint;
}
std::optional<Builder::CompletedPrefixFamilyId> Builder::complete(StateId terminal) {
  if(!charge())return {};
  const auto last=state(terminal);if(!last)return {};
  const auto& frame=frames_[last->frame].request;
  if(last->endpoint!=frame.time.end-1)return {};
  CompletedPrefixFamilyId result;result.owner=owner_;result.terminal=terminal;return result;
}
std::optional<Builder::FrameView> Builder::view(FrameId id) {
  if(!charge() || id.owner!=owner_ || id.id>=frames_.size())return {};
  const auto& f=frames_[id.id].request;
  return FrameView{f.potential_template,f.injection_template,f.entry_state,f.entry_potential,
      f.template_scope,f.state_scope,f.time,f.template_time_axis,f.template_state_axis,
      f.actual_state_axis,f.output_time_axis,f.state_extent,f.carry};
}
std::optional<Builder::StateView> Builder::view(StateId id) {
  if(!charge())return {};const auto value=state(id);if(!value)return {};
  FrameId frame;frame.owner=owner_;frame.id=value->frame;
  return StateView{frame,value->expression,value->potential,value->endpoint};
}
std::optional<Builder::CompletedPrefixView> Builder::view(CompletedPrefixFamilyId id) {
  if(!charge() || id.owner!=owner_)return {};
  const auto last=state(id.terminal);if(!last)return {};
  const auto& request=frames_[last->frame].request;
  if(last->endpoint!=request.time.end-1)return {};
  FrameId frame;frame.owner=owner_;frame.id=last->frame;
  const auto definition=view(frame);if(!definition)return {};
  return CompletedPrefixView{frame,*definition,id.terminal};
}
Statistics Builder::statistics() const{return statistics_;}
bool Builder::exhausted() const{return exhausted_;}
} // namespace sentryward::equivalence::finite_prefix_certificate
