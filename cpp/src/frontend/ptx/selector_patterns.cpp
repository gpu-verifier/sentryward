#include "sentryward/frontend/ptx/selector_patterns.h"
#include "sentryward/value_ir/operations.h"
#include <bit>
#include <limits>
#include <numeric>

namespace sentryward::frontend::ptx {
namespace {
namespace v=value_ir;namespace a=algorithm;
using Piece=SelectorPatterns::Piece;using Pieces=std::vector<Piece>;using I=__int128;
void Check(bool yes){if(!yes)throw std::invalid_argument("thread selector has no supported exact index pattern");}
I Last(const Piece& p){return p.first+p.step*I(p.threads.count-1);}
// Pieces hold unsigned representatives: a piece of a signed `element` operand
// must stay below its sign bit. Always true for other elements.
bool BelowSignBit(const Piece& p,const v::ElementType& element) {
  return element.kind()!=v::ElementKind::SignedInteger || std::max(p.first,Last(p))<(I(1)<<(element.width()-1));
}
Pieces Coalesce(Pieces pieces) {
  for(auto& piece:pieces)if(piece.threads.count==1)piece.threads.step=1;
  std::sort(pieces.begin(),pieces.end(),[](const Piece& x,const Piece& y) {
    return std::tuple{x.threads.step,x.threads.first%x.threads.step,x.threads.first}<
        std::tuple{y.threads.step,y.threads.first%y.threads.step,y.threads.first};
  });
  Pieces result;
  for(auto piece:pieces) {
    if(!result.empty()) {
      auto& last=result.back();
      if(last.threads.step==piece.threads.step && last.step==piece.step &&
          last.threads.first+last.threads.step*last.threads.count==piece.threads.first &&
          last.first+last.step*last.threads.count==piece.first) {
        last.threads.count+=piece.threads.count;continue;
      }
    }
    result.push_back(piece);
  }
  return result;
}
Piece Restrict(Piece p,std::uint64_t begin,std::uint64_t step,std::uint64_t count) {
  return {{p.threads.first+p.threads.step*begin,p.threads.step*step,count},p.first+p.step*begin,p.step*step};
}
Pieces Residues(Piece p,std::uint64_t modulus,bool quotient) {
  Check(modulus && p.first>=0 && Last(p)>=0);
  if(!p.step) {p.first=quotient?p.first/modulus:p.first%modulus;return {p};}
  // A fixed quotient needs no partition (usual coordinate/index arithmetic).
  if(p.first/modulus==Last(p)/modulus) {
    if(quotient){p.first/=modulus;p.step=0;}else p.first%=modulus;return {p};
  }
  const auto magnitude=std::uint64_t(p.step<0?-p.step:p.step);
  const auto period=modulus/std::gcd(modulus,magnitude);
  Pieces result;
  const auto first_quotient=p.first/modulus,last_quotient=Last(p)/modulus;
  const auto quotient_count=1+(last_quotient>first_quotient?last_quotient-first_quotient:first_quotient-last_quotient);
  if(quotient_count<I(period)) {
    for(std::uint64_t begin=0;begin<p.threads.count;) {
      const auto value=p.first+p.step*begin,q=value/modulus;
      const auto available=p.step>0?((q+1)*modulus-1-value)/p.step+1:(value-q*modulus)/(-p.step)+1;
      const auto count=std::min<std::uint64_t>(p.threads.count-begin,std::uint64_t(available));
      auto part=Restrict(p,begin,1,count);
      part.first=quotient?q:value%modulus;part.step=quotient?0:p.step;
      result.push_back(part);begin+=count;
    }
    return result;
  }
  for(std::uint64_t r=0;r<std::min(period,p.threads.count);++r) {
    auto part=Restrict(p,r,period,1+(p.threads.count-1-r)/period);
    part.first=quotient?part.first/modulus:part.first%modulus;
    part.step=quotient?part.step/modulus:0;result.push_back(part);
  }
  return result;
}
I FloorDiv(I a,I b){I q=a/b;if((a%b!=0) && ((a<0)!=(b<0)))--q;return q;}
// An affine difference d over one progression changes sign at most once, so
// each integer comparison of it is constant between its root neighbours.
Pieces Compare(Piece d,v::ComparePredicate predicate) {
  const auto holds=[&](I value) {
    switch(predicate) {
      case v::ComparePredicate::Equal:return value==0;
      case v::ComparePredicate::NotEqual:return value!=0;
      case v::ComparePredicate::Less:return value<0;
      case v::ComparePredicate::LessEqual:return value<=0;
      case v::ComparePredicate::Greater:return value>0;
      case v::ComparePredicate::GreaterEqual:return value>=0;
      default:Check(false);return false;
    }
  };
  const auto n=d.threads.count;
  std::vector<std::uint64_t> cuts{0,n};
  if(d.step) {
    const I root=FloorDiv(-d.first,d.step);
    for(I cut:{root,root+1,root+2})if(cut>0 && cut<I(n))cuts.push_back(std::uint64_t(cut));
  }
  std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
  Pieces result;
  for(std::size_t i=0;i+1<cuts.size();++i) {
    auto part=Restrict(d,cuts[i],1,cuts[i+1]-cuts[i]);
    part.first=holds(part.first)?1:0;part.step=0;result.push_back(part);
  }
  return result;
}
Pieces BitMask(Piece p,std::uint64_t mask,std::uint64_t width_mask) {
  mask&=width_mask;
  if(!mask){p.first=0;p.step=0;return {p};}
  if(mask==width_mask)return {p};
  const bool complement=std::popcount(mask)>std::popcount(width_mask^mask);
  const auto sparse=complement?width_mask^mask:mask;
  const auto bits=std::bit_width(sparse);
  // Coordinates fit uint32_t; a bit above their proven range is constant.
  const auto high=std::max(p.first,Last(p));
  if(I(sparse)>high && (sparse&(sparse-1))==0) {
    if(!complement){p.first=0;p.step=0;}return {p};
  }
  Check(bits<64);
  const auto period=(std::uint64_t(1)<<bits)/std::gcd(std::uint64_t(p.step<0?-p.step:p.step),std::uint64_t(1)<<bits);
  Pieces result;
  for(std::uint64_t r=0;r<std::min(period,p.threads.count);++r) {
    auto part=Restrict(p,r,period,1+(p.threads.count-1-r)/period);
    const auto selected=std::uint64_t(part.first)&sparse;
    if(complement)part.first-=selected;
    else {part.first=selected;part.step=0;}
    result.push_back(part);
  }
  return result;
}
}
SelectorPatterns::SelectorPatterns(const v::Graph& graph):graph_(graph) {
  graph.forEachThreadBinding([&](const v::SourceOrigin& origin) {
    if(!origin.cta || !origin.thread || !origin.thread_coordinates)return;
    for(unsigned axis=0;axis<3;++axis) {
      auto& pieces=coordinates_[{*origin.cta,axis}];
      const I value=origin.thread_coordinates->at(axis);
      if(!pieces.empty()) {
        auto& last=pieces.back();
        if(*origin.thread==last.threads.first+last.threads.count) {
          if(last.threads.count==1)last.step=value-last.first;
          if(value==last.first+last.step*last.threads.count){++last.threads.count;continue;}
        }
      }
      pieces.push_back({{*origin.thread,1,1},value,0});
    }
  });
}
std::vector<SelectorPatterns::Piece> SelectorPatterns::evaluate(v::Value value,std::uint64_t cta,
    std::uint64_t base,a::AffineDomain threads,bool rebind) const {
  Cache cache;
  try {return evaluate(value,cta,base,threads,rebind,0,cache);}
  catch(const std::invalid_argument& error) {
    throw std::invalid_argument(std::string(error.what())+"; reader_cta="+std::to_string(cta)+
        ", reader_base="+std::to_string(base)+", domain=("+std::to_string(threads.first)+","+
        std::to_string(threads.step)+","+std::to_string(threads.count)+")");
  }
}
std::vector<SelectorPatterns::Piece> SelectorPatterns::evaluate(v::Value value,std::uint64_t cta,
    std::uint64_t base,a::AffineDomain threads,bool rebind,unsigned depth,Cache& cache) const {
  if(!threads.count)return {};
  value=graph_.resolve(value);
  // Cache lifetime is one public query; no graph mutation invalidation and no
  // permanent retention of every caller domain. Full Value equality includes
  // conversion and selector interpretations; node identity alone is not a key.
  const auto hash=value.nodeId() ^ (threads.first*131+threads.step*17+threads.count) ^ (base*67+cta);
  if(auto found=cache.find(hash);found!=cache.end())for(const auto& entry:found->second)
    if(entry.value==value && entry.cta==cta && entry.base==base && entry.rebind==rebind &&
        entry.threads.first==threads.first && entry.threads.step==threads.step && entry.threads.count==threads.count)
      return entry.result;
  Pieces result;
  try {result=evaluateImpl(value,cta,base,threads,rebind,depth,cache);}
  catch(const std::invalid_argument& error) {
    // Annotate the first failing expression only, not its entire ancestor DAG.
    const std::string message=error.what();
    if(message.find("selector_node=")!=std::string::npos)throw;
    const auto origin=graph_.instance(value).origin;
    throw std::invalid_argument(message+"; selector_node="+std::to_string(value.nodeId())+
        ", instance="+std::to_string(value.instanceIndex())+", operation="+
        std::string(graph_.definingNode(value).operation().name())+
        ", instruction="+(origin.instruction_index?std::to_string(*origin.instruction_index):"none")+
        ", nested="+std::to_string(bool(value.threadSelector())));
  }
  cache[hash].push_back({value,cta,base,threads,rebind,result});
  return result;
}
std::vector<SelectorPatterns::Piece> SelectorPatterns::evaluateImpl(v::Value value,std::uint64_t cta,
    std::uint64_t base,a::AffineDomain threads,bool rebind,unsigned depth,Cache& cache) const {
  Check(depth<128 && value.indices().empty());
  value=graph_.resolve(value);
  const auto& type=value.type(graph_);
  // Predicates are 0/1 (shuffle validity, lane tests); compare and select them exactly.
  const bool predicate=type.isScalar() && type.element().kind()==v::ElementKind::Predicate;
  Check(type.isScalar() && (type.element().isInteger() || predicate));
  const auto width=predicate?1u:type.element().width();Check(width && width<=64);
  const auto mask=width==64?UINT64_MAX:(std::uint64_t(1)<<width)-1;
  const auto normalize=[&](Pieces parts) {
    Pieces result;
    for(auto part:parts) {
      const auto low=std::min(part.first,Last(part)),high=std::max(part.first,Last(part));
      if(low>=0 && high<=I(mask)){result.push_back(part);continue;}
      Check(width<64);
      const I modulus=I(mask)+1;
      if(low<0){const I add=((-low+modulus-1)/modulus)*modulus;part.first+=add;}
      auto normalized=Residues(part,std::uint64_t(modulus),false);
      result.insert(result.end(),normalized.begin(),normalized.end());
    }
    return Coalesce(std::move(result));
  };
  if(value.conversion()) {
    Check(value.conversion()->source.type(graph_).element().isInteger());
    // Width-preserving reinterpretations and unsigned truncations are exact;
    // signed extension needs a separate pattern, never silently erase it.
    Check(value.conversion()->spec->kind==v::TypeConversionKind::Bitcast ||
        width<=value.conversion()->source.type(graph_).element().width());
    auto source=value.conversion()->source;
    if(value.threadSelector())source=graph_.fromThread(source,*value.threadSelector());
    return normalize(evaluate(source,cta,base,threads,rebind,depth+1,cache));
  }
  const auto& node=graph_.definingNode(value);
  if(value.threadSelector()) {
    // Keep the reader coordinate while evaluating E(S(reader)). In particular,
    // a many-to-one image must not collapse the reader domain to one member.
    auto selectors=evaluate(*value.threadSelector(),cta,base,threads,rebind,depth+1,cache);
    Pieces result;
    for(const auto& selection:selectors) {
      Check(selection.first>=0 && Last(selection)>=0 &&
          selection.first<=UINT64_MAX && Last(selection)<=UINT64_MAX && selection.step>=0);
      const a::AffineDomain image{std::uint64_t(selection.first),std::uint64_t(selection.step),
          selection.step?selection.threads.count:1};
      std::uint64_t covered=0;
      node.forEachInstancePattern([&](const auto&,std::size_t begin,std::size_t count,bool parameterized) {
        const auto origin=node.instanceOrigin(begin);
        Check(origin.cta==cta && origin.thread.has_value());
        const auto& members=node.instanceThreads();
        for(std::size_t at=begin;at<begin+count;) {
          a::AffineDomain domain;
          std::size_t end;
          if(members.empty()) {
            Check(begin==0 && count==1);domain={*origin.thread,1,1};end=1;
          } else {
            const auto segment=members.affineSegment(at);Check(segment.has_value() && segment->stride>=0);
            end=std::min(begin+count,segment->begin+segment->count);
            domain={members.at(at),std::uint64_t(segment->stride),end-at};
          }
          const auto match=a::IntersectAffine(image,domain);
          if(match.count) {
            const a::AffineDomain selected{image.first+image.step*match.left,
                match.count>1?image.step*match.left_step:1,match.count};
            // Use the actual operand pattern and its own base thread, not the
            // consumer's representative. Explicit exceptional instances retain
            // their fixed operands; template ranges rebind only same-thread uses.
            const auto bare=node.result(value.resultIndex(),begin);
            auto values=evaluate(bare,cta,*origin.thread,selected,parameterized,depth+1,cache);
            std::uint64_t piece_covered=0;
            for(const auto& part:values) {
              const auto pull=a::IntersectAffine(image,part.threads);
              if(!pull.count)continue;
              if(!selection.step) {
                result.push_back({selection.threads,part.first+part.step*pull.right,0});
                piece_covered+=selection.threads.count;
              } else {
                result.push_back({{selection.threads.first+selection.threads.step*pull.left,
                    pull.count>1?selection.threads.step*pull.left_step:1,pull.count},
                    part.first+part.step*pull.right,part.step*pull.right_step});
                piece_covered+=pull.count;
              }
            }
            Check(piece_covered==(!selection.step?selection.threads.count:match.count));
            covered+=piece_covered;
          }
          at=end;
        }
      });
      Check(covered==selection.threads.count);
    }
    return Coalesce(std::move(result));
  }
  if(auto* constant=dynamic_cast<const v::ConstantOp*>(&node.operation())) {
    if(predicate) {
      const auto& literal=constant->literal();
      Check(literal=="true" || literal=="false" || literal=="1" || literal=="0");
      return {{threads,I(literal=="true" || literal=="1"),0}};
    }
    return normalize({{threads,I(std::stoll(constant->literal(),nullptr,0)),0}});
  }
  const auto instance=graph_.instance(value);
  bool variable=rebind && instance.origin.cta==cta && instance.origin.thread==base;
  if(auto* coordinate=dynamic_cast<const v::CoordinateOp*>(&node.operation())) {
    const auto axis=unsigned(coordinate->axis());
    if(coordinate->kind()==v::CoordinateKind::ThreadIndex) {
      if(!variable){Check(bool(instance.origin.thread_coordinates));return {{threads,instance.origin.thread_coordinates->at(axis),0}};}
      auto found=coordinates_.find({cta,axis});Check(found!=coordinates_.end());Pieces result;
      for(auto p:found->second)if(auto m=a::IntersectAffine(threads,p.threads);m.count) {
        result.push_back({{threads.first+threads.step*m.left,threads.step*m.left_step,m.count},p.first+p.step*m.right,p.step*m.right_step});
      }
      return result;
    }
    if(coordinate->kind()==v::CoordinateKind::BlockIndex) {
      Check(bool(instance.origin.cta_coordinates));return {{threads,instance.origin.cta_coordinates->at(axis),0}};
    }
  }
  if(dynamic_cast<const v::SelectOp*>(&node.operation())) {
    Check(instance.operands.size()==3);
    Pieces result;
    for(const auto& condition:evaluate(instance.operands[0],cta,base,threads,variable,depth+1,cache))
      for(std::uint64_t member=0;member<condition.threads.count;) {
        // A 0/1 progression is constant unless it has two members.
        const auto count=condition.step?1:condition.threads.count;
        const auto part=Restrict(condition,member,1,count);
        auto chosen=evaluate(instance.operands[part.first?1:2],cta,base,part.threads,variable,depth+1,cache);
        result.insert(result.end(),chosen.begin(),chosen.end());member+=count;
      }
    return Coalesce(std::move(result));
  }
  if(auto* compare=dynamic_cast<const v::CompareOp*>(&node.operation())) {
    Check(instance.operands.size()==2);
    const auto& element=instance.operands[0].type(graph_).element();
    Check(element.isInteger() && instance.operands[1].type(graph_).element()==element);
    auto left=evaluate(instance.operands[0],cta,base,threads,variable,depth+1,cache);
    auto right=evaluate(instance.operands[1],cta,base,threads,variable,depth+1,cache);Pieces result;
    for(auto l:left)for(auto r:right)if(auto m=a::IntersectAffine(l.threads,r.threads);m.count) {
      auto x=Restrict(l,m.left,m.left_step,m.count),y=Restrict(r,m.right,m.right_step,m.count);
      Check(BelowSignBit(x,element) && BelowSignBit(y,element));
      x.first-=y.first;x.step-=y.step;
      auto parts=Compare(x,compare->predicate());result.insert(result.end(),parts.begin(),parts.end());
    }
    return Coalesce(std::move(result));
  }
  const auto* binary=dynamic_cast<const v::BinaryOp*>(&node.operation());Check(binary && instance.operands.size()==2);
  Check(!binary->attributes().saturate && binary->attributes().multiply_mode!=v::IntegerMultiplyMode::High);
  auto left=evaluate(instance.operands[0],cta,base,threads,variable,depth+1,cache);
  auto right=evaluate(instance.operands[1],cta,base,threads,variable,depth+1,cache);Pieces result;
  for(auto l:left)for(auto r:right)if(auto m=a::IntersectAffine(l.threads,r.threads);m.count) {
    auto x=Restrict(l,m.left,m.left_step,m.count),y=Restrict(r,m.right,m.right_step,m.count);
    Piece out=x;Pieces parts;
    const auto nonnegative_signed=[&](const Piece& part,v::Value operand) {
      return BelowSignBit(part,operand.type(graph_).element());
    };
    switch(binary->opcode()) {
      case v::BinaryOpcode::Add:out.first+=y.first;out.step+=y.step;break;
      case v::BinaryOpcode::Subtract:out.first-=y.first;out.step-=y.step;break;
      case v::BinaryOpcode::Multiply:
        Check(!x.step || !y.step);
        Check(width<=instance.operands[0].type(graph_).element().width() ||
            (nonnegative_signed(x,instance.operands[0]) && nonnegative_signed(y,instance.operands[1])));
        // Selector arithmetic cannot overflow the interpreter's signed wide
        // temporary. Fail explicitly instead of silently misbinding threads.
        Check(std::max(x.first,Last(x))<=(I(1)<<63)-1 && std::max(y.first,Last(y))<=(I(1)<<63)-1);
        out.first=x.first*y.first;out.step=x.step*y.first+y.step*x.first;break;
      case v::BinaryOpcode::Divide:case v::BinaryOpcode::Remainder:
        Check(nonnegative_signed(x,instance.operands[0]) && nonnegative_signed(y,instance.operands[1]));
        Check(!y.step && y.first>0 && y.first<=UINT64_MAX);parts=Residues(x,std::uint64_t(y.first),binary->opcode()==v::BinaryOpcode::Divide);break;
      case v::BinaryOpcode::ShiftLeft:case v::BinaryOpcode::ShiftRight:
        Check(!y.step && y.first>=0 && y.first<64);
        if(binary->opcode()==v::BinaryOpcode::ShiftLeft){out.first<<=unsigned(y.first);out.step*=I(1)<<unsigned(y.first);}
        else {Check(nonnegative_signed(x,instance.operands[0]));parts=Residues(x,std::uint64_t(1)<<unsigned(y.first),true);}break;
      case v::BinaryOpcode::BitAnd:case v::BinaryOpcode::BitOr:case v::BinaryOpcode::BitXor: {
        if(y.step && !x.step)std::swap(x,y);Check(!y.step);
        parts=BitMask(x,std::uint64_t(y.first),mask);
        if(binary->opcode()!=v::BinaryOpcode::BitAnd)for(auto& part:parts) {
          auto t=a::IntersectAffine(part.threads,x.threads);
          auto original=Restrict(x,t.right,t.right_step,t.count);
          const I factor=binary->opcode()==v::BinaryOpcode::BitXor?2:1;
          part.first=original.first+y.first-factor*part.first;part.step=original.step-factor*part.step;
        }
        break;
      }
      default:Check(false);
    }
    if(parts.empty())parts.push_back(out);
    auto normalized=normalize(std::move(parts));result.insert(result.end(),normalized.begin(),normalized.end());
  }
  return Coalesce(std::move(result));
}
}
