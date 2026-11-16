#pragma once
#include "importer.hpp"

namespace rg {

inline auto Importer::source_layout_differs(const std::string &raw) -> bool {
  if(auto known=byte_layout_differs.find(raw);known!=byte_layout_differs.end())return known->second;
  bool differs=false;
  const auto &other=mapping.at(side=="left"?"right_inputs":"left_inputs");
  const std::string shared=mapped_name(m,"allocations",raw);
  if(m.contains("layouts") && m["layouts"].contains(raw) && other.contains("layouts"))
    for(auto it=other["layouts"].begin();it!=other["layouts"].end();++it)
      if(mapped_name(other,"allocations",it.key())==shared) {
        const auto &mine=m["layouts"][raw],&theirs=it.value();
        differs=differs || mine["source_strides_bytes"]!=theirs["source_strides_bytes"] ||
                jq(mine.value("source_offset",J(0)))!=jq(theirs.value("source_offset",J(0)));
      }
  byte_layout_differs.emplace(raw,differs);
  return differs;
}

// The canonical address of a view of `bytes_read` bytes at raw `address`: target_offset +
// (offset mod element_bytes) + sum_j target_stride_j * ((offset / stride_j) mod extent_j), with
// offset = address - source_offset. Exact for a dense permuted source layout when the offset lies
// in the layout's extent and the view stays inside one element (both proved from bounds); an
// identity layout only moves the offset. O(rank) Arena terms.
inline auto Importer::canonical_byte_address(const std::string &raw,Id address,const Z &bytes_read) -> std::optional<Id> {
  const auto &l=m["layouts"][raw];
  const auto &target=l.contains("target_strides_bytes")?l["target_strides_bytes"]:l["source_strides_bytes"];
  const Id off=a.sub(address,a.constant(jq(l.value("source_offset",J(0)))));
  const Id target_offset=a.constant(jq(l.value("target_offset",J(0))));
  if(target==l["source_strides_bytes"])return a.add({off,target_offset});
  if(!packed_dense_layouts.contains(raw) || nested_layouts.contains(raw))return {};
  const Z bytes=l["element_bytes"].get<unsigned>();
  Z extent=bytes;
  for(size_t j=0;j<l["shape"].size();++j)
    extent+=(jq(l["shape"][j]).numerator()-1)*jq(l["source_strides_bytes"][j]).numerator();
  const auto bounds=a.nodes[off].bounds;
  if(!bounds || bounds->first<0 || bounds->second+Q(bytes_read)>Q(extent))return {};
  const Id within=a.imod(off,bytes);
  const auto inside=a.nodes[within].bounds;
  if(!inside || inside->first<0 || inside->second+Q(bytes_read)>Q(bytes))return {};
  std::vector<Id> terms{target_offset,within};
  for(size_t j=0;j<l["shape"].size();++j) {
    // A shape-1 axis has coordinate 0; its stride is not checked and may be 0.
    if(jq(l["shape"][j])==1)continue;
    const Id coordinate=a.imod(a.idiv(off,jq(l["source_strides_bytes"][j]).numerator()),
                               jq(l["shape"][j]).numerator());
    terms.push_back(a.mul({a.constant(jq(target[j])),coordinate}));
  }
  ++a.stats["canonical_byte_view_layout_mappings"];
  return a.add(terms);
}

// Whether the other side lays out some input of this side differently. Computed once per side.
inline auto Importer::some_byte_layout_differs() -> bool {
  if(!any_byte_layout_differs) {
    bool any=false;
    if(m.contains("layouts"))
      for(auto it=m["layouts"].begin();it!=m["layouts"].end();++it)any=source_layout_differs(it.key()) || any;
    any_byte_layout_differs=any;
  }
  return *any_byte_layout_differs;
}

// Refuses an output value whose cone holds a raw byte view of such an input; returns the value
// otherwise. Every return of family() passes here. The scan visits each Arena node at most once
// per side (byte_view_scanned).
inline auto Importer::refuse_raw_byte_views(Id n) -> Id {
  if(!some_byte_layout_differs())return n;
  std::vector<Id> todo{n};
  while(!todo.empty()) {
    const Id x=todo.back();todo.pop_back();
    if(byte_view_scanned.size()<=x)byte_view_scanned.resize(a.nodes.size(),0);
    if(byte_view_scanned[x])continue;
    byte_view_scanned[x]=1;a.check();
    const auto &node=a.nodes[x];
    if(node.op=="readbits" && source_layout_differs(node.tag))
      throw Unsupported("A raw byte view of an input laid out differently on the two sides reaches an output",
                        "byte_view_layout_unsupported",
                        {{"allocation",mapped_name(m,"allocations",node.tag)},{"expression",a.render(x,3)}});
    todo.insert(todo.end(),node.args.begin(),node.args.end());
  }
  return n;
}

// An output that is a byte view of such an input is read at its canonical address under the
// shared input name; both sides apply the same rule. Any other raw view of such an input in the
// output is refused. O(rank) for the root, then refuse_raw_byte_views.
inline auto Importer::byte_view_output(Id n) -> Id {
  if(!some_byte_layout_differs())return n;
  const auto &root=a.nodes[n];
  if(root.op=="readbits" && root.args.size()==1 && root.data.size()==1 && source_layout_differs(root.tag) &&
     root.data[0].denominator()==1 && root.data[0]>0 && modz(root.data[0].numerator(),8)==0) {
    const std::string raw=root.tag;const Id address=root.args[0];const Q bits=root.data[0];
    const auto canonical=canonical_byte_address(raw,address,bits.numerator()/8);
    if(!canonical)
      throw Unsupported("A byte-view output of an input laid out differently on the two sides has no proved "
                        "canonical address","byte_view_layout_unsupported",
                        {{"allocation",mapped_name(m,"allocations",raw)},{"bits",str(bits)},
                         {"address_expression",a.render(address,3)}});
    n=a.intern("readbits",{*canonical},{bits},"canonical:"+mapped_name(m,"allocations",raw),
               {true,false,true},std::pair(Q(0),Q(pow2(bits.numerator().convert_to<unsigned>())-1)));
  }
  return refuse_raw_byte_views(n);
}

// Verified fixed read-only inputs (shape arrays) hold their declared values for every read of
// either side: the native run proved that the kernel never writes them.
// A byte view of exactly one such element at a constant offset is that element's stored bits,
// its value modulo 2^bits. Coordinates use the side's source layout, as typed_read does. O(rank).
inline auto Importer::fixed_bits(const std::string &raw,Id offset,unsigned bits) -> std::optional<Id> {
  if(!context || context->fixed_inputs.empty() || !m.contains("layouts") || !m["layouts"].contains(raw))return {};
  const auto &l=m["layouts"][raw];
  const unsigned bytes=l["element_bytes"].get<unsigned>();
  const auto address=a.value(offset);
  if(bits!=8*bytes || !address || address->denominator()!=1)return {};
  const Z off=address->numerator()-jq(l.value("source_offset",J(0))).numerator();
  Z extent=bytes,reconstructed=0;
  for(size_t j=0;j<l["shape"].size();++j)
    extent+=(jq(l["shape"][j]).numerator()-1)*jq(l["source_strides_bytes"][j]).numerator();
  if(off<0 || off+bytes>extent || modz(off,bytes)!=0)return {};
  std::vector<Id> coordinates;
  for(size_t j=0;j<l["shape"].size();++j) {
    // A shape-1 axis has coordinate 0.
    const Z count=jq(l["shape"][j]).numerator(),stride=jq(l["source_strides_bytes"][j]).numerator();
    const Z coordinate=count==1 ? Z(0) : modz(floorz(off,stride),count);
    reconstructed+=coordinate*stride;
    coordinates.push_back(a.constant(Q(coordinate)));
  }
  if(reconstructed!=off)return {};
  const std::string alloc=mapped_name(m,"allocations",raw);
  const auto range=ranges.find(alloc);
  for(const std::string type:{"s"+std::to_string(bits),"u"+std::to_string(bits)})
    if(auto fixed=context->fixed_read(a,alloc,type,coordinates,side!="left")) {
      // Input-contract intervals are checked against fixed values when the context is prepared.
      require_fixed_in_range(alloc,*fixed,range!=ranges.end() ? range->second : Bounds{});
      return a.constant(Q(modz(a.nodes[*fixed].data[0].numerator(),pow2(bits))));
    }
  return {};
}

inline auto Importer::unrolls_fixed_table_fold(Id raw) -> bool {
  if(!context || context->fixed_inputs.empty() || ns[raw].count>fixed_table_unroll_limit ||
     ns[raw].operands.size()!=2 || !unrollable_reducer(ns[raw].reducer))return false;
  if(!fixed_read_levels) {
    std::vector<std::vector<Id>> levels(ns.size());
    for(size_t i=0;i<ns.size();++i) {
      const auto &n=ns[i];
      std::set<Id> s;
      for(size_t j=0;j<n.operands.size();++j)
        for(Id v:levels[n.operands[j].node])
          if(!(n.kind=="Fold" && j==1 && v==n.level))s.insert(v);
      if((n.kind=="InputValue" || n.kind=="InitialCell") && n.operands.size()==1 &&
         context->fixed_input_types.contains(mapped_name(m,"allocations",n.name)))
        s.insert(free[n.operands[0].node].begin(),free[n.operands[0].node].end());
      levels[i].assign(s.begin(),s.end());
    }
    fixed_read_levels=std::move(levels);
  }
  const auto &n=ns[raw];
  const auto &element=(*fixed_read_levels)[n.operands[1].node];
  return std::binary_search(element.begin(),element.end(),n.level);
}

inline auto Importer::unrolls_stacked_fold(Id raw) const -> bool {
  const auto &n=ns[raw];
  if(n.operands.size()!=2 || n.count>stacked_fold_unroll_limit || !unrollable_reducer(n.reducer))return false;
  auto independent=[&](Id node){return !std::binary_search(free[node].begin(),free[node].end(),n.level);};
  Id node=n.operands[1].node;
  for(bool first=true;;first=false) {
    const auto &select=ns[node];
    if(!first && independent(node))return true;
    if(select.kind!="Operation" || select.name!="Select" || !select.attributes.empty() || select.operands.size()!=3)return false;
    const auto &equal=ns[select.operands[0].node];
    if(equal.kind!="Operation" || equal.name!="Equal" || equal.operands.size()!=2)return false;
    bool binder=false,constant=false;
    for(const auto &side:equal.operands) {
      const auto &x=ns[side.node];
      if(x.kind=="Index" && x.level==n.level)binder=true;
      else if(x.kind=="Constant")constant=true;
    }
    if(!binder || !constant || !independent(select.operands[1].node))return false;
    node=select.operands[2].node;
  }
}

// The read R with every occurrence replaced by `to`, rebuilt through the exact Arena constructors of
// add, mul, exp/exp2 and recip; nodes without R are kept. Empty when R sits under another operation
// or the walk exceeds argmax_replace_visit_limit. A post-order walk on an explicit stack (no
// recursion depth), each node rebuilt once: O(min(|root DAG|, limit)).
inline auto Importer::replace_read(Id root,Id from,Id to) -> std::optional<Id> {
  if(root==from)return to;
  std::unordered_map<Id,Id> rebuilt;
  std::vector<std::pair<Id,bool>> todo{{root,false}};
  size_t visits=0;
  while(!todo.empty()) {
    const auto [x,ready]=todo.back();todo.pop_back();
    if(rebuilt.contains(x))continue;
    const auto &node=a.nodes[x];
    if(!ready) {
      if(++visits>argmax_replace_visit_limit) {
        ++a.stats["argmax_gather_visit_limit_declines"];
        return std::nullopt;
      }
      a.check();
      todo.emplace_back(x,true);
      for(size_t j=node.args.size();j-- > 0;)
        if(node.args[j]!=from && !rebuilt.contains(node.args[j]))todo.emplace_back(node.args[j],false);
      continue;
    }
    std::vector<Id> args;bool changed=false;
    args.reserve(node.args.size());
    for(Id arg:node.args) {
      const Id r=arg==from ? to : rebuilt.at(arg);
      changed|=r!=arg;args.push_back(r);
    }
    Id out=x;
    if(changed) {
      if(node.op=="add"){std::vector<Id> t;for(size_t j=0;j<args.size();++j)t.push_back(a.mul({a.constant(node.data[j]),args[j]}));out=a.add(t);}
      else if(node.op=="mul"){args.push_back(a.constant(node.data[0]));out=a.mul(args);}
      else if((node.op=="exp" || node.op=="exp2") && args.size()==1)out=a.exponential(args[0],node.op=="exp2");
      else if(node.op=="recip" && args.size()==1)out=a.recip(args[0]);
      else return std::nullopt;
    }
    rebuilt.emplace(x,out);
  }
  return rebuilt.at(root);
}

// E(R) with R = read(x, .., F, ..), F = first_argmax(v_0..v_{n-1}; d) is v_F when E(read(x, .., d_j, ..)) is v_j
// for every j (equal up to Fold binder names): then E = v_F = max_j v_j. Also when v_j = select(G == d_j, c, u_j)
// with u_j = E(read(.., d_j, ..)), G a first_argmax over the same indices and constant c below every u_j: the
// argmax F of v avoids G (v_G = c < v_k for k != G), so E = u_F = v_F = max v: the reference Max fold that
// defined F, or for masked leaves their masked_max. Checked at Divide and Multiply results containing such a
// read. O(n x min(|E|, argmax_replace_visit_limit)).
inline auto Importer::gathered_at_argmax(const RawNode &n,const std::vector<Id> &xs,Id result) -> Id {
  if(a.nodes[result].op=="read")
    for(Id coordinate:a.nodes[result].args)
      if(a.nodes[coordinate].op=="first_argmax"){argmax_reads.emplace(result,result);return result;}
  Id read=invalid;
  for(Id x:xs)if(auto it=argmax_reads.find(x);it!=argmax_reads.end()){read=it->second;break;}
  if(read==invalid)return result;
  // The result keeps the read for its consumers unless it is a constant.
  auto keep=[&] {
    if(!a.value(result))argmax_reads.emplace(result,read);
    return result;
  };
  if(n.kind!="Operation" || (n.name!="Divide" && n.name!="Multiply"))return keep();
  const auto node=a.nodes[read];
  size_t axis=node.args.size();
  for(size_t j=0;j<node.args.size();++j)if(a.nodes[node.args[j]].op=="first_argmax")axis=j;
  const Id argmax=node.args[axis];
  const auto values=a.nodes[argmax].args;const auto indices=a.nodes[argmax].data;
  if(values.size()<2)return keep();
  Id masked_by=invalid;std::optional<Q> mask;bool nonnegative=true;
  for(size_t j=0;j<values.size();++j) {
    auto coordinates=node.args;coordinates[axis]=a.constant(indices[j]);
    const Id instance_read=a.intern("read",coordinates,node.data,node.tag,node.facts,node.bounds);
    const auto instance=replace_read(result,read,instance_read);
    if(!instance)return keep();
    nonnegative=nonnegative && a.nodes[*instance].facts.finite && a.nodes[*instance].facts.nn;
    if(argmax_alpha.equal(*instance,values[j]))continue;
    const auto &leaf=a.nodes[values[j]];
    const auto &test=a.nodes[leaf.op=="select" ? leaf.args[0] : values[j]];
    const auto c=leaf.op=="select" ? a.value(leaf.args[1]) : std::nullopt;
    const bool masked=leaf.op=="select" && c && test.op=="compare" && test.tag=="eq" && test.args.size()==2 &&
        a.nodes[test.args[0]].op=="first_argmax" && a.nodes[test.args[0]].data==indices &&
        a.value(test.args[1]) && *a.value(test.args[1])==indices[j] &&
        (masked_by==invalid || masked_by==test.args[0]) && (!mask || *mask==*c) &&
        a.nodes[*instance].facts.nn && *c<0 && argmax_alpha.equal(*instance,leaf.args[2]);
    if(!masked)return keep();
    masked_by=test.args[0];mask=c;
  }
  // A masked leaf is below the others only when every instance is nonnegative.
  if(masked_by!=invalid && !nonnegative)return keep();
  if(masked_by!=invalid)
    if(auto masked=a.masked_maximum(std::set<Id>(values.begin(),values.end()))) {
      ++a.stats["argmax_gathers"];
      return *masked;
    }
  const auto fold=argmax_maxima.find(argmax);
  if(fold==argmax_maxima.end())return keep();
  ++a.stats["argmax_gathers"];
  return fold->second;
}

inline auto Importer::first_argmax_fold_pattern(Id raw) const -> bool {
  const auto &n=ns[raw];
  if(n.operands.size()!=2 || n.count>first_argmax_unroll_limit || !n.operands[1].conversion.empty())return false;
  const auto &select=ns[n.operands[1].node];
  if(select.kind!="Operation" || select.name!="Select" || !select.attributes.empty() || select.operands.size()!=3 ||
     !select.operands[0].conversion.empty())return false;
  const auto &equal=ns[select.operands[0].node];
  if(equal.kind!="Operation" || equal.name!="Equal" || equal.operands.size()!=2)return false;
  for(const auto &side:equal.operands) {
    const auto &m=ns[side.node];
    if(side.conversion.empty() && m.kind=="Fold" && m.reducer=="Max" && m.count==n.count &&
       !std::binary_search(free[side.node].begin(),free[side.node].end(),n.level))return true;
  }
  return false;
}

// Min_j select(B(j) == max_k B'(k), j, c) with init >= n-1 (or +inf) and c >= n-1 is the least j attaining
// the max, when B'(k) is B(k) for every k (imported instances, equal up to Fold binder names), Max's init
// is -inf and every B(j) is finite (a finite maximum is attained). Imports B(j) and B'(j) per j. O(n x B).
inline auto Importer::first_argmax_fold(Id raw,const Env &env,std::vector<std::pair<Id,Env>> &missing) -> std::optional<Id> {
  const auto &n=ns[raw];
  const auto &init=n.operands[0];
  const auto &select=ns[n.operands[1].node];
  const auto &equal=ns[select.operands[0].node];
  size_t which=0;
  for(size_t j=0;j<2;++j) {
    const auto &m=ns[equal.operands[j].node];
    if(equal.operands[j].conversion.empty() && m.kind=="Fold" && m.reducer=="Max" && m.count==n.count)which=j;
  }
  const auto &maximum=ns[equal.operands[which].node];
  const auto &body=equal.operands[1-which];
  std::vector<Id> scratch;
  auto need=[&](const Edge &e,const Env &scope)->std::optional<Id> {
    if (const Id *value = cache.find(key(scratch, e.node, scope)))
      return edge(*value, e);
    missing.emplace_back(e.node,scope);return {};
  };
  auto start=need(init,env);
  auto floor=need(maximum.operands[0],env);
  auto whole=need(equal.operands[which],env);
  std::vector<std::pair<Q,Id>> members;
  bool complete=start && floor && whole;
  for(uint64_t j=0;j<n.count;++j) {
    Env scope=env,inner=env;
    scope[n.level]=a.constant(Q(Z(j)));
    inner[maximum.level]=a.constant(Q(Z(j)));
    auto value=need(body,scope),instance=need(maximum.operands[1],inner);
    auto index=need(select.operands[1],scope),other=need(select.operands[2],scope);
    if(!value || !instance || !index || !other){complete=false;continue;}
    const auto d=a.value(*index),c=a.value(*other);
    if(!argmax_alpha.equal(*value,*instance) || !a.nodes[*value].facts.finite || !d || *d!=Q(Z(j)) ||
       !c || *c<Q(Z(n.count-1)))return {};
    members.emplace_back(*d,*value);
  }
  if(!complete)return {};
  const auto lowest=a.value(*start);
  if(!a.neginf(*floor) || !(a.nodes[*start].op=="infinity" && a.nodes[*start].data[0]==1) &&
     !(lowest && *lowest>=Q(Z(n.count-1))))return {};
  ++a.stats["first_argmax_folds"];
  // Copies of the same argmax (one per reference use) are equal up to Fold binder names: share one node.
  const Id argmax=argmax_alpha.representative(a.first_argmax(std::move(members)));
  argmax_maxima.emplace(argmax,*whole);
  return argmax;
}

// Whether the element, imported with the Fold's binder variable, still reads a fixed input at a
// coordinate that depends on the binder (raw offsets can mention the binder and simplify it away).
// Visits only nodes that depend on the binder (cached free sets); stops at the first such read.
inline auto Importer::reads_fixed_table_at(Id element,Id binder) -> bool {
  std::vector<Id> todo{element};
  std::unordered_set<Id> seen;
  while(!todo.empty()) {
    const Id x=todo.back();todo.pop_back();
    if(!a.depends(x,binder) || !seen.insert(x).second)continue;
    a.check();
    const auto &node=a.nodes[x];
    if(node.op=="read")
      if(const auto split=node.tag.find('\n');split!=std::string::npos &&
         context->fixed_input_types.contains(node.tag.substr(0,split)))
        for(Id coordinate:node.args)if(a.depends(coordinate,binder))return true;
    todo.insert(todo.end(),node.args.begin(),node.args.end());
  }
  return false;
}

// The definition of a finite Fold: init + sum_k e(k) (Sum), max(init, e(0), ..., e(count-1)) (Max),
// and Min as -max(-init, -e(k)). Each e(k) is imported with the binder bound to the constant k,
// so a fixed-table read at the binder's coordinate meets its concrete element. Missing operand
// imports are returned for the caller to schedule. O(count x element).
inline auto Importer::unrolled_fold(Id raw,const Env &env,std::vector<std::pair<Id,Env>> &missing,
                                const char *counter) -> std::optional<Id> {
  const auto &n=ns[raw];
  if(!unrollable_reducer(n.reducer))throw Unsupported("unsupported Fold reducer "+n.reducer);
  const auto &init=n.operands[0],&element=n.operands[1];
  std::vector<Id> scratch;
  if(!cache.contains(key(scratch,init.node,env)))missing.emplace_back(init.node,env);
  std::vector<Env> scopes;
  scopes.reserve(n.count);
  for(uint64_t j=0;j<n.count;++j) {
    Env scope=env;
    scope[n.level]=a.constant(Q(Z(j)));
    if(!cache.contains(key(scratch,element.node,scope)))missing.emplace_back(element.node,scope);
    scopes.push_back(std::move(scope));
  }
  if(!missing.empty())return {};
  std::vector<Id> terms{edge(cache.at(key(scratch,init.node,env)),init)};
  for(const auto &scope:scopes)terms.push_back(edge(cache.at(key(scratch,element.node,scope)),element));
  // Taint (hardware_infinity_keys) per term: a Max or Min carries it on, a Sum stops the family.
  std::vector<uint8_t> tainted;
  for(size_t j=0;j<terms.size();++j)
    if(hardware_infinity(j?element.node:init.node,j?scopes[j-1]:env)){tainted.resize(terms.size());tainted[j]=1;}
  if(!tainted.empty() && n.reducer!="Max" && n.reducer!="Min")
    hardware_decline(raw,terms,env,"a reassociated reduction");
  ++a.stats[counter];
  Id result;
  if(n.reducer=="Max")result=a.maximum(terms);
  else if(n.reducer=="Min") {
    std::vector<Id> negated;
    for(Id term:terms)negated.push_back(a.neg(term));
    result=a.neg(a.maximum(negated));
  } else result=a.add(terms);
  if(!tainted.empty())carry_hardware_infinity(raw,key(raw,env),terms,tainted,result);
  return result;
}

// Recognize only the exact e4m3 NaN encoding test on initial byte reads.
// Whole-object constraints are caller premises; address coverage is checked
// separately before selecting a branch. No constraints on computed bytes.
inline auto Importer::finite_scale_guard(const RawNode &select) const -> std::optional<std::pair<Id,std::string>> {
  if (!contract || select.kind != "Operation" || select.name != "Select" ||
      !select.attributes.empty() || select.operands.size()!=3) return {};
  const auto &ce=select.operands[0];
  if(!ce.conversion.empty()) return {};
  const auto &cmp=ns.at(ce.node);
  if(cmp.kind!="Operation" || cmp.name!="Equal" || cmp.type!="pred" ||
     !cmp.attributes.empty() || cmp.operands.size()!=2) return {};
  // E8M0 has exactly one nonfinite code, 255. This implication is about
  // decoded E8M0 values, not generic byte-valued inputs.
  // The byte may be compared directly (u8 255) or after one exact widening conversion to a type that
  // represents every byte value exactly (then compared with 255 of that type, e.g. 255.0 as f32):
  // the conversion is injective on 0..255, so the comparison is still exactly byte == 255.
  static const std::regex widening(R"(\s*Convert(?: rounding=nearest-even)?\(\s*u8\s*->\s*([a-z0-9]+)\s*\)\s*)");
  static const std::regex two_five_five(R"(255(\.0*)?)");
  static const std::set<std::string> exact_byte_types{"u16","u32","u64","s16","s32","s64","f16","bf16","f32","f64"};
  for(unsigned side=0;side<2;++side) {
    const auto &value=cmp.operands[side], &test=cmp.operands[1-side];
    const auto &v=ns.at(value.node),&c=ns.at(test.node);
    if(!test.conversion.empty() || c.kind!="Constant" || (v.kind!="InitialCell" && v.kind!="InputValue") ||
       v.type!="u8" || v.operands.size()!=1) continue;
    std::smatch widened;
    const bool direct=value.conversion.empty() && c.type=="u8" && c.name=="255";
    const bool converted=!value.conversion.empty() && std::regex_match(value.conversion,widened,widening) &&
        exact_byte_types.contains(widened[1].str()) && c.type==widened[1].str() &&
        std::regex_match(c.name,two_five_five);
    if(direct || converted) {
      const auto* rule=contract->find(mapped_name(m,"allocations",v.name),"e8m0");
      if(rule && rule->finite)return std::pair(value.node,std::string("e8m0"));
    }
  }
  auto plain127=[&](const Edge &e) {
    const auto &v=ns.at(e.node);
    return e.conversion.empty() && v.kind=="Constant" && v.type=="u8" && v.name=="127";
  };
  for(unsigned side=0;side<2;++side) {
    if(!plain127(cmp.operands[1-side]) || !cmp.operands[side].conversion.empty()) continue;
    const auto &mask=ns.at(cmp.operands[side].node);
    if(mask.kind!="Operation" || mask.name!="BitAnd" || mask.type!="u8" ||
       !mask.attributes.empty() || mask.operands.size()!=2) continue;
    for(unsigned j=0;j<2;++j) {
      if(!plain127(mask.operands[1-j]) || !mask.operands[j].conversion.empty()) continue;
      Id read=mask.operands[j].node; const auto &v=ns.at(read);
      if((v.kind!="InitialCell" && v.kind!="InputValue") || v.type!="u8" || v.operands.size()!=1) continue;
      auto name=mapped_name(m,"allocations",v.name);
      const auto *rule=contract->find(name,"e4m3");
      if(rule && rule->finite) return std::pair(read,std::string("e4m3"));
    }
  }
  return {};
}

// For each shared allocation name (this mapping side), the 32-bit integer types of its typed reads.
inline auto Importer::integer_reads(const Document &d, const J &side_inputs) -> std::map<std::string, std::set<std::string>> {
  std::map<std::string, std::set<std::string>> out;
  for (const auto &n : d.nodes)
    if ((n.kind == "InitialCell" || n.kind == "InputValue") && (n.type == "s32" || n.type == "u32"))
      out[mapped_name(side_inputs, "allocations", n.name)].insert(n.type);
  return out;
}

// Translate raw-word eq/ne only for one aligned, covered 4-byte input element
// whose opposite-side type is s32 or u32 (typed_read proves the layout).
// Values in [0,2^31) use the same comparison. Constants in [2^31,2^32) use
// v-2^32 for s32 and v for u32. Other predicates and raw reads are unchanged.
// Cost: O(rank) for the typed read.
inline auto Importer::typed_word_predicate(const std::string &op, Id x, Id y) -> std::optional<Id> {
  for (unsigned side_index = 0; side_index != 2; ++side_index) {
    const Id word = side_index ? y : x, other = side_index ? x : y;
    if (!a.raw_word32(word)) continue;
    const auto &bounds = a.nodes[other].bounds;
    if (!a.nodes[other].facts.finite || !bounds || bounds->first < 0) continue;
    const auto constant = a.value(other);
    const bool low_half = bounds->second < Q(pow2(31));
    const bool high_half = constant && constant->denominator() == 1 && *constant >= Q(pow2(31)) &&
                           *constant < Q(pow2(32));
    if (!low_half && !high_half) continue;
    const std::string raw = a.nodes[word].tag;
    const Id offset = a.nodes[word].args[0];
    if (!m.contains("layouts") || !m["layouts"].contains(raw) ||
        m["layouts"][raw].value("element_bytes", 0u) != 4u) continue;
    const auto paired = paired_integer_reads.find(mapped_name(m, "allocations", raw));
    if (paired == paired_integer_reads.end() || paired->second.size() != 1) continue;
    const std::string type = *paired->second.begin();
    Id element;
    try {
      element = typed_read(raw, offset, type, true);
    } catch (const Unsupported &) {
      continue;
    }
    if (a.raw_word32(element)) continue;
    ++a.stats["typed_word_predicates"];
    if (low_half) return a.compare(op, element, other);
    ++a.stats["typed_word_high_half_predicates"];
    return a.compare(op, element, type == "s32" ? a.constant(*constant - Q(pow2(32))) : other);
  }
  return {};
}

// Rule (contracted raw word): a raw word w = readbits(raw, offset) of `bits` bits of an input whose
// elements an input contract bounds, as s<bits> or u<bits>, by an interval inside [0, 2^(bits-1)),
// at an element the contract's region covers: w is that element. The word holds the element's bits,
// and an integer in [0, 2^(bits-1)) has the same bits as a signed and as an unsigned integer, so the
// word read as an unsigned integer is the typed element's value. The element is built by typed_read
// (layout, alignment, extent, region and contract checks; the same read node the other side builds
// for that element), and it is used only when its bounds lie inside [0, 2^(bits-1)). The contract is
// the declared premise on the input's elements; this rule reads that premise for the raw word too
// (input_contract_applied_reads then names this side). Every other raw word stays a raw read. O(rank).
inline auto Importer::contracted_word(const std::string &raw, Id offset, unsigned bits) -> std::optional<Id> {
  if (!contract || bits < 8 || bits > 64) return {};
  const std::string alloc = mapped_name(m, "allocations", raw);
  for (const std::string &type : {"s" + std::to_string(bits), "u" + std::to_string(bits)}) {
    const auto *rule = contract->find(alloc, type);
    if (!rule || !rule->range || rule->range->first < 0 || rule->range->second >= Q(pow2(bits - 1))) continue;
    Id element;
    try {
      element = typed_read(raw, offset, type, true);
    } catch (const Unsupported &) {
      continue;
    }
    const auto &n = a.nodes[element];
    if (!n.facts.finite || !n.bounds || n.bounds->first < 0 || n.bounds->second >= Q(pow2(bits - 1))) continue;
    ++a.stats["contracted_raw_words"];
    return element;
  }
  return {};
}

} // namespace rg
