#pragma once

#include <algorithm>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <type_traits>
#include <variant>
#include <vector>

namespace sentryward::algorithm {

template<class Codec,class=void> struct RecordDelta { using Type=std::monostate; };
template<class Codec> struct RecordDelta<Codec,std::void_t<typename Codec::Delta>> {
  using Type=typename Codec::Delta;
};

// Logical records are independent. Queries reconstruct values without caching them.
template<class T, class Codec>
class TemplateRecords final {
  static constexpr std::uint32_t concrete = 0x80000000u;
  struct Row { std::uint32_t prototype; typename Codec::Binding binding; };
  using Delta=typename RecordDelta<Codec>::Type;
  using GeneratedColumnVisitor=std::function<void(const T&,const Delta&,std::size_t,std::size_t,std::size_t)>;
  using GeneratedColumns=std::function<void(const GeneratedColumnVisitor&)>;
  struct ColumnRun { std::size_t begin, end, row; Delta delta{}; };
  // Ordered iterations of a thread block: logical columns [start, start+width*count) repeat the
  // stored columns [stored, stored+width) `count` times; repetition j adds j*deltas[c] to the
  // binding of stored column stored+c. Stored columns index `columns`, `deltas` and the rows.
  struct Iteration {
    std::size_t start, width, count, stored;
    std::vector<Delta> deltas;
    friend bool operator==(const Iteration&,const Iteration&)=default;
  };
  // Fewest repetitions an ordered iteration stores: with fewer, one stored repetition plus a delta
  // per column is no smaller than the unrolled rows.
  static constexpr std::size_t kMinRepetitions=3;
  // Most loop iterations one repetition spans, which bounds the widths compressIterations tries.
  static constexpr std::size_t kMaxIterationsPerRepetition=16;
  struct Segment {
    std::size_t begin=0, physical=0, width=0, repeats=1;
    std::optional<std::pair<std::uint64_t,std::uintptr_t>> cohort;
    std::vector<Delta> deltas;
    std::function<T(std::size_t)> generate = {};
    std::vector<std::vector<ColumnRun>> columns = {};
    GeneratedColumns generated_columns = {};
    const void* generated_family = nullptr;
    std::vector<Iteration> iterations = {};
  };
  struct StoredColumn { std::size_t column, iteration=0; const Delta* delta=nullptr; };
  // A delta that may step from one member (thread) to the next. A codec can keep binding fields
  // constant across members while letting them advance across iterations.
  static bool memberStep(const Delta& delta) {
    if constexpr(requires { Codec::memberStep(delta); }) return Codec::memberStep(delta);
    else return true;
  }
  // `binding` advanced i times by `a` and j times by `b`. Where the codec has a two-delta translate
  // it is one rewrite, which agrees with translate(translate(binding, a, i), b, j) whenever that
  // succeeds and also accepts an intermediate binding out of range; else that composition.
  template<class Binding> static std::optional<Binding> translateTwice(const Binding& binding,const Delta& a,std::size_t i,
                                                                       const Delta& b,std::size_t j) {
    if constexpr(requires { Codec::translate(binding,a,i,b,j); }) return Codec::translate(binding,a,i,b,j);
    else {
      auto first=Codec::translate(binding,a,i);
      if(!first)return first;
      return Codec::translate(*first,b,j);
    }
  }
  // O(log iteration blocks).
  static StoredColumn storedColumn(const Segment& segment,std::size_t column) {
    if(segment.iterations.empty())return {column};
    const auto it=std::upper_bound(segment.iterations.begin(),segment.iterations.end(),column,
      [](std::size_t x,const Iteration& block){return x<block.start;});
    if(it==segment.iterations.begin())return {column};
    const auto& block=*std::prev(it);
    const auto end=block.start+block.width*block.count;
    if(column>=end)return {column-(end-block.stored-block.width)};
    const auto offset=column-block.start;
    return {block.stored+offset%block.width,offset/block.width,&block.deltas[offset%block.width]};
  }
  // A physical block's stored columns as one run each.
  void runColumns(Segment& segment) {
    const auto width=storedWidth(segment);
    segment.columns.resize(width);
    for(std::size_t i=0;i<width;++i)
      segment.columns[i].push_back({0,segment.repeats,segment.physical+i,
        segment.repeats>1?segment.deltas[segment.deltas.size()==1?0:i]:Delta{}});
    segment.deltas.clear();
  }
  // Thread blocks with different iteration layouts keep only their common iterations (equal
  // positions and deltas); the others are unrolled as without ordered iterations. The tail's rows
  // stay last. O(logical columns of both blocks).
  void unrollIterations(Segment& previous,Segment& tail) {
    std::vector<Iteration> common;
    for(const auto& block:previous.iterations)
      for(const auto& other:tail.iterations)
        if(block.start==other.start && block.width==other.width && block.count==other.count && block.deltas==other.deltas) {
          common.push_back(block);
          break;
        }
    placeIterations(common);
    Segment layout{};layout.width=tail.width;layout.iterations=common;
    const auto stored_width=storedWidth(layout);
    // Logical column of each stored column of the common layout.
    std::vector<std::size_t> logical;logical.reserve(stored_width);
    for(std::size_t i=0;i<tail.width;++i)if(!storedColumn(layout,i).iteration)logical.push_back(i);
    std::vector<Row> tail_rows;tail_rows.reserve(stored_width);
    for(const auto column:logical)tail_rows.push_back(rowAt(tail.begin+column));
    rows_.resize(tail.physical);
    if(previous.iterations!=common) {
      if(previous.columns.empty())runColumns(previous);
      std::vector<std::vector<ColumnRun>> columns(stored_width);
      for(std::size_t i=0;i<stored_width;++i) {
        const auto stored=storedColumn(previous,logical[i]);
        for(auto run:previous.columns[stored.column]) {
          if(stored.iteration) {
            if constexpr(requires { typename Codec::Delta; }) {
              auto row=rows_[run.row];
              auto binding=Codec::translate(row.binding,*stored.delta,stored.iteration);
              if(!binding)throw std::logic_error("invalid iteration binding");
              row.binding=std::move(*binding);run.row=rows_.size();rows_.push_back(std::move(row));
            }
          }
          columns[i].push_back(std::move(run));
        }
      }
      previous.columns=std::move(columns);previous.iterations=common;
    }
    tail.physical=rows_.size();tail.iterations=std::move(common);
    rows_.insert(rows_.end(),std::make_move_iterator(tail_rows.begin()),std::make_move_iterator(tail_rows.end()));
  }
  // A sealed thread block (the last rows) whose edited iterations were sealed unrolled: each
  // iteration folds back over its longest prefix of >= kMinRepetitions repetitions whose columns
  // have the first repetition's runs, prototypes and thread deltas and whose bindings advance by
  // one delta per column. Exact by checking; O(rows of the block).
  void refoldIterations(Segment& segment,const std::vector<Iteration>& edited) {
    if constexpr(requires { typename Codec::Delta; }) {
      std::vector<Iteration> blocks=segment.iterations,folded;
      for(const auto& block:edited) {
        const auto first=storedColumn(segment,block.start).column,width=block.width;
        std::vector<Delta> deltas(width);
        const auto congruent=[&](std::size_t r) {
          for(std::size_t c=0;c<width;++c) {
            const auto& base=segment.columns[first+c];const auto& next=segment.columns[first+r*width+c];
            if(base.size()!=next.size())return false;
            for(std::size_t k=0;k<base.size();++k) {
              const auto& a=base[k];const auto& b=next[k];
              const auto& x=rows_[a.row];const auto& y=rows_[b.row];
              if(a.begin!=b.begin || a.end!=b.end || !(a.delta==b.delta) || x.prototype!=y.prototype || (x.prototype&concrete))
                return false;
              if(r==1 && k==0) {
                auto delta=Codec::difference(x.binding,y.binding);
                if(!delta)return false;
                deltas[c]=std::move(*delta);
              }
              const auto expected=Codec::translate(x.binding,deltas[c],r);
              if(!expected || !(*expected==y.binding))return false;
            }
          }
          return true;
        };
        std::size_t count=1;
        while(count<block.count && congruent(count))++count;
        if(count>=kMinRepetitions)folded.push_back({block.start,width,count,first,std::move(deltas)});
      }
      if(folded.empty())return;
      // Drop the folded repetitions' columns and rows; the block's rows are the last ones.
      std::vector<bool> keep_column(segment.columns.size(),true);
      for(const auto& block:folded)
        for(std::size_t c=block.stored+block.width;c<block.stored+block.width*block.count;++c)keep_column[c]=false;
      std::vector<bool> keep_row(rows_.size()-segment.physical,true);
      for(std::size_t c=0;c<segment.columns.size();++c)
        if(!keep_column[c])for(const auto& run:segment.columns[c])keep_row[run.row-segment.physical]=false;
      std::vector<std::size_t> moved(keep_row.size());
      std::size_t next=segment.physical;
      for(std::size_t i=0;i<keep_row.size();++i)
        if(keep_row[i]) { moved[i]=next;if(next!=segment.physical+i)rows_[next]=std::move(rows_[segment.physical+i]);++next; }
      rows_.resize(next);
      std::vector<std::vector<ColumnRun>> columns;
      for(std::size_t c=0;c<segment.columns.size();++c) {
        if(!keep_column[c])continue;
        columns.push_back(std::move(segment.columns[c]));
        for(auto& run:columns.back())run.row=moved[run.row-segment.physical];
      }
      segment.columns=std::move(columns);
      blocks.insert(blocks.end(),std::make_move_iterator(folded.begin()),std::make_move_iterator(folded.end()));
      std::sort(blocks.begin(),blocks.end(),[](const Iteration& a,const Iteration& b){return a.start<b.start;});
      placeIterations(blocks);
      segment.iterations=std::move(blocks);
    }
  }
  // `stored` of ordered iterations in column order: a block's first repetition is stored after the
  // columns before it, less the later repetitions of earlier blocks.
  static void placeIterations(std::vector<Iteration>& blocks) {
    std::size_t removed=0;
    for(auto& block:blocks) { block.stored=block.start-removed;removed+=block.width*(block.count-1); }
  }
  static std::size_t storedWidth(const Segment& segment) {
    if(segment.iterations.empty())return segment.width;
    const auto& last=segment.iterations.back();
    return segment.width-(last.start+last.width*last.count-last.stored-last.width);
  }
  std::vector<Row> rows_;
  std::vector<Segment> segments_;
  std::unordered_map<std::size_t,Row> edits_;
  std::size_t logical_size_=0;
  bool boundary_=false;
  bool has_generated_=false;
  // rewriteRecords may change a prototype's binding slots, so stored bindings need not be what split() returns.
  bool rewritten_=false;
  const Segment& segmentAt(std::size_t i) const {
    if(i>=logical_size_)throw std::out_of_range("template record index");
    auto it=std::upper_bound(segments_.begin(),segments_.end(),i,
      [](std::size_t x,const Segment& s){return x<s.begin;});
    return *std::prev(it);
  }
  bool generatedAt(std::size_t i) const {
    if(!has_generated_)return false;
    return (edits_.empty() || !edits_.contains(i)) && bool(segmentAt(i).generate);
  }
  Row rowAt(std::size_t i) const {
    if(i>=logical_size_)throw std::out_of_range("template record index");
    if(!edits_.empty())if(auto it=edits_.find(i);it!=edits_.end())return it->second;
    const Segment* segment=&segments_.front();
    if(segments_.size()>1) {
      auto it=std::upper_bound(segments_.begin(),segments_.end(),i,
        [](std::size_t x,const Segment& s){return x<s.begin;});
      segment=&*std::prev(it);
    }
    if(segment->generate)throw std::logic_error("generated record has no stored binding");
    auto position=i-segment->begin;
    const auto member=position/segment->width;
    const auto column=storedColumn(*segment,position%segment->width);
    Row row;
    const Delta* step=nullptr;
    std::size_t steps=0;
    if(!segment->columns.empty()) {
      const auto& runs=segment->columns[column.column];
      const auto& run=*std::prev(std::upper_bound(runs.begin(),runs.end(),member,
        [](std::size_t value,const ColumnRun& r){return value<r.begin;}));
      row=rows_[run.row];
      if(member!=run.begin) { step=&run.delta;steps=member-run.begin; }
    } else {
      row=rows_[segment->physical+column.column];
      if(segment->repeats>1) {
        if constexpr(!requires { typename Codec::Delta; })
          throw std::logic_error("record codec has no binding translation");
        step=&segment->deltas[segment->deltas.size()==1?0:column.column];steps=member;
      }
    }
    if constexpr(requires { typename Codec::Delta; }) {
      // Member and iteration translations compose into one binding rewrite.
      std::optional<typename Codec::Binding> binding;
      if(step && column.iteration)binding=translateTwice(row.binding,*step,steps,*column.delta,column.iteration);
      else if(step)binding=Codec::translate(row.binding,*step,steps);
      else if(column.iteration)binding=Codec::translate(row.binding,*column.delta,column.iteration);
      else return row;
      if(!binding)throw std::logic_error("invalid compressed record binding");
      row.binding=std::move(*binding);
    }
    return row;
  }
  template<class Projection>
  auto projectHeader(std::size_t i, Projection project) const {
    if (i >= logical_size_) throw std::out_of_range("template record index");
    // A projection adds `member` steps of `delta` and `iteration` steps of `step` to the stored binding.
    const auto apply = [&](const Row& row, const Delta* delta, std::size_t member,
                           const Delta* step = nullptr, std::size_t iteration = 0) {
      const auto& prototype = prototypes_.at(row.prototype & ~concrete);
      return project(prototype, row.prototype & concrete ? nullptr : &row.binding,
                     delta, member, step, iteration);
    };
    if (!edits_.empty())
      if (const auto found = edits_.find(i); found != edits_.end())
        return apply(found->second, nullptr, 0);
    const auto& segment = segmentAt(i);
    const auto position = i - segment.begin;
    if (segment.generate) return project(segment.generate(position), nullptr, nullptr, 0, nullptr, 0);
    const auto stored = storedColumn(segment, position % segment.width);
    if (!segment.columns.empty()) {
      const auto& runs=segment.columns[stored.column];
      const auto member=position/segment.width;
      const auto& run=*std::prev(std::upper_bound(runs.begin(),runs.end(),member,
        [](std::size_t value,const ColumnRun& r){return value<r.begin;}));
      return apply(rows_[run.row],member==run.begin?nullptr:&run.delta,member-run.begin,stored.delta,stored.iteration);
    }
    const auto column = stored.column;
    const Delta* delta = segment.repeats > 1
        ? &segment.deltas[segment.deltas.size() == 1 ? 0 : column] : nullptr;
    return apply(rows_[segment.physical + column], delta, position / segment.width, stored.delta, stored.iteration);
  }
  void appendRow(Row row) {
    if(boundary_ || segments_.empty() || segments_.back().repeats!=1 || segments_.back().cohort || !segments_.back().columns.empty() ||
       !segments_.back().iterations.empty()) {
      segments_.push_back({logical_size_,rows_.size(),0,1,{},{}});
      boundary_=false;
    }
    rows_.push_back(std::move(row));++segments_.back().width;++logical_size_;
  }
  void materializeRows() {
    if(has_generated_) {
      TemplateRecords rebuilt;rebuilt.reserve(logical_size_);
      for(std::size_t i=0;i<logical_size_;++i)rebuilt.push_back(at(i));
      *this=std::move(rebuilt);return;
    }
    if(segments_.size()==1 && segments_[0].repeats==1 && segments_[0].columns.empty() && segments_[0].iterations.empty() && edits_.empty()) {
      segments_[0].cohort.reset();boundary_=false;return;
    }
    std::vector<Row> rows;rows.reserve(logical_size_);
    for(std::size_t i=0;i<logical_size_;++i)rows.push_back(rowAt(i));
    rows_=std::move(rows);segments_.clear();edits_.clear();boundary_=false;
    if(logical_size_)segments_.push_back({0,0,logical_size_,1,{},{}});
  }
  std::deque<T> prototypes_;
  std::unordered_map<std::size_t,std::vector<std::uint32_t>> index_;
  Row intern(T value) {
    if(prototypes_.size()>=concrete)throw std::length_error("prototype ID overflow");
    typename Codec::Binding binding{};
    if(!Codec::split(value,binding)) {
      auto id=static_cast<std::uint32_t>(prototypes_.size())|concrete;
      prototypes_.push_back(std::move(value));return {id,{}};
    }
    return {internPrototype(std::move(value)),std::move(binding)};
  }
  // `prototype` is a split() result: its binding fields are cleared.
  std::uint32_t internPrototype(T prototype) {
    if(prototypes_.size()>=concrete)throw std::length_error("prototype ID overflow");
    auto& candidates=index_[Codec::hash(prototype)];
    for(auto id:candidates)if(prototypes_[id]==prototype)return id;
    auto id=static_cast<std::uint32_t>(prototypes_.size());
    prototypes_.push_back(std::move(prototype));candidates.push_back(id);return id;
  }
 public:
  using AffineGenerator=GeneratedColumns;
  using value_type=T;
  class Iterator {
    const TemplateRecords* owner_{};
    std::ptrdiff_t index_{};
   public:
    using value_type=T;
    using reference=T;
    using difference_type=std::ptrdiff_t;
    using iterator_category=std::random_access_iterator_tag;
    struct Arrow {
      std::optional<T> value;
      const T* stable{};
      const T* operator->() const { return stable ? stable : &*value; }
      const T& operator*() const { return *operator->(); }
      explicit operator bool() const { return stable || value.has_value(); }
    };
    using pointer=Arrow;
    Iterator()=default;
    Iterator(const TemplateRecords* owner,std::ptrdiff_t index):owner_(owner),index_(index){}
    T operator*() const { return owner_->at(index_); }
    Arrow operator->() const {
      if (owner_->isConcrete(index_)) return {std::nullopt, &owner_->stored(index_)};
      return {**this, nullptr};
    }
    T operator[](difference_type n) const {return *(*this+n);}
    Iterator& operator++(){++index_;return *this;}
    Iterator operator++(int){auto x=*this;++*this;return x;}
    Iterator& operator--(){--index_;return *this;}
    Iterator operator--(int){auto x=*this;--*this;return x;}
    Iterator& operator+=(difference_type n){index_+=n;return *this;}
    Iterator& operator-=(difference_type n){index_-=n;return *this;}
    friend Iterator operator+(Iterator x,difference_type n){return x+=n;}
    friend Iterator operator+(difference_type n,Iterator x){return x+=n;}
    friend Iterator operator-(Iterator x,difference_type n){return x-=n;}
    friend difference_type operator-(Iterator a,Iterator b){return a.index_-b.index_;}
    friend bool operator==(Iterator,Iterator)=default;
    friend auto operator<=>(Iterator a,Iterator b){return a.index_<=>b.index_;}
  };
  struct Handle {
    const TemplateRecords* owner;
    std::size_t index;
    auto id() const requires requires(const T& value) { value.id.value(); } { return owner->id(index); }
    auto actor() const requires requires(const T& value) { value.actor.value(); } { return owner->actor(index); }
    template<class Op> bool holdsOperation() const { return owner->template holdsOperation<Op>(index); }
    std::size_t effectCount() const { return owner->effectCount(index); }
    T operator*() const {return owner->at(index);}
    typename Iterator::Arrow operator->() const {
      if (owner->isConcrete(index)) return {std::nullopt, &owner->stored(index)};
      return {**this, nullptr};
    }
  };
  Handle ref(std::size_t index) const {return {this,index};}
  typename Iterator::Arrow view(std::size_t index) const { return ref(index).operator->(); }
  TemplateRecords()=default;
  TemplateRecords(const TemplateRecords&)=default;
  TemplateRecords& operator=(const TemplateRecords&)=default;
  TemplateRecords(TemplateRecords&& other){*this=std::move(other);}
  TemplateRecords& operator=(TemplateRecords&& other){
    if(this==&other)return *this;
    rows_=std::move(other.rows_);segments_=std::move(other.segments_);edits_=std::move(other.edits_);
    prototypes_=std::move(other.prototypes_);index_=std::move(other.index_);
    logical_size_=std::exchange(other.logical_size_,0);boundary_=std::exchange(other.boundary_,false);
    has_generated_=std::exchange(other.has_generated_,false);rewritten_=std::exchange(other.rewritten_,false);
    return *this;
  }
  // The first `count` records stored as here: `count` must be where a stored block begins (or
  // size()), so no block is cut. Every prototype stays interned, so records appended afterwards
  // get the prototype IDs they would get here. O(kept rows and blocks + prototypes).
  TemplateRecords prefix(std::size_t count) const {
    if(count>logical_size_)throw std::out_of_range("template record prefix");
    if(count==logical_size_)return *this;
    const auto end=std::lower_bound(segments_.begin(),segments_.end(),count,
        [](const Segment& segment,std::size_t first){return segment.begin<first;});
    if(end==segments_.end() || end->begin!=count)throw std::invalid_argument("template record prefix cuts a stored block");
    TemplateRecords result;
    // A block's rows follow the rows of every earlier block.
    result.rows_.assign(rows_.begin(),rows_.begin()+end->physical);
    result.segments_.assign(segments_.begin(),end);
    for(const auto& [index,row]:edits_)if(index<count)result.edits_.emplace(index,row);
    result.prototypes_=prototypes_;result.index_=index_;
    result.logical_size_=count;result.boundary_=true;
    result.has_generated_=std::any_of(result.segments_.begin(),result.segments_.end(),
        [](const Segment& segment){return bool(segment.generate);});
    result.rewritten_=rewritten_;
    return result;
  }
  TemplateRecords(std::vector<T> values){reserve(values.size());for(auto& value:values)push_back(std::move(value));}
  TemplateRecords(std::initializer_list<T> values){for(const auto& x:values)push_back(x);}
  TemplateRecords& operator=(std::initializer_list<T> values){clear();for(const auto& x:values)push_back(x);return *this;}
  template<class It> TemplateRecords(It first,It last){for(;first!=last;++first)push_back(*first);}
  std::size_t size() const noexcept{return logical_size_;}
  bool empty() const noexcept{return logical_size_==0;}
  std::size_t capacity() const noexcept{return rows_.capacity();}
  std::size_t prototypeCount() const noexcept{return prototypes_.size();}
  std::size_t storedBindingCount() const noexcept{return rows_.size()+edits_.size();}
  // Stored bindings with every ordered iteration unrolled: the rows of the column runs that
  // forEachAffineColumnWithBlock reports, which is what template algorithms over them iterate.
  std::size_t unrolledBindingCount() const noexcept{
    auto count=storedBindingCount();
    for(const auto& segment:segments_)for(const auto& block:segment.iterations)
      for(std::size_t c=block.stored;c<block.stored+block.width;++c)
        count+=(block.count-1)*(segment.columns.empty()?1:segment.columns[c].size());
    return count;
  }
  std::size_t bindingTemplateCount() const noexcept{return segments_.size();}
  std::size_t ownedBytes() const noexcept{
    auto bytes=rows_.capacity()*sizeof(Row)+prototypes_.size()*sizeof(T)+segments_.capacity()*sizeof(Segment);
    for(const auto& segment:segments_)bytes+=segment.deltas.capacity()*sizeof(Delta);
    for(const auto& segment:segments_)for(const auto& block:segment.iterations)
      bytes+=sizeof(Iteration)+block.deltas.capacity()*sizeof(Delta);
    for(const auto& segment:segments_) {
      bytes+=segment.columns.capacity()*sizeof(std::vector<ColumnRun>);
      for(const auto& column:segment.columns)bytes+=column.capacity()*sizeof(ColumnRun);
    }
    return bytes+edits_.size()*sizeof(std::pair<const std::size_t,Row>);
  }
  void reserve(std::size_t n){rows_.reserve(n);}
  void shrink_to_fit(){rows_.shrink_to_fit();segments_.shrink_to_fit();}
  void clear(){rows_.clear();segments_.clear();edits_.clear();logical_size_=0;boundary_=false;has_generated_=false;rewritten_=false;prototypes_.clear();index_.clear();}
  Iterator begin() const{return {this,0};}
  Iterator end() const{return {this,static_cast<std::ptrdiff_t>(size())};}
  auto rbegin() const{return std::make_reverse_iterator(end());}
  auto rend() const{return std::make_reverse_iterator(begin());}
  const T at(std::size_t i) const {
    return at(i,[](T& record,const typename Codec::Binding& binding){Codec::bind(record,binding);});
  }
  // at(i) with `bind(record, binding)` in place of Codec::bind for a stored template record, e.g. a
  // codec's partial binding for a reader of only some fields.
  template<class Bind> const T at(std::size_t i,Bind bind) const {
    if(generatedAt(i)) {
      const auto& segment=segmentAt(i);
      return segment.generate(i-segment.begin);
    }
    const auto row=rowAt(i);
    auto result=prototypes_.at(row.prototype&~concrete);
    if(!(row.prototype&concrete)) bind(result,row.binding);
    return result;
  }
  const T operator[](std::size_t i) const{return at(i);}
  // Header projections do not reconstruct operands of stored templates.
  auto id(std::size_t i) const requires requires(const T& value) { value.id.value(); } {
    return projectHeader(i, [](const T& prototype, const typename Codec::Binding* binding,
                               const Delta* delta, std::size_t member, const Delta* step, std::size_t iteration) {
      if (!binding) return prototype.id;
      auto value = binding->id.value();
      if constexpr (requires { (*delta)[0]; }) {
        if (delta) value = static_cast<std::uint32_t>(value + (*delta)[0]*std::int64_t(member));
        if (step) value = static_cast<std::uint32_t>(value + (*step)[0]*std::int64_t(iteration));
      }
      return decltype(prototype.id)(value);
    });
  }
  auto actor(std::size_t i) const requires requires(const T& value) { value.actor.value(); } {
    return projectHeader(i, [](const T& prototype, const typename Codec::Binding* binding,
                               const Delta* delta, std::size_t member, const Delta* step, std::size_t iteration) {
      if (!binding) return prototype.actor;
      auto value = binding->actor.value();
      if constexpr (requires { (*delta)[1]; }) {
        if (delta) value = static_cast<std::uint32_t>(value + (*delta)[1]*std::int64_t(member));
        if (step) value = static_cast<std::uint32_t>(value + (*step)[1]*std::int64_t(iteration));
      }
      return decltype(prototype.actor)(value);
    });
  }
  std::size_t effectCount(std::size_t i) const
      requires requires(const T& value) { Codec::effectCount(value); } {
    return projectHeader(i, [](const T& prototype, const typename Codec::Binding*,
                               const Delta*, std::size_t, const Delta*, std::size_t) { return Codec::effectCount(prototype); });
  }
  // Any header projection: project(prototype, binding (null for a concrete record), delta (null
  // unless the record repeats its column's binding), member, step (null outside a repetition of an
  // ordered iteration), iteration) reads fixed fields as id() does.
  template<class Projection> auto header(std::size_t i, Projection project) const {
    return projectHeader(i, std::move(project));
  }
  const T front() const{return at(0);}
  template<class Op> bool holdsOperation(std::size_t i) const {
    return projectHeader(i, [](const T& prototype, const typename Codec::Binding*,
                               const Delta*, std::size_t, const Delta*, std::size_t) { return std::holds_alternative<Op>(prototype.op); });
  }
  auto opcode(std::size_t i) const {
    return projectHeader(i, [](const T& prototype, const typename Codec::Binding*,
                               const Delta*, std::size_t, const Delta*, std::size_t) { return Codec::opcode(prototype); });
  }
  const T back() const{return at(size()-1);}
  // A projection: the stored row's prototype decides, so the binding is not translated (view() and
  // operator-> ask this before reading the record).
  bool isConcrete(std::size_t i) const {
    return !generatedAt(i) && projectHeader(i, [](const T&, const typename Codec::Binding* binding, const Delta*,
                                                   std::size_t, const Delta*, std::size_t) { return !binding; });
  }
  const T& stored(std::size_t i) const {
    const auto row=rowAt(i);
    if(!(row.prototype&concrete))throw std::logic_error("template record has no concrete object");
    return prototypes_.at(row.prototype&~concrete);
  }
  T& edit(std::size_t i) {
    auto row=generatedAt(i)?Row{}:rowAt(i);
    if(!(row.prototype&concrete)){
      if(prototypes_.size()>=concrete)throw std::length_error("prototype ID overflow");
      auto value=at(i);
      row.prototype=static_cast<std::uint32_t>(prototypes_.size())|concrete;
      prototypes_.push_back(std::move(value));
      edits_.insert_or_assign(i,row);
    }
    return prototypes_.at(row.prototype&~concrete);
  }
  // Rewrites every logical record without expanding templates. The rewrite may read and
  // change only fields outside the codec binding (stored prototypes hold cleared bindings),
  // so rows and families stay valid. O(prototypes); generated blocks rewrite what they produce.
  void rewriteRecords(std::function<void(T&)> rewrite) {
    rewritten_=true;
    for(auto& prototype:prototypes_)rewrite(prototype);
    std::unordered_map<std::size_t,std::vector<std::uint32_t>> index;
    for(const auto& bucket:index_)for(const auto id:bucket.second)index[Codec::hash(prototypes_[id])].push_back(id);
    index_=std::move(index);
    for(auto& segment:segments_) {
      if(segment.generate)segment.generate=[generate=std::move(segment.generate),rewrite](std::size_t i) {
        auto value=generate(i);rewrite(value);return value;
      };
      if(segment.generated_columns)segment.generated_columns=[columns=std::move(segment.generated_columns),rewrite](
          const GeneratedColumnVisitor& visit) {
        columns([&](const T& value,const Delta& delta,std::size_t first,std::size_t stride,std::size_t count) {
          auto rewritten=value;rewrite(rewritten);visit(rewritten,delta,first,stride,count);
        });
      };
    }
  }
  // The generator owns immutable template bindings and returns independent instances.
  void appendGenerated(std::size_t count,std::function<T(std::size_t)> generate,
      std::optional<std::pair<std::uint64_t,std::uintptr_t>> cohort={},
      GeneratedColumns columns={},const void* family=nullptr) {
    if(!count)return;
    if(!generate)throw std::invalid_argument("missing template record generator");
    if(count>static_cast<std::size_t>(PTRDIFF_MAX)-logical_size_)
      throw std::length_error("template record size overflow");
    if(family && !segments_.empty() && segments_.back().generated_family==family &&
        segments_.back().cohort==cohort) {
      segments_.back().width+=count;logical_size_+=count;boundary_=true;return;
    }
    Segment segment{logical_size_,rows_.size(),count,1,cohort,{}};
    segment.generate=std::move(generate);
    segment.generated_columns=std::move(columns);segment.generated_family=family;
    segments_.push_back(std::move(segment));logical_size_+=count;boundary_=true;has_generated_=true;
  }
  // The codec defines an exact affine family; endpoints validate its finite range.
  void appendAffine(T value,const Delta& delta,std::size_t count)
    requires requires { typename Codec::Delta; } {
    if(!count)return;
    if(count>static_cast<std::size_t>(PTRDIFF_MAX)-logical_size_)
      throw std::length_error("template record size overflow");
    auto prototype=value;
    typename Codec::Binding binding{};
    if(!Codec::split(prototype,binding) || !Codec::translate(binding,delta,count-1))
      throw std::invalid_argument("invalid affine record family");
    boundary_=true;
    push_back(std::move(value));
    auto& segment=segments_.back();
    segment.repeats=count;
    if(count>1)segment.deltas.push_back(delta);
    logical_size_+=count-1;
    boundary_=true;
  }
  void push_back(T value){
    appendRow(intern(std::move(value)));
  }
  // push_back of a record already split elsewhere: `prototype` and `binding` are what
  // Codec::split returned for it (split() succeeded). internSplit gives the prototype's id,
  // pushSplit appends the record: together exactly push_back(the record).
  std::uint32_t internSplit(T prototype){return internPrototype(std::move(prototype));}
  void pushSplit(std::uint32_t prototype,typename Codec::Binding binding){
    if(prototype>=prototypes_.size() || (prototype&concrete))throw std::out_of_range("split record prototype");
    appendRow({prototype,std::move(binding)});
  }
  // Preserve member-major order for a fixed set of interleaved affine columns.
  struct CachedColumn {
    std::size_t column, begin, count;
    T value;
    Delta delta;
  };
  // Restore an exact block without materializing its logical members.
  void appendCachedBlock(std::size_t width, std::size_t members,
      const std::vector<CachedColumn>& columns,
      std::optional<std::pair<std::uint64_t,std::uintptr_t>> cohort = {}) {
    if (!width || !members || width > static_cast<std::size_t>(PTRDIFF_MAX) / members ||
        width * members > static_cast<std::size_t>(PTRDIFF_MAX) - logical_size_)
      throw std::invalid_argument("invalid cached block size");
    if (width > columns.size()) throw std::invalid_argument("missing cached column");
    std::vector<std::size_t> ends(width);
    for (const auto& run : columns) {
      if (run.column >= width || run.begin != ends[run.column] || !run.count ||
          run.count > members - run.begin)
        throw std::invalid_argument("invalid cached column range");
      auto value = run.value;
      typename Codec::Binding binding{};
      if (!Codec::split(value, binding) || !Codec::translate(binding, run.delta, run.count - 1))
        throw std::invalid_argument("invalid cached column binding");
      ends[run.column] += run.count;
    }
    for (auto end : ends) if (end != members) throw std::invalid_argument("incomplete cached column");
    Segment segment{logical_size_, rows_.size(), width, members, cohort, {}};
    segment.columns.resize(width);
    for (const auto& run : columns) {
      segment.columns[run.column].push_back({run.begin, run.begin + run.count, rows_.size(), run.delta});
      rows_.push_back(intern(run.value));
    }
    segments_.push_back(std::move(segment));
    logical_size_ += width * members;
    boundary_ = true;
  }
  void appendAffineColumns(const std::vector<T>& values,const std::vector<Delta>& deltas,
                           std::size_t count) requires requires { typename Codec::Delta; } {
    if(!count)return;
    if(values.empty() || values.size()!=deltas.size())throw std::invalid_argument("affine column width");
    if(count>(static_cast<std::size_t>(PTRDIFF_MAX)-logical_size_)/values.size())
      throw std::length_error("affine column count overflow");
    for(std::size_t i=0;i<values.size();++i) {
      auto value=values[i];typename Codec::Binding binding{};
      if(!Codec::split(value,binding) || !Codec::translate(binding,deltas[i],count-1))
        throw std::invalid_argument("invalid affine column");
    }
    Segment segment{logical_size_,rows_.size(),values.size(),count,{},deltas};
    for(const auto& value:values)rows_.push_back(intern(value));
    segments_.push_back(std::move(segment));logical_size_+=values.size()*count;boundary_=true;
  }
  // Append an exact consecutive affine run without changing record order.
  void pushBackAffine(T value) requires requires { typename Codec::Delta; } {
    auto row=intern(std::move(value));
    if(!segments_.empty() && edits_.empty()) {
      auto& last=segments_.back();
      if(last.width==1 && !last.generate && last.columns.empty() && !last.cohort) {
        const auto& base=rows_[last.physical];
        if(base.prototype==row.prototype && !(row.prototype&concrete)) {
          auto delta=last.repeats==1?Codec::difference(base.binding,row.binding):
              std::optional(last.deltas.front());
          if(delta) {
            auto expected=Codec::translate(base.binding,*delta,last.repeats);
            if(expected && *expected==row.binding) {
              if(last.repeats==1)last.deltas.push_back(std::move(*delta));
              ++last.repeats;++logical_size_;boundary_=true;return;
            }
          }
        }
      }
    }
    boundary_=true;appendRow(std::move(row));boundary_=true;
  }
  // Mutating sequence order requires individual binding rows.
  Iterator erase(Iterator pos){
    auto i=pos-begin();materializeRows();rows_.erase(rows_.begin()+i);--logical_size_;
    segments_.clear();if(logical_size_)segments_.push_back({0,0,logical_size_,1,{},{}});return {this,i};
  }
  Iterator insert(Iterator pos,T value){
    auto i=pos-begin();materializeRows();push_back(std::move(value));
    std::rotate(rows_.begin()+i,rows_.end()-1,rows_.end());return {this,i};
  }
  std::optional<std::pair<std::uint64_t,std::uintptr_t>> threadCohort(std::size_t index) const {
    if(index>=logical_size_)throw std::out_of_range("template record index");
    auto it=std::upper_bound(segments_.begin(),segments_.end(),index,
      [](std::size_t x,const Segment& segment){return x<segment.begin;});
    return std::prev(it)->cohort;
  }
  std::size_t beginThread(){boundary_=true;return logical_size_;}
  // Blocks preserve the original thread-path partition and logical IDs.
  template<class Visitor> void forEachAffineColumnWithBlock(Visitor&& visit,bool include_generated=true,std::size_t first_record=0) const
    requires requires { typename Codec::Delta; } {
    forEachColumnRun([&](const auto& value,const Delta& delta,std::size_t first,std::size_t stride,std::size_t count,
        std::size_t begin,std::size_t width,std::size_t members) {
      visit(value(),delta,first,stride,count,begin,width,members);
    },include_generated,first_record);
  }
  // The same runs, every record included, with visit(delta, first, stride, count, value) where
  // value() reconstructs the run's first record only when called.
  template<class Visitor> void forEachAffineColumnRun(Visitor&& visit) const
    requires requires { typename Codec::Delta; } {
    forEachColumnRun([&](const auto& value,const Delta& delta,std::size_t first,std::size_t stride,std::size_t count,
        std::size_t,std::size_t,std::size_t) { visit(delta,first,stride,count,value); },true);
  }
 private:
  // forEachAffineColumnWithBlock's runs; `value()` reconstructs a run's first record only when called.
  template<class Visitor> void forEachColumnRun(Visitor&& visit,bool include_generated,std::size_t first_record=0) const {
    std::vector<std::size_t> edits;
    for(const auto& [index,row]:edits_) { (void)row;edits.push_back(index); }
    std::sort(edits.begin(),edits.end());
    const auto first_segment=std::lower_bound(segments_.begin(),segments_.end(),first_record,
        [](const auto& segment,std::size_t first){return segment.begin+segment.width*segment.repeats<=first;});
    for(auto segment_it=first_segment;segment_it!=segments_.end();++segment_it) {
      const auto& segment=*segment_it;
      if(segment.generate && !include_generated)continue;
      std::unordered_map<std::size_t,std::vector<std::size_t>> changed;
      auto edit=std::lower_bound(edits.begin(),edits.end(),segment.begin);
      const auto end=segment.begin+segment.width*segment.repeats;
      for(;edit!=edits.end() && *edit<end;++edit)
        changed[(*edit-segment.begin)%segment.width].push_back((*edit-segment.begin)/segment.width);
      const auto emit=[&](std::size_t column,std::size_t member,std::size_t count,const Delta& delta) {
        const auto report=[&](std::size_t first,std::size_t length,const Delta& step) {
          if(!length)return;
          auto begin=segment.begin+column+segment.width*first;
          if(begin<first_record) {
            const auto skip=(first_record-begin+segment.width-1)/segment.width;
            if(skip>=length)return;
            begin+=skip*segment.width;length-=skip;
          }
          visit([&]{return at(begin);},step,begin,segment.width,length,segment.begin,segment.width,segment.repeats);
        };
        const auto found=changed.find(column);
        if(found==changed.end()) { report(member,count,delta);return; }
        auto next=member;
        for(auto it=std::lower_bound(found->second.begin(),found->second.end(),member);
            it!=found->second.end() && *it<member+count;++it) {
          report(next,*it-next,delta);report(*it,1,Delta{});next=*it+1;
        }
        report(next,member+count-next,delta);
      };
      if(segment.generate) {
        if(segment.generated_columns) {
          segment.generated_columns([&](const T& value,const Delta& delta,std::size_t first,std::size_t stride,std::size_t count) {
            if(!count || first>=segment.width)return;
            if(count>1 && !stride)throw std::logic_error("generated source column repeats a logical record");
            // The producer may append to its family after a container snapshot.
            // Earlier coordinates remain immutable; clip to this snapshot's
            // logical extent without visiting any newly appended members.
            if(stride)count=std::min(count,1+(segment.width-1-first)/stride);
            auto cursor=std::size_t{0};
            const auto report=[&](std::size_t start,std::size_t length) {
              if(!length)return;
              const auto begin=segment.begin+first+stride*start;
              if(begin<first_record) {
                if(!stride)return;
                const auto skip=(first_record-begin+stride-1)/stride;
                if(skip>=length)return;
                start+=skip;length-=skip;
              }
              auto instance=value;typename Codec::Binding fields;
              Codec::split(instance,fields);const auto translated=Codec::translate(fields,delta,start);
              if(!translated)throw std::overflow_error("generated source column overflow");
              Codec::bind(instance,*translated);
              visit([&]()->const T&{return instance;},delta,segment.begin+first+stride*start,stride,length,segment.begin,segment.width,1);
            };
            for(auto it=std::lower_bound(edits.begin(),edits.end(),segment.begin+first);
                it!=edits.end() && *it<=segment.begin+first+stride*(count-1);++it) {
              const auto offset=*it-segment.begin-first;
              if(stride && offset%stride)continue;
              const auto member=stride?offset/stride:0;
              report(cursor,member-cursor);
              if(*it>=first_record)visit([&]{return at(*it);},Delta{},*it,stride,1,segment.begin,segment.width,1);cursor=member+1;
            }
            report(cursor,count-cursor);
          });
          continue;
        }
        for(std::size_t i=0;i<segment.width*segment.repeats;++i)
          emit(i,0,1,Delta{});
        continue;
      }
      for(std::size_t i=0;i<segment.width;++i) {
        const auto stored=storedColumn(segment,i).column;
        if(!segment.columns.empty()) {
          for(const auto& run:segment.columns[stored])emit(i,run.begin,run.end-run.begin,run.delta);
          continue;
        }
        const auto delta=segment.repeats>1?segment.deltas[segment.deltas.size()==1?0:stored]:Delta{};
        emit(i,0,segment.repeats,delta);
      }
    }
  }
 public:
  template<class Visitor> void forEachAffineColumn(Visitor&& visit) const
    requires requires { typename Codec::Delta; } {
    forEachAffineColumnWithBlock([&](const T& value,const Delta& delta,std::size_t first,
        std::size_t stride,std::size_t count,auto,auto,auto){visit(value,delta,first,stride,count);});
  }
  template<class Visitor> void forEachAffineColumnAfter(std::size_t first,Visitor&& visit) const
    requires requires { typename Codec::Delta; } {
    forEachAffineColumnWithBlock([&](const T& value,const Delta& delta,std::size_t begin,
        std::size_t stride,std::size_t count,auto,auto,auto){visit(value,delta,begin,stride,count);},true,first);
  }
  // Seal finalized thread bindings; differences split only their own column.
  // A run of a non-concrete row (stored, or an edit that holds a template row) re-interns as its
  // prototype and normalized binding, i.e. what intern(at(index)) returns, without reconstructing
  // the record: one prototype lookup per stored prototype, O(1) per run otherwise. Concrete
  // records (edit() makes its record concrete), and every record after rewriteRecords, are
  // re-interned from the record.
  void sealThreadFamilies() requires requires { typename Codec::Delta; } {
    if(edits_.empty())return;
    TemplateRecords result;
    std::vector<std::size_t> edits;
    for(const auto& [index,row]:edits_){(void)row;edits.push_back(index);}
    std::sort(edits.begin(),edits.end());
    std::vector<std::uint32_t> sealed_prototypes(prototypes_.size(),concrete);
    const auto sealedRow=[&](std::size_t index,const auto& value) {
      if constexpr(requires(const T& prototype,typename Codec::Binding& binding) {
                     Codec::normalize(prototype,binding); }) {
        auto row=rowAt(index);
        if(!rewritten_ && !(row.prototype&concrete)) {
          const auto& prototype=prototypes_[row.prototype];
          auto& id=sealed_prototypes[row.prototype];
          if(id==concrete)id=result.internPrototype(prototype);
          Codec::normalize(prototype,row.binding);
          return Row{id,std::move(row.binding)};
        }
      }
      return result.intern(value());
    };
    const auto generatorsUntil=[&](std::size_t end) {
      while(result.size()<end) {
        const auto& segment=segmentAt(result.size());
        if(!segment.generate)throw std::logic_error("missing family block");
        const auto first=result.size();
        result.appendGenerated(segment.width*segment.repeats,segment.generate,segment.cohort,
            segment.generated_columns,segment.generated_family);
        for(auto it=std::lower_bound(edits.begin(),edits.end(),first);it!=edits.end() && *it<result.size();++it)
          result.edits_.insert_or_assign(*it,result.intern(at(*it)));
      }
    };
    // Ordered iterations without an edited record keep their layout: run decisions depend only
    // on binding differences, which every repetition shares, so the first repetition stands for all.
    // Edited ones seal unrolled and fold back where the sealed repetitions are congruent.
    std::vector<Iteration> edited_iterations;
    const auto uneditedIterations=[&](const Segment& segment) {
      std::vector<Iteration> kept;
      edited_iterations.clear();
      if(segment.iterations.empty())return kept;
      std::vector<bool> edited(segment.iterations.size());
      const auto end=segment.begin+segment.width*segment.repeats;
      for(auto it=std::lower_bound(edits.begin(),edits.end(),segment.begin);it!=edits.end() && *it<end;++it) {
        const auto column=(*it-segment.begin)%segment.width;
        const auto block=std::upper_bound(segment.iterations.begin(),segment.iterations.end(),column,
          [](std::size_t x,const Iteration& b){return x<b.start;});
        if(block!=segment.iterations.begin() && column<std::prev(block)->start+std::prev(block)->width*std::prev(block)->count)
          edited[std::prev(block)-segment.iterations.begin()]=true;
      }
      for(std::size_t i=0;i<segment.iterations.size();++i)
        (edited[i]?edited_iterations:kept).push_back(segment.iterations[i]);
      placeIterations(kept);
      return kept;
    };
    std::optional<std::pair<std::size_t,std::vector<Iteration>>> refolding;
    const auto refold=[&] {
      if(refolding)result.refoldIterations(result.segments_[refolding->first],refolding->second);
      refolding.reset();
    };
    forEachColumnRun([&](const auto& value,const Delta& delta,std::size_t first,
        std::size_t,std::size_t count,std::size_t begin,std::size_t width,std::size_t members) {
      if(result.segments_.empty() || result.segments_.back().begin!=begin) {
        refold();
        generatorsUntil(begin);
        const auto& source=segmentAt(begin);
        Segment block{begin,result.rows_.size(),width,members,source.cohort,{}};
        block.iterations=uneditedIterations(source);
        block.columns.resize(storedWidth(block));result.segments_.push_back(std::move(block));
        result.logical_size_=begin+width*members;result.boundary_=true;
        if(!edited_iterations.empty())refolding.emplace(result.segments_.size()-1,edited_iterations);
      }
      const auto member=(first-begin)/width;
      const auto stored=storedColumn(result.segments_.back(),(first-begin)%width);
      if(stored.iteration)return;
      auto& runs=result.segments_.back().columns[stored.column];
      auto row=sealedRow(first,value);
      if(!runs.empty()) {
        auto& run=runs.back();const auto& base=result.rows_[run.row];
        if(run.end==member && base.prototype==row.prototype && !(row.prototype&concrete)) {
          auto step=run.end-run.begin==1?Codec::difference(base.binding,row.binding):std::optional(run.delta);
          if(step && memberStep(*step) && (count==1 || *step==delta)) {
            auto check=Codec::translate(base.binding,*step,member-run.begin);
            auto end=Codec::translate(base.binding,*step,member+count-1-run.begin);
            if(check && *check==row.binding && end) {
              run.delta=std::move(*step);run.end=member+count;return;
            }
          }
        }
      }
      runs.push_back({member,member+count,result.rows_.size(),delta});
      result.rows_.push_back(std::move(row));
    },false);
    refold();
    generatorsUntil(size());
    *this=std::move(result);
  }
  bool extendThreadFamily(std::size_t begin,std::uint64_t scope,std::uintptr_t history)
    requires requires { typename Codec::Delta; } {
    return compressThreadTail(begin,scope,history,true);
  }
  // Ordered iterations of the tail thread [begin, size()): a block of count >= kMinRepetitions
  // repetitions of `width` consecutive records is stored once when every repetition has the first
  // one's prototypes and advances each column's binding by one delta. `starts` are the
  // thread-relative positions where an iteration of a loop (its header key) begins, in order; a
  // repetition is 1 to kMaxIterationsPerRepetition consecutive iterations of one header. Exact:
  // every record is compared with its prediction. O(records * repetition widths tried); the rows
  // shrink to the stored ones.
  void compressIterations(std::size_t begin,const std::vector<std::pair<std::size_t,std::uint32_t>>& starts)
    requires requires { typename Codec::Delta; } {
    if(segments_.empty() || starts.size()<kMinRepetitions)return;
    auto& tail=segments_.back();
    if(tail.begin!=begin || tail.repeats!=1 || tail.generate || !tail.columns.empty() || tail.cohort ||
       !tail.iterations.empty() || tail.physical+tail.width!=rows_.size())return;
    if(!edits_.empty())for(std::size_t i=0;i<tail.width;++i)if(edits_.contains(begin+i))return;
    const auto n=tail.width;const auto* rows=rows_.data()+tail.physical;
    std::unordered_map<std::uint32_t,std::vector<std::size_t>> headers;
    for(const auto& [offset,header]:starts)headers[header].push_back(offset);
    std::vector<Iteration> blocks;
    std::size_t decided=0;
    for(const auto& [offset,header]:starts) {
      if(offset<decided)continue;
      const auto& same=headers.at(header);
      const auto k=std::size_t(std::lower_bound(same.begin(),same.end(),offset)-same.begin());
      for(std::size_t p=1;p<=kMaxIterationsPerRepetition && k+p<same.size();++p) {
        const auto width=same[k+p]-offset;
        if(!width || kMinRepetitions*width>n-offset)break;
        bool shaped=true;
        for(std::size_t c=0;c<width && shaped;++c) {
          const auto prototype=rows[offset+c].prototype;
          shaped=!(prototype&concrete);
          for(std::size_t r=1;r<kMinRepetitions && shaped;++r)shaped=rows[offset+r*width+c].prototype==prototype;
        }
        if(!shaped)continue;
        std::vector<Delta> deltas;deltas.reserve(width);
        for(std::size_t c=0;c<width && shaped;++c) {
          auto delta=Codec::difference(rows[offset+c].binding,rows[offset+width+c].binding);
          if(delta)deltas.push_back(std::move(*delta));else shaped=false;
        }
        if(!shaped)continue;
        const auto repeats=[&](std::size_t r) {
          for(std::size_t c=0;c<width;++c) {
            const auto& base=rows[offset+c];const auto& next=rows[offset+r*width+c];
            if(next.prototype!=base.prototype)return false;
            const auto expected=Codec::translate(base.binding,deltas[c],r);
            if(!expected || !(*expected==next.binding))return false;
          }
          return true;
        };
        std::size_t count=1;
        while((count+1)*width<=n-offset && repeats(count))++count;
        if(count<kMinRepetitions)continue;
        blocks.push_back({offset,width,count,0,std::move(deltas)});
        decided=offset+width*count;break;
      }
    }
    if(blocks.empty())return;
    std::vector<Row> stored;
    std::size_t next=0;
    for(auto& block:blocks) {
      stored.insert(stored.end(),rows+next,rows+block.start);
      block.stored=stored.size();
      stored.insert(stored.end(),rows+block.start,rows+block.start+block.width);
      next=block.start+block.width*block.count;
    }
    stored.insert(stored.end(),rows+next,rows+n);
    rows_.resize(tail.physical);
    rows_.insert(rows_.end(),std::make_move_iterator(stored.begin()),std::make_move_iterator(stored.end()));
    tail.iterations=std::move(blocks);
  }
  // Column splitting is restricted to explicitly supplied thread families.
  bool compressThreadTail(std::size_t begin,std::uint64_t scope,std::uintptr_t history,bool split_columns=false)
    requires requires { typename Codec::Delta; } {
    boundary_=true;
    if(begin==logical_size_ || segments_.empty())return false;
    auto& tail=segments_.back();
    if(tail.begin!=begin || tail.repeats!=1 || tail.generate || !tail.columns.empty())return false;
    tail.cohort=std::pair{scope,history};
    if(segments_.size()<2)return false;
    auto& previous=segments_[segments_.size()-2];
    if(previous.generate || previous.cohort!=tail.cohort || previous.width!=tail.width)return false;
    for(std::size_t i=0;i<tail.width;++i)if(edits_.contains(tail.begin+i))return false;
    if(previous.iterations!=tail.iterations)unrollIterations(previous,tail);
    // Equal iteration layouts: the stored columns decide every repetition alike.
    const auto width=storedWidth(tail);
    if(!split_columns) {
      if(!previous.columns.empty() || previous.physical+width!=tail.physical)return false;
      std::vector<Delta> deltas;
      for(std::size_t i=0;i<width;++i) {
        const auto& first=rows_[previous.physical+i];const auto& next=rows_[tail.physical+i];
        if(first.prototype!=next.prototype || (first.prototype&concrete))return false;
        auto delta=previous.repeats==1?Codec::difference(first.binding,next.binding):
          std::optional(previous.deltas[previous.deltas.size()==1?0:i]);
        if(!delta || !memberStep(*delta))return false;
        auto expected=Codec::translate(first.binding,*delta,previous.repeats);
        if(!expected || *expected!=next.binding)return false;
        if(previous.repeats==1)deltas.push_back(std::move(*delta));
      }
      if(previous.repeats==1) {
        if(std::all_of(deltas.begin(),deltas.end(),[&](const auto& delta){return delta==deltas.front();}))
          deltas.resize(1);
        previous.deltas=std::move(deltas);
      }
      ++previous.repeats;rows_.resize(tail.physical);segments_.pop_back();return true;
    }
    if(previous.columns.empty())runColumns(previous);
    std::vector<Row> added;
    for(std::size_t i=0;i<width;++i) {
      auto& runs=previous.columns[i];auto& run=runs.back();
      const auto& first=rows_[run.row];const auto& next=rows_[tail.physical+i];
      bool extend=false;
      if(first.prototype==next.prototype && !(first.prototype&concrete)) {
        auto delta=run.end-run.begin==1?Codec::difference(first.binding,next.binding):std::optional(run.delta);
        if(delta && memberStep(*delta)) {
          auto expected=Codec::translate(first.binding,*delta,previous.repeats-run.begin);
          if(expected && *expected==next.binding){run.delta=std::move(*delta);extend=true;}
        }
      }
      if(extend)++run.end;
      else {
        runs.push_back({previous.repeats,previous.repeats+1,tail.physical+added.size(),{}});
        added.push_back(next);
      }
    }
    ++previous.repeats;rows_.resize(tail.physical);
    rows_.insert(rows_.end(),std::make_move_iterator(added.begin()),std::make_move_iterator(added.end()));
    segments_.pop_back();return true;
  }
  friend bool operator==(const TemplateRecords& a,const TemplateRecords& b){return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin());}
};

inline void HashCombine(std::size_t& seed,std::size_t value){seed^=value+0x9e3779b9+(seed<<6)+(seed>>2);}
} // namespace sentryward::algorithm
