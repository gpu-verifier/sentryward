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
  struct Segment {
    std::size_t begin=0, physical=0, width=0, repeats=1;
    std::optional<std::pair<std::uint64_t,std::uintptr_t>> cohort;
    std::vector<Delta> deltas;
    std::function<T(std::size_t)> generate = {};
    std::vector<std::vector<ColumnRun>> columns = {};
    GeneratedColumns generated_columns = {};
    const void* generated_family = nullptr;
  };
  std::vector<Row> rows_;
  std::vector<Segment> segments_;
  std::unordered_map<std::size_t,Row> edits_;
  std::size_t logical_size_=0;
  bool boundary_=false;
  bool has_generated_=false;
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
    if(!segment->columns.empty()) {
      const auto& runs=segment->columns[position%segment->width];
      const auto member=position/segment->width;
      const auto& run=*std::prev(std::upper_bound(runs.begin(),runs.end(),member,
        [](std::size_t value,const ColumnRun& r){return value<r.begin;}));
      auto row=rows_[run.row];
      if(member!=run.begin) {
        if constexpr(requires { typename Codec::Delta; }) {
          auto binding=Codec::translate(row.binding,run.delta,member-run.begin);
          if(!binding)throw std::logic_error("invalid family binding");
          row.binding=std::move(*binding);
        }
      }
      return row;
    }
    auto row=rows_[segment->physical+position%segment->width];
    if(segment->repeats>1) {
      if constexpr(requires { typename Codec::Delta; }) {
        auto binding=Codec::translate(row.binding,segment->deltas[segment->deltas.size()==1?0:position%segment->width],position/segment->width);
        if(!binding)throw std::logic_error("invalid compressed record binding");
        row.binding=*binding;
      } else throw std::logic_error("record codec has no binding translation");
    }
    return row;
  }
  template<class Projection>
  auto projectHeader(std::size_t i, Projection project) const {
    if (i >= logical_size_) throw std::out_of_range("template record index");
    const auto apply = [&](const Row& row, const Delta* delta, std::size_t member) {
      const auto& prototype = prototypes_.at(row.prototype & ~concrete);
      return project(prototype, row.prototype & concrete ? nullptr : &row.binding,
                     delta, member);
    };
    if (!edits_.empty())
      if (const auto found = edits_.find(i); found != edits_.end())
        return apply(found->second, nullptr, 0);
    const auto& segment = segmentAt(i);
    const auto position = i - segment.begin;
    if (segment.generate) return project(segment.generate(position), nullptr, nullptr, 0);
    if (!segment.columns.empty()) {
      const auto& runs=segment.columns[position%segment.width];
      const auto member=position/segment.width;
      const auto& run=*std::prev(std::upper_bound(runs.begin(),runs.end(),member,
        [](std::size_t value,const ColumnRun& r){return value<r.begin;}));
      return apply(rows_[run.row],member==run.begin?nullptr:&run.delta,member-run.begin);
    }
    const auto column = position % segment.width;
    const Delta* delta = segment.repeats > 1
        ? &segment.deltas[segment.deltas.size() == 1 ? 0 : column] : nullptr;
    return apply(rows_[segment.physical + column], delta, position / segment.width);
  }
  void appendRow(Row row) {
    if(boundary_ || segments_.empty() || segments_.back().repeats!=1 || segments_.back().cohort || !segments_.back().columns.empty()) {
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
    if(segments_.size()==1 && segments_[0].repeats==1 && segments_[0].columns.empty() && edits_.empty()) {
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
    auto& candidates=index_[Codec::hash(value)];
    for(auto id:candidates)if(prototypes_[id]==value)return {id,binding};
    auto id=static_cast<std::uint32_t>(prototypes_.size());
    prototypes_.push_back(std::move(value));candidates.push_back(id);return {id,binding};
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
    has_generated_=std::exchange(other.has_generated_,false);
    return *this;
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
  std::size_t bindingTemplateCount() const noexcept{return segments_.size();}
  std::size_t ownedBytes() const noexcept{
    auto bytes=rows_.capacity()*sizeof(Row)+prototypes_.size()*sizeof(T)+segments_.capacity()*sizeof(Segment);
    for(const auto& segment:segments_)bytes+=segment.deltas.capacity()*sizeof(Delta);
    for(const auto& segment:segments_) {
      bytes+=segment.columns.capacity()*sizeof(std::vector<ColumnRun>);
      for(const auto& column:segment.columns)bytes+=column.capacity()*sizeof(ColumnRun);
    }
    return bytes+edits_.size()*sizeof(std::pair<const std::size_t,Row>);
  }
  void reserve(std::size_t n){rows_.reserve(n);}
  void shrink_to_fit(){rows_.shrink_to_fit();segments_.shrink_to_fit();}
  void clear(){rows_.clear();segments_.clear();edits_.clear();logical_size_=0;boundary_=false;has_generated_=false;prototypes_.clear();index_.clear();}
  Iterator begin() const{return {this,0};}
  Iterator end() const{return {this,static_cast<std::ptrdiff_t>(size())};}
  auto rbegin() const{return std::make_reverse_iterator(end());}
  auto rend() const{return std::make_reverse_iterator(begin());}
  const T at(std::size_t i) const {
    if(generatedAt(i)) {
      const auto& segment=segmentAt(i);
      return segment.generate(i-segment.begin);
    }
    const auto row=rowAt(i);
    auto result=prototypes_.at(row.prototype&~concrete);
    if(!(row.prototype&concrete)) Codec::bind(result,row.binding);
    return result;
  }
  const T operator[](std::size_t i) const{return at(i);}
  // Header projections do not reconstruct operands of stored templates.
  auto id(std::size_t i) const requires requires(const T& value) { value.id.value(); } {
    return projectHeader(i, [](const T& prototype, const typename Codec::Binding* binding,
                               const Delta* delta, std::size_t member) {
      if (!binding) return prototype.id;
      auto value = binding->id.value();
      if constexpr (requires { (*delta)[0]; })
        if (delta) value = static_cast<std::uint32_t>(value + (*delta)[0]*std::int64_t(member));
      return decltype(prototype.id)(value);
    });
  }
  auto actor(std::size_t i) const requires requires(const T& value) { value.actor.value(); } {
    return projectHeader(i, [](const T& prototype, const typename Codec::Binding* binding,
                               const Delta* delta, std::size_t member) {
      if (!binding) return prototype.actor;
      auto value = binding->actor.value();
      if constexpr (requires { (*delta)[1]; })
        if (delta) value = static_cast<std::uint32_t>(value + (*delta)[1]*std::int64_t(member));
      return decltype(prototype.actor)(value);
    });
  }
  std::size_t effectCount(std::size_t i) const
      requires requires(const T& value) { Codec::effectCount(value); } {
    return projectHeader(i, [](const T& prototype, const typename Codec::Binding*,
                               const Delta*, std::size_t) { return Codec::effectCount(prototype); });
  }
  const T front() const{return at(0);}
  template<class Op> bool holdsOperation(std::size_t i) const {
    return projectHeader(i, [](const T& prototype, const typename Codec::Binding*,
                               const Delta*, std::size_t) { return std::holds_alternative<Op>(prototype.op); });
  }
  auto opcode(std::size_t i) const {
    return projectHeader(i, [](const T& prototype, const typename Codec::Binding*,
                               const Delta*, std::size_t) { return Codec::opcode(prototype); });
  }
  const T back() const{return at(size()-1);}
  bool isConcrete(std::size_t i) const{return !generatedAt(i) && (rowAt(i).prototype&concrete);}
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
          visit(at(begin),step,begin,segment.width,length,segment.begin,segment.width,segment.repeats);
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
              visit(instance,delta,segment.begin+first+stride*start,stride,length,segment.begin,segment.width,1);
            };
            for(auto it=std::lower_bound(edits.begin(),edits.end(),segment.begin+first);
                it!=edits.end() && *it<=segment.begin+first+stride*(count-1);++it) {
              const auto offset=*it-segment.begin-first;
              if(stride && offset%stride)continue;
              const auto member=stride?offset/stride:0;
              report(cursor,member-cursor);
              if(*it>=first_record)visit(at(*it),Delta{},*it,stride,1,segment.begin,segment.width,1);cursor=member+1;
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
        if(!segment.columns.empty()) {
          for(const auto& run:segment.columns[i])emit(i,run.begin,run.end-run.begin,run.delta);
          continue;
        }
        const auto delta=segment.repeats>1?segment.deltas[segment.deltas.size()==1?0:i]:Delta{};
        emit(i,0,segment.repeats,delta);
      }
    }
  }
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
  void sealThreadFamilies() requires requires { typename Codec::Delta; } {
    if(edits_.empty())return;
    TemplateRecords result;
    std::vector<std::size_t> edits;
    for(const auto& [index,row]:edits_){(void)row;edits.push_back(index);}
    std::sort(edits.begin(),edits.end());
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
    forEachAffineColumnWithBlock([&](const T& value,const Delta& delta,std::size_t first,
        std::size_t,std::size_t count,std::size_t begin,std::size_t width,std::size_t members) {
      if(result.segments_.empty() || result.segments_.back().begin!=begin) {
        generatorsUntil(begin);
        Segment block{begin,result.rows_.size(),width,members,segmentAt(begin).cohort,{}};
        block.columns.resize(width);result.segments_.push_back(std::move(block));
        result.logical_size_=begin+width*members;result.boundary_=true;
      }
      const auto member=(first-begin)/width;
      auto& runs=result.segments_.back().columns[(first-begin)%width];
      auto row=result.intern(value);
      if(!runs.empty()) {
        auto& run=runs.back();const auto& base=result.rows_[run.row];
        if(run.end==member && base.prototype==row.prototype && !(row.prototype&concrete)) {
          auto step=run.end-run.begin==1?Codec::difference(base.binding,row.binding):std::optional(run.delta);
          if(step && (count==1 || *step==delta)) {
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
    generatorsUntil(size());
    *this=std::move(result);
  }
  bool extendThreadFamily(std::size_t begin,std::uint64_t scope,std::uintptr_t history)
    requires requires { typename Codec::Delta; } {
    return compressThreadTail(begin,scope,history,true);
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
    if(!split_columns) {
      if(!previous.columns.empty() || previous.physical+previous.width!=tail.physical)return false;
      std::vector<Delta> deltas;
      for(std::size_t i=0;i<previous.width;++i) {
        const auto& first=rows_[previous.physical+i];const auto& next=rows_[tail.physical+i];
        if(first.prototype!=next.prototype || (first.prototype&concrete))return false;
        auto delta=previous.repeats==1?Codec::difference(first.binding,next.binding):
          std::optional(previous.deltas[previous.deltas.size()==1?0:i]);
        if(!delta)return false;
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
    if(previous.columns.empty()) {
      previous.columns.resize(previous.width);
      for(std::size_t i=0;i<previous.width;++i)
        previous.columns[i].push_back({0,previous.repeats,previous.physical+i,
          previous.repeats>1?previous.deltas[previous.deltas.size()==1?0:i]:Delta{}});
      previous.deltas.clear();
    }
    std::vector<Row> added;
    for(std::size_t i=0;i<previous.width;++i) {
      auto& runs=previous.columns[i];auto& run=runs.back();
      const auto& first=rows_[run.row];const auto& next=rows_[tail.physical+i];
      bool extend=false;
      if(first.prototype==next.prototype && !(first.prototype&concrete)) {
        auto delta=run.end-run.begin==1?Codec::difference(first.binding,next.binding):std::optional(run.delta);
        if(delta) {
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
