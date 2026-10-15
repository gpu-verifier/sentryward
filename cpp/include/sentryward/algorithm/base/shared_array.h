#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sentryward::algorithm {

// A compact, immutable array. The reference count and elements occupy one
// allocation; the handle is one pointer. Copies are O(1) and outlive the source.
// append/set replace the buffer rather than exposing mutable references, so
// modifying a source effect cannot silently alter an already lowered event.
template <class T>
class SharedArray final {
  struct alignas(alignof(T) > alignof(std::atomic<std::size_t>)
                     ? alignof(T) : alignof(std::atomic<std::size_t>)) Buffer {
    std::atomic<std::size_t> references{1};
    std::size_t count;
    explicit Buffer(std::size_t n) : count(n) {}
    T* values() noexcept { return reinterpret_cast<T*>(this + 1); }
    const T* values() const noexcept { return reinterpret_cast<const T*>(this + 1); }
  };

 public:
  using value_type = T;
  using const_iterator = const T*;
  SharedArray() noexcept = default;
  SharedArray(std::initializer_list<T> values)
      : SharedArray(values.begin(), values.end(), values.size()) {}
  SharedArray(const std::vector<T>& values)
      : SharedArray(values.begin(), values.end(), values.size()) {}
  SharedArray(std::vector<T>&& values)
      : SharedArray(std::make_move_iterator(values.begin()),
                    std::make_move_iterator(values.end()), values.size()) {}
  SharedArray(const SharedArray& other) noexcept : buffer_(other.buffer_) {
    if (buffer_) buffer_->references.fetch_add(1, std::memory_order_relaxed);
  }
  SharedArray(SharedArray&& other) noexcept
      : buffer_(std::exchange(other.buffer_, nullptr)) {}
  SharedArray& operator=(SharedArray other) noexcept {
    swap(other);
    return *this;
  }
  ~SharedArray() { release(); }

  void swap(SharedArray& other) noexcept { std::swap(buffer_, other.buffer_); }
  [[nodiscard]] bool empty() const noexcept { return !buffer_; }
  [[nodiscard]] std::size_t size() const noexcept { return buffer_ ? buffer_->count : 0; }
  [[nodiscard]] std::size_t capacity() const noexcept { return size(); }
  [[nodiscard]] const T* data() const noexcept { return buffer_ ? buffer_->values() : nullptr; }
  [[nodiscard]] const_iterator begin() const noexcept { return data(); }
  [[nodiscard]] const_iterator end() const noexcept { return buffer_ ? data() + size() : nullptr; }
  [[nodiscard]] const T& operator[](std::size_t index) const noexcept { return data()[index]; }
  [[nodiscard]] const T& at(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("SharedArray index");
    return (*this)[index];
  }
  [[nodiscard]] const T& front() const noexcept { return (*this)[0]; }
  [[nodiscard]] const T& back() const noexcept { return (*this)[size() - 1]; }
  [[nodiscard]] const void* storage_identity() const noexcept { return buffer_; }
  [[nodiscard]] std::size_t use_count() const noexcept {
    return buffer_ ? buffer_->references.load(std::memory_order_relaxed) : 0;
  }
  static constexpr std::size_t storage_header_bytes() noexcept { return sizeof(Buffer); }

  // Copy once on the first changed element; retain unchanged snapshots.
  template<class Transform>
  [[nodiscard]] SharedArray transformed(Transform transform) const {
    if (empty()) return {};
    Buffer* next = nullptr;
    std::size_t constructed = 0;
    try {
      for (std::size_t i = 0; i < size(); ++i) {
        auto value = transform(i, (*this)[i]);
        if (!next) {
          if constexpr (requires { value == (*this)[i]; })
            if (value == (*this)[i]) continue;
          next = allocate(size());
          for (; constructed < i; ++constructed)
            std::construct_at(next->values() + constructed, (*this)[constructed]);
        }
        std::construct_at(next->values() + constructed, std::move(value));
        ++constructed;
      }
    } catch (...) {
      if (next) {
        std::destroy_n(next->values(), constructed);
        deallocate(next);
      }
      throw;
    }
    if (!next) return *this;
    SharedArray result;
    result.buffer_ = next;
    return result;
  }

  void clear() noexcept { SharedArray{}.swap(*this); }
  void push_back(T value) {
    if (size() == std::numeric_limits<std::size_t>::max()) throw std::length_error("SharedArray size");
    Buffer* next = allocate(size() + 1);
    std::size_t constructed = 0;
    try {
      for (; constructed < size(); ++constructed)
        std::construct_at(next->values() + constructed, (*this)[constructed]);
      std::construct_at(next->values() + constructed, std::move(value));
    } catch (...) {
      std::destroy_n(next->values(), constructed);
      deallocate(next);
      throw;
    }
    release();
    buffer_ = next;
  }
  void set(std::size_t index, T value) {
    if (index >= size()) throw std::out_of_range("SharedArray index");
    Buffer* next = allocate(size());
    std::size_t constructed = 0;
    try {
      for (; constructed < size(); ++constructed) {
        if (constructed == index)
          std::construct_at(next->values() + constructed, std::move(value));
        else
          std::construct_at(next->values() + constructed, (*this)[constructed]);
      }
    } catch (...) {
      std::destroy_n(next->values(), constructed);
      deallocate(next);
      throw;
    }
    release();
    buffer_ = next;
  }
  friend bool operator==(const SharedArray& a, const SharedArray& b) {
    return a.buffer_ == b.buffer_ ||
           (a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()));
  }

 private:
  template <class Iterator>
  SharedArray(Iterator first, Iterator last, std::size_t count) {
    if (!count) return;
    buffer_ = allocate(count);
    try { std::uninitialized_copy(first, last, buffer_->values()); }
    catch (...) { deallocate(buffer_); throw; }
  }
  static Buffer* allocate(std::size_t count) {
    if (count > (std::numeric_limits<std::size_t>::max() - sizeof(Buffer)) / sizeof(T))
      throw std::length_error("SharedArray allocation");
    return new (::operator new(sizeof(Buffer) + count * sizeof(T),
                               std::align_val_t{alignof(Buffer)})) Buffer(count);
  }
  static void deallocate(Buffer* buffer) noexcept {
    buffer->~Buffer();
    ::operator delete(buffer, std::align_val_t{alignof(Buffer)});
  }
  void release() noexcept {
    if (buffer_ && buffer_->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::destroy_n(buffer_->values(), buffer_->count);
      deallocate(buffer_);
    }
  }
  Buffer* buffer_ = nullptr;
};

}  // namespace sentryward::algorithm
