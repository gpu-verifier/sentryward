#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sentryward::algorithm {

class DynamicBitset final {
 public:
  using SizeType = std::size_t;
  static constexpr SizeType npos = std::numeric_limits<SizeType>::max();

  DynamicBitset() = default;

  explicit DynamicBitset(SizeType size, bool value = false)
      : size_(size), words_(wordCount(size), value ? ~Word{0} : Word{0}) {
    clearUnusedBits();
  }

  [[nodiscard]] SizeType size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  void resize(SizeType size, bool value = false) {
    const SizeType old_size = size_;
    words_.resize(wordCount(size), value ? ~Word{0} : Word{0});
    size_ = size;

    if (value && size > old_size && old_size % kWordBits != 0 && old_size != 0) {
      const Word mask = ~Word{0} << (old_size % kWordBits);
      words_[wordIndex(old_size)] |= mask;
    }
    clearUnusedBits();
  }

  [[nodiscard]] bool test(SizeType index) const {
    checkIndex(index);
    return (words_[wordIndex(index)] & bitMask(index)) != 0;
  }

  [[nodiscard]] bool operator[](SizeType index) const { return test(index); }

  DynamicBitset& set(SizeType index, bool value = true) {
    checkIndex(index);
    if (value) {
      words_[wordIndex(index)] |= bitMask(index);
    } else {
      words_[wordIndex(index)] &= ~bitMask(index);
    }
    return *this;
  }

  DynamicBitset& reset(SizeType index) { return set(index, false); }

  DynamicBitset& set() noexcept {
    for (Word& word : words_) {
      word = ~Word{0};
    }
    clearUnusedBits();
    return *this;
  }

  DynamicBitset& reset() noexcept {
    for (Word& word : words_) {
      word = 0;
    }
    return *this;
  }

  DynamicBitset& flip() noexcept {
    for (Word& word : words_) {
      word = ~word;
    }
    clearUnusedBits();
    return *this;
  }

  [[nodiscard]] bool any() const noexcept {
    for (Word word : words_) {
      if (word != 0) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool none() const noexcept { return !any(); }

  [[nodiscard]] SizeType count() const noexcept {
    SizeType result = 0;
    for (Word word : words_) {
      result += static_cast<SizeType>(std::popcount(word));
    }
    return result;
  }

  [[nodiscard]] SizeType find_first() const noexcept {
    for (SizeType word = 0; word < words_.size(); ++word) {
      if (words_[word] != 0) {
        return word * kWordBits +
               static_cast<SizeType>(std::countr_zero(words_[word]));
      }
    }
    return npos;
  }

  [[nodiscard]] SizeType find_next(SizeType previous) const noexcept {
    if (previous == npos || previous + 1 >= size_) {
      return npos;
    }

    const SizeType next = previous + 1;
    SizeType word = wordIndex(next);
    Word candidates = words_[word] & (~Word{0} << (next % kWordBits));
    if (candidates != 0) {
      return word * kWordBits +
             static_cast<SizeType>(std::countr_zero(candidates));
    }

    for (++word; word < words_.size(); ++word) {
      if (words_[word] != 0) {
        return word * kWordBits +
               static_cast<SizeType>(std::countr_zero(words_[word]));
      }
    }
    return npos;
  }

  [[nodiscard]] bool intersects(const DynamicBitset& other) const {
    requireSameSize(other);
    for (SizeType index = 0; index < words_.size(); ++index) {
      if ((words_[index] & other.words_[index]) != 0) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool is_subset_of(const DynamicBitset& other) const {
    requireSameSize(other);
    for (SizeType index = 0; index < words_.size(); ++index) {
      if ((words_[index] & ~other.words_[index]) != 0) {
        return false;
      }
    }
    return true;
  }

  DynamicBitset& operator|=(const DynamicBitset& other) {
    requireSameSize(other);
    for (SizeType index = 0; index < words_.size(); ++index) {
      words_[index] |= other.words_[index];
    }
    return *this;
  }

  DynamicBitset& operator&=(const DynamicBitset& other) {
    requireSameSize(other);
    for (SizeType index = 0; index < words_.size(); ++index) {
      words_[index] &= other.words_[index];
    }
    return *this;
  }

  DynamicBitset& operator^=(const DynamicBitset& other) {
    requireSameSize(other);
    for (SizeType index = 0; index < words_.size(); ++index) {
      words_[index] ^= other.words_[index];
    }
    clearUnusedBits();
    return *this;
  }

  DynamicBitset& subtract(const DynamicBitset& other) {
    requireSameSize(other);
    for (SizeType index = 0; index < words_.size(); ++index) {
      words_[index] &= ~other.words_[index];
    }
    return *this;
  }

  friend DynamicBitset operator|(DynamicBitset left,
                                 const DynamicBitset& right) {
    left |= right;
    return left;
  }

  friend DynamicBitset operator&(DynamicBitset left,
                                 const DynamicBitset& right) {
    left &= right;
    return left;
  }

  friend DynamicBitset operator^(DynamicBitset left,
                                 const DynamicBitset& right) {
    left ^= right;
    return left;
  }

  friend bool operator==(const DynamicBitset&,
                         const DynamicBitset&) noexcept = default;

 private:
  using Word = std::uint64_t;
  static constexpr SizeType kWordBits = std::numeric_limits<Word>::digits;

  [[nodiscard]] static SizeType wordCount(SizeType size) noexcept {
    return size / kWordBits + (size % kWordBits != 0 ? 1 : 0);
  }

  [[nodiscard]] static SizeType wordIndex(SizeType index) noexcept {
    return index / kWordBits;
  }

  [[nodiscard]] static Word bitMask(SizeType index) noexcept {
    return Word{1} << (index % kWordBits);
  }

  void checkIndex(SizeType index) const {
    if (index >= size_) {
      throw std::out_of_range("DynamicBitset index is out of range");
    }
  }

  void requireSameSize(const DynamicBitset& other) const {
    if (size_ != other.size_) {
      throw std::invalid_argument("DynamicBitset sizes differ");
    }
  }

  void clearUnusedBits() noexcept {
    if (words_.empty() || size_ % kWordBits == 0) {
      return;
    }
    words_.back() &= (Word{1} << (size_ % kWordBits)) - 1;
  }

  SizeType size_ = 0;
  std::vector<Word> words_;
};

}  // namespace sentryward::algorithm
