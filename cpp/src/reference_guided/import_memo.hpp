#pragma once
#include "core.hpp"
#include <memory>

namespace rg {
// Shared hash for ordered node-ID keys.
struct IdSequenceHash {
  size_t operator()(std::span<const Id> v) const {
    size_t h = 0;
    for (Id x : v)
      mix(h, x);
    return h;
  }
};
// Maps (raw node, projected environment) to an imported node. Keys live in
// stable chunks; linear probing stays below 3/4 load without copying keys.
class ImportMemo {
  struct Slot {
    size_t hash;
    const Id *key;
  };
  static constexpr size_t chunk_words = size_t(1) << 16;
  std::vector<Slot> slots;
  std::vector<std::unique_ptr<Id[]>> chunks;
  Id *cursor = nullptr, *limit = nullptr;
  size_t entries = 0;
  unsigned shift = 64;
  const Id *find(std::span<const Id> k, size_t h) const {
    if (slots.empty())
      return nullptr;
    for (size_t m = slots.size() - 1, i = Arena::index_home(h, shift);; i = (i + 1) & m) {
      const Slot &s = slots[i];
      if (!s.key)
        return nullptr;
      if (s.hash == h && s.key[0] == k.size() && std::equal(k.begin(), k.end(), s.key + 2))
        return s.key + 1;
    }
  }
  // Grow before writing: a failed allocation leaves the stored entries intact.
  void reserve_slot() {
    if (4 * (entries + 1) <= 3 * slots.size())
      return;
    const size_t n = slots.empty() ? 256 : 2 * slots.size();
    std::vector<Slot> grown(n, Slot{0, nullptr});
    const unsigned grown_shift = 64 - unsigned(std::countr_zero(n));
    for (const auto &x : slots)
      if (x.key)
        place(grown, grown_shift, x);
    slots.swap(grown);
    shift = grown_shift;
  }
  static void place(std::vector<Slot> &v, unsigned home_shift, Slot s) noexcept {
    for (size_t m = v.size() - 1, i = Arena::index_home(s.hash, home_shift);; i = (i + 1) & m)
      if (!v[i].key) {
        v[i] = s;
        return;
      }
  }
public:
  // Slots, cursor and limit point into chunks, so the memo is neither copied
  // nor moved (the deleted copy also suppresses the implicit move).
  ImportMemo() = default;
  ImportMemo(const ImportMemo &) = delete;
  ImportMemo &operator=(const ImportMemo &) = delete;
  // A returned value pointer stays valid: stored entries never move or change.
  const Id *find(std::span<const Id> k) const { return find(k, IdSequenceHash{}(k)); }
  bool contains(std::span<const Id> k) const { return find(k); }
  Id at(std::span<const Id> k) const {
    if (const Id *v = find(k))
      return *v;
    throw std::out_of_range("unordered_map::at");
  }
  void emplace(std::span<const Id> k, Id value) {
    const size_t h = IdSequenceHash{}(k);
    if (find(k, h))
      return;
    reserve_slot();
    const size_t n = k.size() + 2;
    if (size_t(limit - cursor) < n) {
      const size_t words = std::max(chunk_words, n);
      chunks.push_back(std::unique_ptr<Id[]>(new Id[words]));
      cursor = chunks.back().get();
      limit = cursor + words;
    }
    Id *w = cursor;
    cursor += n;
    w[0] = Id(k.size());
    w[1] = value;
    std::copy(k.begin(), k.end(), w + 2);
    place(slots, shift, {h, w});
    ++entries;
  }
};
} // namespace rg
