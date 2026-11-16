#pragma once
#include "sentryward/frontend/ptx/execution.h"
#include "sentryward/frontend/ptx/tensor_maps.h"
#include <deque>
#include <map>
#include <set>
#include <tuple>

namespace sentryward::frontend::ptx {
struct SharedBlocked {};
using SharedActor = std::uint64_t;
// Byte `offset` of a bounded integer (not a table atom) that one store wrote with `bytes`
// bytes: the store keeps the value's low-order bytes, little-endian.
struct BoundedFragment {
  BoundedInteger value;
  unsigned offset{}, bytes{};
  bool operator==(const BoundedFragment &other) const {
    return value == other.value && offset == other.offset && bytes == other.bytes;
  }
};
using SharedByte=std::optional<std::variant<std::uint8_t,PointerBits,DescriptorFragment,BoundedFragment>>;
struct SharedConstantRead {
  Scalar value;
  std::vector<SharedValueSite> sources;
};
class SharedConstants {
public:
  void Write(const SharedValueSite &, std::uint64_t, const std::vector<SharedByte> &);
  void Invalidate(std::uint64_t, std::uint64_t);
  void Clear() { bytes_.clear(); }
  // The value a load of `width` bytes reads (signed when `sign`): an integer, pointer or
  // descriptor fragment reassembled from its bytes, or a bounded integer when every byte is
  // its byte k at address + k from a store of exactly `width` bytes and its range fits the
  // loaded type (as a .local slot: LocalLoaded); otherwise UNKNOWN with no sources.
  [[nodiscard]] SharedConstantRead Read(std::uint64_t, unsigned, bool sign = false) const;
  [[nodiscard]] std::size_t size() const { return bytes_.size(); }
private:
  struct Byte { SharedByte value; SharedValueSite source; };
  std::map<std::uint64_t,Byte> bytes_;
};
struct SharedWrite {
  SharedValueSite source;
  std::uint64_t address;
  std::vector<SharedByte> bytes;
};
struct SharedPhase {
  std::int64_t expected{}, pending{}, tx{}, generation{};
  void Update(std::int64_t arrivals, std::int64_t bytes);
};
class SharedExecution {
public:
  std::map<std::uint64_t, SharedPhase> barriers;
  std::map<std::uint64_t,TensorMapFields> tensor_fields;
  std::map<std::uint64_t,nlohmann::json> tensor_templates;
  nlohmann::json tensor_maps=nlohmann::json::object();
  std::map<std::string, SharedPhase> collectives;
  std::map<std::tuple<std::string, SharedActor, DynamicKey>, std::int64_t> tickets;
  std::map<SharedActor, DynamicKey> cluster_tickets;
  std::map<SharedActor, std::vector<SharedWrite>> pending_copies;
  std::map<SharedActor, std::deque<std::vector<SharedWrite>>> copy_groups;
  std::uint64_t revision = 0;
  SharedConstants constants;
  void Write(const SharedWrite &);
  void Invalidate(std::uint64_t, std::uint64_t);
  void Clear();
  Scalar Read(const SharedValueSite &, std::uint64_t, unsigned, bool sign = false);
  void Export(ExecutionProgram &) const;
  void Init(std::uint64_t, std::int64_t);
  std::int64_t Update(std::uint64_t, std::int64_t, std::int64_t, SharedActor);
  void Wait(std::uint64_t, std::int64_t, SharedActor, bool parity);
  void Collective(std::string, std::int64_t, SharedActor, DynamicKey, bool blocking = true);
  void Defer(std::uint64_t, std::int64_t arrivals, std::int64_t tx,
             std::vector<SharedWrite> writes = {});
  void Progress();
private:
  struct Completion { std::uint64_t address; std::int64_t arrivals, tx; std::vector<SharedWrite> writes; };
  std::vector<Completion> completions_;
  std::map<Coordinates,std::set<SharedValueSite>> writes_;
  std::map<Coordinates,std::set<SharedValueDependency>> dependencies_;
  // Statistics only: the write sites that stored a bounded fragment, and the dependencies
  // whose read reassembled a bounded integer (a subset of dependencies_).
  std::map<Coordinates,std::set<SharedValueSite>> bounded_writes_;
  std::map<Coordinates,std::set<SharedValueDependency>> bounded_dependencies_;
};
std::vector<SharedByte> SharedBytes(const Scalar &, unsigned);
} // namespace sentryward::frontend::ptx
