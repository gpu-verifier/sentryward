#pragma once
#include "sentryward/algorithm/io/source_json.h"
#include "sentryward/frontend/ptx/execution.h"
#include <map>
#include <set>
#include <unordered_map>
namespace sentryward::frontend::ptx {
class AddressValueExtractor;
class RegisterTable;
using namespace sentryward::algorithm;
class SemanticError : public std::runtime_error {
public:
  std::string code;
  nlohmann::json details = nlohmann::json::object();
  SemanticError(std::string code, std::string message, nlohmann::json evidence = nlohmann::json::object())
      : std::runtime_error(std::move(message)), code(std::move(code)), details(std::move(evidence)) {}
};
using Fragment = std::pair<AddressId, std::optional<std::uint64_t>>;
using SharedAddress = std::pair<Coordinates, std::uint64_t>;
std::string CoordinateText(Coordinates);
std::string Opcode(const Instruction &);
std::vector<std::string> OperandTexts(const Module &, const Instruction &);
std::string Trim(std::string_view);
std::string CanonicalSpace(std::string);
std::uint64_t Unsigned(const Integer &, std::string_view,
                       std::uint64_t maximum = UINT64_MAX);
std::int64_t Signed(const Integer &, std::string_view);
bool Has(const Instruction &, std::string_view);
class AddressTable {
public:
  std::vector<ConstantAddressBinding> bindings;
  AddressId Intern(std::string space, std::string allocation,
                   std::uint64_t offset, std::string alias_scope = {});

private:
  struct Key {
    std::uint32_t name;
    std::uint64_t offset;
    bool operator==(const Key &) const = default;
  };
  struct Hash {
    std::size_t operator()(const Key &k) const {
      return k.offset ^ (std::size_t(k.name) * 0x9e3779b97f4a7c15ULL);
    }
  };
  std::map<std::pair<std::string, std::string>, std::uint32_t> names_;
  std::unordered_map<Key, AddressId, Hash> ids_;
};
class BarrierTable {
public:
  BarrierLifetimeId Intern(std::string kind, Coordinates owner,
                           std::string identity);
  std::tuple<std::string, Coordinates, std::string> Location(BarrierLifetimeId) const;

private:
  std::map<std::tuple<std::string, Coordinates, std::string>, BarrierLifetimeId>
      ids_;
};
struct TmaPair {
  std::optional<Fragment> source;
  std::optional<Fragment> destination;
  std::uint64_t logical_offset;
  std::uint32_t target_rank;
};
struct TmaAccesses {
  std::vector<TmaPair> pairs;
  std::vector<std::pair<BarrierLifetimeId, std::uint64_t>> completions;
};
class Resources {
public:
  nlohmann::json dynamic_tensor_maps=nlohmann::json::object();
  std::function<std::optional<nlohmann::json>(const LaneTrace&, const ExecutedInstruction&,
                                           std::string_view)> value_addresses;
  using KnownBits = std::pair<std::uint64_t, std::uint64_t>;
  using BitsResolver = std::function<std::optional<KnownBits>(
      const ExecutedInstruction &, std::string_view, unsigned)>;
  Resources(const ExecutionProgram &, Coordinates, AddressTable &,
            BarrierTable &);
  ~Resources();
  const ExecutionProgram &program;
  Coordinates coordinates;
  AddressTable &addresses;
  BarrierTable &barriers;
  void SetLane(const LaneTrace &);
  std::optional<std::uint32_t> SharedOwner(const ExecutedInstruction &e, std::string_view token) {
    return Owner(e, token);
  }
  Integer ScalarValue(const ExecutedInstruction &, std::string_view,
                      std::string_view field) const;
  KnownBits FixedBits(const ExecutedInstruction &, std::string_view, unsigned = 32,
                      const BitsResolver & = {}) const;
  std::optional<SharedAddress> Shared(const ExecutedInstruction &,
                                      std::string_view);
  BarrierLifetimeId Barrier(const ExecutedInstruction &, std::string_view,
                            const nlohmann::json &contract);
  Fragment Bind(std::string space, std::string resource, std::uint64_t offset,
                std::uint64_t bytes, std::optional<Coordinates> owner = {});
  Fragment Abstract(std::string space, std::string resource,
                    const nlohmann::json &offset, std::optional<std::uint64_t> bytes,
                    std::optional<Coordinates> owner = {});
  Fragment MemoryResource(std::string resource, std::uint64_t offset,
                          std::uint64_t bytes);
  Fragment Memory(const ExecutedInstruction &, std::string_view,
                  const nlohmann::json &, std::uint64_t bytes,
                  std::string space);
  TmaAccesses Tma(const ExecutedInstruction &, const nlohmann::json &,
                  bool store);
  Coordinates Target(std::uint32_t) const;
  std::uint32_t Rank(Coordinates) const;
  void AdmitGrid(const Instruction &) const;
  Integer AddressRegisterBits(const ExecutedInstruction &, std::string_view, const Integer &);

private:
  std::unique_ptr<AddressValueExtractor> address_values_;
  std::unique_ptr<RegisterTable> address_register_types_;
  mutable std::unordered_map<std::string_view, std::uint32_t> address_register_ids_;
  mutable std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> address_instruction_registers_;
  std::map<std::string, nlohmann::json> records_;
  struct MemoryResourceInfo {
    std::string space, object;
    std::uint64_t base = 0;
    std::optional<std::uint64_t> extent;
  };
  std::unordered_map<std::string, MemoryResourceInfo> memory_resources_;
  std::set<std::string> shared_symbols_;
  struct DynamicHash {
    std::size_t operator()(const DynamicKey &key) const noexcept {
      return std::hash<std::uint64_t>{}(key.occurrence) ^
             (std::size_t(key.instruction) * 0x9e3779b97f4a7c15ULL);
    }
  };
  // ExecutionProgram owns immutable lane traces for this CTA. Matrix accesses
  // repeatedly switch supplying lanes; build each index once and keep each
  // lane's provenance answers in its own cache. unordered_map rehash preserves
  // references to elements, so active_index_ remains valid as lanes are added.
  struct OccurrenceIndex {
    std::size_t indexed = 0;
    std::unordered_map<DynamicKey, std::size_t, DynamicHash>
        executions;
  };
  mutable std::unordered_map<const void*,OccurrenceIndex> occurrence_indices_;
  struct LaneIndex {
    std::map<DynamicKey, ExecutedInstruction> requested;
    std::map<DynamicKey, ExecutedInstruction> address_requested;
    std::unordered_map<DynamicKey, std::optional<std::uint32_t>, DynamicHash>
        owners;
    std::unordered_map<DynamicKey, std::optional<std::uint32_t>, DynamicHash> generic_owners;
  };
  std::unordered_map<const LaneTrace *, LaneIndex> lane_indices_;
  const LaneTrace *active_lane_ = nullptr;
  LaneIndex *active_index_ = nullptr;
  const ExecutedInstruction *Definition(DynamicKey, bool address_slice = false) const;
  std::optional<std::uint32_t> Owner(const ExecutedInstruction &,
                                     std::string_view,
                                     std::set<DynamicKey> visiting = {});
  std::optional<std::uint32_t> GenericOwner(const ExecutedInstruction &, std::string_view,
                                           std::set<DynamicKey> visiting = {});
  Scalar Value(const ExecutedInstruction &, std::string_view) const;
  const nlohmann::json &TensorMap(const ExecutedInstruction &, std::string_view,
                                  const nlohmann::json &) const;
};
} // namespace sentryward::frontend::ptx
