#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace sentryward::equivalence {
struct InputIdentityMapping;
struct ComparisonContextFiles { std::string left, right, mapping; };

// Optional evidence for the ORIGINAL supplied fixed read-only input scope.
// No branches, extra tensor values, ranges, or numerical approximations are
// accepted. The on-disk graph/mapping files remain unchanged.
class ComparisonContext {
 public:
  // right_mapping must be decoded from files.mapping and be the same mapping
  // used for comparison. The CLI enforces this with pre/post source hashes;
  // this library entry point does not independently parse the mapping schema.
  static std::shared_ptr<const ComparisonContext> load(
      const std::string& path, const ComparisonContextFiles&,
      const InputIdentityMapping& right_mapping);
  std::optional<std::string> literal(std::string_view allocation,
      std::uint64_t byte_offset, const std::string& type) const;
  const std::string& digest() const { return digest_; }
  const std::map<std::string,std::string>& sourceHashes() const { return source_hashes_; }
  std::size_t cells() const { return cell_count_; }
 private:
  struct Cell { std::string type, literal; std::uint64_t bytes; };
  ComparisonContext() = default;
  std::map<std::string,std::map<std::uint64_t,Cell>,std::less<>> cells_;
  std::map<std::string,std::string> source_hashes_;
  std::string digest_;
  std::size_t cell_count_=0;
};

// Streaming, dependency-free SHA-256; fixed-size storage, no whole-artifact copy.
std::string comparisonSha256(std::istream&);
std::string comparisonFileSha256(const std::string& path);
} // namespace sentryward::equivalence
