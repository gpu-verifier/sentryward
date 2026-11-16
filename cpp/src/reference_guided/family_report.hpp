#pragma once
#include "core.hpp"

namespace rg {
// Groups complete diagnostic records without retaining individual family rows.
class FamilyReport {
  std::vector<J> groups_;
  std::unordered_map<std::string, size_t> index_;
  size_t failed_ = 0, sampled_ = 0;
  static constexpr size_t expression_sample_limit = 8;
public:
  void add(J row) {
    const auto family = row.at("family").get<size_t>();
    row.erase("family");
    J timing = J::object();
    for (auto it = row.begin(); it != row.end();) {
      if (it.key().ends_with("_s")) { timing[it.key()] = *it; it = row.erase(it); }
      else ++it;
    }
    auto [found, inserted] = index_.try_emplace(row.dump(), groups_.size());
    if (inserted)
      groups_.push_back({{"count", 0}, {"families", J::array()}, {"record", std::move(row)}, {"timing_s", J::object()}});
    auto &group = groups_[found->second];
    group["count"] = group["count"].get<size_t>() + 1;
    auto &ranges = group["families"];
    if (!ranges.empty() && ranges.back()[1].get<size_t>() + 1 == family) ranges.back()[1] = family;
    else ranges.push_back({family, family});
    auto &sums = group["timing_s"];
    for (const auto &[key, value] : timing.items()) {
      const double seconds = value.get<double>();
      // cumulative_elapsed_s is a timestamp: keep the group's latest one.
      if (key == "cumulative_elapsed_s") sums[key] = std::max(sums.value(key, 0.0), seconds);
      else sums[key] = sums.value(key, 0.0) + seconds;
    }
  }
  void addFailureExpressions(J &row, const Arena &arena, Id left, Id right) {
    ++failed_;
    if (sampled_ < expression_sample_limit) {
      row["left"] = arena.render_shared(left, 5, 4096);
      row["right"] = arena.render_shared(right, 5, 4096);
      ++sampled_;
    } else {
      row["expressions_omitted"] = true;
    }
  }
  const std::vector<J> &groups() const { return groups_; }
  std::vector<J> takeGroups() { return std::move(groups_); }
  J expressionSamples() const {
    return {{"failed_families", failed_}, {"sampled_families", sampled_},
            {"omitted_families", failed_ - sampled_},
            {"sample_limit", expression_sample_limit}, {"render_byte_limit_per_side", 4096}};
  }
};
} // namespace rg
