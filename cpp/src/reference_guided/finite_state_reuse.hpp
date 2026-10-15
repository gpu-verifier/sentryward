#pragma once
#include "checker.hpp"
#include "linear_reduction_pattern.hpp"
#include "finite_piece_union.hpp"
#include "whole_expression_templates.hpp"
#include <cstdint>
#include <memory>

namespace rg {
// A runner-local collection of COMPLETE output proofs. Read anchors select
// candidates only; every whole body must instantiate one common schema and
// every schema coordinate must be covered before any replacement is enabled.
class FiniteStateReuse {
  static constexpr size_t max_groups = 8, max_pieces = 65536;
  static constexpr size_t anchor_nodes = 32, anchor_depth = 2;
  static constexpr size_t rewrite_limit = 2000000;
  enum class RewriteMode { ReferenceBody, AbstractReference, AbstractOptimized };
  struct Layout {
    std::vector<Z> shape;
    J identity;
    bool valid = true;
  };
  struct AnchorKey {
    std::string tag;
    std::vector<Q> data;
    std::vector<Id> prefix;
    bool operator<(const AnchorKey &other) const {
      return std::tie(tag, data, prefix) < std::tie(other.tag, other.data, other.prefix);
    }
    bool operator==(const AnchorKey &other) const {
      return tag == other.tag && data == other.data && prefix == other.prefix;
    }
  };
  struct Anchor {
    AnchorKey key;
    Id row = invalid, column = invalid;
    const Layout *layout = nullptr;
  };
  struct Piece {
    Id left, right, axis, row, column;
    Z row_value, offset, low, high;
  };
  struct Group {
    AnchorKey key;
    Z rows, columns;
    std::vector<Piece> pieces;
    std::unique_ptr<WholeExpressionTemplates> expressions;
    std::unique_ptr<FinitePieceUnion> cover;
    std::string failure;
    bool overflow = false;
    Z recorded_cells = 0;
    std::unique_ptr<WholeExpressionTemplates> parametric;
    bool parametric_attempted = false;
    size_t seed_cursor = 0, first_seed = SIZE_MAX, second_seed = SIZE_MAX, column_seed = SIZE_MAX;
    J parametric_receipt;
    size_t attempted_pieces = 0;
  };
  Checker &checker;
  Arena &a;
  const uint64_t context;
  std::map<std::string, Layout> layouts;
  std::map<AnchorKey, size_t> group_index;
  std::vector<Group> groups;
  std::unordered_map<Id, Id> rewrite_memo;
  std::unordered_map<Id, Id> reference_abstraction_memo, optimized_abstraction_memo;
  size_t piece_count = 0, rewrite_work = 0, replacements = 0;
  size_t abstract_reference_replacements = 0, abstract_optimized_replacements = 0;
  bool refresh_requested = false, rewrite_exhausted = false;

  static bool small_integer(const Q &value) {
    static const Z bound = pow2(128);
    return value.denominator() == 1 && value.numerator() > -bound && value.numerator() < bound;
  }
  bool integer_var(Id id, Z &low, Z &high) const {
    const auto &n = a.nodes[id];
    if (n.op != "var" || !n.facts.finite || !n.bounds || n.data.size() != 2 ||
        !small_integer(n.data[0]) || !small_integer(n.data[1]) || n.data[0] > n.data[1] ||
        n.bounds->first != n.data[0] || n.bounds->second != n.data[1]) return false;
    low = n.data[0].numerator(); high = n.data[1].numerator();
    return true;
  }
  bool integer_expression(Id id, unsigned &budget) const {
    if (!budget) return false;
    --budget;
    const auto &n = a.nodes[id];
    if (!n.facts.finite) return false;
    if (n.op == "const") return n.data.size() == 1 && small_integer(n.data[0]);
    if (n.op == "var") { Z lo, hi; return integer_var(id, lo, hi); }
    if (n.op != "add" && n.op != "mul" && n.op != "idiv" && n.op != "imod") return false;
    if (n.args.size() > 8) return false;
    for (const Q &coefficient : n.data) if (!small_integer(coefficient)) return false;
    for (Id child : n.args) if (!integer_expression(child, budget)) return false;
    return true;
  }
  bool inside(Id value, const Z &extent) const {
    const auto &n = a.nodes[value];
    if (!n.bounds || n.bounds->first < 0 || n.bounds->second >= Q(extent)) return false;
    unsigned budget = 32;
    return integer_expression(value, budget);
  }
  bool charge(size_t amount = 1) {
    a.check();
    if (amount > rewrite_limit - std::min(rewrite_work, rewrite_limit)) {
      if (!rewrite_exhausted) ++a.stats["finite_state_reuse_work_declines"];
      rewrite_exhausted = true; return false;
    }
    rewrite_work += amount;
    return true;
  }
  void read_layouts(const J &mapping) {
    for (const char *side : {"left_inputs", "right_inputs"}) {
      auto input = mapping.find(side);
      if (input == mapping.end() || !input->contains("layouts")) continue;
      const auto &table = input->at("layouts");
      for (auto it = table.begin(); it != table.end(); ++it) {
        const auto &description = it.value();
        if (!description.contains("shape") || !description.at("shape").is_array()) continue;
        const auto &shape = description.at("shape");
        if (shape.size() < 2 || shape.size() > 4) continue;
        Layout layout;
        for (const J &extent : shape) {
          Q count = jq(extent);
          if (!small_integer(count) || count <= 0) { layout.valid = false; break; }
          layout.shape.push_back(count.numerator());
        }
        if (!layout.valid || !description.contains("source_strides_bytes") ||
            !description.contains("element_bytes")) continue;
        layout.identity = J::array({shape,
            description.value("target_strides_bytes", description.at("source_strides_bytes")),
            description.value("target_offset", J(0)), description.at("element_bytes")});
        std::string allocation = it.key();
        if (input->contains("allocations")) {
          const auto &aliases = input->at("allocations");
          auto alias = aliases.find(allocation);
          if (alias != aliases.end()) allocation = alias->get<std::string>();
        }
        auto [found, fresh] = layouts.emplace(allocation, layout);
        if (!fresh && found->second.identity != layout.identity) found->second.valid = false;
      }
    }
  }
  std::optional<Anchor> read_anchor(Id id) const {
    const auto &n = a.nodes[id];
    if (n.op != "read" || !n.facts.finite || n.args.size() < 2 || n.args.size() > 4) return {};
    auto separator = n.tag.find('\n');
    if (separator == std::string::npos || !n.tag.substr(separator + 1).starts_with("real:")) return {};
    auto layout = layouts.find(n.tag.substr(0, separator));
    if (layout == layouts.end() || !layout->second.valid || layout->second.shape.size() != n.args.size()) return {};
    Anchor result;
    result.key.tag = n.tag; result.key.data = n.data; result.layout = &layout->second;
    for (size_t i = 0; i + 2 < n.args.size(); ++i) {
      auto value = a.value(n.args[i]);
      if (!value || !small_integer(*value) || *value < 0 || *value >= Q(layout->second.shape[i])) return {};
      result.key.prefix.push_back(n.args[i]);
    }
    result.row = n.args[n.args.size() - 2]; result.column = n.args.back();
    return result;
  }
  // This bounded scan does not enter reductions or nonlinear producers. It
  // never proves an expression equality; the whole-template module does that.
  std::vector<Anchor> anchors(Id root, bool rewriting) {
    std::vector<Anchor> result;
    std::vector<std::pair<Id, size_t>> todo{{root, 0}};
    std::unordered_set<Id> seen;
    size_t work = 0;
    while (!todo.empty()) {
      auto [id, depth] = todo.back(); todo.pop_back();
      if (!seen.insert(id).second) continue;
      if (++work > anchor_nodes || (rewriting && !charge())) return {};
      const auto &n = a.nodes[id];
      if (auto anchor = read_anchor(id)) result.push_back(std::move(*anchor));
      if (depth == anchor_depth || (n.op != "add" && n.op != "mul")) continue;
      if (n.args.size() > anchor_nodes - work) return {};
      work += n.args.size();
      if (rewriting && !charge(n.args.size())) return {};
      for (Id child : n.args) todo.emplace_back(child, depth + 1);
    }
    return result;
  }
  std::optional<Id> family_axis(const J &left, const J &right) {
    if (!left.contains("shape") || !right.contains("shape") || left.at("shape") != right.at("shape")) return {};
    const auto &shape = left.at("shape");
    if (!shape.is_array() || shape.size() > 4) return {};
    Id axis = invalid;
    for (size_t i = 0; i < shape.size(); ++i) {
      Q count = jq(shape[i]);
      if (!small_integer(count) || count < 1) return {};
      if (count == 1) continue;
      if (axis != invalid) return {};
      axis = a.var("output:" + std::to_string(i), 0, count - 1);
    }
    // A fully fixed output is a one-point interval, not missing coverage.
    if (axis == invalid && !shape.empty()) axis = a.var("output:0", 0, 0);
    return axis == invalid ? std::optional<Id>{} : axis;
  }
  std::optional<Z> column_offset(Id column, Id axis) const {
    Z low, high;
    if (integer_var(axis, low, high) && low == 0 && high == 0) {
      if (auto value = a.value(column); value && small_integer(*value)) return value->numerator();
    }
    if (column == axis) return Z(0);
    const auto &n = a.nodes[column];
    if (n.op != "add" || n.args.size() != 2 || n.data.size() != 2) return {};
    std::optional<Z> offset;
    bool found_axis = false;
    for (size_t i = 0; i < 2; ++i) {
      if (n.args[i] == axis && n.data[i] == 1) found_axis = true;
      else if (n.args[i] == a.one && small_integer(n.data[i])) offset = n.data[i].numerator();
      else return {};
    }
    return found_axis ? offset : std::optional<Z>{};
  }
  bool preflight(Group &group) {
    if (group.overflow || group.pieces.empty()) { group.failure = "piece_budget"; return false; }
    std::vector<std::pair<Z, Z>> intervals;
    intervals.reserve(group.pieces.size());
    for (const auto &piece : group.pieces) {
      Z start = piece.row_value * group.columns + piece.offset + piece.low;
      intervals.emplace_back(start, piece.high - piece.low + 1);
    }
    std::sort(intervals.begin(), intervals.end());
    Z next = 0;
    for (const auto &[start, length] : intervals) {
      if (start != next) { group.failure = start < next ? "preflight_overlap" : "preflight_hole"; return false; }
      next += length;
    }
    if (next != group.rows * group.columns) { group.failure = "preflight_incomplete"; return false; }
    return true;
  }
  void refresh_complete_schemas() {
    if (!refresh_requested) return;
    refresh_requested = false;
    bool added_schema = false;
    for (size_t index = 0; index < groups.size(); ++index) {
      auto &group = groups[index];
      if (group.cover && group.cover->sealed()) continue;
      // Cardinality is a scheduling filter only. Exact disjoint coverage and
      // complete body proofs below remain mandatory. Do not repeatedly sort
      // a growing incomplete collection for each unresolved output.
      if (group.recorded_cells < group.rows * group.columns) {
        group.failure = "preflight_incomplete"; continue;
      }
      if (group.recorded_cells > group.rows * group.columns) {
        group.failure = "preflight_overlap"; continue;
      }
      if (group.attempted_pieces == group.pieces.size()) continue;
      group.attempted_pieces = group.pieces.size();
      if (!preflight(group)) continue; // Scheduling filter, never a proof receipt.
      const Piece *first = nullptr, *second = nullptr;
      for (const auto &piece : group.pieces) {
        if (piece.row_value <= 0 || piece.offset != 0 || piece.low == piece.high) continue;
        if (!first) first = &piece;
        else if (piece.row_value != first->row_value && piece.axis == first->axis) { second = &piece; break; }
      }
      if (!second) { group.failure = "independent_seed_rows"; continue; }
      const std::string key = "finite-state:" + std::to_string(context) + ":" + std::to_string(index);
      std::vector<Id> axes{a.var(key + ":row", 0, Q(group.rows - 1)),
                           a.var(key + ":column", 0, Q(group.columns - 1))};
      group.expressions = std::make_unique<WholeExpressionTemplates>(checker, axes, context);
      std::vector<WholeTemplateSample> seeds{
          {first->left, first->right, {first->row, first->column}},
          {second->left, second->right, {second->row, second->column}}};
      if (!group.expressions->build(seeds)) {
        group.failure = "template_build:" + group.expressions->last_failure;
        group.expressions.reset(); continue;
      }
      FinitePieceUnionSchema schema{group.expressions->left_template(), group.expressions->right_template(), axes, context};
      group.cover = std::make_unique<FinitePieceUnion>(a, std::move(schema),
          FinitePieceUnionLimits{max_pieces, 8388608});
      bool complete = true;
      for (const auto &piece : group.pieces) {
        a.check();
        auto witness = group.expressions->validate_piece(piece.left, piece.right, {piece.row, piece.column});
        if (!witness || !group.cover->add_proved_piece(
              {piece.left, piece.right, context}, {piece.axis, {piece.row_value}, piece.offset}, *witness)) {
          group.failure = witness ? group.cover->last_failure : "whole_body_instance:" + group.expressions->last_failure;
          complete = false; break;
        }
      }
      if (!complete || !group.cover->seal()) {
        if (complete) group.failure = group.cover->last_failure;
        group.expressions.reset(); group.cover.reset(); continue;
      }
      ++a.stats["finite_state_reuse_sealed_schemas"];
      added_schema = true;
    }
    if (added_schema) {
      // Previously unsuccessful rewrites can now use the new theorem. Existing
      // sealed definitions are immutable; only negative lookup caches expire.
      rewrite_memo.clear(); reference_abstraction_memo.clear(); optimized_abstraction_memo.clear();
    }
  }
  std::optional<Id> replacement(Id target, RewriteMode mode) {
    if (a.nodes[target].op != "add") return {};
    auto candidates = anchors(target, true);
    const Anchor *chosen = nullptr;
    Group *group = nullptr;
    size_t group_id = 0;
    for (const auto &candidate : candidates) {
      auto found = group_index.find(candidate.key);
      if (found == group_index.end()) continue;
      auto &entry = groups[found->second];
      if (!entry.cover || !entry.cover->sealed()) continue;
      if (chosen) return {}; // Ambiguous shallow anchors never select a rule.
      chosen = &candidate; group = &entry; group_id = found->second;
    }
    if (!chosen || !inside(chosen->row, group->rows) || !inside(chosen->column, group->columns)) return {};
    ++a.stats["finite_state_reuse_template_attempts"];
    const std::vector<Id> coordinates{chosen->row, chosen->column};
    std::optional<Id> result;
    if (mode == RewriteMode::ReferenceBody)
      result = group->expressions->rewrite_optimized(target, coordinates);
    else {
      bool matched = mode == RewriteMode::AbstractReference
          ? group->expressions->matches_reference(target, coordinates)
          : group->expressions->matches_optimized(target, coordinates);
      if (matched) {
        // A fresh defined-function symbol abbreviates the certified reference
        // body on this sealed domain. Both sides use the SAME symbol only after
        // full instance verification. Its tag freezes the bank, schema, input
        // identities and fixed prefix; only the integer coordinates vary.
        std::string tag = "finite-state:" + std::to_string(context) + ":" + std::to_string(group_id);
        result = a.intern("certified_state", coordinates, {}, tag, {true, false, false});
      }
    }
    if (!result) ++a.stats["finite_state_reuse_target_decline_" + group->expressions->last_failure];
    return result;
  }
public:
  FiniteStateReuse(Checker &owner, const J &mapping)
      : checker(owner), a(owner.a), context(++a.stats["finite_state_reuse_banks"]) {
    read_layouts(mapping);
  }
  FiniteStateReuse(const FiniteStateReuse &) = delete;
  FiniteStateReuse &operator=(const FiniteStateReuse &) = delete;
  // Call before ordinary normalization: discovery may change the active
  // reference vocabulary, but never creates a second owner for call IDs.
  bool prove_instance(Id left, Id right) {
    if (!checker.exponential_mode || a.nodes[left].op != "add" ||
        a.nodes[right].op != "add") return false;
    auto candidates = anchors(left, false);
    if (candidates.size() != 1) return false;
    const auto &candidate = candidates.front();
    auto found = group_index.find(candidate.key);
    if (found == group_index.end()) return false;
    auto &group = groups[found->second];
    if (!group.parametric_attempted) {
      // Inspect each recorded piece once. Singleton columns cannot identify
      // a varying column coordinate; wait without consuming the proof attempt.
      while (group.seed_cursor < group.pieces.size()) {
        size_t index = group.seed_cursor++;
        const auto &piece = group.pieces[index];
        ++a.stats["parametric_seed_piece_visits"];
        if (piece.row_value <= 0) continue;
        if (group.first_seed == SIZE_MAX) {
          if(piece.offset!=0)continue;
          group.first_seed = index;
        }
        else {
          const auto &first=group.pieces[group.first_seed];
          if(piece.axis!=first.axis || piece.low!=first.low || piece.high!=first.high)continue;
          if(group.second_seed==SIZE_MAX && piece.row_value!=first.row_value && piece.offset==0)group.second_seed=index;
          if(group.column_seed==SIZE_MAX && piece.row_value==first.row_value && piece.offset!=first.offset)group.column_seed=index;
        }
        const auto &first=group.pieces[group.first_seed];
        if(group.second_seed!=SIZE_MAX && (group.columns==1 || first.low!=first.high || group.column_seed!=SIZE_MAX))break;
      }
      if (group.second_seed == SIZE_MAX) return false;
      const Piece *first = &group.pieces[group.first_seed], *second = &group.pieces[group.second_seed];
      if(group.columns>1 && first->low==first->high && group.column_seed==SIZE_MAX)return false;
      group.parametric_attempted = true;
      ++a.stats["parametric_state_attempts"];
      std::string key = "parametric-state:" + std::to_string(context) + ":" + std::to_string(found->second);
      std::vector<Id> axes{a.var(key + ":row", 0, Q(group.rows - 1)),
                           a.var(key + ":column", 0, Q(group.columns - 1))};
      auto proposed = std::make_unique<WholeExpressionTemplates>(checker, axes, context);
      std::vector<WholeTemplateSample> seeds{{first->left, first->right, {first->row, first->column}},
                                            {second->left, second->right, {second->row, second->column}}};
      if(first->low==first->high && group.column_seed!=SIZE_MAX){const auto &third=group.pieces[group.column_seed];seeds.push_back({third.left,third.right,{third.row,third.column}});}
      if (!proposed->build(seeds)) {
        group.parametric_receipt = {{"decline", proposed->last_failure}};
        return false;
      }
      // Samples propose syntax only. Prove the resulting formal-domain
      // identity independently, including all reduction coverage obligations.
      try {
        Id l = checker.normalize(proposed->left_template(), true);
        Id r = checker.normalize(proposed->right_template());
        bool proved = checker.scalar.equal(l, r);
        J reduction;
        if (!proved) {
          LinearReductionPattern linear(checker);
          if (auto proof = linear.prove(l, r)) { proved = true; reduction = proof->receipt; }
          else group.parametric_receipt = {{"decline", linear.last_failure}};
        }
        if (!proved) return false;
        group.parametric_receipt = {{"method", "formal_domain_reduction_and_mandatory_ep"},
          {"left", proposed->left_template()}, {"right", proposed->right_template()},
          {"axes", axes}, {"reduction", reduction}};
        group.parametric = std::move(proposed);
        ++a.stats["parametric_state_theorems"];
      } catch (const Unsupported &e) {
        group.parametric_receipt = {{"decline", e.what()}}; return false;
      }
    }
    if (!group.parametric) return false;
    if (!group.parametric->validate_piece(left, right, {candidate.row, candidate.column})) {
      ++a.stats["parametric_state_instance_declines"]; return false;
    }
    ++a.stats["parametric_state_instance_proofs"];
    return true;
  }
  void seal_ready() { refresh_complete_schemas(); }
  bool has_sealed_schema() const {
    return std::any_of(groups.begin(), groups.end(), [](const Group &group) {
      return group.cover && group.cover->sealed();
    });
  }

  // Caller must first certify equality of the COMPLETE lr and rr bodies.
  // Family bounds scope that proof; a shallow state read is not a proof.
  void record(Id left, Id right, const J &left_family, const J &right_family) {
    if (a.nodes[left].op != "add" || a.nodes[right].op != "add" ||
        !a.nodes[left].facts.finite || !a.nodes[right].facts.finite) return;
    auto axis = family_axis(left_family, right_family);
    if (!axis) return;
    Z low, high;
    if (!integer_var(*axis, low, high)) return;
    for (Id body : {left, right}) {
      const auto &free = a.nodes[body].free;
      if (free.size() > 1 || (!free.empty() && free.front() != *axis)) {
        ++a.stats["finite_state_reuse_open_body_declines"]; return;
      }
    }
    auto left_anchors = anchors(left, false), right_anchors = anchors(right, false);
    std::optional<Anchor> chosen;
    std::optional<Z> offset;
    Z row;
    for (const auto &la : left_anchors) {
      auto fixed = a.value(la.row);
      auto window = column_offset(la.column, *axis);
      const auto &shape = la.layout->shape;
      if (!fixed || !small_integer(*fixed) || *fixed < 0 || *fixed >= Q(shape[shape.size() - 2]) ||
          !window || *window + low < 0 || *window + high >= shape.back()) continue;
      for (const auto &ra : right_anchors) {
        if (!(la.key == ra.key) || la.row != ra.row || la.column != ra.column) continue;
        if (chosen) { ++a.stats["finite_state_reuse_ambiguous_anchor_declines"]; return; }
        chosen = la; offset = *window; row = fixed->numerator();
      }
    }
    if (!chosen) return;
    auto found = group_index.find(chosen->key);
    if (found == group_index.end()) {
      if (groups.size() == max_groups) { ++a.stats["finite_state_reuse_group_declines"]; return; }
      const auto &shape = chosen->layout->shape;
      size_t index = groups.size();
      Group group; group.key = chosen->key; group.rows = shape[shape.size() - 2]; group.columns = shape.back();
      groups.push_back(std::move(group));
      found = group_index.emplace(chosen->key, index).first;
    }
    auto &group = groups[found->second];
    if (group.cover && group.cover->sealed()) return;
    if (piece_count == max_pieces) { group.overflow = true; ++a.stats["finite_state_reuse_piece_declines"]; return; }
    group.pieces.push_back({left, right, *axis, chosen->row, chosen->column, row, *offset, low, high});
    group.recorded_cells += high - low + 1;
    refresh_requested = true;
    ++piece_count; ++a.stats["finite_state_reuse_collected_pieces"];
  }

private:
  std::optional<Id> transform(Id root, RewriteMode mode, std::unordered_map<Id, Id> &memo) {
    refresh_complete_schemas();
    if (std::none_of(groups.begin(), groups.end(), [](const Group &g) { return bool(g.cover); })) return root;
    if (auto found = memo.find(root); found != memo.end()) return found->second;
    if (rewrite_exhausted) return {};
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    while (!todo.empty()) {
      auto [id, ready] = todo.back(); todo.pop_back();
      if (memo.contains(id)) continue;
      if (!charge()) return {};
      const auto &node = a.nodes[id];
      if (!ready) {
        if (auto value = replacement(id, mode)) {
          memo.emplace(id, *value);
          if (*value != id) {
            if (mode == RewriteMode::ReferenceBody) {
              ++replacements; ++a.stats["finite_state_reuse_replacements"];
            } else if (mode == RewriteMode::AbstractReference) {
              ++abstract_reference_replacements;
              ++a.stats["finite_state_reuse_abstract_reference_replacements"];
            } else {
              ++abstract_optimized_replacements;
              ++a.stats["finite_state_reuse_abstract_optimized_replacements"];
            }
          }
          continue; // Do not visit the matched expanded residual or replacement.
        }
        if (rewrite_exhausted || !charge(node.args.size())) return {};
        todo.emplace_back(id, true);
        for (Id child : node.args) if (!memo.contains(child)) todo.emplace_back(child, false);
        continue;
      }
      std::vector<Id> args; args.reserve(node.args.size());
      bool changed = false;
      for (Id child : node.args) { Id value = memo.at(child); args.push_back(value); changed |= value != child; }
      Id result = id;
      try { if (changed) result = checker.reapply(id, args); }
      catch (const Unsupported &) { ++a.stats["finite_state_reuse_reapply_declines"]; return {}; }
      memo.emplace(id, result);
      ++a.stats["finite_state_reuse_stored_node_visits"];
    }
    return memo.at(root);
  }
public:
  std::optional<Id> rewrite(Id root) {
    return transform(root, RewriteMode::ReferenceBody, rewrite_memo);
  }
  // The caller must compare BOTH abstracted expressions. Substitution into
  // just one side does not establish equality to an uninterpreted symbol.
  // Register reduction vocabulary from the abstract reference first, then
  // normalize the abstract optimized expression in that same active context.
  std::optional<Id> abstract_reference(Id root) {
    return transform(root, RewriteMode::AbstractReference, reference_abstraction_memo);
  }
  std::optional<Id> abstract_optimized(Id root) {
    return transform(root, RewriteMode::AbstractOptimized, optimized_abstraction_memo);
  }
  J report() const {
    J schemas = J::array();
    for (const auto &group : groups) {
      J item{{"anchor", group.key.tag}, {"pieces", group.pieces.size()},
             {"rows", group.rows.str()}, {"columns", group.columns.str()}};
      item["parametric"] = group.parametric_receipt;
      if (group.cover) item["certificate"] = group.cover->receipt();
      else item["decline"] = group.failure;
      schemas.push_back(std::move(item));
    }
    return {{"method", "complete_output_equalities_with_finite_piece_union"},
            {"schemas", std::move(schemas)}, {"collected_pieces", piece_count},
            {"replacements", replacements}, {"stored_dag_work", rewrite_work}, {"work_limit", rewrite_limit},
            {"abstract_reference_replacements", abstract_reference_replacements},
            {"abstract_optimized_replacements", abstract_optimized_replacements},
            {"abstraction", "schema-defined functions from sealed equalities; no additional input assumptions"},
            {"late_certificates", "accepted_until_each_schema_is_complete"}};
  }
};
} // namespace rg
