#include "sentryward/algorithm/memory/address_solver.h"

#include <chrono>
#include <exception>

#include <z3++.h>

namespace sentryward::algorithm::address_solver {
namespace {

class Z3Solver final : public Solver {
 public:
  Answer solve(const Query& query, unsigned timeout_ms) override {
    Answer answer;
    try {
      const auto start = std::chrono::steady_clock::now();
      auto remaining = [&]() -> unsigned {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        return elapsed >= timeout_ms ? 0U : timeout_ms - static_cast<unsigned>(elapsed);
      };
      z3::solver solver(context_);
      const auto variables = declare(query);
      environment(solver, query, variables);
      answer.environment = check(solver, remaining(), answer.detail);
      if (answer.environment != Verdict::kSat) return answer;
      const auto goal = encode(query.goal, variables);
      solver.add(goal > number(query.goal_lower) && goal < number(query.goal_upper));
      answer.goal = check(solver, remaining(), answer.detail);
      if (answer.goal == Verdict::kSat) answer.detail = witness(solver.get_model(), query, variables);
    } catch (const std::exception& error) {
      answer.detail = std::string("SMT failure: ") + error.what();
    }
    return answer;
  }

  Verdict consistent(const Query& query, unsigned timeout_ms) override {
    try {
      z3::solver solver(context_);
      environment(solver, query, declare(query));
      std::string ignored;
      return check(solver, timeout_ms, ignored);
    } catch (const std::exception&) {
      return Verdict::kUnknown;
    }
  }

 private:
  z3::context context_;

  z3::expr number(Int value) { return context_.int_val(ToString(value).c_str()); }

  std::vector<z3::expr> declare(const Query& query) {
    std::vector<z3::expr> result;
    result.reserve(query.variables.size());
    for (std::size_t i = 0; i < query.variables.size(); ++i)
      result.push_back(context_.int_const(("v" + std::to_string(i)).c_str()));
    return result;
  }

  z3::expr encode(const Linear& form, const std::vector<z3::expr>& variables) {
    z3::expr result = number(form.constant);
    for (const auto& [variable, coefficient] : form.terms)
      result = result + number(coefficient) * variables.at(variable);
    return result;
  }

  void environment(z3::solver& solver, const Query& query,
                   const std::vector<z3::expr>& variables) {
    for (const auto& bound : query.bounds) {
      const auto term = encode(bound.term, variables);
      if (bound.lower) solver.add(term >= number(*bound.lower));
      if (bound.upper) solver.add(term <= number(*bound.upper));
    }
    for (const auto& [term, modulus] : query.multiples)
      solver.add(z3::mod(encode(term, variables), number(modulus)) == 0);
    for (const auto& clause : query.distinct) {
      z3::expr disjunction = context_.bool_val(false);
      for (const auto& [a, b] : clause)
        disjunction = disjunction || variables.at(a) != variables.at(b);
      solver.add(disjunction);
    }
  }

  static Verdict check(z3::solver& solver, unsigned timeout_ms, std::string& detail) {
    if (timeout_ms == 0) {
      detail = "SMT budget exhausted";
      return Verdict::kUnknown;
    }
    z3::params params(solver.ctx());
    params.set("timeout", timeout_ms);
    solver.set(params);
    switch (solver.check()) {
      case z3::unsat: return Verdict::kUnsat;
      case z3::sat: return Verdict::kSat;
      case z3::unknown: break;
    }
    detail = "SMT unknown: " + solver.reason_unknown();
    return Verdict::kUnknown;
  }

  std::string witness(const z3::model& model, const Query& query,
                      const std::vector<z3::expr>& variables) {
    std::string result;
    for (std::size_t i = 0; i < variables.size(); ++i) {
      if (!result.empty()) result += ", ";
      result += query.variables[i] + "=" + model.eval(variables[i], true).to_string();
    }
    return "candidate valuation {" + result + "}";
  }
};

}  // namespace

std::unique_ptr<Solver> MakeSolver() { return std::make_unique<Z3Solver>(); }
const char* BackendName() noexcept { return "z3"; }

}  // namespace sentryward::algorithm::address_solver
