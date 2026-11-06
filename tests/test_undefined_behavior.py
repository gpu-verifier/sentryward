"""The undefined_behavior result category (docs/output-json.md, "Undefined behavior").

A value-stage undefined-behavior finding after a passed synchronization check is the overall result, its
equivalence check is not_run, and its diagnostic keeps the witness details. These tests need no native executable.
"""
import unittest

from sentryward.results import RESULTS, normalize, publish

WITNESS = {"condition": "tcgen05.mma.sp sparsity metadata", "source": "input_object", "object": "E.metadata",
           "property": "field_values", "declared_values": [0, 4, 8, 9, 12, 13, 14], "witness_value": 0,
           "witness_bits": "0b0000", "line": 6404}


def native_report():
    return {"status": "race_free", "verification_status": "race_free", "mode": "verify",
            "memory_analysis_complete": True,
            "equivalence": {"status": "undefined_behavior", "reason": "sparse_metadata_invalid",
                            "stage": "value_analysis.scalar_fold",
                            "message": "Sparse MMA metadata can hold an index pair the PTX ISA leaves undefined",
                            "details": WITNESS}}


class UndefinedBehaviorTest(unittest.TestCase):
    def test_category_is_known(self):
        self.assertIn("undefined_behavior", RESULTS)

    def test_value_stage_finding_is_the_result(self):
        out = normalize(native_report(), operation="kernel_verification")
        self.assertEqual(out["result"], "undefined_behavior")
        self.assertEqual(out["checks"]["synchronization"], "passed")
        self.assertEqual(out["checks"]["equivalence"], "not_run")
        codes = [d["code"] for d in out["diagnostics"]]
        self.assertIn("sparse_metadata_invalid", codes)
        detail = next(d for d in out["diagnostics"] if d["code"] == "sparse_metadata_invalid")["details"]
        self.assertEqual(detail["witness_bits"], "0b0000")
        self.assertEqual(detail["object"], "E.metadata")

    def test_published_report_keeps_the_witness(self):
        out = publish(normalize(native_report(), operation="kernel_verification"))
        self.assertEqual(out["result"], "undefined_behavior")
        self.assertEqual(out["checks"], {"synchronization": "passed", "equivalence": "not_run"})
        detail = next(d for d in out["diagnostics"] if d["code"] == "sparse_metadata_invalid")["details"]
        self.assertEqual(detail["declared_values"], [0, 4, 8, 9, 12, 13, 14])

    def test_frontend_finding_is_the_result(self):
        report = {"status": "undefined_behavior", "stage": "event_view_construction",
                  "reason": "tmem_address_not_uniform", "verification_performed": False,
                  "memory_analysis_complete": False,
                  "diagnostics": [{"code": "tmem_address_not_uniform",
                                   "message": "tcgen05.ld lanes use different taddr values (undefined behavior)",
                                   "details": {"lane": 1, "taddr": 1, "lane0_taddr": 0, "line": 3247}}]}
        out = publish(normalize(report, operation="kernel_verification"))
        self.assertEqual(out["result"], "undefined_behavior")
        self.assertNotEqual(out["checks"]["synchronization"], "passed")
        detail = next(d for d in out["diagnostics"] if d["code"] == "tmem_address_not_uniform")["details"]
        self.assertEqual(detail["lane0_taddr"], 0)

    def test_synchronization_failure_keeps_precedence(self):
        report = native_report()
        report.update(status="not_wellsynced", verification_status="not_wellsynced")
        out = normalize(report, operation="kernel_verification")
        self.assertEqual(out["result"], "not_wellsynced")


if __name__ == "__main__":
    unittest.main()
