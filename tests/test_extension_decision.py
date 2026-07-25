import unittest

from decide_extension_decision import build_extension_decision


def make_summary(status: str = "WEAK_TRADEOFF_STOP") -> dict[str, object]:
    return {
        "status": status,
        "n_rows": 360,
        "method_summary": {
            "dynamic_int8_cpu": {
                "mean_accuracy_delta": -0.07,
                "mean_macro_f1_delta": -0.11,
                "mean_best_deployment_improvement": 0.56,
            },
            "fp32_gpu": {
                "mean_accuracy_delta": 0.0,
                "mean_macro_f1_delta": 0.0,
                "mean_best_deployment_improvement": 0.9,
            },
            "bf16_gpu": {
                "mean_accuracy_delta": -0.002,
                "mean_macro_f1_delta": -0.002,
                "mean_best_deployment_improvement": 0.92,
            },
        },
    }


def make_stats(paired_cells: int, severe_cells: int) -> dict[str, object]:
    return {
        "dynamic_int8_cpu": {
            "paired_cells": paired_cells,
            "severe_quality_cell_count": severe_cells,
            "accuracy_delta": {"mean": -0.07},
            "macro_f1_delta": {"mean": -0.11},
            "best_deployment_improvement": {"mean": 0.56},
        }
    }


class ExtensionDecisionTests(unittest.TestCase):
    def test_negative_result_decision_when_severe_int8_risk_persists_in_new_seeds(self):
        decision = build_extension_decision(
            formal_summary=make_summary(),
            formal_stats=make_stats(paired_cells=225, severe_cells=102),
            strengthened_summary=make_summary(),
            strengthened_stats=make_stats(paired_cells=90, severe_cells=41),
            formal_rows=900,
            strengthened_rows=360,
        )

        self.assertEqual(decision["status"], "PASS_EXTENSION_DECISION_CLOSED_NEGATIVE_RESULT")
        self.assertTrue(decision["dynamic_int8_quality_risk_persists"])
        self.assertTrue(decision["can_proceed_as_negative_measurement_paper"])

    def test_blocks_when_strengthened_evidence_is_missing(self):
        decision = build_extension_decision(
            formal_summary=make_summary(),
            formal_stats=make_stats(paired_cells=225, severe_cells=102),
            strengthened_summary=make_summary(),
            strengthened_stats=make_stats(paired_cells=0, severe_cells=0),
            formal_rows=900,
            strengthened_rows=0,
        )

        self.assertEqual(decision["status"], "BLOCKED_NEEDS_MANAGER_SCIENCE_DECISION")
        self.assertFalse(decision["can_proceed_as_negative_measurement_paper"])


if __name__ == "__main__":
    unittest.main()
