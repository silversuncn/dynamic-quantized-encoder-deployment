import unittest

from dynamic_quant_core import (
    compute_reference_deltas,
    gate_deployment_tradeoff,
    gate_dynamic_quantization,
    inference_method_configs,
    stratified_train_val_indices,
    validate_result_rows,
)


class DynamicQuantCoreTests(unittest.TestCase):
    def test_stratified_train_val_indices_are_deterministic_disjoint_and_counted(self):
        labels = [0, 1] * 80

        split = stratified_train_val_indices(labels, train_per_class=8, val_per_class=5, seed=23)

        self.assertEqual(split["train_class_counts"], {"0": 8, "1": 8})
        self.assertEqual(split["validation_class_counts"], {"0": 5, "1": 5})
        self.assertEqual(len(set(split["train_indices"]) & set(split["validation_indices"])), 0)
        self.assertEqual(split, stratified_train_val_indices(labels, 8, 5, 23))

    def test_inference_method_configs_include_cpu_reference_and_optional_gpu_na(self):
        configs = inference_method_configs(gpu_available=False, bf16_supported=False)

        self.assertEqual(configs["fp32_cpu"]["reference_method"], "fp32_cpu")
        self.assertEqual(configs["dynamic_int8_cpu"]["quantization"], "dynamic_int8_linear")
        self.assertEqual(configs["fp32_gpu"]["status"], "not_applicable")
        self.assertEqual(configs["bf16_gpu"]["status"], "not_applicable")

    def test_compute_reference_deltas_pairs_by_dataset_backbone_budget_seed(self):
        rows = [
            {
                "dataset": "sst2",
                "backbone": "distilbert-base-uncased",
                "label_budget_per_class": 64,
                "seed": 1,
                "method": "fp32_cpu",
                "status": "PASS",
                "accuracy": 0.84,
                "macro_f1": 0.83,
                "model_size_mb": 250.0,
                "latency_ms_per_example": 3.0,
                "throughput_examples_per_second": 333.3,
                "peak_memory_mb": 900.0,
            },
            {
                "dataset": "sst2",
                "backbone": "distilbert-base-uncased",
                "label_budget_per_class": 64,
                "seed": 1,
                "method": "dynamic_int8_cpu",
                "status": "PASS",
                "accuracy": 0.835,
                "macro_f1": 0.822,
                "model_size_mb": 120.0,
                "latency_ms_per_example": 2.0,
                "throughput_examples_per_second": 500.0,
                "peak_memory_mb": 760.0,
            },
        ]

        enriched = compute_reference_deltas(rows)
        candidate = [row for row in enriched if row["method"] == "dynamic_int8_cpu"][0]

        self.assertAlmostEqual(candidate["accuracy_delta_vs_fp32_cpu"], -0.005)
        self.assertAlmostEqual(candidate["macro_f1_delta_vs_fp32_cpu"], -0.008)
        self.assertAlmostEqual(candidate["model_size_reduction_vs_fp32_cpu"], 0.52)
        self.assertAlmostEqual(candidate["latency_reduction_vs_fp32_cpu"], 1.0 / 3.0)
        self.assertAlmostEqual(candidate["throughput_gain_vs_fp32_cpu"], 0.5001500150015001)
        self.assertAlmostEqual(candidate["peak_memory_reduction_vs_fp32_cpu"], 140.0 / 900.0)
        self.assertAlmostEqual(candidate["best_deployment_improvement_vs_fp32_cpu"], 0.52)

    def test_gate_passes_when_dynamic_int8_is_noninferior_and_smaller(self):
        rows = []
        for dataset in ["sst2", "mrpc"]:
            for seed in [1, 2]:
                rows.append(
                    {
                        "dataset": dataset,
                        "backbone": "distilbert-base-uncased",
                        "label_budget_per_class": 64,
                        "seed": seed,
                        "method": "fp32_cpu",
                        "status": "PASS",
                        "accuracy": 0.80,
                        "macro_f1": 0.79,
                        "model_size_mb": 250.0,
                        "latency_ms_per_example": 4.0,
                        "throughput_examples_per_second": 250.0,
                        "peak_memory_mb": 900.0,
                    }
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "backbone": "distilbert-base-uncased",
                        "label_budget_per_class": 64,
                        "seed": seed,
                        "method": "dynamic_int8_cpu",
                        "status": "PASS",
                        "accuracy": 0.795,
                        "macro_f1": 0.785,
                        "model_size_mb": 130.0,
                        "latency_ms_per_example": 3.5,
                        "throughput_examples_per_second": 285.7,
                        "peak_memory_mb": 840.0,
                    }
                )

        enriched = compute_reference_deltas(rows)
        gate = gate_dynamic_quantization(enriched, margin=-0.01, min_deployment_improvement=0.25)

        self.assertEqual(gate["status"], "PASS_MICRO")
        self.assertEqual(gate["passing_methods"], ["dynamic_int8_cpu"])
        self.assertEqual(gate["expected_paired_cells"], 4)

    def test_gate_does_not_pass_from_optional_gpu_reference_alone(self):
        rows = [
            {
                "dataset": "sst2",
                "backbone": "distilbert-base-uncased",
                "label_budget_per_class": 64,
                "seed": 1,
                "method": "fp32_cpu",
                "status": "PASS",
                "accuracy": 0.80,
                "macro_f1": 0.79,
                "model_size_mb": 250.0,
                "latency_ms_per_example": 10.0,
                "throughput_examples_per_second": 100.0,
                "peak_memory_mb": 900.0,
            },
            {
                "dataset": "sst2",
                "backbone": "distilbert-base-uncased",
                "label_budget_per_class": 64,
                "seed": 1,
                "method": "fp32_gpu",
                "status": "PASS",
                "accuracy": 0.80,
                "macro_f1": 0.79,
                "model_size_mb": 250.0,
                "latency_ms_per_example": 1.0,
                "throughput_examples_per_second": 1000.0,
                "peak_memory_mb": 900.0,
            },
        ]

        enriched = compute_reference_deltas(rows)
        gate = gate_dynamic_quantization(enriched, margin=-0.01, min_deployment_improvement=0.25)

        self.assertEqual(gate["status"], "WEAK_SIGNAL_STOP")
        self.assertEqual(gate["passing_methods"], [])

    def test_tradeoff_gate_passes_bounded_quality_cost_boundary(self):
        rows = []
        int8_metrics = [
            (0.805, 0.795),
            (0.755, 0.735),
            (0.795, 0.785),
            (0.765, 0.745),
        ]
        for seed, (int8_acc, int8_f1) in enumerate(int8_metrics, start=1):
            rows.append(
                {
                    "dataset": "sst2" if seed <= 2 else "mrpc",
                    "backbone": "distilbert-base-uncased",
                    "label_budget_per_class": 64,
                    "seed": seed,
                    "method": "fp32_cpu",
                    "status": "PASS",
                    "accuracy": 0.80,
                    "macro_f1": 0.79,
                    "model_size_mb": 250.0,
                    "latency_ms_per_example": 4.0,
                    "throughput_examples_per_second": 250.0,
                    "peak_memory_mb": 0.0,
                }
            )
            rows.append(
                {
                    "dataset": "sst2" if seed <= 2 else "mrpc",
                    "backbone": "distilbert-base-uncased",
                    "label_budget_per_class": 64,
                    "seed": seed,
                    "method": "dynamic_int8_cpu",
                    "status": "PASS",
                    "accuracy": int8_acc,
                    "macro_f1": int8_f1,
                    "model_size_mb": 130.0,
                    "latency_ms_per_example": 2.5,
                    "throughput_examples_per_second": 400.0,
                    "peak_memory_mb": 0.0,
                }
            )

        enriched = compute_reference_deltas(rows)
        strict_gate = gate_dynamic_quantization(enriched, margin=-0.01, min_deployment_improvement=0.25)
        tradeoff_gate = gate_deployment_tradeoff(enriched, margin=-0.01, min_deployment_improvement=0.25)

        self.assertEqual(strict_gate["status"], "WEAK_SIGNAL_STOP")
        self.assertEqual(tradeoff_gate["status"], "PASS_TRADEOFF_SIGNAL")
        self.assertEqual(tradeoff_gate["tradeoff_methods"], ["dynamic_int8_cpu"])
        self.assertEqual(tradeoff_gate["strict_noninferiority_status"], "WEAK_SIGNAL_STOP")

    def test_tradeoff_gate_rejects_severe_quality_collapse(self):
        rows = [
            {
                "dataset": "sst2",
                "backbone": "distilbert-base-uncased",
                "label_budget_per_class": 64,
                "seed": 1,
                "method": "fp32_cpu",
                "status": "PASS",
                "accuracy": 0.80,
                "macro_f1": 0.79,
                "model_size_mb": 250.0,
                "latency_ms_per_example": 4.0,
                "throughput_examples_per_second": 250.0,
                "peak_memory_mb": 0.0,
            },
            {
                "dataset": "sst2",
                "backbone": "distilbert-base-uncased",
                "label_budget_per_class": 64,
                "seed": 1,
                "method": "dynamic_int8_cpu",
                "status": "PASS",
                "accuracy": 0.55,
                "macro_f1": 0.50,
                "model_size_mb": 130.0,
                "latency_ms_per_example": 2.0,
                "throughput_examples_per_second": 500.0,
                "peak_memory_mb": 0.0,
            },
        ]

        enriched = compute_reference_deltas(rows)
        gate = gate_deployment_tradeoff(enriched, margin=-0.01, min_deployment_improvement=0.25)

        self.assertEqual(gate["status"], "WEAK_TRADEOFF_STOP")
        self.assertEqual(gate["fragile_methods"], ["dynamic_int8_cpu"])

    def test_not_applicable_rows_are_kept_but_excluded(self):
        rows = [
            {
                "dataset": "sst2",
                "backbone": "distilbert-base-uncased",
                "label_budget_per_class": 64,
                "seed": 1,
                "method": "fp32_cpu",
                "status": "PASS",
                "accuracy": 0.84,
                "macro_f1": 0.83,
                "model_size_mb": 250.0,
                "latency_ms_per_example": 3.0,
                "throughput_examples_per_second": 333.3,
                "peak_memory_mb": 900.0,
            },
            {
                "dataset": "sst2",
                "backbone": "distilbert-base-uncased",
                "label_budget_per_class": 64,
                "seed": 1,
                "method": "bf16_gpu",
                "status": "NOT_APPLICABLE",
                "accuracy": "nan",
                "macro_f1": "nan",
                "model_size_mb": "nan",
                "latency_ms_per_example": "nan",
                "throughput_examples_per_second": "nan",
                "peak_memory_mb": "nan",
            },
        ]

        enriched = compute_reference_deltas(rows)

        self.assertEqual(enriched[1]["status"], "NOT_APPLICABLE")
        self.assertNotIn("accuracy_delta_vs_fp32_cpu", enriched[1])

    def test_validate_result_rows_rejects_missing_reference(self):
        rows = [
            {
                "dataset": "sst2",
                "backbone": "distilbert-base-uncased",
                "label_budget_per_class": 64,
                "seed": 1,
                "method": "dynamic_int8_cpu",
                "status": "PASS",
                "accuracy": 0.8,
                "macro_f1": 0.8,
                "model_size_mb": 120.0,
                "latency_ms_per_example": 3.0,
                "throughput_examples_per_second": 333.0,
                "peak_memory_mb": 800.0,
            }
        ]

        with self.assertRaises(ValueError):
            validate_result_rows(rows, require_reference=True)


if __name__ == "__main__":
    unittest.main()
