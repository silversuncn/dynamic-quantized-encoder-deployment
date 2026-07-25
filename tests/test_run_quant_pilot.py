import unittest
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from dynamic_quant_core import inference_method_configs
from run_quant_pilot import (
    FIELDNAMES,
    FORMAL_BACKBONES,
    FORMAL_BUDGETS,
    FORMAL_DATASETS,
    FORMAL_SEEDS,
    MATRIX_CHOICES,
    STRENGTHENED_BACKBONES,
    STRENGTHENED_BUDGETS,
    STRENGTHENED_DATASETS,
    STRENGTHENED_EXPECTED_ROWS,
    STRENGTHENED_SEEDS,
    TASKS,
    TextClassificationDataset,
    build_resume_state,
    effective_inference_batch_size,
    planned_result_rows,
    resolve_budget_plan,
    run_inference,
)


class BatchRecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []
        self.labels_seen = []

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = int(input_ids.shape[0])
        self.batch_sizes.append(batch_size)
        self.labels_seen.append(labels is not None)
        logits = torch.zeros(batch_size, 2)
        logits[:, 0] = 1.0
        return SimpleNamespace(logits=logits)


def make_dataset(n_rows: int) -> TextClassificationDataset:
    return TextClassificationDataset(
        {"input_ids": [[101, 102] for _ in range(n_rows)], "attention_mask": [[1, 1] for _ in range(n_rows)]},
        [0 for _ in range(n_rows)],
    )


def make_result_row(method: str, status: str = "PASS", batch_size: int = 8) -> dict[str, object]:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "matrix": "formal",
            "dataset": "sst2",
            "backbone": "distilbert-base-uncased",
            "label_budget_per_class": 64,
            "validation_budget_per_class": 64,
            "seed": 1,
            "method": method,
            "status": status,
            "batch_size": batch_size,
            "requested_batch_size": batch_size,
            "max_inference_batch_size": batch_size,
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "model_size_mb": 1.0,
            "latency_ms_per_example": 1.0,
            "throughput_examples_per_second": 1.0,
            "peak_memory_mb": 0.0,
            "train_examples": 128,
            "validation_examples": 128,
            "test_examples": 256,
            "train_class_counts": "{\"0\":64,\"1\":64}",
            "validation_class_counts": "{\"0\":64,\"1\":64}",
            "test_class_counts": "{\"0\":128,\"1\":128}",
            "train_index_hash": "train",
            "validation_index_hash": "validation",
            "test_index_hash": "test",
        }
    )
    return row


def write_result_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class RunQuantPilotInferenceTests(unittest.TestCase):
    def test_effective_inference_batch_size_clamps_to_safe_limit(self):
        self.assertEqual(effective_inference_batch_size(4, 8), 4)
        self.assertEqual(effective_inference_batch_size(8, 8), 8)
        self.assertEqual(effective_inference_batch_size(256, 8), 8)

    def test_effective_inference_batch_size_rejects_nonpositive_values(self):
        with self.assertRaises(ValueError):
            effective_inference_batch_size(0, 8)
        with self.assertRaises(ValueError):
            effective_inference_batch_size(8, 0)

    def test_run_inference_uses_loader_microbatches(self):
        model = BatchRecordingModel()
        loader = DataLoader(make_dataset(20), batch_size=8, shuffle=False)

        preds, labels = run_inference(model, loader, torch.device("cpu"), "fp32", max_forward_batch_size=8)

        self.assertEqual(model.batch_sizes, [8, 8, 4])
        self.assertEqual(len(preds), 20)
        self.assertEqual(len(labels), 20)
        self.assertFalse(any(model.labels_seen))

    def test_run_inference_rejects_oversized_forward_batch(self):
        model = BatchRecordingModel()
        loader = DataLoader(make_dataset(9), batch_size=9, shuffle=False)

        with self.assertRaisesRegex(ValueError, "exceeds safe limit"):
            run_inference(model, loader, torch.device("cpu"), "fp32", max_forward_batch_size=8)

    def test_formal_design_has_expected_dimensions_and_registered_tasks(self):
        self.assertEqual(FORMAL_DATASETS, ["sst2", "mrpc", "rte", "cola", "qqp"])
        self.assertEqual(FORMAL_BACKBONES, ["prajjwal1/bert-tiny", "distilbert-base-uncased", "bert-base-uncased"])
        self.assertEqual(FORMAL_BUDGETS, [64, 128, 256])
        self.assertEqual(FORMAL_SEEDS, [1, 2, 3, 4, 5])
        self.assertTrue(set(FORMAL_DATASETS).issubset(TASKS))

        method_count = len(inference_method_configs(gpu_available=True, bf16_supported=True))
        self.assertEqual(planned_result_rows(FORMAL_DATASETS, FORMAL_BACKBONES, FORMAL_BUDGETS, FORMAL_SEEDS, method_count), 900)

    def test_strengthened_design_has_expected_dimensions_and_registered_tasks(self):
        self.assertIn("strengthened", MATRIX_CHOICES)
        self.assertEqual(STRENGTHENED_DATASETS, ["sst2", "mrpc", "rte", "cola", "qqp"])
        self.assertEqual(STRENGTHENED_BACKBONES, ["albert-base-v2", "bert-base-uncased"])
        self.assertEqual(STRENGTHENED_BUDGETS, [64, 128, 256])
        self.assertEqual(STRENGTHENED_SEEDS, [6, 7, 8])
        self.assertTrue(set(STRENGTHENED_DATASETS).issubset(TASKS))

        method_count = len(inference_method_configs(gpu_available=True, bf16_supported=True))
        self.assertEqual(
            planned_result_rows(
                STRENGTHENED_DATASETS,
                STRENGTHENED_BACKBONES,
                STRENGTHENED_BUDGETS,
                STRENGTHENED_SEEDS,
                method_count,
            ),
            STRENGTHENED_EXPECTED_ROWS,
        )

    def test_formal_budget_plan_defaults_validation_budget_to_matching_label_budget(self):
        label_budgets, validation_by_budget = resolve_budget_plan(64, 64, [64, 128, 256], None)

        self.assertEqual(label_budgets, [64, 128, 256])
        self.assertEqual(validation_by_budget, {64: 64, 128: 128, 256: 256})

    def test_budget_plan_rejects_mismatched_validation_budgets(self):
        with self.assertRaises(ValueError):
            resolve_budget_plan(64, 64, [64, 128, 256], [64, 128])

    def test_resume_state_imports_pass_rows_deduplicates_and_marks_complete_base_units(self):
        methods = list(inference_method_configs(gpu_available=True, bf16_supported=True))
        with tempfile.TemporaryDirectory() as tmp:
            resume_csv = Path(tmp) / "formal_matrix_20260724.csv"
            output_csv = Path(tmp) / "formal_matrix_20260725.csv"
            rows = [make_result_row(method) for method in methods]
            duplicate = make_result_row(methods[0])
            duplicate["accuracy"] = 0.5
            write_result_csv(resume_csv, rows + [duplicate])

            state = build_resume_state([resume_csv], output_csv, methods, expected_batch_size=8)

            self.assertEqual(len(state.rows), len(methods))
            self.assertEqual(len(state.completed_row_keys), len(methods))
            self.assertEqual(len(state.completed_base_unit_keys), 1)
            with output_csv.open(newline="", encoding="utf-8") as fh:
                rewritten_rows = list(csv.DictReader(fh))
            self.assertEqual(len(rewritten_rows), len(methods))
            self.assertEqual(rewritten_rows[0]["accuracy"], "0.5")

    def test_resume_state_keeps_partial_base_units_open_for_missing_methods(self):
        methods = list(inference_method_configs(gpu_available=True, bf16_supported=True))
        with tempfile.TemporaryDirectory() as tmp:
            resume_csv = Path(tmp) / "partial.csv"
            output_csv = Path(tmp) / "formal_matrix_20260725.csv"
            write_result_csv(resume_csv, [make_result_row(method) for method in methods[:2]])

            state = build_resume_state([resume_csv], output_csv, methods, expected_batch_size=8)

            self.assertEqual(len(state.completed_row_keys), 2)
            self.assertEqual(len(state.completed_base_unit_keys), 0)

    def test_resume_state_rejects_unsafe_batch_fields(self):
        methods = list(inference_method_configs(gpu_available=True, bf16_supported=True))
        with tempfile.TemporaryDirectory() as tmp:
            resume_csv = Path(tmp) / "unsafe.csv"
            output_csv = Path(tmp) / "formal_matrix_20260725.csv"
            write_result_csv(resume_csv, [make_result_row(methods[0], batch_size=16)])

            with self.assertRaisesRegex(ValueError, "unsafe batch field"):
                build_resume_state([resume_csv], output_csv, methods, expected_batch_size=8)


if __name__ == "__main__":
    unittest.main()
