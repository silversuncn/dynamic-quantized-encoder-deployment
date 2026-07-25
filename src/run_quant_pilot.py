#!/usr/bin/env python3
"""Run Paper18 v3 dynamic quantization micro/bounded/formal matrices."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import random
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dynamic_quant_core import (
    inference_method_configs,
    stable_hash,
    stratified_train_val_indices,
)


TASKS = {
    "sst2": {"hf_path": "glue", "hf_name": "sst2", "text_fields": ("sentence",), "num_labels": 2},
    "mrpc": {"hf_path": "glue", "hf_name": "mrpc", "text_fields": ("sentence1", "sentence2"), "num_labels": 2},
    "rte": {"hf_path": "glue", "hf_name": "rte", "text_fields": ("sentence1", "sentence2"), "num_labels": 2},
    "cola": {"hf_path": "glue", "hf_name": "cola", "text_fields": ("sentence",), "num_labels": 2},
    "qqp": {"hf_path": "glue", "hf_name": "qqp", "text_fields": ("question1", "question2"), "num_labels": 2},
    "qnli": {"hf_path": "glue", "hf_name": "qnli", "text_fields": ("question", "sentence"), "num_labels": 2},
}

FORMAL_DATASETS = ["sst2", "mrpc", "rte", "cola", "qqp"]
FORMAL_BACKBONES = ["prajjwal1/bert-tiny", "distilbert-base-uncased", "bert-base-uncased"]
FORMAL_BUDGETS = [64, 128, 256]
FORMAL_SEEDS = [1, 2, 3, 4, 5]
STRENGTHENED_DATASETS = ["sst2", "mrpc", "rte", "cola", "qqp"]
STRENGTHENED_BACKBONES = ["albert-base-v2", "bert-base-uncased"]
STRENGTHENED_BUDGETS = [64, 128, 256]
STRENGTHENED_SEEDS = [6, 7, 8]
STRENGTHENED_EXPECTED_ROWS = 360
MATRIX_CHOICES = ["micro", "bounded", "formal", "strengthened"]

FIELDNAMES = [
    "matrix",
    "dataset",
    "backbone",
    "label_budget_per_class",
    "validation_budget_per_class",
    "seed",
    "method",
    "status",
    "reference_method",
    "device",
    "inference_dtype",
    "quantization",
    "batch_size",
    "requested_batch_size",
    "max_inference_batch_size",
    "warmup_repeats",
    "timing_repeats",
    "max_epochs",
    "accuracy",
    "macro_f1",
    "model_size_mb",
    "latency_ms_per_example",
    "throughput_examples_per_second",
    "peak_memory_mb",
    "train_examples",
    "validation_examples",
    "test_examples",
    "train_class_counts",
    "validation_class_counts",
    "test_class_counts",
    "train_index_hash",
    "validation_index_hash",
    "test_index_hash",
    "fine_tune_seconds",
    "failure_reason",
]

DEFAULT_INFERENCE_BATCH_SIZE = 8
DEFAULT_MAX_INFERENCE_BATCH_SIZE = 8
RESULT_KEY_FIELDS = (
    "matrix",
    "dataset",
    "backbone",
    "label_budget_per_class",
    "validation_budget_per_class",
    "seed",
    "method",
)
BASE_UNIT_KEY_FIELDS = RESULT_KEY_FIELDS[:-1]
BATCH_SAFETY_FIELDS = ("batch_size", "requested_batch_size", "max_inference_batch_size")


@dataclass
class ResumeState:
    rows: list[dict[str, object]]
    completed_row_keys: set[tuple[str, ...]]
    completed_base_unit_keys: set[tuple[str, ...]]


class TextClassificationDataset(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_task(task: str):
    from datasets import load_dataset

    cfg = TASKS[task]
    return load_dataset(cfg["hf_path"], cfg["hf_name"])


def subset_by_indices(split, indices: list[int]) -> list[dict[str, object]]:
    return [split[int(idx)] for idx in indices]


def class_counts(labels: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        key = str(int(label))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def deterministic_test_indices(labels: list[int], cap_per_class: int, seed: int) -> list[int]:
    split = stratified_train_val_indices(labels, train_per_class=cap_per_class, val_per_class=0, seed=seed)
    return split["train_indices"]


def encode_rows(tokenizer, rows: list[dict[str, object]], task: str, max_length: int) -> TextClassificationDataset:
    fields = TASKS[task]["text_fields"]
    labels = [int(row["label"]) for row in rows]
    if len(fields) == 1:
        encodings = tokenizer([str(row[fields[0]]) for row in rows], truncation=True, padding=True, max_length=max_length)
    else:
        encodings = tokenizer(
            [str(row[fields[0]]) for row in rows],
            [str(row[fields[1]]) for row in rows],
            truncation=True,
            padding=True,
            max_length=max_length,
        )
    return TextClassificationDataset(encodings, labels)


def prepare_task_rows(task: str, train_per_class: int, val_per_class: int, seed: int):
    dataset = load_task(task)
    train_split = dataset["train"]
    test_split = dataset["validation"]
    train_labels = [int(train_split[i]["label"]) for i in range(len(train_split))]
    split = stratified_train_val_indices(train_labels, train_per_class, val_per_class, seed)
    test_labels = [int(test_split[i]["label"]) for i in range(len(test_split))]
    test_indices = deterministic_test_indices(test_labels, cap_per_class=128, seed=seed)
    return {
        "train_rows": subset_by_indices(train_split, split["train_indices"]),
        "validation_rows": subset_by_indices(train_split, split["validation_indices"]),
        "test_rows": subset_by_indices(test_split, test_indices),
        "split": split,
        "test_index_hash": stable_hash(test_indices),
        "test_class_counts": class_counts([test_labels[idx] for idx in test_indices]),
    }


def train_classifier(
    task: str,
    model_name: str,
    seed: int,
    train_rows: list[dict[str, object]],
    max_epochs: int,
    max_length: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[torch.nn.Module, AutoTokenizer, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=TASKS[task]["num_labels"],
        local_files_only=True,
    ).to(device)
    train_ds = encode_rows(tokenizer, train_rows, task, max_length)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _epoch in range(max_epochs):
        model.train()
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs)
            if not torch.isfinite(outputs.loss.detach()):
                raise FloatingPointError("non-finite fine-tuning loss")
            outputs.loss.backward()
            optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fine_tune_seconds = time.perf_counter() - start
    model = model.cpu().eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tokenizer, fine_tune_seconds


def model_size_mb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        torch.save(model.state_dict(), tmp.name)
        tmp.flush()
        return Path(tmp.name).stat().st_size / (1024 * 1024)


def quantize_dynamic_int8(model: torch.nn.Module) -> torch.nn.Module:
    quantize_dynamic = getattr(torch.ao.quantization, "quantize_dynamic")
    return quantize_dynamic(copy.deepcopy(model).cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)


def inference_context(dtype: str, device: torch.device):
    if device.type == "cuda" and dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def effective_inference_batch_size(requested: int, max_allowed: int) -> int:
    if requested < 1:
        raise ValueError("inference batch size must be positive")
    if max_allowed < 1:
        raise ValueError("max inference batch size must be positive")
    return min(requested, max_allowed)


def resolve_model_list(model: str, models: list[str] | None) -> list[str]:
    return models if models else [model]


def resolve_budget_plan(
    label_budget_per_class: int,
    validation_budget_per_class: int,
    budgets: list[int] | None,
    validation_budgets: list[int] | None,
) -> tuple[list[int], dict[int, int]]:
    label_budgets = budgets if budgets else [label_budget_per_class]
    if validation_budgets is not None:
        if len(validation_budgets) != len(label_budgets):
            raise ValueError("--validation-budgets must have the same length as --budgets")
        return label_budgets, dict(zip(label_budgets, validation_budgets, strict=True))
    if budgets is not None:
        return label_budgets, {budget: budget for budget in label_budgets}
    return label_budgets, {label_budget_per_class: validation_budget_per_class}


def planned_result_rows(
    datasets: list[str],
    models: list[str],
    label_budgets: list[int],
    seeds: list[int],
    method_count: int,
) -> int:
    return len(models) * len(datasets) * len(label_budgets) * len(seeds) * method_count


def key_tuple(row: dict[str, object], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in fields)


def result_key(row: dict[str, object]) -> tuple[str, ...]:
    return key_tuple(row, RESULT_KEY_FIELDS)


def base_unit_key(row: dict[str, object]) -> tuple[str, ...]:
    return key_tuple(row, BASE_UNIT_KEY_FIELDS)


def validate_safe_batch_fields(row: dict[str, object], expected_batch_size: int) -> None:
    for field in BATCH_SAFETY_FIELDS:
        try:
            value = int(str(row[field]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"unsafe batch field {field}={row.get(field)!r}") from exc
        if value != expected_batch_size:
            raise ValueError(f"unsafe batch field {field}={value}; expected {expected_batch_size}")


def read_pass_resume_rows(path: Path, expected_batch_size: int) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "PASS":
                continue
            validate_safe_batch_fields(row, expected_batch_size)
            rows.append(dict(row))
    return rows


def write_resume_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_resume_state(
    resume_paths: list[Path],
    output_path: Path,
    expected_methods: list[str],
    expected_batch_size: int,
) -> ResumeState:
    rows_by_key: dict[tuple[str, ...], dict[str, object]] = {}
    sources = list(resume_paths)
    if output_path.exists() and output_path not in sources:
        sources.append(output_path)
    for path in sources:
        for row in read_pass_resume_rows(path, expected_batch_size):
            rows_by_key[result_key(row)] = row

    rows = list(rows_by_key.values())
    methods_by_base: dict[tuple[str, ...], set[str]] = {}
    for row in rows:
        methods_by_base.setdefault(base_unit_key(row), set()).add(str(row["method"]))
    expected = set(expected_methods)
    completed_base_unit_keys = {key for key, methods in methods_by_base.items() if expected.issubset(methods)}
    write_resume_rows(output_path, rows)
    return ResumeState(
        rows=rows,
        completed_row_keys=set(rows_by_key),
        completed_base_unit_keys=completed_base_unit_keys,
    )


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: str,
    max_forward_batch_size: int | None = None,
) -> tuple[list[int], list[int]]:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            actual_batch_size = int(batch["labels"].shape[0])
            if max_forward_batch_size is not None and actual_batch_size > max_forward_batch_size:
                raise ValueError(
                    f"inference forward batch {actual_batch_size} exceeds safe limit {max_forward_batch_size}"
                )
            labels_batch = batch["labels"].to(device)
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            with inference_context(dtype, device):
                outputs = model(**inputs)
            preds.extend(outputs.logits.detach().cpu().argmax(dim=-1).tolist())
            labels.extend(labels_batch.detach().cpu().tolist())
    return preds, labels


def evaluate_variant(
    base_model_cpu: torch.nn.Module,
    tokenizer,
    task: str,
    test_rows: list[dict[str, object]],
    method: str,
    config: dict[str, object],
    max_length: int,
    batch_size: int,
    warmup_repeats: int,
    timing_repeats: int,
) -> dict[str, object]:
    if method == "dynamic_int8_cpu":
        model = quantize_dynamic_int8(base_model_cpu)
    else:
        model = copy.deepcopy(base_model_cpu)

    device_name = str(config["device"])
    dtype = str(config["dtype"])
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and device.type != "cuda":
        raise RuntimeError("cuda requested but unavailable")
    model = model.to(device).eval()
    ds = encode_rows(tokenizer, test_rows, task, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    for _ in range(warmup_repeats):
        _ = run_inference(model, loader, device, dtype, max_forward_batch_size=batch_size)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    preds: list[int] = []
    labels: list[int] = []
    for _ in range(timing_repeats):
        preds, labels = run_inference(model, loader, device, dtype, max_forward_batch_size=batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    examples = len(ds) * timing_repeats
    peak_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0
    size_mb = model_size_mb(model.cpu())
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": "PASS",
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "model_size_mb": size_mb,
        "latency_ms_per_example": (elapsed / examples) * 1000.0 if examples else 0.0,
        "throughput_examples_per_second": examples / elapsed if elapsed > 0 else 0.0,
        "peak_memory_mb": peak_memory_mb,
        "failure_reason": "",
    }


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        out = dict(row)
        for key in ("train_class_counts", "validation_class_counts", "test_class_counts"):
            out[key] = json.dumps(out[key], sort_keys=True, separators=(",", ":"))
        writer.writerow(out)


def write_heartbeat(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", choices=MATRIX_CHOICES, default="micro")
    parser.add_argument("--output", required=True)
    parser.add_argument("--heartbeat", required=True)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=["sst2", "mrpc"])
    parser.add_argument("--label-budget-per-class", type=int, default=64)
    parser.add_argument("--validation-budget-per-class", type=int, default=64)
    parser.add_argument("--budgets", nargs="+", type=int, default=None)
    parser.add_argument("--validation-budgets", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument("--max-inference-batch-size", type=int, default=DEFAULT_MAX_INFERENCE_BATCH_SIZE)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--gpu-available", action="store_true")
    parser.add_argument("--bf16-supported", action="store_true")
    parser.add_argument("--resume-from", nargs="*", type=Path, default=[])
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    effective_batch_size = effective_inference_batch_size(args.inference_batch_size, args.max_inference_batch_size)
    configs = inference_method_configs(args.gpu_available, args.bf16_supported)
    models = resolve_model_list(args.model, args.models)
    label_budgets, validation_budget_by_label_budget = resolve_budget_plan(
        args.label_budget_per_class,
        args.validation_budget_per_class,
        args.budgets,
        args.validation_budgets,
    )
    output = Path(args.output)
    heartbeat = Path(args.heartbeat)
    planned = planned_result_rows(args.datasets, models, label_budgets, args.seeds, len(configs))
    resume_state = build_resume_state(args.resume_from, output, list(configs), effective_batch_size)
    completed = len(resume_state.rows)
    completed_row_keys = set(resume_state.completed_row_keys)
    completed_base_unit_keys = set(resume_state.completed_base_unit_keys)
    failures: list[dict[str, object]] = []
    write_heartbeat(
        heartbeat,
        {
            "status": "RUNNING",
            "matrix": args.matrix,
            "planned_rows": planned,
            "completed_rows": completed,
            "datasets": args.datasets,
            "models": models,
            "label_budgets": label_budgets,
            "requested_batch_size": args.inference_batch_size,
            "batch_size": effective_batch_size,
            "max_inference_batch_size": args.max_inference_batch_size,
            "resume_sources": [str(path) for path in args.resume_from],
            "resumed_rows": len(resume_state.rows),
            "completed_base_units": len(completed_base_unit_keys),
        },
    )

    for model_name in models:
        for task in args.datasets:
            for label_budget in label_budgets:
                validation_budget = validation_budget_by_label_budget[label_budget]
                for seed in args.seeds:
                    base_probe = {
                        "matrix": args.matrix,
                        "dataset": task,
                        "backbone": model_name,
                        "label_budget_per_class": label_budget,
                        "validation_budget_per_class": validation_budget,
                        "seed": seed,
                    }
                    unit_key = base_unit_key(base_probe)
                    if unit_key in completed_base_unit_keys:
                        write_heartbeat(
                            heartbeat,
                            {
                                "status": "RUNNING",
                                "matrix": args.matrix,
                                "planned_rows": planned,
                                "completed_rows": completed,
                                "skipped_completed_base_unit": base_probe,
                                "failure_count": len(failures),
                            },
                        )
                        continue
                    prepared = prepare_task_rows(task, label_budget, validation_budget, seed)
                    base_model, tokenizer, fine_tune_seconds = train_classifier(
                        task,
                        model_name,
                        seed,
                        prepared["train_rows"],
                        args.max_epochs,
                        args.max_length,
                        args.learning_rate,
                        args.train_batch_size,
                    )
                    for method, config in configs.items():
                        base = {
                            "matrix": args.matrix,
                            "dataset": task,
                            "backbone": model_name,
                            "label_budget_per_class": label_budget,
                            "validation_budget_per_class": validation_budget,
                            "seed": seed,
                            "method": method,
                            "reference_method": config["reference_method"],
                            "device": config["device"],
                            "inference_dtype": config["dtype"],
                            "quantization": config["quantization"],
                            "batch_size": effective_batch_size,
                            "requested_batch_size": args.inference_batch_size,
                            "max_inference_batch_size": args.max_inference_batch_size,
                            "warmup_repeats": args.warmup_repeats,
                            "timing_repeats": args.timing_repeats,
                            "max_epochs": args.max_epochs,
                            "train_examples": len(prepared["train_rows"]),
                            "validation_examples": len(prepared["validation_rows"]),
                            "test_examples": len(prepared["test_rows"]),
                            "train_class_counts": prepared["split"]["train_class_counts"],
                            "validation_class_counts": prepared["split"]["validation_class_counts"],
                            "test_class_counts": prepared["test_class_counts"],
                            "train_index_hash": prepared["split"]["train_index_hash"],
                            "validation_index_hash": prepared["split"]["validation_index_hash"],
                            "test_index_hash": prepared["test_index_hash"],
                            "fine_tune_seconds": fine_tune_seconds,
                        }
                        row_key = result_key(base)
                        if row_key in completed_row_keys:
                            continue
                        if config.get("status") == "not_applicable":
                            row = {
                                **base,
                                "status": "NOT_APPLICABLE",
                                "accuracy": math.nan,
                                "macro_f1": math.nan,
                                "model_size_mb": math.nan,
                                "latency_ms_per_example": math.nan,
                                "throughput_examples_per_second": math.nan,
                                "peak_memory_mb": math.nan,
                                "failure_reason": config.get("not_applicable_reason", "not_applicable"),
                            }
                        else:
                            try:
                                result = evaluate_variant(
                                    base_model,
                                    tokenizer,
                                    task,
                                    prepared["test_rows"],
                                    method,
                                    config,
                                    args.max_length,
                                    effective_batch_size,
                                    args.warmup_repeats,
                                    args.timing_repeats,
                                )
                                row = {**base, **result}
                            except Exception as exc:  # noqa: BLE001 - failures are experimental evidence.
                                row = {
                                    **base,
                                    "status": "FAIL",
                                    "accuracy": math.nan,
                                    "macro_f1": math.nan,
                                    "model_size_mb": math.nan,
                                    "latency_ms_per_example": math.nan,
                                    "throughput_examples_per_second": math.nan,
                                    "peak_memory_mb": math.nan,
                                    "failure_reason": f"{type(exc).__name__}: {exc}",
                                }
                                failures.append(
                                    {
                                        "dataset": task,
                                        "backbone": model_name,
                                        "budget": label_budget,
                                        "seed": seed,
                                        "method": method,
                                        "error": row["failure_reason"],
                                    }
                                )
                        append_row(output, row)
                        completed += 1
                        if row["status"] == "PASS":
                            completed_row_keys.add(row_key)
                        write_heartbeat(
                            heartbeat,
                            {
                                "status": "RUNNING",
                                "matrix": args.matrix,
                                "planned_rows": planned,
                                "completed_rows": completed,
                                "last_row": {
                                    "dataset": task,
                                    "backbone": model_name,
                                    "budget": label_budget,
                                    "seed": seed,
                                    "method": method,
                                    "status": row["status"],
                                },
                                "failure_count": len(failures),
                            },
                        )
                    del base_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    terminal = "PASS_RUN_COMPLETE" if not failures else "FAIL_RUN_COMPLETE"
    write_heartbeat(
        heartbeat,
        {
            "status": terminal,
            "matrix": args.matrix,
            "planned_rows": planned,
            "completed_rows": completed,
            "failure_count": len(failures),
            "failures": failures[:8],
            "results": str(output),
        },
    )
    print(json.dumps({"status": terminal, "rows": completed}))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
