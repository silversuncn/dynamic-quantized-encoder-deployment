"""Core utilities for Dynamic quantized encoder deployment."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from statistics import mean
from typing import Iterable


CELL_KEYS = ("dataset", "backbone", "label_budget_per_class", "seed")
REFERENCE_METHOD = "fp32_cpu"

REQUIRED_RESULT_FIELDS = {
    "dataset",
    "backbone",
    "label_budget_per_class",
    "seed",
    "method",
    "status",
    "accuracy",
    "macro_f1",
    "model_size_mb",
    "latency_ms_per_example",
    "throughput_examples_per_second",
    "peak_memory_mb",
}


def stable_hash(values: Iterable[int]) -> str:
    payload = json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stratified_train_val_indices(
    labels: list[int] | list[str],
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> dict[str, object]:
    if train_per_class < 0 or val_per_class < 0:
        raise ValueError("train_per_class and val_per_class must be non-negative")

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped[str(label)].append(idx)

    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    train_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}

    for label in sorted(grouped):
        candidates = grouped[label][:]
        rng.shuffle(candidates)
        train = candidates[:train_per_class]
        validation = candidates[train_per_class : train_per_class + val_per_class]
        train_indices.extend(train)
        validation_indices.extend(validation)
        train_counts[label] = len(train)
        validation_counts[label] = len(validation)

    train_indices.sort()
    validation_indices.sort()
    overlap = sorted(set(train_indices) & set(validation_indices))
    if overlap:
        raise AssertionError(f"train/validation overlap detected: {overlap[:5]}")

    return {
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "train_class_counts": train_counts,
        "validation_class_counts": validation_counts,
        "train_index_hash": stable_hash(train_indices),
        "validation_index_hash": stable_hash(validation_indices),
        "requested_train_per_class": train_per_class,
        "requested_validation_per_class": val_per_class,
    }


def inference_method_configs(
    gpu_available: bool = True,
    bf16_supported: bool = True,
) -> dict[str, dict[str, object]]:
    configs: dict[str, dict[str, object]] = {
        "fp32_cpu": {
            "status": "planned",
            "device": "cpu",
            "dtype": "fp32",
            "quantization": "none",
            "reference_method": REFERENCE_METHOD,
        },
        "dynamic_int8_cpu": {
            "status": "planned",
            "device": "cpu",
            "dtype": "int8_dynamic",
            "quantization": "dynamic_int8_linear",
            "reference_method": REFERENCE_METHOD,
        },
        "fp32_gpu": {
            "status": "planned" if gpu_available else "not_applicable",
            "device": "cuda",
            "dtype": "fp32",
            "quantization": "none",
            "reference_method": "fp32_gpu",
        },
        "bf16_gpu": {
            "status": "planned" if gpu_available and bf16_supported else "not_applicable",
            "device": "cuda",
            "dtype": "bf16",
            "quantization": "none",
            "reference_method": "fp32_gpu",
        },
    }
    if not gpu_available:
        configs["fp32_gpu"]["not_applicable_reason"] = "cuda unavailable"
        configs["bf16_gpu"]["not_applicable_reason"] = "cuda unavailable"
    elif not bf16_supported:
        configs["bf16_gpu"]["not_applicable_reason"] = "bf16 unsupported"
    return configs


def cell_key(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row[key] for key in CELL_KEYS)


def validate_result_rows(rows: list[dict[str, object]], require_reference: bool = False) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    for idx, row in enumerate(rows):
        missing = sorted(REQUIRED_RESULT_FIELDS - set(row))
        if missing:
            raise ValueError(f"row {idx} missing fields: {missing}")
        if row.get("status") != "PASS":
            continue
        for key in (
            "accuracy",
            "macro_f1",
            "model_size_mb",
            "latency_ms_per_example",
            "throughput_examples_per_second",
            "peak_memory_mb",
        ):
            value = float(row[key])
            if math.isnan(value):
                raise ValueError(f"row {idx} has NaN {key}")
            if value < 0:
                raise ValueError(f"row {idx} has negative {key}")

    if require_reference:
        by_cell: dict[tuple[object, ...], set[str]] = defaultdict(set)
        for row in rows:
            if row.get("status") == "PASS":
                by_cell[cell_key(row)].add(str(row["method"]))
        missing_reference = [key for key, methods in by_cell.items() if REFERENCE_METHOD not in methods]
        if missing_reference:
            raise ValueError(f"missing {REFERENCE_METHOD} reference for {len(missing_reference)} cells")


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_reference_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    validate_result_rows(rows, require_reference=True)

    references: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        if row.get("status") == "PASS" and row.get("method") == REFERENCE_METHOD:
            references[cell_key(row)] = row

    enriched: list[dict[str, object]] = []
    for row in rows:
        out = dict(row)
        reference = references.get(cell_key(row))
        if reference and row.get("status") == "PASS":
            accuracy_delta = float(row["accuracy"]) - float(reference["accuracy"])
            f1_delta = float(row["macro_f1"]) - float(reference["macro_f1"])
            size_reduction = _safe_fraction(
                float(reference["model_size_mb"]) - float(row["model_size_mb"]),
                float(reference["model_size_mb"]),
            )
            latency_reduction = _safe_fraction(
                float(reference["latency_ms_per_example"]) - float(row["latency_ms_per_example"]),
                float(reference["latency_ms_per_example"]),
            )
            throughput_gain = _safe_fraction(
                float(row["throughput_examples_per_second"]) - float(reference["throughput_examples_per_second"]),
                float(reference["throughput_examples_per_second"]),
            )
            memory_reduction = _safe_fraction(
                float(reference["peak_memory_mb"]) - float(row["peak_memory_mb"]),
                float(reference["peak_memory_mb"]),
            )
            out.update(
                {
                    "accuracy_delta_vs_fp32_cpu": accuracy_delta,
                    "macro_f1_delta_vs_fp32_cpu": f1_delta,
                    "model_size_reduction_vs_fp32_cpu": size_reduction,
                    "latency_reduction_vs_fp32_cpu": latency_reduction,
                    "throughput_gain_vs_fp32_cpu": throughput_gain,
                    "peak_memory_reduction_vs_fp32_cpu": memory_reduction,
                    "best_deployment_improvement_vs_fp32_cpu": max(
                        size_reduction,
                        latency_reduction,
                        throughput_gain,
                        memory_reduction,
                    ),
                }
            )
        enriched.append(out)
    return enriched


def gate_dynamic_quantization(
    rows: list[dict[str, object]],
    margin: float,
    min_deployment_improvement: float,
) -> dict[str, object]:
    validate_result_rows(rows, require_reference=True)
    paired_cells = {
        cell_key(row)
        for row in rows
        if row.get("status") == "PASS" and row.get("method") == REFERENCE_METHOD
    }
    method_summaries: dict[str, dict[str, object]] = {}

    for method in sorted({str(row["method"]) for row in rows if row["method"] != REFERENCE_METHOD}):
        method_rows = [
            row
            for row in rows
            if row.get("status") == "PASS"
            and str(row["method"]) == method
            and "accuracy_delta_vs_fp32_cpu" in row
        ]
        if not method_rows:
            method_summaries[method] = {
                "paired_cells": 0,
                "noninferior_cells": 0,
                "mean_best_deployment_improvement": None,
                "passes": False,
            }
            continue

        noninferior = [
            row
            for row in method_rows
            if float(row["accuracy_delta_vs_fp32_cpu"]) >= margin
            and float(row["macro_f1_delta_vs_fp32_cpu"]) >= margin
        ]
        mean_improvement = mean(float(row["best_deployment_improvement_vs_fp32_cpu"]) for row in method_rows)
        passes = len(noninferior) > len(method_rows) / 2 and mean_improvement >= min_deployment_improvement
        method_summaries[method] = {
            "paired_cells": len(method_rows),
            "noninferior_cells": len(noninferior),
            "mean_accuracy_delta": mean(float(row["accuracy_delta_vs_fp32_cpu"]) for row in method_rows),
            "mean_macro_f1_delta": mean(float(row["macro_f1_delta_vs_fp32_cpu"]) for row in method_rows),
            "mean_model_size_reduction": mean(float(row["model_size_reduction_vs_fp32_cpu"]) for row in method_rows),
            "mean_latency_reduction": mean(float(row["latency_reduction_vs_fp32_cpu"]) for row in method_rows),
            "mean_throughput_gain": mean(float(row["throughput_gain_vs_fp32_cpu"]) for row in method_rows),
            "mean_peak_memory_reduction": mean(float(row["peak_memory_reduction_vs_fp32_cpu"]) for row in method_rows),
            "mean_best_deployment_improvement": mean_improvement,
            "passes": passes,
        }

    primary_quant_methods = {
        method
        for method in method_summaries
        if "quant" in method or "int8" in method
    }
    passing_methods = sorted(
        method for method, summary in method_summaries.items() if summary["passes"] and method in primary_quant_methods
    )
    return {
        "status": "PASS_MICRO" if passing_methods else "WEAK_SIGNAL_STOP",
        "reference_method": REFERENCE_METHOD,
        "expected_paired_cells": len(paired_cells),
        "passing_methods": passing_methods,
        "method_summaries": method_summaries,
        "margin": margin,
        "min_deployment_improvement": min_deployment_improvement,
    }


def gate_deployment_tradeoff(
    rows: list[dict[str, object]],
    margin: float,
    min_deployment_improvement: float,
    max_mean_quality_drop: float = 0.08,
) -> dict[str, object]:
    validate_result_rows(rows, require_reference=True)
    strict_gate = gate_dynamic_quantization(rows, margin, min_deployment_improvement)
    method_summaries = strict_gate["method_summaries"]

    primary_quant_methods = {
        method
        for method in method_summaries
        if "quant" in method or "int8" in method
    }
    stable_methods = sorted(
        method
        for method in primary_quant_methods
        if method_summaries[method].get("passes")
    )
    tradeoff_methods: list[str] = []
    fragile_methods: list[str] = []

    for method in sorted(primary_quant_methods):
        summary = method_summaries[method]
        paired = int(summary.get("paired_cells") or 0)
        if paired == 0:
            continue
        noninferior = int(summary.get("noninferior_cells") or 0)
        mean_improvement = summary.get("mean_best_deployment_improvement")
        mean_accuracy_delta = summary.get("mean_accuracy_delta")
        mean_macro_f1_delta = summary.get("mean_macro_f1_delta")
        if mean_improvement is None or mean_accuracy_delta is None or mean_macro_f1_delta is None:
            continue
        has_deployment_gain = float(mean_improvement) >= min_deployment_improvement
        quality_drop_bounded = (
            float(mean_accuracy_delta) >= -max_mean_quality_drop
            and float(mean_macro_f1_delta) >= -max_mean_quality_drop
        )
        has_boundary_signal = 0 < noninferior < paired
        if has_deployment_gain and quality_drop_bounded and has_boundary_signal:
            tradeoff_methods.append(method)
        elif has_deployment_gain and not quality_drop_bounded:
            fragile_methods.append(method)

    passing_methods = sorted(set(stable_methods) | set(tradeoff_methods))
    return {
        "status": "PASS_TRADEOFF_SIGNAL" if passing_methods else "WEAK_TRADEOFF_STOP",
        "reference_method": REFERENCE_METHOD,
        "expected_paired_cells": strict_gate["expected_paired_cells"],
        "passing_methods": passing_methods,
        "stable_methods": stable_methods,
        "tradeoff_methods": tradeoff_methods,
        "fragile_methods": fragile_methods,
        "method_summaries": method_summaries,
        "margin": margin,
        "min_deployment_improvement": min_deployment_improvement,
        "max_mean_quality_drop": max_mean_quality_drop,
        "strict_noninferiority_status": strict_gate["status"],
    }
