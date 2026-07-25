#!/usr/bin/env python3
"""Aggregate Paper18 v3 dynamic quantization results and apply gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from dynamic_quant_core import compute_reference_deltas, gate_deployment_tradeoff, gate_dynamic_quantization


NUMERIC_FIELDS = {
    "label_budget_per_class": int,
    "validation_budget_per_class": int,
    "seed": int,
    "batch_size": int,
    "requested_batch_size": int,
    "max_inference_batch_size": int,
    "warmup_repeats": int,
    "timing_repeats": int,
    "max_epochs": int,
    "train_examples": int,
    "validation_examples": int,
    "test_examples": int,
    "accuracy": float,
    "macro_f1": float,
    "model_size_mb": float,
    "latency_ms_per_example": float,
    "throughput_examples_per_second": float,
    "peak_memory_mb": float,
}


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, object] = dict(raw)
            for key, caster in NUMERIC_FIELDS.items():
                if row.get(key) not in {"", None}:
                    row[key] = caster(row[key])
            rows.append(row)
    return rows


def method_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if row["method"] == method and row.get("status") == "PASS"]
        if not method_rows:
            summaries[method] = {"n": 0}
            continue
        summaries[method] = {
            "n": len(method_rows),
            "mean_accuracy": mean(float(row["accuracy"]) for row in method_rows),
            "mean_macro_f1": mean(float(row["macro_f1"]) for row in method_rows),
            "mean_model_size_mb": mean(float(row["model_size_mb"]) for row in method_rows),
            "mean_latency_ms_per_example": mean(float(row["latency_ms_per_example"]) for row in method_rows),
            "mean_throughput_examples_per_second": mean(
                float(row["throughput_examples_per_second"]) for row in method_rows
            ),
            "mean_peak_memory_mb": mean(float(row["peak_memory_mb"]) for row in method_rows),
        }
        if method != "fp32_cpu" and "accuracy_delta_vs_fp32_cpu" in method_rows[0]:
            summaries[method].update(
                {
                    "mean_accuracy_delta": mean(float(row["accuracy_delta_vs_fp32_cpu"]) for row in method_rows),
                    "mean_macro_f1_delta": mean(float(row["macro_f1_delta_vs_fp32_cpu"]) for row in method_rows),
                    "mean_best_deployment_improvement": mean(
                        float(row["best_deployment_improvement_vs_fp32_cpu"]) for row in method_rows
                    ),
                }
            )
    return summaries


def matrix_status(matrix: str, gate_status: str) -> str:
    if matrix == "bounded" and gate_status == "PASS_MICRO":
        return "PASS_BOUNDED_PILOT_READY"
    if matrix == "bounded" and gate_status == "PASS_TRADEOFF_SIGNAL":
        return "PASS_BOUNDED_TRADEOFF_READY"
    if matrix == "formal" and gate_status == "PASS_MICRO":
        return "PASS_FORMAL_MATRIX_READY"
    if matrix == "formal" and gate_status == "PASS_TRADEOFF_SIGNAL":
        return "PASS_FORMAL_TRADEOFF_READY"
    if matrix == "strengthened" and gate_status == "PASS_TRADEOFF_SIGNAL":
        return "PASS_STRENGTHENED_TRADEOFF_READY"
    if matrix == "strengthened" and gate_status == "WEAK_TRADEOFF_STOP":
        return "WEAK_STRENGTHENED_TRADEOFF_STOP"
    return gate_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--matrix", choices=["micro", "bounded", "formal", "strengthened"], default="micro")
    parser.add_argument("--gate-mode", choices=["noninferiority", "tradeoff"], default="noninferiority")
    parser.add_argument("--margin", type=float, default=-0.01)
    parser.add_argument("--min-deployment-improvement", type=float, default=0.25)
    parser.add_argument("--max-mean-quality-drop", type=float, default=0.08)
    args = parser.parse_args()

    rows = load_rows(Path(args.results))
    enriched = compute_reference_deltas(rows)
    if args.gate_mode == "tradeoff":
        gate = gate_deployment_tradeoff(
            enriched,
            args.margin,
            args.min_deployment_improvement,
            args.max_mean_quality_drop,
        )
    else:
        gate = gate_dynamic_quantization(enriched, args.margin, args.min_deployment_improvement)
    status = matrix_status(args.matrix, str(gate["status"]))
    report = {
        "status": status,
        "matrix": args.matrix,
        "gate_mode": args.gate_mode,
        "results": args.results,
        "n_rows": len(enriched),
        "method_summary": method_summary(enriched),
        "gate": gate,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(out)}, sort_keys=True))
    return 0 if report["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
