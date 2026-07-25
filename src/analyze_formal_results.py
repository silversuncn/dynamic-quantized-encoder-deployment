#!/usr/bin/env python3
"""Produce local Phase 4 summaries for Paper18 formal matrix results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

from aggregate_quant_results import load_rows, method_summary
from dynamic_quant_core import compute_reference_deltas


METRIC_FIELDS = [
    "accuracy",
    "macro_f1",
    "model_size_mb",
    "latency_ms_per_example",
    "throughput_examples_per_second",
    "peak_memory_mb",
    "accuracy_delta_vs_fp32_cpu",
    "macro_f1_delta_vs_fp32_cpu",
    "model_size_reduction_vs_fp32_cpu",
    "latency_reduction_vs_fp32_cpu",
    "throughput_gain_vs_fp32_cpu",
    "best_deployment_improvement_vs_fp32_cpu",
]


def finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def phase4_artifact_path(directory: Path, stem: str, artifact_date: str, suffix: str) -> Path:
    return directory / f"{stem}_{artifact_date}.{suffix}"


def phase4_artifact_stem(artifact_prefix: str, artifact_name: str) -> str:
    return f"{artifact_prefix}_{artifact_name}"


def flatten_summary(summary: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, values in sorted(summary.items()):
        row = {"method": method}
        row.update(values)
        rows.append(row)
    return rows


def group_mean_rows(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "PASS":
            continue
        grouped[tuple(row.get(key) for key in keys)].append(row)

    out_rows: list[dict[str, object]] = []
    for key_values, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        out: dict[str, object] = dict(zip(keys, key_values))
        out["n"] = len(group)
        for field in METRIC_FIELDS:
            values = [value for value in (finite_float(row.get(field)) for row in group) if value is not None]
            if values:
                out[f"mean_{field}"] = mean(values)
        out_rows.append(out)
    return out_rows


def describe(values: Iterable[float]) -> dict[str, object]:
    vals = list(values)
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "pstdev": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def dynamic_int8_stats(rows: list[dict[str, object]], gate: dict[str, object]) -> dict[str, object]:
    dyn = [row for row in rows if row.get("status") == "PASS" and row.get("method") == "dynamic_int8_cpu"]
    acc_delta = [value for value in (finite_float(row.get("accuracy_delta_vs_fp32_cpu")) for row in dyn) if value is not None]
    f1_delta = [value for value in (finite_float(row.get("macro_f1_delta_vs_fp32_cpu")) for row in dyn) if value is not None]
    improvement = [
        value
        for value in (finite_float(row.get("best_deployment_improvement_vs_fp32_cpu")) for row in dyn)
        if value is not None
    ]
    noninferior = [
        row
        for row in dyn
        if finite_float(row.get("accuracy_delta_vs_fp32_cpu")) is not None
        and finite_float(row.get("macro_f1_delta_vs_fp32_cpu")) is not None
        and float(row["accuracy_delta_vs_fp32_cpu"]) >= -0.01
        and float(row["macro_f1_delta_vs_fp32_cpu"]) >= -0.01
    ]
    severe_cells = [
        {
            "dataset": row.get("dataset"),
            "backbone": row.get("backbone"),
            "budget": row.get("label_budget_per_class"),
            "seed": row.get("seed"),
            "accuracy_delta": row.get("accuracy_delta_vs_fp32_cpu"),
            "macro_f1_delta": row.get("macro_f1_delta_vs_fp32_cpu"),
        }
        for row in dyn
        if (finite_float(row.get("accuracy_delta_vs_fp32_cpu")) or 0.0) < -0.08
        or (finite_float(row.get("macro_f1_delta_vs_fp32_cpu")) or 0.0) < -0.08
    ]
    gate_dynamic = gate.get("gate", {}).get("method_summaries", {}).get("dynamic_int8_cpu", {})
    return {
        "formal_gate_status": gate.get("status"),
        "dynamic_int8_cpu_gate_summary": gate_dynamic,
        "accuracy_delta": describe(acc_delta),
        "macro_f1_delta": describe(f1_delta),
        "best_deployment_improvement": describe(improvement),
        "noninferior_cells": len(noninferior),
        "paired_cells": len(dyn),
        "severe_quality_cells": severe_cells[:20],
        "severe_quality_cell_count": len(severe_cells),
    }


def write_figures(
    group_rows: list[dict[str, object]],
    method_rows: list[dict[str, object]],
    figure_dir: Path,
    figure_prefix: str = "formal",
) -> dict[str, object]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - figure availability is environment-dependent evidence.
        return {"status": "SKIPPED", "reason": f"{type(exc).__name__}: {exc}"}

    generated: list[str] = []
    dyn_groups = [row for row in group_rows if row.get("method") == "dynamic_int8_cpu"]
    if dyn_groups:
        labels = [f"{row['dataset']}/{row['backbone']}/{row['label_budget_per_class']}" for row in dyn_groups]
        values = [float(row.get("mean_macro_f1_delta_vs_fp32_cpu", 0.0)) for row in dyn_groups]
        height = max(4.0, min(14.0, 0.24 * len(labels)))
        fig, ax = plt.subplots(figsize=(10, height))
        ax.barh(range(len(labels)), values, color="#4C78A8")
        ax.axvline(-0.08, color="#D62728", linestyle="--", linewidth=1)
        ax.axvline(-0.01, color="#7F7F7F", linestyle=":", linewidth=1)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Mean macro-F1 delta vs fp32_cpu")
        ax.set_title("Dynamic INT8 quality deltas by dataset/backbone/budget")
        fig.tight_layout()
        path = figure_dir / f"{figure_prefix}_dynamic_int8_macro_f1_delta.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        generated.append(str(path))

    methods = [row for row in method_rows if row.get("method") != "fp32_cpu" and row.get("mean_best_deployment_improvement") not in {"", None}]
    if methods:
        labels = [str(row["method"]) for row in methods]
        improvements = [float(row["mean_best_deployment_improvement"]) for row in methods]
        acc = [float(row.get("mean_accuracy_delta", 0.0)) for row in methods]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].bar(labels, improvements, color="#59A14F")
        axes[0].axhline(0.25, color="#7F7F7F", linestyle=":", linewidth=1)
        axes[0].set_ylabel("Best deployment improvement")
        axes[0].tick_params(axis="x", rotation=20)
        axes[1].bar(labels, acc, color="#F28E2B")
        axes[1].axhline(-0.08, color="#D62728", linestyle="--", linewidth=1)
        axes[1].axhline(-0.01, color="#7F7F7F", linestyle=":", linewidth=1)
        axes[1].set_ylabel("Mean accuracy delta")
        axes[1].tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = figure_dir / f"{figure_prefix}_method_tradeoff_summary.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        generated.append(str(path))

    return {"status": "PASS", "generated": generated}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    parser.add_argument("--artifact-date", default="20260724")
    parser.add_argument("--artifact-prefix", default="formal_phase4")
    parser.add_argument("--figure-prefix", default=None)
    args = parser.parse_args()

    rows = load_rows(args.results)
    enriched = compute_reference_deltas(rows)
    gate = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    delta_fields = list(rows[0].keys()) + [
        "accuracy_delta_vs_fp32_cpu",
        "macro_f1_delta_vs_fp32_cpu",
        "model_size_reduction_vs_fp32_cpu",
        "latency_reduction_vs_fp32_cpu",
        "throughput_gain_vs_fp32_cpu",
        "peak_memory_reduction_vs_fp32_cpu",
        "best_deployment_improvement_vs_fp32_cpu",
    ]
    figure_prefix = args.figure_prefix or args.artifact_prefix.replace("_phase4", "")
    paired_delta_path = phase4_artifact_path(args.output_dir, phase4_artifact_stem(args.artifact_prefix, "paired_deltas"), args.artifact_date, "csv")
    write_csv(paired_delta_path, enriched, delta_fields)

    method_rows = flatten_summary(method_summary(enriched))
    method_summary_path = phase4_artifact_path(args.output_dir, phase4_artifact_stem(args.artifact_prefix, "method_summary"), args.artifact_date, "csv")
    method_fields = sorted({key for row in method_rows for key in row})
    write_csv(method_summary_path, method_rows, method_fields)

    group_rows = group_mean_rows(enriched, ("dataset", "backbone", "label_budget_per_class", "method"))
    group_summary_path = phase4_artifact_path(args.output_dir, phase4_artifact_stem(args.artifact_prefix, "group_summary"), args.artifact_date, "csv")
    group_fields = sorted({key for row in group_rows for key in row})
    write_csv(group_summary_path, group_rows, group_fields)

    stats = dynamic_int8_stats(enriched, gate)
    strengthened_trigger = {
        "status": "TRIGGER_STRENGTHENED_REVIEW" if stats["severe_quality_cell_count"] else "NO_STRENGTHENED_MATRIX_TRIGGER",
        "reason": "severe dynamic_int8_cpu quality-drop cells beyond -0.08 floor"
        if stats["severe_quality_cell_count"]
        else "formal dynamic_int8_cpu cells stayed within the severe-collapse trigger rule",
        "dynamic_int8_cpu": stats,
    }
    stats_path = phase4_artifact_path(args.output_dir, phase4_artifact_stem(args.artifact_prefix, "statistics"), args.artifact_date, "json")
    stats_path.write_text(json.dumps({"dynamic_int8_cpu": stats, "gate": gate}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trigger_path = phase4_artifact_path(args.output_dir, phase4_artifact_stem(args.artifact_prefix, "strengthened_trigger"), args.artifact_date, "json")
    trigger_path.write_text(json.dumps(strengthened_trigger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    figures = write_figures(group_rows, method_rows, args.figure_dir, figure_prefix=figure_prefix)
    manifest = {
        "status": "PASS_PHASE4_ANALYSIS",
        "results": str(args.results),
        "summary": str(args.summary),
        "artifacts": {
            "paired_deltas_csv": str(paired_delta_path),
            "method_summary_csv": str(method_summary_path),
            "group_summary_csv": str(group_summary_path),
            "statistics_json": str(stats_path),
            "strengthened_trigger_json": str(trigger_path),
            "figures": figures,
        },
    }
    manifest_path = phase4_artifact_path(args.output_dir, phase4_artifact_stem(args.artifact_prefix, "manifest"), args.artifact_date, "json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
