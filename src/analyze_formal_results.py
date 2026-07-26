#!/usr/bin/env python3
"""Produce formal matrix summaries and publication-quality figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

COLORS = {
    "blue": "#2f63a4",
    "teal": "#278b7e",
    "gold": "#be8723",
    "red": "#a64a4a",
    "orange": "#c7782b",
    "gray": "#64707c",
    "grid": "#d8dee6",
    "ink": "#1f262e",
}

FIGURE_SPECS = {
    "method_tradeoff_summary": {"size": (3.45, 2.65), "placement": "one_column"},
    "dynamic_int8_macro_f1_delta": {"size": (3.45, 4.25), "placement": "one_column"},
}

FONT_POLICY = {
    "axis_label_pt": 8,
    "tick_label_pt": 7,
    "legend_pt": 7,
    "annotation_pt": 7,
    "facet_label_pt": 8,
}

FIGURE_OUTPUT_POLICY = {
    "png_dpi": 600,
    "pdf_fonttype": 42,
    "ps_fonttype": 42,
    "bbox_inches": "tight",
    "pad_inches": 0.025,
}

DATASET_LABELS = {
    "cola": "CoLA",
    "mrpc": "MRPC",
    "qqp": "QQP",
    "rte": "RTE",
    "sst2": "SST-2",
}
DATASET_ORDER = ["cola", "mrpc", "qqp", "rte", "sst2"]

BACKBONE_LABELS = {
    "albert-base-v2": "ALBERT",
    "bert-base-uncased": "BERT",
    "distilbert-base-uncased": "DistilBERT",
}
BACKBONE_ORDER = ["albert-base-v2", "bert-base-uncased", "distilbert-base-uncased"]

BUDGET_ORDER = [64, 128, 256]

METHOD_LABELS = {
    "dynamic_int8_cpu": "INT8 CPU",
    "fp32_gpu": "GPU FP32",
    "bf16_gpu": "GPU BF16",
}
METHOD_COLORS = {
    "dynamic_int8_cpu": COLORS["blue"],
    "fp32_gpu": COLORS["teal"],
    "bf16_gpu": COLORS["gold"],
}


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
        out: dict[str, object] = dict(zip(keys, key_values, strict=True))
        out["n"] = len(group)
        for field in METRIC_FIELDS:
            values = [value for value in (finite_float(row.get(field)) for row in group) if value is not None]
            if values:
                out[f"mean_{field}"] = mean(values)
        out_rows.append(out)
    return out_rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def configure_matplotlib(matplotlib: object, plt: object) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_POLICY["axis_label_pt"],
            "axes.labelsize": FONT_POLICY["axis_label_pt"],
            "xtick.labelsize": FONT_POLICY["tick_label_pt"],
            "ytick.labelsize": FONT_POLICY["tick_label_pt"],
            "legend.fontsize": FONT_POLICY["legend_pt"],
            "pdf.fonttype": FIGURE_OUTPUT_POLICY["pdf_fonttype"],
            "ps.fonttype": FIGURE_OUTPUT_POLICY["ps_fonttype"],
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
        }
    )
    plt.rcParams.update(matplotlib.rcParams)


def save_figure(fig: object, figure_dir: Path, stem: str) -> dict[str, object]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / f"{stem}.pdf"
    png_path = figure_dir / f"{stem}.png"
    save_kwargs = {
        "bbox_inches": FIGURE_OUTPUT_POLICY["bbox_inches"],
        "pad_inches": FIGURE_OUTPUT_POLICY["pad_inches"],
    }
    fig.savefig(pdf_path, **save_kwargs)
    fig.savefig(png_path, dpi=FIGURE_OUTPUT_POLICY["png_dpi"], **save_kwargs)
    return {
        "pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path), "size": pdf_path.stat().st_size},
        "png": {"path": str(png_path), "sha256": sha256(png_path), "size": png_path.stat().st_size},
    }


def present_backbones(rows: list[dict[str, object]]) -> list[str]:
    seen = {str(row.get("backbone")) for row in rows}
    ordered = [backbone for backbone in BACKBONE_ORDER if backbone in seen]
    extras = sorted(seen.difference(ordered))
    return ordered + extras


def present_datasets(rows: list[dict[str, object]]) -> list[str]:
    seen = {str(row.get("dataset")) for row in rows}
    ordered = [dataset for dataset in DATASET_ORDER if dataset in seen]
    extras = sorted(seen.difference(ordered))
    return ordered + extras


def present_budgets(rows: list[dict[str, object]]) -> list[int]:
    budgets = {int(float(row["label_budget_per_class"])) for row in rows if row.get("label_budget_per_class") not in {"", None}}
    ordered = [budget for budget in BUDGET_ORDER if budget in budgets]
    extras = sorted(budgets.difference(ordered))
    return ordered + extras


def plot_dynamic_delta_heatmap(dyn_groups: list[dict[str, object]], figure_dir: Path, figure_prefix: str, plt: object) -> dict[str, object]:
    from matplotlib.colors import LinearSegmentedColormap

    backbones = present_backbones(dyn_groups)
    datasets = present_datasets(dyn_groups)
    budgets = present_budgets(dyn_groups)
    values = {
        (str(row["dataset"]), str(row["backbone"]), int(float(row["label_budget_per_class"]))): float(
            row.get("mean_macro_f1_delta_vs_fp32_cpu", 0.0)
        )
        for row in dyn_groups
    }
    height = max(FIGURE_SPECS["dynamic_int8_macro_f1_delta"]["size"][1], 1.25 * len(backbones) + 0.9)
    fig, axes = plt.subplots(
        len(backbones),
        1,
        figsize=(FIGURE_SPECS["dynamic_int8_macro_f1_delta"]["size"][0], height),
        squeeze=False,
    )
    cmap = LinearSegmentedColormap.from_list("dynamic_int8_delta", (COLORS["red"], "#f7f7f7", COLORS["teal"]))
    image = None
    for row_idx, backbone in enumerate(backbones):
        ax = axes[row_idx][0]
        matrix = [
            [values.get((dataset, backbone, budget), float("nan")) for budget in budgets]
            for dataset in datasets
        ]
        image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-0.25, vmax=0.05)
        ax.set_xticks(range(len(budgets)), [str(budget) for budget in budgets])
        ax.set_yticks(range(len(datasets)), [DATASET_LABELS.get(dataset, dataset) for dataset in datasets])
        ax.tick_params(length=0, pad=1.5)
        ax.set_ylabel(BACKBONE_LABELS.get(backbone, backbone), fontsize=FONT_POLICY["facet_label_pt"], labelpad=5)
        for dataset_idx, dataset in enumerate(datasets):
            for budget_idx, budget in enumerate(budgets):
                value = values.get((dataset, backbone, budget))
                if value is None:
                    continue
                text_color = "white" if value < -0.16 else COLORS["ink"]
                ax.text(
                    budget_idx,
                    dataset_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=FONT_POLICY["annotation_pt"],
                    color=text_color,
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([x - 0.5 for x in range(1, len(budgets))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(datasets))], minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
    axes[-1][0].set_xlabel("Labels per class", labelpad=2)
    if image is not None:
        cbar = fig.colorbar(image, ax=[ax[0] for ax in axes], fraction=0.035, pad=0.025)
        cbar.set_label("Mean macro-F1 delta", fontsize=FONT_POLICY["tick_label_pt"], labelpad=2)
        cbar.ax.tick_params(labelsize=FONT_POLICY["tick_label_pt"], length=2, pad=1.5)
    fig.subplots_adjust(left=0.22, right=0.86, top=0.98, bottom=0.09, hspace=0.18)
    return save_figure(fig, figure_dir, f"{figure_prefix}_dynamic_int8_macro_f1_delta")


def plot_method_tradeoff(method_rows: list[dict[str, object]], figure_dir: Path, figure_prefix: str, plt: object) -> dict[str, object]:
    methods = [
        row
        for row in method_rows
        if row.get("method") != "fp32_cpu" and row.get("mean_best_deployment_improvement") not in {"", None}
    ]
    methods.sort(key=lambda row: ["dynamic_int8_cpu", "fp32_gpu", "bf16_gpu"].index(row["method"]) if row["method"] in METHOD_LABELS else 99)
    labels = [METHOD_LABELS.get(str(row["method"]), str(row["method"])) for row in methods]
    improvements = [float(row["mean_best_deployment_improvement"]) for row in methods]
    acc = [float(row.get("mean_accuracy_delta", 0.0)) for row in methods]
    colors = [METHOD_COLORS.get(str(row["method"]), COLORS["blue"]) for row in methods]
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SPECS["method_tradeoff_summary"]["size"])
    axes[0].bar(range(len(methods)), improvements, color=colors, width=0.62)
    axes[0].axhline(0.25, color=COLORS["gray"], linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Best deploy. impr.", labelpad=2)
    axes[0].set_xticks(range(len(methods)), labels, rotation=25, ha="right")
    axes[0].grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    axes[0].set_axisbelow(True)
    axes[1].bar(range(len(methods)), acc, color=colors, width=0.62)
    axes[1].axhline(-0.08, color=COLORS["red"], linestyle="--", linewidth=0.8)
    axes[1].axhline(-0.01, color=COLORS["gray"], linestyle=":", linewidth=0.8)
    axes[1].set_ylabel("Mean Acc. delta", labelpad=2)
    axes[1].set_xticks(range(len(methods)), labels, rotation=25, ha="right")
    axes[1].grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    axes[1].set_axisbelow(True)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", pad=1.5)
    fig.tight_layout(pad=0.05)
    return save_figure(fig, figure_dir, f"{figure_prefix}_method_tradeoff_summary")


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

    configure_matplotlib(matplotlib, plt)
    generated: dict[str, object] = {}
    dyn_groups = [row for row in group_rows if row.get("method") == "dynamic_int8_cpu"]
    if dyn_groups:
        stem = f"{figure_prefix}_dynamic_int8_macro_f1_delta"
        generated[stem] = plot_dynamic_delta_heatmap(dyn_groups, figure_dir, figure_prefix, plt)
        plt.close("all")

    methods = [
        row
        for row in method_rows
        if row.get("method") != "fp32_cpu" and row.get("mean_best_deployment_improvement") not in {"", None}
    ]
    if methods:
        stem = f"{figure_prefix}_method_tradeoff_summary"
        generated[stem] = plot_method_tradeoff(method_rows, figure_dir, figure_prefix, plt)
        plt.close("all")

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
