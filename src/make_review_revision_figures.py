#!/usr/bin/env python3
"""Generate review-revision figures from archived Phase 4 summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from statistics import mean


DATASETS = ["cola", "mrpc", "qqp", "rte", "sst2"]
DATASET_LABELS = {"cola": "CoLA", "mrpc": "MRPC", "qqp": "QQP", "rte": "RTE", "sst2": "SST-2"}
BACKBONES = ["albert-base-v2", "bert-base-uncased", "distilbert-base-uncased"]
BACKBONE_LABELS = {
    "albert-base-v2": "ALBERT",
    "bert-base-uncased": "BERT",
    "distilbert-base-uncased": "DistilBERT",
}
BUDGETS = [64, 128, 256]
METHOD_ORDER = ["dynamic_int8_cpu", "fp32_gpu", "bf16_gpu"]
METHOD_LABELS = {"dynamic_int8_cpu": "INT8 CPU", "fp32_gpu": "GPU FP32", "bf16_gpu": "GPU BF16"}
METHOD_COLORS = {"dynamic_int8_cpu": "#3268a8", "fp32_gpu": "#2b8a6e", "bf16_gpu": "#c28b2c"}
INK = "#1f262e"
GRID = "#d9dee6"
RED = "#a64a4a"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig: object, out_dir: Path, stem: str) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.03)
    return {
        "pdf": {"path": str(pdf), "sha256": sha256(pdf), "size": pdf.stat().st_size},
        "png": {"path": str(png), "sha256": sha256(png), "size": png.stat().st_size},
    }


def configure() -> tuple[object, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
        }
    )
    return matplotlib, plt


def _mix(color: str, target: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    import matplotlib.colors as mcolors

    rgb = mcolors.to_rgb(color)
    return tuple(rgb[i] * (1.0 - t) + target[i] * t for i in range(3))


def plot_tradeoff(
    method_rows: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
    out_dir: Path,
    plt: object,
) -> dict[str, object]:
    import numpy as np
    from matplotlib.patches import Patch

    rows = {row["method"]: row for row in method_rows}
    methods = [method for method in METHOD_ORDER if method in rows]
    labels = [METHOD_LABELS[method] for method in methods]
    colors = [METHOD_COLORS[method] for method in methods]

    reductions = {method: {"size": [], "lat": [], "thru": []} for method in methods}
    for row in paired_rows:
        method = row["method"]
        if method not in reductions:
            continue
        reductions[method]["size"].append(float(row["model_size_reduction_vs_fp32_cpu"]))
        reductions[method]["lat"].append(float(row["latency_reduction_vs_fp32_cpu"]))
        reductions[method]["thru"].append(float(row["throughput_gain_vs_fp32_cpu"]))

    size_red = [mean(reductions[m]["size"]) for m in methods]
    lat_red = [mean(reductions[m]["lat"]) for m in methods]
    thru_gain = [mean(reductions[m]["thru"]) for m in methods]
    acc_deltas = [float(rows[m]["mean_accuracy_delta"]) for m in methods]

    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.4))

    # (a) Size and latency reduction (grouped bars, 0-1 fractions).
    width = 0.38
    for i, method in enumerate(methods):
        axes[0].bar(i - width / 2, size_red[i], width, color=_mix(colors[i], (0, 0, 0), 0.24))
        axes[0].bar(i + width / 2, lat_red[i], width, color=_mix(colors[i], (1, 1, 1), 0.45))
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Reduction (fraction)")
    axes[0].set_title("(a) Size & latency", fontsize=9, pad=4)
    axes[0].legend(
        handles=[
            Patch(facecolor="#4d4d4d", label="Size red."),
            Patch(facecolor="#b7b7b7", label="Latency red."),
        ],
        loc="upper left",
        frameon=False,
        handlelength=1.1,
        handleheight=1.1,
        borderpad=0.2,
        labelspacing=0.3,
    )

    # (b) Throughput gain (isolated high-magnitude axis).
    axes[1].bar(x, thru_gain, color=colors, width=0.62)
    axes[1].set_ylim(0, 35)
    axes[1].set_ylabel("Throughput gain ($\\times$)")
    axes[1].set_title("(b) Throughput", fontsize=9, pad=4)
    for i, value in enumerate(thru_gain):
        axes[1].text(i, value + 0.7, f"{value:.1f}", ha="center", va="bottom", fontsize=7, color=INK)

    # (c) Mean accuracy delta with quality gates.
    axes[2].bar(x, acc_deltas, color=colors, width=0.62)
    axes[2].axhline(-0.01, color="#5e6873", linestyle=":", linewidth=0.9)
    axes[2].axhline(-0.08, color=RED, linestyle="--", linewidth=0.9)
    axes[2].text(len(methods) - 0.5, -0.006, "non-inf.", ha="right", va="bottom", fontsize=7, color="#5e6873")
    axes[2].text(len(methods) - 0.5, -0.078, "severe", ha="right", va="bottom", fontsize=7, color=RED)
    axes[2].set_ylim(-0.095, 0.02)
    axes[2].set_ylabel("Mean accuracy delta")
    axes[2].set_title("(c) Quality", fontsize=9, pad=4)

    for ax in axes:
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(axis="y", color=GRID, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", pad=1.5)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    return save(fig, out_dir, "formal_method_tradeoff_summary")


def group_values(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], float]:
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        if row.get("method") != "dynamic_int8_cpu":
            continue
        key = (row["dataset"], row["backbone"], int(float(row["label_budget_per_class"])))
        grouped.setdefault(key, []).append(float(row["macro_f1_delta_vs_fp32_cpu"]))
    return {key: mean(vals) for key, vals in grouped.items()}


def plot_matrix_panel(ax: object, values: dict[tuple[str, str, int], float], backbones: list[str], title: str) -> object:
    import numpy as np

    row_keys = [(dataset, backbone) for backbone in backbones for dataset in DATASETS]
    matrix = np.array([[values.get((dataset, backbone, budget), math.nan) for budget in BUDGETS] for dataset, backbone in row_keys])
    image = ax.imshow(matrix, cmap="RdBu", vmin=-0.36, vmax=0.06, aspect="auto")
    labels = [f"{DATASET_LABELS[dataset]} / {BACKBONE_LABELS.get(backbone, backbone)}" for dataset, backbone in row_keys]
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(len(BUDGETS)), [str(budget) for budget in BUDGETS])
    ax.set_xlabel("Labels per class", labelpad=2)
    ax.set_title(title, fontsize=10, pad=5)
    ax.tick_params(length=0, pad=1.6)
    for i, (_, backbone) in enumerate(row_keys):
        if i > 0 and row_keys[i - 1][1] != backbone:
            ax.axhline(i - 0.5, color="white", linewidth=1.0)
    for row_i in range(matrix.shape[0]):
        for col_i in range(matrix.shape[1]):
            value = matrix[row_i, col_i]
            if math.isnan(value):
                continue
            ax.text(
                col_i,
                row_i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value < -0.26 else INK,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(BUDGETS))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, matrix.shape[0])], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    return image


def plot_combined_heatmap(formal_rows: list[dict[str, str]], strengthened_rows: list[dict[str, str]], out_dir: Path, plt: object) -> dict[str, object]:
    formal_values = group_values(formal_rows)
    strengthened_values = group_values(strengthened_rows)
    strengthened_backbones = ["albert-base-v2", "bert-base-uncased"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 4.7),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.62},
    )
    image = plot_matrix_panel(axes[0], formal_values, BACKBONES, "(a) Formal matrix")
    plot_matrix_panel(axes[1], strengthened_values, strengthened_backbones, "(b) Strengthened matrix")
    fig.subplots_adjust(left=0.12, right=0.87, top=0.93, bottom=0.09)
    cbar_ax = fig.add_axes([0.91, 0.18, 0.014, 0.66])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("Mean macro-F1 delta vs CPU FP32", labelpad=3)
    cbar.ax.tick_params(labelsize=8, length=2, pad=1.5)
    return save(fig, out_dir, "combined_dynamic_int8_macro_f1_delta")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-method-summary", required=True, type=Path)
    parser.add_argument("--formal-paired-deltas", required=True, type=Path)
    parser.add_argument("--strengthened-paired-deltas", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    _, plt = configure()
    formal_paired = read_csv(args.formal_paired_deltas)
    tradeoff = plot_tradeoff(read_csv(args.formal_method_summary), formal_paired, args.output_dir, plt)
    plt.close("all")
    heatmap = plot_combined_heatmap(
        formal_paired,
        read_csv(args.strengthened_paired_deltas),
        args.output_dir,
        plt,
    )
    plt.close("all")
    print({"status": "PASS_REVIEW_REVISION_FIGURES", "tradeoff": tradeoff, "heatmap": heatmap})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
