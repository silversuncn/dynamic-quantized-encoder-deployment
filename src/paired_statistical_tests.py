#!/usr/bin/env python3
"""Paired statistical tests for deployment deltas."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from scipy import stats


def load_method_rows(path: Path, method: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh) if row.get("method") == method and row.get("status") == "PASS"]


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> dict[str, float | int]:
    if n <= 0:
        raise ValueError("n must be positive")
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    proportion = successes / n
    denom = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denom
    half = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denom
    return {
        "successes": successes,
        "n": n,
        "proportion": proportion,
        "confidence": confidence,
        "ci_low": center - half,
        "ci_high": center + half,
    }


def paired_t_summary(rows: list[dict[str, str]], field: str, confidence: float = 0.95) -> dict[str, float | int]:
    vals = [float(row[field]) for row in rows if row.get(field) not in {"", None}]
    n = len(vals)
    if n < 2:
        raise ValueError(f"need at least two paired values for {field}")
    mean = sum(vals) / n
    sd = math.sqrt(sum((value - mean) ** 2 for value in vals) / (n - 1))
    se = sd / math.sqrt(n)
    t_stat = mean / se
    p_value = float(stats.t.sf(abs(t_stat), n - 1) * 2)
    crit = float(stats.t.ppf(1 - (1 - confidence) / 2, n - 1))
    return {
        "n": n,
        "mean_delta": mean,
        "sample_sd": sd,
        "standard_error": se,
        "t_statistic": t_stat,
        "degrees_of_freedom": n - 1,
        "p_value_two_sided": p_value,
        "confidence": confidence,
        "ci_low": mean - crit * se,
        "ci_high": mean + crit * se,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-deltas", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", default="dynamic_int8_cpu")
    parser.add_argument("--confidence", default=0.95, type=float)
    args = parser.parse_args()

    rows = load_method_rows(args.paired_deltas, args.method)
    noninferior = sum(
        float(row["accuracy_delta_vs_fp32_cpu"]) >= -0.01
        and float(row["macro_f1_delta_vs_fp32_cpu"]) >= -0.01
        for row in rows
    )
    severe = sum(
        float(row["accuracy_delta_vs_fp32_cpu"]) < -0.08
        or float(row["macro_f1_delta_vs_fp32_cpu"]) < -0.08
        for row in rows
    )
    payload = {
        "status": "PASS_PAIRED_STATISTICAL_TESTS",
        "source": str(args.paired_deltas),
        "method": args.method,
        "paired_cells": len(rows),
        "tests": {
            "accuracy_delta_vs_fp32_cpu": paired_t_summary(rows, "accuracy_delta_vs_fp32_cpu", args.confidence),
            "macro_f1_delta_vs_fp32_cpu": paired_t_summary(rows, "macro_f1_delta_vs_fp32_cpu", args.confidence),
        },
        "proportions": {
            "noninferior_cells": wilson_interval(noninferior, len(rows), args.confidence),
            "severe_quality_cells": wilson_interval(severe, len(rows), args.confidence),
        },
        "interpretation": (
            "Two-sided paired t-tests evaluate whether the mean deployment delta differs from zero. "
            "Wilson intervals summarize cell-count proportions; they do not replace the practical "
            "non-inferiority and severe-drop gates."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
