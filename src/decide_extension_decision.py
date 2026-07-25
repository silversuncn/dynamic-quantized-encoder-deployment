#!/usr/bin/env python3
"""Close Strengthened-matrix extension decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def nested_float(payload: dict[str, Any], path: tuple[str, ...], default: float = 0.0) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def method_metric(summary: dict[str, Any], method: str, metric: str, default: float = 0.0) -> float:
    method_summary = summary.get("method_summary", {})
    if not isinstance(method_summary, dict):
        return default
    values = method_summary.get(method, {})
    if not isinstance(values, dict):
        return default
    try:
        return float(values.get(metric, default))
    except (TypeError, ValueError):
        return default


def dynamic_payload(stats: dict[str, Any]) -> dict[str, Any]:
    dynamic = stats.get("dynamic_int8_cpu", {})
    return dynamic if isinstance(dynamic, dict) else {}


def summarize_dynamic_int8(summary: dict[str, Any], stats: dict[str, Any], rows: int) -> dict[str, Any]:
    dynamic = dynamic_payload(stats)
    paired = int(dynamic.get("paired_cells") or 0)
    severe = int(dynamic.get("severe_quality_cell_count") or 0)
    return {
        "rows": rows,
        "summary_status": summary.get("status"),
        "paired_cells": paired,
        "severe_quality_cell_count": severe,
        "severe_quality_rate": severe / paired if paired else 0.0,
        "mean_accuracy_delta": nested_float(dynamic, ("accuracy_delta", "mean")),
        "mean_macro_f1_delta": nested_float(dynamic, ("macro_f1_delta", "mean")),
        "mean_best_deployment_improvement": nested_float(dynamic, ("best_deployment_improvement", "mean")),
    }


def summarize_gpu_methods(summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in ("fp32_gpu", "bf16_gpu"):
        out[method] = {
            "mean_accuracy_delta": method_metric(summary, method, "mean_accuracy_delta"),
            "mean_macro_f1_delta": method_metric(summary, method, "mean_macro_f1_delta"),
            "mean_best_deployment_improvement": method_metric(summary, method, "mean_best_deployment_improvement"),
            "quality_preserved": method_metric(summary, method, "mean_accuracy_delta") >= -0.01
            and method_metric(summary, method, "mean_macro_f1_delta") >= -0.01,
        }
    return out


def build_extension_decision(
    *,
    formal_summary: dict[str, Any],
    formal_stats: dict[str, Any],
    strengthened_summary: dict[str, Any],
    strengthened_stats: dict[str, Any],
    formal_rows: int,
    strengthened_rows: int,
) -> dict[str, Any]:
    formal_dynamic = summarize_dynamic_int8(formal_summary, formal_stats, formal_rows)
    strengthened_dynamic = summarize_dynamic_int8(strengthened_summary, strengthened_stats, strengthened_rows)
    strengthened_gpu = summarize_gpu_methods(strengthened_summary)
    severe_persists = (
        strengthened_dynamic["paired_cells"] > 0
        and strengthened_dynamic["severe_quality_rate"] >= 0.10
        and strengthened_dynamic["mean_best_deployment_improvement"] >= 0.25
        and (
            strengthened_dynamic["mean_accuracy_delta"] <= -0.03
            or strengthened_dynamic["mean_macro_f1_delta"] <= -0.05
        )
    )
    gpu_references_preserve_quality = all(method["quality_preserved"] for method in strengthened_gpu.values())

    if strengthened_rows <= 0 or strengthened_dynamic["paired_cells"] <= 0:
        status = "BLOCKED_NEEDS_MANAGER_SCIENCE_DECISION"
        reason = "strengthened evidence is missing or has no paired dynamic_int8_cpu cells"
    elif severe_persists and gpu_references_preserve_quality:
        status = "PASS_EXTENSION_DECISION_CLOSED_NEGATIVE_RESULT"
        reason = "dynamic INT8 efficiency gain persists with unstable quality in new high-risk seeds; GPU references preserve quality"
    elif gpu_references_preserve_quality:
        status = "PASS_EXTENSION_DECISION_CLOSED_POSITIVE_OR_MIXED_RESULT"
        reason = "strengthened evidence changed the dynamic INT8 risk interpretation but preserves a clear deployment tradeoff narrative"
    else:
        status = "BLOCKED_NEEDS_MANAGER_SCIENCE_DECISION"
        reason = "strengthened evidence does not yield a clear dynamic INT8/GPU deployment narrative"

    return {
        "status": status,
        "reason": reason,
        "formal_rows": formal_rows,
        "strengthened_rows": strengthened_rows,
        "formal_dynamic_int8": formal_dynamic,
        "strengthened_dynamic_int8": strengthened_dynamic,
        "strengthened_gpu_methods": strengthened_gpu,
        "dynamic_int8_quality_risk_persists": severe_persists,
        "gpu_references_preserve_quality": gpu_references_preserve_quality,
        "can_proceed_as_negative_measurement_paper": status == "PASS_EXTENSION_DECISION_CLOSED_NEGATIVE_RESULT",
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-summary", required=True, type=Path)
    parser.add_argument("--formal-statistics", required=True, type=Path)
    parser.add_argument("--strengthened-summary", required=True, type=Path)
    parser.add_argument("--strengthened-statistics", required=True, type=Path)
    parser.add_argument("--formal-rows", required=True, type=int)
    parser.add_argument("--strengthened-rows", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    decision = build_extension_decision(
        formal_summary=read_json(args.formal_summary),
        formal_stats=read_json(args.formal_statistics),
        strengthened_summary=read_json(args.strengthened_summary),
        strengthened_stats=read_json(args.strengthened_statistics),
        formal_rows=args.formal_rows,
        strengthened_rows=args.strengthened_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if str(decision["status"]).startswith("PASS_EXTENSION_DECISION_CLOSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
