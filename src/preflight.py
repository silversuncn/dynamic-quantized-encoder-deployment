#!/usr/bin/env python3
"""Readiness checks for Dynamic quantized encoder deployment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import tempfile
from pathlib import Path

from dynamic_quant_core import (
    compute_reference_deltas,
    gate_dynamic_quantization,
    inference_method_configs,
    stratified_train_val_indices,
)


def check_runtime(torch) -> dict[str, object]:
    info: dict[str, object] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "bf16_supported": False,
        "dynamic_quant_linear_supported": False,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["bf16_supported"] = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    try:
        quantize_dynamic = getattr(torch.ao.quantization, "quantize_dynamic")
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()
        qmodel = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        x = torch.randn(4, 8)
        _ = qmodel(x)
        info["dynamic_quant_linear_supported"] = True
    except Exception as exc:  # noqa: BLE001 - readiness evidence must capture exact failure.
        info["dynamic_quant_error"] = f"{type(exc).__name__}: {exc}"
    return info


def check_offline_assets(model_name: str) -> dict[str, object]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out: dict[str, object] = {"model": model_name, "datasets": {}}
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        local_files_only=True,
    )
    out["model_load"] = "PASS"
    out["tokenizer_vocab_size"] = int(getattr(tokenizer, "vocab_size", 0))
    out["model_num_parameters"] = int(sum(p.numel() for p in model.parameters()))

    for task in ["sst2", "mrpc"]:
        dataset = load_dataset("glue", task)
        train = dataset["train"]
        validation = dataset["validation"]
        split = stratified_train_val_indices(
            [int(train[i]["label"]) for i in range(len(train))],
            train_per_class=4,
            val_per_class=3,
            seed=7,
        )
        out["datasets"][task] = {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "split_train_count": len(split["train_indices"]),
            "split_validation_count": len(split["validation_indices"]),
            "split_overlap_count": len(set(split["train_indices"]) & set(split["validation_indices"])),
            "train_class_counts": split["train_class_counts"],
            "validation_class_counts": split["validation_class_counts"],
        }
    return out


def check_aggregation_schema() -> dict[str, object]:
    rows = [
        {
            "dataset": "synthetic",
            "backbone": "distilbert-base-uncased",
            "label_budget_per_class": 4,
            "seed": 1,
            "method": "fp32_cpu",
            "status": "PASS",
            "accuracy": 0.80,
            "macro_f1": 0.79,
            "model_size_mb": 250.0,
            "latency_ms_per_example": 3.0,
            "throughput_examples_per_second": 333.3,
            "peak_memory_mb": 900.0,
        },
        {
            "dataset": "synthetic",
            "backbone": "distilbert-base-uncased",
            "label_budget_per_class": 4,
            "seed": 1,
            "method": "dynamic_int8_cpu",
            "status": "PASS",
            "accuracy": 0.795,
            "macro_f1": 0.785,
            "model_size_mb": 120.0,
            "latency_ms_per_example": 2.2,
            "throughput_examples_per_second": 454.5,
            "peak_memory_mb": 850.0,
        },
    ]
    enriched = compute_reference_deltas(rows)
    gate = gate_dynamic_quantization(enriched, margin=-0.01, min_deployment_improvement=0.25)
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "synthetic.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted(enriched[0].keys()))
            writer.writeheader()
            writer.writerows(enriched)
        csv_ok = csv_path.exists() and csv_path.stat().st_size > 0
    return {"synthetic_csv_ok": csv_ok, "synthetic_gate_status": gate["status"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="distilbert-base-uncased")
    args = parser.parse_args()

    import torch

    report: dict[str, object] = {
        "status": "RUNNING",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": args.model,
    }
    checks: dict[str, object] = {}
    try:
        checks["runtime"] = check_runtime(torch)
        checks["offline_assets"] = check_offline_assets(args.model)
        checks["method_configs"] = inference_method_configs(
            gpu_available=bool(checks["runtime"].get("cuda_available")),
            bf16_supported=bool(checks["runtime"].get("bf16_supported")),
        )
        checks["aggregation_schema"] = check_aggregation_schema()
        failures = []
        if not checks["runtime"].get("dynamic_quant_linear_supported"):
            failures.append("dynamic_quant_linear_unavailable")
        for task, info in checks["offline_assets"]["datasets"].items():
            if info["split_overlap_count"] != 0:
                failures.append(f"{task}_split_overlap")
        if checks["aggregation_schema"]["synthetic_gate_status"] != "PASS_MICRO":
            failures.append("aggregation_schema_gate_failed")
        report.update(
            {
                "status": "PASS_READINESS" if not failures else "FAIL_READINESS",
                "hard_failures": failures,
                "checks": checks,
            }
        )
    except Exception as exc:  # noqa: BLE001 - compact evidence records blocker cause.
        report.update({"status": "FAIL_READINESS", "error": f"{type(exc).__name__}: {exc}", "checks": checks})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(out)}, sort_keys=True))
    return 0 if report["status"] == "PASS_READINESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
