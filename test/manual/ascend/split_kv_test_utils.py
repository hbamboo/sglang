"""Shared helpers for the split kv_b_proj NPU correctness/profile harnesses."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback

import torch

FRAMEWORK = "/mnt/sfs_turbo/hxd/split-kv-matmul-20261102/framework"
RESULTS_DIR = os.environ.get(
    "SPLIT_KV_RESULTS_DIR", "/mnt/sfs_turbo/hxd/split-kv-matmul-20261102/results"
)
RUN_TAG = os.environ.get("SPLIT_KV_RUN_TAG", "")


def assert_npu5_idle() -> str:
    """Fail fast unless physical NPU5 has no active process/utilization."""
    out = subprocess.run(
        "npu-smi info", shell=True, capture_output=True, text=True, timeout=30
    ).stdout
    lines = out.splitlines()
    idx = next(
        (i for i, line in enumerate(lines) if line.startswith("| 5 ")), None
    )
    if idx is None:
        raise RuntimeError(f"physical NPU5 row not found in npu-smi:\n{out}")
    board = "\n".join(lines[idx : idx + 2])
    ratio = re.search(
        r"\|\s*NA\s*\|\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)", board
    )
    if ratio is None:
        raise RuntimeError(f"cannot parse physical NPU5 npu-smi row:\n{out}")
    util, hbm_used, hbm_total = int(ratio.group(1)), int(ratio.group(4)), int(ratio.group(5))
    if util != 0 or hbm_used * 5 >= hbm_total:
        raise RuntimeError(
            f"physical NPU5 is not idle: util={util}% HBM={hbm_used}/{hbm_total} MB"
        )
    process_section = out.split("Process id", 1)
    if len(process_section) == 2:
        busy = [
            line
            for line in process_section[1].splitlines()
            if line.startswith("| 5 ")
            and str(os.getpid()) not in line  # our own process, if already mapped
        ]
        if busy:
            raise RuntimeError(f"physical NPU5 has active processes: {busy}")
    return board

SOURCE_FILES = [
    f"{FRAMEWORK}/python/sglang/srt/hardware_backend/npu/modules/split_kv_b_proj.py",
    f"{FRAMEWORK}/python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py",
    f"{FRAMEWORK}/python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py",
    f"{FRAMEWORK}/python/sglang/srt/model_executor/model_runner.py",
    f"{FRAMEWORK}/python/sglang/srt/environ.py",
]


def _sha256(path: str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _cmd(cmd: str) -> str:
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()


def collect_context() -> dict:
    npu_smi = _cmd("npu-smi info 2>/dev/null")
    physical_5 = "\n".join(
        line for line in npu_smi.splitlines() if "| 5 " in line or "| 5|" in line
    )
    return {
        "pid": os.getpid(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": os.environ.get("SPLIT_KV_COMMAND", " ".join(sys.argv)),
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "logical_device_count": torch.npu.device_count(),
        "logical_current_device": torch.npu.current_device(),
        "logical_device_name": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": getattr(__import__("torch_npu"), "__version__", None),
        "framework_git_head": _cmd(f"git -C {FRAMEWORK} rev-parse HEAD"),
        "framework_git_status": _cmd(f"git -C {FRAMEWORK} status --porcelain"),
        "source_sha256": {os.path.basename(p): _sha256(p) for p in SOURCE_FILES},
        "physical_5_npu_smi": physical_5,
    }


def record_result(name: str, status: str, metrics: dict, error: str = None) -> str:
    payload = {
        "name": name,
        "run_tag": RUN_TAG,
        "status": status,
        "exit_code": 0 if status == "PASS" else 1,
        "context": collect_context(),
        "metrics": metrics,
    }
    if error:
        payload["error"] = error
    out = pathlib.Path(RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}{RUN_TAG}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"RESULT_JSON {path}")
    print(json.dumps({"name": name, "status": status, "error": error}))
    return str(path)


def run_entrypoint(name: str, fn):
    metrics = {}
    try:
        fn(metrics)
    except Exception:
        err = traceback.format_exc()
        record_result(name, "FAIL", metrics, err)
        raise
    record_result(name, "PASS", metrics)


def close_stats(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict:
    a = actual.detach().float().cpu()
    e = expected.detach().float().cpu()
    if a.numel() == 0:
        return {
            "numel": 0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "max_rel": 0.0,
            "bad": 0,
            "finite": True,
        }
    diff = (a - e).abs()
    allowed = atol + rtol * e.abs()
    return {
        "numel": int(a.numel()),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float((diff / (e.abs() + 1e-6)).max()),
        "bad": int((diff > allowed).sum()),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(e).all()),
    }


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float):
    stats = close_stats(actual, expected, atol, rtol)
    if stats["numel"] and (stats["bad"] or not stats["finite"]):
        raise AssertionError(f"{name} mismatch {stats}")
    return stats


def print_stats(label: str, stats: dict):
    print(
        f"STATS {label} numel={stats['numel']} max_abs={stats['max_abs']:.6g} "
        f"mean_abs={stats['mean_abs']:.6g} max_rel={stats['max_rel']:.6g} "
        f"bad={stats['bad']} finite={stats['finite']}"
    )
