"""Run the original custom dataset as individual requests through 1P1D."""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("/sgl-workspace/p-kv-opt-artifacts"))
    parser.add_argument("--dataset", default="/mnt/sfs_turbo/hxd/CHAT-mxfp8-customer-2000.jsonl")
    parser.add_argument("--model", default="/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--router-port", type=int, default=19082)
    parser.add_argument("--prefill-port", type=int, default=23000)
    parser.add_argument("--server-run-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.artifacts / (
        f"{args.variant}-dataset-c{args.max_concurrency}-n{args.num_prompts}.jsonl"
    )
    metadata_path = output.with_suffix(".run.json")
    assert not output.exists(), f"Refusing to append to existing run: {output}"
    server = requests.get(f"http://127.0.0.1:{args.prefill_port}/get_server_info", timeout=10).json()
    assert server["max_prefill_tokens"] == server["chunked_prefill_size"] == 32768
    server_keys = ["max_prefill_tokens", "chunked_prefill_size", "max_running_requests",
                   "context_length", "mem_fraction_static", "disaggregation_mode", "max_total_num_tokens"]
    prefill_log = args.server_run_dir / "prefill_1.log"
    log_offset = prefill_log.stat().st_size
    command = [
        sys.executable, "-m", "sglang.bench_serving", "--backend", "sglang",
        "--base-url", f"http://127.0.0.1:{args.router_port}",
        "--model", args.model, "--tokenizer", args.model,
        "--dataset-name", "custom", "--dataset-path", args.dataset,
        "--num-prompts", str(args.num_prompts), "--seed", "42",
        "--request-rate", args.request_rate,
        "--max-concurrency", str(args.max_concurrency),
        "--extra-request-body", '{"chat_template_kwargs":{"enable_thinking":true}}',
        "--flush-cache", "--output-details", "--output-file", str(output),
    ]
    metadata = dict(variant=args.variant, command=command, started=time.time(),
                    num_prompts=args.num_prompts, max_concurrency=args.max_concurrency,
                    request_rate=args.request_rate,
                    dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                    server={key: server.get(key) for key in server_keys},
                    server_run_dir=str(args.server_run_dir), log_offset=log_offset)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    result = subprocess.run(command, env=os.environ.copy())
    metadata["returncode"] = result.returncode
    metadata["finished"] = time.time()
    with prefill_log.open("rb") as stream:
        stream.seek(log_offset)
        server_log = stream.read().decode(errors="replace")
    pattern = r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)"
    batches = [dict(sequences=int(a), new_tokens=int(b), cached_tokens=int(c))
               for a, b, c in re.findall(pattern, server_log)]
    metadata["batches"] = batches
    if batches:
        metadata["batch_summary"] = dict(
            count=len(batches), multi_request_batches=sum(b["sequences"] > 1 for b in batches),
            max_new_tokens=max(b["new_tokens"] for b in batches),
            mean_new_tokens=statistics.mean(b["new_tokens"] for b in batches),
            max_sequences=max(b["sequences"] for b in batches),
            total_new_tokens=sum(b["new_tokens"] for b in batches),
            total_cached_tokens=sum(b["cached_tokens"] for b in batches),
        )
    metadata_path.write_text(json.dumps(metadata, indent=2))
    result.check_returncode()
    assert batches and metadata["batch_summary"]["multi_request_batches"] > 0
    assert metadata["batch_summary"]["max_new_tokens"] <= 32768
    data = json.loads(output.read_text().splitlines()[-1])
    assert data["completed"] == args.num_prompts
    assert len(data["ttfts"]) == args.num_prompts
    assert all(value > 0 for value in data["ttfts"])
    assert not any(data["errors"])
    metadata["validation_passed"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata["batch_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
