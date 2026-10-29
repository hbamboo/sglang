"""Compare the paired original-dataset runs, keeping queueing/load explicit."""

import argparse
import json
import math
import statistics
from pathlib import Path


def load(root, variant, concurrency, prompts):
    path = root / f"{variant}-dataset-c{concurrency}-n{prompts}.jsonl"
    result = json.loads(path.read_text().splitlines()[-1])
    run = json.loads(path.with_suffix(".run.json").read_text())
    assert run["validation_passed"]
    assert result["completed"] == len(result["ttfts"]) == prompts
    assert not any(result["errors"])
    assert abs(statistics.mean(result["ttfts"]) * 1000 - result["mean_ttft_ms"]) < 0.001
    return result, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("/sgl-workspace/p-kv-opt-artifacts"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--baseline-variant", default="baseline")
    parser.add_argument("--candidate-variant", default="candidate")
    args = parser.parse_args()
    baseline, base_run = load(args.artifacts, args.baseline_variant, args.concurrency, args.num_prompts)
    candidate, opt_run = load(args.artifacts, args.candidate_variant, args.concurrency, args.num_prompts)
    assert baseline["input_lens"] == candidate["input_lens"]
    assert baseline["output_lens"] == candidate["output_lens"]
    assert base_run["dataset_sha256"] == opt_run["dataset_sha256"]
    assert baseline["total_input_tokens"] == candidate["total_input_tokens"]
    assert baseline["total_output_tokens"] == candidate["total_output_tokens"]
    assert baseline["max_concurrency"] == candidate["max_concurrency"] == args.concurrency
    keys = ["completed", "duration", "total_input_tokens", "total_output_tokens",
            "request_throughput", "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms",
            "p99_ttft_ms", "mean_e2e_latency_ms", "p99_e2e_latency_ms", "mean_tpot_ms"]
    before, after = baseline["mean_ttft_ms"], candidate["mean_ttft_ms"]
    result = dict(
        dataset_sha256=base_run["dataset_sha256"], concurrency=args.concurrency,
        num_prompts=args.num_prompts,
        baseline_variant=args.baseline_variant, candidate_variant=args.candidate_variant,
        request_rate=("inf" if math.isinf(baseline["request_rate"]) else baseline["request_rate"]),
        max_prefill_tokens=32768, chunked_prefill_size=32768,
        baseline={key: baseline.get(key) for key in keys},
        candidate={key: candidate.get(key) for key in keys},
        mean_ttft_saved_ms=before-after,
        mean_ttft_improvement_percent=100 * (before-after) / before,
        identical_generated_texts=sum(a == b for a, b in zip(
            baseline["generated_texts"], candidate["generated_texts"])),
        baseline_batch_summary=base_run["batch_summary"],
        candidate_batch_summary=opt_run["batch_summary"],
        baseline_cache_tokens=base_run["server"]["max_total_num_tokens"],
        candidate_cache_tokens=opt_run["server"]["max_total_num_tokens"],
    )
    output = args.artifacts / f"dataset-comparison-c{args.concurrency}-n{args.num_prompts}.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
