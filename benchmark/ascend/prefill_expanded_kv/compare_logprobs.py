"""Compare teacher-forced probabilities without requiring identical generation."""

import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("/sgl-workspace/p-kv-opt-artifacts"))
    parser.add_argument("--baseline-variant", default="baseline")
    parser.add_argument("--candidate-variant", default="candidate")
    parser.add_argument("--output", default="logprob-comparison.json")
    args = parser.parse_args()
    baseline = json.loads((args.artifacts / f"{args.baseline_variant}-logprobs.json").read_text())
    candidate = json.loads((args.artifacts / f"{args.candidate_variant}-logprobs.json").read_text())
    assert baseline["source_rows"] == candidate["source_rows"]
    assert baseline.get("warm_full_context", False) == candidate.get("warm_full_context", False)
    assert len(baseline["cases"]) == len(candidate["cases"]) == 64
    base_values, opt_values = [], []
    for a, b in zip(baseline["cases"], candidate["cases"]):
        assert a["input_sha256"] == b["input_sha256"]
        assert a["prefix_tokens"] == b["prefix_tokens"] == 640
        assert a["logprob_start_len"] == b["logprob_start_len"]
        assert len(a["input_token_logprobs"]) == len(b["input_token_logprobs"])
        for x, y in zip(a["input_token_logprobs"], b["input_token_logprobs"]):
            assert x[1] == y[1]
            assert math.isfinite(x[0]) and math.isfinite(y[0])
            base_values.append(x[0])
            opt_values.append(y[0])
    differences = [b-a for a, b in zip(base_values, opt_values)]
    base_nll = -math.fsum(base_values) / len(base_values)
    opt_nll = -math.fsum(opt_values) / len(opt_values)
    result = dict(
        cases=64, scored_tokens=len(base_values), cached_prefix_tokens=640,
        warm_full_context=baseline.get("warm_full_context", False),
        baseline_mean_nll=base_nll, candidate_mean_nll=opt_nll,
        mean_nll_delta=opt_nll-base_nll,
        relative_perplexity_change_percent=100 * math.expm1(opt_nll-base_nll),
        mean_abs_logprob_delta=statistics.mean(abs(x) for x in differences),
        max_abs_logprob_delta=max(abs(x) for x in differences),
        identical_logprobs=sum(x == 0 for x in differences),
    )
    (args.artifacts / args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
