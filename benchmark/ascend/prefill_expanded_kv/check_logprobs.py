"""Paired teacher-forced probability check with a real cached prefix."""

import argparse
import hashlib
import json
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("/sgl-workspace/p-kv-opt-artifacts"))
    parser.add_argument("--model", default="/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8")
    parser.add_argument("--dataset", default="/mnt/sfs_turbo/hxd/CHAT-mxfp8-customer-2000.jsonl")
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--warm-full-context", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [json.loads(line) for line in Path(args.dataset).open() if line.strip()]
    selected = [index * len(rows) // args.cases for index in range(args.cases)]
    output = args.artifacts / f"{args.variant}-logprobs.json"
    assert not output.exists(), output
    result = dict(cases=[], source_rows=selected, warm_full_context=args.warm_full_context)
    for index in selected:
        turns = rows[index]["conversations"]
        prompt = tokenizer.encode(turns[0].get("content", turns[0].get("value")), add_special_tokens=True)
        target = tokenizer.encode(turns[1].get("content", turns[1].get("value")), add_special_tokens=False)
        ids = prompt + target
        prefix = min(640, (len(prompt) // 128) * 128)
        for port in [23000, 24000]:
            requests.post(f"http://127.0.0.1:{port}/flush_cache?timeout=20", timeout=30).raise_for_status()
        warm_ids = ids[:prefix]
        if args.warm_full_context:
            warm_ids = ids.copy()
            warm_ids[prefix] = (warm_ids[prefix] + 1) % tokenizer.vocab_size
        warm = requests.post(
            "http://127.0.0.1:19082/generate",
            json=dict(input_ids=warm_ids, sampling_params=dict(temperature=0, max_new_tokens=1, ignore_eos=True)),
            timeout=120,
        )
        warm.raise_for_status()
        start = max(prefix, len(prompt)-256)
        response = requests.post(
            "http://127.0.0.1:19082/generate",
            json=dict(input_ids=ids, return_logprob=True, logprob_start_len=start,
                      sampling_params=dict(temperature=0, max_new_tokens=1, ignore_eos=True)),
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        meta = data["meta_info"]
        values = [item for item in meta.get("input_token_logprobs", []) if item[0] is not None]
        assert values, "PD response did not contain teacher-forced input logprobs"
        assert meta["cached_tokens"] == prefix, meta["cached_tokens"]
        case = dict(source_row=index, prefix_tokens=prefix, prompt_tokens=len(prompt),
                    warm_context_tokens=len(warm_ids),
                    total_tokens=len(ids), logprob_start_len=start,
                    input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    input_token_logprobs=values, output_ids=data.get("output_ids"))
        result["cases"].append(case)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(case=len(result["cases"]), source_row=index,
                              scored_tokens=len(values), cached_tokens=meta["cached_tokens"])), flush=True)


if __name__ == "__main__":
    main()
