"""Sequential 32K streaming TTFT through a real 1P1D router.

Cache flushes and prefix priming are outside the timed requests. Both variants
read the same persisted token IDs. Eight output tokens exercise decode as well.
"""

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import requests


def prepare_prompt(args):
    path = Path(args.prompt_file)
    ids = json.loads(path.read_text()) if path.exists() else None
    if not (isinstance(ids, list) and len(ids) == 32768 and all(isinstance(t, int) for t in ids)):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        rows = [json.loads(line) for line in Path(args.dataset).open() if line.strip()]
        order = list(range(len(rows)))
        random.Random(42).shuffle(order)
        ids, sources = [], []
        for index in order:
            messages = []
            for turn_index, turn in enumerate(rows[index]["conversations"]):
                role = turn.get("role", turn.get("from"))
                if role is None:
                    role = "user" if turn_index % 2 == 0 else "assistant"
                role = {"human": "user", "gpt": "assistant"}.get(role, role)
                messages.append({"role": role, "content": turn.get("content", turn.get("value", ""))})
            if messages and messages[-1]["role"] == "assistant":
                messages.pop()
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=True
            )
            if hasattr(encoded, "keys"):
                encoded = encoded["input_ids"]
            ids.extend(encoded)
            sources.append(index)
            if len(ids) >= 32768:
                break
        ids = ids[:32768]
        assert len(ids) == 32768 and all(isinstance(t, int) for t in ids)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ids))
        path.with_suffix(".manifest.json").write_text(json.dumps(dict(
            model=args.model, dataset=args.dataset, source_rows=sources, seed=42,
            construction="Concatenated templated customer inputs, truncated to exactly 32768 tokens",
        ), indent=2))
    assert len(ids) == 32768
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    print(json.dumps(dict(prompt_tokens=len(ids), prompt_sha256=digest)), flush=True)
    return ids, digest


def flush(args):
    for port in [args.prefill_port, args.decode_port]:
        response = requests.post(f"http://127.0.0.1:{port}/flush_cache?timeout=20", timeout=30)
        response.raise_for_status()


def generate(args, ids, output_tokens=8):
    payload = dict(
        input_ids=ids, stream=True, return_logprob=True,
        sampling_params=dict(temperature=0, max_new_tokens=output_tokens, ignore_eos=True),
    )
    started = time.perf_counter()
    first = None
    last = None
    with requests.post(
        f"http://127.0.0.1:{args.router_port}/generate", json=payload,
        stream=True, timeout=300,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line or not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                continue
            last = json.loads(raw)
            if "error" in last:
                raise RuntimeError(last["error"])
            meta = last.get("meta_info", {})
            if first is None and meta.get("completion_tokens", 0) > 0:
                first = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    if first is None or last is None:
        raise RuntimeError("No generated token arrived in the response stream")
    meta = last["meta_info"]
    assert meta["prompt_tokens"] == len(ids), meta
    assert meta["completion_tokens"] == output_tokens, meta
    return dict(ttft_ms=first * 1000, e2e_ms=elapsed * 1000, meta=meta,
                output_ids=last.get("output_ids"), text=last.get("text"))


def run_case(args, ids, prefix, label, phase, repetition):
    flush(args)
    if prefix:
        generate(args, ids[:prefix], output_tokens=1)
    result = generate(args, ids)
    result.update(case=label, prefix_tokens=prefix, phase=phase, repetition=repetition)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-file", default="/sgl-workspace/p-kv-opt-artifacts/prompt-32k.json")
    parser.add_argument("--model", default="/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8")
    parser.add_argument("--dataset", default="/mnt/sfs_turbo/hxd/CHAT-mxfp8-customer-2000.jsonl")
    parser.add_argument("--router-port", type=int, default=19082)
    parser.add_argument("--prefill-port", type=int, default=23000)
    parser.add_argument("--decode-port", type=int, default=24000)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--profile-dir")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    ids, digest = prepare_prompt(args)
    if args.prepare_only:
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = dict(variant=args.variant, prompt_sha256=digest, prompt_tokens=32768,
                   output_tokens=8, cases={}, warmups=[], measurements=[])
    cases = [(0, "cold_32k"), (16384, "prefix16k_total32k")]
    for prefix, label in cases:
        for index in range(args.warmups):
            results["warmups"].append(run_case(args, ids, prefix, label, "warmup", index))
        for index in range(args.repetitions):
            result = run_case(args, ids, prefix, label, "measured", index)
            results["measurements"].append(result)
            output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        values = [r["ttft_ms"] for r in results["measurements"] if r["case"] == label]
        results["cases"][label] = dict(
            n=len(values), mean_ttft_ms=statistics.mean(values),
            median_ttft_ms=statistics.median(values), min_ttft_ms=min(values),
            max_ttft_ms=max(values), std_ttft_ms=statistics.stdev(values) if len(values)>1 else 0,
        )
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if args.profile_dir:
        flush(args)
        generate(args, ids[:16384], output_tokens=1)
        response = requests.post(
            f"http://127.0.0.1:{args.prefill_port}/start_profile",
            json=dict(output_dir=args.profile_dir, record_shapes=True, with_stack=False),
            timeout=60,
        )
        response.raise_for_status()
        try:
            results["profile_request"] = generate(args, ids)
        finally:
            requests.post(f"http://127.0.0.1:{args.prefill_port}/stop_profile", timeout=300).raise_for_status()
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results["cases"], indent=2), flush=True)


if __name__ == "__main__":
    main()
