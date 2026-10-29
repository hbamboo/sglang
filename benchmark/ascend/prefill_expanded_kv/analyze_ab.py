"""Audit TTFT samples and the actual Ascend kernels from the paired runs."""

import argparse
import collections
import csv
import json
import sqlite3
from pathlib import Path


def profile_summary(folder):
    files = list(folder.rglob("kernel_details.csv"))
    assert len(files) == 1, files
    csv_path = files[0]
    assert (csv_path.parent / "analyse.done").exists()
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    projections = [r for r in rows if "Matmul" in r["Name"] and "8960,512" in r["Input Shapes"]]
    gathers = [r for r in rows if "IndexSelect_Gather" in r["Name"] and (
        ",128,1,512" in r["Input Shapes"] or ",128,1,64" in r["Input Shapes"]
    )]
    cats = [r for r in rows if "aclnnCat_" in r["Name"] and r["Output Shapes"].strip('"') in (
        "1,32768,20,256", "32768,20,256"
    )]
    result = dict(
        csv=str(csv_path), kernels=len(rows),
        current_plus_history_kv_b_projections=len(projections),
        latent_prefix_page_gathers=len(gathers), prefix_chunk_kv_cats=len(cats),
        kv_b_device_us=sum(float(r["Duration(us)"]) for r in projections),
        prefix_gather_device_us=sum(float(r["Duration(us)"]) for r in gathers),
        prefix_cat_device_us=sum(float(r["Duration(us)"]) for r in cats),
        attention_shapes=sorted({r["Input Shapes"] for r in rows if r["Name"] == "FusedInferAttentionScore"}),
    )
    databases = list(csv_path.parent.glob("ascend_pytorch_profiler_*.db"))
    if databases:
        db = sqlite3.connect(databases[0].as_uri() + "?mode=ro", uri=True)
        strings = dict(db.execute("select id,value from STRING_IDS"))
        annotations = [(int(a), int(b), tid) for a, b, tid, name in db.execute(
            "select startNs,endNs,globalTid,name from PYTORCH_API"
        ) if strings[name] == "npu_prefill_expanded_kv"]
        nested = collections.Counter()
        if annotations:
            for a, b, tid, name in db.execute("select startNs,endNs,globalTid,name from PYTORCH_API"):
                if strings[name] in {"aten::cat", "aten::index", "aten::index_select"}:
                    if any(start <= int(a) and int(b) <= end and tid == thread for start, end, thread in annotations):
                        nested[strings[name]] += 1
        result["expanded_attention_annotations"] = len(annotations)
        result["index_or_cat_inside_expanded_attention"] = dict(nested)
        db.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("/sgl-workspace/p-kv-opt-artifacts"))
    args = parser.parse_args()
    root = args.artifacts
    baseline = json.loads((root / "baseline-ttft.json").read_text())
    candidate = json.loads((root / "candidate-ttft.json").read_text())
    assert baseline["prompt_sha256"] == candidate["prompt_sha256"]
    assert baseline["prompt_tokens"] == candidate["prompt_tokens"] == 32768
    cases, quality = {}, {}
    for name, base in baseline["cases"].items():
        opt = candidate["cases"][name]
        before, after = base["mean_ttft_ms"], opt["mean_ttft_ms"]
        cases[name] = dict(baseline=base, candidate=opt,
                           improvement_percent=100 * (before-after) / before,
                           saved_ms=before-after, speedup=before/after)
        a = [r for r in baseline["measurements"] if r["case"] == name]
        b = [r for r in candidate["measurements"] if r["case"] == name]
        assert len(a) == len(b) == 7
        deltas = []
        for source in [a, b]:
            for row in source:
                assert row["meta"]["cached_tokens"] == row["prefix_tokens"]
                assert row["meta"]["prompt_tokens"] == 32768
                assert row["meta"]["completion_tokens"] == 8
                assert row["meta"]["num_retractions"] == 0
        ids_match = all(x["output_ids"] == y["output_ids"] for x, y in zip(a, b))
        for x, y in zip(a, b):
            for p, q in zip(x["meta"]["output_token_logprobs"], y["meta"]["output_token_logprobs"]):
                if p[1] == q[1]:
                    deltas.append(abs(p[0]-q[0]))
        quality[name] = dict(output_ids_match=ids_match,
                            max_matching_token_logprob_delta=max(deltas, default=None))
    profiles = {name: profile_summary(root / "profiles" / name) for name in ["baseline", "candidate"]}
    assert profiles["baseline"]["current_plus_history_kv_b_projections"] == 94
    assert profiles["candidate"]["current_plus_history_kv_b_projections"] == 47
    assert profiles["baseline"]["latent_prefix_page_gathers"] == 94
    assert profiles["candidate"]["latent_prefix_page_gathers"] == 0
    assert profiles["baseline"]["prefix_chunk_kv_cats"] == 94
    assert profiles["candidate"]["prefix_chunk_kv_cats"] == 0
    assert profiles["candidate"]["expanded_attention_annotations"] == 47
    assert not profiles["candidate"]["index_or_cat_inside_expanded_attention"]
    result = dict(prompt_sha256=baseline["prompt_sha256"], cases=cases, quality=quality, profiles=profiles)
    (root / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
