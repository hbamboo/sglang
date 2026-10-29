"""Count historical KV work in a multi-request Ascend profile."""

import argparse
import collections
import csv
import json
import sqlite3
from pathlib import Path


def audit(folder):
    paths = list(folder.rglob("kernel_details.csv"))
    assert len(paths) == 1, paths
    path = paths[0]
    assert (path.parent / "analyse.done").exists()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    database = next(path.parent.glob("ascend_pytorch_profiler_*.db"))
    db = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    strings = dict(db.execute("select id,value from STRING_IDS"))
    apis = [(int(a), int(b), tid, strings[name]) for a, b, tid, name in db.execute(
        "select startNs,endNs,globalTid,name from PYTORCH_API"
    )]
    steps = [name for _, _, _, name in apis if name.startswith("step[EXTEND ")]
    ranges = [(a, b, tid) for a, b, tid, name in apis if name == "npu_prefill_expanded_kv"]
    forbidden = collections.Counter()
    for a, b, tid, name in apis:
        if name in {"aten::cat", "aten::index", "aten::index_select"}:
            if any(start <= a and b <= end and tid == thread for start, end, thread in ranges):
                forbidden[name] += 1
    projections = [r for r in rows if "Matmul" in r["Name"] and "8960,512" in r["Input Shapes"]]
    gathers = [r for r in rows if "IndexSelect_Gather" in r["Name"] and (
        ",128,1,512" in r["Input Shapes"] or ",128,1,64" in r["Input Shapes"]
    )]
    cats = []
    for row in rows:
        shape = row["Output Shapes"].strip('"').split(",")
        if "aclnnCat_" in row["Name"] and len(shape) == 4 and shape[0] == "1" and shape[2:] == ["20", "256"]:
            cats.append(row)
    result = dict(
        profile=str(path.parent), prefill_steps=steps, prefill_step_count=len(steps),
        current_kv_b_projections=47 * len(steps),
        total_kv_b_projections=len(projections),
        historical_kv_b_projections=len(projections) - 47 * len(steps),
        latent_prefix_page_gathers=len(gathers), prefix_chunk_kv_cats=len(cats),
        expanded_attention_ranges=len(ranges), index_or_cat_in_new_core=dict(forbidden),
        attention_shapes=sorted({r["Input Shapes"] for r in rows if r["Name"] == "FusedInferAttentionScore"}),
    )
    assert steps
    if ranges:
        assert len(ranges) == 47 * len(steps)
        assert result["historical_kv_b_projections"] == 0
        assert not gathers and not cats and not forbidden
    else:
        assert result["historical_kv_b_projections"] > 0 and gathers and cats
    db.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
