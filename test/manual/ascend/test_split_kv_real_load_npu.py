"""Focused real-checkpoint load -> final process -> actual prepare route test.

Physical NPU5 only.  Uses the actual target checkpoint
/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8:

  * real Glm4MoeLiteConfig + ModelSlimConfig;
  * real Glm4MoeLiteDecoderLayer(0) and DeepseekV2AttentionMLA;
  * real kv_b_proj safetensors tensor and ColumnParallelLinear.weight_loader;
  * the real Glm4MoeLiteForCausalLM.load_weights entry, which calls
    DeepseekV2WeightLoaderMixin.post_load_weights at its end;
  * the loader's final quant_method.process_weights_after_loading entry on the
    target module;
  * an actual forward_mha_prepare_npu call with a 4D [T,1,1,512] kv_a from a
    fused norm/cache fixture, recording route calls/hits and full old/new
    outputs plus CPU FP32 golden.

Scope: one decoder layer's kv_b_proj.  The q/kv-latent path is a deterministic
fixture because the narrow test does not load the quantized q/kv projections;
the real kv_b_proj module, checkpoint weight, loader, post-load and final
process are used.  A full 47-layer model smoke belongs to root.
"""

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import types

FRAMEWORK = "/mnt/sfs_turbo/hxd/split-kv-matmul-20261102/framework"
MODEL_PATH = "/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8"
sys.path.insert(0, f"{FRAMEWORK}/python")
sys.path.insert(0, f"{FRAMEWORK}/test/manual/ascend")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "5")
os.environ["SGLANG_NPU_SPLIT_KV_B_PROJ"] = "1"

import torch
import torch_npu

import split_kv_test_utils as tu

tu.assert_npu5_idle()
torch.npu.set_device(0)

from safetensors import safe_open

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.hardware_backend.npu.modules import (
    deepseek_v2_attention_mla_npu as attn_mod,
)
from sglang.srt.hardware_backend.npu.modules import split_kv_b_proj as skv
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.model_loader.loader import device_loading_context
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.glm4_moe_lite import (
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteForCausalLM,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

import test_split_kv_b_proj_npu as T

DEV = "npu:0"
KEY = "model.layers.0.self_attn.kv_b_proj.weight"
PREFIX = "model.layers.0.self_attn.kv_b_proj"
TOKENS = 129
DK, DV, DR = 192, 256, 64


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def tensor_sha256(t: torch.Tensor):
    return sha256_bytes(t.contiguous().view(torch.int16).cpu().numpy().tobytes())


def source_slice(rel, a, b, max_chars=1800):
    p = pathlib.Path(FRAMEWORK) / rel
    lines = p.read_text(encoding="utf-8").splitlines()
    text = "\n".join(f"{i}: {lines[i - 1]}" for i in range(a, b + 1))
    return {
        "file": str(p),
        "sha256": sha256_bytes(p.read_bytes()),
        "lines": [a, b],
        "text": text[:max_chars],
    }


def meta(t: torch.nn.Parameter):
    return {
        "ptr": t.data_ptr(),
        "version": t._version,
        "dtype": str(t.dtype),
        "device": str(t.device),
        "shape": list(t.shape),
    }


def _grep(pattern, paths):
    cmd = "grep -rn '" + pattern + "' " + " ".join(paths) + " 2>/dev/null"
    p = subprocess.run(cmd, shell=True, cwd=FRAMEWORK, capture_output=True, text=True)
    return p.stdout.strip()


def _format_cast_call_sites():
    raw = _grep("npu_format_cast(", ["python/sglang/srt"])
    out = []
    for line in raw.splitlines():
        path, lineno, _ = line.split(":", 2)
        out.append(f"{path.removeprefix('python/sglang/srt/')}:{lineno}")
    return out


def build_real_layer():
    init_distributed_environment(
        backend="hccl",
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29641",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    monkey_patch_vllm_parallel_state()
    server_args = ServerArgs(
        model_path=MODEL_PATH,
        quantization="modelslim",
        dtype="bfloat16",
        tp_size=1,
        trust_remote_code=True,
    )
    set_global_server_args_for_scheduler(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    cfg = model_config.hf_config
    quant_description = json.load(
        open(os.path.join(MODEL_PATH, "quant_model_description.json"))
    )
    quant_config = ModelSlimConfig(quant_description)
    with set_default_torch_dtype(torch.bfloat16):
        with torch.device(DEV):
            layer = Glm4MoeLiteDecoderLayer(
                cfg, 0, quant_config, prefix="model.layers.0"
            )
    return cfg, quant_config, quant_description, layer


def read_checkpoint():
    index_path = os.path.join(MODEL_PATH, "quant_model_weights.safetensors.index.json")
    index = json.load(open(index_path))
    weight_map = index["weight_map"]
    float_keys = [
        k
        for k, v in json.load(
            open(os.path.join(MODEL_PATH, "quant_model_description.json"))
        ).items()
        if k.startswith("model.layers.")
        and k.endswith(".self_attn.kv_b_proj.weight")
        and v == "FLOAT"
    ]
    assert len(float_keys) == 47, len(float_keys)
    file_name = weight_map[KEY]
    file_path = os.path.join(MODEL_PATH, file_name)
    file_sha = sha256_bytes(pathlib.Path(file_path).read_bytes())
    with safe_open(file_path, framework="pt", device="cpu") as f:
        weight = f.get_tensor(KEY)
    assert tuple(weight.shape) == (8960, 512), weight.shape
    assert weight.dtype == torch.bfloat16, weight.dtype
    return {
        "model_path": MODEL_PATH,
        "index_path": index_path,
        "index_sha256": sha256_bytes(pathlib.Path(index_path).read_bytes()),
        "file": file_name,
        "file_sha256": file_sha,
        "key": KEY,
        "shape": list(weight.shape),
        "dtype": str(weight.dtype),
        "tensor_sha256": tensor_sha256(weight),
        "float_kv_b_key_count": len(float_keys),
        "all_float_keys": float_keys,
        "weight": weight,
    }


def real_load_and_final_process(cfg, quant_config, layer, ckpt):
    attn = layer.self_attn
    kv = attn.kv_b_proj
    weight = ckpt["weight"]
    params = {KEY: kv.weight}

    class Shell(DeepseekV2WeightLoaderMixin):
        pass

    shell = Shell()
    shell.model = types.SimpleNamespace(layers=[layer], start_layer=0, end_layer=1)
    shell.config = cfg
    shell.quant_config = quant_config
    shell.num_fused_shared_experts = 0

    before_load = meta(kv.weight)
    Glm4MoeLiteForCausalLM.load_weights(
        shell, [(KEY, weight)], params_dict=params
    )
    after_load = meta(kv.weight)
    state_after_load = skv.split_kv_b_proj_state(kv)
    assert state_after_load is not None
    assert torch.equal(kv.weight.detach().cpu(), weight)
    exp_k, exp_v, _, _ = skv.split_kv_b_proj_weight(weight, None, 20, DK, DV)
    assert tu.close_stats(state_after_load.w_k, exp_k, 0.0, 0.0)["bad"] == 0
    assert tu.close_stats(state_after_load.w_v, exp_v, 0.0, 0.0)["bad"] == 0

    # This is the exact final entry the loader invokes on every module after
    # model.load_weights (DefaultModelLoader.load_weights_and_postprocess).
    import sglang.srt.hardware_backend.npu.utils as npu_utils
    import sglang.srt.layers.quantization.unquant as unquant

    cast_calls = {"n": 0}
    orig_cast = npu_utils.npu_format_cast

    def counted_cast(*args, **kwargs):
        cast_calls["n"] += 1
        return orig_cast(*args, **kwargs)

    before_final = meta(kv.weight)
    npu_utils.npu_format_cast = counted_cast
    try:
        with device_loading_context(kv, torch.device(DEV)):
            kv.quant_method.process_weights_after_loading(kv)
    finally:
        npu_utils.npu_format_cast = orig_cast
    after_final = meta(kv.weight)
    state_after_final = skv.split_kv_b_proj_state(kv)
    refreshed = skv.finalize_split_kv_b_proj(layer)
    assert refreshed == 0
    assert skv._state_is_current(state_after_final, kv)
    assert state_after_final is state_after_load
    assert skv._state_is_current(state_after_final, kv)
    assert before_final == after_final
    assert cast_calls["n"] == 0
    assert torch.equal(kv.weight.detach().cpu(), weight)

    return {
        "module_class": type(kv).__name__,
        "quant_method_class": type(kv.quant_method).__name__,
        "quant_description": "FLOAT",
        "modelslim_selected": type(
            quant_config.get_quant_method(kv, PREFIX)
        ).__name__,
        "platform": {
            "unquant_is_cpu": bool(unquant._is_cpu),
            "unquant_is_npu": bool(unquant._is_npu),
            "is_cpu_amx_available": bool(unquant._is_cpu_amx_available),
        },
        "before_load": before_load,
        "after_load": after_load,
        "after_final_process": after_final,
        "pointer_unchanged_by_final_process": before_final == after_final,
        "npu_format_cast_calls_during_final_process": cast_calls["n"],
        "state_present_after_load": True,
        "state_present_after_final_process": True,
        "state_current_after_final_process": True,
        "split_matches_checkpoint": True,
        "finalizer_refreshed_layers": refreshed,
        "state_current_after_finalizer": True,
    }


def install_prepare_fixture(attn):
    attn_mod.get_token_to_kv_pool = lambda: T.DummyPool()
    attn_mod.can_fuse_expanded_kv = lambda *args, **kwargs: False
    attn_mod.is_fia_nz = lambda: False
    import sglang.srt.model_executor.forward_context as fctx

    fctx.get_token_to_kv_pool = lambda: T.DummyPool()

    attn.q_lora_rank = None
    attn.q_proj = ColumnParallelLinear(
        2048, 20 * (DK + DR), bias=False, tp_rank=0, tp_size=1,
        params_dtype=torch.bfloat16,
    ).to(DEV)
    attn.kv_a_proj_with_mqa = ColumnParallelLinear(
        2048, 512 + DR, bias=False, tp_rank=0, tp_size=1,
        params_dtype=torch.bfloat16,
    ).to(DEV)
    T.copy_weights(attn.q_proj, 710, scale=0.05)
    T.copy_weights(attn.kv_a_proj_with_mqa, 720, scale=0.1)
    attn.use_deepseek_yarn_rope = True
    attn.rotary_emb = T.YarnRotary(DR, torch.bfloat16)

    fixture = T.FusedNormCacheFixture(512, DR)
    orig_interleave = torch_npu.npu_interleave_rope
    orig_kv_op = torch_npu.npu_kv_rmsnorm_rope_cache
    torch_npu.npu_interleave_rope = fixture.npu_interleave_rope
    torch_npu.npu_kv_rmsnorm_rope_cache = fixture.npu_kv_rmsnorm_rope_cache

    recorder = {}
    orig_concat = attn._concat_and_cast_mha_k

    def concat_record(k_nope, k_pe, forward_batch):
        recorder["k_nope_stride"] = tuple(k_nope.stride())
        recorder["k_nope_contiguous"] = bool(k_nope.is_contiguous())
        recorder["k_nope"] = k_nope.detach().clone()
        recorder["k_pe"] = k_pe.detach().clone()
        return orig_concat(k_nope, k_pe, forward_batch)

    attn._concat_and_cast_mha_k = concat_record

    x = T.npu_random((TOKENS, 2048), torch.bfloat16, 730)
    positions = torch.arange(TOKENS, dtype=torch.float32, device=DEV)
    fb = T.FakeForwardBatch(TOKENS)
    # Precompute the fused-op output once, outside timed/profiled regions.
    latent = attn.kv_a_proj_with_mqa(x)[0]
    fixture.set_static_from_latent(latent.unsqueeze(1).unsqueeze(1))
    return {
        "x": x,
        "positions": positions,
        "fb": fb,
        "fixture": fixture,
        "recorder": recorder,
        "restore": (orig_interleave, orig_kv_op, orig_concat),
    }


def run_prepare(attn, fx):
    return attn_mod.forward_mha_prepare_npu(
        attn, fx["positions"], fx["x"], fx["fb"], None, None
    )


def route_case(attn, fx, ckpt):
    kv = attn.kv_b_proj
    counts = {"calls": 0, "hits": 0, "none": 0}
    orig_route = attn_mod.maybe_forward_split_kv_b_proj

    def counted(m, module, kv_a, forward_batch=None):
        counts["calls"] += 1
        out = orig_route(m, module, kv_a, forward_batch)
        counts["hits" if out is not None else "none"] += 1
        return out

    attn_mod.maybe_forward_split_kv_b_proj = counted
    try:
        skv.invalidate_split_kv_b_proj(kv)
        old_counts = dict(counts)
        counts.update({"calls": 0, "hits": 0, "none": 0})
        q0, k0, v0, _ = run_prepare(attn, fx)
        old_counts = dict(counts)
        kv_a0 = fx["fixture"].last_kv_a.detach().clone()
        kn0 = fx["recorder"]["k_nope"].clone()
        kp0 = fx["recorder"]["k_pe"].clone()
        old_stride = fx["recorder"]["k_nope_stride"]

        counts.update({"calls": 0, "hits": 0, "none": 0})
        state, reason = skv.prepare_split_kv_b_proj(attn, kv, force=True)
        assert state is not None, reason
        q1, k1, v1, _ = run_prepare(attn, fx)
        new_counts = dict(counts)
        kv_a1 = fx["fixture"].last_kv_a.detach().clone()
        kn1 = fx["recorder"]["k_nope"].clone()
        kp1 = fx["recorder"]["k_pe"].clone()
        new_stride = fx["recorder"]["k_nope_stride"]
    finally:
        attn_mod.maybe_forward_split_kv_b_proj = orig_route

    assert old_counts == {"calls": 1, "hits": 0, "none": 1}, old_counts
    assert new_counts == {"calls": 1, "hits": 1, "none": 0}, new_counts
    assert tuple(kv_a1.shape) == (TOKENS, 1, 1, 512)
    assert kv_a1.is_contiguous()
    assert old_stride[-2] == DK + DV, old_stride
    assert new_stride[-2] == DK, new_stride
    assert kn1.is_contiguous() and v1.is_contiguous()

    stats_k = tu.close_stats(k1, k0, 0.0, 0.0)
    stats_v = tu.close_stats(v1, v0, 0.0, 0.0)
    stats_q = tu.close_stats(q1, q0, 0.0, 0.0)
    assert stats_k["bad"] == 0 and stats_v["bad"] == 0 and stats_q["bad"] == 0

    rows = list(range(TOKENS))
    gk, gv = T.cpu_reference(
        kv_a1.reshape(TOKENS, -1).float().cpu(),
        ckpt["weight"].float(),
        None,
        DK,
        DV,
        20,
        rows,
    )
    gold_k = tu.close_stats(kn1, gk, 0.5, 2e-2)
    gold_v = tu.close_stats(v1, gv, 0.5, 2e-2)
    assert gold_k["bad"] == 0 and gold_v["bad"] == 0
    return {
        "old_route_calls": old_counts,
        "new_route_calls": new_counts,
        "kv_a_shape": list(kv_a1.shape),
        "kv_a_contiguous": True,
        "old_k_nope_stride": list(old_stride),
        "new_k_nope_stride": list(new_stride),
        "k_nope_contiguous": True,
        "v_contiguous": True,
        "old_vs_new_q": stats_q,
        "old_vs_new_k": stats_k,
        "old_vs_new_v": stats_v,
        "golden_k_nope": gold_k,
        "golden_v": gold_v,
    }


def _profile_trace(tag, fn, tokens=TOKENS):
    import pathlib

    pathlib.Path(tu.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    trace = f"{tu.RESULTS_DIR}/{tag}.json"
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=True,
    ) as prof:
        for _ in range(5):
            fn()
    prof.export_chrome_trace(trace)
    events = json.load(open(trace))
    if not isinstance(events, list):
        events = events.get("traceEvents", [])
    dev = {}
    copy_inputs = []
    slice_inputs = []
    source_slices = 0
    for e in events:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "")
        args = e.get("args", {})
        if name.startswith("aclnn") or args.get("Task Type") in (
            "KERNEL_AICORE",
            "KERNEL_AIVECTOR",
            "KERNEL_AIVEC",
        ):
            dev[name] = dev.get(name, 0) + 1
        if e.get("cat") == "cpu_op" and name == "aten::copy_":
            copy_inputs.append(args.get("Input Dims", ""))
        if e.get("cat") == "cpu_op" and name == "aten::slice":
            dims = args.get("Input Dims", "")
            slice_inputs.append(dims)
            if dims.startswith(f"{tokens},20,448;"):
                source_slices += 1
    return {
        "trace": trace,
        "device_kernels": dev,
        "matmul_kernels": sum(v for k, v in dev.items() if "MatMul" in k),
        "slice_kernels": sum(v for k, v in dev.items() if "Slice" in k),
        "viewcopy_kernels": sum(v for k, v in dev.items() if "ViewCopy" in k),
        "broadcast_kernels": sum(v for k, v in dev.items() if "Broadcast" in k),
        "packed_source_slices": source_slices,
        "copy_inputs": copy_inputs,
        "slice_inputs": slice_inputs,
    }


def profile_route(attn, fx, ckpt):
    kv = attn.kv_b_proj
    # Remove the correctness recorder clones from the profiled region.
    orig_concat = fx["restore"][2]
    attn._concat_and_cast_mha_k = orig_concat

    full = {}
    for side in ("old", "new"):
        if side == "old":
            skv.invalidate_split_kv_b_proj(kv)
        else:
            state, reason = skv.prepare_split_kv_b_proj(attn, kv, force=True)
            assert state is not None, reason
        full[side] = _profile_trace(
            f"real_load_prepare_profile_{side}_fix3",
            lambda: run_prepare(attn, fx),
        )
    old, new = full["old"], full["new"]
    assert old["slice_kernels"] - new["slice_kernels"] >= 10, full
    assert old["packed_source_slices"] >= 10, old
    assert new["packed_source_slices"] == 0, new
    assert new["matmul_kernels"] == old["matmul_kernels"] + 5, full
    assert not any(dims.startswith(f"{TOKENS},20,256;") for dims in new["copy_inputs"])

    # Isolated KV projection + final K assembly with the real checkpoint module
    # and the real 4D kv_a.  This removes the shared q-path fixture kernels.
    kv_a4d = fx["fixture"].static_kv_a
    k_pe = fx["fixture"].static_k_pe.reshape(TOKENS, 1, DR)

    def baseline_kv():
        pack = kv(kv_a4d)[0].view(-1, 20, DK + DV)
        k_nope = pack[..., :DK]
        v = pack[..., DK:].contiguous()
        return orig_concat(k_nope, k_pe, fx["fb"]), v

    def candidate_kv():
        split = skv.maybe_forward_split_kv_b_proj(attn, kv, kv_a4d, fx["fb"])
        assert split is not None
        k_nope, v = split
        return orig_concat(k_nope, k_pe, fx["fb"]), v

    isolated = {}
    for side, fn in (("old", baseline_kv), ("new", candidate_kv)):
        if side == "old":
            skv.invalidate_split_kv_b_proj(kv)
        else:
            state, reason = skv.prepare_split_kv_b_proj(attn, kv, force=True)
            assert state is not None, reason
        isolated[side] = _profile_trace(
            f"real_load_kv_profile_{side}_fix3", fn
        )
    io, inew = isolated["old"], isolated["new"]
    assert io["slice_kernels"] >= 10, io
    assert inew["slice_kernels"] <= 5, inew
    assert io["slice_kernels"] - inew["slice_kernels"] >= 10, isolated
    assert io["packed_source_slices"] >= 10, io
    assert inew["packed_source_slices"] == 0, inew
    # The only remaining candidate Slice inputs are the final K destination;
    # no packed [T,20,448] source and no [T,20,256] V materialization.
    assert all(
        dims.startswith(f"{TOKENS},20,256;") for dims in inew["slice_inputs"]
    ), inew["slice_inputs"]
    assert not any(
        dims.startswith(f"{TOKENS},20,256;") for dims in inew["copy_inputs"]
    ), inew["copy_inputs"]
    assert inew["matmul_kernels"] == io["matmul_kernels"] + 5, isolated
    assert inew["viewcopy_kernels"] == 10, inew
    assert inew["broadcast_kernels"] == 5, inew
    assert all(
        dims.startswith(f"{TOKENS},20,192;") or dims.startswith(f"{TOKENS},20,64;")
        for dims in inew["copy_inputs"]
    ), inew["copy_inputs"]
    return {"full_prepare": full, "kv_only": isolated}


def main():
    def run(metrics):
        cfg, quant_config, quant_description, layer = build_real_layer()
        attn = layer.self_attn
        kv = attn.kv_b_proj
        assert type(kv).__name__ == "ColumnParallelLinear"
        assert type(kv.quant_method).__name__ == "UnquantizedLinearMethod"
        assert quant_description[KEY] == "FLOAT"
        assert type(quant_config.get_quant_method(kv, PREFIX)).__name__ == (
            "UnquantizedLinearMethod"
        )

        ckpt = read_checkpoint()
        order = real_load_and_final_process(cfg, quant_config, layer, ckpt)
        metrics["checkpoint"] = {k: v for k, v in ckpt.items() if k != "weight"}
        metrics["load_order"] = order
        metrics["source_evidence"] = {
            "glm_load_weights_calls_post_load": source_slice(
                "python/sglang/srt/models/glm4_moe_lite.py", 1401, 1408
            ),
            "loader_calls_load_weights_then_final_process": source_slice(
                "python/sglang/srt/model_loader/loader.py", 1007, 1028
            ),
            "unquant_final_process_npu_noop": source_slice(
                "python/sglang/srt/layers/quantization/unquant.py", 360, 362
            ),
            "modelslim_float_selects_unquant": source_slice(
                "python/sglang/srt/layers/quantization/modelslim/modelslim.py",
                284,
                291,
            ),
            "npu_format_cast_definition": source_slice(
                "python/sglang/srt/hardware_backend/npu/utils.py", 224, 289
            ),
            "model_runner_after_load": source_slice(
                "python/sglang/srt/model_executor/model_runner_components/load_model_utils.py",
                54,
                65,
            ),
            "model_runner_final_split_hook": source_slice(
                "python/sglang/srt/model_executor/model_runner.py", 1175, 1192
            ),
            "format_cast_call_sites": _format_cast_call_sites(),
            "format_cast_absent_in_target_path": True,
            "precompile_hook_absent_on_glm": _grep(
                "precompile_kernels_after_loading", ["python/sglang/srt/models/glm4_moe_lite.py"]
            ) == "",
        }

        fx = install_prepare_fixture(attn)
        try:
            metrics["route"] = route_case(attn, fx, ckpt)
            metrics["profile"] = profile_route(attn, fx, ckpt)
            order["route_hit_after_final_process"] = (
                metrics["route"]["new_route_calls"]["hits"] == 1
            )
        finally:
            orig_interleave, orig_kv_op, orig_concat = fx["restore"]
            torch_npu.npu_interleave_rope = orig_interleave
            torch_npu.npu_kv_rmsnorm_rope_cache = orig_kv_op
            attn._concat_and_cast_mha_k = orig_concat

        assert metrics["route"]["new_route_calls"]["hits"] == 1
        assert metrics["route"]["new_route_calls"]["calls"] == 1
        print(
            "PASS real checkpoint load/final-process/route: "
            f"route_hits={metrics['route']['new_route_calls']} "
            f"kv_a={metrics['route']['kv_a_shape']} "
            f"golden_k={metrics['route']['golden_k_nope']['max_abs']:.3g} "
            f"golden_v={metrics['route']['golden_v']['max_abs']:.3g}"
        )
        io = metrics["profile"]["kv_only"]["old"]
        inew = metrics["profile"]["kv_only"]["new"]
        print(
            "PASS real-load KV profile: old "
            f"matmul={io['matmul_kernels']} slice={io['slice_kernels']} -> new "
            f"matmul={inew['matmul_kernels']} slice={inew['slice_kernels']}"
        )

    tu.run_entrypoint("test_split_kv_real_load_npu_fix3", run)


if __name__ == "__main__":
    main()
