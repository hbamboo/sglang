"""Staged shared/routed MXFP8 expert overlap for GLM-4.7-Flash on Ascend.

Main:    router -> top-k / dispatch -> GMM1 -> GMM2 -> finalize -> join
Shared:           gate/up           ->       act+quant -> down -> join

The second and third stages wait for GMM1 and GMM2 respectively, avoiding
concurrent expert GEMMs while using the complementary vector/cube work.
"""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import lru_cache

import torch

from sglang.srt.utils import get_bool_env_var

_active_overlap = ContextVar("npu_shared_expert_overlap", default=None)


@lru_cache(maxsize=None)
def _shared_stream(device_index):
    # Calls run on the current NPU, keyed to prevent sharing across devices.
    return torch.npu.Stream(device=device_index)


def _linear_kernel(layer):
    scheme = getattr(layer, "scheme", None)
    return getattr(scheme, "kernel", layer.quant_method)


def can_overlap_shared_expert(block, hidden_states, forward_batch):
    """The staged path is for ordinary, single-rank MXFP8 prefill only."""
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPUMXFP8LinearMethod,
    )
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUMXFP8MoEMethod

    shared = getattr(block, "shared_experts", None)
    experts = block.experts
    return (
        forward_batch is not None
        and forward_batch.forward_mode.is_extend_without_speculative()
        and hidden_states.shape[0] > 0
        and hidden_states.dtype == torch.bfloat16
        and block.tp_size == 1
        and not block._enable_a2a_moe
        and not getattr(experts, "_use_ascend_fuseep", False)
        and block.num_fused_shared_experts == 0
        and shared is not None
        and isinstance(getattr(experts, "w13_kernel", None), NPUMXFP8MoEMethod)
        and isinstance(getattr(experts, "w2_kernel", None), NPUMXFP8MoEMethod)
        and isinstance(_linear_kernel(shared.gate_up_proj), NPUMXFP8LinearMethod)
        and isinstance(_linear_kernel(shared.down_proj), NPUMXFP8LinearMethod)
        and shared.down_proj.tp_size == 1
        and shared.down_proj.bias is None
    )


def get_shared_expert_overlap(experts):
    overlap = _active_overlap.get()
    return overlap if overlap is not None and overlap.experts is experts else None


@contextmanager
def shared_expert_overlap_context(overlap):
    token = _active_overlap.set(overlap)
    try:
        yield
    finally:
        # Also protect error paths: input references can be released by the
        # caller only after pending shared work has been joined to its stream.
        overlap.close()
        _active_overlap.reset(token)


class SharedExpertOverlap:
    def __init__(self, experts, shared):
        self.experts = experts
        self.shared = shared
        self.stream = _shared_stream(torch.npu.current_device())
        self.input_ready = torch.npu.Event()
        self.gmm1_done = torch.npu.Event()
        self.gmm2_done = torch.npu.Event()
        self.shared_done = torch.npu.Event()
        self.annotate = get_bool_env_var("SGLANG_NPU_SHARED_EXPERT_OVERLAP_ANNOTATE")
        self.main = None
        self._clear()

    def _clear(self):
        self.gate_up = self.quantized = self.scales = self.output = None
        self.stage = 0

    def _label(self, name):
        return torch.profiler.record_function(name) if self.annotate else nullcontext()

    def start(self, hidden_states):
        if self.main is not None:
            raise RuntimeError("Shared expert overlap cannot be reentered")
        self.main = torch.npu.current_stream()
        self.output_dtype = hidden_states.dtype
        self.input_ready.record(self.main)
        self.stream.wait_event(self.input_ready)
        hidden_states.record_stream(self.stream)
        with torch.npu.stream(self.stream), self._label("shared_overlap/gate_up"):
            self.gate_up = self.shared.gate_up_proj(hidden_states)[0]
        self.stage = 1

    def before_gmm2(self):
        if self.stage != 1:
            raise RuntimeError("Shared activation must follow gate/up")
        self.gmm1_done.record(self.main)
        self.stream.wait_event(self.gmm1_done)
        with torch.npu.stream(self.stream), self._label("shared_overlap/swiglu_quant"):
            activated = self.shared.act_fn(self.gate_up)
            self.quantized, self.scales = torch.ops.npu.npu_dynamic_mx_quant(
                activated, dst_type=torch.float8_e4m3fn
            )
        self.gate_up = None
        self.stage = 2

    def after_gmm2(self):
        if self.stage != 2:
            raise RuntimeError("Shared down projection must follow activation")
        self.gmm2_done.record(self.main)
        self.stream.wait_event(self.gmm2_done)
        with torch.npu.stream(self.stream), self._label("shared_overlap/down_proj"):
            self.output = _linear_kernel(self.shared.down_proj).apply_quantized(
                self.shared.down_proj, self.quantized, self.scales, self.output_dtype
            )
            self.shared_done.record(self.stream)
        self.quantized = self.scales = None
        self.stage = 3

    def finish(self):
        if self.stage != 3:
            raise RuntimeError("The Ascend runner did not execute shared overlap stages")
        self.main.wait_event(self.shared_done)
        self.output.record_stream(self.main)
        output = self.output
        self.output = None
        self.stage = 4
        return output

    def close(self):
        if self.main is not None:
            if self.stage != 4:
                self.main.wait_stream(self.stream)
            self.main = None
        self._clear()
