"""
NPU MoE init routing components.

Prepare token routing before expert computation. Two API versions are provided:
- v1: legacy routing using ``npu_moe_init_routing``.
- v2: improved routing using ``npu_moe_init_routing_v2``.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_disagg
from sglang.srt.utils import get_bool_env_var

# ``npu_moe_init_routing_v2`` quant_mode selecting MXFP8: the op emits an
# float8_e4m3fn payload plus an e8m0 block scale, fusing the activation quant
# that would otherwise need a separate ``npu_dynamic_mx_quant`` pass.
MXFP8_QUANT_MODE = 3

# Opt-in for a single-process mixed prefill+decode server (disaggregation_mode
# "null"). It only relaxes the role test; every row/dtype/geometry guard below
# still applies, and decode batches (<=512 rows) can never reach 8192 rows.
_NPU_MIXED_PREFILL_OPT = get_bool_env_var("SGLANG_NPU_MIXED_PREFILL_OPT", "0")


def _prefill_or_mixed_opt_role() -> bool:
    """Original PD prefill, or the explicit mixed-process opt-in under "null"."""
    mode = get_disagg().disaggregation_mode
    if mode == "prefill":
        return True
    return mode == "null" and _NPU_MIXED_PREFILL_OPT


def _normalize_mxfp_scale(scale: torch.Tensor) -> torch.Tensor:
    """Reshape a flat 2D e8m0 block scale ``[N, M]`` into pair-split ``[N, M//2, 2]``.

    ``npu_moe_init_routing_v2(quant_mode=3)`` emits the scale flat, while the
    grouped matmul wants the pair-split view. Already-3D scales (what
    ``npu_dynamic_mx_quant`` returns) pass through untouched.
    """
    if scale is None or scale.ndim != 2:
        return scale
    return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)


def _can_prequantize_mxfp8(
    hidden_states, topk_ids, num_experts, top_k, quant_mode
):
    """Keep the default and all unqualified routes on the existing fused kernel."""
    if (
        quant_mode != MXFP8_QUANT_MODE
        or not envs.SGLANG_NPU_MOE_PREQUANT_ROUTING.get()
    ):
        return False
    return (
        hidden_states.device.type == "npu"
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.ndim == 2
        and 8192 <= hidden_states.shape[0] <= 36000
        and hidden_states.shape[1] == 2048
        and hidden_states.is_contiguous()
        and top_k == 4
        and num_experts == 64
        and topk_ids.shape == (hidden_states.shape[0], 4)
        and topk_ids.dtype == torch.int32
        and topk_ids.device == hidden_states.device
        and topk_ids.is_contiguous()
        and hasattr(torch, "float8_e8m0fnu")
        and _prefill_or_mixed_opt_role()
    )


class BaseInitRouting(ABC):
    """Abstract base for NPU MoE init routing."""

    @abstractmethod
    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: ...


class NPUMoEInitRouting_v1(BaseInitRouting):
    """
    NPU MoE init routing (v1 API).

    Uses ``npu_moe_init_routing`` with a manually constructed ``row_idx`` tensor.
    """

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]
        row_idx_len = num_tokens * top_k
        row_idx = (
            torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_ids.device)
            .view(topk_ids.shape[1], -1)
            .permute(1, 0)
            .contiguous()
        )

        hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch.ops.npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=num_tokens,
            )
        )
        expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts
        )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, None


class NPUMoEInitRouting_v2(BaseInitRouting):
    """
    NPU MoE init routing (v2 API).

    Uses ``npu_moe_init_routing_v2``, which integrates expert token counting.
    """

    def __init__(self, quant_mode: int = -1):
        self.quant_mode = quant_mode

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]
        prequant = _can_prequantize_mxfp8(
            hidden_states, topk_ids, num_experts, top_k, self.quant_mode
        )
        scale_args = {}
        if prequant:
            # Quantize distinct rows once, instead of once per expanded top-k row.
            # The dispatcher still sees the original BF16 tensor, preserving its
            # top-k weight dtype and the router/shared-expert input contract.
            hidden_states, scale = torch.ops.npu.npu_dynamic_mx_quant(
                hidden_states,
                axis=-1,
                round_mode="rint",
                dst_type=torch.float8_e4m3fn,
                block_size=32,
                scale_alg=0,
            )
            if scale.dtype == torch.uint8:
                scale = scale.view(torch.float8_e8m0fnu)
            if scale.dtype != torch.float8_e8m0fnu:
                raise RuntimeError("Prequantized MoE routing requires E8M0 scale bytes")
            scale_args["scale"] = scale
        routed = None
        if prequant and envs.SGLANG_NPU_MOE_STABLE_ROUTING.get():
            from sglang.srt.hardware_backend.npu.moe.stable_routing import (
                try_stable_prequantized_routing,
            )

            routed = try_stable_prequantized_routing(
                hidden_states, topk_ids, scale_args["scale"], num_experts, top_k
            )
        if routed is None:
            hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
                torch.ops.npu.npu_moe_init_routing_v2(
                    hidden_states,
                    topk_ids,
                    active_num=num_tokens * top_k,
                    expert_num=num_experts,
                    expert_tokens_num_type=1,
                    expert_tokens_num_flag=True,
                    active_expert_range=[0, num_experts],
                    quant_mode=-1 if prequant else self.quant_mode,
                    **scale_args,
                )
            )
        else:
            hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = routed
        if self.quant_mode == -1:
            pertoken_scale = None
        elif self.quant_mode == MXFP8_QUANT_MODE:
            pertoken_scale = _normalize_mxfp_scale(pertoken_scale)
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, pertoken_scale


class NPUMoEInitRouting_Quant(BaseInitRouting):
    """
    NPU MoE init routing (Quant API).

    Uses ``npu_moe_init_routing_quant``, which integrates expert token counting.
    """

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]

        hidden_states, expanded_row_idx, expert_tokens, _, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_quant(
                hidden_states,
                topk_ids,
                active_num=num_tokens * topk_ids.shape[1],
                expert_num=num_experts,
                expert_tokens_num_mode=1,
                expert_tokens_before_capacity_flag=False,
                quant_mode=1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, pertoken_scale
