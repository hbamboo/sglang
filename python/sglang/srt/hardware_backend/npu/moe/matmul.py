from abc import ABC, abstractmethod
import logging
from typing import Tuple

import torch


logger = logging.getLogger(__name__)
_logged_mxfp8_counts = False


def _use_mxfp8_gmm1_counts(hidden_states, expert_tokens, weight, weight_prefix,
                           group_list_type, transposed, scale_args):
    # Short-circuit graph tracing and the default before any runtime/NPU query.
    if torch.compiler.is_compiling():
        return False
    from sglang.srt.environ import envs

    if not envs.SGLANG_NPU_GMM1_COUNTS.get():
        return False
    if weight_prefix != "w13" or group_list_type != 1 or not transposed:
        return False
    if (
        hidden_states.device.type != "npu"
        or hidden_states.dtype != torch.float8_e4m3fn
        or hidden_states.ndim != 2
        or hidden_states.shape[1] != 2048
        or not 4 <= hidden_states.shape[0] <= 65536
        or hidden_states.shape[0] % 4 != 0
        or not hidden_states.is_contiguous()
        or weight.dtype != torch.float8_e4m3fn
        or tuple(weight.shape) != (64, 2048, 3072)
        or weight.device != hidden_states.device
        or expert_tokens.dtype != torch.int64
        or tuple(expert_tokens.shape) != (64,)
        or expert_tokens.device != hidden_states.device
        or not expert_tokens.is_contiguous()
        or scale_args.get("dequant_mode") != 2
        or scale_args.get("quant_mode") != 2
        or scale_args.get("dequant_dtype") != torch.float32
        or scale_args.get("quant_dtype") != torch.float8_e4m3fn
        or scale_args.get("weight_scale_dtype") is None
        or scale_args.get("x_scale_dtype") != scale_args.get("weight_scale_dtype")
    ):
        return False
    from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    if get_disagg().disaggregation_mode != "prefill" or get_parallel().world_size != 1:
        return False
    graph = get_exec().graph.cuda_graph_config
    if (
        graph is None
        or graph.prefill.backend != "disabled"
        or get_tc_piecewise_forward_context() is not None
    ):
        return False
    return not torch.npu.is_current_stream_capturing()


class BaseMatmul(ABC):
    @abstractmethod
    def forward(
        self,
        layer: torch.nn.Module,
        weight_prefix: str,
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        output_dtype: torch.dtype,
        group_list_type: int,
        transposed: bool,
        **scale_args,
    ) -> torch.Tensor:
        pass


class GroupedMatmul(BaseMatmul):
    def forward(
        self,
        layer: torch.nn.Module,
        weight_prefix: str,
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        output_dtype: torch.dtype,
        group_list_type: int,
        transposed: bool,
        **scale_args,
    ) -> torch.Tensor:
        # Access the weight attribute directly from the layer
        weight = getattr(layer, f"{weight_prefix}_weight", None)
        if weight is None:
            raise AttributeError(
                f"Weight attribute '{weight_prefix}_weight' not found in layer"
            )
        return torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[weight] if transposed else [weight.transpose(1, 2)],
            **scale_args,
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=output_dtype,
        )[0]


class GroupedMatmulSwigluQuant(BaseMatmul):
    """Grouped matmul with swiglu and requantisation fused into one kernel.

    Used for the gate/up projection (gmm1) of block-scaled MoE: the kernel emits
    activations already quantised for the following down projection, so the
    caller has no separate activation step. Unlike ``GroupedMatmul`` it returns
    ``(quantized_activations, block_scale)`` instead of a single tensor, and it
    takes no ``output_dtype`` — the output dtype is set through ``quant_dtype``
    in ``scale_args``.
    """

    def forward(
        self,
        layer: torch.nn.Module,
        weight_prefix: str,
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        output_dtype: torch.dtype = None,
        group_list_type: int = 1,
        transposed: bool = True,
        **scale_args,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = getattr(layer, f"{weight_prefix}_weight", None)
        if weight is None:
            raise AttributeError(
                f"Weight attribute '{weight_prefix}_weight' not found in layer"
            )
        if _use_mxfp8_gmm1_counts(
            hidden_states, expert_tokens, weight, weight_prefix,
            group_list_type, transposed, scale_args,
        ):
            result = torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2(
                x=hidden_states, weight=[weight], group_list=expert_tokens,
                group_list_type=1, **scale_args,
            )
            global _logged_mxfp8_counts
            if not _logged_mxfp8_counts:
                _logged_mxfp8_counts = True
                logger.info(
                    "MXFP8 GMM1 direct counts queued: M=%d E=64 K=2048 N=3072; eager single-rank PD prefill",
                    hidden_states.shape[0],
                )
            return result
        # This op wants a cumulative group_list while the plain grouped matmul
        # keeps the COUNT form the dispatcher produces (group_list_type=1). The
        # asymmetry is intentional.
        group_list = expert_tokens.cumsum(0) if group_list_type == 1 else expert_tokens
        return torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2(
            x=hidden_states,
            weight=[weight] if transposed else [weight.transpose(1, 2)],
            group_list=group_list,
            **scale_args,
        )
