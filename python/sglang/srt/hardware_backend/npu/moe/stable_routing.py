"""Conservative optional E64/K4 raw-byte routing after existing MX quantization."""
from functools import lru_cache
import logging

import torch

from sglang.srt.runtime_context import get_disagg, get_exec

logger = logging.getLogger(__name__)
_logged_hit = False


def _eligible(x, ids, scale, num_experts, top_k):
    return (
        num_experts == 64
        and top_k == 4
        and x.device.type == "npu"
        and x.dtype == torch.float8_e4m3fn
        and x.ndim == 2
        and 8192 <= x.shape[0] <= 36000
        and x.shape[1] == 2048
        and x.is_contiguous()
        and ids.dtype == torch.int32
        and ids.shape == (x.shape[0], 4)
        and ids.device == x.device
        and ids.is_contiguous()
        and scale is not None
        and scale.dtype == torch.float8_e8m0fnu
        and scale.shape == (x.shape[0], 32, 2)
        and scale.device == x.device
        and scale.is_contiguous()
        and get_disagg().disaggregation_mode == "prefill"
        and not torch.compiler.is_compiling()
        and get_exec().graph.cuda_graph_config is not None
        and get_exec().graph.cuda_graph_config.prefill.backend == "disabled"
        and not torch.npu.is_current_stream_capturing()
    )


@lru_cache(maxsize=1)
def _kernel_route():
    try:
        from sgl_kernel_npu.moe_stable_routing import route_prequantized
    except ModuleNotFoundError as exc:
        if exc.name not in ("sgl_kernel_npu", "sgl_kernel_npu.moe_stable_routing"):
            raise
        return None
    return route_prequantized


def try_stable_prequantized_routing(x, ids, scale, num_experts, top_k):
    """None means original native fallback; execution/load errors propagate."""
    if not _eligible(x, ids, scale, num_experts, top_k):
        return None
    route = _kernel_route()
    if route is None:
        return None
    output = route(x, ids, scale)
    if output is None:
        return None
    global _logged_hit
    if not _logged_hit:
        _logged_hit = True
        # CPU metadata only; no device synchronization or tensor value access.
        logger.info(
            "Stable MoE routing queued: T=%d H=2048 E=64 K=4 FP8/E8; eager PD prefill",
            x.shape[0],
        )
    return output
