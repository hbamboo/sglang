import os
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


@lru_cache(maxsize=1)
def is_mla_preprocess_enabled() -> bool:
    # MLAPO writes the latent KV through two separate cache buffers, which is
    # incompatible with the single fused 576-wide pool flash_mla needs.
    if (
        envs.ENABLE_ASCEND_FLASH_MLA.get()
        or envs.SGLANG_NPU_MLA_UNIFIED_KV.get()
    ):
        # The ported MlaPrologV3 path stages its KV in a scratch and lands it
        # through the pool, so it does run on top of the fused layout; the
        # legacy int8 prologue still addresses the split buffers and stays off.
        return is_mlaprolog_v3_enabled()
    # The ported MlaPrologV3 path lives inside the MLAPO prologue, so enabling
    # it implies the fused prologue (and the save_kv_cache=False handshake).
    return get_bool_env_var("SGLANG_NPU_USE_MLAPO") or is_mlaprolog_v3_enabled()


@lru_cache(maxsize=1)
def is_fia_nz() -> bool:
    is_fia_nz_ = get_bool_env_var("SGLANG_USE_FIA_NZ")
    if is_fia_nz_:
        assert (
            is_mla_preprocess_enabled()
        ), "SGLANG_USE_FIA_NZ must be enable with SGLANG_NPU_USE_MLAPO"
    return is_fia_nz_


@lru_cache(maxsize=1)
def is_mlaprolog_v3_enabled() -> bool:
    """Use the ported MlaPrologV3 kernel for the MLA prologue (A/B switch).

    ``SGLANG_NPU_USE_MLAPO_V3=1`` (in addition to ``SGLANG_NPU_USE_MLAPO=1``)
    replaces the fused int8 ``npu_mla_preprocess`` / the unfused prologue with
    the MlaPrologV3 operator from sgl-kernel-npu.  It accepts the GLM-4.7-Flash
    geometry (He=2048, Hcq=768, Hckv=512, D=192, Dr=64, N=20) and either MXFP8
    (``float8_e4m3`` + per-32 e8m0 scale) or bf16 projections.
    """
    return get_bool_env_var("SGLANG_NPU_USE_MLAPO_V3")


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def transdata(nd_mat, block_size: tuple = (16, 16)):
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, ((0, r_pad, 0, c_pad)))
    nz_mat = torch.permute(
        torch.reshape(
            nd_mat,
            (r // block_size[0], block_size[0], c // block_size[1], block_size[1]),
        ),
        [2, 0, 1, 3],
    )
    nz_mat = torch.reshape(
        nz_mat, (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3])
    )
    return nz_mat


def trans_rope_weight(weight, rope_dim):
    weight_1 = weight[..., -rope_dim::2, :].contiguous()
    weight_2 = weight[..., -rope_dim + 1 :: 2, :].contiguous()
    weight[..., -rope_dim:, :] = torch.cat([weight_1, weight_2], dim=-2)

    return weight.contiguous()


class NPUFusedMLAPreprocess(torch.nn.Module):
    def __init__(
        self,
        fused_qkv_a_proj_with_mqa,
        q_a_layernorm,
        kv_a_layernorm,
        q_b_proj,
        w_kc,
        rotary_emb,
        layer_id,
        num_local_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        super().__init__()
        self.qkv_a_proj = fused_qkv_a_proj_with_mqa
        self.q_a_layernorm = q_a_layernorm
        self.kv_a_layernorm = kv_a_layernorm
        self.q_b_proj = q_b_proj
        self.w_kc = w_kc.contiguous()
        self.rotary_emb = rotary_emb
        self.layer_id = layer_id
        self.quant_config = quant_config
        self.has_preprocess_weights = False
        self.dtype = None

        self.q_lora_rank = self.q_b_proj.input_size  # 1536
        self.kv_lora_rank = self.kv_a_layernorm.hidden_size  # 512
        self.num_local_heads = num_local_heads  # tp
        self.qk_nope_head_dim = qk_nope_head_dim  # 128
        self.qk_rope_head_dim = qk_rope_head_dim  # 64
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        # W8A8-int8 checkpoints expose `weight_scale`; an MXFP8 checkpoint keeps
        # its scale in `weight_scale_inv` (per-block e8m0) and needs no
        # per-tensor dequant scale here.
        weight_scale = getattr(self.q_b_proj, "weight_scale", None)
        self.q_b_proj_weight_scale = (
            weight_scale.view(1, -1).to(torch.float)
            if weight_scale is not None
            else None
        )

    def preprocess_weights(self, hidden_states):
        self.dummy = torch.zeros(
            (hidden_states.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self.qkv_a_proj_input_offset = self.qkv_a_proj.input_offset.to(dtype=torch.int8)
        self.q_b_proj_input_offset = self.q_b_proj.input_offset.to(dtype=torch.int8)

        # matmul_0 weight [7168, 2112]
        fused_qkv_a_proj_with_mqa_weight_q = self.qkv_a_proj.weight.data[
            :, : self.q_lora_rank
        ].clone()  # [7168, 1536]
        fused_qkv_a_proj_with_mqa_weight_kv = self.qkv_a_proj.weight.data[
            :, self.q_lora_rank :
        ].clone()  # [7168, 576]
        # rope fit
        fused_qkv_a_proj_with_mqa_weight_kv_t = (
            fused_qkv_a_proj_with_mqa_weight_kv.t().contiguous()
        )
        fused_qkv_a_proj_with_mqa_weight_kv_t = trans_rope_weight(
            fused_qkv_a_proj_with_mqa_weight_kv_t, self.qk_rope_head_dim
        )
        fused_qkv_a_proj_with_mqa_weight_kv = (
            fused_qkv_a_proj_with_mqa_weight_kv_t.t().contiguous()
        )
        # cat nz
        fused_qkv_a_proj_with_mqa_weight_new = torch.cat(
            (fused_qkv_a_proj_with_mqa_weight_kv, fused_qkv_a_proj_with_mqa_weight_q),
            dim=-1,
        )
        fused_qkv_a_proj_with_mqa_weight = (
            fused_qkv_a_proj_with_mqa_weight_new.t().contiguous()
        )
        fused_qkv_a_proj_with_mqa_weight_nz = (
            transdata(fused_qkv_a_proj_with_mqa_weight, block_size=(16, 32))
            .unsqueeze(0)
            .contiguous()
        )
        self.qkv_a_proj_weight_nz = npu_format_cast(fused_qkv_a_proj_with_mqa_weight_nz)

        # matmul_0 deq_scale [2112]
        fused_qkv_a_proj_with_mqa_deq_scale_q = self.qkv_a_proj.deq_scale.data[
            : self.q_lora_rank
        ].clone()  # [7168, 1536]
        fused_qkv_a_proj_with_mqa_deq_scale_kv = self.qkv_a_proj.deq_scale.data[
            self.q_lora_rank :
        ].clone()  # [7168, 576]
        # rope fit
        fused_qkv_a_proj_with_mqa_deq_scale_kv = (
            fused_qkv_a_proj_with_mqa_deq_scale_kv.reshape(
                self.kv_lora_rank + self.qk_rope_head_dim, -1
            ).contiguous()
        )
        fused_qkv_a_proj_with_mqa_deq_scale_kv = trans_rope_weight(
            fused_qkv_a_proj_with_mqa_deq_scale_kv, self.qk_rope_head_dim
        )
        fused_qkv_a_proj_with_mqa_deq_scale_kv = (
            fused_qkv_a_proj_with_mqa_deq_scale_kv.view(
                self.kv_lora_rank + self.qk_rope_head_dim
            ).contiguous()
        )
        self.qkv_a_proj_deq_scale_kvq = torch.cat(
            (
                fused_qkv_a_proj_with_mqa_deq_scale_kv,
                fused_qkv_a_proj_with_mqa_deq_scale_q,
            ),
            dim=-1,
        )

        # matmul_0 quant_bias [2112]
        fused_qkv_a_proj_with_mqa_quant_bias_q = self.qkv_a_proj.quant_bias.data[
            : self.q_lora_rank
        ].clone()  # [7168, 1536]
        fused_qkv_a_proj_with_mqa_quant_bias_kv = self.qkv_a_proj.quant_bias.data[
            self.q_lora_rank :
        ].clone()  # [7168, 576]
        # rope fit
        fused_qkv_a_proj_with_mqa_quant_bias_kv = (
            fused_qkv_a_proj_with_mqa_quant_bias_kv.reshape(
                self.kv_lora_rank + self.qk_rope_head_dim, -1
            ).contiguous()
        )
        fused_qkv_a_proj_with_mqa_quant_bias_kv = trans_rope_weight(
            fused_qkv_a_proj_with_mqa_quant_bias_kv, self.qk_rope_head_dim
        )
        fused_qkv_a_proj_with_mqa_quant_bias_kv = (
            fused_qkv_a_proj_with_mqa_quant_bias_kv.view(
                self.kv_lora_rank + self.qk_rope_head_dim
            ).contiguous()
        )
        self.qkv_a_proj_quant_bias_kvq = torch.cat(
            (
                fused_qkv_a_proj_with_mqa_quant_bias_kv,
                fused_qkv_a_proj_with_mqa_quant_bias_q,
            ),
            dim=-1,
        )

        # matmul_1 weight [1536, num_head * 192]
        q_b_proj_weight = self.q_b_proj.weight.data.clone()
        q_b_proj_weight = q_b_proj_weight.t().reshape(
            self.num_local_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1
        )
        q_b_proj_weight = trans_rope_weight(q_b_proj_weight, self.qk_rope_head_dim)
        q_b_proj_weight = q_b_proj_weight.reshape(
            self.num_local_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), -1
        )
        q_b_proj_weight_nz = (
            transdata(q_b_proj_weight, block_size=(16, 32)).unsqueeze(0).contiguous()
        )
        self.q_b_proj_weight_nz = npu_format_cast(q_b_proj_weight_nz)

        # matmul_1 deq_scale [num_head * 192]
        q_b_proj_deq_scale = self.q_b_proj.deq_scale.data.clone()
        q_b_proj_deq_scale = q_b_proj_deq_scale.reshape(
            self.num_local_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1
        )
        q_b_proj_deq_scale = trans_rope_weight(
            q_b_proj_deq_scale, self.qk_rope_head_dim
        )
        self.q_b_proj_deq_scale = q_b_proj_deq_scale.reshape(
            self.num_local_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        )

        # matmul_1 quant_bias [num_head * 192]
        q_b_proj_quant_bias = self.q_b_proj.quant_bias.data.clone()
        q_b_proj_quant_bias = q_b_proj_quant_bias.reshape(
            self.num_local_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1
        )
        q_b_proj_quant_bias = trans_rope_weight(
            q_b_proj_quant_bias, self.qk_rope_head_dim
        )
        self.q_b_proj_quant_bias = q_b_proj_quant_bias.reshape(
            self.num_local_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        )

    def mlaprolog_preprocess_weight(self):
        self.qkv_a_proj.weight.data = self.qkv_a_proj.weight.data.transpose(0, 1)
        qkv_a_proj_weight_q = self.qkv_a_proj.weight.data[:, : self.q_lora_rank].clone()
        qkv_a_proj_weight_kv = self.qkv_a_proj.weight.data[
            :, self.q_lora_rank :
        ].clone()
        self.q_a_proj_weight = npu_format_cast(qkv_a_proj_weight_q)
        self.kv_a_proj_weight = npu_format_cast(qkv_a_proj_weight_kv)

    def get_sin_cos(self, positions):
        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 2)
        sin = sin.repeat(1, 2)
        return cos, sin

    def get_kv_cache_and_cache_idx(self, forward_batch):
        k_cache, v_cache = get_token_to_kv_pool().get_kv_buffer(self.layer_id)
        slot_mapping = forward_batch.out_cache_loc.to(dtype=torch.int32)
        return k_cache, v_cache, slot_mapping

    def forward_absorb_prepare_npu_rms_norm_cache(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch,
        zero_allocator,
    ):
        bsz, _ = hidden_states.view(-1, hidden_states.shape[-1]).shape
        self.dtype = hidden_states.dtype
        if self.layer_id == 0:
            self.cos, self.sin = self.get_sin_cos(positions)
            self.rotary_emb.cos_cached, self.rotary_emb.sin_cache = self.cos, self.sin
        else:
            self.cos, self.sin = self.rotary_emb.cos_cached, self.rotary_emb.sin_cache

        self.kvCache, self.kvCacheRope, self.slotmapping = (
            self.get_kv_cache_and_cache_idx(forward_batch)
        )

        if not self.has_preprocess_weights:
            self.has_preprocess_weights = True

        cos, sin = self.cos, self.sin

        if self.q_lora_rank is not None:
            fused_qkv_a_proj_out = self.qkv_a_proj(hidden_states)[0]
            q_lowrank, latent_cache = fused_qkv_a_proj_out.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q_lowrank)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )  # b*s,n,d

        q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim)
        q_nope = torch.matmul(q_nope.transpose(0, 1), self.w_kc).transpose(0, 1)

        q_pe = q_pe.view(-1, self.num_local_heads, 1, self.qk_rope_head_dim)
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)
        q_pe = torch.ops.npu.npu_interleave_rope(q_pe, cos, sin)  # (B,N,S,D)
        q_pe = q_pe.view(cos.shape[0], self.num_local_heads, self.qk_rope_head_dim)

        latent_cache = latent_cache.view(
            -1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim
        )  # (B*S,N,1,D)

        cache_mode = "PA_NZ" if is_fia_nz() else "PA_BNSD"
        self.kvCache = self.kvCache.view(
            -1,
            get_attn_backend().page_size,
            1,
            get_attn_backend().kv_lora_rank,
        )
        self.kvCacheRope = self.kvCacheRope.view(
            -1,
            get_attn_backend().page_size,
            1,
            get_attn_backend().qk_rope_head_dim,
        )
        k_rope, k_nope, _, _ = torch.ops.npu.npu_kv_rmsnorm_rope_cache(
            latent_cache,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            self.slotmapping.to(torch.int64),
            self.kvCacheRope,
            self.kvCache,
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode=cache_mode,
        )

        return (q_pe, k_rope, q_nope, k_nope, forward_batch, zero_allocator, positions)

    def forward_mlapo(self, positions, hidden_states, forward_batch, zero_allocator):
        input_dtype = hidden_states.dtype
        if not self.has_preprocess_weights:
            self.preprocess_weights(hidden_states)
            self.has_preprocess_weights = True
            self.dtype = hidden_states.dtype

        if self.layer_id == 0:
            cos, sin = self.get_sin_cos(positions)
            self.rotary_emb.cos_cached, self.rotary_emb.sin_cache = cos, sin
        else:
            cos, sin = self.rotary_emb.cos_cached, self.rotary_emb.sin_cache

        k_cache, v_cache, slot_mapping = self.get_kv_cache_and_cache_idx(forward_batch)

        q_nope_out = torch.empty(
            (hidden_states.shape[0], self.w_kc.shape[0], k_cache.shape[-1]),
            dtype=input_dtype,
            device=hidden_states.device,
        )
        q_rope_out = torch.empty(
            (hidden_states.shape[0], self.w_kc.shape[0], v_cache.shape[-1]),
            dtype=input_dtype,
            device=hidden_states.device,
        )
        if is_fia_nz():
            kv_shape, kv_rope_shape = k_cache.shape, v_cache.shape
            num_blocks, block_size, num_heads, _ = kv_shape
            k_cache = k_cache.view(
                num_blocks, num_heads * self.kv_lora_rank // 16, block_size, 16
            )
            v_cache = v_cache.view(
                num_blocks, num_heads * self.qk_rope_head_dim // 16, block_size, 16
            )
        # TODO: dummy inputs to be removed
        # https://github.com/sgl-project/sgl-kernel-npu/issues/78
        if hasattr(self.q_a_layernorm, "bias"):
            q_a_layernorm_bias = self.q_a_layernorm.bias
        else:
            q_a_layernorm_bias = self.dummy

        torch.ops.npu.mla_preprocess(
            hidden_states,
            self.dummy,
            self.dummy,
            self.qkv_a_proj_weight_nz,
            self.qkv_a_proj_deq_scale_kvq,
            self.q_a_layernorm.weight,
            q_a_layernorm_bias,
            self.q_b_proj_weight_nz,
            self.q_b_proj_deq_scale,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            self.w_kc,
            k_cache,
            v_cache,
            slot_mapping,
            quant_scale0=self.qkv_a_proj.input_scale,
            quant_offset0=self.qkv_a_proj_input_offset,
            bias0=self.qkv_a_proj_quant_bias_kvq,
            quant_scale1=self.q_b_proj.input_scale,
            quant_offset1=self.q_b_proj_input_offset,
            bias1=self.q_b_proj_quant_bias,
            cache_mode="nzcache" if is_fia_nz() else "krope_ctkv",
            quant_mode="per_tensor_quant_asymm",
            q_out0=q_nope_out,
            kv_cache_out0=k_cache,
            q_out1=q_rope_out,
            kv_cache_out1=v_cache,
        )

        if is_fia_nz():
            k_cache = k_cache.view(kv_shape)
            v_cache = v_cache.view(kv_rope_shape)

        return (
            q_rope_out,
            v_cache,
            q_nope_out,
            k_cache,
            forward_batch,
            zero_allocator,
            positions,
        )

    def forward_mlaprolog(self, positions, hidden_states, forward_batch):
        if not self.has_preprocess_weights:
            self.mlaprolog_preprocess_weight()
            self.has_preprocess_weights = True
        self.cos, self.sin = self.get_sin_cos(positions)
        k_cache, v_cache, slot_mapping = self.get_kv_cache_and_cache_idx(forward_batch)
        mla_prolog_input_args = {
            "token_x": hidden_states,
            "weight_dq": self.q_a_proj_weight,
            "weight_uq_qr": self.q_b_proj.weight,
            "weight_uk": self.w_kc,
            "weight_dkv_kr": self.kv_a_proj_weight,
            "rmsnorm_gamma_cq": self.q_a_layernorm.weight,
            "rmsnorm_gamma_ckv": self.kv_a_layernorm.weight,
            "rope_sin": self.sin,
            "rope_cos": self.cos,
            "kv_cache": k_cache,
            "kr_cache": v_cache,
            "cache_index": slot_mapping.to(dtype=torch.int64),
            "dequant_scale_w_uq_qr": self.q_b_proj_weight_scale,
            "rmsnorm_epsilon_cq": self.q_a_layernorm.variance_epsilon,
            "rmsnorm_epsilon_ckv": self.kv_a_layernorm.variance_epsilon,
            "cache_mode": "PA_BSND",
            "query_norm_flag": True,
            "weight_quant_mode": 1,  # 0:no quant; 1:uq_qr: quant; 2: weight_dq,weight_uq_qr,weight_dkv_kr: quant
        }
        q_nope, q_pe, dequant_scale_q_nope, qr, dequant_q_norm = (
            torch.ops.custom.npu_mla_prolog_v3(**mla_prolog_input_args)
        )
        dequant_q_norm = dequant_q_norm.view(hidden_states.shape[0])
        return (
            q_pe,
            v_cache,
            q_nope,
            k_cache,
            qr,
            forward_batch,
            positions,
            dequant_q_norm,
        )

    # ------------------------------------------------------- MlaPrologV3 (ported)
    def uses_mlaprolog(self) -> bool:
        """Whether this layer's prologue goes through an MLA-prolog operator.

        Both the ported kernel (``SGLANG_NPU_USE_MLAPO_V3``) and the W8A8-int8
        ``forward_mlaprolog`` path return the 8-tuple consumed by the callers;
        everything else returns the legacy 7-tuple.
        """
        if is_mlaprolog_v3_enabled():
            return True
        return hasattr(self.quant_config, "ignore") and any(
            re.fullmatch(r".*kv_b_proj", str(entry))
            for entry in self.quant_config.ignore
        )

    def _mlaprolog_v3_preprocess_weight(self):
        """Build the three projection weights in the operator's bf16 + NZ ABI.

        The checkpoint fuses q_a/kv_a into `qkv_a_proj_with_mqa` as [out, in]
        (or [in, out] for some schemes) and stores MXFP8 pairs, so the fused
        matrix is dequantised once, split on the output axis and cast to
        FRACTAL_NZ.  `w_kc` is already the absorbed bf16 [N, D, Hckv] weight.
        """
        from sgl_kernel_npu.mla_prolog_v3_compat import dequant_to_in_out, to_nd

        import torch_npu

        def to_nz(t):
            return torch_npu.npu_format_cast(t.contiguous(), 29)  # FRACTAL_NZ

        def as_bf16_in_out(module):
            weight = module.weight
            scale = getattr(module, "weight_scale_inv", None)
            plain = to_nd(weight.data)
            out = dequant_to_in_out(
                weight.data, scale.data if scale is not None else None
            )
            if out is None and plain is not None and plain.dtype == torch.bfloat16:
                out = plain
            if out is None:
                raise RuntimeError(
                    "MlaPrologV3: cannot build a bf16 %s from dtype %s"
                    % (type(module).__name__, weight.dtype)
                )
            return out

        fused = as_bf16_in_out(self.qkv_a_proj)
        fused_out = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        if fused.shape[1] != fused_out:
            raise RuntimeError(
                "MlaPrologV3: fused qkv_a output %d != FQ_KC+KV_KC+ROPE_H=%d"
                % (fused.shape[1], fused_out)
            )
        self.weight_dq = to_nz(fused[:, : self.q_lora_rank])
        self.weight_dkv_kr = to_nz(fused[:, self.q_lora_rank :])
        self.weight_uq_qr = to_nz(as_bf16_in_out(self.q_b_proj))
        self.weight_uk = self.w_kc.contiguous()
        if self.weight_uk.dtype != torch.bfloat16:
            self.weight_uk = self.weight_uk.to(torch.bfloat16)

    def _forward_mlaprolog_v3_bf16_prefill(self, positions, hidden_states, forward_batch, sin, cos):
        from sgl_kernel_npu.mla_prolog_v3_bf16_prefill import GlmMlaPrologPrefillBf16

        pool = get_token_to_kv_pool()
        if pool.store_dtype != pool.dtype:
            raise NotImplementedError("BF16 prefill requires an unquantized KV pool")
        if not hasattr(self, "_v3_bf16_prefill"):
            self._v3_bf16_prefill = GlmMlaPrologPrefillBf16(
                self.weight_dq, self.weight_uq_qr, self.weight_uk, self.weight_dkv_kr,
                self.q_a_layernorm.weight, self.kv_a_layernorm.weight,
                epsilon_cq=self.q_a_layernorm.variance_epsilon,
                epsilon_ckv=self.kv_a_layernorm.variance_epsilon)
            if get_bool_env_var("SGLANG_NPU_MLAPO_V3_DEBUG"):
                print("[mla_prolog_v3 BF16 prefill] layer", self.layer_id,
                      "tokens", hidden_states.shape[0], "quant_mode=0", flush=True)
        with torch.profiler.record_function("mla_prolog_v3_prefill_bf16"):
            q, qr, norm, kv, kr = self._v3_bf16_prefill(hidden_states, sin, cos)
            pool = get_token_to_kv_pool()
            pool.set_unified_mla_kv(self.layer_id, forward_batch.out_cache_loc, kv, kr)
        offset = self.layer_id - pool.start_layer
        return (qr, pool.v_buffer[offset], q, pool.k_buffer[offset], norm,
                forward_batch, positions, None)

    def forward_mlaprolog_v3(self, positions, hidden_states, forward_batch):
        """MlaPrologV3 prologue with a separately gated large-prefill path."""
        if not self.has_preprocess_weights:
            self._mlaprolog_v3_preprocess_weight()
            self.has_preprocess_weights = True
            if get_bool_env_var("SGLANG_NPU_MLAPO_V3_DEBUG"):
                print(
                    "[mlaprolog_v3] layer %d weights ready: dq=%s uq_qr=%s uk=%s dkv_kr=%s"
                    % (
                        self.layer_id,
                        tuple(self.weight_dq.shape),
                        tuple(self.weight_uq_qr.shape),
                        tuple(self.weight_uk.shape),
                        tuple(self.weight_dkv_kr.shape),
                    ),
                    flush=True,
                )

        cos, sin = self.get_sin_cos(positions)
        pool = get_token_to_kv_pool()
        unified_kv = bool(getattr(pool, "flash_mla", False))
        capture_dir = os.environ.get("SGLANG_NPU_MLAPO_V3_CAPTURE_DIR")
        if capture_dir and self.layer_id == 2 and hidden_states.shape[0] >= 8192:
            import json
            from pathlib import Path
            import torch_npu
            seen = getattr(self, "_v3_capture_shapes", set())
            tokens = hidden_states.shape[0]
            if tokens not in seen and len(seen) < 3:
                folder = Path(capture_dir)
                folder.mkdir(parents=True, exist_ok=True)
                inputs = dict(token_x=hidden_states, weight_dq=self.weight_dq,
                    weight_uq_qr=self.weight_uq_qr, weight_uk=self.weight_uk,
                    weight_dkv_kr=self.weight_dkv_kr,
                    rmsnorm_gamma_cq=self.q_a_layernorm.weight,
                    rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
                    rope_sin=sin, rope_cos=cos)
                meta = dict(layer=self.layer_id, tokens=tokens, weight_quant_mode=0,
                    kv_cache_quant_mode=0, query_norm_flag=True,
                    tensors={k: dict(shape=list(v.shape), stride=list(v.stride()), dtype=str(v.dtype),
                        format=int(torch_npu.get_npu_format(v))) for k,v in inputs.items()})
                print("[mla_prolog_v3 actual input]", json.dumps(meta), flush=True)
                (folder / ("T%d.json" % tokens)).write_text(json.dumps(meta, indent=2))
                torch.save({k: (torch_npu.npu_format_cast(v,2) if torch_npu.get_npu_format(v) not in (0,2) else v).cpu()
                    for k,v in inputs.items()}, folder / ("T%d.pt" % tokens))
                seen.add(tokens)
                self._v3_capture_shapes = seen

        if (
            get_bool_env_var("SGLANG_NPU_MLAPO_V3_PREFILL_BF16_NATIVE")
            and forward_batch.forward_mode.is_extend()
            and hidden_states.shape[0] >= 1024
            and unified_kv
            and tuple(self.weight_uk.shape) == (20, 192, 512)
            and self.weight_dq.shape == (2048, 768)
            and self.weight_uq_qr.shape == (768, 5120)
        ):
            return self._forward_mlaprolog_v3_bf16_prefill(
                positions, hidden_states, forward_batch, sin, cos)

        if unified_kv:
            kv_cache, kr_cache, cache_index = self._v3_scratch_cache(
                hidden_states, pool
            )
        else:
            k_cache, v_cache, slot_mapping = self.get_kv_cache_and_cache_idx(
                forward_batch
            )
            page_size = get_attn_backend().page_size
            kv_cache = k_cache.view(-1, page_size, 1, self.kv_lora_rank)
            kr_cache = v_cache.view(-1, page_size, 1, self.qk_rope_head_dim)
            cache_index = slot_mapping.to(dtype=torch.int64)

        q_nope, q_pe, _, q_norm, _ = torch.ops.npu.mla_prolog_v3(
            token_x=hidden_states,
            weight_dq=self.weight_dq,
            weight_uq_qr=self.weight_uq_qr,
            weight_uk=self.weight_uk,
            weight_dkv_kr=self.weight_dkv_kr,
            rmsnorm_gamma_cq=self.q_a_layernorm.weight,
            rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
            rope_sin=sin,
            rope_cos=cos,
            kv_cache=kv_cache,
            kr_cache=kr_cache,
            cache_index=cache_index,
            rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
            rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_BSND",
            query_norm_flag=True,
            weight_quant_mode=0,  # weights are bf16 (dequantised above)
        )

        if unified_kv:
            # The kernel can only address a packed (blocks, page, 1, W) cache, so
            # it wrote the one-step scratch above; land those rows in the fused
            # pool the same way set_kv_buffer() does.  The attention backend
            # reads the pool directly, MLAPO reports save_kv_cache=False.
            num_tokens = hidden_states.shape[0]
            pool.set_unified_mla_kv(
                self.layer_id,
                forward_batch.out_cache_loc,
                kv_cache.reshape(-1, self.kv_lora_rank)[:num_tokens],
                kr_cache.reshape(-1, self.qk_rope_head_dim)[:num_tokens],
            )
            kv_cache = pool.k_buffer[self.layer_id - pool.start_layer]
            kr_cache = pool.v_buffer[self.layer_id - pool.start_layer]

        # q_norm is the normalised q_lora: only the DSA indexer consumes it, and
        # the attention backend already reports save_kv_cache=False for MLAPO.
        return (
            q_pe,
            kr_cache,
            q_nope,
            kv_cache,
            q_norm,
            forward_batch,
            positions,
            None,
        )

    def _v3_scratch_cache(self, hidden_states, pool):
        """One step of packed cache for MlaPrologV3 on the fused 576-wide pool.

        The kernel assumes ``kv_cache``/``kr_cache`` hold latent and rope in two
        separate packed buffers (``cacheMode="PA_BSND"`` addresses row
        ``cache_index[t]`` of a flat ``(rows, headSize)`` view), while the fused
        pool interleaves both per token, so the prologue writes a throw-away
        one-step cache here and the rows are landed in the pool afterwards.

        The page dimension below is only there to satisfy the kernel's
        ``blockSize`` check (16..1024, 16-aligned); it does not have to match the
        pool's page size because the addressing is flat.
        """
        if pool.store_dtype != pool.dtype:
            raise NotImplementedError(
                "MlaPrologV3 on the unified MLA KV layout requires an unquantised "
                "KV pool (store_dtype %s != dtype %s)"
                % (pool.store_dtype, pool.dtype)
            )
        page_size = getattr(pool, "page_size", 0) or 0
        if page_size < 16 or page_size > 1024 or page_size % 16:
            page_size = 128
        num_tokens = hidden_states.shape[0]
        num_blocks = (num_tokens + page_size - 1) // page_size
        device = hidden_states.device
        dtype = pool.store_dtype
        kv_cache = torch.empty(
            (num_blocks, page_size, 1, self.kv_lora_rank),
            dtype=dtype,
            device=device,
        )
        kr_cache = torch.empty(
            (num_blocks, page_size, 1, self.qk_rope_head_dim),
            dtype=dtype,
            device=device,
        )
        cache_index = torch.arange(num_tokens, dtype=torch.int64, device=device)
        return kv_cache, kr_cache, cache_index

    def forward(self, positions, hidden_states, forward_batch, zero_allocator):
        # assert self.quant_config and self.quant_config.get_name() == "modelslim"
        # route by `qkv_a_proj` quant type as MTP layers can be unquantized
        _is_w8a8 = (
            hasattr(self.qkv_a_proj.quant_method, "quantization_config")
            and self.qkv_a_proj.quant_method.quantization_config.get_name()
            == "modelslim"
        )
        # with the mlaprolog enabled, the kv_b_proj layers are unquantized
        _is_mlaprolog = hasattr(self.quant_config, "ignore") and any(
            re.fullmatch(r".*kv_b_proj", l) for l in self.quant_config.ignore
        )
        if is_mlaprolog_v3_enabled():
            return self.forward_mlaprolog_v3(
                positions, hidden_states, forward_batch
            )
        if _is_w8a8:
            return self.forward_mlapo(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif _is_mlaprolog:
            return self.forward_mlaprolog(positions, hidden_states, forward_batch)
        else:
            return self.forward_absorb_prepare_npu_rms_norm_cache(
                positions, hidden_states, forward_batch, zero_allocator
            )
