"""Ascend HiCache host storage for split and unified MLA KV pools."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Optional, Sequence

import torch

from sglang.srt.environ import envs

from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import get_bool_env_var


def _transfer_kv_pages(device, host, device_indices, host_indices, page_size, h2d):
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_dim_exchange

    # The native implementation reads sizes()[4] even for an empty V. Its
    # optional-index wrapper uses a 1-D empty tensor, so transfer each component
    # as K instead of using that wrapper's optional index arguments.
    empty = torch.empty((0, 0, 0, 0, 0), dtype=host.dtype)
    transfer_kv_dim_exchange(
        device_k=device,
        host_k=host,
        device_v=empty,
        host_v=empty,
        device_indices=device_indices,
        host_indices=host_indices,
        page_size=page_size,
        direction=TransferDirection.H2D if h2d else TransferDirection.D2H,
    )


class NPUMLATokenToKVPoolHost(MLATokenToKVPoolHost):
    """Page-major host payloads, using Ascend's all-layer async page copies.

    Each host page is one contiguous allocation containing all KV components.
    A unified MLA device buffer is copied as one component, never through its
    non-contiguous latent/rope views. Legacy split KV and an optional indexer
    are stored in separate, contiguous sections *within* the same host page.
    This also gives storage backends one complete payload/pointer per page.
    """

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        override_kv_cache_dim: Optional[int] = None,
        mtp_draft_device_pools: Sequence = (),
        dcp_size: int = 1,
        dcp_rank: int = 0,
        *,
        pool_label: str = "kv",
    ):
        if layout != "page_first_kv_split":
            raise ValueError("Ascend MLA HiCache requires layout='page_first_kv_split'")
        if page_size != device_pool.page_size:
            raise ValueError("Ascend HiCache host/device page sizes must match")
        if dcp_size != 1 or dcp_rank != 0 or device_pool.layer_shard_enabled:
            raise ValueError("Ascend MLA HiCache does not support DCP or layer sharding")
        if mtp_draft_device_pools:
            raise ValueError("Ascend MLA HiCache does not support packed MTP pools")
        if getattr(device_pool, "prefill_expanded_geometry", None) is not None:
            raise ValueError("Ascend MLA HiCache cannot restore expanded prefill KV")
        self._cache_layout_flags = (
            get_bool_env_var("SGLANG_USE_FIA_NZ"),
            envs.SGLANG_NPU_DISABLE_YARN_KV_SINK.get(),
            get_bool_env_var("ENABLE_QKV_NORM_ROPE_OPT"),
        )
        self.mtp_draft_device_pools = ()
        self.unified_kv = bool(getattr(device_pool, "unified_kv", False))
        width = device_pool.kv_lora_rank + device_pool.qk_rope_head_dim
        if override_kv_cache_dim is not None and override_kv_cache_dim != width:
            raise ValueError("Ascend MLA HiCache requires the complete latent/rope KV")
        self._component_names = ("kv_buffer",) if self.unified_kv else (
            "k_buffer",
            "v_buffer",
        )
        if getattr(device_pool, "index_k_buffer", None) is not None:
            self._component_names += ("index_k_buffer",)
        for name in self._component_names:
            buf = getattr(device_pool, name)
            if (
                not isinstance(buf, torch.Tensor)
                or buf.ndim != 5
                or not buf.is_contiguous()
                or buf.shape[0] != device_pool.layer_num
                or buf.shape[2:4] != (page_size, 1)
                or buf.dtype != device_pool.store_dtype
            ):
                raise ValueError(f"Unsupported Ascend MLA HiCache device buffer {name}")
        self.can_use_jit = False
        # Keep the MLA type for storage-backend dispatch, but bypass its CUDA
        # pointer-array initialization and its layer-first buffer assumptions.
        HostKVCache.__init__(
            self,
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            pool_label=pool_label,
        )

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.target_layer_num = self.layer_num = self.device_pool.layer_num
        self.kv_cache_dim = sum(
            getattr(self.device_pool, name).shape[-1] for name in self._component_names
        )
        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num
        return self.layout_dim

    def get_ksize_per_token(self):
        return self.size_per_token

    def init_kv_buffer(self):
        page_elements = self.layer_num * self.page_size * self.kv_cache_dim
        device_type = str(self.device_pool.device).split(":")[0]
        buffer = ALLOC_MEMORY_FUNCS[device_type](
            (self.page_num, page_elements),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
            registration_granularity_bytes=page_elements * self.dtype.itemsize,
        )
        self._host_components = []
        offset = 0
        for name in self._component_names:
            width = getattr(self.device_pool, name).shape[-1]
            count = self.layer_num * self.page_size * width
            view = buffer[:, offset : offset + count].view(
                self.page_num, self.layer_num, self.page_size, 1, width
            )
            self._host_components.append(view)
            if name != "kv_buffer":
                setattr(self, name, view)
            else:
                self.k_buffer = view[..., : self.kv_lora_rank]
                self.v_buffer = view[..., self.kv_lora_rank :]
            offset += count
        if "index_k_buffer" not in self._component_names:
            self.index_k_buffer = None
        return buffer

    @staticmethod
    def _page_indices(indices, page_size, capacity, label):
        if indices.ndim != 1 or indices.dtype != torch.int64:
            raise ValueError(f"Ascend HiCache {label} indices must be a 1-D int64 tensor")
        indices = indices.to(device="cpu").contiguous()
        if indices.numel() % page_size:
            raise ValueError("Ascend HiCache transfers require complete pages")
        pages = indices.view(-1, page_size)
        if pages.numel() and (
            (pages[:, 0] % page_size != 0).any()
            or not torch.equal(pages, pages[:, :1] + torch.arange(page_size))
            or (indices < 0).any()
            or (indices >= capacity).any()
        ):
            raise ValueError(f"Invalid Ascend HiCache {label} page indices")
        return indices

    def _transfer(self, device_pool, host_indices, device_indices, io_backend, h2d):
        if io_backend != "kernel_ascend":
            raise ValueError("Ascend MLA HiCache requires io_backend='kernel_ascend'")
        if device_pool is not self.device_pool:
            raise ValueError("Ascend MLA HiCache cannot transfer a different device pool")
        if host_indices.numel() != device_indices.numel():
            raise ValueError("Ascend HiCache host/device index lengths must match")
        host_indices = self._page_indices(
            host_indices, self.page_size, self.size, "host"
        )
        device_capacity = getattr(device_pool, self._component_names[0]).shape[1]
        device_indices = self._page_indices(
            device_indices, self.page_size, device_capacity * self.page_size, "device"
        )
        if host_indices.numel() == 0:
            return
        for name, host in zip(self._component_names, self._host_components):
            _transfer_kv_pages(
                getattr(device_pool, name),
                host,
                device_indices,
                host_indices,
                self.page_size,
                h2d,
            )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        self._transfer(device_pool, host_indices, device_indices, io_backend, False)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if is_draft:
            raise ValueError("Ascend MLA HiCache does not support packed MTP loads")
        if not 0 <= layer_id < self.layer_num:
            raise ValueError(f"Invalid Ascend HiCache local layer {layer_id}")
        # All layers are enqueued on the transfer stream before the layer-0
        # completion event. Later layer events remain ordered after these copies.
        if layer_id == 0:
            self._transfer(device_pool, host_indices, device_indices, io_backend, True)

    def _page_id(self, index):
        if index < 0 or index % self.page_size or index >= self.size:
            raise ValueError(f"Invalid Ascend HiCache host page start {index}")
        return index // self.page_size

    def get_data_page(self, index, flat=True):
        page = self.kv_buffer[self._page_id(index)]
        return page if flat else page.view(1, -1)

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            self.kv_buffer.shape[1],
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def set_from_flat_data_page(self, index, data_page):
        page = self.get_data_page(index)
        if data_page.numel() != page.numel() or data_page.dtype != self.dtype:
            raise ValueError("Ascend HiCache storage page size/dtype mismatch")
        page.copy_(data_page.reshape(-1))

    def get_page_buffer_meta(self, indices):
        indices = self._page_indices(indices, self.page_size, self.size, "host")
        pages = indices[:: self.page_size].tolist()
        return (
            [self.get_data_page(index).data_ptr() for index in pages],
            [self.kv_buffer.shape[1] * self.dtype.itemsize] * len(pages),
        )

    def get_hybrid_pool_buffer(self):
        return [self.kv_buffer]

    @property
    def storage_format(self):
        # Equal-sized unified and split pages have different byte ordering.
        # Include every physical-format discriminator so neither layout can
        # consume the other's persisted pages (or old CUDA/split payloads).
        components = tuple(
            (name, getattr(self.device_pool, name).shape[-1])
            for name in self._component_names
        )
        signature = repr(
            (
                str(self.dtype),
                self.page_size,
                self.layer_num,
                self.start_layer,
                self.end_layer,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self._cache_layout_flags,
                components,
            )
        )
        return "ascend-mla-v1-" + hashlib.sha256(signature.encode()).hexdigest()[:16]

    def prepare_storage_config(self, backend_name, storage_config):
        if backend_name != "file":
            raise NotImplementedError(
                "Ascend MLA HiCache currently supports only the 'file' L3 backend; "
                "omit --hicache-storage-backend for device/host caching"
            )
        return replace(
            storage_config,
            model_name=f"{storage_config.model_name or ''}--{self.storage_format}",
        )

    def get_contiguous_buf_infos(self):
        # This is storage registration of the complete host allocation. It is
        # not the per-layer PD direct-to-host ABI of NPUMLATokenToKVPool.
        return (
            [self.kv_buffer.data_ptr()],
            [self.kv_buffer.nbytes],
            [self.kv_buffer.shape[1] * self.dtype.itemsize],
        )

    def is_stride_page_aligned(self, page_size_bytes=4096):
        return (
            self.kv_buffer.data_ptr() % page_size_bytes == 0
            and self.kv_buffer.stride(0) * self.dtype.itemsize % page_size_bytes == 0
        )

    def destroy(self):
        super().destroy()
        self._host_components = []
        self.k_buffer = self.v_buffer = self.index_k_buffer = None
