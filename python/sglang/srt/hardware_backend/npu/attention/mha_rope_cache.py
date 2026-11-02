"""Per-forward cross-layer cache for MHA (deepseek-YARN) RoPE cos/sin.

The MHA prepare path in ``deepseek_v2_attention_mla_npu.py`` asks the shared
``DeepseekScalingRotaryEmbedding`` for cos/sin on every layer.  That lookup
performs two gathers into the FP32 source tables plus two casts, even though
every layer of one model forward sees the same rotary object, positions and
requested dtype.  This module reuses those tensors for the remainder of the
forward.

Lifecycle and support scope
---------------------------
* Opt-in only: ``SGLANG_NPU_MHA_ROPE_REUSE=1`` (read once at import, matching
  the project's boolean-env convention).
* The cache is stored *inside* the existing
  ``ForwardBatch.npu_mlaprolog_rope_cache`` dictionary.  The model forward
  (``glm4_moe_lite.py``) replaces that dictionary at every Python forward,
  which is what bounds the lifetime.  This module never creates a process-wide
  cache; when the attribute is missing or ``None`` it falls back.
* Supported route: eager NPU inference only.  ``torch.compile`` and NPU graph
  capture both fall back to the original lookup because warmup results cannot
  be reused and a replay must still execute its graph-side gather.
* Inference tensors are supported on the enabled eager path.  They have no
  ``_version`` counter, so the key records an explicit inference marker plus
  the framework's per-forward immutability precondition (positions and source
  tables are not mutated within one model forward, and the model replaces the
  dictionary before every step/capture invocation).  No fabricated version 0
  is used.  A caller that must mutate an inference tensor in place within a
  forward must call :func:`invalidate_mha_rope_cache` first; arbitrary
  mutation detection is not claimed.
  ``_version`` also does not observe ``.data.copy_`` on normal tensors, so the
  per-forward reset remains the invalidation boundary for that operation.
* Cache identity keeps live object references (not only a bare ``id``) for
  the rotary module, both FP32 source tables and the positions/offsets
  tensors, together with mutation markers (normal ``_version`` counters or
  the explicit inference marker), requested dtype/device and the current NPU
  stream.  Different views, table replacement, normal in-place mutation, a
  different dtype/device/offsets or a different stream therefore miss instead
  of reusing stale values.
* On a hit the exact cached ``(cos, sin)`` tensors are returned: no gather,
  cast, reshape or copy.  On a miss the original
  ``rotary.get_cos_sin_cache(positions, dtype, offsets=offsets)`` is called
  exactly once and its return value is cached unchanged, preserving the
  current ``[T, 1, 1, 64]`` shape, dtype, stride and rounding.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

#: ForwardBatch attribute that the model resets at every Python forward.
CACHE_ATTR = "npu_mlaprolog_rope_cache"

_ENV_VAR = "SGLANG_NPU_MHA_ROPE_REUSE"
# Read once at import.  Truthy spellings intentionally match
# sglang.srt.utils.get_bool_env_var("1"/"true") without importing the heavy
# sglang package from this leaf module.
_ENABLED = os.environ.get(_ENV_VAR, "").strip().lower() in ("1", "true")

# Private namespace inside the shared per-forward dict.  MLA-preprocess entries
# use ``(id(rotary_emb), positions)`` tuple keys and different cos/sin math;
# this sentinel keeps the two schemes from ever colliding.
_NAMESPACE = object()
_UNKNOWN_STREAM = object()
# Explicit "this tensor is immutable-by-forward-contract" marker.  It is a
# unique object, never an integer, so it can never be confused with version 0.
_INFERENCE_MARKER = object()

__all__ = [
    "CACHE_ATTR",
    "get_mha_rope_cos_sin",
    "get_or_compute_cached",
    "invalidate_mha_rope_cache",
    "is_enabled",
]


def is_enabled() -> bool:
    """Whether the opt-in reuse path is enabled for this process."""
    return _ENABLED


def _original(rotary_emb, positions, dtype, offsets):
    """Call the pre-existing lookup exactly as the MHA prepare path did."""
    return rotary_emb.get_cos_sin_cache(positions, dtype, offsets=offsets)


def _is_compiling() -> bool:
    """Dynamo guard, checked before any tensor metadata or stream query."""
    compiler = getattr(torch, "compiler", None)
    fn = getattr(compiler, "is_compiling", None)
    if fn is None:
        # Cannot prove the runtime is eager: do not reuse.
        return True
    return bool(fn())


def _positions_device_supported(positions) -> bool:
    return isinstance(positions, torch.Tensor) and positions.device.type == "npu"


def _is_stream_capturing() -> bool:
    """NPU capture guard.  Unknown/absent probing answers mean no reuse."""
    npu = getattr(torch, "npu", None)
    fn = getattr(npu, "is_current_stream_capturing", None)
    if fn is None:
        return True
    try:
        return bool(fn())
    except Exception:
        # This is an introspection-only probe (no kernels submitted).  If the
        # backend cannot answer, conservatively disable reuse.
        return True


def _current_stream(positions):
    npu = getattr(torch, "npu", None)
    fn = getattr(npu, "current_stream", None)
    if fn is None:
        return _UNKNOWN_STREAM
    try:
        return fn(positions.device)
    except Exception:
        return _UNKNOWN_STREAM


def _stream_token(stream):
    """Stable identity for the stream used to produce the cached tensors.

    ``torch.npu.current_stream()`` may return a fresh Python wrapper for the
    same native stream, so compare the native handle when available.  The key
    also keeps the Stream object alive, keeping that handle valid.
    """
    if stream is None:
        return None
    handle = getattr(stream, "npu_stream", None)
    if isinstance(handle, int):
        return handle
    return id(stream)


def _is_inference_tensor(tensor: torch.Tensor) -> bool:
    fn = getattr(torch, "is_inference", None)
    if fn is None:
        # Older runtimes without inference tensors: fall through to _version.
        return False
    return bool(fn(tensor))


def _mutation_marker(tensor: torch.Tensor):
    """Mutation identity for a tensor used by the cache key.

    Normal tensors expose a real ``_version`` counter.  Inference tensors
    deliberately have none (accessing it raises RuntimeError); for them the key
    records ``_INFERENCE_MARKER`` and relies on the documented within-forward
    immutability contract plus :func:`invalidate_mha_rope_cache`.  ``None``
    means no safe marker exists and the caller must fall back.
    """
    if _is_inference_tensor(tensor):
        return _INFERENCE_MARKER
    try:
        return tensor._version
    except (AttributeError, NotImplementedError, RuntimeError):
        return None


def _cache_metadata(rotary_emb, positions, offsets):
    """Return the tables and mutation markers, or ``None`` to fall back.

    Inference positions and inference source tables are supported here; the
    compatibility check is about tensor layout and marker availability, not
    about requiring a version counter.
    """
    if not isinstance(positions, torch.Tensor):
        return None
    if not callable(getattr(rotary_emb, "get_cos_sin_cache", None)):
        return None
    cos_table = getattr(rotary_emb, "cos_cached_total", None)
    sin_table = getattr(rotary_emb, "sin_cached_total", None)
    if not isinstance(cos_table, torch.Tensor):
        return None
    if not isinstance(sin_table, torch.Tensor):
        return None
    if offsets is not None:
        if not isinstance(offsets, torch.Tensor):
            return None

    positions_marker = _mutation_marker(positions)
    cos_marker = _mutation_marker(cos_table)
    sin_marker = _mutation_marker(sin_table)
    offsets_marker = None if offsets is None else _mutation_marker(offsets)
    if positions_marker is None or cos_marker is None or sin_marker is None:
        return None
    if offsets is not None and offsets_marker is None:
        return None
    return (
        cos_table,
        sin_table,
        positions_marker,
        cos_marker,
        sin_marker,
        offsets_marker,
    )


def _cache_inputs_supported(rotary_emb, positions, offsets) -> bool:
    """Check the exact tensor metadata this cache relies on."""
    return _cache_metadata(rotary_emb, positions, offsets) is not None


class _MhaRopeCacheKey:
    """Cache key that holds live references plus the metadata it compares.

    Keeping the tensors/module/stream referenced is deliberate: ``id()`` is
    only used for hashing and a recycled id can never compare equal because
    the original object cannot be collected while its key is in the dict.
    """

    __slots__ = (
        "rotary",
        "cos_table",
        "sin_table",
        "positions",
        "offsets",
        "dtype",
        "device",
        "positions_dtype",
        "offsets_dtype",
        "stream",
        "stream_token",
        "cos_marker",
        "sin_marker",
        "positions_marker",
        "offsets_marker",
        "_hash",
    )

    def __init__(
        self,
        rotary_emb,
        cos_table,
        sin_table,
        positions,
        offsets,
        dtype,
        device,
        stream,
        cos_marker,
        sin_marker,
        positions_marker,
        offsets_marker,
    ):
        self.rotary = rotary_emb
        self.cos_table = cos_table
        self.sin_table = sin_table
        self.positions = positions
        self.offsets = offsets
        self.dtype = dtype
        self.device = device
        self.positions_dtype = positions.dtype
        self.offsets_dtype = None if offsets is None else offsets.dtype
        self.stream = stream
        self.stream_token = _stream_token(stream)
        # Precomputed by _cache_metadata so this class never touches
        # Tensor._version (which would raise for inference tensors).
        self.cos_marker = cos_marker
        self.sin_marker = sin_marker
        self.positions_marker = positions_marker
        self.offsets_marker = offsets_marker
        self._hash = hash(
            (
                id(self.rotary),
                id(self.cos_table),
                id(self.sin_table),
                id(self.positions),
                id(self.offsets),
                self.dtype,
                self.device,
                self.positions_dtype,
                self.offsets_dtype,
                self.stream_token,
                self.cos_marker,
                self.sin_marker,
                self.positions_marker,
                self.offsets_marker,
            )
        )

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        # Tensors are compared by live identity only; every other field is
        # non-tensor metadata (markers are ints or the inference sentinel).
        # Nothing here performs elementwise Tensor equality or boolean tests.
        if not isinstance(other, _MhaRopeCacheKey):
            return NotImplemented
        return (
            self.rotary is other.rotary
            and self.cos_table is other.cos_table
            and self.sin_table is other.sin_table
            and self.positions is other.positions
            and self.offsets is other.offsets
            and self.dtype == other.dtype
            and self.device == other.device
            and self.positions_dtype == other.positions_dtype
            and self.offsets_dtype == other.offsets_dtype
            and self.stream_token == other.stream_token
            and self.cos_marker == other.cos_marker
            and self.sin_marker == other.sin_marker
            and self.positions_marker == other.positions_marker
            and self.offsets_marker == other.offsets_marker
        )


def get_or_compute_cached(
    rotary_emb,
    positions,
    dtype,
    cache,
    offsets=None,
    stream=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure cache logic.

    Used by :func:`get_mha_rope_cos_sin` after its eager/NPU guards, and by the
    CPU tests with a mock rotary and an explicit cache dict.  A missing or
    non-dict ``cache`` cleanly falls back to the original lookup.
    """
    if not isinstance(cache, dict):
        return _original(rotary_emb, positions, dtype, offsets)
    metadata = _cache_metadata(rotary_emb, positions, offsets)
    if metadata is None:
        return _original(rotary_emb, positions, dtype, offsets)

    (
        cos_table,
        sin_table,
        positions_marker,
        cos_marker,
        sin_marker,
        offsets_marker,
    ) = metadata
    key = _MhaRopeCacheKey(
        rotary_emb,
        cos_table,
        sin_table,
        positions,
        offsets,
        dtype,
        positions.device,
        stream,
        cos_marker,
        sin_marker,
        positions_marker,
        offsets_marker,
    )

    namespace = cache.get(_NAMESPACE)
    if isinstance(namespace, dict):
        hit = namespace.get(key)
        if hit is not None:
            # Exact tensors from the original lookup: no Index/Cast/copy.
            return hit

    cos, sin = _original(rotary_emb, positions, dtype, offsets)
    if not isinstance(namespace, dict):
        namespace = {}
        cache[_NAMESPACE] = namespace
    namespace[key] = (cos, sin)
    return cos, sin


def invalidate_mha_rope_cache(forward_batch_or_cache) -> None:
    """Explicitly drop this module's entries for the current forward.

    The normal lifecycle is the model's per-forward dictionary reset.  This
    helper exists for the documented exception: a caller that mutates an
    inference positions tensor or inference source table in place *within* one
    model forward (inference tensors have no ``_version`` counter) must call
    this before asking for cos/sin again.  It removes only this module's
    private namespace, leaving MLA-preprocess entries intact.
    """
    cache = forward_batch_or_cache
    if not isinstance(cache, dict):
        cache = getattr(cache, CACHE_ATTR, None)
    if isinstance(cache, dict):
        cache.pop(_NAMESPACE, None)


def get_mha_rope_cos_sin(
    rotary_emb,
    positions,
    dtype,
    forward_batch=None,
    offsets=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for ``rotary_emb.get_cos_sin_cache`` in MHA prepare.

    Falls back to the original call whenever reuse cannot be proven safe:
    disabled flag, Dynamo tracing, non-NPU positions, NPU graph capture,
    missing/None cache dict, unavailable stream identity, an unexpected table
    layout, or an unreadable mutation marker on a non-inference tensor.
    Inference tensors are supported under the within-forward immutability
    contract; call :func:`invalidate_mha_rope_cache` if that contract is
    explicitly broken in place.
    """
    if not _ENABLED:
        return _original(rotary_emb, positions, dtype, offsets)
    # Must precede version-counter and stream queries so Dynamo never
    # graph-breaks on them while tracing the eager path.
    if _is_compiling():
        return _original(rotary_emb, positions, dtype, offsets)
    if not _positions_device_supported(positions):
        return _original(rotary_emb, positions, dtype, offsets)
    # Checked early: a captured graph must record its own gather to refresh the
    # output on replay, so warmup values must never be served here.
    if _is_stream_capturing():
        return _original(rotary_emb, positions, dtype, offsets)

    cache = getattr(forward_batch, CACHE_ATTR, None)
    if not isinstance(cache, dict):
        return _original(rotary_emb, positions, dtype, offsets)

    stream = _current_stream(positions)
    if stream is _UNKNOWN_STREAM:
        # Without a stream identity, serving a tensor produced on another
        # stream would lack a real dependency.
        return _original(rotary_emb, positions, dtype, offsets)

    return get_or_compute_cached(
        rotary_emb,
        positions,
        dtype,
        cache,
        offsets=offsets,
        stream=stream,
    )
