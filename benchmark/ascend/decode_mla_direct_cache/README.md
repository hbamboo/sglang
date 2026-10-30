# Decode MLA prolog: direct latent-cache writes

This opt-in change removes the KV `Cat` and the following standalone
`ScatterNdUpdate` before FlashMLA. **The Query Cat remains unchanged.**

The previous unified-cache path writes V3's outputs into step-sized packed
512/64 scratch buffers, concatenates their rows and scatters them into the final
576-wide paged latent cache. The direct path passes the final cache's 512/64
component views and the real token slots to the existing V3 call. The native
writer honours their physical row and page strides. All projection, norm, RoPE
and Query calculations retain the existing implementation.

## Enable

Set on the **decode worker only**:

```bash
ENABLE_ASCEND_FLASH_MLA=1
SGLANG_NPU_USE_MLAPO_V3=1
SGLANG_NPU_MLAPO_V3_DIRECT_CACHE=1
```

The new flag defaults to off. It requires an updated `sgl-kernel-npu` whose
direct module advertises `mla_prolog_v3_direct_cache_version >= 1`; enabling it
with an older library fails explicitly before launching an invalid write.
The current fast path supports unquantized BF16, PA_BSND, one KV head and the
BF16 split-N V3 route used by this GLM configuration. Unsupported physical
layouts are rejected rather than silently treated as packed tensors.

The P worker and PD transport remain unchanged. Only latent KV is stored on D;
no expanded K/V cache is introduced. The final cache dimensions, allocation,
page table, slot mapping and PD registration descriptors stay the same.

## Native implementation

Two physical row-stride fields are appended to V3's private tiling data. Zero
preserves legacy packed addressing. The host validates the logical cache
shapes, then supplies actual page and row strides for supported strided views.
The complete resulting tiling blob is hashed by the existing device-tiling
cache, so packed and strided layouts cannot alias during graph replay.

The native patch changes cache address calculation and component writes only.
The original contiguous-cache path and all Query output code remain intact.
The existing prolog operator schema is unchanged.

## Regression

```bash
ASCEND_RT_VISIBLE_DEVICES=6 \
PYTHONPATH=/path/to/private/sgl-kernel-npu/python/sgl_kernel_npu:$PWD/python \
python3 -m pytest -q \
  test/registered/ascend/test_mlaprolog_v3_direct_cache.py \
  test/registered/ascend/test_decode_mlaprolog_v3.py
```

The tests compare packed-cache and direct-cache outputs bitwise, cover
permuted slots and page boundaries, preserve untouched slots and other layers,
verify the PD descriptor and transfer wait, assert the scratch/landing path is
bypassed and replay a captured graph with updated real slot indices. Existing
decode scope and prefill sink tests remain included.

## Build isolation

Build the updated native source in a separate checkout with the target CANN
and SoC settings. For this validation the source and private Python package are
under `/sgl-workspace/sgl-kernel-npu-direct-cache`; D's `PYTHONPATH` selects that
package. The globally installed library and CANN SDK are not modified.
Turning the feature off retains the existing scratch path.

Timed 1P1D runs and D-only profiler captures are kept separate. The acceptance
trace must retain V3 and FlashMLA, remove the KV Cat and standalone scatter,
and retain one Query Cat per attention layer per recorded decode step.

For the validated Ascend950 / CANN 9.1.0 environment:

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
NATIVE_ROOT=/sgl-workspace/sgl-kernel-npu-direct-cache
cmake -S "$NATIVE_ROOT" -B "$NATIVE_ROOT/build" \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.1.0/aarch64-linux/include \
  -DSOC_VERSION=Ascend950PR_9599 \
  -DBUILD_CATLASS_MODULE=OFF -DBUILD_DEEPEP_MODULE=OFF \
  -DBUILD_KERNELS_MODULE=ON -DBUILD_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX="$NATIVE_ROOT/output"
cmake --build "$NATIVE_ROOT/build" --target sgl_kernel_npu -j8
export PYTHONPATH="$NATIVE_ROOT/python/sgl_kernel_npu:$PYTHONPATH"
```

The private library must be built and selected before enabling the flag. Hardware
regressions that require stride support skip when an old native module is loaded;
the production flag still fails explicitly with an old module.

See [validation.md](validation.md) in the SGLang benchmark directory for the
measured 1P1D results and trace counts.

Native companion commit: [sgl-kernel-npu f6dd0b163dd1](https://github.com/hbamboo/sgl-kernel-npu/commit/f6dd0b163dd1d90fa9d7a341bf8df28fa7271058).
