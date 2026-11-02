# Qualified codex-1031 r97 runtime snapshot

Runtime source commit: `f4ea39f662ab6fff24c9f4cc9fd3b1431d3969bf`.

The wheel and runtime file manifest freeze the exact already-qualified image bytes. The local package version retains its original packaging baseline identifier. This commit adds only immutable release artifacts; it does not change model execution. For exact runtime restoration, verify the wheel SHA256 from release-lock.json and install this wheel with `python3 -m pip install --no-index --no-deps --force-reinstall sglang-0.5.19+codex1031.g73a2cbb6-py3-none-any.whl`.

Full performance reproduction additionally requires the central release-lock repository: CANN/ATB, the actual Triton-Ascend installation, memfabric-hybrid, fastokens/router, the original compiler cache, and the pinned startup/benchmark protocol. The source commit is not a bit-rebuild claim.
