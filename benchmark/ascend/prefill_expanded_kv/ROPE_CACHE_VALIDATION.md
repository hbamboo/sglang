# Decode RoPE 输入缓存：opt/p_kv_shared_tune 验证

优化已移植到 hxd-test 容器的 `/sgl-workspace/sglang-p-kv-opt`，分支 `opt/p_kv_shared_tune`，基线为 `31fcca23560fb3f1d3427f6e62b398489fa95c30`。基线由该 HEAD 建立的独立工作树运行，候选使用目标分支工作区；两组均重新加载完整模型、重新捕获图。

## 变更

GLM 每次模型 forward 初始化父、TBO 子批次的 RoPE 结果缓存；V3 分支在同一次 forward 内按 RoPE 实例及 positions Tensor 复用展开后的连续 cos/sin。图捕获保留一次准备链，每次 replay 根据更新后的 positions 刷新共享结果。TBO 字段拆分显式清空新字段。

每层按 device 和 2 的幂容量缓存只读 INT64 scratch 索引。图 warmup 初始化之后，capture/replay 只取连续前缀 view；保留旧容量的存储，保证旧图有效。已有 decode-only V3 路由、统一 576 维 latent KV 传输和 P 的展开 KV/融合 sink 保持原有语义。

## 环境与方法

- Ascend950DT / CANN 9.1.0，物理 5、6 为两个 TP1 prefill，7 为 TP1 decode。
- 模型 `/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8`，modelslim；BF16 latent KV，无 MTP。
- P：expanded KV=1、shared overlap=1、threshold=8192、transient reserve=5120 MiB；FlashMLA=0、MLAPO_V3=0、统一 KV=1、DISABLE_YARN_KV_SINK=0。
- D：FlashMLA=1、MLAPO_V3=1、expanded KV/shared overlap/reserve=0。max graph BS64。
- 两侧显存比例 0.90、max running64、显式拆分共享专家；P max/chunk prefill tokens=32768。
- 数据集 `CHAT-mxfp8-customer-2000.jsonl`，seed42，thinking=true。每组先准备100请求，然后固定并发8运行两轮、每轮100请求，每轮预热后 flush 两台 P。400 条正式测量请求、200 条准备请求均成功，两组逐请求输入/输出长度一致。各轮均有真实多请求 prefill batch，新 token 未超过32768。
- 计时期间关闭 profiler；质量和 profiling 独立执行。采用顺序 A/B，未做随机交叉顺序；固定并发属于闭环负载，两组实际 QPS 不相同。

## 正确性与回归

48 个唯一回归用例通过：首轮41通过、7项硬件 opt-in 跳过，随后启用 `P_KV_TEST_DEVICE=npu:0` 补跑展开 KV 测试文件10项全部通过，补齐全部7项。包括16项新增缓存测试，以及已有 decode MLA、统一池、原融合 sink 和展开 KV 回归。

10 个固定质量用例覆盖短文本、1K/8K/16K/32K上下文，各输出16 token。优化前后 **160个 token 的 ID 和 logprob 全部完全一致**。这次比较的两组均已启用 V3，未改变既有 BF16 投影计算路径。

两组请求阶段没有 RuntimeError、IndexError、KVTransferError 或传输线程失败。停服按 SIGTERM 标记分段：各 worker 都已排空请求；两组退出时有 SystemExit:0/CancelledError，基线两台 P 另有既有 uvloop 退出 segfault，候选没有。退出期诊断与请求期运行错误分开保留，未宣称修复退出问题。

`git diff --check` 通过；新增测试使用仓库指定 black26.1.0/isort7.0.0 格式化，并核对 AST 不变。仓库全局注册检查在基线和候选均报告相同3个旧测试缺少 CI 注册（decode_mlaprolog_v3、prefill_expanded_kv、prefill_transient_reserve）；新增测试位于 manual，未增加注册错误。

## 实际服务性能

| 指标 | 基线 | 优化后 | 变化 |
|---|---:|---:|---:|
| 平均 TPOT / ms | 8.376 | 7.645 | -8.72% |
| 平均 TTFT / ms | 315.474 | 317.816 | +0.74% |
| 平均 E2E / ms | 549.993 | 531.881 | -3.29% |
| 实际 QPS | 14.179 | 14.615 | +3.08% |

TPOT 两轮分别为：基线 8.187/8.565 ms；候选 7.611/7.679 ms。
TTFT 两轮分别为：基线 322.701/308.248 ms；候选 323.168/312.465 ms。

本次实测平均 TPOT 下降约8.72%；TTFT增加约0.74%，两项均完整报告。结果适用于上述配置和负载，不能替代其他并发、上下文或拓扑的测量。

## 完整模型 decode profiling

来自真实 SGLang 服务的独立 CPU+NPU 采集，record_shapes=true、with_stack=false、48列 kernel CSV，op/api statistic 与 trace JSON 完整。每组对8个请求各生成32 token，完整 decode annotations 的活跃 token 数均为256。两份 trace 的调度 batch 分布不同，因此不直接比较总 kernel 数或总设备时长。

| 图 token bucket | 基线 / 候选步数 | 准备 kernel 中位时间 us | 捕获图 span 中位时间 ms |
|---|---:|---:|---:|
| 1 | 2 / 60 | 397.862 → 8.665 | 7.815 → 7.212 |
| 2 | 10 / 3 | 558.624 → 12.802 | 8.465 → 7.993 |
| 4 | 1 / 10 | 572.103 → 12.898 | 9.493 → 8.986 |
| 8 | 32 / 27 | 598.044 → 12.513 | 11.052 → 10.700 |

bucket8 是图的 padding 后大小，实际活跃请求数可能为5/6/7/8。上表是对应图 bucket 的观测值；两组 step 没有按活跃 batch/上下文逐步配对，不能仅将 span 差值归因于本次改动。

每一步均经过同流 norm→MLA 区间的严格序列校验：

| 每个47层 decode step 的准备 kernel | 基线 | 候选 |
|---|---:|---:|
| BS≥2：Index | 47 | 1 |
| BS≥2：Slice | 94 | 2 |
| BS≥2：Tile | 94 | 2 |
| BS≥2：Arange | 47 | 0 |
| BS≥2：合计 | **282** | **5** |
| BS=1：合计（原本无需Slice） | **188** | **3** |

基线45个 decode step、2115次 MLA prolog，候选100个 step、4700次 MLA prolog；各自 FlashMla 次数与 MLA prolog 完全相等。所有候选 step 的非首层 norm 后都没有这组准备 kernel，首层仅保留一次 RoPE 准备。

## Profiling 与复现

原始数据保存在 hxd-test 的 `/sgl-workspace/rope-cache-shared-validation/`，包含 baseline/candidate 的服务器日志、配置、逐请求结果、质量输出、profiling，以及 `audit.json` 和 `source-manifest.json`。

下载包 `decode-profiling-opt-p_kv_shared_tune.zip` 包含两组完整 `ASCEND_PROFILER_OUTPUT`、profiler配置、逐层窗口与审计；SHA256：`989d4c412e15b0ea6544d4e87b91d87d2d4633e1f0b42ce703229f5a84365423`，大小 91771466 字节。

复现服务使用本分支的 `benchmark/ascend/prefill_expanded_kv/launch_decode_v3_2p1d.sh`；加载 `/mnt/sfs_turbo/hxd/pkgs/flash_mla_delivery/env.sh` 后设置 BASELINE_ROOT/CANDIDATE_ROOT 为当前 checkout，并选择空闲的 P5/6、D7。准确的启动命令和角色环境见原始 `status.json`；用户指定分支的实际 worker PYTHONPATH 已记录。

回归命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 P_KV_TEST_DEVICE=npu:0 \
PYTHONPATH=/sgl-workspace/sglang-p-kv-opt/python:/opt/flash_mla_pylibs \
python3 -m pytest -q \
  test/manual/ascend/test_mlaprolog_v3_input_cache.py \
  test/registered/ascend/test_decode_mlaprolog_v3.py \
  test/registered/ascend/test_prefill_expanded_kv.py
```

本次两组测试进程组均已退出；没有停止之后新占用卡的其他任务。原 `/sgl-workspace/sglang` 实验工作区未用于此次目标分支验证。
