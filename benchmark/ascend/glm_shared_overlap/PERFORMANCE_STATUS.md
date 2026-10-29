# GLM-4.7-Flash 三阶段多流：性能与验证报告

状态：实现、NPU 数值回归、44 组组件筛选和真实客户负载 profiling 已完成。当前推荐配置在一轮完整客户测试中平均 TTFT 降低 4.12%；反向运行顺序的复测及完整 GSM8K 对照尚未结束，整个目标仍在执行。

## 实现与范围

目标环境是 `root@113.46.12.25` 的 `hxd-glm`，物理 NPU 5，芯片 Ascend950DT_9582。权重 `/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8`。仓库 `/sgl-workspace/sglang`，分支 `feat/glm-npu-shared-overlap`，作者 huangxudong。

主流执行路由专家；共享流严格执行三个阶段：gate/up 与 TopK/init routing 并行；等待 routed GMM1 后执行 SwiGLU+MX quant，与 GMM2 并行；等待 GMM2 后执行 down，与 finalize 并行。主流在最终加法前等待 shared-done event。显式 event、输入/输出 record_stream 和异常清理保护依赖及内存生命周期。

路径限普通 BF16 prefill、TP=EP=1、独立共享专家、原生 Ascend MXFP8 量化；decode 与不满足条件的路径保留原实现。功能默认关闭。容器预先存在的七个运行时修改保持原样，没有纳入优化提交。

推荐环境变量：

```bash
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
SGLANG_NPU_SHARED_EXPERT_OVERLAP=1
SGLANG_NPU_SHARED_EXPERT_OVERLAP_MIN_TOKENS=8192
```

基线使用相同 allocator，关闭 overlap。提前量化开关为 0，副流 cube/vector 上限为 -1。门槛针对实际调度出的每次 prefill；小尾批次回退串行。门槛的代码默认值仍为 0，推荐部署显式设置 8192。

## 主要性能负载

使用原始 `/mnt/sfs_turbo/hxd/CHAT-mxfp8-customer-2000.jsonl`，SHA256 `bb722901ab4a6857571837bbef58b30a4ca00ce08c1aaf170ada3c94694d0f0a`。直接使用 `conversations[0].content`，2000 条请求共 7566887 输入 token，保持文件顺序，不把请求拼成一条 32K 文本。

客户端形成 248 个受控多请求 burst，每组不超过 32768 个新 token、32 条请求；实际每组 21333..32743 token，最多 27 条请求。每请求严格输出 1 token，greedy、ignore_eos，prefix cache 关闭。每个新服务先预热 3 组，profiling 与计时分开。TTFT 计至首个包含生成 token 的 SSE 数据包。该负载是受控 burst，并非开放到达率或最大 QPS 测试。

服务参数：TP=1，mem-fraction-static=0.80，context-length=40960，max-prefill-tokens=chunked-prefill-size=32768，max-running-requests=32，disable-radix-cache、disable-cuda-graph、disable-shared-experts-fusion，attention-backend=ascend，ASCEND_USE_FIA=1。

### 相同 allocator 下的完整测试

| 指标 | 串行基线 | 三阶段全量 prefill | 三阶段 + 8192 门槛 |
| --- | ---: | ---: | ---: |
| 平均 TTFT | 899.505 ms | 868.400 ms | 862.426 ms |
| P50 TTFT | 915.780 ms | 890.071 ms | 882.213 ms |
| P95 TTFT | 1012.993 ms | 984.429 ms | 980.298 ms |
| 全部 burst 总耗时 | 232.151 s | 224.632 s | 223.106 s |
| 请求吞吐 | 8.6151 req/s | 8.9034 req/s | 8.9644 req/s |
| 变快的配对请求组 | — | 246/248 | 248/248 |
| 首 token 一致性 | — | 2000/2000 | 2000/2000 |

推荐门槛版平均 TTFT 降低 4.122%，P95 降低 3.227%，burst 总耗时降低 3.896%。配对组平均节省 36.474 ms，中位节省 30.439 ms，按组 bootstrap 95% 区间 [34.372,38.676] ms。全部样本都保留，未剔除慢样本。该区间仅反映这轮请求组的变异，不包含不同运行次序的方差；正在按“优化 → 基线”反向复测。

旧的默认 allocator 新服务对照平均 TTFT 891.379→887.049 ms，但 P95 1008.688→1058.228 ms，完整耗时几乎持平。原始数据仍保留，不能把这轮称为稳定收益。新实验双方都改用相同 allocator，不能由此断言 allocator 是旧慢批次的已证实根因，也不能把旧基线与新优化直接相除当作收益。

单条 32768-token、1-output 微基准中位 TTFT 2578.401→2567.290 ms，降低 0.431%。这是历史诊断数据，不替代用户指定的多请求验收。

## Profiling 证据

相同客户首组为 27 请求、32467 token，实际调度拆成 30861 和 1606 token。每版采集 5 次，丢掉前 2 次，剩余 3 次各 46 个 MoE 层实例。

| 实际 prefill | 串行 MLP 中位数 | 全量并行 MLP 中位数 | 门槛版 MLP 中位数 |
| --- | ---: | ---: | ---: |
| 30861 token | 6393.133 us | 5797.436 us | 5794.706 us |
| 1606 token | 1209.750 us | 1577.149 us | 1281.055 us |

大批次 138/138 层实例的三个阶段 overlap_pct 都达到 0.5，三个阶段中位数均为 1.0。小尾批次没有足够的工作掩盖跨流提交开销，门槛版直接回退。MLP 窗口按 post-attention AddRmsNorm 结束至最终 shared+routed Add 结束计算；原始 5 次及 selected 标记都保存在逐层 CSV。

并发仍拉长主流：大批次 init routing / GMM2 / finalize 平均从 1377/956/343 us 增至 1447/1057/404 us。收益来自整个 MLP 窗口变短，不能把三个阶段毛重叠相加当成 TTFT 节省。提前量化和 stream 限核的 44 组组件实验全部数值一致，但没有稳定胜过原调度，保留为默认关闭的实验选项。

## 精度验证

真实 NPU 共享投影测试使用 GLM 的 2048→3072→1536→2048 维度，覆盖 1/128/1024/32768 token、重复复用、非默认父流、allocator churn、异常清理和 decode fallback。共享输出逐位一致，最大绝对误差为 0。组件筛选的 44 个 shape/config 组合也全部逐位一致。真实客户测试 2000 个首 token 全部一致。

完整 GSM8K 对照仍待完成：本地 test 1319 题、固定 train 前 4 个示例、greedy、最多 8192 新 token，EvalScope extract_answer/math_equal 判分。最终两组使用相同的 full decode graph、并发 32，prefill graph 关闭；与旧并发 16/eager 的 100 条基线、44 条优化 journal 分开。每条输出、prompt SHA256、完成原因和截断均保留。质量运行的延迟不用于客户性能结论。

## 交付与复算

远端结果根目录 `/mnt/paas/hxdnpu/glm-shared-overlap-20261030`，本地 `C:/Users/Administrator/Desktop/workspace/glm-shared-overlap`。实现和实验脚本保存在同一分支，提交作者为 huangxudong。

- `compare_customer.py`：全部 2000 请求 / 248 组配对，无异常值剔除。
- `analyze_moe_windows.py`：五次 capture 的逐层窗口及真实 overlap_pct。
- `compare_gsm8k.py`：完整 1319 题、逐样本 prompt/答案/截断审计。
- `TUNING_ROUND2.md`：候选、资源配额和选择依据。
- `results/*-comparison.json`、`*-paired-packs.csv`、`*-mlp-analysis.json`：可复算结果。
- 最终 profiling 包将包含 matched baseline 与门槛版的 trace、CSV、profiler metadata 及 SHA256 manifest。

整个目标仅在反向性能复测、两组完整 GSM8K 与交付审计完成后才结束。
