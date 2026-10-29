# P 展开 KV 优化验证

实现分支：`opt/p_kv_opt`，请求基线 `fc79e7caa00cad115dcc4f78f2277b6c17ee1c15`。
基线启动所需的共同修复为 `bc85c8ef1499140d497d84c8c0cd46920b36d94d`，仅增加缺失的 `envs` import，两组完全相同。

## 验收负载

- 物理 **6 卡 P、7 卡 D，1P1D，TP=1**；按用户后续指定替换原先的 0/1 卡。
- 原始 `CHAT-mxfp8-customer-2000.jsonl`，seed 42，**固定并发 8、正式测量 100 条**。每组先额外跑 100 条预热，客户端在正式阶段再次预热一条并清空 server cache。
- 沿用原版 `sglang.bench_serving` 的 custom 数据集逻辑与 `bench_glm.sh` 的 thinking 选项。未将多条请求拼接成单个 32K 请求。
- `max_prefill_tokens=chunked_prefill_size=32768`，是单批新增 token 上限。两组实测最大均为 26368，最多 7 条请求同批。
- 两组输入长度数组、输出长度数组和文件 SHA256 一致：正式输入共 **384096 tokens**，输出 **2900 tokens**，均 100/100 成功，无客户端错误。
- MXFP8 权重、ModelSlim、BF16 KV、静态显存比例 0.9；P 最大请求数 64，D graph 最大 BS 64，上下文 33024。D 的代码、latent 格式和 FlashMLA 路径相同。

## 结果

平均 TTFT **882.243 → 678.787 ms**，减少 **203.456 ms（23.06%）**。

| 指标 | 基线 | 优化 | 单位 |
| --- | ---: | ---: | --- |
| 平均 TTFT | 882.243 | 678.787 | ms |
| 中位 TTFT | 905.124 | 693.215 | ms |
| P90 TTFT | 1157.704 | 916.336 | ms |
| P99 TTFT | 1302.030 | 979.316 | ms |
| 平均 E2E | 1072.839 | 893.439 | ms |
| P99 E2E | 1501.683 | 1321.922 | ms |
| 请求吞吐 | 7.282 | 8.722 | req/s |
| 平均 TPOT | 6.807 | 7.666 | ms/token |

吞吐是此固定并发闭环下的观测值，没有进行开放请求速率的最大 QPS 搜索。平均 TPOT 本轮有所上升；平均 E2E 和 TTFT 均下降。
平均每请求 E2E 减少 **179.401 ms（16.72%）**，已经包括 PD 传输与后续 decode。首 token 之后的平均耗时由 190.596 增至 214.652 ms，因此最终 E2E 收益小于 TTFT 收益。该测量是并发 8 下的每请求统计，未验证最终方案在并发 1 下的独立请求收益，也不保证每条请求都变快。
首轮基线出现约 4.3–4.8 秒的个别 TTFT，首轮完整数据单独保留。上表采用双方相同的完整预热协议，未选择或剔除正式 100 条中的单个请求。

## 历史操作消除证据

另外使用原始数据集 16 条请求采集 profiling，独立于计时阶段；两组各记录 6 个 prefill step。

| 设备 kernel 调用 | 基线 | 优化 |
| --- | ---: | ---: |
| 历史 kv_b_proj | 188 | 0 |
| latent prefix page gather | 376 | 0 |
| prefix/chunk K/V cat | 1410 | 0 |
| 当前 token 的 kv_b_proj | 282 | 282 |

优化的 282 个 `npu_prefill_expanded_kv` host 范围中，没有 `aten::index`、`aten::index_select` 或 `aten::cat`。
trace 中存在 BS=7、24721 个新 token 和 BS=6、21170 个新 token 的真实多请求步骤。当前 token 投影、元数据索引和缓存写入仍按需要执行。

## 正确性

- 6 卡 NPU 回归 **10 passed**，覆盖多请求、共享 prefix、跨页、非页对齐、KV 长度超过 2048、零 query、padding、真实缓存写入、容量规划和 PD 注册。
- 对 FP32 causal attention：最大相对 L2 **0.002094773**，最小余弦相似度 **0.999997806286**，误差统计使用 FP64 reduction。
- 完整上下文 prefix 复用检查：从原数据集均匀选择 64 条。先运行完整长请求，在第 640 token 处改变其后续内容；再运行原始请求，验证恰好命中前 640 tokens。这覆盖 prefix 在较大投影中生成后被复用的场景。
- 对相同输入的 **18176 个 teacher-forced logprob**：基线平均 NLL **12.639856272220**，优化 **12.639856272220**，差值 **0**；平均绝对 logprob 差 **0**，最大差 **0**，perplexity 相对变化 **0%**。
- 动态 batch 的完整生成文本并不逐字一致：正式对照为 18/100；同一基线两次运行也只有 29/100 一致。本次使用算子 golden 和固定输入概率核对作为数值证据，没有声称任意动态 batch 下逐字相同。

## 实现与显存代价

P 增加分页展开 K/V，新的 token 只写一次；有 prefix 时 FIA 按 block table 直接读取历史页，省去历史重投影、gather 和 cat。无 prefix 时使用已有连续 K/V 计算 attention，同时保留展开缓存。P 继续保存统一 576 维 latent，PD 只注册和传输 latent；D 不增加展开缓存。

没有“展开 KV 压回 latent”的转换。原有 `forward_mha_prepare_npu` 先生成并写入 latent（norm/RoPE/cache write），再以当前 token 的 `kv_b_proj` 展开 K/V；优化额外把展开结果写入 K/V 页，每层新增两次 scatter。PD 从已经保存的 latent 页直接发送，传输字节/token 和 D 布局不变。原有 latent 生成与写入开销仍在 P 的 prepare 阶段，新增展开 K/V 写入开销在 `set_prefill_expanded_kv`，均已计入客户端 TTFT/E2E。未单独测得这些写入的独占耗时，不把它称作零开销。

如果单请求既不命中 prefix、又无需分 chunk，就没有历史重投影/gather/cat 可消除，同时仍有展开缓存写入，不能保证这一场景有收益。

单 token、单层的缓存元素从 576 增到 `576 + 20×(256+256) = 10816`，即 **18.78 倍**。同样约 51.3 GiB 的 P KV 预算下，容量从 **1019008 → 54016 tokens**。本次收益包含缓存布局、容量和动态调度批次的共同影响，不能把全部端到端收益归因于单个 kernel。

功能默认关闭，P 设置 `SGLANG_NPU_PREFILL_EXPANDED_KV=1` 启用；该基线版本两端还需 `ENABLE_ASCEND_FLASH_MLA=1` 保持统一 latent 传输布局。

## 复现与证据

复现入口和约束见同目录 README。对照命令使用 `--max-concurrency 8 --num-prompts 100`；正式结果标签为 `baseline-steady`、`candidate-steady`。
远端完整原始结果与 trace：`/sgl-workspace/p-kv-opt-artifacts`。本地原始结果位于 `results/`，包括每条请求 TTFT、长度、概率、kernel CSV 与审计 JSON。
