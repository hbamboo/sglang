# Decode FlashMLA + MLA Prolog V3 integration and 2P1D acceptance

四个指定提交已按顺序 cherry-pick 到 `opt/p_kv_shared_tune`，并补齐 decode 阶段判断、保留 P 原融合缓存路径。最终 P 不执行 FlashMLA/MLA Prolog V3；D 同时执行两算子。

## 代码与实际执行路径

| 来源提交 | 合入提交 | 内容 |
|---|---|---|
| `5ac6236cf3` | `9363481a42` | 自定义 `torch.ops.npu.mla_prolog_v3` 接入和 MXFP8 权重准备 |
| `5588c687dc` | `30c6e442d6` | 独立统一 latent KV 布局开关 |
| `d1404dba2b` | `566227cd2a` | merge 内容与首提交重合；保留来源记录并去重实现 |
| `53c58880c7` | `987314860c` | V3 小步 scratch + 统一池 scatter，与 FlashMLA 共存 |

修正提交 `75507b4153` 用相同的 batch 条件控制 prepare 与 attention 的 KV 写入：V3 只在 decode 执行，prefill 保持原分支并正常保存 latent。旧 int8 MLAPO 在统一布局下仍不启用。

修正提交 `f0af590e68` 保留 P 原有 `KvRmsNormRopeCache` 融合路径。被合入的统一布局提交原先一并禁用了这个路径；初轮对照出现 rate10 TTFT 约15%劣化。真实 NPU 检查证明当前 `torch_npu` 适配层正确处理统一 KV 池的非连续 512/64 view，输出与缓存稀疏写入均逐位一致，因此恢复原路径，而非接受该劣化。显式 `SGLANG_NPU_DISABLE_YARN_KV_SINK` 仍可用于其他环境的手动回退。

D 的 V3 使用本步大小的 packed 512/64 scratch，再 scatter 到统一 `[blocks,128,1,576]` latent 池；FlashMLA 随后读取同一池。P 的展开 K/V 只供本地 FIA 使用，PD 注册与传输仍是每层一个 576 维 latent 缓冲，BF16 时每页 147456 字节，不传展开缓存。

## 配置与测试口径

物理卡5、6分别运行两个 TP1 prefill 实例，7运行一个 TP1 decode。模型 `/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8`，CANN9.1.0/Ascend950DT，`modelslim`，BF16 latent KV，无 MTP。

| 设置 | P1/P2 | D |
|---|---|---|
| `ENABLE_ASCEND_FLASH_MLA` | 0 | 1 |
| `SGLANG_NPU_MLA_UNIFIED_KV` | 1 | FlashMLA 隐含启用 |
| `SGLANG_NPU_USE_MLAPO_V3` | 0 | 1 |
| `SGLANG_NPU_USE_MLAPO` | 0 | 0（V3 开关已隐含启用 prolog 路由） |
| 展开 KV / shared overlap | 1 / 1 | 0 / 0 |
| transient reserve | 5120 MiB | 0 |

P max/chunk prefill tokens=32768，overlap threshold=8192，P/D mem fraction=.90，max running=64，D graph max BS64；所有阶段使用 `expandable_segments:True` 和显式拆分共享专家。原版本无独立布局开关，P 使用 FlashMLA 开关选择统一池，但实际 prefill 一直是 FIA。

原数据集 SHA256 `bb722901ab4a6857571837bbef58b30a4ca00ce08c1aaf170ada3c94694d0f0a`，seed42、thinking=true。每次新服务准备100条；正式固定并发8每轮100条、三轮，另在rate10/concurrency500每轮500条、两轮。每轮预热后通过 router flush，两台P日志都验证200 OK。所有正式计时关闭 profiler/诊断标注；质量与 profiling 单独进行。每个100条用例输入384096/输出2900 token；每个500条用例输入1951677/输出14497 token，跨版本逐请求长度完全一致。

## 正式结果

| 版本 | c8平均TTFT ms（3轮） | c8平均TPOT ms | c8实际QPS | rate10平均TTFT ms（2轮） | rate10平均TPOT ms |
|---|---:|---:|---:|---:|---:|
| 原分支 | 297.569 | 9.460 | 13.871 | 322.460 | 8.173 |
| 合入修正后 | 308.785 | 8.405 | 14.326 | 320.463 | 7.335 |
| 同版本关闭展开KV | 584.896 | 4.303 | 10.970 | 566.626 | 4.724 |
| 末尾原分支复测 | 297.909 | 9.341 | 13.913 | 325.910 | 8.204 |

固定 offered rate10 时，新版本平均TTFT相对首次原版本变化 -0.62%，相对末尾原版本复测变化 -1.67%；相对同代码关闭展开KV，平均TTFT降低 43.44%。
固定并发8时，相对关闭展开KV，平均TTFT降低 47.21%。同时应注意闭环固定并发下 D 变快会增加实际输入速率，故也完整列出QPS及TTFT；不能只把该口径的TTFT差值归因于P的单步性能。

各轮原始TTFT如下，避免把顺序实验的波动隐藏在均值里：

- 原分支：c8 302.801, 293.300, 296.607 ms；rate10 283.937, 360.984 ms。
- 合入修正后：c8 309.090, 309.616, 307.648 ms；rate10 281.681, 359.244 ms。
- 关闭展开KV：c8 578.983, 589.249, 586.456 ms；rate10 559.135, 574.116 ms。
- 末尾原分支复测：c8 309.291, 292.539, 291.898 ms；rate10 285.565, 366.255 ms。

## 正确性、路径和传输证据

36项硬件回归全部通过：原展开KV/容量预算测试、V3仅decode的路由、scratch大小和量化池拒绝、真实V3 scratch与packed分页结果逐位一致、统一池落地与未写区域保护、原P融合sink对统一view的稀疏写入。

正式验收四组各1300条测量请求，共5200条均成功；两个P均有真实多请求batch，所有batch新token<=32768，日志未发现 KVTransferError、IndexError 或传输线程失败。此处验证的是2P1D，未声称完成多D或不同TP拓扑验收。

独立框架profile包含CPU与NPU，Level1+PipeUtilization，CSV48列、op/api statistic齐全且trace JSON有效。两个P恢复原融合sink，展开KV区间中无历史index/index_select/cat；P无FlashMLA或V3设备算子；D每层decode同时出现V3与FlashMLA，次数相等。

固定10个质量用例包含短文本及1K/8K/16K/32K上下文：P首token ID及logprob全部与原版本一致=True。decode生成token共160个，其中112个同位置token相同、3/10用例完整相同。V3接入按来源提交把MXFP8投影准备为BF16计算，decode并非逐位等价；这些生成检查不是完整模型准确率评测，未新增GSM8K质量结论。

日志按SIGTERM标记区分请求期与退出期。各组均在请求数排空为0之后停止；停服时存在tokenizer/uvloop的SystemExit:0与CancelledError栈，原版本的部分P进程另有uvloop退出segfault，新版本最终组未见该segfault。原始栈保留在shutdown_diagnostics及服务器日志中；这些退出期问题未计为请求期传输失败，也没有被当作本次修复内容。

P的分页展开attention实现、容量预算实现，以及MHA prepare/core函数都与合入前保持一致，源码哈希/AST片段比对记录在summary.json。独立profile与正式计时分开，初轮取消P融合sink的诊断数据仍留在`after/`、`no-expanded/`，不混作最终结果。

## 复现与交付

最终应用代码提交：`f0af590e683dcd5cef771e89bba3154cb6056304`。后续提交仅补充启动器/验收工具/报告。

在目标仓库执行：

```bash
bash benchmark/ascend/prefill_expanded_kv/launch_decode_v3_2p1d.sh
```

此启动器使用5、6为P，7为D，可通过PREFILL_DEVICE/DECODE_DEVICE覆盖；需要空闲卡和端口，并已安装包含`npu::mla_prolog_v3`及`mla_prolog_v3_compat`的sgl-kernel-npu、自定义FlashMLA环境。原站点脚本通过临时副本加入共享专家配置，原文件未改。

远端完整原始证据：`/sgl-workspace/p-kv-opt-artifacts/decode-v3-2p1d-20261030-093809`。summary.json、每轮jsonl、配置、两侧日志、质量输出、profile以及补丁均保留；所有测试进程和本任务端口在最终审计时释放。

实际KV容量（token）：before: P1=48896, P2=48896, D=1019904；after-fused: P1=48896, P2=48896, D=1019904；no-expanded-fused: P1=920576, P2=920576, D=1019904。展开缓存会降低P的容量，关闭展开KV对照保留同样的显存预算，属于完整策略对比，不能把端到端差值全部归因于某一个kernel。
