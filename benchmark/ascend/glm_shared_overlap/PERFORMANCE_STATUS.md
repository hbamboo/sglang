# GLM-4.7-Flash 多流实现与阶段性性能报告

状态：三阶段多流已实现并提交，真实 NPU 数值测试和客户负载性能/profiling 对照已完成。尚未证明稳定端到端收益；完整 GSM8K 对照正在运行，整个目标尚未完成。

## 环境与实现

- `root@113.46.12.25`，容器 `hxd-glm`，物理 NPU 5，Ascend950DT。
- 权重 `/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8`。
- 仓库 `/sgl-workspace/sglang`，分支 `feat/glm-npu-shared-overlap`。
- 实现提交 `203c0c9bfc`；客户负载/断点验证提交 `f1065f282e`，作者 huangxudong。
- 开关 `SGLANG_NPU_SHARED_EXPERT_OVERLAP=1`，默认关闭。
- 共享 gate/up 对齐 top-k/init routing；共享 SwiGLU+MXFP8 quant 等待 routed GMM1 完成后与 GMM2 并行；共享 down 等待 GMM2 后与 finalize 并行。最终加法前等待共享完成。
- 使用作用域 context、可复用 NPU stream/event 和跨流 allocation lifetime 管理；decode 保留原分支。此前容器已有的 7 个未提交运行时修改未纳入本次提交。

## 主要性能负载

用户澄清的负载是多条请求、prefill 新 token 预算最多 32768、每请求输出 1 token。此前单请求 32K 测量仅是微基准，不能替代本节数据。

数据集 `/mnt/sfs_turbo/hxd/CHAT-mxfp8-customer-2000.jsonl`，SHA256 `bb722901ab4a6857571837bbef58b30a4ca00ce08c1aaf170ada3c94694d0f0a`。
直接使用 `conversations[0].content`，与实际 tokenizer 对应 2000 条请求、7566887 输入 token。保持文件顺序，分成 248 个受控 burst，每组不超过 32768 token 和 32 条请求；实际每组 21333..32743 token，最多 27 条请求。

两版均从新服务启动后执行同样的三个多请求组预热、同样的数据顺序、相同模型/缓存/调度参数；prefix cache 关闭，greedy、ignore_eos、每请求严格输出一个 token。客户端测量首个包含生成 token 的 SSE 数据包。这是受控 burst 对照，不是给定到达率下的排队性能测试。

服务主要参数：TP=1，mem-fraction-static=0.80，context-length=40960，max-prefill-tokens=32768，chunked-prefill-size=32768，max-running-requests=32，disable-radix-cache、disable-cuda-graph、disable-shared-experts-fusion，attention-backend=ascend，ASCEND_USE_FIA=1。两版只切换多流开关。

| 指标 | 基线 | 新服务优化版 | 变化 |
| --- | ---: | ---: | ---: |
| 平均 TTFT | 891.379 ms | 887.049 ms | 下降 0.486% |
| TTFT 中位数 | 909.187 ms | 903.263 ms | 下降 0.652% |
| P95 TTFT | 1008.688 ms | 1058.228 ms | 上升 4.911% |
| 完整 burst 总耗时 | 230.239 s | 230.231 s | 下降 0.0036% |
| 请求吞吐 | 8.68663 req/s | 8.68694 req/s | 基本持平 |
| 首 token 一致性 | — | 2000/2000 | 全部一致 |

248 组配对中 222 组更快，配对批次节省中位数 23.494 ms；但平均节省仅 0.033 ms/组，bootstrap 95% 区间为 [-15.716,12.072] ms，包含 0。全部样本和慢批次均保留，没有剔除异常值。结论是有典型批次收益，但尚未取得稳定整体收益，不能称为已完成性能优化。

初次优化客户测试是在长解码/中止之后进行，运行历史与新服务基线不一致，结果另行保留，不纳入上述公平对照。

## Profiling 实证

客户首组是 27 条请求、32467 token，调度器实际拆成 30861 与 1606 token 两次 prefill，均没有超过预算。

| 实际 prefill | MoE 层数 | gate/up 有重叠层数 | act+quant 有重叠层数 | down 有重叠层数 |
| --- | ---: | ---: | ---: | ---: |
| 30861 token | 46 | 46 | 46 | 46 |
| 1606 token | 46 | 0 | 3 | 0 |

大批次的三个阶段毛重叠总量分别约 21.421、6.544、14.475 ms。主流 57、共享流 54；这些是按实际 kernel 时间区间求交集的结果。毛重叠不能相加后当作 TTFT 节省，因为并发会竞争资源，且 host 提交与尾批次影响整次请求。

所采客户窗口内，MoE init routing、routed GMM2、finalize 的 kernel 累计时间分别从 67.412/49.549/16.854 ms 增至 70.593/54.220/19.712 ms，说明共享工作被掩盖的同时，主路由工作也被拉长。

下一步应基于证据检查：小尾批次选择串行路径的阈值、共享流 vector/cube 资源限制、跨流 allocator 的等待/回收与慢批次的关系。当前 torch_npu 提供 set_stream_limit/get_stream_limit/reset_stream_limit，但尚未改动资源配额。不能把 allocator 问题当成已证实根因。

原始 profiling 位于结果目录的 `baseline-customer-profile` 与 `optimized-fresh-customer-profile`。本地镜像保留 `kernel_details.csv`、统计文件，`overlap-customer-by-layer.csv` 为逐层求交结果。

## 精度验证状态

真实 NPU MXFP8 共享投影测试使用 2048→3072→1536→2048 维度；1/128/1024/32768 token、重复复用、非默认父流、allocator churn、异常清理和 decode fallback 均通过。已测共享输出逐位一致，最大绝对误差 0。

客户数据 2000 个首 token 全部一致。这不等价于完整 GSM8K 精度结论。

完整 GSM8K 正在进行：本地 test split 1319 题，固定 train 前 4 个示例，greedy，最多 8192 新 token，并发 16，EvalScope extract_answer/math_equal 判分。每条输出、prompt hash、完成原因、截断标记和分数均保存。当前先执行基线全量，再执行优化全量；已完成的 44 条优化记录保留在同版 journal 中用于续跑。完成后还需核对题数、逐样本 prompt、截断、正确率和差异。

## 可复算材料

远端结果根目录：`/mnt/paas/hxdnpu/glm-shared-overlap-20261030`。
本地工作目录：`C:\Users\Administrator\Desktop\workspace\glm-shared-overlap`。

- `results/baseline-customer.json`、`optimized-fresh-customer.json`：完整逐请求/逐批次记录。
- `compare_customer.py`：248 组配对统计、首 token 对照和 bootstrap。
- `analyze_overlap.py --customer --optimized-prefix optimized-fresh-customer`：逐层重叠证据。
- `results/optimized-fresh-customer-paired-packs.csv`、`overlap-customer-by-layer.csv`：审核明细。
- `results/npu-tests.log`：真实 NPU 数值测试。
- `TASK_STATE.md`：持续目标、活进程接续和未完成项；必须先核对实际进程，不能只信状态文件。
