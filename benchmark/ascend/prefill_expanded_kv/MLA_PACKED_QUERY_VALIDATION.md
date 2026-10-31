# MLA Prolog V3 decode：576 维 Query 直写阶段报告

576 维直写与 SGLang 无 Query concat 接入已实现并通过回归。**按业务常见 batch 验收，30% 目标尚未达到，目标保持进行中。** Prefill 不使用本算子，本次没有改动其计算路线。

## 执行口径与权重

物理卡7，CANN9.1.0，编译目标 Ascend950PR_9599。checkpoint 为 `/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8`。
`q_a_proj [768,2048]`、`q_b_proj [5120,768]`、`kv_a_proj_with_mqa [576,2048]` 均存为 F8_E4M3 配 U8 每32元素 scale；`kv_b_proj [8960,512]` 为 BF16。框架将 Q-A 与 KV-A 合并，看起来是两组量化线性层。当前 V3 加载准备阶段反量化为 BF16，运行 `weight_quant_mode=0`；本次保持这一口径，没有新增量化。

## 已实现

* `allocate_outputs(..., packed_query=True)` 和高层 `mla_prolog_v3(..., packed_query=True)` 生成一个 `[T,H,576]` 分配，原有 Query/Query-RoPE 返回值保留为其512/64维视图。
* Host 验证 BF16、设备、规则且对齐的物理 head stride；私有 tiling 传递步长。Qn 的 Fixpipe 与 RoPE 的最终搬出直接落到目标行，消除后续拼接流量。
* SGLang `forward_mlaprolog_v3` 对 FlashMLA decode 使用 packed 输出，`_fmla_decode` 经 `combine_query` 检查共享存储、偏移与 stride 后恢复完整连续视图。验证中禁止调用 `torch.cat`，旧的独立 Tensor 调用保留兼容 fallback。
* 现有 KV direct-cache 路线保留。算子仓合入其前置提交 f6dd0b1，形成 fabae3a；Query 优化可和它一起使用。
* 小 batch 使用16核；精确 GLM BF16 形状在 T17..128 使用20核。Packed T17..128 按完整 head 同核生产/消费 QC/Qn；T41..128 将 CQ 与 KV/KR 分给不相交的10/9核组并行。其他 dtype/形状及大 T 使用原调度。环境调优参数纳入 canonical tiling 缓存键，完整物理 tiling blob 分设备缓存。

## 原始库 A/B/A

真实 checkpoint 第2层权重，固定种子42。原始库使用已保存的 cache-only 基线；新库 6cfe9b18fe1fb07bc13a34e12f41e6d9065b1dee70727537bab72e6b46d6a94b256：`6cfe9b18fe1fb07bc13a34e12f41e6d9065b1dee70727537bab72e6b46d6a94b`。各进程先校验 Query、Qnorm 和完整写入缓存逐位一致，再测47次完整调用/图、31组 NPU event 中位数。先 A、再 B、再 A；表中基线为两次 A 的平均值。A/B 只包含 prolog 与它的 Query concat/直写，保持 KV direct-cache 相同，不包含 FlashMLA、其他层算子或服务排队。

| Batch | 原始库+concat µs | 新库直写 µs | 加速比 | 时延下降 |
|---:|---:|---:|---:|---:|
| 1 | 20.186 | 18.275 | 1.105x | 9.47% |
| 32 | 24.744 | 21.690 | 1.141x | 12.34% |
| 64 | 34.253 | 27.134 | 1.262x | 20.78% |
| 92 | 48.013 | 34.948 | 1.374x | 27.21% |

这是同一层权重重复调用的算子微基准，不是整网吞吐、TPOT 或冷权重访问结论。相同二进制内部开关的交替对照另覆盖 T1/7/16/17/32/33/40/41/44/48/64/65/80/92/127/128/129；T65以上可出现超过30%的反时延收益，但不能代表常见业务 batch。

## 业务 batch 核对

用户确认以业务常见 batch 验收。现有 CHAT 压测 decode worker 日志在16:50:00..16:55:40窗口的108条周期采样中，100条是 batch1..8，其余为9/10。日志不是每步记录，不能当作完整概率分布，但足以说明不能仅以64/92验收。来源、完整采样直方图见 results JSON；目前小 batch 的收益约10%–15%，**尚不满足该业务口径的30%目标**。

## 正确性和兼容性

* T1..92逐个检查，加127/128/129边界共95个形状；五路输出非空值、cache写入、canary与图回放均通过逐位比对。未量化占位的合法空形状差异按numel=0处理。
* 原有27项一键验收全部通过，包括CPU golden、4层真实权重、厂商逐位对照、边界、out缓冲、多流与图捕获。新增packed项首次只在写结果时因套件OUT是目录而失败，保存失败日志后修正命令并重跑通过；没有掩盖数值失败。
* SGLang原有及新增接入47项通过；另13项包含实际分配式wrapper的图捕获、scratch/direct-cache两种路径与真实FlashMLA一致性。`query_norm_flag=False` 的5个边界额外通过。
* 当前没有对本次 Query 改动运行新的全模型服务吞吐对照，不能声称业务TPOT或QPS提升30%。

## 调优推演与实测取舍

Step1：四个 BF16 Cube MM 单元分别为 DQ `(T,768,2048)`、DKV/KR `(T,576,2048)`、QC/QR `(T,5120,768)` 和20个 Qn `(T,512,192)`，保留中间 BF16 舍入。路由为现有低阶Cube MMAD/Fixpipe流水调优；Blaze迁移未实测，不能声称不支持或一定更慢。

Step2：本算子没有卡间通信，跳过卡间流水。Step3：CQ各核写完后通过CV与向量汇合同步，RMSCQ完成后再生产QC/QR；head-local确保Qn只读本核QC，用FIX→MTE2事件保持可见性；并行下投影在覆盖共享空间前仍等前一步KV消费者。原子归约顺序与舍入位置没有变化。

Step4：逐类检查 Scalar/小case、访存、Vector及流水不足。已保存业务profile的一个原始V3核记录 scalar约26.7%/28.2%(AIC/AIV)、MAC24.8%、MTE2 58.1%、Vector0.1%；这些流水有重叠，不能相加，也不足以把某屏障的成本单独归因。小batch固定开销仍需继续隔离。没有新Simulator指令时间线，不能声称已定位唯一bound或达到硬件上限。

尝试并归档：Qn完整K192/N64、单48KiB L0B的K192/N128、保持K64x3的N256，精度通过但没有稳定胜出；16/20/24/32核及多组CQ/KV核组扫描；21/22/23核的12+9平衡切分也没有稳定优于保留的20核策略。实验patch对应当时源码上下文，复现时须核对上下文，不能直接把历史patch覆盖生产文件。

同步静态脚本原始报告保留于任务交付目录：pair检查给出跨函数匹配等50条候选，flow检查0条；这些候选没有被宣称全部清零。case_retriever数据库在只读技能目录重建失败；正则脚本不替代模板展开与运行验证。被拒绝的wide-L0B实现已从生产代码移除。

## 运行与回退

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /mnt/sfs_turbo/hxd/pkgs/flash_mla_delivery/env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export PYTHONPATH=/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu:/sgl-workspace/sglang-p-kv-opt/python:$PYTHONPATH
# D：现有 ENABLE_ASCEND_FLASH_MLA=1 / SGLANG_NPU_USE_MLAPO_V3=1
export SGLANG_NPU_MLAPO_V3_PACKED_QUERY=1
export MLA_PROLOG_V3_DECODE_TUNE=1
# 原始输出+原始调度对照：同时将以上两个开关置0
cd /mnt/paas/hxdnpu/sgl-kernel-npu
python3 csrc/mla_prolog_v3/tests/final_compare.py
bash csrc/mla_prolog_v3/tests/run_acceptance.sh 7
```

启动器 `benchmark/ascend/prefill_expanded_kv/launch_decode_v3_2p1d.sh` 仅为D选择匹配的新包，P仍关闭V3。可用 `DECODE_KERNEL_PYTHONPATH` 覆盖包路径。当前共享工作区另有FlashMLA迁移改动，未混入本阶段的提交。

原始日志和未裁剪数据保留 `/tmp/mla-output576-20261030`。业务样本原文、模型权重及大profiler文件不纳入提交。后续工作聚焦常见小batch固定开销；在达到业务口径30%前不宣布目标完成。
