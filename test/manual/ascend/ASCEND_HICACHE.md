# Ascend HiCache：PD 三条 prefill 路径

基于 `release/codex-1031-best-r97-20261102` 的 `bba2139690fc499ce7b39e18bfd4222f2a8051bd` 开发。
工作区 `/sgl-workspace/sglang-hicache`，分支 `opt/codex-1031-hicache`。
原工作区、release 引用、算子包、已发布镜像均未替换。

## 实现

MHA 修复 FIA 逐层列表、非对称 K/V 和主机层轴引用；MLA 使用 NPU 专用 HostPool，避免访问 CUDA 指针数组，避免把统一576维缓存中的非连续512/64视图当成连续内存。

按用户指定，expanded 模式主机保存双份：compact latent/RoPE（及存在时的 indexer）继续供 PD 传输，expanded K/V 供 P 侧直接回载。没有回载后的重投影，没有新增 attention 量化。默认路径没有 expanded pool，仍只保存 compact，并执行该路径原有计算。未提交的重投影草稿已归档到证据目录 `draft-reproject`，不属于交付代码。

每个 host page 是包含全部本地层、全部组件的连续 payload。compact 沿用 native 全层复制；expanded 每层独立分配，逐层调用原生 K/V pair copy，不能 stack 或假设层间连续。预算、flat page、pointer/size、文件序列化都包含完整双份数据。

H2D stream 提交全部复制后才记录完成事件。`get_prefill_expanded_kv` 独立等待对应层事件；不借用 compact getter 的等待。CPU 确定性回归及 expanded-first 硬件测试覆盖这个顺序。

当前 dense GLM 每层每 token 的 compact 宽度576，expanded宽度 `20×(256+256)=10240`，双份10816元素，约18.8倍于 compact。BF16为21632 bytes。`--hicache-size`、比例容量与主机预算均计入双份；没有把额外数据藏在预算外。

## P 侧配置

P 启用 `--enable-hierarchical-cache`。本轮为 `--hicache-ratio 4 --hicache-write-policy write_through`。

| 路径 | `PREFILL_EXPANDED_KV` | `PREFILL_FUSED_KV_SCATTER` | `PREFILL_FAS` | `PREFILL_FAS_EXTENDED_SHAPES` |
|---|---:|---:|---:|---:|
| 默认 | 0 | 0 | 0 | 0 |
| expanded | 1 | 1 | 0 | 0 |
| FAS | 1 | 1 | 1 | 1 |

表中开关全名都加 `SGLANG_NPU_` 前缀。FAS 依赖 expanded pool；单开 FAS 而关闭 expanded 不代表命中。FAS 保留 batch、dtype、layout、prefix、token 数 guard，不满足时正常 fallback。

P 使用 unified latent、FlashMLA关闭；D 使用之前已验证的 unified+FlashMLA。此次验收关注 P。Ascend 自动解析 IO 为 `kernel_ascend`，MLA host layout 为 `page_first_kv_split`，MHA为 `page_first_direct`。

运行新功能需切换到本工作区源码，并同步更新启动脚本固定 commit 校验。原 `r97_script` 仍引用旧冻结源码，逐角色开关也可能覆盖外层 export；本轮没有替换用户已有脚本。FAS 还必须满足实际 shape guard。

## 文件后端

MLA L3 当前开放 `file`：`--hicache-storage-backend file`，目录由 `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` 指定。其他 MLA L3 后端在加载前拒绝。

Namespace 自动区分 split/unified、dtype、page、层范围、latent/RoPE/index维度、expanded几何及ND/NZ相关开关。开关在初始化时冻结。等长不同格式不串读，同格式不同容量可复用；startup/runtime attach共用校验且不修改原配置。双份 file round trip 做了真实文件逐字节CPU验证，未做真实PD L3压测。

## 功能验收

硬件为 Ascend950DT，P7/D6各TP1。服务使用 eager、关闭调度重叠，未验收图模式或多P/D拓扑。测试模型为固定随机权重的2层BF16模型：20头、latent512、QK192+64、V256，与目标GLM核心attention几何相同，但hidden320/vocab1024，不是生产47层MXFP8模型。所有输入为合成token IDs。

每case独立启动并检查P/D均注册且健康：cold golden四prompt各生成2token→标准P/D flush成功并settle1秒→四条257token seed留下256前缀→九条4096token不同前缀驱逐设备缓存→四条target要求每条host256/device0及两token与本case golden相同。

默认/expanded target为256 prefix+2048 new。PD实际拆成1+3批，因此FAS专属case使用256+4096，保持guard原样。实际后3条组成batch3/new12288并执行FAS，首条走原FIA fallback。形状不同，只比较各case自身cold/回载结果，不作跨case性能AB。

| 路径 | 有效结果 | 实际路径证据 |
|---|---|---|
| 默认 | 21逻辑请求成功；4target host256/device0、两token一致 | PD传输与loadback |
| expanded（补fence后） | 同上 | actual expanded分支及dual mirror H2D |
| FAS（补fence后） | 同上 | 算子返回后记录batch3/new12288/prefix256，生产guard复核 |

默认路径结果来自 `4db0619`。之后到生产代码冻结点 `20bcb8b`，唯一生产变更是 expanded getter 的等待；默认路径关闭 expanded，不调用该 getter，因此保留已通过结果。expanded、FAS 均在补等待后重测通过，未把此前结果冒充重测。

不能将 `first eligible eager step` 当成实际FAS执行；新counter/日志只在真实算子返回后增加。FAS覆盖3条target，不宣称4条全部或全部业务100%命中。

DMA位级比较通过；服务token一致不代表logits位级一致。默认/expanded最大logprob差约 `5.197525e-5`，FAS约 `0.007791519`。这些结果不替代生产整网精度评测。

CPU累计85项通过、1项原条件skip：MHA8、MLA9、file8、DCP host10、staged write-back15、FlashMLA短表6、原FAS contract10、新FAS dispatcher7、dual7、expanded fence5。DCP项是原通用代码回归，不表示NPU新增DCP支持；重投影和旧hook草稿不计入。

实际NPU矩阵：MHA8、MLA8、dual4种组合。dual4在补fence后改为先读取expanded，再读取所有compact getter，重新通过。包含BF16/FP16、独立逐层分配、乱序页面、换页回载、未选页、完整序列化及完成事件。

最终验收选用的目录：

- 默认：`pd-three-paths/pd-default-20261102-194024`
- expanded：`pd-three-paths/post-fence/pd-expanded-20261102-200831`
- FAS：`pd-three-paths/post-fence/pd-fas-20261102-201032`

证据根目录：`/mnt/sfs_turbo/hxd/p_kv_shared_tune_test/codex-1031/hicache-20261102`。
聚合：`pd-three-paths/pd-service-agent-summary-post-fence.json`；报告：`pd-three-paths/post-fence/PD-POST-FENCE-VALIDATION.md`。所有旧失败/pre-fence结果保留。P/D通过后owner停止超时再SIGKILL，与运行crash分开记录；全部自建组清空、卡6/7释放。

## 范围与原镜像

DCP、layer sharding、packed MTP、HiSparse、PD offload-only未在本轮支持或验收。旧QK192/V128小模型的ATB RingMLA前缀错误，在无HiCache对照同样复现，未在本次修改。unified decode经FIA读strided RoPE也不支持，应保留FlashMLA。另单独修复了eager FlashMLA短上下文bucket超出block-table容量，graph动态 `-1` 不变。

本轮没有生产QPS benchmark，没有恢复暂停的11QPS优化任务，没有生产47层MXFP8整网精度结论。

原镜像 `sglang:codex-1031-best-r97-20261102`、ID `sha256:0e87d41c14db6b258376dafc160ec71345bb636abe5ffeb017f9d3e8f4ddb8f1` 先构建，再固化同运行内容到release；不是checkout该release分支后直接打的。冻结核对4679 tracked runtime路径+1 generated version内容一致。镜像未包含此次HiCache提交，本轮未重打或改内部脚本；旧10.7342 QPS/card、E2E1445.54ms不是HiCache收益。
