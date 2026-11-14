# BC 输入管线设计取舍

当前接口和用法见 [Python 指南](../../../python/README.md#loading-and-batching)，
职责与交互见[神经适配架构](../../../docs/architecture/neural.md#offline-python-data-and-training)。

## 驻留与并行边界

连续 RAM 复用精确内置 `PackedDecisionDataset`，保留列式 gather 与有效候选评分，
不通过包装子类引入回退。旧 `mmap=False` 仍分 shard，不能消除跨 shard 调度。
预分配后逐 shard 映射复制，避免同时积累完整 RAM shards；但随机 gather 和
源 page cache 与目标副本共存的内存成本仍在。

| 选择 | 原因与代价 |
| --- | --- |
| 单 CPU producer | 保留列式 gather，避免 worker 逐样本回退及 spawn 复制；收益受 NumPy/GIL 工作比例影响 |
| 两个按候选上界定容的 pinned slots | 可预检预算、无热路径扩容；少量大候选行会增大预留量 |
| 普通 gather 后复制到 pinned slot | 复用原 gather 作数值参照，先隔离 DMA 生命周期改动；增加一次 host 复制 |
| 主线程拥有独立 copy stream | 集中设备调度；producer 等 H2D event 即可复用 slot，无须等待 compute |
| 每批独立 GPU Tensor | 通过 allocator 与 `record_stream` 管理寿命，避免持久 GPU slots 的计算完成复用协议 |

只提交已 READY 的下一批，不让当前计算等 producer。队列、slots 和 events
不扩展 epoch 提交边界；finite 检查与 metric 回传时机不随供给优化改变。

## 后续性能决策

默认保留 mmap/sync：工作集可能已在 page cache，checksum 验证也会预热文件，
连续 RAM 与前瞻的端到端收益须分别测量。

比较 mmap、分 shard `mmap=False`、连续 RAM、无重叠 pinned 搬运、copy-stream
前瞻和完整 producer；预驻 GPU 的真实 batches 用于估计供给上限。固定 cache、
revision、seed、batch 序列、模型、线程、依赖、硬件、重复次数、噪声界和接受
阈值，区分冷缓存与 checksum 后的 warm-cache。测量启动物化、含验证和
checkpoint 的 epoch、吞吐、gather/staging/H2D、GPU 空隙与内存峰值；profiler
与正式计时分开，避免逐步全局同步。

只有端到端收益超过噪声、物化成本能在目标 epoch 数内摊销且内存可接受时，
才考虑默认开启。若 staging 成为主要 host 成本，再评估直接 gather 到 pinned
输出；finite/metric 同步另作独立优化。
