# BC 数据供给管线设计

为离线 BC 提供独立可选的连续 RAM 驻留与 CUDA 一批前瞻。
接口、行为和验收见 [spec.md](spec.md)。

## 职责与数据表示

数据驻留归 prepared-cache 层：它掌握 shard inventory、split identity 和加载
资源，多 split context 可以在分配前核算合计预算，并统一清理构建失败。
训练与评估共用 `_input_pipeline.py` 管理 device-batch 迭代，将线程、队列、
stream/event 和清理集中在一个边界，模型和 epoch 循环只消费设备 batch。

RAM 模式选择每 split 一个按列连续的 `PackedDecisionDataset`：

- [`cache.py`](../../../python/mahjong_bc/cache.py) 的 `mmap=False` 仍保留
  shard，不能消除跨 shard 分组与 gather。
- [`data.py`](../../../python/mahjong_bc/data.py) 的 packed 快路径依赖精确内置
  类型；复用该类型，避免包装子类退出列式 gather 和有效候选评分。
- 保留 metadata、forced rows 和原始全局索引，避免为性能缓存另建数据身份。

构建时预分配目标、逐 shard 映射复制，避免额外积累完整 RAM shards。输出只读
以支持异步消费，旧加载 API 保持原有可变行为。连续布局消除 shard 调度，但
仍需随机 gather；源 page cache 与目标副本也可能同时占用内存。

## 并行策略

| 选择 | 取舍 |
| --- | --- |
| 单 CPU producer | 保留列式 gather，避免多 worker 的逐样本回退和 spawn 复制；并行收益取决于 NumPy/GIL 工作比例 |
| 两个按候选上界定容的 pinned slots | 预算可预检，无热路径扩容；少量大候选行会增加预留量 |
| 普通 gather 后复制到 pinned slot | 保留现有 gather 作为数值参照，先隔离 DMA 生命周期改动；代价是一次额外 host 复制 |
| 主线程拥有独立 copy stream | GPU 调度集中；producer 只等待 slot 的 copy event，不依赖主线程计算结束后释放 |
| 每批独立 GPU Tensor | 使用 allocator 与 `record_stream` 管理跨流生命周期，不另建持久 GPU slots 的计算完成复用协议 |

主线程只对已 READY 的下一批提交前瞻，避免为等待下一批而延迟当前计算。
显式请求不适用的异步路径时报错，默认同步路径保留旧回退，使执行策略可解释。

Host 数据错误可按 batch 定位，按序交付；CUDA 错误可能影响设备状态，终止
当前 epoch。队列与 DMA 保持为临时状态，不扩展 `last.pt` 的 epoch 提交边界。
驻留和搬运属于执行选项，不进入训练身份与 checkpoint schema，可在恢复时
重新选择；详细兼容约束见规格。

## 性能判断与后续选择

显式选择驻留与搬运，便于独立归因，默认策略暂不调整。mmap 工作集可能已在
page cache 中，默认 checksum 验证也读过文件；RAM 复制的收益需要实测。

逐项对照当前 mmap、分 shard `mmap=False`、连续 RAM、pinned/non-blocking
无重叠对照、copy-stream 前瞻和完整 producer。多个真实 batch 预驻 GPU 可
估计供给上限。

固定 cache、revision、seed、batch 序列、模型、线程、依赖和硬件。测量启动
物化、含验证和 checkpoint 的 epoch、decision/候选吞吐、gather/staging、H2D、
GPU 空隙及内存峰值。Profiler 与正式计时分开，不逐步全局同步；冷缓存与
checksum 后 warm-cache 分开比较。

采集前固定工作负载、重复次数、噪声界和接受阈值。端到端收益超过噪声、
物化成本能在目标 epoch 数内摊销且内存可接受后，再决定是否默认开启。
若 staging 成为主要 host 成本，再评估直接 gather 到 pinned 输出；finite/
metric 同步则留作独立优化，保留本 Change 的检查时机。

未采用的实验方案按[研究保留规则](../../../docs/governance.md#research-retention)
保留可复现方法。
