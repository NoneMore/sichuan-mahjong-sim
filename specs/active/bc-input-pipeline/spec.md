# BC 数据供给管线规格

## 范围

为离线训练、逐 epoch 验证和冻结 checkpoint 评估增加连续 RAM 驻留与 CUDA
一批前瞻，默认保持 mmap 加同步搬运。设计取舍见 [design.md](design.md)。
编码、cache/checkpoint 格式、模型、训练策略和指标语义不变；首版不做直接
gather 到 pinned 输出，也不调整 finite/metric 检查时机。

## 接口与配置

### 数据驻留

新增 `PreparedCache.open_splits(names, *, storage, ram_limit_bytes=None)`，返回
context manager，通过 `inputs.datasets` 和 `inputs.loading_record` 提供数据与
加载记录。退出时关闭其拥有的 datasets；batch 消费 context 必须先退出。

| storage | 每 split 表示 | 预算 |
| --- | --- | --- |
| `mmap` | 只读 `ShardedDecisionDataset`，保留 shard 布局 | 不接受 `ram_limit_bytes` |
| `ram` | 只读、精确类型 `PackedDecisionDataset`，具有完整 identity | `ram_limit_bytes` 为必填正整数 |

沿用 `PreparedCache` 的验证及显式 checksum 选项，只消费 prepared cache。
旧 `load_split(split, mmap=...)` 的类型、默认、所有权和可变数组行为不变。

### Batch 搬运

新增 `BatchTransferConfig(mode="sync", pinned_limit_bytes=None)`，由 `Trainer`、
`evaluate_dataset`、`evaluate_checkpoint` 的可选 `batch_transfer` 参数接收。

- `sync`：沿用原 batch 校验与搬运，不接受 pinned 预算，不创建 producer、
  pinned slots 或 copy stream。
- `cuda-prefetch`：要求可用 CUDA、`num_workers=0`、显式正整数 pinned 预算，
  且满足以下条件：dataset 为精确内置 `PackedDecisionDataset`，或由该精确类型
  shards 组成的精确内置 `ShardedDecisionDataset`；具有 `target_kinds`，输入
  数组只读；policy 为精确内置 `CandidatePolicy`，使用 flat candidates；offsets
  与缓存 counts 一致，评估满足 eligible-only 条件。

不适用的显式异步请求在 producer 启动前抛出 `ConfigurationError`。
异步消费期间不得修改、替换或关闭输入；自定义类型、可变输入、缺失 metadata、
worker 和布局变化等用法继续由 `sync` 保留原回退。

### CLI

训练和冻结评估均支持：

| 选项 | 默认/要求 |
| --- | --- |
| `--data-storage mmap\|ram` | 默认 `mmap` |
| `--ram-limit-mib N` | 仅 `ram` 接受且必填 |
| `--batch-transfer sync\|cuda-prefetch` | 默认 `sync` |
| `--pinned-limit-mib N` | 仅 `cuda-prefetch` 接受且必填 |

MiB 按 `2**20` 转为 bytes，预算须为正整数。连续 RAM 和 CUDA 前瞻均要求
`num_workers=0`。配置错误在 producer 启动前拒绝，不自动切换策略。
两种优化独立，支持 mmap+prefetch 和 RAM+sync。

## 连续 RAM

按 manifest shard 顺序合并全部 decisions，保留 forced rows、metadata、dtype、
候选顺序、局部 target、全局索引和 split identity：

```text
states               float32 [N, 1013]
candidate_features   float32 [K, 39]
offsets              int64   [N+1]
targets              int64   [N]
其余 metadata                 [N]
```

offsets 加此前候选总数并去掉重复边界。`eligible_indices()` 和 forced 统计不变；
空 split 使用合法空特征和 `[0]` offsets。完成共同 packed-layout 验证后，目标及
派生布局数组均只读。

预分配目标，逐 shard 映射复制。分配特征前依据 manifest shape/dtype 核算全部
请求 split 的目标与派生数组容量，包括 candidate counts、eligible mask 和
eligible indices 的保守上界。合计超出 `ram_limit_bytes` 即失败；构建失败释放
已分配输入，成功记录实际持有字节数。

预算约束目标和派生数组，不是 RSS 限制；源 mmap/page cache、校验临时量、
epoch 索引、batch 和模型不计入。

## Batch 与 pinned 容量

保留原 indices 顺序、batch 划分、候选顺序、shape、dtype 和尾批；显式重复
索引按原序保留，正常训练 epoch 对 eligible decisions 无重无漏。producer
使用主线程确定的切片，不消耗 RNG，不运行模型或更新统计。

| 消费者 | 收集、stage 和搬运的字段 |
| --- | --- |
| 训练 | states、flat candidates、mask、targets、candidate_positions |
| 评估 | 上述字段，加 stages、target_kinds |

不收集 batch provenance 或生成 eligible Tensor。保留 gather 对 targets、候选
范围和训练 eligibility 的校验，以及同步完整 batch 的原校验谓词与错误。

非空遍历使用单 producer、两个定容 pinned slots。令 M 为 indices 数量，
`B=min(batch_size, M)`，C 为所选行的最大候选数。每 slot 预留 B 行 states/targets、
B×C 行 flat candidates/positions 和 B×C 个 mask 元素；评估另加 B 行 stages/kinds。
两个 slots 的元素容量为：

```text
训练：2 × (4060B + 165BC) bytes
评估：2 × (4076B + 165BC) bytes
```

启动 producer 前核算并分配，超出 `pinned_limit_bytes` 即失败。空遍历不分配。
该预算只限制活跃 slots 的元素容量，不包括普通 batch、临时索引、Tensor 对象
及 allocator 缓存，不是 OS 锁页量上限。

先 gather 普通 batch，再复制到 slot；从一维 backing buffer 截取真实元素数
并 view 为本批 shape，保证源连续、不暴露容量尾部。slot 不扩容，超过容量
时报错。两个 slots 与单 producer 限制准备中数据的存活数量。

## CUDA 调度与生命周期

- batch n 使用 slot n mod 2，按 FIFO 交付。H2D event 完成后才可重写 slot；
  producer 可等待交还的 event，不依赖训练主线程返回后轮询释放。
- 主线程在独立 copy stream 上调用 `.to(device, non_blocking=True)`，全部字段
  复制后记录 ready event。producer 不建流、提交设备工作或持有 GPU batch。
- 调用方当前 compute stream 等待本批 event，各 GPU Tensor 对该 stream 调用
  `record_stream()`；host 引用保留到 H2D 完成。
- yield 前尝试提交下一批已 READY 的 H2D，不为尚未 READY 的下一批延迟当前
  计算。下一批 copy 不等本批 compute，本批 compute 也不等后续 copy。
- 管线只保留待交付批和一个前瞻批的 GPU 引用，不复用持久 GPU slots。常规
  供给不调用全局 `torch.cuda.synchronize()`，释放 host slots 不等待整条计算流。

全 forced 评估沿用 metadata 快路径，不评分或创建 producer/staging。训练与
验证使用各自的消费 context，不共享传输中的 slots。

## 失败与恢复

Host gather/填充错误按 batch 序号交付，先前 batch 可正常消费，失败批次不
计算或计入 samples/steps。CUDA 分配/设备错误则终止当前 epoch，停止后续
更新与消费。失败不自动重试或降级。

耗尽、提前退出和异常退出均停止发布、取消队列/slot 等待、join producer，
待已提交 H2D 结束后释放 staging，再关闭输入；清理错误不覆盖原异常。

finite-logit 检查仍在 update 前拒绝非有限输出；metric 回传与 loss chunk 的
host 累加顺序不变。训练和验证管线在保存 checkpoint 前退出。`last.pt` 只提交
完成的 epoch；中断后丢弃前瞻状态，从原边界恢复模型、optimizer、遍历与 RNG，
不保存队列/slots/events，不增加 batch 级恢复。

## 记录与兼容

CLI 将 `loading_record` 传入 Trainer 的 `dataset_provenance.input_loading`，
对应 `run.json` 的 `dataset.provenance.input_loading`；运行 payload 和
`run.json` 另记 `batch_transfer`。冻结评估 JSON 同样记录加载和搬运配置。
记录模式、预算、split 表示和 RAM 实际持有字节；mmap 文件大小不作 RAM
驻留量，旧加载 API 的库调用不生成 cache 加载记录。

供给选项不进入 `TrainingConfig` 或 checkpoint schema。resume 可重新选择，
省略则用 mmap/sync，不算训练超参 override。记录仅描述本次启动，不改写旧
epoch 配置。原 dataset/model/training/device/dependency 约束与精确恢复保证不变。

## 验收

1. **兼容与数据等价**：旧测试通过；覆盖单/多 shard、空 split、混合/全 forced、
   乱序/重复 indices、变长候选及尾批。比较 RAM 与原加载的样本、identity、
   索引和统计，以及 sync/prefetch 的 batch 字段。
2. **配置与预算**：覆盖合法组合、缺失/非法/多余预算及不适用输入；检查合计
   RAM 预检、构建失败清理。Pinned 预算等于公式值时可容纳，少 1 byte 拒绝，
   无扩容、尾部搬运或未消费字段读取。
3. **退出与错误顺序**：注入未来 host 错误、CUDA 错误，并在 producer 等待、
   DMA 未完成时退出；无死锁、提前复用或错误覆盖，失败 epoch 不提交 checkpoint。
4. **真实 CUDA**：覆盖延迟 H2D、交替复用、非默认 compute stream 和 allocator
   压力；在支持并发且供给充足的负载下，以时间线核验下一批 copy 与本批计算
   重叠。无适用硬件时记录未验证项；不设统一吞吐提升门槛。
5. **训练与恢复**：固定初始化、数据和 seed，对照四种配置组合；核对数值、
   steps/samples、模型/optimizer、指标、best 选择和中断重放，维持原有精确性。
   finite 错误仍在 update 前拒绝，运行记录反映本次配置。

实现后更新 [`python/README.md`](../../../python/README.md) 的加载、CLI、恢复
说明及[神经适配架构](../../../docs/architecture/neural.md)的职责与交互。
默认开启和 gather/pinned 融合的后续判断见[设计](design.md#性能判断与后续选择)。
