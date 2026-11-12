# 精简默认 batch gather 与 device transfer 字段

## 目标与范围

按训练和评估的实际用途收集、搬运字段，消除完整 `DecisionBatch` 带来的冗余。
范围限于 `num_workers=0`、具有 `target_kinds` 的内置 `PackedDecisionDataset`
及其组成的 `ShardedDecisionDataset`，评估沿用现有 eligible-only 优化条件。

## 行为

| 路径 | 收集并搬运的 Tensor 字段 |
| --- | --- |
| 训练 | `states`、`candidates`、`mask`、`targets` |
| 评估 | 上述四项，加 `stages`、`target_kinds` |

两条路径均不按批读取、转换或分配 `actor_seats`、`decision_indices`、
`episode_ids`，也不生成独立的 batch `eligible` Tensor。
训练在 host 端根据本批实际候选范围确认所有行均 eligible，随后对全部行的
NLL 求均值；评估继续由指标内部从 `mask` 推导 eligibility。

## 兼容边界

- 精简路径供训练和评估内部使用，完整 sample、batch、collate、默认
  `iter_batches()` 及 `DecisionBatch.to()` 的契约不变。
- 自定义 dataset/子类、缺失 `target_kinds`、多 worker 及不满足 eligible-only
  评估条件的情况，保留现有回退语义。
- 保持 batch 顺序、候选顺序、padding、dtype、数据校验、训练权重和指标语义。
  缓存中的溯源字段及其加载、验证不变；不涉及格式或版本变更。

## 验收

1. 通过字段访问、分配和搬运观测，确认热路径只处理表中字段，而非先收集再丢弃；
   未消费的溯源字段不触发 batch 专属转换或溢出检查。
2. 对单/多 shard、乱序与重复索引、变长候选及尾批，与完整 collate 逐项比较；
   训练仍拒绝 forced 行和无效 target，覆盖构造后 offsets 变化的情况。
3. 固定 CPU 数据、初始化、seed 和配置，与完整 batch 参照验证 loss、参数更新、
   样本/步数、验证报告及 best 选择一致，通过恢复重放测试。
   评估覆盖 eligible/forced 混合及全 forced split。
4. 完整 batch 元数据、handoff 和回退测试继续通过；有 CUDA 时验证实际设备搬运，
   无 CUDA 时记录未覆盖项。验收不设吞吐提升门槛。

实现后更新[神经适配架构](../../../docs/architecture/neural.md)中的字段职责与路径边界。
