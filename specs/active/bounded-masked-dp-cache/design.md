# Bounded Masked DP Cache — Design

关联：[Issue #17](https://github.com/NoneMore/sichuan-mahjong-sim/issues/17)。
接口、行为与验收见 [spec.md](spec.md)。

## 背景

基线 `9390264` 的缓存仅在一次决策内有效，见
[`analysis.rs`](../../../src/policy/analysis.rs)。
[历史研究](https://github.com/NoneMore/sichuan-mahjong-sim/blob/955b1785a44791c6444f4a51e41fb7d329976fc6/docs/studies/heuristic-space-time.md)
支持增加跨决策缓存；旧原型和测量脚本已删除，需重新实现候选与采集证据。

## 取舍

- **显式上下文，而非 TLS 或整组每席私有表：** 兼顾计算复用、内存预算、释放和实验隔离。
- **保留本地缓存：** 高频查询仍走本地快速路径，共享层只承接需要重建的花色表。
- **局部轮替，而非整表清空：** 容量满后保留大部分积累，命中无需维护 LRU 顺序。
- **先按精确 bound 分键：** 简化截断结果的隔离与证明；若重复存储影响收益，
  再独立比较带过滤的跨 bound 复用。
- **按字节比较，而非条目数：** 编码、对齐和替换元数据都会影响实际空间与吞吐。

## 集成边界

上下文归策略分析管理，封装存储与替换；`Game`、通用 `Policy` trait 和 `Runner`
不承担缓存管理。benchmark 和 `policy_ab` 在组装策略时绑定上下文。

跨决策 key 需显式包含原 evaluator 固定的容量和目标面子数。花色编号可省略，
因为 kernel 只计算九种牌的局部表，mask 在组合阶段才平移。

现有 benchmark 的预热与正式阶段使用不同实例；新增缓存生命周期需要明确区分
冷暖与初始化成本，因此采用新版报告协议。
