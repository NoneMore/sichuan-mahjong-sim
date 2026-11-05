# One-pass discard：历史验收摘要（2026-11-05）

**结论：保留优化，Change 已归档。** 原合同见 [spec.md](spec.md)。生产改动仅在
`src/policy/analysis.rs` 和 `src/policy/heuristic.rs`，不改变规则、策略选择、信息边界或公开 API。

按用户提交前的清理要求，已删除一次性工具、源码副本、EXE、原始计时、计数 JSON 和轨迹，
替代原规格的实验材料保留要求。下述数字仅为历史摘要，不能从仓库原始样本重新核验，
也不是性能保证；必要实现和永久回归测试保留。

## 实现与正确性

原手牌一次精确分析返回 deficit 与所有最优目标的 surplus 并集，推导全部候选距离。
标准型共享 masked DP，七对使用小型 pair-unit DP；surplus 缓存惰性分配，与普通距离及
Ukeire needed-mask 缓存分离。空/无效候选短路、最低距离筛选、ukeire、番数及碰杠门控不变。

新增 20 项回归测试覆盖逐候选普通 deficit 参考、标准/七对并列、缺门、0–4 副露与容量、
候选顺序及无效候选、缓存和排序/门控接线。分析层含 150 个物理采样手牌及 30 个边界手牌，
排序层含 60 个物理状态；生成方法、固定 seeds 和样本量断言均留在测试源码。
排序参考复用未改动的 ukeire/fan/门控 helper，不声称重新独立验证这些 helper。

跨版本 seeds `7..38`、`10007..10038` 共 64 局 / 3514 决策，逐动作、合法列表指纹、
得分、胜者、决策数与终局原因一致。每局第 20 个动作后 clone，换新策略续局也完全一致。
清理后 fmt、warnings-denied Clippy、debug/release 全目标测试均通过，各 **258 tests**。

## 性能方法与结果

基线 `2d7d2d0`（生产代码与规格基线 `a243b2f` 相同）；候选为该提交上的本次实现，
测量后仅修改生产注释和测试。环境为 Windows 10 build 19045、Ryzen 7 3700X、
`x86_64-pc-windows-msvc`、rustc/Cargo 1.98.0、LLVM 22.1.8，默认 release，无自定义
flags 或 CPU affinity，系统后台活动未严格控制。

工作负载为 `heuristic-benchmark-v1` 四席启发式自对弈，2000 局、预热 20 局、起始 seed
7 / 10007。先各做 3 对 B/B 校准；再各做 5 对 B/C、C/B、B/C、C/B、B/C，串行、无剔除，
不与构建、测试、计数或轨迹采集并发。以下降幅为每对 `1 - candidate / baseline` 的中位数。

| seed | total 配对中位降幅（范围） | policy 配对中位降幅 | total / policy 改善对 | 最大绝对 B/B total / policy 差 |
| --- | --- | --- | --- | --- |
| 7 | 36.23%（34.11–37.57%） | 36.84% | 5/5，5/5 | 3.43% / 3.40% |
| 10007 | 34.20%（32.98–35.15%） | 34.80% | 5/5，5/5 | 3.73% / 3.71% |

两组均满足预定的 ≥2%、超过 B/B 波动且至少 4/5 对改善的保留门槛，非计时结果一致。
独立 total 中位秒为 6.018268→3.837655 / 6.159982→4.078815，不用于配对判定。

隔离插桩统计预热后的实际 lookup/hit/miss 和 miss 分支 DP 构建，检查 `request=hit+miss`、
`build=miss`。两 seed 的 rank 次数为 53504 / 54751，候选数为 494023 / 504494；
候选分布不变，每次 rank 恰好一次原手牌 surplus 分析。旧候选生成区间的 plain suit 缓存
命中率约 55.9%，不能按候选数推算 DP 节省；全批次实际花色构建从 1051439→557925 /
1078498→574452，减少 46.94% / 46.74%，Ukeire mask 构建数不变。

另用无详细计数事件的 region-only 构建，每 seed 三对 B/C、C/B、B/C。计时从 rank 入口
到 scored vector 建成，包含 evaluator 初始化与全部 surplus 成本，排除 ukeire/fan/选择；
区间中位秒为 3.171724→1.018121 / 3.251635→1.047118。该计时仍有每 rank 一对 Instant
的扰动，不替代正式端到端结果；详细计数版的耗时未用于结论。未测 allocator 或峰值内存。

复测时在独立 checkout 构建 `cargo build --release --example heuristic_bench`，保存二进制，
按上述顺序串行运行 `heuristic_bench --games 2000 --seed 7 --warmup 20` 及 seed 10007。
须重新采集原始观测、逐动作轨迹；计数及区间成本需按上述边界重新隔离插桩，不能恢复已删除数据。
