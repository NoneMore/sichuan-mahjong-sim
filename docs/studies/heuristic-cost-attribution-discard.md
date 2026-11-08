# 启发式策略成本归因：one-pass discard 后复测（2026-11-06）

## 结论

测量提交 **`06fb9198ba76324ef80dc29e4118caca7feff303`**。下一轮应优先研究
**`suit_table_masked` 的逐花色精确 DP，首先关注 ukeire 的 needed-mask 路径**。

主采样共 6,346 个样本：masked DP self **79.09%**（needed **57.25%**、surplus
**21.84%**），普通距离 `suit_table` 仅 **3.50%**。重复、换种子、长批次和采样方式
交叉检查均支持这一排序；策略计算仍占正式批次约 **97.4%**。

[上一轮](heuristic-cost-attribution-refresh.md)的普通距离 DP 优先级已不适用：
one-pass discard 将逐候选 `deficit` 改为一次原手牌 surplus 分析，当前主要路径为：

- 有效进张：`ukeire → deficit_with_mask → suit_table_masked<false>`（needed）。
- 候选距离：`rank_discards → deficit_with_surplus → suit_table_masked<true>`（surplus）。

两条路径共享泛型 DP，但不能把全部 masked 成本归给 ukeire。份额不是调用次数、
单调用延迟或预期加速比。本轮未修改生产 Rust、未插桩、未实施优化。

## 测量条件

沿用[归档方法](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/specs/archive/heuristic-cost-attribution/design.md)和
[采样工具](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/specs/archive/heuristic-cost-attribution/tools/README.md)。
工作负载为 `heuristic-benchmark-v1` 四席启发式自对弈，每阶段新建 Runner/策略，
阶段内跨局复用，每局决策上限 10,000；预热不计入正式批次。

- 测量前工作树干净；Windows 10 Pro 19045 x64，Ryzen 7 3700X（8 核/16 线程）。
- rustc/Cargo 1.98.0、LLVM 22.1.8，`x86_64-pc-windows-msvc`；64 位 CPython 3.12.10，
  Windows Kits dbghelp 10.0.26100.9457。
- 独立 release 构建：default 无自定义 flags；fp 使用
  `-C force-frame-pointers=yes -C debuginfo=2`；nofp 仅使用 `-C debuginfo=2`。
- 性能运行串行，不与构建、测试重叠；未绑定 CPU，后台活动未严格控制。

## 正式基线

默认 release，2,000 局、预热 20 局；各字段独立取中位数，括号为范围。

| 指标 | seed 7，5 次 | seed 10007，3 次 |
| --- | ---: | ---: |
| 决策数 | 113,841 | 115,351 |
| 批次时间（秒） | 3.931742（3.814684–4.013775） | 3.991322（3.974791–4.152507） |
| 策略 / 规则时间（秒） | 3.831368 / 0.078137 | 3.891275 / 0.078876 |
| 吞吐量（局/秒） | 508.680 | 501.087 |
| 每轮策略时间 / 批次时间 | 97.367–97.471% | 97.430–97.493% |

本轮未重建旧版做交错 A/B，不把跨日期耗时或份额差当作受控加速比。
one-pass discard 的历史受控实验见[验收摘要](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/specs/archive/one-pass-discard/results.md)。

## 采样与归因

`STARTUP_SKIP_MS=150`、`MAX_DEPTH=48`；低扰动方式捕获上下文后恢复线程再展开栈，
sleep 1 ms；完整栈检查保持暂停至展开结束，sleep 5 ms。
下表均为**整个工作线程采样窗口**的 self 份额，不是策略内部归一化比例。

| 配置（局数 / seed / warmup） | 样本数 | needed | surplus | 普通 DP | 完整栈 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp，2000 / 7 / 0 | 644 | 58.70% | 22.36% | 2.80% | 62.42% |
| 同配置重复 | 669 | 56.05% | 22.27% | 4.48% | 58.59% |
| fp，2000 / 10007 / 0 | 683 | 55.05% | 23.43% | 3.22% | 58.57% |
| fp，10000 / 777 / 0 | 3,688 | 57.97% | 21.18% | 3.52% | 59.63% |
| fp，2000 / 7 / 20 | 662 | 55.29% | 22.96% | 3.32% | 60.57% |
| **以上主采样合计** | **6,346** | **57.25%** | **21.84%** | **3.50%** | **59.79%** |
| nofp，10000 / 7 / 0 | 5,151 | 56.11% | 23.22% | 3.05% | 23.43% |
| fp，10000 / 7 / 0，保持暂停 | 3,530 | 53.85% | 21.50% | 3.43% | 97.45% |

PDB 最内层名称 `suit_table_masked` 不带 const 参数，因此按外层**完整符号名**
`policy::analysis::ukeire` / `policy::analysis::DistanceEvaluator::deficit_with_surplus`
分别归为 needed / surplus；普通 DP 按最内层完整名 `policy::analysis::suit_table` 单列。
主采样分别为 3,633 / 1,386 / 222 个样本，无未归属 masked 样本。
旧聚合器的子串桶 `dp.suit_table` 会合并为 82.59%，不能解释成普通 DP 份额。

保持暂停的一轮有 3,440/3,530 个完整栈（同时包含 `Runner::drive` 与 `main`）；
全部 2,660 个 masked 和 121 个普通 DP self 样本均在完整策略链下。
哈希 146、VCRUNTIME 154 个样本也在策略链下；共享 rules/eval 的 53 个样本中，
仅 24 个属于策略。主采样哈希占 3.53%、VCRUNTIME 模块占 4.24%，均远低于 DP。
VCRUNTIME 不等于分配器成本，无策略帧的截断栈也不能当作非策略链。

## 局限与验证

- 采样未精确门控。无预热短批次的 `sampling_wall_s` 比该轮正式批次墙钟短约
  9.7–9.9%，长批次约 2.0%；可能混入末尾输出。主要判断由无预热和长批次支撑。
- fp/nofp 低扰动实际 cadence 为 5.27–5.55 / 3.72 ms，栈完整率有限。
  保持暂停为 11.35 ms、批次 40.413 秒，同负载非采样控制为 19.716–20.265 秒，
  故完整栈一轮仅作上下文检查，不与主采样合并。
- 26 次运行均成功、完成数等于请求数、总分为零；同负载非计时聚合字段在构建和采样间
  一致，也与上一轮匹配负载一致。这不是逐动作等价证明；本轮未重做跨版本 replay。
- fmt、warnings-denied Clippy、debug/release 全目标测试（各 **258 项**）及归档工具
  测试（**24 项**）通过。未测调用数、缓存命中率、分配次数、内存或单调用延迟。

## 复测与材料清理

相关 Windows 采样工具已从当前工作树退役；如需重做采样，须从[清理前快照](https://github.com/NoneMore/sichuan-mahjong-sim/tree/f57e48fdd39a9571371194b46d61bfba2b20d795/specs/archive/heuristic-cost-attribution/tools)取回，并重建所述历史环境。

从上述提交构建 `cargo build --release --example heuristic_bench`，直接运行：

```sh
target/release/examples/heuristic_bench.exe --games 2000 --seed 7 --warmup 20
target/release/examples/heuristic_bench.exe --games 2000 --seed 10007 --warmup 20
```

seed 7 重复 5 次、10007 重复 3 次；前三轮按 default/fp/nofp、nofp/fp/default、
fp/default/nofp 交错。按采样表逐轮复测，命令见归档工具说明；仅合并同一镜像的前五轮，
nofp 和保持暂停分别分析。先校验 EXE/PDB SHA256，再用内联符号和上述外层名拆分路径；
若外层也被内联，应重建归属证据，不能照搬分类假设。

按用户清理要求，本轮原始样本、专用构建、日志、一次性脚本和逐轮 JSON 已删除，
仅保留本文历史摘要。不能从现存材料重新核验原始计数，但可按提交、参数和归档方法
重新采集同类证据；这些数字不是当前性能保证。
