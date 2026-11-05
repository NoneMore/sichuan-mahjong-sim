# 启发式策略成本归因：优化后复测（2026-11-05）

> 本文保留 one-pass discard 之前 `6d3173b` 的历史结论。新版见
> [one-pass discard 后复测](heuristic-cost-attribution-discard.md)，普通距离 DP 已非首要热点。

## 结论

在 **`6d3173b2857f42ff2681061e5d9f203172b52d20`** 上，下一轮优化仍应优先研究**逐花色精确 DP**：主采样中纯距离 `suit_table` self 为 **48.70%**，掩码版 `suit_table_with_mask` 为 **35.23%**，合计 **83.93%**。重复、换种子、长批次和采样方式交叉检查均支持这一排序。

[第一次归因](heuristic-cost-attribution.md)之后已合入七对子闭式/花色 DP 剪枝（`dc6ff5c`）及 one-pass ukeire（`6d3173b`）。旧的“ukeire 对假想摸牌反复计算 deficit”已被消除，当前主要路径是：

- 候选排序：`rank_discards → deficit → suit_table`。
- 有效进张：`ukeire → deficit_with_mask → suit_table_with_mask`。

若只选择一个入口，先看纯距离版的总成本；但候选排序对每个候选计算距离，只对最小距离候选计算 ukeire，**份额不能说明单次构建谁更贵**，也不决定具体优化方案。本轮未修改生产 Rust、未插桩或实施优化。

## 测量条件与证据

沿用[归档方法](../../specs/archive/heuristic-cost-attribution/design.md)和[采样工具](../../specs/archive/heuristic-cost-attribution/tools/README.md)，使用未改变的 `heuristic-benchmark-v1` 四席启发式自对弈入口。

- 测量前工作树干净；Windows 10 Pro 19045 x64，Ryzen 7 3700X（8 核/16 线程）。
- rustc/Cargo 1.98.0，LLVM 22.1.8，`x86_64-pc-windows-msvc`；64 位 CPython 3.12.10，Windows Kits dbghelp 10.0.26100.9457。
- 三种 release 构建：默认无自定义 flags；`fp` 加 `-C force-frame-pointers=yes -C debuginfo=2`；`nofp` 仅加 `-C debuginfo=2`。release 不执行 debug 距离一致性断言。
- 直接运行保存的 EXE，单线程，每局决策上限 10,000；每阶段新建 Runner/策略，阶段内跨局复用。预热不计入正式结果。
- 性能进程串行执行，不与构建/测试重叠；未绑定 CPU，其他系统活动未严格控制。采样按 120 ms CPU 增量选工作线程，其他线程的累计 CPU 均为 0。

[逐轮证据](heuristic-cost-attribution-refresh/measurements.json)保留 benchmark 字段、构建哈希、调用上下文和稳定性计数。原始样本、日志、符号输出及 EXE/PDB 留在本地 `target/hca-refresh-6d3173b/`，未纳入仓库。分析旧样本前必须保留原构建；工具会重新校验 EXE/PDB SHA256。

## 正式基线

默认 release，2,000 局、预热 20 局。表中为独立中位数，括号为范围。

| 指标 | seed 7，5 次 | seed 10007，3 次 |
| --- | ---: | ---: |
| 决策数 | 113,841 | 115,351 |
| 批次时间（秒） | 6.500741（6.402077–6.623493） | 6.615957（6.593407–6.710023） |
| 策略时间（秒） | 6.388596 | 6.504026 |
| 规则时间（秒） | 0.088271 | 0.088514 |
| 吞吐量（局/秒） | 307.657 | 302.299 |
| 每轮策略时间 / 批次时间 | 98.246–98.295% | 98.308–98.312% |

策略仍占约 **98.3%**。批次还包含建局、聚合和循环等成本，组件时间无需相加等于批次时间。本轮未重建旧版本做交错 A/B，不将跨日期历史差值当作受控优化收益。

## 函数采样与调用上下文

`STARTUP_SKIP_MS=150`、`MAX_DEPTH=48`；通常捕获上下文后立即恢复线程再展开栈（sleep 1 ms），完整栈检查则保持暂停直到展开结束（sleep 5 ms）。

| 配置 | 样本数 | 纯距离 DP self | 掩码 DP self | 合计 | 完整栈 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp，2,000 局，seed 7，warmup 0 | 1,040 | 47.12% | 35.87% | 82.98% | 63.17% |
| 同配置重复 | 1,048 | 48.09% | 35.97% | 84.06% | 64.60% |
| fp，2,000 局，seed 10007，warmup 0 | 1,082 | 48.98% | 33.64% | 82.62% | 60.72% |
| fp，10,000 局，seed 777，warmup 0 | 5,480 | 49.29% | 35.24% | 84.53% | 64.73% |
| fp，2,000 局，seed 7，warmup 20 | 1,045 | 47.46% | 35.50% | 82.97% | 64.11% |
| **以上主采样合计** | **9,695** | **48.70%（4,721）** | **35.23%（3,416）** | **83.93%** | **64.03%** |
| nofp，10,000 局，seed 7，warmup 0 | 6,289 | 48.93% | 35.28% | 84.21% | 49.74% |
| fp，10,000 局，seed 7，保持暂停 | 5,908 | 43.69% | 35.19% | 78.88% | 96.46% |

**必须按最内层内联完整符号名拆开两个 DP。** 旧 `aggregate_attribution.py` 的 `dp.suit_table` 用子串匹配，会合并 `suit_table_with_mask`；其他距离桶也有同类问题。本轮旧桶合并值 83.93% 不是纯距离函数份额，不能与旧研究的 80.70% 直接相减来衡量收益。

主采样的次要桶为哈希 2.58%、共享规则/eval 1.12%、VCRUNTIME 4.19%。后者被旧脚本命名为 `allocator`，实际只是模块分类，不能全算作分配；同理系统 DLL 桶不等于真实等待时间。

保持暂停的一轮中，5,699/5,908 个样本同时含 `Runner::drive` 与 `main`，作为完整栈判据：

- 两个 DP 的 2,581/2,079 个 self 样本全部位于完整策略栈下，外层符号分别为 `deficit` / `ukeire`，支持上述两条路径。
- 外层 self 为 `deficit` 的 2,682 个样本中，2,578（96.12%）由 `rank_discards` 直接调用，没有 ukeire 调用者；外层 self 为 `ukeire` 的 2,226 个样本中，2,071（93.04%）由 `rank_discards` 直接调用。
- `choose`、`rank_discards`、`deficit`、`ukeire` 的 outer-symbol inclusive 分别为 94.14%、88.64%、50.58%、41.50%。这里包含被采样帧自身，沿用聚合脚本口径；`symbolize_inline.py` 的 inclusive 只计调用者帧，两者不可混用。
- 199 个哈希、230 个 VCRUNTIME 样本均在策略链下；80 个共享规则/eval 样本中，34 个属于策略、46 个属于完整非策略链，不能全部计入策略。

这些份额是**整个工作线程采样窗口**的权重，不是策略内部归一化比例。Inclusive 包含子调用，不能相加；内联会隐藏中间栈帧，路径解释结合了 PDB 和源码。

## 可信度与局限

- **排序稳定但不承诺收益。** 所有低扰动配置中纯距离为 47.12–49.29%、掩码为 33.64–35.97%；保持暂停也未改变排序。未测调用次数、缓存命中率、分配次数、峰值内存或单调用延迟。
- **采样有扰动，符号构建不代表默认吞吐量。** 前三轮交错运行的默认/fp/nofp 批次中位数为 6.416/6.868/6.805 秒。低扰动实际 cadence 为 5.13–6.01 ms；部分采样运行反而快于非采样控制，不能据此宣称无开销。保持暂停的 cadence 为 11.80 ms，耗时 70.126 秒，相同构建/负载的后补非采样控制为 33.080 秒（约 2.12 倍），仅用于上下文交叉检查，不与主采样合并。
- **窗口不等于正式批次。** 启动等待、哈希、符号初始化和线程选择会错过开头，末尾可能包含输出。无预热短批次的窗口长度差约 5.6–6.5%，seed 777 长批次约 1.1%，排序保持。warmup 20 的采样窗口 6.041 秒短于正式批次 6.365 秒且持续至进程退出，支持采样在正式阶段开始后启动；没有精确阶段门控。
- **工作负载可比不等于逐动作等价。** 所有进程成功退出，构建身份校验通过；同一 seed/game 范围下，各构建、采样和预热配置的完成数、决策数、结束原因、得分及胜局聚合字段一致，总分为零。详细字段见证据 JSON；本轮没有重新做跨版本逐动作 replay。

## 复现

从上述提交及仓库根目录，先完成三种构建，再串行运行。构建/采样命令沿用[工具说明](../../specs/archive/heuristic-cost-attribution/tools/README.md)，输出目录换成新目录，二进制保留至分析结束。基线参数为 `--games 2000 --seed 7 --warmup 20`（5 次）及 seed 10007（3 次）。前三轮按 `default/fp/nofp`、`nofp/fp/default`、`fp/default/nofp` 交错，每轮随后运行默认 seed 10007，再补 seed 7 两次。

采样按上表逐行运行。只将前五轮同一 fp 镜像的目录传给聚合器合并，nofp 和保持暂停分别分析。非采样控制使用同一 EXE 和完全相同参数，完整栈控制在所有采样后补跑。逐轮参数和次序也记录在证据 JSON 中。

归档符号化器给出 innermost self；额外上下文和外层/内联交叉计数可按下列片段重建。将归档工具目录加入 `sys.path`，`d` 指向一轮样本目录；使用 Windows Python。

```python
import json
from collections import Counter
from pathlib import Path
import symbolize_inline as S
import aggregate_attribution as A

mods, image = S.load_image(d)  # 校验 EXE/PDB SHA256
S.inline_base = 0x140000000
inline = S.Inline(mods[0]["path"], S.inline_base, mods[0]["size"])
P = "sichuan_mahjong_sim::policy::analysis::"
dp_names = (P + "suit_table", P + "suit_table_with_mask")
self_names, contexts, callers, outer_inner = Counter(), Counter(), Counter(), Counter()
for s in json.loads((Path(d) / "samples.json").read_text())["samples"]:
    rip = s["rip"]
    key = S.label(rip)
    if rip["module"] == mods[0]["name"]:
        key = inline.innermost(S.inline_base + rip["addr"] - mods[0]["base"])[0] or key
    self_names[key] += 1
    names = {S.label(f) for f in [rip] + s["frames"]}
    full = "sichuan_mahjong_sim::runner::Runner::drive" in names and "main" in names
    policy = "sichuan_mahjong_sim::policy::heuristic::impl$1::choose" in names
    category = key if key in dp_names else A.bucket(key, rip["module"])
    contexts[category, full, policy] += 1
    if rip["symbol"] in (P + "DistanceEvaluator::deficit", P + "ukeire") and s["frames"]:
        callers[rip["symbol"], S.label(s["frames"][0])] += 1
    if key in dp_names:
        outer_inner[rip["symbol"], key] += 1
print(self_names.most_common(10), contexts, callers, outer_inner)
S.P.dbg.SymCleanup(inline.proc)
```

不含策略帧的截断栈仍须保留未归属，不能当作完整非策略链。后续优化须另外验证策略等价、冷暖缓存行为及默认 release 的交错性能对照。

## 验证

本轮 profiling 已通过 `cargo fmt --all -- --check`、`cargo clippy --all-targets -- -D warnings`、debug/release 的 `cargo test --all-targets`（各 238 项），以及归档工具 `python -B -m unittest test_tools -v`（24 项）。复现片段已在本地样本执行核验；测量记录不构成跨机器性能保证。
