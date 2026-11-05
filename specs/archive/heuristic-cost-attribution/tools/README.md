# 归因研究的一次性复现材料

这些脚本仅用于复现[成本归因研究](../../../../docs/studies/heuristic-cost-attribution.md)，随[已完成的 Change](../design.md)归档。它们不被 Rust crate 依赖，不是生产能力或长期维护的 profiling 框架。

## 环境要求

- 64 位 Windows、64 位 CPython 3.10 或更新版本（使用了 `Counter.total()`）。本次使用 CPython 3.12.10。
- `dbghelp.dll`：依次尝试 `DBGHELP_DLL`、Windows Kits 的 `Debuggers/x64`、`System32`、裸 DLL 名。本次使用 Windows Kits 版本 10.0.26100.9457。符号解析需支持 PDB 内联记录。
- 带调试信息的 release 可执行文件及相邻 PDB；无需管理员权限或附加调试器。

## 运行

从仓库根目录执行。Git Bash 示例（`python` 必须是 Windows Python）：

```sh
CARGO_TARGET_DIR=target/prof3 RUSTFLAGS="-C debuginfo=2" \
  cargo build --release --example heuristic_bench
SAMPLE_INTERVAL_MS=1 STARTUP_SKIP_MS=150 MAX_DEPTH=48 \
  python specs/archive/heuristic-cost-attribution/tools/sample_cpu_windows.py \
  target/hca-sample -- target/prof3/release/examples/heuristic_bench.exe \
  --games 2000 --seed 7 --warmup 0
python specs/archive/heuristic-cost-attribution/tools/symbolize_inline.py target/hca-sample
python specs/archive/heuristic-cost-attribution/tools/aggregate_attribution.py target/hca-sample
```

PowerShell 等价命令：

```powershell
$env:CARGO_TARGET_DIR = "target/prof3"
$env:RUSTFLAGS = "-C debuginfo=2"
cargo build --release --example heuristic_bench
$env:SAMPLE_INTERVAL_MS = "1"
$env:STARTUP_SKIP_MS = "150"
$env:MAX_DEPTH = "48"
python specs/archive/heuristic-cost-attribution/tools/sample_cpu_windows.py `
  target/hca-sample -- target/prof3/release/examples/heuristic_bench.exe `
  --games 2000 --seed 7 --warmup 0
python specs/archive/heuristic-cost-attribution/tools/symbolize_inline.py target/hca-sample
python specs/archive/heuristic-cost-attribution/tools/aggregate_attribution.py target/hca-sample
```

每次采样使用不同的输出目录。多目录聚合只接受同一个可执行文件及 PDB；两个分析脚本会检查记录的镜像路径、镜像大小及当前 PDB 大小，拒绝混用构建。分析前须保留采样时的二进制与 PDB，不能原地重新构建后继续解析旧样本。

复现研究中的主要构建时，另用 `target/prof2`，将 `RUSTFLAGS` 设为 `-C force-frame-pointers=yes -C debuginfo=2`。无 frame-pointer 构建须单独分析，不能与它合并符号化。

## 脚本与采样选项

- `sample_cpu_windows.py`：启动目标进程，以线程 CPU 增量选择工作线程，周期性暂停并捕获上下文，用 `StackWalk64` 展开栈；保存样本、模块信息、运行日志及目标输出。
- `symbolize_inline.py`：解析 PDB 内联函数和行号，输出 self、inclusive 及调用者统计。
- `aggregate_attribution.py`：按成本类别聚合，并输出阶段归属、栈完整率和多次运行的对比。

| 环境变量 | 含义 |
| --- | --- |
| `SAMPLE_INTERVAL_MS` | 样本间 sleep，默认 1 毫秒；不是实际采样间隔，日志另报 `cadence_ms`。 |
| `STARTUP_SKIP_MS` | 启动后等待，默认 150 毫秒；之后还有约 120 毫秒的线程选择及符号初始化。 |
| `MAX_DEPTH` | 最大展开帧数，默认 96；研究使用 48。 |
| `HOLD_SUSPENDED=1` | 展开期间保持线程暂停，获得较完整调用栈，但本次使目标耗时增加到约 2.25–2.29 倍。交叉检查时同时设 `SAMPLE_INTERVAL_MS=5`。 |
| `DBGHELP_DLL` | 优先尝试的 dbghelp 路径。 |
| `CALLEE` | `symbolize_inline.py` 查询直接调用者的符号。 |

很短的进程可能在线程选择完成前退出，脚本会报告没有工作线程。采样窗口和方法的适用限制以研究正文为准，不把这些脚本的全进程函数榜直接视为策略内部排序。
