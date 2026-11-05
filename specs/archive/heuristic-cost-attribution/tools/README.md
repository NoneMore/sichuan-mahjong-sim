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

每次采样必须使用新的输出目录：若目录已存在且非空，采样脚本直接拒绝运行，避免旧 `samples.json` 把失败的一轮伪装成成功。一轮采样只有同时满足“目标进程自身退出码为 0”和“至少捕获到一个样本”才算有效；目标非零退出、采样/符号化异常、零样本都会以非零码结束，并且不写 `samples.json`。因此两个分析脚本看不到失败的采样结果。

镜像身份按内容校验，而不只是路径和大小：采样阶段在 `samples.json` 中记录 EXE 与 PDB 的绝对路径、字节数和 SHA256；`symbolize_inline.py` 与 `aggregate_attribution.py` 在解析之前都会重新计算磁盘上这两个文件的 SHA256，与记录不符即拒绝。多目录聚合还要求各目录记录同一构建身份（EXE 与 PDB 的 SHA256 均相同），因此同路径同大小的原地重建、混用不同构建都会被拒绝。旧采样若没有记录哈希会 fail closed 并提示重新采样，不会默默信任当前磁盘上的文件。分析前须保留采样时的二进制与 PDB。

复现研究中的主要构建时，另用 `target/prof2`，将 `RUSTFLAGS` 设为 `-C force-frame-pointers=yes -C debuginfo=2`。无 frame-pointer 构建须单独分析，不能与它合并符号化。

## 测试

从工具目录运行单元测试（用 mock 替代 Win32/dbghelp 调用，不需要目标构建或已有采样证据）：

```sh
python -B -m unittest test_tools -v
```

## 脚本与采样选项

- `sample_cpu_windows.py`：启动目标进程，以线程 CPU 增量选择工作线程，周期性暂停并捕获上下文，用 `StackWalk64` 展开栈；保存样本、模块信息、运行日志及目标输出。暂停后使用 `finally` 尝试恢复线程；恢复失败会使本轮采样失败，失败时终止自己启动的子进程并关闭句柄。退出码：`0` 成功，`2` 参数或输出目录不可用，`3` 采样/符号化失败，`4` 目标非零退出，`5` 零样本；除 `0` 外都不写 `samples.json`。
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
