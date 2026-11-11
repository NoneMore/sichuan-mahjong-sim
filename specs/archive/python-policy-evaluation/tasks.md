# Python policy evaluation 交付拆分

七个交付单元，共用 [spec](spec.md) 和 [design](design.md)。每项随代码交付
测试及受影响的使用/架构说明，可作为 PR 边界。

## 依赖

```text
T1 -> T2 -> T4 --+
                 +-> T5 --+
            T3 --+        +-> T7
                     T6 --+
```

T3 使用数值 fixture，T6 使用终局记录 fixture，均可与绑定工作并行。
绑定构建选择在 T1 解决；`last.pt` 诊断入口不纳入本计划。

## T1：构建与首个决策

**交付：** 独立 extension 可安装、import、reset 并读取首个决策。

- 验证 PyO3/NumPy、打包工具、Python 支持范围与 Cargo workspace 安排，
  将构建结论写回 design。
- 建立绑定及在线 Python 包，接通 Game -> encoder -> owned arrays。
- 增加构建、import 和 reset smoke 检查。

**完成信号：** 项目 `.venv` 可读取与 Rust fixture 一致的首决策；构建证据
覆盖现有 Rust 1.75 baseline、core/collector 及独立离线安装路径。

## T2：完整环境生命周期（依赖 T1）

**交付：** Python 用简单索引策略完成整局并取得终局结果。

- 实现 reset generation、动作缓存、step、计数、decision limit 和终局返回。
- 扩展离线/在线编码 fixture，覆盖各 stage 和观测边界。

**完成信号：** 完整对局 smoke 与场景测试覆盖响应、forced choices、获胜后
继续游戏、reset、快照隔离及无状态变更的请求拒绝；错误与正常终局可区分。

## T3：无标签 batch 与冻结推理

**交付：** 无需 extension，数值 fixture 经共享 checkpoint 推理得到本地索引。

- 提取共享 padding/mask 能力，离线 collator 在其上附加监督字段。
- 复用 loader、policy 和 greedy selection，接通 forced bypass。

**完成信号：** padding、partial batch、ties、非有限 logits、forced bypass
和版本兼容测试通过；推理不依赖训练数据或 run 目录，现有离线测试保持通过。

## T4：Teacher replay 与 Rust 元数据（依赖 T2）

**交付：** Python teacher 驱动与同 seed native 执行一致，并可读取编译来源和
既定 seed split。

- 接通原始 Decision 上的 teacher 查询、canonical target lookup 和显式 cache。
- 复用 Rust split/编译来源 API，建立 native/Python replay probe。

**完成信号：** 语义动作序列、计数和终局一致；teacher 查询保持 pending
不变，cache 不改变选择。split 与 Rust 一致，来源不受调用目录影响。

## T5：多局混合执行（依赖 T3、T4）

**交付：** 同步运行多个 neural/teacher 混合游戏并返回逐局结果。

- 按座位策略分组 pending decisions，批量推理并路由索引。
- 处理进度差异、forced bypass、计数/追踪、终局退出和错误传播。

**完成信号：** 冻结 checkpoint 与 heuristic 完成多局；fixture 检查跨 handle/
decision 路由、不同 stage/候选数、partial batch 和先后终局。游戏或推理
故障停止执行，无替代提交。

## T6：计划、原始记录与汇总

**交付：** 终局 fixture 经计划、持久化记录生成可复算 summary。

- 定义记录协议和六种 placement 计划，按 seed block 汇总规格指标。
- 接通 checkpoint 内容身份、Python 来源采集及 Rust 元数据记录字段。
- 实现计划先行持久化、逐局写入和完整性校验；bucket 查询留给 T7 接入。

**完成信号：** fixture 统计可从记录独立复算；内容身份及实际来源可辨识。
输出失败或缺失/重复/不一致记录保留完成数据，不产生完整 summary。

## T7：正式比较与集成验证（依赖 T5、T6）

**交付：** 可复现的六 placement 比较入口及分项成本记录。

- 接通配置入口、checkpoint、Rust split、运行身份、T5 执行与 T6 记录。
- 完成 validation 诊断/test 冻结比较，输出失败时停止调度。
- 测量 environment、teacher、inference 成本；更新命令说明、架构与验证记录，
  在 CI 接通绑定/在线路径并保留独立离线测试。

**完成信号：** 小型真实 checkpoint 比较完成每 seed 六局，元数据齐全，
summary 可复算；集成故障测试验证 partial-run 语义。适用的 Rust、离线
Python 和绑定/在线检查通过，保留输入、环境、命令及测量结果。
