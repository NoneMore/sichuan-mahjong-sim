# Offline BC 分阶段交付计划

## 范围与来源

将 `bc-offline-training` 拆为四个依次合并、可独立验证的实现 PR。代码抽取来源为 `298bbee`，拆分基线为 `main` 的 `ea5acec`（teacher collection）。完整行为范围见 `specs/archive/bc-offline-training/{spec,design}.md`。

保持规则、teacher、encoding v1 和 seed-based splits 不变。PyO3、在线游戏评估和 RL 不在本次实现范围；`python-policy-evaluation` 保留为独立提案。

## 分支与交付方式

先将本次 Change 文档直接提交到 `main`：BC 的原 spec/design、本计划，以及独立的 `python-policy-evaluation` 提案。保留现有文档路径和内容。

```text
更新后的 main -> feat/bc-prepare       -> PR 1 合并
更新后的 main -> feat/bc-dataset       -> PR 2 合并
更新后的 main -> feat/bc-training      -> PR 3 合并
更新后的 main -> feat/bc-recovery      -> PR 4 合并
```

1. 从包含上述文档的最新 `main` 创建 PR 1 分支，按阶段抽取代码与测试。新文件可按路径恢复，共享文件按改动内容拆分；原大分支保留为抽取来源。
2. 完成本阶段验收后，向 `main` 提交 PR；合并后再创建下一阶段分支。
3. README、当前架构与 validation 的实现描述随对应代码 PR 更新。
4. 最终对照原 diff 核对交付覆盖，再关闭被替代的大 PR。

每个实现 PR 包含对应测试与文档，在自己的 checkout 上独立构建、安装和运行。数据校验、安全写入等正确性要求随所属能力交付。标准检查沿用 [开发指南](../../../docs/development.md)，进度和验证结果记录在 PR 中。

## PR 1：Rust 数据准备与协议

**交付：** corpus snapshot -> 自包含 NumPy cache，可独立 prepare / verify。

### 抽取范围

- `CorpusSnapshot`、fallible episode visitor，以及 collector 读取与 provenance 的配套修改。
- `prepare/`、`bc-prepare`、crate 导出、contract/CLI 测试和协议文档。
- 固定 manifest snapshot、有界缓冲、验证后发布；中断 preparation 重新执行。
- Rust/Python 交接协议：数组布局、offset/target、metadata、版本、checksum、cache identity 和 provenance。

### 验收

- 小型 corpus 完成 prepare；移开原 corpus 后 cache 仍可 verify。
- round-trip 测试覆盖特征、split、候选顺序、target、metadata 和 forced decisions。
- snapshot 后 append 不改变数据成员；损坏输入或 visitor 失败不发布 cache。

## PR 2：Python Dataset Loader

**交付：** Rust cache -> mmap Dataset -> padded batch，尚无训练功能。

### 抽取范围

- Python 包基础、`cache.py`、`data.py`、dataset identity 和 fixtures。
- manifest、inventory、checksum、数组结构与 metadata 校验；`cache_id` 作为 Rust 定义的不透明 identity。
- sharded mmap、空 split、forced/eligible 查询、padding/mask 和确定性 batch iteration；保留最后 partial batch，mmap 使用 `num_workers=0`。
- `cache-info` / `version` CLI、Python CPU CI、安装说明与 `.gitignore`。

### 验收

- 独立安装、import、CLI 和 loader/data tests 通过。
- 测试覆盖多 shard、空 split、forced 标识、候选/target 对齐、padding 和错误输入。
- 用真实 Rust prepare 产物验证 Python load/batch，对照 counts、metadata、候选和 targets，记录可复现输入与命令。

## PR 3：最小可用 BC Training

**交付：** 全新训练 -> validation 选 best -> 保存/加载 -> frozen test evaluation；暂不支持 resume。

### 抽取范围

- `policy.py`、`metrics.py`，以及配置、训练循环、checkpoint 和 CLI 的非 recovery 部分。
- 按原规格实现 candidate scorer、masked loss、greedy/sampling、eligible decision 等权训练和 seeded 完整 epoch。
- 完整离线指标、validation NLL 选 best、冻结 test evaluation，以及配置、dataset identity、来源、依赖和 metrics 的 run 记录。
- versioned policy checkpoint 与安全发布，明确 policy payload 和可选 recovery state 的格式边界；不兼容变更显式调整版本。
- `train` / `evaluate` / `inspect` CLI，支持 CPU 和单本地 GPU。

### 验收

- CPU fixture 的 loss 下降；每 epoch eligible decisions 恰好遍历一次，包含 partial batch。
- padding/permutation、canonical-first ties、指标分母与分组、完整评估和 validation-only selection 测试通过。
- CLI 完成训练、保存、重新加载和 test evaluation；版本拒绝、安全写入和已有输出保护测试通过。

## PR 4：Recovery 和工程增强

**交付：** 从最后完成的 epoch 恢复，重放中断 epoch，并支持输出目录迁移。

### 抽取范围

- `checkpoint.py`、`train.py`、配置与 CLI 的 recovery 部分及相关测试。
- optimizer、epoch/step、RNG、遍历顺序与 best 状态的保存和恢复；scheduler 如有使用则一并恢复。
- `last.pt` / `--resume`、identity/configuration/device type/dependency 一致性检查、原子发布与故障注入。
- 跨目录恢复时重新发布 best/last，覆盖无新 improvement 和训练已完成的情况。
- 真实 corpus 端到端验证、单 GPU smoke，以及最终覆盖核对和文档收尾。

### 验收

- 固定 CPU runtime/configuration 下，恢复与连续训练的后续 sample order、step counts 完全一致；model/optimizer states 满足测试声明的数值容差。
- 不兼容恢复被拒绝；中断写入保留上一个可用 checkpoint；目录迁移的边界测试通过。
- prepare -> load -> train -> save/resume -> frozen test evaluation 完成，记录输入、命令、source、环境和结果。
- 单 GPU training/checkpoint reload smoke 通过并记录 CUDA 环境。

## 跨阶段抽取注意点

- `__init__.py`、`config.py`、`cli.py` 和测试 helpers 按阶段引入，解除对尚未交付模块的 imports。
- `test_cache.py` 中的 Trainer 检查归 PR 3；training/checkpoint/CLI 测试按非 recovery 与 recovery 拆分。
- README、架构、开发说明和 validation 记录随对应能力更新；重新抽取后的提交重新验证。
