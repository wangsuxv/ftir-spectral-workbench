# Merge Audit

审计日期：2026-08-23

## 结论

统一项目满足《FTIR Unified Workbench：基线处理与 2D-COS 项目合并实施规格书》的首版完成定义。最终独立审计发现的跨包 lineage、UI 失效与跨 Prepared 确认缺口均已修复并加入反向回归；当前没有 P0/P1 阻断项。科学核心没有被复制到 UI 或协调层，活跃工作流不存在第二套预处理路径。

## 完成定义逐项核对

| # | 规格要求 | 实现证据 | 状态 |
|---:|---|---|---|
| 1 | 基线是唯一预处理权威 | `BaselineWorkflowService` 只调用 `ftir_baseline`；Prepared adapter 固定读取 `PipelineResult.analysis_data` | 通过 |
| 2 | 2D 不重新基线 | `TwoDCOSWorkflowService` 只接受 `PreparedSpectralDataset`，直接调用 `compute_2dcos` / `compute_cross_2dcos` | 通过 |
| 3 | baseline-only 是正式完成状态 | 项目状态机与 UI 均支持导出并结束；mock 测试证明不调用 2D | 通过 |
| 4 | baseline ZIP 可独立复现 | bundle 含输入、吸光度、基线、校正谱、QC、recipe、Prepared CSV/meta 和 manifest | 通过 |
| 5 | 当前校正谱以内存继续 2D | `continue_to_twodcos()` 直接传递当前 baseline result，经唯一 adapter 构造 Prepared | 通过 |
| 6 | corrected CSV 可稍后载入 | 支持裸 CSV、CSV+sidecar、baseline ZIP 与 `PreparedExport`；精确往返测试通过 | 通过 |
| 7 | 2D 与原项目结果一致 | 保留原 2D 回归；新内存链与 legacy baseline-none 链矩阵一致测试通过 | 通过 |
| 8 | 原测试与新增测试全部通过 | 135 baseline + 186 2D + 75 integration/UI = 396 passed | 通过 |
| 9 | recipe、QC、hash、manifest 串联 | 三类 bundle、三项 parent lineage、子结果逐项验证和项目嵌套校验通过 | 通过 |
| 10 | UI 不复制科学算法 | UI 只组织 service/config/state/render；科学公式仍位于两个来源包 | 通过 |

## 关键禁止项审计

- 统一 service/UI 不调用 `ftir2dcos.pipeline.run_pipeline` 或 legacy `preprocess_dataset()`。
- Prepared 默认不读取 `view_data` 或 `optional_normalized`。
- 默认 2D 输入不做归一化；显式科学归一化会生成新的 Prepared hash。
- 显示归一化、色阶、等高线、字体和线宽不改变科学 fingerprint，也不触发矩阵重算。
- 不连续区间分别保留为 self/cross block，不拼接为伪连续波数轴。
- Cross block 会逐元素验证扰动值、标签、谱数和顺序。
- 不同 Prepared blocks 的 cross 即使上述内容一致也默认阻断，必须显式确认；确认状态与双方 fingerprint/hash 写入结果。
- 项目 builder/verifier 会把每个 2D 的三项 parent lineage 与同包 baseline sidecar/manifest 交叉核对；重新签名的错配包也会被拒绝。
- Baseline 配置或原始数据变化会清除 baseline、Prepared 与 2D；2D 科学配置变化只清除 2D。
- `BASELINE.dpt` 默认排除；文件名按数值扰动排序，不采用词典序。

## 修复与兼容面

- 已修复 frozen/slots 下 `PipelineConfig.to_dict()` 的 zero-argument `super()` 问题。
- 已修复旧 baseline Streamlit 对导出 ZIP 含“2D-COS 矩阵”的错误描述。
- 旧 `ftir2dcos.pipeline`、conversion 和 preprocessing 仅作为 legacy 回归兼容面保留，不是统一工作流入口。
- 两套来源测试放在独立包目录并使用 pytest importlib 模式，避免同名测试模块冲突。

## 数据与真实演算审计

- 私有 DPT 序列已在本机完成格式、有限值、轴一致性和扰动排序审计。
- 非等间隔扰动会被显式警告，并在 metadata/manifest 中记录 index-order Noda policy。
- 从 baseline ZIP 重新载入 Prepared 后复算 self/cross 结果，交付矩阵与复算结果逐元素一致，QC 通过。
- baseline、2D、project 三层清单以及 `.ftirw` 内嵌子包均通过哈希验证。
- 原始谱、数据派生指纹、产物哈希与生成包均已从公开 Git 历史排除，只保留在本机忽略路径中。

## 自动化证据

```text
.venv/bin/python -m pytest -q -p no:cacheprovider
396 passed in 36.81s

.venv/bin/ruff check src tests ui scripts
All checks passed!

.venv/bin/mypy src/ftir_workbench --no-incremental
Success: no issues found in 15 source files

私有数据演算与 manifest 重放：本机通过，不进入公开 CI 或公开制品。
```

## 保留风险与后续项

1. 真实样品的默认演示 recipe 可能触发负残差等 QC 警告，需要根据样品、仪器和研究目的由领域专家调整并签核参数；程序不会隐藏或裁剪结果。
2. 非等间隔扰动目前使用标准 index-order Noda 策略。如果研究需要真实时间加权，应作为新的、经文献和基准验证的科学 convention 实现，不能静默改变现有定义。
3. PySide6/macOS `.app` 按规格留待科学合并稳定后的后续阶段；它不影响本次 Streamlit 首版完成状态。
