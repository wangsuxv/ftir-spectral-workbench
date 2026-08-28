# Project Status

更新日期：2026-08-29

## 当前状态

FTIR Spectral Workbench 的 v0.2.0 基线冻结版已在本地分支 `feat/v0.2-baseline-frozen` 完成发布验收并形成本地阶段提交。项目版本 metadata 为 `0.2.0`；本次没有打 tag、创建 GitHub release 或推送远端，远程公开项目未被改变。

本版本从冻结的 v0.1.0 科学基线开始，只实施四项 UI/协调层增量，不改变 baseline 数值模型、配置 schema、workflow state 或 baseline bundle。

## 已交付

- 保留 `ftir_baseline` 与 `ftir2dcos` 两个经过回归测试的科学包。
- 新增 `ftir_workbench` 协调包，提供不可变 Prepared 数据合同、配置封装、fingerprint、状态失效、baseline/2D/project service 与导出校验。
- 新增统一 Streamlit 工作台和统一 CLI。
- baseline-only 是正式成功路径；用户可下载结果后结束，不会触发 2D 计算。
- 当前基线结果可直接通过内存继续 2D，不写回 CSV、不重新基线。
- corrected CSV、sidecar metadata、baseline ZIP 均可作为以后继续 2D 的入口。
- 支持多个 self 区间、矩形 cross-range、峰序分析、canonical 与 2dpy-compatible convention。
- 不同 Prepared blocks 的 cross 默认阻断；逐项兼容检查通过后仍需调用方显式确认，并记录双方完整血缘。
- baseline、2D 和 `.ftirw` 项目包均有 SHA-256 manifest；2D 子结果保存三项父级血缘。
- CI 使用锁定环境覆盖 Python 3.11、3.12、3.13，并运行基于合成/示例数据的 pytest、Ruff、Mypy 与 wheel 构建；私有数据重放仅在本机执行。

## v0.2.0 本地增量

- Coarse Preview 在完整序列上运行临时 draft，可查看实际谱或 first/middle/last/mean/median；Candidate Gallery 复用既有六类候选 API，且不会自动采用第一名。
- Fine Preview 显示 `A_raw`、`A_for_baseline`、`B_coarse`、`B_fine`、`B_total`、Corrected、粗调残余及 fitted anchor diagnostics。
- Preview 与正式状态分离；只有 Adopt/Apply 才写入 `baseline_config` 并使 baseline/Prepared/2D 后代失效。
- Series Consistency & QC 显示五张原始结果热图、完整逐谱 QC、三组趋势、筛选/CSV 和单谱 drill-down；页面不重新计算 QC。
- A↔T 仅创建显示副本或独立派生 CSV。转换不裁剪；负 A 可产生 `%T>100`；派生文件明确标注非仪器原始透过率，也不加入 baseline ZIP。
- Cross 服务仍以 `combinations(ranges, 2)` 计算 `C(n,2)` 个 unique pairs。stored/reverse 两个 orientation 形成 `n(n-1)` 个查看方向，不增加 `CrossRangeResult`，也不重复 peak-order evidence。
- Cross 2 严格使用 `Phi_reverse=Phi_stored.T`、`Psi_reverse=-Psi_stored.T`，实际 row/column ranges 随 `nu1`/`nu2` metadata 映射；完整 N×N Self/Cross overview 不重算矩阵。
- 2D bundle 增加 reverse matrices 与 `orientations.json`；verifier 校验矩阵、轴和 metadata，同时兼容完整的 v0.1 stored-only bundle。

## Baseline Freeze

- 冻结范围：`src/ftir_baseline/**`、`tests/baseline_regression/**`、`legacy/baseline_streamlit_app.py`。
- 起始哈希：`artifacts/v0.1_baseline_freeze_manifest.json`，共记录 41 个文件。
- 已逐项复核 41 个 manifest 条目的大小与 SHA-256，冻结路径相对起始 commit 没有代码差异，完整 baseline regression 为 135 passed。
- Series QC、Preview 和 A→T 都位于 UI/`ftir_workbench` 层，未向 baseline result 或 config 增加字段。

## 最终验证

| 检查 | 结果 |
|---|---:|
| v0.2 全量 pytest | 445 passed in 37.13s |
| Baseline 冻结回归 | 135 passed in 14.91s |
| Phase 4 Cross/UI/导出定向回归 | 68 passed |
| Ruff | passed |
| Mypy（`ftir_workbench`） | passed，17 source files |
| sdist 与 wheel 构建 | passed，0.2.0 |
| wheel 安装、import 与三组 CLI help smoke | passed |
| Baseline freeze manifest | 41/41 size + SHA-256 matched |
| 冻结路径相对起始 commit 的 Git diff | empty |
| Git 跟踪范围隐私审计 | passed，无原始谱/私有 bundle |

可公开复核的完整结果保存在 `artifacts/validation/final/`。这些结果说明本地候选通过验收，但不表示已经发布或同步到 GitHub。

## 私有数据验证

项目曾在本机实验数据上完成 baseline、Prepared、self/cross 2D-COS、精确往返、矩阵独立复算以及三层 manifest 验证。为保护数据，原始谱、生成的 ZIP/`.ftirw`、数据维度、数据指纹和产物哈希均不进入公开 Git 历史；本次 v0.2 文档和代码也不包含原始数据。

公开 CI 只使用 `examples/` 下的合成/示例数据。将自己的谱放入被忽略的 `data/original/` 后，仍可在本机执行相同的演算和验证脚本。

## 科学注意事项

- 对非等间隔扰动，当前 Noda 矩阵按数值排序后的采集序号构造，不执行非均匀时间加权；该策略会写入 warning 和 provenance。
- 演示 recipe 只用于证明数据链、血缘和数值可复现性；真实样品的基线参数仍需领域专家审阅。
- `BASELINE.dpt` 不会自动并入扰动谱，也不会被擅自解释为算法基线。
- v0.2 的 A→T 是 corrected absorbance 的数学表示，不应解释为恢复了仪器采集的原始透过率。
- Cross 1/Cross 2 是同一 unique pair 的两个矩阵方向，不表示因果方向。

## 明确排除

本版本不实施多个独立 Baseline Blocks、全局基线后的局部 range correction、processing/analysis 双范围模型、AnalysisRangePreparationService、global/local-fine 双分支、baseline sensitivity Run A/B/C、新基线方法、baseline schema 或 bundle 变化、独立区间端点强制归零、PySide6/macOS `.app`，也不进行大规模 Streamlit 架构重写。

## 启动

```bash
source .venv/bin/activate
streamlit run ui/streamlit_app.py
```

完整命令与数据说明见 `README.md` 和 `docs/original_data.md`。
