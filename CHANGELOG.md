# Changelog

本文件记录 FTIR Spectral Workbench 的用户可见变化。当前 `0.2.0` 条目描述已通过本地与 GitHub Actions 验收并发布到默认分支的公开版本。

## [0.2.0] - 2026-08-29

### Added

- Coarse current-recipe Preview、first/middle/last/mean/median/实际谱选择，以及复用既有 API 的六类 Candidate Gallery。
- Fine decomposition、coarse residual、fitted anchor window/representative diagnostics。
- Series Consistency & QC 的五张热图、完整逐谱 QC 表、搜索/CSV、三组趋势和单谱 drill-down。
- `ftir_workbench.display_units` 中 view-only `A → T/%T` 转换与两份独立派生 CSV。
- Cross stored/reverse view helper、Cross 1/Cross 2 UI、30×30 numeric preview 和完整 N×N Self/Cross overview。
- 2D bundle 的 reverse synchronous/asynchronous CSV、`orientations.json` 及相应完整性验证。
- 独立 2Dpy-compatible hetero oracle 与新的显示、Cross、bundle、UI 回归测试。

### Changed

- 页面名称由 `Series QC` 调整为 `Series Consistency & QC`。
- 项目版本 metadata 从 `0.1.0` 更新为 `0.2.0`。
- 2D bundle verifier 识别 v0.2 additive reverse contract，同时继续接受完整的 v0.1 stored-only bundle。

### Preserved

- `src/ftir_baseline/**`、`tests/baseline_regression/**` 与 `legacy/baseline_streamlit_app.py` 冻结。
- baseline config/schema、数值模型、处理区间、workflow state、Prepared contract、baseline ZIP 和 baseline-only 成功路径保持 v0.1 行为。
- 每个 unique cross pair 仍只计算一次；Cross 2 不增加结果对象、fingerprint 或 peak-order evidence。

### Privacy and release status

- 未加入实验原始数据、私有派生产物、私有 fingerprint 或产物 hash。
- 已通过 PR #1 发布到公开默认分支；未加入原始数据、私有 bundle 或环境文件。

## [0.1.0] - 2026-08-23

- 首个冻结版本：统一 baseline-first Streamlit/CLI 工作流、baseline-only 导出、Prepared checkpoint、可选 self/cross 2D-COS、fingerprints 和可验证 bundles。
