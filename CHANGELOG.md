# Changelog

本文件记录 FTIR Spectral Workbench 的用户可见变化。当前公开版本为 `0.2.1`。

## [0.2.1] - 2026-09-01

### Added

- `.tsv`、`.tab`、`.asc`、`.dat`、`.xy` 原始文本入口；`.csv`、`.txt`、`.dpt` 旧入口继续可用。
- comma、tab、semicolon、whitespace 内容检测，以及非 comma delimiter 下的 decimal-comma 严格解析。
- UTF-8、UTF-8 BOM、UTF-16 LE/BE BOM、GB18030 和 CP1252 文本解码。
- `TextImportOptions`、`ImportProbe`、`probe_spectrum_file`，并由 Probe 与正式读取共享同一个内部 parser。
- leading metadata/preamble、blank/comment lines、可选 header、科学计数法和全空边缘列处理。
- Streamlit Import Diagnosis/高级文本选项，以及 CLI delimiter、decimal mark、encoding、header、skip rows 和 edge-column 控件。
- 混合文本扩展名多文件序列的逐文件解析诊断和 provenance。
- [`docs/input_formats.md`](docs/input_formats.md) 与六个无私有数据的合成输入示例。

### Strict validation

- 原始 SHA-256 针对解码前 bytes 计算；检测证据、物理行号和 warning 写入 metadata。
- 数值块内的坏行、ragged row、内部缺失值、非有限值、千位分隔歧义和同等可信的候选块会明确失败。
- 多文件波数轴继续要求 point-for-point 一致；不会静默排序、插值、去重、裁剪或翻转。
- OMNIC/OPUS、SPC、PerkinElmer、JCAMP-DX、Excel 和 raw ZIP 不会通过扩展名白名单伪装为受支持文本。

### Unchanged

- Coarse/Fine Preview、Candidate Gallery、Series Consistency & QC、A↔T、Cross 2、full block overview 和 baseline-only 行为不变。
- Prepared handoff、self/cross 2D-COS、peak-order、bundle/project schema、manifest 和 verifier 合同不变。
- baseline 和 2D-COS 的科学配置、公式、结果路径及非 Import 页面保持 v0.2.0 行为。

### Release status

- 已合并到公开默认分支，并发布 `v0.2.1` tag 与 GitHub Release。
- 本地验收：597 passed；Ruff、Mypy、sdist/wheel build 与安装后 CLI smoke 均通过。
- 科学冻结 24/24 文件匹配；exact v0.2.0 基线源码生成的 baseline/2D/project bundle 通过 17 项验签与 Prepared 精确重载。
- 真实命令输出与机器可读审计见 `artifacts/validation/v0.2.1/final/`。

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
