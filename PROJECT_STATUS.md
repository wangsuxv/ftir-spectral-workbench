# Project Status

更新日期：2026-09-01

## 当前状态

FTIR Spectral Workbench 的本地版本为 v0.2.5；开发分支为 `feat/v0.2.5-post-baseline-smoothing`，严格基于 v0.2.1 提交 `92513def080001de4c226fcea0fde484ae8d97fb`。本次任务只升级本地仓库，未推送、打 tag 或创建 GitHub Release。

v0.2.5 的唯一新增科学功能是 Post-Baseline Smoothing。v0.2.1 的输入兼容、baseline 算法与结果、Coarse/Fine Preview、Series Consistency & QC、A↔T、baseline-only、Prepared handoff、Cross 1/Cross 2、完整 block overview、peak-order、bundle/project compatibility 保持不变；许可证仍为 MIT，公开内容不包含实验原始数据或私有派生产物。

## v0.2.5 发布范围

- 数据链：`PipelineResult.analysis_data → primary unsmoothed Prepared → optional smoothing → smoothed Prepared → existing prepared-only 2D-COS`。
- 方法：Savitzky–Golay、Gaussian、Moving Average、Median / Despike；默认 `enabled=false`。
- 共同合同：finite float64、输入不可变、全序列统一参数、只沿 `axis=1`、轴/标签/顺序不变、不插值、不自动重采样。
- 轴策略：先检查 `abs(diff(wavenumber))` 近似均匀；默认失败，只有显式 expert override 可按 index-space 处理并记录 warning/provenance。
- Prepared lineage：child 保留 `baseline_run_id` 和 `baseline_fingerprint`，重新计算 `prepared_data_sha256`，recipe 记录 parent hash、smoothing fingerprint、方法、有效参数、QC 和 warning。
- UI：新增第 8 页 Post-Baseline Smoothing；2D Setup/Results 顺延为第 9/10 页。Preview 不修改正式状态，Apply 只创建分支且不自动激活。
- 分支切换：复用 `_activate_prepared_for_twodcos()`，只清除旧 2D 后代并保留 baseline 与 committed smoothing。
- 导出：新增独立 smoothing bundle 和 verifier；baseline bundle 不变，smoothed 2D bundle 继续用既有结构并嵌入实际 source Prepared。
- CLI：新增 `ftir-workbench smooth`；旧 `baseline`、`twodcos` 参数与语义不变，2D 阶段不执行 smoothing。

## v0.2.5 Science Freeze

- 冻结清单为 `artifacts/v0.2.1_science_freeze_manifest.json`，覆盖 34 个 v0.2.1 输入、baseline、2D 与 workbench service 文件。
- 禁止修改的 baseline/2D engine、peak-order、Cross/display helper 和 baseline/2D service 路径保持与起始提交逐文件 size/SHA-256 一致。
- Post-baseline smoothing 只在新增 workbench core/adapter/service/export 层实现；不复用或修改 baseline estimate-only smoothing，也不向 2D engine 注入 smoothing。
- 旧 `.csv/.txt/.dpt` 以及 v0.2.1 的全部回归继续由同一测试集覆盖；旧 bundle/project 由起始版本实际生成后交给当前代码验签和 Prepared 精确重载。

## v0.2.5 本地发布验收

| 检查 | 实际结果 |
|---|---:|
| 全量 pytest | 723 passed，5 个既有小型单谱绘图 warning |
| Ruff `src tests ui scripts` | passed |
| Mypy `src/ftir_workbench` | passed，20 source files |
| sdist 与 wheel 构建 | passed，0.2.5 |
| 全新 wheel venv install/import/CLI | passed；依赖复用已验证工作区层，三包导入、四个 help、`smooth → verify → twodcos → verify` |
| v0.2.1 science freeze | 34/34 start/worktree size + SHA-256；无新增、缺失或 Git diff |
| exact v0.2.1 bundle/project reload | baseline/2D/project 验签、嵌套 byte identity 与 Prepared exact reload 通过 |
| smoothing bundle roundtrip | verifier、parent/child、residual、fingerprint 与 child exact reload 通过 |
| unsmoothed/smoothed 2D | self 2 + cross 1、Cross reverse identities、不同 fingerprint、smoothed source exact reload 通过 |
| Git 与 distribution 隐私审计 | passed；sdist 仅含 `data/original/README.md`，wheel 无原始数据成员，ignored raw spectra 未读取 |

最终实际结果与机器可读审计保存在 `artifacts/validation/v0.2.5/final/`。审计使用 4×81 的确定性合成 Prepared；真实实验谱、原始数据、私有 bundle、本机路径和临时路径均未写入发布证据。

## v0.2.1 发布范围

- 文本扩展名：`.csv`、`.tsv`、`.tab`、`.txt`、`.dpt`、`.asc`、`.dat`、`.xy`。
- 布局：单个二列光谱、单个宽表光谱序列、多个二列单谱文件。
- 内容：comma/tab/semicolon/whitespace，dot 与受约束的 decimal comma，UTF-8/BOM、UTF-16 LE/BE BOM、GB18030、CP1252，blank/comment lines、leading preamble、可选 header、科学计数法和全空边缘列。
- 公共接口：旧 reader API 保留；新增 `TextImportOptions`、`ImportProbe` 和 `probe_spectrum_file`，Probe 与正式加载共享 parser。
- Import 页面：新增 Advanced text import options 和 Import Diagnosis；单位继续由用户明确选择。
- CLI：新增可选 delimiter、decimal mark、encoding、header、skip rows 与 edge-column flags；旧命令无需新参数。
- Provenance：原始 bytes SHA-256、解析证据、数值块物理行号、跳过行、warning 和多文件逐文件诊断均可审计。
- 文档与数据：[`docs/input_formats.md`](docs/input_formats.md) 给出完整合同；`examples/import_formats/` 和 `tests/fixtures/import_compat/` 只包含合成数据。

明确不支持 OMNIC `.spa/.spg/.srs`、Bruker OPUS `.0/.1/...`、`.spc`、`.sp`、JCAMP-DX、Excel 和 raw ZIP。重命名扩展名不能把这些二进制或结构化格式变为受支持文本；必须先用仪器软件导出。

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

## v0.2.1 Science Freeze

- 冻结 baseline config/pipeline/算法、单位转换、归一化、QC、gallery、导出，以及 2D-COS 数学、peak-order 和 workbench baseline/2D service。
- v0.2.0 科学冻结文件的 size/SHA-256 基准保存在 `artifacts/v0.2_science_freeze_manifest.json`；v0.2.1 最终审计必须逐项匹配。
- 旧 `.csv/.txt/.dpt` 的 wavenumber、spectra、perturbation、labels、baseline、Prepared hash 及 self/cross 2D matrices 由冻结 reference 做逐元素/哈希回归。
- parser 不排序波数轴、不插值、不去重、不裁剪、不删除内部缺失值、不根据 header 推断单位，也不自动翻转多文件轴；多文件轴继续 point-for-point 一致。

## v0.2.0 历史 Baseline Freeze

- 冻结范围：`src/ftir_baseline/**`、`tests/baseline_regression/**`、`legacy/baseline_streamlit_app.py`。
- 起始哈希：`artifacts/v0.1_baseline_freeze_manifest.json`，共记录 41 个文件。
- 已逐项复核 41 个 manifest 条目的大小与 SHA-256，冻结路径相对起始 commit 没有代码差异，完整 baseline regression 为 135 passed。
- Series QC、Preview 和 A→T 都位于 UI/`ftir_workbench` 层，未向 baseline result 或 config 增加字段。

## v0.2.0 已发布版本的历史验证

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

可公开复核的完整结果保存在 `artifacts/validation/final/`。这些结果只对应已合并到默认分支的 v0.2.0 源码；GitHub Actions 也已通过，不能当作 v0.2.1 的最终结果。

## v0.2.1 发布验收

| 检查 | 实际结果 |
|---|---:|
| 全量 pytest | 597 passed，5 个小型单谱绘图 warning |
| 输入兼容测试 | 127 passed |
| 旧 CSV/TXT/DPT 科学 reference | 3 passed，逐元素/哈希一致 |
| Ruff | passed |
| Mypy `src/ftir_workbench` | passed，17 source files |
| sdist 与 wheel 构建 | passed，0.2.1 |
| 全新临时环境 wheel install/import/CLI | passed，三包可导入，三个入口可启动，TSV inspect/baseline/verify 成功 |
| v0.2.0 science freeze | 24/24 size 与 SHA-256 匹配 |
| exact v0.2.0 bundle/project reload | 17/17 验签、嵌套检查与 Prepared 精确重载通过 |
| Git 跟踪范围隐私审计 | passed，仅跟踪 `data/original/README.md` |

真实命令输出统一保存在 `artifacts/validation/v0.2.1/final/`。科学子组件 `ftir_baseline==0.1.0` 和 `ftir2dcos==0.4.0` 保留其冻结的历史版本；发行包与统一工作台版本为 `0.2.1`。

该版本已推送并发布；公开 release 仅包含源码与合成测试资料，不包含实验原始数据或私有派生产物。

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

v0.2.5 不实施多个独立 Baseline Blocks、全局基线后的局部 range correction、processing/analysis 双范围模型、AnalysisRangePreparationService、global/local-fine 双分支、baseline sensitivity Run A/B/C、新基线方法、baseline schema 或 bundle 变化、独立区间端点强制归零、PySide6/macOS `.app`，也不进行大规模 Streamlit 架构重写。

本补丁也不加入厂商二进制 reader、JCAMP-DX parser、Excel sheet/column mapping 或 raw ZIP ingestion。极短文件、多个同等可信数值块、encoding 多解和 delimiter/decimal 冲突需要用户显式选择；千位分隔符需要在导入前清理。

Post-baseline smoothing 不提供自动最佳算法/参数、SNR 驱动推荐、Butterworth/Whittaker/wavelet/PCA、resampling、沿扰动轴平滑、连续多次 smoothing 或 smoothing + scientific normalization 组合。旧 `.ftirw` 不强制嵌入 smoothing-only branch；该分支通过独立 smoothing bundle 保存，而 smoothed 2D bundle 自包含其实际 source Prepared。

## 启动

```bash
source .venv/bin/activate
streamlit run ui/streamlit_app.py
```

完整命令与数据说明见 `README.md` 和 `docs/original_data.md`。
