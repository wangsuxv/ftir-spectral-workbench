# FTIR Spectral Workbench

FTIR Spectral Workbench 将可审计的 FTIR 基线处理与二维相关光谱（2D-COS）放在同一条可复现工作流中。基线处理本身是正式完成状态；用户可以导出校正谱后结束，也可以把同一份内存中的校正吸光度继续送入 2D-COS。

当前本地版本为 `0.2.5`。它在完整保留 v0.2.1 输入兼容、基线结果与 prepared-only 2D-COS 行为的前提下，增加了默认关闭、必须显式创建和激活的 **Post-Baseline Smoothing** 科学分支。项目采用 MIT 许可证，公开内容不包含实验原始数据。

## v0.2.5 Post-Baseline Smoothing

v0.2.5 的数据流固定为：

```text
PipelineResult.analysis_data
→ primary unsmoothed Prepared（默认）
→ optional post-baseline smoothing
→ smoothed Prepared（独立分支）
→ existing prepared-only 2D-COS
```

支持 Savitzky–Golay、Gaussian、Moving Average 和 Median / Despike 四种方法。所有方法对完整二维谱矩阵使用一套参数，只沿 `axis=1` 波数轴处理；不会沿扰动轴平滑，不排序、插值、重采样或覆盖 `PipelineResult.analysis_data`。非均匀波数轴默认拒绝，只有显式 expert override 才允许按 index-space 处理并记录 warning/provenance。

Streamlit 第 8 页提供独立 Preview、实际谱/first/middle/last/mean/median 对比、removed component、QC、正式分支创建及显式 2D 分支切换。Preview 不提交结果；Apply 只创建 smoothed Prepared，不自动激活。2D Setup/Results 仍调用既有 prepared-only 服务，2D 引擎本身不执行 smoothing。

命令行可从 baseline ZIP 或 Prepared CSV + sidecar 创建可验证 bundle：

```bash
ftir-workbench smooth baseline_run.zip \
  --method savgol \
  --window-length 7 \
  --polyorder 2 \
  --output post_baseline_smoothing_run.zip

ftir-workbench twodcos post_baseline_smoothing_run.zip \
  --range 1736:1509:amide \
  --range 1250:1140:fingerprint
```

`post_baseline_smoothing_run.zip` 自包含 parent/child Prepared、removed component、完整配置、QC、图和 SHA-256 manifest；专用 verifier 会从 parent + config 重跑权威 smoothing core 并核对 child、fingerprint 与残差。详细合同、参数和限制见 [`docs/post_baseline_smoothing.md`](docs/post_baseline_smoothing.md)，合成示例见 [`examples/smoothing/`](examples/smoothing/)。

v0.2.5 本地发布验收为 723 passed、5 个既有小型单谱绘图 warning；Ruff、Mypy、sdist/wheel build、全新 wheel 虚拟环境安装（复用已验证依赖层）与 `smooth → verify → twodcos → verify` CLI 链路均通过。v0.2.1 science freeze 为 34/34，冻结根无新增/缺失/修改；精确起始提交生成的旧 baseline/2D/project bundle 可由当前代码验签并精确重载。真实日志与机器审计保存在 [`artifacts/validation/v0.2.5/final/`](artifacts/validation/v0.2.5/final/)。本次本地升级未推送或发布 GitHub tag。

## v0.2.1 文本输入兼容性

v0.2.1 接受 `.csv`、`.tsv`、`.tab`、`.txt`、`.dpt`、`.asc`、`.dat` 和 `.xy` 分隔文本，覆盖单个二列表、单个宽表和多个二列单谱文件。内容解析支持 comma、tab、semicolon、whitespace，dot/decimal-comma、UTF-8/BOM、UTF-16 LE/BE BOM、GB18030、CP1252，以及 blank/comment lines、leading preamble、可选 header、科学计数法和可证明的全空边缘列。

导入 Probe 与正式加载共享同一 parser，并记录原始 bytes 的 SHA-256、编码/分隔符/小数符号证据、header、数值块物理行号、跳过行、边缘空列和 warning。单位仍由用户明确选择；parser 不排序波数轴、不插值、不去重、不裁剪、不删除内部缺失值，也不自动翻转多文件波数轴。

该输入层不读取 OMNIC `.spa/.spg/.srs`、Bruker OPUS `.0/.1/...`、`.spc`、`.sp`、JCAMP-DX、Excel 或 raw ZIP。请先在仪器软件中导出为受支持的纯文本表。完整支持矩阵、检测规则、CLI/API 示例与排错说明见 [`docs/input_formats.md`](docs/input_formats.md)；纯合成示例见 [`examples/import_formats/`](examples/import_formats/)。v0.2.1 发布验收达到 597 passed，并完成 wheel 安装、CLI smoke、24/24 科学冻结哈希及旧 bundle/project 重载；证据保存在 `artifacts/validation/v0.2.1/final/`。

## v0.2.0 四项更新

1. **Coarse/Fine Preview**：可选择实际谱或 first/middle/last/mean/median 代表谱，查看原吸光度、估计通道、粗/细/总基线和校正谱。Coarse 还提供六类既有候选的人工比较画廊。Preview 始终在完整序列上运行；候选排序只是启发式诊断，不是真实物理基线的证明。
2. **Series Consistency & QC**：恢复 Raw、Coarse、Fine、Total、Corrected 五张热图、完整逐谱 QC 表、分量趋势和单谱 drill-down。该页面只展示既有 `PipelineResult` / `QCResult`，不重新实现 QC，也不会删除、重排或裁剪光谱。
3. **A↔T 显示与派生导出**：保留 `%T/T → A` 的科学入口，并增加 `T=10^-A`、`%T=100×10^-A` 的显示副本和独立 CSV。负吸光度可以得到大于 100 的 `%T`，不会被静默裁剪。
4. **Cross 2**：每个 unique cross pair 仍只计算一次；界面和 2D bundle 同时提供 stored orientation（Cross 1）与由转置恒等式得到的 reverse orientation（Cross 2），并显示完整 N×N Self/Cross block overview。

详细的冻结范围与排除项见 [`docs/v0.2_baseline_frozen_scope.md`](docs/v0.2_baseline_frozen_scope.md)，版本变化见 [`CHANGELOG.md`](CHANGELOG.md)。

## 科学边界

- `ftir_baseline` 是单位转换、连续波数区间、estimate-only 平滑、粗/细/序列基线、归一化分支、QC 和基线导出的唯一实现。
- `ftir2dcos.twodcos` 是 Hilbert–Noda、同步/异步、canonical/2dpy-compatible、homo 和 cross-range 计算的唯一实现。
- `ftir_workbench` 只负责数据合同、状态、fingerprint、服务和跨阶段导出，不复制科学公式。
- 默认 2D 输入始终是未归一化的 corrected absorbance，即 `PipelineResult.analysis_data`。
- 2D 服务不调用旧 `ftir2dcos.pipeline`，因此不会重复单位转换、平滑、基线或归一化。
- 不连续 2D 区间分别计算 self/cross 矩阵，不会伪装成连续波数轴。
- 不同 Prepared blocks 的 cross 默认阻断；兼容性检查后仍需显式确认，并记录双方血缘。
- v0.2.1 的输入、baseline 与 2D 科学路径在 v0.2.5 继续冻结；post-baseline smoothing 仅位于独立 workbench 分支层。对应逐文件 SHA-256 审计记录保存在 `artifacts/`。
- Preview、显示单位、热图色阶和 Cross orientation 都是状态隔离的查看操作，不改变 Prepared、2D fingerprint 或已存在矩阵。

## Preview 与正式配置

Coarse 和 Fine 页面中的控件先形成临时 draft。`Preview` 使用临时 validated config 对完整光谱序列运行现有 `ftir_baseline.run_pipeline`，结果保存在独立 UI state 中；它不会覆盖正式 `baseline_config`、正式 baseline result、Prepared 或 2D。

只有显式点击 `Adopt this recipe` 或 `Apply fine settings` 才会提交正式配置，并按 v0.1 依赖关系使 baseline result、Prepared 和 2D 失效。仅切换代表谱、候选查看项、显示单位或 Cross 1/Cross 2 不会触发科学失效。

## 四种入口

1. 原始 FTIR → 基线、QC、导出后结束。
2. 原始 FTIR → 基线 → 当前内存校正谱 → self/cross 2D-COS。
3. 已导出的 corrected CSV、metadata 或 baseline ZIP → 直接继续 2D-COS。
4. Primary Prepared → 显式创建 smoothed Prepared → 选择该分支继续既有 self/cross 2D-COS。

裸 corrected CSV 可以载入，但会留下 provenance 不完整警告；附带 `prepared_spectrum.meta.json` 或从 baseline ZIP 载入时可恢复完整父级哈希链。

## 安装

需要 Python 3.11 或更高版本。建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

启动统一界面：

```bash
streamlit run ui/streamlit_app.py
```

运行测试：

```bash
pytest -q
ruff check src tests ui scripts
```

## 私有数据

公开仓库不包含实验原始数据，也不包含由私有数据生成的指纹清单。`data/original/` 仅作为本机数据入口；其中除说明文件外的所有内容均被 Git 忽略。你也可以直接在 Streamlit 界面上传自己的 `.dpt`、`.csv`、`.tsv`、`.tab`、`.txt`、`.asc`、`.dat` 或 `.xy` 文本文件。

导入目录时会：

- 只读取支持的谱文件；
- 默认排除 `BASELINE.dpt`；
- 从文件名提取扰动值并进行显式数值排序；
- 保留降序波数轴；
- 对非等间隔时间序列给出警告，Noda 矩阵仍按排序后的采集序号计算。

将自己的 DPT 序列放入该目录后，可运行完整演算：

```bash
ftir-workbench demo \
  --input-dir data/original \
  --output outputs/real-data-demo
```

需要覆盖自动文本检测时，可在 `inspect`、`baseline` 或 `demo` 的原始输入命令中使用 `--delimiter`、`--decimal-mark`、`--encoding`、`--header`、`--skip-rows`、`--trim-empty-edge-columns` 或 `--no-trim-empty-edge-columns`。旧命令不要求这些参数。例如：

```bash
ftir-workbench inspect examples/import_formats/semicolon_decimal_comma.csv \
  --unit absorbance \
  --delimiter semicolon \
  --decimal-mark comma \
  --header present
```

对演算产物重新载入、复算并验证三层 manifest：

```bash
python scripts/validate_real_data_demo.py outputs/real-data-demo
```

演算默认先在 1800–900 cm⁻¹ 上完成唯一一次基线处理，再从 prepared data 截取 1736–1509 和 1250–1140 cm⁻¹，计算两个 self 结果和一个矩形 cross 结果。运行前请根据自己的数据范围审阅这些参数。

## 导出

Baseline bundle 包含原始输入、吸光度、基线分解、校正谱、QC、recipe、`corrected_absorbance_for_2dcos.csv`、`prepared_spectrum.meta.json` 和带 SHA-256 的 manifest。

Baseline Result 页面另提供：

- `derived_fraction_transmittance_from_corrected_absorbance.csv`
- `derived_percent_transmittance_from_corrected_absorbance.csv`

二者只从 `PipelineResult.analysis_data` 数学派生，文件内明确声明它们不是仪器原始透过率。它们是独立下载，不加入 baseline ZIP，因此 v0.1 baseline bundle 合同保持不变。

2D bundle 包含父级 prepared CSV/meta、配置、动态谱、同步/异步矩阵、QC、图和 manifest。manifest 会记录：

- `parent_baseline_run_id`
- `parent_baseline_fingerprint`
- `parent_prepared_data_sha256`

每个 cross pair 的 stored 文件继续保留，同时新增 reverse synchronous、reverse asynchronous 和 `orientations.json`。定义为：

```text
Phi_reverse = Phi_stored.T
Psi_reverse = -Psi_stored.T
```

row/column range 由结果中的实际 `row_variable` / `column_variable` 映射，不能假设 stored rows 恒为第一个配置区间。对于 n 个分析区间，计算仍为 `C(n,2)` 个 unique pairs；界面可查看 `n(n-1)` 个有方向的 cross maps。新版 verifier 验证新增方向文件，同时仍接受没有 reverse 合同的完整 v0.1 2D bundle。

Smoothing bundle 与 baseline bundle 分离，baseline ZIP 的字节合同不变。若 active source 是 smoothed Prepared，既有 2D bundle 会嵌入该 child 的完整 Prepared sidecar 和 smoothing lineage；2D 阶段不会再次平滑。

可选归一化文件命名为 `normalized_optional_for_sensitivity_analysis.csv`，不会被误标为默认 2D 输入。

## 目录

```text
src/ftir_baseline/       基线科学核心（保留原 API）
src/ftir2dcos/           2D-COS 科学核心与 legacy API
src/ftir_workbench/      统一合同、服务、状态和导出
ui/streamlit_app.py      统一 Streamlit 工作台
legacy/                  两个来源界面，仅用于回归
tests/                   两套原回归 + 新集成/UI 测试
data/original/           本机私有数据入口（谱文件不纳入 Git）
```

## 结果失效

原始数据、单位、扰动顺序或已提交的基线科学配置变化会失效 Prepared、smoothing 和全部 2D 结果。Smoothing draft/Preview 不修改正式分支；创建 smoothed branch 也不自动激活。显式切换 primary/smoothed Prepared 时只使旧 2D 后代失效。2D 区间、convention 或网格策略变化只失效 2D。代表谱、A/%T/T 显示、Cross 1/Cross 2、色阶、等高线、字体和线宽不会触发科学重算。

本版本不增加多个 Baseline Blocks、局部 range correction、processing/analysis 双范围模型、baseline sensitivity A/B/C、新基线方法或 schema，也不实施自动参数优化、自动 SNR 推荐、resampling、连续多次 smoothing、smoothing + scientific normalization 组合、沿扰动轴平滑、PySide6/macOS `.app` 或大规模 UI 重写。

最终实施状态与逐项合并审计见 `PROJECT_STATUS.md` 和 `MERGE_AUDIT.md`。
私有数据处理约定见 `docs/original_data.md`。
