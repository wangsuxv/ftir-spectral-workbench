# FTIR Spectral Workbench

FTIR Spectral Workbench 将可审计的 FTIR 基线处理与二维相关光谱（2D-COS）放在同一条可复现工作流中。基线处理本身是正式完成状态；用户可以导出校正谱后结束，也可以把同一份内存中的校正吸光度继续送入 2D-COS。

当前工作树是 `0.2.0` 候选版本。它以冻结的 v0.1 基线科学路径为基础，只增加 Coarse/Fine Preview、Series Consistency & QC、A↔T 显示与派生导出，以及 Cross 2 / 完整 Self-Cross overview。候选代码已同步到公开 GitHub 功能分支；在 Draft PR 合并前，默认 `main` 仍保持 v0.1。

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
- `src/ftir_baseline/**`、`tests/baseline_regression/**` 和 `legacy/baseline_streamlit_app.py` 在 v0.2 中冻结；文件哈希记录于 `artifacts/v0.1_baseline_freeze_manifest.json`。
- Preview、显示单位、热图色阶和 Cross orientation 都是状态隔离的查看操作，不改变 Prepared、2D fingerprint 或已存在矩阵。

## Preview 与正式配置

Coarse 和 Fine 页面中的控件先形成临时 draft。`Preview` 使用临时 validated config 对完整光谱序列运行现有 `ftir_baseline.run_pipeline`，结果保存在独立 UI state 中；它不会覆盖正式 `baseline_config`、正式 baseline result、Prepared 或 2D。

只有显式点击 `Adopt this recipe` 或 `Apply fine settings` 才会提交正式配置，并按 v0.1 依赖关系使 baseline result、Prepared 和 2D 失效。仅切换代表谱、候选查看项、显示单位或 Cross 1/Cross 2 不会触发科学失效。

## 三种入口

1. 原始 FTIR → 基线、QC、导出后结束。
2. 原始 FTIR → 基线 → 当前内存校正谱 → self/cross 2D-COS。
3. 已导出的 corrected CSV、metadata 或 baseline ZIP → 直接继续 2D-COS。

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

公开仓库不包含实验原始数据，也不包含由私有数据生成的指纹清单。`data/original/` 仅作为本机数据入口；其中除说明文件外的所有内容均被 Git 忽略。你也可以直接在 Streamlit 界面上传自己的 `.dpt`、`.csv` 或 `.txt` 文件。

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

原始数据、单位、扰动顺序或已提交的基线科学配置变化会失效 Prepared 和全部 2D 结果。2D 区间、convention 或网格策略变化只失效 2D。Preview draft、代表谱、候选查看项、A/%T/T 显示、Cross 1/Cross 2、色阶、等高线、字体、线宽和显示归一化不会触发科学重算。

本版本不增加多个 Baseline Blocks、局部 range correction、processing/analysis 双范围模型、baseline sensitivity A/B/C、新基线方法或 schema，也不实施 PySide6/macOS `.app` 和大规模 UI 重写。

最终实施状态与逐项合并审计见 `PROJECT_STATUS.md` 和 `MERGE_AUDIT.md`。
私有数据处理约定见 `docs/original_data.md`。
