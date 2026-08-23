# FTIR Spectral Workbench

FTIR Spectral Workbench 将可审计的 FTIR 基线处理与二维相关光谱（2D-COS）放在同一条可复现工作流中。基线处理本身是正式完成状态；用户可以导出校正谱后结束，也可以把同一份内存中的校正吸光度继续送入 2D-COS。

## 科学边界

- `ftir_baseline` 是单位转换、连续波数区间、estimate-only 平滑、粗/细/序列基线、归一化分支、QC 和基线导出的唯一实现。
- `ftir2dcos.twodcos` 是 Hilbert–Noda、同步/异步、canonical/2dpy-compatible、homo 和 cross-range 计算的唯一实现。
- `ftir_workbench` 只负责数据合同、状态、fingerprint、服务和跨阶段导出，不复制科学公式。
- 默认 2D 输入始终是未归一化的 corrected absorbance，即 `PipelineResult.analysis_data`。
- 2D 服务不调用旧 `ftir2dcos.pipeline`，因此不会重复单位转换、平滑、基线或归一化。
- 不连续 2D 区间分别计算 self/cross 矩阵，不会伪装成连续波数轴。
- 不同 Prepared blocks 的 cross 默认阻断；兼容性检查后仍需显式确认，并记录双方血缘。

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

2D bundle 包含父级 prepared CSV/meta、配置、动态谱、同步/异步矩阵、QC、图和 manifest。manifest 会记录：

- `parent_baseline_run_id`
- `parent_baseline_fingerprint`
- `parent_prepared_data_sha256`

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

原始数据、单位、扰动顺序或基线科学配置变化会失效 prepared 和全部 2D 结果。2D 区间、convention 或网格策略变化只失效 2D。色阶、等高线、字体、线宽和显示归一化不会触发科学重算。

桌面版 PySide6/macOS `.app` 属于后续阶段；首版先以统一 Streamlit 验证科学工作流。

最终实施状态与逐项合并审计见 `PROJECT_STATUS.md` 和 `MERGE_AUDIT.md`。
私有数据处理约定见 `docs/original_data.md`。
