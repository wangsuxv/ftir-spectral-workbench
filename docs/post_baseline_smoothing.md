# 基线后平滑与独立 Prepared 分支（v0.2.5）

v0.2.5 在现有基线结果之后增加可选的光谱平滑。它是一项会改变科学数据、Prepared SHA-256 和后续 2D-COS 数值的预处理，不是显示效果，也不会偷偷进入既有 baseline pipeline 或 2D-COS engine。

这里的“基线后平滑”与 baseline 算法中的 estimate-only smoothing 不同：后者只辅助估计基线，不改写最终 corrected absorbance；本功能直接变换 Primary Prepared 中已经完成基线校正的 spectra，因此必须作为新的科学分支记录和激活。

```text
PipelineResult.analysis_data
  ↓
Primary Prepared（未平滑、默认科学分支）
  ├─→ 直接用于 2D-COS
  └─→ Post-Baseline Smoothing
        ↓
      Smoothed Prepared（显式创建、独立 fingerprint）
        ↓
      用户显式激活后用于 2D-COS
```

Primary Prepared 始终保留。创建 smoothed branch 不会覆盖或自动激活它，用户也可以随时切回未平滑分支。

## 默认行为与数据方向

- 默认 `enabled=False`；升级后不操作本页，v0.2.1 的科学结果不变。
- disabled result 是严格 identity：`smoothed_spectra` 与 parent spectra 逐元素相同，`removed_component` 逐元素为零。
- Prepared 的矩阵形状为 `(n_spectra, n_wavenumbers)`。
- 所有算法只沿波数轴工作，即 `axis=1`；不会沿扰动/时间方向平滑。
- 每条谱使用完全相同的算法和参数。
- 波数轴及扰动顺序不会被排序、翻转、插值、裁剪或去重。
- 未平滑 corrected absorbance 与 smoothed corrected absorbance 都被保留并写入 smoothing bundle。

## 四种算法

| 方法 | 主要参数 | 默认值 | 边界模式 | 使用提示 |
|---|---|---|---|---|
| Savitzky–Golay (`savgol`) | 奇数 `window_length`、`polyorder` | 7、2 | `interp`；也可 `mirror`/`nearest` | 窗口须为不小于 3 的奇数且不超过点数，`0 <= polyorder < window_length`，固定 `deriv=0`；近似跨度为 `(window_length - 1) × median spacing` cm⁻¹ |
| Gaussian (`gaussian`) | `sigma_points`、`truncate` | 1.0、4.0 | `reflect`；也可 `mirror`/`nearest` | 两参数都须大于零；`sigma` 以点数表示，`sigma_cm1 = sigma_points × median spacing`，`FWHM ≈ 2.35482 × sigma_cm1` |
| Moving average (`moving_average`) | 奇数 `window_length` | 3 | `reflect`；也可 `mirror`/`nearest` | 窗口须为不小于 3 的奇数且不超过点数；简单线性低通，可能压低窄峰或扩大边缘影响 |
| Median (`median`) | 奇数 `window_length` | 3 | `reflect`；也可 `mirror`/`nearest` | 窗口须为不小于 3 的奇数且不超过点数；非线性、面向脉冲/尖刺的 expert 方法，启用时会明确 warning，且可能削平真实窄峰 |

方法选择不会改变未启用参数的 scientific fingerprint。例如选择 Gaussian 时，SG 的窗口值不会进入 fingerprint。完整可编辑配置仍会写入 `smoothing_config.json`，而 manifest 的 `parameters` 只记录当前方法的有效参数。

QC 是中性的“变换幅度与峰形保真诊断”，包括 removed RMS、导数相关、面积变化、roughness 和 edge effect。默认 warning 阈值为 relative RMS removed `> 0.10`、first-derivative correlation `< 0.95`、relative absolute-area change `> 0.02`、edge-effect ratio `> 2.0`。阈值只产生 warning，不会自动拒绝、修改数据或替用户选择“最佳”算法。没有已知噪声区或 ground truth 时，Workbench 不声称平滑自动提高了 SNR。

## 波数轴均匀性

平滑前会检查：

```text
spacing = abs(diff(wavenumber))
allclose(spacing, median(spacing), rtol=1e-3, atol=1e-8)
```

是否在 `uniformity_rtol=1e-3` 下近似均匀。升序和降序轴都可使用，因为检查基于绝对间距。

默认策略为：

```text
nonuniform_axis_policy = error
```

非均匀轴会明确失败。Workbench 不自动重采样，因为插值本身会改变峰形与 2D-COS 结果。只有用户明确选择：

```text
allow_index_space_with_warning
```

时，才按数组索引空间应用滤波，并在 result、Prepared recipe、bundle 与 CLI 输出中记录 warning。该 override 不会把波数轴变成均匀网格。

## Preview、Apply 与激活

Streamlit 的第 8 页为 `Post-Baseline Smoothing`。

页面 parent source 优先使用 `state.prepared` 中的 Primary Prepared；只有 Primary 不存在时才考虑 active Prepared fallback。fallback 如果是已经平滑的 branch 或 scientific-normalization sensitivity branch，会被 guard 拒绝，避免 chained smoothing 或两种科学变换被静默组合。

1. `Generate Preview` 对当前 Primary Prepared 和 draft config 调用权威 smoothing core，但不创建或替换 committed branch。
2. actual spectrum、first、middle、last、mean、median 只改变预览选择。mean/median 是显示聚合，不是新的科学光谱。
3. plot range、代表谱和曲线显示变化不会使科学结果失效。
4. `Create Smoothed Scientific Branch` 只有在 preview config 等于当前 draft、且 preview parent hash 等于当前 parent Prepared hash 时可用。
5. Apply 创建 smoothed Prepared 和可验证 bundle，但不会自动激活该分支。
6. `Use Unsmoothed Branch for 2D-COS` 与 `Use Smoothed Branch for 2D-COS` 是平级选择。切换只清除旧的 2D config/result/bundle/peak-order，保留 baseline、Primary Prepared、preview 和 committed smoothing。

改变 baseline science 会清除 preview、committed smoothing 和其 bundle。只改变 smoothing draft 会使旧 preview 过期，但不会删除已提交的 smoothed branch；页面会显示 draft 与 committed branch 不同。

Child 会复制 parent 的波数轴、labels、source、baseline run/fingerprint、方向与顺序、normalization 状态、baseline recipe 和 baseline QC，并用 smoothed spectra 重算 Prepared hash。常量谱或某些 SG 多项式输入可能产生数值 no-op，因而 child hash 可以恰好等于 parent hash；这不是错误，branch recipe 与 smoothing fingerprint 仍明确记录该科学分支。2D-COS 输入 fingerprint 绑定所激活 Prepared，数据实际变化时也会自然变化。

## Python API

```python
from ftir_workbench import (
    PostBaselineSmoothingConfig,
    PostBaselineSmoothingService,
    verify_smoothing_bundle,
)
from ftir_workbench.export import load_prepared

parent = load_prepared(
    "examples/smoothing/synthetic_corrected_prepared.csv"
)
config = PostBaselineSmoothingConfig(
    enabled=True,
    method="savgol",
    savgol_window_length=7,
    savgol_polyorder=2,
    savgol_mode="interp",
)

service = PostBaselineSmoothingService()
preview = service.preview(parent, config)
result, child = service.apply(parent, config)
bundle = service.build_bundle(result, child)
assert verify_smoothing_bundle(bundle)

# 对 smoothing ZIP 调用公共 loader 会先执行严格 verifier，再返回 child。
reloaded_child = load_prepared(bundle)
```

`preview` 和 `apply` 使用同一个权威 core。两者的数值差异只来自传入的 parent/config；UI 是否提交结果不改变算法。

## CLI

`smooth` 从 Prepared CSV + sidecar 或 baseline ZIP 创建显式 enabled branch。`--method` 是必填项：

```bash
ftir-workbench smooth \
  examples/smoothing/synthetic_corrected_prepared.csv \
  --method savgol \
  --window-length 7 \
  --polyorder 2 \
  --mode interp \
  --output outputs/smoothing-example
```

其他方法示例：

```bash
ftir-workbench smooth prepared_spectrum.meta.json \
  --method gaussian --sigma-points 1.25 --truncate 3 --mode reflect

ftir-workbench smooth baseline_run.zip \
  --method moving_average --window-length 5 --mode mirror

ftir-workbench smooth baseline_run.zip \
  --method median --window-length 3 --mode nearest
```

非均匀轴的显式 expert override：

```bash
ftir-workbench smooth prepared_spectrum.meta.json \
  --method gaussian \
  --nonuniform-axis-policy allow_index_space_with_warning
```

验证与继续 2D-COS：

```bash
ftir-workbench verify \
  outputs/smoothing-example/post_baseline_smoothing_run.zip

ftir-workbench twodcos \
  outputs/smoothing-example/post_baseline_smoothing_run.zip \
  --range 1800:1550:upper \
  --range 1500:1200:lower \
  --output outputs/smoothing-2d
```

CLI 不提供 disabled smoothing command：调用 `smooth` 就表示用户明确请求创建科学分支。重复把 smoothing bundle 作为 `smooth` 输入会由 chained-smoothing guard 拒绝；同一个 bundle 可以直接作为 `twodcos` 输入。

CLI 会拒绝不属于当前算法的参数：`--polyorder` 仅用于 Savitzky–Golay，`--sigma-points/--truncate` 仅用于 Gaussian，`--window-length` 仅用于 Savitzky–Golay、Moving average 或 Median。`savgol --mode reflect` 无效；其余三种算法的 `--mode interp` 无效。错误参数不会被静默忽略。

## Smoothing bundle

`post_baseline_smoothing_run.zip` 包含：

```text
source_corrected_absorbance.csv
source_prepared_spectrum.meta.json
smoothed_corrected_absorbance.csv
prepared_spectrum.meta.json
smoothing_removed_component.csv
smoothing_config.json
smoothing_metrics.json
smoothing_metrics.csv
figures/
  selected_spectrum_overlay.png
  selected_spectrum_residual.png
manifest.json
```

Manifest 的 `artifact_type` 为 `post_baseline_smoothing_run`，并绑定 parent baseline lineage、parent/child Prepared SHA-256、smoothing fingerprint、方法和有效参数。专用 verifier 除检查 ZIP 成员、大小与 SHA-256 外，还会从 embedded source Prepared 和 config 重跑权威 core，核对 smoothed child、removed identity、配置、QC、fingerprint 和完整 Prepared reload。因此，只重新计算 manifest 文件哈希不能掩盖科学语义篡改。

Bundle 只包含 baseline-corrected source 和派生结果，不包含原仪器 raw spectra、baseline bundle、2D bundle 或 project archive。`load_prepared(smoothing_zip)` 返回 `prepared_spectrum.meta.json` 指向的 smoothed child。

“不含 raw”不等于“可公开”：真实 smoothing bundle 仍包含完整的 baseline-corrected 光谱及其派生结果，可能属于敏感研究数据。请把真实 bundle 保存在被 Git 忽略的输出目录或私有存储中，公开仓库只提交明确生成的合成示例。

## 明确限制

- 不自动重采样非均匀波数轴。
- 不允许对 smoothed Prepared 再次平滑（no chained smoothing）。
- 不把 scientific normalization 与 post-baseline smoothing 组合在同一 Prepared 分支。
- 不沿扰动/时间轴平滑。
- 不在 2D-COS engine 内执行 smoothing；2D 服务只消费用户已经激活的 Prepared。
- 不自动选择“最佳”方法或参数，不做自动 SNR 优化、峰拟合或 post-2D filtering。
- 不提供 Butterworth、FFT、wavelet、PCA 或其他未列出的滤波方法。
- 不改变既有 baseline bundle schema，也不强制把 smoothing bundle 嵌入 `.ftirw` project。
- Median 是非线性方法；所有 QC warning 都是诊断，不是自动科学结论。
- 点数参数的物理宽度依赖实际波数间距；跨仪器/网格比较时必须检查 axis diagnostics。

## 纯合成示例

[`examples/smoothing/`](../examples/smoothing/) 中的 CSV 和 sidecar 是小型、确定性的合成 corrected-absorbance Prepared，只用于演示 API、CLI、bundle 和测试合同。曲线数值是手工固定的演示值，并非实验结果的抽样、聚合或派生；它不含实验数据、原始仪器数据、样本或个人标识、本机路径、仪器序列号、真实采集 metadata 或私有 fingerprint。
