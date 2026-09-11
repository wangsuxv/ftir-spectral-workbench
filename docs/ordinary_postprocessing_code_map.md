# 普通光谱后处理：实施前本地代码映射

记录日期：2026-09-11。本文依据实际本地源码、`FTIR_Workbench_v0.3.1_Ordinary_Smoothing_Normalization_Spec.md` 全文及 `Acceptance_Tests.md` 的 66 项待实施场景编写。它是 v0.3.1 阶段 0 的代码映射；“现状”与“建议增量”分开，不表示新功能已实施或通过验收。

## 1. 实际基准与环境

| 项目 | 只读核对结果 |
|---|---|
| 实施基准 HEAD | `9fc13c10ae34bd53cb220d0807edb18c9e2e41dc` |
| 当前开发分支 | `feat/v0.3.1-ordinary-postprocessing` |
| 检查时工作树 | `git status --short` 无输出，编写本映射前干净 |
| 发行版本 | `pyproject.toml` 为 `0.3.0`，普通模式前置功能确实存在 |
| Python | 3.12.14，沿用现有 `.venv` |
| 实际数值依赖 | NumPy 2.5.2、SciPy 1.18.1、pandas 3.0.5、pybaselines 1.2.1、Pydantic 2.13.4 |
| 实际 UI/测试依赖 | Streamlit 1.62.0、Plotly 6.9.0、pytest 9.1.1 |
| lock 与工具 | 以上版本与 `requirements.lock` 对应条目一致；现有 pytest、Ruff、Mypy、build 配置保留 |

规格引用的远程 `c0751bc2fa0e3a3716c70bc9da33766fcad2b746` 是旧 v0.2.5 审阅证据，**不作为本轮实施基准**。本映射未 reset、切回旧远程版本、修改运行代码或读取实验数据。README/PROJECT_STATUS 的 v0.3.0 最终 906 passed 是上一轮历史结果，不能作为 v0.3.1 本轮测试结果；本轮测试由阶段日志另行记录。

## 2. 普通模式入口与选择状态

现状见 [统一入口](../ui/streamlit_app.py) 的 `main()` 与 [普通界面](../ui/batch_workflow.py) 的 `render_batch_workflow()`：

- `st.sidebar.selectbox("数据模式", ..., key="workflow_mode")` 在上传前分流。普通模式调用 `render_batch_workflow()` 后立即返回；原位十页流程及全局原位状态仍走旧路径。
- 普通业务对象为 `st.session_state.batch_workspace: BatchWorkspace`，不会复用原位 `baseline_config`、`baseline_result` 或 Prepared。
- 现有五页为导入检查、单位/处理范围、粗调、细调、批量检查/导出；尚无普通后处理页。
- 当前编辑条目保存在 `BatchWorkspace.selected_spectrum_id`。`_sidebar_selection()` 使用同一 `display_order` 生成搜索/状态过滤后的显示列表；勾选集单独保存在 `batch_target_ids` 及 `batch_targets_<workspace_id>` widget，不等于当前条目。
- `_values()` 将尚在编辑的值保存在对应 `SpectrumProcessingState.display_preferences["editor_<stage>"]`；`_changed()` / `_sync()` 将有效值写入独立 typed draft。无效编辑仍保留编辑值与 `error_<stage>`，不会把旧正式结果覆盖掉。`_key()` 包含 workspace、spectrum、stage、field。
- `_reset_editor()` 在恢复已确认参数、复制成功、工作区恢复时重建必要控件。切谱、过滤、重排和显示缩放不调用科学服务。确认并下一条通过 `batch_selection_revision` 更新同一选择器，而不是新建第二个谱身份。

新页应复用这些工作区/选择器和当前稳定交互方式。普通基线页目前只有逐谱确认和“确认并下一条”，没有服务级批量确认；v0.3.1 的“确认所选有效后处理预览”需要新增明确操作，不能误称现有能力。

## 3. 身份、原始输入与可用母谱 B

现有模型见 [batch/models.py](../src/ftir_workbench/batch/models.py)，导入见 [batch/importing.py](../src/ftir_workbench/batch/importing.py)。

| 实际对象 | 已有字段和合同 |
|---|---|
| `ImportedSource` | 持久 `source_id`、原文件名、原 bytes/SHA-256、默认输入单位、解析选项和 probe |
| `SpectrumRecord` | 持久 `spectrum_id`、`source_id`、原始列序号（x=0，强度从1起）、原列名、展示名、原 x/y、确认单位、方向、科学输入 hash、重复候选/排除标志 |
| `BatchWorkspace` | schema `1.0`、`workspace_id`、mode、sources/records/states 字典、显示顺序/当前条目、导入错误、导出历史/摘要 |
| `StageSnapshot` | `stage`、谱 ID、输入 hash、`PipelineConfig`、`fingerprint`、完整不可变 `PipelineResult`、coarse 父 fingerprint、当前及父实现 fingerprint |

ID 为显式导入时产生并在工作区恢复时保留的 UUID hex；文件名、列名、当前行号都不是身份。原 x/y 为不可变 bytes-backed float64 数组。`scientific_input_hash(x,y,unit)` 编码数组 shape、little-endian float64 bytes 和单位；`source_selection_fingerprint(workspace,spectrum_id)` 另编码原文件 SHA、原列序号和 import options。完全同值的不同列可有相同科学输入 hash，但身份/来源 fingerprint 不同。

`import_sources(workspace, uploads: Sequence[tuple[str,bytes]], *, input_unit="absorbance", options: TextImportOptions|None=None) -> list[str]` 逐文件公开 probe/read、逐列拆分，保留异轴和失败隔离。新后处理无需也不应再次导入或重新构建普通工作区。当前普通模型没有连续分段/多区间 metadata 合同；不能凭空声明支持跨多个不连续段的滤波，也不能把明显分段数据拼接后跨间隙处理。

**B 的唯一正确现有入口：**

```python
from ftir_workbench.batch.state import get_ready_snapshot

baseline_snapshot = get_ready_snapshot(workspace, spectrum_id, "fine")
x = baseline_snapshot.result.absorbance_selected.wavenumber
B = baseline_snapshot.result.analysis_data[0]
baseline_fingerprint = baseline_snapshot.fingerprint
```

`get_ready_snapshot(...,"fine")` 同时要求有效正式粗调和有效最终状态，并校验来源、配方、数组及 coarse 父级。

- `fine_decision="applied"`：已确认有效 fine。
- `fine_decision="explicitly_skipped"`：`skip_fine()` 已显式将有效 coarse 作为最终结果，仍产生 `stage="fine"` 的快照，其结果复用 coarse `PipelineResult`，fine 分量严格为零。
- `fine_decision="not_decided"`：即使有正式 coarse，也会抛 `FINE_REQUIRED`，不能直接作为 B。
- `coarse_stale` / `fine_stale` 或父级不匹配均拒绝当前结果。不能只检查 `fine_snapshot is not None`，也不能取当前图的 preview。

`PipelineResult.analysis_data` 在此处确实是未归一化校正吸光度；**这与后面 `NormalizationResult.analysis_data` 的“保持调用输入”语义不能混淆**。

## 4. 已有草稿、预览、确认及失效接口

现有 [batch/service.py](../src/ftir_workbench/batch/service.py) 仅编排基线，签名为：

```python
preview_coarse(workspace, spectrum_id, *, runner=None) -> StageSnapshot
preview_fine(workspace, spectrum_id, *, runner=None) -> StageSnapshot
preview_selected(workspace, spectrum_ids, *, stage, runner=None) -> dict[str, str | None]
run_singleton(workspace, spectrum_id, config, *, runner=None) -> PipelineResult
to_spectrum_set(workspace, spectrum_id) -> SpectrumSet
IndependentBatchBaselineService(runner=None)
```

服务对象是上述 preview API 的 facade，不拥有 `_last_result`。每谱仅保留一个 coarse/fine preview 及各自正式快照。同依赖重用有限工作区缓存；所有拟合仅由 `run_pipeline` 执行。普通后处理不能调用这些拟合入口，只能解析它们已确认的 B。

现有 [batch/state.py](../src/ftir_workbench/batch/state.py) 的状态接口：

```python
set_preparation_draft(workspace, spectrum_id, preparation: PreparationConfig) -> None
confirm_preparation(workspace, spectrum_id) -> None
set_coarse_draft(workspace, spectrum_id, coarse: CoarseBaselineConfig) -> None
set_fine_draft(workspace, spectrum_id, fine: FineBaselineConfig) -> None
confirm_coarse(workspace, spectrum_id) -> StageSnapshot
confirm_fine(workspace, spectrum_id) -> StageSnapshot
skip_fine(workspace, spectrum_id) -> StageSnapshot
get_ready_snapshot(workspace, spectrum_id, stage) -> StageSnapshot
restore_committed_draft(workspace, spectrum_id, stage) -> None
draft_changes(workspace, spectrum_id) -> dict[str, bool]
copy_parameters(workspace, source_id, target_ids, *, stage,
                include_range=False, include_smoothing=False) -> dict[str, str | None]
```

这里 `source_id` 参数名实际接收**源 spectrum_id**，不是 `ImportedSource.source_id`。`copy_parameters` 默认只复制相应阶段独立 draft；两个可选参数指基线处理范围和 **estimate-only smoothing**，不是本次后基线平滑。它不复制单位、floor、父结果或确认状态。普通批量预览逐目标读取自己的 draft、逐条隔离错误；UI 还先检查无效编辑器，避免用旧有效 draft 冒充当前无效编辑值。

现有失效方式：

| 变化 | 现有行为 | 后处理需要观察的依赖 |
|---|---|---|
| 仅改变准备/粗/细 draft | 正式快照不变，旧 preview 可能不匹配 | 不使 B/S/N 的正式结果失效 |
| 确认改变准备单位/范围/估计平滑 | 更新解释，当前 coarse/fine 标 stale | B 不可用，全部派生不可作为当前输出 |
| 确认改变 coarse | coarse 被新快照替代；该谱旧 fine stale | 直到新 fine/skip 确认前不能解析 B |
| 确认 fine 或明确 skip | 更新该谱最终快照/父级/decision | 比较实际 B fingerprint，变化时 S/N_B/N_S 过期 |
| 相同 coarse 预览再次确认 | 不无故使 fine 失效 | 后处理确认也必须遵循幂等规则 |
| 改名/排序/选谱/缩放 | 科学 fingerprint 不变，必要导出 metadata 重建 | 不运行后处理或改变科学依赖 |

[batch/fingerprints.py](../src/ftir_workbench/batch/fingerprints.py) 的 `stage_fingerprint(workspace,spectrum_id,config,stage,parent_coarse_fingerprint=None,*,implementation=None)` 已绑定 workspace/谱/来源、列选择、输入、完整基线 config 和实现。`implementation_fingerprint()` 是进程内 `lru_cache(maxsize=1)`，读取冻结 baseline 所有 `.py` 及六个 batch 核心文件、数值依赖版本和 Python 版本。保留已保存的旧 `implementation` 可校验历史快照；不应给旧 B 重算新 fingerprint 冒充原版本。新派生 fingerprint 应另用有效后处理参数及 output hash，而不是改变旧 stage hash 的定义。

## 5. 可复用平滑 API 与最小公开门面

本地 [post_baseline_smoothing.py](../src/ftir_workbench/post_baseline_smoothing.py) 当前公开符号只有：

```python
PostBaselineSmoothingConfig(...)
PostBaselineSmoothingResult(...)
apply_post_baseline_smoothing(prepared: PreparedSpectralDataset,
                             config: PostBaselineSmoothingConfig) -> PostBaselineSmoothingResult
post_baseline_smoothing_fingerprint(parent_prepared, config, smoothed_spectra) -> str
```

现有 config 默认为 disabled、savgol 7点/2阶/interp；Gaussian sigma=1、truncate=4；Moving/Median 窗口3；convolution reflect；uniformity rtol=1e-3、非均匀轴默认 error。构造器严格拒绝 bool 窗口、非整数、非法 mode、非法 sigma/truncate。`to_dict()` 保存全部可编辑字段；`scientific_dict()` 只返回所选算法的有效参数，disabled 精确为 `{"enabled":false}`，可直接复用“闲置参数不改科学 hash”的规则。

可在**同一模块内部**复用的实际数值 helpers：

```python
_axis_diagnostics(wavenumber, *, uniformity_rtol) -> (median_spacing, relative_deviation, uniform)
_validate_active_window(config, *, n_points) -> None
_apply_filter(source: FloatArray, config) -> FloatArray
_compute_qc(before, after, wavenumber, config) -> (per_spectrum, summary, warnings)
_approximate_physical_width(config, *, median_spacing) -> dict[str, float]
```

滤波输入为 `(n_spectra,n_points)`，SG/Gaussian/Moving 明确 `axis=1`，Median 为 `size=(1,window)`；不混合谱行。非均匀性使用绝对间距、`allclose(...,rtol=config.uniformity_rtol,atol=1e-8)`。宽度及中性 QC 已完整实现，不需要复制四套 SciPy 调用或重新写 QC 算法。

**建议增量、当前尚不存在：** 同模块增加不接触 Prepared 的 `ArraySmoothingResult` 与 `smooth_spectral_arrays(wavenumber,spectra,config)`，验证真实 x/y 并复用上述 helpers。普通层以 `(1,n)` 调用，返回不可变 x/y、`B-S`、有效参数、物理宽度与诊断。公开门面还需处理输入形状、单调/finite/短谱、disabled identity 及拒绝未支持的分段结构；不能借构造假的 Prepared 获得这些检查。

原 `PostBaselineSmoothingResult.parent_prepared` 必须是真 Prepared，构造器还核对其 Prepared fingerprint；不能直接复用该结果类型装普通谱。原位 `apply_post_baseline_smoothing()` 在运算前拒绝 `normalization_state="scientific_explicit"` 和已有 post-baseline smoothing branch。旧 [PostBaselineSmoothingService](../src/ftir_workbench/services/smoothing_service.py) 的 `preview(prepared,config)` 返回该结果；`apply(prepared,config)` 返回 `(result,Prepared child)`；`build_bundle(result,prepared)` 返回旧 smoothing ZIP。新普通后处理不能调用这些 Prepared 服务，也不能放宽它们的组合/重复平滑 guards。

## 6. 冻结归一化的精确接口与适配缺口

现有 [ftir_baseline/normalization.py](../src/ftir_baseline/normalization.py) 公开签名：

```python
normalize_spectra(wavenumber, spectra, method="none", *, interval=None,
                  reference_interval=None, target=1.0, use_absolute=True,
                  instability_cv_threshold=0.10) -> NormalizationResult
apply_normalization(wavenumber, spectra, config=None, **overrides) -> NormalizationResult
```

允许 1D 或 2D spectra，保持调用维数；`factors` 始终为每谱系数数组。wrapper 将 `internal_reference_range→reference_interval`、`integration_range→interval`、`absolute→use_absolute`、`target_value→target`，并过滤非认可参数，所以普通适配必须主动验证有效参数，不能依靠“传入了某字段”证明核心使用了它。

| 普通语义 | 调用既有核心 | 必须读取的输出 |
|---|---|---|
| 不处理 | `none` | 保留父数据，不能标为已归一化 |
| 全域最大正峰 | `internal_peak_height`，reference interval=父谱完整实际域，`use_absolute=False` | `optional_normalized`，检查不是 None |
| 窗口正峰 | `internal_peak_height`，reference interval=明确窗口，`use_absolute=False` | `optional_normalized` |
| 参考窗口面积 | `internal_peak_area`，reference interval=窗口，absolute/signed 显式选择 | `optional_normalized` |
| 全域/指定范围面积 | `area`，interval=实际选定范围，absolute/signed 显式选择 | `optional_normalized` |
| L2 | `vector`，全父谱点集 | `optional_normalized` |
| Min–Max 0–1 | `minmax_display`，target不开放 | `view_data`，purpose=display_only |

`analysis_data` **始终是调用方输入**，不能用于 N 输出。Min–Max 的 `optional_normalized is None`，`factors=1/span`；核心不保存 offset/min/max/span，普通 metadata 必须补充 `offset=-min/span` 才能完整表达 `z=scale*y+offset`。所有 N 输出标为 normalized/scaled intensity，禁止再把它作为 A 做 T/%T 转换。

明确需要在新普通适配层补充而不改冻结 core 的保护：

- `_interval_mask` 只掩码选点且至少2点，并不要求请求 bounds 完全在父域内；适配需拒绝部分越界、记录请求/实际 bounds 和点数，不插值，不裁剪全谱输出。
- 默认 `use_absolute=True` 对峰高会选负谷绝对值；普通正峰入口必须覆盖为 False，并拒绝非正分母。
- 面积内部按 x 升序副本作梯形积分，输出方向不变；signed 负面积在 core 并未拒绝，普通层需正分母及近抵消保护。absolute 只用于分母，不改输出信号符号。
- core 的 near-zero 检查主要基于 float64 tiny；新增有版本记录的 scale-aware 数值阈值，并检查 reference、scale/offset/output finite/正scale。vector 的 `np.linalg.norm` 与极端数据可能 overflow/underflow，应明确失败，不能静默修谱或改分母。
- 单谱内部参考 `reference_cv=0` 不是跨样品稳定性证据；现有 series 文本只保留审计语义，普通展示不能宣称稳定内标已验证。
- B 与 S 必须各自求系数。保留两套 normalization draft/result；不能把 B 的 factors 复制到 S 来源或另一条谱。

## 7. 旧普通导出与工作区合同

现有 [batch/export.py](../src/ftir_workbench/batch/export.py)：

```python
build_batch_export(workspace, spectrum_ids, *, stage, ready_only=False,
                   include_wide=False, derived_units=()) -> BatchExportArtifact
can_export_wide(workspace, spectrum_ids, *, stage, ready_only=False) -> tuple[bool,str]
verify_batch_export(payload: bytes) -> bool
```

`stage` 仅 `coarse/fine`。`_select()` 对每条调用 `get_ready_snapshot`，只用已确认 `result.analysis_data[0]` 序列化。默认遇未就绪/排除项阻断；明确 ready-only 才导出其余并报告。宽表对**实际输出 x** 作 `np.array_equal`。CSV 用 `.17g`，文本域防公式注入，安全文件名带完整稳定 ID。结果类型 `independent_baseline_batch`、schema `1.0`、`is_2d_ready=false`，无 Prepared/for_2dcos。旧 verifier 只检验归档/manifest 完整性，不声称重算算法。新 S/N 的独立包不能静默替换此旧导出按钮的数组。

现有 [batch/workspace.py](../src/ftir_workbench/batch/workspace.py)：

```python
save_batch_workspace(workspace: BatchWorkspace) -> bytes
load_batch_workspace(bundle_bytes: bytes) -> BatchWorkspace
make_archive(members: dict[str,bytes], artifact_type: str) -> bytes
read_verified_archive(payload: bytes, artifact_type: str) -> dict[str,bytes]
```

workspace artifact 为 `independent_baseline_workspace`。**manifest 与 workspace.json 两层均硬编码 schema `1.0`**；目前没有 v0.3.1 新字段迁移逻辑。保存原文件、来源/列映射、typed draft/committed、完整 PipelineResult 和历史 stale 快照；所有 preview 故意置空。恢复调用公开 parser 重读来源，校验实际原列 x/y、业务结构、单位解释、数组形状/分解、recipe、saved fingerprint/parent implementation；不重跑基线拟合或平滑。来自旧实现的快照保留旧 fingerprint 并附“未由当前 pipeline 重新计算/确认”提示。

`_snapshot_payload/_load_snapshot/_validate_result` 已承担 B 完整序列化与验证；可在现有 workspace 模块内部复用这些能力保存新分支所需 B 历史，不复制第二套基线数据格式。`_Arrays` 只接受受限 float64 NPY、C-order、NPY v1/v2、最多3维及实际字节一致；`np.load(...,allow_pickle=False)`。反复引用同一数组也计入解码总预算。ZIP 限额为压缩256 MiB、单成员64 MiB、解压/解码总量512 MiB、成员20,000；拒绝重复/危险路径、符号链接、加密及不支持压缩。严格 JSON 拒绝重复 key/非标准 NaN/Inf，业务 metadata 不接受数值 codec 标记；无效编辑缺失单元格转 null 并保留错误。

`ui._workspace_context(...,for_save=True)` 当前只包含旧普通字段；新增状态必须进入工作区保存下载的失效依据，否则新后处理改变后可能仍下载旧保存包。旧基线导出 context 则应维持原含义，不因只创建 S/N 而切换输出分支。

## 8. 冻结边界与需要保护的现有测试

实际 [v0.2.1 science manifest](../artifacts/v0.2.1_science_freeze_manifest.json) 为34文件，冻结根为：

```text
src/ftir_baseline/**
src/ftir2dcos/twodcos/**
src/ftir2dcos/preprocessing/smoothing.py
src/ftir2dcos/peak_order.py
src/ftir_workbench/cross_views.py
src/ftir_workbench/display_units.py
src/ftir_workbench/services/baseline_service.py
src/ftir_workbench/services/twodcos_service.py
```

本映射期间实际调用 `scripts.audit_v025_release.audit_science_freeze(Path.cwd())`：`status=pass`，start/worktree各检查34，extra/missing/Git diff/mismatches均空。此旧审计的 manifest 基准仍是 `92513def...`；它只回答旧科学冻结是否完整，**不替换本轮开发 HEAD `9fc13c10...`**。

`ftir_baseline/normalization.py` 在冻结范围内，不修改。`post_baseline_smoothing.py` **不在**这份 manifest/冻结根内，允许最小新增数组公开门面；其现有 Prepared 行为仍由测试保护，不移除 guards，也不重写旧 manifest。

| 保护对象 | 实际现有测试与关键断言 |
|---|---|
| 普通输入/独立性 | `tests/batch/test_importing.py`、`test_adversarial.py`：异轴/同名/同值来源隔离、无关B编辑删除重排不改A、拒绝联合模式/基线归一化、错谱/过期preview及父级错配 |
| 母谱与状态 | `test_state_and_stages.py`、`test_method_parity.py`：全部旧粗细方法权威等价、estimate-only两通道、明确skip、草稿/正式隔离、准备确认失效 |
| 普通导出/安全恢复 | `test_export.py`、`test_workspace_roundtrip.py`：无重算、真正同轴、stale阻断、保存old implementation/旧fine父实现、恶意NPY/路径/结构/重复引用预算 |
| 普通UI | `test_ui.py`、`test_ui_roundtrip.py`：A→B→A、未渲染widget恢复、按各自配方预览、无效编辑隔离、真实媒体存储下载、上传恢复、下一条/同轴导出 |
| 原归一化 | `tests/baseline_regression/test_normalization.py`：analysis保持输入、optional分支、面积方向、MinMax仅view、config aliases、零/非有限/少点拒绝 |
| 旧平滑数值 | `tests/smoothing/test_post_baseline_smoothing.py`、`test_smoothing_axis.py`、`test_smoothing_config.py`、`test_smoothing_metrics.py`：四方法SciPy等价、axis1、Median(1,w)、disabled exact、闲置参数hash、有限/immutable、非均匀策略/物理宽度 |
| 旧Prepared guards/导出/2D | `test_smoothing_prepared.py`、`test_smoothing_export.py`、`test_smoothing_2d_integration.py`；`test_scientific_normalization_and_chained_smoothing_are_rejected` 显式锁定禁组合/禁chained |
| 原位界面/CLI/科学冻结 | `tests/ui/test_smoothing_page.py`、`test_smoothing_preview.py`、原 unified/preview/Cross UI；`tests/integration/` Prepared/workflow/smoothing_cli；`tests/release/test_v025_release_audit.py`、`test_v021_release_audit.py` |

以上是已存在且已审阅相关实现/断言的保护位置，不是本轮通过数量。新66项 PP-001…PP-066 需分别落到数值、状态、导出、迁移、AppTest及真实浏览器证据；不能把66场景数当pytest计数，也不能以现有测试文件存在宣称新测试通过。

## 9. 建议增量落点（待实现）

采用已有普通工作区扩展，保持基线状态/算法服务不变：

1. `BatchWorkspace` 新增 `postprocessing: dict[spectrum_id, OrdinaryPostprocessingState]`，默认空；在 `batch/postprocessing.py` 定义有限分支状态、不可变派生快照和服务。复用现有 `records`、谱选择及 B，不新增第二个普通 workspace；无需改造 `SpectrumProcessingState` 或原基线 `state.py`。
2. 后处理 resolver 每次对照当前有效 B/S fingerprint 计算 current/stale；新快照保留不可变 B `StageSnapshot`，N_S 另保留所需 S 父快照。B更新只使当前谱 S/N_B/N_S 过期，S更新只使 N_S 过期；N_B与N_S各自有draft/preview/committed。数组相同但配方/provenance不同仍是不同父级。
3. `post_baseline_smoothing.py` 仅新增数组门面；`batch/normalization_adapter.py` 完成普通语义、范围/近零保护及输出字段选择。后处理服务不得 import Streamlit、调用 `run_pipeline`、Prepared adapter 或2D引擎。
4. 在 `ui/batch_workflow.py` 增加后处理区域/页，沿用当前谱和勾选集，明确B/S来源、物理宽度、quantity、独立双图/残差和批量确认。默认导出及N来源仍为B；创建S不自动切换。
5. 在现有导出体系新增明确的普通后处理 artifact 路径，保留旧 coarse/fine exporter；指定分支缺失不得fallback。N_B包带B，N_S包带B/S，系数、offset、参考区间及purpose完整；所有导出只序列化正式不可变快照。
6. 扩展现有 workspace 读写为新 schema `2.0`，明确同时读取旧 `1.0`，旧缺失后处理字段初始化为空/disabled；保留旧B数组/配方/ID/fingerprint。协调manifest及workspace payload版本，避免旧reader把新包误认成旧schema。沿用同一ZIP/NPY安全校验，保存新draft/committed/stale/选择项并省略preview。

这些是依据真实代码提出的最小接入点；后续实现与每阶段真实验证还需单独记录。本映射本身只新增此文档。
