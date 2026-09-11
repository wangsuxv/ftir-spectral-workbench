# 普通光谱后处理：实际验收证据矩阵

本文件按父目录 `Acceptance_Tests.md` 的 **66 项场景**逐项记录证据，记录范围截至 **2026-09-11，Phase 5 的 v0.3.1 最终全量检查与真实浏览器下载审计**。最终全量 **1238 passed**，完整静态检查、构建和冻结审计通过。版本更新前的完整浏览器闭环与最终 v0.3.1 重启后的恢复、S/N_S 实际计算、确认和下载均通过；各批产物保留真实版本元信息。完成提交与最终差异审计见 [本次完成报告](../artifacts/validation/v0.3.1/REPORT.md)。场景数量不是 pytest 通过数量。

## 实际基准、环境与执行批次

实际起点是本地 `9fc13c10ae34bd53cb220d0807edb18c9e2e41dc`（v0.3.0）；起点干净，实施分支为 `feat/v0.3.1-ordinary-postprocessing`。Phase 2 完成提交为 `1485405be29eadc07b42b926964cca19bea3a8d8`；Phase 3/4 实现提交为 `c5072946508e5c0f6f9a71da803a99221bada743`，P5 前置全量在该实现上执行；两者均不是最终完成 SHA，最终审计 HEAD 及完成记录见 [本次完成报告](../artifacts/validation/v0.3.1/REPORT.md)。未 reset 到远程 v0.2.5，也未升级数值依赖。来源见 [Phase 0 说明](../artifacts/validation/v0.3.1/phase0/README.md) 与 [环境记录](../artifacts/validation/v0.3.1/phase0/environment.json)。

执行环境为 macOS arm64、Python 3.12.14、NumPy 2.5.2、SciPy 1.18.1、Streamlit 1.62.0、pytest 9.1.1。活动 lock 条目与安装版本 64/64 一致，见 [lock 核对](../artifacts/validation/v0.3.1/phase0/lock_comparison.json)。

下表每行是一次独立执行。**各行相互重叠，不能相加成为总通过数量。** Phase 2 服务测试的 66 passed 恰巧与规格场景总数相同，二者没有一一对应关系。

| 证据代号 | 实际命令（共同前缀 `.venv/bin/python`） | 实际结果 | 原始执行日志（隐私路径已脱敏） |
| --- | --- | --- | --- |
| P0 | `-m pytest -q` | 906 passed，5 个既有合成 CLI 图形范围 warning；92.28 s | [Phase 0 全量](../artifacts/validation/v0.3.1/phase0/pytest.log) |
| P0S | `-m ruff check src tests ui scripts`；`-m mypy` | Ruff 通过；mypy 29 源文件通过 | [Ruff](../artifacts/validation/v0.3.1/phase0/ruff.log)、[mypy](../artifacts/validation/v0.3.1/phase0/mypy.log) |
| P1 | `-m pytest tests/postprocessing/test_numerical_adapters.py tests/smoothing tests/baseline_regression/test_normalization.py -q` | 206 passed = 当时 105 个新增参数化 case + 101 个旧 case；2.77 s | [Phase 1 数值/旧回归](../artifacts/validation/v0.3.1/phase1/numerical-and-existing-regression.log) |
| N | 同 P1 命令，增加只读 request validator 测试后重跑 | 218 passed = 117 个新增 case + 同一组 101 个旧 case；2.73 s | [Phase 2 数值/旧回归](../artifacts/validation/v0.3.1/phase2/request-validation-numerical-regression.log) |
| S | `-m pytest tests/postprocessing/test_state_contracts.py tests/postprocessing/test_service_quantities.py -q` | 66 passed = 47 个状态 case + 19 个数量语义/完整性 case；3.15 s | [Phase 2 服务最终执行](../artifacts/validation/v0.3.1/phase2/final-service-snapshot-guards.log) |
| B0 | `-m pytest tests/batch -q` | 181 个既有普通基线 case passed；27.27 s | [旧 batch 回归](../artifacts/validation/v0.3.1/phase2/old-batch-regression.log) |
| F | `scripts/audit_v025_release.py --expected-version 0.3.0 --output …` | 34/34 冻结文件精确字节及根路径集合通过；旧 bundle、Prepared/smoothing、2D、Cross 审计通过 | [Phase 1 冻结/兼容审计](../artifacts/validation/v0.3.1/phase1/freeze-and-legacy-audit.log) |
| A | 对 `git show HEAD:src/ftir_workbench/post_baseline_smoothing.py` 的起点内容核对新增前缀 | 原 35,105 字节未改，只追加新公开数组入口 | [追加范围检查](../artifacts/validation/v0.3.1/phase1/smoothing-append-only.log) |
| P2S | 数值源码 Ruff/mypy；完整 batch 与新增测试 Ruff | 均通过；数值 mypy 2 源文件通过 | [数值 Ruff](../artifacts/validation/v0.3.1/phase2/request-validation-ruff.log)、[数值 mypy](../artifacts/validation/v0.3.1/phase2/request-validation-mypy.log)、[Phase 2 最终 Ruff](../artifacts/validation/v0.3.1/phase2/final-static.log) |
| U3 | `-m pytest tests/postprocessing/test_ui_postprocessing.py tests/postprocessing/test_ui_adversarial.py tests/batch/test_ui.py tests/batch/test_ui_roundtrip.py tests/ui/test_smoothing_page.py tests/ui/test_smoothing_preview.py -q` | 48 passed；56.20 s | [Phase 3 新旧 UI](../artifacts/validation/v0.3.1/phase3/ui-gate-final.log) |
| U4 | U3 文件组再加入 `tests/postprocessing/test_ui_roundtrip.py` | 54 passed；71.01 s；该轮不包含后续新增“全部可用分支”严格策略回归 | [Phase 4 新旧 UI](../artifacts/validation/v0.3.1/phase4/ui-gate.log) |
| U4A | `-m pytest tests/postprocessing/test_ui_roundtrip.py -q` | 6 passed；15.52 s；包含“全部分支”须显式 valid-only 的修补后回归 | [UI 全部分支策略复核](../artifacts/validation/v0.3.1/phase4/ui-all-explicit-policy.log) |
| E4 | `-m pytest tests/postprocessing/test_export.py tests/postprocessing/test_export_adversarial.py tests/batch/test_export.py -q` | 96 passed；3.72 s；含 33 个独立攻击 case | [新旧导出最终执行](../artifacts/validation/v0.3.1/phase4/export-tests-final.log) |
| W4 | `-m pytest -q tests/postprocessing/test_workspace_migration.py tests/postprocessing/test_service_edge_lineage.py` | 39 passed = 36 个迁移/安全 case + 3 个来源链路 case；4.70 s | [迁移与来源链路](../artifacts/validation/v0.3.1/phase4/workspace-migration-final.log) |
| OW4 | `-m pytest tests/batch/test_workspace_roundtrip.py -q` | 73 个旧 workspace case passed；3.34 s | [旧工作区回归](../artifacts/validation/v0.3.1/phase4/workspace-existing-first.log) |
| F4 | `scripts/audit_v025_release.py --expected-version 0.3.0` | 34/34 冻结字节与路径集合、旧 bundle/Prepared/smoothing/2D/Cross、Git index 隐私检查通过；当时 workbench 仍为 0.3.0 | [Phase 4 冻结/兼容](../artifacts/validation/v0.3.1/phase4/freeze-and-legacy-audit.log) |
| P4S | UI、导出、迁移新增文件 Ruff；导出 mypy 与 UI `--follow-imports=skip` mypy | 所列范围通过；不替代完成版完整静态检查 | [UI Ruff](../artifacts/validation/v0.3.1/phase4/ui-static.log)、[UI mypy](../artifacts/validation/v0.3.1/phase4/ui-mypy.log)、[导出 Ruff](../artifacts/validation/v0.3.1/phase4/export-ruff-final.log)、[导出 mypy](../artifacts/validation/v0.3.1/phase4/export-mypy.log)、[迁移 Ruff](../artifacts/validation/v0.3.1/phase4/workspace-migration-ruff.002.log) |
| P5P | `-m pytest -q` | 1230 passed，5 个与基准相同的 CLI 图形范围 warning；136.87 s；版本更新前执行，未包含随后发现的 Min–Max 恢复攻击回归 | [版本更新前全量](../artifacts/validation/v0.3.1/phase5/pytest-pre-version.log) |
| P5S | `-m ruff check src tests ui scripts`；`-m mypy` | 完整 Ruff 通过；mypy 32 源文件通过；与 P5P 同一阶段 | [完整 Ruff](../artifacts/validation/v0.3.1/phase5/ruff-pre-version.log)、[完整 mypy](../artifacts/validation/v0.3.1/phase5/mypy-pre-version.log) |
| P5M | `-m pytest -q tests/postprocessing/test_state_contracts.py tests/postprocessing/test_service_quantities.py tests/postprocessing/test_workspace_migration.py tests/postprocessing/test_export.py tests/postprocessing/test_export_adversarial.py tests/batch/test_workspace_roundtrip.py tests/batch/test_export.py` | 279 passed；13.00 s；包含新增 8 个 Min–Max scale/output 成对重哈希攻击 case。改动文件 Ruff 和相关 3 源文件 mypy 通过 | [修复定向回归](../artifacts/validation/v0.3.1/phase5/minmax-integrity-regression.log)、[修复 Ruff](../artifacts/validation/v0.3.1/phase5/minmax-integrity-ruff.log)、[修复 mypy](../artifacts/validation/v0.3.1/phase5/minmax-integrity-mypy.log) |
| BR5 | Playwright 真实上传、点击、Plotly zoom、下载和新浏览器标签上传恢复；30 条工具事件记录（包含失败定位重试） | 合成 3 条谱、12 个 B/S/N_B/N_S 节点闭环通过；事件保留真实异常与后续成功，非 30 个独立测试 case；当时 workbench 元信息为 0.3.0 | [真实浏览器事件](../artifacts/validation/v0.3.1/phase5/browser_events.json) |
| BD5 | `scripts/verify_v031_browser_artifacts.py` 实际下载文件审计（完整命令见日志） | 7 个下载文件及全部审计检查通过；12 个分支 CSV 与正式数组精确相同，恢复比较 660 个数组 / 70,860 个元素；审计没有生成替代下载文件 | [下载审计日志](../artifacts/validation/v0.3.1/phase5/browser-artifacts.log)、[结构化结果](../artifacts/validation/v0.3.1/phase5/browser_artifact_audit.json) |
| P5F | `-m pytest -q` | **1238 passed，5 个既有 CLI 图形范围 warning；132.47 s**，包含 8 个 Min–Max 恢复攻击回归；workbench 0.3.1 | [最终全量](../artifacts/validation/v0.3.1/phase5/pytest-final.log) |
| P5FS | `-m ruff check src tests ui scripts`；`-m mypy` | 完整 Ruff 通过，mypy 32 源文件通过 | [最终 Ruff](../artifacts/validation/v0.3.1/phase5/ruff-final.log)、[最终 mypy](../artifacts/validation/v0.3.1/phase5/mypy-final.log) |
| P5B | `-m build --no-isolation`；`-m pip install --no-deps dist/ftir_spectral_workbench-0.3.1-py3-none-any.whl` | v0.3.1 sdist/wheel 构建及 wheel 安装通过；无依赖升级 | [最终构建](../artifacts/validation/v0.3.1/phase5/build-final.log)、[wheel 安装](../artifacts/validation/v0.3.1/phase5/install-wheel-v031.log) |
| P5A | `scripts/audit_v025_release.py --expected-version 0.3.1` | 34/34 冻结精确字节与路径集合、旧 bundle/Prepared/smoothing/2D/Cross、workbench 0.3.1 元信息及 Git index/ignore 隐私检查全部通过 | [最终冻结与旧包审计](../artifacts/validation/v0.3.1/phase5/freeze-and-legacy-final.log) |
| BFR5 | v0.3.1 重启后真实 Playwright 新标签恢复、计算确认 A 的 S 窗口 9 与 N_S L2、下载两分支 ZIP 并保存工作区 | 2 条实际工具事件通过，浏览器异常计数均为 0；恢复历史 B，执行当前 S/N_S 计算 | [最终版本浏览器事件](../artifacts/validation/v0.3.1/phase5/browser_final_events.json) |
| BFD5 | 最终版本实际 3 个下载 ZIP 审计（完整命令见日志） | 全部通过；A 的 S/N_S 计算及 exporter 版本为 0.3.1；204 个历史 B 数组、A 的 N_B 与无关 B/C 不变，两导出包共 10 个 CSV 节点与工作区正式数组精确相同 | [最终版本下载审计](../artifacts/validation/v0.3.1/phase5/browser-final-version-artifacts.log)、[结构化结果](../artifacts/validation/v0.3.1/phase5/browser_final_artifact_audit.json) |
| P5D | 基准与最终安装依赖逐包比较 | 64 个依赖版本全部不变；仅 workbench 源码及安装元信息更新为 0.3.1 | [最终依赖比较](../artifacts/validation/v0.3.1/phase5/dependency_comparison.json) |

测试、静态检查、构建与下载审计命令由 `scripts/validate_v031.py <phase> <name> -- <command>` 实际运行；浏览器记录来自实际 Playwright 工具响应。未脱敏日志只保存在忽略目录 `outputs/validation-private/v031/`；已有失败日志也保留，没有以成功日志覆盖。初跑发现并修复了 SG 单/多 RHS 舍入容差、有效区间 tuple/list 规范化、Min–Max 大偏置重构校验以及参数复制目标范围验证；对应记录位于 [Phase 1 首跑](../artifacts/validation/v0.3.1/phase1/numerical-first.log)、[Min–Max 复现](../artifacts/validation/v0.3.1/phase2/review-smoke.log) 和 [复制范围首跑](../artifacts/validation/v0.3.1/phase2/state-contracts.log)。

Phase 4 首轮独立攻击测试发现单位记录/轴方向元数据遗漏校验，以及 CSV/浮点异常未按布尔接口拒绝；修复后全部回归通过。见 [攻击首轮失败](../artifacts/validation/v0.3.1/phase4/export-adversarial.log) 与 E4。旧导出首次采用 ZIP 全字节比较失败，实查只因旧工作区 JSON 对 QC 字典键排序，恢复后 QC 行序变化；后续比较要求光谱、配方及其余成员逐字节相同，QC 整行内容精确相同。见 [迁移首轮](../artifacts/validation/v0.3.1/phase4/workspace-migration.log) 与 W4。

## 测试选择器约定

为使表格可读，下列别名表示真实测试文件；`别名::test_name` 是 pytest 函数选择器。参数化函数选择器涵盖该函数全部已收集 case，表中不把函数数当作 case 数。将别名替换为对应路径即可运行，例如：

```bash
.venv/bin/python -m pytest \
  tests/postprocessing/test_numerical_adapters.py::test_pp009_four_methods_match_existing_prepared_result_and_qc -q
```

| 别名 | 实际文件 |
| --- | --- |
| N | [tests/postprocessing/test_numerical_adapters.py](../tests/postprocessing/test_numerical_adapters.py) |
| S | [tests/postprocessing/test_state_contracts.py](../tests/postprocessing/test_state_contracts.py) |
| Q | [tests/postprocessing/test_service_quantities.py](../tests/postprocessing/test_service_quantities.py) |
| L | [tests/smoothing/test_post_baseline_smoothing.py](../tests/smoothing/test_post_baseline_smoothing.py) |
| I | [tests/batch/test_importing.py](../tests/batch/test_importing.py) |
| U | [tests/postprocessing/test_ui_postprocessing.py](../tests/postprocessing/test_ui_postprocessing.py) |
| UA | [tests/postprocessing/test_ui_adversarial.py](../tests/postprocessing/test_ui_adversarial.py) |
| UR | [tests/postprocessing/test_ui_roundtrip.py](../tests/postprocessing/test_ui_roundtrip.py) |
| E | [tests/postprocessing/test_export.py](../tests/postprocessing/test_export.py) |
| EA | [tests/postprocessing/test_export_adversarial.py](../tests/postprocessing/test_export_adversarial.py) |
| W | [tests/postprocessing/test_workspace_migration.py](../tests/postprocessing/test_workspace_migration.py) |
| G | [tests/postprocessing/test_service_edge_lineage.py](../tests/postprocessing/test_service_edge_lineage.py) |

“Pass（数值/服务）”仅覆盖已列出的自动测试；UI、下载及恢复另列真实执行证据，执行范围限制也保留，不能由数值通过推定浏览器操作已执行。

## 66 项场景到真实证据的映射

| ID | 场景 | 当前真实测试选择器 / 证据批次 | 当前结果与剩余验收 |
| --- | --- | --- | --- |
| PP-001 | 功能未启用，旧普通结果不变 | W::test_pp001_pp062_exact_v030_workspace_preserves_all_baseline_arrays_and_progress；W::test_pp001_legacy_export_spectra_recipes_and_qc_values_remain_exact；E::test_old_baseline_export_bytes_remain_unchanged_after_postprocessing_and_new_export / W4、E4 | Pass；读取精确基准旧包，B/x/基线分量/父指纹/进度不变；后处理默认关闭。旧导出除 QC 行序外成员字节相同，QC 值精确相同。 |
| PP-002 | 后处理不重跑基线 | S::test_pp002_pp003_postprocessing_never_calls_baseline_prepared_or_twod；N::test_array_facades_never_construct_prepared_or_call_baseline / S、N | Pass；基线入口替换成抛错 spy 后 S/N 仍成功。 |
| PP-003 | 无 Prepared、无 2D | S::test_pp002_pp003_postprocessing_never_calls_baseline_prepared_or_twod；E::test_pp059_requested_branch_includes_exact_complete_parent_graph / S、E4 | Pass；Prepared/2D 构造 spy 未触发；导出节点明确 is_2d_ready=false。 |
| PP-004 | 明确最终基线前置 | S::test_pp004_coarse_only_requires_explicit_final_baseline_decision；U::test_defaults_and_final_baseline_prerequisite / S、U3 | Pass；服务与真实 AppTest 均要求细调确认或明确 skip，不能由粗调自动形成最终 B。 |
| PP-005 | 母数据与引用不可变 | S::test_pp005_parent_arrays_and_recipe_stay_immutable_across_all_branches；U::test_all_smoothing_methods_preview_confirm_and_plots；UA::test_new_preview_is_shown_and_display_controls_do_not_compute / S、U3 | Pass；父数组/配方不变，派生数组只读；实际图形渲染、版本和显示单位切换未改母数据或计算。浏览器 zoom 另见 PP-018。 |
| PP-006 | 拒绝非法处理链 | S::test_pp006_smoothing_rejects_every_nonbaseline_source；S::test_pp006_normalization_rejects_every_noncontract_source / S | Pass；S→S、N→S、N→N 均由服务拒绝。 |
| PP-007 | 单谱与无关批次等价 | S::test_pp007_single_and_unrelated_batch_have_identical_derived_arrays；N::test_all_normalization_methods_match_public_core_and_affine_evidence / S、N | Pass；普通服务逐谱单 row 数组逐元素相同。 |
| PP-008 | 原位、冻结与兼容 | L::test_scientific_normalization_and_chained_smoothing_are_rejected；P0、N、F4、A、P5F、P5A | Pass；v0.3.1 最终全量 1238 passed；34/34 冻结精确字节/路径集合及旧 bundle、Prepared/smoothing、2D/Cross 审计通过。 |
| PP-009 | 四种平滑等价 | N::test_pp009_four_methods_match_existing_prepared_result_and_qc / N | Pass；同 shape 输入与旧合法 Prepared fixture 输出/QC 一致，Median 不跨 row。 |
| PP-010 | 平滑关闭精确 identity | N::test_pp010_disabled_smoothing_is_detached_immutable_identity / N | Pass；逐元素 identity、零移除分量、独立只读内存。 |
| PP-011 | SG 参数严格验证 | N::test_pp011_savgol_invalid_active_controls_rejected / N | Pass；偶数、bool、非整数窗口、polyorder 及 mode 错误被拒绝。 |
| PP-012 | 超长窗口拒绝 | N::test_pp012_active_window_not_silently_shrunk；N::test_smoothing_request_validates_each_targets_axis_and_window / N | Pass；保留真实 n/window 错误，不自动缩窗。 |
| PP-013 | Gaussian 参数验证 | N::test_pp013_gaussian_invalid_active_controls_rejected / N | Pass；sigma/truncate 的 0、负值、NaN、Inf、bool 被拒绝。 |
| PP-014 | 升降序等价 | N::test_pp014_axis_reversal_preserves_old_tolerance / N | Pass；保留各自方向，旧输出容差 1e-14、QC 容差 1e-12。 |
| PP-015 | 非均匀轴默认拒绝 | N::test_pp015_016_nonuniform_axis_requires_explicit_index_space_policy / N | Pass；无自动重采样。 |
| PP-016 | 显式索引空间例外 | N::test_pp015_016_nonuniform_axis_requires_explicit_index_space_policy / N、BR5；UI `_smoothing_editor` 只读审查 | Pass（数值及控件）；非均匀原轴、override、warning 数值契约通过。真实浏览器在均匀轴 A 上显式选择索引空间，警告和独立草稿正确显示并保存。浏览器未另上传非均匀轴来运行该例外，保留此执行范围限制。 |
| PP-017 | 非有限或非单调拒绝 | N::test_pp017_invalid_array_contract_rejected_without_repair；I::test_i09_strict_invalid_axis_or_data_rejected / N、B0 | Pass；不填值、不排序、不丢点。 |
| PP-018 | zoom 不裁剪或计算 | S::test_pp018_reads_confirmation_cached_preview_and_copy_do_not_recalculate；UA::test_new_preview_is_shown_and_display_controls_do_not_compute / S、U3、BR5 | Pass；服务/实际 AppTest 显示和单位切换零计算 spy 通过。真实 Plotly Zoom in 将显示 [1900,900] 改为 [1650,1150]，图形数据和已确认版本/父指纹不变；该浏览器事件未冒称服务端调用计数。 |
| PP-019 | 物理窗口随网格变化 | N::test_pp019_physical_width_uses_actual_spacing；U::test_batch_explicit_preview_confirm_with_per_spectrum_widths / N、U3 | Pass；7 点窗口间距 1/2 对应跨度 6/12；实际批量预览表显示各谱物理窗口，不自动换窗。 |
| PP-020 | 移除分量恒等式/名称 | N::test_pp020_021_022_removed_component_median_warning_and_unclipped_outputs；U::test_all_smoothing_methods_preview_confirm_and_plots；E::test_pp059_requested_branch_includes_exact_complete_parent_graph / N、U3、E4 | Pass；removed=B-S，实际页面双图及 CSV 分量使用 removed component/移除分量语义，不作为真实噪声或自动 SNR 结论。 |
| PP-021 | Median 非线性提示 | N::test_pp020_021_022_removed_component_median_warning_and_unclipped_outputs；U::test_all_smoothing_methods_preview_confirm_and_plots / N、U3 | Pass；Median 数值/风险 warning 与实际方法页面预览均通过；界面保留真实窄峰可能受损的提示。 |
| PP-022 | 不裁负值或重置端点 | 同 PP-021 / N | Pass；保留负输出与实际端点变化，无基线追加调用。 |
| PP-023 | 读取真实归一化分支 | N::test_pp023_025_026_positive_maximum_reads_scientific_branch；E::test_pp023_csv_reads_corrected_normalized_output_not_parent_analysis_data / N、E4 | Pass；[0,2,4]→[0,0.5,1]，实际导出读取归一化输出。 |
| PP-024 | Min–Max view 与偏移 | N::test_pp024_minmax_reads_view_and_records_complete_affine_transform；Q::test_minmax_large_offset_uses_stable_core_expression_for_validation / N、S | Pass；[-1,1,3]→[0,0.5,1]，scale/offset 均 .25，purpose=display_only；大偏置校验使用核心稳定表达式。 |
| PP-025 | 峰高不暗含平移 | N::test_pp023_025_026_positive_maximum_reads_scientific_branch / N | Pass；[-1,1,3] 保留负值，offset=0。 |
| PP-026 | 正峰高而非负谷 | 同 PP-025 / N | Pass；[-.3,.1,.2] 的分母是 .2。 |
| PP-027 | 全非正峰高拒绝 | N::test_pp027_nonpositive_height_rejected / N | Pass；无负 scale 翻转。 |
| PP-028 | L2 解析例与零向量 | N::test_pp028_l2_analytic_example_and_zero_guard / N | Pass；[3,4]→[.6,.8]，零向量拒绝，保留采样密度说明。 |
| PP-029 | 实际 x 面积 | N::test_pp029_030_real_x_area_not_sample_sum_and_direction_invariant / N | Pass；x=[1000,1002,1004] 分母 2，不用样本求和。 |
| PP-030 | 降序面积不翻号 | 同 PP-029 / N | Pass；分母同为正 2，输出轴保持降序。 |
| PP-031 | absolute/signed 定义 | N::test_pp031_absolute_and_signed_area_keep_original_output_signs / N | Pass；两种面积配方及分母不同，absolute 仅影响分母。 |
| PP-032 | 带符号近抵消拒绝 | N::test_pp032_signed_area_zero_negative_or_rounding_cancellation_rejected / N | Pass；零、负值及舍入尺度近抵消拒绝。 |
| PP-033 | 请求区间整体覆盖 | N::test_pp033_partial_outside_reference_interval_rejected；N::test_normalization_request_validates_target_range_samples_and_reference_guards / N | Pass；部分越界也拒绝，不取交集。 |
| PP-034 | 实际点数与区间端点 | N::test_pp034_request_bounds_actual_samples_and_at_least_two_points / N | Pass；一实际点拒绝，非对齐端点不插值，记录 requested/actual bounds。 |
| PP-035 | 窗口不裁剪输出 | N::test_pp035_reference_interval_controls_denominator_not_output_domain；Q::test_reference_recipe_survives_immutable_json_conversion / N、S | Pass；窗口系数应用于完整 B/S 数组，tuple/list 序列化不误判配方。 |
| PP-036 | target 与 Min–Max UI | N::test_pp036_active_target_is_applied；N::test_pp036_minmax_target_is_inactive_and_effective_target_fixed_to_one；U::test_normalization_active_controls_and_selected_output / N、U3、BR5、BD5 | Pass；target 按有效方法生效，Min–Max 固定 0–1；AppTest 与真实浏览器均核对 target 隐藏，下载的 Min–Max 数组/量纲匹配确认版；raw 闲置值不进入科学配方。 |
| PP-037 | 近零与溢出保护 | N::test_pp037_scale_relative_reference_floor_not_fixed_absolute_epsilon；N::test_pp037_core_extremes_fail_with_clear_numerical_error；Q::test_minmax_large_offset_uses_stable_core_expression_for_validation / N、S | Pass；记录版本化 64×eps 保护、用户最小参考阈值，无法有限计算时明确失败。 |
| PP-038 | CV=0 不是稳定内标证据 | N::test_pp038_reference_cv_zero_is_not_reported_as_cross_sample_evidence；U::test_normalization_active_controls_and_selected_output / N、U3；UI `_normalization_editor` 只读审查 | 数值 metadata/warning 与实际内标页面通过；界面明确单谱 CV=0 不能验证跨样品内标稳定性。 |
| PP-039 | B/S 各自求分母 | N::test_pp039_040_normalize_exact_selected_parent_in_correct_order；G::test_pp039_area_normalization_resolves_each_actual_b_and_s_parent；UA::test_source_specific_normalization_drafts_and_export_choice_survive_new_s_and_switches / N、W4、U3 | Pass；最大峰高及面积分别从 B/S 求分母；专门面积例两父级积分不同，UI 两来源草稿独立。 |
| PP-040 | S 后再 N 的顺序 | N::test_pp039_040_normalize_exact_selected_parent_in_correct_order / N | Pass；[0,0,9,0,0] 移动平均后归一化得到 [0,1,1,1,0]。 |
| PP-041 | 量纲与物理 T 限制 | Q::test_pp041_normalized_quantities_cannot_be_shown_as_physical_transmittance；U::test_normalization_active_controls_and_selected_output；UR::test_negative_absorbance_display_retains_values_and_reports_overflow；E::test_pp058_minmax_export_quantity_offset_and_filename_are_explicit / S、U4、E4 | Pass；N 使用 normalized/scaled 标签且无物理 T 转换；B/S 负吸光度及派生 >100%T 不静默裁剪，显示溢出明确报告；Min–Max 输出说明 display-only。 |
| PP-042 | 异质配方提示 | E::test_pp042_heterogeneous_recipes_warn_even_when_output_axes_match；UA::test_normalization_copy_and_batch_confirmation_report_missing_s_and_window_per_target / E4、U3 | Pass；不同有效配方/网格有异质比较 warning，不声称定量可比；复制缺 S/缺区间逐目标报错。 |
| PP-043 | 预览不提交 | S::test_pp043_preview_alone_is_never_a_confirmed_branch；U::test_all_smoothing_methods_preview_confirm_and_plots；E::test_pp056_missing_branch_default_blocks_and_valid_only_explicitly_reports_no_fallback / S、U3、E4 | Pass；预览不自动提交，正式导出仅接受当前已确认分支。 |
| PP-044 | 草稿与正式版共存 | S::test_pp044_pp045_edit_preserves_confirmed_but_blocks_old_preview_commit；U::test_committed_draft_restore_and_save_download_context；UR::test_explicit_exports_include_parents_report_missing_and_preserve_download_bytes / S、U4 | Pass；草稿与确认版共存，页面提醒未确认修改；下载采用已确认版。 |
| PP-045 | 过期预览拒绝确认 | 同 PP-044；S::test_preview_from_another_identity_cannot_be_confirmed / S | Pass；配置、父级或身份不符不能提升旧预览。 |
| PP-046 | S 更新只使 N_S 过期 | S::test_pp046_new_smoothing_invalidates_only_its_normalized_child；UA::test_source_specific_normalization_drafts_and_export_choice_survive_new_s_and_switches；E::test_pp056_stale_smoothing_child_and_stale_final_baseline_are_not_exported / S、U3、E4、BR5、BD5 | Pass；浏览器确认 A 新 S 窗口 9 后，仅 A 的 N_S 过期，N_B 仍 ready；严格导出阻断，明确 valid-only 后实际 ZIP 排除 A 的 N_S，B/C 数组不变。 |
| PP-047 | B 更新按谱失效 | S::test_pp047_confirmed_new_baseline_invalidates_only_that_records_descendants；G::test_pp047_new_confirmed_preparation_and_b_invalidate_only_own_derived_lineage / S、W4 | Pass；coarse 配方、输入单位或范围确认并生成新 B 后，仅该谱派生过期；其他谱不变。 |
| PP-048 | N_B 与 N_S 隔离 | S::test_pp048_normalization_branches_do_not_supersede_each_other / S | Pass；只替换所选 N 分支。 |
| PP-049 | 同数组不同配方仍失效 | S::test_pp049_equal_smoothing_arrays_with_different_recipe_still_change_parent / S | Pass；恒定谱不同 Median 窗口输出相同但科学父指纹不同。 |
| PP-050 | 幂等与闲置参数 | S::test_pp050_reconfirm_and_unused_parameters_do_not_invalidate_descendants；N::test_pp050_effective_config_ignores_inactive_invalid_fields_but_preserves_raw_draft；UA::test_inactive_invalid_sg_draft_survives_switching_records_and_unrendered_pages / S、N、U3 | Pass；闲置无效参数保留在独立草稿，不进入 effective hash；切谱/切页/搜索清理后仍恢复。 |
| PP-051 | 切谱/来源草稿隔离 | UA::test_inactive_invalid_sg_draft_survives_switching_records_and_unrendered_pages；UA::test_source_specific_normalization_drafts_and_export_choice_survive_new_s_and_switches / U3、BR5、BD5 | Pass；AppTest 切页/过滤/N 来源切换恢复草稿与确认版；真实浏览器 A→B→C→A 恢复 A 的 S 来源与 L2 草稿，新标签工作区恢复也精确保留来源与独立草稿。 |
| PP-052 | 复制逐目标校验 | S::test_pp052_copy_resolves_each_targets_smoothed_parent_without_fallback_or_coefficient_copy；UA::test_normalization_copy_and_batch_confirmation_report_missing_s_and_window_per_target / S、U3 | Pass；实际复制/批量预览/确认按每条 S 和范围验证；无 fallback，无跨谱系数复制。 |
| PP-053 | 批量失败隔离 | S::test_pp053_batch_preview_and_confirm_isolate_invalid_windows_from_good_and_committed_items；UA::test_batch_preview_rejects_short_target_without_losing_existing_result / S、U3 | Pass；短谱非法窗口逐条失败，有效项可确认，已有正式结果保留。 |
| PP-054 | 宽表/同名/cache 隔离 | S::test_pp054_identical_values_from_different_sources_never_share_snapshot_provenance；S::test_pp054_wide_columns_have_independent_derived_cache_and_parent_identity；E::test_pp060_rename_and_edit_do_not_mutate_existing_payload_or_refit / S、E4 | Pass；逐列输入/稳定身份/cache 隔离；改名只更新新导出元信息，旧科学指纹与下载 bytes 不变。 |
| PP-055 | 导出不执行计算 | E::test_pp055_export_and_nonreplay_verifier_never_call_numerical_entrypoints；UR::test_explicit_exports_include_parents_report_missing_and_preserve_download_bytes / E4、U4 | Pass；导出及 recompute=False 校验基线/S/N 抛错 spy 均未触发。显式 recompute=True 独立校验会重放 S/N，不能算作纯导出。 |
| PP-056 | 导出明确分支无回退 | E::test_pp056_missing_branch_default_blocks_and_valid_only_explicitly_reports_no_fallback；E::test_pp056_stale_smoothing_child_and_stale_final_baseline_are_not_exported；UR::test_explicit_exports_include_parents_report_missing_and_preserve_download_bytes；UR::test_all_branch_export_requires_explicit_valid_only_and_preserves_minmax_quantity / E4、U4、U4A | Pass；缺失/过期默认阻断，显式 valid-only 才排除并报告；UI 选择全部分支时同样要求显式授权 valid-only，不自动放宽策略。 |
| PP-057 | 相同/不同 x 导出 | E::test_pp057_only_exact_actual_x_arrays_allow_wide；UR::test_heterogeneous_axes_disable_wide_but_selected_same_axes_download_exactly / E4、U4、BR5、BD5 | Pass；真实 3 谱异轴的 12 分支 ZIP 禁用宽表；改选同轴 A/B 后，独立下载宽表 201 行 / 8 输出列，与 ZIP 对应成员和各正式数组 float64 精确相同；无补齐或插值。 |
| PP-058 | 浮点精度及文件名 | E::test_pp058_float64_tiny_negative_and_ordinary_values_roundtrip_exactly；E::test_pp058_minmax_export_quantity_offset_and_filename_are_explicit / E4 | Pass；CSV float64 精确往返，负/微小值保留；稳定 id、branch、quantity 和 Min–Max display-only 名称明确。 |
| PP-059 | 完整 B/S 父级和系数 | E::test_pp059_requested_branch_includes_exact_complete_parent_graph；E::test_baseline_only_versions_distinguish_saved_core_from_export_environment / E4 | Pass；N_B 带 B、N_S 带 B/S；recipe 有父指纹、scale/offset/reference、版本及用途，is_2d_ready=false。B 仅作完整性来源锚点，不宣称重拟合或原始来源认证。 |
| PP-060 | 下载 payload 不随编辑变化 | E::test_pp060_rename_and_edit_do_not_mutate_existing_payload_or_refit；UR::test_explicit_exports_include_parents_report_missing_and_preserve_download_bytes / E4、U4 | Pass；实际 AppTest 下载 payload 被保存并检查；编辑/改名不改旧 bytes，新下载标签更新且仍只用确认版。浏览器文件核验另见 PP-065。 |
| PP-061 | 篡改、安全 ZIP 恢复 | W::test_pp061_rehashed_derived_corruption_is_rejected_before_install；W::test_pp061_derived_archive_retains_path_and_resource_boundaries；W::test_pp061_paired_minmax_scale_and_output_rehash_cannot_change_fixed_span；EA::test_rehashed_identity_graph_quantity_and_report_conflicts_return_false；EA::test_malicious_csv_payloads_return_false_without_uncaught_parser_errors；EA::test_rehashed_extreme_minmax_parent_returns_false_instead_of_floating_point_exception / W4、E4、OW4、P5M | Pass；重哈希身份/父链/量纲/成员/请求、伪预览、路径、NPY/pickle 与资源边界均拒绝；恶意 CSV/极端 Min–Max 导出包校验返回失败。Phase 5 补查并修复 scale=1/span 契约；N_B/N_S、零/大偏置、成对放大/缩小攻击 8 case 全部拒绝且未重放数值计算。 |
| PP-062 | 旧 workspace 迁移 | W::test_pp001_pp062_exact_v030_workspace_preserves_all_baseline_arrays_and_progress；W::test_pp001_legacy_export_spectra_recipes_and_qc_values_remain_exact / W4 | Pass；使用 Phase 0 精确 9fc13 Git archive 生成的旧 bytes；稳定 ID、原来源/B/草稿/进度保留，派生默认空/关闭；缺夹具环境会从同一提交重新生成。 |
| PP-063 | 新 workspace 往返 | W::test_pp063_roundtrip_preserves_drafts_arrays_selected_ids_and_historical_parent_graph；UR::test_real_uploaded_workspace_restores_selection_sources_drafts_parents_and_no_previews / W4、U4、BR5、BD5 | Pass；schema 2.0 内层工作区与原 1.0 ZIP manifest 往返。实际浏览器下载→新标签上传→恢复后再下载，660 个数组 / 70,860 元素、稳定 ID、原始来源、草稿、正式历史父级和选择/显示偏好精确相同；预览不保存。 |
| PP-064 | AppTest 状态闭环 | U::test_normalization_active_controls_and_selected_output；UA::test_confirm_and_next_visits_three_postprocessing_records_and_preserves_each_branch；UR::test_real_uploaded_workspace_restores_selection_sources_drafts_parents_and_no_previews / U3、U4 | Pass；真实已安装 Streamlit AppTest 上传、参数/来源、切谱/页、preview/confirm、A→B→C→A 和下载上传恢复闭环；不以 mock widget API 代替。 |
| PP-065 | 浏览器上传与下载验收 | BR5、BD5、BFR5、BFD5；30+2 条真实工具事件，7+3 个实际下载文件 | Pass；完整闭环覆盖合成宽表 A/B 加异轴 C、显式 skip 形成 B、逐谱不同 S/N_B/N_S、A→B→C→A、真实 zoom、stale 阻断/明确排除、同异轴下载与新标签恢复。最终 v0.3.1 重启后再次恢复，计算确认 A 的新 S/N_S 并下载；实际数组、父链、隔离及版本全部通过。前批元信息 0.3.0、最终批 0.3.1 均如实保留。 |
| PP-066 | 报告与隐私 | P0 环境/lock、各阶段真实日志、P5F/P5FS/P5B/P5A/P5D；本次受版本控制修改范围只读隐私复核 | v0.3.1 最终 1238 passed、完整静态/构建/冻结/旧包审计通过，64 个依赖版本不变，未读取真实忽略数据。完成提交 SHA、最终 diff/隐私复核见 [本次完成报告](../artifacts/validation/v0.3.1/REPORT.md)；未执行 push/tag/release。 |

Phase 5 独立只读复核用纯合成 B=[0,1,2]，同时将 Min–Max 的 scale/output 放大 4 倍并重算结果 hash：原工作区 save/load 接受 [0,2,4]，但导出 verifier 返回 False。问题是工作区共用 snapshot 校验漏查 scale=1/span；原复现见 [真实失败日志](../artifacts/validation/v0.3.1/phase5/minmax-restored-scale-review.log)。随后以父谱 span 核对 scale，新增成对重哈希攻击及相关状态/恢复/导出回归 279 passed（P5M），再完成 v0.3.1 最终全量 1238 passed（P5F）。P5P 的 1230 passed 保留为发现和修复该缺口之前的完整测试证据。

真实浏览器证据分为操作记录 BR5 和实际下载审计 BD5。浏览器记录包含定位超时与后续成功，未隐藏失败重试。审计只读取实际下载的 6 个 ZIP 和 1 个独立宽表 CSV，没有重新生成替代产物；纯完整性校验和恢复的数值入口 spy 为零，显式重放校验只执行 S/N，不执行基线/Prepared/2D。这些 spy 只观测审计本身，不冒称记录了此前浏览器服务器调用。全部产物保留运行当时 exporter_versions=0.3.0；后续工作区恢复后重新导出的 12 节点 ZIP 与恢复前字节相同。

最终 v0.3.1 重新启动浏览器会话后，实际上传恢复同一工作区，预览并确认 A 的 S 窗口 9 和 N_S L2，再下载 S ZIP、N_S ZIP 和工作区 ZIP。BFR5/BFD5 核实实际计算与 exporter 版本为 0.3.1；204 个历史 B 数组未变，A 的 N_B 与无关 B/C 完全不变，两导出包的 4+6 个 CSV 节点均与已确认工作区数组精确相同。两个 verifier 通过，审计恢复和纯校验未调用数值入口；显式重放只计算 S/N，未调用基线、Prepared 或 2D。spy 的观测范围仍只限本次审计。

## 完成记录与执行范围

P5F/P5FS/P5B/P5A/P5D 已完成修复后、版本更新后的全量测试、完整静态、构建、冻结/旧包审计与依赖核对；BFR5/BFD5 完成最终版本实际浏览器计算和下载复核。完成提交、最终差异和隐私证据统一见 [本次完成报告](../artifacts/validation/v0.3.1/REPORT.md)。P0 的 906 passed 和历史 723 passed 均不是本次最终通过数量。

执行范围限制：浏览器实际选择并保存了专家索引空间选项，但所用 A 为均匀轴；没有另传非均匀轴并在浏览器重跑此例外。该输入的数值拒绝/明确 override 契约由 N 对应测试验证。

本文件只依据真实日志更新结果。对受版本控制变更与未忽略新增文件的隐私复核只读取源码、文档和已脱敏合成日志；不读取真实实验忽略目录。合成示例由固定 seed 数学脚本生成，实际 CSV、旧兼容 ZIP/NPZ 与私有原始日志均保留在 Git 忽略目录。

本次文档检查实际核对 66 行场景、AST 测试函数选择器及本地链接，并只读检查自基准以来的变更/未忽略新增文件；未发现待提交 CSV、ZIP、NPY/NPZ、二进制或日志中的实际私有绝对路径。见 [文档与公开差异检查](../artifacts/validation/v0.3.1/phase4/acceptance-document-audit.002.log)。这是该时点的检查，不替代最终提交前复核。

补入真实浏览器与 v0.3.1 最终全量结果后，再次核对 66 行场景、94 个实际 AST 函数选择器、当时 61 个有效链接，以及浏览器事件/下载审计/最终 pytest 中的实际数量，均通过，见 [Phase 5 文档证据复核](../artifacts/validation/v0.3.1/phase5/acceptance-document-browser-audit.log)。
