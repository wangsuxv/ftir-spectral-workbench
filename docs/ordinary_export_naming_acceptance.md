# v0.3.2 普通光谱导出命名验收记录

本次升级让普通光谱的导出文件以原文件名、原曲线列名或用户指定名称为主体，追加**已确认**的处理步骤。UUID、来源、quantity、父级和科学指纹继续完整保留在元数据中。普通基线与后处理共用命名规则，`legacy_id` 保留旧脚本所需的路径和包结构。

最终本地全量为 **1392 passed，5 条既有 QC 绘图警告**；静态检查、冻结检查、真实旧包兼容和安装包预检通过。真实浏览器闭环及下载内容审计通过，范围见下文。GitHub 的准确合并提交、CI 和正式发行产物证据由文末链接的 Release 记录。

## 基准、范围与科学不变

实施起点是干净的 `14a514d7032f5f7c302f4f47d79a72d895251c1d`（v0.3.1），分支 `feat/readable-spectrum-export-names`。实际环境为 macOS arm64、Python 3.12.14；起点状态、64 项依赖锁比较、模块来源见 [environment.json](../artifacts/validation/v0.3.2/phase0/environment.json)。实施前本次实跑 **1238 passed，5 warnings，146.91s**，不是引用历史发布数字。

旧默认路径把显示名、UUID 和状态混在一起；二列文件即使来自不同样品，显示名还可能同为 `Absorbance`。升级将“科学身份”与“对人可读的导出名称”分开：全工作区先分配基础名，再选本次结果；N_S 的后缀使用它实际绑定的 S 父快照；UI 预览与导出共享同一个命名计划。

主要修改范围如下，完整文件清单以最终 Git diff 为准：

| 文件 | 改动 |
| --- | --- |
| `src/ftir_workbench/batch/export_naming.py` | 新增纯命名模块：策略、覆盖、布局判断、Unicode/路径清理、稳定消歧、路径映射与校验 |
| `src/ftir_workbench/batch/export.py` | 基线预览和新名称、schema 1.1、名称/身份/报告/成员校验；保留 legacy 行为 |
| `src/ftir_workbench/batch/postprocessing_export.py` | B/S/N_B/N_S 及父节点命名、新旧版本分流、映射与原严格图结构校验联动 |
| `src/ftir_workbench/batch/workspace.py` | 在既有 `display_preferences.export_naming` 中严格保存/读取偏好；仍为 workspace 2.0 |
| `ui/batch_export_naming.py`、`ui/batch_workflow.py`、`ui/batch_postprocessing.py` | 共享命名控件、正式结果文件名预览、全工作区命名变化使下载缓存失效 |
| 新增命名、兼容、工作区、AppTest、科学冻结测试 | 验证新格式及真实旧格式；原有固定旧路径断言转为显式 legacy 或通过权威映射定位 |
| `scripts/generate_v031_naming_fixtures.py`、`scripts/validate_v032.py`、`scripts/audit_v032_distribution.py` | 精确旧源码夹具、可复现日志、构建和安装包审计 |
| `pyproject.toml`、`src/ftir_workbench/__init__.py`、版本审计测试及用户文档 | workbench 目标版本 0.3.2；旧 baseline/2D 包版本保留 |

`requirements.lock`、旧科学 manifest、34 个冻结文件及冻结路径集合完全未改。额外的 `test_n22_naming_does_not_change_scientific_implementation_or_lock` 将 batch 的 models/importing/fingerprints/recipes/service/state/postprocessing/normalization adapter、后基线 smoothing、旧 baseline/2D/services 源码和 CI 配置逐字节与起点比较；不是仅核对输出近似相等。见 [科学及版本定向回归](../artifacts/validation/v0.3.2/phase4/version-and-science-regression.log) 和 [冻结及旧流程审计](../artifacts/validation/v0.3.2/phase4/freeze-and-legacy.log)。

命名、预览、打包和恢复不运行 pipeline、smoothing 或 normalization，不创建 Prepared。CSV 序列化保留正式数组、方向、范围和 `.17g` 精度；异轴仍拒绝宽表。显式后处理 verifier 的 `recompute=True` 仍可重放 S/N 校验，这是独立接口；不会重新拟合 baseline。历史快照的版本与指纹不被改写，新版本主动计算仍可产生包含新软件版本的指纹。

## 真实旧版本兼容证据

通过 `git archive` 隔离上述精确 v0.3.1 提交后生成合成夹具，归档 SHA-256 为 `505c7837c120501cbaf0c7fbeff01393a7b0233f063bc9a22e65fc0a051a22b4`。不是在新代码下生成一个标成旧版的 ZIP。生成器两次独立执行的 **11 个产物摘要及完整 producer metadata 一致**；工作区有 **176 个逐数组精确比较项**。证据见 [归档信息](../artifacts/validation/v0.3.2/phase0/archive.json)、[首次生成](../artifacts/validation/v0.3.2/phase0/old-fixtures-first.log)、[重复生成](../artifacts/validation/v0.3.2/phase0/old-fixtures-repeat.log)。

兼容测试包含 3 个旧基线包、6 个旧后处理包、旧 workspace 2.0，以及精确 v0.3.0 提交 `9fc13c10ae34bd53cb220d0807edb18c9e2e41dc` 生成的 workspace 1.0。新版验证旧包；新版 `legacy_id` 导出与旧产物的成员集合、CSV 表格值和标签比较，并在隔离的 **v0.3.1 verifier 子进程**中验证，后处理含 replay/no-replay 两种模式。QC 行序允许 JSON 恢复后的合法排序差异，既有环境版本字段记录实际版本，不宣称整个新旧 ZIP 字节必然相同。

新可读包 metadata/index 为 1.1，命名策略版本 1.0；外层 manifest 和后处理节点仍为 1.0。legacy 包为旧 1.0，不增加 `export_name_map.csv` 或新命名字段。未知版本、不一致映射和未引用成员拒绝。旧基线 verifier 保留原有归档完整性能力，不声称具备后处理包同等级的语义重放。

新增命名夹具和真实浏览器验收均使用合成输入；夹具、下载包、原始日志保存在 ignored `outputs/validation-private/v032/`。完整本地旧回归按既有设计运行了两项私有目录测试，会读取可用的本地实验文件；它们不是本次新增的合成测试。仓库只记录生成代码、合成测试和脱敏验收日志，没有提交或上传实验原始数据及私有派生谱。

## N01–N22 验收映射

以下编号对应规格书场景；测试函数名中的历史 `nXX` 不作为场景编号解释。参数化测试数量不能当作独立验收场景数量，也不能把各阶段测试数相加。下列 pytest 选择器均被最终 **1392 passed** 覆盖；补充命令探测单独列证，不计入 pytest 数量。

为避免重复长路径，使用下列可定位别名：

| 别名 | 实际测试文件 |
| --- | --- |
| BN | [tests/batch/test_export_naming.py](../tests/batch/test_export_naming.py) |
| PN | [tests/postprocessing/test_export_naming.py](../tests/postprocessing/test_export_naming.py) |
| BC | [tests/batch/test_export_naming_compatibility.py](../tests/batch/test_export_naming_compatibility.py) |
| NW | [tests/batch/test_export_naming_workspace.py](../tests/batch/test_export_naming_workspace.py) |
| BU | [tests/batch/test_export_naming_ui.py](../tests/batch/test_export_naming_ui.py) |
| PU | [tests/postprocessing/test_export_naming_ui.py](../tests/postprocessing/test_export_naming_ui.py) |
| BE | [tests/batch/test_export.py](../tests/batch/test_export.py) |
| PE | [tests/postprocessing/test_export.py](../tests/postprocessing/test_export.py) |
| SF | [tests/batch/test_naming_science_freeze.py](../tests/batch/test_naming_science_freeze.py) |

| 场景 | 已验证的结果与实际测试选择器 |
| --- | --- |
| N01 二列 G2_90RH / Absorbance | 默认文件主体是原文件名，无完整 UUID。`BN::test_n01_n02_original_filename_and_wide_column_selection`；`PN::test_requested_ns_preview_matches_readable_graph_and_wide_with_exact_parent_arrays` |
| N02 相同通用表头的不同单谱 | 按原文件主名区分，各自保留身份。`BN::test_n01_n02_original_filename_and_wide_column_selection` |
| N03 有表头宽表及列子集 | 采用原列名，子集不重新分配名字。`PN::test_wide_column_subset_and_collision_names_use_whole_workspace`；BN 的 N01 同名函数覆盖只选第 2 列 |
| N04 无表头单谱/宽表 | 明确 parser 选项，宽表以文件名及原始列索引回溯；旧 probe 缺失时使用来源计数证据。`BN::test_n03_headerless_and_old_probe_fallback_uses_original_indices`；[补充实跑](../artifacts/validation/v0.3.2/phase4/naming-matrix-edge-probe.log) 验证无表头二列的原文件主体与 probe |
| N05 展示名/单谱覆盖 | 两入口共用优先级，空覆盖安全回退；API `None` 只对本次请求清除覆盖。`BN::test_n04_display_override_and_explicit_empty_fallback_do_not_rewrite_records`；`PN::test_explicit_null_override_clears_only_this_export_without_restoring_saved_value`；`BU::test_shared_policy_override_download_and_workspace_roundtrip` |
| N06 同名、重复上传、重名列 | 不覆盖；不同 ID 对应不同成员。`BN::test_n06_collision_tokens_and_subset_order_stability`；`BN::test_n07_cross_file_wide_labels_use_readable_file_qualifier`；`BU::test_unselected_collision_import_and_override_invalidate_frozen_download` |
| N07 Unicode 及合法符号 | 保留中文、希腊字母、空格、括号和温湿度符号，JSON 保存原文。`BN::test_n05_safe_unicode_names`；`PN::test_display_override_map_escapes_csv_but_not_original_json` |
| N08 大小写/NFC/清洗/截断碰撞 | 统一碰撞键，稳定 token，选择顺序不改变结果。`BN::test_n06_collision_tokens_and_subset_order_stability` |
| N09 字节预算及路径约束 | 长中文、保留名、路径字符串和空名均有明确处理；最长 NS/MinMax 后缀仍在 200 UTF-8 字节内。`BN::test_n05_safe_unicode_names`；`BN::test_n09_longest_suffix_budget_includes_tokens_and_extension`；BN 的 N01 同名函数覆盖 Windows/POSIX 源路径 |
| N10 强制 token 前缀碰撞 | 8 位不足则扩展到 12 位；完整 token 冲突明确失败。`BN::test_n08_short_token_collision_extends_or_fails_without_overwrite` |
| N11 子集、重排、保存恢复 | 同工作区子集/重排名称不变；偏好、ID、来源和正式快照恢复。`PN::test_wide_column_subset_and_collision_names_use_whole_workspace`；`NW::test_naming_preferences_roundtrip_keeps_science_and_source_exact`；`BU::test_shared_policy_override_download_and_workspace_roundtrip`；[补充实跑](../artifacts/validation/v0.3.2/phase4/naming-matrix-edge-probe.log) 比较 5 条完整命名决策及碰撞 token，恢复前后完全一致 |
| N12 B/S/N_B/N_S 及父级 | 四分支后缀正确；仅请求 N_S 时 B/S 标为 parent，预览等于实际成员。`PN::test_requested_ns_preview_matches_readable_graph_and_wide_with_exact_parent_arrays`；`PU::test_ns_preview_uses_confirmed_methods_parent_roles_and_real_download_arrays` |
| N13 未确认草稿 | 4 种 S 方法及 NS 父方法取正式版本，不受随后草稿修改影响。`PN::test_names_use_confirmed_smoothing_and_ns_parent_not_unconfirmed_drafts` |
| N14 quantity / MinMax / T | 6 种归一化后缀和 quantity 正确；MinMax 为 display_only；派生 T/%T 保留科学标识与负吸光度行为。`PN::test_confirmed_normalization_method_suffix_preserves_quantity`；`BE::test_e09_derived_transmittance_preserves_negative_absorbance_and_provenance`；`BU::test_derived_name_preview_and_new_import_inherit_shared_policy` |
| N15 同轴/异轴宽表 | 同轴列名关联权威映射且逐值相等；长度、方向、范围或坐标不同均拒绝。`PN::test_requested_ns_preview_matches_readable_graph_and_wide_with_exact_parent_arrays`；`PE::test_pp057_only_exact_actual_x_arrays_allow_wide`；`BE::test_e03_actual_different_axes_reject_wide_but_export_zip` |
| N16 缓存失效与旧 bytes | 改策略、覆盖或未选中的碰撞条目使下载失效；已生成 bytes 原样保留。`BU::test_unselected_collision_import_and_override_invalidate_frozen_download`；`PU::test_post_policy_peer_override_and_source_rename_refresh_download_without_compute` |
| N17 真旧包兼容 | 精确 v0.3.1 旧包由新 verifier 接受；新 legacy 仍由旧 verifier 接受。`BC::test_current_verifier_accepts_actual_v031_legacy_bundle`；`BC::test_new_legacy_id_member_paths_arrays_and_old_verifier_remain_compatible` |
| N18 重哈希攻击 | 路径/身份/quantity/父级/节点/版本/CSV映射/额外成员被拒绝；同一 SID 跨分支决策、同源布局证据和来源条目数一致。`PN::test_rehashed_readable_name_mapping_and_member_attacks_fail_closed`；`PN::test_rehashed_same_source_columns_must_share_consistent_naming_evidence`；`BN::test_n18_rehashed_processing_report_must_match_confirmed_export_identity` |
| N19 旧 workspace 1.0/2.0 | 默认 original；ID、来源、历史指纹和数组保持，schema 2.0 的 176 项数组精确比较。`BC::test_actual_v030_schema1_workspace_defaults_naming`；`BC::test_actual_v031_workspace_preserves_every_array_and_defaults_naming` |
| N20 新偏好恢复 | 三种策略、覆盖及显示偏好恢复；错误类型、冲突策略和 NaN/Inf 拒绝，不被修补成合法 null。`NW::test_naming_preferences_roundtrip_keeps_science_and_source_exact`；`NW::test_malformed_naming_namespace_is_rejected_even_with_valid_manifest`；`NW::test_nonfinite_override_is_rejected_before_editor_nan_to_null_repair` |
| N21 禁止额外科学计算 | 命名/预览/导出不调用科学入口，不改变数组、父级、正式对象和指纹。`PN::test_naming_preview_export_and_changed_options_do_not_call_scientific_entrypoints`；安装包另封锁 14 个入口，9 种导出组合为 0 调用 |
| N22 原位/Prepared/2D/smoothing | `SF::test_n22_naming_does_not_change_scientific_implementation_or_lock`、完整 pytest 和冻结审计通过；审计实际包含旧 bundle、Prepared、smoothing、self/cross 2D |

## 本次实际运行与日志

日志中的 `COMMAND` 是完整实跑命令；`EXIT_CODE`、时间和输出均保留。下表只列有判定意义的运行，不将修复前失败替换为通过。公开日志位于 `artifacts/validation/v0.3.2/`，本机原始输出在 ignored `outputs/validation-private/v032/`。

| 检查 | 实际命令/范围 | 结果与日志 |
| --- | --- | --- |
| Phase 0 起点全量 | `.venv/bin/python -m pytest -q -p no:cacheprovider` | 1238 passed / 5 warnings，146.91s；[pytest.log](../artifacts/validation/v0.3.2/phase0/pytest.log) |
| 命名及旧工作区 | `python -m pytest -q -p no:cacheprovider`，命名 workspace/compatibility + 原 workspace_roundtrip/migration 四文件 | 168 passed，17.76s；[workspace-compat-final.log](../artifacts/validation/v0.3.2/phase1/workspace-compat-final.log) |
| 两导出器及严格校验 | `python -m pytest`，两 naming 文件 + 原后处理 export/adversarial + migration 五文件 | 216 passed，9.96s；其中新后处理命名测试 46 例；[postprocessing-naming-review-final.log](../artifacts/validation/v0.3.2/phase2/postprocessing-naming-review-final.log) |
| AppTest 定向闭环 | 两个 naming_ui 文件 + 原 all-branch/MinMax 明确排除策略测试 | 8 passed，14.04s；[naming-final.log](../artifacts/validation/v0.3.2/phase3-ui/naming-final.log) |
| 版本及科学不变 | `.venv/bin/pytest -q -p no:cacheprovider tests/release/test_v025_release_audit.py tests/batch/test_naming_science_freeze.py` | 9 passed，4.25s；[version-and-science-regression.log](../artifacts/validation/v0.3.2/phase4/version-and-science-regression.log) |
| 最终全量 | `.venv/bin/pytest -q -p no:cacheprovider` | **1392 passed / 5 warnings，163.68s**，包含最终同源及 null-override 修复；[full-pytest.002.log](../artifacts/validation/v0.3.2/phase4/full-pytest.002.log) |
| Ruff | `.venv/bin/ruff check src tests ui scripts` | 通过；[ruff.log](../artifacts/validation/v0.3.2/phase4/ruff.log) |
| mypy | `.venv/bin/mypy src/ftir_workbench --no-incremental` | 33 source files 通过；[mypy.log](../artifacts/validation/v0.3.2/phase4/mypy.log) |
| 依赖一致性 | `.venv/bin/python -m pip check` | 无损坏依赖；[pip-check.log](../artifacts/validation/v0.3.2/phase4/pip-check.log) |
| 冻结及旧流程 | `.venv/bin/python scripts/audit_v025_release.py --expected-version 0.3.2` | 34/34 精确字节、完整路径集合、旧流程均通过；[freeze-and-legacy.log](../artifacts/validation/v0.3.2/phase4/freeze-and-legacy.log) |
| wheel/sdist 预检构建 | `.venv/bin/python -m pip wheel . --no-deps --no-build-isolation -w outputs/release-v032/preflight`；`.venv/bin/python -m hatchling build -t sdist -d outputs/release-v032/preflight` | 两种产物构建通过；[wheel](../artifacts/validation/v0.3.2/phase4/wheel-preflight-final.log)、[sdist](../artifacts/validation/v0.3.2/phase4/sdist-preflight-final.log) |
| 安装包审计 | `outputs/release-v032/install-env/bin/python scripts/audit_v032_distribution.py --wheel outputs/release-v032/preflight/ftir_spectral_workbench-0.3.2-py3-none-any.whl --sdist outputs/release-v032/preflight/ftir_spectral_workbench-0.3.2.tar.gz --output artifacts/validation/v0.3.2/phase4/installed_distribution_scope_audit.json` | 通过；[日志](../artifacts/validation/v0.3.2/phase4/installed-distribution-scope.log)、[逐项 JSON](../artifacts/validation/v0.3.2/phase4/installed_distribution_scope_audit.json) |

最终 5 条警告来自旧单谱扩展名 CLI 测试中 `ftir_baseline/qc.py` 的相同 y 轴上下限绘图提示，与起点相同；没有用修改冻结科学文件来消除警告。

功能 [PR #4](https://github.com/wangsuxv/ftir-spectral-workbench/pull/4) 的 [实际 CI](https://github.com/wangsuxv/ftir-spectral-workbench/actions/runs/41223266740) 在 Python 3.11、3.12、3.13 下均为 **1390 passed、2 skipped、5 warnings**，三环境 Ruff、Mypy 和 wheel 构建全部通过。两项 skip 为既有 `test_private_local_directory_loads_when_available` 和 `test_private_local_data_runs_through_all_streamlit_pages`：GitHub 未提供私有本地数据，按原测试规则跳过。没有删改或关闭这些测试；本地 1392 与 CI 1390+2 的范围应分别理解。Phase 0 公开日志首行只是发布说明，现已澄清为“不包含实验光谱数组”；实际命令输出及原始日志摘要保持不变。

安装包审计核对 77 个项目运行模块的 wheel、源码和独立安装字节，34 个冻结文件、MIT 文件、三个 CLI `--help` 和中立目录 `python -I` 导入位置。三种命名策略 × coarse/fine/post-all 共 **9 种组合**验证真实包、预览、正式数组、MinMax 和工作区偏好恢复，封锁 14 个科学入口且记录 **0 调用**。64 项依赖与 Phase 0/锁文件一致。

安装环境将项目 wheel 独立安装在 `install-env`，通过只读 `.pth` 复用原 `.venv` 的锁定科学依赖，**不是全新安装全部依赖的干净环境测试**。这里是提交前 preflight：正式合并后必须重新构建、安装审计并生成校验和，不能把这些预检产物的 SHA-256 当作 Release 产物证据。

## 已保留的失败与关闭原因

| 初始现象 | 原因、处理与关闭证据 |
| --- | --- |
| 无表头小夹具未导入，1 failed / 59 passed | 夹具未指定严格 parser 所需的分隔符/小数点选项；修正测试输入选项，未改变 parser。原日志 [naming-baseline.log](../artifacts/validation/v0.3.2/phase1-2/naming-baseline.log)，复验 [60 passed](../artifacts/validation/v0.3.2/phase1-2/naming-baseline.002.log) |
| 后处理命名首轮 6 failed / 32 passed | 测试使用错误参数名 `reference_range`；改为实际 API `reference_interval`。保留 [首轮日志](../artifacts/validation/v0.3.2/phase2/postprocessing-naming-new-tests-first.log) |
| 1 failed / 116 passed 的布局攻击 | validator 未拒绝宽表布局与二列计数矛盾；补布局/计数约束。随后追加同 SID 多分支决策及同 source 多列一致性、包内 distinct SID 数约束；真实重哈希反例均纳入最终 216 通过。保留 [失败日志](../artifacts/validation/v0.3.2/phase2/postprocessing-naming-full-second.log) 和 [同源回归](../artifacts/validation/v0.3.2/phase2/source-evidence-regression.log) |
| 重哈希报告错误身份仍被接受 | 新基线 verifier 原来只比 CSV/JSON 报告一致，未把报告和正式输出关联；补 requested/exported/stage/来源/指纹/决策及 strict-policy 约束。保留 [接受错误包的实际探测](../artifacts/validation/v0.3.2/phase1/naming-adversarial-probe.log)；[49 项回归通过](../artifacts/validation/v0.3.2/phase1/report-identity-regression.log) |
| 显式 `None` 清除覆盖名失败 | 后处理已解析 options 再次合并保存偏好，导致旧覆盖重新出现；预览/导出改为从原请求分配名字。原始/展示两策略修前 [2 failed](../artifacts/validation/v0.3.2/phase2/null-override-first.log)，最终联合通过 |
| AppTest 首轮及旧 UI 断言失败 | 修复 fine/B 状态映射导致的重复预览排除行；测试读取改为真实 `batch_metadata.json`。旧 MinMax 测试改用权威映射定位谱图，而不是要求新路径含 UUID，保留 quantity/数组断言。保留 [首轮](../artifacts/validation/v0.3.2/phase3-ui/naming-initial.log)、[旧 UI 联合首轮](../artifacts/validation/v0.3.2/phase3-ui/naming-and-existing-ui.log)；定向 8 例及最终全量关闭 |
| Editable 安装首次失败 | `--no-build-isolation` 环境缺 `editables`；使用标准隔离构建完成 editable metadata 更新，没有升级科学依赖。保留 [失败](../artifacts/validation/v0.3.2/phase4/editable-install.log)、[成功](../artifacts/validation/v0.3.2/phase4/editable-install-isolated.log) |
| 首次升级后全量 1383 passed / 2 failed | 两个 release 测试仍显式期待 0.3.1；只更新应用版本目标断言，审计 API 的历史默认仍为 0.2.5。保留 [首次全量](../artifacts/validation/v0.3.2/phase4/full-pytest.log)，最终 **1392 passed** 关闭；不把首次运行写成全量通过 |

## 真实浏览器闭环

在稳定服务 `8767` 上实际上传两个合成 CSV，分别确认粗调及明确跳过细调，切回 A 确认 S/N_S，下载默认及覆盖命名包，保存工作区并在新标签页上传恢复，最后下载两谱粗调 ZIP 和同轴宽表。共审计 **7 个真实 ZIP 和 1 个独立宽表 CSV**；每个导出数组与工作区正式 float64 数组逐元素一致，来源、稳定 ID、草稿、指纹、父级和命名偏好恢复一致。覆盖命名 N_S 包在恢复前后 SHA-256 完全相同。

详细操作、路径、截图与 trace 位置、失败尝试及边界见 [真实浏览器验收](ordinary_export_naming_browser_acceptance.md)、[机器可读记录](../artifacts/validation/v0.3.2/phase4/browser-summary.json)及[下载内容审计日志](../artifacts/validation/v0.3.2/phase4/browser-artifact-audit-recorded.log)。开发服务器 `8766` 的热重载失配未计为通过；稳定重启后重新执行成功。实际浏览器未另执行 display/legacy_id 策略、N_B 计算或异轴宽表禁用；这些仅按对应自动化测试记录，不冒充浏览器结果。

## 发布审计与准确提交

本文件保存本地验收及发行前预检证据。发布使用原仓库功能分支 → 面向 main 的 PR → Python 3.11/3.12/3.13 实际 CI → 正常合并 → 精确提交的 tag/Release；不强推、不覆盖已有 tag、不绕过检查。

最终 PR、三版本 CI、main 合并完整 SHA、tag 指向及正式 wheel/sdist/SHA256SUMS 的证据统一附在 [v0.3.2 Release](https://github.com/wangsuxv/ftir-spectral-workbench/releases/tag/v0.3.2) 的说明中；可从 [tag 对应源码](https://github.com/wangsuxv/ftir-spectral-workbench/tree/v0.3.2) 核对实际发布树。本地完整发布记录位于 ignored `outputs/release-v032/PUBLISH_REPORT.md`。只有合并后的准确提交才用于正式构建，随后重新进行安装/冻结审计并下载 GitHub 附件核对 SHA256SUMS；本报告中的 preflight 摘要不能替代正式发行产物摘要。

提交前日志里的 `audited_head_commit=14a514d…` 表示当时 HEAD 加上所检查工作树，不能当作完成 SHA。发行提交的精确身份可在同步 tags 后核验：

```bash
git rev-parse 'v0.3.2^{}'
git show --no-patch --format=fuller v0.3.2
```

使用与本地更新见 [ordinary_export_naming.md](ordinary_export_naming.md)。GitHub 发布不会自动更新已经运行的本地进程，升级前需保存工作区并停止旧服务。
