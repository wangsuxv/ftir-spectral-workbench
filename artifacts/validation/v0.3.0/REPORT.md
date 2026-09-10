# v0.3.0 独立批量基线实施验收

2026-09-11，本地实现完成。最终全量 **906 passed，5 个既有 warning，82.47 秒**；Ruff、Mypy、构建/安装、冻结及旧 bundle/smoothing/2D 审计通过。真实浏览器完成三文件、五条 A/%T/T 异轴光谱的上传、逐谱粗调与细调、两阶段 ZIP 下载、新会话保存/恢复闭环，下载数组与正式快照精确一致。

## 基准与提交

- 实际仓库：`ftir-spectral-workbench` 子目录，起始 HEAD `8cf6cc8bd38491d6549595564f2c57715bc69ef0`，起始分支 `feat/v0.2.5-post-baseline-smoothing`，工作区干净。
- 规格书所述 `c0751bc2fa0e3a3716c70bc9da33766fcad2b746` 不在本地 Git 对象库。未 fetch/reset，不声称已对该远程提交作逐文件比较。
- 实施分支：`feat/v0.3-independent-batch-baseline`。
- 完成实现提交：`a42c7424c46fecaa4fa4e03d8837a05654c3e91e`。后续仅提交此报告与最终验证证据；交付 HEAD 在最终回复给出，也可执行 `git rev-parse HEAD` 查看。
- 阶段提交：Phase 0 `82f2bf2`，Phase 1 `c028f72`，Phase 2 `4eb45fa`，Phase 3 `1201e2f`，Phase 4 `9570400`，Phase 5 实现 `a42c742`。
- 未推送、创建 PR、打 tag 或发布 Release。未读取/提交实验原始数据或私有派生光谱；只有明确生成的三份合成示例加入 Git。

## 实现与修改文件

[完整逐文件列表](changed_files.txt) 为实际基准至实现提交的 Git diff；本目录随后增加的是验证证据。

- `src/ftir_workbench/batch/`：独立模型、逐文件/列导入、配方规范化、来源/阶段 fingerprint、单谱 service、状态机、分阶段 ZIP/宽表导出、工作区安全保存恢复。
- `ui/batch_workflow.py`：五页普通模式、每谱参数与草稿、预览/确认、确认后下一条、复制参数、搜索/筛选/选择、粗细调导出及工作区恢复。
- `ui/streamlit_app.py`：导入前模式分流；模式切换时保留原位业务状态及上传来源。原位页面、baseline/Prepared/smoothing/2D 流程保留。
- `tests/batch/`：导入、独立性、全部公开粗细调方法的核心等价、estimate-only smoothing 开关、父级 stale、草稿/复制/缓存、真实 AppTest 上传下载恢复、恶意归档与业务结构校验。
- `pyproject.toml`、`src/ftir_workbench/__init__.py`：验收后版本统一为 0.3.0；Mypy 增加本地 `src` 查找路径。数值依赖未变。
- `scripts/audit_v025_release.py` 及其测试：保留默认严格 0.2.5 的历史检查，增加显式 `--expected-version 0.3.0`，不放松旧科学断言。
- `scripts/validate_v030.py`、性能与真实下载审计脚本；README/CHANGELOG/PROJECT_STATUS、普通模式说明和可重现合成示例。

每个普通条目保留自身轴、单位、完整域默认范围、原始列哈希、独立准备/粗细调草稿、预览及正式快照。数值只调用旧 `run_pipeline`，每次单谱 `SpectrumSet`，显式 `independent_locked`、`normalization=none`；fine 从原始输入与确认 coarse 配方重新执行并核对 coarse 父级。占位 perturbation 仅为内部适配。普通模式不创建 Prepared、运行 2D 或输出 `for_2dcos`。

## 真实测试记录

以下是各阶段独立命令的结果，不能相加。命令工作目录为仓库；`scripts/validate_v030.py` 保存命令、环境、stdout/stderr、退出码和耗时。提交日志仅脱敏本机路径；原始日志位于 ignored `outputs/validation-private/`。

| 阶段 | 真实结果 | 日志 |
|---|---|---|
| 0 | 起始旧 pytest 723 passed；Ruff pass；默认 Mypy 首次失败，指定 MYPYPATH 后 20 文件通过 | [起点说明](phase0/README.md)、[pytest](phase0/pytest.log)、[Mypy 初始](phase0/mypy.log) |
| 1 | 新 importer 与旧导入回归 139 passed | [导入测试](phase1/importing_and_regression.log) |
| 2 | 独立核心及方法/通道等价联合 74 passed | [独立核心](phase2/independent_core.log)、[方法等价](phase2/method_parity.log) |
| 3 | 普通及旧 UI 48 passed | [UI 回归](phase3/ui_and_legacy.log) |
| 4 | batch 与旧 UI 联合 177 passed；随后加强父级验证的 core/export 联合 115 passed | [联合](phase4/batch_and_ui.log)、[加强验证](phase4/final_export_workspace.log) |
| 5 | 版本更新前 865 passed；版本契约更新后 867 passed；最终工作区结构修复后 **906 passed** | [最终全量](phase5/pytest_after_schema_fix.log) |
| 5 | 最终 Ruff pass；默认 Mypy 29 source files pass | [Ruff](phase5/ruff_after_schema_fix.log)、[Mypy](phase5/mypy_after_schema_fix.log) |
| 5 | 损坏工作区结构/重复数组引用预算新增 39 例，相关导出/恢复/UI 联合 96 passed | [结构校验](phase5/workspace-schema-tests.log) |
| 5 | sdist/wheel 0.3.0 构建成功，wheel 无依赖重装成功 | [构建](phase5/build_delivery.log)、[安装](phase5/wheel_install_delivery.log) |

最终核心命令为 `.venv/bin/python -m pytest -q`、`.venv/bin/python -m ruff check src tests ui scripts`、`env -u MYPYPATH .venv/bin/python -m mypy`。最终 5 个 warning 均为旧单谱导入 CLI 图中相同 y 范围的 Matplotlib 警告，与起点相同。

首次失败记录未删除：静态格式、测试调用误用 fine-disabled、UI 状态切换、归档验证及下载审计脚本自身的比较方式均保留失败与后续修复日志。editable 安装因环境缺少 `editables` 失败，改用已经成功构建的 wheel，使用 `--no-deps` 安装，未新增数值或构建依赖。

## 关键验收证据

- A 单独运行与批次运行数组精确一致；改变/删除 B 或重排不改变 A 数组与 fingerprint；同列缓存、同值不同来源身份、相互独立草稿/正式状态均有服务和 UI 回归。
- A→B→A、切页、搜索过滤恢复草稿/正式结果；单纯查看、参数复制、导出、恢复不拟合。AppTest 使用本机 Streamlit 1.62 实际 FileUploader，以及真实媒体存储和下载控件，未虚构测试 API。
- coarse 后 fine 从同一 raw 完整重跑；estimate-only smoothing 开关及全部既有方法与权威核心逐数组比较。coarse 更新仅使同谱 fine stale；stale fine 拒绝导出。未处理、失败、排除、明确跳过均在报告中区分。
- 默认按各谱原始轴输出 CSV + ZIP；同轴宽表由实际 x 完全相等的服务与 AppTest 用例验证，长度/方向/范围/坐标任一不同均禁止。负 A 与派生 >100%T 保留；普通模式时间连续性为 N/A。
- 工作区包含原始文件、稳定 ID、来源列、完整原始轴、草稿、正式数组及父级，恢复不拟合；hash、结构、非有限原始数组、非法 NPY、路径、成员大小及解码预算均严格验证。

浏览器实际操作与重试见 [会话记录](phase5/browser_session.json)，实际下载内容检查见 [机器报告](phase5/browser_download_audit.json)、[最终代码审计日志](phase5/browser_download_audit_after_schema_fix.log)。截图：[导出](phase5/browser-export.png)、[恢复后](phase5/browser-restored.png)。浏览器上传三份示例后形成五条 291/401/401/401/581 点光谱，确认五条 coarse、四条 fine 及一条明确跳过；所有阶段实际下载。刷新为空会话后上传工作区恢复，再次导出。最终结构修复后重启服务重复恢复/下载，下载字节与之前恢复后的包一致，界面异常数为 0。

恢复前后光谱/分量 CSV、配方、metadata 与处理报告逐字节一致；QC 指标值也一致，但 JSON 映射排序使 QC 行序变化，导致 QC 文件和对应 manifest 哈希不同，因此不宣称整个恢复前后 ZIP 字节相同。浏览器实际操作 Plotly 缩放/复位；未注入浏览器计算计数器，禁止重算由 AppTest/服务和下载审计独立验证。

## 冻结、兼容与环境

[最终审计](phase5/release_audit_final.json) 对完成实现提交执行：34/34 冻结文件按原始字节 size/SHA-256 通过；冻结根集合无新增、缺失、Git diff。manifest 与本次起点完全相同，未改写 manifest。旧 baseline/2D/project bundle 从历史提交真实生成再由当前代码验签及精确恢复；旧 smoothing 分支与 unsmoothed/smoothed 2D/Cross 全部审计通过。

[环境核对](phase5/final_environment.json)：原环境 Python 3.12.14，64 项锁定依赖全部保持初始版本，安装 distribution 为 0.3.0。旧 `ftir_baseline` 仍 0.1.0，`ftir2dcos` 仍 0.4.0。[分发隐私检查](phase5/distribution_privacy.json) 验证 wheel 包含最终 workspace 源码，sdist 原始数据目录只有 README，wheel 无原始数据、无私有输出。

[性能报告](phase5/performance.json)：固定 seed 3030、100×4000 点合成光谱，导入 2.662 秒；逐谱 arPLS 预览并确认 1.072 秒，单条中位 0.0103 秒；重复预览 0.0276 秒且新增 pipeline 调用 0。整个过程 4.020 秒，进程峰值 RSS 300.48 MiB（含依赖、导入及进程已有内存，并非工作区增量）。A-alone 全数组精确相等、最大误差 0；修改/删除 B、重排后 A fingerprint 不变。此为本机实测，未承诺固定响应时间。

## 运行与限制

在仓库目录执行：

```bash
PYTHONPATH=src:. .venv/bin/python -m streamlit run ui/streamlit_app.py
```

导入前选“普通光谱模式”。示例及生成方式见 `examples/independent_batch/README.md`；完整说明见 `docs/independent_batch_baseline.md`。本次验收服务监听本机 `127.0.0.1:8503`。

范围内功能已完成。只支持现有公开文本 parser；没有 Excel、厂商二进制或普通批次新 CLI。每谱仅一个连续处理范围，不自动对齐、不插值；异轴批次使用逐谱 ZIP。普通模式只支持 baseline estimate-only smoothing，原位 v0.2.5 post-baseline smoothing 仍走原流程。

预览不写入工作区，恢复正式数组并保留旧实现警告，不声称重算；升级代码应先保存、停止/重启进程、再恢复，热重载不是迁移。工作区 ZIP 限额为 256 MiB 压缩、512 MiB 解压/累计数组解码、64 MiB 单成员、20,000 成员。未执行其他操作系统/Python 版本矩阵、外部仪器实测或远程 CI；所有科学/性能/UI 数据均为合成数据。本次没有发布动作。
