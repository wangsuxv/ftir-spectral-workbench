# Changelog

本文件记录 FTIR Spectral Workbench 的用户可见变化。v0.3.1 为普通光谱增加独立平滑与归一化；下方旧版本的发布状态和测试数字均保留为历史记录。

## [0.3.1] - 2026-09-11

本地功能、真实浏览器导出/恢复验收和统一 workbench v0.3.1 版本更新完成，尚未对外发布。

### Added

- 普通模式新增默认关闭的逐谱后处理：B 为已确认最终基线，S=Smooth(B)，N_B=Normalize(B)，N_S=Normalize(S)，分支互不覆盖。
- 公开数组平滑门面复用原 Savitzky–Golay、Gaussian、Moving Average、Median helpers；归一化适配器复用冻结 core 的最大正峰、窗口峰高/面积、全域/指定域面积、L2 和 Min–Max。
- 普通后处理页提供独立草稿、预览、明确确认、来源选择、移除分量、QC、参数复制与逐项批量操作。切谱/切页保留状态，确认 S 不自动选择其他来源。
- 独立 `ordinary_spectrum_postprocess_bundle` 导出与专用 verifier；逐谱 CSV 使用 17 位有效数字，包含完整/有效配方、source/parent/hash、必要父级数组、scale/offset、参考区间、QC、警告和真实版本信息。
- 工作区内层 schema 2.0 保存新增草稿、确认版和历史父级，兼容读取 schema 1.0；批量勾选集和来源选择可恢复，预览不保存。
- [普通后处理说明](docs/ordinary_postprocessing.md)、[可再生纯合成输入](examples/ordinary_postprocessing/README.md) 和对应数值、状态、UI、导出与迁移回归。

### Scientific and state boundaries

- 后处理只从有效 B 或 S 读取完整数组，不重新运行 baseline、不造 Prepared、不调用 2D；旧基线、原位 smoothing/Prepared/2D、CLI 和旧 bundle 合同保留。
- 未激活或禁用参数保留在完整草稿中，但不改变有效科学 hash。每个目标独立验证实际轴、点数及参考区间；复制、确认、导出和恢复不执行滤波或归一化。
- 草稿变化不覆盖正式快照。新 S 只使旧 N_S 过期；新 B 使同谱后处理过期，其他谱不受影响。禁止 S→S、N→S、N→N，缺失 S 时不退回 B。
- 面积使用真实 x 与显式 absolute/signed 定义；区间完整包含且至少有两个实际点，零/近零或不稳定参考拒绝，不插值、不加 epsilon 继续运算。Min–Max 使用 core 的 `view_data` 与正确 offset，仅作 0–1 显示缩放。
- 归一化量不转换为物理 T/%T。平滑移除分量不宣称真实噪声或自动 SNR 改善，不进行普通样品时间连续性或跨样品稳定性评分。
- 导出只序列化已确认数组，默认缺失或过期即阻止；“全部分支”同样严格，只有显式“只导出有效项”才排除并报告。宽表要求实际输出 x 完全一致。
- verifier 可重放 S/N 核对父子结果，B 仅做锚点完整性校验；SHA-256 不是数字签名，历史 B 未记录的依赖版本不以当前导出环境冒充。
- 冻结 manifest 与根文件集合不改写，旧平滑模块只追加数组门面，不升级数值依赖或增加算法。

### Validation status

- 实际基准为 `9fc13c10ae34bd53cb220d0807edb18c9e2e41dc`；完成提交 SHA 与完整证据见 [交付报告](artifacts/validation/v0.3.1/REPORT.md)。
- Phase 0 实际起点 906 passed；Phase 1 数值及旧回归 206 passed；Phase 2 数值 218、状态 66、旧 batch 181 passed；Phase 3 实际 AppTest 48 passed；Phase 4 导出 96、工作区 39、UI 54 passed。各命令覆盖重叠，不能相加为验收总数。
- 版本更新前全量 1230 passed；最终 v0.3.1 实际全量 **1238 passed，5 个既有 warning，132.47 秒**。Ruff、配置范围内 Mypy（32 个源码文件）、sdist/wheel 构建和当前环境 wheel `--no-deps` 安装通过。最终冻结 34/34、旧 bundle/project/Prepared smoothing/self/cross 2D 及包版本审计通过。真实命令见 [Phase 5 日志](artifacts/validation/v0.3.1/phase5/)。
- 真实浏览器完成三条合成异轴谱的 12 分支下载、201 行/8 结果列同轴宽表、stale 阻断与明确排除、新会话工作区恢复；660 个数组、70,860 个元素与来源/草稿/父级精确核对。原始下载发生在版本号统一前，metadata 的实际 0.3.0 保留；30 步原始响应包括失败定位尝试，详见 [浏览器验收记录](docs/ordinary_postprocessing_browser_acceptance.md)。
- 最终 v0.3.1 服务重启后，再次通过真实浏览器恢复、确认新 S/N_S、下载与保存；新快照计算版本及两个包的导出版本均为 0.3.1。204 个基线数组、A 的 N_B 和无关 B/C 不变，4/6 个导出 CSV 节点精确匹配确认数组；64 项依赖版本与起点全部一致。
- 首次无隔离 editable 安装因环境缺 `editables` 失败，日志保留；随后本地 wheel 安装成功，未升级依赖，不冒充全新隔离环境测试。
- 初始失败与修复后日志保留在 `artifacts/validation/v0.3.1/`。本次不自动推送、创建 PR、打 tag 或发布 Release；新增示例与测试只使用合成数据。

## [0.3.0] - 2026-09-11

### Added

- 导入前选择原位序列或普通光谱模式，两种工作区在同一 session 中隔离。
- 普通模式逐文件使用公开文本 reader/probe，再将宽表每列拆成独立条目；支持异轴、混合 A/%T/T、同名文件及逐文件失败隔离。
- 每谱稳定身份、完整域默认范围、准备/粗调/细调草稿、有限预览缓存、正式快照、显式 fine 跳过与 stale 父级状态。
- 五页普通模式 UI，包含切谱草稿恢复、搜索/状态筛选、逐谱预览/确认、独立参数复制和按各自配方批量预览。
- 每谱 CSV ZIP、分量/recipe/QC/排除报告、实际 x 完全一致时的可选宽表及明确标注的派生 T/%T。
- 独立 workspace ZIP 保存/恢复，包含原始来源、稳定 ID、草稿、正式数组和父级关系；严格 JSON、非 pickle NPY、SHA-256 和 ZIP 资源限额验证。
- [普通模式文档](docs/independent_batch_baseline.md)、[可重现纯合成示例](examples/independent_batch/) 与 100×4000 点性能基准脚本。

### Scientific and state boundaries

- 新逻辑位于 `ftir_workbench.batch` 和独立普通 UI；保留旧 manifest 与冻结根文件集合，不改变既有科学默认值或数值依赖。
- 每次仅以单谱 `SpectrumSet` 调用 `run_pipeline`，显式固定 `independent_locked`、无归一化；普通模式不创建 Prepared、不调用 2D、不生成 `for_2dcos` 文件。
- 粗调使用规范的 fine-disabled 配置；细调从相同原始输入和已确认 coarse 配方重跑，核对粗调父级并保持 estimate-only smoothing 的真实拟合通道。
- 草稿不覆盖正式结果；改变 A 的 coarse 只使 A 的 fine 过期。导出不运行 pipeline，不自动确认草稿，也不把尚未决定 fine 当作明确跳过。
- 不插值或自动对齐异轴，不静默裁剪负 A/派生 >100%T，不给普通批次做跨样品时间连续性评分。
- 旧原位 baseline/Prepared/project/bundle、Post-Baseline Smoothing 和 2D 功能保留；没有新增普通批次 CLI 命令，也未改造桌面 GUI。
- 实现 fingerprint 在进程内缓存。更新代码应先保存工作区、停止并重启 Streamlit、再恢复；热重载不替代迁移。旧实现恢复结果会注明未由当前 pipeline 重算。

### Validation status

- 本次实际阶段结果：Phase 0 旧测试 723 passed，Phase 1 导入及 importer 回归 139 passed，Phase 2 独立核心 74 passed，Phase 3 普通及旧 UI 48 passed，Phase 4 batch/UI 联合 177 passed。
- 以上为各阶段独立命令结果，不作最终全量测试总数。初始失败与修复后的日志见 `artifacts/validation/v0.3.0/phase0/` 至 `phase4/`。
- 最终全量 906 passed（5 个既有 warning），Ruff/Mypy、sdist/wheel 构建、64 项依赖不变、冻结 34/34、旧 bundle/smoothing/2D 审计通过；真实浏览器完成五条合成混合单位异轴谱的导出/恢复闭环。实际日志与局限见 `artifacts/validation/v0.3.0/REPORT.md` 和 `phase5/`。
- 本次任务未自动推送、创建 PR、打 tag 或发布 Release；示例全为合成数据。

## [0.2.5] - 2026-09-01

### Added

- 默认关闭的 Post-Baseline Smoothing 科学分支，包含 Savitzky–Golay、Gaussian、Moving Average 和 Median / Despike 四种方法。
- `PostBaselineSmoothingConfig`、不可变 `PostBaselineSmoothingResult`、统一计算入口、科学 fingerprint、QC 与显式 `PostBaselineSmoothingService`。
- 从 primary unsmoothed Prepared 创建 child Prepared 的完整 lineage：保留 baseline run/fingerprint，重新计算 Prepared SHA-256，并记录 parent hash、方法、有效参数、QC 与 warning。
- Streamlit 第 8 页的 uniform-axis diagnostics、实际谱/first/middle/last/mean/median Preview、overlay、removed component、QC、Apply 和 primary/smoothed 2D 分支切换。
- 可确定生成、严格验签和精确重载的 `post_baseline_smoothing_run.zip`；bundle 同时保存 parent/child Prepared、removed component、配置、QC 和图。
- `ftir-workbench smooth`，可接受 baseline ZIP 或 Prepared CSV + sidecar；既有 `twodcos` 可直接读取 smoothing bundle。
- [`docs/post_baseline_smoothing.md`](docs/post_baseline_smoothing.md) 与无实验数据的 `examples/smoothing/` 合成示例。

### Scientific boundaries

- Primary unsmoothed Prepared 仍是默认数据；Preview 不提交，Apply 不自动激活 smoothed branch。
- 所有 smoothing 只沿 `axis=1` 波数轴，对全部光谱使用同一参数；不沿扰动轴处理，不插值、不重排、不自动重采样，也不覆盖 `PipelineResult.analysis_data`。
- 非均匀轴默认拒绝；显式 index-space override 会写入 warning 和 provenance。
- 禁止 chained smoothing 和 smoothing + scientific normalization 组合；2D 阶段只消费 active Prepared，本身不调用 smoothing。
- baseline bundle 不变；smoothed 2D bundle 使用既有结构并嵌入实际 child Prepared。
- v0.2.1 的输入、baseline、2D-COS、Cross、peak-order 与 frozen workbench service 文件保持逐文件 size/SHA-256 不变。

### Release status

- 本版本已在本地按六个 Phase 分别提交；尚未由本次任务推送、打 tag 或创建 GitHub Release。
- 本地验收：723 passed、5 个既有小型单谱绘图 warning；Ruff、Mypy、0.2.5 sdist/wheel build 与全新 wheel 虚拟环境（复用已验证依赖层）import/CLI 链路通过。
- Science freeze 34/34，冻结根的文件集合与 Git diff 均精确匹配；精确 v0.2.1 起始快照生成的 baseline/2D/project bundle 通过当前 verifier 与 Prepared exact reload。
- Smoothing bundle roundtrip、unsmoothed/smoothed self + cross 2D、Cross reverse identities 和 smoothed 2D source Prepared exact reload 全部通过。
- 最终测试、构建、wheel 安装、CLI、science freeze、旧 bundle/project 与 smoothing/2D 集成的真实输出保存在 `artifacts/validation/v0.2.5/final/`。

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
