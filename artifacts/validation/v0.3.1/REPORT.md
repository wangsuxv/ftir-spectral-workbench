# v0.3.1 普通光谱后处理：完成报告

日期：2026-09-11。本轮在已存在的 v0.3.0 普通工作区上完成增量升级，新增 B→S→N_S 与 B→N_B；原 B、旧普通基线导出、原位/Prepared/2D 流程保留。**最终实跑 1238 passed，5 个既有 warning；Ruff、mypy、构建、冻结与真实浏览器文件验收通过。**

## 实际基准与提交

- 实际本地基准：`9fc13c10ae34bd53cb220d0807edb18c9e2e41dc`，v0.3.0，开始时工作树干净。
- 开发分支：`feat/v0.3.1-ordinary-postprocessing`。
- Phase 0 映射/基准：`a1069ed52741b7da66f0d7c5581f0a123b08f7d7`。
- Phase 1–2 数值/业务状态：`1485405be29eadc07b42b926964cca19bea3a8d8`。
- Phase 3–4 UI/导出/迁移：`c5072946508e5c0f6f9a71da803a99221bada743`。
- **完成实现、恢复完整性修复及版本统一的代码提交：`b76d5dd579c6450613142967efbac0f769df2b27`。** 最终全量测试及最终浏览器运行代码与此提交相同；随后提交的是文档、验证工具和真实证据，不再改产品数值实现。
- 最终 workbench 源码、pyproject、安装发行包统一为 **0.3.1**。本地未发现已占用的同名版本 tag；未查询或假定远程版本。旧 `c0751bc...` 仅是规格历史证据，未 reset、切回或覆盖本地升级。
- 没有 push、PR、tag 或 Release；实验谱及派生数据没有加入 Git。

## 已实现的数据流与用户操作

```text
当前有效、已确认最终基线 B
├─ S = Smooth(B)
│  └─ N_S = Normalize(S)
└─ N_B = Normalize(B)
```

B 只来自原普通模式已确认 fine，或明确“确认不做细调”的 coarse 最终决策；未处理 fine 不构成 B。新增步骤默认关闭，默认归一化来源和导出均为 B。确认 S 不自动切换来源，不修改 B，也不能对 S/N 再平滑或对 N 再归一化。

平滑通过新增数组公共门面复用原四种 helpers；SG 7/2/interp、Gaussian 1/4、Moving Average 3、Median 3 的预览默认值保持。普通谱以 (1,n) 沿波数轴处理；完整父域与方向保留。非均匀轴默认拒绝，显式 index-space override 才继续并记录其含义；不插值、重采样或自动改窗口。界面显示 B、S、B−S“移除分量”、实际采样间距和窗口跨度及既有 QC。

归一化调用冻结公开 `apply_normalization`：峰高/面积/L2 读取 `optional_normalized`，Min–Max 读取 `view_data`，保持 `analysis_data` 原语义。支持 none、全范围最大正峰高、窗口正峰高、窗口/指定域/全域面积、L2、固定 0–1 显示缩放。区间须完整覆盖且至少两个实际点；正峰参考、真实 x 积分、正且有限 scale、近零/非有限保护与 scale/offset 均记录。输出仍为父谱全域；负值不裁剪，N 不作为吸光度或物理 T/%T。

每谱按已有 stable spectrum_id 保存三分支的独立 draft/preview/committed。确认检查 id/source/config/parent，重复有效确认幂等；无效非激活参数不改变科学 hash。draft 修改保留有效 committed；新 S 仅使本谱 N_S stale，新 B 使本谱全部后处理 stale。复制只写目标草稿，目标各自校验；批量预览与批量确认分开，表格报告成功、失败和跳过。切谱、缩放、显示单位变化不运行计算。

新导出明确选择 B/S/N_B/N_S/全部，默认严格要求所选分支有效；只有显式 valid-only 才排除不可用项，并报告原因。只序列化不可变确认数组，不计算基线、平滑或归一化。ZIP 含逐谱 17 位有效数字 CSV、稳定 ID、必要 B/S 父数据、完整/有效配方、来源/父 hash、scale/offset/参考区间、QC、warnings、软件版本和 manifest。实际 x 逐元素相等才提供宽表。artifact 为 `ordinary_spectrum_postprocess_bundle`、`is_2d_ready=false`，无 `for_2dcos` 文件。

## 修改范围与入口

| 职责 | 实际文件 |
|---|---|
| 原平滑 helpers 的追加公共数组门面 | [post_baseline_smoothing.py](../../../src/ftir_workbench/post_baseline_smoothing.py) |
| 普通归一化数值/区间保护 | [normalization_adapter.py](../../../src/ftir_workbench/batch/normalization_adapter.py) |
| 三分支草稿/预览/确认/失效/复制/批处理 | [postprocessing.py](../../../src/ftir_workbench/batch/postprocessing.py) |
| 普通 artifact、严格导出及专用 verifier | [postprocessing_export.py](../../../src/ftir_workbench/batch/postprocessing_export.py) |
| 复用工作区，schema/选择集扩展 | [models.py](../../../src/ftir_workbench/batch/models.py)、[workspace.py](../../../src/ftir_workbench/batch/workspace.py) |
| 普通后处理界面与原普通导航接入 | [batch_postprocessing.py](../../../ui/batch_postprocessing.py)、[batch_workflow.py](../../../ui/batch_workflow.py) |
| workbench 版本及当前版本审计断言 | [pyproject.toml](../../../pyproject.toml)、[workbench 包信息](../../../src/ftir_workbench/__init__.py)、[release 测试](../../../tests/release/test_v025_release_audit.py) |
| 新数值/状态/导出/迁移/UI 测试 | [tests/postprocessing](../../../tests/postprocessing/)，原 [batch UI 测试](../../../tests/batch/test_ui.py) 仅适应插入新导航页 |
| 文档、合成输入与验证工具 | [代码映射](../../../docs/ordinary_postprocessing_code_map.md)、[用户指南](../../../docs/ordinary_postprocessing.md)、[66 场景矩阵](../../../docs/ordinary_postprocessing_acceptance.md)、[浏览器记录](../../../docs/ordinary_postprocessing_browser_acceptance.md)、[合成示例](../../../examples/ordinary_postprocessing/README.md)、[scripts](../../../scripts/) |

全部相对基准的公开路径清单与隐私范围见 [repository_audit.json](phase5/repository_audit.json)。没有改旧 ordinary importer/service/state/export 或统一 Streamlit 入口；没有新增第二个普通工作区框架。

## 实际执行与结果

以下测试批次有重叠，**不能相加**。命令通过 `scripts/validate_v031.py <phase> <name> -- <command>` 保留 UTC、命令、退出码、耗时与真实 stdout/stderr；失败日志保留，重复运行使用新文件名。

| 阶段/命令 | 实际结果 | 日志 |
|---|---|---|
| Phase 0：`.venv/bin/python -m pytest -q` | 906 passed，5 warnings，92.28s；本次重新执行的起点 | [pytest](phase0/pytest.log) |
| Phase 0：Ruff / mypy / freeze | Ruff pass，mypy 29 files pass，34/34 freeze | [阶段起点](phase0/README.md) |
| Phase 1：新增数值 + 原 smoothing/normalization | 206 passed；后增加 request guards 后同组 218 passed | [206](phase1/numerical-and-existing-regression.log)、[218](phase2/request-validation-numerical-regression.log) |
| Phase 2：状态/quantity + 旧普通回归 | 66 passed；旧普通 181 passed | [状态](phase2/state-contracts-final.log)、[旧普通](phase2/old-batch-regression.log) |
| Phase 3：新旧 AppTest | 48 passed | [UI gate](phase3/ui-gate-final.log) |
| Phase 4：迁移/来源，旧 workspace | 39 passed；旧 workspace 73 passed | [迁移](phase4/workspace-migration-final.log)、[旧恢复](phase4/workspace-existing-first.log) |
| Phase 4：新旧导出 + 攻击测试 | 96 passed | [导出](phase4/export-tests-final.log) |
| Phase 4：新旧 UI，最终全部分支策略 | 54 passed；策略修补后相关 6 passed | [UI](phase4/ui-gate.log)、[显式 valid-only](phase4/ui-all-explicit-policy.log) |
| Phase 5：Min–Max 恢复完整性修复回归 | 279 passed，含新增 8 个 scale/output 成对重哈希攻击 case | [修复回归](phase5/minmax-integrity-regression.log) |
| **最终 `.venv/bin/python -m pytest -q`** | **1238 passed，5 warnings，132.47s** | [最终全量](phase5/pytest-final.log) |
| `.venv/bin/python -m ruff check src tests ui scripts` | pass | [Ruff](phase5/ruff-final.log) |
| `.venv/bin/python -m mypy` | 32 source files pass；原 unused-override note | [mypy](phase5/mypy-final.log) |
| `.venv/bin/python -m build --no-isolation` | wheel + sdist 成功 | [build](phase5/build-final.log) |
| `pip install --no-deps dist/ftir_spectral_workbench-0.3.1-py3-none-any.whl` | 当前环境安装 0.3.1 成功 | [安装](phase5/install-wheel-v031.log) |
| `scripts/audit_v025_release.py --expected-version 0.3.1` | 冻结、旧包、原位 smoothing、self/cross 2D、版本通过 | [最终审计](phase5/freeze-and-legacy-final.log) |
| `scripts/audit_v031_distribution.py` | 76 运行模块、12 batch 模块、安装源/三 CLI/公共 API/Min–Max 数值/包内容通过 | [构建内容/安装审计](phase5/distribution-audit-final.log)、[JSON](phase5/distribution_audit.json) |
| 实际浏览器文件审计与最终版复核 | pass，细节如下 | [完整闭环](phase5/browser-artifacts.log)、[最终版](phase5/browser-final-version-reproducible.log) |

5 个 warning 与基准相同，来自旧 CLI 合成 fixture 的 identical imshow y-limits；不是新增计算告警。曾尝试 editable 安装但环境缺 `editables`，[真实失败日志](phase5/install-editable-v031.log) 保留；未为此升级依赖，改用 wheel 安装。完整 64 项依赖与起点一致，见 [dependency_comparison.json](phase5/dependency_comparison.json)。

文档完成后还执行了 [交付构建](phase5/build-delivery.log)、[交付 wheel 安装](phase5/install-delivery-wheel.log)、[包内容与安装入口复核](phase5/distribution-delivery.log)、[完整 Ruff](phase5/ruff-delivery.log) 和 [211 个本地链接/66 场景检查](phase5/docs-final.log)，均通过。当前构建文件 hash 见 [distribution_delivery_audit.json](phase5/distribution_delivery_audit.json)；公开差异与隐私检查见 [repository-delivery.log](phase5/repository-delivery.log)。这些是验证工具和文档补齐后的检查，不再次累加 pytest 数量。

独立审查实际发现并修复了 Min–Max 工作区恢复漏核对 scale=1/span 的完整性缺口；没有通过修改 manifest 掩盖问题。原 [失败复现](phase5/minmax-restored-scale-review.log)、8 个攻击回归及修复后全量均保留。其他阶段首跑失败及具体修正见 66 场景矩阵。

## 真实浏览器验收

Chromium/Playwright 在真实 Streamlit 服务上执行上传、控件、预览/确认、下载和新会话恢复，没有虚构 AppTest uploader/zoom API：

1. 合成宽表拆为 A/B，另一个异轴文件得到 C；共三谱、两来源。三条粗调分别确认，随后逐条明确 skip fine，形成独立 B。
2. A：SG 7/2、N_B 最大正峰高、N_S L2；B：Gaussian sigma1.2、N_B Min–Max、N_S 全域绝对面积；C：Median3、N_B 全域绝对面积、N_S 最大正峰高。
3. A→B→C→A 恢复草稿及来源；真实 Plotly Zoom in 将显示范围 [1900,900] 改为 [1650,1150]，数组与正式指纹不变。均匀 A 上实际选择专家 override 显示警告，修改窗口9只形成 draft，正式 S7 保留。
4. 实际下载三谱全部12分支 ZIP；异轴宽表禁用。明确选择 A/B 后下载 201 点×8 数据列宽表及 ZIP，独立 CSV 与 ZIP 成员相同、各列与正式数组逐元素相同。初次多选等待不足曾只选 A，保留初次文件，核对实际选择集后重新下载 A/B，没有将初次文件当作双谱通过。
5. 确认新 A S9，A N_S stale，A N_B 仍有效；strict 导出阻断。明确 valid-only 后下载 B/C 的 N_S，报告 A stale，无 fallback。
6. 新浏览器标签上传保存的工作区并再次保存；660 个数组、70,860 个元素、稳定 ID、原始来源 bytes、草稿、正式数组、父级、选择集和显示/来源偏好完全一致。恢复后重新导出的12节点也一致。
7. 完整闭环先于版本号统一，原包真实记录0.3.0；保留原始证据。随后重启最终0.3.1，新标签恢复旧包、确认新 A S/N_S 并下载和保存，新计算与 exporter 均记录0.3.1；原 B 的204数组、A N_B、无关 B/C不变。两个最终包的4/6 CSV节点与正式快照精确相等。

[原始工具事件](phase5/browser_events.json) 保留失败定位/等待尝试及后续真实重试；[最终版事件](phase5/browser_final_events.json)、[完整文件审计](phase5/browser_artifact_audit.json)、[最终版文件审计](phase5/browser_final_artifact_audit.json) 不包含谱数组。下载文件本体位于被忽略的 `outputs/validation-private/v031/browser/`。审计 spy 证明其加载/校验没有基线/Prepared/2D调用；它不是对过去浏览器服务器调用的追溯监控。服务及 AppTest 中另有禁止调用 spy；专用 replay verifier 仅允许重算 S/N。

## 兼容与冻结

- `artifacts/v0.2.1_science_freeze_manifest.json` 字节不变；34 个冻结文件与冻结根路径集合不变，extra/missing/mismatch 均空。
- 原 `post_baseline_smoothing.py` 的35,105字节前缀完全保留，数组门面仅追加；旧拒绝归一化 Prepared/chained smoothing 保护仍通过。
- 旧普通 schema1.0 加载后新增后处理为空/关闭，旧 B、输入、ID、粗细父级和状态不变。新写 schema2.0，保存历史 B/S、正式分支及独立 draft；preview 不保存，恢复不自动计算。旧 reader 明确拒绝2.0，避免静默丢字段。
- 使用旧基准的 `git archive` 生成纯合成迁移 fixture，两次生成一致。旧导出除 QC 行序外成员逐字节相同；QC 整行/数值相同。行序差异来自既有工作区 JSON 键排序，没有改科学结果。
- ZIP 路径穿越、重复/错配引用、预算及数组结构检查保持；NPY 禁 pickle。
- 原位、旧 baseline/2D/project 包、v0.2.5 smoothing 分支及旧格式保留；没有改数值依赖、算法或冻结默认值。

## 实际运行入口与产物

在仓库根目录：

```bash
PYTHONPATH=src:. .venv/bin/python -m streamlit run ui/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8504
```

本次最终服务实际运行于 [http://127.0.0.1:8504](http://127.0.0.1:8504)，选择“普通光谱模式”→导入/恢复→确认最终 B→第5页后处理。第6页保留旧基线导出与工作区保存。实际构建在忽略的 `dist/` 下，普通后处理目前由该网页与公共 Python API 操作，未新增普通批次 CLI 子命令。

合成数据生成：

```bash
.venv/bin/python scripts/generate_v031_examples.py --help
```

按[示例说明](../../../examples/ordinary_postprocessing/README.md)指定新的忽略输出目录，生成器拒绝覆盖；不读取实验谱。

## 已知限制与未执行项

- 此轮实际环境为 macOS arm64/Python3.12.14/Streamlit1.62.0/Chromium。未执行 Windows/Linux、Safari/Firefox 或不同依赖矩阵；没有进行桌面 PySide GUI 的人工验收。
- 浏览器专家控件在均匀合成 A 上实际操作；没有另行在浏览器上传非均匀轴。其拒绝/override 数值与轴保留由实际自动测试覆盖。
- wheel 在当前环境安装和隔离 import/CLI 验证通过；未创建全新依赖环境。editable 安装未通过，使用已成功的 wheel 路径。
- B verifier 只检查保存锚点/配方/数组完整性，不重拟合基线，也不证明来源真实性。旧 B 快照没有历史数值依赖清单，导出明确区分其已知 baseline 版本与当前 exporter 版本，未伪造历史依赖。
- 专用 S/N verifier 可因不同数值依赖产生精确重放差异；Gaussian verifier 对超大重放核施加资源上限。普通导出不调用 verifier 重放。
- 不支持连续分段/局部多区间、自动对齐/插值、Excel/厂商二进制或跨样品评分。L2随采样点密度变化；不同配方/坐标不会自动成为可定量比较的谱。
