# 普通光谱导出名称（v0.3.2）

普通粗调、最终基线和后处理导出现在默认使用样品原始名称，加上**已确认结果**的处理步骤。文件名用于辨认样品；稳定 ID、配方和科学指纹仍由 JSON 和索引保存。原位、Prepared、smoothing、2D-COS 的历史格式没有改变。

## 在界面中使用

在粗调/细调页的导出区域，或“普通光谱后处理 → 结果导出”页，展开“导出文件命名设置”。两处共享同一份工作区偏好。

1. 默认“原始名称”：二列文件取文件主名；真正的宽表取原始强度列名；无表头宽表使用文件主名加 `__colNN`。NN 是原始强度列索引（第一条强度列为 1，波数列为 0）。只勾选宽表的一列仍按宽表命名。
2. “展示名称”：使用当前已保存的展示名称。不会猜测展示名是否曾被人工修改。
3. “旧版 ID 命名”：恢复 v0.3.1 的路径、宽表列头和 schema 1.0 合同，供依赖旧名称的脚本使用。
4. 可勾选“覆盖当前光谱的导出基础名”；只影响当前谱，优先于前两种策略，不改原始来源、展示名称或 ID。空覆盖采用安全后备名称。legacy 模式保留覆盖偏好，但不应用它。
5. 检查预览中的最终 CSV 路径、处理状态、请求结果/自动父级和排除原因，再打包下载。名称或冲突环境变化后需要重新打包；已经下载的 ZIP 内容不会被改写。

粗调、细调和后处理确认规则不变。细调尚未决定时不能自动导出最终 B。修改草稿后，文件名仍描述有效的正式结果；缺失或过期的目标不会被其他分支替代。“仅导出有效项”仍须明确选择。

## 名称与处理步骤

对二列 `G2_90RH.csv`，即使表头只是 `Absorbance`，实际示例为：

| 正式结果 | 主谱 CSV |
|---|---|
| 粗调 | `G2_90RH__baseline-coarse.csv` |
| 最终 B（细调或明确跳过） | `G2_90RH__baseline.csv` |
| Savitzky–Golay S | `G2_90RH__baseline__smooth-savgol.csv` |
| B 的最大正峰归一化 | `G2_90RH__baseline__norm-max.csv` |
| B 的窗口峰高归一化 | `G2_90RH__baseline__norm-peak-window.csv` |
| S 的面积归一化 | `G2_90RH__baseline__smooth-savgol__norm-area.csv` |
| B 的 L2 归一化 | `G2_90RH__baseline__norm-l2.csv` |
| B 的 Min–Max 显示缩放 | `G2_90RH__baseline__minmax_scaled_intensity_display_only.csv` |

其他平滑后缀为 `smooth-gaussian`、`smooth-moving_average`、`smooth-median`；窗口面积归一化为 `norm-area-window`。N_S 的平滑方法来自它实际绑定的 S 父快照。未知方法会明确失败。参数、参考窗口、面积定义、scale/offset 和 warnings 保存在配方中；相同简短后缀不表示不同样品的配方相同或可直接定量比较。

每谱两列 CSV 的 y 列仍表示 quantity：`corrected_absorbance`、`smoothed_absorbance`、`normalized_intensity` 或 `minmax_scaled_intensity_display_only`。同轴宽表的 y 列用上述可读样品名与处理后缀。归一化量不当作吸光度，不生成物理 T/%T；原基线导出可选的派生 T/%T 继续明确标为由校正吸光度数学派生。

## 安全、消歧与稳定范围

文件名取 Windows/POSIX 路径最后一级，再去掉最后一个扩展名。中文、希腊字母、空格、括号、温湿度符号保留；统一 NFC，移除控制字符，替换非法路径字符，处理 Windows 保留名、末尾点/空格。JSON 仍保留完整原始文本。

统一在**全工作区**分配基础名，然后选择本次输出。比较键使用 NFC 和大小写折叠。不同宽表中重名列优先加文件主名；仍冲突时追加稳定身份 token，从 8 位开始，前缀碰撞时逐步扩展，不能唯一分配时明确失败。文件名不会覆盖已有成员。

可读模式的单个文件名（含后缀、token、扩展名）最多 200 UTF-8 字节。基础名统一预留 100 字节给所有支持的后缀，较长中文名可能提前截断；不会切断字符。该限制不保证任意解压目录的整条路径长度；legacy 保留历史长度规则，可能超过此新预算。清理、截断、后备名称和 token 理由在预览与映射中记录。

未变化工作区内，改变选择顺序、导出子集或保存再恢复不改变名称。新增/删除重名条目、改变展示名/覆盖，可能改变消歧分配；重新导入会产生新 ID，不保证与旧导入使用同一个 token。

## ZIP 内的名称对照与版本

新版可读模式保留各自 artifact 类型，内部 metadata/index 为 schema **1.1**，`naming_policy_version="1.0"`。外层 manifest 仍为 **1.0**，后处理节点 JSON 仍为 **1.0**。每个实际节点在 `export_name_map.csv` 和 metadata 的权威 `name_map` 中有一行，记录：

- spectrum/source ID、原文件、原列、列索引、源摘要、布局证据；
- 展示名、策略、覆盖、采用的名称来源、基础名、清理/截断/消歧决定；
- branch、quantity、purpose、fingerprint、parent fingerprint；
- 主谱、配方、分量、移除分量、派生输出路径、宽表列名；
- `included_as=requested|parent`。

只请求 N_B 会自动包含 B；只请求 N_S 会包含 B 和 S。它们是验证来源需要的父节点，预览和映射会说明。配方 JSON 仍可使用内部身份文件名。CSV 文本沿用公式转义规则，JSON 保留原始文字。

`legacy_id` 输出旧 schema **1.0**，没有名称映射成员或新增命名字段。新版验证器支持真实旧包及新包，拒绝未知 schema。后处理 verifier 保留来源、父子、数组、有效配方及完整成员集合检查；可显式重放 S/N，但从不重拟合基线。新版基线 verifier 检查名称/路径/身份/报告和宽表数组一致性；旧基线包仍只有原有归档完整性能力，不声称等同于后处理语义复算。摘要是完整性检查，不是来源签名。

```python
from ftir_workbench.batch.export import build_batch_export, preview_batch_export
from ftir_workbench.batch.postprocessing_export import build_postprocessing_export

preview = preview_batch_export(workspace, ids, stage="fine")
readable = build_batch_export(workspace, ids, stage="fine", include_wide=True)
old_contract = build_batch_export(workspace, ids, stage="fine", naming="legacy_id")
post = build_postprocessing_export(
    workspace, ids, branches=("normalized_smoothed",),
    naming={"policy": "original", "overrides": {ids[0]: "样品 α"}},
)
```

宽表仍只在**实际输出 x 数组逐元素一致**时可用；不会插值、对齐、排序、删点或裁负值。CSV 保持 `.17g` 和 float64 回读精度。生成预览与导出仅序列化正式快照，不运行 baseline/smoothing/normalization，不创建 Prepared，`is_2d_ready=false`。

## 保存、升级与实现位置

工作区仍是 schema **2.0**，兼容读取旧 **1.0/2.0**。命名偏好保存在既有 `state.display_preferences["export_naming"]` 中；载入/保存严格检查类型、策略和跨条目一致性。缺少命名设置时默认 `original`。保存正式快照、来源、草稿和父级；不保存预览，不自动重算。升级前先保存备份；旧版读取器并不承诺理解新命名偏好。

保存工作区、停止原 Streamlit 进程后，在原仓库及虚拟环境执行（有未提交工作应先保护）：

```bash
git switch main
git pull --ff-only origin main
python -m pip install -e . --no-deps
python -m streamlit run ui/streamlit_app.py
```

再从“恢复普通光谱工作区”上传旧 ZIP。GitHub 发布不会自动替换已经运行的进程。wheel 是 Python 包，Streamlit UI 仍从仓库脚本运行，没有新增桌面安装器。

集中模块 `batch/export_naming.py` 只解析保存的来源与名称，不导入计算入口。两个 exporter 的 preview/build 共用分配计划；两个 UI 的下载缓存绑定整个工作区名称分配和实际节点计划。工作区只新增偏好校验入口；科学模型、导入器、配方、状态机、指纹实现及所有冻结文件未改动。

验证与已知限制见 [本次验收报告](ordinary_export_naming_acceptance.md) 和 [真实浏览器记录](ordinary_export_naming_browser_acceptance.md)。历史科学计算指纹保持原样；新版主动计算时因软件版本改变而形成新指纹是正常行为。
