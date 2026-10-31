# 普通光谱导出命名：真实浏览器验收

2026 年 11 月 1 日（北京时间）在版本 **0.3.2** 完成真实 Playwright 浏览器闭环。最终服务为 `http://127.0.0.1:8767`，使用 `PYTHONPATH=src`、已安装的 editable 包和 `fileWatcherType=none`。本记录不把 AppTest 或 Python helper 生成的包当作浏览器下载。

结果：**通过本文列明的浏览器与实际下载范围**。[机器可读记录](../artifacts/validation/v0.3.2/phase4/browser-summary.json)保存每个文件的完整 SHA-256、字节数、ZIP 成员数、主谱 CSV 数、路径和核对项目；[最终实际审计日志](../artifacts/validation/v0.3.2/phase4/browser-artifact-audit-recorded.log)记录命令、输出、退出码和时间。

## 输入与实际操作

仅使用本次生成的两个合成吸光度文件 `G2_90RH.csv`、`样品 α (25°C 80%RH).csv`，各 181 点，轴均为 1800→900 cm⁻¹；原始强度列均名为 `Absorbance`。文件、下载 ZIP、截图及审计脚本保存在忽略目录 `outputs/validation-private/v032/browser/`，没有提交原始或派生数组。

| 顺序 | 实际 UI 操作 | 观察结果 |
| --- | --- | --- |
| 1 | 新标签页选择普通模式，用文件上传控件同时上传两个文件并确认 | 页面明确显示导入 2 条独立光谱 |
| 2 | A 选择 linear，计算粗调预览，再确认；点击正式 ZIP 下载 | 默认路径为 `spectra/G2_90RH__baseline-coarse.csv`，不是通用表头 Absorbance |
| 3 | A 明确跳过细调；切换 B，预览并确认 arPLS，再明确跳过细调 | 两条独立结果保留；随后切回 A，显示已跳过细调 |
| 4 | A 启用 Savitzky–Golay 平滑，预览后确认；明确选择 S 为归一化来源 | S 已确认；没有把归一化来源自动切换当作用户操作 |
| 5 | A 选择全范围最大正峰高，启用、预览并确认 N_S | N_S 已确认，导出选择明确改为 N_S |
| 6 | 默认原始命名打包并下载后处理 ZIP | 包含请求的 N_S，以及角色明确为 parent 的 B、S |
| 7 | 开启当前谱导出覆盖，输入 `样品 G2 90%RH` 并提交 | 旧后处理下载按钮消失；重新打包下载使用新基础名 |
| 8 | 进入批量检查页，查看粗调、最终两组命名控件并下载工作区 | 两组覆盖值完全相同；保存成功 |
| 9 | 新标签页、新 Streamlit 会话，从真实文件上传控件恢复下载的工作区 | 2 条谱恢复；覆盖名称和 N_S 导出选择恢复；没有点击任何重新计算按钮 |
| 10 | 恢复后再次打包 N_S 并下载，再下载恢复后的工作区 | 正式数组、身份和父级保持；两次覆盖命名 N_S ZIP 的 SHA-256 完全一致 |
| 11 | 恢复会话的粗调导出范围设为全部条目，明确勾选同轴宽表 | 真实下载包含两条主谱的批量 ZIP，以及 181 行、3 列的独立宽表 CSV |

所有正式 ZIP、工作区和宽表均由点击真实下载按钮产生的 Playwright `download` 事件取得，并通过 `download.saveAs` 保存；每次正式下载均返回 `failure=null`。

## 实际下载与核对

| 文件 | ZIP 成员（含 manifest） | 主谱 CSV | 内部 schema |
| --- | ---: | ---: | --- |
| `coarse_original.zip` | 8 | 1 | 1.1 |
| `post_original.zip` | 12 | 3 | 1.1 |
| `post_override.zip` | 12 | 3 | 1.1 |
| `workspace_before.zip` | 256 | 不适用 | 2.0 |
| `post_restored.zip` | 12 | 3 | 1.1 |
| `workspace_restored.zip` | 256 | 不适用 | 2.0 |
| `coarse_two_wide.zip` | 12 | 2 | 1.1 |
| `coarse_two_wide.csv` | 不适用 | 两条强度列 | CSV |

实际下载审计没有生成替代导出文件，调用 `read_verified_archive`、`verify_batch_export` 和 `verify_postprocessing_export(recompute=False/True)`。审计期间两个基线 `run_pipeline` 入口被禁止；后处理专用验证器的 `recompute=True` 只执行它既有的验证重放。

对实际工作区文件的检查覆盖稳定 ID、显示顺序、当前/多选项、原始文件字节、解析记录、原始 x/y、准备/粗调/细调草稿与正式配方、基线分量、结果数组、指纹、明确跳过决策，以及 S/N_S 的正式数组、草稿、父级关系、归一化来源和命名偏好。恢复文件没有保留预览。工作区 ZIP 的整体 SHA 不相同，因为恢复后的导出历史增加；正式数据和上述状态均相同。

每个实际导出谱图 CSV 的 float64 x/y 均与恢复工作区的对应正式快照逐元素相等；N_S 的强度列仍为 `normalized_intensity`。没有生成 Prepared 或 `for_2dcos` 成员，导出声明 `is_2d_ready=false`。

批量宽表文件与 ZIP 内宽表字节一致；两个强度表头等于名称映射中的 `wide_column_label`，波数列和两条强度列逐元素等于各自正式粗调结果。两谱没有自动对齐或插值。

实际可读预览表截图 `post_original_preview.png` 和 `post_override_preview.png` 已查看，显示以下三条路径及 B/S 为父级、N_S 为请求结果；截图可见内容与实际 ZIP 权威名称映射一致：

```text
spectra/G2_90RH__baseline.csv
spectra/G2_90RH__baseline__smooth-savgol.csv
spectra/G2_90RH__baseline__smooth-savgol__norm-max.csv
```

覆盖后对应基础名全部变为 `样品 G2 90%RH`。另保留 `coarse_two_preview.png` 显示两个实际粗调条目。

## 失败尝试与范围边界

早期 `8766` 服务在代码热重载、安装变化后出现 `COARSE_STALE`。该会话的页面快照和 trace 保留，没有将其计为通过。按已记录的升级要求重启为稳定的 `8767` 并从新会话重新上传后，相同操作成功；工作区升级应先保存，再 restart/load，热重载不能代替工作区迁移。

浏览器工具还遇到底层 radio 被标签遮挡、模式切换尚未完成时导入按钮禁用、下拉选项定位超时，以及表格工具栏 CSV 没有触发下载事件。这些超时均未被当作成功：分别使用可见标签、等待实际控件、正常组合框键盘选择，以及可读预览截图完成对应检查。正式 ZIP/工作区/宽表下载使用正常下载事件成功。

实际下载审计脚本初次有三个检查器自身假设错误：后处理 artifact type、`wide_headers` 是否包含波数列、安装 distribution 名。均改为真实公开合同后重跑；[首轮](../artifacts/validation/v0.3.2/phase4/browser-artifact-audit.log)、[第二轮](../artifacts/validation/v0.3.2/phase4/browser-artifact-audit-final.log)、[第三轮](../artifacts/validation/v0.3.2/phase4/browser-artifact-audit-pass.log)失败日志保留，最终通过日志见文首。

本次真实浏览器没有另外执行 display、legacy_id 策略、N_B 计算或异轴宽表禁用分支。这些不冒充浏览器结果；命名策略、legacy 合同、缓存和工作区等自动化结果见 [UI 定向 8 项通过日志](../artifacts/validation/v0.3.2/phase3-ui/naming-final.log)及对应测试。

最终浏览器 trace 为 `.playwright-mcp/traces/trace-1793465578901.trace`，恢复后双谱宽表补充 trace 为 `.playwright-mcp/traces/trace-1793466213814.trace`。浏览器验收期间未修改产品源码、安装依赖或业务 session state。
