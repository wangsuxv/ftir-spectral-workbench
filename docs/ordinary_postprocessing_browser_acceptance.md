# v0.3.1 普通光谱后处理：真实浏览器验收

记录日期：2026-09-11。本机真实浏览器已完成上传、逐谱确认 B/S/N_B/N_S、下载、过期排除及新会话工作区恢复；下载文件独立审计为 `pass`。本记录描述实际证据，不将 AppTest、脚本生成物或预期结果冒充浏览器操作。

## 环境与证据

浏览器由实际 Playwright 工具控制，访问本机 Streamlit `http://127.0.0.1:8504/`。使用现有 Python 3.12.14、Streamlit 1.62.0、NumPy 2.5.2、SciPy 1.18.1 环境，没有安装新数值依赖。输入来自固定 seed=3101 的纯合成生成器，不含实验数据。

| 证据 | 内容 |
|---|---|
| [browser_events.json](../artifacts/validation/v0.3.1/phase5/browser_events.json) | 30 条真实工具响应，包含实际 Playwright 操作、返回值、下载事件及失败定位尝试；仓库绝对路径已脱敏 |
| [browser-artifacts.log](../artifacts/validation/v0.3.1/phase5/browser-artifacts.log) | 实际下载审计命令、完整输出、退出码 0 |
| [browser_artifact_audit.json](../artifacts/validation/v0.3.1/phase5/browser_artifact_audit.json) | 机器可读完整审计，`mode=complete`、缺失文件为空、失败项为空、`status=pass` |
| [browser_final_events.json](../artifacts/validation/v0.3.1/phase5/browser_final_events.json) | 最终 v0.3.1 重启后的两条真实工具响应：恢复、确认新 S/N_S、下载及保存工作区 |
| [browser-final-version-artifacts.log](../artifacts/validation/v0.3.1/phase5/browser-final-version-artifacts.log)、[browser_final_artifact_audit.json](../artifacts/validation/v0.3.1/phase5/browser_final_artifact_audit.json) | 最终版本实际下载审计 pass；计算/导出版本、基线不变、独立性和正式数组精确核对 |
| [审计脚本](../scripts/verify_v031_browser_artifacts.py) | 只读取真实上传源和下载文件，校验正式数组、身份、父级、恢复、宽表及 stale 排除；不生成替代证据 |
| [最终 pytest](../artifacts/validation/v0.3.1/phase5/pytest-final.log) | v0.3.1 全量 1238 passed、5 个既有 warning，132.47 秒；独立于浏览器步骤计数 |

第一轮完整浏览器下载发生在包版本统一之前，其 `exporter_versions` 中的 workbench **实际为 0.3.0**。下载内容已经包含本次后处理实现；原文件、原 metadata 和原日志保留，不追改成 0.3.1。随后工作台与发行包统一为 0.3.1，sdist/wheel 构建、当前环境 wheel `--no-deps` 安装、最终全量测试和冻结/旧功能审计通过；见 [构建](../artifacts/validation/v0.3.1/phase5/build-final.log)、[安装](../artifacts/validation/v0.3.1/phase5/install-wheel-v031.log) 与 [最终冻结审计](../artifacts/validation/v0.3.1/phase5/freeze-and-legacy-final.log)。最终版本服务已实际重启并完成下文补充闭环，保留两轮各自真实版本；[64 项依赖](../artifacts/validation/v0.3.1/phase5/dependency_comparison.json) 与起点完全一致。实际基准与完成提交 SHA 见 [交付报告](../artifacts/validation/v0.3.1/REPORT.md)。

## 实际输入与确认配方

上传 `synthetic_wide.csv` 和 `synthetic_other_axis.csv`，均明确选择吸光度。宽表拆为 A/B 两条，另一个文件为 C；实际原始 bytes 与恢复工作区中的来源 bytes 精确匹配。

| 条目 | 轴 | B | S | N_B | N_S |
|---|---|---|---|---|---|
| synthetic_A | 1900→900 cm⁻¹，201 点 | coarse `none` 明确预览/确认，明确跳过 fine | SG 7 点、2 阶 | 全域最大正峰高 | L2 向量 |
| synthetic_B | 与 A 完全相同 | 同上 | Gaussian σ=1.2 点 | Min–Max 0–1 显示缩放 | absolute 全域面积 |
| synthetic_C | 900→1900 cm⁻¹，251 点 | 同上 | Median 3 点 | absolute 全域面积 | 全域最大正峰高 |

这里选择 coarse `none` 是为了使已知合成 B 易于检查；本轮浏览器没有用非零算法基线证明拟合质量。普通 coarse/fine 科学等价、estimate-only smoothing 和旧算法回归由对应 pytest 及冻结审计覆盖。每个后处理分支都经过实际预览与明确确认，没有直接写入服务器业务状态来替代 UI。

## 已执行闭环

1. 在导入前选择普通模式，上传两份文件并确认单位。页面返回“已导入 3 条独立光谱”，保留每谱稳定 ID、来源文件及原列位置。事件 1–4 记录导入后状态及三条 coarse `none` 的实际确认。
2. 对三条光谱分别点击“确认不做细调”，随后进入“普通光谱后处理”。两个新增步骤初始关闭；仅 coarse、未决定 fine 的状态没有被自动当作 B。事件 5–6 记录实际完成及定位重试。
3. 对 A、B、C 分别预览/确认上表的 S、N_B、N_S。A→B→C→A 后，A 的归一化来源 S 和 L2 草稿恢复；B 的 Min–Max 页面无目标值控件。事件 8–15 记录对应操作与返回状态。
4. 在 A 的已确认 S 图上真实点击 Plotly `Zoom in`。x 显示范围从 `[1900,900]` 改为 `[1650,1150]`；图形实际数据和显示的 S/父级版本均未变化（事件 9）。这是实际图形缩放，不是只修改服务层 display preference。
5. 在“结果导出”选择全部条目、全部四分支，下载 `all_branches.zip`。包含三条谱共 12 个结果节点；由于 C 异轴，宽表控件实际禁用（事件 16–17）。
6. 明确勾选 A/B，下载 `selected_wide.zip` 和独立 `selected_wide.csv`。宽表含 201 行、8 个独立结果列，外加一列波数；逐列与正式 float64 数组完全相同，独立 CSV 与 ZIP 成员 byte 相同（事件 18–19、24）。
7. 在均匀轴 A 上实际展开“采样轴专家选项”，将策略改为明确 index-space override，并将 SG 窗口草稿由 7 改为 9，保留 7 点正式快照。确认图中仍显示原 S 版本；保存 `workspace_before.zip` 时 A/B 勾选集、草稿和正式结果分开保留（事件 20–24）。
8. 预览/确认 A 的新 9 点 S。A 的旧 N_S 显示父级过期，N_B 仍显示已确认。选择所有条目的 N_S 时，默认严格打包按钮禁用；明确勾选“只导出有效项”后，实际下载 `stale_valid_only.zip`，只包含 B/C 的 N_S 及必要 B/S 父数据。报告明确排除 A，不回退其他来源（事件 25–26）。
9. 新建浏览器页形成新的 Streamlit 会话，实际上传此前的 `workspace_before.zip` 并点击恢复，再下载 `workspace_restored.zip`。A/B 勾选集恢复；A 窗口草稿是 9，正式 S 仍是旧 7 点版本，页面提示草稿与确认版不同。未执行预览或确认即可再下载 `all_branches_restored.zip`（事件 27–30）。
10. 独立运行审计脚本读取上述实际下载，核对来源、数组、父级、配方、fingerprint、quantity、宽表和排除项。审计未生成替代工作区或导出包。

## 下载核对结果与解释边界

| 检查 | 实际结果 |
|---|---|
| 完整 B/S/N_B/N_S 导出 | 三条谱 12 个节点，12 个分支 CSV 与已确认工作区数组逐元素相同 |
| 工作区恢复 | 660 个数组、70,860 个元素精确匹配；稳定 ID、来源、草稿、只读正式数组、历史父级、勾选集、来源/导出选择与显示偏好一致 |
| 恢复后的完整导出 | 12 节点数组、配方、来源与 fingerprint 相同；本轮两个完整导出 ZIP 的 bytes 也相同 |
| A/B 宽表 | 201 行、8 个结果列逐列 float64 精确一致；加入 C 则禁用宽表，不伪对齐 |
| stale 隔离 | A 的 N_S 明确排除，没有过期数组或其他分支回退；B/C 的数组与 fingerprint 保持不变 |
| 非重放验证/恢复 | 审计调用期间 baseline、Prepared、2D、smoothing、normalization 禁止调用计数均为 0 |
| 显式重放验证 | 每个已导出 S 重放一次、每个已导出 N 重放一次；完整包为 3 次平滑和 6 次归一化，baseline/Prepared/2D 调用为 0 |

恢复核对中的 660 个数组包括快照及嵌套父级数组，不等于 660 条独立光谱。工作区 ZIP 自身的保存记录可不同，因此判据是恢复后的完整业务/科学状态；不要求两份工作区 ZIP byte 相同。

审计 spy 只观察**下载后的加载与 verifier 调用**，不能回溯证明浏览器服务器过去所有操作的调用次数。导出/恢复/页面不运算的合同另有服务和 AppTest spy 测试。显式 verifier 重放用于检验导出父子关系，不能说成导出时进行了计算；B 只做已存锚点完整性检查，未重新拟合。SHA-256 不是数字签名，不证明原始实验来源真实性。

## 最终 v0.3.1 重启后补充闭环

停止旧服务、以 v0.3.1 重新启动后，在新浏览器页实际上传并恢复第一轮的工作区。页面恢复 A 的 9 点平滑草稿和旧 7 点确认版；明确预览/确认新的 9 点 S，导出当前勾选 A/B 的 S 为 `v031_smoothed.zip`。随后明确预览/确认 A 新的 L2 N_S，导出勾选 A/B 的 N_S 为 `v031_normalized_smoothed.zip`，并保存 `workspace_v031.zip`。这两条实际工具响应均返回异常计数 0。

独立下载审计确认：A 新 S 与 N_S 的**实际计算版本**为 0.3.1，两个包的**导出环境版本**也为 0.3.1；A/B 的既有基线及所有条目的原始来源保持。204 个基线数组精确不变，A 的 N_B 和无关 B/C 状态与数组完全不变。S 包为 2 个请求结果加 B 父级共 4 个 CSV 节点，N_S 包为 2 个请求结果加 B/S 父级共 6 个节点，全部与新保存工作区中的正式数组精确相同。两个包均通过非重放完整性检查和显式 S/N 重放检查，后者没有 baseline、Prepared 或 2D 调用。

此补充步骤验证最终版本的恢复、新确认和实际下载；没有声称它重新执行第一轮全部 30 条操作。历史 B 及 B/C 的既有 S/N 快照保留其真实旧计算版本，不因重新导出而伪装为全部重算。

## 保留的失败尝试

30 条原始响应不是“30 次全部一次通过”。事件 5、7、10、12、16、21、22 保留 `TimeoutError`：包括等待不匹配的 fine/Min–Max 文本、Streamlit checkbox 的包裹元素拦截指针、分支切换后尚未稳定的下拉选项及专家策略下拉定位失败。后续通过读取实际 DOM、点击可见标签、等待分支特定控件、键盘展开下拉并选择实际选项完成操作；没有把失败响应改成成功。

事件 1 是上传确认等待超时后的真实页面检查，页面已显示两份上传文件与三条导入结果；下载审计另外核对来源 bytes。UI 检查没有以服务层注入状态替代上传。带显式 exception 计数的成功检查返回 0，但本记录不扩大声称完整浏览器过程中具有连续异常遥测。

## 复现步骤与未执行范围

在新的、未使用过的忽略目录生成输入，保留旧证据：

```bash
.venv/bin/python scripts/generate_v031_examples.py \
  --output outputs/v031-browser-recheck/inputs
PYTHONPATH=src:. .venv/bin/python -m streamlit run ui/streamlit_app.py \
  --server.port 8504 --server.fileWatcherType none
```

若端口已被当前工作台占用，使用另一空闲端口。按上文完成操作，将七份真实下载以 `all_branches.zip`、`workspace_before.zip`、`workspace_restored.zip`、`all_branches_restored.zip`、`selected_wide.zip`、`selected_wide.csv`、`stale_valid_only.zip` 保存到 `outputs/v031-browser-recheck/`，保留其 `inputs/` 子目录中的实际上传源。再运行：

```bash
PYTHONPATH=src:. .venv/bin/python scripts/verify_v031_browser_artifacts.py \
  --directory outputs/v031-browser-recheck \
  --output outputs/v031-browser-recheck/audit.json
```

此命令只审计已有真实下载；缺少文件时不算完整通过。不要以脚本调用导出服务生成替代文件来“补全”浏览器证据。

本轮浏览器专家 override 控件在均匀 A 上实际操作；没有将其写成非均匀轴上传/拒绝/override 的浏览器闭环，后者的科学行为由数值测试覆盖。未执行 Windows/Linux、Safari/Firefox、多浏览器矩阵、移动端、远程部署、真实实验数据或百万点性能验收。未以此验证新算法优越性、真实噪声、仪器分辨率、样品定量可比性或实验来源真实性。
