# Project Status

更新日期：2026-08-23

## 当前状态

FTIR Spectral Workbench 首版已经完成，可用于统一的基线处理、baseline-only 导出，以及从同一份校正吸光度继续进行 self/cross 2D-COS。项目已建立独立 Git 仓库，当前分支为 `main`，首版冻结标签为 `v0.1.0`。

PySide6/macOS `.app` 不属于本次首版范围；按照合并规格，先以 Streamlit 验证统一科学工作流。

## 已交付

- 保留 `ftir_baseline` 与 `ftir2dcos` 两个经过回归测试的科学包。
- 新增 `ftir_workbench` 协调包，提供不可变 Prepared 数据合同、配置封装、fingerprint、状态失效、baseline/2D/project service 与导出校验。
- 新增统一 Streamlit 工作台和统一 CLI。
- baseline-only 是正式成功路径；用户可下载结果后结束，不会触发 2D 计算。
- 当前基线结果可直接通过内存继续 2D，不写回 CSV、不重新基线。
- corrected CSV、sidecar metadata、baseline ZIP 均可作为以后继续 2D 的入口。
- 支持多个 self 区间、矩形 cross-range、峰序分析、canonical 与 2dpy-compatible convention。
- 不同 Prepared blocks 的 cross 默认阻断；逐项兼容检查通过后仍需调用方显式确认，并记录双方完整血缘。
- baseline、2D 和 `.ftirw` 项目包均有 SHA-256 manifest；2D 子结果保存三项父级血缘。
- CI 使用锁定环境覆盖 Python 3.11、3.12、3.13，并运行基于合成/示例数据的 pytest、Ruff、Mypy 与 wheel 构建；私有数据重放仅在本机执行。

## 最终验证

| 检查 | 结果 |
|---|---:|
| 全量 pytest | 396 passed in 36.81s |
| Baseline 原回归 | 135 passed |
| 2D-COS 原回归 | 186 passed |
| 新集成与统一 UI 测试 | 75 passed |
| Ruff | passed |
| Mypy（`ftir_workbench`） | passed，15 source files |
| Wheel 构建 | passed |
| Streamlit AppTest 启动 | passed，9 个页面，无异常 |
| 私有数据独立复算与三层 manifest（仅本机） | passed |

构建产物：`ftir_spectral_workbench-0.1.0-py3-none-any.whl`，SHA-256 `d793e724610d4b3dbd09c9808e6bbc14275e7ed76fb7f7200e946dc7e0ffb1a7`。

## 私有数据验证

项目已在本机实验数据上完成 baseline、Prepared、self/cross 2D-COS、精确往返、矩阵独立复算以及三层 manifest 验证。为保护数据，原始谱、生成的 ZIP/`.ftirw`、数据维度、数据指纹和产物哈希均不进入公开 Git 历史。

公开 CI 只使用 `examples/` 下的合成/示例数据。将自己的谱放入被忽略的 `data/original/` 后，仍可在本机执行相同的演算和验证脚本。

## 科学注意事项

- 对非等间隔扰动，当前 Noda 矩阵按数值排序后的采集序号构造，不执行非均匀时间加权；该策略会写入 warning 和 provenance。
- 演示 recipe 只用于证明数据链、血缘和数值可复现性；真实样品的基线参数仍需领域专家审阅。
- `BASELINE.dpt` 不会自动并入扰动谱，也不会被擅自解释为算法基线。

## 启动

```bash
source .venv/bin/activate
streamlit run ui/streamlit_app.py
```

完整命令与数据说明见 `README.md` 和 `docs/original_data.md`。
