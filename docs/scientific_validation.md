# Scientific validation

验证由三层组成：精确不变量、已知真值合成谱、真实序列的无真值诊断。三层不能互相替代。

v0.2.0 仍以 v0.1 的 baseline 数值结果为冻结参照。`artifacts/v0.1_baseline_freeze_manifest.json` 记录 `src/ftir_baseline/**`、`tests/baseline_regression/**` 和 `legacy/baseline_streamlit_app.py` 的文件哈希；发布验收必须同时通过逐文件哈希比较和冻结路径 `git diff`。

## 1. 精确不变量

测试覆盖：

- `%T=100 → A=0`、`%T=10 → A=1`、`T=0.1 → A=1`；
- NaN、Inf、`T<=0`、重复与非单调波数必须明确失败；
- 原始数组不可修改，处理数组均为 float64；
- endpoint 对纯线性基线、strict 两端点和升降序轴的精确行为；
- PCHIP 通过代表锚点、两锚点线性退化、禁止外推；
- 自动算法形状、参数锁定、收敛记录及方向等价；
- `B_total=B_coarse+B_fine` 和 `raw=B_total+corrected`；
- shared-shape 每谱只有常数和一次斜率两个自由度；
- 归一化主/显示支路隔离；
- JSON recipe 重放逐元素一致、输入哈希和软件版本完整；
- ZIP 中每个文件和 manifest 的 SHA-256 可验证。

v0.2 新增但不改变 baseline 数值的不变量：

- Coarse/Fine Preview 对完整序列调用现有 pipeline，选中实际行与同一 preview result 逐元素一致；
- Preview 不修改 committed `baseline_config`，不覆盖正式 baseline result，也不清除 Prepared 或 2D；Adopt/Apply 才执行既有失效规则；
- 五张 Series heatmap 分别绑定 raw/coarse/fine/total/corrected 原数组，Corrected 显示色阶以 0 为中心但不改值；
- QC table、趋势与 drill-down 逐行对齐 `QCResult.per_spectrum` 和 perturbation labels，页面不调用新的 QC 计算；
- A→T→A 与 A→%T→A 在浮点容差内往返，identity 返回 owned copy；complex、NaN、Inf 和 overflow 明确失败；
- 负 A 允许得到 `T>1`、`%T>100`，不得裁剪；
- 改变 A/%T/T 显示不修改 baseline config、Prepared hash、2D result 或 fingerprint；
- 对 n 个 ranges，cross pair count 为 `C(n,2)`，oriented map count 为 `n(n-1)`；
- `Phi_reverse=Phi_stored.T`、`Psi_reverse=-Psi_stored.T`，reverse row/column axes 与 variables 正确交换；
- Cross 2 不增加 `CrossRangeResult`、不重新调用 cross 核心、不改变 2D fingerprint、不复制 peak-order evidence；
- full block overview 的每个对角/非对角单元绑定正确的 self/stored/reverse 原矩阵；
- v0.2 2D bundle 校验 reverse matrices、实际轴和 `orientations.json`，完整 v0.1 stored-only bundle 仍通过 verifier。

## 2. 合成谱

`synthetic.py` 以固定种子生成真实化学谱、真实基线和噪声。峰型包含 Gaussian、Lorentzian、
Voigt；背景包含常数、线性、二次、指数和宽弯曲项；噪声包含 Gaussian 与少量脉冲。

覆盖场景：

1. constant；
2. linear；
3. quadratic；
4. exponential；
5. broad-and-narrow；
6. broad-OH；
7. smooth-drift；
8. abrupt-jump；
9. low-noise；
10. high-noise；
11. 同一场景的升序和降序等价检查。

评价包括 baseline / corrected RMSE、峰高偏差、峰面积偏差、峰位移动和基线时间粗糙度。

固定 seed `20260821`、arPLS λ=10⁶、每场景 5 条谱 × 500 点、Gaussian σ=0.002 的参考结果：

| 指标 | 结果 |
|---|---:|
| 10 场景平均 baseline RMSE | 0.008754 |
| 10 场景平均 corrected RMSE | 0.009352 |
| 普通 9 类 baseline RMSE 范围 | 0.00276–0.00917 |
| broad-OH baseline RMSE | 0.05163 |
| smooth-drift time roughness | 0.000486 |
| abrupt-jump time roughness | 0.01314 |

broad-OH 的显著退化是预期的失败检测：宽峰几乎占满区间时，自动方法无法只凭数据知道它是
化学峰还是背景。工具应把这一风险暴露给用户，而不是用“看起来更平”作为成功标准。

## 3. 私有序列验证

实验数据没有已知真基线，因此本机验证只检查可证的格式、数据合同、
数值重建、QC、血缘和导出往返性质。原始谱、样本数量、采集范围、
数据派生统计与指纹均不在公开仓库中发布。

验证规则仍包括：数值扰动排序不会静默改变、`BASELINE.dpt` 被显式
排除并记录、确认吸光度后不做对数转换、负残差只报警不裁剪，以及
候选扫描中的最低启发式分数不会自动成为最终配方。

## 4. 质量指标

- Anchor residual：固定锚点窗口中 `median(abs(corrected))`；
- Negative fraction：`corrected < -k·noise_sigma` 的比例；
- Baseline roughness：波数方向二阶差分平方均值；
- Peak preservation：导数相关、关键区峰位与相对峰高变化；
- Time roughness：沿扰动顺序的基线二阶差分平方均值；
- Reconstruction：`max(abs(raw-baseline-corrected))`；
- Baseline area / adjacent RMS：用于序列趋势和异常跳变提示。

时间突变可能是真实接触变化或仪器事件，因此任何异常只提示，不自动删除数据。

## 5. 显示单位与派生导出

基线和 Prepared 的科学单位保持 absorbance。反向显示只使用：

```text
T = 10 ** (-A)
%T = 100 * 10 ** (-A)
```

派生 CSV 的唯一数据源是 `PipelineResult.analysis_data`。CSV metadata 明确说明该结果是 baseline-corrected absorbance 的数学表示，不是原仪器透过率。文件不进入 baseline ZIP，因而不能作为 baseline 科学产物或仪器信号恢复来解读。

## 6. Cross 2 独立验证

每个 unique pair 只由 `compute_cross_2dcos` 计算一次。Cross 2 从结果对象的只读 reverse properties 取得，并依据 `row_variable` / `column_variable` 映射实际 ranges。验收同时覆盖 canonical 与 2dpy-compatible convention。

2dpy-compatible hetero oracle 必须独立读取宽表、转置输入、独立均值扣除、独立构造 Noda 矩阵、以 nested loop 计算 hetero sync/async 并执行 final transpose；oracle 不得调用 production cross helper。这样可以在不把 reverse view 误当作第二次计算的前提下验证 stored orientation。

## 7. 复现实验

```bash
pytest -q
pytest --cov=ftir_baseline --cov-report=term-missing
ruff check src tests ui scripts
mypy src/ftir_workbench --no-incremental
ftir-baseline demo --input-dir data/original --output demo_output
```

导出的 `10_processing_recipe.json`、输入 SHA-256、所有中间矩阵和 manifest 共同构成审计记录。

本地私有数据验证只报告合同、重建、QC、血缘和往返是否通过，不在公开文档或 Git 历史中记录原始谱、样本数量、采集范围、派生统计、数据 fingerprint 或产物 hash。当前 v0.2 工作树尚未发布到 GitHub。
