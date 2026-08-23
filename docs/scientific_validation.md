# Scientific validation

验证由三层组成：精确不变量、已知真值合成谱、真实序列的无真值诊断。三层不能互相替代。

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

## 5. 复现实验

```bash
pytest -q
pytest --cov=ftir_baseline --cov-report=term-missing
ftir-baseline demo --input-dir data/original --output demo_output
```

导出的 `10_processing_recipe.json`、输入 SHA-256、所有中间矩阵和 manifest 共同构成审计记录。
