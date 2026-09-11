# 普通光谱后处理合成示例

所有示例由数学函数和固定随机 seed=3101 生成，无实验原始数据、无仪器性能含义。

在仓库根目录运行：

```bash
.venv/bin/python scripts/generate_v031_examples.py
PYTHONPATH=src:. .venv/bin/python -m streamlit run ui/streamlit_app.py
```

生成位置 `outputs/ordinary-postprocessing-synthetic/` 已被 Git 忽略。脚本拒绝覆盖同名旧文件。

1. 导入前选择“普通光谱模式”；同时上传 `synthetic_wide.csv` 与 `synthetic_other_axis.csv`。宽表拆为 A/B 两项，另一个文件为 C；各自保留真实轴。
2. 每条完成粗调，再明确确认细调或跳过细调。没有作出细调决定的条目不能后处理。
3. 在后处理页启用平滑，逐谱预览并确认；也可复制配方到选中项、批量预览，再明确批量确认。
4. 归一化来源先选 B，使用全范围最大正峰高或面积，预览并确认 N_B。再明确切换来源 S，独立预览并确认 N_S。
5. 确认 S 不改变归一化来源。编辑草稿不覆盖正式快照；正式导出使用已确认结果。
6. B/S/N_B/N_S 可分别导出；全部导出包含四分支及必要父级。A/B 同轴可合并宽表，含 C 时禁止伪对齐宽表，使用逐谱 CSV 的 ZIP。
7. 保存普通工作区，重新上传恢复。确认版、草稿、父级和过期状态保留；预览不保存，确认未提交草稿前要重新预览。

`synthetic_spike.csv` 可演示 Moving Average 与 Normalize 的顺序：3 点 reflect 平滑后再最大峰高归一化，会形成连续三个 1。Median 可能削弱真实窄峰；该变化不自动代表成功去噪。

操作细节与限制见 [普通后处理说明](../../docs/ordinary_postprocessing.md)。实际测试结果在 `artifacts/validation/v0.3.1/`，不能以示例文字替代验收结果。
