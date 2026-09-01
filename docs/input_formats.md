# FTIR 文本输入格式（v0.2.1）

v0.2.1 只扩大**原始 FTIR 文本表**的输入兼容性。它不会从表头猜测物理单位，也不会改变 v0.2.0 的基线、Prepared、2D-COS、peak-order 或 bundle/project 科学合同。导入时仍须由用户明确选择 `absorbance`、`percent_transmittance` 或 `fraction_transmittance`。

仓库中的 [`examples/import_formats/`](../examples/import_formats/) 全部为纯合成数据，不含实验原始数据。

## 支持矩阵

| 维度 | v0.2.1 支持 | 说明 |
|---|---|---|
| 扩展名 | `.csv`, `.tsv`, `.tab`, `.txt`, `.dpt`, `.asc`, `.dat`, `.xy` | 扩展名只限制文本入口，不决定实际分隔符 |
| 布局 | 单个二列表、单个宽表、多个二列单谱文件 | 宽表第一列是波数，其余每列是一条谱；多文件模式要求每个文件恰好一个强度列 |
| 分隔符 | comma、tab、semicolon、whitespace | `auto` 检查内容；`.tsv`/`.tab` 只提供 tab 提示 |
| 小数点 | dot；decimal comma | decimal comma 仅能与非 comma 分隔符组合 |
| 数值 | 整数、小数、正负号、科学计数法 | 例如 `1`、`.25`、`-0.25`、`1.25e-3` |
| 编码 | UTF-8、UTF-8 BOM、UTF-16 LE/BE BOM、GB18030、CP1252 | 可自动检测，也可显式指定 |
| 文件外层 | blank lines、`#`/`//`/`%` 注释、leading metadata/preamble、可选 header | 跳过的物理行号会被记录 |
| 空列 | 可证明整列为空的首尾边缘列 | 仅在 `trim_empty_edge_columns=True` 时移除并记录数量 |

每种受支持的扩展名、分隔符、编码和布局都由仓库中的合成 fixture 覆盖；发布验收记录保存在 `artifacts/validation/v0.2.1/final/`。

## 表格布局

单个二列、无表头文件：

```text
1800    0.0200
1700    0.0250
1600    0.0400
```

单个宽表文件：

```text
Wavenumber    0 min    5 min    10 min
1800          0.0200   0.0210   0.0220
1700          0.0250   0.0270   0.0290
1600          0.0400   0.0450   0.0520
```

宽表表头中的谱列名称可用于扰动标签；能从标签解析数值时才可能按显式请求排序。表头名称不会用于推断 `A`、`T` 或 `%T`。

多文件序列由若干二列表组成。显式传入路径列表时保留调用方顺序；目录没有可移植的采集顺序，因此目录入口要求显式启用数值扰动排序。每个文件的波数轴必须 point-for-point 完全相同，方向也必须相同。

## 自动检测规则

Probe 与正式读取调用同一个内部 parser，处理顺序为：

```text
read raw bytes and calculate SHA-256
→ reject known binary signatures / inspect BOM and NUL pattern
→ decode text
→ apply explicit skip_rows
→ identify blank and comment lines
→ evaluate delimiter and decimal profiles from file content
→ identify optional header and one rectangular numeric block
→ verify/trim all-empty edge columns when enabled
→ strict float64 and SpectrumSet validation
→ record ImportProbe and metadata
```

### 分隔符与 decimal mark

`auto` 会比较 comma、tab、semicolon 和 whitespace 产生的连续矩形数值块，而不是只看扩展名或第一行。`.tsv` 和 `.tab` 的扩展名提示仅在内容证据等价时用于稳定选择；扩展名与内容不一致会留下 warning。若多个候选产生不同且同样可信的解释，导入失败并要求用户显式选择。

delimiter 为 comma 时，decimal mark 不能也是 comma。delimiter 为 tab、semicolon 或 whitespace 时，可以自动识别或显式选择 decimal comma，例如：

```text
1000,5;0,123
999,5;0,124
```

整数文件在没有小数符号证据时稳定选择 dot，并记录“无 decimal evidence”。`1,234.56` 和 `1.234,56` 之类带千位分隔符的 token 不受支持；请在导入前移除千位分组，不要让 parser 猜测其含义。

### 编码

自动模式先检查 BOM；无 BOM 时识别合理的 UTF-16 交替 NUL-byte pattern，再按确定性 fallback 处理 UTF-8-SIG、GB18030 和 CP1252。若多个编码都能解码却得到不同文本，Probe 会给出 warning；此时应查看表头并显式选择编码。

UTF-16 LE/BE 的 BOM 是可审计证据。明显的 ZIP、OLE/Excel、PDF、图片 signature，异常 NUL 分布或二进制控制字符会在文本解析前被拒绝，不能通过 CP1252 fallback 伪装成谱表。

### Preamble、header 和短文件

默认 comment prefixes 为 `#`、`//`、`%`。自动模式允许 blank/comment lines 以及数值块前的文字 metadata，并记录所有跳过的行号。与数据列数一致且主要为文字的紧邻前一行可作为 header。

为了避免把 metadata 中偶然出现的两个数字误认成光谱，带 preamble 的自动检测数值块至少需要三行。只有 1–2 个数据点时，应显式设置 `skip_rows` 和/或 `header_mode`。若存在多个等长数值块或其他不能唯一解释的内容，也必须显式消除歧义。

## 高级导入选项

Streamlit 的 Import 页提供 “Advanced text import options” 和只读的 “Import Diagnosis”。诊断显示编码、分隔符、小数符号、header、数据块行号、布局、边缘空列、跳过的行和 warning；它不会替用户选择单位，也不会运行基线或 2D-COS。

Python API 保留旧调用，并提供可选对象：

```python
from ftir_baseline import TextImportOptions, probe_spectrum_file, read_spectrum_file

options = TextImportOptions(
    delimiter="semicolon",
    decimal_mark="comma",
    encoding="utf-8",
    header_mode="present",
    skip_rows=0,
    trim_empty_edge_columns=True,
)

probe = probe_spectrum_file("sample.csv", options=options)
data = read_spectrum_file(
    "sample.csv",
    input_unit="absorbance",
    import_options=options,
)
```

旧的 `read_spectrum_file`、`load_spectrum_files`、`load_spectrum_directory`、`read_spectrum` 和 `read_spectrum_series` 调用不要求新增参数。

CLI 的原始文本入口支持同一组核心覆盖项：

```bash
ftir-workbench inspect sample.csv \
  --unit absorbance \
  --delimiter semicolon \
  --decimal-mark comma \
  --encoding utf-8 \
  --header present \
  --skip-rows 0 \
  --trim-empty-edge-columns
```

不提供这些 flags 时仍使用向后兼容的自动模式。需要保留边缘空列用于严格排错时使用 `--no-trim-empty-edge-columns`。

## 严格科学验证

parser 不会为了“成功导入”而修改光谱：

- 不静默排序波数轴或谱序列；
- 不插值、不重采样、不去重、不裁剪；
- 不删除内部空列、内部缺失值或数值块中的坏行；
- 不从 header 自动决定物理单位；
- 不自动翻转多文件波数轴；
- 不宽容处理 `NaN`、`Inf`、ragged rows 或混用的小数格式；
- 不把数值块后的 footer 当成可忽略内容。

错误会尽可能给出源文件、物理行号、列号/列标签、安全截断的 offending token，以及已选 encoding、delimiter 和 decimal mark。多文件序列仍要求波数轴逐点相等；任何一个文件解析失败都会使整个加载失败。

## Provenance

原始 SHA-256 对文件的**原始 bytes**计算，发生在解码和换行处理之前。单文件 metadata 保留旧字段，并增加：

```text
import_parser / import_parser_version
source_format / source_sha256
encoding / encoding_evidence
delimiter / delimiter_name / delimiter_detection_candidates
decimal_mark / decimal_detection_evidence
header_mode / header / skip_rows
skipped_preamble_lines / skipped_comment_lines
trimmed_empty_edge_columns
numeric_block_start_line / numeric_block_end_line
input_layout / import_warnings / import_probe
```

多文件 metadata 另外保存 `import_probe_by_file`、`encoding_by_file`、`delimiter_by_file`、`decimal_mark_by_file`、`preamble_lines_by_file` 和 `trimmed_empty_columns_by_file`，同时继续记录原始/最终文件顺序、排除项和每文件 SHA-256。

## 明确不支持

以下不是普通分隔文本，v0.2.1 不会仅靠增加扩展名白名单来声称支持：

| 类型 | 扩展名/形式 | 处理方式 |
|---|---|---|
| Thermo OMNIC | `.spa`, `.spg`, `.srs` | 先从 OMNIC 导出为文本表 |
| Bruker OPUS | `.0`, `.1`, … | 先从 OPUS 导出为 Data Point Table/ASCII/CSV |
| Galactic/SPC | `.spc` | 使用原软件或专用转换器导出文本 |
| PerkinElmer | `.sp` | 使用仪器软件导出文本 |
| JCAMP-DX | `.jdx`, `.dx`, `.jcamp` | 需要 JCAMP-DX 专用解析器；本版不读取 |
| Excel | `.xls`, `.xlsx` | 先另存为单一、明确的 CSV/TSV 文本表 |
| raw ZIP archives | `.zip` | 解压并从仪器软件导出受支持的文本文件；不要直接上传 ZIP |

OMNIC/OPUS 的菜单名称会随版本变化。通用做法是打开已处理的目标谱，选择 ASCII/text/data-point-table 导出，输出 `x`（wavenumber）和 `y`（强度），关闭千位分组，选择明确的 delimiter 和 decimal mark，并为序列保留可解析且唯一的扰动标签。导出后先用 Import Diagnosis 核对行列数、轴方向、编码和数值块范围，再明确选择输入单位。

## 排错清单

1. 诊断为 ambiguous：显式选择 delimiter、decimal mark、encoding 和 header mode，不要反复改扩展名。
2. 只有 1–2 个点且有 metadata：设置精确的 `skip_rows`，并指定 `header=present` 或 `absent`。
3. decimal-comma 文件：使用 semicolon/tab/whitespace delimiter；comma/comma 组合本身不可判定。
4. 千位分隔符报错：在导出软件中关闭 digit grouping 后重新导出。
5. 内部 missing/ragged row：回到源数据修复；parser 不会删除该点或补值。
6. 多文件轴不一致：在合并前进行有记录的外部重采样；Workbench 不会自动插值或翻转。
7. 表头乱码或 encoding warning：在高级选项中显式选择已知编码并重新诊断。
8. vendor binary/structured 错误：从 OMNIC、OPUS 或对应仪器软件导出受支持文本，而不是重命名文件扩展名。

## 合成示例

| 文件 | 覆盖内容 |
|---|---|
| [`wide_utf8.tsv`](../examples/import_formats/wide_utf8.tsv) | UTF-8、tab、带 header 的宽表 |
| [`single_headerless.tab`](../examples/import_formats/single_headerless.tab) | headerless 二列 tab |
| [`single_whitespace.asc`](../examples/import_formats/single_whitespace.asc) | whitespace 与科学计数法 |
| [`semicolon_decimal_comma.csv`](../examples/import_formats/semicolon_decimal_comma.csv) | semicolon + decimal comma |
| [`utf16le_wide.tsv`](../examples/import_formats/utf16le_wide.tsv) | 真实 UTF-16 LE BOM bytes 的宽表 |
| [`preamble_dat.dat`](../examples/import_formats/preamble_dat.dat) | leading preamble、blank/comment lines、header |

这些文件只用于演示输入合同，不代表真实 FTIR 科学信号，也不应用于评价基线参数。
