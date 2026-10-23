# 两张示意图说明三项下游任务

**论文排版版**：`figures/downstream_tasks_publication.pdf`，同名SVG可编辑、PNG可预览，由 `tools/build_downstream_publication_schematic.py` 生成。两个紧凑面板a/b，移除说明框和流程文字；下文旧biology/workflow文件仅为内部讲解，不建议放正文。所有蛋白形状、箭头和突变效应网格是概念示意，不是测量数据。图a解释蛋白接触与变构调控的不同标签，并标识AlloBench来源读出到AM迁移；图b解释Src构建体及位置级功能幅度，不能解释为新的实验测量或已证明的变构通路。

论文图注建议：**Downstream regulatory and functional readout tasks.** (a) Conceptual distinction between a protein-binding interface and an allosteric regulatory site. Binding-trained residue representations are evaluated for recovery of AlloBench site annotations and frozen transfer to AM mutation-site annotations. The dashed arrow illustrates regulatory coupling, not an inferred communication pathway. (b) Full-length Src and its kinase-domain construct provide experimentally derived mutation-effect annotations. Absolute activity effects are averaged across measured non-wild-type substitutions at each position to define functional sensitivity, which is evaluated through a separately fitted Src readout. Shapes and effect-grid colors are schematic, not measured coordinates or values. Functional sensitivity is not a binary allosteric-site label. Dataset counts, folds, source-test checkpoint selection, AM routing and performance metrics are reported in Methods and the quantitative result panels, not embedded as explanatory paragraphs in the artwork.

两图是可编辑的矢量概念图，不是实验结构展示，也不是新结果。生成命令：`python tools/build_downstream_task_schematics.py`。输出SVG、PDF和PNG。

## 图一：结合、变构与功能敏感性不是同一种标签

文件：`figures/downstream_task_biology.*`。

在同一蛋白概念示意上，蓝色标注与蛋白伙伴直接接触的界面，紫色标注变构调节扰动与功能位点的耦联，红色星号表示可以发生突变的不同位置。虚线仅解释变构概念，不代表模型推断或实验确定的信号路径。

- **原始结合任务**：预测残基是否位于蛋白–蛋白接触界面，回答“哪里接触”。不是泛指所有配体结合，也不是变构标签。
- **AlloBench**：以公开变构残基映射为正例，当前可评价蛋白位置中其余位置为操作性背景，回答“能否找回变构位点注释”。背景不等于实验已证明没有变构作用。AlloBench的变构口袋可能结合小分子；这与蛋白–蛋白接触标签不是同一概念。
- **AM**：使用给定突变数据的二元注释，按野生型位置映射；同位点替换标签冲突时排除，未知位置不补负例。不能把AM画成有全蛋白稠密阴性标签的口袋数据，更不能为其补造未经确认的原始实验机制。其出版来源仍待完全核实。
- **Src**：预测位置突变的功能影响幅度，回答“哪些位置的变化更能影响功能”。该连续终点不是变构位点的二元真值。催化、结构稳定性、直接接触、调控等因素可能影响功能，因此不能将所有高影响位置都标为远端变构位点。当前代码没有把催化位置、接触位置或稳定性相关位置统一排除。

结合与变构不能画成绝对互斥的两个集合；同样，不应将界面到活性中心的箭头标成已被GSBind发现的传播路径。三类标签可有重叠，但当前没有系统量化这种重叠。

## 图二：这三张结果图分别评价什么

文件：`figures/downstream_task_workflow.*`。

共同起点是结合任务训练后的冻结GSBind表示，不是直接拿结合概率当变构概率。四种输入对照分别为hidden、GSBind score、显式结构、PeSTo score；各自训练线性读出。

| 结果面板 | 输入与任务 | 标签 | 读出训练与评估 |
| --- | --- | --- | --- |
| A AlloBench PR | 残基变构注释分类 | 383蛋白、216556位置，7409正例 | AlloBench来源组件五折logistic读出；完整来源OOF pooled梯形AUPR |
| B AM PR | 不在AM适配的跨数据集位点注释迁移 | 127蛋白、1801已注释位置，839正例 | 复用AlloBench头；排除包含目标组件的来源训练头；不使用AM标签拟合或选轮 |
| C Src功能 | 冻结表征对实验功能敏感性的回归/排序 | 全长与激酶结构域，各269位置 | 单独拟合Src ridge读出；对应位置共同留出并purge±5邻居；完整OOF Spearman后平均两构建体 |

Src第i位点的主终点为已测非野生型替换的 `mean(abs(activity_effect))`；该字段来自原研究实验数据的处理/建模结果，不是GSBind自己开展的新实验。它是位置级聚合，不预测每一种突变，不区分激活与失活方向。`folding_effect`另存，但不是本次主终点，也不表示已经做了功能效应对稳定性效应的条件校正。

AlloBench与Src的来源头采用测试折选优，属探索性结果；AM不训练、不按目标标签选优，但相关来源组件可能存在来源选优标签暴露。示意图不改变这些既有边界。

## 论文叙事

建议总标题：**Beyond binding interfaces: regulatory-site annotations and experimental functional sensitivity**。

三图的共同问题：**为识别结合界面学习的残基表示，是否也能用于识别调控位点注释，并预测实验测得/推导的功能敏感性？**

结果应分两层描述：AlloBench→AM说明变构相关注释的来源读出与迁移；Src说明在独立定义的实验功能终点上，隐藏层读出优于当前结构/分数基线。Src不是AlloBench头的迁移，也不证明高影响位点都是变构位点。不能将跨任务可读出性等同于发现新的因果变构通路或严格条件非冗余的信息增量。

## 可核对来源

- 结合与当前下游协议：`docs/DOWNSTREAM_SELECTED_PROTOCOL_CN.md`。
- AlloBench材料：Maity & Qiao, ACS Omega 2025，DOI `10.1021/acsomega.5c01263`；本工程标签见 `tools/allosteric_data/materialize.py`。
- AM映射与排除规则：`tools/allosteric_data/materialize.py`、`datasets/allosteric/am_position_audit.csv`。未确认的出版引用不补造。
- Src：Beltran et al., Science Advances 2026，DOI `10.1126/sciadv.aea2726`；作者源代码仓库 `lehner-lab/src_allostery`；本工程终点重建见 `src/gsbind/labels.py`。
