# 论文结构案例：ABL1变构口袋与Src激活段

## 1. ABL1：可用的真实变构案例

**展示结论修正：该例不能称为“精准发现pocket”。** 当前探针逐残基预测变构注释相关分数，不输出口袋边界、连通口袋或配体可结合性。现有连续百分位着色会把中高秩大片区域也画成黄/红，尤其排名参照是全部1130个蛋白位置，而显示的是429个观测激酶片段残基，导致画面整体偏暖。不是所有暖色区域都被判为口袋。

本例核对：27个来源口袋注释位点中，hidden有18个进入全蛋白前10%，14个进入前5%；429个映射残基中前10%有94个，其中仅18个属于显示注释，即18/94。其余76个是“来源口袋注释之外的高排名”，不能未经实验就全部认定为新口袋，也不能全部断言是真实生物学阴性。原生score探针在27个注释位点中没有前10%命中，结构探针有7个。hidden说明注释位点的排序富集改善，但预测并不集中于唯一口袋，不能靠配色制造精准定位结论。AUPR 0.325278是排序指标，不是32.5%的定位准确率；连续百分位不等于预测概率。该初版不宜直接作为“发现口袋”的正文证据，需在候选展示中如实保留注释外高分及固定top-k的precision/recall，必要时更换更清晰的说明性案例，但不得隐藏挑选过程。

### 固定top-k复核版

`abl1_topk_comparison.pdf/png`用灰色背景、红色命中、橙色遗漏、蓝色注释外高分，比较score、structure、hidden三个探针的全蛋白前5%和前10%。所有图同相机，两个工程突变位置始终灰色；保留所有可映射的注释外高分，不只显示口袋周围。

这里改用精确 `k=ceil(N×比例)`，N=1130，即k=57/113。按分数降序，平分按序列位置升序，不沿用百分位≥0.9的边界规则。**因此前述百分位着色的94个含一个序列不一致残基；精确top113在图中可评价高分为93个，其中18命中、75注释外。** 不能将两个分母混用。全蛋白前113另有19个位置不在显示结构内，1个是保持灰色的序列不一致位置。

| 选择 | 方法 | 注释命中/27 | 全蛋白precision | 注释recall | 图内可评价注释外高分 |
| --- | --- | --- | --- | --- | --- |
| top57 | score | 0 | 0.0% | 0.0% | 3 |
| top57 | structure | 3 | 5.3% | 11.1% | 52 |
| top57 | hidden | 14 | 24.6% | 51.9% | 38 |
| top113 | score | 0 | 0.0% | 0.0% | 7 |
| top113 | structure | 7 | 6.2% | 25.9% | 103 |
| top113 | hidden | 18 | 15.9% | 66.7% | 75 |

precision以全蛋白k为分母，不以画面内可见高分数代替；表中precision是相对现有注释，不是实验验证的真实口袋精度。逐残基分类和所有分母保存于 `ABL1_TOPK_RESIDUES.csv`、`ABL1_TOPK_SUMMARY.csv`。此版本用于检验是否有清晰富集，而不是改标签、模型或挑选最佳阈值。两个比例都展示，不能只保留看起来最好的一个。

复现：`tools/prepare_abl1_topk.py` → 开源PyMOL环境运行 `tools/render_abl1_topk.py` → `tools/assemble_abl1_topk.py`。PSE与渲染参数同目录保留。

选择AlloBench内的P00519 / ABL1，展示PDB **5MO4**。该实验结构同时含asciminib（AY7，链A，602）和nilotinib（NIL，链A，601），分别提供变构口袋与ATP竞争抑制剂口袋的空间参照。出处为Wylie et al., Nature 543:733–737 (2017), DOI `10.1038/nature21702`。

建议正文三幅并排：**来源注释的变构口袋 → GSBind score探针 → hidden探针**。三图使用同一结构和相机，两个真实配体一直保留。来源注释为coral，asciminib为紫色，nilotinib为绿色。预测图用完整蛋白可评价残基中的分数百分位，白–黄–红表示低到高；不是跨方法可直接比较的校准概率。该例是经过筛选的说明性案例，不是随机代表或独立验证。

图片：`figures/structural_cases/abl1_case_comparison.pdf/png`。原图和PyMOL会话包括 `abl1_annotation`、`abl1_binding_score`、`abl1_hidden` 的PNG/PSE。

完整蛋白AlloBench AUPR：hidden 0.325278，structure 0.055126，GSBind score 0.014312，PeSTo score 0.013359。这是**该蛋白AlloBench全部来源记录的并集标签**上的梯形AUPR，不是仅5MO4这27个位点的AUPR。真实复合物仅用于展示，预测来自已保存的单体输入，不把配体作为预测输入。

映射审计：5MO4链A共有429个观测CA残基，27个来源变构注释位点全部映射，与工程原始映射逐位点一致。必须强调5MO4是**T334I/D382N工程突变构建体**，对应工程参考序列的315/363位；两个不一致位置在预测着色中保持灰色，不能当作野生型的精确坐标。PDB编号也不能直接当作UniProt编号；完整对应见 `ABL1_RESIDUE_MAPPING.csv`。

## 2. Src：“激活域”应区分激酶结构域和激活环/激活段

使用工程实际输入的全长AF2与KD AF2结构，不重新换一个漂亮构象冒充预测输入。UniProt **P12931人Src**：SH3 84–145，SH2 151–248，激酶结构域270–523。按本序列DFG407–409至APE433–435定义**激活段407–435**，其中Y419为已注释自磷酸化位点；Y530为C端调控磷酸化位置。注意经典Src文献常见其他物种/构建体编号，本文统一用人P12931，不把Y416直接套到当前序列。

`SRC_ANNOTATIONS.json`保存UniProt来源特征摘录、序列定义和下载内容哈希。激活段范围是明确的motif边界定义，不是假称UniProt直接给出了该区间。

两种展示用途：

- **讲清楚区域位置**：`src_activation_context.pdf/png`，全长域背景＋KD的DFG、Y419、APE放大；全长中的低置信无序N端1–83不显示cartoon，以免遮挡。AF2图不能证明活化/失活状态，也没有在模型上伪造pY419修饰。
- **支撑实验读出**：`src_functional_comparison.pdf/png`，同一KD结构上并排映射实验功能幅度、structure-only预测、hidden预测，采用相同相机、各自同构建体内百分位色标。KD的真实完整OOF相关为hidden 0.666977、structure 0.510877。灰色位置是无主终点评价的结构残基，不是功能零值。

**当前正文候选版为单行四幅**：激活段位置参考 → 实验功能敏感性 → structure-only → GSBind hidden。同一KD结构、相机、统一裁剪窗口，后面三幅用相同白–黄–红百分位色标，不为hidden另调色域。图下直接写结构ρ=0.511、hiddenρ=0.667；hidden标题强调色，并标出绝对增益Δρ=+0.156，而不夸大成16%准确率提升。相关系数重新从全部269个位点原始实验幅度和已保存OOF预测计算，非挑选的局部、激活段或top-k指标。元数据与数值底表为 `SRC_CASE_FIGURE.json`、`SRC_CASE_AGREEMENT.csv`，保留FL/KD两构建体；全长结果和整体性能图不删除。由 `tools/assemble_src_functional_case.py` 生成，原总拼图脚本也调用这一版本。

不能保证或暗示每一个激活段位置都被hidden准确识别。图上颜色必须由完整底表生成，不手挑位点调色；当前仅展示功能效应**幅度**，并不预测激活/抑制方向。因此“隐藏层恢复激活机制”“发现激活环”都不是现有结果能够支持的结论。

### 新增：单张结构改善图（当前展示候选）

`src_spatial_gain.pdf/png` 将“哪里更接近实验”直接映射到同一张Src KD结构上，而不要求读者比较三张相似热图。红色表示hidden比structure-only更接近实验，蓝色表示更差，白色接近无变化；灰色为未评价残基。Y419仅作位置参考，不代表该处特别显著。

对全部269个评价位置分别计算实验功能幅度、structure-only预测、hidden预测的构建体内百分位秩（并列取平均秩），令颜色值为 `|rank(structure)-rank(experiment)| - |rank(hidden)-rank(experiment)|`。采用零中心、正负对称的色标，范围取全部位点的最大绝对值，不筛掉退步位置、不平滑、不独立增强红色。157处改善、111处变差、1处持平；平均绝对百分位秩误差由0.233413降至0.185874，减少20.4%。KD的完整OOF Spearman相关为0.510877与0.666977。误差改善图是辅助解释，不是Spearman的逐位置分解，不替代论文主指标，也不能将KD结果当成FL/KD宏平均结果。

数据与定义保存在 `SRC_SPATIAL_GAIN.csv/json`，渲染会话为 `src_spatial_gain_render.pse`。`tools/build_src_spatial_gain.py --action prepare/render/assemble`依次生成数据、开源PyMOL结构和论文拼图。原四幅对照保留作核查，不作为这个单图方案的必需组成。

英文图注：**Spatial distribution of improved agreement with Src functional measurements.** The actual Src kinase-domain AF2 input is colored by the reduction in absolute rank-percentile error of the hidden-feature probe relative to the structure-only probe. Red indicates lower error for the hidden-feature probe; blue indicates higher error. All 269 evaluated positions are retained with a symmetric, zero-centered color scale; gray denotes unevaluated residues. Mean absolute rank-percentile error decreases by 20.4%, while Spearman correlation increases from 0.511 to 0.667. Y419 provides a spatial reference only. These saved out-of-fold predictions retain test-fold checkpoint selection and are exploratory, not independent-test estimates; functional magnitude does not identify activation direction.

## 3. 建议正文取舍

不建议把所有预览全塞进正文。优先取**ABL1口袋对比＋Src激活段上下文/实验映射**两个案例单元，整体性能仍由PR图和Src完整OOF比较支撑。全长上下文或结构基线额外图可放补充。案例不能替代整体统计，且沿用来源测试折选优的探索性边界。

## 4. 可复现文件

`figures/structural_cases/`包含结构输入、逐残基映射、全部预测、注释、文件哈希、渲染元数据与PSE。脚本：

```bash
python tools/prepare_structural_readout_cases.py --experiment-root ../GSbind
python tools/render_structural_readout_cases.py
python tools/assemble_structural_readout_cases.py
```

第一步只整理已有数据，不训练。第二步需可合法使用的PyMOL环境，正文出图使用**开源PyMOL**而不是商业评估版水印输出；不得通过擦除水印伪造无水印图。`PYMOL_RENDER.json`记录实际版本。第三步保留图中文字和色标为PDF矢量元素，蛋白部分为ray-traced位图。

### 英文图注草稿

**Structural illustrations of regulatory-site and functional readouts.** ABL1 site annotations and saved binding-score/hidden-feature readout ranks are mapped onto the same 5MO4 display structure. Asciminib and nilotinib mark the allosteric and ATP-competitive ligand pockets. The engineered T334I/D382N structure is used for visualization only; readouts were computed from the saved monomer input. Sequence-inconsistent residues remain uncolored, and case-level metrics use the full-protein AlloBench annotation union, not only the displayed pocket. For human Src, the sequence-defined DFG407–APE435 activation segment and Y419 are shown on the actual AF2 input, without assigning an experimentally established activation state. Experimental functional-magnitude ranks and complete-OOF probe ranks are mapped using identical cameras and within-construct percentile scales. Functional magnitude does not identify activation direction or establish allosteric causality. Cases are selected illustrations, not independent performance estimates; source readouts retain test-fold checkpoint selection.
