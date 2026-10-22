# 生物信息读出：固定50/50/10次迭代的唯一当前口径

## 1. 替换范围

正文仅保留 AlloBench、AM、Src。三任务分别训练；不再使用旧 AlloBench→功能迁移、geometry/amino/hidden/full 四拆分、155维组合探针、KRAS 或其他四个变构数据集的旧结果支撑当前正文。旧结果、冻结头、统计图和执行命令已删除。`datasets/allosteric`、`datasets/functional` 的原始来源、坐标映射与原始实验数据保留作标签审计，不是另一套活动结果。

## 2. 数据、任务与划分

| 任务 | 当前评价单位 | 标签行 | 正例 | 同源组件 / 构建体 | 终点 |
|---|---|---:|---:|---|---|
| AlloBench | 383蛋白，全部可评价残基 | 216556 | 7409 | 282组件 | 已知变构位点 vs 同蛋白其他可评价位点 |
| AM | 127蛋白，有实验注释的位置 | 1801 | 839 | 112组件 | 实验变构突变注释分类；不是全蛋白未知位置作负例 |
| Src | 同一P12931的FL/KD，各269位置 | 538 | 不适用 | 2构建体、269共享物理位置 | 位点突变功能影响幅度的排序 |

AlloBench/AM沿用原始同源组件外层5折，成员见各任务 `labels.csv.gz` 的 `outer_fold`。组件来自MMseqs序列一致性≥0.30、双向覆盖≥0.80的连通分量；同组件/同蛋白不跨训练测试。不得重新调用另一版本的 GroupKFold 改写同分数组件分配。AlloBench标签来源见 `datasets/allosteric/raw/AlloBench.csv`、`allobench_positive_positions.csv` 与记录映射；AM见 `raw/supplied/AM/`、`am_position_audit.csv`，保留原有可评价位置筛选，不补造未知标签。

Src保留原始实验全长（FL）和孤立激酶域（KD）构建体，物理位置268–536，各269位置；对应结构为 `src_fl_af2`（536残基）、`src_kd_af2`（276残基），共812个完整结构特征行。两构建体的相同物理位点共同留出：第p位点的折为 `floor((p-1)/10) mod 5`；训练时另排除距任意测试位点≤5的位置。评价包括全部538条实验记录，不按binding score筛选。这里FL/KD是两种构建体，不是两个互作partner，更不是两种独立蛋白。标签来自 `datasets/functional/raw/energy_single_variants.csv`；每构建体、每物理位点对非野生型替换的 `abs(activity_effect)` 取均值。`abs(folding_effect)` 均值为另存的稳定性读出，不作本次主终点或部署输入。原始来源文件为 `sciadv.aea2726_table_s1.txt` / `table_s2.txt`（Science Advances，DOI:10.1126/sciadv.aea2726）；原始表含多突变/终止等记录，不能不经原筛选直接平均。

任务标签与特征的键为 `case_id,residue_index_0`；功能实验另含 `interaction,physical_position`。`matrix_row` 指向当前公共特征行，`source_matrix_row` 保留历史全矩阵索引，只作溯源。

## 3. 四种输入，主干冻结

1. GSBind score：E11原生蛋白结合概率，1维。
2. Structure-only：AlloBench/AM为RSA、8Å接触数、DSSP helix/strand/turn、20种氨基酸one-hot、pLDDT，共26维。Src沿用原功能矩阵的上述26维，并追加三个Cα空间邻域计数，共29维；半径与列顺序见 `FEATURE_SCHEMA.json`，不含binding score和功能标签。**跨任务结构维度不完全一致，不应写成统一26维。**
3. GSBind hidden：原始128维蛋白内中心化表示，单独读出，不拼score/显式结构。中心是完整蛋白中原生score<0.5残基hidden均值；若没有低分残基则取完整蛋白均值。中心化不使用实验标签。
4. PeSTo score：固定PeSTo的标量蛋白结合概率，1维；不训练PeSTo隐藏层探针。来源分数文件哈希随 `SOURCE_INPUT_AUDIT.json` 发布。

特征块保持原训练浮点类型；不能随意把AlloBench/AM的float32 hidden转成float64，否则有限步优化轨迹可能改变。特征完整发布在 `datasets/downstream`，无需历史950MB条件hidden或外部服务器目录即可重新训练全部60个头。E11权重为 `checkpoints/direct128_e11.pt`，张量身份另见原有 `provenance/e11_tensor_identity.csv`。

## 4. 模型与固定迭代

令冻结特征为x，分类探针为 `sigmoid(wᵀ StandardScaler(x)+b)`，使用带L2惩罚的加权LogisticRegression/lbfgs；每折训练侧非加权StandardScaler，权重为 `1/(组件内蛋白数×蛋白可评价残基数)` 再归一至均值1。C沿用此前训练侧选择结果（候选0.1、1、10），每折具体值完整保存在 `SELECTION.json` / `downstream_fixed_checkpoint_manifest.csv`。历史C选择使用训练部分内部划分；**当前固定轮次重训不新增验证数据，不再内层选轮，不做测试集早停。** 分类tol=1e-4；max_iter=50，取第50次接受迭代，提前收敛则保留实际最后一次。

Src为带截距加权Ridge。仅在当前训练折内按case/interaction计算 `rank(functional_magnitude)/n_train`，同位置目标取均值；按parent内构建体数和每构建体训练位置数均衡、权重归一至均值1。StandardScaler也仅用训练折、同权重拟合。alpha沿用此前训练侧内部划分确定值，逐折完整公开，不重新用测试集选择。对加权中心化正规方程 `(XᵀWX+alpha I)w=XᵀW(y-ȳ)` 从零初值做共轭梯度，rtol=1e-10、atol=0、maxiter=10，取第10次，提前收敛保留最后一次；截距恢复为 `ȳ-x̄ᵀw`。

四探针同一任务使用相同折和固定预算；预算是优化器迭代，不是神经网络epoch。1维Src探针可能1次就收敛，不伪造第10个不同模型。当前读出训练无主干更新，无新增hidden/score拼接，无held-out checkpoint选优。训练使用BLAS线程数2以复现原有限步数值轨迹；不同BLAS硬件/版本仍可能带来差异，数值核对失败会显式报错，而不会静默通过。

历史预算50/50/10是在已查看探索性held-out表现后确定，因此这些结果属于探索后固定协议的OOF结果，**不能宣称预注册的独立测试性能**。固定迭代不意味着已经消除了历史模型选择偏差。

## 5. 指标与当前结果

分类AUPR严格为 `auc(recall,precision)`，来自完整precision_recall_curve；不是average_precision_score。每蛋白计算，再同源组件内平均、最后组件等权平均；pooled数值单列，不与macro混用。Src先对完整OOF序列、每case/interaction计算Spearman，再case宏平均；不得用五个fold的rho均值冒充完整OOF rho。未经训练的两种raw score另列补充对照，不翻转方向后冒充原生分数。

| 方法 | AlloBench AUPR | AM AUPR | Src ρ |
|---|---:|---:|---:|
| GSBind score probe | 0.074599 | 0.548396 | 0.388337 |
| Structure-only | 0.076044 | 0.647281 | 0.401681 |
| GSBind hidden128 probe | **0.154688** | **0.654665** | **0.517147** |
| PeSTo score probe | 0.080793 | 0.539383 | 0.146099 |

相对结构基线的绝对提升分别约0.078644、0.007385、0.115466。AM提升较小，不把数值领先直接写成统计显著。当前未发布新协议下的置信区间/显著性检验，不沿用已删除旧模型的p值。AlloBench不同组件与AM可能存在来源交叠，三任务不合并为独立样本总体。

## 6. 可直接用于正文的英文方法

### Frozen-representation biological readouts

We assessed task-specific biological information in frozen GSBind representations using AlloBench, AM, and Src. AlloBench and AM comprised 216,556 residues from 383 proteins and 1,801 annotated positions from 127 proteins, respectively. We retained five homology-component-disjoint folds, with 282 and 112 components, and fitted separate weighted L2-logistic probes for each dataset. Four inputs were compared on identical held-out positions: the native GSBind binding score, structure-only covariates, the original protein-centered 128-dimensional GSBind representation, and the scalar PeSTo binding score. PeSTo hidden representations were not used. Standardization used training data only; regularization strengths were reused from previously recorded training-side selections. Classification probes were evaluated at optimizer iteration 50, or their last available iterate after earlier convergence.

For Src, we predicted position-level mean absolute mutational activity effects in full-length and isolated kinase-domain constructs (269 shared physical positions; 538 construct-position observations). Five interleaved 10-residue-block folds jointly held out corresponding FL/KD positions and excluded training positions within five residues of the test positions. Separate weighted ridge readouts used within-training-fold ranked effects and training-only weighted standardization. We retained conjugate-gradient iteration 10, or the last earlier converged iterate. Classification performance was trapezoidal precision–recall area averaged first within proteins and then across homology components; Src performance was Spearman correlation computed on complete out-of-fold predictions within each construct/interaction and macro-averaged across constructs. Iteration budgets were fixed after exploratory inspection of held-out results; these analyses are not independent prospective test evaluations.

### Results wording

The GSBind hidden-state probe achieved component-macro AUPR values of 0.1547 and 0.6547 on AlloBench and AM, compared with 0.0760 and 0.6473 for structure-only probes and 0.0808 and 0.5394 for PeSTo score probes. On Src, the hidden-state probe attained an out-of-fold Spearman correlation of 0.5171, versus 0.4017 for structure-only features, 0.3883 for the GSBind score probe, and 0.1461 for the PeSTo score probe. These task-specific readouts indicate information accessible beyond scalar binding scores, without establishing zero-shot functional transfer or an independently selected test performance estimate.

## 7. 输出、图1和来源

核对后的参考文献见 `downstream_references.bib`：Maity与Qiao（ACS Omega，2025，10:17973–17982；DOI:10.1021/acsomega.5c01263）；Krapp等的PeSTo（Nature Communications，2023，14:2175；DOI:10.1038/s41467-023-37701-8）；Beltran等的Src实验（Science Advances，2026，12:eaea2726；DOI:10.1126/sciadv.aea2726）。作者/题名/卷页已通过DOI注册元数据核对。AM仍只登记来源表，不伪造出版引用。

`results/downstream/METRICS.csv` 是唯一主表；`FOLD_METRICS.csv` 含实际迭代次数；`DETAILED_METRICS.csv` 含蛋白/interaction层指标；每任务 `OOF_predictions.csv.gz` 是标签、fold、预测共同底表。图1可按“冻结GSBind/PeSTo → 四输入 → 分任务线性头 → 两个组件分类与Src位置回归”绘制，标明50/50/10与分组规则，不能再画跨任务统一Allo训练头。三面板结果图和CSV位于 `figures/downstream_fixed_iterations.*`。

来源登记：AlloBench官方表源标识 `djmaity/allobench/AlloBench.csv`，记录级PubMed/source PDB保存在原始表与映射表；AM为提供的 `AM.csv, AM_add.csv, AM_processed.csv`，当前仓库未提供可核实的独立AM出版DOI，不能编造引用。Src实验DOI如上，GSBind/PeSTo输入及软件出处见 `SOURCE_INPUT_AUDIT.json`、原始raw文件和对应工程。若主文需要完整作者/题名/卷页，请按实际原始论文bibliography补齐，而不要把二次整理表误写成新发表数据集。
