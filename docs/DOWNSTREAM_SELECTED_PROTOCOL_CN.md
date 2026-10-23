# 生物信息读出：AlloBench来源最优头迁移AM，Src内部五折最优

日期：2026-10-23。此为唯一活动口径，替换固定50/50/10。旧AM自训练头、固定迭代参考预测与图表已删除。原始来源与实验标签保留溯源；KRAS和其他变构数据集的探索不进入本次主文。

## 1. 两条主线

**变构线：冻结E11表示 → AlloBench五折各自选择来源测试最优checkpoint → 原头直接迁移AM。** 不再单独训练AM，不用AM标签选C、选轮、拟合scaler、反转预测方向或决定集成权重。

**功能线：冻结E11表示 → Src内部五折回归读出 → 每折选该折测试Spearman最好的checkpoint → 汇总完整OOF。** 不是AlloBench→Src迁移，也不是固定第10次。

每条线包含四个对照：GSBind原生score探针、structure-only探针、GSBind中心化hidden128探针、PeSTo原生score探针。各探针仅用自己的来源标签训练/选轮；PeSTo始终只有标量score，不使用隐藏层。每个来源每方法保存五个checkpoint，两个来源共40个；AM没有自训练checkpoint。

## 2. 数据、标签与划分

| 数据 | 标签位置数 | 蛋白/构建体 | 同源组件 | 正例 |
|---|---:|---:|---:|---:|
| AlloBench | 216556 | 383蛋白 | 282 | 7409 |
| AM（迁移全体） | 1801 | 127蛋白 | 112 | 839 |
| AM（来源组件独立子集） | 371 | 26蛋白 | 25 | 175 |
| Src | 538 | 同一P12931的FL/KD各269位置 | 不按蛋白划分 | 连续终点，不适用 |

AlloBench按同源组件分组外层5折；同组件、同蛋白不跨训练/测试。组件来自MMseqs identity≥0.30、双向coverage≥0.80连通分量，沿用公开 `labels.csv.gz` 的原始fold，不因软件版本重分。分类采用全部当前可评价残基；AM仅有实验注释的位置，未知位置不补负例。AlloBench原始来源见 `datasets/allosteric/raw/AlloBench.csv`、阳性位点与记录映射；AM见 `raw/supplied/AM/` 与 `am_position_audit.csv`。

AM `labels.csv.gz` 中保留旧 `outer_fold` 字段仅作溯源，**本协议完全不用这个AM fold训练、选轮或路由**。它不是目标集重新五折训练。主路由由AlloBench来源组件/来源fold决定。

Src输入为 `src_fl_af2`（536结构残基）和 `src_kd_af2`（276），共812特征行。物理位置268–536，两构建体各269实验位置。第p位点fold为 `floor((p-1)/10) mod 5`；同物理位置FL/KD共同留出，训练另外排除距任何测试位点≤5的位置。标签为每构建体每位置非野生型替换 `abs(activity_effect)` 均值，稳定性另存 `abs(folding_effect)` 均值，不作本次主终点或部署输入。来源为 `datasets/functional/raw/energy_single_variants.csv`，原补充表 `sciadv.aea2726_table_s1.txt/table_s2.txt`。FL/KD是同一蛋白的不同构建体，不是两个互作partner或两个独立蛋白。

## 3. 输入与模型

- GSBind score：冻结E11蛋白结合概率，1维。
- Structure-only：分类为RSA、8Å接触数、DSSP helix/strand/turn、20AA one-hot和pLDDT，共26维；Src为同类26维加6/10/14Å三个Cα邻域计数，共29维。Src原列顺序不同，准确顺序见 `datasets/downstream/FEATURE_SCHEMA.json`。不含binding score或功能标签。
- Hidden：原始128维蛋白内中心化表示，单独读出，不拼接score/显式结构。中心来自完整蛋白原生score<0.5残基hidden均值；若无低分残基则用完整蛋白均值，中心化不使用实验标签。
- PeSTo score：冻结PeSTo的标量蛋白结合概率，1维；来源缓存哈希见各任务 `SOURCE_INPUT_AUDIT.json`。

AlloBench是带截距L2加权LogisticRegression/lbfgs；StandardScaler只在当前来源训练折拟合，非加权；残基训练权重为 `1/(组件内蛋白数×蛋白可评价残基数)` 再归一至均值1。C沿用历史训练侧内部划分选择，候选0.1/1/10，逐折保存在 `SELECTION.json`，不是按AM重选。原来源搜索tol=1e-4、max_iter=1500，保存每个接受迭代，并将历史收敛头纳入候选；以该来源留出折组件宏平均梯形AUPR最大者为最佳，平局保留先出现的候选。

Src是加权Ridge：在当前训练fold内按case/interaction计算 `rank(functional_magnitude)/n_train`，同位置目标平均；按parent内构建体和训练位置均衡、权重均值1。Scaler只用训练数据并采用同权重。alpha沿用历史训练侧选择（候选1000/100/10/1），精确值随各fold公开。对加权中心化正规方程做共轭梯度，零初值、rtol=1e-10、atol=0、最多512次，并把历史Ridge解纳入候选；每个接受迭代按该来源测试fold的case/interaction宏平均Spearman选优。不同fold的预测拼接后再计算完整OOF指标，**不是选一个让完整OOF最高的共同迭代，也不是取fold指标均值**。

AlloBench hidden选中迭代为 **28/49/69/44/74**；Src hidden为 **1/7/2/4/7**。有些1维探针历史收敛头与迭代候选并列，由于历史头先入候选，最终保留历史头；其 `selected_iteration=null` 是明确的历史模型候选，不是假造“第0轮”。其余方法完整选中记录与搜索指标保存在40个 `SELECTION.json/CANDIDATE_METRICS.csv` 和 `provenance/downstream_selected_checkpoint_manifest.csv`。

## 4. AM冻结迁移路由

对目标残基所属组件c，定义可用来源头集合 `F(c)={f: c不在AlloBench第f折训练组件中}`，预测为这些头输出概率的等权平均。若c与AlloBench来源有关，通常只使用对应来源留出头；如果是完全新组件，则平均五头。路由不读取AM标签、AM旧fold或AM预测好坏。每蛋白的可用fold、头数、来源独立标记见 `datasets/downstream/AM/TRANSFER_ROUTING.csv`。

主结果使用AM全部1801位置，hidden AUPR **0.693699**。这不是不作排除的五头平均结果，不能混写为0.701284。

另列AM来源组件独立子集：排除出现于AlloBench任意折（包括选轮fold）的组件，剩371位置、26蛋白、25组件，hidden AUPR **0.726231**。该子集所有位置均使用五头均值。不能把这个子集分数冒充AM全体成绩，也不能与全体成绩直接相减解释性能变化。来源独立性限于既有组件规则；未额外排除主干预训练暴露或低双向coverage的局部domain同源。

## 5. 指标与唯一当前结果

分类严格为 `auc(recall,precision)` 梯形AUPR，而非average_precision_score：先每蛋白，再组件内平均，最后组件等权平均。Src为完整OOF序列上每case/interaction的Spearman，先interaction再构建体宏平均。pooled分类数值、逐fold相关、原生raw score另存补充，不与主指标混用。

| 方法 | AlloBench来源选优OOF AUPR | AM冻结迁移AUPR | Src内部选优OOF ρ |
|---|---:|---:|---:|
| GSBind score probe | 0.074599 | 0.509336 | 0.388337 |
| Structure-only | 0.077680 | 0.648387 | 0.400463 |
| GSBind hidden128 | **0.159236** | **0.693699** | **0.512691** |
| PeSTo score probe | 0.080793 | 0.485176 | 0.146099 |

AM来源组件独立子集：hidden **0.726231**、structure **0.675900**、GSBind score **0.500983**、PeSTo score **0.428022**。本次没有新协议下的置信区间或显著性检验，不沿用旧模型的p值，不把数值领先直接写作显著。

**报告边界必须保留：AlloBench与Src用来源留出/测试表现选checkpoint，因此源任务OOF是test-selected探索结果，不是独立测试估计。AM标签虽未训练或选轮，但全体AM含来源相关组件，其来源相似标签可能参与来源选轮，不能把全体AM称完全独立外部测试；来源组件独立子集单列。** 固定保存已选checkpoint不会消除先前选轮使用测试标签的事实。

## 6. 可替换正文的英文方法与结果

### Biological readouts and frozen transfer

We evaluated biological information in frozen GSBind representations along two analysis branches: AlloBench-trained readouts transferred without adaptation to AM, and within-system Src readouts. Four inputs were compared: the native GSBind binding probability, structure-only covariates, the original protein-centered 128-dimensional GSBind representation, and the scalar PeSTo binding probability. PeSTo hidden representations were not used.

AlloBench comprised 216,556 residues from 383 proteins in 282 homology components. We retained five component-disjoint folds and trained weighted L2-logistic readouts with training-only standardization and previously selected training-side regularization strengths. For each source fold and input, the checkpoint maximizing that fold's trapezoidal precision–recall area was retained from the original optimizer history, including the historical converged head as a candidate. The selected AlloBench heads were then applied directly to 1,801 annotated AM positions from 127 proteins, without AM fitting, hyperparameter selection, checkpoint selection, or score-direction adjustment. Target components related to the source used only heads whose source training folds excluded that component; source-disjoint components used an equal-weight mean of all five heads. We additionally reported a source-component-disjoint AM subset comprising 371 positions from 26 proteins in 25 components.

For Src, we used full-length and isolated kinase-domain constructs with 269 shared physical positions (538 construct-position observations). Five interleaved 10-position-block folds jointly held out corresponding FL/KD positions and excluded training positions within five residues of the held-out positions. Weighted ridge readouts used within-training-fold ranked mean absolute mutational activity effects and training-only weighted standardization. Each fold retained the checkpoint with the highest held-out case/interaction-macro Spearman correlation over the conjugate-gradient trajectory and historical ridge solution. Final Src correlation was computed on the concatenated out-of-fold predictions, not averaged across fold correlations.

Classification performance was trapezoidal AUPR computed within proteins and averaged within and then across homology components. AlloBench and Src checkpoint selection used their source held-out/test folds; these source OOF results are explicitly test-selected exploratory estimates. Full AM transfer does not establish complete source independence because source-related components remain, and the source-component-disjoint subset is reported separately.

### Results

The GSBind hidden-state readout achieved a source-selected AlloBench OOF AUPR of 0.1592 and, without AM adaptation, an AM transfer AUPR of 0.6937, compared with 0.6484 for structure-only covariates and 0.4852 for PeSTo score readouts. On the source-component-disjoint AM subset, the hidden-state readout attained an AUPR of 0.7262 versus 0.6759 for structure-only covariates. The within-system Src hidden-state readout achieved a test-selected complete-OOF Spearman correlation of 0.5127, compared with 0.4005 for structure-only covariates, 0.3883 for the GSBind score readout, and 0.1461 for the PeSTo score readout. These analyses support accessible biological information beyond scalar binding probabilities, while distinguishing frozen annotation transfer from source-test-selected within-system evaluation.

## 7. 图1、工程与引用

图1建议明确画两条箭头：`冻结GSBind/PeSTo → 四类输入 → AlloBench来源五折选头 → AM不适配迁移`；另一路 `冻结输入 → Src内部五折回归选头 → 完整OOF功能关联`。AM旁标记来源训练组件排除路由及来源组件独立补充子集；删除“AM单独训练”和“固定50/50/10”，不要把两条线画成统一AlloBench功能迁移。

当前主表 `results/downstream/METRICS.csv`：`scope=all_positions` 是正文；AM的 `scope=source_component_disjoint` 是补充。`FOLD_METRICS.csv`只记录两个训练来源共40头，AM没有伪造训练fold指标。每任务 `predictions.csv.gz`保存全部标签、原始fold和预测；AM另含可用来源头数、来源独立标记。图为 `figures/downstream_selected_transfer.*`。

### 结果分布图：两个小提琴面板和一个Src对比面板

完整图注和面板定义见 `docs/DOWNSTREAM_FIGURE_CN.md`。A–B以组件为单位画小提琴＋白色箱线，分别展示AlloBench的282个组件和AM全部112个组件，不再叠加散点或均值菱形。每个组件值先按蛋白计算梯形AUPR，再在组件内平均；不能把全部残基当作独立样本，也不能把五折均值当作这幅小提琴的原始分布。箱体表示四分位间距，箱内线为中位数。

C以方法为横轴、两构建体平均完整OOF相关为柱高，不叠加构建体点或显著性括号。隐藏层相对结构的平均相关增益仍为+0.112，不改变指标。该比较不是严格条件非冗余信息检验。Src只有两个构建体，不画小提琴。新增D为AlloBench/AM的pooled梯形AUPR分组柱图，与PR曲线面积一致，不是组件宏平均。四图合图为 `figures/downstream_results_four_panel.*`，并提供四幅独立PNG/PDF。不再单画来源组件独立子集、配对增益或逐位点秩散点，以免信息重复。底表在 `figures/source_data/downstream_selected/`；其中保留的来源独立子集、配对差值和位点秩只是补充分析数据，不是新增主图面板，不改变任何评估口径。

`main.py downstream --action replay`只做冻结推断；`--action train`重建两个来源的优化器搜索和test-best选择，使用随仓库提供的历史收敛参考头作为原候选，不冒充从零重训历史超参数选择。`--task AM --action train`只重建来源AlloBench搜索并迁移AM，不拟合AM。输入、40个选中头、40个历史参考头、全部候选指标、来源选择参数均在Git中。数值版本见 `requirements-downstream.txt` / `provenance/downstream_runtime.json`，BLAS线程数2；轨迹不一致会报错，不悄悄替换结果。

参考文献见 `docs/downstream_references.bib`：Maity与Qiao，AlloBench，ACS Omega 2025，DOI:10.1021/acsomega.5c01263；Krapp等，PeSTo，Nature Communications 2023，DOI:10.1038/s41467-023-37701-8；Beltran等，The allosteric landscape of the Src kinase，Science Advances 2026，DOI:10.1126/sciadv.aea2726。AM仍为提供的整理表，当前缺可核验的独立出版书目，不能捏造DOI。
