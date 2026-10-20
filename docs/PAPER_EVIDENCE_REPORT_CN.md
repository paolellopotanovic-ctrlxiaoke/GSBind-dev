# GSbind论文最终证据报告

更新：2026-10-20。当前变构分析仅保留geometry/amino/hidden/full四种设计；本报告由GSbind本地代码与真实冻结读出重算，不依赖旁边的研究工作目录。

公开说明：本报告描述完整主工程证据。Git已提供代码、权重、名单、标签、结果CSV和保存预测；大结构/compact/七条件hidden等外置，下载链接待补，见data_release/README.md。普通克隆用main.py summarize从保存预测重算；完整冻结头重放和训练需要安装外部数据。

## 1. 论文主线

GSbind以原子编码和Gaussian-surface处理、反馈与融合完成结合位点预测。主任务在统一MMseqs两种测试态评价；下游用变构注释/突变位置验证表示的应用价值；SSM220验证高binding score应用；KRAS/Src提供实验功能关联及Gaussian-surface计算依赖。模型机制依赖不自动等同于真实生物信号传递。

## 2. 创新架构与完整工程

固定Gaussian-direct128 E11：128维、vg、三轮co-update、3,355,001参数。AtomEncoder提供原子/残基表示，Gaussian field/pair与多尺度surface构建表面上下文，经core_residue_gate/update与same_level_fusion回传残基，direct head输出结合score。完整主架构包括表面处理、反馈和融合，不仅是pair参数。

20,801训练记录、988 holo与874 APO/AF2官方MMseqs测试记录，原始结构、compact缓存、标签、配置、完整前向和Lightning训练代码均保留。详见ARCHITECTURE.md和ENGINEERING_CONTRACT_CN.md。下游主模型不重训。

## 3. 结合位点主任务：正文AUPR

| panel | model | cases | aupr_mixed |
| --- | --- | --- | --- |
| mmseq_holo_clean1000 | gaussian_direct128 | 988 | 0.791500 |
| mmseq_holo_clean1000 | gpsfun_gpsite_pdb | 988 | 0.554290 |
| mmseq_holo_clean1000 | pesto_official | 988 | 0.737962 |
| mmseq_holo_clean1000 | scannet_ppi_no_msa | 988 | 0.656418 |
| mmseq_state_clean886 | gaussian_direct128 | 874 | 0.728115 |
| mmseq_state_clean886 | gpsfun_gpsite_pdb | 874 | 0.538721 |
| mmseq_state_clean886 | pesto_official | 874 | 0.676970 |
| mmseq_state_clean886 | scannet_ppi_no_msa | 874 | 0.601400 |

四模型使用相同测试记录但训练来源可能不同，是任务级比较。没有BC30或逐case oracle主表。MCC/F1/AUROC等完整保存于results/main，不放正文主表。

### Gaussian-surface截断

| panel | condition | cases | aupr_mixed |
| --- | --- | --- | --- |
| mmseq_holo_clean1000 | baseline | 988 | 0.791500 |
| mmseq_holo_clean1000 | identity | 988 | 0.791500 |
| mmseq_holo_clean1000 | surface_half | 988 | 0.782697 |
| mmseq_holo_clean1000 | surface_zero | 988 | 0.521965 |
| mmseq_state_clean886 | baseline | 874 | 0.728115 |
| mmseq_state_clean886 | identity | 874 | 0.728115 |
| mmseq_state_clean886 | surface_half | 874 | 0.717405 |
| mmseq_state_clean886 | surface_zero | 874 | 0.476520 |

## 4. 变构生物信息：四种唯一设计

| 方法 | 显式输入 | 维度 |
| --- | --- | --- |
| geometry | RSA、8Å堆积、三类DSSP | 5 |
| amino | 20类残基身份one-hot，无序列窗口 | 20 |
| hidden | 冻结模型融合hidden | 128 |
| full | score、geometry、AA、pLDDT、hidden，无额外logit | 155 |

所有四种均为训练侧标准化+逻辑回归。hidden参考集合由native score<0.5定义，逐蛋白中心化；因此不显式输入score不等于完全score-independent。纯几何/AA也是监督线性读出，并非未训练的经验规则。

### 4.1 数据集与生物问题

| 来源 | 本轮可用蛋白 | 标签及角色 |
| --- | --- | --- |
| AlloBench | 383 | 变构位点注释，任务内nested五折 |
| AM | 127 | 变构突变WT位置/供方随机背景，任务内五折及固定迁移 |
| ASBench | 168 | 指定变构配体5Å接触，固定迁移 |
| PASSer_ASD_source | 288 | 作者ASD源表位点，固定迁移 |
| CASBench | 70 | 官方酶条目代表结构的变构位点，固定迁移 |
| ASD_AlloPPI | 21 | PPI调节体系的注释位点，固定迁移 |

AlloBench430原始蛋白/2257记录完整注册，383通过当前映射，其余47原因完整记录；不是430全预测。AM原始128蛋白/1969替换，冲突位置与未追溯记录排除后127蛋白/1801位置，不预测替换强度，不称全部背景为实验中性。PASSer源表不是匿名准备包1049组的精确复现；CASBench91酶条目/2871结构完整保留，按固定字典序代表策略映射70蛋白，不声称全部2871结构均评价。AlloPPI完整50关系/748调节剂注册，但只21蛋白通过坐标映射；不预测PPI调节强度。

### 4.2 任务内五折AUPR

| experiment | subset | proteins | geometry | amino | hidden | full |
| --- | --- | --- | --- | --- | --- | --- |
| AM | all_evaluable | 127 | 0.648789 | 0.640152 | 0.696926 | 0.721639 |
| AlloBench | all_evaluable | 383 | 0.077062 | 0.068598 | 0.164986 | 0.169727 |

AlloBench383蛋白/282组件，AM127蛋白/112组件，各自outer5/inner3。MMseqs identity≥0.30且双向coverage≥0.80连通组件。C=0.1/1/10仅训练内折选择；组件/蛋白均衡权重。所有方法共享相同位置与划分。

### 4.3 固定Allo读出迁移：全队列

| experiment | subset | proteins | geometry | amino | hidden | full |
| --- | --- | --- | --- | --- | --- | --- |
| AlloBench_to_AM | all_evaluable | 127 | 0.672292 | 0.634919 | 0.731538 | 0.742908 |
| AlloBench_to_ASBench | all_evaluable | 168 | 0.071139 | 0.061792 | 0.155350 | 0.161144 |
| AlloBench_to_ASD_AlloPPI | all_evaluable | 21 | 0.046787 | 0.063514 | 0.129414 | 0.145404 |
| AlloBench_to_CASBench | all_evaluable | 70 | 0.072508 | 0.064279 | 0.135557 | 0.145109 |
| AlloBench_to_PASSer_ASD_source | all_evaluable | 288 | 0.070212 | 0.064960 | 0.162038 | 0.170413 |

目标标签不拟合。相关组件用来源对应留出模型；来源组件独立蛋白使用Allo全数据头。全队列不是全部独立外部样本，不将重叠注释来源累加为独立证据。

### 4.4 固定迁移：来源组件独立

| experiment | subset | proteins | geometry | amino | hidden | full |
| --- | --- | --- | --- | --- | --- | --- |
| AlloBench_to_AM | source_component_disjoint | 26 | 0.670549 | 0.641633 | 0.743914 | 0.755877 |
| AlloBench_to_ASBench | source_component_disjoint | 14 | 0.101368 | 0.052964 | 0.151451 | 0.157643 |
| AlloBench_to_ASD_AlloPPI | source_component_disjoint | 9 | 0.034188 | 0.069990 | 0.161933 | 0.186481 |
| AlloBench_to_CASBench | source_component_disjoint | 2 | 0.137799 | 0.075710 | 0.281695 | 0.270422 |
| AlloBench_to_PASSer_ASD_source | source_component_disjoint | 29 | 0.084754 | 0.071572 | 0.180350 | 0.189443 |

### 4.5 全部正文比较与Gaussian-surface依赖

| region | e11_overlap_filter | comparison | mean | lower | upper | p | components | cases | bootstrap | sign_flips | q_bh | experiment | subset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | all | full_minus_geometry | 0.092665 | 0.076934 | 0.109045 | 0.000500 | 282 | 383 | 10000 | 2000 | 0.000692 | AlloBench | all_evaluable |
| all | all | full_minus_amino | 0.101129 | 0.084783 | 0.118271 | 0.000500 | 282 | 383 | 10000 | 2000 | 0.000692 | AlloBench | all_evaluable |
| all | all | full_minus_hidden | 0.004740 | 0.000650 | 0.008939 | 0.027486 | 282 | 383 | 10000 | 2000 | 0.033733 | AlloBench | all_evaluable |
| all | all | S_retain50_minus_baseline | -0.014418 | -0.022051 | -0.007155 | 0.000500 | 282 | 383 | 10000 | 2000 | 0.000692 | AlloBench | all_evaluable |
| all | all | S_retain00_minus_baseline | -0.038888 | -0.052656 | -0.025781 | 0.000500 | 282 | 383 | 10000 | 2000 | 0.000692 | AlloBench | all_evaluable |
| all | all | S_noise101_minus_baseline | -0.032544 | -0.041715 | -0.023783 | 0.000500 | 282 | 383 | 10000 | 2000 | 0.000692 | AlloBench | all_evaluable |
| all | all | S_noise211_minus_baseline | -0.032556 | -0.040348 | -0.025350 | 0.000500 | 282 | 383 | 10000 | 2000 | 0.000692 | AlloBench | all_evaluable |
| all | all | S_noise307_minus_baseline | -0.035981 | -0.044045 | -0.028311 | 0.000500 | 282 | 383 | 10000 | 2000 | 0.000692 | AlloBench | all_evaluable |
| all | all | S_retain00_minus_mean_noise | -0.005194 | -0.015330 | 0.004783 | 0.322839 | 282 | 383 | 10000 | 2000 | 0.370921 | AlloBench | all_evaluable |
| all | all | full_minus_geometry | 0.072850 | 0.036556 | 0.109741 | 0.001000 | 112 | 127 | 10000 | 2000 | 0.017991 | AM | all_evaluable |
| all | all | full_minus_amino | 0.081487 | 0.051530 | 0.113598 | 0.000500 | 112 | 127 | 10000 | 2000 | 0.013493 | AM | all_evaluable |
| all | all | full_minus_hidden | 0.024713 | 0.001781 | 0.047141 | 0.037981 | 112 | 127 | 10000 | 2000 | 0.107946 | AM | all_evaluable |
| all | all | S_retain50_minus_baseline | -0.000159 | -0.016204 | 0.014337 | 0.983008 | 112 | 127 | 10000 | 2000 | 0.983008 | AM | all_evaluable |
| all | all | S_retain00_minus_baseline | -0.030497 | -0.055584 | -0.006553 | 0.015492 | 112 | 127 | 10000 | 2000 | 0.064352 | AM | all_evaluable |
| all | all | S_noise101_minus_baseline | -0.015497 | -0.039494 | 0.006911 | 0.215392 | 112 | 127 | 10000 | 2000 | 0.361456 | AM | all_evaluable |
| all | all | S_noise211_minus_baseline | -0.031627 | -0.055267 | -0.009663 | 0.006997 | 112 | 127 | 10000 | 2000 | 0.044978 | AM | all_evaluable |
| all | all | S_noise307_minus_baseline | -0.013095 | -0.032379 | 0.004030 | 0.175412 | 112 | 127 | 10000 | 2000 | 0.315742 | AM | all_evaluable |
| all | all | S_retain00_minus_mean_noise | -0.010424 | -0.032984 | 0.012589 | 0.360820 | 112 | 127 | 10000 | 2000 | 0.541229 | AM | all_evaluable |
| all | all | full_minus_geometry | 0.070616 | 0.031745 | 0.108757 | 0.001499 | 112 | 127 | 10000 | 2000 | 0.016192 | AlloBench_to_AM | all_evaluable |
| all | all | full_minus_amino | 0.107989 | 0.074598 | 0.142320 | 0.000500 | 112 | 127 | 10000 | 2000 | 0.006747 | AlloBench_to_AM | all_evaluable |
| all | all | full_minus_hidden | 0.011370 | -0.004217 | 0.025585 | 0.155922 | 112 | 127 | 10000 | 2000 | 0.227562 | AlloBench_to_AM | all_evaluable |
| all | all | S_retain50_minus_baseline | -0.004856 | -0.019855 | 0.009900 | 0.512244 | 112 | 127 | 10000 | 2000 | 0.542376 | AlloBench_to_AM | all_evaluable |
| all | all | S_retain00_minus_baseline | -0.033299 | -0.060786 | -0.006565 | 0.020490 | 112 | 127 | 10000 | 2000 | 0.051520 | AlloBench_to_AM | all_evaluable |
| all | all | S_noise101_minus_baseline | -0.016013 | -0.037725 | 0.005875 | 0.149425 | 112 | 127 | 10000 | 2000 | 0.224138 | AlloBench_to_AM | all_evaluable |
| all | all | S_noise211_minus_baseline | -0.013669 | -0.031752 | 0.004744 | 0.143428 | 112 | 127 | 10000 | 2000 | 0.221289 | AlloBench_to_AM | all_evaluable |
| all | all | S_noise307_minus_baseline | -0.020388 | -0.041091 | 0.000581 | 0.054473 | 112 | 127 | 10000 | 2000 | 0.105055 | AlloBench_to_AM | all_evaluable |
| all | all | S_retain00_minus_mean_noise | -0.016610 | -0.043021 | 0.009831 | 0.236382 | 112 | 127 | 10000 | 2000 | 0.303919 | AlloBench_to_AM | all_evaluable |
| all | all | full_minus_geometry | 0.085328 | -0.011413 | 0.178626 | 0.113443 | 25 | 26 | 10000 | 2000 | 0.875134 | AlloBench_to_AM | source_component_disjoint |
| all | all | full_minus_amino | 0.114244 | 0.036738 | 0.188719 | 0.007996 | 25 | 26 | 10000 | 2000 | 0.431784 | AlloBench_to_AM | source_component_disjoint |
| all | all | full_minus_hidden | 0.011964 | -0.041097 | 0.050421 | 0.718641 | 25 | 26 | 10000 | 2000 | 0.958286 | AlloBench_to_AM | source_component_disjoint |
| all | all | S_retain50_minus_baseline | -0.008414 | -0.043148 | 0.023445 | 0.635682 | 25 | 26 | 10000 | 2000 | 0.958286 | AlloBench_to_AM | source_component_disjoint |
| all | all | S_retain00_minus_baseline | -0.037687 | -0.085330 | 0.011860 | 0.145427 | 25 | 26 | 10000 | 2000 | 0.958286 | AlloBench_to_AM | source_component_disjoint |
| all | all | S_noise101_minus_baseline | -0.013414 | -0.045136 | 0.015886 | 0.439780 | 25 | 26 | 10000 | 2000 | 0.958286 | AlloBench_to_AM | source_component_disjoint |
| all | all | S_noise211_minus_baseline | -0.004774 | -0.043367 | 0.040340 | 0.838581 | 25 | 26 | 10000 | 2000 | 0.958286 | AlloBench_to_AM | source_component_disjoint |
| all | all | S_noise307_minus_baseline | -0.002744 | -0.031006 | 0.024336 | 0.844578 | 25 | 26 | 10000 | 2000 | 0.958286 | AlloBench_to_AM | source_component_disjoint |
| all | all | S_retain00_minus_mean_noise | -0.030709 | -0.076094 | 0.012561 | 0.194403 | 25 | 26 | 10000 | 2000 | 0.958286 | AlloBench_to_AM | source_component_disjoint |
| all | all | full_minus_geometry | 0.090005 | 0.066270 | 0.116412 | 0.000500 | 120 | 168 | 10000 | 2000 | 0.001000 | AlloBench_to_ASBench | all_evaluable |
| all | all | full_minus_amino | 0.099352 | 0.075541 | 0.125862 | 0.000500 | 120 | 168 | 10000 | 2000 | 0.001000 | AlloBench_to_ASBench | all_evaluable |
| all | all | full_minus_hidden | 0.005793 | -0.000202 | 0.011924 | 0.062469 | 120 | 168 | 10000 | 2000 | 0.093703 | AlloBench_to_ASBench | all_evaluable |
| all | all | S_retain50_minus_baseline | -0.006566 | -0.018422 | 0.004453 | 0.276362 | 120 | 168 | 10000 | 2000 | 0.363989 | AlloBench_to_ASBench | all_evaluable |
| all | all | S_retain00_minus_baseline | -0.023322 | -0.046113 | -0.002075 | 0.039980 | 120 | 168 | 10000 | 2000 | 0.061683 | AlloBench_to_ASBench | all_evaluable |
| all | all | S_noise101_minus_baseline | -0.027483 | -0.037819 | -0.017888 | 0.000500 | 120 | 168 | 10000 | 2000 | 0.001000 | AlloBench_to_ASBench | all_evaluable |
| all | all | S_noise211_minus_baseline | -0.030321 | -0.041963 | -0.019722 | 0.000500 | 120 | 168 | 10000 | 2000 | 0.001000 | AlloBench_to_ASBench | all_evaluable |
| all | all | S_noise307_minus_baseline | -0.031188 | -0.042189 | -0.021034 | 0.000500 | 120 | 168 | 10000 | 2000 | 0.001000 | AlloBench_to_ASBench | all_evaluable |
| all | all | S_retain00_minus_mean_noise | 0.006341 | -0.011233 | 0.023091 | 0.486757 | 120 | 168 | 10000 | 2000 | 0.602290 | AlloBench_to_ASBench | all_evaluable |
| all | all | full_minus_geometry | 0.056275 | -0.026011 | 0.129036 | 0.215392 | 10 | 14 | 10000 | 2000 | 0.368816 | AlloBench_to_ASBench | source_component_disjoint |
| all | all | full_minus_amino | 0.104679 | 0.042598 | 0.181075 | 0.005497 | 10 | 14 | 10000 | 2000 | 0.134933 | AlloBench_to_ASBench | source_component_disjoint |
| all | all | full_minus_hidden | 0.006192 | -0.024422 | 0.032026 | 0.613193 | 10 | 14 | 10000 | 2000 | 0.807621 | AlloBench_to_ASBench | source_component_disjoint |
| all | all | S_retain50_minus_baseline | -0.016723 | -0.041332 | 0.009270 | 0.225387 | 10 | 14 | 10000 | 2000 | 0.368816 | AlloBench_to_ASBench | source_component_disjoint |
| all | all | S_retain00_minus_baseline | -0.026857 | -0.071108 | 0.020463 | 0.275362 | 10 | 14 | 10000 | 2000 | 0.424845 | AlloBench_to_ASBench | source_component_disjoint |
| all | all | S_noise101_minus_baseline | -0.019266 | -0.042867 | 0.005053 | 0.174413 | 10 | 14 | 10000 | 2000 | 0.324769 | AlloBench_to_ASBench | source_component_disjoint |
| all | all | S_noise211_minus_baseline | -0.025849 | -0.044200 | -0.009385 | 0.019990 | 10 | 14 | 10000 | 2000 | 0.134933 | AlloBench_to_ASBench | source_component_disjoint |
| all | all | S_noise307_minus_baseline | -0.016471 | -0.028891 | -0.006654 | 0.009995 | 10 | 14 | 10000 | 2000 | 0.134933 | AlloBench_to_ASBench | source_component_disjoint |
| all | all | S_retain00_minus_mean_noise | -0.006328 | -0.041458 | 0.030537 | 0.789605 | 10 | 14 | 10000 | 2000 | 0.926928 | AlloBench_to_ASBench | source_component_disjoint |
| all | all | full_minus_geometry | 0.100201 | 0.081944 | 0.119867 | 0.000500 | 211 | 288 | 10000 | 2000 | 0.000843 | AlloBench_to_PASSer_ASD_source | all_evaluable |
| all | all | full_minus_amino | 0.105453 | 0.085465 | 0.126498 | 0.000500 | 211 | 288 | 10000 | 2000 | 0.000843 | AlloBench_to_PASSer_ASD_source | all_evaluable |
| all | all | full_minus_hidden | 0.008375 | 0.003262 | 0.013765 | 0.001499 | 211 | 288 | 10000 | 2000 | 0.002381 | AlloBench_to_PASSer_ASD_source | all_evaluable |
| all | all | S_retain50_minus_baseline | -0.013174 | -0.022776 | -0.004235 | 0.006997 | 211 | 288 | 10000 | 2000 | 0.009942 | AlloBench_to_PASSer_ASD_source | all_evaluable |
| all | all | S_retain00_minus_baseline | -0.037113 | -0.054392 | -0.020475 | 0.000500 | 211 | 288 | 10000 | 2000 | 0.000843 | AlloBench_to_PASSer_ASD_source | all_evaluable |
| all | all | S_noise101_minus_baseline | -0.032541 | -0.042996 | -0.022865 | 0.000500 | 211 | 288 | 10000 | 2000 | 0.000843 | AlloBench_to_PASSer_ASD_source | all_evaluable |
| all | all | S_noise211_minus_baseline | -0.032872 | -0.042636 | -0.023646 | 0.000500 | 211 | 288 | 10000 | 2000 | 0.000843 | AlloBench_to_PASSer_ASD_source | all_evaluable |
| all | all | S_noise307_minus_baseline | -0.037952 | -0.047618 | -0.029275 | 0.000500 | 211 | 288 | 10000 | 2000 | 0.000843 | AlloBench_to_PASSer_ASD_source | all_evaluable |
| all | all | S_retain00_minus_mean_noise | -0.002657 | -0.015563 | 0.010259 | 0.701649 | 211 | 288 | 10000 | 2000 | 0.714888 | AlloBench_to_PASSer_ASD_source | all_evaluable |
| all | all | full_minus_geometry | 0.104690 | 0.045017 | 0.173888 | 0.001499 | 24 | 29 | 10000 | 2000 | 0.008996 | AlloBench_to_PASSer_ASD_source | source_component_disjoint |
| all | all | full_minus_amino | 0.117871 | 0.060743 | 0.184411 | 0.000500 | 24 | 29 | 10000 | 2000 | 0.005397 | AlloBench_to_PASSer_ASD_source | source_component_disjoint |
| all | all | full_minus_hidden | 0.009094 | -0.010939 | 0.031599 | 0.459270 | 24 | 29 | 10000 | 2000 | 0.635913 | AlloBench_to_PASSer_ASD_source | source_component_disjoint |
| all | all | S_retain50_minus_baseline | -0.000612 | -0.019890 | 0.020827 | 0.961019 | 24 | 29 | 10000 | 2000 | 0.961019 | AlloBench_to_PASSer_ASD_source | source_component_disjoint |
| all | all | S_retain00_minus_baseline | -0.024592 | -0.067410 | 0.018319 | 0.271864 | 24 | 29 | 10000 | 2000 | 0.473570 | AlloBench_to_PASSer_ASD_source | source_component_disjoint |
| all | all | S_noise101_minus_baseline | -0.021357 | -0.044996 | -0.000325 | 0.071464 | 24 | 29 | 10000 | 2000 | 0.148426 | AlloBench_to_PASSer_ASD_source | source_component_disjoint |
| all | all | S_noise211_minus_baseline | -0.040200 | -0.064905 | -0.018626 | 0.000500 | 24 | 29 | 10000 | 2000 | 0.005397 | AlloBench_to_PASSer_ASD_source | source_component_disjoint |
| all | all | S_noise307_minus_baseline | -0.024474 | -0.045728 | -0.006815 | 0.012494 | 24 | 29 | 10000 | 2000 | 0.032127 | AlloBench_to_PASSer_ASD_source | source_component_disjoint |
| all | all | S_retain00_minus_mean_noise | 0.004085 | -0.024641 | 0.036484 | 0.821089 | 24 | 29 | 10000 | 2000 | 0.916440 | AlloBench_to_PASSer_ASD_source | source_component_disjoint |
| all | all | full_minus_geometry | 0.072601 | 0.046072 | 0.101628 | 0.000500 | 60 | 70 | 10000 | 2000 | 0.001420 | AlloBench_to_CASBench | all_evaluable |
| all | all | full_minus_amino | 0.080830 | 0.053326 | 0.111619 | 0.000500 | 60 | 70 | 10000 | 2000 | 0.001420 | AlloBench_to_CASBench | all_evaluable |
| all | all | full_minus_hidden | 0.009552 | 0.000616 | 0.019629 | 0.058471 | 60 | 70 | 10000 | 2000 | 0.098669 | AlloBench_to_CASBench | all_evaluable |
| all | all | S_retain50_minus_baseline | -0.008191 | -0.027961 | 0.007628 | 0.421289 | 60 | 70 | 10000 | 2000 | 0.505547 | AlloBench_to_CASBench | all_evaluable |
| all | all | S_retain00_minus_baseline | -0.019158 | -0.050217 | 0.009037 | 0.225387 | 60 | 70 | 10000 | 2000 | 0.304273 | AlloBench_to_CASBench | all_evaluable |
| all | all | S_noise101_minus_baseline | -0.037531 | -0.057966 | -0.020572 | 0.000500 | 60 | 70 | 10000 | 2000 | 0.001420 | AlloBench_to_CASBench | all_evaluable |
| all | all | S_noise211_minus_baseline | -0.030417 | -0.046642 | -0.016085 | 0.000500 | 60 | 70 | 10000 | 2000 | 0.001420 | AlloBench_to_CASBench | all_evaluable |
| all | all | S_noise307_minus_baseline | -0.026994 | -0.041514 | -0.014119 | 0.000500 | 60 | 70 | 10000 | 2000 | 0.001420 | AlloBench_to_CASBench | all_evaluable |
| all | all | S_retain00_minus_mean_noise | 0.012490 | -0.010671 | 0.036284 | 0.312844 | 60 | 70 | 10000 | 2000 | 0.401660 | AlloBench_to_CASBench | all_evaluable |
| all | all | full_minus_geometry | 0.132623 | -0.135855 | 0.401101 | 1.000000 | 2 | 2 | 10000 | 2000 | 1.000000 | AlloBench_to_CASBench | source_component_disjoint |
| all | all | full_minus_amino | 0.194712 | -0.024531 | 0.413954 | 1.000000 | 2 | 2 | 10000 | 2000 | 1.000000 | AlloBench_to_CASBench | source_component_disjoint |
| all | all | full_minus_hidden | -0.011273 | -0.030236 | 0.007690 | 1.000000 | 2 | 2 | 10000 | 2000 | 1.000000 | AlloBench_to_CASBench | source_component_disjoint |
| all | all | S_retain50_minus_baseline | -0.030240 | -0.056499 | -0.003981 | 0.491754 | 2 | 2 | 10000 | 2000 | 1.000000 | AlloBench_to_CASBench | source_component_disjoint |
| all | all | S_retain00_minus_baseline | -0.124833 | -0.238179 | -0.011487 | 0.491754 | 2 | 2 | 10000 | 2000 | 1.000000 | AlloBench_to_CASBench | source_component_disjoint |
| all | all | S_noise101_minus_baseline | -0.039193 | -0.089890 | 0.011505 | 1.000000 | 2 | 2 | 10000 | 2000 | 1.000000 | AlloBench_to_CASBench | source_component_disjoint |
| all | all | S_noise211_minus_baseline | -0.121165 | -0.239022 | -0.003309 | 0.491754 | 2 | 2 | 10000 | 2000 | 1.000000 | AlloBench_to_CASBench | source_component_disjoint |
| all | all | S_noise307_minus_baseline | -0.039351 | -0.077428 | -0.001274 | 0.491754 | 2 | 2 | 10000 | 2000 | 1.000000 | AlloBench_to_CASBench | source_component_disjoint |
| all | all | S_retain00_minus_mean_noise | -0.058263 | -0.102732 | -0.013794 | 0.491754 | 2 | 2 | 10000 | 2000 | 1.000000 | AlloBench_to_CASBench | source_component_disjoint |
| all | all | full_minus_geometry | 0.098618 | 0.047924 | 0.157615 | 0.000500 | 20 | 21 | 10000 | 2000 | 0.013493 | AlloBench_to_ASD_AlloPPI | all_evaluable |
| all | all | full_minus_amino | 0.081890 | 0.030313 | 0.141644 | 0.004998 | 20 | 21 | 10000 | 2000 | 0.053973 | AlloBench_to_ASD_AlloPPI | all_evaluable |
| all | all | full_minus_hidden | 0.015991 | 0.001681 | 0.032807 | 0.066467 | 20 | 21 | 10000 | 2000 | 0.143236 | AlloBench_to_ASD_AlloPPI | all_evaluable |
| all | all | S_retain50_minus_baseline | 0.004856 | -0.016447 | 0.024408 | 0.632184 | 20 | 21 | 10000 | 2000 | 0.682759 | AlloBench_to_ASD_AlloPPI | all_evaluable |
| all | all | S_retain00_minus_baseline | 0.000138 | -0.056685 | 0.051744 | 0.996502 | 20 | 21 | 10000 | 2000 | 0.996502 | AlloBench_to_ASD_AlloPPI | all_evaluable |
| all | all | S_noise101_minus_baseline | -0.038535 | -0.069718 | -0.012058 | 0.008496 | 20 | 21 | 10000 | 2000 | 0.065539 | AlloBench_to_ASD_AlloPPI | all_evaluable |
| all | all | S_noise211_minus_baseline | -0.037130 | -0.067931 | -0.010479 | 0.020990 | 20 | 21 | 10000 | 2000 | 0.084815 | AlloBench_to_ASD_AlloPPI | all_evaluable |
| all | all | S_noise307_minus_baseline | -0.040837 | -0.073875 | -0.013447 | 0.005997 | 20 | 21 | 10000 | 2000 | 0.053973 | AlloBench_to_ASD_AlloPPI | all_evaluable |
| all | all | S_retain00_minus_mean_noise | 0.038972 | -0.008845 | 0.087293 | 0.137931 | 20 | 21 | 10000 | 2000 | 0.219067 | AlloBench_to_ASD_AlloPPI | all_evaluable |
| all | all | full_minus_geometry | 0.152293 | 0.065561 | 0.244089 | 0.003498 | 9 | 9 | 10000 | 2000 | 0.161919 | AlloBench_to_ASD_AlloPPI | source_component_disjoint |
| all | all | full_minus_amino | 0.116492 | 0.032189 | 0.215388 | 0.005997 | 9 | 9 | 10000 | 2000 | 0.161919 | AlloBench_to_ASD_AlloPPI | source_component_disjoint |
| all | all | full_minus_hidden | 0.024549 | -0.003388 | 0.057065 | 0.163918 | 9 | 9 | 10000 | 2000 | 0.354063 | AlloBench_to_ASD_AlloPPI | source_component_disjoint |
| all | all | S_retain50_minus_baseline | -0.001304 | -0.008959 | 0.006125 | 0.742129 | 9 | 9 | 10000 | 2000 | 0.871195 | AlloBench_to_ASD_AlloPPI | source_component_disjoint |
| all | all | S_retain00_minus_baseline | -0.020885 | -0.082960 | 0.034254 | 0.625187 | 9 | 9 | 10000 | 2000 | 0.756822 | AlloBench_to_ASD_AlloPPI | source_component_disjoint |
| all | all | S_noise101_minus_baseline | -0.048716 | -0.103325 | -0.004019 | 0.118941 | 9 | 9 | 10000 | 2000 | 0.310931 | AlloBench_to_ASD_AlloPPI | source_component_disjoint |
| all | all | S_noise211_minus_baseline | -0.060696 | -0.111525 | -0.019528 | 0.012994 | 9 | 9 | 10000 | 2000 | 0.175412 | AlloBench_to_ASD_AlloPPI | source_component_disjoint |
| all | all | S_noise307_minus_baseline | -0.066858 | -0.127553 | -0.017302 | 0.013493 | 9 | 9 | 10000 | 2000 | 0.175412 | AlloBench_to_ASD_AlloPPI | source_component_disjoint |
| all | all | S_retain00_minus_mean_noise | 0.037871 | -0.034517 | 0.111279 | 0.374813 | 9 | 9 | 10000 | 2000 | 0.549212 | AlloBench_to_ASD_AlloPPI | source_component_disjoint |

AP为混合标签蛋白先计算、组件内平均再组件宏平均；10000次组件bootstrap，2000次符号翻转，seed20261020。full相对geometry/amino支持完整融合表示的应用价值，不把全部差值归因hidden单独贡献；full相对hidden反映显式基础特征的补充。新四设计不单独展示无hidden完整结构基线，因此不从当前四行比较新声称严格条件化hidden归因。

表面干预对本轮155维full头重新计算：score/结构/AA/pLDDT固定为native，替换七条件hidden，不重训头。identity等价；保留50%、完全截断与三次同模长噪声分别报告。计算通路依赖不等于真实生物因果。

### 4.6 可复现资产

datasets/allosteric包含504个完整输入、结构/compact、六任务标签、完整430注册审计、七条件hidden、实际MMseqs序列与命中。checkpoints/allosteric仅四种头×两任务×五折及全数据头；src/gsbind/allostery.py与probe_training.py负责冻结重算和独立探针训练。tools/allosteric_data保留下载、源结构映射、同源分组、前向和任务构建代码。所有活动路径本地化。

### 4.7 高结合分数内部的额外变构信息

七组主结果均为完整任务可评价位置，不是低分筛选结果；本次预测与标签逐残基一对一核对。高分固定为native score≥0.5，在同一位置集合比较各方法。

| experiment | proteins | mixed_proteins | geometry | amino | hidden | full | binding_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AM | 114 | 70 | 0.788324 | 0.706531 | 0.770955 | 0.817027 | 0.593806 |
| AlloBench | 383 | 258 | 0.157063 | 0.133391 | 0.224182 | 0.229375 | 0.114803 |
| AlloBench_to_AM | 114 | 70 | 0.807157 | 0.706204 | 0.819183 | 0.837182 | 0.593806 |
| AlloBench_to_ASBench | 168 | 126 | 0.177883 | 0.142012 | 0.234978 | 0.234839 | 0.134736 |
| AlloBench_to_ASD_AlloPPI | 21 | 15 | 0.138568 | 0.144094 | 0.146604 | 0.147056 | 0.104428 |
| AlloBench_to_CASBench | 70 | 49 | 0.155770 | 0.143772 | 0.195658 | 0.215363 | 0.113325 |
| AlloBench_to_PASSer_ASD_source | 288 | 194 | 0.163922 | 0.135781 | 0.230337 | 0.231834 | 0.117920 |

| experiment | comparison | mean | lower | upper | components | cases |
| --- | --- | --- | --- | --- | --- | --- |
| AM | full_minus_geometry | 0.028703 | -0.024526 | 0.082224 | 64 | 70 |
| AlloBench | full_minus_geometry | 0.072312 | 0.048036 | 0.096868 | 190 | 258 |
| AlloBench_to_AM | full_minus_geometry | 0.030025 | -0.026643 | 0.084934 | 64 | 70 |
| AlloBench_to_ASBench | full_minus_geometry | 0.056956 | 0.029057 | 0.084552 | 92 | 126 |
| AlloBench_to_ASD_AlloPPI | full_minus_geometry | 0.008488 | -0.050079 | 0.068462 | 15 | 15 |
| AlloBench_to_CASBench | full_minus_geometry | 0.059594 | 0.016289 | 0.105678 | 43 | 49 |
| AlloBench_to_PASSer_ASD_source | full_minus_geometry | 0.067912 | 0.042577 | 0.093351 | 145 | 194 |

AlloBench、ASBench、PASSer和CASBench高分位置中，full相对geometry分别提升7.23、5.70、6.79和5.96个百分点，组件bootstrap区间均高于零。隐藏表示可以细分已经具有较高结合倾向的位置。AM与AlloPPI的完整分层结果同表保留；跨来源存在重叠，不将七组视为七个互相独立队列。分层比较未做跨全部新比较的多重校正。

## 5. SSM220：高binding score应用

| endpoint | region | condition | aupr | aupr_mixed | auroc | mcc | f1 | precision | recall | accuracy | top10_hits | top10_precision | top10_recall | cases | mixed_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| is_direct_interface | all | S_retain00 | 0.657430 | 0.657430 | 0.836099 | 0.444880 | 0.590221 | 0.455875 | 0.864292 | 0.698929 | 6.727273 | 0.672727 | 0.445360 | 220 | 220 |
| is_direct_interface | all | baseline | 0.819607 | 0.819607 | 0.912964 | 0.606203 | 0.701664 | 0.583443 | 0.913322 | 0.799193 | 8.127273 | 0.812727 | 0.540886 | 220 | 220 |
| is_direct_interface | baseline_high | S_retain00 | 0.732168 | 0.730945 | 0.656347 | 0.149446 | 0.717175 | 0.614382 | 0.901057 | 0.614821 | 7.077273 | 0.707727 | 0.512340 | 220 | 219 |
| is_direct_interface | baseline_high | baseline | 0.839074 | 0.838339 | 0.788852 | 0.000000 | 0.723660 | 0.583443 | 1.000000 | 0.583443 | 8.127273 | 0.812727 | 0.585097 | 220 | 219 |
| primary18_label | all | S_retain00 | 0.786289 | 0.682173 | 0.476068 | 0.002956 | 0.296068 | 0.405519 | 0.262086 | 0.543368 | 4.000000 | 0.428349 | 0.443504 | 107 | 39 |
| primary18_label | all | baseline | 0.844312 | 0.768463 | 0.626417 | 0.092093 | 0.328983 | 0.443605 | 0.294704 | 0.686768 | 4.261682 | 0.454517 | 0.477037 | 107 | 39 |
| primary18_label | baseline_high | S_retain00 | 0.871973 | 0.743945 | 0.454096 | -0.026823 | 0.471611 | 0.493213 | 0.480339 | 0.468615 | 4.273684 | 0.496725 | 0.690081 | 95 | 28 |
| primary18_label | baseline_high | baseline | 0.925829 | 0.851658 | 0.706778 | 0.000000 | 0.526018 | 0.499640 | 0.589474 | 0.499640 | 4.473684 | 0.516725 | 0.715357 | 95 | 28 |
| relaxed1_label | all | S_retain00 | 0.776537 | 0.670391 | 0.471478 | 0.003229 | 0.260464 | 0.352331 | 0.228253 | 0.552994 | 3.637097 | 0.372849 | 0.407750 | 124 | 40 |
| relaxed1_label | all | baseline | 0.840407 | 0.764600 | 0.643518 | 0.082454 | 0.289693 | 0.386220 | 0.265416 | 0.673337 | 3.887097 | 0.397849 | 0.460825 | 124 | 40 |
| relaxed1_label | baseline_high | S_retain00 | 0.860329 | 0.725475 | 0.427156 | -0.031801 | 0.406006 | 0.423962 | 0.409618 | 0.439195 | 3.945946 | 0.428378 | 0.657178 | 111 | 29 |
| relaxed1_label | baseline_high | baseline | 0.917787 | 0.838409 | 0.709685 | 0.000000 | 0.454261 | 0.431453 | 0.513514 | 0.431453 | 4.135135 | 0.447297 | 0.700627 | 111 | 29 |
| sensitive_ge_1 | all | S_retain00 | 0.771244 | 0.701623 | 0.506486 | 0.028140 | 0.522038 | 0.713557 | 0.463792 | 0.495563 | 7.540909 | 0.754091 | 0.187015 | 220 | 161 |
| sensitive_ge_1 | all | baseline | 0.806823 | 0.748030 | 0.612835 | 0.114860 | 0.503066 | 0.757200 | 0.428511 | 0.497399 | 7.954545 | 0.795455 | 0.196546 | 220 | 161 |
| sensitive_ge_1 | baseline_high | S_retain00 | 0.835698 | 0.728771 | 0.518700 | -0.017193 | 0.741782 | 0.753762 | 0.803408 | 0.679708 | 7.636364 | 0.763636 | 0.440498 | 220 | 126 |
| sensitive_ge_1 | baseline_high | baseline | 0.864834 | 0.776869 | 0.654050 | 0.000000 | 0.804756 | 0.757200 | 0.945455 | 0.757200 | 7.954545 | 0.795455 | 0.452911 | 220 | 126 |
| sensitive_ge_2 | all | S_retain00 | 0.724922 | 0.666090 | 0.510221 | 0.033472 | 0.506453 | 0.663475 | 0.468141 | 0.502538 | 7.154545 | 0.715455 | 0.193603 | 220 | 173 |
| sensitive_ge_2 | all | baseline | 0.774529 | 0.726307 | 0.617779 | 0.136224 | 0.500235 | 0.717623 | 0.444109 | 0.519182 | 7.722727 | 0.772273 | 0.212391 | 220 | 173 |
| sensitive_ge_2 | baseline_high | S_retain00 | 0.800055 | 0.713182 | 0.523441 | -0.029747 | 0.715338 | 0.711438 | 0.799297 | 0.647918 | 7.245455 | 0.724545 | 0.440556 | 220 | 145 |
| sensitive_ge_2 | baseline_high | baseline | 0.845304 | 0.778091 | 0.666919 | 0.000000 | 0.777872 | 0.717623 | 0.945455 | 0.717623 | 7.722727 | 0.772273 | 0.466982 | 220 | 145 |

SSM220保留完整220输入、原始替换测量和标签构建。只使用binding score，不加入SSM hidden分类器。primary18保留质量/区间过滤及足够有效替换，未知不作为阴性。不能以此声称hidden超越埋藏度。

## 6. KRAS/Src：全位置实验功能应用与表面依赖

主分析覆盖全部实验位置，不先按score或非接触条件筛选。KRAS为187个唯一位置、六partner共1118条记录；Src两个构建体各269个位置，共538条记录。主模型和读出均冻结，未使用目标功能标签拟合。

| dataset | method | rho | top10_effect_mean | top10_fold_enrichment | top20_effect_mean | top20_fold_enrichment |
| --- | --- | --- | --- | --- | --- | --- |
| KRAS | basic_probe | 0.187373 | 0.407387 | 1.259126 | 0.323220 | 0.999271 |
| SRC | basic_probe | 0.139094 | 0.991882 | 1.272640 | 0.981677 | 1.257646 |
| KRAS | burial | 0.303495 | 0.544821 | 1.705356 | 0.388822 | 1.210604 |
| SRC | burial | 0.370039 | 0.720105 | 0.923639 | 0.964372 | 1.237143 |
| KRAS | current_full | 0.330615 | 0.539018 | 1.687240 | 0.514614 | 1.606335 |
| SRC | current_full | 0.358122 | 1.419240 | 1.824978 | 1.284483 | 1.644403 |
| KRAS | current_hidden | 0.364367 | 0.464302 | 1.446847 | 0.510348 | 1.589793 |
| SRC | current_hidden | 0.362412 | 1.795674 | 2.302890 | 1.325425 | 1.696461 |
| KRAS | full_probe | 0.427150 | 0.592040 | 1.848508 | 0.505707 | 1.576223 |
| SRC | full_probe | 0.368311 | 1.943697 | 2.494177 | 1.375519 | 1.763081 |
| KRAS | packing | 0.341765 | 0.509257 | 1.592543 | 0.426463 | 1.331151 |
| SRC | packing | 0.323612 | 0.954753 | 1.224677 | 0.958481 | 1.228074 |
| KRAS | raw_score | -0.128343 | 0.488159 | 1.483951 | 0.422750 | 1.308493 |
| SRC | raw_score | -0.432925 | 0.406384 | 0.516859 | 0.427333 | 0.543164 |

### 6.1 Gaussian-surface通路依赖

| version | condition | region | target | method | dataset | rho | all_effect_mean | top10_budget | top10_effect_mean | top10_fold_enrichment | top10_target_top_hits | top10_target_top_precision | top20_budget | top20_effect_mean | top20_fold_enrichment | top20_target_top_hits | top20_target_top_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fresh_E11 | S_retain00 | all | functional_magnitude | full_probe | KRAS | 0.169168 | 0.321926 | 10.000000 | 0.493980 | 1.534232 | 1.166667 | 0.116667 | 37.666667 | 0.499044 | 1.550919 | 14.500000 | 0.385254 |
| fresh_E11 | S_retain00 | all | functional_magnitude | full_probe | SRC | 0.171064 | 0.781744 | 10.000000 | 1.563997 | 2.002343 | 1.500000 | 0.150000 | 54.000000 | 1.131826 | 1.450827 | 21.000000 | 0.388889 |
| fresh_E11 | S_retain00 | all | stability_magnitude | full_probe | KRAS | -0.192937 | 0.511001 | 10.000000 | 0.346566 | 0.678203 | 0.000000 | 0.000000 | 37.666667 | 0.444575 | 0.869994 | 3.500000 | 0.092935 |
| fresh_E11 | S_retain00 | all | stability_magnitude | full_probe | SRC | -0.145014 | 0.693645 | 10.000000 | 0.680757 | 0.990050 | 0.000000 | 0.000000 | 54.000000 | 0.580223 | 0.844538 | 6.500000 | 0.120370 |
| fresh_E11 | S_retain50 | all | functional_magnitude | full_probe | KRAS | 0.343745 | 0.321926 | 10.000000 | 0.575765 | 1.789220 | 1.500000 | 0.150000 | 37.666667 | 0.545129 | 1.695587 | 17.000000 | 0.451399 |
| fresh_E11 | S_retain50 | all | functional_magnitude | full_probe | SRC | 0.322394 | 0.781744 | 10.000000 | 1.880180 | 2.410864 | 3.000000 | 0.300000 | 54.000000 | 1.297957 | 1.664564 | 26.000000 | 0.481481 |
| fresh_E11 | S_retain50 | all | stability_magnitude | full_probe | KRAS | -0.000390 | 0.511001 | 10.000000 | 0.484000 | 0.947172 | 1.000000 | 0.100000 | 37.666667 | 0.495974 | 0.970586 | 6.000000 | 0.159317 |
| fresh_E11 | S_retain50 | all | stability_magnitude | full_probe | SRC | -0.015990 | 0.693645 | 10.000000 | 0.772597 | 1.108806 | 0.500000 | 0.050000 | 54.000000 | 0.647819 | 0.943452 | 7.500000 | 0.138889 |
| fresh_E11 | baseline | all | functional_magnitude | full_probe | KRAS | 0.427150 | 0.321926 | 10.000000 | 0.592040 | 1.848508 | 1.333333 | 0.133333 | 37.666667 | 0.505707 | 1.576223 | 17.166667 | 0.456022 |
| fresh_E11 | baseline | all | functional_magnitude | full_probe | SRC | 0.368311 | 0.781744 | 10.000000 | 1.943697 | 2.494177 | 2.500000 | 0.250000 | 54.000000 | 1.375519 | 1.763081 | 28.500000 | 0.527778 |
| fresh_E11 | baseline | all | stability_magnitude | full_probe | KRAS | 0.181468 | 0.511001 | 10.000000 | 0.839539 | 1.642954 | 2.000000 | 0.200000 | 37.666667 | 0.734679 | 1.437726 | 14.500000 | 0.384898 |
| fresh_E11 | baseline | all | stability_magnitude | full_probe | SRC | 0.063918 | 0.693645 | 10.000000 | 0.611121 | 0.892695 | 0.000000 | 0.000000 | 54.000000 | 0.726782 | 1.058112 | 10.000000 | 0.185185 |

原完整冻结读出的全位置功能rho：KRAS 0.4272、Src 0.3683；完全截断后为0.1692和0.1711，下降0.2580和0.1972。下降区间分别为[0.0758,0.4549]与[0.1288,0.2650]。所有条件使用统一GPU后端的新E11前向，固定native score/基础协变量，仅替换隐藏表示。

### 6.2 控制结构与折叠后的实验功能关联

| dataset | region | method | controls | estimate | lower | upper | bootstrap | block_length | refit_controls_each_draw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KRAS | all | full_probe | score_geometry_AA_plddt | 0.307566 | 0.125312 | 0.463511 | 10000 | 10 | True |
| KRAS | all | full_probe | score_geometry_AA_plddt_folding | 0.259481 | 0.073794 | 0.418631 | 10000 | 10 | True |
| SRC | all | full_probe | score_geometry_AA_plddt | 0.348077 | 0.213078 | 0.457189 | 10000 | 10 | True |
| SRC | all | full_probe | score_geometry_AA_plddt_folding | 0.348670 | 0.213340 | 0.455322 | 10000 | 10 | True |

控制score、几何、AA、pLDDT后关联仍为0.3076/0.3481；进一步控制折叠后为0.2595/0.3487。前10个候选的实验功能效应均值富集1.85×/2.49×。这两部分分别体现补充功能信息与实验优先排序价值。

两个体系、三个结构输入；先assay、后case、再parent汇总，不把partner/构建体当作独立家族。full_probe为原冻结应用头；current_hidden/current_full补充列出当前正式Allo读出的迁移结果，不混称。全部高低分与接触分层、预算新增/丢失、读出来源和后端审计在results/functional及FUNCTIONAL_PROTOCOL_CN.md。

### 6.3 相近结合分数内的功能信息分离

每个具体实验按native binding score固定分为五个等频组，在组内评价功能排序，先组、后assay、case和parent平均。没有重训主模型或读出。该比较直接回答：在结合倾向相近的位置中，隐藏读出是否还能区分实验功能影响？

| dataset | comparison | estimate | lower | upper | iterations |
| --- | --- | --- | --- | --- | --- |
| KRAS | rho | 0.307952 | 0.133884 | 0.448261 | 10000 |
| KRAS | full_probe_minus_raw_score | 0.276273 | 0.083171 | 0.440402 | 10000 |
| KRAS | full_probe_minus_burial | 0.116539 | -0.120903 | 0.317980 | 10000 |
| KRAS | full_probe_minus_packing | 0.055912 | -0.123851 | 0.220794 | 10000 |
| SRC | rho | 0.374870 | 0.287446 | 0.441446 | 10000 |
| SRC | full_probe_minus_raw_score | 0.453701 | 0.343134 | 0.564486 | 10000 |
| SRC | full_probe_minus_burial | 0.305819 | 0.119687 | 0.494064 | 10000 |
| SRC | full_probe_minus_packing | 0.289805 | 0.087622 | 0.474167 | 10000 |

原完整冻结读出组内rho为KRAS 0.3080和Src 0.3749；相对直接使用score分别增加0.2763 [0.0832,0.4404]和0.4537 [0.3431,0.5645]。Src相对burial/packing的增加也有正区间；KRAS相对这两个结构指标的区间跨零。此处与全协变量偏相关共同支持额外功能信息。

另外计算观测折叠效应较低一半的功能排序、功能减折叠百分位秩差，全部结果保留于results/functional_separation。它们回答不同实验问题，不把较低折叠影响等同于完全无折叠影响，也不按哪个方法变正来选择阈值。详见FUNCTIONAL_SEPARATION_CN.md。

## 7. 图表、附录与复算

正文性能表只放AUPR，MCC/F1/AUROC及阈值在附录。四图结构见FOUR_FIGURE_PLAN_CN.md；python main.py figures生成可复算基础图及source_data。python main.py summarize --task functional重算全位置功能读出；python main.py functional-statistics重算序列块配对区间；python main.py allosteric-strata重算全位置/高低分变构统计。python main.py evidence --task allosteric重放四方法与完整来源迁移；python main.py fit-probes --cohort both --output runs/refit_probes只训练独立探针。

当前活动工程覆盖原112→171/旧HGB变构展示，不再同时维护旧论文口径；主任务、SSM220和KRAS/Src不同端点保留。代码与数据完整不代表各来源自动取得公开再分发许可。
