# GSbind 生物信息增量：主文结果与图表方案

日期：2026-10-20。基于 Gaussian-direct128 E11；全位置功能应用与高分变构分析已纳入当前正式工程，覆盖原低分主分析。

## 1. 主文故事

**为结合位点预测训练的 GSbind，不仅输出结合分数，其内部表示还支持变构位点读出和实验功能影响排序；Gaussian-surface 通路对 KRAS/Src 的功能排序有明显贡献。**

主文按“整体变构能力 → 高结合分数内部的功能分辨 → 实验体系的候选排序 → Gaussian-surface 通路依赖”展开，不再把低分非接触残基作为唯一入口。

## 2. 其他变构数据集是否只分析低分？

**不是。七组正式主表均使用各任务完整可评价位置。** 本次将预测表与任务标签逐残基一对一核对，位置集合完全一致，并以 1e-12 容差复现原四方法主表。

| experiment | proteins | complete_positions | low_positions | high_positions | positives |
| --- | --- | --- | --- | --- | --- |
| AM | 127 | 1801 | 1234 | 567 | 839 |
| AlloBench | 383 | 216556 | 152848 | 63708 | 7409 |
| AlloBench_to_AM | 127 | 1801 | 1234 | 567 | 839 |
| AlloBench_to_ASBench | 168 | 87832 | 63051 | 24781 | 2989 |
| AlloBench_to_ASD_AlloPPI | 21 | 12509 | 8448 | 4061 | 236 |
| AlloBench_to_CASBench | 70 | 35619 | 25837 | 9782 | 1252 |
| AlloBench_to_PASSer_ASD_source | 288 | 145870 | 104461 | 41409 | 4774 |

AM 的完整位置指数据集提供的实验变异 WT 位置，不是把未测量的整条序列当作负例。AlloBench 与 AM 的任务内读出采用原分组交叉验证；AlloBench_to_* 为原冻结迁移读出，本次没有额外训练。低分位置可作为隐藏层中心化的参考集合，但这不等于评价只取低分。

## 3. 全位置 AUPR：主结果

| experiment | geometry | amino | hidden | full | binding_score |
| --- | --- | --- | --- | --- | --- |
| AM | 0.6488 | 0.6402 | 0.6969 | 0.7216 | 0.5666 |
| AlloBench | 0.0771 | 0.0686 | 0.1650 | 0.1697 | 0.0817 |
| AlloBench_to_AM | 0.6723 | 0.6349 | 0.7315 | 0.7429 | 0.5666 |
| AlloBench_to_ASBench | 0.0711 | 0.0618 | 0.1554 | 0.1611 | 0.0998 |
| AlloBench_to_ASD_AlloPPI | 0.0468 | 0.0635 | 0.1294 | 0.1454 | 0.0795 |
| AlloBench_to_CASBench | 0.0725 | 0.0643 | 0.1356 | 0.1451 | 0.0870 |
| AlloBench_to_PASSer_ASD_source | 0.0702 | 0.0650 | 0.1620 | 0.1704 | 0.0823 |

geometry 为原五维几何对照，amino 为氨基酸对照，hidden 为独立隐藏层探针，full 为原完整基础信息＋隐藏层探针。binding_score 为最终结合分数直接排序，作为额外解释性比较。主表维持四方法；MCC、阈值混淆矩阵和逐蛋白结果保留在数据表中。

AUPR 使用原协议的 aupr_mixed：先在同时含正负标签的蛋白上计算 AP，再先组件内、后组件间平均。七组实验存在蛋白和来源交叠，不将其包装为七个互相独立的盲测试集。

## 4. 高分位置也能进一步分辨变构注释

高分预定义为 native binding_score ≥ 0.5。各方法共享同一位置集合，评价规则不变。

| experiment | geometry | amino | hidden | full | binding_score | proteins | mixed_proteins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AM | 0.7883 | 0.7065 | 0.7710 | 0.8170 | 0.5938 | 114 | 70 |
| AlloBench | 0.1571 | 0.1334 | 0.2242 | 0.2294 | 0.1148 | 383 | 258 |
| AlloBench_to_AM | 0.8072 | 0.7062 | 0.8192 | 0.8372 | 0.5938 | 114 | 70 |
| AlloBench_to_ASBench | 0.1779 | 0.1420 | 0.2350 | 0.2348 | 0.1347 | 168 | 126 |
| AlloBench_to_ASD_AlloPPI | 0.1386 | 0.1441 | 0.1466 | 0.1471 | 0.1044 | 21 | 15 |
| AlloBench_to_CASBench | 0.1558 | 0.1438 | 0.1957 | 0.2154 | 0.1133 | 70 | 49 |
| AlloBench_to_PASSer_ASD_source | 0.1639 | 0.1358 | 0.2303 | 0.2318 | 0.1179 | 288 | 194 |

**最清晰的结果是：即使已经具有较高结合分数，隐藏表示仍能区分其中哪些位置更符合变构注释。**

- AlloBench：full 0.2294，geometry 0.1571，最终 score 0.1148；full−geometry +0.0723，95% CI [0.0480, 0.0969]。
- ASBench 迁移：full 0.2348，geometry 0.1779；增益 +0.0570，CI [0.0291, 0.0846]。
- PASSer 迁移：full 0.2318，geometry 0.1639；增益 +0.0679，CI [0.0426, 0.0934]。
- CASBench 迁移：full 0.2154，geometry 0.1558；增益 +0.0596，CI [0.0163, 0.1057]。

上述四组比较的区间均高于零；这是高结合分数内部仍存在额外生物信息的直接证据。AM 两种协议的高分 full 优于最终 score，但与几何的差值区间跨零；AlloPPI 高分仅 15 个混合标签蛋白，差值区间也跨零。完整七组结果同图呈现。

高分层与全位置的 AUPR 不能直接用绝对值比较，因为正例率与可评价蛋白集合随分层变化；结论来自同层配对比较。CI 为组件 bootstrap 10,000 次，符号置换 2,000 次；这里是后续分层分析，未做跨全部比较的多重校正。

## 5. KRAS/Src：全部实验位置的应用

KRAS 为 187 个唯一实验位置、六个 partner 的 1,118 条位置×实验记录；Src 为两个构建体各 269 个位置，合计 538 条记录。全部纳入，不依赖 low-score 筛选；不把 partner/构建体当成新增独立蛋白家族。

| method | KRAS | SRC |
| --- | --- | --- |
| basic_probe | 0.1874 | 0.1391 |
| burial | 0.3035 | 0.3700 |
| current_full | 0.3306 | 0.3581 |
| current_hidden | 0.3644 | 0.3624 |
| full_probe | 0.4272 | 0.3683 |
| packing | 0.3418 | 0.3236 |
| raw_score | -0.1283 | -0.4329 |

full_probe 为原冻结功能应用中的完整读出；current_hidden/current_full 是当前正式 AlloBench 读出的补充迁移结果。两类头明确分列，不将联合头中的 hidden 贡献项冒充独立纯 hidden 探针。本次不使用 KRAS/Src 功能标签训练分类器。

rho 是预测排序与实验突变影响大小排序的 Spearman 相关系数，不是准确率。KRAS 六个实验的 full_probe rho 为 0.3549–0.4748；Src FL/KD 分别为 0.3362/0.4005。

### 5.1 实验预算

| dataset | method | top10_effect_mean | top10_fold_enrichment | top20_effect_mean | top20_fold_enrichment |
| --- | --- | --- | --- | --- | --- |
| KRAS | basic_probe | 0.4074 | 1.2591 | 0.3232 | 0.9993 |
| SRC | basic_probe | 0.9919 | 1.2726 | 0.9817 | 1.2576 |
| KRAS | burial | 0.5448 | 1.7054 | 0.3888 | 1.2106 |
| SRC | burial | 0.7201 | 0.9236 | 0.9644 | 1.2371 |
| KRAS | full_probe | 0.5920 | 1.8485 | 0.5057 | 1.5762 |
| SRC | full_probe | 1.9437 | 2.4942 | 1.3755 | 1.7631 |
| KRAS | packing | 0.5093 | 1.5925 | 0.4265 | 1.3312 |
| SRC | packing | 0.9548 | 1.2247 | 0.9585 | 1.2281 |
| KRAS | raw_score | 0.4882 | 1.4840 | 0.4228 | 1.3085 |
| SRC | raw_score | 0.4064 | 0.5169 | 0.4273 | 0.5432 |

每个具体实验选前 10 个位置时，full_probe 的实验功能影响均值相对全体均值富集 KRAS 1.85×、Src 2.49×。前 20% 预算下，相对 burial 的平均效应增加分别为 0.1169 [0.0011, 0.2594] 和 0.4111 [0.1311, 0.7241]。不同体系效应单位不同，不跨体系比较绝对幅度。

### 5.2 结构控制后的功能关联

| dataset | region | method | controls | estimate | lower | upper | bootstrap | block_length | refit_controls_each_draw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KRAS | all | full_probe | score_geometry_AA_plddt | 0.3076 | 0.1253 | 0.4635 | 10000 | 10 | True |
| KRAS | all | full_probe | score_geometry_AA_plddt_folding | 0.2595 | 0.0738 | 0.4186 | 10000 | 10 | True |
| SRC | all | full_probe | score_geometry_AA_plddt | 0.3481 | 0.2131 | 0.4572 | 10000 | 10 | True |
| SRC | all | full_probe | score_geometry_AA_plddt_folding | 0.3487 | 0.2133 | 0.4553 | 10000 | 10 | True |

控制 score、几何、氨基酸和 pLDDT 后，两体系偏秩相关仍为 0.3076 和 0.3481；进一步控制折叠影响后为 0.2595 和 0.3487，区间均高于零。这是功能信息与上述协变量不能完全互相替代的证据；偏相关不是额外的测试集预测头。

### 5.3 Gaussian-surface 通路

| dataset | S_retain00 | S_retain50 | baseline |
| --- | --- | --- | --- |
| KRAS | 0.1692 | 0.3437 | 0.4272 |
| SRC | 0.1711 | 0.3224 | 0.3683 |

冻结读出不变，同一 GPU 后端重新前向，固定 native score/基础协变量；只改变传入读出的隐藏表示。S_retain50/00 是原 Gaussian-surface 表面反馈／融合干预，不是重新训练的无高斯架构，也不单独等价于 pair 参数删除。

- KRAS：rho 0.4272 → 0.1692，下降 0.2580；配对下降区间 [0.0758, 0.4549]。
- Src：rho 0.3683 → 0.1711，下降 0.1972；配对下降区间 [0.1288, 0.2650]。

这支持“实验功能排序依赖模型实际使用的 Gaussian-surface 上下文整合通路”。不是对真实蛋白变构因果的湿实验结论。高分变构子集的表面截断损失本次未得到一致显著性，因此主文的强通路依赖图使用实际支持较强的 KRAS/Src 全位置实验；高分子图用于表达信息增量，不混用两个论证。

KRAS/Src 高分子集的功能排序结果另见完整审计：KRAS full_probe rho 0.1912，Src −0.0979；不能将高分变构注释的增益推广成两个功能体系中每个分层都更好。

### 5.4 新增：相近结合分数内的功能区分

将每个实验内的native score固定分为五个等频组，在组内计算功能排序，再按组、assay、case与parent汇总。原完整冻结读出的组内rho为KRAS **0.3080**、Src **0.3749**；相对直接score排序分别增加**0.2763 [0.0832,0.4404]**和**0.4537 [0.3431,0.5645]**。

这比仅对比总相关性更直接说明：相近结合倾向的位置仍有可由隐藏读出区分的实验功能差异。与全协变量控制关联配合，作为主文信息增量的重点展示。较低观测折叠效应和功能减折叠秩差的另外两种分离也已完整计算，详见`FUNCTIONAL_SEPARATION_CN.md`；不以所有分离终点均高于零为结论要求。

## 6. 主文图安排

- Figure 3a：冻结模型与读出示意，清楚区分结合任务训练、变构标签的探针训练、独立功能实验评价。
- Figure 3b：全位置七组四方法 AUPR，总览任务内读出和冻结迁移。
- Figure 3c：全部七组高分子集 full−geometry 配对差值和 95% CI；不只选择显著的数据集。
- Figure 3d：AlloBench 高分中 score、geometry、amino、hidden、full 的排序比较，配合明确选择标准的结构实例。实例结构图需基于已有逐残基表另行制作，不用示意图冒充实际结构结果。
- Figure 4a：KRAS/Src 全位置实验设计及原冻结读出。
- Figure 4b：全部位置功能 rho 与固定预算前 20% 实验效应；同时呈现 score、burial、packing、基础读出和完整读出。
- Figure 4c：native → retain50 → retain00 的功能 rho；配对区间体现 Gaussian-surface 贡献。
- Figure 4d：相近score内的功能区分，配合控制结构与折叠后的关联；`figures/functional_information_separation.pdf`已生成对应矢量草图。

主文围绕真实增益组织，不需要以反向 score 为叙事主线；完整方向性、全部分层、缓存后端与同源审计放入补充材料。当前结果支持变构注释读出和实验功能排序，不以完整生化通路作为本节结论。

## 7. 文件入口

- results/functional/PAPER_*.csv 与 results/allosteric_strata/PAPER_*.csv：主文作图的直接输入。
- results/allosteric_strata/：全部变构分层、逐蛋白指标和配对统计。
- results/functional_separation/：三种分离分析、分组位置、逐实验结果和10,000次序列块区间。
- results/functional/native_experimental_positions.csv：全部功能实验位置、实验值、score、结构与各读出。
- results/functional/all_conditions_experimental_positions.csv.gz：全部干预位置级读出。
- docs/FUNCTIONAL_PROTOCOL_CN.md：冻结读出、输入契约、统计与补充审计说明。
- figures/allosteric_main_and_high_score.pdf：全任务 AUPR 与高分配对增益，矢量草图。
- FILE_INVENTORY.csv：结果和图文件的大小与 SHA256。

本报告、复算代码、主文与补充结果同步到 projects/GSbind 和 projects/GSbind_github；远端 GitHub 尚未推送本次更新。
