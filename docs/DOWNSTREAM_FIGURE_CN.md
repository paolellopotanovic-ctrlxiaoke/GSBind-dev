# 下游结果图：清晰小提琴与柱状图

在展示结果前，可用 `figures/downstream_task_biology.*` 解释结合界面、变构注释与功能敏感性的区别，再用 `figures/downstream_task_workflow.*` 对应A/B/C的标签、读出与指标。两图的讲述逻辑见 `docs/DOWNSTREAM_TASK_EXPLANATION_CN.md`。

日期：2026-10-23。图：`figures/downstream_selected_transfer.pdf`及同名PNG。由 `python main.py figures` 中的下游绘图步骤生成，不重新训练或选择模型。

当前版模仿用户指定的GSBind Figure 2配色、黑色轮廓、白色箱体、简洁坐标轴；去掉散点、菱形、显著性外观括号和图内长说明。方法配色固定为score青绿、Structure浅黄、hidden浅蓝、PeSTo粉红。三图原路径更新，另有四结果面板合图 `figures/downstream_results_four_panel.png/pdf`。独立PNG/PDF为 `downstream_A_AlloBench`、`downstream_B_AM`、`downstream_C_Src`、`downstream_D_pooled_AUPR`，均在 `figures/`。两张结构案例图不改动。

另提供 `figures/downstream_pooled_pr.pdf`及同名PNG：分别将AlloBench和AM全部评估残基拼接后，绘制四种读出的真实PR曲线。横轴Recall、纵轴Precision，图例为梯形AUPR而非AP；隐藏层曲线下浅色阴影对应其梯形面积，灰色虚线是该数据集正例比例。两个数据集的正例比例不同，不直接把两者AUPR高低解释为迁移提升。沿用已有checkpoint，不按pooled指标重新选择。曲线可以交叉，面积最大不等于所有Recall下Precision均最高。完整阈值、Recall、Precision在 `figures/source_data/downstream_pooled_pr/PR_CURVES.csv.gz`，8个面积及样本数、正例比例在同目录 `AUPR_SUMMARY.csv`。图中各线不作step后再用矩形面积冒充梯形积分。

| 面板 | 内容 | 观测单位 |
| --- | --- | --- |
| A | AlloBench四种方法的AUPR小提琴＋白色箱线，无散点 | 282个组件 |
| B | 冻结AlloBench探针迁移至AM的四种方法AUPR分布 | AM全部112个组件 |
| C | 四种读出的实验功能预测性能 | 纯柱状图，柱高为全长与KD两个构建体的平均相关 |
| D | AlloBench与AM的四方法AUPR分组柱状比较 | pooled梯形AUPR，不是组件均值 |

不再画来源组件独立子集的小提琴，不再画配对优势面板，也不画逐位点秩散点。它们的底表仍保留为补充分析，但不是本图的面板。

## 之前E、F的含义，以及为何简化

旧E分别计算全长Src与激酶结构域的预测功能分数和实验功能幅度的Spearman相关，并比较四种读出方法。它不是两个新的数据集，也不是训练曲线。C现在以方法比较为主：方法配色与A、B一致，只画两构建体的均值柱；各构建体数值完整保留在底表。

旧F将各构建体269个位点的预测值和实验值分别转成秩百分位后画散点，只是相关性的可视化，没有引入新指标。与C信息重复，已经删除。

## 图形定义

- A、B先逐蛋白计算梯形AUPR，再在组件内平均；每个核密度观测为一个组件，而不是残基或一次交叉验证。
- 小提琴表示组件值的核密度，密度显示限定在合法AUPR区间[0,1]。白色箱体表示25%–75%分位，黑线为中位数，须为1.5倍四分位距内的观测；不叠加散点，箱线不另画离群点，全部原始值保留在底表和核密度输入中。
- 不再叠加均值菱形或顶部数字；组件均值仍在主表和 `panel_summary.csv` 中。小提琴宽度不是置信区间，不添加显著性星号。
- C为每个构建体完整OOF预测的Spearman相关，再平均两个构建体；不是五折相关系数的平均，也不能用仅有两个构建体生成有意义的小提琴。
- C柱顶仅标均值，不添加误差条或显著性括号。隐藏层减结构的平均绝对相关增益为0.512691 − 0.400463 = +0.112228。两个构建体均提升：全长0.358404 vs 0.290049（+0.068356），激酶结构域0.666977 vs 0.510877（+0.156101）。这是描述性差值，不是显著性检验。
- D直接读取主表的pooled_AUPR，与现有真实PR曲线面积一致；不是AP，不改变checkpoint，也不虚构重复实验误差条。A/B的组件均值和D的pooled指标不能混用。D底表为 `figures/source_data/downstream_selected/pooled_AUPR_bars.csv`。
- 该比较支持“冻结隐藏表征在线性读出下能更好预测实验功能变化，优于当前结构特征和结合分数基线”。尚未比较结构与隐藏表征的联合探针，因此不能据此声称严格条件独立的信息增量、机制因果证明或独立测试泛化。
- AlloBench和Src保留来源测试折选ckpt的探索性说明。AM不训练、不用AM标签选ckpt；删去独立子集图不等于改变数据路由或消除来源选择暴露。

## 英文图注

**Allosteric-site and functional readouts.** (A, B) Component-level trapezoidal AUPR distributions for GSBind score, structural features, GSBind hidden features, and PeSTo score readouts on AlloBench (282 components) and frozen AlloBench-to-AM transfer (112 components). AUPR is computed per protein and averaged within each component. Violins show kernel densities; white boxes show interquartile ranges and black median lines, with whiskers extending to observations within 1.5 interquartile ranges. Individual observations and boxplot outliers are not overlaid. (C) Bars show the arithmetic mean of the complete-OOF Spearman correlations for full-length Src and its kinase-domain construct (269 positions each), not the mean of fold correlations. (D) Bars compare pooled trapezoidal AUPR across all evaluated residues within AlloBench and AM, using the same checkpoints as A–C. Pooled AUPR is distinct from component-macro AUPR. No replicate-based error bars or significance tests are implied. This is a descriptive predictive comparison, not a test of conditional non-redundancy or causal mechanism. AlloBench and Src source checkpoints are selected using source held-out-fold performance and constitute exploratory test-selected OOF results. AM labels are not used for fitting or checkpoint selection; eligible source heads exclude the target component from their training data. Source-related AM components may nevertheless have source-label selection exposure.

## 底表

D柱顶数字横排；各数据集柱组内的灰色虚线为随机排序的参考精确率（全部评价残基的正例比例），AlloBench与AM分别计算，不共用一条基线。它是PR的无信息参考水平，不是随机置换后有限样本梯形AUPR的精确期望，也不作为第五个训练方法。基线样本数、正例数和比例保存在 `figures/source_data/downstream_selected/random_baselines.csv`。

`figures/source_data/downstream_selected/component_AUPR.csv`和`panel_summary.csv`支持A、B；`src_construct_rho.csv`和`src_functional_gains.csv`支持C及逐构建体增益。目录中其余子集、配对差值及位点秩表为补充分析，保留溯源但不展示为主图面板。指标一致性测试在 `tests/test_downstream_figures.py`。
