# 统一指标与统计契约

## 指标与阈值

1. AUPR使用`sklearn.average_precision_score`，即Average Precision，不用梯形PR曲线面积替换。
2. `aupr_mixed`仅在同时有正负标签的case计算；`aupr`在有正例时计算，全正例为1，无正例记NaN。每张汇总表同时给`cases`和`mixed_cases`，明确有效分母。
3. `AUROC`仅在正负兼有时定义。不得用0补足未定义的case。
4. 主任务/SSM的MCC、F1等以固定score≥0.5判正。新变构探针阈值由训练侧内折OOF选MCC最大，同时另存固定0.5结果；不在测试集选阈值。TP/TN/FP/FN全部保存。MCC分母为0给0，并保存mcc_defined=false。
5. 主任务原始protein score的0.5有分类意义。SSM下游沿用这个binding score阈值，不将其说成校准过的“突变敏感概率”；变构探针也未做生化概率校准，稀疏阳性下0.5-MCC/F1可能接近0。它们的主要生物应用是**排名和相同实验预算富集**，不能挑测试阈值让MCC变高。
6. KRAS/Src的功能/折叠效应是连续测量。报告Spearman rho、固定预算top20%效应均值，不凭空二值化产生MCC。这是“该任务不适用”，不是漏算指标。

## 人群与未知标签

- 主任务使用`label_mask > 0.5`；四模型必须同名单、同残基、同标签。
- 所有intervention的`baseline_high`/`baseline_low`都按原始未扰动binding score分组，不随干预重选有利位置。
- AlloBench all使用383蛋白，AM使用127蛋白；baseline_low固定native score<0.5。both_clean排除与E11训练达到30% identity且双向coverage≥80%的命中；either_clean按任一方向coverage≥80%排除。source_component_disjoint另外排除与Allo读出来源的组件共享，不能混淆这三种分层。
- AlloBench383与AM127各自nested五折，其他来源固定迁移；AlloBench是变构配体/位点**注释识别**，不能说背景残基已被实验排除变构功能。四种方法仅geometry/amino/hidden/full。
- SSM `sensitive_ge_1/ge_2`重现原始中心估计标签；`primary18`采用原始替换测量的置信区间与质量过滤，至少18个有效非WT替换。至少一个有效替换的下界≥2则阳性；所有有效替换上界均<2才是阴性；其余未知，不当阴性。`relaxed1`为至少一个有效非WT替换的单独补充口径。

## 汇总层级

`case_metrics.csv`是所有分析的底表。`case_macro_metrics.csv`为case等权，`pooled_metrics.csv`把同一评价人群的残基拼接后计算。

AlloBench主增量用component等权：先case计算，再同源component内均值，最后component等权。SSM的干预差值也按`cluster_rep`聚合。主任务干预按Pinder system聚合，避免左右链当作独立样本。KRAS/Src先同一case内partner均值，再构建体均值，最终按parent分别报告；不把大量残基当作大量独立蛋白。

**case-macro、component-macro、pooled是不同指标，数值不得混引。** 原始四模型主表以case-macro AUPR展示，并提供pooled F1/MCC和case-macro F1/MCC。

## 排名预算

Top-10固定每个蛋白最多10个位置；不足10取全部。按预测降序、残基索引升序打破同分。输出命中数、precision、recall。不同方法比较必须固定同一蛋白、同一位置集合及预算。AlloBench另保存新增命中/丢失命中，不能只列新找到的位置。

KRAS/Src预算为该有效人群位置数的`ceil(0.2*N)`，原始`position`升序打破同分，功能和折叠效应各自报告，不因目标效应改变候选排序。

## 配对统计

`src/gsbind/metrics.py`固定10,000次cluster bootstrap、2,000次随机sign-flip；新变构实验显式使用seed20261020，其他任务沿用各自固定种子。每次以同源组件或system为重采样单元。先在case上计算“comparison−baseline”，再组件内均值，再bootstrap；配对缺失只在同一对的完整数据上剔除。

95% CI用bootstrap 2.5%/97.5%分位数。双侧p为`(1 + count(|null|≥|observed|))/2001`。同一任务输出中按metric对全部生成比较进行BH校正，保存q；不只校正显著者。

**方向：** `surface_paired_statistics.csv`中负增量表示干预造成损失；要报告“损失”时取负号并交换CI端点。当前变构比较总表为`results/allosteric/ALL_AUPR_CONTRASTS.csv`，差值方向写在`comparison`中：`A_minus_B`表示A减B。

`S_zero_minus_noise`单独比较完整截断与三次噪声的均值，负值表示截断比同模长噪声造成更大损失。主任务已保存预测采用的sigmoid数值转换与四模型原始CSV最多相差约1.2e-7，按2e-7容差检查；identity仍要求同协议逐值完全相同。

既有结果保留于`evidence/`，本工程使用统一随机种子重算区间。点估计应一致，Monte Carlo CI可有小幅差异；不能把两次随机抽样的CI混成同一次实验。
