# 功能信息分离：相近结合倾向内部还学到了什么？

## 研究问题与固定方法

总相关性混合了不同结合分数、结构环境和实验后果。本分析进一步检查冻结隐藏读出是否在结合倾向相近的位置内仍能区分功能影响。模型仍为Gaussian-direct128 E11；原功能读出与当前Allo头均固定，不使用KRAS/Src功能标签重新拟合分类器。

分析使用当前全部实验位置，并一次性保留三种有不同生物含义的分离方式：

1. **相近score内的功能排序**：每个assay按native score分五个等频组，保留相同score的并列关系；组内计算功能Spearman，等权平均组，然后按assay→case→parent汇总。每组至少8位置。所有方法固定同一分组。
2. **较低观测折叠影响中的功能排序**：实验折叠影响百分位秩≤0.5的位置中，评价功能排序。它不是“完全不影响折叠”的实验判定。
3. **相对功能偏向**：实验指标`[rank(functional magnitude)−rank(folding magnitude)]/N`，检查读出是否偏向功能相对折叠更突出的残基。此指标使用实验标签定义，不是对新蛋白可直接计算的物理量。

固定5分位和0.5阈值，没有搜索到所有指标为正再选阈值。上述为已查看体系上的后续分析；完整方法和结果同时保存，不将换终点视为推翻原始负相关。

## 核心结果：相近score内仍有功能信息

| 方法 | KRAS组内功能rho | Src组内功能rho |
| --- | ---: | ---: |
| 原始结合score | 0.0317 | -0.0788 |
| Burial | 0.1914 | 0.0691 |
| Packing | 0.2520 | 0.0851 |
| 原基础读出 | 0.1678 | 0.2542 |
| 原完整冻结读出 | **0.3080** | **0.3749** |
| 当前独立纯hidden读出 | 0.3519 | 0.3250 |
| 当前完整Allo读出 | 0.2948 | 0.3315 |

原完整冻结读出相对原始score：

- **KRAS：+0.2763，95% CI [0.0832, 0.4404]。**
- **Src：+0.4537，95% CI [0.3431, 0.5645]。**

两体系的完整读出本身组内rho区间为KRAS [0.1339,0.4483]、Src [0.2874,0.4414]。这回答了“相似结合倾向为什么还要读隐藏层”：它能进一步排序实际实验功能影响。

Src相对burial增加0.3058 [0.1197,0.4941]，相对packing增加0.2898 [0.0876,0.4742]。KRAS相对两者的点增益为0.1165和0.0559，区间跨零。更全面的结构控制证据由原全协变量偏秩相关提供：加入score、几何、AA、pLDDT和折叠影响后，KRAS为0.2595 [0.0738,0.4186]，Src为0.3487 [0.2133,0.4553]。

## 另外两种分离结果

较低观测折叠影响一半中，原完整读出功能rho为KRAS 0.3946、Src 0.2900；score为0.0257、-0.4016。Src的burial为0.4038，因此这一子集不支持完整读出超越所有结构单指标。

功能减折叠百分位秩差中，原完整读出rho为KRAS 0.1725、Src 0.2382；当前纯hidden为0.2888、0.2105。KRAS原始score为0.2561，不能将此新指标称为原完整头全面优于score。所有方法区间和差值均在完整CSV中，不只保留主文使用的结果。

## 统计与图表

每种分析10,000次序列块bootstrap，块长10；同parent位置跨assay共享抽样权重。分组与位置集合在原生数据上定义后固定，区间条件于这些观测分组。每次抽样精确重算带权并列秩。全部主比较的有效抽样次数保存，不把构建体或partner当成独立家族；新增分离比较未作多重校正。

主文适合以**score内功能区分＋全协变量控制关联**组合展示，不要求所有指标或所有分层都大于零。图为`figures/functional_information_separation.pdf`。总体应用排序和Gaussian-surface截断仍使用全部实验位置，不被新分离结果替代。

## 复算与数据

```bash
python main.py functional-separation --output runs/functional_separation --bootstrap 10000
```

代码：`src/gsbind/functional_separation.py`。

正式结果：`results/functional_separation/parent_metrics.csv`、`assay_metrics.csv`、`paired_block_bootstrap.csv`、`coverage.csv`、`position_membership.csv`、`PROTOCOL.json`。输入：`results/functional/native_experimental_positions.csv`。保留反向score作为补充诊断，主文不以反向解释为主线。
