# 四张主图：最终证据与可复算来源

目标：以GSbind结合位点主任务为主线，以变构信息增量、实验敏感性排序和Gaussian-surface依赖性构成生物应用证据；不再让旧候选生化通路承担Fig4核心结论。

## Figure 1：架构与任务

| 小图 | 叙事 | 代码/来源 |
|---|---|---|
| 1a | partner-free结构→残基结合score的整体任务；holo仅提供对应界面标签 | 三个main manifest、工程契约 |
| 1b | AtomEncoder→Gaussian field/pair→多尺度surface→三轮反馈/融合→direct head | `model.py`及`modules/`，`ARCHITECTURE.md` |
| 1c | 明确surface反馈/融合在完整Gaussian模块中的位置；标出唯一正文截断位置 | `surface.py`三组hook |
| 1d | 训练/两个MMseqs测试/下游任务的不同角色 | 20,801；988；874；AlloBench383/AM127五折；四来源迁移；SSM220；KRAS/Src |

架构图是实际代码的示意，不把模型计算route画成已经验证的生物信号传递链。

## Figure 2：结合位点主任务性能与表面上下文作用

| 小图 | 指标 | 精确表 |
|---|---|---|
| 2a | holo四模型case-macro AUPR | `results/main/case_macro_metrics.csv` |
| 2b | APO/AF2四模型case-macro AUPR | 同上，state panel |
| 2c | 同人群pooled AUPR；MCC/F1/AUROC移至附录 | `results/main/pooled_metrics.csv` |
| 2d | baseline→surface50→surface0剂量效应；holo和state分别 | `results/main/surface_case_macro_metrics.csv` |
| 2e | system级配对损失与CI，完整逐case底表放补充 | `surface_paired_statistics.csv`、`surface_case_metrics.csv` |

不放BC30，不放逐case最优oracle，不把训练数据差异说成严格相同训练集架构对照。若绘制PR曲线，必须从四模型残基预测重新计算，不能从AUPR常数反推曲线。

## Figure 3：AlloBench的隐藏信息增量与实验候选价值

| 小图 | 问题 | 精确表 |
|---|---|---|
| 3a | 纯几何、纯AA、纯hidden、完整155维融合读出的AUPR | `results/allosteric/MAIN_AUPR_TABLE.csv`，AlloBench与AM任务内五折 |
| 3b | 固定Allo读出到ASBench/PASSer源表/CASBench/AlloPPI/AM迁移 | 同表，来源相关保护全队列与来源组件独立子集分别标注 |
| 3c | 全人群、低score、E11同源清洁分层上的比较与CI | `ALL_AUPR_CONTRASTS.csv` |
| 3d | 每蛋白10个实验预算：完整读出相对三种对照新增/丢失 | 各实验的`top10_added_lost.csv` |
| 3e | 新155维full读出的真实S通路截断及同模长噪声 | 各实验`summary.csv`和`paired_aupr_statistics.csv` |

结构实例应从上述底表预先确定选择标准（如最大增益、代表中位数、无增益各一），再展示真实输入坐标上的注释/读出；不要只挑最好看图片，也不要在没有结构图时声称已完成具体结构展示。

## Figure 4：实验功能后果与Gaussian-surface应用依赖

| 小图 | 内容 | 精确表 |
|---|---|---|
| 4a | SSM220 native高binding score是否富集敏感位点：全220+两种阈值 | `results/ssm220/case_metrics.csv`、`endpoint_coverage.csv` |
| 4b | primary18原始置信区间支持标签；全人群/高score的固定预算与表面截断 | `case_macro_metrics.csv`、`surface_paired_statistics.csv` |
| 4c | KRAS/Src低score非接触位置的功能效应关联：full/basic/score | `results/functional/parent_metrics.csv` |
| 4d | 功能与折叠效应并列，展示不是同一个测量目标 | 同表的两个target；逐partner在`interaction_metrics.csv` |
| 4e | KRAS、Src各自baseline→surface50→surface0及噪声，突出rho变化 | `parent_metrics.csv` |
| 4f | 代表性真实实验位置散点/残基映射，完整测量而非只展示命中 | `experimental_residues.csv`与`frozen_predictions.parquet` |

本图中SSM只使用原始score，不包含隐藏SSM探针。KRAS/Src代表有价值的两个实验体系应用，不把三个构建体或多个partner扩成跨大量蛋白独立验证。

## 现有绘图入口

`python main.py figures`导出基础证据图PDF，并把所有最终CSV复制到`figures/source_data/`。这些是可修改的分析图和完整作图底表；四张主图的最终排版、结构渲染和面板标注仍应围绕此表设计完成，不能把基础柱状图称为已完成的期刊四图。
