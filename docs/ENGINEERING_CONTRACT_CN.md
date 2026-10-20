# GSbind 完整工程与前向契约

公开仓库提供完整代码、正式结果、权重和小案例；大数据按`data_release/`单独发布，当前链接待补。全数据命令需要安装该数据，保存预测的指标复算使用`main.py summarize`。

## 模型身份

- 论文模型：GSbind，权重为Gaussian-direct128 E11，128维，`vg`架构、3轮co-update、direct protein head。
- checkpoint SHA256：`9d920481b5bf7ddd3e4b6542b2b1951b5a4683ce50f5b2392d22e7eb7945f397`。
- 3,355,001个参数：AtomEncoder 1,474,957；Gaussian/direct路径1,880,044。
- 核心网络是精确E11实现。包装层增加本地路径、统一指标和S通路hook，不替换权重或改变正常前向。

## 文件级职责

| 层级 | 实现 | 输入→输出 |
|---|---|---|
| PDB读取和对齐 | `src/gaussian_direct/atom_encoding/structure_io.py`、`dataset.py` | 原子坐标、原始残基ID→严格对齐原子张量 |
| 拓扑与缓存 | `atom_encoding/encoding.py`、`surface_cache.py`、`preprocessing.py` | 64邻接、16/24/32残基图、表面点及图 |
| AtomEncoder | `atom_encoding/atom_encoder.py` | 原子30维输入→标量/向量更新→残基状态 |
| Gaussian投影 | `modules/gaussian_projection.py` | 残基条件带宽→残基到表面的各向异性Gaussian投影 |
| Pair Gaussian | `modules/pair_gaussian.py` | 残基pair上下文→表面消息修正 |
| 表面上下文 | `modules/surface_encoder.py` | 表面图GVP、多尺度传播、连通patch表示 |
| 跨尺度反馈 | `modules/cross_scale.py`、`global_context.py`、`model.py` | 表面→残基attention、全局上下文、门控co-update与融合 |
| 主输出 | `GaussianDirectModel.forward` | 5通道logit；第0通道sigmoid是protein interface score |
| 训练 | `lightning.py`、`objectives.py` | LightningModule/DataModule、加权损失、优化器、epoch导出 |
| 任务入口 | `gsbind/cli.py`、`evidence.py` | 固定配置、统一分析、可追溯输出 |

## 主任务数据

训练manifest完整20,801条，测试仅官方MMseqs统一988 holo与874 APO/AF2。名称`clean1000`和`clean886`是历史panel标识，不是实际纳入数量。没有BC30 panel和逐case最优oracle。

每条记录有`apex_case_id`、输入状态、链、严格残基顺序、labels/mask、本地PDB和compact路径。测试输入是单侧/单体结构；结合标签来自对应holo界面。**apo/AF2是分析输入，holo提供目标标签，不能把“使用holo标签”误写成“以holo坐标推断apo/AF2”。**

`compact_path`优先于缓存目录推导，便于多个任务独立使用各自输入。无显式compact路径时，按`receptor_pdb_relpath`的SHA1推导缓存键。拓扑缓存的0-based索引在加载时转换为网络的padding语义，不能自行整体再加1。

预处理先严格按`input_residue_ids`保留并对齐结构，检查labels和mask长度，不对错位样本静默截断。Surface缓存包含坐标、法向、几何描述、表面图邻接/距离及最近残基关系。缓存是原始结构的数值化输入，而不是另行训练的模型。

## 训练配置

`configs/main.json`：seed42，20训练epoch，20 epoch学习率计划，双GPU，每卡batch4，global batch8；atom/Gaussian学习率均1e-4，weight decay1e-4，warmup比例0.05，最低学习率比例0.1，dropout0.1，mixed precision。

正例权重从训练manifest计算：`0.5 * (1-positive_rate)/positive_rate`。优化器和目标的精确行为以`lightning.py`、`objectives.py`为准；配置不通过测试结果重选。提供的是精确E11权重和训练工程，不声称本次打包重新跑完20 epoch或在不同GPU后端逐bit重现训练。

## 表面通路截断

本文Gaussian模块口径包含原子编码之后的残基/表面整合、反馈和融合，不只指pair参数。

每一轮的`core_residue_gate`、`core_residue_update`以及最终`same_level_fusion`，其拼接输入为`[residue_state, surface_context]`。干预将第二块替换为：

```text
baseline / identity:  surface_context
S_retain50:          0.5 × surface_context
S_retain00:          0 × surface_context
S_noise{seed}:       surface_context + unit_noise × ||surface_context||₂
```

噪声按每个残基独立归一化，seed为101/211/307，并按阶段偏移1009。噪声位移模长等于完整截断所删除的surface上下文模长。AtomEncoder输出和拼接的residue一半不被hook直接修改；后续残基状态因真实前向反馈改变是干预目的。表面计算本身仍执行，切断的是它向残基更新和融合的有效输入。

因此这是**模型内Gaussian-surface有效通路的依赖性检验**，不是重新训练无表面网络，也不表示剩余模型完全没有历史表面信息，更不等价于真实蛋白物理传递通路被证明。保存identity逐bit不变、局部hook审计和完整forward案例。

## 下游读出协议

AlloBench与AM分别进行同源组件分组nested外5内3折，候选C=0.1/1/10；E11完全冻结。仅保留四种线性读出：geometry（5维RSA/packing/DSSP）、amino（20维残基身份）、hidden（128维融合表示）、full（score+geometry+AA+pLDDT+hidden，共155维，不含额外logit）。四种使用同一数据人群、分组、正则选择和训练权重。

全部504输入的完整E11前向与七条件surface重放已保存。每个蛋白在native score<0.5的固定残基集合上对各条件hidden分别中心化，无低分位置时用完整蛋白均值。因此hidden读出不显式输入score，但中心化使用native score定义参考集合；不称完全score-independent。

AlloBench383蛋白/282组件，AM127蛋白/112组件。训练侧内折选择正则并产生OOF概率以选MCC阈值；正文AUPR不依赖分类阈值。外部ASBench/PASSer_ASD_source/CASBench/ASD_AlloPPI及Allo→AM固定迁移，目标标签不拟合。相关组件用来源外层留出头，组件独立目标用来源全数据头。

full的Gaussian-surface干预固定native score、几何、AA、pLDDT，只替换hidden并使用同一个冻结155维头；identity逐值等价。KRAS/Src使用其原有功能读出，不被这次变构读出替换，也不能称接受新变构标签训练。
