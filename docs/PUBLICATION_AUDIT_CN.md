# GSbind 主工程发布审查

日期：2026-10-20。范围是当前主文工程，不是历史研究档案。旧远端全部文件由历史分支及标签完整保留；当前默认分支从GSbind主工程建立。

## 1. 主任务框架

`main.py → gsbind.cli → ExperimentConfig → GaussianDirectApp`统一处理预处理、训练、前向和评价。训练使用`GaussianDirectLightningModule`与`GaussianDirectDataModule`；优化器、学习率计划、masked BCE、DDP和保存契约集中在`src/gaussian_direct/lightning.py`。不是用Lightning名字包装另一套隐藏训练脚本。

`configs/main.json`为20,801记录、双GPU、20 epoch计划；论文固定E11。`configs/apo_af2.json`是另一正式测试态；`configs/smoke.json`为CPU两例。主任务训练和下游探针训练是两个明确不同的入口。

## 2. 各下游任务的边界

| 任务 | 输入与监督 | 是否训练 | 指标与代码 |
|---|---|---|---|
| AlloBench / AM | 固定E11 hidden及结构/AA特征；各自标签 | 仅四种线性探针，组件分组outer5/inner3 | component-macro AP；`allostery.py`、`probe_training.py` |
| 固定来源迁移 | AlloBench冻结头；相关组件用来源留出头 | 不用目标标签拟合 | AP、组件独立分层、预算命中；`allostery.py` |
| SSM220 | 原始binding score与实验敏感性标签 | 不训练SSM hidden头 | AP、Top10、覆盖数；`evidence.ssm` |
| KRAS/Src | 原冻结功能读出与实验功能/折叠效应 | 不使用目标实验标签重新拟合 | Spearman rho、Top20%效应；`evidence.functional` |
| Gaussian-surface截断 | 残基gate/update及最终fusion的surface半部 | 不重训模型或读出 | 同集合配对损失、identity、噪声；`surface.py`、`forward.py` |

四种变构特征固定为geometry5、amino20、hidden128、full155。full无重复logit。hidden逐蛋白中心化的参考集合由native score<0.5定义，不能称完全score-independent预处理。

## 3. 数据与指标

训练/测试名单、labels、掩码、逐残基预测、各队列CSV、冻结头和映射审计进入Git。大结构、compact、七条件hidden及大SSM源表进入外部数据清单，明确待公开，不伪装下载已完成。

主任务是相同测试人群的case-macro AP；变构主结果为component-macro AP；pooled独立标识。SSM不同端点覆盖不同，保留mixed_cases，不把220结构输入说成全部严格实验标签可评价。KRAS/Src为连续实验后果，不强行制造MCC。

## 4. 当前主文范围与目录精简

只保留结合位点主任务、四方法变构应用、score-only SSM220、KRAS/Src和少量Gaussian-surface干预。不引入旧CPAS/MPP/GCBAR、SSM hidden分类器、历史HGB、112→171探针展示、BC30面板或逐case oracle。

`results/`是当前最终数字；`figures/source_data/`是绘图入口的CSV副本；`datasets/*/*.parquet`保存必要细粒度预测/标签。`evidence/main/`少量表仅作既有冻结主表的数值核对，不是另一套主结果。`provenance/`保存来源与验证，不保存一堆探索运行日志。

## 5. 发布审查中修正的内容

1. 删除架构说明中“已有bypass/参数匹配结果”的歧义：代码开关存在不等于该重训练实验已进入当前结果。
2. 修正指标文档的旧`increment_statistics.csv`引用，指向当前`ALL_AUPR_CONTRASTS.csv`。
3. 增加`summarize`入口，在缺少大hidden张量时仍可从公开保存的预测重算主文表格；明确这不是重新模型前向。
4. 区分Git文件校验与完整数据校验，外置数据未安装时不声称全数据复现已通过。
5. 保留完整server包不变，公开版在独立工作树构建；发布文件使用独立清单。

## 6. 后续工作不阻塞主工程发布

最终期刊四图布局、精细结构渲染、附录扩展分析及完整数据公共托管链接后续加入。当前基础图不是最终排版图；模型通路截断也不替代所有重训练架构消融，更不等同真实生化因果。
