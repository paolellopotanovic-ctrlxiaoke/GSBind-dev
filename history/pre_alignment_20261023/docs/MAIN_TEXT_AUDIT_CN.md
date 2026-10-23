# 当前主文证据与公开工程复核

日期：2026-10-23。当前下游仅采用“AlloBench来源测试最优头直接迁移AM；Src内部五折测试最优”。固定50/50/10、AM单独训练、旧迁移/组合探针的结果不再并列为活动口径。

## 当前数值与数据

- AlloBench hidden选中28/49/69/44/74次，来源选优OOF AUPR **0.159236**。AM主迁移AUPR **0.693699**；来源组件独立补充子集 **0.726231**。Src hidden选中1/7/2/4/7次，完整选优OOF Spearman **0.512691**。
- 两个来源×四方法×五折，共40个选中头及40个历史训练参考头，均发布候选指标历史。AM没有训练头，标签不参与scaler、参数、选轮或路由；主迁移按来源训练组件排除路由，而不是无条件平均五头。
- AlloBench216556行/7409正例/383蛋白/282组件；AM1801行/839正例/127蛋白/112组件，独立补充371行/26蛋白/25组件；Src538条构建体位置记录、269唯一位置、812完整特征行。
- 所有当前正文/补充下游指标、完整预测、路由和来源候选由公共输入重放核对；来源搜索重建检查另见本次provenance，不沿用旧固定轮次审查结果。
- 主任务/SSM从保存预测重新计算CSV并与原数字核对；主干没有因本次更新重训。主任务20801训练链、988 holo、874 apo/AF2；E11 case-macro AP为0.7914999169、0.7281149439。
- Figure3保留原Gaussian-surface和配对结构可靠性数据/代码，原公开读出核对312行汇总、3152次配对指标；没有声称全部原始结构干预前向在公共克隆重跑。
- forward仍只有前两次surface pass后更新残基，第三次surface context与第二次反馈后的残基状态融合；本次不改E11权重或计算图。

## 报告边界

1. **来源测试选优要明确标注。** AlloBench与Src选轮使用各来源留出/测试折；保存成固定checkpoint并不使这些OOF成绩成为独立测试估计。Src不是新蛋白泛化，FL/KD也不是两个partner。
2. **AM不训练不等于AM全体来源独立。** 来源相关组件仍可能参与AlloBench来源选轮；全体AM按训练组件排除路由，完全排除来源组件的子集另列。不要把0.726231子集分数冒充全体0.693699，不按AM表现再改源模型。
3. **AP与梯形AUPR不能混称。** 主任务/SSM旧`aupr/aupr_mixed`字段实际是average_precision_score；本次下游是梯形AUPR，Figure3的ap是AP。原数字保留；如全文要统一为梯形AUPR，需要另算全部相应主表，不能只改标签。
4. 当前公共克隆含全部下游输入、源头、历史参考候选及搜索重建代码；全主干20轮训练、全体结构特征生成仍需公共URL待定的大数据包。
5. 本仓库没有完整LaTeX主稿。另一个本地`GSBind_RECOMB27`草稿有已有未提交修改，abstract/experimental_setup/results仍含legacy pooled AUPR、cryptic、MPP及TODO；本次不覆盖该仓库。这里提供的当前英文Methods/Results需整合入实际主稿后逐句审查，不能仅凭工程更新宣布可投稿。
6. 本协议没有新置信区间或显著性检验，不沿用旧模型p值。AM独立出版书目仍未核实，其余引用见 `downstream_references.bib`。

## 复核入口

```bash
python main.py downstream --action replay --output runs/selected_replay
python main.py downstream --action train --output runs/source_search_reconstruction
python main.py summarize --task all --output runs/current_maintext_audit
python tools/audit_delivery.py
python main.py verify
python -m pytest -q
python tools/check_publication.py
```

当前重放、搜索重建、文件校验与测试以本次 `provenance/current_release_checks.json`、`current_maintext_numerical_audit.json` 和 `downstream_selected_search_validation.json` 为准，不把历史固定预算报告当作当前通过记录。
