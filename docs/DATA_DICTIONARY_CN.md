# GSbind数据字典

本页描述完整数据契约。Git公开版保留全部名单、标签、预测与结果表；大结构/compact/七条件hidden和SSM大源表由`data_release/EXTERNAL_FILES.csv`精确登记，下载链接待补。是否已包含某文件以根目录`FILE_INVENTORY.csv`为准。

## 主任务

datasets/main/train.csv为20801训练记录；holo_test.csv为988记录；apo_af2_test.csv为874记录。PDB、compact、标签、四模型逐残基预测和surface预测均保留。manifest路径相对于GSbind根目录，运行入口自动切换至根目录。

## 变构四对照

datasets/allosteric/input_manifest.csv注册全部504输入；raw保存AlloBench、AM、ASBench、PASSer源表、CASBench和ASD AlloPPI原始来源、结构及下载审计。compact和features按输入身份保存真实模型输入及七条件前向结果。

tasks下六个parquet均以uniprot+residue_index_0对齐，label为任务特定标签；matrix_row索引hidden_centered_conditions.npy的共同残基轴。完整矩阵形状为7×N×128，条件顺序由conditions.json定义。每个features/<UniProt>/surface.npz保存7×L×128原hidden及7×L×1原输出logits。

四种设计固定为geometry/amino/hidden/full。geometry为RSA、8Å堆积、三类DSSP共5维；amino为20维残基身份；hidden为128维融合表示；full为score+geometry+AA+pLDDT+hidden共155维，无额外logit。hidden按每蛋白固定native低score集合中心化，不直接输入score，但不称完全score-independent。

AlloBench未注释位置是背景，不是已证实无变构功能。AM仅在已收录/随机背景位置评价，冲突替换标签排除；WT位置不能解释为特定替换效应。CASBench是代表结构子集，PASSer_ASD_source不是匿名原包精确复现。注册、失败和映射原因均保留。

checkpoints/allosteric/<AlloBench|AM>下是四种冻结线性读出，包含五个外层头、全数据头、标准化器、正则、训练阈值、调参与fold_membership。reference_predictions只保留四种native预测，用于独立冻结复算核对。

results/allosteric/<experiment>/predictions.parquet保存全部位置、method、condition、prediction、threshold、outer_fold、transfer_role。case_metrics为逐蛋白；component_metrics先组件内平均；summary再组件宏平均。MAIN_AUPR_TABLE为正文唯一四列；SUPPLEMENTARY_CLASSIFICATION_METRICS保存MCC/F1/AUROC。ALL_AUPR_CONTRASTS保存全部配对bootstrap及符号翻转。

source_component_disjoint是与Allo读出来源无组件共享的迁移子集；E11训练同源清洁分层为另一审计维度。各来源有重叠，不把重复蛋白或注释当独立证据。

## SSM220

datasets/ssm220包含220结构、原始替换和score预测。primary18按替换质量、置信区间及至少18个有效非WT替换构建；未知不变阴性。本任务不包含hidden敏感性探针。

## KRAS/Src

datasets/functional包含两个体系的三个结构输入、原始实验能量、残基功能/折叠效应、完整hidden、surface条件和原冻结功能预测。checkpoints/functional_probe是原功能读出专用权重，不能替换为新Allo分类头。functional指标按parent分别报告，不与位点AUPR混淆。

当前主分析为全部实验位置。`results/functional/complete_predictions.parquet`包含七条件全蛋白读出，`experimental_predictions.parquet`为实验标签对齐结果，`native_experimental_positions.csv`为便于作图的原生位置表；`assay_metrics.csv`/`case_metrics.csv`/`parent_metrics.csv`对应三级汇总。`PAPER_*.csv`为主文表，其他高低分与接触分层供补充使用。实验源表不夹带旧预测列。

`results/allosteric_strata/`保存七组变构完整位置核对、同一native score定义的高低分比较、逐蛋白指标和配对组件区间，不改变原正式任务及迁移协议。

## 数据许可

保留原始来源、URL及身份审计；各来源数据保留其原许可。完整服务器数据目录使用普通硬链接，禁止原地修改只读输入；Git公开工作树使用独立副本，归档解压后也是独立内容。
