# 当前论文与工程证据对应

日期：2026-10-23。当前工程、数据仓库与论文采用同一四图故事；发布版本以各仓库提交与主稿证据清单为准。

| 图 | 活动内容 | 唯一结果来源 |
|---|---|---|
| Fig1 | 结构输入、Gaussian表面通路、score与融合hidden两种输出 | 数据仓库 `GSBind_fig1/src/plot_all.py` 的示意图 |
| Fig2 | Bound/Unbound主任务，AP与配对增量 | 数据仓库 `GSBind_fig2`、`source_data/fig2` |
| Fig3 | 合并Unbound构象偏移、界面恢复与空间集中、单独AF2置信度 | 数据仓库 `GSBind_fig3`、`source_data/fig3`；分类为梯形AUPRC |
| Fig4 | AlloBench、AM迁移、Src监督功能读出，主图ABC | 本工程 `results/downstream/METRICS.csv` 与数据仓库 `source_data/fig4/shared/METRICS.csv` |

主稿 `evidence/current_release.json` 保存当前图、协议与指标快照的SHA256。其 `tools/align_current_evidence.py` 校验本工程活动协议与图4协议完全一致、汇总结果一致及主要数值一致。主干E11不重训；下游训练只拟合线性探针，不是端到端微调。Src为同一蛋白的FL/KD构建体位置留出。AM不训练，不参与checkpoint选择。

来源checkpoint选择、候选历史和AM组件路由保留在 `DOWNSTREAM_SELECTED_PROTOCOL_CN.md`、配置及逐fold元数据中，不因论文聚焦信息增量而删除。旧固定50/50/10与AM自训练版本保存于 `history/pre_alignment_20261023/public_downstream_snapshot.tar.gz`；旧Fig3设计说明移至同目录的 `docs/`。历史代码仍可供复查，但不作为新Fig3的活动入口。

当前全主干训练与结构/表面特征再生成仍依赖外部大数据输入，其公开URL待补；不能把下游回放成功解释成已完整重训全部主干。作者信息、完整文献引用和投稿排版由主稿单独检查。
