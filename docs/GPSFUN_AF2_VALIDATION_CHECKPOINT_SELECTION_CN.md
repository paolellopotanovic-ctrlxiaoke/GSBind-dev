# GPSFun AF2 full375 验证集与 E11 选择协议

## 边界

主 backbone 的 E11 不是在最终测试集上选出的，也不是 Lightning early stopping。历史流程先完整训练 E0–E19、逐 epoch 保存 checkpoint，然后在固定的 GPSFun 官方 protein split AF2 版本 full375 上统一验证，最后用性能-only 综合分选择 checkpoint。最终 Bound/Unbound 等测试集只用于之后的报告。

E20 不存在；候选 epoch 严格为 0–19。

## 验证集契约

验证 manifest 的 SHA-256 为：

```text
e6c10cba5ec4a2b43b39d7509fc95400f924baeaacaa30afa046c3051a9273ce
```

固定契约如下：

| 项目 | 值 |
|---|---:|
| protein cases | 375 |
| valid residues | 77,978 |
| positive residues | 11,350 |

构建方式是 GPSFun 官方 canonical sequence 到 AF2 结构的严格序列映射和 `write_crop`。前 330 例使用既有 strict exact-domain AF2 版本；后 45 例由本地 ColabFold/AF2 pTM model 5 按官方 canonical sequence 预测后，复用同一套 `official_canonical_v1` 与 Biopython `PairwiseAligner` 逻辑重建。45 例的序列 coverage 和 identity 均为 1.0；alignment 没有丢弃残基，但 v1 label mask 使 8,175 个结构残基进入契约时为 8,160 个 valid residues，正例保持 1,202。

这里的 “对齐/裁剪” 是序列级别 strict-domain 映射，不是把 AF2 结构向 holo 结构做三维 superposition。不能把该集合描述为未裁剪的 full-sequence mask-all 版本；早先 `official_direct_v2` 的 8,175 valid-residue 口径已被本契约取代。

## 指标

每个 epoch 计算 16 个性能指标：

- pooled：AUPR、AUROC、P@10、P@20、P@50；
- case-mean：AUPR、AUROC、P@10、P@20、P@50；
- case-median：AUPR、AUROC、P@10、P@20、P@50；
- macro recall：每 case 在阈值 0.5 的 `TP/(TP+FN)`，再对 375 个 case 不加权平均。

验证 loss、weighted loss 和 Brier/calibration 指标只可作为辅助诊断输出；主选择分明确排除它们。

## 主选择公式

先在四个指标家族内部平均原始值，再用候选 E0–E19 的 population standard deviation（`ddof=0`）分别 z-score，最后四个家族等权平均：

```text
score =
  0.25 * z(mean pooled family)
+ 0.25 * z(mean case-mean family)
+ 0.25 * z(mean case-median family)
+ 0.25 * z(macro recall)
```

该写法先压缩家族内指标，避免 P@10/20/50 重复放大权重。16 个指标各自等权 z-score 后平均（`all16_equal_z`）只作为敏感性检查，不是主选择器。

历史结果中 E11 的 `family_balanced_z=0.9788990205747575`，高于 E5 的 `0.850614` 和 E1 的 `0.533716`。`all16_equal_z`、16 指标平均排名和家族平衡排名的敏感性检查也均为 E11 第一。

## 代码入口

配置位于 `configs/main.json` 的 `training.checkpoint_selection`。训练完成后，`GaussianDirectApp.train()` 会在 global rank 0 自动执行：

```bash
python main.py train --config configs/main.json
```

对已保存的完整 E0–E19 run，可以单独执行：

```bash
python main.py --config configs/main.json select-checkpoint
```

执行前必须安装 GPSFun AF2 full375 manifest 和 compact NPZ cache 到配置指定的外部 datasets 目录。代码会校验 manifest SHA、case/residue/positive 契约、完整 epoch 集合和所有 16 项指标有限性；任一项不匹配都会失败而不是静默降级。

输出目录包含：

- `epoch_XX/metrics.json`
- `epoch_XX/case_metrics.csv`
- `epoch_XX/residue_predictions.csv`
- `composite_summary.csv`
- `selected_checkpoint.json`
- `selection_contract.json`
- `VALIDATION_SELECTION_COMPLETE`

`tests/test_gpsfun_checkpoint_selection.py` 冻结了历史 E0–E19 composite 表，要求重算后仍选择 E11 且分数逐位接近 `0.9788990205747575`。
