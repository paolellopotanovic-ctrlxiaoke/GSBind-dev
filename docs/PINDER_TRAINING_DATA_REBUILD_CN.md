# PINDER API 训练/测试集完整重建流程

## 目标

本流程把历史 E11 训练集构建逻辑整理为一条可恢复、可审计、直接使用 PINDER 公开资源的命令。默认数据源是官方 `pinder` Python API 的 **2024-02** release；索引通过 `pinder.core.get_index()` 获取，缺失的 native holo PDB 通过官方发布清单下载，不依赖项目服务器或私有中间 CSV。

输出包括：

- `datasets/pinder_rebuild/train.csv`：可直接被 `gaussian_direct.preprocessing` 和训练 loader 消费的 manifest；
- `datasets/pinder_rebuild/test.csv`：PINDER 官方 test split 的双侧 receptor manifest；
- `datasets/pinder_rebuild/structures/`：从 native holo complex 裁剪出的单链 receptor PDB；
- `datasets/pinder_rebuild/events/`：directed event 与同 receptor-holo union 中间层；
- `datasets/pinder_rebuild/build_report.json`：来源、过滤计数、选择计数和不变量；
- `datasets/pinder_rebuild/split_contract.json`：最终 train/test 文件哈希与泄漏检查。

## 安装与运行

```bash
cd GSbind
python -m pip install -e '.[pinder,test]'

# 建议先做 4 个系统的 API/坐标/manifest smoke；这不是正式数据集。
python main.py rebuild-pinder \
  --output-dir datasets/pinder_rebuild_smoke \
  --cache-dir "$HOME/.cache/pinder/2024-02" \
  --workers 2 --chunk-size 2 --limit 2

# 完整重建。显式 anchor 到仓库中已发布的 E11 20,801 行历史选择，
# 因此选择流程保留 E11 当时使用的样本总体，而不是重新发明一个训练池。
python main.py rebuild-pinder \
  --output-dir datasets/pinder_rebuild \
  --cache-dir "$HOME/.cache/pinder/2024-02" \
  --anchor-manifest datasets/main/train.csv \
  --workers 16 --chunk-size 512

# 生成现有训练代码所需的 compact NPZ。
python main.py --config configs/pinder_rebuild.json preprocess
```

`--cache-dir` 就是官方 pinder 包的 `PINDER_DATA_DIR`。流水线会固定 `PINDER_RELEASE=2024-02`；这是当前官方 R2 下载器支持的公开 release，也是历史清单的来源版本。全量重建会下载和处理大量 native PDB，磁盘、网络和运行时间均显著高于只下载 index。

不需要保留 E11 历史样本总体时，可省略 `--anchor-manifest`。此时使用同一质量、泄漏、union、cap5 与序列多样性规则重新做确定性选择；它是一个干净的 API 原生训练池，但行数和逐行样本不承诺等于历史 20,801。

## 训练来源与测试集

训练候选来自官方 PINDER `train`、`val` 和 `invalid` 三个非 test split。官方 test split 完整保留为独立测试集，不参与训练筛选。

泄漏规则与历史 E11 最终构建一致：

1. 收集官方 test systems 的 `cluster_id_R` 与 `cluster_id_L`，得到双侧 side-cluster union；
2. 任何 train/val/invalid system，只要任一侧 cluster 落在该 union 中，即从训练候选删除；
3. 最终 `train.csv` 再次检查不得含任何官方 test side cluster；
4. train/test case ID 必须无交集，cluster 内样本数不得超过 cap5。

官方 test manifest 保留每个 system 的 R/L 两个 receptor 方向；实际测试统计应明确使用 case/system 聚合，避免把同源双侧方向误当作两个独立生物 case。

## 标签与数据清理

每个官方 native holo complex 下载后执行：

1. 读取 R/L 两条蛋白链；
2. 保留标准氨基酸 residue，删除氢和替代构象；
3. 用 non-hydrogen heavy atom 计算 5 Å inter-chain contact；
4. 分别生成 R→L 与 L→R 两个 directed receptor event；
5. 对同一 `receptor_holo_pdb`、同一序列和同一有序残基契约的事件做 label OR、mask intersection；
6. 重新计算 `num_residues`、`num_positive_labels` 和 positive rate。

最终质量契约为：

- `40 <= num_residues <= 1000`；
- `num_positive_labels >= 3`；
- `mask_coverage >= 0.95`；
- `low_mapping_quality == 0`；
- `coordinate_union_status == "ok"`；
- 重算 positive rate `<= 0.75`；
- receptor 序列可用且非空。

代码从不信任中间 CSV 中旧序列化形式的 `positive_rate` 字段；该值始终由 `num_positive_labels / num_residues` 重算。这个修正对应历史清单中已发现的 stale-field 问题。

## cap5 与序列多样性

strict cluster 规则保留历史逻辑：`seq:*` 本地 fallback 统一 recollapse 为 `cluster_p`，其余官方 cluster 原样保留。同一 strict cluster 内每个 receptor 序列 SHA1 只能出现一次，全局 cluster cap 为 5。

打分 tuple 从小到大优先选择：

```text
1 - mask_coverage
holo_event_failure_count
-holo_event_success_count
abs(num_residues - median_num_residues) / median_num_residues
sha256(receptor_holo_pdb)
```

在满足唯一序列约束时，算法还按全局序列使用计数均衡重复家族，最后再次强制 cap5。

## Anchor 模式

`datasets/main/train.csv` 是 E11 实际使用的 20,801 行历史选择。显式传入该文件时：

- `receptor_pdb_relpath` 被转换回 PINDER holo ID；
- 只有重新进入 API 坐标质量池的 anchor 才保留；
- anchor 优先于新样本；
- 最后仍必须通过官方 test side leakage、cap5 和 cluster 内唯一序列检查；
- `anchor_rows_not_in_candidate` 等审计写入 `build_report.json`。

因此 anchor 模式不是跳过筛选，而是把历史选择作为优先候选重新过完整契约。历史 20,801 样本总体来自四条来源链：strict official train、official validation、invalid train-side overlap、invalid val-side overlap，以及最终 full rebuild 新增样本。完整来源计数在 `build_report.json` 与事件 parquet 中保留。

## 从重建结果继续训练

```bash
python main.py --config configs/pinder_rebuild.json preprocess
python main.py --config configs/pinder_rebuild.json train
```

`configs/pinder_rebuild.json` 的 `expected_train_rows=0` 表示不把新 API 选择硬绑到历史行数；如果使用 `--anchor-manifest datasets/main/train.csv` 且要求严格复现历史样本总体，可在完整构建完成后把该值改回 20,801 作为本机契约检查。

## 与历史 20,801 的关系

- 样本总体：anchor 模式以公开的 `datasets/main/train.csv` 固定，不依赖本地私有 anchor。
- 原始来源：index 与 PDB 全部改为官方 2024-02 API。
- 标签：从 native holo 坐标重算 5 Å heavy-atom contact，并重新做同 holo union。
- 不变量：非 test 来源、双侧 test-cluster 排除、长度/正例/覆盖率/rate、cap5、cluster 内唯一序列均强制执行。
- 不承诺：不同 PINDER mirror 或上游文件变更时，新下载字节的文件级 SHA 必须等于历史 CSV。样本与标签契约由 `split_contract.json` 和 `build_report.json` 验证。

本流程使公开用户能够在没有历史项目服务器或私有 work 目录的情况下，从官方 API 得到可预处理、可训练的完整划分。

## 公开 API smoke 验证记录

2026-10-27 已使用官方 `pinder==0.6.0`、release `2024-02` 完成真实网络 smoke：官方 index 2,319,564 行、test side clusters 2,124 个；下载 native PDB、重算 5 Å 标签、写出单链结构、train/test manifest、split contract 和 SHA256 全部通过。该验证只覆盖 deterministic prefix，不代表已完成全量 20,801 anchor 复核。机器无关摘要见 `results/pinder_api_smoke/VALIDATION.json`。
