# 全位置功能应用：输入、冻结读出与补充分析协议

## 正式评价范围

主分析为全部具有配对功能/折叠实验测量的位置：KRAS 187个唯一位置、六partner共1118条记录；Src FL和KD各269个位置。结果先按实验、再构建体、最后parent汇总；两个体系不合并成一个独立蛋白总体。低分、非接触及其交叉只是补充分层。

`datasets/functional/experimental_residues.csv`仅保留实验标签、映射、测量与接触元数据，不再夹带旧hidden/score预测。全部当前预测在`results/functional/`。`complete_features.parquet`与`hidden.npy`按相同完整蛋白残基顺序对齐；native score及七条件hidden来自同一GPU后端完整E11前向。

## 冻结读出的来源

- `full_probe/basic_probe/score_probe`来自原功能应用使用的AlloBench注释线性头；KRAS采用原来源留出fold2，Src采用原来源五头均值。权重与原训练来源不改，扩展到全部功能实验位置，不用功能标签重新训练。
- `current_geometry/current_amino/current_hidden/current_full`为当前正式AlloBench四方法头的补充迁移。实际MMseqs命中在`datasets/functional/source_homology_hits.tsv`；新来源中KRAS父全长命中fold1，Src父全长命中fold2，使用对应来源留出头。Src KD继承父全长规则，但存在横跨其他fold的激酶域局部同源，不能称完整域级同源独立。
- `joint_hidden_logit`是原联合头中hidden项的线性贡献，`joint_basic_logit`是其基础项，不是另训的纯hidden探针。独立纯hidden结果使用`current_hidden`。

每个条件的hidden分别以固定native score<0.5残基集合中心化；集合为空时用全部位置。参考集合不随干预变化。中心化使用低分不等于评价只使用低分。

原功能头训练在原来源低分位置上；应用到全部功能位置是固定读出的范围扩展，不将其概率解释为经过高分子集校准的功能概率。

## 实验终点与排名

功能/折叠终点为原始实验替换效应绝对值的位置级均值。功能影响大不区分激活或抑制，不等价于只预测单个替换。Src原始功能/丰度终点另列补充表。KRAS的noncontact相对具体partner界面；Src沿用ANP重原子距离标记，两者不混称同一个生化定义。

主要指标为Spearman rho、Top10与前20%位置的实验效应均值和富集。前20%使用ceil(0.2N)，同分按位置升序打破。新增/丢失只在同一位置集合与预算比较，真值前列集合也采用相同预算。不同体系实验尺度不同，不比较绝对效应单位大小。

## Gaussian-surface计算干预

保持模型、原冻结读出及native score/几何/AA/pLDDT不变，只替换干预后的hidden。baseline、identity、retain50、retain00和三个同模长噪声均保存；identity要求逐值一致。表面反馈和融合均属于本项目Gaussian-surface模块。

三个输入的当前`surface/*.npz`均来自统一GPU前向，不再用原混合后端缓存的差量叠加。原CPU/GPU比较与缓存差异记录保留在`results/functional/cpu_gpu_baseline_audit.csv`和`forward_cache_audit.csv`，它们是来源审计，不是当前预测输入。

## 置信区间与控制分析

序列块bootstrap 10,000次、块长10。同一parent位置在不同partner/构建体共享抽样权重；使用带权并列秩精确计算重采样Spearman。结果只表示两个实验体系内的位置抽样不确定性。预算重采样在每次抽样中重新执行相同位置预算排序。

偏秩相关控制score、RSA/packing/DSSP、AA、pLDDT，另加折叠影响。每次bootstrap重算秩和控制回归；这个回归是标签关联调整，不是部署时另训预测头。

全部高低分、非接触、反向score、预算与噪声结果完整保留。主文的高分信息增量来自变构注释评价，强表面依赖来自KRAS/Src全位置功能实验；不把两个不同终点合为一个逐层均成立的结论。

## 可复算入口

```bash
python main.py summarize --task functional --output /absolute/path/to/recompute
python main.py functional-statistics --output /absolute/path/to/recompute/functional --bootstrap 10000
python main.py allosteric-strata --output /absolute/path/to/recompute/allosteric_strata
python tools/build_evidence_report.py
python main.py figures
```

`summarize`重新执行冻结头、位置对齐和指标计算，不需要大体积allosteric hidden。`functional-statistics`重算较耗时的全部块bootstrap；交付的完整10,000次结果已在`results/functional/`。重新生成结构前向可使用`python main.py surface-forward --manifest datasets/functional/manifest.csv --output runs/functional_forward --device cuda:0`，CPU与GPU可能有不可忽略的后端差异，应对全部条件使用一致后端。
