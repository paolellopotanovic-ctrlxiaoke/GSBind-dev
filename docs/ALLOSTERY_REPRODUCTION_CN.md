# 变构四对照：完整复现与论文替换口径

## 1. 唯一当前设计

geometry=RSA/8Å packing/三类DSSP共5维；amino=20维残基身份；hidden=128维融合表示；full=score+geometry+AA+pLDDT+hidden共155维。没有额外logit，没有HGB/MLP，没有旧112→171活动结果。所有标签、划分和方法固定，未按测试表现选择。

AlloBench383蛋白和AM127蛋白分别nested outer5/inner3。MMseqs identity≥0.30且双向coverage≥0.80连通组件；C=.1/1/10；训练侧StandardScaler/组件蛋白均衡权重/训练侧OOF阈值。正文为组件宏平均AP：蛋白内计算，再同源组件内均值，最后组件均值。完整MCC等在附录。

## 2. 无训练重算当前全部结果

普通Git克隆无需大hidden即可运行`python main.py summarize --task allosteric --output runs/public_recompute`，从公开保存的预测重算指标。下面的冻结头重放需要先按`data_release/README.md`安装完整数据；两种运行模式不混称为重新前向。

```bash
cd GSbind
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python main.py evidence --task allosteric
python main.py rebuild-allosteric-labels
python tools/build_evidence_report.py
python main.py figures
```

重放本地冻结四头，并核对reference_predictions中的native概率；七条件真实surface hidden重放到本轮155维full头。只固定native score及显式结构，不重新拟合截断条件。输出全队列/低score、两类E11清洁分层、来源独立迁移、Top10新增/丢失及10000次组件bootstrap/2000次符号翻转。

## 3. 从冻结表示重新训练四种探针

```bash
python main.py fit-probes --cohort both --output runs/refit_probes
python - <<'PY'
from pathlib import Path
from gsbind.allostery import run
root = Path.cwd()
run(root, root/'runs/refit_evidence', models=root/'runs/refit_probes')
PY
```

重新拟合不覆盖checkpoints；因软件版本变化出现细微数值差异需说明。AlloBench和AM各自训练只证明任务内可读性；Allo→AM/ASBench/PASSer_ASD_source/CASBench/ASD_AlloPPI固定迁移不用目标标签训练。来源相关组件用相应来源外层留出头，source_component_disjoint才是来源完全独立子集。

## 4. 原始数据到输入/标签

全部当前输入/标签在服务器完整包中，安装后不需要联网即可复算；Git不包含全部大文件。要审计或从源文件重建，可从仓库根目录依次运行：

```bash
python tools/allosteric_data/download_sources.py
python tools/allosteric_data/prepare.py
python tools/allosteric_data/remap.py
python tools/allosteric_data/external.py
python tools/allosteric_data/complete_ppi.py
MMSEQS_BIN=/path/to/mmseqs python tools/allosteric_data/homology.py
python tools/allosteric_data/materialize.py
python tools/allosteric_data/casbench.py
```

prepare优先使用本地参考manifest和已存在结构，不重复下载当前504输入；AFDB新版本需要重新序列校验。源PDB/ligand到AFDB的映射检查残基身份，失败原因保留。CASBench使用固定每酶条目PDB字典序首个代表，不能重新按性能选代表。

`materialize`从完整七条件hidden构建六任务；应先物化共同特征，再运行CASBench添加该任务。新增结构需用主工程preprocessing生成compact；已有504个compact全部提供。完整前向入口：

```bash
python tools/allosteric_data/features.py --manifest datasets/allosteric/input_manifest.csv --device cuda:0 --fresh-full
```

默认缓存存在则复用；若确需重算全部前向，应先将features移至明确备份位置再运行，不要原地覆盖只读硬链接数据。完整E11原子前向捕获实际Gaussian输入，重放七条件并检查baseline/identity。DSSP仅此阶段需要；标准复算只用保存特征。

## 5. 数据版本及生物端点

AlloBench430身份/2257记录完整注册，383严格映射；AM原始128蛋白/1969替换最终127/1801位置。ASBench168蛋白、PASSer源表288、CASBench70代表、AlloPPI21。各来源有重叠，不能累加为独立证据。

AM随机背景不等于全部实验中性，不预测具体替换强度。位点注释任务不证明完整变构传播路径。CASBench本轮不是催化位点预测；AlloPPI本轮不是PPI调节强度预测。PASSer匿名口袋包缺失身份映射，当前保留源表重建命名。

hidden不显式读入score，但中心化参考集合由native score定义；不称score-independent。当前四对照显示表示可用性和完整融合收益，不由full−geometry独自隔离hidden贡献；Gaussian特异依赖另看实际S通路干预。

## 6. 其他生物应用

SSM220仅高binding score应用和原表面截断保留。KRAS/Src的实验标签、功能/折叠读出和原功能权重保留；那是另一端点，不被新Allo分类头替换。旧论文的变构章节、Figure3底表、命令和旧HGB结果已替换为当前四对照，不在本工程并列维护历史。
