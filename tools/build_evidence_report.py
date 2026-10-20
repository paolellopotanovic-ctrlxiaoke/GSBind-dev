from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def markdown(frame):
    def cell(value):
        return f'{value:.6f}' if isinstance(value,float) else str(value)
    lines = ['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines.extend('| '+' | '.join(cell(value) for value in row)+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)


def main():
    main = pd.read_csv(ROOT/'results/main/case_macro_metrics.csv')
    surface = pd.read_csv(ROOT/'results/main/surface_case_macro_metrics.csv')
    allo = pd.read_csv(ROOT/'results/allosteric/MAIN_AUPR_TABLE.csv')
    statistics = pd.read_csv(ROOT/'results/allosteric/ALL_AUPR_CONTRASTS.csv')
    primary = statistics[statistics.region.eq('all')&statistics.e11_overlap_filter.eq('all')]
    ssm = pd.read_csv(ROOT/'results/ssm220/case_macro_metrics.csv')
    ssm = ssm[ssm.region.isin(['all','baseline_high'])&ssm.condition.isin(['baseline','S_retain00'])]
    functional = pd.read_csv(ROOT/'results/functional/parent_metrics.csv')
    functional = functional[functional.region.eq('all')&functional.method.eq('full_probe')&functional.condition.isin(['baseline','S_retain50','S_retain00'])]
    strata = pd.read_csv(ROOT/'results/allosteric_strata/MAIN_AUPR_BY_SCORE_STRATUM.csv')
    high = strata[strata.region.eq('baseline_high')]
    high_statistics = pd.read_csv(ROOT/'results/allosteric_strata/paired_aupr_contrasts.csv')
    high_statistics = high_statistics[high_statistics.region.eq('baseline_high')&high_statistics.comparison.eq('full_minus_geometry')]
    functional_main = pd.read_csv(ROOT/'results/functional/MAIN_ALLPOSITION_RESULTS.csv')
    functional_main = functional_main[functional_main.target.eq('functional_magnitude')&functional_main.region.eq('all')&functional_main.method.isin(['raw_score','burial','packing','basic_probe','full_probe','current_hidden','current_full'])]
    partial = pd.read_csv(ROOT/'results/functional/partial_rank_bootstrap.csv')
    partial = partial[partial.region.eq('all')&partial.method.eq('full_probe')]
    separation = pd.read_csv(ROOT/'results/functional_separation/paired_block_bootstrap.csv')
    separation = separation[separation.analysis.eq('within_score_quintiles')&separation.method.eq('full_probe')
        &separation.comparison.isin(['rho','full_probe_minus_raw_score','full_probe_minus_burial','full_probe_minus_packing'])]
    columns = ['experiment','subset','proteins','geometry','amino','hidden','full']
    text = ['# GSbind论文最终证据报告','','更新：2026-10-20。当前变构分析仅保留geometry/amino/hidden/full四种设计；本报告由GSbind本地代码与真实冻结读出重算，不依赖旁边的研究工作目录。',
        '', '公开说明：本报告描述完整主工程证据。Git已提供代码、权重、名单、标签、结果CSV和保存预测；大结构/compact/七条件hidden等外置，下载链接待补，见data_release/README.md。普通克隆用main.py summarize从保存预测重算；完整冻结头重放和训练需要安装外部数据。',
        '', '## 1. 论文主线','',
        'GSbind以原子编码和Gaussian-surface处理、反馈与融合完成结合位点预测。主任务在统一MMseqs两种测试态评价；下游用变构注释/突变位置验证表示的应用价值；SSM220验证高binding score应用；KRAS/Src提供实验功能关联及Gaussian-surface计算依赖。模型机制依赖不自动等同于真实生物信号传递。',
        '', '## 2. 创新架构与完整工程','',
        '固定Gaussian-direct128 E11：128维、vg、三轮co-update、3,355,001参数。AtomEncoder提供原子/残基表示，Gaussian field/pair与多尺度surface构建表面上下文，经core_residue_gate/update与same_level_fusion回传残基，direct head输出结合score。完整主架构包括表面处理、反馈和融合，不仅是pair参数。',
        '', '20,801训练记录、988 holo与874 APO/AF2官方MMseqs测试记录，原始结构、compact缓存、标签、配置、完整前向和Lightning训练代码均保留。详见ARCHITECTURE.md和ENGINEERING_CONTRACT_CN.md。下游主模型不重训。',
        '', '## 3. 结合位点主任务：正文AUPR','',markdown(main[['panel','model','cases','aupr_mixed']]),
        '', '四模型使用相同测试记录但训练来源可能不同，是任务级比较。没有BC30或逐case oracle主表。MCC/F1/AUROC等完整保存于results/main，不放正文主表。',
        '', '### Gaussian-surface截断','',markdown(surface[['panel','condition','cases','aupr_mixed']]),
        '', '## 4. 变构生物信息：四种唯一设计','',
        '| 方法 | 显式输入 | 维度 |\n| --- | --- | --- |\n| geometry | RSA、8Å堆积、三类DSSP | 5 |\n| amino | 20类残基身份one-hot，无序列窗口 | 20 |\n| hidden | 冻结模型融合hidden | 128 |\n| full | score、geometry、AA、pLDDT、hidden，无额外logit | 155 |',
        '', '所有四种均为训练侧标准化+逻辑回归。hidden参考集合由native score<0.5定义，逐蛋白中心化；因此不显式输入score不等于完全score-independent。纯几何/AA也是监督线性读出，并非未训练的经验规则。',
        '', '### 4.1 数据集与生物问题','',
        '| 来源 | 本轮可用蛋白 | 标签及角色 |\n| --- | --- | --- |\n| AlloBench | 383 | 变构位点注释，任务内nested五折 |\n| AM | 127 | 变构突变WT位置/供方随机背景，任务内五折及固定迁移 |\n| ASBench | 168 | 指定变构配体5Å接触，固定迁移 |\n| PASSer_ASD_source | 288 | 作者ASD源表位点，固定迁移 |\n| CASBench | 70 | 官方酶条目代表结构的变构位点，固定迁移 |\n| ASD_AlloPPI | 21 | PPI调节体系的注释位点，固定迁移 |',
        '', 'AlloBench430原始蛋白/2257记录完整注册，383通过当前映射，其余47原因完整记录；不是430全预测。AM原始128蛋白/1969替换，冲突位置与未追溯记录排除后127蛋白/1801位置，不预测替换强度，不称全部背景为实验中性。PASSer源表不是匿名准备包1049组的精确复现；CASBench91酶条目/2871结构完整保留，按固定字典序代表策略映射70蛋白，不声称全部2871结构均评价。AlloPPI完整50关系/748调节剂注册，但只21蛋白通过坐标映射；不预测PPI调节强度。',
        '', '### 4.2 任务内五折AUPR','',markdown(allo[allo.experiment.isin(['AlloBench','AM'])][columns]),
        '', 'AlloBench383蛋白/282组件，AM127蛋白/112组件，各自outer5/inner3。MMseqs identity≥0.30且双向coverage≥0.80连通组件。C=0.1/1/10仅训练内折选择；组件/蛋白均衡权重。所有方法共享相同位置与划分。',
        '', '### 4.3 固定Allo读出迁移：全队列','',markdown(allo[~allo.experiment.isin(['AlloBench','AM'])&allo.subset.eq('all_evaluable')][columns]),
        '', '目标标签不拟合。相关组件用来源对应留出模型；来源组件独立蛋白使用Allo全数据头。全队列不是全部独立外部样本，不将重叠注释来源累加为独立证据。',
        '', '### 4.4 固定迁移：来源组件独立','',markdown(allo[allo.subset.eq('source_component_disjoint')][columns]),
        '', '### 4.5 全部正文比较与Gaussian-surface依赖','',markdown(primary),
        '', 'AP为混合标签蛋白先计算、组件内平均再组件宏平均；10000次组件bootstrap，2000次符号翻转，seed20261020。full相对geometry/amino支持完整融合表示的应用价值，不把全部差值归因hidden单独贡献；full相对hidden反映显式基础特征的补充。新四设计不单独展示无hidden完整结构基线，因此不从当前四行比较新声称严格条件化hidden归因。',
        '', '表面干预对本轮155维full头重新计算：score/结构/AA/pLDDT固定为native，替换七条件hidden，不重训头。identity等价；保留50%、完全截断与三次同模长噪声分别报告。计算通路依赖不等于真实生物因果。',
        '', '### 4.6 可复现资产','',
        'datasets/allosteric包含504个完整输入、结构/compact、六任务标签、完整430注册审计、七条件hidden、实际MMseqs序列与命中。checkpoints/allosteric仅四种头×两任务×五折及全数据头；src/gsbind/allostery.py与probe_training.py负责冻结重算和独立探针训练。tools/allosteric_data保留下载、源结构映射、同源分组、前向和任务构建代码。所有活动路径本地化。',
        '', '### 4.7 高结合分数内部的额外变构信息','',
        '七组主结果均为完整任务可评价位置，不是低分筛选结果；本次预测与标签逐残基一对一核对。高分固定为native score≥0.5，在同一位置集合比较各方法。',
        '',markdown(high[['experiment','proteins','mixed_proteins','geometry','amino','hidden','full','binding_score']]),
        '',markdown(high_statistics[['experiment','comparison','mean','lower','upper','components','cases']]),
        '', 'AlloBench、ASBench、PASSer和CASBench高分位置中，full相对geometry分别提升7.23、5.70、6.79和5.96个百分点，组件bootstrap区间均高于零。隐藏表示可以细分已经具有较高结合倾向的位置。AM与AlloPPI的完整分层结果同表保留；跨来源存在重叠，不将七组视为七个互相独立队列。分层比较未做跨全部新比较的多重校正。',
        '', '## 5. SSM220：高binding score应用','',markdown(ssm),
        '', 'SSM220保留完整220输入、原始替换测量和标签构建。只使用binding score，不加入SSM hidden分类器。primary18保留质量/区间过滤及足够有效替换，未知不作为阴性。不能以此声称hidden超越埋藏度。',
        '', '## 6. KRAS/Src：全位置实验功能应用与表面依赖','',
        '主分析覆盖全部实验位置，不先按score或非接触条件筛选。KRAS为187个唯一位置、六partner共1118条记录；Src两个构建体各269个位置，共538条记录。主模型和读出均冻结，未使用目标功能标签拟合。',
        '',markdown(functional_main[['dataset','method','rho','top10_effect_mean','top10_fold_enrichment','top20_effect_mean','top20_fold_enrichment']]),
        '', '### 6.1 Gaussian-surface通路依赖','',markdown(functional),
        '', '原完整冻结读出的全位置功能rho：KRAS 0.4272、Src 0.3683；完全截断后为0.1692和0.1711，下降0.2580和0.1972。下降区间分别为[0.0758,0.4549]与[0.1288,0.2650]。所有条件使用统一GPU后端的新E11前向，固定native score/基础协变量，仅替换隐藏表示。',
        '', '### 6.2 控制结构与折叠后的实验功能关联','',markdown(partial),
        '', '控制score、几何、AA、pLDDT后关联仍为0.3076/0.3481；进一步控制折叠后为0.2595/0.3487。前10个候选的实验功能效应均值富集1.85×/2.49×。这两部分分别体现补充功能信息与实验优先排序价值。',
        '', '两个体系、三个结构输入；先assay、后case、再parent汇总，不把partner/构建体当作独立家族。full_probe为原冻结应用头；current_hidden/current_full补充列出当前正式Allo读出的迁移结果，不混称。全部高低分与接触分层、预算新增/丢失、读出来源和后端审计在results/functional及FUNCTIONAL_PROTOCOL_CN.md。',
        '', '### 6.3 相近结合分数内的功能信息分离','',
        '每个具体实验按native binding score固定分为五个等频组，在组内评价功能排序，先组、后assay、case和parent平均。没有重训主模型或读出。该比较直接回答：在结合倾向相近的位置中，隐藏读出是否还能区分实验功能影响？',
        '',markdown(separation[['dataset','comparison','estimate','lower','upper','iterations']]),
        '', '原完整冻结读出组内rho为KRAS 0.3080和Src 0.3749；相对直接使用score分别增加0.2763 [0.0832,0.4404]和0.4537 [0.3431,0.5645]。Src相对burial/packing的增加也有正区间；KRAS相对这两个结构指标的区间跨零。此处与全协变量偏相关共同支持额外功能信息。',
        '', '另外计算观测折叠效应较低一半的功能排序、功能减折叠百分位秩差，全部结果保留于results/functional_separation。它们回答不同实验问题，不把较低折叠影响等同于完全无折叠影响，也不按哪个方法变正来选择阈值。详见FUNCTIONAL_SEPARATION_CN.md。',
        '', '## 7. 图表、附录与复算','',
        '正文性能表只放AUPR，MCC/F1/AUROC及阈值在附录。四图结构见FOUR_FIGURE_PLAN_CN.md；python main.py figures生成可复算基础图及source_data。python main.py summarize --task functional重算全位置功能读出；python main.py functional-statistics重算序列块配对区间；python main.py allosteric-strata重算全位置/高低分变构统计。python main.py evidence --task allosteric重放四方法与完整来源迁移；python main.py fit-probes --cohort both --output runs/refit_probes只训练独立探针。',
        '', '当前活动工程覆盖原112→171/旧HGB变构展示，不再同时维护旧论文口径；主任务、SSM220和KRAS/Src不同端点保留。代码与数据完整不代表各来源自动取得公开再分发许可。']
    (ROOT/'docs/PAPER_EVIDENCE_REPORT_CN.md').write_text('\n'.join(text)+'\n')
    print('Updated complete paper evidence report')


if __name__=='__main__':
    main()
