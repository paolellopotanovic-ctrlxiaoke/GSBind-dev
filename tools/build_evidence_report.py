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
    functional = functional[functional.region.eq('low_noncontact')&functional.method.eq('full_probe')&functional.condition.isin(['baseline','S_retain50','S_retain00'])]
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
        '', '## 5. SSM220：高binding score应用','',markdown(ssm),
        '', 'SSM220保留完整220输入、原始替换测量和标签构建。只使用binding score，不加入SSM hidden分类器。primary18保留质量/区间过滤及足够有效替换，未知不作为阴性。不能以此声称hidden超越埋藏度。',
        '', '## 6. KRAS/Src：实验功能后果与表面依赖','',markdown(functional),
        '', '两个体系、三个结构输入；按parent分别汇总，不把partner/构建体当作独立家族。保留原冻结功能读出及其训练来源，不替换为本轮Allo新探针；原始实验能量、功能/折叠测量和完整干预均保留。其数值是实验体系应用，不是新变构位点分类指标。',
        '', '## 7. 图表、附录与复算','',
        '正文性能表只放AUPR，MCC/F1/AUROC及阈值在附录。四图结构见FOUR_FIGURE_PLAN_CN.md；python main.py figures生成可复算基础图及source_data。python main.py evidence --task allosteric重算四对照、全部来源迁移、新full干预、预算及CI；不触发训练。python main.py fit-probes --cohort both --output runs/refit_probes只训练四种独立线性探针。',
        '', '当前活动工程覆盖原112→171/旧HGB变构展示，不再同时维护旧论文口径；主任务、SSM220和KRAS/Src不同端点保留。代码与数据完整不代表各来源自动取得公开再分发许可。']
    (ROOT/'docs/PAPER_EVIDENCE_REPORT_CN.md').write_text('\n'.join(text)+'\n')
    print('Updated complete paper evidence report')


if __name__=='__main__':
    main()
