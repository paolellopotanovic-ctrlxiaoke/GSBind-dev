# 当前主文证据与公开工程复核

日期：2026-10-22。复核对象为本仓库，不把别的论文草稿的未核实结论视作已完成。

## 数值与方法检查

- 当前下游唯一主表：`results/downstream/METRICS.csv`。3任务×4探针共60头，另存2种未经校准raw score，18行指标。每一折指定50/50/10上限，提前收敛取实际末次，绝不按测试指标或测试loss取最优。原逐次预测、发布头重放、公开输入重训三者数值逐行核对。
- 输入与标签：AlloBench216556行/7409正例/383蛋白/282组件；AM1801行/839正例/127蛋白/112组件；Src538条构建体位置记录、269唯一共享物理位置、完整特征812行。分类组件/蛋白互斥、Src同位点共折和±5排除均由代码验证。
- 主任务与SSM：从公开保存预测重新计算全部CSV，逐表核对原发布结果，见 `provenance/current_maintext_numerical_audit.json`；不是重跑20轮主干训练。主任务20801训练链、988 holo、874 apo/AF2；主要E11 case-macro AP为0.7914999169、0.7281149439，具体末位以CSV为准。
- Figure3：保留原Gaussian-surface与配对结构可靠性数据/代码，纳入原单元测试。部分从原始结构重建的脚本依赖外部结构/原始逐残基干预源目录，不能声称所有原始前向均已在公共克隆独立复跑。公共CSV、绘图底表与已有测试不等于全部原始实验重新执行。
- forward核对：`model.py`仅前两次surface pass后更新残基；第三次surface context与第二次反馈后的残基状态融合。已修正文档“三轮完整残基反馈”的误述，未改动E11计算图和权重。

## 必须保留的诚实边界

1. **AP ≠ 梯形AUPR。** `gsbind.metrics.binary_metrics`中的历史`aupr`/`aupr_mixed`实际调用average_precision_score，因此主任务/SSM旧CSV是AP。当前下游严格为梯形AUPR。旧数字未偷换；全文如统一写AUPR，必须统一定义并另算所有相关主表，而不是只改列名。Figure3中ap亦为AP。
2. 固定50/50/10是探索后确定的预算，不是从未看过测试集。当前代码不逐折选择测试checkpoint，但无法消除此前看结果决定预算的选择偏差；应报告探索性OOF，不包装独立泛化检验。
3. 下游任务分别训练，不再讲“只在AlloBench训练、直接迁移到Src”；Src的FL/KD属于同一蛋白，不能算两个独立蛋白或两个互作partner。AM对结构基线只小幅领先，不沿用旧模型显著性。
4. 当前公开仓库足以重训全部下游线性头；全主干20轮训练、全体结构前向仍需尚未有公共URL的旧大数据包。`data_release/bundle.json`状态没有伪装成已公开。
5. 本仓库没有完整LaTeX主稿。另一个本地`GSBind_RECOMB27`草稿的abstract/experimental_setup/results还保留legacy pooled AUPR、cryptic小分子和MPP等主张，并有明确TODO边界。该草稿不能因本次工程更新就宣布核对完毕/可投稿；本次不覆盖那份仓库已有未提交修改。新方法与结果替换正文已提供在 `DOWNSTREAM_FIXED_PROTOCOL_CN.md`，需应用到实际主稿后再逐句核对。
6. AM来自提供的原始整理表，目前缺可核验的独立出版书目；不能捏造作者/DOI。完整引用仍需按原始文献表补足。Src来源保留Science Advances DOI与原始补充表，AlloBench保留官方来源标识和记录级PubMed。

## 复核入口

```bash
python main.py downstream --action train --output runs/fixed_refit
python main.py summarize --task all --output runs/current_maintext_audit
python tools/audit_delivery.py
python main.py verify
python -m pytest -q
python tools/check_publication.py
```

`current_maintext_numerical_audit.json`、`downstream_fixed_refit_validation.json`、`downstream_runtime.json`与本次工程执行记录分别说明检查内容；不沿用旧模型的绿色审查报告替代当前执行。
