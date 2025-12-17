# 数据分析可视化

## 文件命名方法

- **评估结果文件**：遵循 `*_eval.jsonl` 命名格式，其中包含温度参数信息。例如：`amc_baseline_temp06_eval.jsonl`（温度为0.6）、`amc_instruct_temp10_eval.jsonl`（温度为1.0）。
- **可视化文件**：以 `amc_` 开头，后面接可视化类型的描述，例如：`amc_temperature_effects.png`（温度影响图）、`amc_model_comparison.png`（模型对比图）。
- **结果汇总文件**：以 `amc_results_summary.csv` 命名，包含所有模型和温度组合的性能指标。

## 文件夹内容介绍

```
├── analyze_amc_example.py    # 主分析脚本
├── requirements.txt          # 项目依赖
├── res                       # 结果数据目录
│   └── cleaned               # 清洗后的评估数据
│       ├── aime_instruct     # AIME指令模型评估数据
│       ├── amc_baseline      # AMC基线模型评估数据
│       ├── amc_instruct      # AMC指令模型评估数据
│       └── math500_baseline  # Math500基线模型评估数据
└── visualizations            # 可视化结果目录
    ├── amc_majority_vote_improvement.png  # 多数投票改进图
    ├── amc_model_comparison.png           # 模型对比图
    ├── amc_pass_at_k_curve.png            # pass@k曲线图
    ├── amc_results_summary.csv            # 结果汇总CSV
    └── amc_temperature_effects.png        # 温度影响图
```

### 核心文件说明

- **analyze_amc_example.py**：包含数据加载、指标计算、结果分析和可视化的完整脚本。
- **requirements.txt**：列出项目所需的Python依赖库（pandas、numpy、matplotlib）。

### 数据目录说明

- **res/cleaned/amc_baseline/**：存放Qwen/Qwen2.5-Math-1.5B模型在AMC2023数据集不同温度下的评估结果文件。（我多跑了一组0.8）
- **res/cleaned/amc_instruct/**：存放Qwen/Qwen2.5-Math-1.5B-Instruct模型在AMC2023数据集不同温度下的评估结果文件。（我多跑了一组0.8）
- **res/cleaned/aime_instruct/**：存放Qwen/Qwen2.5-Math-1.5B-Instruct模型在AIME2025数据集不同温度下的评估结果文件。
- **res/cleaned/math500_baseline/**：存放Qwen/Qwen2.5-Math-1.5B模型在Math500数据集不同温度下的评估结果文件。

### 可视化目录说明

- **amc_temperature_effects.png**：展示不同温度对模型性能的影响。
- **amc_model_comparison.png**：对比基线模型和指令模型在不同温度下的性能。
- **amc_pass_at_k_curve.png**：展示不同k值下的pass@k性能曲线。
- **amc_majority_vote_improvement.png**：展示多数投票相对于pass@1的性能改进。
- **amc_results_summary.csv**：包含所有模型和温度组合的性能指标汇总。

## 运行方法

1. 确保已安装所需依赖：

```bash
pip install -r requirements.txt
```

2. 运行分析脚本：

```bash
python analyze_amc_example.py
```

3. 查看结果：
   - 控制台将输出性能指标表格
   - 可视化结果将保存到 `visualizations/` 目录
   - 结果汇总将保存为 `visualizations/amc_results_summary.csv`

## 性能指标说明

- **pass@k**：表示在k次尝试中至少有一次正确的概率。
- **maj@1**：表示多数投票的正确率。

## 注意事项

- 脚本默认从 `res/cleaned/` 目录加载数据，如需修改路径，请在 `main()` 函数中调整 `base_dir` 参数。
- 可视化结果将自动保存到 `visualizations/` 目录，如该目录不存在将自动创建。
