# 数据分析可视化

## 文件命名方法

- **评估结果文件**：遵循 `*_eval.jsonl` 命名格式，其中包含温度参数信息。例如：`amc_baseline_temp06_eval.jsonl`（温度为0.6）、`amc_instruct_temp10_eval.jsonl`（温度为1.0）。
- **可视化文件**：以 `amc_` 开头，后面接可视化类型的描述，例如：`amc_temperature_effects.png`（温度影响图）、`amc_model_comparison.png`（模型对比图）。
- **结果汇总文件**：以 `amc_results_summary.csv` 命名，包含所有模型和温度组合的性能指标。

## 文件夹内容介绍

```text
├── analyze_amc_example.py    # AMC数据集专项分析脚本
├── analyze_results.py        # 多数据集综合分析脚本
├── README.md                 # 项目说明文档
├── requirements.txt          # 项目依赖
├── res                       # 结果数据目录
│   └── cleaned               # 清洗重命名后的评估数据
│       ├── aime_baseline     # AIME基线模型评估数据
│       ├── aime_instruct     # AIME指令模型评估数据
│       ├── amc_baseline      # AMC基线模型评估数据
│       ├── amc_instruct      # AMC指令模型评估数据
│       └── math500_baseline  # Math500基线模型评估数据
└── visualizations            # 可视化结果目录
```

### 核心文件说明

- **analyze_amc_example.py**：AMC数据集专项分析脚本，包含数据加载、指标计算、结果分析和可视化功能。
- **analyze_results.py**：多数据集综合分析脚本，支持AMC、AIME和Math500数据集的批量分析和对比。
- **README.md**：项目说明文档，包含文件结构、运行方法和性能指标说明。
- **requirements.txt**：列出项目所需的Python依赖库（pandas、numpy、matplotlib等）。

## 运行方法

1. 确保已安装所需依赖：

```bash
pip install -r requirements.txt
```

2. 运行AMC数据集专项分析脚本：

```bash
python analyze_amc_example.py
```

3. 运行多数据集综合分析脚本：

```bash
python analyze_results.py
```

4. 查看结果：
   - 控制台将输出性能指标表格
   - 可视化结果将保存到 `visualizations/` 目录
   - 结果汇总将保存为 `visualizations/amc_results_summary.csv`（专项分析）和各数据集的结果文件（综合分析）

## 性能指标说明

- **pass@k**：表示在k次尝试中至少有一次正确的概率。
- **maj@1**：表示多数投票的正确率。

## 注意事项

- 脚本默认从 `res/cleaned/` 目录加载数据，如需修改路径，请在 `main()` 函数中调整 `base_dir` 参数。
- 可视化结果将自动保存到 `visualizations/` 目录，如该目录不存在将自动创建。
