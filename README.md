# Commends

## 评估
默认从 prediction 中找文件
- 评估指定文件（带扩展名）：
```bash
python eval.py -f 250915-random.json
```
- 评估指定文件（不带扩展名也可）：
```bash
python eval.py -f 250915-random
```



# FutureX 预测评估系统

一个用于评估AI模型未来事件预测能力的评估框架，支持多种题型的自动化评分。

## 项目概述

FutureX评估系统是一个专门用于评估AI模型在预测未来事件方面表现的工具。系统支持4个难度级别的题型评估，并提供详细的评分机制。

## 功能特性

### 支持的题型级别

- **Level 1**: 单选题 - 二元选择题（Yes/No）
- **Level 2**: 多选题 - 多个选项的F1评分
- **Level 3**: 排序题/数值预测 - 排序匹配或数值误差评估
- **Level 4**: 复杂预测 - 高难度排序或数值预测

### 评分机制

1. **单选题 (Level 1)**: `score = 1` 如果预测正确，否则 `score = 0`
2. **多选题 (Level 2)**: 使用F1-Score评分
3. **数值预测 (Level 3/4)**: `score = max(0, 1 - ((Y - Ŷ) / σ(Y))²)`
4. **排序题 (Level 3/4)**: 完全匹配得1分，部分匹配得 `0.8 × |交集| / k`

### 智能数值标准化

- 支持LLM辅助的单位转换和数值标准化
- 自动处理不同单位间的转换（如：亿元 ↔ 万元，billion ↔ million）
- 回退机制确保在LLM失败时仍能进行基础评估

## 项目结构

```
Prediction/
├── README.md              # 项目说明文档
├── eval.py               # 核心评估脚本
├── dataset/              # 数据集目录
│   └── 250915.json      # 原始数据集
└── prediction/           # 预测结果目录
    └── 250915-random.json # 预测结果文件
```

## 数据格式

### 输入数据格式 (dataset/*.json)

```json
[
  {
    "id": "唯一标识符",
    "prompt": "预测任务的完整提示词",
    "end_time": "预测截止时间",
    "level": 1,  // 题目难度级别 (1-4)
    "ground_truth": "标准答案",
    "Std": null,  // 数值题的标准差（可选）
    "additional values": "额外信息",
    "Description": "题目描述"
  }
]
```

### 输出数据格式 (prediction/*.json)

在输入数据基础上添加 `answer` 字段：

```json
[
  {
    // ... 原有字段 ...
    "answer": "\\boxed{模型预测答案}"  // 新增字段
  }
]
```

## 使用方法

### 基本使用

```bash
# 评估默认文件 (20250915-random.json)
python eval.py

# 评估指定文件
python eval.py -f your_prediction_file.json

# 或者指定完整文件名
python eval.py -f your_prediction_file
```

### 编程接口

```python
from eval import FutureXEvaluator
import json

# 初始化评估器（带LLM支持）
evaluator = FutureXEvaluator(
    api_key="your_api_key",
    base_url="your_api_base_url"
)

# 或者不使用LLM（仅基础评估）
evaluator = FutureXEvaluator()

# 加载数据
with open('prediction/your_file.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 执行评估
results = evaluator.evaluate_all(data)

# 查看结果
print(f"总分: {results['overall_score']:.4f}")
for level in [1, 2, 3, 4]:
    score = results['level_scores'][level]
    count = results['level_counts'][level]
    print(f"Level {level}: {score:.4f} (共 {count} 题)")
```

## 评分权重

系统采用加权平均计算总分：

- Level 1: 权重 0.1 (10%)
- Level 2: 权重 0.2 (20%)  
- Level 3: 权重 0.3 (30%)
- Level 4: 权重 0.4 (40%)

## 答案格式要求

所有预测答案必须使用 `\\boxed{}` 格式包装：

- 单选题: `\\boxed{Yes}` 或 `\\boxed{No}`
- 多选题: `\\boxed{A, B, C}`
- 排序题: `\\boxed{项目1, 项目2, 项目3}`
- 数值题: `\\boxed{123.45}`

## 相关链接

- **追踪记录**: https://app.phoenix.arize.com/s/Ianlan
- **飞书文档**: https://splffs0qwk.feishu.cn/wiki/H45QwCecpizvy6kINXhci7T2nlh

## 注意事项

1. 确保预测文件位于 `prediction/` 目录下
2. 预测文件必须是有效的JSON格式
3. 每个预测项必须包含 `answer` 字段
4. 答案格式必须严格遵循 `\\boxed{}` 规范
5. 数值题建议配置LLM API以获得更准确的单位标准化
