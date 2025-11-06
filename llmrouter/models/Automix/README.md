# Automix Router

## 概述

Automix是一个基于自验证(self-verification)的LLM路由策略，能够智能地决定何时将查询从小模型路由到大模型，从而在保证性能的同时降低推理成本。

本实现将原始的Automix代码重构为符合LLMRouter框架的格式，提供了标准化的接口和训练流程。

## 架构说明

### 核心组件

1. **methods.py** - 路由方法
   - `Threshold`: 单阈值路由
   - `DoubleThreshold`: 双阈值路由
   - `TripleThreshold`: 三阈值路由
   - `SelfConsistency`: 自一致性路由
   - `POMDPSimple`: 简单POMDP路由
   - `GreedyPOMDP`: 贪心POMDP路由
   - `POMDP`: 组合POMDP方法(推荐)

2. **data_pipeline.py** - 数据准备流程
   - `solve_queries()`: Step1 - 获取小模型和大模型的预测
   - `self_verify()`: Step2 - 对小模型预测进行自验证
   - `prepare_automix_data()`: 完整的数据准备流程

3. **model.py** - AutomixModel类
   - 封装Automix路由逻辑的PyTorch模块
   - 提供训练、推理和评估接口

4. **router.py** - AutomixRouter类
   - 继承`MetaRouter`的路由器实现
   - 实现`route()`方法

5. **trainer.py** - AutomixRouterTrainer类
   - 继承`BaseTrainer`的训练器实现
   - 实现参数搜索和评估逻辑

## 使用方法

### 1. 数据准备

首先，需要准备包含查询的数据文件(JSONL格式):

```python
from llmrouter.models.Automix import prepare_automix_data

# 准备数据(包括获取模型预测和自验证)
df = prepare_automix_data(
    input_data_path="./data/queries.jsonl",
    output_dir="./data",
    engine_small="meta/llama-3.1-8b-instruct",
    engine_large="meta/llama-3.1-70b-instruct",
)
```

输入数据格式示例:
```json
{"query": "What is the capital of France?", "gt": "Paris", "split": "train"}
{"query": "Who wrote Romeo and Juliet?", "gt": "Shakespeare", "split": "test"}
```

### 2. 创建路由器

```python
from llmrouter.models.Automix import (
    AutomixRouter,
    AutomixModel,
    POMDP,
    Threshold,
    SelfConsistency
)

# 选择路由方法
# 方法1: POMDP(推荐,组合多种策略)
method = POMDP(num_bins=8)

# 方法2: 简单阈值
# method = Threshold(num_bins=8)

# 方法3: 自一致性
# method = SelfConsistency(num_bins=8)

# 创建模型
model = AutomixModel(
    method=method,
    slm_column="llama13b_f1",
    llm_column="llama70b_f1",
    verifier_column="p_ver_13b",
    costs=[1, 50],  # [小模型成本, 大模型成本]
    verifier_cost=1,
    verbose=True
)

# 创建路由器
router = AutomixRouter(model=model)
```

### 3. 训练和评估

```python
from llmrouter.models.Automix import AutomixRouterTrainer

# 创建训练器
trainer = AutomixRouterTrainer(
    router=router,
    device="cpu"  # Automix主要使用pandas操作,不需要GPU
)

# 分割训练集和测试集
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

# 训练和评估
results = trainer.train_and_evaluate(train_df, test_df)

# 查看结果
print(f"训练集 IBC Lift: {results['train']['metrics']['ibc_lift']:.4f}")
print(f"测试集 IBC Lift: {results['test']['ibc_lift']:.4f}")
print(f"测试集性能: {results['test']['avg_performance']:.4f}")
print(f"测试集成本: {results['test']['avg_cost']:.2f}")
```

### 4. 使用训练好的路由器进行推理

```python
# 对新数据进行推理
new_data = pd.DataFrame([
    {
        "question": "Who painted the Mona Lisa?",
        "llama13b_pred_ans": "Leonardo",
        "llama70b_pred_ans": "Leonardo da Vinci",
        "p_ver_13b": 0.8,
        "llama13b_f1": 0.85,
        "llama70b_f1": 0.95
    }
])

# 获取路由决策
batch = {"data": new_data, "mode": "infer"}
outputs = router.route(batch)

# outputs["decisions"] 是布尔张量
# True = 路由到大模型, False = 使用小模型
print(f"路由决策: {outputs['decisions']}")
print(f"预估性能: {outputs['performance']:.4f}")
print(f"预估成本: {outputs['cost']:.2f}")
```

## 数据格式说明

### 输入数据(Step1)

```json
{
  "query": "问题文本",
  "gt": "标准答案或答案列表",
  "split": "train或test",
  "dataset": "数据集名称(可选)"
}
```

### Step1输出(模型预测)

包含以下列:
- `question`: 问题文本
- `output`: 标准答案
- `llama13b_pred_ans`: 13B模型预测
- `llama70b_pred_ans`: 70B模型预测
- `split`: train/test分割
- `dataset`: 数据集名称

### Step2输出(自验证)

在Step1基础上添加:
- `llama13b_ver`: 验证响应列表
- `p_ver_13b`: 验证正确的比例(0-1)
- `llama13b_f1`: 13B模型F1分数
- `llama70b_f1`: 70B模型F1分数
- `category`: 样本类别(NEEDY/GOOD/HOPELESS/UNDEFINED)

### 类别说明

- **NEEDY**: 小模型表现不如大模型,适合路由到大模型
- **GOOD**: 两个模型表现相同且良好,使用小模型即可
- **HOPELESS**: 两个模型都表现不佳
- **UNDEFINED**: 未分类

## 路由方法说明

### Threshold(阈值法)

当验证分数 ≤ 阈值时,路由到大模型。

```python
method = Threshold(num_bins=8)
```

### POMDP(部分可观察马尔可夫决策过程)

综合多种策略的组合方法,包括:
- POMDPSimple: 基于观测概率的POMDP
- GreedyPOMDP: 贪心POMDP
- DoubleThreshold: 双阈值
- Threshold: 单阈值

```python
method = POMDP(num_bins=8, init_belief=False)
```

### SelfConsistency(自一致性)

使用固定阈值0.5进行路由。

```python
method = SelfConsistency(num_bins=8)
```

## 评估指标

### IBC Lift(Incremental Benefit over Cost Lift)

衡量路由策略相对于基准(仅使用小模型)的性能提升与成本增加的比率。

```
IBC Lift = (automix_slope - slm_llm_slope) / slm_llm_slope
```

其中:
- `automix_slope`: Automix路由的性能-成本斜率
- `slm_llm_slope`: 小模型到大模型的性能-成本斜率

### 其他指标

- `avg_performance`: 平均任务性能(F1分数)
- `avg_cost`: 平均推理成本
- `routing_percentage`: 路由到大模型的查询百分比

## 完整示例

参见 `example_usage.py`:

```bash
cd /HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/lx/llm_code/router/LLMRouter/llmrouter/models/Automix
python example_usage.py
```

## 与原始代码的对应关系

| 原始文件 | 新文件 | 说明 |
|---------|--------|------|
| `Step1_SolveQueries.py` | `data_pipeline.py::solve_queries()` | 获取模型预测 |
| `Step2_SelfVerify.py` | `data_pipeline.py::self_verify()` | 自验证和分类 |
| `Step3_MetaVerify.py` | `trainer.py::train_and_evaluate()` | 训练和评估 |
| `automix_methods.py` | `methods.py` | 路由方法 |
| `automix.py::Automix类` | `model.py::AutomixModel类` | 核心模型逻辑 |
| - | `router.py::AutomixRouter类` | LLMRouter框架集成 |

## 依赖项

- torch
- pandas
- numpy
- scipy
- transformers
- openai
- huggingface_hub
- tqdm

## 注意事项

1. **API配置**: 在使用数据准备流程前,确保配置了OpenAI/NVIDIA API密钥。

2. **成本控制**: 可以通过`cost_constraint`参数限制路由成本范围。

3. **模型选择**: 当前实现默认使用Llama 3.1的8B和70B版本,可以根据需要修改。

4. **验证次数**: `self_verify()`中的`n`参数控制验证采样次数,增加可提高可靠性但会增加成本。

5. **训练时间**: Automix训练是参数搜索过程,时间取决于`num_bins`和方法复杂度。

## 引用

如果使用Automix,请引用原始论文:

```bibtex
@article{automix2023,
  title={Automix: Automatically Mixing Language Models},
  author={...},
  journal={...},
  year={2023}
}
```

## 许可证

本实现遵循LLMRouter项目的许可证。
