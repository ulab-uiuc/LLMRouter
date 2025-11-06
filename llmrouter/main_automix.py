"""
Automix Router - 完整使用示例
==============================

这个脚本展示了如何使用Automix路由器进行完整的训练和推理流程。

运行方式:
    python example_usage.py [--config CONFIG_PATH]

参数:
    --config: YAML配置文件路径 (默认: config/automix_config.yaml)
"""

import os
import sys
import argparse
import pandas as pd
import yaml

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print("runing in: ", project_root)


from llmrouter.models.Automix import (
    AutomixRouter,
    AutomixRouterTrainer,
    AutomixModel,
    POMDP,
    Threshold,
    SelfConsistency,
    prepare_automix_data,
)


def load_config(config_path: str = None) -> dict:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径。如果为None,使用默认路径

    Returns:
        配置字典
    """
    if config_path is None:
        # 使用默认配置路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config", "automix_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"已加载配置文件: {config_path}")
    return config


def get_routing_method(method_name: str, num_bins: int):
    """
    根据方法名称创建路由方法实例

    Args:
        method_name: 方法名称 ("Threshold", "SelfConsistency", "POMDP")
        num_bins: bin数量

    Returns:
        路由方法实例
    """
    method_map = {
        "Threshold": Threshold,
        "SelfConsistency": SelfConsistency,
        "POMDP": POMDP,
    }

    if method_name not in method_map:
        raise ValueError(
            f"未知的路由方法: {method_name}. "
            f"可用方法: {list(method_map.keys())}"
        )

    return method_map[method_name](num_bins=num_bins)


def train_and_evaluate(config: dict):
    """
    使用真实数据进行训练和评估
    需要先准备数据文件

    Args:
        config: 从YAML文件加载的配置字典
    """
    cfg = config["real_data"]
    display_cfg = config["display"]
    sep_width = display_cfg["separator_width"]

    print("=" * sep_width)
    print("示例1: 使用真实数据进行训练和评估")
    print("=" * sep_width)

    # 从配置获取路径
    data_path = cfg["data_path"]
    output_dir = cfg["output_dir"]

    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        print("请先准备数据文件或运行数据准备流程")
        return

    print(f"\n步骤1: 准备数据(获取模型预测和自验证)")
    print("-" * sep_width)

    # 如果已经准备好数据,可以跳过这一步
    skip_data_prep = os.path.exists(
        os.path.join(output_dir, "router_automix_llamapair_ver_outputs.jsonl")
    )

    if skip_data_prep:
        print("检测到已准备好的数据,跳过数据准备步骤")
        df = pd.read_json(
            os.path.join(output_dir, "router_automix_llamapair_ver_outputs.jsonl"),
            lines=True,
            orient="records",
        )
    else:
        df = prepare_automix_data(
            input_data_path=data_path,
            output_dir=output_dir,
            engine_small=cfg["engine_small"],
            engine_large=cfg["engine_large"],
        )

    print(f"\n准备完成! 数据集大小: {len(df)}")
    print(f"训练集大小: {len(df[df['split'] == 'train'])}")
    print(f"测试集大小: {len(df[df['split'] == 'test'])}")

    print(f"\n步骤2: 创建Automix路由器")
    print("-" * sep_width)

    # 从配置创建路由方法
    method = get_routing_method(cfg["routing_method"], cfg["num_bins"])
    print(f"路由方法: {cfg['routing_method']} (num_bins={cfg['num_bins']})")

    # 创建模型
    model = AutomixModel(
        method=method,
        slm_column=cfg["columns"]["slm"],
        llm_column=cfg["columns"]["llm"],
        verifier_column=cfg["columns"]["verifier"],
        costs=[cfg["costs"]["small_model"], cfg["costs"]["large_model"]],
        verifier_cost=cfg["costs"]["verifier"],
        verbose=cfg["training"]["verbose"],
    )
    print(
        f"模型配置: 小模型成本={cfg['costs']['small_model']}, "
        f"大模型成本={cfg['costs']['large_model']}, "
        f"验证成本={cfg['costs']['verifier']}"
    )

    # 创建路由器
    router = AutomixRouter(model=model)
    print("路由器创建成功")

    print(f"\n步骤3: 训练路由器")
    print("-" * sep_width)

    # 创建训练器
    trainer = AutomixRouterTrainer(router=router, device=cfg["training"]["device"])

    # 分割数据
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    # 训练和评估
    results = trainer.train_and_evaluate(train_df, test_df)

    print(f"\n步骤4: 查看结果")
    print("-" * sep_width)

    # 获取显示精度配置
    prec = display_cfg["precision"]

    print("\n训练集结果:")
    print(f"  最佳参数: {results['train']['best_param']}")
    print(f"  IBC Lift: {results['train']['metrics']['ibc_lift']:.{prec['ibc_lift']}f}")
    print(f"  平均性能: {results['train']['metrics']['avg_performance']:.{prec['performance']}f}")
    print(f"  平均成本: {results['train']['metrics']['avg_cost']:.{prec['cost']}f}")

    print("\n测试集结果:")
    print(f"  IBC Lift: {results['test']['ibc_lift']:.{prec['ibc_lift']}f}")
    print(f"  平均性能: {results['test']['avg_performance']:.{prec['performance']}f}")
    print(f"  平均成本: {results['test']['avg_cost']:.{prec['cost']}f}")

    # 计算路由统计
    test_decisions = results["test"]["route_to_llm"]
    num_routed = int(test_decisions.sum())
    total = len(test_decisions)
    print(f"  路由到大模型: {num_routed}/{total} ({num_routed/total*100:.{prec['percentage']}f}%)")

    print(f"\n步骤5: 使用训练好的路由器进行推理")
    print("-" * sep_width)

    # 选择几个测试样本进行推理
    num_samples = cfg["inference"]["num_samples"]
    sample_data = test_df.head(num_samples)

    for idx, row in sample_data.iterrows():
        decision = router.model.infer(row)
        model_used = "大模型(70B)" if decision else "小模型(13B)"
        print(f"\n问题: {row['question'][:60]}...")
        print(f"  验证分数: {row[cfg['columns']['verifier']]:.3f}")
        print(f"  小模型F1: {row[cfg['columns']['slm']]:.3f}")
        print(f"  大模型F1: {row[cfg['columns']['llm']]:.3f}")
        print(f"  路由决策: {model_used}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Automix Router 使用示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML配置文件路径 (默认: configs/model_config_test/automix_config.yaml)",
    )
    args = parser.parse_args()

    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保配置文件存在或使用 --config 参数指定配置文件路径")
        return

    sep_width = config["display"]["separator_width"]

    print("\n" + "=" * sep_width)
    print("Automix Router 使用示例")
    print("=" * sep_width)

    try:
        train_and_evaluate(config)
    except Exception as e:
        print(f"\n真实数据示例失败: {e}")
        print("提示: 请确保数据文件存在并且配置正确")

    print("\n" + "=" * sep_width)
    print("示例完成!") 
    print("=" * sep_width)


if __name__ == "__main__":
    main()
