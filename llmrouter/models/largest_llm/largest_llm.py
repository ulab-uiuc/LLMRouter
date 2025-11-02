from llmrouter.models.meta_router import MetaRouter


def parse_size(size_str: str) -> float:
    """
    Convert a model size string (e.g., '7B', '13B', '512M') into a numeric value (in billions).
    Only supports numeric parsing with unit suffixes K, M, B, T.
    """
    size_str = str(size_str).strip().upper()
    try:
        if size_str.endswith("K"):
            return float(size_str[:-1]) / 1e6
        elif size_str.endswith("M"):
            return float(size_str[:-1]) / 1e3
        elif size_str.endswith("B"):
            return float(size_str[:-1])
        elif size_str.endswith("T"):
            return float(size_str[:-1]) * 1e3
        else:
            return float(size_str)
    except Exception:
        return 0.0


class LargestLLM(MetaRouter):
    """
    Always select the largest LLM (based on the 'size' field) that ends with 'B'.
    """

    def __init__(self, yaml_path: str):
        super().__init__(yaml_path)
        print("✅ LargestLLM initialized successfully")

    def inference(self):
        if not hasattr(self, "llm_data") or not self.llm_data:
            raise ValueError("LLM data not loaded or missing in YAML configuration.")

        # Filter only models whose size ends with 'B'
        filtered_names = [
            name for name, info in self.llm_data.items()
            if isinstance(info.get("size", ""), str) and info["size"].upper().endswith("B")
        ]

        if not filtered_names:
            raise ValueError("No models with size ending in 'B' found in llm_data.")

        # Find the largest model among those
        largest_model_name = max(
            filtered_names,
            key=lambda k: parse_size(self.llm_data[k].get("size", "0B"))
        )

        largest_model = self.llm_data[largest_model_name]
        print(f"🚀 Largest model (ending with 'B') selected: {largest_model_name} ({largest_model.get('size')})")

        return {
            "model_name": largest_model_name,
            "model_size": largest_model.get("size"),
            "model_info": largest_model
        }
