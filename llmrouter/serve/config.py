"""
Serve Configuration
===================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import yaml


@dataclass
class LLMConfig:
    """Single LLM configuration"""
    name: str
    provider: str
    model_id: str
    base_url: str
    api_key: Optional[str] = None
    input_price: float = 0.0
    output_price: float = 0.0
    max_tokens: int = 4096
    context_limit: int = 32768


@dataclass
class ServeConfig:
    """Server configuration"""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Router settings
    router_name: str = "randomrouter"
    router_config_path: Optional[str] = None

    # LLM backend settings
    llms: Dict[str, LLMConfig] = field(default_factory=dict)

    # API Keys
    api_keys: Dict[str, List[str]] = field(default_factory=dict)

    # Show model name prefix
    show_model_prefix: bool = True

    fail_on_routing_error: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ServeConfig":
        """Load configuration from YAML file"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = cls()

        # Server settings
        serve_config = data.get("serve", {})
        config.host = serve_config.get("host", config.host)
        config.port = serve_config.get("port", config.port)
        config.show_model_prefix = serve_config.get("show_model_prefix", config.show_model_prefix)
        config.fail_on_routing_error = serve_config.get("fail_on_routing_error", config.fail_on_routing_error)

        # Router settings
        router_config = data.get("router", {})
        config.router_name = router_config.get("name", config.router_name)
        config.router_config_path = router_config.get("config_path")

        # API Keys
        config.api_keys = data.get("api_keys", {})

        # LLM configuration
        llms_data = data.get("llms", {})
        for name, llm_config in llms_data.items():
            config.llms[name] = LLMConfig(
                name=name,
                provider=llm_config.get("provider", "openai"),
                model_id=llm_config.get("model", name),
                base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
                api_key=llm_config.get("api_key"),
                input_price=llm_config.get("input_price", 0.0),
                output_price=llm_config.get("output_price", 0.0),
                max_tokens=llm_config.get("max_tokens", 4096),
                context_limit=llm_config.get("context_limit", 32768),
            )

        return config

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key"""
        keys = self.api_keys.get(provider, [])
        if isinstance(keys, str):
            # Support environment variables
            if keys.startswith("${") and keys.endswith("}"):
                return os.environ.get(keys[2:-1])
            return keys
        elif isinstance(keys, list) and keys:
            # Round-robin multiple keys
            if not hasattr(self, "_key_index"):
                self._key_index = {}
            idx = self._key_index.get(provider, 0)
            key = keys[idx % len(keys)]
            self._key_index[provider] = idx + 1
            return key
        return None
