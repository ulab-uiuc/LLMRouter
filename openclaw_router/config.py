"""
OpenClaw Router Configuration
==============================
"""

import os
import re
import yaml
import itertools
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off", ""):
            return False
    return default


@dataclass
class LLMConfig:
    """Single LLM configuration"""
    name: str
    provider: str
    model_id: str
    base_url: str
    provider_type: str = "openai_compatible"
    auth_mode: str = "auto"  # auto, bearer, none
    chat_path: str = "/chat/completions"
    local: Optional[bool] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    description: str = ""
    input_price: float = 0.0
    output_price: float = 0.0
    max_tokens: int = 4096
    context_limit: int = 32768


@dataclass
class RouterConfig:
    """Router strategy configuration"""
    strategy: str = "random"  # llm, rules, random, round_robin, llmrouter

    # For LLM strategy
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    provider_type: str = "openai_compatible"
    auth_mode: str = "auto"  # auto, bearer, none
    chat_path: str = "/chat/completions"
    local: Optional[bool] = None

    # For rules strategy
    rules: List[Dict] = field(default_factory=list)

    # For random strategy
    weights: Dict[str, int] = field(default_factory=dict)

    # For llmrouter strategy (ML-based routers)
    llmrouter_name: Optional[str] = None  # knnrouter, mlprouter, thresholdrouter, etc.
    llmrouter_config: Optional[str] = None  # Path to router config
    llmrouter_model_path: Optional[str] = None  # Path to trained model

    # Ordered list of model names to retry, in order, if the model selected by
    # the routing strategy fails to respond (e.g. the provider is down or
    # rate-limited). Models not present in `llms` are ignored. Empty by
    # default, which preserves the previous behavior of failing the request.
    fallback_models: List[str] = field(default_factory=list)


@dataclass
class MemoryConfig:
    """
    Optional routing memory.

    When enabled, the server persists (query -> selected model) pairs to disk and
    retrieves top-k similar past queries for future routing decisions.

    Current implementation only uses memory to augment `router.strategy: llm`.
    """

    enabled: bool = False
    path: str = ""  # JSONL path; relative paths are resolved against the config file directory.
    top_k: int = 10

    # Dense retriever (Contriever by default)
    retriever_model: str = "facebook/contriever-msmarco"
    device: str = "cpu"  # "cpu" or "cuda"
    max_length: int = 256  # retriever tokenizer max_length

    # Guardrails for stored/prompt text size
    max_query_chars: int = 500
    max_prompt_chars: int = 200

    # If true and request provides a user id, retrieve from the same user only.
    per_user: bool = False


@dataclass
class MediaConfig:
    """
    Media understanding configuration.

    When enabled, converts images/audio/video to text descriptions using Together AI APIs.
    The text descriptions are stored in memory alongside the original text query.
    """

    enabled: bool = False

    # Together AI settings
    api_key: Optional[str] = None
    api_key_env: str = "TOGETHER_API_KEY"
    base_url: str = "https://api.together.xyz/v1"

    # Vision model for images/video frames
    vision_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    # Audio transcription model
    audio_model: str = "openai/whisper-large-v3"

    # Prompts
    image_prompt: str = "Describe this image concisely in 2-3 sentences."
    video_prompt: str = "Describe what you see in these video frames."

    # Video settings
    video_max_frames: int = 4  # Max frames to extract from video

    # Max content length in memory
    max_description_chars: int = 500


@dataclass
class OpenClawConfig:
    """Main OpenClaw Router configuration"""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    show_model_prefix: bool = True

    # Router settings
    router: RouterConfig = field(default_factory=RouterConfig)

    # LLM backends
    llms: Dict[str, LLMConfig] = field(default_factory=dict)

    # API Keys
    api_keys: Dict[str, Any] = field(default_factory=dict)

    # Routing memory (optional)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Media understanding (optional)
    media: MediaConfig = field(default_factory=MediaConfig)

    # Origin metadata (useful for resolving relative paths)
    config_path: Optional[str] = None
    config_dir: Optional[str] = None

    # Key cycling state
    _nvidia_key_cycle: Any = field(default=None, repr=False)
    _nvidia_key_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OpenClawConfig":
        """Load configuration from YAML file"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Expand environment variables
        data = cls._expand_env_vars(data)

        config = cls()
        config.config_path = yaml_path
        config.config_dir = os.path.dirname(os.path.abspath(yaml_path))

        # Server settings
        serve_config = data.get("serve", {})
        config.host = serve_config.get("host", config.host)
        config.port = serve_config.get("port", config.port)
        config.show_model_prefix = serve_config.get("show_model_prefix", config.show_model_prefix)

        # API Keys
        config.api_keys = data.get("api_keys", {})

        # Initialize NVIDIA key cycling
        nvidia_keys = config.api_keys.get("nvidia", [])
        if isinstance(nvidia_keys, str):
            nvidia_keys = [nvidia_keys]
        nvidia_keys = [k for k in nvidia_keys if k and not k.startswith("$")]
        if nvidia_keys:
            config._nvidia_key_cycle = itertools.cycle(nvidia_keys)
            print(f"[Config] Loaded {len(nvidia_keys)} NVIDIA API key(s)")

        # Router settings
        router_data = data.get("router", {})
        config.router = RouterConfig(
            strategy=router_data.get("strategy", "random"),
            provider=router_data.get("provider"),
            model=router_data.get("model"),
            base_url=router_data.get("base_url"),
            provider_type=router_data.get("provider_type", "openai_compatible"),
            auth_mode=router_data.get("auth_mode", "auto"),
            chat_path=router_data.get("chat_path", "/chat/completions"),
            local=router_data.get("local"),
            rules=router_data.get("rules", []),
            weights=router_data.get("weights", {}),
            llmrouter_name=router_data.get("llmrouter", {}).get("name") or router_data.get("name"),
            llmrouter_config=router_data.get("llmrouter", {}).get("config_path") or router_data.get("config_path"),
            llmrouter_model_path=router_data.get("llmrouter", {}).get("model_path") or router_data.get("model_path"),
            fallback_models=list(router_data.get("fallback_models", []) or []),
        )

        # Memory settings
        memory_data = data.get("memory", {}) or {}
        default_memory = MemoryConfig()
        config.memory = MemoryConfig(
            enabled=_parse_bool(memory_data.get("enabled", default_memory.enabled), default_memory.enabled),
            path=str(memory_data.get("path", default_memory.path) or ""),
            top_k=int(memory_data.get("top_k", default_memory.top_k) or default_memory.top_k),
            retriever_model=str(memory_data.get("retriever_model", default_memory.retriever_model) or default_memory.retriever_model),
            device=str(memory_data.get("device", default_memory.device) or default_memory.device),
            max_length=int(memory_data.get("max_length", default_memory.max_length) or default_memory.max_length),
            max_query_chars=int(memory_data.get("max_query_chars", default_memory.max_query_chars) or default_memory.max_query_chars),
            max_prompt_chars=int(memory_data.get("max_prompt_chars", default_memory.max_prompt_chars) or default_memory.max_prompt_chars),
            per_user=_parse_bool(memory_data.get("per_user", default_memory.per_user), default_memory.per_user),
        )

        # Media understanding settings
        media_data = data.get("media", {}) or {}
        default_media = MediaConfig()
        config.media = MediaConfig(
            enabled=_parse_bool(media_data.get("enabled", default_media.enabled), default_media.enabled),
            api_key=media_data.get("api_key"),
            api_key_env=str(media_data.get("api_key_env", default_media.api_key_env) or default_media.api_key_env),
            base_url=str(media_data.get("base_url", default_media.base_url) or default_media.base_url),
            vision_model=str(media_data.get("vision_model", default_media.vision_model) or default_media.vision_model),
            audio_model=str(media_data.get("audio_model", default_media.audio_model) or default_media.audio_model),
            image_prompt=str(media_data.get("image_prompt", default_media.image_prompt) or default_media.image_prompt),
            video_prompt=str(media_data.get("video_prompt", default_media.video_prompt) or default_media.video_prompt),
            video_max_frames=int(media_data.get("video_max_frames", default_media.video_max_frames) or default_media.video_max_frames),
            max_description_chars=int(media_data.get("max_description_chars", default_media.max_description_chars) or default_media.max_description_chars),
        )

        # LLM configurations
        llms_data = data.get("llms", data.get("models", {}))
        for name, llm_config in llms_data.items():
            config.llms[name] = LLMConfig(
                name=name,
                provider=llm_config.get("provider", "openai"),
                model_id=llm_config.get("model", name),
                base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
                provider_type=llm_config.get("provider_type", "openai_compatible"),
                auth_mode=llm_config.get("auth_mode", "auto"),
                chat_path=llm_config.get("chat_path", "/chat/completions"),
                local=llm_config.get("local"),
                api_key=llm_config.get("api_key"),
                api_key_env=llm_config.get("api_key_env"),
                description=llm_config.get("description", ""),
                input_price=llm_config.get("input_price", 0.0),
                output_price=llm_config.get("output_price", 0.0),
                max_tokens=llm_config.get("max_tokens", 4096),
                context_limit=llm_config.get("context_limit", 32768),
            )

        return config

    @staticmethod
    def _expand_env_vars(value: Any) -> Any:
        """Recursively expand environment variables ${VAR_NAME}"""
        if isinstance(value, str):
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, value)
            for match in matches:
                env_value = os.getenv(match, "")
                value = value.replace(f"${{{match}}}", env_value)
            return value
        elif isinstance(value, dict):
            return {k: OpenClawConfig._expand_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [OpenClawConfig._expand_env_vars(item) for item in value]
        return value

    def get_api_key(self, provider: str, model_config: Optional[LLMConfig] = None) -> Optional[str]:
        """Get API key for provider"""
        # 1. Check model-specific api_key
        if model_config and model_config.api_key:
            return model_config.api_key

        # 2. Check model-specific api_key_env
        if model_config and model_config.api_key_env:
            return os.getenv(model_config.api_key_env)

        # 3. Get from global api_keys
        if provider == "nvidia":
            if self._nvidia_key_cycle:
                with self._nvidia_key_lock:
                    return next(self._nvidia_key_cycle)
            return os.getenv("NVIDIA_API_KEY")

        configured = self.api_keys.get(provider)
        if isinstance(configured, list):
            normalized = []
            for item in configured:
                text = str(item).strip()
                if text and not text.startswith("$"):
                    normalized.append(text)
            if normalized:
                # Keep behavior simple for non-NVIDIA providers:
                # use the first configured key.
                return normalized[0]
            # Allow explicit empty key (local OpenAI-compatible backends).
            if any(str(item).strip() == "" for item in configured):
                return ""

        key = configured if isinstance(configured, str) else None
        if not key:
            key = os.getenv(f"{provider.upper()}_API_KEY")
        if key and not key.startswith("$"):
            return key
        return None


# Models that don't support system role
MODELS_WITHOUT_SYSTEM_ROLE = {
    "google/gemma-2-9b-it",
    "gemma-2-9b-it",
    "meta/llama-3.1-8b-instruct",
    "meta/llama3-70b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "nvidia/llama3-chatqa-1.5-8b",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
}

# Model context limits
MODEL_CONTEXT_LIMITS = {
    "google/gemma-2-9b-it": 8192,
    "meta/llama-3.1-8b-instruct": 128000,
    "qwen/qwen2.5-7b-instruct": 32768,
    "mistralai/mistral-7b-instruct-v0.3": 32768,
    "nvidia/llama3-chatqa-1.5-8b": 8192,
    "mistralai/mixtral-8x22b-instruct-v0.1": 65536,
    "meta/llama3-70b-instruct": 8192,
    "mistralai/mixtral-8x7b-instruct-v0.1": 32768,
    "nvidia/llama-3.3-nemotron-super-49b-v1": 32768,
}
