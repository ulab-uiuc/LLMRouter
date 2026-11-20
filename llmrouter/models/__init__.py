from .meta_router import MetaRouter
from .base_trainer import BaseTrainer

from .smallest_llm import SmallestLLM
from .largest_llm import LargestLLM

from .knnrouter import KNNRouter
from .knnrouter import KNNRouterTrainer

from .svmrouter import SVMRouter
from .svmrouter import SVMRouterTrainer

from .mlprouter import MLPRouter
from .mlprouter import MLPTrainer

from .racerrouter import RACERRouter, RACERTrainer

from .mfrouter import MFRouter
from .mfrouter import MFRouterTrainer

from .elorouter import EloRouter
from .elorouter import EloRouterTrainer

from .automix import AutomixRouter
from .automix import AutomixRouterTrainer

from .routerdc import DCRouter
from .routerdc import DCTrainer

from .hybrid_llm import HybridLLMRouter
from .hybrid_llm import HybridLLMTrainer

try:
    from .graphrouter import GraphRouter
    from .graphrouter import GraphTrainer
except Exception:
    GraphRouter = None
    GraphTrainer = None

try:
    from .causallm_router import CausalLMRouter
    from .causallm_router import CausalLMTrainer
except Exception:
    CausalLMRouter = None
    CausalLMTrainer = None

try:
    from .router_r1 import RouterR1
except Exception:
    RouterR1 = None

try:
    from .gmtrouter import GMTRouter
    from .gmtrouter import GMTRouterTrainer
except Exception:
    GMTRouter = None
    GMTRouterTrainer = None

try:
    from .personalizedrouter import PersonalizedRouter
    from .personalizedrouter import PersonalizedRouterTrainer
except Exception:
    PersonalizedRouter = None
    PersonalizedRouterTrainer = None

try:
    from .tsrouter import TSRouter
    from .tsrouter import TSRouterTrainer
except Exception:
    TSRouter = None
    TSRouterTrainer = None

from .gnosis_latent_router import GnosisLatentRouter
from .gnosis_latent_router import GnosisLatentRouterTrainer

__all__ = [
    "MetaRouter",
    "BaseTrainer",
    "SmallestLLM",
    "LargestLLM",

    "KNNRouter",
    "KNNRouterTrainer",

    "SVMRouter",
    "SVMRouterTrainer",

    "MLPRouter",
    "MLPTrainer",

    "RACERRouter",
    "RACERTrainer",

    "MFRouter",
    "MFRouterTrainer",

    "EloRouter",
    "EloRouterTrainer",

    "DCRouter",
    "DCTrainer",

    "AutomixRouter",
    "AutomixRouterTrainer",

    "HybridLLMRouter",
    "HybridLLMTrainer",

    "GraphRouter",
    "GraphTrainer",

    "CausalLMRouter",
    "CausalLMTrainer",

    "RouterR1",

    "GMTRouter",
    "GMTRouterTrainer",

    "PersonalizedRouter",
    "PersonalizedRouterTrainer",

    "TSRouter",
    "TSRouterTrainer",

    "GnosisLatentRouter",
    "GnosisLatentRouterTrainer",
]
