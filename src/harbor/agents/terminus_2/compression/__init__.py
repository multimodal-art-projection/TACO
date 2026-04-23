"""Self-evolving terminal output compression system.

Components:
- models: Data models (CompressionRule, CompressionPlan, FeedbackSignal)
- planner: Generates compression plans at task start
- dynamic_filter: Wraps rules as BaseFilter for SafeOutputFilter
- feedback: Zero-LLM feedback detection
- evolver: Rule evolution via LLM (spawn replacement/new)
- rule_cache: Cross-task persistence
- seed_rules: Initial rules for cold start
"""

from .models import CompressionPlan, CompressionRule, FeedbackSignal
from .planner import CompressionPlanner
from .dynamic_filter import DynamicCompressionFilter
from .feedback import FeedbackCollector
from .evolver import RuleEvolver
from .rule_cache import RuleCache
from .seed_rules import get_seed_rules
from .evo_logger import SelfEvoLogger

__all__ = [
    "CompressionRule",
    "CompressionPlan",
    "FeedbackSignal",
    "CompressionPlanner",
    "DynamicCompressionFilter",
    "FeedbackCollector",
    "RuleEvolver",
    "RuleCache",
    "get_seed_rules",
    "SelfEvoLogger",
]
