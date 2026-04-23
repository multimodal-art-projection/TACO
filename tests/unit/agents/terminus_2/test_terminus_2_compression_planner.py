import importlib.util
import json
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPRESSION_DIR = REPO_ROOT / "src/harbor/agents/terminus_2/compression"


def _ensure_package(name: str) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module


def _load_module(fullname: str, path: Path):
    spec = importlib.util.spec_from_file_location(fullname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_planner_types():
    for package_name in (
        "harbor",
        "harbor.agents",
        "harbor.agents.terminus_2",
        "harbor.agents.terminus_2.compression",
    ):
        _ensure_package(package_name)

    models_module = _load_module(
        "harbor.agents.terminus_2.compression.models",
        COMPRESSION_DIR / "models.py",
    )
    _load_module(
        "harbor.agents.terminus_2.compression.evo_logger",
        COMPRESSION_DIR / "evo_logger.py",
    )
    planner_module = _load_module(
        "harbor.agents.terminus_2.compression.planner",
        COMPRESSION_DIR / "planner.py",
    )
    return planner_module.CompressionPlanner, models_module.CompressionRule


class TestCompressionPlannerRuleIdNormalization:
    def test_modified_rule_reusing_cached_id_is_renamed(self):
        planner_cls, rule_cls = _load_planner_types()
        planner = planner_cls(llm_client=None)
        cached_rule = rule_cls(rule_id="seed_pip_install", trigger_regex="pip install")

        response = json.dumps(
            {
                "selected_rule_ids": ["seed_pip_install"],
                "modified_rules": [
                    {
                        "rule_id": "seed_pip_install",
                        "trigger_regex": "pip install",
                        "description": "tuned pip compression",
                    }
                ],
                "new_rules": [],
            }
        )

        plan = planner._parse_plan_with_cache(response, "general", [cached_rule])

        assert plan.selected_rule_ids == []
        assert [rule.rule_id for rule in plan.modified_rules] == [
            "seed_pip_install_mod"
        ]

    def test_no_cache_duplicate_generated_ids_are_uniquified(self):
        planner_cls, _ = _load_planner_types()
        planner = planner_cls(llm_client=None)

        response = json.dumps(
            {
                "rules": [
                    {"rule_id": "pip_rule", "trigger_regex": "pip install"},
                    {"rule_id": "pip_rule", "trigger_regex": "pip3 install"},
                ]
            }
        )

        plan = planner._parse_plan_no_cache(response, "general")

        assert [rule.rule_id for rule in plan.new_rules] == [
            "pip_rule",
            "pip_rule_new",
        ]
