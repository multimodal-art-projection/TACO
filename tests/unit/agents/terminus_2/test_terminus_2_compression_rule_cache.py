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


def _load_rule_cache_types():
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
        "harbor.agents.terminus_2.compression.seed_rules",
        COMPRESSION_DIR / "seed_rules.py",
    )
    rule_cache_module = _load_module(
        "harbor.agents.terminus_2.compression.rule_cache",
        COMPRESSION_DIR / "rule_cache.py",
    )
    return rule_cache_module.RuleCache, models_module.CompressionRule


def _make_rule(rule_cls, *, confidence: float, times_applied: int, times_complained: int):
    return rule_cls(
        rule_id="make_compilation_output",
        trigger_regex=r"\b(make|gcc|cc)\b",
        description="Compilation output compressed - errors/warnings preserved",
        keep_patterns=[r"\berror:", r"\bwarning:"],
        strip_patterns=[r"gcc .+", r"cc .+"],
        keep_first_n=5,
        keep_last_n=10,
        max_lines=None,
        summary_header="[Compilation output compressed - errors/warnings preserved]",
        priority=42,
        confidence=confidence,
        times_applied=times_applied,
        times_complained=times_complained,
    )


class TestRuleCacheComplaintDelta:
    def test_save_merges_only_new_complaints(self, tmp_path):
        rule_cache_cls, rule_cls = _load_rule_cache_types()
        cache_path = tmp_path / "compression_rules_cache.json"
        cache = rule_cache_cls(cache_path=cache_path)

        existing_rule = _make_rule(
            rule_cls,
            confidence=0.9,
            times_applied=29,
            times_complained=10,
        )
        cache_path.write_text(
            json.dumps({"general": [existing_rule.model_dump()]}),
            encoding="utf-8",
        )

        updated_rule = _make_rule(
            rule_cls,
            confidence=0.8,
            times_applied=29,
            times_complained=11,
        )
        cache.save("general", [updated_rule])

        saved_rule = json.loads(cache_path.read_text(encoding="utf-8"))["general"][0]
        assert saved_rule["times_applied"] == 29
        assert saved_rule["times_complained"] == 11

    def test_save_frozen_rule_degrades_confidence_and_adds_only_delta_complaints(
        self, tmp_path
    ):
        rule_cache_cls, rule_cls = _load_rule_cache_types()
        cache_path = tmp_path / "compression_rules_cache.json"
        cache = rule_cache_cls(cache_path=cache_path)

        existing_rule = _make_rule(
            rule_cls,
            confidence=0.9,
            times_applied=29,
            times_complained=10,
        )
        cache_path.write_text(
            json.dumps({"general": [existing_rule.model_dump()]}),
            encoding="utf-8",
        )

        frozen_rule = _make_rule(
            rule_cls,
            confidence=0.0,
            times_applied=29,
            times_complained=11,
        )
        cache.save("general", [frozen_rule])

        saved_rule = json.loads(cache_path.read_text(encoding="utf-8"))["general"][0]
        assert saved_rule["times_complained"] == 11
        assert saved_rule["confidence"] == 0.45
