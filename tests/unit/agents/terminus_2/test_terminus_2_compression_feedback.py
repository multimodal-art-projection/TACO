import importlib.util
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


def _load_feedback_types():
    for package_name in (
        "harbor",
        "harbor.agents",
        "harbor.agents.terminus_2",
        "harbor.agents.terminus_2.compression",
    ):
        _ensure_package(package_name)

    _load_module(
        "harbor.agents.terminus_2.compression.models",
        COMPRESSION_DIR / "models.py",
    )
    _load_module(
        "harbor.agents.terminus_2.compression.evo_logger",
        COMPRESSION_DIR / "evo_logger.py",
    )
    feedback_module = _load_module(
        "harbor.agents.terminus_2.compression.feedback",
        COMPRESSION_DIR / "feedback.py",
    )
    return (
        feedback_module.FeedbackCollector,
        feedback_module.COMPRESSION_FEEDBACK_NEED_FULL_OUTPUT,
    )


class TestCompressionFeedbackStructuredField:
    def test_detect_complaint_requires_structured_feedback(self):
        collector_cls, feedback_value = _load_feedback_types()
        collector = collector_cls()
        collector.record_compression(
            episode=1,
            command="pytest -q",
            raw_output="full output",
            compressed_output="[compressed]",
            rules_applied=["pytest_rule"],
        )

        signal = collector.detect_complaint(
            episode=2,
            agent_response="I am missing details from the compressed output.",
            compression_feedback=feedback_value,
        )

        assert signal is not None
        assert signal.signal_type == "complaint"
        assert signal.rules_applied == ["pytest_rule"]

    def test_missing_structured_feedback_does_not_trigger(self):
        collector_cls, _ = _load_feedback_types()
        collector = collector_cls()
        collector.record_compression(
            episode=1,
            command="pytest -q",
            raw_output="full output",
            compressed_output="[compressed]",
            rules_applied=["pytest_rule"],
        )

        signal = collector.detect_complaint(
            episode=2,
            agent_response="Show me the full output",
            compression_feedback="",
        )

        assert signal is None
