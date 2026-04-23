import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TERMINUS_DIR = REPO_ROOT / "src/harbor/agents/terminus_2"


def _load_module(fullname: str, path: Path):
    spec = importlib.util.spec_from_file_location(fullname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_json_parser():
    module = _load_module(
        "harbor.agents.terminus_2.terminus_json_plain_parser",
        TERMINUS_DIR / "terminus_json_plain_parser.py",
    )
    return module.TerminusJSONPlainParser


def _load_xml_parser():
    module = _load_module(
        "harbor.agents.terminus_2.terminus_xml_plain_parser",
        TERMINUS_DIR / "terminus_xml_plain_parser.py",
    )
    return module.TerminusXMLPlainParser


class TestTerminusStructuredCompressionFeedback:
    def test_json_parser_extracts_compression_feedback(self):
        parser = _load_json_parser()()
        response = """{
  "analysis": "Compressed pytest output is missing the failing test name.",
  "plan": "I need the uncompressed output before deciding the next command.",
  "commands": [],
  "compression_feedback": "need_full_output",
  "task_complete": false
}"""

        result = parser.parse_response(response)

        assert result.error == ""
        assert result.compression_feedback == "need_full_output"

    def test_json_parser_rejects_unknown_compression_feedback_value(self):
        parser = _load_json_parser()()
        response = """{
  "analysis": "Something is missing.",
  "plan": "Pause until I can inspect the omitted output.",
  "commands": [],
  "compression_feedback": "please_help"
}"""

        result = parser.parse_response(response)

        assert result.error == ""
        assert result.compression_feedback == ""
        assert "compression_feedback" in result.warning

    def test_xml_parser_extracts_compression_feedback(self):
        parser = _load_xml_parser()()
        response = """<response>
<analysis>The compressed build log omitted the actual linker error.</analysis>
<plan>I need the uncompressed output before proceeding.</plan>
<commands></commands>
<compression_feedback>need_full_output</compression_feedback>
</response>"""

        result = parser.parse_response(response)

        assert result.error == ""
        assert result.compression_feedback == "need_full_output"
