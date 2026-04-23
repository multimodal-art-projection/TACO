"""Feedback detection for the self-evolving compression system.

Monitors agent responses to detect signs that compression was too aggressive
or that a command produced large output with no rule coverage. Complaint
detection is intentionally explicit: the agent must emit a structured
compression feedback value in its parsed response.

Two signal types:
- "complaint": Agent emitted the explicit structured compression feedback value
- "uncovered_big_output": Large output had no dynamic rule match
"""

import logging
import re

from .evo_logger import SelfEvoLogger
from .models import FeedbackSignal

logger = logging.getLogger(__name__)


COMPRESSION_FEEDBACK_NEED_FULL_OUTPUT = "need_full_output"

ANALYSIS_COMPLAINT_PATTERNS = [
    r"need\s+(the\s+)?full\s+output",
    r"need\s+(the\s+)?raw\s+output",
    r"need\s+more\s+(context|details|logs|output)",
    r"missing\s+(critical\s+)?(details|lines|logs|output|error)",
    r"compressed\s+output\s+(hid|omitted|missed|removed)",
    r"cannot\s+see\s+(the\s+)?(full|raw|complete)\s+(output|log|error)",
]

RETRY_COMMAND_HINT_PATTERNS = [
    r"\bcat\b",
    r"\btail\b",
    r"\bhead\b",
    r"\bsed\b",
    r"\bgrep\b",
    r"\bless\b",
    r"\bmore\b",
    r"\bawk\b",
    r"\bwc\b",
]


class FeedbackCollector:
    """Collects and analyzes feedback signals from agent behavior.

    Usage in the agent loop:
    1. After compression: collector.record_compression(...)
    2. After LLM responds: signal = collector.detect_complaint(...)
    3. After compression: signal = collector.detect_uncovered(...)
    4. If signal is not None: pass to RuleEvolver for action
    """

    def __init__(self, evo_logger: SelfEvoLogger | None = None):
        self._last_command: str | None = None
        self._last_rules_applied: list[str] = []
        self._last_raw_output: str = ""
        self._last_compressed_output: str = ""
        self._last_episode: int = -1
        self._history: list[FeedbackSignal] = []
        self._evo_logger = evo_logger

    @property
    def history(self) -> list[FeedbackSignal]:
        """Read-only access to feedback history."""
        return list(self._history)

    def record_compression(
        self,
        episode: int,
        command: str,
        raw_output: str,
        compressed_output: str,
        rules_applied: list[str],
    ) -> None:
        """Record what happened in the latest compression for feedback analysis.

        Must be called AFTER each _smart_compress() call.
        """
        self._last_episode = episode
        self._last_command = command
        self._last_rules_applied = list(rules_applied)
        self._last_raw_output = raw_output
        self._last_compressed_output = compressed_output

    def detect_complaint(
        self,
        episode: int,
        agent_response: str,
        compression_feedback: str = "",
        analysis: str = "",
        commands: list[str] | None = None,
    ) -> FeedbackSignal | None:
        """Detect if the agent is unhappy with the previous compression.

        Must be called AFTER the LLM responds, with the response text
        and the parsed compression_feedback field from that structured response.

        Returns FeedbackSignal if negative feedback detected, None otherwise.
        """
        # Nothing to check if no dynamic rules fired last round
        if not self._last_rules_applied:
            return None

        signal_source, signal_details = self._detect_complaint_source(
            compression_feedback=compression_feedback,
            analysis=analysis,
            commands=commands or [],
        )
        if not signal_source:
            return None

        signal = FeedbackSignal(
            episode=episode,
            signal_type="complaint",
            command=self._last_command or "",
            rules_applied=list(self._last_rules_applied),
            raw_output_length=len(self._last_raw_output),
            compressed_output_length=len(self._last_compressed_output),
            signal_source=signal_source,
            signal_details=signal_details,
            agent_response_snippet=agent_response[:500],
            raw_output_snippet=self._last_raw_output[:2000],
        )
        self._history.append(signal)
        if self._evo_logger:
            self._evo_logger.log_feedback(
                signal.episode,
                signal.signal_type,
                signal.command,
                signal.rules_applied,
                signal.raw_output_length,
                signal.compressed_output_length,
                signal.agent_response_snippet,
                signal.raw_output_snippet,
            )
        logger.info(
            f"Feedback detected: complaint at episode {episode} "
            f"for command '{self._last_command}', "
            f"rules involved: {self._last_rules_applied}, "
            f"source={signal_source}"
        )
        return signal

    def _detect_complaint_source(
        self,
        compression_feedback: str,
        analysis: str,
        commands: list[str],
    ) -> tuple[str, str]:
        normalized_feedback = compression_feedback.strip().lower()
        if normalized_feedback == COMPRESSION_FEEDBACK_NEED_FULL_OUTPUT:
            return ("explicit_feedback", "structured compression_feedback requested full output")

        analysis_text = analysis.strip().lower()
        if analysis_text and any(
            re.search(pattern, analysis_text) for pattern in ANALYSIS_COMPLAINT_PATTERNS
        ):
            return ("analysis_heuristic", "analysis text indicates compressed output hid critical details")

        if self._looks_like_retry_for_details(commands):
            return ("retry_command_heuristic", "follow-up commands appear to re-open compressed output for missing details")

        return ("", "")

    def _looks_like_retry_for_details(self, commands: list[str]) -> bool:
        if not self._last_command:
            return False

        normalized_last = self._last_command.strip().lower()
        if not normalized_last:
            return False

        for command in commands:
            normalized = command.strip().lower()
            if not normalized:
                continue
            if normalized_last and normalized == normalized_last:
                return True
            if any(re.search(pattern, normalized) for pattern in RETRY_COMMAND_HINT_PATTERNS):
                if any(token in normalized for token in ["log", "output", "error", "trace", "stdout", "stderr"]):
                    return True
        return False

    def detect_uncovered(
        self,
        episode: int,
        command: str,
        raw_output: str,
        rules_applied: list[str],
        threshold: int = 5000,
    ) -> FeedbackSignal | None:
        """Detect if a large output had no dynamic rule coverage.

        Must be called AFTER compression, when we know which rules fired.
        Only triggers if output was big AND no dynamic rules matched.

        Returns FeedbackSignal if uncovered big output detected, None otherwise.
        """
        if len(raw_output) <= threshold:
            return None
        if rules_applied:  # At least one dynamic rule handled it
            return None

        signal = FeedbackSignal(
            episode=episode,
            signal_type="uncovered_big_output",
            command=command,
            rules_applied=[],
            raw_output_length=len(raw_output),
            compressed_output_length=len(raw_output),  # Wasn't compressed
            raw_output_snippet=raw_output[:2000],
        )
        self._history.append(signal)
        if self._evo_logger:
            self._evo_logger.log_feedback(
                signal.episode,
                signal.signal_type,
                signal.command,
                signal.rules_applied,
                signal.raw_output_length,
                signal.compressed_output_length,
                raw_output_snippet=signal.raw_output_snippet,
            )
        logger.info(
            f"Uncovered big output at episode {episode}: "
            f"command '{command}', {len(raw_output)} chars with no dynamic rules"
        )
        return signal
