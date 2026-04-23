"""Structured event logger for the self-evolving compression system.

Writes all self-evo events to ``self_evo_log.jsonl`` in the agent's
logging directory.  Each line is a JSON object with a ``type`` field
indicating the event kind, a ``timestamp``, and event-specific payload.

Event types
-----------
- ``plan_request``   – LLM prompt sent to the planner
- ``plan_response``  – raw LLM response + parsed plan
- ``rule_applied``   – a dynamic rule fired during compression
- ``feedback``       – complaint / uncovered signal
- ``evolve_request`` – LLM prompt for spawn_replacement / spawn_new
- ``evolve_response``– raw LLM response + parsed new rule
- ``boost``          – confidence boost applied to rules
- ``final_state``    – snapshot of all rules at task end
"""

import json
import time
from pathlib import Path
from typing import Any


class SelfEvoLogger:
    """Append-only JSONL logger for self-evo events."""

    def __init__(self, log_dir: Path):
        self._log_path = log_dir / "self_evo_log.jsonl"
        log_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "type": event_type,
            "timestamp": time.time(),
            **payload,
        }
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Plan phase
    # ------------------------------------------------------------------

    def log_plan_request(
        self,
        task_category: str,
        instruction: str,
        cached_rule_ids: list[str],
        prompt: str,
    ) -> None:
        self._write(
            "plan_request",
            {
                "task_category": task_category,
                "instruction_snippet": instruction[:500],
                "cached_rule_ids": cached_rule_ids,
                "prompt": prompt,
            },
        )

    def log_plan_response(
        self,
        raw_response: str,
        selected_rule_ids: list[str],
        modified_rules: list[dict],
        new_rules: list[dict],
        used_fallback: bool = False,
    ) -> None:
        self._write(
            "plan_response",
            {
                "raw_response": raw_response,
                "selected_rule_ids": selected_rule_ids,
                "modified_rules": modified_rules,
                "new_rules": new_rules,
                "used_fallback": used_fallback,
            },
        )

    # ------------------------------------------------------------------
    # Rule application
    # ------------------------------------------------------------------

    def log_rule_applied(
        self,
        episode: int,
        command: str,
        rule_id: str,
        raw_length: int,
        filtered_length: int,
        lines_stripped: int,
    ) -> None:
        self._write(
            "rule_applied",
            {
                "episode": episode,
                "command": command,
                "rule_id": rule_id,
                "raw_length": raw_length,
                "filtered_length": filtered_length,
                "lines_stripped": lines_stripped,
            },
        )

    # ------------------------------------------------------------------
    # Feedback detection
    # ------------------------------------------------------------------

    def log_feedback(
        self,
        episode: int,
        signal_type: str,
        command: str,
        rules_applied: list[str],
        raw_output_length: int,
        compressed_output_length: int,
        agent_response_snippet: str = "",
        raw_output_snippet: str = "",
    ) -> None:
        self._write(
            "feedback",
            {
                "episode": episode,
                "signal_type": signal_type,
                "command": command,
                "rules_applied": rules_applied,
                "raw_output_length": raw_output_length,
                "compressed_output_length": compressed_output_length,
                "agent_response_snippet": agent_response_snippet[:500],
                "raw_output_snippet": raw_output_snippet[:2000],
            },
        )

    # ------------------------------------------------------------------
    # Evolve operations
    # ------------------------------------------------------------------

    def log_evolve_request(
        self,
        operation: str,
        prompt: str,
        old_rule_id: str | None = None,
    ) -> None:
        self._write(
            "evolve_request",
            {
                "operation": operation,
                "old_rule_id": old_rule_id,
                "prompt": prompt,
            },
        )

    def log_evolve_response(
        self,
        operation: str,
        raw_response: str,
        new_rule: dict | None,
        old_rule_id: str | None = None,
    ) -> None:
        self._write(
            "evolve_response",
            {
                "operation": operation,
                "old_rule_id": old_rule_id,
                "raw_response": raw_response,
                "new_rule": new_rule,
            },
        )

    def log_boost(
        self,
        episode: int,
        boosted_rule_ids: list[str],
        boost_factor: float,
    ) -> None:
        self._write(
            "boost",
            {
                "episode": episode,
                "boosted_rule_ids": boosted_rule_ids,
                "boost_factor": boost_factor,
            },
        )

    # ------------------------------------------------------------------
    # Final state
    # ------------------------------------------------------------------

    def log_final_state(self, rules: list[dict]) -> None:
        self._write(
            "final_state",
            {
                "rules": rules,
            },
        )
