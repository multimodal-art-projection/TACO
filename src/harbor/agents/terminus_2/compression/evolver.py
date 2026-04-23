"""Rule evolution — spawns new/replacement compression rules via LLM.

This module handles the "evolve" phase of the self-evo loop:
- spawn_replacement(): When feedback shows a rule compressed too aggressively,
  freeze the old rule and generate a better one via LLM.
- spawn_new(): When a large output has no rule coverage, generate a new
  rule specifically for that command type.
- boost_confidence(): When rules work well, gradually increase their confidence.

LLM calls here are RARE events — typically 0-3 per task.
"""

import json
import logging
import re

from .evo_logger import SelfEvoLogger
from .models import CompressionRule, FeedbackSignal

logger = logging.getLogger(__name__)


# --- Prompt Templates ---

SPAWN_REPLACEMENT_PROMPT = """You are a terminal output compression rule expert.

The following rule compressed terminal output too aggressively, causing the
agent to miss critical information.

Old rule (JSON):
{old_rule_json}

Command that was executed: {command}

Original terminal output (first 2000 chars):
{raw_output_snippet}

Agent's feedback (what it complained about):
{agent_feedback}

Generate a NEW replacement rule that:
1. Keeps the same trigger_regex (targets same command type)
2. Is MORE CONSERVATIVE — preserves more information
3. Stays SPECIFIC to this command type (don't make a generic "keep everything" rule)
4. Adds the missing information type to keep_patterns
5. Only strips content that is 100% guaranteed noise (progress bars, blank lines, etc.)
6. Uses a new rule_id (suggest: old_id + "_v2" or similar)

Output a single JSON object with these fields:
{{
  "rule_id": "string",
  "trigger_regex": "string",
  "description": "string",
  "keep_patterns": ["regex1", "regex2"],
  "strip_patterns": ["regex1", "regex2"],
  "keep_first_n": 5,
  "keep_last_n": 10,
  "max_lines": null,
  "summary_header": "[description of what was compressed]",
  "priority": 42
}}

Output ONLY the JSON object, no other text."""

SPAWN_NEW_PROMPT = """You are a terminal output compression rule expert.

The agent executed a command that produced a very long output ({output_length} chars),
but no compression rule exists for this command type.

Command: {command}

Output (first 2000 chars):
{raw_output_head}

Output (last 500 chars):
{raw_output_tail}

Task context: {task_instruction}

Generate a compression rule for this type of command. The rule should:
1. Have a trigger_regex that matches this CATEGORY of command (not just this exact command)
2. Identify repetitive/progress/noise patterns in the output to strip
3. Preserve all error messages, results, and actionable information
4. Be conservative — when in doubt, keep the line

Output a single JSON object with these fields:
{{
  "rule_id": "string",
  "trigger_regex": "string",
  "description": "string",
  "keep_patterns": ["regex1", "regex2"],
  "strip_patterns": ["regex1", "regex2"],
  "keep_first_n": 5,
  "keep_last_n": 10,
  "max_lines": null,
  "summary_header": "[description of what was compressed]",
  "priority": 42
}}

Output ONLY the JSON object, no other text."""


class RuleEvolver:
    """Evolves compression rules based on feedback signals.

    Uses LLM to generate replacement or new rules when the current rule set
    is insufficient. Also handles confidence boosting for well-performing rules.

    LLM calls are rare — only triggered by feedback signals (typically 0-3 per task).
    """

    def __init__(self, llm_client, evo_logger: SelfEvoLogger | None = None):
        """Initialize with an LLM client.

        Args:
            llm_client: An LLMClient instance (from module.client) with .chat() method.
        """
        self._client = llm_client
        self._evo_logger = evo_logger
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _accumulate_usage(self, usage: dict) -> None:
        self.total_input_tokens += usage.get("prompt_tokens", 0)
        self.total_output_tokens += usage.get("completion_tokens", 0)

    def spawn_replacement(
        self,
        old_rule: CompressionRule,
        signal: FeedbackSignal,
    ) -> CompressionRule | None:
        """Freeze the old rule and generate a replacement via LLM.

        Args:
            old_rule: The rule that caused the complaint. Will be frozen (confidence=0).
            signal: The feedback signal with context about what went wrong.

        Returns:
            A new CompressionRule, or None if LLM call/parsing failed.
        """
        # Step 1: Freeze the old rule immediately
        old_rule.confidence = 0.0
        old_rule.times_complained += 1
        logger.info(f"Froze rule '{old_rule.rule_id}' (confidence → 0)")

        # Step 2: Generate replacement via LLM
        prompt = SPAWN_REPLACEMENT_PROMPT.format(
            old_rule_json=old_rule.model_dump_json(indent=2),
            command=signal.command,
            raw_output_snippet=signal.raw_output_snippet[:2000],
            agent_feedback=signal.agent_response_snippet[:500],
        )

        if self._evo_logger:
            self._evo_logger.log_evolve_request(
                "spawn_replacement", prompt, old_rule.rule_id
            )

        new_rule = self._call_llm_for_rule(prompt)

        if self._evo_logger:
            self._evo_logger.log_evolve_response(
                "spawn_replacement",
                "",
                new_rule.model_dump() if new_rule else None,
                old_rule.rule_id,
            )

        if new_rule:
            logger.info(
                f"Spawned replacement rule '{new_rule.rule_id}' "
                f"for frozen '{old_rule.rule_id}'"
            )
        else:
            logger.warning(f"Failed to spawn replacement for '{old_rule.rule_id}'")
        return new_rule

    def spawn_new(
        self,
        signal: FeedbackSignal,
        task_instruction: str,
    ) -> CompressionRule | None:
        """Generate a new rule for an uncovered command type.

        Args:
            signal: The feedback signal indicating uncovered big output.
            task_instruction: The task description for context.

        Returns:
            A new CompressionRule, or None if LLM call/parsing failed.
        """
        raw = signal.raw_output_snippet
        prompt = SPAWN_NEW_PROMPT.format(
            output_length=signal.raw_output_length,
            command=signal.command,
            raw_output_head=raw[:2000],
            raw_output_tail=raw[-500:] if len(raw) > 2500 else "",
            task_instruction=task_instruction[:500],
        )

        if self._evo_logger:
            self._evo_logger.log_evolve_request("spawn_new", prompt)

        new_rule = self._call_llm_for_rule(prompt)

        if self._evo_logger:
            self._evo_logger.log_evolve_response(
                "spawn_new",
                "",
                new_rule.model_dump() if new_rule else None,
            )

        if new_rule:
            logger.info(
                f"Spawned new rule '{new_rule.rule_id}' "
                f"for uncovered command '{signal.command[:50]}'"
            )
        else:
            logger.warning(
                f"Failed to spawn new rule for command '{signal.command[:50]}'"
            )
        return new_rule

    @staticmethod
    def boost_confidence(
        rules: list[CompressionRule],
        rule_ids: list[str],
        boost_factor: float = 1.05,
        max_confidence: float = 1.0,
    ) -> None:
        """Slightly increase confidence for rules that worked well.

        Called when no negative feedback is detected after a compression event.

        Args:
            rules: All active rules to search through.
            rule_ids: IDs of rules to boost (those that fired last episode).
            boost_factor: Multiplicative factor (default 1.05 = +5%).
            max_confidence: Upper bound on confidence.
        """
        for rule in rules:
            if rule.rule_id in rule_ids:
                old = rule.confidence
                rule.confidence = min(max_confidence, rule.confidence * boost_factor)
                if rule.confidence != old:
                    logger.debug(
                        f"Boosted rule '{rule.rule_id}' confidence: "
                        f"{old:.2f} → {rule.confidence:.2f}"
                    )

    def _call_llm_for_rule(self, prompt: str) -> CompressionRule | None:
        """Call LLM and parse the response into a CompressionRule.

        Handles JSON extraction, validation, and error recovery.
        """
        try:
            response, usage = self._client.chat_with_usage(user_content=prompt)
            self._accumulate_usage(usage)
            return self._parse_rule_from_response(response)
        except Exception:
            logger.exception("LLM call failed for rule generation")
            return None

    @staticmethod
    def _parse_rule_from_response(response: str) -> CompressionRule | None:
        """Extract a CompressionRule from an LLM response string.

        Handles common LLM output quirks:
        - JSON wrapped in markdown code blocks
        - Extra text before/after JSON
        - Missing optional fields
        """
        text = response.strip()

        # Strip <think>...</think> blocks (Qwen3 reasoning output)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.endswith("```"):
                text = text[:-3].strip()

        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return None

        # Ensure required fields exist
        if "rule_id" not in data or "trigger_regex" not in data:
            logger.warning(
                "LLM response missing required fields (rule_id, trigger_regex)"
            )
            return None

        # Set defaults for optional fields
        data.setdefault("description", "")
        data.setdefault("keep_patterns", [])
        data.setdefault("strip_patterns", [])
        data.setdefault("keep_first_n", 5)
        data.setdefault("keep_last_n", 10)
        data.setdefault("max_lines", None)
        data.setdefault("summary_header", None)
        data.setdefault("priority", 42)

        # Reset evolution tracking for new rules
        data["confidence"] = 1.0
        data["times_applied"] = 0
        data["times_complained"] = 0

        # Validate the trigger_regex compiles
        try:
            re.compile(data["trigger_regex"])
        except re.error as e:
            logger.warning(f"LLM generated invalid trigger_regex: {e}")
            return None

        try:
            return CompressionRule(**data)
        except Exception as e:
            logger.warning(f"Failed to create CompressionRule from LLM data: {e}")
            return None
