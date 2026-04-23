"""Compression plan generation — the Proposal phase of self-evo.

At task start, the Planner:
1. Loads cached rules (or seed rules if cold start)
2. Sends them + task description to LLM
3. LLM selects applicable rules, modifies some, creates new ones
4. Returns a CompressionPlan ready for injection into SafeOutputFilter

This is ONE LLM call per task — the primary cost of self-evo.
"""

import json
import logging
import re

from .evo_logger import SelfEvoLogger
from .models import CompressionPlan, CompressionRule

logger = logging.getLogger(__name__)


# --- Prompt Templates ---

# Used when cached rules exist — LLM selects/modifies/supplements
PROPOSAL_PROMPT_WITH_CACHE = """You are a terminal output compression strategy expert.

The system already has these baseline filters (you do NOT need to generate rules for these):
- ANSI escape code removal
- System login banner (Ubuntu MOTD) removal
- Empty command polling state handling

Below are historical compression rules from previous tasks. Select the ones
relevant to the current task, modify any that need adjustment, and create
new rules if needed.

Historical rules:
{cached_rules_json}

Current task description:
{instruction}

Task category: {task_category}

Current terminal environment (first 500 chars):
{terminal_state}

Instructions:
1. "selected_rule_ids": List rule_ids of rules to use AS-IS from the historical set
2. "modified_rules": For rules that are close but need adjustment, output the full
   modified rule with a NEW rule_id (e.g., original_id + "_mod")
3. "new_rules": For command types not covered by any historical rule, create new rules

Requirements:
- Only create rules for HIGH-OUTPUT commands (pip, apt, make, pytest, git, docker, etc.)
- Do NOT create rules for short-output commands (ls, cat, echo, pwd, cd)
- NEVER compress error output — errors must always be fully preserved
- Be conservative: when in doubt, KEEP the line rather than strip it
- Total rules (selected + modified + new) should be 3-7

Output a single JSON object:
{{
  "selected_rule_ids": ["id1", "id2"],
  "modified_rules": [
    {{
      "rule_id": "string",
      "trigger_regex": "string",
      "description": "string",
      "keep_patterns": ["regex1"],
      "strip_patterns": ["regex1"],
      "keep_first_n": 5,
      "keep_last_n": 10,
      "max_lines": null,
      "summary_header": "[description]",
      "priority": 42
    }}
  ],
  "new_rules": [
    {{same format as modified_rules}}
  ]
}}

Output ONLY the JSON object, no other text."""

# Used on cold start — LLM generates rules from scratch
PROPOSAL_PROMPT_NO_CACHE = """You are a terminal output compression strategy expert.

The system already has these baseline filters (you do NOT need to generate rules for these):
- ANSI escape code removal
- System login banner (Ubuntu MOTD) removal
- Empty command polling state handling

Given the task below, predict which terminal commands will produce long outputs,
and create compression rules for them.

Task description:
{instruction}

Task category: {task_category}

Current terminal environment (first 500 chars):
{terminal_state}

Requirements:
- Only create rules for HIGH-OUTPUT commands (pip, apt, make, pytest, git, docker, etc.)
- Do NOT create rules for short-output commands (ls, cat, echo, pwd, cd)
- NEVER compress error output — errors must always be fully preserved
- Be conservative: when in doubt, KEEP the line rather than strip it
- Generate 3-7 rules

For each rule, provide:
- trigger_regex: regex to match the command string
- description: what this rule does
- keep_patterns: regex patterns for lines that MUST be preserved
- strip_patterns: regex patterns for lines safe to remove
- keep_first_n: always keep first N lines (default 5)
- keep_last_n: always keep last N lines (default 10)
- max_lines: cap on body lines after filtering (null = no cap)
- summary_header: text to show when lines are removed

Output a single JSON object:
{{
  "rules": [
    {{
      "rule_id": "string",
      "trigger_regex": "string",
      "description": "string",
      "keep_patterns": ["regex1"],
      "strip_patterns": ["regex1"],
      "keep_first_n": 5,
      "keep_last_n": 10,
      "max_lines": null,
      "summary_header": "[description]",
      "priority": 42
    }}
  ]
}}

Output ONLY the JSON object, no other text."""


class CompressionPlanner:
    """Generates compression plans at task start via one LLM call.

    The planner is the entry point of the self-evo system. It takes cached/seed
    rules and the current task context, then produces a CompressionPlan containing
    the optimal rule set for this specific task.
    """

    def __init__(self, llm_client, evo_logger: SelfEvoLogger | None = None):
        """Initialize with an LLM client.

        Args:
            llm_client: An LLMClient instance (from module.client) with
                       .chat(user_content=...) method.
            evo_logger: Optional structured event logger for self-evo diagnostics.
        """
        self._client = llm_client
        self._evo_logger = evo_logger
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _accumulate_usage(self, usage: dict) -> None:
        self.total_input_tokens += usage.get("prompt_tokens", 0)
        self.total_output_tokens += usage.get("completion_tokens", 0)

    def generate_plan(
        self,
        instruction: str,
        task_category: str,
        terminal_state: str,
        cached_rules: list[CompressionRule],
    ) -> CompressionPlan:
        """Generate a compression plan for the current task.

        Args:
            instruction: The task description.
            task_category: Category from step1_design (e.g., "software-engineering").
            terminal_state: Current terminal screen content.
            cached_rules: Rules loaded from cache (or seed rules for cold start).

        Returns:
            CompressionPlan with selected, modified, and new rules.
            Falls back to a plan using all cached rules if LLM fails.
        """
        try:
            if cached_rules:
                plan = self._generate_with_cache(
                    instruction, task_category, terminal_state, cached_rules
                )
            else:
                plan = self._generate_no_cache(
                    instruction, task_category, terminal_state
                )
            logger.info(
                f"Generated compression plan: "
                f"{len(plan.selected_rule_ids)} selected, "
                f"{len(plan.modified_rules)} modified, "
                f"{len(plan.new_rules)} new"
            )
            return plan
        except Exception:
            logger.exception("Plan generation failed, using fallback")
            fallback = self._fallback_plan(task_category, cached_rules)
            if self._evo_logger:
                self._evo_logger.log_plan_response(
                    "",
                    fallback.selected_rule_ids,
                    [],
                    [],
                    used_fallback=True,
                )
            return fallback

    def _generate_with_cache(
        self,
        instruction: str,
        task_category: str,
        terminal_state: str,
        cached_rules: list[CompressionRule],
    ) -> CompressionPlan:
        """Generate plan when cached rules are available."""
        # Serialize rules for the prompt (only include fields LLM needs to see)
        rules_for_prompt = [
            {
                "rule_id": r.rule_id,
                "trigger_regex": r.trigger_regex,
                "description": r.description,
                "keep_patterns": r.keep_patterns,
                "strip_patterns": r.strip_patterns,
                "keep_first_n": r.keep_first_n,
                "keep_last_n": r.keep_last_n,
                "max_lines": r.max_lines,
                "summary_header": r.summary_header,
                "confidence": r.confidence,
                "times_applied": r.times_applied,
                "times_complained": r.times_complained,
            }
            for r in cached_rules
        ]

        prompt = PROPOSAL_PROMPT_WITH_CACHE.format(
            cached_rules_json=json.dumps(
                rules_for_prompt, ensure_ascii=False, indent=2
            ),
            instruction=instruction[:1000],
            task_category=task_category,
            terminal_state=terminal_state[:500],
        )

        if self._evo_logger:
            self._evo_logger.log_plan_request(
                task_category,
                instruction,
                [r.rule_id for r in cached_rules],
                prompt,
            )

        response, usage = self._client.chat_with_usage(user_content=prompt)
        self._accumulate_usage(usage)
        plan = self._parse_plan_with_cache(response, task_category, cached_rules)

        if self._evo_logger:
            self._evo_logger.log_plan_response(
                response,
                plan.selected_rule_ids,
                [r.model_dump() for r in plan.modified_rules],
                [r.model_dump() for r in plan.new_rules],
            )

        return plan

    def _generate_no_cache(
        self,
        instruction: str,
        task_category: str,
        terminal_state: str,
    ) -> CompressionPlan:
        """Generate plan from scratch when no cache exists."""
        prompt = PROPOSAL_PROMPT_NO_CACHE.format(
            instruction=instruction[:1000],
            task_category=task_category,
            terminal_state=terminal_state[:500],
        )

        if self._evo_logger:
            self._evo_logger.log_plan_request(
                task_category,
                instruction,
                [],
                prompt,
            )

        response, usage = self._client.chat_with_usage(user_content=prompt)
        self._accumulate_usage(usage)
        plan = self._parse_plan_no_cache(response, task_category)

        if self._evo_logger:
            self._evo_logger.log_plan_response(
                response,
                plan.selected_rule_ids,
                [r.model_dump() for r in plan.modified_rules],
                [r.model_dump() for r in plan.new_rules],
            )

        return plan

    def _parse_plan_with_cache(
        self,
        response: str,
        task_category: str,
        cached_rules: list[CompressionRule],
    ) -> CompressionPlan:
        """Parse LLM response for the with-cache prompt format."""
        data = self._extract_json(response)

        # Parse selected_rule_ids — validate they actually exist in cache
        cached_ids = {r.rule_id for r in cached_rules}
        selected_ids = [
            rid for rid in data.get("selected_rule_ids", []) if rid in cached_ids
        ]

        used_rule_ids = set(cached_ids)

        # Parse modified rules. They must not reuse historical rule_ids, otherwise
        # save() will treat them as the old cached rule and delta-counting breaks.
        modified, replaced_selected_ids = self._normalize_generated_rule_ids(
            self._parse_rule_list(data.get("modified_rules", [])),
            used_rule_ids=used_rule_ids,
            cached_ids=cached_ids,
            rule_kind="modified",
        )

        if replaced_selected_ids:
            selected_ids = [
                rid for rid in selected_ids if rid not in replaced_selected_ids
            ]

        # Parse new rules. These also get normalized to avoid collisions with
        # historical rule_ids or each other.
        new, _ = self._normalize_generated_rule_ids(
            self._parse_rule_list(data.get("new_rules", [])),
            used_rule_ids=used_rule_ids,
            cached_ids=cached_ids,
            rule_kind="new",
        )

        return CompressionPlan(
            task_category=task_category,
            selected_rule_ids=selected_ids,
            modified_rules=modified,
            new_rules=new,
        )

    def _parse_plan_no_cache(
        self,
        response: str,
        task_category: str,
    ) -> CompressionPlan:
        """Parse LLM response for the no-cache prompt format."""
        data = self._extract_json(response)
        rules, _ = self._normalize_generated_rule_ids(
            self._parse_rule_list(data.get("rules", [])),
            used_rule_ids=set(),
            cached_ids=set(),
            rule_kind="new",
        )

        return CompressionPlan(
            task_category=task_category,
            selected_rule_ids=[],
            modified_rules=[],
            new_rules=rules,
        )

    @staticmethod
    def _make_unique_rule_id(base_rule_id: str, used_rule_ids: set[str]) -> str:
        """Return a rule_id that doesn't collide with any existing/generated rule."""
        candidate = base_rule_id
        suffix = 2
        while candidate in used_rule_ids:
            candidate = f"{base_rule_id}_{suffix}"
            suffix += 1
        return candidate

    @classmethod
    def _normalize_generated_rule_ids(
        cls,
        rules: list[CompressionRule],
        *,
        used_rule_ids: set[str],
        cached_ids: set[str],
        rule_kind: str,
    ) -> tuple[list[CompressionRule], set[str]]:
        """Rename generated rules that collide with cached/generated identifiers."""
        normalized_rules: list[CompressionRule] = []
        replaced_cached_ids: set[str] = set()
        suffix = "mod" if rule_kind == "modified" else "new"

        for rule in rules:
            original_id = rule.rule_id

            if original_id in cached_ids:
                replaced_cached_ids.add(original_id)
                new_rule_id = cls._make_unique_rule_id(
                    f"{original_id}_{suffix}", used_rule_ids
                )
                logger.warning(
                    "%s rule reused cached rule_id '%s'; renamed to '%s'",
                    rule_kind.capitalize(),
                    original_id,
                    new_rule_id,
                )
                rule = rule.model_copy(update={"rule_id": new_rule_id})
            elif original_id in used_rule_ids:
                new_rule_id = cls._make_unique_rule_id(
                    f"{original_id}_{suffix}", used_rule_ids
                )
                logger.warning(
                    "Duplicate generated %s rule_id '%s'; renamed to '%s'",
                    rule_kind,
                    original_id,
                    new_rule_id,
                )
                rule = rule.model_copy(update={"rule_id": new_rule_id})

            used_rule_ids.add(rule.rule_id)
            normalized_rules.append(rule)

        return normalized_rules, replaced_cached_ids

    @staticmethod
    def _parse_rule_list(raw_rules: list[dict]) -> list[CompressionRule]:
        """Parse a list of raw dicts into CompressionRule instances."""
        rules = []
        for raw in raw_rules:
            if not isinstance(raw, dict):
                continue
            raw = raw.copy()
            if "rule_id" not in raw or "trigger_regex" not in raw:
                logger.warning(f"Skipping rule without rule_id or trigger_regex: {raw}")
                continue

            # Validate regex compiles
            try:
                re.compile(raw["trigger_regex"])
            except re.error as e:
                logger.warning(f"Skipping rule with bad trigger_regex: {e}")
                continue

            # Set defaults
            raw.setdefault("description", "")
            raw.setdefault("keep_patterns", [])
            raw.setdefault("strip_patterns", [])
            raw.setdefault("keep_first_n", 5)
            raw.setdefault("keep_last_n", 10)
            raw.setdefault("max_lines", None)
            raw.setdefault("summary_header", None)
            raw.setdefault("priority", 42)

            # Fresh confidence for generated rules
            raw["confidence"] = 1.0
            raw["times_applied"] = 0
            raw["times_complained"] = 0

            try:
                rules.append(CompressionRule(**raw))
            except Exception as e:
                logger.warning(f"Failed to parse rule '{raw.get('rule_id')}': {e}")
        return rules

    @staticmethod
    def _extract_json(response: str) -> dict:
        """Extract JSON object from LLM response text."""
        text = response.strip()

        # Strip <think>...</think> blocks (Qwen3 reasoning output)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Remove markdown code block wrapper
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.endswith("```"):
                text = text[:-3].strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object via regex
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to extract JSON from LLM response")
        logger.warning(f"Raw response (first 500 chars): {response[:500]}")
        return {}

    @staticmethod
    def _fallback_plan(
        task_category: str,
        cached_rules: list[CompressionRule],
    ) -> CompressionPlan:
        """Fallback plan if LLM call fails — just use all cached rules."""
        return CompressionPlan(
            task_category=task_category,
            selected_rule_ids=[r.rule_id for r in cached_rules],
            modified_rules=[],
            new_rules=[],
        )
