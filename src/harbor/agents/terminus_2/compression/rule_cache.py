"""Cross-task compression rule persistence.

Stores evolved compression rules in a JSON file organized by task_category.
Falls back to seed rules when no cached rules exist for a category.

Cache lifecycle:
1. Task start: load() → returns cached rules or seed rules
2. Task end: save() → persists good rules, degrades frozen ones
3. Cross-task: rules accumulate and improve via confidence fusion
"""

import fcntl
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from .models import CompressionRule
from .seed_rules import get_seed_rules

logger = logging.getLogger(__name__)

# Default cache location
DEFAULT_CACHE_PATH = Path.home() / ".harbor" / "compression_rules_cache.json"


def _counter_delta(current_value: int, previous_value: int) -> int:
    """Return the new increments accumulated during the current task only."""
    return max(0, current_value - previous_value)


class RuleCache:
    """Manages persistent storage of compression rules across tasks.

    Rules are organized by task_category. Each category has its own
    list of rules that evolve over time.
    """

    def __init__(self, cache_path: Path = DEFAULT_CACHE_PATH):
        self._cache_path = cache_path
        self._lock_path = cache_path.with_suffix(".lock")

    @contextmanager
    def _file_lock(self, *, shared: bool = False) -> Generator[None, None, None]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = open(self._lock_path, "w")  # noqa: SIM115
        try:
            lock_type = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(lock_fd, lock_type)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def load(self, task_category: str, max_rules: int = 30) -> list[CompressionRule]:
        """Load cached rules for a task category, falling back to seeds.

        Args:
            task_category: The task category to load rules for.
            max_rules: Maximum number of rules to return (sorted by quality).

        Returns:
            List of CompressionRule instances. If no cache exists for this
            category, returns seed rules as a baseline.
        """
        cached = self._read_category(task_category)

        if not cached:
            logger.info(f"No cached rules for '{task_category}', using seed rules")
            return get_seed_rules()

        # Sort by quality score: confidence × (times_applied + 1)
        # The +1 avoids zero-multiplication for fresh rules
        cached.sort(
            key=lambda r: r.confidence * (r.times_applied + 1),
            reverse=True,
        )

        if len(cached) > max_rules:
            logger.info(
                f"Trimming {len(cached)} cached rules to {max_rules} for '{task_category}'"
            )
            cached = cached[:max_rules]

        logger.info(f"Loaded {len(cached)} cached rules for '{task_category}'")
        return cached

    def save(self, task_category: str, rules: list[CompressionRule]) -> None:
        """Persist rules after a task completes.

        Rules are merged with existing cache:
        - Good rules (confidence > 0.2, applied >= 1): stored/updated
        - Frozen rules (confidence = 0): existing cache entry degraded but not deleted
        - Unused rules (applied = 0): ignored (don't pollute cache with bad predictions)

        Cross-task confidence fusion: blends old and new confidence values
        to avoid single-task bias.

        The entire read-modify-write is protected by an exclusive file lock
        so concurrent tasks don't overwrite each other's updates.
        """
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

        with self._file_lock(shared=False):
            # Read existing cache inside the lock
            existing_map: dict[str, CompressionRule] = {}
            if self._cache_path.exists():
                try:
                    full_cache = json.loads(
                        self._cache_path.read_text(encoding="utf-8")
                    )
                    for r in full_cache.get(task_category, []):
                        try:
                            parsed = CompressionRule(**r)
                            existing_map[parsed.rule_id] = parsed
                        except Exception:
                            continue
                except (json.JSONDecodeError, TypeError):
                    full_cache = {}
            else:
                full_cache = {}

            for rule in rules:
                if rule.confidence <= 0.0:
                    if rule.rule_id in existing_map:
                        old = existing_map[rule.rule_id]
                        old.confidence *= 0.5

                        # rule.times_complained includes the historical value loaded
                        # from cache, so only persist the new complaints from this task.
                        delta_complained = _counter_delta(
                            rule.times_complained, old.times_complained
                        )
                        old.times_complained += delta_complained
                        logger.debug(f"Degraded frozen rule '{rule.rule_id}' in cache")
                    continue

                if rule.times_applied < 1:
                    continue

                if rule.confidence < 0.2:
                    continue

                if rule.rule_id in existing_map:
                    old = existing_map[rule.rule_id]
                    merged_confidence = 0.7 * old.confidence + 0.3 * rule.confidence
                    old.confidence = merged_confidence

                    # Only add the delta (new applications in this task)
                    # rule.times_applied includes the old value loaded from cache
                    # So we use max() to handle the increment correctly
                    delta_applied = _counter_delta(
                        rule.times_applied, old.times_applied
                    )
                    old.times_applied += delta_applied

                    delta_complained = _counter_delta(
                        rule.times_complained, old.times_complained
                    )
                    old.times_complained += delta_complained
                    old.keep_patterns = rule.keep_patterns
                    old.strip_patterns = rule.strip_patterns
                    old.description = rule.description
                    logger.debug(
                        f"Updated cached rule '{rule.rule_id}': "
                        f"confidence={merged_confidence:.2f}, "
                        f"delta_applied={delta_applied}, "
                        f"delta_complained={delta_complained}, "
                        f"total_applied={old.times_applied}"
                    )
                else:
                    existing_map[rule.rule_id] = rule.model_copy(deep=True)
                    logger.debug(f"Added new rule '{rule.rule_id}' to cache")

            # Write back atomically inside the same lock
            full_cache[task_category] = [r.model_dump() for r in existing_map.values()]
            tmp_path = self._cache_path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(full_cache, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.rename(self._cache_path)

        logger.info(f"Saved {len(existing_map)} rules for '{task_category}'")

    def _read_category(self, task_category: str) -> list[CompressionRule]:
        """Read rules for a single category from the cache file."""
        if not self._cache_path.exists():
            return []
        try:
            with self._file_lock(shared=True):
                cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            raw_rules = cache.get(task_category, [])
            return [CompressionRule(**r) for r in raw_rules]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to read cache: {e}")
            return []
