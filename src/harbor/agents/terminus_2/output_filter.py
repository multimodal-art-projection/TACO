"""
Terminal Output Filter - Modular Filter Chain Architecture

Core Principle:
"If this content is compressed from the context, the Agent will still make
the exact same correct decision, and the task result will remain unchanged."

Only content that is 100% certain not to affect agent decision will be compressed.

Architecture:
- FilterConfig: Configuration class for controlling filter behavior
- FilterContext: Context passed between filters
- FilterResult: Result of filtering operation
- BaseFilter: Abstract base class for all filters
- SafeOutputFilter: Main filter orchestrator using filter chain
"""

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class FilterState(Enum):
    """Filter state machine states"""

    IDLE = "idle"  # Normal execution state
    WAITING = "waiting"  # Waiting for command completion (1st empty command)
    POLLING = "polling"  # Continuous waiting (consecutive empty commands)


# ============================================================
# Configuration Classes
# ============================================================


@dataclass
class FilterConfig:
    """
    Filter configuration - One-stop control center

    Controls:
    - LLM compression behavior
    - Which filters are enabled/disabled
    - Various thresholds
    - Debug mode
    """

    # ===== LLM Compression Control =====
    # Always run LLM compression after rule-based filtering
    always_llm_compress: bool = False

    # Also run LLM compression for error outputs
    llm_compress_errors: bool = False

    # Length threshold for triggering LLM compression
    llm_compress_threshold: int = 2000

    # Package install output LLM compression threshold (lower, as these are more redundant)
    package_install_llm_threshold: int = 200

    # ===== Filter Control =====
    # Enabled filters: ["all"] means all enabled
    # Can also specify: ["ansi", "git", "pip", "apt", ...]
    enabled_filters: List[str] = field(default_factory=lambda: ["all"])

    # Disabled filters (takes priority over enabled)
    disabled_filters: List[str] = field(default_factory=list)

    # ===== Thresholds =====
    # Minimum length to apply filtering
    min_length_to_filter: int = 100

    # Consecutive empty commands to trigger POLLING state
    polling_trigger_count: int = 2

    # ===== Debug =====
    # Debug mode: record detailed filter execution info
    debug: bool = False

    def is_filter_enabled(self, filter_name: str) -> bool:
        """Check if a specific filter is enabled"""
        if filter_name in self.disabled_filters:
            return False
        if "all" in self.enabled_filters:
            return True
        return filter_name in self.enabled_filters


@dataclass
class FilterContext:
    """
    Filter context - Passed between filters

    Contains information about the current filtering operation
    and allows filters to communicate with each other.
    """

    command: str  # Current command
    has_error: bool  # Whether output contains error
    state: FilterState  # Current state (IDLE/WAITING/POLLING)
    last_command: Optional[str]  # Previous command
    original_length: int  # Original output length

    # Flags that filters can set (affects subsequent logic)
    is_package_install: bool = False  # Detected as package installation
    is_redundant: bool = False  # Output is redundant (e.g., polling)
    suggested_llm_type: Optional[str] = None  # Suggested LLM prompt type
    short_circuit: bool = False  # Skip remaining filters

    # Debug info (populated when debug=True)
    filter_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class FilterResult:
    """
    Filter result - Output of the filtering operation
    """

    output: str  # Filtered output
    need_llm_compress: bool  # Whether to trigger LLM compression

    # LLM compression type for selecting appropriate prompt
    # Values: "default" | "error" | "install" | "running"
    llm_compress_type: str = "default"

    is_redundant: bool = False  # Is redundant information
    status: str = "success"  # success | failed | running
    compression_applied: bool = False  # Was any compression applied
    original_length: int = 0  # Original length
    compressed_length: int = 0  # Compressed length

    # Debug: which filters were applied
    applied_filters: List[str] = field(default_factory=list)

    # Debug: detailed stats per filter
    debug_info: Optional[Dict[str, Any]] = None


# ============================================================
# Filter Statistics
# ============================================================


@dataclass
class FilterStats:
    """
    Filter statistics - For analyzing filter effectiveness
    """

    total_calls: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    llm_compress_triggered: int = 0
    filter_hits: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Per-filter compression stats
    filter_compression: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(
            lambda: {"before": 0, "after": 0, "calls": 0}
        )
    )

    def record_call(
        self,
        original_len: int,
        compressed_len: int,
        applied_filters: List[str],
        llm_triggered: bool,
    ) -> None:
        """Record a filtering operation"""
        self.total_calls += 1
        self.total_original_bytes += original_len
        self.total_compressed_bytes += compressed_len
        if llm_triggered:
            self.llm_compress_triggered += 1
        for f in applied_filters:
            self.filter_hits[f] += 1

    def record_filter_compression(
        self, filter_name: str, before: int, after: int
    ) -> None:
        """Record compression stats for a specific filter"""
        stats = self.filter_compression[filter_name]
        stats["before"] += before
        stats["after"] += after
        stats["calls"] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        compression_ratio = (
            1 - self.total_compressed_bytes / self.total_original_bytes
            if self.total_original_bytes > 0
            else 0
        )
        return {
            "total_calls": self.total_calls,
            "total_original_bytes": self.total_original_bytes,
            "total_compressed_bytes": self.total_compressed_bytes,
            "compression_ratio": f"{compression_ratio:.1%}",
            "llm_compress_triggered": self.llm_compress_triggered,
            "filter_hits": dict(self.filter_hits),
            "filter_compression": {
                k: dict(v) for k, v in self.filter_compression.items()
            },
        }


# ============================================================
# Base Filter Class
# ============================================================


class BaseFilter(ABC):
    """
    Abstract base class for all filters

    To create a new filter:
    1. Inherit from BaseFilter
    2. Set name and priority
    3. Implement filter() method
    4. Optionally override should_skip()
    """

    # Filter name (used for enable/disable control)
    name: str = "base"

    # Priority (lower = executed first)
    priority: int = 50

    # Whether this filter suggests LLM compression after processing
    suggests_llm_compress: bool = False

    # Suggested LLM prompt type (if suggests_llm_compress is True)
    llm_compress_type: Optional[str] = None

    @abstractmethod
    def filter(self, text: str, context: FilterContext) -> str:
        """
        Execute filtering

        Args:
            text: Input text
            context: Filter context (can be modified to pass info to later filters)

        Returns:
            Filtered text
        """
        pass

    def should_skip(self, context: FilterContext) -> bool:
        """
        Whether to skip this filter (subclasses can override)

        Args:
            context: Filter context

        Returns:
            True to skip this filter
        """
        return False


# ============================================================
# Concrete Filter Implementations
# ============================================================


class AnsiCleanFilter(BaseFilter):
    """
    ANSI escape code cleaner - Always executed first

    Removes:
    - Color codes (\x1b[32m etc.)
    - Cursor movement codes
    - Terminal title codes
    """

    name = "ansi"
    priority = 10  # Highest priority

    def filter(self, text: str, context: FilterContext) -> str:
        result = text
        # Color and style codes
        result = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result)
        # Terminal title codes
        result = re.sub(r"\x1b\].*?\x07", "", result)
        # Carriage return cleanup (progress bar overwrite)
        result = re.sub(r"\r(?!\n)", "\n", result)
        # Compress multiple blank lines
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result


class SystemBannerFilter(BaseFilter):
    """
    System banner and welcome message cleaner

    Removes:
    - Ubuntu welcome message
    - MOTD
    - Last login info
    """

    name = "banner"
    priority = 15

    def filter(self, text: str, context: FilterContext) -> str:
        result = text
        safe_banners = [
            r"Welcome to Ubuntu.*?\n",
            r"\* Documentation:.*?\n",
            r"\* Management:.*?\n",
            r"\* Support:.*?\n",
            r"System information as of.*?\n",
            r"Last login:.*?\n",
        ]
        for pattern in safe_banners:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        return result


class PollingFilter(BaseFilter):
    """
    Polling state handler

    When in POLLING state (consecutive empty commands),
    extracts and summarizes progress information.
    """

    name = "polling"
    priority = 20  # High priority, can short-circuit

    def should_skip(self, context: FilterContext) -> bool:
        # Only process in POLLING state without errors
        return context.state != FilterState.POLLING or context.has_error

    def _extract_progress(
        self, output: str, last_command: Optional[str]
    ) -> Optional[str]:
        """Extract progress information from output"""
        tail = output[-1000:] if len(output) > 1000 else output

        extracted = []

        # Percentage progress
        match = re.search(r"(\d{1,3})%", tail)
        if match:
            extracted.append(f"Progress: {match.group(1)}%")

        # pip/apt download progress
        match = re.search(r"Downloading\s+(\S+)", tail)
        if match:
            pkg = match.group(1)[:30]
            extracted.append(f"Downloading {pkg}")

        # apt status
        if "Reading package lists" in tail:
            extracted.append("Reading package lists")
        elif "Building dependency tree" in tail:
            extracted.append("Building dependency tree")
        elif re.search(r"Setting up\s+(\S+)", tail):
            match = re.search(r"Setting up\s+(\S+)", tail)
            extracted.append(f"Setting up {match.group(1)}")
        elif re.search(r"Unpacking\s+(\S+)", tail):
            match = re.search(r"Unpacking\s+(\S+)", tail)
            extracted.append(f"Unpacking {match.group(1)}")

        if extracted:
            return "; ".join(extracted)

        return None

    def filter(self, text: str, context: FilterContext) -> str:
        progress_info = self._extract_progress(text, context.last_command)

        if progress_info:
            cmd_desc = (
                context.last_command[:50]
                if context.last_command
                else "previous command"
            )
            if len(cmd_desc) == 50:
                cmd_desc += "..."

            context.is_redundant = True
            context.short_circuit = True  # Skip remaining filters

            return f"[WAITING] {cmd_desc}\nCurrent status: {progress_info}"

        # Cannot determine progress, don't compress
        return text


# ============================================================
# Main Filter Orchestrator
# ============================================================


class SafeOutputFilter:
    """
    Safe Output Filter - Main orchestrator using filter chain

    Design Principles:
    1. Safety first - Don't compress when uncertain
    2. Only compress clearly safe patterns
    3. Always preserve error messages
    4. Always preserve content that may affect decisions

    Usage:
        # Default configuration
        filter = SafeOutputFilter()

        # Custom configuration
        filter = SafeOutputFilter(FilterConfig(
            always_llm_compress=True,
            llm_compress_errors=True,
        ))

        # Process output
        result = filter.process(terminal_output, current_command)
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self.stats = FilterStats()

        # State machine
        self.consecutive_empty_commands = 0
        self.last_command: Optional[str] = None
        self.state = FilterState.IDLE

        # Initialize filter chain (sorted by priority)
        # Baseline filters only; command-specific filters injected via add_filter()
        self.filters: list[BaseFilter] = sorted(
            [
                AnsiCleanFilter(),
                SystemBannerFilter(),
                PollingFilter(),
            ],
            key=lambda f: f.priority,
        )

    def add_filter(self, filter_instance: BaseFilter) -> None:
        """
        Add a custom filter to the chain

        Args:
            filter_instance: Instance of a BaseFilter subclass
        """
        self.filters.append(filter_instance)
        self.filters.sort(key=lambda f: f.priority)

    def remove_filter(self, filter_name: str) -> bool:
        """
        Remove a filter from the chain by name

        Args:
            filter_name: Name of the filter to remove

        Returns:
            True if filter was found and removed
        """
        original_len = len(self.filters)
        self.filters = [f for f in self.filters if f.name != filter_name]
        return len(self.filters) < original_len

    def process(self, terminal_output: str, current_command: str) -> FilterResult:
        """
        Process terminal output through the filter chain

        Args:
            terminal_output: Raw terminal output
            current_command: Current command being executed

        Returns:
            FilterResult with filtered output and metadata
        """
        original_length = len(terminal_output)

        # Update state machine
        self._update_state(current_command)

        # Detect errors
        has_error = self._has_error(terminal_output)

        # Create filter context
        context = FilterContext(
            command=current_command,
            has_error=has_error,
            state=self.state,
            last_command=self.last_command,
            original_length=original_length,
        )

        # Run filter chain
        current_text = terminal_output
        applied_filters = []

        for f in self.filters:
            # Check if filter is enabled
            if not self.config.is_filter_enabled(f.name):
                continue

            # Check if filter should be skipped
            if f.should_skip(context):
                continue

            # Record before length for stats
            before_len = len(current_text)

            # Apply filter
            current_text = f.filter(current_text, context)

            # Record stats
            after_len = len(current_text)
            if after_len != before_len:
                applied_filters.append(f.name)
                if self.config.debug:
                    self.stats.record_filter_compression(f.name, before_len, after_len)

            # Check for short-circuit
            if context.short_circuit:
                break

        # If output is short, skip further processing
        if len(current_text) < self.config.min_length_to_filter:
            result = FilterResult(
                output=current_text,
                need_llm_compress=False,
                status="failed" if has_error else "success",
                original_length=original_length,
                compressed_length=len(current_text),
                compression_applied=(len(current_text) < original_length),
                applied_filters=applied_filters,
            )
            self._record_stats(result)
            return result

        # Determine if LLM compression is needed
        need_llm = self._should_trigger_llm_compress(
            current_text, context, has_error, applied_filters
        )

        # Determine LLM prompt type
        llm_type = self._determine_llm_type(context, has_error)

        # Determine status
        status = self._detect_status(current_text, has_error)

        # Build result
        result = FilterResult(
            output=current_text,
            need_llm_compress=need_llm,
            llm_compress_type=llm_type,
            is_redundant=context.is_redundant,
            status=status,
            compression_applied=(len(current_text) < original_length),
            original_length=original_length,
            compressed_length=len(current_text),
            applied_filters=applied_filters,
        )

        # Add debug info if enabled
        if self.config.debug:
            result.debug_info = {
                "context": {
                    "is_package_install": context.is_package_install,
                    "suggested_llm_type": context.suggested_llm_type,
                    "state": self.state.value,
                },
                "filter_stats": context.filter_stats,
            }

        self._record_stats(result)
        return result

    def _update_state(self, current_command: str) -> None:
        """Update state machine"""
        is_empty = not current_command.strip()

        if is_empty:
            self.consecutive_empty_commands += 1
            if self.consecutive_empty_commands >= self.config.polling_trigger_count:
                self.state = FilterState.POLLING
            elif self.consecutive_empty_commands == 1:
                self.state = FilterState.WAITING
        else:
            self.consecutive_empty_commands = 0
            self.state = FilterState.IDLE
            self.last_command = current_command

    def _has_error(self, output: str) -> bool:
        """Detect if output contains error messages"""
        error_patterns = [
            r"\bError\b",
            r"\bERROR\b",
            r"\berror:",
            r"\bFailed\b",
            r"\bFAILED\b",
            r"\bfailed\b",
            r"\bException\b",
            r"\bTraceback\b",
            r"command not found",
            r"No such file or directory",
            r"Permission denied",
            r"\bfatal:",
            r"\bE:",  # apt errors
            r"ModuleNotFoundError",
            r"ImportError",
            r"SyntaxError",
            r"NameError",
            r"TypeError",
            r"ValueError",
            r"KeyError",
            r"AttributeError",
            r"RuntimeError",
        ]
        for pattern in error_patterns:
            if re.search(pattern, output):
                return True
        return False

    def _should_trigger_llm_compress(
        self,
        text: str,
        context: FilterContext,
        has_error: bool,
        applied_filters: List[str],
    ) -> bool:
        """Determine if LLM compression should be triggered"""
        # NOTE: Analysis command detection removed - now relying on LLM with task context
        # to make intelligent decisions about what to preserve.

        # If always_llm_compress is True, always trigger (respecting threshold)
        if self.config.always_llm_compress:
            if has_error and not self.config.llm_compress_errors:
                return False
            return len(text) > self.config.min_length_to_filter

        # Handle errors
        if has_error:
            return (
                self.config.llm_compress_errors
                and len(text) > self.config.llm_compress_threshold
            )

        # Package install output - lower threshold
        if context.is_package_install:
            return len(text) > self.config.package_install_llm_threshold

        # Check if any filter suggests LLM compression
        for f in self.filters:
            if f.name in applied_filters and f.suggests_llm_compress:
                threshold = (
                    self.config.package_install_llm_threshold
                    if f.llm_compress_type == "install"
                    else self.config.llm_compress_threshold
                )
                if len(text) > threshold:
                    return True

        # Default: trigger if output is long and rules didn't compress much
        return len(text) > self.config.llm_compress_threshold

    def _determine_llm_type(self, context: FilterContext, has_error: bool) -> str:
        """Determine which LLM prompt type to use"""
        if has_error:
            return "error"
        if context.suggested_llm_type:
            return context.suggested_llm_type
        if context.is_package_install:
            return "install"
        if context.state == FilterState.POLLING:
            return "running"
        return "default"

    def _detect_status(self, output: str, has_error: bool) -> str:
        """Detect command execution status"""
        if has_error:
            return "failed"

        # Running detection
        running_patterns = [r"\d+%", r"\.\.\."]
        completion_patterns = [
            r"Successfully installed",
            r"done\.",
            r"Done\.",
            r"DONE",
            r"complete",
            r"Complete",
            r"completed",
            r"Completed",
            r"finished",
            r"Finished",
        ]

        tail = output[-500:] if len(output) > 500 else output
        has_running = any(re.search(p, tail) for p in running_patterns)
        has_completion = any(re.search(p, output) for p in completion_patterns)

        if has_running and not has_completion:
            return "running"

        return "success"

    def _record_stats(self, result: FilterResult) -> None:
        """Record statistics for a filtering operation"""
        self.stats.record_call(
            result.original_length,
            result.compressed_length,
            result.applied_filters,
            result.need_llm_compress,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics summary"""
        return self.stats.get_summary()

    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = FilterStats()

    def reset(self) -> None:
        """Reset state machine"""
        self.consecutive_empty_commands = 0
        self.last_command = None
        self.state = FilterState.IDLE


# ============================================================
# Backward Compatibility
# ============================================================

# Keep old name for backward compatibility
OutputFilter = SafeOutputFilter
