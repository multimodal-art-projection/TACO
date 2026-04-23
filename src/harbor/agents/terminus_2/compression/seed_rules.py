"""Seed compression rules — converted from the 6 static filters removed from SafeOutputFilter.

These rules serve as the initial rule cache for cold-start tasks.
When no cached rules exist for a task category, the Planner loads these
as a baseline, then lets the LLM select/modify/supplement as needed.

Unlike the old static filters, seed rules:
- Are subject to LLM selection (not always applied)
- Can be evolved via feedback (spawn replacements)
- Get persisted to cache after the first task
"""

from .models import CompressionRule


# --- Seed Rule Definitions ---

SEED_GIT_NOISE = CompressionRule(
    rule_id="seed_git_noise",
    trigger_regex=r"\bgit\b\s+(clone|fetch|pull|push|checkout|submodule)",
    description=(
        "Removes git transfer progress lines (Counting/Compressing/Receiving/"
        "Resolving objects) and remote enumeration noise. Compresses hint blocks."
    ),
    keep_patterns=[
        r"\bfatal:",
        r"\berror:",
        r"\bwarning:",
        r"Already up to date",
        r"Cloning into",
        r"branch\s+\S+\s+->",
    ],
    strip_patterns=[
        r"Enumerating objects: \d+, done\.",
        r"Counting objects: +\d+% \(\d+/\d+\)",
        r"Counting objects: \d+, done\.",
        r"Compressing objects: +\d+% \(\d+/\d+\)",
        r"Compressing objects: \d+, done\.",
        r"Receiving objects: +\d+% \(\d+/\d+\)",
        r"Receiving objects: \d+, done\.",
        r"Resolving deltas: +\d+% \(\d+/\d+\)",
        r"Resolving deltas: \d+, done\.",
        r"remote: Enumerating objects:",
        r"remote: Counting objects:",
        r"remote: Compressing objects:",
        r"remote: Total \d+",
        r"^hint: ",
    ],
    keep_first_n=3,
    keep_last_n=5,
    max_lines=None,
    summary_header="[git output compressed — transfer progress removed]",
    priority=30,
    confidence=0.8,
    times_applied=10,
)

SEED_HEREDOC = CompressionRule(
    rule_id="seed_heredoc",
    trigger_regex=r"(cat|tee)\s+.*<<\s*['\"]?\w+['\"]?|cat\s+>\s*\S+\s*<<",
    description=(
        "Detects heredoc commands (cat > file << EOF) and compresses the echoed "
        "lines. Terminal echoes every line back during heredoc writes; these are "
        "pure noise once the write succeeds. Errors are preserved."
    ),
    keep_patterns=[
        r"^bash: ",
        r"^sh: ",
        r": event not found",
        r": command not found",
        r"syntax error",
        r": No such file or directory",
    ],
    strip_patterns=[
        r"^> ",
    ],
    keep_first_n=2,
    keep_last_n=3,
    max_lines=10,
    summary_header="[heredoc echo compressed]",
    priority=35,
    confidence=0.8,
    times_applied=10,
)

SEED_PIP_INSTALL = CompressionRule(
    rule_id="seed_pip_install",
    trigger_regex=r"\bpip3?\s+install\b",
    description=(
        "Compresses pip install output: removes Collecting/Downloading/"
        "Requirement-already-satisfied lines while preserving errors and "
        "the final 'Successfully installed' summary."
    ),
    keep_patterns=[
        r"\bERROR:",
        r"\berror:",
        r"Successfully installed",
        r"Could not",
        r"Traceback",
        r"WARNING:",
    ],
    strip_patterns=[
        r"^\s*Collecting \S+",
        r"^\s*Downloading \S+",
        r"^\s*Requirement already satisfied",
        r"^\s*Using cached",
        r"^\s*Installing collected packages",
    ],
    keep_first_n=3,
    keep_last_n=5,
    max_lines=None,
    summary_header="[pip install output compressed]",
    priority=50,
    confidence=0.8,
    times_applied=10,
)

SEED_APT_INSTALL = CompressionRule(
    rule_id="seed_apt_install",
    trigger_regex=r"\bapt(?:-get)?\s+(install|update|upgrade)\b",
    description=(
        "Compresses apt/apt-get output: removes repetitive Setting-up, "
        "Unpacking, and Get: download lines while keeping first/last few "
        "and preserving errors."
    ),
    keep_patterns=[
        r"\bE:",
        r"\bErr:",
        r"\berror:",
        r"dpkg: error",
        r"Unable to locate package",
        r"Reading package lists",
        r"Building dependency tree",
    ],
    strip_patterns=[
        r"^Setting up \S+",
        r"^Unpacking \S+",
        r"^Get:\d+",
        r"^Preparing to unpack",
        r"^Selecting previously unselected package",
    ],
    keep_first_n=2,
    keep_last_n=2,
    max_lines=None,
    summary_header="[apt install output compressed]",
    priority=55,
    confidence=0.8,
    times_applied=10,
)

SEED_COMPILER_OUTPUT = CompressionRule(
    rule_id="seed_compiler_output",
    trigger_regex=r"\b(gcc|g\+\+|clang|cc|make|cmake)\b",
    description=(
        "Truncates long gcc/g++/clang/cc compiler command lines (>200 chars) "
        "that clutter the output. Preserves error and warning messages."
    ),
    keep_patterns=[
        r"\berror:",
        r"\bwarning:",
        r"\bundefined reference",
        r": fatal error:",
        r"make\[\d+\]: \*\*\*",
        r"^ld:",
    ],
    strip_patterns=[
        r"^\s*(gcc|g\+\+|clang|cc)\s+.{200,}",
    ],
    keep_first_n=5,
    keep_last_n=10,
    max_lines=30,
    summary_header="[compiler output compressed — long command lines truncated]",
    priority=60,
    confidence=0.8,
    times_applied=10,
)

SEED_OPENSSL = CompressionRule(
    rule_id="seed_openssl",
    trigger_regex=r"\b(openssl|ssh-keygen|gpg)\b",
    description=(
        "Removes OpenSSL/SSH key generation dot-progress noise. These are long "
        "strings of dots and plus signs emitted during random number generation."
    ),
    keep_patterns=[
        r"\berror:",
        r"unable to",
        r"Generating",
        r"Your identification has been saved",
        r"The key fingerprint is",
    ],
    strip_patterns=[
        r"[.+]{20,}",
    ],
    keep_first_n=5,
    keep_last_n=5,
    max_lines=None,
    summary_header="[key generation progress compressed]",
    priority=65,
    confidence=0.8,
    times_applied=10,
)

# All seed rules collected
_SEED_RULES: list[CompressionRule] = [
    SEED_GIT_NOISE,
    SEED_HEREDOC,
    SEED_PIP_INSTALL,
    SEED_APT_INSTALL,
    SEED_COMPILER_OUTPUT,
    SEED_OPENSSL,
]


def get_seed_rules() -> list[CompressionRule]:
    """Return deep copies of seed rules so callers can modify without affecting originals."""
    return [rule.model_copy(deep=True) for rule in _SEED_RULES]
