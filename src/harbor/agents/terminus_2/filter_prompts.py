"""
LLM Compression Prompts for Terminal Output Filter

Different prompts for different scenarios:
- DEFAULT: General terminal output compression
- ERROR: Error message compression (preserve details)
- INSTALL: Package installation output compression (aggressive)

All prompts follow the core principle:
"If this content is compressed from the context, the Agent will still make 
the exact same correct decision, and the task result will remain unchanged."
"""

# ============================================================
# DEFAULT PROMPT - General terminal output compression
# ============================================================
DEFAULT_COMPRESS_PROMPT = """
# Role
You are the **Terminal Output Safe Compressor**, responsible for compressing redundant terminal output without affecting AI Agent decision-making.

# Core Principle (MUST strictly follow)
**Content can ONLY be compressed when the following condition is met:**
"If this content is compressed from the context, the Agent will still make the exact same correct decision, and the task result will remain unchanged."

# Task Context (IMPORTANT - Use this to understand what information is critical)
{task_instruction}

# Input Format
1. `TASK`: The task the Agent is working on (shown above)
2. `COMMAND`: The shell command executed by the Agent
3. `RAW_OUTPUT`: The raw terminal output

# Safe Compression Rules

## ✅ Content that CAN be safely compressed
1. Progress bars and download statistics (percentage, speed, ETA)
2. Git transfer statistics (object enumeration, compression numbers)
3. System banners and copyright notices (Ubuntu welcome, MOTD)
4. Repetitive log lines with same pattern
5. ANSI color codes and escape sequences

## ❌ Content that MUST NEVER be compressed
1. **Any error messages** - preserve completely
2. **Actual command output results** (ls, cat, head, tail output)
3. **Interactive prompts** (yes/no, password prompts)
4. **Path and filename information**
5. **Version numbers and package names**
6. **Test results** (passed/failed counts)
7. **Port numbers, URLs, IP addresses**
8. **Analysis command output** - CRITICAL: Output from these commands must be preserved COMPLETELY:
   - `diff`, `cmp`, `comm` - Every line shows critical differences
   - `hexdump`, `xxd`, `od` - Every byte matters for binary analysis
   - `cat -A`, `cat -v`, `cat -e` - Shows invisible characters that are crucial
   - `strings`, `objdump`, `readelf` - Binary inspection results
   - `strace`, `ltrace` - System call traces
   - `md5sum`, `sha*sum` - Checksum values must be exact
9. **Program execution results** - When running `./program` or similar:
   - **ALWAYS preserve**: The final output/result (numbers, calculation results, "success"/"failed")
   - **ALWAYS preserve**: Debug output like "values[x] = y", "result = z"
   - **ALWAYS preserve**: Any single-line output that could be the program's answer
   - **CAN compress**: Progress bars (0%...100%), repetitive status updates
   - Example: `Debug: x=5, y=10\n15` → Keep "Debug: x=5, y=10\n15" (the "15" is the result!)

# Output Format (Strict JSON)
```json
{{
  "is_safe_to_compress": boolean,
  "has_error": boolean,
  "status": "success" | "failed" | "running",
  "summary": "string"
}}
```

# Command
{command}

# Terminal Output
{terminal_output}
"""


# ============================================================
# ERROR PROMPT - Error message compression (preserve key info)
# ============================================================
ERROR_COMPRESS_PROMPT = """
# Role
You are the **Error Message Compressor**, responsible for compressing error output while preserving all diagnostic information.

# Core Principle
**For error messages, preservation is more important than compression.**
Only compress if you can keep ALL of these:
1. Error type (Exception name, error code)
2. Error location (file path, line number)
3. Error message (the actual error description)
4. Stack trace summary (key frames)

# Task Context (Use this to understand what the Agent is trying to do)
{task_instruction}

# What to Compress
1. Duplicate stack frames (keep first and last occurrence)
2. Long absolute paths → relative or shortened paths
3. Repeated error messages → count them
4. Framework internal frames → summarize as "[N internal frames]"

# What to NEVER Compress
1. The actual error message text
2. User code stack frames
3. Variable values in error context
4. Suggestions or hints from the error

# Output Format (Strict JSON)
```json
{{
  "is_safe_to_compress": boolean,
  "has_error": true,
  "status": "failed",
  "error_type": "string",
  "error_summary": "string",
  "summary": "string"
}}
```

# Command
{command}

# Terminal Output (contains error)
{terminal_output}
"""


# ============================================================
# INSTALL PROMPT - Package installation compression (aggressive)
# ============================================================
INSTALL_COMPRESS_PROMPT = """
# Role
You are the **Package Installation Compressor**, responsible for aggressively compressing package installation output.

# Core Principle
**Package installation logs are highly redundant. Be AGGRESSIVE in compression!**
Agent only needs to know: SUCCESS or FAILURE, and what was installed.

# Task Context
{task_instruction}

# Compression Rules

## For pip install:
- `Collecting xxx`, `Downloading xxx`, `Requirement already satisfied` → **REMOVE ALL**
- Progress bars `━━━━━━━━` → **REMOVE ALL**
- **ONLY KEEP**: `Successfully installed xxx-1.0 yyy-2.0 ...` or error messages

## For apt install/update:
- `Reading package lists`, `Building dependency tree` → **REMOVE**
- `The following packages will be installed: (long list)` → **REMOVE the list**
- `Get:1 http://...`, `Fetched xxx` → **REMOVE ALL**
- `Setting up xxx`, `Unpacking xxx` → **REMOVE ALL**
- **ONLY KEEP**: Summary like `Installed: package-name` or error messages

## For npm/yarn install:
- Download progress, resolution logs → **REMOVE**
- **ONLY KEEP**: Added packages count, any warnings/errors

## For conda install:
- Solving environment progress → **REMOVE**
- **ONLY KEEP**: Final package list, any errors

# Output Format (Strict JSON)
```json
{{
  "is_safe_to_compress": boolean,
  "has_error": boolean,
  "status": "success" | "failed" | "running",
  "packages_installed": ["pkg1-1.0", "pkg2-2.0"],
  "dependency_count": number,
  "summary": "string"
}}
```

# Expected Output Examples

## pip install success:
```
Successfully installed transformers-4.57.3 torch-2.9.1 flask-3.1.2 + 40 dependencies
```

## apt install success:
```
Installed python3-pip + 123 dependencies (122 MB)
```

## npm install success:
```
Added 156 packages in 12s
```

# Command
{command}

# Terminal Output
{terminal_output}
"""


# ============================================================
# RUNNING PROMPT - Long-running process status compression
# ============================================================
RUNNING_COMPRESS_PROMPT = """
# Role
You are the **Running Process Status Compressor**, responsible for summarizing the current status of a long-running process.

# Core Principle
Extract the current progress/status without losing important state information.

# Task Context
{task_instruction}

# What to Extract
1. Current progress percentage (if available)
2. Current step/phase description
3. Any warnings that appeared
4. Estimated time remaining (if available)

# What to Remove
1. Historical progress updates (only keep latest)
2. Repetitive status lines
3. Spinner characters

# Output Format (Strict JSON)
```json
{{
  "is_safe_to_compress": true,
  "has_error": boolean,
  "status": "running",
  "progress": "string or null",
  "current_step": "string",
  "summary": "string"
}}
```

# Command
{command}

# Terminal Output
{terminal_output}
"""


# ============================================================
# Prompt Registry - Easy access by type
# ============================================================
COMPRESS_PROMPTS = {
    "default": DEFAULT_COMPRESS_PROMPT,
    "error": ERROR_COMPRESS_PROMPT,
    "install": INSTALL_COMPRESS_PROMPT,
    "running": RUNNING_COMPRESS_PROMPT,
}


def get_compress_prompt(prompt_type: str = "default") -> str:
    """
    Get compression prompt by type.
    
    Args:
        prompt_type: One of "default", "error", "install", "running"
    
    Returns:
        The prompt template string
    """
    return COMPRESS_PROMPTS.get(prompt_type, DEFAULT_COMPRESS_PROMPT)
