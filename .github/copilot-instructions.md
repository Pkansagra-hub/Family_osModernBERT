# GitHub Copilot Instructions for Modeling Studio

## PRODUCTION CODE STANDARDS

This is **production code**. Follow these rules strictly:

1. **NO `await` usage** - This codebase does not use async/await patterns
2. **NO emojis in code or comments** - Keep output professional and clean
3. **NO decorative characters** - Use plain text for logging and messages

## WHEN UNCERTAIN - ASK, DON'T GUESS

If you are uncertain about:
- The intended behavior or design pattern
- Which file to modify
- The correct approach to a problem
- Whether a change might break existing functionality

**ASK THE USER FOR CLARIFICATION** instead of guessing. Wrong assumptions waste time.

## Python Best Practices

1. **Type hints** - Use type annotations for all function parameters and return values
2. **Docstrings** - All public functions/classes must have docstrings (Google style)
3. **Error handling** - Use specific exceptions, not bare `except:`
4. **Imports** - Use absolute imports, group by stdlib/third-party/local
5. **Constants** - Use UPPER_CASE for module-level constants
6. **No magic numbers** - Define named constants for numeric values
7. **Single responsibility** - Functions should do one thing well
8. **DRY principle** - Don't repeat yourself; extract common logic
9. **Explicit over implicit** - Be clear about what code does
10. **Test coverage** - New code should have corresponding tests

## Terminal Commands - CRITICAL RULES

### DO NOT USE PowerShell Output Piping with `2>&1 | Select-Object`

**NEVER** use patterns like:
```powershell
# ❌ BAD - DO NOT USE
command 2>&1 | Select-Object -Last 10
command 2>&1 | Select-Object -First 20
```

**Why this is forbidden:**
1. It breaks error handling and exit codes
2. It can truncate critical error messages
3. It causes unpredictable behavior with stderr redirection
4. It makes debugging significantly harder

**Instead, run commands directly:**
```powershell
# ✅ GOOD - Use direct command execution
command
pytest tests/v3/test_something.py
python script.py
```

If output is too long, use proper pytest flags:
```powershell
# ✅ GOOD - Use tool-specific options to limit output
pytest tests/v3/test_something.py -q
pytest tests/v3/test_something.py --tb=short
pytest tests/v3/test_something.py -x  # Stop on first failure
```

## General Guidelines

1. **Run tests using the `runTests` tool** when available, not terminal commands
2. **Use absolute paths** for all file operations
3. **Prefer Python's native test runners** over shell piping
4. **Keep terminal commands simple** - avoid complex pipelines
