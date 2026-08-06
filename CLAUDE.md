# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Lint (Black + ruff) — run before every commit
./lint.sh

# Run any script
uv run python scripts/main_sim.py ...

# Add a dependency
uv add package_name
```

## Code style

- **Formatter**: Black, 120-char line length (`./lint.sh` runs Black + ruff)
- **Strict inputs**: code must throw errors if the input is invalid — never replace with default values or use a fallback. This is research code; silent failures mask bugs and cause unintended behavior.
  - If args/fields/format do not match what the caller is required to provide, **fail immediately** with a clear error. Do not invent a "backup path" (e.g. reading missing CLI flags from a sidecar JSON, guessing `sim_backend`, coercing shapes, substituting defaults).
  - Required CLI parameters must be explicit. Do not make a required option optional by supplying an implicit default in the script.
  - Do not add "helpful" recovery, probing, or multi-source resolution unless the user explicitly asks for it. Prefer requiring the expected inputs over making the code robust to incomplete ones.
  - Example of what **not** to do: `if args.obs_mode is None: obs_mode = json_data[...]` — instead raise `ValueError("obs_mode (-o) is required")`.
- **Assertions**: prefer `assert condition, "clear message"` over `if not condition: raise ...` for input validation and invariants.
- **Error surfacing**: do not use `try`/`finally`. Let errors surface directly instead of adding defensive cleanup machinery around failing paths.
- **CLI scripts**: the `if __name__ == "__main__":` block must only parse arguments with argparse and call `main()`. All logic belongs inside `main()`.
- **Helper functions**: prefer module-level helpers for distinct logical steps in CLI scripts, especially when they are reused or make `main()` easier to read. Do not create tiny one-off helpers for trivial expressions.
  - If a helper is used **2 or more times** in a file, define it at module scope.
  - Use nested (local) functions only when the function is called **once** and requires access to enclosing-scope variables. Never use local functions just to avoid giving something a name.
