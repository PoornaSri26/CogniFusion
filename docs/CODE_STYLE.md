# CODE_STYLE - CogniFusion

## Python
- snake_case functions/vars; PascalCase classes; UPPER_SNAKE for constants
- Type hints on public function signatures; docstrings on classes and non-obvious functions
- Compact code: no dead branches, no commented-out experiments, no print-debug left in
- Errors: handle at boundaries (file IO, hardware), let math errors surface
- numpy/torch tensor shape comments where non-obvious (B,T,F)

## Notebook
- Section markdown before each code cell; code cells self-contained but reuse earlier vars deliberately
- Print summary stats, not raw dumps; display() DataFrames
- All `!` commands single-line (no backslash continuation - Colab-fragile)

## Git
- Conventional-ish messages: `Add X`, `Fix Y`, lowercase imperative
- Author: Poorna Sri <poornasri.n24@gmail.com>
- Small focused commits; verify (compile/train smoke) before push

## Verification before done
`python -m compileall src data dashboard.py` + one short training run for train.py changes + notebook compile check for notebook changes.

---
## For AI agents
Follow these conventions. Run compile checks before finishing. No dead code, no commented experiments, no `!`-line continuations in notebook cells.
