@AGENTS.md

## Python Code Preferences

The `Python Authoring Standard` in `AGENTS.md` is mandatory for all first-party Python changes. In particular:

- preserve behavior unless a behavior change is explicitly requested;
- prefer direct, typed, domain-specific functions and visible control flow;
- order direct imports before `from` imports, with standard-library, third-party, and repository-local groups;
- avoid speculative abstractions, broad exception handling, silent fallbacks, and narrative comments;
- maintain complete, accurate NumPy and PyTorch shape traces using `b`, `l`, `d`, `h`, `c`, and `n` conventions;
- keep entry points thin and modules cohesive; never weaken tests, and limit test edits to scoped mechanical cleanup or changes required by a real interface, module boundary, configuration, or workflow change;
- exclude vendored `src/protify/fastplms/` code unless the task explicitly includes it;
- establish a CPU test baseline before structural cleanup and compare the same tests afterward.
