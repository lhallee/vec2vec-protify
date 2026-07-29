# Protify

## Purpose and Sources

This is the standalone Synthyra Protify checkout. Keep its public training, evaluation, packaging, and cloud interfaces independent of the parent synth workspace.

- `docs/getting_started.md` and `docs/cli_and_config.md`: entrypoints and configuration
- `docs/probes_and_training.md`: probe behavior
- `docs/testing.md`: test scopes and working directories
- `src/protify/base_models/supported_models.py`: authoritative model registry
- `src/protify/data/supported_datasets.py`: authoritative dataset registry

## Architectural Invariants

- CLI, YAML, GUI, and cloud dispatch share the same `MainProcess` pipeline.
- Keep heavy model and dataset loading lazy where the registries and factories are lazy.
- `src/protify/fastplms/` is a vendored repository with its own guidance. Do not mix FastPLMs changes into a Protify task.
- `--parallel_probe_runs` applies only to compatible pooled, sequence-level linear probes. Matrix or tokenwise probes, transformer probes, Lyra probes, PPI run-specific datasets, and full fine-tuning use the sequential fallback.

## Python Authoring Standard

Write first-party Python in Logan's direct, readable, team-oriented style. Treat cleanup as behavior-preserving unless the task explicitly changes behavior. Preserve exceptions, CLI output, serialization, random-number use, dtype, device, tensor shape, import side effects, and documented performance guarantees.

- Prefer visible data flow, domain language, and narrow typed functions over cleverness, speculative abstractions, compatibility shims, or helpers that only rename one call.
- Keep a module docstring first and `from __future__` immediately after it. Put every direct `import` before ordinary `from` imports. Within those blocks, order standard-library, third-party, then repository-local imports. Do not move guarded or initialization-sensitive imports across their barriers.
- Use precise parameterized types and concrete domain names. Use `Any` only at genuinely dynamic boundaries. Do not change framework-discovered fields or serialization schemas merely to improve annotations.
- Use ordinary exceptions for invalid external input and assertions for internal invariants. Avoid broad `except Exception`, silent fallback chains, and unsupported compatibility branches.
- Comment intent, biological conventions, units, assumptions, and non-obvious mechanics. Remove comments and docstrings that narrate syntax, preserve debugging history, or repeat clear names and types.
- For NumPy, PyTorch, and similar numerical code, maintain complete shape traces. Use `b` for batch, `l` for sequence length, `d` for hidden width, `h` for heads, `c` for classes, and `n` for a generic count. Annotate inputs, transformations, reductions, broadcasting, and returned tensors without inventing unsupported dimensions.
- Keep modules cohesive and entry points thin. Split by ownership and dependency direction, not line count. Extract a helper only when it names a real concept or isolates a separately testable phase.
- Exclude generated and vendored code from style passes unless the task explicitly includes it. In particular, do not style-edit `src/protify/fastplms/` as part of a Protify task.

For repository-scale Python cleanup, inspect every eligible first-party module and classify it as compliant, mechanical edit, structural edit, excluded, or blocked. Run the smallest relevant CPU baseline before editing, rerun it afterward, review the final diff for behavior drift, and report the coverage matrix. Do not apply repository-wide formatting churn merely for uniformity.

## Canonical Commands

Build and run the broad CPU suite from the repository root:

```powershell
docker build -t protify-env:latest .
docker run --rm --ipc=host -v ${PWD}:/workspace -w /workspace protify-env:latest \
  python -m pytest src/protify/testing_suite -v -m "not gpu and not slow"
```

Focused parallel-probe tests run with `src/protify/` as the working directory:

```powershell
docker run --rm --gpus all --ipc=host -v ${PWD}:/workspace -e PYTHONPATH=/workspace \
  -w /workspace/src/protify protify-env:latest python -m pytest \
  testing_suite/test_parallel_probe_plan.py testing_suite/test_parallel_linear_probe.py -v
```
