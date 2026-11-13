# Contributing

## Setup

```bash
uv sync --all-extras
uv run pre-commit install
```

## Quality gates (same as Jenkins)

```bash
bash scripts/dev/check.sh
```

or individually:

```bash
uv run ruff format --check . && uv run ruff check .
uv run mypy                         # strict
uv run pytest -q                    # unit, property, integration, e2e (CPU only)
bash scripts/ci/secret_scan.sh
uv run bandit -c pyproject.toml -r src -q
uv build
```

## Ground rules

* **No GPU in tests.** Use `MockBackend`, `SimulatedInferenceBackend` and `VirtualClock`.
  Tests must not sleep on wall-clock time to observe behaviour; use the injected clock.
* **Never fabricate.** Metrics an engine does not expose stay `None`; estimated token counts
  are labelled; simulated results carry the SIMULATED label; percentiles follow the
  sample-size rule.
* **No prices in code or configs.** Prices are experiment inputs with a source and date.
* **No secrets, weights or request dumps in git.** API keys come from environment variable
  *names*; `results/` is ignored; curated reports go to `reports/`.
* **Pin versions.** Engine images by tag (+ digest), models by commit SHA. Re-verify engine
  flags against current official docs before changing a launch builder, and record the
  version in `backends/launch.py:ENGINE_VERSIONS` and `docs/backends/`.
* Keep the data plane free of control-plane imports, and keep FastAPI out of domain logic.
* Add or update an ADR in `docs/adr/` for architectural decisions.

## Commits

Conventional-style messages (`feat(routing): …`, `fix(gateway): …`, `docs: …`), small and
logical. Stage files explicitly and review `git diff --cached` before committing.
