# Jenkins CI

Jenkins is the official CI for InferenceLab (ADR-003). The pipeline is defined in the
repository's `Jenkinsfile` (declarative). It needs **no GPU, no CUDA, no cloud
credentials and no model downloads**.

## Agent requirements

| Requirement | Notes |
|---|---|
| Linux agent labelled `linux` | 4+ cores, 8–16 GB RAM is enough; `disableConcurrentBuilds()` keeps one build at a time |
| `git`, `curl`, `bash` | standard |
| `uv` | installed automatically into `~/.local/bin` on first run if missing; Python 3.12 is fetched by uv |
| Docker (optional) | for the image build stage; the stage is skipped with a message when no daemon is available |
| `gitleaks` (optional) | history secret scan with the repository's `.gitleaks.toml`; `detect-secrets` always runs |
| Plugins | Pipeline, Git, JUnit, Timestamper, Workspace Cleanup; *Coverage* plugin optional (coverage XML is archived either way) |

The agent user must be able to talk to Docker (member of the `docker` group, or rootless
Docker) if the image stage should run. Do not give the agent cloud or GPU credentials.

Without a local `gitleaks` binary, the history scan can run from the official image:

```bash
docker run --rm -v "$PWD:/repo:ro" -e GIT_CONFIG_COUNT=1 \
  -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0='*' \
  ghcr.io/gitleaks/gitleaks:v8.28.0 git --no-banner --redact /repo
```

## Job setup

1. *New Item → Multibranch Pipeline* (or *Pipeline from SCM*) pointing at the repository.
2. Credentials: a read-only deploy key or token for private clones; public clones need none.
3. Build configuration: *by Jenkinsfile*, script path `Jenkinsfile`.
4. Parameters (defaults are fine): `RUN_DOCKER_BUILD`, `RUN_DEMO`.

## Triggers

* **Webhook** (preferred): GitHub → *Settings → Webhooks* → `https://<jenkins>/github-webhook/`,
  content type JSON, "push" and "pull request" events; enable *GitHub hook trigger for
  GITScm polling* (or the multibranch equivalent).
* **Polling** fallback: `H/5 * * * *` if Jenkins is not reachable from GitHub.
* **Manual**: *Build with Parameters*.

## Stages

| Stage | Command (all via `uv run --frozen`) |
|---|---|
| Checkout | `checkout scm` |
| Environment Validation | uv / Python 3.12 present; CPU and memory printed |
| Dependency Lock Validation | `uv lock --check`, `uv sync --frozen --all-extras` |
| Format | `ruff format --check .` |
| Ruff | `ruff check .` (JUnit report) |
| Strict mypy | `mypy` (strict, JUnit report) |
| Unit / Property / Integration Tests | `pytest tests/<suite>` with coverage appended (`HYPOTHESIS_PROFILE=ci`: 200 derandomized examples) |
| Mock Serving E2E | `pytest tests/e2e/test_mock_serving.py` |
| Gateway E2E | `pytest tests/e2e/test_gateway_e2e.py tests/e2e/test_cli.py`; coverage XML; `--fail-under=85` |
| Benchmark Smoke Test | virtual-time simulated benchmark, local prefix-cache experiment, quick CPU demo |
| Security Scan | `scripts/ci/secret_scan.sh` (detect-secrets baseline + optional gitleaks), `bandit -r src` |
| Docker Build | `docker build`, `--version` smoke, non-root check (skipped without a daemon) |
| Python Package Build | `uv build`; wheel contents listed |

## Artifacts and retention

Archived: JUnit XML, `coverage.xml`, `bandit.xml`, the wheel and sdist, benchmark
`summary.json`/`report.md` and the demo report with plots — all small. Never archived:
model weights, `requests.parquet` dumps, caches, `.env` files. `buildDiscarder` keeps 30
builds and artifacts of the last 10.

## Running the same gates locally

```bash
bash scripts/dev/check.sh
```
