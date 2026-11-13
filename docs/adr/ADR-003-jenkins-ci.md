# ADR-003 — Jenkins instead of GitHub Actions

**Status:** accepted · **Date:** 2026-11-13

## Context

GitHub Actions minutes are not available for this project. CI must run on a self-hosted
worker that may have only 16 GB of RAM and no GPU.

## Decision

Jenkins (declarative `Jenkinsfile`) is the official CI; no active `.github/workflows/`
exist. The pipeline runs entirely on CPU with `disableConcurrentBuilds()` and
`CUDA_VISIBLE_DEVICES=''`, with stages for lock validation, formatting, Ruff, strict mypy,
unit / property / integration tests, mock-serving and gateway end-to-end tests, a
benchmark smoke test (including the quick CPU demo), security scans, the Docker image
(when a daemon is available) and the Python package. JUnit, coverage and small benchmark
summaries are archived; model weights never are. `scripts/dev/check.sh` runs the same gates
locally.

## Consequences

* Full control of the worker, no minute limits, identical commands locally and in CI.
* The worker must be maintained (docs/ci/jenkins.md).
* GPU validation is a separate, manual process on rented machines (ADR-004,
  docs/cloud/gpu-runbook.md).
