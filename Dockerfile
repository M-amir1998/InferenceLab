# syntax=docker/dockerfile:1
# InferenceLab CPU image: gateway, simulated engines, benchmark client and tooling.
# No CUDA, no GPU runtime. GPU engines (vLLM / SGLang / TensorRT-LLM) run in their own
# official, version-pinned images (see deployment/docker/).
ARG PYTHON_IMAGE=python:3.12.16-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.11.21 AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra tracing
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra tracing

FROM ${PYTHON_IMAGE}
LABEL org.opencontainers.image.title="inferencelab" \
      org.opencontainers.image.description="LLM inference gateway, simulated engines and benchmark tooling (CPU)" \
      org.opencontainers.image.source="https://github.com/M-amir1998/InferenceLab" \
      org.opencontainers.image.licenses="Apache-2.0"
RUN groupadd --system --gid 10001 inferencelab \
 && useradd --system --uid 10001 --gid inferencelab --home-dir /app --shell /usr/sbin/nologin inferencelab
COPY --from=builder /opt/venv /opt/venv
COPY --chown=10001:10001 configs /app/configs
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"]
ENTRYPOINT ["inferencelab"]
CMD ["gateway", "serve", "/app/configs/gateway/compose.yaml"]
