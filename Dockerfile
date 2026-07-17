# Multi-step Build
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Omit dev dependencies
ENV UV_NO_DEV=1
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv \
	--mount=type=bind,src=uv.lock,target=uv.lock \
	--mount=type=bind,src=pyproject.toml,target=pyproject.toml \
	uv sync --locked --no-install-project

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
	uv sync --locked

FROM python:3.12-slim-trixie AS runner

RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITECODE=1

COPY --from=builder --chown=nonroot:nonroot /app /app

ENV PATH="/app/.venv/bin:$PATH"

ENV PYTHONUNBUFFERED=1

USER nonroot

WORKDIR /app

EXPOSE 8050

ENTRYPOINT ["python", "-m", "src.app"]
