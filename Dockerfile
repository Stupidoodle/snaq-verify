# syntax=docker/dockerfile:1
# =============================================================================
# Stage 1 — builder: install Python dependencies into an isolated venv
# =============================================================================
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

WORKDIR /app

# Copy only the files uv needs to resolve and install dependencies.
# Keeping these separate from the source code means this layer is cached
# when only source files change (and vice versa).
COPY pyproject.toml uv.lock ./

# Install production dependencies into /app/.venv with a frozen lock.
# --no-dev:    skip dev tooling (pytest, ruff, mypy) in the runtime image.
# --frozen:    fail if uv.lock is out of date — ensures reproducibility.
# UV_LINK_MODE=copy: avoid hard-link issues when COPY moves the venv later.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --no-dev --no-install-project

# Now copy the project source so the package itself is installed.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --no-dev

# =============================================================================
# Stage 2 — runtime: slim image with only the venv and source
# =============================================================================
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS runtime

WORKDIR /app

# Create a non-root user and group for security.
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup \
               --shell /bin/sh --create-home appuser

# Bring over the fully-installed virtual environment and source tree.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
COPY --from=builder /app/uv.lock /app/uv.lock

# Pre-create the output directory with correct ownership so the container
# can write reports without running as root.
RUN mkdir -p /app/output && chown -R appuser:appgroup /app

USER appuser

# uv run resolves snaq-verify from pyproject.toml [project.scripts].
# --frozen ensures the lock file is not re-evaluated at container start-up.
ENTRYPOINT ["uv", "run", "--frozen", "snaq-verify"]
