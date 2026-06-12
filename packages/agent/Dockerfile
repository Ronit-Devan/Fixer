# syntax=docker/dockerfile:1.7

# Multi-stage build for the ET live agent (Product A).
#
# Final image size (uncompressed): ~230-280 MB
#   python:3.12-slim base .................... ~125 MB
#   agent + engine + transitive runtime deps ... ~110-150 MB
#     dominated by pandas (~75 MB) + numpy (~40 MB);
#     pydantic/typer/rich/pynvml are small.
# The final layer carries only the venv + a non-root user — no uv, no pip
# cache, no apt cache, no source trees, no build wheels.
#
# Build context is the repo root so the agent's `../engine` path dep can
# be copied in alongside the agent itself:
#
#   # from the repo root
#   docker build -f packages/agent/Dockerfile -t et-agent:dev .

# -----------------------------------------------------------------------------
# Build stage — installs uv, materializes a self-contained venv at /opt/venv.
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# uv via pip keeps the version coupling shallow and avoids dragging an extra
# base image into the build pipeline. The builder layer is discarded anyway.
RUN pip install --no-cache-dir "uv>=0.5,<1"

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Build context is the repo root; copy each package by its real path.
# Engine is copied first because the agent's pyproject declares it as
# a path dependency — installing the engine into the venv first means
# uv resolves the agent's `gpu-doctor-engine` dep against an already-
# installed package rather than re-resolving the path source.
COPY packages/engine /build/engine
COPY packages/agent  /build/agent

# Build everything into /opt/venv so the final stage copies a single dir.
# --no-cache prevents uv from leaving its download cache inside the layer.
# Install the engine first, then the agent, for the resolution reason
# documented above.
RUN uv venv /opt/venv --python 3.12 \
 && VIRTUAL_ENV=/opt/venv uv pip install --no-cache \
        --python /opt/venv/bin/python /build/engine \
 && VIRTUAL_ENV=/opt/venv uv pip install --no-cache \
        --python /opt/venv/bin/python /build/agent

# -----------------------------------------------------------------------------
# Runtime stage — minimal, non-root, just the venv.
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# No apt installs; trim the apt cache and any pip leftovers for a clean layer.
RUN rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

# Non-root system user. UID 10001 sits above the typical host UID range and
# satisfies the K8s PodSecurity "must be > 0" guard used by runAsNonRoot.
RUN groupadd --system --gid 10001 agent \
 && useradd --system --uid 10001 --gid 10001 \
            --home /home/agent --shell /usr/sbin/nologin agent \
 && install -d -o agent -g agent /home/agent

COPY --from=builder --chown=agent:agent /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/agent

USER agent
WORKDIR /home/agent

ENTRYPOINT ["gpu-doctor-agent"]

# Default: Tier-1 sampling + mock attribution at 1 Hz, forever.
# Runs on any node — no GPU, no NVML library, no profiler traces required.
# Override `args` in the DaemonSet for any other profile (real NVML, file
# trace replay, etc.). Intentionally no --max-iters so the container behaves
# like a daemon and respects K8s lifecycle signals.
CMD ["run", "--mock", "--scenario", "idle", "--attribution-source", "mock", "--interval", "1.0"]
