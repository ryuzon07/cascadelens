# python:3.11.14-slim-bookworm, pinned by digest.
FROM python@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d AS base

# LightGBM's wheel links against the OpenMP runtime, which the slim image omits.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 make \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.13.6

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --extra api

# /model-info and the report read these. .dockerignore keeps out everything else, and all of data/.
COPY configs ./configs
COPY experiments ./experiments

RUN useradd --create-home --uid 10001 cascadelens \
    && chown -R cascadelens /app
ENV PATH="/app/.venv/bin:$PATH"

# Build with --target smoke, then run `make smoke RUN=` as this non-root user.
FROM base AS smoke
COPY Makefile ./
COPY scripts ./scripts
COPY docs ./docs
COPY tests/fixtures ./tests/fixtures
RUN chown -R cascadelens /app
USER cascadelens
CMD ["make", "smoke", "RUN="]

# The default target serves a saved model. MODEL_DIR is a folder inside the build context.
FROM base AS api
ARG MODEL_DIR=model
COPY ${MODEL_DIR} /app/model
RUN chown -R cascadelens /app/model
USER cascadelens
EXPOSE 8000
ENTRYPOINT ["cascadelens"]
# Inside a container the port must bind all interfaces. Publish it to localhost only.
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--model", "/app/model"]
