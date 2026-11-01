FROM python:3.11-slim

# Directory of a saved model, inside the build context. It holds no benchmark data.
ARG MODEL_DIR=model

# LightGBM's wheel links against the OpenMP runtime, which the slim image omits.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.13.6

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --extra api

# /model-info reads these. .dockerignore keeps out everything else, and all of data/.
COPY configs ./configs
COPY experiments ./experiments
COPY ${MODEL_DIR} /app/model

RUN useradd --create-home --uid 10001 cascadelens \
    && chown -R cascadelens /app
USER cascadelens

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
ENTRYPOINT ["cascadelens"]
# Inside a container the port must bind all interfaces. Publish it to localhost only.
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--model", "/app/model"]
