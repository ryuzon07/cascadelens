UV ?= uv
RUN ?= $(UV) run --frozen
export UV

.PHONY: install lint typecheck test smoke all report report-check dev web-install web-build \
	web-check e2e openapi

install:
	$(UV) sync --frozen --all-extras

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

typecheck:
	$(RUN) mypy

test:
	$(RUN) pytest --cov=cascadelens --cov-report=term-missing

smoke:
	$(RUN) python scripts/smoke.py

all:
	$(RUN) python scripts/run_stages.py

report:
	$(RUN) python -m cascadelens.report build

report-check:
	$(RUN) python -m cascadelens.report check

# Start the API and the Vite dev server together.
dev:
	$(RUN) python scripts/dev.py

web-install:
	cd web && npm ci

web-build:
	cd web && npm run build

web-check:
	cd web && npm run lint && npm run typecheck && npm test

e2e:
	cd web && npm run e2e

openapi:
	$(RUN) python scripts/export_openapi.py
	cd web && npm run gen:api:snapshot
