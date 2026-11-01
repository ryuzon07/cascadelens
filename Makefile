UV ?= uv
RUN ?= $(UV) run --frozen
export UV

.PHONY: install lint typecheck test smoke all report report-check

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
