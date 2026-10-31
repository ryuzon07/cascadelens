UV ?= uv
RUN := $(UV) run --frozen

.PHONY: install lint typecheck test smoke all

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
	$(RUN) cascadelens version

all:
	@echo "TODO: make all will run the full pipeline once later parts exist"
