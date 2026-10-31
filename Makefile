UV ?= uv
ifeq ($(OS),Windows_NT)
# The bash on PATH on Windows is the WSL launcher, which hangs when WSL is missing.
BASH ?= C:/Program Files/Git/bin/bash.exe
else
BASH ?= bash
endif
RUN := $(UV) run --frozen
export UV

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
	$(RUN) python scripts/smoke.py

all:
	"$(BASH)" scripts/reproduce_all.sh
