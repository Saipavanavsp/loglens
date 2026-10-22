# Run from WSL/Linux/macOS. On Windows PowerShell use ./tasks.ps1 <target> instead.
PY ?= python

.PHONY: install lint format test check data-download data-parse up down

install:
	$(PY) -m pip install -e ".[data,gateway,eval,dashboard,dev]"
	pre-commit install

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

test:
	pytest

check: lint test

data-download:
	$(PY) -m loglens.data.download

data-parse:
	$(PY) -m loglens.data.parse

up:
	docker compose -f infra/docker-compose.yml up -d --build

down:
	docker compose -f infra/docker-compose.yml down
