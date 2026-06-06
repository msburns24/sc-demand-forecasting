.PHONY: setup lint format lint-fix test run help

setup:
  git config core.hooksPath .githooks
	python -m venv .venv --upgrade-deps
	.venv/bin/python -m pip install -r requirements-dev.txt


# Dev
lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .

lint-fix:
	.venv/bin/ruff check --fix .

test:
	.venv/bin/python -m pytest tests/ -v

run:
	.venv/bin/uvicorn docker.app.main:app --reload --port 8000


help:
	@echo "Available targets:"
	@echo ""
	@echo "  General"
	@echo ""
	@echo "    setup        Create venv and install dependencies"
	@echo ""
	@echo "  Dev"
	@echo ""
	@echo "    lint         Check code with ruff"
	@echo "    format       Format code with ruff"
	@echo "    lint-fix     Auto-fix ruff issues"
	@echo "    test         Run pytest"
	@echo "    run          Start FastAPI dev server on :8000"
