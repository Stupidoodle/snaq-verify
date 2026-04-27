.PHONY: setup install lint lint-fix fmt typecheck test test-unit test-int verify eval run-and-eval verify-determinism docker-build docker-run run all clean help

# ============================================================================
# Setup
# ============================================================================

setup: install ## Install dependencies via uv sync

install: ## Install all dependencies
	uv sync

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run ruff linter
	uv run ruff check src/ tests/

lint-fix: ## Run ruff linter with auto-fix
	uv run ruff check --fix src/ tests/

fmt: ## Run ruff formatter + lint --fix
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

typecheck: ## Run mypy type checking
	uv run mypy src/

# ============================================================================
# Testing
# ============================================================================

test: test-unit ## Run unit tests (alias)

test-unit: ## Run unit tests only (no network)
	uv run pytest tests/unit/ -v

test-int: ## Run integration tests (real APIs)
	uv run pytest tests/integration/ -v -m integration

# ============================================================================
# Run
# ============================================================================

verify: ## Verify food_items.json (writes verification_report.json)
	uv run snaq-verify run --input food_items.json --output verification_report.json

eval: ## Evaluate report against golden set (writes eval_report.json)
	uv run snaq-verify eval --report verification_report.json \
		--ground-truth tests/data/ground_truth.json \
		--output eval_report.json

run-and-eval: ## Verify + evaluate in one shot
	uv run snaq-verify run-and-eval \
		--input food_items.json \
		--output verification_report.json \
		--eval-output eval_report.json \
		--ground-truth tests/data/ground_truth.json

verify-determinism: ## Run verify twice, diff value-fields, fail on mismatch
	uv run snaq-verify run --input food_items.json --output /tmp/snaq_run1.json
	uv run snaq-verify run --input food_items.json --output /tmp/snaq_run2.json
	uv run python -m snaq_verify.cli.diff_runs /tmp/snaq_run1.json /tmp/snaq_run2.json

# ============================================================================
# Docker (single-command path)
# ============================================================================

docker-build: ## Build docker image
	docker build -t snaq-verify:latest .

docker-run: docker-build ## Build + run via docker compose (single-command)
	docker compose up --build

run: docker-run ## Alias for docker-run

# ============================================================================
# Aggregate
# ============================================================================

all: setup lint typecheck test verify ## setup + lint + typecheck + test + verify

clean: ## Remove caches and artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .cache/ htmlcov/

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
