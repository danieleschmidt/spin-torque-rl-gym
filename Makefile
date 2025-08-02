# Makefile for Spin-Torque RL-Gym

.PHONY: help install install-dev clean test test-cov lint format type-check security-check build docs serve-docs publish pre-commit all-checks

# Default target
help: ## Show this help message
	@echo "Spin-Torque RL-Gym Development Commands"
	@echo "======================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install package for production
	pip install .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev,jax,viz]"
	pre-commit install

# Cleaning
clean: ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Testing
test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=spin_torque_gym --cov-report=html --cov-report=term-missing

test-fast: ## Run tests in parallel (fast)
	pytest tests/ -v -n auto

test-slow: ## Run all tests including slow ones
	pytest tests/ -v -m "not slow" --durations=10

# Code quality
lint: ## Run linter (ruff)
	ruff check spin_torque_gym/ tests/ --fix

format: ## Format code (black + isort)
	black spin_torque_gym/ tests/
	isort spin_torque_gym/ tests/

type-check: ## Run type checker (mypy)
	mypy spin_torque_gym/

security-check: ## Run security checks (bandit + safety)
	bandit -r spin_torque_gym/ -c pyproject.toml
	safety check

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

all-checks: lint type-check security-check test ## Run all code quality checks

# Build and distribution
build: clean ## Build distribution packages
	python -m build

docs: ## Build documentation
	@echo "Building documentation..."
	# Add sphinx-build command when docs are set up
	@echo "Documentation not yet configured"

serve-docs: ## Serve documentation locally
	@echo "Serving documentation..."
	# Add documentation serving command
	@echo "Documentation not yet configured"

publish: build ## Publish to PyPI (requires authentication)
	twine upload dist/*

publish-test: build ## Publish to Test PyPI
	twine upload --repository testpypi dist/*

# Development workflow
dev-setup: install-dev pre-commit ## Set up development environment
	@echo "Development environment ready!"

dev-test: format lint type-check test ## Run development workflow checks

ci-test: ## Run CI tests (same as GitHub Actions)
	pytest tests/ -v --cov=spin_torque_gym --cov-report=xml --cov-report=term-missing

# Benchmarking and profiling
benchmark: ## Run performance benchmarks
	@echo "Running benchmarks..."
	python -m pytest tests/benchmarks/ -v --benchmark-only

profile: ## Run profiling
	@echo "Running profiler..."
	python -m cProfile -s tottime -m spin_torque_gym.benchmarks.profile_main

# Environment management
env-create: ## Create virtual environment
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

env-update: ## Update dependencies
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev,jax,viz]" --upgrade

# Docker targets (when Dockerfile is added)
docker-build: ## Build Docker image
	docker build -t spin-torque-rl-gym .

docker-run: ## Run Docker container
	docker run -it --rm spin-torque-rl-gym

# Data and experiment management
data-download: ## Download sample datasets
	@echo "Downloading sample data..."
	# Add data download commands when available

experiments-clean: ## Clean experiment artifacts
	rm -rf experiments/
	rm -rf results/
	rm -rf wandb/
	rm -rf mlruns/

# Version management
version-patch: ## Bump patch version
	@echo "Bumping patch version..."
	# Add version bumping logic

version-minor: ## Bump minor version
	@echo "Bumping minor version..."
	# Add version bumping logic

version-major: ## Bump major version
	@echo "Bumping major version..."
	# Add version bumping logic

# IDE helpers
open-coverage: ## Open coverage report in browser
	@if [ -f htmlcov/index.html ]; then \
		python -m webbrowser htmlcov/index.html; \
	else \
		echo "No coverage report found. Run 'make test-cov' first."; \
	fi

# Git helpers
git-clean: ## Clean git repository
	git clean -fdx

git-reset: ## Reset to last commit (WARNING: destructive)
	git reset --hard HEAD
	git clean -fdx

# Quick development commands
quick-test: ## Quick test of core functionality
	python -c "import spin_torque_gym; print('Import successful')"
	pytest tests/test_core.py -v

quick-format: ## Quick format and lint
	black spin_torque_gym/ --check || (black spin_torque_gym/ && echo "Formatted!")
	ruff check spin_torque_gym/ --fix