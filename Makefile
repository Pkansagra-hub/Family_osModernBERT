.PHONY: install install-dev format lint test clean

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

# Code quality
format:
	black src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

lint:
	ruff check src/ scripts/ tests/
	mypy src/

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/modeling_studio --cov-report=html

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-outputs:
	rm -rf outputs/
	rm -rf checkpoints/
	rm -rf logs/
	rm -rf runs/
	rm -rf wandb/

# Pre-commit
pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files
