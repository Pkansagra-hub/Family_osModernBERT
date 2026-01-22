.PHONY: install install-dev format lint test clean clean-outputs pre-commit-install release-prep release-test

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

# Release preparation
release-prep:
	python scripts/prepare_release.py --version $(VERSION) --create-notes

release-test:
	python scripts/prepare_release.py --version $(VERSION) --test-install

# Cleaning
clean:
	if exist build rmdir /s /q build
	if exist dist rmdir /s /q dist
	if exist *.egg-info rmdir /s /q *.egg-info
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist .mypy_cache rmdir /s /q .mypy_cache
	if exist .ruff_cache rmdir /s /q .ruff_cache
	if exist htmlcov rmdir /s /q htmlcov
	for /d /r . %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d"
	del /s /q *.pyc 2>nul || echo.

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
