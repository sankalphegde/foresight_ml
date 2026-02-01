.PHONY: help setup sync install local-up local-down terraform-init terraform-plan terraform-apply terraform-destroy lint format typecheck

help:
	@echo "Foresight-ML Data Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make setup           - Install uv and initialize project"
	@echo "  make sync            - Sync dependencies with uv"
	@echo "  make install         - Install package in development mode"
	@echo ""
	@echo "Local Development:"
	@echo "  make local-up        - Start local Airflow"
	@echo "  make local-down      - Stop local Airflow"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            - Run ruff linter"
	@echo "  make format          - Format code with ruff"
	@echo "  make typecheck       - Run mypy type checker"
	@echo "  make check           - Run all checks (lint + typecheck)"
	@echo ""
	@echo "GCP Deployment:"
	@echo "  make terraform-init  - Initialize Terraform"
	@echo "  make terraform-plan  - Preview infrastructure changes"
	@echo "  make terraform-apply - Deploy infrastructure"
	@echo "  make terraform-destroy - Destroy infrastructure"

setup:
	@echo "Installing uv..."
	command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "Initializing project..."
	uv sync
	@echo "Setup complete!"

sync:
	uv sync

install:
	uv pip install -e .

local-up:
	docker-compose up -d
	@echo "Airflow UI: http://localhost:8080 (admin/admin)"

local-down:
	docker-compose down

lint:
	uv run ruff check src/ airflow/

format:
	uv run ruff format src/ airflow/
	uv run ruff check --fix src/ airflow/

typecheck:
	uv run mypy src/

check: lint typecheck
	@echo "All checks passed"

terraform-init:
	cd terraform && terraform init

terraform-plan:
	cd terraform && terraform plan

terraform-apply:
	cd terraform && terraform apply

terraform-destroy:
	cd terraform && terraform destroy