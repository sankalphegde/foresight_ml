.PHONY: help setup local-up local-down lint format typecheck terraform-check test check

help:
	@echo "Foresight-ML Data Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make setup           - Install uv and initialize project"
	@echo ""
	@echo "Local Development:"
	@echo "  make local-up        - Start local Airflow"
	@echo "  make local-down      - Stop local Airflow"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            - Run ruff linter (check only)"
	@echo "  make format          - Format and fix code (runs pre-commit)"
	@echo "  make typecheck       - Run mypy type checker"
	@echo "  make terraform-check - Validate Terraform configuration"
	@echo "  make test            - Run pytest"
	@echo "  make check           - Run all checks (format + typecheck + terraform)"

setup:
	@echo "Installing uv..."
	command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "Initializing project..."
	uv sync
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install
	@echo "Setup complete!"

local-up:
	docker-compose up -d
	@echo "Airflow UI: http://localhost:8080 (admin/admin)"

local-down:
	docker-compose down

lint:
	uv run ruff check .

# Run pre-commit hooks twice to format and fix code.
# Hook may modify files and have exit code 1, so we run it twice to ensure all issues are resolved.
format:
	uv run pre-commit run --all-files || uv run pre-commit run --all-files

typecheck:
	uv run mypy src/

terraform-check:
	@cd infra && terraform fmt -check -recursive \
		&& terraform init -backend=false -input=false > /dev/null \
		&& terraform validate

check: format typecheck terraform-check
	@echo "All checks passed"

test:
	uv run pytest tests/
