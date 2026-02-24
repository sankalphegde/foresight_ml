.PHONY: help setup local-up local-down demo-airflow lint format typecheck terraform-check test check dvc-setup dvc-push dvc-pull

help:
	@echo "Foresight-ML Data Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make setup           - Install uv and initialize project"
	@echo "  make dvc-setup       - Initialize DVC with GCS remote"
	@echo ""
	@echo "Local Development:"
	@echo "  make local-up        - Start local Airflow"
	@echo "  make local-down      - Stop local Airflow"
	@echo "  make demo-airflow    - Start, unpause, trigger DAG, and show task states"
	@echo ""
	@echo "Data Version Control:"
	@echo "  make dvc-push        - Push tracked data to GCS"
	@echo "  make dvc-pull        - Pull tracked data from GCS"
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

demo-airflow:
	@RUN_ID=$${RUN_ID:-demo_manual_$$(date +%Y%m%d_%H%M%S)}; \
	echo "Using run_id=$$RUN_ID"; \
	docker-compose up -d airflow; \
	docker-compose exec airflow airflow dags unpause foresight_ingestion; \
	docker-compose exec airflow airflow dags trigger foresight_ingestion --run-id $$RUN_ID; \
	sleep 8; \
	docker-compose exec airflow airflow tasks states-for-dag-run foresight_ingestion $$RUN_ID

lint:
	uv run ruff check src/ tests/ scripts/ monitoring/

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

check: format typecheck lint terraform-check
	@echo "All checks passed"

test:
	uv run pytest tests/

dvc-setup:
	@if [ -z "$$GCS_BUCKET" ]; then \
		echo "Error: GCS_BUCKET environment variable not set"; \
		echo "Run: source .env"; \
		exit 1; \
	fi
	@echo "Initializing DVC..."
	uv run dvc init --force
	@echo "Configuring GCS remote: gs://$$GCS_BUCKET/dvc-storage"
	uv run dvc remote add -d storage gs://$$GCS_BUCKET/dvc-storage --force
	@if [ -n "$$GOOGLE_APPLICATION_CREDENTIALS" ]; then \
		echo "Setting credentials path..."; \
		uv run dvc remote modify storage credentialpath $$GOOGLE_APPLICATION_CREDENTIALS; \
	fi
	@echo "DVC setup complete!"
	@echo "Next steps:"
	@echo "  1. Track data: dvc add data/companies.csv"
	@echo "  2. Commit: git add data/companies.csv.dvc .gitignore && git commit -m 'Track data'"
	@echo "  3. Push: make dvc-push"

dvc-push:
	@echo "Pushing data to GCS..."
	uv run dvc push

dvc-pull:
	@echo "Pulling data from GCS..."
	uv run dvc pull
