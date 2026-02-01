.PHONY: help init deploy test clean

help:
	echo "Foresight-ML Minimal Pipeline"
	echo ""
	echo "Available commands:"
	echo "  make init       - Initialize project (DVC, deps)"
	echo "  make deploy     - Deploy infrastructure with Terraform"
	echo "  make test       - Test API clients locally"
	echo "  make clean      - Clear caches"
	echo ""

init:
	pip install -r requirements.txt
	dvc init
	echo "✓ Initialized"

deploy:
	cd terraform && terraform init && terraform apply
	echo "✓ Infrastructure deployed"

test:
	python -c "from src.sec_client import SECClient; print('SEC client OK')"
	python -c "from src.fred_client import FREDClient; print('FRED client OK')"
	echo "✓ Clients working"

clean:
	rm -rf .cache/
	echo "✓ Caches cleared"