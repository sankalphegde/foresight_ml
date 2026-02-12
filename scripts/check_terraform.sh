#!/bin/bash

# Define colors for readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo "--- Starting Infrastructure Linting ---"

FAILED=0

# Change to infra directory
pushd "infra" > /dev/null || exit 1

echo -e "\n${YELLOW}Scope: infra${NC}"

# 1. Check Formatting
if ! terraform fmt -check -recursive > /dev/null; then
  echo -e "${RED}✘ Formatting error:${NC} Files are not formatted. Run 'terraform fmt' locally."
  terraform fmt -check -recursive # Print the specific filenames
  FAILED=1
else
  echo -e "${GREEN}✔ Formatting looks great!${NC}"
fi

# 2. Validate Config
terraform init -backend=false -input=false > /dev/null 2>&1
if ! terraform validate; then
  echo -e "${RED}✘ Validation failed:${NC} Syntax error in infra."
  FAILED=1
else
  echo -e "${GREEN}✔ Configuration is valid.${NC}"
fi

# Return to previous directory
popd > /dev/null

# Final Exit Logic
if [ $FAILED -eq 1 ]; then
  echo -e "\n${RED}Pipeline Failed.${NC} Please review the errors above."
  exit 1
else
  echo -e "\n${GREEN}Pipeline Passed!${NC} All checks cleared."
  exit 0
fi
