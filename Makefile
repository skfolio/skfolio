# Run the checks CI runs, locally, in one command: `make check`.
#
# Each target mirrors a step in `.github/workflows/ci.yml`. CI additionally tests the
# minimum and latest dependency versions across Python versions and operating
# systems; see CONTRIBUTING.md. Without `make`, run the commands below directly.

.DEFAULT_GOAL := help
.PHONY: help check lint deps test

help: ## List the targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-6s %s\n", $$1, $$2}'

check: lint deps test ## Run every check: lint, deps and test

lint: ## Ruff, docstring presence, YAML and end-of-file checks on all files
	uv run pre-commit run --all-files

deps: ## Unused, missing or misplaced dependencies (deptry)
	uv run deptry src

test: ## Test suite, including the docstring examples
	uv run pytest
