.PHONY: uv-download
uv-download:
ifeq ($(OS),Windows_NT)
	# Windows-specific uv installation
	powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
else
	# Unix-like systems (Linux/macOS)
	curl -LsSf https://astral.sh/uv/install.sh | sh
endif

.PHONY: install-tests
install-tests:
	uv venv
ifeq ($(OS),Windows_NT)
	# Windows-specific virtual environment activation
	. .venv/Scripts/activate && uv pip install -e .[tests]
else
	# Unix-like systems (Linux/macOS)
	. .venv/bin/activate && uv pip install -e .[tests]
endif

.PHONY: install-dev
install-dev:
	uv venv
ifeq ($(OS),Windows_NT)
	# Windows-specific virtual environment activation
	. .venv/Scripts/activate && uv pip install -e .[dev]
else
	# Unix-like systems (Linux/macOS)
	. .venv/bin/activate && uv pip install -e .[dev]
endif
	uv run pre-commit install

.PHONY: codestyle
codestyle:
	uv run ruff check --output-format=github --fix ./
	uv run ruff format ./

.PHONY: check-codestyle
check-codestyle:
	uv run ruff check --output-format=github
	uv run ruff format --check

.PHONY: test
test:
	uv run pytest --cov=./ --cov-report=xml
