.PHONY: uv-download
uv-download:
	curl -LsSf https://astral.sh/uv/install.sh | sh

.PHONY: installdev
installdev:
	uv venv
	. .venv/bin/activate && \
	uv pip install -e .[dev] && \
	uv run pre-commit install

.PHONY: install-tests
install-tests:
	uv venv
	. .venv/bin/activate && \
	uv pip install -e .[tests]

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
