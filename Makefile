.PHONY: pip-format-and-lint pip-test uv-format-and-lint uv-test

pip-format-and-lint:
	ruff check --fix
	ruff format

pip-test:
	pytest

uv-format-and-lint:
	uv run ruff check --fix
	uv run ruff format

uv-test:
	uv run pytest
