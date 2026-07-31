# Contributing

Thank you for contributing to skfolio. Contributions of all sizes are welcome,
including bug reports, feature proposals, documentation improvements, tests, and
code changes.

By participating in the project, you agree to follow our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- Report reproducible bugs through [GitHub issues][gh-issues].
- Propose improvements or new features.
- Improve the documentation, docstrings, or tutorials.
- Fix open issues.
- Add tests or improve existing test coverage.

Please report security vulnerabilities according to our
[Security Policy](SECURITY.md), rather than opening a public issue.

For substantial changes, consider opening an issue first. This gives maintainers and
contributors an opportunity to discuss the scope and approach before implementation.

### Report a bug

A useful bug report includes:

- Your operating system and Python version.
- Relevant details about your environment.
- A minimal reproducible example.
- The observed behavior and the expected behavior.
- The complete error message or traceback, when applicable.

### Propose a feature

A useful feature proposal:

- Explains the problem or use case.
- Describes the expected behavior.
- Keeps the initial scope as narrow as practical.
- Mentions alternatives or prior approaches when relevant.

## Development setup

Local development requires Python 3.10 or later and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

1. Fork the repository on GitHub and clone your fork:

   ```shell
   git clone git@github.com:your-name/skfolio.git
   cd skfolio
   ```

2. Create a virtual environment and install the development dependencies:

   ```shell
   uv venv
   uv pip install --editable ".[dev]"
   ```

3. Create a branch for your changes:

   ```shell
   git checkout -b name-of-your-bugfix-or-feature
   ```

   A recommended branch name is
   `category/reference/description-in-kebab-case`, where `category` is one of
   `feature`, `fix`, `refactor`, or `chore`, and `reference` is an issue such as
   `issue-34` or `no-ref`. For example:
   `feature/issue-34/factor-model`.

## Tests and code quality

Add tests for new behavior and bug fixes. While developing, you can run a focused
test file:

```shell
uv run pytest tests/path/to/test_file.py
```

Run the complete test suite when appropriate:

```shell
uv run pytest
```

Format and lint your changes with:

```shell
uv run ruff check --fix
uv run ruff format
```

## Documentation

If your change affects the documentation, install the documentation dependencies:

```shell
uv pip install --editable ".[dev,docs]"
cd docs
```

For a fast build without executing the tutorials:

```shell
uv run sphinx-build -b html -D plot_gallery=0 . _build
```

To execute a single tutorial, replace the filename in `filename_pattern`:

```shell
uv run sphinx-build -b html -D 'sphinx_gallery_conf.filename_pattern=plot_characteristics_factor_model\.py$' -D sphinx_gallery_conf.run_stale_examples=True . _build
```

To execute the complete gallery locally when needed:

```shell
uv run sphinx-build -b html . _build
```

The complete gallery can take about 30 minutes and is also executed by the
documentation deployment workflow.

Sphinx-Gallery generates `docs/auto_examples` and `docs/jupyterlite_contents`.
Edit the source tutorials under `examples` rather than editing these generated files.

## Submit your changes

Commit your changes using a message that follows
[Conventional Commits](https://www.conventionalcommits.org):

Use `type(scope): description`, where the scope is optional. Common types are:

- `feat`: New functionality.
- `fix`: Bug fix.
- `docs`: Documentation changes.
- `test`: Test changes.
- `refactor`: Code changes that preserve behavior.
- `perf`: Performance improvements.
- `build`: Packaging or dependency changes.
- `ci`: Continuous integration changes.
- `chore`: Repository maintenance.

Add `!` for a breaking change, for example
`feat!: remove deprecated parameter`.

```shell
git add .
git commit -m "feat(scope): describe your change"
git push origin name-of-your-bugfix-or-feature
```

Open a pull request through GitHub, or use the GitHub CLI:

```shell
gh pr create --fill
```

Draft pull requests are welcome and are a good place to discuss work in progress.
Before requesting a review:

- Include tests for feature changes and bug fixes.
- Update the documentation when behavior or public APIs change.
- Keep the pull request focused on one coherent change.
- Confirm that the relevant local checks pass.
- Ensure that continuous integration passes.

[gh-issues]: https://github.com/skfolio/skfolio/issues
