<!--
Welcome to skfolio, and thanks for contributing!
-->

#### Reference Issues/PRs
<!--
Example: Fixes #1234. See also #3456.

Use a closing keyword such as "Fixes" so that the issue is closed when the pull
request is merged. See
https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue.

Write "None" if there is no related issue or pull request. For substantial changes,
consider opening an issue first: https://github.com/skfolio/skfolio/issues.
-->

#### What does this implement/fix? Explain your changes.
<!--
A clear and concise description of what you have implemented.
-->

#### Does your contribution introduce a new dependency? If yes, which one?
<!--
Only relevant if you changed `pyproject.toml`.
We try to minimize dependencies in the core dependency set.
-->

#### What should a reviewer concentrate their feedback on?
<!--
Point reviewers to the parts that are ready for feedback, especially in a draft
pull request.
-->

#### Any other comments?
<!--
Add anything else that reviewers should know.
-->

#### PR checklist
<!--
Remove the items that do not apply.
-->

##### For all contributions
- [ ] The PR title follows [Conventional Commits](https://www.conventionalcommits.org), for example `fix(portfolio): ...`, with a type listed in [CONTRIBUTING.md](https://github.com/skfolio/skfolio/blob/main/CONTRIBUTING.md#submit-your-changes).
- [ ] New behavior and bug fixes are covered by tests.
- [ ] The [tests and code quality checks](https://github.com/skfolio/skfolio/blob/main/CONTRIBUTING.md#tests-and-code-quality) pass locally.
- [ ] The documentation is updated when behavior or public APIs change.

##### For new estimators
- [ ] The estimator is listed in the API reference in `docs/api.rst`.
- [ ] The docstring and the `examples` gallery include a usage example.
