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

#### AI use

AI-assisted and fully AI-generated PRs are welcome, including submissions by AI
agents. All PRs must follow our [contributing guidelines](https://github.com/skfolio/skfolio/blob/main/CONTRIBUTING.md#ai-contributions).

If no AI was used to prepare this PR, write **None** and remove the fields below.
Otherwise, complete them:

- **Tools and models:** List the AI tools used, including model names or versions if known.
- **What AI did:** Briefly describe its involvement—for example, editing wording, investigating a bug, writing tests, or generating the entire PR.
- **Human review before submission:** All changes / Some changes / None / Unknown.

See [how we review PRs](https://github.com/skfolio/skfolio/blob/main/CONTRIBUTING.md#how-we-review-prs).

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
