# Security Policy

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Preferred channels:

- Use GitHub Private Vulnerability Reporting (Security → Report a vulnerability)
- Or email the maintainers at `security@skfoliolabs.com`

Please include the following in your report:

- Affected component and version(s), including Python version and OS
- Reproduction steps or proof-of-concept, and the expected vs. actual behavior
- Impact assessment and suggested CVSS v3.1 vector (if possible)
- Any relevant logs, stack traces, and configuration details

### Response targets (best effort)

- Acknowledgement: within 1 business days
- Triage and initial assessment: within 3 business days
- Fix window (from triage):
  - Critical: target 3 days
  - High: target 5 days
  - Moderate: target 20 days
  - Low: best effort

We will coordinate a disclosure timeline with you. Please keep reports private until a fix is released.

### Coordinated disclosure and CVE

- We handle advisories via GitHub Security Advisories and will request a CVE when appropriate
- Credit is offered to reporters who wish to be acknowledged
- Once a fix is available, we will publish release notes and an advisory with remediation guidance

## Supported Versions

- Actively supported: latest released version
- Older versions: security fixes may be backported at maintainers’ discretion based on severity and feasibility

For production use, we recommend pinning to a specific released version.

## Security Controls in This Repository

We use GitHub-native controls to reduce supply chain risk and detect vulnerabilities:

- Dependency Review is enabled on pull requests and blocks introducing high-severity vulnerabilities and non-approved licenses (MIT, BSD-3-Clause, BSD-2-Clause, Apache-2.0, ISC)
- Dependabot is enabled for both alerts and automated update PRs; it checks Python dependencies and GitHub Actions, opening grouped PRs for minor/patch updates
- CodeQL code scanning is enabled via GitHub Security
- SBOM (SPDX JSON) is exported and attached to each release
- A minimal and strict dependency set is maintained; skfolio depends only on a small number of well-maintained, widely used scientific Python libraries

## Secure Development Practices

- All changes are contributed via pull requests and are reviewed by maintainers before merge
- Linting and formatting is enforced with Ruff in CI
- Unit tests are run on Linux, macOS, and Windows across multiple Python versions
- GitHub Actions workflows are configured with minimal permissions
- The license policy is restricted to permissive, BSD-3-Clause–compatible licenses (MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, ISC)

## CI/CD Security Controls

- Tests workflow:
  - Sets up Python and `uv`, creates a virtual environment, installs deps from `pyproject.toml` extras
  - Runs `ruff check` and `ruff format --check`.
  - Executes `pytest` with coverage and uploads to Codecov
  - Uses least-privilege workflow permissions (`contents: read`)
- Coverage and PR gates:
  - Codecov reports coverage on every PR and branch push
  - Coverage thresholds are enforced via `codecov.yml` (project: 95%, patch: 97%)
  - The Codecov status check is required for PR merge (via branch protection)
- Dependency Review workflow:
  - Fails PRs on high-severity vulnerabilities or disallowed licenses
  - Posts a summary comment to PRs for visibility
- SBOM export workflow:
  - On tags `v*`, exports SPDX JSON from GitHub Dependency Graph and uploads to the matching GitHub Release
- Code scanning:
  - CodeQL is enabled via GitHub Security (Default setup) and runs on PRs/default branch and on a schedule

## Release and Provenance

- Versioning and tagging follow semantic-release; tags are formatted as `v{version}`.
- Release automation publishes distributions to PyPI and uploads artifacts to GitHub Releases.
- An SPDX SBOM is attached to each `v*` release.
- For reproducibility, consumers should pin to an exact release tag/version.

## Data and Privacy

- This library does not collect telemetry or transmit user data.
- Any datasets used in examples/tests are local or publicly available and are not sent to remote services by the library.
