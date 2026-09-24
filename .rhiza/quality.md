# Quality charter

skfolio is not rhiza-managed. This file only records decisions the maintainers have
already taken, so that `/rhiza:quality` runs stop re-filing them.

## Accepted deviations

### GitHub Actions use version tags, not commit SHAs

Actions are referenced by major-version tag (`actions/checkout@v7`) rather than pinned to
a full commit SHA. Don't file findings asking for every `uses:` to be SHA-pinned.

Reason: decided in
[#299](https://github.com/skfolio/skfolio/issues/299#issuecomment-5795253187). SHA pinning
is not maintenance-free: every Dependabot SHA bump still needs CI, triage and a merge,
while a major-version tag picks up minor and patch releases with no repository change.
That extra review step only pays off if the upstream changes are actually audited, and
the project is unlikely to audit every release of `actions/checkout`, `setup-python` or
`upload-artifact`. Routinely merging SHA bumps because CI passes adds PR volume without
the security benefit that manual review implies. Mature open-source projects are also
split on the practice.

Hardening effort goes to privileged workflows instead. The release workflow can publish
packages, write repository contents, mint OIDC credentials and create a GitHub App token,
so it deserves more scrutiny than ordinary CI. A finding may still propose pinning an
individual high-privilege action, but only with a concrete reason for that action. #351
and its PR #352 were closed on these grounds.

## Filing

- Before filing, check closed issues and their discussion as well as open ones. A
  finding that was already discussed and closed should not be re-filed without new
  evidence that answers the reason it was closed.
