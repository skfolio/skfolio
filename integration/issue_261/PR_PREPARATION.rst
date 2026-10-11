Algorithm PR preparation
========================

Proposed title: feat(optimization): integrate Carlo's online portfolio algorithms

Head branch: online-framework-port, in the lcrmorin/skfolio fork.
Target: skfolio/skfolio main. Proposed initial status: draft.
The GitHub body is in PR_DESCRIPTION.txt, following the repository template.
Neither the branch nor the PR has been published by this preparation.

Scope and review boundaries
---------------------------

Expose EG, AdaGrad and moving-average reversion through shared Carlo kernels.
Keep family factories internal. Include the common lifecycle, causal priors,
native geometry, shared constraints and one-period Portfolio evaluation required
to make these usable estimators. Preserve original source and test attribution.
Regret utilities and internal hindsight benchmark helpers are included. Broader
execution extensions remain separate.

Carlo's remaining algorithm tests are now ported. The base test file was renamed
so pytest collects it. All 46 inventoried source/test files are ported, and the
regret-dependent robustness case is enabled. Mathematical
reference changes are recorded in PORTING.rst and the inventory.

Upstream compatibility
----------------------

Merged upstream v1.8.1 (120880d7fd1cf525a43c46e0c180c45c9b21e4b0) without rewriting
the incremental history. Upstream's runtime AssetPanel fields property supersedes
the earlier local type-only workaround; the merged file is identical to upstream.
The algorithm PR consequently has no AssetPanel diff against its current base.
The upstream docs and CombinatorialPurgedCV changes are retained.

Review checkpoints before marking ready
--------------------------------------

* Agree concrete public names/defaults and the retained private family scope.
* Review native constrained objectives and adaptive state ordering.
* Review turnover entry anchors, causal prior warmup and failure/reset behavior.
* Decide whether the private objective catalogue warrants a separate contribution.
* Run GitHub CI after the user pushes; local validation does not substitute for
  the supported Python/platform matrix.

Final refreshed-branch validation
--------------------------------

The complete source-doctest and repository test run passes: 10,845 tests, no
skips (including 672 online tests). Ruff lint and formatting, repository type
checks, and numpydoc presence checks pass. Sphinx HTML builds successfully with
warnings. Source and destination provenance hashes are verified. These are local
checks, not the GitHub platform/version matrix.

Publication commands for the user
--------------------------------

These commands have not been run. From /Users/lucasmorin/skfolio-online-port::

    git push -u origin online-framework-port
    gh pr create --repo skfolio/skfolio --base main --head lcrmorin:online-framework-port --draft --title "feat(optimization): integrate Carlo's online portfolio algorithms" --body-file integration/issue_261/PR_DESCRIPTION.txt
