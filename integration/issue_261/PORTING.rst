Incremental integration of Carlo's online framework
===================================================

Source and boundaries
---------------------

Source: CarloNicolini/skfolio, universal_portfolio, commit
c1bd1f168cf7d660c81f74710aadd9941e64a2c7. Upstream base: skfolio v1.8.0,
33923e0. The full previous integration remains on online-integration-ci as a
reference, not as the starting tree for this branch. This branch is
online-framework-port.

PORT_INVENTORY.json inventories every source and test file in Carlo's online
folders, with original line counts and SHA-256 digests. Status pending means it
has not been moved, adapted means it has been moved with listed changes, and
unchanged means a byte-identical copy. Deferred will require an explicit reason.
Source package exports are not considered moved just because a package directory
exists. Supporting changes outside these folders are listed separately below.

Each functional commit must update its inventory entries, keep attribution,
include its dependent tests and explain deviations from Carlo's source. Port only
a dependency-complete slice. Prefer original tests; add tests for changed semantics.
Do not treat passing imports as evidence that an estimator is integrated.

Progress
--------

Iteration 1: mirror maps

* Adapted: src/skfolio/optimization/online/_mirror_maps.py.
* Adapted: tests/test_optimization/test_online/mirror_maps/test_mirror_maps_composite.py.
* New scaffold: online/__init__.py contains only a module docstring. Carlo's family
  exports have not been copied; that source inventory entry remains pending.
* All other 44 inventoried files remain pending. In particular the OCO engine,
  projector, remaining mirror-map tests and estimator families are not moved yet.

The two files derive directly from Carlo's pinned source. Their compatibility
edits reuse reference commit df64d34: annotations, missing docstrings, lint cleanup,
the equivalent Tsallis entropy-map presence check, and corrected Burg doctests.
No new update formula or estimator-specific implementation is introduced. The full
set of mirror maps is retained to preserve composition and adaptive-state behavior.
The inventory includes destination digests and the exact adaptation reference.

For the exact source-to-port delta, when Carlo's commit is present locally, run::

    git diff c1bd1f168cf7d660c81f74710aadd9941e64a2c7 -- src/skfolio/optimization/online/_mirror_maps.py tests/test_optimization/test_online/mirror_maps/test_mirror_maps_composite.py

This branch does not yet expose a usable online portfolio estimator. The next
functional commit should move the shared OCO engine with its projector/predictor
dependencies and tests, not add a standalone EG implementation.

Next iterations
---------------

1. Move mirror maps and their standalone composite-map tests. Preserve adaptive
   and composite geometry rather than reducing the code to a standalone EG update.
2. Move FirstOrderOCO, projector interfaces/implementations and gradient predictors
   with Carlo's engine and remaining mirror-map tests. Record limitations of rich
   projections explicitly; passing a Euclidean projection through a generic API
   does not establish correctness for arbitrary adaptive geometry.
3. Adapt the shared lifecycle and both algorithm families to fit-reset,
   partial_fit-continue and next-period target timing. Move affected tests together.
   Keep algorithm state, optional statistical state and execution holdings separate.
4. Integrate with existing Portfolio and online evaluation. Test EG, AdaGrad and
   OLMAR as representative configurations of the shared machinery. Agree public
   estimator names and explicit sklearn-compatible constructors before export.
5. Extract/reuse common constraints, preserve native geometry, and add optional
   causal priors with metadata routing. Follow existing skfolio parameter names.
6. Port universe, recovery and larger-window extensions as separate changes after
   the core contract is established. These already exist on the reference branch;
   they are not yet present here.

Supporting dependencies and scope decisions
------------------------------------------

The earlier full import also changed measures/__init__.py, measures/_enums.py,
measures/_measures.py, portfolio/_base.py and tests/test_measures/test_measures.py.
Those changes have not been copied. Audit them when the objectives/estimators
need them, against current upstream functionality rather than copying whole files.

Carlo's external datasets and legacy model-selection code are excluded: reuse
skfolio datasets and existing online evaluation infrastructure. Generic objective
catalogues, regret tooling and benchmark APIs remain pending scope decisions,
not silently deleted. Autograd must not become a production dependency merely to
validate analytical gradients.

Open design decisions
---------------------

* Concrete public estimators versus lightweight aliases, with shared internals
  retained in either case. Explicit constructors are needed for clean sklearn
  parameter discovery and tuning.
* Which constrained adaptive/composite updates are mathematically supported.
* Whether reference-branch extensions should use newer upstream facilities instead
  of carrying forward compatibility APIs and duplicated failure handling.

Validation
----------

The initial planning commit contains no executable implementation changes.
Iteration 1 validation is recorded below after its checks complete. The upstream AssetPanel.fields type-contract fix is carried in its own commit
(cherry-picked from df64d34), separate from the Carlo port. It changes the shared
readable type contract without altering runtime panel behavior. Its 503-test
reference validation included the container suite; the new branch rechecks types.
Do not bring the personal CI branch trigger into an upstream contribution.

Iteration 1 checks: 44 tests passed, including the original standalone composite-map
tests and source doctests. Two scipy line-search warnings occur in coupled-map
tests; their numerical assertions pass. Ruff lint/format and numpydoc presence
checks pass. Full-repository type checking passes with the separate upstream fix.
These checks validate this slice; they do not validate an online portfolio estimator.
