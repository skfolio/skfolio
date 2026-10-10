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
* At the end of iteration 1, the other 44 inventoried files were pending. In particular the OCO engine,
  projector, remaining mirror-map tests and estimator families are not moved yet.

The two files derive directly from Carlo's pinned source. Their compatibility
edits reuse reference commit df64d34: annotations, missing docstrings, lint cleanup,
the equivalent Tsallis entropy-map presence check, and corrected Burg doctests.
No new update formula or estimator-specific implementation is introduced. The full
set of mirror maps is retained to preserve composition and adaptive-state behavior.
The inventory includes destination digests and the exact adaptation reference.

For the exact source-to-port delta, when Carlo's commit is present locally, run::

    git diff c1bd1f168cf7d660c81f74710aadd9941e64a2c7 -- src/skfolio/optimization/online/_mirror_maps.py tests/test_optimization/test_online/mirror_maps/test_mirror_maps_composite.py

Iteration 2a: projector and predictor dependencies

Moved _projection.py, _prediction.py, _utils.py and Carlo's test_projection.py.
The projector retains Carlo's fast/convex selection. Corrections are deliberately
recorded: infeasible budgets raise; budget=None means no sum constraint; zero
turnover cannot change holdings; positive turnover solves the constraint
intersection when needed; short exposure uses a convex upper bound; variance
and tracking limits are both applied when both are supplied.

Predictor formulas are preserved, with annotations/docstrings from the reference
branch. OLMAR1 currently returns the last observation through W warmup rows, then
averages inverse products over W relatives without a unit term for current price.
The standalone regression records that source convention, not a claim of agreement
with the literature. Review its indexing/window interpretation before exposing the
OLMAR estimator. Predictor histories are also currently unbounded.

The utility module is copied for CLIP_EPSILON; its legacy return clipping must not
become an implicit policy for missing or bankrupt assets in the future estimator.
New tests cover reset behavior, source OLMAR timing and corrected projection bounds.

The branch still exposes no public portfolio estimator. Common constraint
extraction, native constrained mirror geometry and estimator integration remain
pending. The shared OCO engine and dependent tests are moved in iteration 2b below.

Iteration 2b: shared OCO engine

Moved _foco.py, winner/test_foco_engine.py and mirror_maps/test_mirror_maps.py.
Carlo's OMD/FTRL paths, adaptive state ordering, optimistic predictors, scheduled
learning rates and SwordMeta remain shared. Compatibility edits add annotations,
docstrings and scalar conversion; the tests retain Carlo's reference assertions.
The original projector.project interface is retained deliberately. Native KL and
adaptive/composite constrained geometry are not supplied by this commit.

Current inventory: 9 of 46 source/test files moved (5 implementation modules and
4 original test files); 37 remain pending. Two new regression test files cover
projector corrections and predictor state/timing. No public estimator is exported.

Next iteration: shared lifecycle and algorithm-family adapters, bringing their
objective and learning-rate dependencies only as needed. Keep optional covariance
estimates separate from gradient state and keep account execution in Portfolio.
The current projector validates a mathematical candidate against supplied holdings;
it does not decide when those holdings were executed or drift them itself.

Next iterations
---------------

1. Completed: mirror maps and their standalone composite-map tests. Preserve adaptive
   and composite geometry rather than reducing the code to a standalone EG update.
2. Completed: FirstOrderOCO, projector interfaces/implementations and gradient predictors
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

Iteration 2a checks: 65 tests and doctests pass. Ruff, formatting, numpydoc and
full-repository type checking pass. The two known coupled-map line-search warnings
remain. The imported fast/convex projector performs Euclidean projections; its
interface alone does not establish native adaptive or entropy constrained updates.

Iteration 2b checks: all 109 online-core tests and source doctests pass. Full
repository Ruff lint/format and type checks pass; numpydoc presence checks pass.
The source engine docstring has nonstandard section headings reported as warnings;
these do not fail the presence checks. The two existing scipy line-search warnings
remain. Source/destination inventory digests are verified before committing.
