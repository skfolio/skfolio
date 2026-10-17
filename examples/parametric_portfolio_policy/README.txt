.. _parametric_portfolio_policy_examples:

Parametric Portfolio Policy
---------------------------

Examples about :class:`~skfolio.optimization.ParametricPortfolioPolicy`: mapping
asset characteristics directly to portfolio weights by maximizing the realized
utility of the policy.

Supervised portfolio policies fit a regressor to allocations generated over
training windows. The supervised policy reuses the characteristic-processing
helpers contributed by Cashubski in PR #341.

Two command-line examples use bundled historical prices. Run from the repository
root::

    PYTHONPATH=src python examples/parametric_portfolio_policy/historical_walk_forward.py --output-dir /tmp/skfolio-supervised-historical
    PYTHONPATH=src python examples/parametric_portfolio_policy/cash_execution_example.py --output-dir /tmp/skfolio-supervised-cash

The first compares a random forest policy with a constrained equal-weight baseline
across yearly refits. The second demonstrates cash, transaction costs and synthetic
eligibility changes. Generated reports are software-validation outputs, not
investment-performance evidence; the fixed universe is not point-in-time data.
