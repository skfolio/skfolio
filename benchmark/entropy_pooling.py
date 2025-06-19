import numpy as np

from skfolio.datasets import load_sp500_dataset
from skfolio.measures import cvar, mean, standard_deviation, variance
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EntropyPooling
from skfolio.utils.figure import plot_kde_distributions

if __name__ == "__main__":
    # Load price data, keep two stocks and convert to returns
    prices = load_sp500_dataset()
    prices = prices[["AMD", "BAC"]]
    X = prices_to_returns(prices)

    # Mean, Vol and CVaR stats
    print(f"Ann. Mean: {mean(X['AMD']) * 252:0.1%}")
    print(f"Ann. Vol: {standard_deviation(X['AMD']) * np.sqrt(252):0.1%}")
    print(f"CVaR-95%: {cvar(X['AMD']):0.1%}")

    # Entropy Pooling with only a CVaR view to see how it affects the mean
    ep = EntropyPooling(cvar_views=["AMD == 0.15"])
    ep.fit(X)

    sample_weight = ep.return_distribution_.sample_weight

    # Results
    print(f"Relative Entropy : {ep.relative_entropy_:.3f}")
    print(f"Ann. Mean: {mean(X['AMD'], sample_weight=sample_weight) * 252:0.0%}")
    print(
        f"Ann. Vol: {standard_deviation(X['AMD'], sample_weight=sample_weight) * np.sqrt(252):0.0%}"
    )
    print(f"CVaR-95%: {cvar(X['AMD'], sample_weight=sample_weight):0.1%}")

    # As expected, the posterior mean is smaller than its prior.

    # Plot the distribution
    plot_kde_distributions(
        X[["AMD"]],
        sample_weight=sample_weight,
        percentile_cutoff=0.1,
        title="Distribution of AMD Returns (Prior vs. Posterior)",
        unweighted_suffix="Prior",
        weighted_suffix="Posterior",
    ).show()

    # Entropy Pooling with a CVaR view plus a variance view.
    # We compare by applying the CVaR at the first stage vs the final stage.

    vol_view = 0.8
    variance_view = vol_view**2/252
    for apply_cvar_last in [True, False]:
        print("\n")
        print(f"Apply CVaR Last: {apply_cvar_last}")

        ep = EntropyPooling(
            variance_views=[f"AMD == {variance_view}"],
            cvar_views=["AMD == 0.15"],
            apply_cvar_last=apply_cvar_last,
        )
        ep.fit(X)

        sample_weight = ep.return_distribution_.sample_weight

        # Results
        print(f"Relative Entropy : {ep.relative_entropy_:.3f}")
        print(f"Ann. Mean: {mean(X['AMD'], sample_weight=sample_weight) * 252:0.0%}")
        print(
            f"Ann. Vol: {standard_deviation(X['AMD'], sample_weight=sample_weight) * np.sqrt(252):0.0%}"
        )
        print(f"CVaR-95%: {cvar(X['AMD'], sample_weight=sample_weight):0.1%}")

        # Plot the distribution
        plot_kde_distributions(
            X[["AMD"]],
            sample_weight=ep.return_distribution_.sample_weight,
            percentile_cutoff=0.1,
            title="Distribution of AMD Returns (Prior vs. Posterior)",
            unweighted_suffix="Prior",
            weighted_suffix="Posterior",
        ).show()
