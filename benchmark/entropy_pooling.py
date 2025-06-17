from skfolio.datasets import load_sp500_dataset
from skfolio.measures import (
    mean,
    cvar,
    variance
)
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EntropyPooling
from skfolio.utils.figure import plot_kde_distributions

prices = load_sp500_dataset()
prices = prices[["AMD", "BAC", "GE", "JNJ", "JPM", "LLY", "PG"]]
X = prices_to_returns(prices)

mean(X["AMD"])
variance(X["AMD"])
cvar(X["AMD"])

ep = EntropyPooling(cvar_views=["AMD == 0.20"])
ep.fit(X)

mean(X["AMD"], sample_weight=ep.return_distribution_.sample_weight)
variance(X["AMD"], sample_weight=ep.return_distribution_.sample_weight)
cvar(X["AMD"], sample_weight=ep.return_distribution_.sample_weight)

print(f"Relative Entropy : {ep.relative_entropy_:.3f}")

plot_kde_distributions(
    X[["AMD"]],
    sample_weight=ep.return_distribution_.sample_weight,
    percentile_cutoff=0.1,
    title="Distribution of Asset Returns (Prior vs. Posterior)",
    unweighted_suffix="Prior",
    weighted_suffix="Posterior",
).show()


for apply_cvar_last in [True, False]:
    print("\n")
    print(f"Apply CVaR Last: {apply_cvar_last}")

    ep = EntropyPooling(variance_views=["AMD == 0.004"], cvar_views=["AMD == 0.20"], apply_cvar_last=apply_cvar_last)
    ep.fit(X)
    print(f"Relative Entropy : {ep.relative_entropy_:.3f}")

    print(f"Mean: {mean(X['AMD'], sample_weight=ep.return_distribution_.sample_weight):0.4f}")
    print(f"Variance: {variance(X['AMD'], sample_weight=ep.return_distribution_.sample_weight):0.4f}")
    print(f"CVaR: {cvar(X['AMD'], sample_weight=ep.return_distribution_.sample_weight):0.4f}")


    plot_kde_distributions(
        X[["AMD"]],
        sample_weight=ep.return_distribution_.sample_weight,
        percentile_cutoff=0.1,
        title="Distribution of Asset Returns (Prior vs. Posterior)",
        unweighted_suffix="Prior",
        weighted_suffix="Posterior",
    ).show()



