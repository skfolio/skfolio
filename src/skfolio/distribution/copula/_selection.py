import numpy as np
import numpy.typing as npt
import scipy.stats as st
import sklearn as sk

from skfolio.distribution.copula._base import BaseBivariateCopula
from skfolio.distribution.copula._independent import IndependentCopula


def select_bivariate_copula(
    X: npt.ArrayLike,
    copula_candidates: list[BaseBivariateCopula],
    aic: bool = True,
    independence_level: float = 0.05,
) -> BaseBivariateCopula:
    """
    Select the best bivariate copula from a list of candidates using an information
    criterion.

    This function first tests the dependence between the two variables in X using
    Kendall's tau independence test. If the p-value is greater than or equal to
    `independence_level`, the null hypothesis of independence is not rejected, and the
    `IndependentCopula` is returned. Otherwise, each candidate copula in
    `copula_candidates` is fitted to the data X. For each candidate, either the
    Akaike Information Criterion (AIC) or the Bayesian Information Criterion (BIC) is
    computed, and the copula with the lowest criterion value is selected.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        An array of bivariate inputs (u, v) with uniform marginals (values in [0, 1]).

    copula_candidates : list[BaseBivariateCopula]
        A list of candidate copula models. Each candidate must inherit from
        `BaseBivariateCopula`.

    aic : bool, default=True
        If True, the Akaike Information Criterion (AIC) is used for model selection;
        otherwise, the Bayesian Information Criterion (BIC) is used.

    independence_level : float, default=0.05
        The significance level for the Kendall tau independence test. If the p-value is
        greater than or equal to this level, the independence hypothesis is not
        rejected, and the `IndependentCopula` is returned.

    Returns
    -------
    selected_copula : BaseBivariateCopula
        The fitted copula model among the candidates that minimizes the selected
        information criterion (AIC or BIC).

    Raises
    ------
    ValueError
        If X is not a 2D array with exactly two columns, or if any candidate in
        `copula_candidates` does not inherit from `BaseBivariateCopula`.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must contains two columns for Bivariate Copula")

    kendall_tau, p_value = st.kendalltau(X[:, 0], X[:, 1])
    if p_value >= independence_level:
        return IndependentCopula().fit(X)

    results = {}
    for copula in copula_candidates:
        if not isinstance(copula, BaseBivariateCopula):
            raise ValueError(
                "The candidate copula must inherit from BaseBivariateCopula"
            )
        copula = sk.clone(copula)
        if copula.itau and copula.kendall_tau is None:
            # Faster computation by reusing kendall tau if itau
            copula.kendall_tau = kendall_tau
        copula.fit(X)
        results[copula] = copula.aic(X) if aic else copula.bic(X)
    selected_copula = min(results, key=results.get)
    # noinspection PyTypeChecker
    return selected_copula
