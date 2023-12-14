import datetime as dt

import numpy as np
import pytest

from skfolio.datasets import load_sp500_dataset
from skfolio.moments import EWMu, EmpiricalMu, EquilibriumMu, ShrunkMu, ShrunkMuMethods
from skfolio.preprocessing import prices_to_returns


@pytest.fixture(scope="module")
def X():
    prices = load_sp500_dataset()
    prices = prices.loc[dt.date(2014, 1, 1) :]
    X = prices_to_returns(X=prices, log_returns=False)
    return X


def test_empirical_mu(X):
    model = EmpiricalMu()
    model.fit(X)
    np.testing.assert_almost_equal(model.mu_, np.array(X).mean(axis=0))


def test_ew_mu(X):
    model = EWMu()
    model.fit(X)
    np.testing.assert_almost_equal(
        model.mu_,
        np.array(
            [
                -1.24726372e-02,
                -9.45067113e-03,
                1.61657194e-03,
                -1.83966453e-03,
                2.83363890e-03,
                9.19409711e-04,
                -2.64229605e-03,
                -5.19784162e-04,
                1.54353089e-03,
                6.95047714e-05,
                4.90700510e-04,
                6.30137508e-04,
                -6.19996526e-03,
                -4.15448322e-04,
                -3.22012113e-03,
                4.39614645e-05,
                -1.16913798e-02,
                -5.00326373e-04,
                -4.56242603e-03,
                1.86622274e-03,
            ]
        ),
    )


def test_equilibrium_mu(X):
    model = EquilibriumMu()
    model.fit(X)
    np.testing.assert_almost_equal(
        model.mu_,
        np.array(
            [
                1.39281815e-04,
                2.24154782e-04,
                1.69252898e-04,
                1.58471289e-04,
                1.54063130e-04,
                1.55075717e-04,
                1.23265081e-04,
                8.27725049e-05,
                1.52196578e-04,
                8.34019782e-05,
                1.00899188e-04,
                8.80173294e-05,
                1.37526282e-04,
                8.88137066e-05,
                9.35022944e-05,
                8.03837339e-05,
                2.14880096e-04,
                1.22283550e-04,
                7.42303223e-05,
                1.36071903e-04,
            ]
        ),
    )


def test_shrinkage_mu(X):
    model = ShrunkMu()
    model.fit(X)
    np.testing.assert_almost_equal(
        model.mu_,
        np.array(
            [
                9.47940177e-04,
                1.59095814e-03,
                6.01237831e-04,
                7.18130766e-04,
                5.45759046e-04,
                9.15282071e-05,
                7.68939857e-04,
                5.15035127e-04,
                6.31494691e-04,
                4.59445265e-04,
                9.92082170e-04,
                6.12075663e-04,
                9.40928476e-04,
                5.67185534e-04,
                5.35844339e-04,
                5.14453650e-04,
                3.23942637e-04,
                9.59637316e-04,
                4.91763301e-04,
                4.40458356e-04,
            ]
        ),
    )

    model = ShrunkMu(vol_weighted_target=True)
    model.fit(X)
    np.testing.assert_almost_equal(
        model.mu_,
        np.array(
            [
                9.26914220e-04,
                1.62077956e-03,
                5.52795992e-04,
                6.78932369e-04,
                4.92930159e-04,
                2.78050083e-06,
                7.33759246e-04,
                4.59776711e-04,
                5.85445448e-04,
                3.99791018e-04,
                9.74546792e-04,
                5.64490839e-04,
                9.19348060e-04,
                5.16050971e-04,
                4.82231436e-04,
                4.59149253e-04,
                2.53573366e-04,
                9.39536323e-04,
                4.34664640e-04,
                3.79302697e-04,
            ]
        ),
    )

    model = ShrunkMu(method=ShrunkMuMethods.BAYES_STEIN)
    model.fit(X)
    np.testing.assert_almost_equal(
        model.mu_,
        np.array(
            [
                0.00083316,
                0.00121765,
                0.00062585,
                0.00069574,
                0.00059267,
                0.00032106,
                0.00072612,
                0.0005743,
                0.00064394,
                0.00054106,
                0.00085955,
                0.00063233,
                0.00082896,
                0.00060548,
                0.00058674,
                0.00057395,
                0.00046004,
                0.00084015,
                0.00056038,
                0.00052971,
            ]
        ),
    )

    model = ShrunkMu(method=ShrunkMuMethods.BODNAR_OKHRIN)
    model.fit(X)
    np.testing.assert_almost_equal(
        model.mu_,
        np.array(
            [
                5.95933693e-05,
                -1.35794787e-04,
                1.64942731e-04,
                1.29423513e-04,
                1.81800576e-04,
                3.19823666e-04,
                1.13984606e-04,
                1.91136381e-04,
                1.55748848e-04,
                2.08027978e-04,
                4.61803341e-05,
                1.61649535e-04,
                6.17239528e-05,
                1.75289900e-04,
                1.84813270e-04,
                1.91313069e-04,
                2.49201958e-04,
                5.60390638e-05,
                1.98207784e-04,
                2.13797362e-04,
            ]
        ),
    )
