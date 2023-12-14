import numpy as np
import pytest

import skfolio.measures as skm
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns


@pytest.fixture(scope="module")
def returns():
    prices = load_sp500_dataset()
    X = prices_to_returns(X=prices[["AAPL"]], log_returns=False)
    returns = X.to_numpy().reshape(-1)
    return returns


def test_semi_variance(returns):
    np.testing.assert_almost_equal(skm.semi_variance(returns), 0.00036832171792503356)


def test_kurtosis(returns):
    np.testing.assert_almost_equal(
        skm.fourth_central_moment(returns), 1.4712882298202872e-05
    )


def test_semi_kurtosis(returns):
    np.testing.assert_almost_equal(
        skm.fourth_lower_partial_moment(returns), 1.067510944174677e-05
    )


def test_mean_absolute_deviation(returns):
    np.testing.assert_almost_equal(
        skm.mean_absolute_deviation(returns), 0.01860775246296703
    )


def test_cvar(returns):
    np.testing.assert_almost_equal(skm.cvar(returns), 0.05924007327154102)


def test_value_at_risk(returns):
    np.testing.assert_almost_equal(skm.value_at_risk(returns), 0.039568345323741094)


def test_worst_return(returns):
    np.testing.assert_almost_equal(skm.worst_realization(returns), 0.5184729064039408)


def test_first_lower_partial_moment(returns):
    np.testing.assert_almost_equal(
        skm.first_lower_partial_moment(returns), 0.009303876231483517
    )
    np.testing.assert_almost_equal(
        skm.first_lower_partial_moment(returns, min_acceptable_return=0),
        0.008732408884216343,
    )


def test_entropic_risk_measure(returns):
    np.testing.assert_almost_equal(
        skm.entropic_risk_measure(returns), 2.9949847733889547
    )
    np.testing.assert_almost_equal(
        skm.entropic_risk_measure(returns, theta=0.5, beta=0.5), 0.3462084546654301
    )


def test_evar(returns):
    np.testing.assert_almost_equal(skm.evar(returns), 0.21399369255094944)


def test_drawdown_at_risk(returns):
    np.testing.assert_almost_equal(
        skm.drawdown_at_risk(skm.get_drawdowns(returns)), 0.8498386636151526
    )
    np.testing.assert_almost_equal(
        skm.drawdown_at_risk(skm.get_drawdowns(returns, compounded=True)),
        0.752285191956124,
    )


def test_cdar(returns):
    np.testing.assert_almost_equal(
        skm.cdar(skm.get_drawdowns(returns)), 0.92763054634099
    )
    np.testing.assert_almost_equal(
        skm.cdar(skm.get_drawdowns(returns, compounded=True)), 0.7828217574177616
    )


def test_max_drawdown(returns):
    np.testing.assert_almost_equal(
        skm.max_drawdown(skm.get_drawdowns(returns)), 1.2480532424452897
    )
    np.testing.assert_almost_equal(
        skm.max_drawdown(skm.get_drawdowns(returns, compounded=True)),
        0.8180987202925042,
    )


def test_average_drawdown(returns):
    np.testing.assert_almost_equal(
        skm.average_drawdown(skm.get_drawdowns(returns)), 0.2444492558204457
    )
    np.testing.assert_almost_equal(
        skm.average_drawdown(skm.get_drawdowns(returns, compounded=True)),
        0.2851824165766825,
    )


def test_edar(returns):
    np.testing.assert_almost_equal(
        skm.edar(skm.get_drawdowns(returns)), 0.9962309765185212
    )


def test_ulcer_index(returns):
    np.testing.assert_almost_equal(
        skm.ulcer_index(skm.get_drawdowns(returns)), 0.36064200471643704
    )
    np.testing.assert_almost_equal(
        skm.ulcer_index(skm.get_drawdowns(returns, compounded=True)), 0.3830786822243506
    )


def test_gini_mean_difference(returns):
    np.testing.assert_almost_equal(
        skm.gini_mean_difference(returns), 0.027802253166037096
    )
