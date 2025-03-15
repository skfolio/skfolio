import numpy as np
import plotly.graph_objects as go
import pytest

from skfolio.distribution import GaussianCopula
from skfolio.distribution.copula._base import _RHO_BOUNDS


@pytest.fixture
def X():
    # Using same convention as other libraries for Benchmark
    X = np.array(
        [
            [0.05, 0.1],
            [0.4, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.2, 0.3],
        ]
    )
    return X


@pytest.fixture
def fitted_model():
    # Using same convention as other libraries for Benchmark
    fitted_model = GaussianCopula(random_state=42)
    fitted_model.rho_ = 0.8090169943749475
    return fitted_model


def test_gaussian_copula_init():
    """Test initialization of GaussianCopula with default parameters."""
    model = GaussianCopula()
    assert model.itau is True
    assert model.kendall_tau is None


def test_gaussian_copula_fit_with_kendall_tau(random_data):
    """Test fit() using Kendall's tau inversion (default)."""
    model = GaussianCopula(itau=False).fit(random_data)
    # Check that rho_ has been fitted and lies within the valid range
    assert hasattr(model, "rho_")
    assert _RHO_BOUNDS[0] < model.rho_ < _RHO_BOUNDS[1]


def test_gaussian_copula_fit_with_provided_kendall_tau(random_data):
    """Test fit() using a manually provided Kendall's tau."""
    model = GaussianCopula(itau=True, kendall_tau=0.25)
    model.fit(random_data)
    # Check that the resulting rho_ is sin(pi * 0.25 / 2) = sin(pi/8)
    expected_rho = np.sin(np.pi * 0.25 / 2.0)
    assert np.isclose(model.rho_, expected_rho)


def test_gaussian_copula_fit_mle(random_data):
    """Test fit() using MLE optimization for rho_."""
    model = GaussianCopula(itau=False)
    model.fit(random_data)
    # Just check it's between -1 and 1
    assert -1 < model.rho_ < 1


def test_gaussian_cdf_shape(random_data):
    """Test cdf() returns correct shape."""
    model = GaussianCopula().fit(random_data)
    cdf = model.cdf(random_data)
    assert cdf.shape == (100,)
    # All values should remain in (0,1) for a well-behaved CDF
    assert np.all(cdf >= 0) and np.all(cdf <= 1)


@pytest.mark.parametrize(
    "first_margin",
    [
        True,
        False,
    ],
)
def test_gaussian_partial_derivative_shape(random_data, first_margin):
    """Test partial_derivative() returns correct shape."""
    model = GaussianCopula().fit(random_data)
    h = model.partial_derivative(random_data, first_margin=first_margin)
    assert h.shape == (100,)
    # All values should remain in (0,1) for a well-behaved CDF
    assert np.all(h >= 0) and np.all(h <= 1)


@pytest.mark.parametrize(
    "first_margin",
    [
        True,
        False,
    ],
)
def test_gaussian_inverse_partial_derivative_shape(random_data, first_margin):
    """Test inverse_partial_derivative() returns correct shape."""
    model = GaussianCopula().fit(random_data)
    h_inv = model.inverse_partial_derivative(random_data, first_margin=first_margin)
    assert h_inv.shape == (100,)
    # Should lie within [0,1]
    assert np.all(h_inv >= 0) and np.all(h_inv <= 1)


def test_gaussian_score_samples(random_data):
    """Test score_samples() for shape and type."""
    model = GaussianCopula().fit(random_data)
    log_pdf = model.score_samples(random_data)
    assert log_pdf.shape == (100,)
    # log-pdf can be negative or positive, so we won't do a bound check here.


def test_gaussian_score(random_data):
    """Test the total log-likelihood via score()."""
    model = GaussianCopula().fit(random_data)
    total_ll = model.score(random_data)
    # It's a scalar
    assert isinstance(total_ll, float)


def test_gaussian_aic_bic(random_data):
    """Test AIC and BIC computation."""
    model = GaussianCopula().fit(random_data)
    aic_val = model.aic(random_data)
    bic_val = model.bic(random_data)

    # Both are floats
    assert isinstance(aic_val, float)
    assert isinstance(bic_val, float)

    # Typically, BIC >= AIC for large n, but not guaranteed. Just check they're finite.
    assert np.isfinite(aic_val)
    assert np.isfinite(bic_val)


def test_gaussian_sample():
    """Test sample() method for shape and range."""
    model = GaussianCopula(random_state=123).fit(np.random.rand(100, 2))
    samples = model.sample(n_samples=50)
    assert samples.shape == (50, 2)
    # Should lie strictly in (0,1).
    assert np.all(samples >= 1e-8) and np.all(samples <= 1 - 1e-8)


def test_gaussian_rho_out_of_bounds():
    """Check that setting rho_ out of bounds triggers an error in score_samples()."""
    model = GaussianCopula(itau=False)
    # Manually set rho_ out-of-bounds after fitting
    model.rho_ = 1.2
    with pytest.raises(ValueError, match="rho must be between -1 and 1."):
        _ = model.score_samples(np.random.rand(5, 2))


@pytest.mark.parametrize(
    "itau,expected",
    [
        [True, 0.8090169943749475],
        [False, 0.9273756082619797],
    ],
)
def test_gaussian_fit(X, itau, expected):
    model = GaussianCopula(itau=itau)
    model.fit(X)
    assert np.isclose(model.rho_, expected)


def test_gaussian_score_exact(X, fitted_model):
    assert np.isclose(fitted_model.score(X), 3.2732998035445258)
    np.testing.assert_almost_equal(
        fitted_model.score_samples(X),
        np.array([1.34908291, 0.29895068, 0.5212166, 0.47059694, 0.63345267]),
    )


def test_gaussian_aic_bic_exact(X, fitted_model):
    np.isclose(fitted_model.aic(X), -4.5465996070890515)
    np.isclose(fitted_model.bic(X), -4.937161694654951)


def test_cdf_exact(
    X,
    fitted_model,
):
    np.testing.assert_almost_equal(
        fitted_model.cdf(X),
        np.array([0.0359141, 0.18021753, 0.24926745, 0.44315041, 0.16167046]),
    )


@pytest.mark.parametrize(
    "first_margin,expected",
    [
        (True, np.array([0.53332908, 0.1393711, 0.61438086, 0.66677303, 0.6049685])),
        (False, np.array([0.15045411, 0.76650108, 0.29340619, 0.36365638, 0.23882845])),
    ],
)
def test_gaussian_partial_derivative_exact(X, fitted_model, first_margin, expected):
    h = fitted_model.partial_derivative(X, first_margin=first_margin)
    np.testing.assert_almost_equal(h, expected)


@pytest.mark.parametrize(
    "first_margin",
    [
        True,
        False,
    ],
)
def test_gaussian_partial_derivative_numeric(X, fitted_model, first_margin):
    h = fitted_model.partial_derivative(X, first_margin=first_margin)

    delta = 1e-6
    i = 0 if first_margin else 1
    X1 = X.copy()
    X1[:, i] += delta
    X2 = X.copy()
    X2[:, i] -= delta

    h_num = (fitted_model.cdf(X1) - fitted_model.cdf(X2)) / delta / 2

    np.testing.assert_almost_equal(h, h_num)


@pytest.mark.parametrize(
    "first_margin,expected",
    [
        (True, np.array([0.01858046, 0.2420715, 0.2832673, 0.55918913, 0.16130203])),
        (False, np.array([0.02255551, 0.20332606, 0.30390676, 0.58119914, 0.17906309])),
    ],
)
def test_gaussian_inverse_partial_derivative_exact(
    X, fitted_model, first_margin, expected
):
    p = fitted_model.inverse_partial_derivative(X, first_margin=first_margin)
    np.testing.assert_almost_equal(p, expected)


@pytest.mark.parametrize(
    "first_margin",
    [
        True,
        False,
    ],
)
def test_gaussian_partial_derivative_inverse_partial_derivative(
    fitted_model, X, first_margin
):
    """h(u | v) = p => h^-1(p | v) = u"""
    p = fitted_model.partial_derivative(X, first_margin=first_margin)
    if first_margin:
        PV = np.stack([X[:, 0], p], axis=1)
    else:
        PV = np.stack([p, X[:, 1]], axis=1)

    u = fitted_model.inverse_partial_derivative(PV, first_margin=first_margin)
    if first_margin:
        np.testing.assert_almost_equal(X[:, 1], u)
    else:
        np.testing.assert_almost_equal(X[:, 0], u)


@pytest.mark.parametrize("itau", [True, False])
def test_gaussian_sample_refitting(X, fitted_model, itau):
    samples = fitted_model.sample(n_samples=int(1e5))
    m = GaussianCopula(itau=itau).fit(samples)
    assert np.isclose(fitted_model.rho_, m.rho_, 1e-2)


def test_itau_bounds():
    X = np.arange(1, 5) / 5
    X = np.stack((X, X)).T
    m = GaussianCopula(itau=True).fit(X)
    assert m.rho_ == _RHO_BOUNDS[1]
    assert not np.isnan(m.score(X))


def test_tail_concentration(fitted_model):
    quantiles = np.linspace(0.01, 0.99, 50)
    tc = fitted_model.tail_concentration(quantiles)
    assert tc.shape == quantiles.shape, "tail_concentration output shape mismatch"
    assert np.all(tc >= 0), "tail_concentration contains negative values"


def test_tail_concentration_raise(fitted_model):
    quantiles = np.linspace(0.01, 1.5, 50)
    with pytest.raises(ValueError, match="quantiles must be between 0.0 and 1.0."):
        _ = fitted_model.tail_concentration(quantiles)


def test_plot_tail_concentration(fitted_model):
    fig = fitted_model.plot_tail_concentration(title="Test Tail Concentration")
    assert isinstance(fig, go.Figure), "plot_tail_concentration did not return a Figure"
    # Check that the title is set
    assert "Tail Concentration" in fig.layout.title.text, (
        "plot_tail_concentration title missing"
    )
    fig = fitted_model.plot_tail_concentration()
    assert "Tail Concentration" in fig.layout.title.text


def test_plot_pdf_2d(fitted_model):
    fig = fitted_model.plot_pdf_2d(title="Test PDF 2D")
    assert isinstance(fig, go.Figure), "plot_pdf_2d did not return a Figure"
    fig = fitted_model.plot_pdf_2d()
    assert "PDF of the Bivariate" in fig.layout.title.text


def test_plot_pdf_3d(fitted_model):
    fig = fitted_model.plot_pdf_3d(title="Test PDF 3D")
    assert isinstance(fig, go.Figure), "plot_pdf_3d did not return a Figure"
    fig = fitted_model.plot_pdf_3d()
    assert "PDF of the Bivariate" in fig.layout.title.text


def test_lower_tail_dependence(fitted_model):
    result = fitted_model.lower_tail_dependence
    assert result == 0.0


def test_upper_tail_dependence(fitted_model):
    result = fitted_model.upper_tail_dependence
    assert result == 0.0


def test_fitted_repr(fitted_model):
    rep = fitted_model.fitted_repr
    assert "GaussianCopula" in rep, "fitted_repr does not contain class name"
    param_str = f"{fitted_model.rho_:0.3f}"
    assert param_str in rep, (
        f"fitted_repr does not contain formatted param: {param_str}"
    )
