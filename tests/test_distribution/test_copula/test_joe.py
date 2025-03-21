import numpy as np
import plotly.graph_objects as go
import pytest

from skfolio.distribution import CopulaRotation, JoeCopula
from skfolio.distribution.copula._joe import _THETA_BOUNDS


@pytest.fixture
def X():
    X = np.array(
        [
            [0.05, 0.1],
            [0.4, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.9, 0.8],
        ]
    )
    return X


@pytest.fixture
def fitted_model():
    # Using same convention as other libraries for Benchmark
    fitted_model = JoeCopula(random_state=42)
    fitted_model.theta_ = 3.4781710077039865
    fitted_model.rotation_ = CopulaRotation.R0
    return fitted_model


def test_joe_copula_init():
    """Test initialization of JoeCopula with default parameters."""
    model = JoeCopula()
    assert model.itau is True
    assert model.kendall_tau is None


def test_joe_copula_fit_with_kendall_tau(random_data):
    """Test fit() using Kendall's tau inversion (default)."""
    model = JoeCopula(itau=True).fit(random_data)
    # Check that theta_ has been fitted and lies within the valid range
    assert hasattr(model, "theta_")
    assert hasattr(model, "rotation_")
    assert _THETA_BOUNDS[0] < model.theta_ < _THETA_BOUNDS[1]


def test_joe_copula_fit_mle(random_data):
    """Test fit() using MLE optimization for theta_."""
    model = JoeCopula(itau=False)
    model.fit(random_data)
    assert hasattr(model, "theta_")
    assert hasattr(model, "rotation_")
    assert _THETA_BOUNDS[0] < model.theta_ < _THETA_BOUNDS[1]


def test_joe_partial_derivative_shape(random_data):
    """Test partial_derivative() returns correct shape."""
    model = JoeCopula().fit(random_data)
    h = model.partial_derivative(random_data)
    assert h.shape == (100,)
    # All values should remain in (0,1) for a well-behaved CDF
    assert np.all(h >= 0) and np.all(h <= 1)


def test_joe_inverse_partial_derivative_shape(random_data):
    """Test inverse_partial_derivative() returns correct shape."""
    model = JoeCopula().fit(random_data)
    h_inv = model.inverse_partial_derivative(random_data)
    assert h_inv.shape == (100,)
    # Should lie within [0,1]
    assert np.all(h_inv >= 0) and np.all(h_inv <= 1)


def test_joe_score_samples(random_data):
    """Test score_samples() for shape and type."""
    model = JoeCopula().fit(random_data)
    log_pdf = model.score_samples(random_data)
    assert log_pdf.shape == (100,)
    # log-pdf can be negative or positive, so we won't do a bound check here.


def test_joe_score(random_data):
    """Test the total log-likelihood via score()."""
    model = JoeCopula().fit(random_data)
    total_ll = model.score(random_data)
    # It's a scalar
    assert isinstance(total_ll, float)


def test_joe_aic_bic(random_data):
    """Test AIC and BIC computation."""
    model = JoeCopula().fit(random_data)
    aic_val = model.aic(random_data)
    bic_val = model.bic(random_data)

    # Both are floats
    assert isinstance(aic_val, float)
    assert isinstance(bic_val, float)

    # Typically, BIC >= AIC for large n, but not guaranteed. Just check they're finite.
    assert np.isfinite(aic_val)
    assert np.isfinite(bic_val)


def test_joe_sample():
    """Test sample() method for shape and range."""
    model = JoeCopula(random_state=123).fit(np.random.rand(100, 2))
    samples = model.sample(n_samples=50)
    assert samples.shape == (50, 2)
    # Should lie strictly in (0,1).
    assert np.all(samples >= 1e-8) and np.all(samples <= 1 - 1e-8)


def test_joe_theta_out_of_bounds():
    """Check that setting theta_ out of bounds triggers an error in score_samples()."""
    model = JoeCopula(itau=False)
    # Manually set theta_ out-of-bounds after fitting
    model.theta_ = 0.5
    model.rotation_ = CopulaRotation.R0
    with pytest.raises(
        ValueError, match="Theta must be greater than 1 for the Joe copula."
    ):
        _ = model.score_samples(np.random.rand(5, 2))


@pytest.mark.parametrize(
    "itau,expected_theta,expected_rotation",
    [
        [True, 8.767706807353818, CopulaRotation.R0],
        [False, 3.4781710077039865, CopulaRotation.R180],
    ],
)
def test_joe_fit(X, itau, expected_theta, expected_rotation):
    model = JoeCopula(itau=itau)
    model.fit(X)
    assert np.isclose(model.theta_, expected_theta)
    assert model.rotation_ == expected_rotation


@pytest.mark.parametrize(
    "rotation,expected",
    [
        (
            CopulaRotation.R0,
            np.array([0.93185286, 0.30864155, 0.43967997, 0.49862684, 0.65420809]),
        ),
        (
            CopulaRotation.R90,
            np.array([-5.90333622, -0.49019459, -0.35173268, 0.49862684, -2.47029007]),
        ),
        (
            CopulaRotation.R180,
            np.array([1.34539806, -0.01732094, 0.61611089, 0.43511225, 0.69371343]),
        ),
        (
            CopulaRotation.R270,
            np.array([-4.32949845, -1.26787994, -0.16722499, 0.43511225, -3.85282546]),
        ),
    ],
)
def test_joe_score_exact(X, fitted_model, rotation, expected):
    fitted_model.rotation_ = rotation
    assert np.isclose(fitted_model.score(X), np.sum(expected))
    np.testing.assert_almost_equal(fitted_model.score_samples(X), expected)


@pytest.mark.parametrize(
    "rotation,expected",
    [
        (
            CopulaRotation.R0,
            [0.01467877, 0.15725618, 0.22640759, 0.44708776, 0.79501503],
        ),
        (
            CopulaRotation.R90,
            [
                3.41756792e-06,
                1.09545891e-02,
                1.25330869e-02,
                1.52912245e-01,
                7.00424163e-01,
            ],
        ),
        (
            CopulaRotation.R180,
            [0.04749933, 0.19039541, 0.26372482, 0.43359505, 0.75072786],
        ),
        (
            CopulaRotation.R270,
            [
                1.77386069e-05,
                3.11887404e-03,
                1.97263140e-02,
                1.66404953e-01,
                7.00089707e-01,
            ],
        ),
    ],
)
def test_cdf_exact(X, fitted_model, rotation, expected):
    fitted_model.rotation_ = rotation
    np.testing.assert_almost_equal(fitted_model.cdf(X), expected)


def test_joe_aic_bic_exact(X, fitted_model):
    fitted_model.rotation_ = CopulaRotation.R180
    np.isclose(fitted_model.aic(X), -4.146027394639978)
    np.isclose(fitted_model.bic(X), -4.536589482205877)


@pytest.mark.parametrize(
    "first_margin,rotation,expected",
    [
        (True, CopulaRotation.R0, [0.28028, 0.2326, 0.64851, 0.74718, 0.16822]),
        (True, CopulaRotation.R90, [0.00024, 0.09367, 0.14166, 0.74718, 0.99513]),
        (True, CopulaRotation.R180, [0.83124, 0.0606, 0.621, 0.59233, 0.52698]),
        (True, CopulaRotation.R270, [0.00038, 0.01642, 0.1051, 0.59233, 0.99688]),
        (False, CopulaRotation.R0, [0.13055, 0.73027, 0.37865, 0.40808, 0.9405]),
        (False, CopulaRotation.R90, [4e-05, 0.07307, 0.06437, 0.59192, 0.99263]),
        (False, CopulaRotation.R180, [0.0594, 0.83777, 0.2058, 0.29823, 0.79921]),
        (False, CopulaRotation.R270, [0.00062, 0.05389, 0.16578, 0.70177, 0.99939]),
    ],
)
def test_joe_partial_derivative_exact(
    X, fitted_model, first_margin, rotation, expected
):
    fitted_model.rotation_ = rotation
    h = fitted_model.partial_derivative(X, first_margin=first_margin)
    np.testing.assert_almost_equal(h, expected, 5)


@pytest.mark.parametrize(
    "first_margin",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize("rotation", list(CopulaRotation))
def test_student_t_partial_derivative_numeric(X, fitted_model, first_margin, rotation):
    fitted_model.rotation_ = rotation
    h = fitted_model.partial_derivative(X, first_margin=first_margin)

    delta = 1e-2
    i = 0 if first_margin else 1
    X1 = X.copy()
    X1[:, i] += delta
    X2 = X.copy()
    X2[:, i] -= delta

    h_num = (fitted_model.cdf(X1) - fitted_model.cdf(X2)) / delta / 2

    np.testing.assert_almost_equal(h, h_num, 2)


@pytest.mark.parametrize(
    "first_margin,rotation,expected",
    [
        (True, CopulaRotation.R0, [0.0336, 0.17561, 0.25016, 0.51483, 0.925]),
        (True, CopulaRotation.R90, [0.87488, 0.3379, 0.61106, 0.51483, 0.40953]),
        (True, CopulaRotation.R180, [0.02949, 0.29685, 0.30165, 0.60498, 0.92218]),
        (True, CopulaRotation.R270, [0.49865, 0.4321, 0.63062, 0.60498, 0.18525]),
        (False, CopulaRotation.R0, [0.01884, 0.20163, 0.24779, 0.55208, 0.88212]),
        (False, CopulaRotation.R90, [0.39201, 0.68633, 0.50114, 0.44792, 0.48386]),
        (False, CopulaRotation.R180, [0.04742, 0.20232, 0.34807, 0.62429, 0.95019]),
        (False, CopulaRotation.R270, [0.66938, 0.73737, 0.42668, 0.37571, 0.56049]),
    ],
)
def test_joe_inverse_partial_derivative_exact(
    X, fitted_model, first_margin, rotation, expected
):
    fitted_model.rotation_ = rotation
    p = fitted_model.inverse_partial_derivative(X, first_margin=first_margin)
    np.testing.assert_almost_equal(p, expected, 5)


@pytest.mark.parametrize(
    "first_margin",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize("rotation", list(CopulaRotation))
def test_joe_partial_derivative_inverse_partial_derivative(
    X, fitted_model, first_margin, rotation
):
    """h(u | v) = p => h^-1(p | v) = u"""
    fitted_model.rotation_ = rotation
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
@pytest.mark.parametrize("rotation", list(CopulaRotation))
def test_clayton_sample_refitting(X, fitted_model, itau, rotation):
    fitted_model.rotation_ = rotation
    samples = fitted_model.sample(n_samples=int(1e5))
    m = JoeCopula(itau=itau).fit(samples)
    assert np.isclose(fitted_model.theta_, m.theta_, 1e-2)


def test_itau_bounds():
    X = np.arange(1, 5) / 5
    X = np.stack((X, X)).T
    m = JoeCopula(itau=True).fit(X)
    assert m.theta_ == _THETA_BOUNDS[1]
    assert not np.isnan(m.score(X))


def test_tail_concentration(fitted_model):
    quantiles = np.linspace(0.01, 0.99, 50)
    tc = fitted_model.tail_concentration(quantiles)
    assert tc.shape == quantiles.shape, "tail_concentration output shape mismatch"
    assert np.all(tc >= 0), "tail_concentration contains negative values"


def test_plot_tail_concentration(fitted_model):
    fig = fitted_model.plot_tail_concentration(title="Test Tail Concentration")
    assert isinstance(fig, go.Figure), "plot_tail_concentration did not return a Figure"
    # Check that the title is set
    assert "Tail Concentration" in fig.layout.title.text, (
        "plot_tail_concentration title missing"
    )


def test_plot_pdf_2d(fitted_model):
    fig = fitted_model.plot_pdf_2d(title="Test PDF 2D")
    assert isinstance(fig, go.Figure), "plot_pdf_2d did not return a Figure"


# Test plot_pdf_3d
def test_plot_pdf_3d(fitted_model):
    fig = fitted_model.plot_pdf_3d(title="Test PDF 3D")
    assert isinstance(fig, go.Figure), "plot_pdf_3d did not return a Figure"


def test_lower_tail_dependence(fitted_model):
    result = fitted_model.lower_tail_dependence
    assert result == 0.0


def test_upper_tail_dependence(fitted_model):
    expected = 2.0 - np.power(2.0, 1.0 / fitted_model.theta_)
    result = fitted_model.upper_tail_dependence
    assert np.isclose(result, expected)


def test_fitted_repr(fitted_model):
    rep = fitted_model.fitted_repr
    assert "JoeCopula" in rep, "fitted_repr does not contain class name"
    theta_str = f"theta={fitted_model.theta_:0.2f}"
    assert theta_str in rep, (
        f"fitted_repr does not contain formatted theta: {theta_str}"
    )
    rotation_str = str(fitted_model.rotation_)
    assert rotation_str in rep, f"fitted_repr does not include rotation: {rotation_str}"
