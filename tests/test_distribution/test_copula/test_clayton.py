import numpy as np
import plotly.graph_objects as go
import pytest

from skfolio.distribution import ClaytonCopula, CopulaRotation
from skfolio.distribution.copula._clayton import _THETA_BOUNDS


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
    fitted_model = ClaytonCopula(random_state=42)
    fitted_model.theta_ = 2.718413896014
    fitted_model.rotation_ = CopulaRotation.R0
    return fitted_model


def test_clayton_copula_init():
    """Test initialization of ClaytonCopula with default parameters."""
    model = ClaytonCopula()
    assert model.itau is True
    assert model.kendall_tau is None


def test_clayton_copula_fit_with_kendall_tau(random_data):
    """Test fit() using Kendall's tau inversion (default)."""
    model = ClaytonCopula(itau=True).fit(random_data)
    # Check that theta_ has been fitted and lies within the valid range
    assert hasattr(model, "theta_")
    assert hasattr(model, "rotation_")
    assert _THETA_BOUNDS[0] < model.theta_ < _THETA_BOUNDS[1]


def test_clayton_copula_fit_mle(random_data):
    """Test fit() using MLE optimization for theta_."""
    model = ClaytonCopula(itau=False)
    model.fit(random_data)
    assert hasattr(model, "theta_")
    assert hasattr(model, "rotation_")
    assert _THETA_BOUNDS[0] < model.theta_ < _THETA_BOUNDS[1]


def test_clayton_partial_derivative_shape(random_data):
    """Test partial_derivative() returns correct shape."""
    model = ClaytonCopula().fit(random_data)
    h = model.partial_derivative(random_data)
    assert h.shape == (100,)
    # All values should remain in (0,1) for a well-behaved CDF
    assert np.all(h >= 0) and np.all(h <= 1)


def test_clayton_inverse_partial_derivative_shape(random_data):
    """Test inverse_partial_derivative() returns correct shape."""
    model = ClaytonCopula().fit(random_data)
    h_inv = model.inverse_partial_derivative(random_data)
    assert h_inv.shape == (100,)
    # Should lie within [0,1]
    assert np.all(h_inv >= 0) and np.all(h_inv <= 1)


def test_clayton_score_samples(random_data):
    """Test score_samples() for shape and type."""
    model = ClaytonCopula().fit(random_data)
    log_pdf = model.score_samples(random_data)
    assert log_pdf.shape == (100,)
    # log-pdf can be negative or positive, so we won't do a bound check here.


def test_clayton_score(random_data):
    """Test the total log-likelihood via score()."""
    model = ClaytonCopula().fit(random_data)
    total_ll = model.score(random_data)
    # It's a scalar
    assert isinstance(total_ll, float)


def test_clayton_aic_bic(random_data):
    """Test AIC and BIC computation."""
    model = ClaytonCopula().fit(random_data)
    aic_val = model.aic(random_data)
    bic_val = model.bic(random_data)

    # Both are floats
    assert isinstance(aic_val, float)
    assert isinstance(bic_val, float)

    # Typically, BIC >= AIC for large n, but not guaranteed. Just check they're finite.
    assert np.isfinite(aic_val)
    assert np.isfinite(bic_val)


def test_clayton_sample():
    """Test sample() method for shape and range."""
    model = ClaytonCopula(random_state=123).fit(np.random.rand(100, 2))
    samples = model.sample(n_samples=50)
    assert samples.shape == (50, 2)
    # Should lie strictly in (0,1).
    assert np.all(samples >= 1e-8) and np.all(samples <= 1 - 1e-8)


def test_clayton_theta_out_of_bounds():
    """Check that setting theta_ out of bounds triggers an error in score_samples()."""
    model = ClaytonCopula(itau=False)
    # Manually set theta_ out-of-bounds after fitting
    model.theta_ = 0
    model.rotation_ = CopulaRotation.R0
    with pytest.raises(
        ValueError, match="Theta must be greater than 1 for the Clayton copula."
    ):
        _ = model.score_samples(np.random.rand(5, 2))


@pytest.mark.parametrize(
    "itau,expected_theta,expected_rotation",
    [
        [True, 7.999999999999, CopulaRotation.R180],
        [False, 2.718413896014, CopulaRotation.R0],
    ],
)
def test_clayton_fit(X, itau, expected_theta, expected_rotation):
    model = ClaytonCopula(itau=itau)
    model.fit(X)
    assert np.isclose(model.theta_, expected_theta)
    assert model.rotation_ == expected_rotation


@pytest.mark.parametrize(
    "rotation,expected",
    [
        (CopulaRotation.R0, [1.39729, 0.03641, 0.61795, 0.43688, 0.70493]),
        (CopulaRotation.R90, [-4.75603, -1.25039, -0.15239, 0.43688, -4.12012]),
        (CopulaRotation.R180, [0.96547, 0.30684, 0.44028, 0.49993, 0.70748]),
        (CopulaRotation.R270, [-6.4388, -0.50604, -0.31586, 0.49993, -2.67991]),
    ],
)
def test_clayton_score_exact(X, fitted_model, rotation, expected):
    fitted_model.rotation_ = rotation
    assert np.isclose(fitted_model.score(X), np.sum(expected))
    np.testing.assert_almost_equal(fitted_model.score_samples(X), expected, 5)


@pytest.mark.parametrize(
    "rotation,expected",
    [
        (CopulaRotation.R0, [0.04747, 0.19063, 0.26372, 0.43532, 0.75255]),
        (CopulaRotation.R90, [1e-05, 0.00272, 0.01828, 0.16468, 0.70006]),
        (CopulaRotation.R180, [0.01543, 0.1597, 0.229, 0.44781, 0.79499]),
        (CopulaRotation.R270, [0.0, 0.00971, 0.01168, 0.15219, 0.70031]),
    ],
)
def test_cdf_exact(X, fitted_model, rotation, expected):
    fitted_model.rotation_ = rotation
    np.testing.assert_almost_equal(fitted_model.cdf(X), expected, 5)


def test_clayton_aic_bic_exact(X, fitted_model):
    fitted_model.rotation_ = CopulaRotation.R180
    np.isclose(fitted_model.aic(X), -3.840007906839)
    np.isclose(fitted_model.bic(X), -4.2305699944)


@pytest.mark.parametrize(
    "first_margin,rotation,expected",
    [
        (True, CopulaRotation.R0, [0.82437, 0.06355, 0.61925, 0.59746, 0.51411]),
        (True, CopulaRotation.R90, [0.00023, 0.01599, 0.10489, 0.59746, 0.99782]),
        (True, CopulaRotation.R180, [0.29298, 0.22782, 0.64707, 0.74066, 0.17404]),
        (True, CopulaRotation.R270, [0.00013, 0.08735, 0.13733, 0.74066, 0.9963]),
        (False, CopulaRotation.R0, [0.06263, 0.83656, 0.21247, 0.30331, 0.79665]),
        (False, CopulaRotation.R90, [0.00039, 0.04958, 0.15965, 0.69669, 0.99956]),
        (False, CopulaRotation.R180, [0.13555, 0.73506, 0.37393, 0.4054, 0.93725]),
        (False, CopulaRotation.R270, [2e-05, 0.06933, 0.06554, 0.5946, 0.99432]),
    ],
)
def test_clayton_partial_derivative_exact(
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
        (True, CopulaRotation.R0, [0.02903, 0.29321, 0.3009, 0.60164, 0.92503]),
        (True, CopulaRotation.R90, [0.51606, 0.42877, 0.62973, 0.60164, 0.18826]),
        (True, CopulaRotation.R180, [0.03187, 0.17918, 0.25099, 0.51818, 0.92574]),
        (True, CopulaRotation.R270, [0.8736, 0.34348, 0.61017, 0.51818, 0.39878]),
        (False, CopulaRotation.R0, [0.04667, 0.20252, 0.34507, 0.62136, 0.95084]),
        (False, CopulaRotation.R90, [0.6736, 0.735, 0.42941, 0.37864, 0.5551]),
        (False, CopulaRotation.R180, [0.01809, 0.19939, 0.25107, 0.55375, 0.884]),
        (False, CopulaRotation.R270, [0.40632, 0.68929, 0.49786, 0.44625, 0.47982]),
    ],
)
def test_clayton_inverse_partial_derivative_exact(
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
def test_clayton_partial_derivative_inverse_partial_derivative(
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
    m = ClaytonCopula(itau=itau).fit(samples)
    assert np.isclose(fitted_model.theta_, m.theta_, 1e-2)


def test_itau_bounds():
    X = np.arange(1, 5) / 5
    X = np.stack((X, X)).T
    m = ClaytonCopula(itau=True).fit(X)
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
    expected = np.power(2.0, -1.0 / fitted_model.theta_)
    result = fitted_model.lower_tail_dependence
    assert np.isclose(result, expected)


def test_upper_tail_dependence(fitted_model):
    result = fitted_model.upper_tail_dependence
    assert result == 0.0


def test_fitted_repr(fitted_model):
    rep = fitted_model.fitted_repr
    assert "ClaytonCopula" in rep, "fitted_repr does not contain class name"
    theta_str = f"theta={fitted_model.theta_:0.2f}"
    assert theta_str in rep, (
        f"fitted_repr does not contain formatted theta: {theta_str}"
    )
    rotation_str = str(fitted_model.rotation_)
    assert rotation_str in rep, f"fitted_repr does not include rotation: {rotation_str}"
