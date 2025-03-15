import numpy as np
import plotly.graph_objects as go
import pytest

from skfolio.distribution import CopulaRotation, GumbelCopula
from skfolio.distribution.copula._gumbel import _THETA_BOUNDS


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
    fitted_model = GumbelCopula(random_state=42)
    fitted_model.theta_ = 3.40806952683
    fitted_model.rotation_ = CopulaRotation.R0
    return fitted_model


def test_gumbel_copula_init():
    """Test initialization of GumbelCopula with default parameters."""
    model = GumbelCopula()
    assert model.itau is True
    assert model.kendall_tau is None


def test_gumbel_copula_fit_with_kendall_tau(random_data):
    """Test fit() using Kendall's tau inversion (default)."""
    model = GumbelCopula(itau=True).fit(random_data)
    # Check that theta_ has been fitted and lies within the valid range
    assert hasattr(model, "theta_")
    assert hasattr(model, "rotation_")
    assert _THETA_BOUNDS[0] < model.theta_ < _THETA_BOUNDS[1]


def test_gumbel_copula_fit_mle(random_data):
    """Test fit() using MLE optimization for theta_."""
    model = GumbelCopula(itau=False)
    model.fit(random_data)
    assert hasattr(model, "theta_")
    assert hasattr(model, "rotation_")
    assert _THETA_BOUNDS[0] < model.theta_ < _THETA_BOUNDS[1]


def test_gumbel_partial_derivative_shape(random_data):
    """Test partial_derivative() returns correct shape."""
    model = GumbelCopula().fit(random_data)
    h = model.partial_derivative(random_data)
    assert h.shape == (100,)
    # All values should remain in (0,1) for a well-behaved CDF
    assert np.all(h >= 0) and np.all(h <= 1)


def test_gumbel_inverse_partial_derivative_shape(random_data):
    """Test inverse_partial_derivative() returns correct shape."""
    model = GumbelCopula().fit(random_data)
    h_inv = model.inverse_partial_derivative(random_data)
    assert h_inv.shape == (100,)
    # Should lie within [0,1]
    assert np.all(h_inv >= 0) and np.all(h_inv <= 1)


def test_gumbel_score_samples(random_data):
    """Test score_samples() for shape and type."""
    model = GumbelCopula().fit(random_data)
    log_pdf = model.score_samples(random_data)
    assert log_pdf.shape == (100,)
    # log-pdf can be negative or positive, so we won't do a bound check here.


def test_gumbel_score(random_data):
    """Test the total log-likelihood via score()."""
    model = GumbelCopula().fit(random_data)
    total_ll = model.score(random_data)
    # It's a scalar
    assert isinstance(total_ll, float)


def test_gumbel_aic_bic(random_data):
    """Test AIC and BIC computation."""
    model = GumbelCopula().fit(random_data)
    aic_val = model.aic(random_data)
    bic_val = model.bic(random_data)

    # Both are floats
    assert isinstance(aic_val, float)
    assert isinstance(bic_val, float)

    # Typically, BIC >= AIC for large n, but not guaranteed. Just check they're finite.
    assert np.isfinite(aic_val)
    assert np.isfinite(bic_val)


def test_gumbel_sample():
    """Test sample() method for shape and range."""
    model = GumbelCopula(random_state=123).fit(np.random.rand(100, 2))
    samples = model.sample(n_samples=50)
    assert samples.shape == (50, 2)
    # Should lie strictly in (0,1).
    assert np.all(samples >= 1e-8) and np.all(samples <= 1 - 1e-8)


def test_gumbel_theta_out_of_bounds():
    """Check that setting theta_ out of bounds triggers an error in score_samples()."""
    model = GumbelCopula(itau=False)
    # Manually set theta_ out-of-bounds after fitting
    model.theta_ = 0
    model.rotation_ = CopulaRotation.R0
    with pytest.raises(
        ValueError, match="Theta must be greater than 1 for the Gumbel copula."
    ):
        _ = model.score_samples(np.random.rand(5, 2))


@pytest.mark.parametrize(
    "itau,expected_theta,expected_rotation",
    [
        [True, 4.9999999999, CopulaRotation.R0],
        [False, 3.40806952683, CopulaRotation.R0],
    ],
)
def test_gumbel_fit(X, itau, expected_theta, expected_rotation):
    model = GumbelCopula(itau=itau)
    model.fit(X)
    assert np.isclose(model.theta_, expected_theta)
    assert model.rotation_ == expected_rotation


@pytest.mark.parametrize(
    "rotation,expected",
    [
        (CopulaRotation.R0, [1.41564, 0.19141, 0.70036, 0.71391, 0.63522]),
        (CopulaRotation.R90, [-8.39376, -1.3788, -0.70126, 0.71391, -4.68222]),
        (CopulaRotation.R180, [1.34754, -0.13262, 0.7683, 0.68753, 0.87787]),
        (CopulaRotation.R270, [-7.3659, -1.90492, -0.55808, 0.68753, -5.54506]),
    ],
)
def test_gumbel_score_exact(X, fitted_model, rotation, expected):
    fitted_model.rotation_ = rotation
    assert np.isclose(fitted_model.score(X), np.sum(expected))
    np.testing.assert_almost_equal(fitted_model.score_samples(X), expected, 5)


@pytest.mark.parametrize(
    "rotation,expected",
    [
        (CopulaRotation.R0, [0.03644, 0.18724, 0.26519, 0.46883, 0.79606]),
        (CopulaRotation.R90, [0.0, 0.00187, 0.00423, 0.13117, 0.70002]),
        (CopulaRotation.R180, [0.04768, 0.19478, 0.2764, 0.46478, 0.7834]),
        (CopulaRotation.R270, [0.0, 0.00087, 0.00555, 0.13522, 0.70001]),
    ],
)
def test_cdf_exact(X, fitted_model, rotation, expected):
    fitted_model.rotation_ = rotation
    np.testing.assert_almost_equal(fitted_model.cdf(X), expected, 5)


def test_gumbel_aic_bic_exact(X, fitted_model):
    fitted_model.rotation_ = CopulaRotation.R0
    np.isclose(fitted_model.aic(X), -5.3130643319)
    np.isclose(fitted_model.bic(X), -5.7036264194)


@pytest.mark.parametrize(
    "first_margin,rotation,expected",
    [
        (True, CopulaRotation.R0, [0.57236, 0.10945, 0.69891, 0.75716, 0.13772]),
        (True, CopulaRotation.R90, [1e-05, 0.02054, 0.05669, 0.75716, 0.99951]),
        (True, CopulaRotation.R180, [0.8425, 0.04833, 0.71101, 0.70423, 0.30523]),
        (True, CopulaRotation.R270, [2e-05, 0.00786, 0.05419, 0.70423, 0.99969]),
        (False, CopulaRotation.R0, [0.15186, 0.84992, 0.2716, 0.30256, 0.94395]),
        (False, CopulaRotation.R90, [0.0, 0.02312, 0.03771, 0.69744, 0.99955]),
        (False, CopulaRotation.R180, [0.05907, 0.90286, 0.19928, 0.27599, 0.85336]),
        (False, CopulaRotation.R270, [2e-05, 0.01653, 0.06, 0.72401, 0.99989]),
    ],
)
def test_gumbel_partial_derivative_exact(
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
        (True, CopulaRotation.R0, [0.0185, 0.26175, 0.27333, 0.53089, 0.92534]),
        (True, CopulaRotation.R90, [0.88025, 0.44212, 0.64526, 0.53089, 0.25255]),
        (True, CopulaRotation.R180, [0.02936, 0.30034, 0.28931, 0.55165, 0.93855]),
        (True, CopulaRotation.R270, [0.74415, 0.45837, 0.64545, 0.55165, 0.17365]),
        (False, CopulaRotation.R0, [0.02263, 0.19099, 0.31378, 0.58281, 0.88085]),
        (False, CopulaRotation.R90, [0.58523, 0.73622, 0.51182, 0.41719, 0.39678]),
        (False, CopulaRotation.R180, [0.04752, 0.19606, 0.34241, 0.59932, 0.9207]),
        (False, CopulaRotation.R270, [0.71843, 0.75444, 0.49874, 0.40068, 0.44859]),
    ],
)
def test_gumbel_inverse_partial_derivative_exact(
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
def test_gumbel_partial_derivative_inverse_partial_derivative(
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
    m = GumbelCopula(itau=itau).fit(samples)
    assert np.isclose(fitted_model.theta_, m.theta_, 1e-2)


def test_itau_bounds():
    X = np.arange(1, 5) / 5
    X = np.stack((X, X)).T
    m = GumbelCopula(itau=True).fit(X)
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
    assert "GumbelCopula" in rep, "fitted_repr does not contain class name"
    theta_str = f"theta={fitted_model.theta_:0.2f}"
    assert theta_str in rep, (
        f"fitted_repr does not contain formatted theta: {theta_str}"
    )
    rotation_str = str(fitted_model.rotation_)
    assert rotation_str in rep, f"fitted_repr does not include rotation: {rotation_str}"
