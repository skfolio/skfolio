import numpy as np
import plotly.graph_objects as go
import pytest

from skfolio.distribution import StudentTCopula
from skfolio.distribution.copula._base import _RHO_BOUNDS
from skfolio.distribution.copula._student_t import _DOF_BOUNDS


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
    fitted_model = StudentTCopula(random_state=42)
    fitted_model.rho_ = 0.8090169943749475
    fitted_model.dof_ = 2.918296662088029
    return fitted_model


def test_student_t_copula_init():
    """Test initialization of StudentTCopula with default parameters."""
    model = StudentTCopula()
    assert model.itau is True
    assert model.kendall_tau is None


def test_student_t_copula_fit_with_kendall_tau(random_data):
    """Test fit() using Kendall's tau inversion (default)."""
    model = StudentTCopula(itau=True).fit(random_data)
    # Check that rho_ has been fitted and lies within the valid range
    assert hasattr(model, "rho_")
    assert hasattr(model, "dof_")
    assert _RHO_BOUNDS[0] < model.rho_ < _RHO_BOUNDS[1]
    assert _DOF_BOUNDS[0] < model.dof_ < _DOF_BOUNDS[1]


def test_student_t_copula_fit_with_provided_kendall_tau(random_data):
    """Test fit() using a manually provided Kendall's tau."""
    model = StudentTCopula(itau=True, kendall_tau=0.25)
    model.fit(random_data)
    # Check that the resulting rho_ is sin(pi * 0.25 / 2) = sin(pi/8)
    expected_rho = np.sin(np.pi * 0.25 / 2.0)
    assert np.isclose(model.rho_, expected_rho)


def test_student_t_copula_fit_mle(random_data):
    """Test fit() using MLE optimization for rho_."""
    model = StudentTCopula(itau=False)
    model.fit(random_data)
    # Just check it's between -1 and 1
    assert -1 < model.rho_ < 1
    assert 1 <= model.dof_ <= 50


def test_student_t_copula_fit_mle_with_provided_kendall_tau(random_data):
    """Test fit() using a manually provided Kendall's tau."""
    model = StudentTCopula(itau=False, kendall_tau=0.25)
    model.fit(random_data)
    assert -1 < model.rho_ < 1
    assert 1 <= model.dof_ <= 50


def test_student_t_cdf_shape(random_data):
    """Test cdf() returns correct shape."""
    model = StudentTCopula().fit(random_data)
    cdf = model.cdf(random_data)
    assert cdf.shape == (100,)
    # All values should remain in (0,1) for a well-behaved CDF
    assert np.all(cdf >= 0) and np.all(cdf <= 1)


def test_student_t_shape(random_data):
    """Test cdf() returns correct shape."""
    model = StudentTCopula().fit(random_data)
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
def test_student_t_partial_derivative_shape(random_data, first_margin):
    """Test partial_derivative() returns correct shape."""
    model = StudentTCopula().fit(random_data)
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
def test_student_t_inverse_partial_derivative_shape(random_data, first_margin):
    """Test inverse_partial_derivative() returns correct shape."""
    model = StudentTCopula().fit(random_data)
    h_inv = model.inverse_partial_derivative(random_data, first_margin=first_margin)
    assert h_inv.shape == (100,)
    # Should lie within [0,1]
    assert np.all(h_inv >= 0) and np.all(h_inv <= 1)


def test_student_t_score_samples(random_data):
    """Test score_samples() for shape and type."""
    model = StudentTCopula().fit(random_data)
    log_pdf = model.score_samples(random_data)
    assert log_pdf.shape == (100,)
    # log-pdf can be negative or positive, so we won't do a bound check here.


def test_student_t_score(random_data):
    """Test the total log-likelihood via score()."""
    model = StudentTCopula().fit(random_data)
    total_ll = model.score(random_data)
    # It's a scalar
    assert isinstance(total_ll, float)


def test_student_t_aic_bic(random_data):
    """Test AIC and BIC computation."""
    model = StudentTCopula().fit(random_data)
    aic_val = model.aic(random_data)
    bic_val = model.bic(random_data)

    # Both are floats
    assert isinstance(aic_val, float)
    assert isinstance(bic_val, float)

    # Typically, BIC >= AIC for large n, but not guaranteed. Just check they're finite.
    assert np.isfinite(aic_val)
    assert np.isfinite(bic_val)


def test_student_t_sample():
    """Test sample() method for shape and range."""
    model = StudentTCopula(random_state=123).fit(np.random.rand(100, 2))
    samples = model.sample(
        n_samples=50,
    )
    assert samples.shape == (50, 2)
    # Should lie strictly in (0,1).
    assert np.all(samples >= 1e-8) and np.all(samples <= 1 - 1e-8)


def test_student_t_rho_out_of_bounds():
    """Check that setting rho_ out of bounds triggers an error in score_samples()."""
    model = StudentTCopula(itau=False)
    # Manually set rho_ out-of-bounds after fitting
    model.rho_ = 1.2
    model.dof_ = 3.0
    with pytest.raises(ValueError, match="rho must be between -1 and 1."):
        _ = model.score_samples(np.random.rand(5, 2))


def test_student_t_dof_out_of_bounds():
    """Check that setting rho_ out of bounds triggers an error in score_samples()."""
    model = StudentTCopula(itau=False)
    # Manually set rho_ out-of-bounds after fitting
    model.rho_ = 0.5
    model.dof_ = 0.3
    with pytest.raises(
        ValueError, match="Degrees of freedom `dof` must be between 1 and 50."
    ):
        _ = model.score_samples(np.random.rand(5, 2))


@pytest.mark.parametrize(
    "itau,expected_rho,expected_dof",
    [
        [True, 0.8090169943749475, 2.918296662088029],
        [False, 0.9254270224728093, 50.0],
    ],
)
def test_student_t_fit(X, itau, expected_rho, expected_dof):
    model = StudentTCopula(itau=itau, tolerance=1e-5)
    model.fit(X)
    assert np.isclose(model.rho_, expected_rho)
    assert np.isclose(model.dof_, expected_dof)


def test_student_t_score_exact(X, fitted_model):
    assert np.isclose(fitted_model.score(X), 3.42716024087639)
    np.testing.assert_almost_equal(
        fitted_model.score_samples(X),
        np.array([1.38938848, 0.15984856, 0.61188097, 0.56991331, 0.69612892]),
    )


def test_student_t_aic_bic_exact(X, fitted_model):
    np.isclose(fitted_model.aic(X), -2.8543204817527803)
    np.isclose(fitted_model.bic(X), -3.6354446568845797)


def test_cdf_exact(
    X,
    fitted_model,
):
    np.testing.assert_almost_equal(
        fitted_model.cdf(X),
        np.array([0.03933961, 0.17777323, 0.24908186, 0.44199897, 0.1629778]),
        4,
    )


@pytest.mark.parametrize(
    "first_margin,expected",
    [
        (True, np.array([0.61622819, 0.10773059, 0.63375585, 0.6928992, 0.6302627])),
        (False, np.array([0.10692092, 0.7863787, 0.26049836, 0.34268922, 0.19885133])),
    ],
)
def test_student_t_partial_derivative_exact(X, fitted_model, first_margin, expected):
    h = fitted_model.partial_derivative(X, first_margin=first_margin)
    np.testing.assert_almost_equal(h, expected)


@pytest.mark.parametrize(
    "first_margin",
    [
        True,
        False,
    ],
)
def test_student_t_partial_derivative_numeric(X, fitted_model, first_margin):
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
    "first_margin,expected",
    [
        (True, np.array([0.02441904, 0.26535277, 0.29025729, 0.55021996, 0.17163888])),
        (False, np.array([0.03369421, 0.20617276, 0.32085672, 0.58137266, 0.20057199])),
    ],
)
def test_student_t_inverse_partial_derivative_exact(
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
def test_student_t_partial_derivative_inverse_partial_derivative(
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
    samples = fitted_model.sample(n_samples=int(1e4))
    m = StudentTCopula(itau=itau).fit(samples)
    assert np.isclose(fitted_model.rho_, m.rho_, 1e-2)
    assert np.isclose(fitted_model.dof_, m.dof_, 1e-1)


def test_itau_bounds():
    X = np.arange(1, 5) / 5
    X = np.stack((X, X)).T
    m = StudentTCopula(itau=True).fit(X)
    assert m.rho_ == _RHO_BOUNDS[1]
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
    assert np.isclose(result, 0.555818574869954)


def test_upper_tail_dependence(fitted_model):
    result = fitted_model.upper_tail_dependence
    assert np.isclose(result, 0.555818574869954)


def test_fitted_repr(fitted_model):
    rep = fitted_model.fitted_repr
    assert "StudentTCopula" in rep, "fitted_repr does not contain class name"
    rho_str = f"rho={fitted_model.rho_:0.3f}"
    assert rho_str in rep, f"fitted_repr does not contain formatted rho: {rho_str}"
    dof_str = f"dof={fitted_model.dof_:0.2f}"
    assert dof_str in rep, f"fitted_repr does not contain formatted dof: {dof_str}"
