r"""
=================
Bivariate Copulas
=================

This tutorial introduces Bivariate Copulas estimators that are the building blocks of
:class:`~skfolio.distribution.VineCopula`.

Introduction
============

Bivariate copulas are mathematical functions that allow us to construct a joint
distribution by combining the individual marginal distributions of two variables
with a separate model for their dependence structure. This approach enables the
marginal behavior of each variable to be modeled independently, while the copula
captures how these variables move together.

There are two primary families of copulas used in finance:

1. **Elliptical Copulas:**

   - **Gaussian Copula:**
     Based on the multivariate normal distribution with a correlation parameter
     :math:`\rho \in [-1, 1]`. It is symmetric but does not exhibit tail dependence.

   - **Student's t Copula:**
     Derived from the multivariate Student's t-distribution with :math:`\rho \in [-1, 1]`
     and degrees of freedom :math:`\nu > 2`. It captures symmetric tail dependence,
     making it more suitable for modeling extreme co-movements.

2. **Archimedean Copulas:**

   - **Gumbel Copula:**
     Characterized by a parameter :math:`\theta \in [1, \infty)` and models upper tail
     dependence.

   - **Clayton Copula:**
     Uses a parameter :math:`\theta \in (0, \infty)` and capture lower tail dependence.

   - **Joe Copula:**
     Defined with :math:`\theta \in [1, \infty)` and models upper tail dependence.

+-----------------+----------------+-------------------------------------------+---------------------------------------------+---------------+
| **Copula**      | **Family**     | **Parameters**                            | **Tail Dependence**                         | **Symmetry**  |
+=================+================+===========================================+=============================================+===============+
| Gaussian        | Elliptical     | :math:`\rho \in [-1, 1]`                  | None                                        | Symmetric     |
+-----------------+----------------+-------------------------------------------+---------------------------------------------+---------------+
| Student's t     | Elliptical     | :math:`\rho \in [-1, 1]`; :math:`\nu > 2` | Both upper and lower tail dependence        | Symmetric     |
+-----------------+----------------+-------------------------------------------+---------------------------------------------+---------------+
| Gumbel          | Archimedean    | :math:`\theta \in [1, \infty)`            | Upper tail dependence                       | Asymmetric    |
+-----------------+----------------+-------------------------------------------+---------------------------------------------+---------------+
| Clayton         | Archimedean    | :math:`\theta \in (0, \infty)`            | Strong lower tail dependence                | Asymmetric    |
+-----------------+----------------+-------------------------------------------+---------------------------------------------+---------------+
| Joe             | Archimedean    | :math:`\theta \in [1, \infty)`            | Strong upper tail dependence                | Asymmetric    |
+-----------------+----------------+-------------------------------------------+---------------------------------------------+---------------+

Rotation of Archimedean Copulas
===============================
Standard Archimedean copulas are inherently designed to capture dependence in one
specific tail:

- **Gumbel and Joe:** Naturally capture upper tail dependence.
- **Clayton:** Naturally captures lower tail dependence.

However, financial data may exhibit tail behavior opposite to a copula’s inherent design,
or even negative dependence. **Rotation** is a transformation that adjusts the copula to
model the opposite tail. In effect, rotation swaps the roles of the upper and lower tails,
enabling the model to capture tail dependence where it is most relevant.

Available rotations include 0° (unrotated), 90°, 180°, and 270°. During the fitting
process, both the copula parameters and the optimal rotation are estimated.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` and select Bank of America (BAC) and
# JPMorgan (JPM) stocks starting from 1990-01-02 up to 2022-12-28:
import numpy as np
from plotly.io import show

from skfolio.datasets import load_sp500_dataset
from skfolio.distribution import (
    ClaytonCopula,
    Gaussian,
    GaussianCopula,
    GumbelCopula,
    JoeCopula,
    JohnsonSU,
    StudentT,
    StudentTCopula,
    select_bivariate_copula,
    select_univariate_dist,
)
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
prices = prices[["BAC", "JPM"]]
X = prices_to_returns(prices)
print(X.tail())

# %%
# Marginal Distribution
# =====================
# First, we fit the marginal distributions for each asset independently.
# We use the utility function `select_univariate_dist` to select the optimal
# univariate distribution based on information criterion (BIC or AIC).
candidates = [Gaussian(), StudentT(), JohnsonSU()]
X1, X2 = X[["BAC"]], X[["JPM"]]

bac_dist = select_univariate_dist(X=X1, distribution_candidates=candidates)
print(f"BAC: {bac_dist.fitted_repr}")

jpm_dist = select_univariate_dist(X=X2, distribution_candidates=candidates)
print(f"JPM: {jpm_dist.fitted_repr}")

# %%
# Let's plot the PDF of the fitted distribution versus the historical data:
jpm_dist.plot_pdf(X2)

# %%
# Let's analyse the Q-Q plot:
jpm_dist.qq_plot(X2)

# %%
# Let's explore the difference versus the Gaussian distribution:
gaussian = Gaussian()
gaussian.fit(X2)
gaussian.plot_pdf(X2)

# %%
gaussian.qq_plot(X2)

# %%
# The Q-Q plot and zooming in on the tail of the Johnson Su distribution shows that its
# fat tails are well captured, which is not the case for the Gaussian distribution.

# %%
# Uniform Marginals
# =================
# Before working with copulas, we transform the asset returns into uniform marginals
# using their univariate CDFs:
X = np.hstack([bac_dist.cdf(X1), jpm_dist.cdf(X2)])

# %%
# Bivariate Copulas
# =================
# We use the utility function `select_bivariate_copula` to select the optimal
# bivariate copula based on information criterion (BIC or AIC).
candidates = [
    GaussianCopula(),
    StudentTCopula(),
    ClaytonCopula(),
    GumbelCopula(),
    JoeCopula(),
]
copula = select_bivariate_copula(X, copula_candidates=candidates)
print(copula.fitted_repr)
print(f"AIC: {copula.aic(X):,.2f}")

# %%
# Let's plot the 2D probability density function (PDF) of the Student-t copula.
# The x-axis and y-axis represent the uniform variates (u and v) obtained by applying
# the marginal cumulative distribution functions (CDFs) to the returns of BAC and JPM,
# respectively.
fig = copula.plot_pdf_2d()
fig.update_layout(height=700)

# %%
# Each contour line connects points of equal copula density, effectively delineating
# regions with the same level of joint dependence between the two assets. Areas where
# the contours are closely spaced indicate a rapid change in density, reflecting regions
# with higher concentrations of joint probability.
#
# The Student-t copula captures tail dependence: extreme positive or negative returns
# in one asset tend to occur simultaneously with extreme returns in the other. This is
# visualized as pronounced "bulges" in the tail regions of the contour plot.
#
# Note: The copula density is defined on the scale of the transformed uniform marginals.
# To recover the joint density of the original asset returns, the copula density must
# be multiplied by the marginal probability density functions, which take on lower
# values in the tails.
#
# Now, let's plot the 3D PDF:
fig = copula.plot_pdf_3d()
fig.update_layout(scene_camera=dict(eye=dict(x=-1.2, y=1.4, z=0.8)))
fig

# %%
# Tail Dependencies
# -----------------
# For a bivariate random vector :math:`(X,Y)` with marginal distribution functions
# :math:`F_X` and :math:`F_Y`, the tail dependence coefficients quantify the probability
# that one variable is extreme given that the other is extreme. They are defined as
# follows:
#
# Upper Tail Dependence Coefficient:
#
# .. math::
#     \lambda_U = \lim_{u \to 1^-} P\left( Y > F_Y^{-1}(u) \mid X > F_X^{-1}(u) \right)
#
# Lower Tail Dependence Coefficient:
#
# .. math::
#     \lambda_L = \lim_{u \to 0^+} P\left( Y \le F_Y^{-1}(u) \mid X \le F_X^{-1}(u) \right)
#
# A positive :math:`\lambda_U` (or :math:`\lambda_L`) indicates that extreme high
# (or low) values of :math:`X` and :math:`Y` occur together more frequently than if
# the variables were independent.
#
# Let's print the Student's t Copula tail dependence.
# The model indicates a tail dependence coefficient of approximately 51%, suggesting
# a relatively high likelihood that extreme returns (both negative and positive) occur
# simultaneously for the assets. Since the Student's t copula is symmetric, the tail
# dependence is the same for both the lower and upper tails.
print(f"Lower Tail Dependence: {copula.lower_tail_dependence:.2%}")
print(f"Upper Tail Dependence: {copula.upper_tail_dependence:.2%}")

# %%
# Tail Concentration
# --------------------
# Tail concentration refers to how much probability mass is concentrated at the tails
# of a distribution.
# Let's plot the tail concentration of the copula model versus the historical data:
copula.plot_tail_concentration(X)

# %%
# Comparison with the Gaussian Copula
# ===================================
# Let's explore the difference versus the Gaussian Copula:
copula = GaussianCopula()
copula.fit(X)
print(copula.fitted_repr)
print(f"Rho: {copula.rho_:0.2f}")
print(f"AIC: {copula.aic(X):,.2f}")
print(f"Lower Tail Dependence: {copula.lower_tail_dependence:.2%}")
print(f"Upper Tail Dependence: {copula.upper_tail_dependence:.2%}")

# %%
# Let's plot the 2D PDF:
fig = copula.plot_pdf_2d()
fig.update_layout(height=700)

# %%
# Let's plot the tail concentration of the copula model versus the historical data:
copula.plot_tail_concentration(X)

# %%
# As expected, the tail concentration plot shows that the Gaussian Copula cannot capture
# the tail dependencies of the historical data.

# %%
# Comparison with the Joe Copula
# ==============================
# Let's now compare with the Joe Copula:
copula = JoeCopula()
copula.fit(X)
print(copula.fitted_repr)
print(f"Rotation: {copula.rotation_}")
print(f"Rho: {copula.theta_:0.2f}")
print(f"AIC: {copula.aic(X):,.2f}")
print(f"Lower Tail Dependence: {copula.lower_tail_dependence:.2%}")
print(f"Upper Tail Dependence: {copula.upper_tail_dependence:.2%}")

# %%
# Let's plot the 2D PDF:
fig = copula.plot_pdf_2d()
fig.update_layout(height=700)
show(fig)

# %%
# |
# Let's plot the tail concentration of the copula model versus the historical data:
copula.plot_tail_concentration(X)

# %%
# The Joe Copula, when rotated at 180° (as in this example), exhibits strong lower
# tail dependence. Although Archimedean copulas are typically not appropriate for stock
# returns, they can be a good fit for other asset classes such as agricultural
# commodities, derivatives, or CDSs.

# %%
# Conclusion
# ==========
# Bivariate copulas are used to model complex dependencies between financial assets.
# Elliptical copulas, such as the Gaussian and Student's t, offer a straightforward
# approach with symmetric dependence, while Archimedean copulas (Gumbel, Clayton, Joe)
# provide specialized modeling of tail dependencies. The ability to rotate Archimedean
# copulas further enhances their flexibility, allowing for a more accurate
# representation of the observed tail behavior in financial data.
