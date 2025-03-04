r"""
===========
Vine Copula
===========

This tutorial presents the :class:`~skfolio.distribution.VineCopula` estimator.
An introduction to Bivariate Copulas can be found in
:ref:`this previous tutorial <sphx_glr_auto_examples_3_synthetic_data_plot_1_bivariate_copulas.py>`.

Introduction
============
A Vine copula is a highly flexible multivariate copula model that decomposes a complex
dependency structure into a cascade of bivariate copulas (Gaussian, Student's, Clayton,
Gumbel, Joe, etc.). This approach allows each pair of variables to be modeled with its
own copula, capturing intricate dependencies that may be asymmetric or exhibit distinct
tail behaviors.

Moreover, the marginal distributions are modeled independently using a variety of
univariate candidate distributions (Gaussian, Student's t, Johnson Su, etc.). This
separation of marginal modeling and dependence modeling provides greater flexibility
when fitting multivariate data.

Mathematical Foundations
========================
At the core of copula theory lies **Sklar's Theorem**, which states that for any
multivariate cumulative distribution function :math:`F(x_1, \dots, x_d)` with marginals
:math:`F_1(x_1), \dots, F_d(x_d)`, there exists a copula :math:`C` such that:

.. math::
    F(x_1, \dots, x_d) = C\left(F_1(x_1), \dots, F_d(x_d)\right).

A Regular Vine copula leverages Sklar's Theorem by expressing the joint density function
:math:`f(x_1, \dots, x_d)` as a product of the marginal densities and a series of
bivariate copula densities arranged in a vine structure. Mathematically, this
decomposition is given by:

.. math::
    f(x_1, \dots, x_d) = \prod_{i=1}^{d} f_i(x_i) \times
    \prod_{k=1}^{d-1} \prod_{j=1}^{d-k}
    c_{j, j+k \mid 1, \dots, j-1}\Bigl(F\bigl(x_j \mid x_1, \dots, x_{j-1}\bigr),
    F\bigl(x_{j+k} \mid x_1, \dots, x_{j-1}\bigr)\Bigr),

where:

- :math:`f_i(x_i)` denotes the marginal density of :math:`x_i`.
- :math:`c_{j, j+k \mid 1, \dots, j-1}` represents the density of the bivariate copula
  linking the conditional distributions of :math:`x_j` and :math:`x_{j+k}` given the
  variables :math:`x_1, \dots, x_{j-1}`.

Advantages of Vine Copulas
==========================
Vine copulas offer several advantages in financial modeling compared to deep learning
approaches such as out-of-the-box Generative Adversarial Networks (GANs) or Variational
Autoencoders (VAEs):

- **Interpretability:**
  Vine copulas provide a clear, parametric decomposition of the joint distribution into
  marginal and pairwise components.

- **Tail Dependence Modeling:**
  They explicitly model tail dependencies, allowing for better estimation of joint
  extreme events.

- **Flexibility and Parsimony:**
  By selecting the best-fitting parametric copula for each pair of variables, vine
  copulas can capture a wide range of dependency structures while limiting overfitting,
  even in high-dimensional settings.

- **Data Efficiency:**
  They require less data than deep learning models to accurately
  model dependencies, making them suitable for financial datasets that may be limited in
  size.

- **Conditional Sampling and Stress Testing:**
  They support advanced conditional sampling techniques that enable the generation
  of scenario-specific simulations. This is especially beneficial for generating stress
  tests that are both **extreme** and **plausible**.

Types of Vine Structures
========================
Vine copulas encompass several specific structures that organize the pair-copula
decomposition differently. The most common types are:

- **R-vine (Regular Vine):**
  The most general vine structure, which is constructed using a maximum spanning tree
  (MST) algorithm. At each tree level, the MST selects the pairwise dependencies that
  maximize a chosen dependence measure, ensuring that the strongest dependencies are
  captured.

- **C-vine (Canonical Vine):**
  A special case of the R-vine where one variable acts as a central node and is
  connected to all other variables in the first tree. Subsequent trees condition on this
  central variable. This structure is useful when one variable exerts a dominant
  influence over the others.

- **D-vine (Drawable Vine):**
  A special case where variables are arranged in a sequential chain with each variable
  conditionally dependent only on its immediate neighbors. While this chain-like
  structure is intuitive, it is often too simplistic for financial applications, where
  dependencies tend to be more complex and multidimensional.

- **Clustered Vine:**
  An extension of the vine framework that explicitly accounts for clustered dependency
  structures. In this approach, variables are grouped around central assets, capturing
  hierarchical relationships within clusters. This is especially beneficial in finance,
  where assets often exhibit strong intra-cluster dependencies, enhancing conditional
  sampling and stress testing.

Skfolio Implementation
======================
The Vine Copula in skfolio is a novel, state-of-the-art implementation designed
specifically for financial data. It constructs a Regular Vine copula where asset
centrality can be controlled to capture clustered or C-like dependency structures.
This allows for a more nuanced representation of hierarchical relationships among
assets, enhancing conditional sampling and stress testing.

Key features include:

- **Inference Methods:**
  The implementation supports both inverse Kendall’s tau (itau) and Maximum Likelihood
  Estimation (MLE) approaches to estimate both optimal marginal distributions and pair
  copula parameters.

- **Dependence Structure:**
  The maximum spanning tree construction supports multiple dependence measures such as
  Kendall's tau, mutual information, or Wasserstein distance.

- **Performance Enhancements:**
  It leverages parallelization and vine truncation to improve computational efficiency.

- **Sampling Capabilities:**
  The model supports both unconditional sampling and complex conditional sampling for
  stress testing.

"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` and select 6 stocks (for demonstration
# purposes) starting from 1990-01-02 up to 2022-12-28:
from plotly.io import show

from skfolio.datasets import load_sp500_dataset
from skfolio.distribution import StudentTCopula, VineCopula, compute_pseudo_observations
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
prices = prices[["AMD", "BAC", "HD", "JPM", "LLY", "CVX"]]
X = prices_to_returns(prices)
print(X.tail())


# %%
# Vine Copula
# ===========
# Let's fit a Regular :class:`~skfolio.distribution.VineCopula` in parallel using all
# processors (`n_jobs=-1`) and applying a log transform for improved statistical
# properties:
vine = VineCopula(n_jobs=-1, log_transform=True)
vine.fit(X)
vine.display_vine()

# %%
# We note that the marginals are composed of Johnson Su and Student's t distributions
# while the pair copulas are only composed of Student's t Copulas.
#
# Let's break it down:
#
# * `Node(0): JohnsonSU(-0.0462, 1.24, -0.00123, 0.0332)`: This represents the marginal
#   distribution of variable 1 (BAC).
#
# * `Edge((0, 2), StudentTCopula(0.316, 5.671))`: This represents the unconditional
#   bivariate copula between variables 0 (AMD) and 2 (HD) modeled by a Student's t Copula
#   with :math:`\rho=31.6\%` and :math:`\nu=5.6`.
#
# * `Edge((1, 2) | {3, 5}, StudentTCopula(0.123, 7.732))`: This represents the
#   conditional bivariate copula between variables 1 (BAC) and 2 (HD) given variables
#   3 (JPM) and 5 (CVX) modeled by a Student's t Copula with :math:`\rho=12.3\%` and
#   :math:`\nu=7.7`.

# %%
# Sampling from the Vine
# ======================
# Let's generate 10,000 synthetic returns from the vine copula model. In the next
# tutorial, we will show how this can be used for minimizing portfolio CVaR when
# historical tail data is limited. The use of parametric copulas enables the
# extrapolation of tail dependencies, and by generating a larger sample of returns,
# we can achieve enhanced accuracy in capturing tail co-dependencies during the
# optimization process.
samples = vine.sample(n_samples=10000)
print(samples.shape)

# %%
# Let's plot the Scatter Plots of the generated returns from the Vine model and compare
# them with the historical returns `X`.
fig = vine.plot_scatter_matrix(X=X, random_state=2)
fig.update_layout(height=600)
show(fig)

# %%
# Tractability & Interpretability
# ===============================
# As mentioned above, one of the advantages of Vine Copula is its tractability.
# First, let's plot the marginal distribution of AMD and compare it versus the
# historical data:
vine.marginal_distributions_[0].plot_pdf(X[["AMD"]])

# %%
# Now, let's investigate the bivariate copula between variables 0 (AMD) and 2 (HD):
edge = vine.trees_[0].edges[0]
copula = edge.copula
print(edge)
print(f"Lower Tail Dependence: {copula.lower_tail_dependence:.2f}")
print(f"Upper Tail Dependence: {copula.upper_tail_dependence:.2f}")

# %%
# The model indicates a tail dependence coefficient of approximately 11%, suggesting
# a positive likelihood that extreme returns (both negative and positive) occur
# simultaneously for the assets.
#
# Let's plot the tail concentration of the copula model versus the historical data:
U = compute_pseudo_observations(X[["AMD", "HD"]])
copula.plot_tail_concentration(U)

# %%
# We notice that the model properly captured the fat tail dependencies.


# %%
# Conditional Sampling & Stress Testing
# =====================================
# One of the main advantages of Vine Copula is its ability to produce accurate
# conditional sampling in extreme scenarios by leveraging the accurate computation of
# inverse CFD of the marginal distributions as well as the partial derivatives and
# inverse partial derivatives of the bivariate copulas.
#
# Let's generate 1000 samples conditioned on AMD returns being at -20%.
#
# When using conditional sampling, it is recommended that the assets you condition on
# are set as central during the vine copula construction. This can be specified via
# `central_assets`:
vine = VineCopula(n_jobs=-1, log_transform=True, central_assets=["AMD"])
vine.fit(X)
vine.display_vine()

# %%
# As expected, we notice that the variable 0 (AMD) is the central node in Tree 0 and
# the common conditioning variable in subsequent trees.
cond_samples = vine.sample(n_samples=1000, conditioning={"AMD": -0.2})
print(cond_samples.shape)

# %%
# Note that you can also provide an array of scenarios for the AMD returns
# (e.g. `[-0.2,-0.21,-0.22,...]`).

# %%
# Let's now see a more complex example by generating samples conditional to the
# following:
#
# * HD between -10% and -15%
# * JPM below -5%
vine = VineCopula(n_jobs=-1, log_transform=True, central_assets=["HD", "JPM"])
vine.fit(X)

conditioning = {"HD": (-0.15, -0.10), "JPM": (None, -0.05)}
cond_samples = vine.sample(n_samples=1000, conditioning=conditioning)
print(cond_samples.shape)

# %%
# Let's plot the marginals conditional returns and compare them with the
# historical returns `X`.
vine.plot_marginal_distributions(X=X, conditioning=conditioning, random_state=0)

# %%
# In the graph, by selecting HD and JPM, you can see that the conditioning has been
# respected and has impacted the other assets following the vine structure. This allows
# the creation of Stress Tests that are both **extreme** and **plausible**.


# %%
# Advanced Stress Testing
# =======================
# Another approach for generating stress samples, which can be used in conjunction
# with conditional sampling, involves directly stressing the vine structure. Since all
# pair copulas are accessible, we can modify their parameters to simulate stressed
# market conditions. This approach is particularly useful when the vine structure is
# considered to be dynamic and regime-dependent (i.e. dynamic vine).
# In our example, we increase the correlation parameter of the Student's t copula by
# 10% and decrease its degrees of freedom by 20%, resulting in heavier tails and more
# pronounced joint extreme events. In practice, these stressed parameters should be
# calibrated based on specific market regimes.
for tree in vine.trees_:
    for edge in tree.edges:
        if isinstance(edge.copula, StudentTCopula):
            edge.copula.rho_ *= 1.1
            edge.copula.dof_ *= 0.8

samples = vine.sample(n_samples=1000)


# %%
# Conclusion
# ==========
# The flexibility, interpretability, and explicit modeling of tail dependencies make
# vine copulas an attractive choice for financial applications. In the next tutorial,
# we will show how to use them in a portfolio pipeline for optimization and stress
# testing.

# %%
# References
# ==========
# [1] Selecting and estimating regular vine copulae and application to financial returns
#     Dißmann, Brechmann, Czado, and Kurowicka (2013).
#
# [2] Growing simplified vine copula trees: improving Dißmann's algorithm
#     Krausa and Czado(2017).
#
# [3] Pair-copula constructions of multiple dependence"
#     Aas, Czado, Frigessi, Bakken(2009).
#
# [4] Pair-Copula Constructions for Financial Applications: A Review
#     Aas and Czado(2016).
#
# [5] Conditional copula simulation for systemic risk stress testing
#     Brechmann, Hendrich, Czado(2013)
