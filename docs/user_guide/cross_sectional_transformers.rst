.. _cross_sectional_transformers:

.. currentmodule:: skfolio.preprocessing

****************************
Cross-Sectional Transformers
****************************

A :ref:`Cross-Sectional Transformer <preprocessing_ref>` normalizes each value
within an observation's cross-section using only values from that same
cross-section, i.e. row by row across assets.

All transformers follow the scikit-learn API and accept any array-like input
(numpy array, pandas DataFrame, etc.). They are stateless: `fit` only
validates, and `transform` returns the normalized array. NaNs are treated as
missing values, ignored when computing cross-sectional statistics, and
preserved in the output.

Available transformers:
    * :class:`CSStandardScaler`: cross-sectional z-score, optionally computed
      within groups (e.g. sectors).
    * :class:`CSPercentileRankScaler`: cross-sectional percentile rank in
      :math:`(0, 1)`.
    * :class:`CSGaussianRankScaler`: cross-sectional rank gaussianization via
      the inverse standard normal CDF :math:`\Phi^{-1}`.
    * :class:`CSWinsorizer`: cross-sectional clipping at low and high
      percentiles.
    * :class:`CSTanhShrinker`: smooth shrinkage of extreme values toward the
      cross-sectional center, preserving the original scale.

Shared arguments
================

`cs_weights`
    Cross-sectional weights as a non-negative array of shape
    `(n_observations, n_assets)`. Assets with `cs_weights > 0` define the
    *estimation universe* used to compute the cross-sectional statistics,
    while assets outside still receive a transformed value relative to it.
    :class:`CSStandardScaler` and :class:`CSGaussianRankScaler` also use
    `cs_weights` to weight the cross-sectional mean used for centering.

`cs_groups`
    Cross-sectional groups as an integer array of shape
    `(n_observations, n_assets)` with labels `>= -1`. Statistics are then
    computed within each group rather than over the full cross-section. Use
    `-1` to mark unclassified assets. Such assets, together with groups
    smaller than `min_group_size`, fall back to the global cross-section.
    Useful for keeping exposures neutral within sectors or countries.
    Supported by :class:`CSStandardScaler`, :class:`CSPercentileRankScaler`
    and :class:`CSGaussianRankScaler`.

Choosing a transformer
======================

===============================  ================  ================  ===========  ============================
Transformer                      Output            Outlier handling  `cs_groups`  `cs_weights`
===============================  ================  ================  ===========  ============================
:class:`CSStandardScaler`        Z-scores          None              Yes          Universe + weighted mean
:class:`CSGaussianRankScaler`    Gaussian scores   Rank-based        Yes          Universe + weighted recenter
:class:`CSPercentileRankScaler`  Percentile ranks  Rank-based        Yes          Universe mask only
:class:`CSTanhShrinker`          Original scale    Smooth tails      No           Universe mask only
:class:`CSWinsorizer`            Original scale    Hard clip         No           Universe mask only
===============================  ================  ================  ===========  ============================

Across all transformers, `cs_weights > 0` picks the assets that enter the *estimation universe*. Ranks,
medians, MAD, percentiles and standard deviations are always equal-weighted on that set. Only the
cross-sectional mean used for centering depends on the magnitude of the weights: :class:`CSStandardScaler`
centers `X` by its weighted mean (*Universe + weighted mean*), and :class:`CSGaussianRankScaler` recenters
the Gaussianized scores by their weighted mean (*Universe + weighted recenter*).

Example
=======

The example below uses :class:`CSStandardScaler` to illustrate the common
cross-sectional API, including NaN preservation.

.. code-block:: python

    import numpy as np

    from skfolio.preprocessing import CSStandardScaler

    X = np.array([[1.0, np.nan, 3.0, 4.0],
                  [4.0, 3.0, 2.0, 1.0],
                  [10.0, 20.0, np.nan, 40.0]])

    transformer = CSStandardScaler()
    transformer.fit_transform(X)
    # array([[-1.09108945,         nan,  0.21821789,  0.87287156],
    #        [ 1.161895  ,  0.38729833, -0.38729833, -1.161895  ],
    #        [-0.87287156, -0.21821789,         nan,  1.09108945]])

Here `cs_weights` defines a custom estimation universe and weighted mean, while
`cs_groups` applies the scaling within groups first. `min_group_size=2` is
required because these are two-asset groups (the default is 8); when a group's
estimation universe shrinks below this threshold (e.g. due to NaN or zero
weights), it falls back to the global cross-section:

.. code-block:: python

    cs_weights = np.array([[3.0, 0.0, 1.0, 2.0],
                           [4.0, 0.0, 2.0, 3.0],
                           [2.0, 3.0, 0.0, 5.0]])
    cs_groups = np.array([[0, 0, 1, 1],
                          [0, 0, 1, 1],
                          [0, 0, 1, 1]])

    transformer = CSStandardScaler(min_group_size=2)
    transformer.fit_transform(X, cs_weights=cs_weights, cs_groups=cs_groups)
    # array([[-0.55454325,         nan, -0.62182063,  1.1427252 ],
    #        [ 0.62254586, -0.15324206,  0.5035012 , -1.16572861],
    #        [-1.33736075,  0.20821245,         nan,  0.41001683]])
