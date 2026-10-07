.. _seriation:

.. currentmodule:: skfolio.seriation

********************
Seriation Estimators
********************

Seriation orders assets from their pairwise distances. Estimators in `skfolio.seriation`
take a square distance matrix and store the computed order in `ordering_`. This order
can be used to rearrange a matrix or form groups for portfolio allocation.

The available estimators are:

* :class:`HierarchicalSeriation`, which orders the leaves of a hierarchical clustering
  tree. By default, it uses Ward linkage and optimal leaf ordering to minimize distances
  between adjacent leaves without changing the tree. The
  `hierarchical_clustering_estimator` parameter accepts a configured
  :class:`~skfolio.cluster.HierarchicalClustering` estimator.
* :class:`SpectralSeriation`, which sorts coordinates computed from all pairwise
  distances. Its `partial_fit` method uses the previous coordinates and ordering to
  choose consistently between equivalent spectral solutions and resolve ties.

**Example:**

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.distance import PearsonDistance
    from skfolio.preprocessing import prices_to_returns
    from skfolio.seriation import HierarchicalSeriation, SpectralSeriation

    X = prices_to_returns(load_sp500_dataset())
    distance = PearsonDistance().fit(X.iloc[:252]).distance_

    hierarchical = HierarchicalSeriation().fit(distance)
    print(hierarchical.ordering_)

    spectral = SpectralSeriation().fit(distance)
    print(spectral.ordering_)

`ordering_` contains the original asset column positions in the computed order.


Online Learning
***************

:class:`SpectralSeriation` supports online learning through `partial_fit`. Each call
recomputes spectral coordinates from a complete, updated distance matrix. It uses the
previous coordinates and ordering to choose consistently between equivalent solutions
and resolve ties. Calling `fit` starts a new seriation without this history.

:class:`HierarchicalSeriation` supports batch fitting only. Each `fit` call builds a new
clustering tree from the supplied distance matrix, independently of previous calls.

For the spectral estimator fitted above, the next 252 observations can supply a new
distance matrix:

.. code-block:: python

    next_distance = PearsonDistance().fit(X.iloc[252:504]).distance_
    spectral.partial_fit(next_distance)
    print(spectral.ordering_)

The distance estimator determines which observations contribute to each matrix.
`SpectralSeriation.partial_fit` receives that matrix, rather than new rows of returns.


Changing Asset Universes
***********************

Both estimators follow skfolio's :ref:`NaN-aware convention <native_nan_aware>`:
non-investable assets remain present in the input arrays. For seriation, a NaN diagonal
entry excludes the asset from `ordering_`, and `investable_mask_` identifies the included
assets. For columns A, B and C, excluding B leaves positions 0 and 2 in the computed
order. Distances between included assets must be finite.

The included assets can change with each `fit` or `partial_fit` call. During
`SpectralSeriation.partial_fit`, the full matrix must keep the same asset rows and
columns in the same order. Adding an asset outside this set requires a new `fit` on the
expanded distance matrix. See :class:`BaseSeriation` for the full input requirements.


.. _seriation_turnover:

Seriation and Turnover
**********************

Spectral seriation avoids discrete cluster merges and can preserve allocation groups
under small changes in distances. Its `partial_fit` method also uses the previous result
to resolve equivalent solutions and ties consistently. Both effects can reduce turnover
in recursive portfolio allocation.

Lower turnover is not guaranteed. A small `spectral_gap_` indicates sensitivity to
distance changes, and allocation groups can change when spectral coordinates cross.
The benefit depends on the data, allocation method and rebalance schedule, and is
assessed by comparing realized turnover, risk and performance.
