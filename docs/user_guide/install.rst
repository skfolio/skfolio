************
Installation
************

Install using pip
*****************
``skfolio`` is available on PyPI and can be installed with:

.. code:: console

    $ pip install skfolio

Install using conda
*******************

.. code:: console

    $ conda install -c conda-forge skfolio

Install Additional Solvers
**************************

By default, the solver `Clarabel`  is installed.
To install additional solvers (`SCIP`, `GUROBI`, `MOSEK`), please refer to
`the cvxpy documentation <https://www.cvxpy.org/install/index.html>`_

Install Optional Dependencies
*****************************

Feature Extraction
==================

To use the feature extraction module with probabilistic PCA methods:

.. code:: console

    $ pip install 'skfolio[feature_extraction]'

Or using uv:

.. code:: console

    $ uv pip install 'skfolio[feature_extraction]'

This installs the `gen_fex` package which provides:

- **PPCA**: Probabilistic Principal Component Analysis
- **PKPCA**: Probabilistic Kernel PCA with Wishart process priors

See :ref:`feature_extraction` for usage examples.


Dependencies
************

`skfolio` requires:

- python (>= 3.10)
- numpy (>= 1.23.4)
- scipy (>= 1.8.0)
- pandas (>= 1.4.1)
- cvxpy-base (>= 1.5.0)
- clarabel (>= 0.9.0)
- scikit-learn (>= 1.6.0)
- joblib (>= 1.3.2)
- plotly (>= 5.22.0)