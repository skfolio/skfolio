************
Installation
************

Install using pip
*****************
`skfolio` is available on PyPI and can be installed with:

.. code:: console

    $ pip install skfolio

Install using conda
*******************

.. code:: console

    $ conda install -c conda-forge skfolio

Install Additional Solvers
**************************

The solver `Clarabel` is installed by default. Cardinality and threshold constraints
require a mixed-integer solver. To install additional solvers (e.g. `SCIP`, `GUROBI`,
`MOSEK`), please refer to
`the cvxpy documentation <https://www.cvxpy.org/install/index.html>`_


Dependencies
************

`skfolio` requires:

- python (>= 3.10)
- numpy (>= 1.24.0)
- scipy (>= 1.15.2)
- pandas (>= 2.1.0)
- cvxpy-base (>= 1.5.0)
- clarabel (>= 0.10.0)
- scs (>= 3.2.0)
- scikit-learn (>= 1.6.0)
- joblib (>= 1.3.2)
- plotly (>= 6.0.0)
