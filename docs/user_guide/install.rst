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
- numpy (>= 1.23.4)
- scipy (>= 1.15.2)
- pandas (>= 1.4.1)
- cvxpy-base (>= 1.5.0)
- clarabel (>= 0.9.0)
- scikit-learn (>= 1.6.0)
- joblib (>= 1.3.2)
- plotly (>= 5.22.0)