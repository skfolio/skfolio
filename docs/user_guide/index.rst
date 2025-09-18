:og:description:  Comprehensive user guide for skfolio: step-by-step tutorials to install, configure, and use the Python library for portfolio optimization and risk management.

.. meta::
    :keywords: python portfolio optimization,
               quantitative finance,
               risk management,
               portfolio backtesting,
               algorithmic trading,
               robust optimization,
               scikit-learn integration,
               financial modeling,
               stress testing,
               skfolio
    :description: Comprehensive user guide for skfolio: step-by-step tutorials to
                  install, configure, and use the Python library for portfolio
                  optimization and risk management.


.. _user_guide:

==========
User Guide
==========

.. warning::
    The API is already stable and follows scikit-learn conventions.
    However, the version number remains below 1.0.0 to allow for rapid iteration
    and development. A first official stable release (1.0.0) is planned for 2025.
    Until then, we recommend pinning versions in production environments to guard
    against minor breaking changes, or connecting with `Skfolio Labs <https://skfoliolabs.com>`_
    for enterprise support and dedicated SLAs.


`skfolio` is a portfolio optimization and risk management framework build on top of
scikit-learn to perform model selection, validation, parameter tuning and stress-test
while reducing the risk of data leakage and overfitting.

.. toctree::
    :maxdepth: 2
    :hidden:

    Install <install>
    Optimization <optimization>
    Portfolio <portfolio>
    Population <population>
    Prior <prior>
    Expected Returns <expected_returns>
    Covariance <covariance>
    Distance <distance>
    Clustering <cluster>
    Uncertainty Set <uncertainty_set>
    Pre-Selection <pre_selection>
    Model Selection <model_selection>
    Hyper-Parameters Tuning <hyper_parameters_tuning>
    Metadata Routing <metadata_routing>
    Datasets <datasets>
    Data Preparation <data_preparation>
