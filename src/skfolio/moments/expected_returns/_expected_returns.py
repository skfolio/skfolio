"""Expected returns estimators."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from enum import auto

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm  

from skfolio.moments.covariance import BaseCovariance, EmpiricalCovariance
from skfolio.moments.expected_returns._base import BaseMu
from skfolio.utils.tools import AutoEnum, check_estimator


class EmpiricalMu(BaseMu):
    """Empirical Expected Returns (Mu) estimator.

    Estimates the expected returns with the historical mean.

    Parameters
    ----------
    window_size : int, optional
        Window size. The model is fitted on the last `window_size` observations.
        The default (`None`) is to use all the data.

    Attributes
    ----------
    mu_ : ndarray of shape (n_assets,)
        Estimated expected returns of the assets.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.
    """

    def __init__(self, window_size: int | None = None):
        self.window_size = window_size

    def fit(self, X: npt.ArrayLike, y=None) -> "EmpiricalMu":
        """Fit the Mu Empirical estimator model.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : EmpiricalMu
            Fitted estimator.
        """
        X = self._validate_data(X)
        if self.window_size is not None:
            X = X[-self.window_size :]
        self.mu_ = np.mean(X, axis=0)
        return self


class EWMu(BaseMu):
    r"""Exponentially Weighted Expected Returns (Mu) estimator.

    Estimates the expected returns with the exponentially weighted mean (EWM).

    Parameters
    ----------
    window_size : int, optional
        Window size. The model is fitted on the last `window_size` observations.
        The default (`None`) is to use all the data.

    alpha : float, default=0.2
        Exponential smoothing factor. The default value is `0.2`.

        :math:`0 < \alpha \leq 1`.

    Attributes
    ----------
    mu_ : ndarray of shape (n_assets,)
       Estimated expected returns of the assets.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.
    """

    def __init__(self, window_size: int | None = None, alpha: float = 0.2):
        self.window_size = window_size
        self.alpha = alpha

    def fit(self, X: npt.ArrayLike, y=None) -> "EWMu":
        """Fit the EWMu estimator model.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : EWMu
            Fitted estimator.
        """
        X = self._validate_data(X)
        if self.window_size is not None:
            X = X[-self.window_size :]
        self.mu_ = pd.DataFrame(X).ewm(alpha=self.alpha).mean().iloc[-1, :].to_numpy()
        return self


class EquilibriumMu(BaseMu):
    r"""Equilibrium Expected Returns (Mu) estimator.

    The Equilibrium is defined as:

        .. math:: risk\_aversion \times \Sigma \cdot w^T

    For Market Cap Equilibrium, the weights are the assets Market Caps.
    For Equal-weighted Equilibrium, the weights are equal-weighted (1/N).

    Parameters
    ----------
    risk_aversion : float, default=1.0
        Risk aversion factor.
        The default value is `1.0`.

    weights : array-like of shape (n_assets,), optional
        Asset weights used to compute the Expected Return Equilibrium.
        The default is to use the equal-weighted equilibrium (1/N).
        For a Market Cap weighted equilibrium, you must provide the asset Market Caps.

    covariance_estimator : BaseCovariance, optional
        :ref:`Covariance estimator <covariance_estimator>` used to estimate the
        covariance in the equilibrium formula.
        The default (`None`) is to use :class:`~skfolio.moments.EmpiricalCovariance`.

    Attributes
    ----------
    mu_ : ndarray of shape (n_assets,)
          Estimated expected returns of the assets.

    covariance_estimator_ : BaseCovariance
        Fitted `covariance_estimator`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.
    """

    covariance_estimator_: BaseCovariance

    def __init__(
        self,
        risk_aversion: float = 1,
        weights: np.ndarray | None = None,
        covariance_estimator: BaseCovariance | None = None,
    ):
        self.risk_aversion = risk_aversion
        self.weights = weights
        self.covariance_estimator = covariance_estimator

    def fit(self, X: npt.ArrayLike, y=None) -> "EquilibriumMu":
        """Fit the EquilibriumMu estimator model.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : EquilibriumMu
            Fitted estimator.
        """
        # fitting estimators
        self.covariance_estimator_ = check_estimator(
            self.covariance_estimator,
            default=EmpiricalCovariance(),
            check_type=BaseCovariance,
        )
        self.covariance_estimator_.fit(X)

        # we validate and convert to numpy after all models have been fitted to keep
        # features names information.
        X = self._validate_data(X)
        n_assets = X.shape[1]
        if self.weights is None:
            weights = np.ones(n_assets) / n_assets
        else:
            weights = np.asarray(self.weights)
        self.mu_ = self.risk_aversion * self.covariance_estimator_.covariance_ @ weights
        return self


class ShrunkMuMethods(AutoEnum):
    """Shrinkage methods for the ShrunkMu estimator

    Parameters
    ----------
    JAMES_STEIN : str
        James-Stein method

    BAYES_STEIN : str
        Bayes-Stein method

    BODNAR_OKHRIN : str
        Bodnar Okhrin Parolya method
    """

    JAMES_STEIN = auto()
    BAYES_STEIN = auto()
    BODNAR_OKHRIN = auto()


class ShrunkMu(BaseMu):
    r"""Shrinkage Expected Returns (Mu) estimator.

    Estimates the expected returns using shrinkage.

    The sample mean estimator is unbiased but has high variance.
    Stein (1955) proved that it's possible to find an estimator with reduced total
    error using shrinkage by trading a small bias against high variance.

    The estimator shrinks the sample mean toward a target vector:

        .. math:: \hat{\mu} = \alpha\bar{\mu}+\beta \mu_{target}

    with :math:`\bar{\mu}` the sample mean, :math:`\mu_{target}` the target vector
    and :math:`\alpha` and :math:`\beta` two constants to determine.

    There are two choices for the target vector :math:`\mu_{target}` :

        * Grand Mean: constant vector of the mean of the sample mean
        * Volatility-Weighted Grand Mean: volatility-weighted sample mean

    And three methods for :math:`\alpha` and :math:`\beta` :

        * James-Stein
        * Bayes-Stein
        * Bodnar Okhrin Parolya

    Parameters
    ----------
    covariance_estimator : BaseCovariance, optional
        :ref:`Covariance estimator <covariance_estimator>` used to estimate the
        covariance in the shrinkage formulae.
        The default (`None`) is to use :class:`~skfolio.moments.EmpiricalCovariance`.

    vol_weighted_target : bool, default=False
        If this is set to True, the target vector :math:`\mu_{target}` is the
        Volatility-Weighted Grand Mean otherwise it is the Grand Mean.
        The default is `False`.

    method : ShrunkMuMethods, default=ShrunkMuMethods.JAMES_STEIN
        Shrinkage method :class:`ShrunkMuMethods`.

        Possible values are:

            * JAMES_STEIN
            * BAYES_STEIN
            * BODNAR_OKHRIN

        The default value is `ShrunkMuMethods.JAMES_STEIN`.

    Attributes
    ----------
    mu_ : ndarray of shape (n_assets,)
        Estimated expected returns of the assets.

    covariance_estimator_ : BaseCovariance
        Fitted `covariance_estimator`.

    mu_target_ : ndarray of shape (n_assets,)
        Target vector :math:`\mu_{target}`.

    alpha_ : float
        Alpha value :math:`\alpha`.

    beta_ : float
        Beta value :math:`\beta`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1] "Risk and Asset Allocation",
        Attilio Meucci (2005)

    .. [2] "Bayes-stein estimation for portfolio analysis",
        Philippe Jorion (1986)

    .. [3] "Optimal shrinkage estimator for high-dimensional mean vector"
        Bodnar, Okhrin and Parolya (2019)
    """

    covariance_estimator_: BaseCovariance
    mu_target_: np.ndarray
    alpha_: float
    beta_: float

    def __init__(
        self,
        covariance_estimator: BaseCovariance | None = None,
        vol_weighted_target: bool = False,
        method: ShrunkMuMethods = ShrunkMuMethods.JAMES_STEIN,
    ):
        self.covariance_estimator = covariance_estimator
        self.vol_weighted_target = vol_weighted_target
        self.method = method

    def fit(self, X: npt.ArrayLike, y=None) -> "ShrunkMu":
        """Fit the ShrunkMu estimator model.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : ShrunkMu
            Fitted estimator.
        """
        if not isinstance(self.method, ShrunkMuMethods):
            raise ValueError(
                "`method` must be of type ShrunkMuMethods, got"
                f" {type(self.method).__name__}"
            )
        # fitting estimators
        self.covariance_estimator_ = check_estimator(
            self.covariance_estimator,
            default=EmpiricalCovariance(),
            check_type=BaseCovariance,
        )
        self.covariance_estimator_.fit(X)

        # we validate and convert to numpy after all models have been fitted to keep
        # features names information.
        X = self._validate_data(X)
        n_observations, n_assets = X.shape

        covariance = self.covariance_estimator_.covariance_

        sample_mu = np.mean(X, axis=0)
        cov_inv = None

        # Calculate target vector
        if self.vol_weighted_target:
            cov_inv = np.linalg.inv(covariance)
            self.mu_target_ = np.sum(cov_inv, axis=1) @ sample_mu / np.sum(cov_inv)
        else:
            self.mu_target_ = np.mean(sample_mu)
        self.mu_target_ *= np.ones(n_assets)

        # Calculate Estimators
        match self.method:
            case ShrunkMuMethods.JAMES_STEIN:
                eigenvalues = np.linalg.eigvals(covariance)
                self.beta_ = (
                    (np.sum(eigenvalues) - 2 * np.max(eigenvalues))
                    / np.sum((sample_mu - self.mu_target_) ** 2)
                    / n_observations
                )
                self.alpha_ = 1 - self.beta_
            case ShrunkMuMethods.BAYES_STEIN:
                if cov_inv is None:
                    cov_inv = np.linalg.inv(covariance)
                self.beta_ = (n_assets + 2) / (
                    n_observations
                    * (sample_mu - self.mu_target_).T
                    @ cov_inv
                    @ (sample_mu - self.mu_target_)
                    + (n_assets + 2)
                )
                self.alpha_ = 1 - self.beta_
            case ShrunkMuMethods.BODNAR_OKHRIN:
                if cov_inv is None:
                    cov_inv = np.linalg.inv(covariance)
                u = sample_mu.T @ cov_inv @ sample_mu
                v = sample_mu.T @ cov_inv @ self.mu_target_
                w = self.mu_target_.T @ cov_inv @ self.mu_target_
                self.alpha_ = (
                    (u - n_assets / (n_observations - n_assets)) * w - v**2
                ) / (u * w - v**2)
                self.beta_ = (1 - self.alpha_) * v / u
            case _:
                raise ValueError(f"method {self.method} is not valid")

        self.mu_ = self.alpha_ * sample_mu + self.beta_ * self.mu_target_
        return self



class MertonMu(BaseMu):
    """Merton Estimator for the returns (first one)
    
    Estimates the expected returns based on the paper of Merton.
    
    Parameters
    ----------
    h : int, optional
        size of the intervall of time where we assume the returns almost constant
    T : int
        Size of the window where "Yj" is constant i.e. the market is in a stationary state.
    j: int, optional 
        Model used (1,2 or 3).
    b: float, optional
        Parameter for the posterior distribution of Yj in Merton's paper. Equal to infinity by default
    likelihood : boolean, optional
        asking to the user if he wants to use the likelihood estimator. By default yes.
        
    Attributes
    ---------- 
     mu_ : ndarray of shape (n_assets,)
        Estimated expected returns of the assets.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.
    """

class MertonMu(BaseMu):
    def __init__(self,
                 h: int = 30,
                 T:int = 365,
                 j: int = 1,
                 b:float = np.inf,
                 likelihood:bool=True,
                 covariance_estimator: BaseCovariance | None = None
                 ):
        self.tau = len(X) 
        self.h = h 
        self.T = T 
        self.j = j 
        self.b = b 
        self.N = int(self.T/self.h)
        self.likelihood = likelihood
        self.covariance_estimator = covariance_estimator
        #self.rendements_taux_sans_risques = rendements_taux_sans_risques #en théorie on donne un array de taux sans risques ? 

    def fit(self, X:npt.ArrayLike,short_term_rate: npt.ArrayLike |None = None, y=None)->"MertonMu":
        """Fit of the Merton Estimator 

         Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        short_term_rate : array-like of shape (n_observations,1)
            short term rates observed on the market 

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : MertonMu
            Fitted estimator.
        """
        if short_term_rate is None:
            short_term_rate = np.zeros(len(X)) #Default short-term rate set to 0

        # fitting estimators
        self.covariance_estimator_ = check_estimator(
            self.covariance_estimator,
            default=EmpiricalCovariance(),
            check_type=BaseCovariance,
        )

        self.mu_ = None

        riskless_rate = np.exp(self.h*short_term_rate) #return per dollar on the riskless asset between t and t + h 

        X_t = np.log(X.shift(-self.h)/X).div(riskless_rate, axis=0).dropna()
        #X_t denote the log return per dollar on the market portfolio between time t and t + h. (R(t+h)-R(t))/riskless_rate(t)
       

        self.covariance_estimator_.fit((np.log(X)).diff(-self.h).dropna()[-self.N:]) 
        #We consider the period during which the variance is assumed to be constant.
        covariance = self.covariance_estimator_.covariance_ 


        
       
        if self.j == 1:
            #choice of the model 
            Y = np.mean(X_t[-self.N:],axis=0)/(np.diag(covariance)) + 0.5 #We assumed that the covariance was constant over the N periods.
            self.mu_ = (Y*(np.diag(covariance))).values + short_term_rate[-1] #model 1 in Merton's article 
        
        

        if self.likelihood: 
            #For now, we have only implemented the case where we want to work with the likelihood method and where the parameter b is set to +inf.
            omega_j =np.sqrt(self.N*np.diag(covariance)**(4-2*self.j))
            lambda_j = Y/omega_j 
            rho_j = omega_j*(self.b-lambda_j)
            eta_j = - lambda_j*omega_j #quantities explicitly described in Merton's paper.

            Y_barre = Y + np.exp(-0.5*eta_j**2)/(np.sqrt(2*np.pi)*omega_j*(1-norm.cdf(eta_j))) #b is set to +inf
            self.mu_ = (Y_barre*np.diag(covariance)).values + short_term_rate[-1] #new returns 
        
        return self







