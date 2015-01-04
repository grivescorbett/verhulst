"""Routines to bootstrap confidence intervals on coefficients.

.. autosummary::

    confint
    bootstrap
    jackknife
"""


import multiprocessing

import numpy as np
import numpy.ma as ma
import scipy.stats


def confint(X, y, clf, n=100, alpha=[0.025, 0.975], n_jobs=None):
    """Compute coefficient confidence intervals using non-parametric BCa.

    Parameters
    ----------
    X : array
        Design matrix.
    y : array
        Response vector.
    clf : sklearn.base.BaseEstimator
        A scikit-learn estimator with `fit` and `predict` methods and
        a `coef_` attribute.
    n : int, optional
        The number of bootstrap samples. Default is 100.
    alpha : array, optional
        An array of percentiles used to compute confidence intervals.
        Default is a 95% confidence interval.
    n_jobs : int, optional
        The number of jobs to execute in parallel. Default is None, which
        uses the number returned by `os.cpu_count()`

    Returns
    -------
    confint : array
        Estimated BCa confidence intervals.
    z0 : array
        Estimated bias correction.
    acc : array
        Estimated acceleration constants.
    u : array
        Jackknife influence values.

    References
    ----------
    .. [1] Efron, Bradley, and Robert J. Tibshirani. 1993. *An Introduction
       to the Bootstrap*. London: Chapman & Hall.
    """
    if n_jobs is not None and not (isinstance(n_jobs, int) and n_jobs > 1):
        raise ValueError('Number of jobs must be a positive integer')

    # Bias correction
    theta_hat = clf.fit(X, y).coef_
    theta_star = _g(X, y, clf, bootstrap(X, n), n_jobs)
    theta_star.sort(axis=0)

    # Acceleration
    u = _g(X, y, clf, jackknife(X), n_jobs)
    uu = np.mean(u, axis=0) - u
    acc = np.sum(uu ** 3, axis=0) / (6 * np.sum(uu ** 2, axis=0) ** 1.5)

    z0 = scipy.stats.norm.ppf(np.sum(theta_star < theta_hat, axis=0) / n)
    zalpha = scipy.stats.norm.ppf(alpha)[:, np.newaxis]

    tt = scipy.stats.norm.cdf(z0 + (z0 + zalpha) / (1 - acc * (z0 + zalpha)))
    ooo = (tt * (n - 1)).astype(np.int64)

    confint = theta_star[ooo].diagonal(axis1=1, axis2=2)

    return confint, z0, acc, u


def _f(X, y, clf):
    """Returns the flattened coefficients of a fitted classifier.

    This function exists at the module level instead of as an anonymous or
    subordinate function so that it is importable by `multiprocessing`.

    Parameters
    ----------
    X : array
        Design matrix.
    y : array
        Response vector.
    clf : sklearn.base.BaseEstimator
        A scikit-learn estimator with `fit` and `predict` methods and a
        `coef_` attribute.

    Returns
    -------
    coef : array
        The flattened coefficients of the fitted classifier.
    """
    return clf.fit(X, y).coef_.flatten()


def _g(X, y, clf, g, n_jobs=100):
    """Returns an array of bootstrapped coefficients.

    Parameters
    ----------
    X : array
        Design matrix.
    y : array
        Response vector.
    clf : sklearn.base.BaseEstimator
        A scikit-learn estimator with `fit` and `predict` methods and a
        `coef_` attribute.
    g : generator
        A generator that yields indices, which are used to create subsets
        of `X` and `y` to pass to `clf`.
    n_jobs : int, optional
        The number of jobs to execute in parallel. Default is None, which
        uses the number returned by `os.cpu_count()`

    Returns
    -------
    coef : array
        An array bootstrapped coefficients.
    """
    with multiprocessing.Pool(n_jobs) as pool:
        return np.array(pool.starmap(_f, ((X[i], y[i], clf) for i in g)))


def bootstrap(X, n=100):
    """Yields bootstrap indices.

    Parameters
    ----------
    X : array
        Design matrix.
    n : int
        The number of bootstrap samples. Default is 100.

    Returns
    -------
    indices : array
        An array in which rows correspond to bootstrap indices.
    """
    nrow, __ = X.shape
    for i in range(n):
        yield np.random.randint(nrow, size=(nrow, ))


def jackknife(X):
    """Yields jackknife indices.

    Parameters
    ----------
    X : array
        Design matrix.

    Returns
    -------
    index : array
        An array of jackknife indices.
    """
    nrow, __ = X.shape
    masked = ma.array(np.arange(nrow), mask=False)
    for i in range(nrow):
        masked.mask = False
        masked.mask[i] = True
        yield masked.compressed()
