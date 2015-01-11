"""Statistical routines for evaluating binary logistic regressions.

Pearson Chi-Square Statistic and Deviance
-----------------------------------------

.. autosummary::

    pearson_residuals
    pearson_chisquare
    deviance_residuals
    deviance

Hosmer–Lemeshow Tests
---------------------

.. autosummary::

    classification_balance
    hosmer_lemeshow_table
    hosmer_lemeshow_test
    osius_rojek_test

Likelihood Measures
-------------------

.. autosummary::

    log_likelihood
    aic

Summary Measures
----------------

.. autosummary::

    pearson_r2
    efrons_r2
    count_r2
    adjusted_count_r2
    mckelvey_zavoina_r2
    brier_score
    expected_pcp
    summary_measures

Casewise Statistics
-------------------

.. autosummary::

    pregibon_leverages
    standardize_residuals
    pregibon_dbetas
    case_deltas
"""


from collections import namedtuple

import numpy as np
import numpy.lib.recfunctions as rf
import scipy.linalg
import scipy.stats

from .bootstrap import confint  # noqa
from .util import binarize, make_recarray


########################################################################
# Pearson Chi-Square Statistic and Deviance


def pearson_residuals(y_true, y_pred):
    """Computes Pearson residuals.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    r : array
        Pearson residuals.
    """
    return (y_true - y_pred) / np.sqrt(y_pred * (1 - y_pred))


def pearson_chisquare(X, y_true, y_pred):
    """Computes the Pearson chi-square statistic.

    Parameters
    ----------
    X : array
        Design matrix.
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    chisquare : float
        The test statistic :math:`X^2`.
    df : int
        The degrees of freedom of the test.
    p : float
        The p-value of the test.
    """
    __, p = X.shape

    r = pearson_residuals(y_true, y_pred)

    chisquare = np.sum(np.square(r))
    df = len(y_true) - (p + 1)
    p = scipy.stats.chisqprob(chisquare, df)

    TestResult = namedtuple('PearsonChiSquare', ('chisquare', 'df', 'p'))
    return TestResult(chisquare, df, p)


def deviance_residuals(y_true, y_pred):
    """Computes deviance residuals.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    d : array
        Deviance residuals.
    """
    y0 = -np.sqrt(2 * np.abs(np.log(1 - y_pred)))
    y1 = np.sqrt(2 * np.abs(np.log(y_pred)))
    d = np.where(y_true, y1, y0)
    return d


def deviance(X, y_true, y_pred):
    """Computes the deviance statistic.

    Parameters
    ----------
    X : array
        Design matrix.
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    D : float
        The test statistic :math:`D`.
    df : int
        The degrees of freedom of the test.
    p : float
        The p-value of the test.
    """
    __, p = X.shape

    d = deviance_residuals(y_true, y_pred)

    D = np.sum(np.square(d))
    df = len(y_true) - (p + 1)
    p = scipy.stats.chisqprob(D, df)

    TestResult = namedtuple('Deviance', ('D', 'df', 'p'))
    return TestResult(D, df, p)


########################################################################
# Hosmer–Lemeshow Tests


def classification_balance(y_true, y_pred, n_groups=10):
    """Computes classification balance by groups of observations.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].
    n_groups : int, optional
        The number of groups to create. The default value is 10, which
        corresponds to deciles of predicted probabilities.

    Returns
    -------
    table : recarray
        A record array with `n_groups` rows and five columns: Group Size,
        Number of Observed 0, Number of Observed 1, Proportion of
        Observed 0, Proportion of Observed 1.
    """
    if n_groups < 2:
        raise ValueError('Number of groups must be greater than or equal to 2')

    if n_groups > len(y_true):
        raise ValueError('Number of predictions must exceed number of groups')

    table = make_recarray(y_true, y_pred)

    table = [(len(g), (g.y_true == 0).sum(), (g.y_true == 1).sum())
             for g in np.array_split(table, n_groups)]
    names = ('group_size', 'n_0', 'n_1')
    table = np.rec.fromrecords(table, names=names)

    field = table.n_0 / table.group_size
    table = rf.append_fields(table, 'prop_0', field, usemask=False,
                             asrecarray=True)

    field = table.n_1 / table.group_size
    table = rf.append_fields(table, 'prop_1', field, usemask=False,
                             asrecarray=True)

    return table


def hosmer_lemeshow_table(y_true, y_pred, n_groups=10):
    """Constructs a Hosmer–Lemeshow table.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].
    n_groups : int, optional
        The number of groups to create. The default value is 10, which
        corresponds to deciles of predicted probabilities.

    Returns
    -------
    table : recarray
        A record array with `n_groups` rows and four columns: Group Size,
        Observed Frequency, Predicted Frequency, and Mean Probability.
    """
    if n_groups < 2:
        raise ValueError('Number of groups must be greater than or equal to 2')

    if n_groups > len(y_true):
        raise ValueError('Number of predictions must exceed number of groups')

    table = make_recarray(y_true, y_pred)

    table = [(len(g), g.y_true.sum(), g.y_pred.sum(), g.y_pred.mean())
             for g in np.array_split(table, n_groups)]
    names = ('group_size', 'obs_freq', 'pred_freq', 'mean_prob')
    table = np.rec.fromrecords(table, names=names)

    return table


def hosmer_lemeshow_test(y_true, y_pred, n_groups=10):
    """Performs a Hosmer–Lemeshow test.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].
    n_groups : int, optional
        The number of groups to create. The default value is 10, which
        corresponds to deciles of predicted probabilities.

    Returns
    -------
    C_hat : float
        The test statistic :math:`\hat{C}`.
    df : int
        The degrees of freedom of the test.
    p : float
        The p-value of the test.

    References
    ----------
    .. [1] Hosmer, David W. and Stanley Lemesbow. "Goodness of fit tests
       for the multiple logistic regression model." *Communications in
       Statistics – Theory and Methods* 9 (1980): 1043–1069.
    """
    table = hosmer_lemeshow_table(y_true, y_pred, n_groups=n_groups)

    num = np.square(table.obs_freq - table.mean_prob)
    den = table.group_size * table.mean_prob * (1 - table.mean_prob)

    C_hat = np.sum(num / den)

    df = len(table) - 2
    p = scipy.stats.chisqprob(C_hat, df)

    TestResult = namedtuple('HosmerLemeshowTest', ('C_hat', 'df', 'p'))
    return TestResult(C_hat, df, p)


def osius_rojek_test(X, y_true, y_pred):
    """Tests the normal approximation to the distribution of the Pearson
    chi-square statistic.

    Parameters
    ----------
    X : array
        Design matrix.
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    z : float
        The test statistic :math:`z`.
    df : int
        The degrees of freedom of the test.
    p : float
        The p-value of the test.

    References
    ----------
    .. [1] Osius, Gerhard, and Dieter Rojek. "Normal goodness-of-fit tests
       for multinomial models with large degrees of freedom." *Journal of
       the American Statistical Association* 87 (1992): 1145–1152.
    """
    v = y_pred * (1 - y_pred)
    c = (1 - 2 * y_pred) / v

    chisquare, __, __ = pearson_chisquare(X, y_true, y_pred)

    # Compute the weighted least-squares solution.
    Xv = X * np.sqrt(v)[:, np.newaxis]
    cv = c * np.sqrt(v)

    __, residuals, __, __ = scipy.linalg.lstsq(Xv, cv)

    A = 0  # No variance correction factor; assume unique covariate patterns

    df = len(y_true) - (2 + 1)  # ``p`` = 2 for binary logistic regression
    z = (chisquare - df) / np.sqrt(A + residuals)
    p = scipy.stats.chisqprob(z, df)

    TestResult = namedtuple('OsiusRojekTest', ('z', 'df', 'p'))
    return TestResult(z, df, p)


########################################################################
# Likelihood Measures


def log_likelihood(y_true, y_pred):
    """Computes the log-likelihood.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    l : float
        The log-likelihood.
    """
    l = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    l = l.sum()
    return l


def aic(X, y_true, y_pred):
    """Computes the Akaike information criterion.

    Parameters
    ----------
    X : array
        Design matrix.
    y_true : array_like
        Observed labels, either 0 or 1.
    y_pred : array_like
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    aic : float
        The Akaike information criterion.
    """
    __, k = X.shape
    return 2 * k - 2 * log_likelihood(y_true, y_pred)


########################################################################
# Summary Measures


def pearson_r2(y_true, y_pred):
    """Computes the squared Pearson correlation coefficient.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    R2 : float
        The squared Pearson correlation coefficient.
    """
    r, __ = scipy.stats.pearsonr(y_true, y_pred)
    return np.square(r)


def efrons_r2(y_true, y_pred):
    """Computes Efron's R-Squared.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    R2 : float
        The squared Pearson correlation coefficient.

    References
    ----------
    .. [1] Efron, Bradley. "Regression and ANOVA with Zero-One Data:
       Measures of Residual Variation." *Journal of the American
       Statistical Association* 73 (1978): 113–121.
    """
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - y_true.mean()))
    return 1 - ss_res / ss_tot


def count_r2(y_true, y_pred):
    """Computes the count R-squared.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    R2 : float
        The count R-squared.
    """
    return (binarize(y_pred) == y_true).sum() / len(y_true)


def adjusted_count_r2(y_true, y_pred):
    """Computes the adjusted count R-squared.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    R2 : float
        The adjusted count R-squared.
    """
    __, n = scipy.stats.mode(y_true)
    n = n[0]  # Extract the frequency of the most common label
    return ((binarize(y_pred) == y_true).sum() - n) / (len(y_true) - n)


def mckelvey_zavoina_r2(y_true, y_pred, link='logit'):
    """Computes McKelvey and Zavoina's R-squared.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].
    link : {'logit', 'probit'}
        The link function used in the GLM.

    Returns
    -------
    R2 : float
        McKelvey and Zavoina's R-squared.

    References
    ----------
    .. [1] McKelvey, Richard D., and William Zavoina. "A statistical model
       for the analysis of ordinal level dependent variables." *The Journal
       of Mathematical Sociology* 4 (1975): 103–120.
    """
    if link == 'logit':
        e = np.square(np.pi) / 3
    elif link == 'probit':
        e = 1
    else:
        raise ValueError('Link function must be logit or probit')
    var = np.sum(y_pred * (1 - y_pred))
    return var / (var + e)


def brier_score(y_true, y_pred):
    """Computes the Brier score.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    B : float
        The Brier score.

    References
    ----------
    .. [1] Brier, Glenn W. "Verification of Forecasts Expressed in Terms of
       Probability." *Monthly Weather Review* 78 (1950): 1–3.
    """
    return np.mean(np.square(y_pred - y_true))


def expected_pcp(y_true, y_pred):
    """Computes the expected percentage of correct predictions.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    ePCP : float
        The expected percentage of correct predictions.

    References
    ----------
    .. [1] Herron, Michael C. "Postestimation Uncertainty in Limited
       Dependent Variable Models." *Political Analysis* 8 (1999): 83–98.
    """
    ePCP = y_pred[y_true == 1].sum() + (1 - y_pred[y_true == 0]).sum()
    ePCP /= len(y_true)
    return ePCP


def summary_measures(y_true, y_pred):
    """Produces a text report of summary measures.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    report : str
        A text report of summary measures.
    """
    measures = [('Squared Pearson Correlation Coefficient', pearson_r2),
                ("Efron's R^2", efrons_r2),
                ("McKelvey and Zavoina's R^2", mckelvey_zavoina_r2),
                ('Count R^2', count_r2),
                ('Adjusted Count R^2', adjusted_count_r2),
                ('Brier Score', brier_score),
                ('Expected Percentage of Correct Predictions', expected_pcp)]

    report = '\n'.join('{:<45}{:> 5.3f}'.format(name, method(y_true, y_pred))
                       for name, method in measures)

    return report


########################################################################
# Casewise Statistics


def pregibon_leverages(X, y_pred):
    """Computes Pregibon leverages.

    Parameters
    ----------
    X : array
        Design matrix.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    leverages : array
        Pregibon leverages.

    References
    ----------
    .. [1] Pregibon, Daryl. "Logistic regression diagnostics." *The Annals
       of Statistics* 9 (1981) 705–724.
    """
    v = np.sqrt(y_pred * (1 - y_pred))
    VX = X * v[:, np.newaxis]
    H = VX.dot(np.linalg.solve(VX.T.dot(VX), VX.T))  # Hat matrix
    leverages = H.diagonal()

    if not np.isclose(leverages.sum(), X.shape[1]):
        raise ValueError('Sum of leverages must equal the number of features')

    return leverages


def standardize_residuals(residuals, leverages):
    """Standardizes residuals.

    Parameters
    ----------
    residuals : array
        Pearson or deviance residuals.
    leverages : array
        Pregibon leverages.

    Returns
    -------
    standardized : array
        Standardized residuals.
    """
    return residuals / np.sqrt(1 - leverages)


def pregibon_dbetas(residuals, leverages):
    """Computes Pregibon DBetas.

    Parameters
    ----------
    residuals : array
        Pearson residuals.
    leverages : array
        Pregibon leverages.

    Returns
    -------
    dbetas : array
        Pregibon DBetas.

    References
    ----------
    .. [1] Pregibon, Daryl. "Logistic regression diagnostics." *The Annals
       of Statistics* 9 (1981) 705–724.
    """
    return (np.square(residuals) * leverages) / np.square(1 - leverages)


def case_deltas(residuals, leverages):
    """Computes changes in :math:`X^2` or :math:`D` due to case deletion.

    Parameters
    ----------
    residuals : array
        Pearson or deviance residuals.
    leverages : array
        Pregibon leverages.

    Returns
    -------
    delta_chisq : array
        :math:`\Delta X^2` or :math:`\Delta D`.

    References
    ----------
    .. [1] Pregibon, Daryl. "Logistic regression diagnostics." *The Annals
       of Statistics* 9 (1981) 705–724.
    """
    return np.square(residuals) / (1 - leverages)
