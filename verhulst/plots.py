"""Plotting routines for evaluating binary logistic regressions.

.. autosummary::

    binned_plot
    diagnostic_plots
    ecdf_by_observed_label
    hosmer_lemeshow_plot
    influence_plot
    odds_ratio_plot
    predicted_probabilities
    predicted_probabilities_by_observed_label
    roc_plot
    separation_plot
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from .stats import (pearson_residuals, deviance_residuals, pregibon_dbetas,
                    case_deltas, pregibon_leverages, hosmer_lemeshow_table)
from .util import ntile_name

from .distributions import ECDF  # Vendorized statsmodels.distributions


def binned_plot(y_true, y_pred, n_groups=None, **kwargs):
    """Plots binned residuals by binned predicted probabilities.

    Parameters
    ----------
    y_true : array_like
        Observed labels, either 0 or 1.
    y_pred : array_like
        Predicted probabilities, floats on [0, 1].
    n_groups : int, optional
        The number of groups to create. Automatically computed if ommitted.

    Notes
    -----
    .. plot:: pyplots/binned_plot.py

    References
    ----------
    .. [1] Gelman, Andrew, and Jennifer Hill. 2007. *Data Analysis Using
       Regression and Multilevel/Hierarchical Models*. New York: Cambridge
       University Press.
    """
    n = len(y_true)

    if n_groups is None:
        if n >= 100:
            n_groups = np.floor(np.sqrt(n))
        elif 10 < n < 100:
            n_groups = 10
        else:
            n_groups = np.floor(n / 2)

    if n_groups < 2:
        raise ValueError('Number of groups must be greater than or equal to 2')

    if n_groups > len(y_true):
        raise ValueError('Number of predictions must exceed number of groups')

    recarray = np.recarray(len(y_true), [('y_pred', 'f8'), ('residual', 'f8')])
    recarray['y_pred'] = y_pred
    recarray['residual'] = y_true - y_pred
    recarray.sort(order='y_pred')

    table = [(len(g), g.y_pred.mean(), g.residual.mean(), g.residual.std(), 0.)
             for g in np.array_split(recarray, n_groups)]
    names = ['n', 'y_pred_mean', 'residual_mean', 'residual_std', 'error']
    table = np.rec.fromrecords(table, names=names)

    table.error = 2 * table.residual_std / np.sqrt(table.n)

    plt.scatter(table.y_pred_mean, table.residual_mean, **kwargs)
    plt.plot(table.y_pred_mean, table.error, c='k')
    plt.plot(table.y_pred_mean, -table.error, c='k')

    plt.axhline(0, c='k', ls='--')

    __, __, y1, y2 = plt.axis()
    plt.axis((0, 1, y1, y2))

    plt.title('Binned Residual Plot')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Average Residual')

    plt.tight_layout()


def diagnostic_plots(X, y_true, y_pred, **kwargs):
    """Produces essential diagnostic plots for binary logistic regressions.

    Parameters
    ----------
    X : array
        Design matrix.
    y_true : array_like
        Observed labels, either 0 or 1.
    y_pred : array_like
        Predicted probabilities, floats on [0, 1].

    Notes
    -----
    .. plot:: pyplots/diagnostic_plots.py

    References
    ----------
    .. [1] Hosmer, David W., Jr., Stanley Lemeshow, and Rodney X.
       Sturdivant. *Applied Logistic Regression*. 3rd ed. New York: Wiley,
       2013.
    """
    r = pearson_residuals(y_true, y_pred)
    d = deviance_residuals(y_true, y_pred)
    leverages = pregibon_leverages(X, y_pred)

    delta_X2 = case_deltas(r, leverages)
    delta_D = case_deltas(d, leverages)
    dbetas = pregibon_dbetas(r, leverages)

    ax = plt.subplot(311)
    plt.scatter(y_pred, delta_X2, **kwargs)
    plt.ylabel(r'$\Delta \chi^2$')

    plt.subplot(312, sharex=ax)
    plt.scatter(y_pred, delta_D, **kwargs)
    plt.ylabel(r'$\Delta D$')

    plt.subplot(313, sharex=ax)
    plt.scatter(y_pred, dbetas, **kwargs)
    plt.xlabel('Predicted Probability')
    plt.ylabel(r'$\Delta \hat{\beta}$')

    __, __, y1, y2 = plt.axis()
    plt.axis((0, 1, y1, y2))

    plt.tight_layout()


def ecdf_by_observed_label(y_true, y_pred):
    """Plots the empirical cumulative density functions by observed label.

    Parameters
    ----------
    y_true : array_like
        Observed labels, either 0 or 1.
    y_pred : array_like
        Predicted probabilities, floats on [0, 1].

    Notes
    -----
    .. plot:: pyplots/ecdf_by_observed_label.py
    """
    x = np.linspace(0, 1)

    ecdf = ECDF(y_pred[y_true == 0])
    y_0 = ecdf(x)

    ecdf = ECDF(y_pred[y_true == 1])
    y_1 = ecdf(x)

    plt.step(x, y_0, label='Observed label 0')
    plt.step(x, y_1, label='Observed label 1')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Proportion')
    plt.title('Empirical Cumulative Density Functions by Observed Label')
    plt.legend(loc='lower right')


def hosmer_lemeshow_plot(y_true, y_pred, n_groups=10, colors=('red', 'blue')):
    """Plots a Hosmer–Lemeshow table.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].
    n_groups : int, optional
        The number of groups to create. The default value is 10, which
        corresponds to deciles of predicted probabilities.
    colors : tuple, optional
        The colors of the predicted and observed bars, respectively.

    Notes
    -----
    .. plot:: pyplots/hosmer_lemeshow_plot.py
    """
    table = hosmer_lemeshow_table(y_true, y_pred, n_groups)

    index = np.arange(n_groups)
    width = 0.35

    fig, ax = plt.subplots()

    plt.bar(index, table.pred_freq, width, color=colors[0],
            label='Predicted')

    plt.bar(index + width, table.obs_freq, width, color=colors[1],
            label='Observed')

    plt.xlabel('{} of Predicted Probabilities'.format(ntile_name(n_groups)))
    plt.ylabel('Frequency')
    plt.title('Observed versus Predicted Frequencies')
    plt.xticks(index + width, index + 1)
    plt.legend(loc='upper left')
    plt.tight_layout()


def influence_plot(X, y_true, y_pred, **kwargs):
    """Produces an influence plot.

    Parameters
    ----------
    X : array
        Design matrix.
    y_true : array_like
        Observed labels, either 0 or 1.
    y_pred : array_like
        Predicted probabilities, floats on [0, 1].

    Notes
    -----
    .. plot:: pyplots/influence_plot.py
    """
    r = pearson_residuals(y_true, y_pred)
    leverages = pregibon_leverages(X, y_pred)

    delta_X2 = case_deltas(r, leverages)
    dbetas = pregibon_dbetas(r, leverages)

    plt.scatter(y_pred, delta_X2, s=dbetas * 800, **kwargs)

    __, __, y1, y2 = plt.axis()
    plt.axis((0, 1, y1, y2))

    plt.xlabel('Predicted Probability')
    plt.ylabel(r'$\Delta \chi^2$')

    plt.tight_layout()


def odds_ratio_plot(coef, confint, names):
    """Plots confidence intervals around odds ratios.

    Note that `vs.confint` returns an array of shape `(2, n)`. This
    preserves similarity to the `bca` function in the `bootstrap` R
    package, which served as the reference implementation. This function
    expects the transposed array of shape `(2, n)` and will raise a
    `ValueError` if this is not the case.

    Parameters
    ----------
    coef : array
        An array of odds ratios.
    confint : array
        An array of confidence intervals around the odds ratios.
    names : array
        A collection of strings corresponding to the feature names.
    """
    nrow, ncol = confint.shape
    if ncol != 2:
        raise ValueError('Ambiguous lower and upper confidence bounds')

    if not len(coef) == nrow == len(names):
        raise ValueError('Arguments have mismatched dimensions')

    index = np.arange(nrow)
    confint = confint.T  # `plt.errorbar` expects shape `(2, n)`

    plt.errorbar(coef, index, xerr=confint, fmt='o')
    plt.axvline(1, c='k', ls='--')

    plt.yticks(index, names)
    plt.ylim(-0.5, nrow - 0.5)

    plt.title('Bootstrapped Odds Ratio Confidence Intervals')
    plt.xlabel('Odds Ratio')

    plt.tight_layout()


def predicted_probabilities(y_true, y_pred, n_groups=30):
    """Plots the distribution of predicted probabilities.

    Parameters
    ----------
    y_true : array_like
        Observed labels, either 0 or 1.
    y_pred : array_like
        Predicted probabilities, floats on [0, 1].
    n_groups : int, optional
        The number of groups to create. The default value is 30.

    Notes
    -----
    .. plot:: pyplots/predicted_probabilities.py
    """
    plt.hist(y_pred, n_groups)
    plt.xlim([0, 1])
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')

    title = 'Distribution of Predicted Probabilities (n = {})'
    plt.title(title.format(len(y_pred)))

    plt.tight_layout()


def predicted_probabilities_by_observed_label(y_true, y_pred, n_groups=30):
    """Plots distributions of predicted probabilities by observed label.

    Parameters
    ----------
    y_true : array_like
        Observed labels, either 0 or 1.
    y_pred : array_like
        Predicted probabilities, floats on [0, 1].
    n_groups : int, optional
        The number of groups to create. The default value is 30.

    Notes
    -----
    .. plot:: pyplots/predicted_probabilities_by_observed_label.py
    """
    x = y_pred[y_true == 0]
    n = len(x)
    weights = np.zeros_like(x) + 1 / n

    ax = plt.subplot(211)
    plt.hist(x, n_groups, (0, 1), weights=weights)
    plt.title('Observed Label 0 (n = {})'.format(n), fontsize=16)

    x = y_pred[y_true == 1]
    n = len(x)
    weights = np.zeros_like(x) + 1 / n

    plt.subplot(212, sharex=ax, sharey=ax)
    plt.hist(x, n_groups, (0, 1), weights=weights)
    plt.title('Observed Label 1 (n = {})'.format(n), fontsize=16)

    plt.tight_layout()

    plt.suptitle('Distributions of Predicted Probabilities by Observed Label',
                 fontsize=18)

    fig = plt.gcf()
    fig.text(0.5, 0.04, 'Predicted Probability', ha='center', va='center')
    fig.text(0.02, 0.5, 'Proportion', ha='center', va='center',
             rotation='vertical')
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.1, top=0.9)


def roc_plot(y_true, y_pred):
    """Plots a receiver operating characteristic.

    Parameters
    ----------
    y_true : array_like
        Observed labels, either 0 or 1.
    y_pred : array_like
        Predicted probabilities, floats on [0, 1].

    Notes
    -----
    .. plot:: pyplots/roc_plot.py

    References
    ----------
    .. [1] Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python."
       *Journal of Machine Learning Research* 12 (2011): 2825–2830.
    .. [2] scikit-learn developers. "Receiver operating characteristic (ROC)."
       Last modified August 2013.
       http://scikit-learn.org/stable/auto_examples/plot_roc.html.
    """
    fpr, tpr, __ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC curve (area = {:0.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')


def separation_plot(y_true, y_pred, colors=('#eeeeee', '#348ABD'),
                    original_style=False):
    """Plots a separation plot.

    Parameters
    ----------
    y_true : array_like
        Observed labels, either 0 or 1.
    y_pred : array_like
        Predicted probabilities, floats on [0, 1].
    colors : tuple, optional
        The colors of the observed 0 and 1 cases, respectively.
    original_style : bool, optional
        Whether to use the plot style presented in the original paper.
        If ``True``, overrides the ``colors`` argument.

    Notes
    -----
    .. plot:: pyplots/separation_plot.py

    References
    ----------
    .. [1] Greenhill, Brian, Michael D. Ward, and Audrey Sacks. "The
       Separation Plot: A New Visual Method for Evaluating the Fit of
       Binary Models." *American Journal of Political Science* 55 (2011):
       991–1002. doi:10.1111/j.1540-5907.2011.00525.x.
    .. [2] Davidson-Pilon, Cam. 2014. "Bayesian Methods for Hackers."
       https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers.
    """
    n = len(y_true)
    x = np.arange(n)

    if original_style:
        colors = ('#fdf2db', '#e44a32')

    colors = np.array(colors)

    index = np.argsort(y_pred)

    plt.bar(x, np.ones(n), width=1, color=colors[y_true[index].astype(int)],
            edgecolor='none')
    plt.plot(x, y_pred[index], 'k', linewidth=1, drawstyle='steps-post')

    plt.vlines([(1 - y_pred[index]).sum()], [0], [1])  # Expected value line

    plt.xlim(0, n - 1)

    if original_style:
        plt.grid(False)
        plt.axis('off')

    plt.tight_layout()
