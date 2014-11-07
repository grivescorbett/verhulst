"""Utility functions."""


import numpy as np


def binarize(array, threshold=0.5):
    """Converts continuous predicted probabilities into binary labels.

    Parameters
    ----------
    array : array
        Predicted probabilities, floats on [0, 1].
    threshold : float, optional
        Threshold at or above which to assign the label ``1``.

    Returns
    -------
    array : array
        Observed labels, either 0 or 1.
    """
    if threshold < 0 or threshold > 1:
        raise ValueError('`threshold` must be between 0 and 1')
    return np.where(array >= threshold, 1, 0)


def make_recarray(y_true, y_pred):
    """Combines arrays into a recarray.

    Parameters
    ----------
    y_true : array
        Observed labels, either 0 or 1.
    y_pred : array
        Predicted probabilities, floats on [0, 1].

    Returns
    -------
    table : recarray
        A record array with observed label and predicted probability
        columns, sorted by predicted probability.
    """
    recarray = np.recarray(len(y_true), [('y_true', 'u8'), ('y_pred', 'f8')])
    recarray['y_true'] = y_true
    recarray['y_pred'] = y_pred
    recarray.sort(order='y_pred')
    return recarray


def ntile_name(n):
    """Returns the ntile name corresponding to an ntile integer.

    Parameters
    ----------
    n : int
        An ntile integer.

    Returns
    -------
    ntile_name : str
        The corresponding ntile name.
    """
    ntile_names = {
        4: 'Quartile',
        5: 'Quintile',
        6: 'Sextile',
        10: 'Decile',
        12: 'Duodecile',
        20: 'Vigintile',
        100: 'Percentile'
    }
    return ntile_names.get(n, '{}-tile'.format(n))
