Verhulst
========

Verhulst is a MIT-licensed_ Python library for evaluating binary logistic
regressions fitted with scikit-learn_.

scikit-learn takes a machine learning approach to data analysis and executes
numerical routines using liblinear_ return certain intermediate results of
the logistic regression fitting. statsmodels_ takes an econometric approach
to data analysis but is not fully compatible with scikit-learn classifiers.
Verhulst aims to fill this gap by providing a consistent API to statistical
analysis and plotting routines commonly used to evaluate logistic regression
fit.

Features
--------

Statistical Analysis

- Pearson Chi-Square Statistic and Deviance
- Hosmer–Lemeshow Tests
- Likelihood Measures
- Summary Measures
- Casewise Statistics

Plotting

- Diagnostic Plots
- Residual Plots
- Goodness-of-Fit Plots

scikit-learn fits logistic regressions using liblinear_, which does not return
the likelihoods of the null or fitted models. Verhulst therefore omits popular
likelihood-based (Cox-Snell, Nagelkerke) and log-likelihood-based (McFadden)
summary measures.

A Remark About Interpretation
-----------------------------

Some of the statistical tests and measures implemented in Verhulst are not
intuitively easy to explain. Some are outdated and not recommended for general
use. Nevertheless, they are included here because they continue to be used.

In general, these statistical tests and measures are most useful when initially
building a model. Given their problematic interpretations and potential to
mislead, however, many authors discourage routinely including them in published
work.

Installation
------------

Verhulst supports Python 3.2, 3.3, and 3.4.

pip_ can install Verhulst from GitHub:

::

   pip install git+git://github.com/rpetchler/verhulst.git

The following packages are dependencies for Verhulst:

- numpy_
- scipy_
- matplotlib_

In addition, the ``statsmodels.distributions`` module is vendorized.

Documentation
-------------

Documentation is written in `reStructured Text`_ and numpydoc_ and generated
by Sphinx_. The Makefile_ in ``docs/`` contains targets for HTML and PDF
documentation. The following command generates HTML documentation:

::

   $ make html

Run a local webserver (e.g., ``python3 -m http.server``) in the directory
``docs/_build/html/`` in order to view the documentation in a web browser.

References
----------

.. [1] Hosmer, David W., Jr., Stanley Lemeshow, and Rodney X. Sturdivant.
   *Applied Logistic Regression*. 3rd ed. New York: Wiley, 2013.

.. _MIT-licensed: http://opensource.org/licenses/MIT
.. _scikit-learn: http://scikit-learn.org/
.. _liblinear: http://www.csie.ntu.edu.tw/~cjlin/liblinear/
.. _statsmodels: http://statsmodels.sourceforge.net/
.. _pip: https://github.com/pypa/pip
.. _numpy: http://www.numpy.org/
.. _scipy: http://www.scipy.org/
.. _matplotlib: http://matplotlib.org/
.. _`reStructured Text`: http://docutils.sourceforge.net/rst.html
.. _numpydoc: https://github.com/numpy/numpydoc
.. _Sphinx: http://sphinx-doc.org/
.. _Makefile: https://www.gnu.org/software/make/
