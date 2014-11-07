Verhulst
========

Verhulst is a `BSD-licensed`_ Python library for evaluating binary logistic
regressions fitted with `scikit-learn`_.

scikit-learn takes a machine learning approach to data analysis and executes
numerical routines using `liblinear`_ return certain intermediate results of
the logistic regression fitting. `statsmodels`_ takes an econometric approach
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

Installation
------------

Verhulst supports Python 3.2, 3.3, and 3.4.

`pip`_ can install Verhulst from GitHub:

::

   pip install git+git://github.com/rpetchler/verhulst.git

The following packages are dependencies for Verhulst:

- `numpy`_
- `scipy`_
- `matplotlib`_

In addition, the `statsmodels.distributions` module is vendorized.

Documentation
-------------

Documentation is written in `reStructured Text`_ and `numpydoc`_ and generated
by `Sphinx`_. The `Makefile`_ in `docs/` contains targets for HTML and PDF
documentation. The following command generates HTML documentation:

::

   $ make html

Run a local webserver (e.g., `python3 -m http.server`) in the directory
`docs/_build/html/` in order to view the documentation in a web browser.

Contribute
----------

Fork this repository on GitHub and make changes to a branch. Write tests to
demonstrate that your changes fix a bug or implement a feature as expected.
Send a pull request to @rpetchler to have your fork integrated.

.. _BSD-licensed: http://opensource.org/licenses/BSD-3-Clause
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
