.. _tutorial:

Tutorial
========

Import Libraries
----------------

.. doctest::

  >>> import matplotlib.pyplot as plt
  >>> import numpy as np
  >>> import pandas as pd
  >>> from sklearn.cross_validation import train_test_split
  >>> from sklearn.linear_model import LogisticRegression

.. doctest::

  >>> import verhulst.plots as vp
  >>> import verhulst.stats as vs

Acquire Data
------------

.. doctest::

   >>> url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

.. doctest::

   >>> names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
   ...         'marital_status', 'occupation', 'relationship', 'race', 'sex',
   ...         'capital_gain', 'capital_loss', 'hours_per_week',
   ...         'native_country', 'income']
   >>> df = pd.read_csv(url, sep=', ', header=None, names=names, na_values=['?'])

Preprocess
----------

.. doctest::

   >>> df.dropna(inplace=True)

.. doctest::

   >>> df['government'] = df.workclass.str.endswith('gov')
   >>> df['selfemp'] = df.workclass.str.startswith('Self-emp')
   >>> df['white'] = df.race == 'White'
   >>> df['male'] = df.sex == 'Male'
   >>> df['parttime'] = df.hours_per_week <= 35
   >>> df['native'] = df.native_country == 'United-States'
   >>> df.income = df.income == '>50K'

.. doctest::

   >>> columns = ['age', 'government', 'selfemp', 'education_num', 'white',
   ...            'male', 'parttime', 'native', 'income']
   >>> df = df.loc[:, columns].astype(int)

Partition
---------

.. doctest::

   >>> y = df.pop('income').values
   >>> X = df.values

.. doctest::

   >>> np.random.seed(18490215)
   >>> X_train, X_test, y_train, y_test = train_test_split(X, y)

.. doctest::

   >>> clf = LogisticRegression()
   >>> clf.fit(X_train, y_train)

.. doctest::

   >>> X = X_test
   >>> y_true = y_test
   >>> y_pred = clf.predict_proba(X_test)[:, 1]

Analyze
-------

.. doctest::

   >>> pd.Series(np.exp(clf.coef_.flatten()), index=df.columns)

.. doctest::

   >>> pd.Series(y_pred).describe()

.. doctest::

   >>> table = vs.hosmer_lemeshow_table(y_true, y_pred)
   >>> pd.DataFrame(table)
      group_size  obs_freq   pred_freq  mean_prob
   0         755         9   11.926261   0.015796
   1         754        31   34.290538   0.045478
   2         754        74   60.268854   0.079932
   3         754        66   93.830936   0.124444
   4         754       121  125.826623   0.166879
   5         754       186  162.613887   0.215668
   6         754       244  217.049333   0.287864
   7         754       280  285.457396   0.378591
   8         754       405  378.368113   0.501814
   9         754       548  532.861238   0.706713

.. doctest::

   >>> vs.hosmer_lemeshow_test(y_true, y_pred)
   (4207.0817350647221, 8, 0.0)

.. doctest::

   >>> vp.hosmer_lemeshow_plot(y_true, y_pred)

.. doctest::

   >>> vp.diagnostic_plots(X, y_true, y_pred)

.. doctest::

   >>> vp.binned_plot(y_true, y_pred)

.. doctest::

   >>> print(vs.summary_measures(y_true, y_pred))
   Squared Pearson Correlation Coefficient       0.252
   Efron's R^2                                   0.251
   McKelvey and Zavoina's R^2                    0.997
   Count R^2                                     0.790
   Adjusted Count R^2                            0.194
   Brier Score                                   0.144
   Expected Percentage of Correct Predictions    0.712

.. doctest::

   >>> vp.ecdf_by_observed_label(y_true, y_pred)

.. doctest::

   >>> vp.predicted_probabilities_by_observed_label(y_true, y_pred)
