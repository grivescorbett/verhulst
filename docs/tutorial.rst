.. _tutorial:

Tutorial
========

Import Libraries
----------------

.. ipython:: python

   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   from sklearn.cross_validation import train_test_split
   from sklearn.linear_model import LogisticRegression

.. ipython:: python

   import verhulst.plots as vp
   import verhulst.stats as vs

Acquire Data
------------

.. ipython:: python

   url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

.. ipython:: python
   :okwarning:

   names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 'hours_per_week',
            'native_country', 'income']
   df = pd.read_csv(url, sep=', ', header=None, names=names, na_values=['?'])

Preprocess
----------

.. ipython:: python

   df.dropna(inplace=True)

.. ipython:: python

   df['government'] = df.workclass.str.endswith('gov')
   df['selfemp'] = df.workclass.str.startswith('Self-emp')
   df['white'] = df.race == 'White'
   df['male'] = df.sex == 'Male'
   df['parttime'] = df.hours_per_week <= 35
   df['native'] = df.native_country == 'United-States'
   df.income = df.income == '>50K'

.. ipython:: python

   columns = ['age', 'government', 'selfemp', 'education_num', 'white',
              'male', 'parttime', 'native', 'income']
   df = df.loc[:, columns].astype(int)

Partition
---------

.. ipython:: python

   y = df.pop('income').values
   X = df.values

.. ipython:: python

   np.random.seed(18490215)
   X_train, X_test, y_train, y_test = train_test_split(X, y)

.. ipython:: python

   clf = LogisticRegression()
   clf.fit(X_train, y_train)

.. ipython:: python

   X = X_test
   y_true = y_test
   y_pred = clf.predict_proba(X_test)[:, 1]

Analyze
-------

.. ipython:: python

   pd.Series(np.exp(clf.coef_.flatten()), index=df.columns)

.. ipython:: python

   pd.Series(y_pred).describe()

.. ipython:: python

   table = vs.hosmer_lemeshow_table(y_true, y_pred)

.. ipython:: python

   vs.hosmer_lemeshow_test(y_true, y_pred)

.. ipython:: python

   @savefig hosmer_lemeshow_plot.png width=6in
   vp.hosmer_lemeshow_plot(y_true, y_pred, colors=('#348ABD', '#A60628'))

.. ipython:: python

   @savefig diagnostic_plots.png width=6in
   vp.diagnostic_plots(X, y_true, y_pred)

.. ipython:: python

   @savefig binned_plot.png width=6in
   vp.binned_plot(y_true, y_pred)

.. ipython:: python

   print(vs.summary_measures(y_true, y_pred))

.. ipython:: python

   @savefig ecdf_by_observed_label.png width=6in
   vp.ecdf_by_observed_label(y_true, y_pred)

.. ipython:: python

   @savefig predicted_probabilities_by_observed_label.png width=6in
   vp.predicted_probabilities_by_observed_label(y_true, y_pred)
