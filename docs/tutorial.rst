.. _tutorial:

Tutorial
========

Import Libraries
----------------

First we import numpy_ and pandas_ using the standard abbreviations. (pandas is
not a dependency of Verhulst but it simplifies importing and manipulating data
prior to fitting a model.) We also import components of scikit-learn_.

.. ipython:: python

   import numpy as np
   import pandas as pd
   from sklearn.cross_validation import train_test_split
   from sklearn.linear_model import LogisticRegression

We also need the plotting and statistical analysis modules from Verhulst.

.. ipython:: python

   import verhulst.plots as vp
   import verhulst.stats as vs

Acquire data
------------

For this tutorial we will use the popular Adult_ data set from the `UCI Machine
Learning Repository`_. This data set contains 48,842 records extracted from the
1994 census database. The response is a binary variable indicating whether a
person earned more than $50,000 annually. The data set comes partitioned into
training and testing sets; in order to keep this tutorial simple we download
only the training set of 32,561 records and partition it ourselves.

The following URL directs to the comma-delimited input file:

.. ipython:: python

   url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

pandas can read CSV files from HTTP sources. We specify that the file contains
no header and add column names, and we convert ``?`` characters to ``np.nan``,
the internal representation pandas uses for missing data.

.. ipython:: python
   :okwarning:

   names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 'hours_per_week',
            'native_country', 'income']
   df = pd.read_csv(url, sep=', ', header=None, names=names, na_values=['?'])

Preprocess
----------

Many records have missing values for ``workclass``, ``occupation``, or
``native_country``. Rather than fill these missing values we simply drop the
incomplete records. Our data set now contains 30,162 records, indicating that
only 2,399 records, or 7.3% of the records in the original data set, contained
missing values.

.. ipython:: python

   df.isnull().sum()

   df.dropna(inplace=True)

   df.shape

Next we construct features that might predict whether a person earned more than
$50,000 in 1994. We manually create these features based on theoretical
intuition rather than expanding every categorical variable (using
``pd.get_dummies``, for example). This keeps our feature space small and
simplifies the interpretation of our results.

.. ipython:: python

   df['government'] = df.workclass.str.endswith('gov')
   df['selfemp'] = df.workclass.str.startswith('Self-emp')
   df['white'] = df.race == 'White'
   df['male'] = df.sex == 'Male'
   df['parttime'] = df.hours_per_week <= 35
   df['native'] = df.native_country == 'United-States'
   df.income = df.income == '>50K'

Finally, we extract the features we created and convert them from Boolean to
integer type if necessary.

.. ipython:: python

   columns = ['age', 'government', 'selfemp', 'education_num', 'white',
              'male', 'parttime', 'native', 'income']
   df = df.loc[:, columns].astype(int)

(It is good practice to conduct exploratory analysis before and during the
preprocessing step, but we omit this step in order to keep this tutorial brief.)

Partition
---------

scikit-learn_ requires that features and the response be separate rather than
combined. We therefore set up a feature matrix and response vector.

.. ipython:: python

   y = df.pop('income').values
   X = df.values

Next we partition the feature matrix and response vector into training and
testing sets. (We set the seed to make our results reproducible.) Our training
set contains 22,621 records and our testing set contains 7,541 observations.

.. ipython:: python

   np.random.seed(18490215)
   X_train, X_test, y_train, y_test = train_test_split(X, y)
   X_train.shape
   X_test.shape

Our response is unbalanced: only 25.1% of the individuals in our training set
earned more than $50,000 in 1994.

.. ipython:: python

   pd.value_counts(y_train)

We initialize and fit a logistic regression using scikit-learn.

.. ipython:: python

   clf = LogisticRegression()
   clf.fit(X_train, y_train)

For convenience we rename the feature matrix and response vector of the test set
and extract the predicted probability of a record being in the ``1`` class.

.. ipython:: python

   X = X_test
   y_true = y_test
   y_pred = clf.predict_proba(X_test)[:, 1]

Is the model well-specified?
----------------------------

The first step in analyzing our model is to determine whether the logistic
model is well-specified. The simplest way to do this is to compute the Pearson
chi-square statistic and deviance, which are summary statistics based on the
Pearson residual and deviance residual, respectively. Under the assumption that
the fitted logistic regression is correct, these statistics are supposed to
follow a chi-square distribution with :math:`J - (p + 1)` degrees of freedom,
where :math:`J` is the number of distinct values observed and :math:`p` is the
number of features.

Verhulst can compute these statistics and their associated *p*-values. For this
model both statistics have large *p*-values. We therefore fail to reject the
null hypothesis that there is no difference between the true probabilities and
those predicted by the logistic model.

.. ipython:: python

   vs.pearson_chisquare(X, y_true, y_pred)
   vs.deviance(X, y_true, y_pred)

Although popular, these omnibus goodness-of-fit tests produce incorrect results
when :math:`J \approx n`. (The theory underlying this statement is beyond the
scope of this tutorial.) In order to address this problem, Hosmer and Lemeshow
(1980) and Lemeshow and Hosmer (1982) proposed grouping values based on
predicted probabilities. The most common approach is calculate the chi-square
statistic based on ten groups of observed and predicted frequencies.

Verhulst can easily create and plot such a Hosmer–Lemeshow table. We convert
the returned numpy record array to a pandas DataFrame for better printing
alignment.

.. ipython:: python

   pd.DataFrame(vs.hosmer_lemeshow_table(y_true, y_pred))

   @savefig hosmer_lemeshow_plot.png width=6in
   vp.hosmer_lemeshow_plot(y_true, y_pred, colors=('#348ABD', '#A60628'))

Here we formed ten groups of approximately 754 records each. The mean predicted
probability of each group ascends monotonically, and the observed and predicted
frequency are very close in each decile of predicted probabilities. Both of
these characteristics suggest that the logistic model is appropriate.

Verhulst can conduct a Hosmer–Lemeshow test to verify these observations:

.. ipython:: python

   vs.hosmer_lemeshow_test(y_true, y_pred)

The *p*-value is nearly zero, which indicates that we should reject the null
hypothesis that there is no difference between the true probabilities and those
predicted by the logistic model. In other words, our logistic model does not
fit our data. This is common with large sample sizes, however, which cause even
small differences between the observed and predicted frequencies to inflate the
*p*-value. In this case we can safely ignore the Hosmer–Lemeshow because our
sample size is 7,541 records.

How do outliers affect the model?
---------------------------------

.. ipython:: python

   @savefig diagnostic_plots.png width=6in
   vp.diagnostic_plots(X, y_true, y_pred)

Verhulst makes it easy to find observations with large case deltas. For example,
we can investigate the records that have :math:`\Delta X^2` values greater than
60. These records correspond to the two points in the upper-left corner of the
first subplot in the previous figure.

.. ipython:: python

   residuals = vs.pearson_residuals(y_true, y_pred)
   leverages = vs.pregibon_leverages(X, y_pred)
   delta_X2 = vs.case_deltas(residuals, leverages)
   pd.DataFrame(X[delta_X2 > 60], columns=df.columns)

Both individuals are middle-aged females. The first has only three years of
education; the second has only seven years of education, works part-time, and is
non-native. Both individuals earned more than $50,000 in 1994, making them clear
outliers based on their demographic attributes.

.. ipython:: python

   @savefig binned_plot.png width=6in
   vp.binned_plot(y_true, y_pred)

How effectively does the model discriminate between classes?
------------------------------------------------------------

.. ipython:: python

   print(vs.summary_measures(y_true, y_pred))

We can also compute descriptive statistics for the predicted probabilities. The
mean predicted probability is greater than the median; this suggests that the
distribution of predicted probability is skewed left. In addition, the range of
predicted probabilities reaches a lower bound of nearly 0 but an upper bound of
0.965. Both of these observations make sense because our response has a greater
proportion of records in the ``0`` class.

.. ipython:: python

   pd.Series(y_pred).describe()

.. ipython:: python

   @savefig ecdf_by_observed_label.png width=6in
   vp.ecdf_by_observed_label(y_true, y_pred)

.. ipython:: python

   @savefig predicted_probabilities_by_observed_label.png width=6in
   vp.predicted_probabilities_by_observed_label(y_true, y_pred)

What is the effect of each feature?
-----------------------------------

We can exponentiate the coefficients of the logistic regression in order to
obtain odds ratios. Gender has the largest effect size; males are 3.27 times as
likely to earn more than $50,000 per year than are females. Years of education
and ethnicity also have large effect sizes. Part-time status has a large,
negative effect: part-time workers are only 0.26 times as likely to earn more
than $50,000 compared to full-time workers.

.. ipython:: python

   pd.Series(np.exp(clf.coef_.flatten()), index=df.columns)

.. _numpy: http://www.numpy.org/
.. _pandas: http://pandas.pydata.org/
.. _scikit-learn: http://scikit-learn.org/
.. _Adult: http://archive.ics.uci.edu/ml/datasets/Adult
.. _`UCI Machine Learning Repository`: http://archive.ics.uci.edu/ml/
