import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import verhulst.plots as vp


np.random.seed(18490215)
X, y = make_classification(n_samples=1200, n_features=20,
                           n_informative=20, n_redundant=0,
                           n_repeated=0)

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_true = y_test
y_pred = clf.predict_proba(X_test)[:, 1]


vp.predicted_probabilities(y_true, y_pred, 20)
plt.show()
