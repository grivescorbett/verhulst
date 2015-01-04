"""Test suite for the bootstrap module."""


import unittest

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import verhulst.bootstrap as vb


class TestBootstrap(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=1200, n_features=20,
                                             n_informative=20, n_redundant=0,
                                             n_repeated=0)

        self.clf = LogisticRegression()
        self.clf.fit(self.X, self.y)

        self.coef = self.clf.coef_.flatten()
        self.confint, __, __, __ = vb.confint(self.X, self.y, self.clf)

    def test_order(self):
        lower, upper = self.confint
        self.assertTrue(all(lower <= upper))

    def test_contains(self):
        lower, upper = self.confint
        self.assertTrue(all((lower <= self.coef) & (self.coef <= upper)))

    def test_n_jobs(self):
        with self.assertRaises(ValueError):
            vb.confint(self.X, self.y, self.clf, n_jobs=-1)


if __name__ == '__main__':
    unittest.main()
