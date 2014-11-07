"""Test suite for the utility module."""


import unittest

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import verhulst.util


class TestBinarize(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([0, 0, 1, 0, 1, 1])
        self.y_pred = np.array([0.774, 0.364, 0.997, 0.728, 0.961, 0.422])

    def test_threshold_default(self):
        result = verhulst.util.binarize(self.y_pred)
        np.testing.assert_equal(result, np.array([1, 0, 1, 1, 1, 0]))

    def test_threshold_argument(self):
        result = verhulst.util.binarize(self.y_pred, threshold=0.3)
        np.testing.assert_equal(result, np.array([1, 1, 1, 1, 1, 1]))

    def test_threshold_minimum(self):
        with self.assertRaises(ValueError):
            verhulst.util.binarize(self.y_pred, threshold=-0.5)

    def test_threshold_maximum(self):
        with self.assertRaises(ValueError):
            verhulst.util.binarize(self.y_pred, threshold=1.5)


class TestMakeRecarray(unittest.TestCase):

    def setUp(self):
        np.random.seed(18490215)

        X, y = make_classification(n_samples=1200, n_features=20,
                                   n_informative=20, n_redundant=0,
                                   n_repeated=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        self.y_true = y_test
        self.y_pred = clf.predict_proba(X_test)[:, 1]

    def test_group_balance_type(self):
        result = verhulst.util.make_recarray(self.y_true, self.y_pred)
        self.assertIsInstance(result, np.recarray)

    def test_group_balance_shape(self):
        result = verhulst.util.make_recarray(self.y_true, self.y_pred).shape
        expected = (len(self.y_true), )
        self.assertEqual(result, expected)


class TestNtileName(unittest.TestCase):

    def test_argument_valid(self):
        result = verhulst.util.ntile_name(4)
        self.assertEqual(result, 'Quartile')

    def test_argument_default(self):
        result = verhulst.util.ntile_name('4')
        self.assertEqual(result, '4-tile')


if __name__ == '__main__':
    unittest.main()
