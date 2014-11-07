"""Test suite for the statistics module."""


import unittest

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import verhulst.stats as vs


class SmallFixture(unittest.TestCase):

    def setUp(self):
        """Set up sample data.

        References
        ----------
        .. [1] Greenhill, Brian, Michael D. Ward, and Audrey Sacks. "The
           Separation Plot: A New Visual Method for Evaluating the Fit of
           Binary Models." *American Journal of Political Science* 55
           (2011): 991–1002. doi:10.1111/j.1540-5907.2011.00525.x.
        """
        self.y_true = np.array([0, 0, 1, 0, 1, 1])
        self.y_pred = np.array([0.774, 0.364, 0.997, 0.728, 0.961, 0.422])


class LargeFixture(unittest.TestCase):

    def setUp(self):
        np.random.seed(18490215)

        X, y = make_classification(n_samples=1200, n_features=20,
                                   n_informative=20, n_redundant=0,
                                   n_repeated=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        self.X = X_test
        self.y_true = y_test
        self.y_pred = clf.predict_proba(X_test)[:, 1]


class TestPearsonChiSquareDeviance(SmallFixture):

    def test_pearson_residuals(self):
        result = vs.pearson_residuals(self.y_true, self.y_pred)
        expected = np.array([-1.85061578, -0.756523, 0.0548546, -1.63599223,
                             0.20145155, 1.17032826])
        np.testing.assert_allclose(result, expected)

    def test_pearson_chisquare(self):
        result = vs.pearson_chisquare(self.y_true, self.y_pred)
        expected = (8.0868363931758829, 3, 0.044250738758029588)
        np.testing.assert_allclose(result, expected)

    def test_deviance_residuals(self):
        result = vs.deviance_residuals(self.y_true, self.y_pred)
        expected = np.array([-1.72465665, -0.9513745, 0.07751786, -1.61366243,
                             0.28206691, 1.31358286])
        np.testing.assert_allclose(result, expected)

    def test_deviance(self):
        result = vs.deviance(self.y_true, self.y_pred)
        expected = (8.2945311040325489, 3, 0.040301114529601764)
        np.testing.assert_allclose(result, expected)


class TestHosmerLemeshowTests(LargeFixture):

    def test_classification_balance_n_groups_minimum(self):
        with self.assertRaises(ValueError):
            vs.classification_balance(self.y_true, self.y_pred, 1)

    def test_classification_balance_n_groups_maximum(self):
        nrow, __ = self.X.shape
        with self.assertRaises(ValueError):
            vs.classification_balance(self.y_true, self.y_pred, nrow + 1)

    def test_classification_balance_type(self):
        result = vs.classification_balance(self.y_true, self.y_pred)
        self.assertIsInstance(result, np.recarray)

    def test_classification_balance_shape(self):
        result = vs.classification_balance(self.y_true, self.y_pred).shape
        expected = (10, )
        self.assertEqual(result, expected)

    def test_classification_balance_names(self):
        result = vs.classification_balance(self.y_true, self.y_pred)
        result = result.dtype.names
        expected = ('group_size', 'n_0', 'n_1', 'prop_0', 'prop_1')
        self.assertEqual(result, expected)

    def test_hosmer_lemeshow_table_n_groups_minimum(self):
        with self.assertRaises(ValueError):
            vs.hosmer_lemeshow_table(self.y_true, self.y_pred, 1)

    def test_hosmer_lemeshow_table_n_groups_maximum(self):
        nrow, __ = self.X.shape
        with self.assertRaises(ValueError):
            vs.hosmer_lemeshow_table(self.y_true, self.y_pred, nrow + 1)

    def test_hosmer_lemeshow_table_type(self):
        result = vs.hosmer_lemeshow_table(self.y_true, self.y_pred)
        self.assertIsInstance(result, np.recarray)

    def test_hosmer_lemeshow_table_shape(self):
        result = vs.hosmer_lemeshow_table(self.y_true, self.y_pred).shape
        expected = (10, )
        self.assertEqual(result, expected)

    def test_hosmer_lemeshow_table_names(self):
        result = vs.hosmer_lemeshow_table(self.y_true, self.y_pred).dtype.names
        expected = ('group_size', 'obs_freq', 'pred_freq', 'mean_prob')
        self.assertEqual(result, expected)

    def test_hosmer_lemeshow_test_C_hat_type(self):
        result, __, __ = vs.hosmer_lemeshow_test(self.y_true, self.y_pred)
        self.assertIsInstance(result, np.float64)

    def test_hosmer_lemeshow_test_C_hat_minimum(self):
        result, __, __ = vs.hosmer_lemeshow_test(self.y_true, self.y_pred)
        self.assertTrue(result > 0)

    def test_hosmer_lemeshow_test_df_type(self):
        __, result, __ = vs.hosmer_lemeshow_test(self.y_true, self.y_pred)
        self.assertIsInstance(result, int)

    def test_hosmer_lemeshow_test_df_minimum(self):
        __, result, __ = vs.hosmer_lemeshow_test(self.y_true, self.y_pred)
        self.assertTrue(result > 0)

    def test_hosmer_lemeshow_test_p_type(self):
        __, __, result = vs.hosmer_lemeshow_test(self.y_true, self.y_pred)
        self.assertIsInstance(result, np.float64)

    def test_osius_rojek_test_z_type(self):
        result, __, __ = vs.osius_rojek_test(self.X, self.y_true, self.y_pred)
        self.assertIsInstance(result, np.float64)

    def test_osius_rojek_test_z_minimum(self):
        result, __, __ = vs.osius_rojek_test(self.X, self.y_true, self.y_pred)
        self.assertTrue(result > 0)

    def test_osius_rojek_test_df_type(self):
        __, result, __ = vs.osius_rojek_test(self.X, self.y_true, self.y_pred)
        self.assertIsInstance(result, int)

    def test_osius_rojek_test_df_minimum(self):
        __, result, __ = vs.osius_rojek_test(self.X, self.y_true, self.y_pred)
        self.assertTrue(result > 0)

    def test_osius_rojek_test_p_type(self):
        __, __, result = vs.osius_rojek_test(self.X, self.y_true, self.y_pred)
        self.assertIsInstance(result, np.float64)


class TestLikelihoodMeasures(LargeFixture):

    def test_log_likelihood(self):
        result = vs.log_likelihood(self.y_true, self.y_pred)
        self.assertIsInstance(result, np.float64)

    def test_aic(self):
        result = vs.aic(self.X, self.y_true, self.y_pred)
        self.assertIsInstance(result, np.float64)


class TestSummaryMeasures(SmallFixture):

    def test_pearson_r2(self):
        result = vs.pearson_r2(self.y_true, self.y_pred)
        np.testing.assert_approx_equal(result, 0.1249447)

    def test_efrons_r2(self):
        result = vs.efrons_r2(self.y_true, self.y_pred)
        np.testing.assert_approx_equal(result, -0.0647800)

    def test_count_r2(self):
        result = vs.count_r2(self.y_true, self.y_pred)
        np.testing.assert_approx_equal(result, 0.5)

    def test_adjusted_count_r2(self):
        result = vs.adjusted_count_r2(self.y_true, self.y_pred)
        np.testing.assert_approx_equal(result, 0.0)

    def test_mckelvey_zavoina_r2(self):
        result = vs.mckelvey_zavoina_r2(self.y_true, self.y_pred)
        np.testing.assert_approx_equal(result, 0.2127050)

    def test_mckelvey_zavoina_r2_probit(self):
        result = vs.mckelvey_zavoina_r2(self.y_true, self.y_pred, 'probit')
        np.testing.assert_approx_equal(result, 0.4705717)

    def test_mckelvey_zavoina_r2_invalid_link(self):
        with self.assertRaises(ValueError):
            vs.mckelvey_zavoina_r2(self.y_true, self.y_pred, 'foobar')

    def test_brier_score(self):
        result = vs.brier_score(self.y_true, self.y_pred)
        np.testing.assert_approx_equal(result, 0.2661950)

    def test_expected_pcp(self):
        result = vs.expected_pcp(self.y_true, self.y_pred)
        np.testing.assert_approx_equal(result, 0.5856667)

    def test_summary_measures(self):
        result = vs.summary_measures(self.y_true, self.y_pred)
        self.assertIsInstance(result, str)


class TestCasewiseStatistics(LargeFixture):

    def test_pregibon_leverages(self):
        result = vs.pregibon_leverages(self.X, self.y_pred)
        self.assertIsInstance(result, np.ndarray)

    def test_pregibon_leverages_minimum(self):
        result = vs.pregibon_leverages(self.X, self.y_pred).min()
        self.assertTrue(result > 0)

    def test_pregibon_leverages_maximum(self):
        result = vs.pregibon_leverages(self.X, self.y_pred).max()
        self.assertTrue(result < 1)

    def test_pregibon_leverages_sum(self):
        # The sum of the leverages does not equal the number of features
        # when the feature matrix contains redundant data, as is the
        # default for ``sklearn.datasets.make_classification``.
        X, y = make_classification(n_samples=1200)

        clf = LogisticRegression()
        clf.fit(X, y)

        y_pred = clf.predict_proba(X)[:, 1]

        with self.assertRaises(ValueError):
            vs.pregibon_leverages(X, y_pred)

    def test_standardize_residuals(self):
        residuals = vs.pearson_residuals(self.y_true, self.y_pred)
        leverages = vs.pregibon_leverages(self.X, self.y_pred)
        result = vs.standardize_residuals(residuals, leverages)
        self.assertIsInstance(result, np.ndarray)

    def test_pregibon_dbetas(self):
        residuals = vs.pearson_residuals(self.y_true, self.y_pred)
        leverages = vs.pregibon_leverages(self.X, self.y_pred)
        result = vs.pregibon_dbetas(residuals, leverages)
        self.assertIsInstance(result, np.ndarray)

    def test_case_deltas(self):
        residuals = vs.pearson_residuals(self.y_true, self.y_pred)
        leverages = vs.pregibon_leverages(self.X, self.y_pred)
        result = vs.case_deltas(residuals, leverages)
        self.assertIsInstance(result, np.ndarray)


if __name__ == '__main__':
    unittest.main()
