"""
Unit tests confirming zero data leakage across preprocessing transformers using unittest.
"""
import unittest
import numpy as np
import pandas as pd
from src.preprocessing.transformers import OmicsStandardizer, InFoldImputer, ClinicalEncoder
from src.pathways.scorer import PathwayZScorer


class TestPreprocessingLeakage(unittest.TestCase):
    def test_omics_standardizer_no_test_leakage(self):
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.normal(loc=10.0, scale=2.0, size=(50, 5)), columns=[f"g{i}" for i in range(5)])
        X_test_1 = pd.DataFrame(np.random.normal(loc=10.0, scale=2.0, size=(20, 5)), columns=[f"g{i}" for i in range(5)])
        X_test_extreme = pd.DataFrame(np.random.normal(loc=1000.0, scale=50.0, size=(20, 5)), columns=[f"g{i}" for i in range(5)])

        scaler = OmicsStandardizer()
        scaler.fit(X_train)

        train_mean = scaler.mean_.copy()
        train_scale = scaler.scale_.copy()

        # Transform normal test set
        res1 = scaler.transform(X_test_1)
        self.assertTrue(np.allclose(scaler.mean_, train_mean))
        self.assertTrue(np.allclose(scaler.scale_, train_scale))

        # Transform extreme test set
        res_extreme = scaler.transform(X_test_extreme)
        # Stored training parameters must remain strictly identical
        self.assertTrue(np.allclose(scaler.mean_, train_mean))
        self.assertTrue(np.allclose(scaler.scale_, train_scale))

        # Output shapes match
        self.assertEqual(res1.shape, (20, 5))
        self.assertEqual(res_extreme.shape, (20, 5))

    def test_infold_imputer_no_test_leakage(self):
        np.random.seed(42)
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [np.nan, 8.0]])
        imputer = InFoldImputer(strategy="median")
        imputer.fit(X_train)

        # Median of col 0 in train is 3.0, col 1 is 5.0
        self.assertEqual(imputer.statistics_[0], 3.0)
        self.assertEqual(imputer.statistics_[1], 5.0)

        # Transforming a test set with missing values uses train medians
        X_test = np.array([[np.nan, np.nan], [100.0, 200.0]])
        X_test_imp = imputer.transform(X_test)
        self.assertEqual(X_test_imp[0, 0], 3.0)
        self.assertEqual(X_test_imp[0, 1], 5.0)

    def test_pathway_zscorer_no_leakage(self):
        np.random.seed(42)
        genes = [f"ENSG_{i}" for i in range(10)]
        X_train = pd.DataFrame(np.random.normal(5.0, 1.0, (40, 10)), columns=genes)
        X_test = pd.DataFrame(np.random.normal(20.0, 5.0, (15, 10)), columns=genes)

        pathway_map = {
            "PWAY_1": genes[:6],
            "PWAY_2": genes[4:],
        }

        scorer = PathwayZScorer(pathway_to_features=pathway_map)
        scorer.fit(X_train)

        fitted_means = scorer.feature_means_.copy()

        # Transform test set
        test_scores = scorer.transform(X_test)

        # Verify fitted means unchanged
        self.assertEqual(scorer.feature_means_, fitted_means)
        self.assertEqual(test_scores.shape, (15, 2))
        self.assertEqual(list(test_scores.columns), ["PWAY_1", "PWAY_2"])


if __name__ == "__main__":
    unittest.main()
