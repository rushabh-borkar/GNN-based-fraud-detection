import unittest

from elliptic_pipeline import illicit_metrics, temporal_split_indices


class PipelineUtilsTest(unittest.TestCase):
    def test_temporal_split_indices(self):
        steps = [1, 34, 35, 43, 44, 49]
        splits = temporal_split_indices(steps, train_end_step=34, val_end_step=43)
        self.assertEqual(splits["train"], [0, 1])
        self.assertEqual(splits["val"], [2, 3])
        self.assertEqual(splits["test"], [4, 5])

    def test_illicit_metrics_binary_outputs(self):
        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.3, 0.6, 0.2]
        metrics = illicit_metrics(y_true, y_score, threshold=0.5)

        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertGreaterEqual(metrics["pr_auc"], 0.0)
        self.assertLessEqual(metrics["pr_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
