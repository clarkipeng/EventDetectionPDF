import unittest

import numpy as np

from eval import enforce_alternating_interval_candidates


class PostprocessingTests(unittest.TestCase):
    def test_enforce_alternating_interval_candidates_keeps_legal_pairs(self):
        candidates = [
            np.array([10, 20, 50, 80]),
            np.array([5, 30, 60, 70]),
        ]
        scores = [
            np.array([0.2, 0.9, 0.4, 0.8]),
            np.array([0.7, 0.6, 0.3, 0.9]),
        ]
        filtered_candidates, filtered_scores = enforce_alternating_interval_candidates(
            candidates,
            scores,
        )

        np.testing.assert_array_equal(filtered_candidates[0], np.array([20, 50]))
        np.testing.assert_array_equal(filtered_candidates[1], np.array([30, 70]))
        np.testing.assert_allclose(filtered_scores[0], np.array([0.9, 0.4]))
        np.testing.assert_allclose(filtered_scores[1], np.array([0.6, 0.9]))


if __name__ == "__main__":
    unittest.main()
