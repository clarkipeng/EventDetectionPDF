import unittest

import numpy as np
import torch

from src.utils import (
    DENSITY_OBJECTIVES,
    apply_objective_overrides,
    density_logits_to_rates,
    get_loss,
    get_targets,
)


class DummyDataClass:
    event_type = "interval"
    gaussian_sigma = 2
    tolerances = [1, 3]
    target_tolerances = [1, 3]
    day_length = 100


class ObjectiveTests(unittest.TestCase):
    def test_density_targets_are_nonnegative_unit_mass_per_boundary(self):
        dataclass = DummyDataClass()
        locations = (np.array([10, 30]), np.array([20, 40]))
        for objective in DENSITY_OBJECTIVES:
            target = get_targets(dataclass, 60, locations, objective, normalize=True)
            self.assertEqual(target.shape, (60, 2))
            self.assertGreaterEqual(target.min(), 0)
            np.testing.assert_allclose(target.sum(axis=0), np.array([2.0, 2.0]))

    def test_density_loss_is_finite_for_empty_and_multi_event_windows(self):
        dataclass = DummyDataClass()
        for locations in [
            (np.array([]), np.array([])),
            (np.array([10, 30]), np.array([20, 40])),
        ]:
            target = get_targets(dataclass, 60, locations, "density_gau")
            prediction = torch.zeros((1, 60, 2))
            loss = get_loss(
                "density_gau",
                dataclass=dataclass,
                downsample=1,
            )(prediction, torch.tensor(target[None], dtype=torch.float32))
            self.assertTrue(torch.isfinite(loss).all())

    def test_density_prior_modes_produce_positive_rates(self):
        dataclass = DummyDataClass()
        prediction = torch.zeros((1, 5, 2))
        for prior in ["sparse", "none"]:
            rates = density_logits_to_rates(
                prediction,
                dataclass=dataclass,
                downsample=1,
                density_prior=prior,
            )
            self.assertTrue(torch.all(rates > 0))

    def test_segmentation_losses_are_finite(self):
        prediction = torch.zeros((2, 10, 1))
        target = torch.zeros((2, 10, 1))
        target[:, 2:5, 0] = 1
        for objective in ["seg", "seg_weighted", "seg_focal"]:
            loss = get_loss(objective)(prediction, target)
            self.assertTrue(torch.isfinite(loss).all())

    def test_target_tolerance_scale_does_not_change_eval_tolerances(self):
        dataclass = DummyDataClass()
        apply_objective_overrides(dataclass, tolerance_scale=2.0)
        self.assertEqual(dataclass.tolerances, [1, 3])
        self.assertEqual(dataclass.target_tolerances, [2, 6])


if __name__ == "__main__":
    unittest.main()
