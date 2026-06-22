import unittest

import torch

from models.load_model import get_model


class DummyDataClass:
    event_type = "interval"
    num_feats = 3
    cat_feats = 0
    cat_uniq = 0


class ModelTests(unittest.TestCase):
    def test_forward_rnn_and_causal_transformer_output_shapes(self):
        dataclass = DummyDataClass()
        x = torch.randn(2, 8, 3)
        for model_name in [
            "fgru",
            "gru_forward",
            "causal_transformer_2l_32h_4a",
            "transformer_2l_32h_4a",
        ]:
            model = get_model(
                dataclass,
                model_name,
                objective="density_custom",
                sequence_length=8,
                downsample=1,
                agg_feats="none",
            )
            y = model(x)
            self.assertEqual(y.shape, (2, 8, 2))

    def test_causal_transformer_does_not_depend_on_future_inputs(self):
        dataclass = DummyDataClass()
        model = get_model(
            dataclass,
            "causal_transformer_1l_32h_4a",
            objective="density_custom",
            sequence_length=8,
            downsample=1,
            agg_feats="none",
        )
        model.eval()
        x = torch.randn(1, 8, 3)
        x_changed = x.clone()
        x_changed[:, 4:, :] = torch.randn_like(x_changed[:, 4:, :])
        with torch.no_grad():
            y = model(x)
            y_changed = model(x_changed)
        torch.testing.assert_close(y[:, :4, :], y_changed[:, :4, :])

    def test_offline_transformer_can_depend_on_future_inputs(self):
        dataclass = DummyDataClass()
        model = get_model(
            dataclass,
            "transformer_1l_32h_4a",
            objective="density_custom",
            sequence_length=8,
            downsample=1,
            agg_feats="none",
        )
        model.eval()
        x = torch.randn(1, 8, 3)
        x_changed = x.clone()
        x_changed[:, 4:, :] = torch.randn_like(x_changed[:, 4:, :])
        with torch.no_grad():
            y = model(x)
            y_changed = model(x_changed)
        self.assertGreater(torch.max(torch.abs(y[:, :4, :] - y_changed[:, :4, :])).item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
