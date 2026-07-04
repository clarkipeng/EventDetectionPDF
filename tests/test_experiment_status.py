import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from paper.experiment_status import load_observed_runs, merge_status, write_text_summary
from paper.backfill_epoch_results import parse_epoch_segments


class ExperimentStatusTests(unittest.TestCase):
    def test_parse_epoch_segments_splits_sequential_objectives(self):
        text = "\n".join(
            [
                "fold 0, epoch 1/2: train loss: 0.100, valid loss: 0.200, valid mAP: 0.300",
                "fold 0, epoch 2/2: train loss: 0.090, valid loss: 0.190, valid mAP: 0.310",
                "fold 0, epoch 1/2: train loss: 0.050, valid loss: 0.150, valid mAP: 0.400",
                "fold 0, epoch 2/2: train loss: 0.040, valid loss: 0.140, valid mAP: 0.420",
            ]
        )

        segments = parse_epoch_segments(text)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0][-1]["epoch"], 2)
        self.assertAlmostEqual(segments[1][-1]["valid_mAP"], 0.420)

    def test_epoch_results_populate_incomplete_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = (
                root
                / "seizure"
                / "gru_3l_128h"
                / "density_custom"
                / "seed_0_seizure_highscore_gru3l128h_ds256_bs8_e20"
                / "results"
            )
            results_dir.mkdir(parents=True)
            (results_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "dataset": "seizure",
                        "model": "gru_3l_128h",
                        "objective": "density_custom",
                        "seed": 0,
                        "run_tag": "seizure_highscore_gru3l128h_ds256_bs8_e20",
                        "epochs": 20,
                        "folds": 4,
                        "batch_size": 8,
                        "downsample": 256,
                        "eval_every": 1,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "fold": 0,
                        "best_epoch": 17,
                        "best_valid_loss": 0.002,
                        "best_valid_mAP": 0.225,
                    },
                    {
                        "fold": 1,
                        "best_epoch": 15,
                        "best_valid_loss": 0.003,
                        "best_valid_mAP": 0.342,
                    },
                ]
            ).to_csv(results_dir / "fold_results.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "fold": 2,
                        "epoch": 14,
                        "epochs": 20,
                        "train_loss": 0.003,
                        "valid_mAP": 0.204,
                        "evaluated": True,
                        "epoch_runtime_sec": 120,
                    },
                    {
                        "fold": 2,
                        "epoch": 15,
                        "epochs": 20,
                        "train_loss": 0.003,
                        "valid_mAP": 0.239,
                        "evaluated": True,
                        "epoch_runtime_sec": 120,
                    },
                    {
                        "fold": 2,
                        "epoch": 16,
                        "epochs": 20,
                        "train_loss": 0.002,
                        "evaluated": False,
                        "epoch_runtime_sec": 80,
                    },
                ]
            ).to_csv(results_dir / "epoch_results.csv", index=False)

            observed = load_observed_runs(root)
            self.assertEqual(int(observed.loc[0, "recorded_epochs"]), 3)
            self.assertEqual(int(observed.loc[0, "latest_fold"]), 2)
            self.assertEqual(int(observed.loc[0, "latest_epoch"]), 16)
            self.assertTrue(pd.isna(observed.loc[0, "latest_valid_mAP"]))
            self.assertEqual(int(observed.loc[0, "latest_eval_epoch"]), 15)
            self.assertAlmostEqual(float(observed.loc[0, "latest_eval_valid_mAP"]), 0.239)
            self.assertEqual(int(observed.loc[0, "best_eval_epoch"]), 15)
            self.assertAlmostEqual(float(observed.loc[0, "best_eval_valid_mAP"]), 0.239)
            self.assertAlmostEqual(
                float(observed.loc[0, "estimated_current_fold_remaining_min"]),
                (120 + 120 + 120 + 120) / 60,
            )

            plan = pd.DataFrame(
                [
                    {
                        "phase": "P12b",
                        "name": "seizure_highscore_stride_ablation",
                        "dataset": "seizure",
                        "model": "gru_3l_128h",
                        "objective": "density_custom",
                        "seed": 0,
                        "run_tag": "seizure_highscore_gru3l128h_ds256_bs8_e20",
                        "target_epochs": 20,
                        "target_folds": 4,
                        "target_batch_size": 8,
                        "target_downsample": 256,
                        "target_eval_every": 1,
                        "score_required": True,
                        "purpose": "test",
                    }
                ]
            )
            status = merge_status(plan, observed)
            self.assertEqual(status.loc[0, "status"], "partial")

            summary_path = root / "status.txt"
            write_text_summary(status, summary_path)
            summary = summary_path.read_text(encoding="utf-8")
            self.assertIn("Active Rows", summary)
            self.assertIn("Incomplete Rows", summary)
            self.assertIn("| phase | name | status | model | objective | seed | run_tag |", summary)
            self.assertIn("0.239", summary)
            self.assertIn("best_eval_valid_mAP", summary)
            self.assertIn("estimated_current_fold_remaining_min", summary)


if __name__ == "__main__":
    unittest.main()
