import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.evaluate_routed_checkpoints import parse_args
from scripts.evaluate_task_checkpoint import load_model
from train_streaming import configure_ablation


class CheckpointAblationTest(unittest.TestCase):
    def test_no_hsg_checkpoint_rejects_full_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory)
            torch.save(
                {"ablation": "no_hsge", "additional_components": {}},
                checkpoint / "training_state.pt",
            )
            with patch(
                "scripts.evaluate_task_checkpoint.StreamingSceneInstructionQwenModel"
            ) as model_constructor:
                with self.assertRaisesRegex(ValueError, "ablation mismatch"):
                    load_model("unused", checkpoint, torch.device("cpu"), "full")
                model_constructor.assert_not_called()

    def test_routed_parser_accepts_no_hsg_ablation(self) -> None:
        arguments = [
            "evaluate_routed_checkpoints.py",
            "--global-checkpoint",
            "global",
            "--local-checkpoint",
            "local",
            "--model-path",
            "model",
            "--dataset",
            "dataset.jsonl",
            "--output-dir",
            "output",
            "--ablation",
            "no_hsge",
        ]
        with patch.object(sys, "argv", arguments):
            self.assertEqual(parse_args().ablation, "no_hsge")

    def test_component_ablation_flags_are_independent(self) -> None:
        self.assertEqual(
            configure_ablation("no_global_topology", rank=1),
            (True, True, True, False),
        )
        self.assertEqual(
            configure_ablation("no_object_tokens", rank=1),
            (True, False, True, True),
        )
        self.assertEqual(
            configure_ablation("no_graph_updates_history", rank=1),
            (True, True, False, True),
        )


if __name__ == "__main__":
    unittest.main()
