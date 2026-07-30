import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.build_mixed_recovery_dataset import build_dataset
from utils.dataloader_streaming import StreamingSceneGraphInstructionDataset


class MixedRecoveryDatasetTest(unittest.TestCase):
    def test_builder_adds_global_and_local_recovery_samples(self) -> None:
        task = {
            "instruction": "Go to room b.",
            "task_info": {"type": "guidance", "difficulty": "easy"},
            "execution_summary": {
                "global_plan": ["goto(room_b): pass()"],
                "subtasks": ["goto(room_b)"],
            },
            "scene_name": "test_scene",
            "streaming_samples": [
                {
                    "mode": "global",
                    "context": "instruction: Go to room b.",
                    "target": ["goto(room_b): pass()"],
                    "completed": [],
                    "pending": [],
                    "scene_graph": {
                        "name": "test_scene",
                        "agent": {"position": "room_a"},
                        "macro_zones": {},
                        "rooms": {
                            "room_a": {"floor": "1f", "neighbor": ["room_b"]},
                            "room_b": {"floor": "1f", "neighbor": ["room_a"]},
                        },
                    },
                },
                {
                    "mode": "local",
                    "context": "instruction: Go to room b.",
                    "target": "goto(room_b)",
                    "completed": [],
                    "pending": ["goto(room_b): pass()"],
                    "scene_graph": {
                        "name": "test_scene",
                        "current_room": "room_a",
                        "agent": {"position": "room_a", "state": "hand-free"},
                        "room": {
                            "neighbor": ["room_b"],
                            "small_objects": {},
                            "large_objects": {},
                        },
                    },
                },
            ],
            "sample_count": 2,
        }

        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.jsonl"
            output_path = Path(temporary_directory) / "output.jsonl"
            input_path.write_text(json.dumps(task) + "\n", encoding="utf-8")

            totals = build_dataset(
                input_path,
                output_path,
                global_replay_copies=1,
                global_repair_variants=1,
                local_recovery_samples=1,
            )
            augmented_task = json.loads(output_path.read_text(encoding="utf-8"))
            dataset = StreamingSceneGraphInstructionDataset(
                str(output_path), chunk_size=10
            )
            dataset._load_chunk(0)

        self.assertEqual(totals["total_samples"], 5)
        self.assertEqual(augmented_task["sample_count"], 5)
        augmentation_types = {
            sample.get("metadata", {}).get("augmentation_type")
            for sample in augmented_task["streaming_samples"]
        }
        self.assertIn("global_semantic_replay", augmentation_types)
        self.assertIn("global_plan_repair", augmentation_types)
        self.assertIn("local_temporary_recovery", augmentation_types)
        recovery_samples = [
            sample
            for sample in augmented_task["streaming_samples"]
            if sample.get("metadata", {}).get("augmentation_type", "").startswith(
                "local_"
            )
        ]
        self.assertEqual(len(recovery_samples), 1)
        self.assertIn(
            "RECOVERY CONTEXT", recovery_samples[0]["instruction_override"]
        )
        recovery_instructions = [
            sample["instruction"]
            for sample in dataset.current_chunk_data
            if "RECOVERY CONTEXT" in sample["instruction"]
        ]
        self.assertEqual(len(recovery_instructions), 1)


if __name__ == "__main__":
    unittest.main()
