import unittest

from scripts.build_static_history_ablation_dataset import transform_task


class StaticHistoryAblationDatasetTest(unittest.TestCase):
    def test_reuses_first_global_and_per_room_local_views(self) -> None:
        task = {
            "streaming_samples": [
                {
                    "mode": "global",
                    "scene_graph": {"agent": {"state": "initial"}, "rooms": {}},
                },
                {
                    "mode": "local",
                    "scene_graph": {
                        "current_room": "room_a",
                        "agent": {"state": "hand-free"},
                        "room": {"small_objects": {"cup": {}}},
                    },
                },
                {
                    "mode": "global",
                    "scene_graph": {"agent": {"state": "updated"}, "rooms": {}},
                },
                {
                    "mode": "local",
                    "scene_graph": {
                        "current_room": "room_a",
                        "agent": {"state": "holding"},
                        "room": {"small_objects": {}},
                    },
                },
            ]
        }

        transformed, stats = transform_task(task)
        samples = transformed["streaming_samples"]

        self.assertEqual(samples[2]["scene_graph"], samples[0]["scene_graph"])
        self.assertEqual(samples[3]["scene_graph"], samples[1]["scene_graph"])
        self.assertEqual(stats["global_samples"], 2)
        self.assertEqual(stats["local_samples"], 2)
        self.assertEqual(stats["local_rooms"], 1)
        self.assertEqual(
            transformed["ablation_transform"]["name"],
            "static_graph_no_history",
        )

    def test_accepts_local_only_tasks(self) -> None:
        task = {
            "streaming_samples": [
                {
                    "mode": "local",
                    "scene_graph": {
                        "current_room": "room_a",
                        "agent": {"state": "idle"},
                        "room": {},
                    },
                }
            ]
        }

        transformed, stats = transform_task(task)

        self.assertEqual(len(transformed["streaming_samples"]), 1)
        self.assertEqual(stats["global_samples"], 0)
        self.assertEqual(stats["local_samples"], 1)


if __name__ == "__main__":
    unittest.main()
