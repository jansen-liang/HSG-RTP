import json
import tempfile
import unittest
from pathlib import Path

from scripts.convert_hsg_rtp_to_grid import convert_dataset


class GridConversionTest(unittest.TestCase):
    def test_converts_all_hsg_action_shapes(self) -> None:
        local_view = {
            "name": "test_scene",
            "current_room": "room_a",
            "agent": {"position": "room_a", "state": "holding-parcel"},
            "room": {
                "floor": "floor_1",
                "neighbor": ["room_b"],
                "large_objects": {"table": {"type": "furniture", "is_container": True}},
                "small_objects": {
                    "parcel": {
                        "type": "item",
                        "affordance": ["pick", "place"],
                        "state": {"availability": "available"},
                        "relation": {"on": "table"},
                    },
                    "button": {
                        "type": "control",
                        "affordance": ["press"],
                        "state": {"pressed": False},
                        "relation": {"on": "table"},
                    },
                },
            },
        }
        actions = [
            "scan(room_a)",
            "goto(room_b)",
            "pick(parcel)",
            "place(parcel, table)",
            "press(button)",
            "wait(elevator_up_clear)",
        ]
        record = {
            "instruction": "Handle the parcel.",
            "task_info": {"type": "delivery"},
            "streaming_samples": [
                {"mode": "local", "target": action, "scene_graph": local_view}
                for action in actions
            ],
            "execution_summary": {
                "final_state": {
                    "name": "test_scene",
                    "rooms": {"room_a": local_view["room"]},
                    "agent": {
                        "position": "room_a",
                        "state": "hand-free",
                        "inventory": {},
                    },
                }
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")
            output = root / "grid"
            manifest = convert_dataset(source, output, max_nodes=87, overwrite=False)

            self.assertEqual(manifest["tasks"], 1)
            self.assertEqual(manifest["samples"], 7)
            self.assertEqual(manifest["actions"]["finish"], 1)
            commands = json.loads((output / "scene.0.instr.json").read_text(encoding="utf-8"))
            low = commands["commands"][0]["low"]
            self.assertEqual([command.split()[0] for command in low], [
                "scan", "goto", "pick", "place", "press", "wait", "finish"
            ])
            wait_graph = json.loads(
                (output / "scene.0.graphs/scene.0.instr.0.sg.5.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                any("elevator up clear" in node["attributes"]["label"] for node in wait_graph["nodes"])
            )
            self.assertTrue(
                all("entity_id" in node["attributes"] for node in wait_graph["nodes"])
            )


if __name__ == "__main__":
    unittest.main()
