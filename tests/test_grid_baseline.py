import unittest

from evaluation.grid_baseline import rollout_grid_policy


class SequencePredictor:
    def __init__(self, actions):
        self.actions = iter(actions)

    def predict(self, instruction, local_view):
        return next(self.actions), 0.01


def make_record():
    return {
        "instruction": "Move the parcel to room b.",
        "task_info": {
            "type": "delivery",
            "parameters": {"objects": ["parcel"]},
        },
        "execution_summary": {
            "subtasks": ["pick(parcel)", "goto(room_b)", "place(parcel, table_b)"],
            "final_state": {
                "rooms": {
                    "room_a": {"small_objects": {}, "large_objects": {}},
                    "room_b": {
                        "small_objects": {
                            "parcel": {"relation": {"on": "table_b"}}
                        },
                        "large_objects": {"table_b": {}},
                    },
                },
                "agent": {"position": "room_b", "inventory": {}},
            },
        },
    }


def make_scene():
    return {
        "name": "test",
        "rooms": {
            "room_a": {
                "floor": "floor_1",
                "neighbor": ["room_b"],
                "large_objects": {},
                "small_objects": {
                    "parcel": {
                        "affordance": ["pick", "place"],
                        "relation": {"on": "floor"},
                    }
                },
            },
            "room_b": {
                "floor": "floor_1",
                "neighbor": ["room_a"],
                "large_objects": {"table_b": {"is_container": True}},
                "small_objects": {},
            },
        },
        "agent": {
            "position": "room_a",
            "state": "hand-free",
            "inventory": {},
        },
    }


class GridBaselineTest(unittest.TestCase):
    def test_strict_rollout_requires_finish_after_goal(self):
        predictor = SequencePredictor(
            ["pick(parcel)", "goto(room_b)", "place(parcel, table_b)", "finish"]
        )
        result = rollout_grid_policy(make_record(), make_scene(), predictor)
        self.assertTrue(result.success)
        self.assertTrue(result.finished)
        self.assertEqual(result.model_calls, 4)

    def test_invalid_action_fails_immediately(self):
        predictor = SequencePredictor(["pick(table_b)"])
        result = rollout_grid_policy(make_record(), make_scene(), predictor)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_type, "execution_error")


if __name__ == "__main__":
    unittest.main()
