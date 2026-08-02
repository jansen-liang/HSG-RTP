import json
import unittest

from evaluation.action_parser import ParseError, parse_local_action, parse_prediction
from evaluation.goal_evaluator import build_goal_spec, evaluate_goal
from evaluation.plan_evaluator import evaluate_global_plan
from evaluation.policies import OraclePolicy
from evaluation.runner import evaluate_policy_dataset
from evaluation.rollout_evaluator import evaluate_action_sequence, rollout_policy


def make_scene() -> dict:
    return {
        "name": "test_scene",
        "macro_zones": {},
        "agent": {"position": "room_a", "state": "hand-free", "inventory": {}},
        "rooms": {
            "room_a": {
                "floor": "1f",
                "neighbor": ["room_b"],
                "small_objects": {
                    "parcel": {
                        "type": "object",
                        "affordance": ["pick", "place"],
                        "state": {"availability": "available"},
                        "relation": {"on": "shelf"},
                    }
                },
                "large_objects": {"shelf": {"type": "furniture"}},
            },
            "room_b": {
                "floor": "1f",
                "neighbor": ["room_a"],
                "small_objects": {},
                "large_objects": {"table": {"type": "furniture"}},
            },
        },
    }


def make_record() -> dict:
    final_state = make_scene()
    parcel = final_state["rooms"]["room_a"]["small_objects"].pop("parcel")
    parcel["relation"] = {"on": "table"}
    final_state["rooms"]["room_b"]["small_objects"]["parcel"] = parcel
    final_state["agent"]["position"] = "room_b"
    return {
        "instruction": "Deliver the parcel to room b.",
        "scene_name": "test_scene",
        "task_info": {
            "type": "delivery",
            "difficulty": "easy",
            "parameters": {
                "objects": ["parcel"],
                "source_room": "room_a",
                "target_rooms": ["room_b"],
                "objects_goal_state": {},
            },
        },
        "execution_summary": {
            "global_plan": [
                "goto(room_a): pick(parcel)",
                "goto(room_b): place(parcel)",
            ],
            "subtasks": ["pick(parcel)", "goto(room_b)", "place(parcel, table)"],
            "final_state": final_state,
        },
    }


class ActionParserTest(unittest.TestCase):
    def test_local_action_is_normalized(self) -> None:
        self.assertEqual(parse_local_action("place(parcel,table)").canonical, "place(parcel, table)")

    def test_prediction_requires_expected_mode(self) -> None:
        with self.assertRaises(ParseError):
            parse_prediction(json.dumps({"mode": "global", "task": []}), "local")


class TaskEvaluationTest(unittest.TestCase):
    def test_reference_plan_and_execution_succeed(self) -> None:
        record = make_record()
        scene = make_scene()
        plan_result = evaluate_global_plan(
            record, record["execution_summary"]["global_plan"], scene
        )
        execution_result = evaluate_action_sequence(
            record, scene, record["execution_summary"]["subtasks"]
        )
        self.assertTrue(plan_result.success, plan_result.errors)
        self.assertTrue(execution_result.success, execution_result.failure_message)

    def test_goal_requires_reference_relation(self) -> None:
        record = make_record()
        goal = build_goal_spec(record)
        success, failures = evaluate_goal(make_scene(), goal)
        self.assertFalse(success)
        self.assertTrue(any("expected in room_b" in failure for failure in failures))

    def test_oracle_policy_rollout_succeeds(self) -> None:
        record = make_record()
        policy = OraclePolicy(
            record["execution_summary"]["global_plan"],
            record["execution_summary"]["subtasks"],
        )
        result = rollout_policy(record, make_scene(), policy)
        self.assertTrue(result.success, result.failure_message)

    def test_dataset_runner_reports_task_metrics_and_similarity(self) -> None:
        record = make_record()
        results, summary = evaluate_policy_dataset(
            [record],
            {"test_scene": make_scene()},
            lambda current: OraclePolicy(
                current["execution_summary"]["global_plan"],
                current["execution_summary"]["subtasks"],
            ),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(summary["plan_sr"], 1.0)
        self.assertEqual(summary["exec_sr"], 1.0)
        self.assertEqual(summary["global_jaccard"], 1.0)
        self.assertEqual(summary["global_lcs_ratio"], 1.0)
        self.assertEqual(summary["local_jaccard"], 1.0)
        self.assertEqual(summary["local_lcs_ratio"], 1.0)
        self.assertEqual(summary["avg_total_tokens"], 0.0)

    def test_unsupported_state_goal_is_rejected(self) -> None:
        record = make_record()
        record["task_info"]["parameters"]["objects_goal_state"] = {
            "parcel": {"temperature": "hot"}
        }
        result = evaluate_global_plan(
            record, record["execution_summary"]["global_plan"], make_scene()
        )
        self.assertFalse(result.success)
        self.assertTrue(any("object-state transitions" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
