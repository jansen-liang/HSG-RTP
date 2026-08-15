import json
import unittest

from evaluation.action_parser import ParseError, parse_local_action, parse_prediction
from evaluation.goal_evaluator import build_goal_spec, evaluate_goal
from evaluation.plan_evaluator import evaluate_global_plan
from evaluation.policies import OraclePolicy
from evaluation.runner import evaluate_policy_dataset
from evaluation.rollout_evaluator import (
    evaluate_action_sequence,
    ground_global_plan,
    ground_local_action,
    rollout_policy,
)


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

    def test_goal_requires_target_room_but_allows_alternate_support(self) -> None:
        record = make_record()
        goal = build_goal_spec(record)
        success, failures = evaluate_goal(make_scene(), goal)
        self.assertFalse(success)
        self.assertTrue(any("expected in room_b" in failure for failure in failures))

        alternate = make_record()["execution_summary"]["final_state"]
        alternate["rooms"]["room_b"]["small_objects"]["parcel"]["relation"] = {
            "on": "floor"
        }
        success, failures = evaluate_goal(alternate, goal)
        self.assertTrue(success, failures)

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

    def test_local_navigation_is_grounded_to_next_hop(self) -> None:
        scene = make_scene()
        scene["rooms"]["room_b"]["neighbor"].append("room_c")
        scene["rooms"]["room_c"] = {
            "floor": "1f",
            "neighbor": ["room_b"],
            "small_objects": {},
            "large_objects": {},
        }
        action, change = ground_local_action("goto(room_c)", scene)
        self.assertEqual(action, "goto(room_b)")
        self.assertEqual(change["operation"], "route_next_hop")

    def test_neighbor_scan_is_grounded_to_goto(self) -> None:
        action, change = ground_local_action("scan(room_b)", make_scene())
        self.assertEqual(action, "goto(room_b)")
        self.assertEqual(change["operation"], "room_scan_to_goto")

    def test_local_entity_alias_is_normalized_to_visible_object(self) -> None:
        scene = make_scene()
        parcel = scene["rooms"]["room_a"]["small_objects"].pop("parcel")
        scene["rooms"]["room_a"]["small_objects"]["chinese_takeout"] = parcel
        action, change = ground_local_action("scan(china_takeout_box)", scene)
        self.assertEqual(action, "scan(chinese_takeout)")
        self.assertEqual(change["operation"], "normalize_entity_alias")

    def test_global_grounding_adds_missing_delivery_object(self) -> None:
        scene = make_scene()
        second = dict(scene["rooms"]["room_a"]["small_objects"]["parcel"])
        scene["rooms"]["room_a"]["small_objects"]["letter"] = second
        record = make_record()
        record["task_info"]["parameters"]["objects"] = ["parcel", "letter"]
        record["task_info"]["parameters"]["target_rooms"] = ["room_b"]
        repaired, changes = ground_global_plan(
            [
                "goto(room_a): pick(parcel)",
                "goto(room_b): place(parcel)",
            ],
            scene,
            record,
        )
        self.assertIn("goto(room_a): pick(letter)", repaired)
        self.assertIn("goto(room_b): place(letter)", repaired)
        self.assertTrue(
            any(change["operation"] == "repair_task_route_and_coverage" for change in changes)
        )

    def test_global_grounding_inserts_shortest_path_rooms(self) -> None:
        scene = make_scene()
        scene["rooms"]["room_a"]["neighbor"] = ["room_mid"]
        scene["rooms"]["room_b"]["neighbor"] = ["room_mid"]
        scene["rooms"]["room_mid"] = {
            "floor": "1f",
            "neighbor": ["room_a", "room_b"],
            "small_objects": {},
            "large_objects": {},
        }
        repaired, _ = ground_global_plan(
            [
                "goto(room_a): pick(parcel)",
                "goto(room_b): place(parcel)",
            ],
            scene,
            make_record(),
        )
        self.assertIn("goto(room_mid): pass()", repaired)

    def test_minimal_global_grounding_does_not_complete_task_plan(self) -> None:
        scene = make_scene()
        scene["rooms"]["room_a"]["small_objects"]["letter"] = dict(
            scene["rooms"]["room_a"]["small_objects"]["parcel"]
        )
        record = make_record()
        record["task_info"]["parameters"]["objects"] = ["parcel", "letter"]
        plan = [
            "goto(room_a): pick(parcel)",
            "goto(room_b): place(parcel)",
        ]
        grounded, changes = ground_global_plan(
            plan,
            scene,
            record,
            normalize_rooms=False,
            repair_task_plan=False,
        )
        self.assertEqual(grounded, plan)
        self.assertFalse(
            any(change["operation"] == "repair_task_route_and_coverage" for change in changes)
        )

    def test_lightweight_grounding_routes_without_adding_missing_objects(self) -> None:
        scene = make_scene()
        scene["rooms"]["room_a"]["neighbor"] = ["room_mid"]
        scene["rooms"]["room_b"]["neighbor"] = ["room_mid"]
        scene["rooms"]["room_mid"] = {
            "floor": "1f",
            "neighbor": ["room_a", "room_b"],
            "small_objects": {},
            "large_objects": {},
        }
        scene["rooms"]["room_a"]["small_objects"]["letter"] = dict(
            scene["rooms"]["room_a"]["small_objects"]["parcel"]
        )
        record = make_record()
        record["task_info"]["parameters"]["objects"] = ["parcel", "letter"]
        grounded, _ = ground_global_plan(
            [
                "goto(room_a): pick(parcel)",
                "goto(room_b): place(parcel)",
            ],
            scene,
            record,
            task_semantic_repair_budget=0,
        )
        self.assertIn("goto(room_mid): pass()", grounded)
        self.assertFalse(any("letter" in step for step in grounded))

    def test_lightweight_grounding_repairs_at_most_one_missing_object(self) -> None:
        scene = make_scene()
        scene["rooms"]["room_a"]["small_objects"]["letter"] = dict(
            scene["rooms"]["room_a"]["small_objects"]["parcel"]
        )
        record = make_record()
        record["task_info"]["parameters"]["objects"] = ["parcel", "letter"]
        grounded, changes = ground_global_plan(
            [
                "goto(room_a): pick(parcel)",
                "goto(room_b): place(parcel)",
            ],
            scene,
            record,
            task_semantic_repair_budget=2,
        )
        self.assertTrue(any("letter" in step for step in grounded))
        self.assertTrue(
            any(change["operation"] == "bounded_task_semantic_repair" for change in changes)
        )

    def test_pending_transition_advances_after_floor_button_press(self) -> None:
        scene = {
            "agent": {
                "position": "elevator_cabin",
                "pressed_buttons": ["elevator_button_2"],
            },
            "rooms": {
                "elevator_1f": {"neighbor": ["elevator_cabin"]},
                "elevator_2f": {"neighbor": ["elevator_cabin"]},
                "elevator_cabin": {
                    "neighbor": ["elevator_1f", "elevator_2f"],
                    "small_objects": {},
                    "large_objects": {},
                },
            },
        }
        action, change = ground_local_action(
            "scan(elevator_cabin)",
            scene,
            ["goto(elevator_1f): trans from(1f) to(2f)"],
        )
        self.assertEqual(action, "goto(elevator_2f)")
        self.assertEqual(change["operation"], "advance_pending_transition")

    def test_pending_transition_selects_unpressed_floor(self) -> None:
        scene = {
            "agent": {"position": "elevator_cabin", "pressed_buttons": []},
            "rooms": {
                "elevator_1f": {"neighbor": ["elevator_cabin"]},
                "elevator_2f": {"neighbor": ["elevator_cabin"]},
                "elevator_cabin": {
                    "neighbor": ["elevator_1f", "elevator_2f"],
                    "small_objects": {
                        "elevator_button_2": {"affordance": ["press"]}
                    },
                    "large_objects": {},
                },
            },
        }
        action, change = ground_local_action(
            "scan(elevator_cabin)",
            scene,
            ["goto(elevator_1f): trans from(1f) to(2f)"],
        )
        self.assertEqual(action, "press(elevator_button_2)")
        self.assertEqual(change["operation"], "select_pending_transition_floor")

    def test_repeated_room_scan_advances_pending_pick(self) -> None:
        scene = make_scene()
        scene["agent"]["scan_history"] = ["room_a", "shelf", "parcel"]
        action, change = ground_local_action(
            "scan(room_a)",
            scene,
            ["goto(room_a): pick(parcel)"],
        )
        self.assertEqual(action, "pick(parcel)")
        self.assertEqual(change["operation"], "advance_pending_object_interaction")

    def test_repeated_room_scan_routes_to_pending_room(self) -> None:
        scene = make_scene()
        scene["agent"]["scan_history"] = ["room_a"]
        action, change = ground_local_action(
            "scan(room_a)",
            scene,
            ["goto(room_b): place(parcel)"],
        )
        self.assertEqual(action, "goto(room_b)")
        self.assertEqual(change["operation"], "advance_pending_route")

    def test_minimal_local_grounding_does_not_inject_pending_action(self) -> None:
        scene = make_scene()
        scene["agent"]["scan_history"] = ["room_a", "shelf", "parcel"]
        action, change = ground_local_action(
            "scan(room_a)",
            scene,
            ["goto(room_a): pick(parcel)"],
            repair_stalled_action=False,
        )
        self.assertEqual(action, "scan(room_a)")
        self.assertIsNone(change)


if __name__ == "__main__":
    unittest.main()
