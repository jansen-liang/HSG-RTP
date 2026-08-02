import json
import unittest

from evaluation.perturbations import (
    BlockEdgePerturbation,
    FailActionOncePerturbation,
    MoveObjectPerturbation,
)
from evaluation.recovery import RecoveryConfig
from evaluation.rollout_evaluator import rollout_policy, subgoal_satisfied
from tests.test_task_evaluation import make_record, make_scene


class ScriptedRecoveryPolicy:
    def __init__(self, global_plans: list[list[str]], local_actions: list[str]):
        self.global_plans = iter(global_plans)
        self.local_actions = iter(local_actions)
        self.global_instructions = []
        self.local_instructions = []

    def generate_global(self, instruction, scene_graph, completed):
        self.global_instructions.append(instruction)
        try:
            plan = next(self.global_plans)
        except StopIteration as error:
            raise RuntimeError("No scripted global plan remains") from error
        return json.dumps({"mode": "global", "task": plan})

    def generate_local(self, instruction, scene_graph, completed, pending):
        self.local_instructions.append(instruction)
        try:
            action = next(self.local_actions)
        except StopIteration as error:
            raise RuntimeError("No scripted local action remains") from error
        return json.dumps({"mode": "local", "task": [action]})


def make_three_room_scene() -> dict:
    scene = make_scene()
    scene["rooms"]["room_a"]["neighbor"] = ["room_b", "room_c"]
    scene["rooms"]["room_b"]["neighbor"] = ["room_a", "room_c"]
    scene["rooms"]["room_c"] = {
        "floor": "1f",
        "neighbor": ["room_a", "room_b"],
        "small_objects": {},
        "large_objects": {"counter": {"type": "furniture"}},
    }
    return scene


class ClosedLoopRecoveryTest(unittest.TestCase):
    def test_strict_rollout_reports_the_original_execution_failure(self) -> None:
        record = make_record()
        policy = ScriptedRecoveryPolicy(
            [record["execution_summary"]["global_plan"]],
            ["place(parcel, table)"],
        )

        result = rollout_policy(
            record,
            make_scene(),
            policy,
            recovery_config=RecoveryConfig(
                max_initial_plan_retries=0,
                max_local_retries=0,
                max_global_replans=0,
            ),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.failure_type, "execution_error")

    def test_transition_subgoal_uses_elevator_destination_room(self) -> None:
        scene = {
            "agent": {"position": "elevator_2f"},
            "rooms": {
                "elevator_1f": {"floor": "floor_1_public"},
                "elevator_2f": {"floor": "floor_2_guest"},
            },
        }

        self.assertTrue(
            subgoal_satisfied(
                "goto(elevator_1f): trans from(1f) to(2f)", scene, []
            )
        )

    def test_temporary_failure_retries_the_same_local_action(self) -> None:
        record = make_record()
        policy = ScriptedRecoveryPolicy(
            [record["execution_summary"]["global_plan"]],
            ["pick(parcel)", "goto(room_b)", "place(parcel, table)"],
        )

        result = rollout_policy(
            record,
            make_scene(),
            policy,
            perturbations=[FailActionOncePerturbation("pick(parcel)")],
        )

        self.assertTrue(result.success, result.failure_message)
        self.assertEqual(result.actions.count("pick(parcel)"), 1)
        self.assertTrue(
            any(
                event.get("event") == "local_failure"
                and event.get("retryable_same_action")
                for event in result.recovery_trace
            )
        )
        self.assertTrue(
            any(
                event.get("event") == "automatic_retry_attempt"
                and event.get("action") == "pick(parcel)"
                for event in result.recovery_trace
            )
        )
        self.assertFalse(
            any("RECOVERY CONTEXT" in instruction for instruction in policy.local_instructions)
        )

    def test_invalid_initial_plan_is_repaired_with_room_constraints(self) -> None:
        record = make_record()
        policy = ScriptedRecoveryPolicy(
            [
                ["goto(missing_room): pick(parcel)"],
                record["execution_summary"]["global_plan"],
            ],
            ["pick(parcel)", "goto(room_b)", "place(parcel, table)"],
        )

        result = rollout_policy(
            record,
            make_scene(),
            policy,
            recovery_config=RecoveryConfig(max_initial_plan_retries=2),
        )

        self.assertTrue(result.success, result.failure_message)
        self.assertEqual(len(policy.global_instructions), 2)
        self.assertNotIn("Valid room IDs", policy.global_instructions[0])
        self.assertIn('Valid room IDs: ["room_a", "room_b"]', policy.global_instructions[1])
        self.assertIn("missing_room", policy.global_instructions[1])
        self.assertTrue(
            any(
                event.get("event") == "initial_global_plan_failure"
                and event.get("failure_type") == "invalid_global_plan"
                for event in result.recovery_trace
            )
        )
        self.assertTrue(
            any(
                event.get("event") == "initial_global_plan_success"
                and event.get("attempt") == 2
                for event in result.recovery_trace
            )
        )

    def test_ungrounded_pass_step_is_dropped_before_validation(self) -> None:
        record = make_record()
        policy = ScriptedRecoveryPolicy(
            [[
                "goto(room_a): pick(parcel)",
                "goto(phantom_corridor): pass()",
                "goto(room_b): place(parcel)",
            ]],
            ["pick(parcel)", "goto(room_b)", "place(parcel, table)"],
        )

        result = rollout_policy(record, make_scene(), policy)

        self.assertTrue(result.success, result.failure_message)
        self.assertEqual(len(policy.global_instructions), 1)
        self.assertNotIn("phantom_corridor", " ".join(result.plan))
        self.assertTrue(
            any(
                event.get("event") == "initial_global_plan_grounded"
                for event in result.recovery_trace
            )
        )

    def test_grounding_removes_off_path_pass_detour(self) -> None:
        record = make_record()
        scene = make_three_room_scene()
        policy = ScriptedRecoveryPolicy(
            [[
                "goto(phantom_corridor): pass()",
                "goto(room_c): pass()",
                "goto(room_a): pick(parcel)",
                "goto(room_b): place(parcel)",
            ]],
            ["pick(parcel)", "goto(room_b)", "place(parcel, table)"],
        )

        result = rollout_policy(record, scene, policy)

        self.assertTrue(result.success, result.failure_message)
        self.assertNotIn("room_c", " ".join(result.plan))
        grounding_event = next(
            event
            for event in result.recovery_trace
            if event.get("event") == "initial_global_plan_grounded"
        )
        self.assertTrue(
            any(
                change.get("operation") == "drop_off_path_pass"
                and change.get("from") == "room_c"
                for change in grounding_event["changes"]
            )
        )

    def test_guidance_scan_global_step_is_normalized_to_pass(self) -> None:
        record = make_record()
        record["task_info"] = {
            "type": "guidance",
            "difficulty": "easy",
            "parameters": {
                "waypoints": ["room_a", "room_b"],
                "start_room": "room_a",
                "end_room": "room_b",
            },
        }
        record["execution_summary"]["global_plan"] = ["goto(room_b): pass()"]
        record["execution_summary"]["subtasks"] = ["goto(room_b)"]
        record["execution_summary"]["final_state"]["agent"]["position"] = "room_b"
        policy = ScriptedRecoveryPolicy(
            [["goto(room_b): scan(room_b)"]],
            ["goto(room_b)"],
        )

        result = rollout_policy(record, make_scene(), policy)

        self.assertTrue(result.success, result.failure_message)
        self.assertEqual(result.plan, ("goto(room_b): pass()",))
        grounding_event = next(
            event
            for event in result.recovery_trace
            if event.get("event") == "initial_global_plan_grounded"
        )
        self.assertTrue(
            any(
                change.get("operation") == "normalize_guidance_scan"
                for change in grounding_event["changes"]
            )
        )

    def test_bare_global_goto_is_normalized_to_pass(self) -> None:
        record = make_record()
        policy = ScriptedRecoveryPolicy(
            [[
                "goto(room_a)",
                "goto(room_a): pick(parcel)",
                "goto(room_b): place(parcel)",
            ]],
            ["pick(parcel)", "goto(room_b)", "place(parcel, table)"],
        )

        result = rollout_policy(record, make_scene(), policy)

        self.assertTrue(result.success, result.failure_message)
        self.assertEqual(
            result.plan,
            (
                "goto(room_a): pass()",
                "goto(room_a): pick(parcel)",
                "goto(room_b): place(parcel)",
            ),
        )
        grounding_event = next(
            event
            for event in result.recovery_trace
            if event.get("event") == "initial_global_plan_grounded"
        )
        self.assertTrue(
            any(
                change.get("operation") == "normalize_bare_goto"
                for change in grounding_event["changes"]
            )
        )

    def test_moved_object_triggers_global_plan_replacement(self) -> None:
        record = make_record()
        revised_plan = [
            "goto(room_c): pick(parcel)",
            "goto(room_b): place(parcel)",
        ]
        policy = ScriptedRecoveryPolicy(
            [record["execution_summary"]["global_plan"], revised_plan],
            [
                "pick(parcel)",
                "goto(room_c)",
                "pick(parcel)",
                "goto(room_a)",
                "goto(room_b)",
                "place(parcel, table)",
            ],
        )

        result = rollout_policy(
            record,
            make_three_room_scene(),
            policy,
            recovery_config=RecoveryConfig(max_local_retries=0, max_global_replans=1),
            perturbations=[
                MoveObjectPerturbation(0, "parcel", "room_a", "room_c")
            ],
        )

        self.assertTrue(result.success, result.failure_message)
        self.assertEqual(result.plan, tuple(revised_plan))
        self.assertTrue(
            any(
                event.get("event") == "global_replan_success"
                for event in result.recovery_trace
            )
        )
        self.assertTrue(
            any(
                event.get("event") == "local_recovery_exhausted"
                for event in result.recovery_trace
            )
        )
        self.assertTrue(
            any(
                event.get("event") == "global_replan_attempt"
                for event in result.recovery_trace
            )
        )
        self.assertTrue(
            any(
                "GLOBAL REPLANNING CONTEXT" in instruction
                for instruction in policy.global_instructions
            )
        )

    def test_blocked_edge_replans_through_an_alternate_room(self) -> None:
        record = make_record()
        revised_plan = ["goto(room_b): place(parcel)"]
        policy = ScriptedRecoveryPolicy(
            [record["execution_summary"]["global_plan"], revised_plan],
            [
                "pick(parcel)",
                "goto(room_b)",
                "goto(room_c)",
                "goto(room_b)",
                "place(parcel, table)",
            ],
        )

        result = rollout_policy(
            record,
            make_three_room_scene(),
            policy,
            recovery_config=RecoveryConfig(max_local_retries=0, max_global_replans=1),
            perturbations=[BlockEdgePerturbation(1, "room_a", "room_b")],
        )

        self.assertTrue(result.success, result.failure_message)
        self.assertIn("goto(room_c)", result.actions)
        self.assertTrue(
            any(
                event.get("event") == "perturbation"
                and event.get("perturbation_type") == "block_edge"
                for event in result.recovery_trace
            )
        )


if __name__ == "__main__":
    unittest.main()
