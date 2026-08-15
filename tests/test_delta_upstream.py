import json
import sys
import unittest
from pathlib import Path

from evaluation.delta_upstream import (
    UpstreamDeltaAdaptationPolicy,
    UpstreamDeltaPDDLPlanner,
    build_adapted_problem,
    build_upstream_decomposition_prompt,
    build_upstream_problem_prompt,
    extract_pddl_problem,
    hsg_scene_to_delta,
    parse_upstream_subgoals,
)


DELTA_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "DELTA"
sys.path.insert(0, str(DELTA_ROOT))
from data.scene_graph import extract_accessible_items_from_sg, prune_sg_with_item


class DeltaUpstreamSchemaTest(unittest.TestCase):
    def test_converted_scene_runs_through_upstream_pruner(self):
        scene = {
            "name": "test",
            "rooms": {
                "room_a": {
                    "floor": "floor_1",
                    "neighbor": ["room_b"],
                    "large_objects": {
                        "table": {"is_container": True},
                    },
                    "small_objects": {
                        "parcel": {
                            "affordance": ["pick", "place"],
                            "state": {"availability": "available"},
                            "relation": {"on": "table"},
                        }
                    },
                },
                "room_b": {
                    "floor": "floor_1",
                    "neighbor": ["room_a"],
                    "large_objects": {},
                    "small_objects": {},
                },
            },
            "agent": {"position": "room_a", "state": "hand-free"},
        }
        converted = hsg_scene_to_delta(scene)
        self.assertEqual(
            set(extract_accessible_items_from_sg(converted)), {"table", "parcel"}
        )
        self.assertIn("drop", converted["rooms"]["room_a"]["items"]["parcel"]["affordance"])
        pruned = prune_sg_with_item(converted, ["parcel"])
        self.assertEqual(
            set(pruned["rooms"]["room_a"]["items"]), {"parcel"}
        )
        self.assertEqual(pruned["rooms"]["room_a"]["neighbor"], ["room_b"])

    def test_upstream_planner_executes_adapted_delivery_problem(self):
        try:
            planner = UpstreamDeltaPDDLPlanner(DELTA_ROOT)
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        scene = {
            "agent": {"position": "room_a"},
            "rooms": {
                "room_a": {
                    "floor": "floor_1",
                    "neighbor": ["room_b"],
                    "small_objects": {"parcel": {}},
                    "large_objects": {},
                },
                "room_b": {
                    "floor": "floor_1",
                    "neighbor": ["room_a"],
                    "small_objects": {},
                    "large_objects": {},
                },
            },
        }

        actions, _, failures = planner.plan(
            scene,
            [{"kind": "deliver", "object": "parcel", "room": "room_b"}],
        )

        self.assertEqual(failures, 0)
        self.assertEqual(actions[-1], ("place", "parcel", "room_b"))

    def test_builds_and_parses_upstream_delivery_decomposition(self):
        scene = {
            "agent": {"position": "room_a"},
            "rooms": {
                "room_a": {
                    "neighbor": ["room_b"],
                    "small_objects": {"parcel": {}},
                    "large_objects": {},
                },
                "room_b": {
                    "neighbor": ["room_a"],
                    "small_objects": {},
                    "large_objects": {},
                },
            },
        }
        record = {
            "instruction": "Deliver the parcel to room B.",
            "task_info": {
                "type": "delivery",
                "parameters": {
                    "objects": ["parcel"],
                    "target_rooms": ["room_b"],
                },
            },
        }
        problem, room_symbols, object_symbols = build_adapted_problem(scene, record)

        self.assertIn(
            f"(object-at {object_symbols['parcel']} {room_symbols['room_b']})",
            problem,
        )
        parsed = parse_upstream_subgoals(
            "```\n(:goal (object-at o0 r1))\n```",
            record,
            room_symbols,
            object_symbols,
        )
        self.assertEqual(
            parsed,
            [{"kind": "deliver", "object": "parcel", "room": "room_b"}],
        )

    def test_uses_upstream_decomposition_prompt(self):
        scene = {
            "agent": {"position": "room_a"},
            "rooms": {
                "room_a": {
                    "neighbor": ["room_b"],
                    "small_objects": {},
                    "large_objects": {},
                },
                "room_b": {
                    "neighbor": ["room_a"],
                    "small_objects": {},
                    "large_objects": {},
                },
            },
        }
        record = {
            "instruction": "Visit room B.",
            "task_info": {
                "type": "guidance",
                "parameters": {"intermediate_points": [], "end_room": "room_b"},
            },
        }

        content, prompt, room_symbols, _ = build_upstream_decomposition_prompt(
            DELTA_ROOT, record, scene
        )

        self.assertIn("decomposing long-term tasks", content)
        self.assertIn("Visit room B.", prompt)
        self.assertIn(f"(visited {room_symbols['room_b']})", prompt)

    def test_uses_upstream_problem_generation_prompt(self):
        scene = {
            "name": "test_scene",
            "agent": {"position": "room_a", "state": "hand-free"},
            "rooms": {
                "room_a": {
                    "floor": "floor_1",
                    "neighbor": ["room_b"],
                    "small_objects": {},
                    "large_objects": {},
                },
                "room_b": {
                    "floor": "floor_1",
                    "neighbor": ["room_a"],
                    "small_objects": {},
                    "large_objects": {},
                },
            },
        }
        record = {"instruction": "Visit room B."}

        content, prompt = build_upstream_problem_prompt(DELTA_ROOT, record, scene)

        self.assertIn("PDDL problem file generator", content)
        self.assertIn("Visit room B.", prompt)
        self.assertIn("hsg-delta", prompt)

    def test_extracts_complete_generated_problem(self):
        response = "text before\n```pddl\n" + (
            "(define (problem test) (:domain hsg-delta) "
            "(:objects r0 - room) (:init (robot-at r0)) "
            "(:goal (visited r0)))"
        ) + "\n```"

        problem = extract_pddl_problem(response)

        self.assertTrue(problem.startswith("(define"))
        self.assertTrue(problem.endswith(")\n"))

    def test_upstream_policy_uses_pddl_decomposition_and_planner(self):
        class Backend:
            def __init__(self, generated_problem):
                self.calls = []
                self.generated_problem = generated_problem

            def reset_usage(self):
                self.calls = []

            def generate_text(self, stage, system_prompt, user_prompt):
                self.calls.append(
                    {
                        "stage": stage,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "inference_time": 0.0,
                    }
                )
                if stage == "upstream_problem_generation":
                    return self.generated_problem
                return "```\n(:goal (object-at o0 r1))\n```"

        scene = {
            "agent": {"position": "room_a"},
            "rooms": {
                "room_a": {
                    "neighbor": ["room_b"],
                    "small_objects": {"parcel": {}},
                    "large_objects": {},
                },
                "room_b": {
                    "neighbor": ["room_a"],
                    "small_objects": {},
                    "large_objects": {"desk": {}},
                },
            },
        }
        record = {
            "instruction": "Deliver the parcel to room B.",
            "task_info": {
                "type": "delivery",
                "parameters": {
                    "objects": ["parcel"],
                    "target_rooms": ["room_b"],
                },
            },
        }
        problem, _, _ = build_adapted_problem(scene, record)
        policy = UpstreamDeltaAdaptationPolicy(
            Backend(problem),
            record,
            scene,
            UpstreamDeltaPDDLPlanner(DELTA_ROOT),
            DELTA_ROOT,
        )

        prediction = json.loads(policy.generate_global(record["instruction"], scene, []))

        self.assertEqual(prediction["mode"], "global")
        self.assertTrue(prediction["task"])
        self.assertEqual(policy.decomposition_calls, 1)
        self.assertEqual(policy.problem_generation_calls, 1)
        self.assertTrue(policy.local_actions)


if __name__ == "__main__":
    unittest.main()
