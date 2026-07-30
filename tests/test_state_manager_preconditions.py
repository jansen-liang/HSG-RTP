from copy import deepcopy
import unittest

from pipeline.utils.state_manager import SceneGraphStateManager
from tests.test_task_evaluation import make_record, make_scene


def make_strict_scene() -> dict:
    scene = make_scene()
    scene["rooms"]["room_a"]["small_objects"]["fixed_item"] = {
        "type": "object",
        "affordance": ["place"],
        "relation": {"on": "shelf"},
    }
    scene["rooms"]["room_a"]["large_objects"]["bin"] = {
        "type": "furniture",
        "is_container": True,
        "placement_relation": "in",
    }
    scene["rooms"]["room_c"] = {
        "floor": "1f",
        "neighbor": [],
        "small_objects": {},
        "large_objects": {},
    }
    return scene


class StateManagerPreconditionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = SceneGraphStateManager(verbose=False)
        self.manager.load_initial_state(make_strict_scene())

    def assert_failed_without_state_change(self, action: str) -> str:
        before = deepcopy(self.manager.current_state)
        history_length = len(self.manager.state_history)
        action_count = before["state_metadata"]["action_count"]
        version = before["state_metadata"]["version"]

        success, returned_state, error = self.manager.execute_action(action)

        self.assertFalse(success)
        self.assertIsNotNone(error)
        self.assertEqual(returned_state, before)
        self.assertEqual(self.manager.current_state, before)
        self.assertEqual(len(self.manager.state_history), history_length)
        self.assertEqual(
            self.manager.current_state["state_metadata"]["action_count"], action_count
        )
        self.assertEqual(
            self.manager.current_state["state_metadata"]["version"], version
        )
        self.assertFalse(self.manager.execution_log[-1]["success"])
        self.assertEqual(self.manager.execution_log[-1]["state_version"], version)
        return error

    def test_remote_room_scan_fails(self) -> None:
        error = self.assert_failed_without_state_change("scan(room_b)")
        self.assertIn("not local", error)

    def test_local_room_scan_succeeds(self) -> None:
        success, state, error = self.manager.execute_action("scan(room_a)")
        self.assertTrue(success, error)
        self.assertEqual(state["agent"]["last_scanned"], "room_a")

    def test_local_object_and_surface_scans_succeed(self) -> None:
        for target in ("parcel", "shelf"):
            success, _, error = self.manager.execute_action(f"scan({target})")
            self.assertTrue(success, error)
        self.assertEqual(
            self.manager.current_state["agent"]["scan_history"],
            ["parcel", "shelf"],
        )

    def test_missing_scan_target_fails(self) -> None:
        self.assert_failed_without_state_change("scan(missing)")

    def test_floor_scan_is_local(self) -> None:
        success, state, error = self.manager.execute_action("scan(floor)")
        self.assertTrue(success, error)
        self.assertEqual(state["agent"]["last_scanned"], "floor")

    def test_pick_without_affordance_fails(self) -> None:
        error = self.assert_failed_without_state_change("pick(fixed_item)")
        self.assertIn("not pickable", error)

    def test_pick_local_pickable_object_succeeds(self) -> None:
        success, state, error = self.manager.execute_action("pick(parcel)")
        self.assertTrue(success, error)
        self.assertIn("parcel", state["agent"]["inventory"])
        self.assertNotIn("parcel", state["rooms"]["room_a"]["small_objects"])

    def test_place_on_missing_surface_preserves_inventory(self) -> None:
        self.assertTrue(self.manager.execute_action("pick(parcel)")[0])
        error = self.assert_failed_without_state_change("place(parcel, missing)")
        self.assertIn("Surface missing not found", error)
        self.assertIn("parcel", self.manager.current_state["agent"]["inventory"])

    def test_place_on_local_surface_succeeds(self) -> None:
        self.assertTrue(self.manager.execute_action("pick(parcel)")[0])
        success, state, error = self.manager.execute_action("place(parcel, shelf)")
        self.assertTrue(success, error)
        self.assertEqual(
            state["rooms"]["room_a"]["small_objects"]["parcel"]["relation"],
            {"on": "shelf"},
        )

    def test_place_into_container_uses_in_relation(self) -> None:
        self.assertTrue(self.manager.execute_action("pick(parcel)")[0])
        success, state, error = self.manager.execute_action("place(parcel, bin)")
        self.assertTrue(success, error)
        self.assertEqual(
            state["rooms"]["room_a"]["small_objects"]["parcel"]["relation"],
            {"in": "bin"},
        )

    def test_non_neighbor_goto_fails(self) -> None:
        error = self.assert_failed_without_state_change("goto(room_c)")
        self.assertIn("Cannot reach", error)

    def test_neighbor_goto_succeeds(self) -> None:
        success, state, error = self.manager.execute_action("goto(room_b)")
        self.assertTrue(success, error)
        self.assertEqual(state["agent"]["position"], "room_b")

    def test_unknown_wait_fails(self) -> None:
        error = self.assert_failed_without_state_change("wait(anything)")
        self.assertIn("Unknown wait condition", error)

    def test_known_elevator_wait_requires_elevator_location(self) -> None:
        error = self.assert_failed_without_state_change("wait(elevator_up_clear)")
        self.assertIn("invalid in room_a", error)

    def test_success_updates_metadata_and_history_consistently(self) -> None:
        success, state, error = self.manager.execute_action("scan(room_a)")
        self.assertTrue(success, error)
        self.assertEqual(len(self.manager.state_history), 2)
        self.assertEqual(state["state_metadata"]["version"], 2)
        self.assertEqual(state["state_metadata"]["action_count"], 1)
        self.assertEqual(self.manager.state_history[-1], state)
        self.assertTrue(self.manager.execution_log[-1]["success"])
        self.assertEqual(self.manager.execution_log[-1]["state_version"], 2)

    def test_existing_reference_execution_still_succeeds(self) -> None:
        manager = SceneGraphStateManager(verbose=False)
        manager.load_initial_state(make_scene())
        for action in make_record()["execution_summary"]["subtasks"]:
            success, _, error = manager.execute_action(action)
            self.assertTrue(success, f"{action}: {error}")


if __name__ == "__main__":
    unittest.main()
