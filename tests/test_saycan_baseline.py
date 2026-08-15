from evaluation.saycan_baseline import (
    SayCanAdaptationPredictor,
    enumerate_saycan_skills,
    evaluate_saycan_dataset,
)


def sample_view():
    return {
        "current_room": "room_a",
        "agent": {"position": "room_a", "state": "hand-free"},
        "room": {
            "neighbor": ["room_b"],
            "small_objects": {
                "parcel": {"affordance": ["pick"]},
                "button": {"affordance": ["press"]},
            },
            "large_objects": {"desk": {"placement_relation": "on"}},
        },
    }


def test_skill_affordances_enforce_symbolic_preconditions():
    skills = enumerate_saycan_skills(
        sample_view(), ["room_a", "room_b", "room_c"], []
    )

    assert skills["goto(room_b)"] == 1.0
    assert skills["goto(room_c)"] == 0.0
    assert skills["pick(parcel)"] == 1.0
    assert skills["press(parcel)"] == 0.0
    assert skills["press(button)"] == 1.0
    assert skills["finish"] == 0.2


def test_skill_affordances_offer_place_only_while_holding():
    view = sample_view()
    view["agent"]["state"] = "holding-parcel"

    skills = enumerate_saycan_skills(view, ["room_a", "room_b"], [])

    assert skills["pick(parcel)"] == 0.0
    assert skills["place(parcel, floor)"] == 1.0
    assert skills["place(parcel, desk)"] == 1.0


def test_predictor_multiplies_language_and_affordance_scores():
    class Backend:
        def __init__(self):
            self.calls = []

        def reset_usage(self):
            self.calls = []

        def score_options(self, stage, system_prompt, user_prompt, options):
            assert "goto(room_c)" not in options
            self.calls.append(
                {
                    "stage": stage,
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "inference_time": 0.1,
                }
            )
            return {option: (5.0 if option == "finish" else 4.0) for option in options}

    predictor = SayCanAdaptationPredictor(Backend())
    predictor.reset({}, {"rooms": {"room_a": {}, "room_b": {}, "room_c": {}}})

    action, elapsed = predictor.predict("Go to room B", sample_view())

    assert action != "finish"
    assert elapsed == 0.1


def test_direct_rollout_finishes_without_room_level_plan():
    class Backend:
        def __init__(self):
            self.calls = []

        def reset_usage(self):
            self.calls = []

        def score_options(self, stage, system_prompt, user_prompt, options):
            at_destination = '"current_room":"room_b"' in user_prompt
            self.calls.append(
                {
                    "stage": stage,
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "inference_time": 0.1,
                }
            )
            return {
                option: (
                    10.0
                    if (at_destination and option == "finish")
                    or (not at_destination and option == "goto(room_b)")
                    else 0.0
                )
                for option in options
            }

    initial_scene = {
        "name": "test",
        "agent": {"position": "room_a", "state": "hand-free"},
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
    final_state = {
        **initial_scene,
        "agent": {"position": "room_b", "state": "hand-free"},
    }
    record = {
        "instruction": "Go to room B.",
        "scene_name": "test",
        "task_info": {
            "type": "guidance",
            "difficulty": "easy",
            "parameters": {"end_room": "room_b"},
        },
        "execution_summary": {
            "subtasks": ["goto(room_b)"],
            "final_state": final_state,
        },
    }
    predictor = SayCanAdaptationPredictor(Backend())

    results, summary = evaluate_saycan_dataset(
        [record], {"test": initial_scene}, predictor
    )

    assert results[0]["actions"] == ["goto(room_b)"]
    assert summary["exec_sr"] == 1.0
    assert summary["plan_sr"] is None
    assert summary["local_jaccard"] == 1.0
