from evaluation.external_baselines import (
    HuggingFaceJSONBackend,
    collapsed_scene_graph,
    expanded_scene_subgraph,
    parse_room_selection,
)


def sample_scene():
    return {
        "name": "test",
        "agent": {"position": "room_a"},
        "rooms": {
            "room_a": {
                "floor": "1f",
                "neighbor": ["room_b"],
                "large_objects": {"desk": {"type": "furniture"}},
                "small_objects": {"cup": {"type": "container"}},
            },
            "room_b": {
                "floor": "1f",
                "neighbor": ["room_a"],
                "large_objects": {},
                "small_objects": {},
            },
        },
    }


def test_collapsed_graph_hides_object_identity_but_keeps_types():
    collapsed = collapsed_scene_graph(sample_scene())

    assert "cup" not in str(collapsed)
    assert collapsed["rooms"]["room_a"]["object_types"] == {
        "furniture": 1,
        "container": 1,
    }


def test_expanded_subgraph_keeps_only_selected_valid_rooms():
    expanded = expanded_scene_subgraph(sample_scene(), ["room_b", "missing"])

    assert list(expanded["rooms"]) == ["room_b"]


def test_room_selection_deduplicates_and_filters_unknown_ids():
    selected = parse_room_selection(
        '{"rooms":["room_b","missing","room_b","room_a"]}',
        {"room_a", "room_b"},
    )

    assert selected == ["room_b", "room_a"]


def test_json_backend_delegates_payload_to_text_generation():
    backend = object.__new__(HuggingFaceJSONBackend)
    captured = {}

    def generate_text(stage, system_prompt, user_prompt):
        captured.update(
            stage=stage,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return "result"

    backend.generate_text = generate_text

    result = backend.generate("stage", "system", {"room": "大厅", "step": 1})

    assert result == "result"
    assert captured == {
        "stage": "stage",
        "system_prompt": "system",
        "user_prompt": '{"room":"大厅","step":1}',
    }
