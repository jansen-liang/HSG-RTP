import unittest

from evaluation.policies import RoutedStreamingModelPolicy, StreamingModelPolicy


class RecordingModel:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        mode = "global" if kwargs["generation_config"]["do_sample"] is False else "local"
        task = ["goto(room_a): pass()"] if mode == "global" else ["scan(room_a)"]
        return {
            "predictions": [f'{{"mode":"{mode}","task":{task!r}}}'],
            "usage": [{"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}],
        }


class StreamingModelPolicyTest(unittest.TestCase):
    def test_global_is_deterministic_and_local_is_sampled(self) -> None:
        model = RecordingModel()
        policy = StreamingModelPolicy(model)

        policy.generate_global("instruction", {"rooms": {"room_a": {}}}, [])
        policy.generate_local(
            "instruction",
            {"current_room": "room_a"},
            [],
            ["goto(room_a): pass()"],
        )

        global_config = model.calls[0]["generation_config"]
        local_config = model.calls[1]["generation_config"]
        self.assertEqual(
            global_config,
            {"do_sample": False},
        )
        self.assertTrue(local_config["do_sample"])
        self.assertEqual(local_config["temperature"], 0.1)
        self.assertEqual(local_config["top_p"], 0.95)
        self.assertEqual(
            model.calls[0]["generation_config"],
            {"do_sample": False},
        )

        usage = policy.usage_summary()
        self.assertEqual(usage["model_calls"], 2)
        self.assertEqual(usage["input_tokens"], 20)
        self.assertEqual(usage["output_tokens"], 8)
        self.assertEqual(usage["total_tokens"], 28)

        policy.reset_usage()
        self.assertEqual(policy.usage_summary()["model_calls"], 0)

    def test_static_scene_reuses_first_local_view(self) -> None:
        model = RecordingModel()
        policy = StreamingModelPolicy(model, static_scene=True)
        first = {"room": {"id": "room_a", "items": {"cup": {}}}}
        second = {"room": {"id": "room_a", "items": {}}}

        policy.generate_local("instruction", first, [], [])
        policy.generate_local("instruction", second, [], [])

        self.assertEqual(
            model.calls[0]["scene_graphs"], model.calls[1]["scene_graphs"]
        )
        self.assertIn("cup", first["room"]["items"])

    def test_routed_policy_uses_separate_models(self) -> None:
        global_model = RecordingModel()
        local_model = RecordingModel()
        policy = RoutedStreamingModelPolicy(
            global_model,
            local_model,
            local_generation_config={"do_sample": False},
        )

        policy.generate_global("instruction", {"rooms": {"room_a": {}}}, [])
        policy.generate_local(
            "instruction", {"current_room": "room_a"}, [], []
        )

        self.assertEqual(len(global_model.calls), 1)
        self.assertEqual(len(local_model.calls), 1)
        self.assertEqual(policy.usage_summary()["model_calls"], 2)


if __name__ == "__main__":
    unittest.main()
