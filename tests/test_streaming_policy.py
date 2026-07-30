import unittest

from evaluation.policies import StreamingModelPolicy


class RecordingModel:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        mode = "global" if kwargs["generation_config"]["do_sample"] is False else "local"
        task = ["goto(room_a): pass()"] if mode == "global" else ["scan(room_a)"]
        return {"predictions": [f'{{"mode":"{mode}","task":{task!r}}}']}


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


if __name__ == "__main__":
    unittest.main()
