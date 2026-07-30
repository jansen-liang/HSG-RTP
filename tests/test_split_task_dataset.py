import unittest

from scripts.split_task_dataset import split_records, stratum, task_id


def make_record(index: int, scene: str, task_type: str, difficulty: str) -> dict:
    return {
        "instruction": f"instruction {index}",
        "scene_name": scene,
        "task_info": {"type": task_type, "difficulty": difficulty},
        "execution_summary": {"global_plan": [f"plan {index}"], "subtasks": [f"action {index}"]},
        "streaming_samples": [{"target": f"action {index}"}],
    }


class SplitTaskDatasetTest(unittest.TestCase):
    def test_split_is_deterministic_and_task_disjoint(self) -> None:
        records = [
            make_record(index, f"scene-{index % 5}", f"type-{index % 3}", f"difficulty-{index % 2}")
            for index in range(100)
        ]
        duplicate = dict(records[0])
        duplicate["timestamp"] = 1
        records.append(duplicate)

        first_train, first_test = split_records(records, test_ratio=20 / 101, seed=42)
        second_train, second_test = split_records(records, test_ratio=20 / 101, seed=42)

        self.assertEqual(len(first_train), 81)
        self.assertEqual(len(first_test), 20)
        self.assertEqual(
            [task_id(record) for record in first_train],
            [task_id(record) for record in second_train],
        )
        self.assertEqual(
            [task_id(record) for record in first_test],
            [task_id(record) for record in second_test],
        )
        self.assertTrue(
            {task_id(record) for record in first_train}.isdisjoint(
                task_id(record) for record in first_test
            )
        )
        self.assertLessEqual(
            {stratum(record) for record in first_test},
            {stratum(record) for record in records},
        )


if __name__ == "__main__":
    unittest.main()
