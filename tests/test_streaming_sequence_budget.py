import unittest

import torch

from utils.streaming_hlr import StreamingSceneInstructionQwenModel


class StreamingSequenceBudgetTest(unittest.TestCase):
    class FakeTokenizer:
        def __init__(self) -> None:
            self.padding_side = "left"

        def __call__(
            self,
            texts: list[str],
            *,
            padding: bool,
            truncation: bool,
            return_tensors: str,
            add_special_tokens: bool,
        ) -> dict[str, torch.Tensor]:
            self.assert_call_options = (
                padding,
                truncation,
                return_tensors,
                add_special_tokens,
                self.padding_side,
            )
            token_rows = []
            for text in texts:
                content = text.removesuffix("<|endoftext|>")
                token_rows.append([len(piece) for piece in content.split()] + [99])

            padded_length = max(len(row) for row in token_rows)
            input_ids = torch.zeros((len(token_rows), padded_length), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for row_index, row in enumerate(token_rows):
                input_ids[row_index, : len(row)] = torch.tensor(row)
                attention_mask[row_index, : len(row)] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def make_model(self) -> StreamingSceneInstructionQwenModel:
        model = StreamingSceneInstructionQwenModel.__new__(
            StreamingSceneInstructionQwenModel
        )
        torch.nn.Module.__init__(model)
        model.max_prefix_length = 12
        model.segment_token_limits = {
            "system": 2,
            "instruction": 2,
            "completed": 3,
            "pending": 3,
        }
        return model

    def make_segment(self, values: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = torch.tensor(values, dtype=torch.float32).view(1, -1, 1)
        mask = torch.ones((1, len(values)), dtype=torch.long)
        return embeds, mask

    def test_prefix_preserves_mandatory_segments_and_trims_history(self) -> None:
        model = self.make_model()
        system = self.make_segment([1, 2, 3])
        scene = self.make_segment([10, 11, 12])
        instruction = self.make_segment([20, 21, 22])
        completed = self.make_segment([30, 31, 32, 33])
        pending = self.make_segment([40, 41, 42, 43])
        output = self.make_segment([50])

        embeds, mask, lengths = model._assemble_segmented_prefix(
            *system,
            *scene,
            *instruction,
            *completed,
            *pending,
            *output,
        )

        active = embeds[0][mask[0].bool()].flatten().tolist()
        self.assertEqual(lengths, [12])
        self.assertEqual(active[:7], [1, 2, 10, 11, 12, 21, 22])
        self.assertEqual(active[-4:], [40, 41, 42, 50])
        self.assertNotIn(3, active)

    def test_prefix_rejects_mandatory_segment_overflow(self) -> None:
        model = self.make_model()
        model.max_prefix_length = 5
        segment = self.make_segment([1, 2])
        empty = (
            torch.zeros((1, 0, 1), dtype=torch.float32),
            torch.zeros((1, 0), dtype=torch.long),
        )

        with self.assertRaisesRegex(ValueError, "exceed the prefix budget"):
            model._assemble_segmented_prefix(
                *segment,
                *segment,
                *segment,
                *empty,
                *empty,
                *segment,
            )

    def test_zero_optional_budget_drops_history(self) -> None:
        model = self.make_model()
        model.max_prefix_length = 8
        mandatory = self.make_segment([1, 2])
        history = self.make_segment([30, 31, 32])

        embeds, mask, _ = model._assemble_segmented_prefix(
            *mandatory,
            *mandatory,
            *mandatory,
            *history,
            *history,
            *mandatory,
        )

        active = embeds[0][mask[0].bool()].flatten().tolist()
        self.assertEqual(active, [1, 2, 1, 2, 1, 2, 1, 2])
        self.assertNotIn(30, active)

    def test_targets_use_dynamic_padding_and_keep_eos(self) -> None:
        model = self.make_model()
        model.max_output_length = 8
        tokenizer = self.FakeTokenizer()

        encoded = model._encode_training_targets(
            tokenizer, ["short", "a longer target"]
        )

        self.assertEqual(encoded["input_ids"].shape, (2, 4))
        self.assertEqual(encoded["attention_mask"].sum(dim=1).tolist(), [2, 4])
        self.assertEqual(encoded["input_ids"][0, 1].item(), 99)
        self.assertEqual(encoded["input_ids"][1, 3].item(), 99)
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual(
            tokenizer.assert_call_options,
            (True, False, "pt", False, "right"),
        )

    def test_targets_over_limit_raise_without_truncation(self) -> None:
        model = self.make_model()
        model.max_output_length = 3
        tokenizer = self.FakeTokenizer()

        with self.assertRaisesRegex(ValueError, "refusing to truncate supervision"):
            model._encode_training_targets(tokenizer, ["one two three"])

        self.assertEqual(tokenizer.padding_side, "left")


if __name__ == "__main__":
    unittest.main()
