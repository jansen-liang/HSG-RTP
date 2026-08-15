import unittest

import torch
from torch import nn

from utils.graph_encoder import HierarchicalSceneGraphEncoder


class EdgeIgnoringIdentity(nn.Module):
    def forward(self, node_features, edge_index):
        return node_features


class FailIfCalled(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("Topology GNN should be disabled")


class LocalGraphEncoderTest(unittest.TestCase):
    def make_encoder(self):
        encoder = HierarchicalSceneGraphEncoder.__new__(HierarchicalSceneGraphEncoder)
        nn.Module.__init__(encoder)
        encoder.llm_dim = 4
        encoder.item_proj = nn.Identity()
        encoder.type_embedding = nn.Embedding(4, 4)
        nn.init.zeros_(encoder.type_embedding.weight)
        encoder.type_to_idx = {"macro": 0, "room": 1, "item": 2, "agent": 3}
        descriptions = []

        def encode_text(text):
            descriptions.append(text)
            return torch.full((1, 4), float(len(descriptions)))

        encoder.encode_text = encode_text
        return encoder, descriptions

    def test_local_scene_preserves_each_object_token(self):
        encoder, descriptions = self.make_encoder()
        scene = {
            "current_room": "office",
            "room": {
                "small_objects": {
                    "bottle_b": {"type": "bottle", "state": "closed"},
                    "bottle_a": {"type": "bottle", "state": "open"},
                },
                "large_objects": {"desk": {"type": "surface", "state": "clear"}},
            },
            "agent": {"position": "office", "state": "idle", "inventory": []},
        }

        sequence = encoder.encode_local_scene(scene)

        self.assertEqual(sequence.shape, (5, 4))
        self.assertEqual(encoder.sequence_length(scene), 5)
        self.assertIn("bottle_a", descriptions[1])
        self.assertIn("bottle_b", descriptions[2])
        self.assertIn("desk", descriptions[3])

    def test_empty_room_contains_room_and_agent_only(self):
        encoder, _ = self.make_encoder()
        scene = {
            "current_room": "empty_room",
            "room": {},
            "agent": {"position": "empty_room", "state": "idle"},
        }

        self.assertEqual(encoder.encode_local_scene(scene).shape, (2, 4))
        self.assertEqual(encoder.sequence_length(scene), 2)


class GlobalGraphEncoderTest(unittest.TestCase):
    def make_encoder(self):
        encoder = HierarchicalSceneGraphEncoder.__new__(HierarchicalSceneGraphEncoder)
        nn.Module.__init__(encoder)
        encoder.llm_dim = 4
        encoder.item_proj = nn.Identity()
        encoder.item_attn_gate = nn.Linear(4, 1)
        encoder.room_gnn = EdgeIgnoringIdentity()
        encoder.room_post_gnn = nn.Identity()
        encoder.room_residual_proj = nn.Identity()
        encoder.use_room_gnn = True
        encoder.type_embedding = nn.Embedding(4, 4)
        nn.init.zeros_(encoder.type_embedding.weight)
        encoder.type_to_idx = {"macro": 0, "room": 1, "item": 2, "agent": 3}
        descriptions = []

        def encode_text(text):
            descriptions.append(text)
            return torch.full((1, 4), float(len(descriptions)))

        encoder.encode_text = encode_text
        return encoder, descriptions

    def test_global_scene_consumes_hierarchical_metadata(self):
        encoder, descriptions = self.make_encoder()
        scene = {
            "name": "hotel",
            "agent": {
                "position": "lobby",
                "state": "hand-free",
                "battery": 87,
                "type": "service_robot",
            },
            "macro_zones": {
                "floor_1_public": {
                    "rooms": ["lobby", "restaurant"],
                }
            },
            "rooms": {
                "lobby": {
                    "floor": "floor_1_public",
                    "type": "reception",
                    "neighbor": ["restaurant"],
                },
                "restaurant": {
                    "floor": "floor_1_public",
                    "type": "dining",
                    "neighbor": ["lobby"],
                },
            },
        }

        sequence = encoder.encode_single_scene(scene)
        encoded_text = "\n".join(descriptions)

        self.assertEqual(sequence.shape, (4, 4))
        self.assertIn("Macro zone 'floor_1_public'", encoded_text)
        self.assertIn('member rooms: ["lobby", "restaurant"]', encoded_text)
        self.assertIn("Room identity: 'lobby'", encoded_text)
        self.assertIn("floor: floor_1_public", encoded_text)
        self.assertIn('macro zones: ["floor_1_public"]', encoded_text)
        self.assertIn("type: reception", encoded_text)
        self.assertIn("Agent type: service_robot", encoded_text)
        self.assertIn("state: hand-free", encoded_text)
        self.assertIn("battery: 87", encoded_text)

    def test_floor_field_supplies_membership_when_zone_list_is_incomplete(self):
        memberships = HierarchicalSceneGraphEncoder.build_zone_memberships({
            "macro_zones": {"floor_2_guest": {"rooms": []}},
            "rooms": {
                "room_201": {
                    "floor": "floor_2_guest",
                    "neighbor": [],
                }
            },
        })

        self.assertEqual(memberships["room_201"], ["floor_2_guest"])

    def test_disabling_topology_keeps_all_global_tokens(self):
        encoder, _ = self.make_encoder()
        encoder.use_room_gnn = False
        encoder.room_gnn = FailIfCalled()
        encoder.room_post_gnn = FailIfCalled()
        scene = {
            "agent": {"position": "room_a", "state": "idle"},
            "macro_zones": {"floor_1": {"rooms": ["room_a", "room_b"]}},
            "rooms": {
                "room_a": {"floor": "floor_1", "neighbor": ["room_b"]},
                "room_b": {"floor": "floor_1", "neighbor": ["room_a"]},
            },
        }

        sequence = encoder.encode_single_scene(scene)

        self.assertEqual(sequence.shape, (4, 4))
        self.assertEqual(encoder.sequence_length(scene), 4)


if __name__ == "__main__":
    unittest.main()
