from types import SimpleNamespace

import torch

from exllamav3.loader.frozen_tensors import FrozenTensorSource
from exllamav3.modules.arch_specific.gemma4 import Gemma4VisionPatchEmbedder
from exllamav3.modules.arch_specific.glm4v import Glm4VPosEmbedding
from exllamav3.modules.arch_specific.qwen3_vl import Qwen3VLPosEmbedding

CPU = torch.device("cpu")


def source_config(tensors):
    """A config whose stc is the frozen source itself, which is what restore installs."""
    return SimpleNamespace(stc=FrozenTensorSource(tensors))


def freeze_module(module):
    """The snapshot Model.freeze() would build for this module, children included."""
    return {key: value for child in module for key, value in child.get_tensors().items()}


def make_qwen3_vl_pos_embedding(config):
    return Qwen3VLPosEmbedding(
        config,
        "model.visual.pos_embed",
        num_position_embeddings=16,
        hidden_size=8,
        spatial_merge_size=2,
        out_dtype=torch.float,
    )


def test_qwen3_vl_pos_embedding_round_trips_through_a_frozen_source():
    weight = torch.randn((16, 8), dtype=torch.float16)

    live = make_qwen3_vl_pos_embedding(source_config({"model.visual.pos_embed.weight": weight}))
    live.load(CPU)

    restored = make_qwen3_vl_pos_embedding(source_config(freeze_module(live)))
    restored.load(CPU)

    torch.testing.assert_close(restored.embedding.weight, live.embedding.weight)


def make_glm4v_pos_embedding(config):
    return Glm4VPosEmbedding(
        config,
        "model.visual.embeddings.position_embedding",
        num_position_embeddings=16,
        hidden_size=8,
        spatial_merge_size=2,
        out_dtype=torch.float,
    )


def test_glm4v_pos_embedding_round_trips_through_a_frozen_source():
    weight = torch.randn((16, 8), dtype=torch.float16)

    live = make_glm4v_pos_embedding(
        source_config({"model.visual.embeddings.position_embedding.weight": weight})
    )
    live.load(CPU)

    restored = make_glm4v_pos_embedding(source_config(freeze_module(live)))
    restored.load(CPU)

    torch.testing.assert_close(restored.pos_embed_2d, live.pos_embed_2d)


def make_gemma4_patch_embedder(config):
    return Gemma4VisionPatchEmbedder(
        config,
        "model.vision_tower.embedder",
        hidden_size=8,
        patch_dim=4,
        position_embedding_size=16,
        out_dtype=torch.float,
    )


def test_gemma4_patch_embedder_round_trips_through_a_frozen_source():
    live = make_gemma4_patch_embedder(source_config({
        "model.vision_tower.embedder.position_embedding_table": torch.randn((16, 8), dtype=torch.float16),
        "model.vision_tower.embedder.input_proj.weight": torch.randn((8, 4), dtype=torch.float16),
    }))
    live.load(CPU)

    snapshot = freeze_module(live)
    assert "model.vision_tower.embedder.position_embedding_table" in snapshot

    # load(), not a hand-assignment: the embedder owns a raw tensor and a child Linear, and restore
    # has to bring both back from the snapshot alone
    restored = make_gemma4_patch_embedder(source_config(snapshot))
    restored.load(CPU)

    torch.testing.assert_close(restored.position_embedding_table, live.position_embedding_table)
    restored_snapshot = freeze_module(restored)
    assert set(restored_snapshot) == set(snapshot)
    for key, value in snapshot.items():
        torch.testing.assert_close(restored_snapshot[key], value)
