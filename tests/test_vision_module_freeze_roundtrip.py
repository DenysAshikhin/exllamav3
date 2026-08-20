import torch

from exllamav3.loader.frozen_tensors import FrozenTensorSource
from exllamav3.modules.arch_specific.gemma4 import Gemma4VisionPatchEmbedder
from exllamav3.modules.arch_specific.glm4v import Glm4VPosEmbedding
from exllamav3.modules.arch_specific.qwen3_vl import Qwen3VLPosEmbedding


class SourceCollection:
    """Minimal stc stand-in serving tensors from a frozen source, exactly as restore does."""

    def __init__(self, source):
        self.source = source

    def has_tensor(self, key):
        return self.source.has_tensor(key)

    def has_tensor_group(self, key, subkeys):
        return self.source.has_tensor_group(key, subkeys)

    def get_tensor(self, key, device=None, **kwargs):
        return self.source.get_tensor(key, device, **kwargs)

    def get_tensors(self, prefix, device=None, allow_bf16=False):
        return self.source.get_tensors(prefix, device, allow_bf16)


class SourceConfig:
    def __init__(self, source):
        self.stc = SourceCollection(source)


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
    disk = FrozenTensorSource({"model.visual.pos_embed.weight": weight})

    live = make_qwen3_vl_pos_embedding(SourceConfig(disk))
    live.load(torch.device("cpu"))

    frozen = FrozenTensorSource(live.get_tensors())
    restored = make_qwen3_vl_pos_embedding(SourceConfig(frozen))
    restored.load(torch.device("cpu"))

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
    disk = FrozenTensorSource({"model.visual.embeddings.position_embedding.weight": weight})

    live = make_glm4v_pos_embedding(SourceConfig(disk))
    live.load(torch.device("cpu"))

    frozen = FrozenTensorSource(live.get_tensors())
    restored = make_glm4v_pos_embedding(SourceConfig(frozen))
    restored.load(torch.device("cpu"))

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
    disk = FrozenTensorSource({
        "model.vision_tower.embedder.position_embedding_table": torch.randn((16, 8), dtype=torch.float16),
        "model.vision_tower.embedder.input_proj.weight": torch.randn((8, 4), dtype=torch.float16),
    })

    live = make_gemma4_patch_embedder(SourceConfig(disk))
    live.load(torch.device("cpu"))

    frozen = FrozenTensorSource(live.get_tensors())
    assert "model.vision_tower.embedder.position_embedding_table" in frozen.tensors

    restored = make_gemma4_patch_embedder(SourceConfig(frozen))
    restored.position_embedding_table = frozen.get_tensor(
        "model.vision_tower.embedder.position_embedding_table",
        torch.device("cpu"),
        float2half=True,
        allow_bf16=True,
    )

    torch.testing.assert_close(restored.position_embedding_table, live.position_embedding_table)
