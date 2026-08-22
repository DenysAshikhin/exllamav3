import pytest
import torch

from freeze_fakes import FakeCollection, FakeModule, LedgerlessCollection, make_model

CUDA0 = torch.device("cuda:0")


def test_freeze_rejects_a_module_that_consumed_tensors_and_contributes_none():
    model = make_model(
        [
            FakeModule(CUDA0, {"model.layers.0.mlp.up.weight": torch.ones(1)}, "model.layers.0.mlp"),
            FakeModule(CUDA0, {}, "model.visual.pos_embed"),
        ],
        stc=FakeCollection([
            "model.layers.0.mlp.up.weight",
            "model.visual.pos_embed.weight",
        ]),
    )

    with pytest.raises(RuntimeError, match="model.visual.pos_embed"):
        model.freeze()


def test_freeze_accepts_a_parent_covered_only_by_its_children():
    """
    Coverage is subtree-wide on purpose, and this is why: vision Attention reads one fused
    <key>.qkv.weight and snapshots it under its child Linears' keys, contributing nothing at its
    own level. Requiring every module to contribute at its own level refuses this model - verified
    against the real 3.8_27b preset, where it rejected all model.visual.blocks.N.attn modules.
    The cost is that a module owning a raw tensor *and* contributing children slips through here;
    tests/test_vision_module_freeze_roundtrip.py is what pins that case.
    """
    model = make_model(
        [
            FakeModule(CUDA0, {}, "model.visual.blocks.0.attn"),
            FakeModule(
                CUDA0,
                {"model.visual.blocks.0.attn.q_proj.weight": torch.ones(1)},
                "model.visual.blocks.0.attn.q_proj",
            ),
        ],
        stc=FakeCollection(["model.visual.blocks.0.attn.qkv.weight"]),
    )

    source = model.freeze()

    assert set(source.tensors) == {"model.visual.blocks.0.attn.q_proj.weight"}


def test_freeze_accepts_a_parent_and_child_that_both_contribute():
    model = make_model(
        [
            FakeModule(
                CUDA0,
                {"model.vision_tower.embedder.position_embedding_table": torch.ones(1)},
                "model.vision_tower.embedder",
            ),
            FakeModule(
                CUDA0,
                {"model.vision_tower.embedder.input_proj.weight": torch.ones(1)},
                "model.vision_tower.embedder.input_proj",
            ),
        ],
        stc=FakeCollection([
            "model.vision_tower.embedder.position_embedding_table",
            "model.vision_tower.embedder.input_proj.weight",
        ]),
    )

    source = model.freeze()

    assert set(source.tensors) == {
        "model.vision_tower.embedder.position_embedding_table",
        "model.vision_tower.embedder.input_proj.weight",
    }


def test_freeze_accepts_a_module_that_recanonicalised_its_keys():
    model = make_model(
        [FakeModule(
            CUDA0,
            {"model.layers.0.mlp.experts.0.up.weight": torch.ones(1)},
            "model.layers.0.mlp",
        )],
        stc=FakeCollection(["model.layers.0.mlp.experts.gate_up_proj_blocks"]),
    )

    source = model.freeze()

    assert source.has_tensor("model.layers.0.mlp.experts.0.up.weight")


def test_freeze_ignores_read_keys_owned_by_another_component():
    model = make_model(
        [FakeModule(CUDA0, {"model.visual.pos_embed.weight": torch.ones(1)}, "model.visual.pos_embed")],
        stc=FakeCollection([
            "model.visual.pos_embed.weight",
            "mtp.layers.0.fc.weight",
            "tokenizer.json",
        ]),
    )

    source = model.freeze()

    assert set(source.tensors) == {"model.visual.pos_embed.weight"}


def test_freeze_refuses_a_collection_that_kept_no_ledger():
    model = make_model(
        [FakeModule(CUDA0, {"model.visual.pos_embed.weight": torch.ones(1)}, "model.visual.pos_embed")],
        stc=LedgerlessCollection(),
    )

    with pytest.raises(RuntimeError, match="no record of loaded keys"):
        model.freeze()
