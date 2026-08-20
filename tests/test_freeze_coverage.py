import pytest
import torch

from exllamav3.model.model import Model


class LedgerCollection:
    def __init__(self, read_keys):
        self.read_keys = set(read_keys)


class BareCollection:
    pass


class LedgerConfig:
    def __init__(self, collection):
        self.stc = collection
        self.moe_cpu_hosts = {}


class KeyedModule:
    def __init__(self, key, tensors):
        self.key = key
        self.device = torch.device("cuda:0")
        self._tensors = tensors

    def __iter__(self):
        yield self

    def get_tensors(self):
        return self._tensors


def make_model(modules, collection):
    model = Model.__new__(Model)
    model.modules = modules
    model.output_device = torch.device("cuda:0")
    model.loaded_tp = False
    model.active_devices = [0]
    model.config = LedgerConfig(collection)
    return model


def test_freeze_rejects_a_module_that_consumed_tensors_and_contributes_none():
    model = make_model(
        [
            KeyedModule("model.layers.0.mlp", {"model.layers.0.mlp.up.weight": torch.ones(1)}),
            KeyedModule("model.visual.pos_embed", {}),
        ],
        LedgerCollection([
            "model.layers.0.mlp.up.weight",
            "model.visual.pos_embed.weight",
        ]),
    )

    with pytest.raises(RuntimeError, match="model.visual.pos_embed"):
        model.freeze()


def test_freeze_accepts_a_module_that_recanonicalised_its_keys():
    model = make_model(
        [KeyedModule("model.layers.0.mlp", {"model.layers.0.mlp.experts.0.up.weight": torch.ones(1)})],
        LedgerCollection(["model.layers.0.mlp.experts.gate_up_proj_blocks"]),
    )

    source = model.freeze()

    assert source.has_tensor("model.layers.0.mlp.experts.0.up.weight")


def test_freeze_ignores_read_keys_owned_by_another_component():
    model = make_model(
        [KeyedModule("model.visual.pos_embed", {"model.visual.pos_embed.weight": torch.ones(1)})],
        LedgerCollection([
            "model.visual.pos_embed.weight",
            "mtp.layers.0.fc.weight",
            "tokenizer.json",
        ]),
    )

    source = model.freeze()

    assert set(source.tensors) == {"model.visual.pos_embed.weight"}


def test_freeze_refuses_a_collection_that_kept_no_ledger():
    model = make_model(
        [KeyedModule("model.visual.pos_embed", {"model.visual.pos_embed.weight": torch.ones(1)})],
        BareCollection(),
    )

    with pytest.raises(RuntimeError, match="no record of loaded keys"):
        model.freeze()
