import torch

from exllamav3.loader.frozen_tensors import FrozenTensorSource
from exllamav3.modules.linear import Linear
from exllamav3.modules.quant.fp16 import LinearFP16


class SourceCollection:
    def __init__(self, source):
        self.source = source

    def has_tensor_group(self, key, subkeys):
        return self.source.has_tensor_group(key, subkeys)

    def get_tensor(self, key, device=None, **kwargs):
        return self.source.get_tensor(key, device, **kwargs)


class SourceConfig:
    def __init__(self, source):
        self.stc = SourceCollection(source)


def make_linear(config=None):
    return Linear(
        config,
        "model.linear",
        in_features=2,
        out_features=2,
        pad_to=2,
        weight_scale=3.0,
    )


def test_fp16_freeze_round_trip_applies_weight_scale_once_to_weight_and_bias():
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
    bias = torch.tensor([5.0, 6.0], dtype=torch.float16)
    scale = 3.0
    live = make_linear()
    live.inner = LinearFP16(
        2,
        2,
        weight * scale,
        bias * scale,
        2,
        2,
        key="model.linear",
    )
    live.device = torch.device("cpu")
    live.quant_type = "fp16"

    source = FrozenTensorSource(live.get_tensors())
    restored = make_linear(SourceConfig(source))
    restored.load(torch.device("cpu"))

    torch.testing.assert_close(restored.inner.weight, live.inner.weight)
    torch.testing.assert_close(restored.inner.bias, live.inner.bias)
