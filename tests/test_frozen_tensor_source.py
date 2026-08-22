import pytest
import torch
from safetensors.torch import save_file

from exllamav3.loader import safetensors as safetensors_module
from exllamav3.loader.frozen_tensors import FrozenTensorSource
from exllamav3.loader.safetensors import SafetensorsCollection, VariantSafetensorsCollection


def test_frozen_source_lookup_is_complete_and_never_falls_back_to_disk(tmp_path, monkeypatch):
    save_file(
        {
            "model.required": torch.tensor([1.0]),
            "model.optional": torch.tensor([2.0]),
        },
        tmp_path / "model.safetensors",
    )
    collection = SafetensorsCollection(str(tmp_path))
    source = FrozenTensorSource({"model.present": torch.tensor([7.0])})
    collection.set_frozen_source(source)

    def fail_disk_read(*args, **kwargs):
        raise AssertionError("frozen source attempted a disk read")

    monkeypatch.setattr(safetensors_module.ext, "stloader_open_file", fail_disk_read)

    assert collection.has_tensor("model.present")
    assert not collection.has_tensor("model.required")
    assert not collection.has_tensor_group("model", ["required"])
    assert torch.equal(collection.get_tensor("model.present"), source.tensors["model.present"])
    assert collection.get_tensor("model.optional", optional=True) is None
    with pytest.raises(ValueError, match="model.required"):
        collection.get_tensor("model.required")


def test_frozen_source_preserves_tensor_shape_dtype_and_requested_transforms():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source = FrozenTensorSource({"model.weight": tensor})

    transposed = source.get_tensor("model.weight", transpose=True)
    converted = source.get_tensor("model.weight", float2half=True)

    assert transposed.shape == (3, 2)
    assert transposed.dtype == torch.float32
    assert converted.shape == tensor.shape
    assert converted.dtype == torch.float16
    assert torch.equal(converted.float(), tensor)


def test_frozen_source_requires_cpu_tensors():
    with pytest.raises(ValueError, match="CPU tensors"):
        FrozenTensorSource({"model.weight": "not a tensor"})


def test_new_tensor_overlay_still_falls_back_to_disk_and_clears_frozen_source(tmp_path):
    disk_tensor = torch.tensor([2.0])
    save_file({"model.required": disk_tensor}, tmp_path / "model.safetensors")
    collection = SafetensorsCollection(str(tmp_path))

    collection.set_new_tensors({"model.present": torch.tensor([7.0])})
    assert torch.equal(collection.get_tensor("model.required"), disk_tensor)

    collection.set_frozen_source(FrozenTensorSource({"model.required": torch.tensor([8.0])}))
    assert torch.equal(collection.get_tensor("model.required"), torch.tensor([8.0]))

    collection.set_new_tensors(None)
    assert collection.frozen_source is None
    assert torch.equal(collection.get_tensor("model.required"), disk_tensor)


def test_variant_frozen_source_is_authoritative_for_all_lookups(tmp_path, monkeypatch):
    main_dir = tmp_path / "main"
    child_dir = tmp_path / "child"
    main_dir.mkdir()
    child_dir.mkdir()
    save_file(
        {"model.main": torch.tensor([1.0]), "model.missing": torch.tensor([2.0])},
        main_dir / "model.safetensors",
    )
    save_file({"model.child": torch.tensor([3.0])}, child_dir / "model.safetensors")

    main = SafetensorsCollection(str(main_dir))
    child = SafetensorsCollection(str(child_dir))
    variant = VariantSafetensorsCollection(main)
    variant.add_stc(["model.child"], child)
    source = FrozenTensorSource({"model.source": torch.tensor([7.0])})
    variant.set_frozen_source(source)

    def fail_disk_read(*args, **kwargs):
        raise AssertionError("variant frozen source attempted a disk read")

    monkeypatch.setattr(safetensors_module.ext, "stloader_open_file", fail_disk_read)

    assert variant.has_tensor("model.source")
    assert not variant.has_tensor("model.main")
    assert variant.has_tensor_group("model", ["source"])
    assert not variant.has_tensor_group("model", ["missing"])
    assert variant.get_tensor_size("model.source") == source.tensors["model.source"].nbytes
    assert variant.get_tensor_sizes("model") == [source.tensors["model.source"].nbytes]
    assert variant.get_tensor_meta("model.source") == source.get_tensor_meta("model.source")
    assert set(variant.list_tensors("model")) == {"model.source"}
    assert set(variant.get_tensors("model")) == {"model.source"}
    assert torch.equal(variant.get_tensor("model.source"), source.tensors["model.source"])
    assert variant.get_tensor("model.missing", optional=True) is None
    assert variant.max_key_len() == len("model.source")
