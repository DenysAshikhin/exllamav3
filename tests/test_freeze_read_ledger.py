import torch
from safetensors.torch import save_file

from exllamav3.loader.frozen_tensors import FrozenTensorSource
from exllamav3.loader.safetensors import SafetensorsCollection, VariantSafetensorsCollection


def test_collection_records_keys_it_served_from_disk(tmp_path):
    save_file(
        {"model.weight": torch.ones(2), "model.bias": torch.zeros(2)},
        str(tmp_path / "model.safetensors"),
    )
    stc = SafetensorsCollection(str(tmp_path))

    stc.get_tensor("model.weight")

    assert stc.read_keys == {"model.weight"}


def test_collection_does_not_record_absent_optional_probes(tmp_path):
    save_file({"model.weight": torch.ones(2)}, str(tmp_path / "model.safetensors"))
    stc = SafetensorsCollection(str(tmp_path))

    assert stc.get_tensor("model.missing", optional=True) is None
    assert stc.read_keys == set()


def test_collection_does_not_record_reads_served_from_a_frozen_source(tmp_path):
    save_file({"model.weight": torch.ones(2)}, str(tmp_path / "model.safetensors"))
    stc = SafetensorsCollection(str(tmp_path))
    stc.set_frozen_source(FrozenTensorSource({"model.weight": torch.ones(2)}))

    stc.get_tensor("model.weight")

    assert stc.read_keys == set()


def test_prefix_reads_are_recorded_key_by_key(tmp_path):
    save_file(
        {"model.a.weight": torch.ones(2), "model.a.bias": torch.zeros(2)},
        str(tmp_path / "model.safetensors"),
    )
    stc = SafetensorsCollection(str(tmp_path))

    stc.get_tensors("model.a")

    assert stc.read_keys == {"model.a.weight", "model.a.bias"}


def test_variant_collection_unions_the_ledgers_of_its_children(tmp_path):
    main_dir = tmp_path / "main"
    child_dir = tmp_path / "child"
    main_dir.mkdir()
    child_dir.mkdir()
    save_file({"model.main": torch.ones(1)}, str(main_dir / "model.safetensors"))
    save_file({"model.child": torch.ones(1)}, str(child_dir / "model.safetensors"))
    stc = VariantSafetensorsCollection(SafetensorsCollection(str(main_dir)))
    stc.add_stc(["model.child"], SafetensorsCollection(str(child_dir)))

    stc.get_tensor("model.main")
    stc.get_tensor("model.child")

    assert stc.read_keys == {"model.main", "model.child"}
