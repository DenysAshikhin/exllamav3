import pytest
import torch
from types import SimpleNamespace
from safetensors.torch import save_file

import exllamav3.model.model as model_module
from exllamav3.loader import safetensors as safetensors_module
from exllamav3.loader.frozen_tensors import FrozenTensorSource
from exllamav3.loader.safetensors import SafetensorsCollection, VariantSafetensorsCollection
from exllamav3.model.model import Model


class FakeCollection:
    def __init__(self):
        self.read_keys = set()


class FakeConfig:
    def __init__(self, moe_cpu_hosts=None):
        self.moe_cpu_hosts = {} if moe_cpu_hosts is None else moe_cpu_hosts
        self.stc = FakeCollection()


class FakeModule:
    def __init__(self, device, tensors, key="model.fake"):
        self.device = device
        self.key = key
        self._tensors = tensors

    def __iter__(self):
        yield self

    def get_tensors(self):
        return self._tensors

    def unload(self):
        self.device = None


def make_model(
    modules,
    *,
    output_device=torch.device("cuda:0"),
    loaded_tp=False,
    active_devices=None,
    moe_cpu_hosts=None,
):
    model = Model.__new__(Model)
    model.modules = modules
    model.output_device = output_device
    model.loaded_tp = loaded_tp
    model.active_devices = [0] if active_devices is None else active_devices
    model.config = FakeConfig(moe_cpu_hosts)
    return model


def test_freeze_rejects_an_unloaded_model():
    model = make_model(
        [FakeModule(None, {})],
        output_device=None,
        active_devices=[],
    )

    with pytest.raises(RuntimeError, match="fully loaded"):
        model.freeze()


def test_freeze_rejects_tensor_parallel_models():
    model = make_model(
        [FakeModule(torch.device("cuda:0"), {"model.weight": torch.ones(1)})],
        loaded_tp=True,
    )

    with pytest.raises(RuntimeError, match="tensor-parallel"):
        model.freeze()


def test_freeze_rejects_models_using_multiple_cuda_devices():
    model = make_model(
        [
            FakeModule(torch.device("cuda:0"), {"model.one": torch.ones(1)}),
            FakeModule(torch.device("cuda:1"), {"model.two": torch.ones(1)}),
        ],
        active_devices=[0, 1],
    )

    with pytest.raises(RuntimeError, match="one CUDA device"):
        model.freeze()


def test_freeze_rejects_cpu_offloaded_moe():
    model = make_model(
        [FakeModule(torch.device("cuda:0"), {"model.weight": torch.ones(1)})],
        moe_cpu_hosts={"moe": object()},
    )

    with pytest.raises(RuntimeError, match="CPU-offloaded MoE"):
        model.freeze()


def test_freeze_rejects_partially_loaded_modules():
    model = make_model(
        [
            FakeModule(torch.device("cuda:0"), {"model.one": torch.ones(1)}),
            FakeModule(None, {"model.two": torch.ones(1)}),
        ]
    )

    with pytest.raises(RuntimeError, match="fully loaded"):
        model.freeze()


def test_freeze_returns_independent_contiguous_cpu_copies():
    live_weight = torch.arange(6.0).reshape(2, 3).t()
    mutated_live_value = live_weight.clone()
    model = make_model(
        [FakeModule(torch.device("cuda:0"), {"model.weight": live_weight})]
    )

    frozen = model.freeze()

    assert isinstance(frozen, FrozenTensorSource)
    assert frozen.tensors["model.weight"].device.type == "cpu"
    assert frozen.tensors["model.weight"].is_contiguous()
    assert frozen.tensors["model.weight"].data_ptr() != live_weight.data_ptr()
    assert torch.equal(frozen.tensors["model.weight"], mutated_live_value.cpu())
    assert not hasattr(model, "frozen_tensors")

    live_weight.zero_()
    model.unload()

    assert torch.equal(frozen.tensors["model.weight"], mutated_live_value.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_freeze_copies_real_cuda_tensor_to_independent_cpu_storage():
    live_weight = torch.arange(6.0, device="cuda:0").reshape(2, 3).t()
    original_live_value = live_weight.clone()
    module = FakeModule(torch.device("cuda:0"), {"model.weight": live_weight})
    model = make_model([module])

    frozen = model.freeze()

    assert isinstance(frozen, FrozenTensorSource)
    assert frozen.tensors["model.weight"].device == torch.device("cpu")
    assert frozen.tensors["model.weight"].is_contiguous()
    assert frozen.tensors["model.weight"].data_ptr() != live_weight.data_ptr()
    assert torch.equal(frozen.tensors["model.weight"], original_live_value.cpu())

    live_weight.zero_()
    model.unload()

    assert torch.equal(frozen.tensors["model.weight"], original_live_value.cpu())


def test_freeze_rejects_conflicting_duplicate_tensor_keys():
    model = make_model(
        [
            FakeModule(torch.device("cuda:0"), {"model.weight": torch.ones(1)}),
            FakeModule(torch.device("cuda:0"), {"model.weight": torch.zeros(1)}),
        ]
    )

    with pytest.raises(RuntimeError, match="duplicate tensor key"):
        model.freeze()


class FailingTensor:
    device = torch.device("cuda:0")

    def detach(self):
        raise RuntimeError("copy failed")


class FailingToTensor:
    device = torch.device("cuda:0")

    def __init__(self):
        self.detach_called = False
        self.to_called = False

    def detach(self):
        self.detach_called = True
        return self

    def to(self, *args, **kwargs):
        self.to_called = True
        raise RuntimeError("copy failed")


def test_freeze_propagates_copy_failures_without_mutating_model_state():
    module = FakeModule(
        torch.device("cuda:0"),
        {"model.weight": FailingTensor()},
    )
    model = make_model([module])

    with pytest.raises(RuntimeError, match="copy failed"):
        model.freeze()

    assert model.output_device == torch.device("cuda:0")
    assert module.device == torch.device("cuda:0")
    assert not hasattr(model, "frozen_tensors")


def test_freeze_propagates_to_failures_without_mutating_model_state():
    failing_tensor = FailingToTensor()
    module = FakeModule(
        torch.device("cuda:0"),
        {"model.weight": failing_tensor},
    )
    model = make_model([module])

    with pytest.raises(RuntimeError, match="copy failed"):
        model.freeze()

    assert failing_tensor.detach_called
    assert failing_tensor.to_called
    assert model.output_device == torch.device("cuda:0")
    assert module.device == torch.device("cuda:0")
    assert module.get_tensors()["model.weight"] is failing_tensor
    assert not hasattr(model, "frozen_tensors")


class LoadSTC:
    def __init__(self, disk_tensors=None):
        self.disk_tensors = {} if disk_tensors is None else disk_tensors
        self.frozen_source = None

    def set_frozen_source(self, source):
        self.frozen_source = source

    def get_tensor(self, key, device=None):
        if self.frozen_source is not None:
            target = device
            if target is not None and torch.device(target).type == "cuda" and not torch.cuda.is_available():
                target = None
            return self.frozen_source.get_tensor(key, target)
        else:
            value = self.disk_tensors[key]
        target = torch.device(device) if device is not None else torch.device("cpu")
        if target.type == "cuda" and not torch.cuda.is_available():
            return value
        return value.to(target)

    def abort_deferred_load(self):
        pass

    def close(self):
        self.set_frozen_source(None)


class DeferredSTC(LoadSTC):
    def __init__(self):
        super().__init__()
        self.deferred_mode = False
        self.deferred_loads = []

    def begin_deferred_load(self):
        assert not self.deferred_mode
        self.deferred_mode = True

    def abort_deferred_load(self):
        assert self.deferred_mode
        self.deferred_mode = False
        self.deferred_loads = []


class LoadModule:
    def __init__(self, stc, key="model.weight", fail=False):
        self.stc = stc
        self.key = key
        self.fail = fail
        self.device = None
        self.caps = {}
        self.loaded = None

    def __iter__(self):
        yield self

    def can_defer_load(self):
        return False

    def load(self, device, **kwargs):
        self.loaded = self.stc.get_tensor(self.key, device)
        self.device = torch.device(device)
        if self.fail:
            raise RuntimeError("load failed")

    def unload(self):
        self.loaded = None
        self.device = None


class DeferredLoadModule(LoadModule):
    def can_defer_load(self):
        return True

    def load(self, device, **kwargs):
        self.stc.deferred_loads.append(self.key)
        super().load(device, **kwargs)


def make_load_model(stc, *, fail=False):
    module = LoadModule(stc, fail=fail)
    model = Model.__new__(Model)
    model.modules = [module]
    model.output_device = None
    model.loaded_tp = False
    model.active_devices = []
    model.cache_weakrefs = {}
    model.config = SimpleNamespace(
        stc=stc,
        moe_cpu_hosts={},
        infer_params=SimpleNamespace(),
    )
    return model, module


def make_deferred_load_model(stc):
    module = DeferredLoadModule(stc, key="model.missing")
    model = Model.__new__(Model)
    model.modules = [module]
    model.output_device = None
    model.loaded_tp = False
    model.active_devices = []
    model.cache_weakrefs = {}
    model.config = SimpleNamespace(
        stc=stc,
        moe_cpu_hosts={},
        infer_params=SimpleNamespace(),
    )
    return model, module


def make_multi_load_model(stc):
    modules = [
        LoadModule(stc, key="model.one"),
        LoadModule(stc, key="model.two"),
    ]
    model = Model.__new__(Model)
    model.modules = modules
    model.output_device = None
    model.loaded_tp = False
    model.active_devices = []
    model.cache_weakrefs = {}
    model.config = SimpleNamespace(
        stc=stc,
        moe_cpu_hosts={},
        infer_params=SimpleNamespace(),
    )
    return model, modules


def source_device():
    return torch.device("cuda:0")


def test_load_source_restores_weights_without_mutating_source_and_clears_it():
    disk_value = torch.tensor([1.0])
    frozen_value = torch.tensor([7.0])
    stc = LoadSTC({"model.weight": disk_value})
    model, module = make_load_model(stc)
    source = FrozenTensorSource({"model.weight": frozen_value})
    original_source = source.tensors.copy()

    model.load(device=source_device(), source=source)

    assert torch.equal(module.loaded.cpu(), frozen_value)
    assert source.tensors.keys() == original_source.keys()
    assert source.tensors["model.weight"] is original_source["model.weight"]
    assert stc.frozen_source is None

    model.load(device="cpu")
    assert torch.equal(module.loaded, disk_value)


def test_complete_load_source_never_reads_disk_and_missing_source_key_does_not_fallback(
    tmp_path,
    monkeypatch,
):
    disk_value = torch.tensor([1.0])
    save_file({"model.weight": disk_value}, tmp_path / "model.safetensors")

    stc = SafetensorsCollection(str(tmp_path))
    model, module = make_load_model(stc)
    original_get_tensor = stc.get_tensor

    def get_tensor_without_cuda_transfer(key, device=None, **kwargs):
        if device is not None and torch.device(device).type == "cuda" and not torch.cuda.is_available():
            device = None
        return original_get_tensor(key, device, **kwargs)

    stc.get_tensor = get_tensor_without_cuda_transfer

    def fail_disk_read(*args, **kwargs):
        raise AssertionError("source load attempted a disk read")

    with monkeypatch.context() as patch:
        patch.setattr(safetensors_module.ext, "stloader_open_file", fail_disk_read)
        model.load(device=source_device(), source=FrozenTensorSource({"model.weight": torch.tensor([9.0])}))
        assert torch.equal(module.loaded.cpu(), torch.tensor([9.0]))

        with pytest.raises(ValueError, match="model.weight"):
            model.load(device=source_device(), source=FrozenTensorSource({}))

    model.load(device="cpu")
    assert torch.equal(module.loaded, disk_value)


def test_load_source_clears_it_when_module_loading_fails():
    stc = LoadSTC()
    model, _ = make_load_model(stc, fail=True)

    with pytest.raises(RuntimeError, match="load failed"):
        model.load(device=source_device(), source=FrozenTensorSource({"model.weight": torch.ones(1)}))

    assert stc.frozen_source is None


def test_load_source_unloads_earlier_modules_when_a_late_key_is_missing():
    stc = LoadSTC()
    model, modules = make_multi_load_model(stc)

    with pytest.raises(ValueError, match="model.two"):
        model.load(
            device=source_device(),
            source=FrozenTensorSource({"model.one": torch.ones(1)}),
        )

    assert all(module.loaded is None and module.device is None for module in modules)
    assert model.output_device is None
    assert model.active_devices == []
    assert stc.frozen_source is None


def test_failed_source_restore_aborts_deferred_loader_state():
    stc = DeferredSTC()
    model, _ = make_deferred_load_model(stc)

    with pytest.raises(ValueError, match="model.missing"):
        model.load(
            device=source_device(),
            source=FrozenTensorSource({"model.other": torch.ones(1)}),
        )

    assert not stc.deferred_mode
    assert stc.deferred_loads == []
    assert stc.frozen_source is None


def test_failed_source_restore_aborts_variant_child_deferred_loader_state(tmp_path):
    main_dir = tmp_path / "main"
    child_dir = tmp_path / "child"
    main_dir.mkdir()
    child_dir.mkdir()
    save_file({"model.main": torch.ones(1)}, main_dir / "model.safetensors")
    save_file({"model.weight": torch.ones(1)}, child_dir / "model.safetensors")

    main = SafetensorsCollection(str(main_dir))
    child = SafetensorsCollection(str(child_dir))
    stc = VariantSafetensorsCollection(main)
    stc.add_stc(["model.weight"], child)
    child.begin_deferred_load()
    model, _ = make_load_model(stc, fail=True)

    with pytest.raises(RuntimeError, match="load failed"):
        model.load(
            device=source_device(),
            source=FrozenTensorSource({"model.weight": torch.ones(1)}),
        )

    assert not child.deferred_mode
    child.begin_deferred_load()
    child.abort_deferred_load()


def test_failed_source_restore_drops_global_tensor_cache(monkeypatch):
    stc = LoadSTC()
    model, _ = make_load_model(stc, fail=True)
    drop_calls = []
    monkeypatch.setattr(
        model_module.g_tensor_cache,
        "drop_all",
        lambda: drop_calls.append(True),
    )

    with pytest.raises(RuntimeError, match="load failed"):
        model.load(device=source_device(), source=FrozenTensorSource({"model.weight": torch.ones(1)}))

    assert drop_calls == [True]
    assert stc.frozen_source is None


def test_load_source_rejects_non_cpu_tensor_values_before_loading():
    stc = LoadSTC()
    model, module = make_load_model(stc)

    with pytest.raises(ValueError, match="CPU tensors"):
        source = object.__new__(FrozenTensorSource)
        source.tensors = {"model.weight": "not a tensor"}
        model.load(device=source_device(), source=source)

    assert module.loaded is None
    assert stc.frozen_source is None


def test_load_source_cleanup_runs_when_generator_is_closed(monkeypatch):
    stc = LoadSTC()
    model, _ = make_load_model(stc)

    def fake_autosplit(*args):
        yield

    model._load_autosplit = fake_autosplit
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    load = model.load_gen(source=FrozenTensorSource({"model.weight": torch.ones(1)}))

    next(load)
    assert stc.frozen_source is not None
    load.close()

    assert stc.frozen_source is None


def test_load_source_unloads_modules_when_generator_is_closed(monkeypatch):
    stc = LoadSTC()
    model, modules = make_multi_load_model(stc)

    def fake_autosplit(*args):
        modules[0].load(source_device())
        yield

    model._load_autosplit = fake_autosplit
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    load = model.load_gen(
        source=FrozenTensorSource({"model.one": torch.ones(1), "model.two": torch.ones(1)})
    )

    next(load)
    assert modules[0].loaded is not None
    load.close()

    assert all(module.loaded is None and module.device is None for module in modules)
    assert model.output_device is None
    assert model.active_devices == []
    assert stc.frozen_source is None


def test_load_source_autosplit_terminal_callback_clears_before_collection_close(monkeypatch):
    stc = LoadSTC()
    model, _ = make_load_model(stc)
    model.modules = [LoadModule(stc)]
    callback_states = []
    close_states = []

    def callback(module, modules):
        callback_states.append(stc.frozen_source is not None)

    def close():
        close_states.append(stc.frozen_source is None)

    stc.close = close

    def fake_autosplit(*args):
        assert args[-1]
        callback_sync = args[7]
        callback_sync(len(model.modules), len(model.modules))
        stc.close()
        yield

    model._load_autosplit = fake_autosplit
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    load = model.load_gen(source=FrozenTensorSource({"model.weight": torch.ones(1)}), callback=callback)
    next(load)
    load.close()

    assert callback_states == [True]
    assert close_states == [True]
    assert stc.frozen_source is None


def test_source_load_rejects_tensor_parallel_before_installing_source():
    stc = LoadSTC()
    model, _ = make_load_model(stc)
    model.loaded_tp = True

    with pytest.raises(RuntimeError, match="tensor-parallel"):
        model.load(device=source_device(), source=FrozenTensorSource({"model.weight": torch.ones(1)}))

    assert stc.frozen_source is None


def test_source_load_rejects_tensor_parallel_request_before_installing_source():
    stc = LoadSTC()
    model, _ = make_load_model(stc)

    with pytest.raises(RuntimeError, match="tensor-parallel"):
        model.load(device=source_device(), tensor_p=True, source=FrozenTensorSource({"model.weight": torch.ones(1)}))

    assert stc.frozen_source is None


def test_source_load_rejects_default_multi_gpu_autosplit(monkeypatch):
    stc = LoadSTC()
    model, _ = make_load_model(stc)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(RuntimeError, match="one CUDA device"):
        model.load(source=FrozenTensorSource({"model.weight": torch.ones(1)}))

    assert stc.frozen_source is None


def test_source_load_allows_default_single_gpu_restore(monkeypatch):
    stc = LoadSTC()
    model, module = make_load_model(stc)

    def fake_autosplit(*args):
        module.load(source_device())
        args[7](1, 1)
        stc.close()
        yield

    model._load_autosplit = fake_autosplit
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    model.load(source=FrozenTensorSource({"model.weight": torch.ones(1)}))

    assert torch.equal(module.loaded.cpu(), torch.ones(1))
    assert stc.frozen_source is None


def test_source_load_explicit_cuda_uses_single_target_when_multiple_are_visible(monkeypatch):
    stc = LoadSTC()
    model, module = make_load_model(stc)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    model.load(device=source_device(), source=FrozenTensorSource({"model.weight": torch.ones(1)}))

    assert torch.equal(module.loaded.cpu(), torch.ones(1))
    assert stc.frozen_source is None


def test_source_load_rejects_explicit_cpu_device():
    stc = LoadSTC()
    model, _ = make_load_model(stc)

    with pytest.raises(RuntimeError, match="CUDA device"):
        model.load(device="cpu", source=FrozenTensorSource({"model.weight": torch.ones(1)}))

    assert stc.frozen_source is None


def test_source_load_rejects_cpu_offloaded_moe():
    stc = LoadSTC()
    model, _ = make_load_model(stc)
    model.config.moe_cpu_hosts = {"moe": object()}

    with pytest.raises(RuntimeError, match="CPU-offloaded MoE"):
        model.load(device=source_device(), source=FrozenTensorSource({"model.weight": torch.ones(1)}))

    assert stc.frozen_source is None


@pytest.mark.parametrize(
    ("component", "budget_name"),
    [("text", "moe_cpu_offload"), ("mtp", "draft_moe_cpu_offload")],
)
def test_source_load_rejects_configured_cpu_moe_budget_before_mutation(
    component,
    budget_name,
    monkeypatch,
):
    stc = LoadSTC()
    model, _ = make_load_model(stc)
    model.component = component
    setattr(model.config.infer_params, budget_name, 1)

    def fail_free_mem():
        raise AssertionError("source validation ran after free_mem")

    monkeypatch.setattr(model_module, "free_mem", fail_free_mem)

    with pytest.raises(RuntimeError, match="CPU-offloaded MoE"):
        model.load(device=source_device(), source=FrozenTensorSource({"model.weight": torch.ones(1)}))

    assert stc.frozen_source is None
