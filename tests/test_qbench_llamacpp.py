import sys
from types import SimpleNamespace

from eval.qbench import engines


class FakeLlama:
    kwargs = None

    def __init__(self, **kwargs):
        FakeLlama.kwargs = kwargs


def test_llamacpp_backend_passes_cpu_moe_and_thread_options(monkeypatch):
    split_mode = SimpleNamespace(
        LLAMA_SPLIT_MODE_LAYER=1,
        LLAMA_SPLIT_MODE_ROW=2,
        LLAMA_SPLIT_MODE_NONE=0,
    )
    module = SimpleNamespace(
        Llama=FakeLlama,
        llama_cpp=SimpleNamespace(llama_split_mode=split_mode),
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setattr(
        engines,
        "gguf_storage_info",
        lambda source: {
            "bpw_layer": 4.0,
            "bpw_head": 4.0,
            "vram_gb": 1.0,
        },
    )

    engines.LlamaCppBackend(
        "model.gguf",
        4096,
        SimpleNamespace(),
        {
            "n_gpu_layers": "all",
            "cpu_moe": True,
            "n_cpu_moe": 0,
            "n_threads": 12,
            "n_threads_batch": 12,
        },
    )

    assert FakeLlama.kwargs["n_gpu_layers"] == "all"
    assert FakeLlama.kwargs["cpu_moe"] is True
    assert FakeLlama.kwargs["n_cpu_moe"] == 0
    assert FakeLlama.kwargs["n_threads"] == 12
    assert FakeLlama.kwargs["n_threads_batch"] == 12
