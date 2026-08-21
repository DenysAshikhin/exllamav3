import pytest

from eval.qbench.data import model_cache_key, prepare_output_dirs


def model_spec(source):
    return {
        "cache_id": "laguna-s21-bf16-poolside-main",
        "engine": "transformers",
        "source": source,
        "options": {
            "streaming": True,
            "streaming_rows": 2,
            "trust_remote_code": True,
        },
    }


def test_model_cache_key_is_independent_of_source_path():
    first = model_cache_key(model_spec("D:/models/laguna"))
    relocated = model_cache_key(model_spec("E:/deleted/laguna"))

    assert first == relocated


def test_model_cache_key_changes_with_execution_shape():
    first_spec = model_spec("D:/models/laguna")
    second_spec = model_spec("D:/models/laguna")
    second_spec["options"]["streaming_rows"] = 3

    assert model_cache_key(first_spec) != model_cache_key(second_spec)


def test_model_cache_key_changes_for_noise_pass():
    spec = model_spec("D:/models/laguna")

    assert model_cache_key(spec) != model_cache_key(spec, noise_eps=0.0001)


@pytest.mark.parametrize("cache_id", [None, "", "   "])
def test_model_cache_key_requires_nonempty_cache_id(cache_id):
    spec = model_spec("D:/models/laguna")
    spec["cache_id"] = cache_id

    with pytest.raises(ValueError, match="cache_id"):
        model_cache_key(spec)


def test_prepare_output_dirs_creates_plain_and_structured_parents(tmp_path):
    results = tmp_path / "quality" / "results.json"
    graph = tmp_path / "quality" / "graphs" / "kld.png"
    project = {
        "output": {
            "results": str(results),
            "plot_kld_vram": {
                "file": str(graph),
                "x_log": True,
            },
        },
    }

    prepare_output_dirs(project)

    assert results.parent.is_dir()
    assert graph.parent.is_dir()
