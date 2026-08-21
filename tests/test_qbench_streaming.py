from eval.qbench.engines import TransformersBackend


class FakeShardHandle:
    def __init__(self):
        self.closed = False

    def __exit__(self, exception_type, exception, traceback):
        self.closed = True


def test_streaming_backend_releases_open_shard_mappings():
    first = FakeShardHandle()
    second = FakeShardHandle()
    backend = object.__new__(TransformersBackend)
    backend.shard_handles = {"first": first, "second": second}

    backend._release_shards()

    assert first.closed
    assert second.closed
    assert backend.shard_handles == {}
