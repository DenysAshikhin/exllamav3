"""Shared doubles for the freeze tests.

A freeze walks the module tree, collects tensors and checks the result against the collection's
read ledger, so every freeze test needs the same three stand-ins: a collection with a ledger, a
config owning that collection, and a module that reports a key, a device and some tensors. They
live here so a new attribute on the real `Model` is a one-file change rather than three.
"""

import torch

from exllamav3.model.model import Model


class FakeCollection:
    """Stands in for SafetensorsCollection, carrying only the read ledger freeze consults."""

    def __init__(self, read_keys=None):
        self.read_keys = set() if read_keys is None else set(read_keys)


class LedgerlessCollection:
    """A collection from before the ledger existed; freeze must refuse it."""


class FakeConfig:
    def __init__(self, moe_cpu_hosts=None, stc=None):
        self.moe_cpu_hosts = {} if moe_cpu_hosts is None else moe_cpu_hosts
        self.stc = FakeCollection() if stc is None else stc


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
    stc=None,
):
    model = Model.__new__(Model)
    model.modules = modules
    model.output_device = output_device
    model.loaded_tp = loaded_tp
    model.active_devices = [0] if active_devices is None else active_devices
    model.cache_weakrefs = {}
    model.config = FakeConfig(moe_cpu_hosts, stc)
    return model
