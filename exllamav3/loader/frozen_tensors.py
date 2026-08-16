from __future__ import annotations

from collections.abc import Mapping

import torch


class FrozenTensorSource:
    """Source-only lookup for the CPU tensors retained by a frozen model."""

    def __init__(self, tensors: Mapping[str, torch.Tensor]):
        if not isinstance(tensors, Mapping):
            raise TypeError("Frozen tensor source must be a mapping")

        self.tensors = dict(tensors)
        for key, tensor in self.tensors.items():
            if not isinstance(key, str):
                raise ValueError("Frozen tensor source keys must be strings")
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                raise ValueError("Frozen tensor source must contain only CPU tensors")

    def has_tensor(self, key: str) -> bool:
        return key in self.tensors

    def has_tensor_group(self, key: str | list[str], subkeys: list) -> bool:
        if isinstance(key, list):
            return all(self.has_tensor_group(k, subkeys) for k in key)

        return all(
            (
                f"{key}.{subkey}" in self.tensors if isinstance(subkey, str) else
                any(f"{key}.{sk}" in self.tensors for sk in subkey)
            ) for subkey in subkeys
        )

    def get_tensor(
        self,
        key: str,
        device: torch.device | str | int | None = None,
        optional: bool = False,
        allow_bf16: bool = False,
        float2half: bool = False,
        no_defer: bool = False,
        transpose: bool = False,
        pad_to: tuple | None = None,
        fidx: int | None = None,
    ) -> torch.Tensor | None:
        if key not in self.tensors:
            if optional:
                return None
            raise ValueError(f"Required tensor {key} not found in supplied tensors")

        tensor = self.tensors[key]
        if fidx is not None:
            if not no_defer:
                raise AssertionError("Cannot load fused tensor in deferred mode")
            if not tensor.shape or not 0 <= fidx < tensor.shape[0]:
                raise ValueError(f"Batch tensor {key} has shape {list(tensor.shape)}, index {fidx} is out of bounds")
            tensor = tensor[fidx]

        target = torch.device("cpu") if device is None else torch.device(device)
        tensor = tensor.to(target)
        if tensor.dtype == torch.bfloat16 and not allow_bf16:
            tensor = tensor.to(torch.float16)
        if tensor.dtype == torch.float32 and float2half:
            tensor = tensor.to(torch.float16)
        if transpose:
            tensor = tensor.T if tensor.ndim > 0 else tensor
        if pad_to is not None:
            padded = torch.zeros(pad_to, dtype=tensor.dtype, device=tensor.device)
            padded[tuple(slice(0, size) for size in tensor.shape)].copy_(tensor)
            tensor = padded
        return tensor.contiguous()

    def get_tensors(
        self,
        prefix: str,
        device: torch.device | str | int | None = None,
        allow_bf16: bool = False,
    ) -> dict[str, torch.Tensor]:
        keys = sorted(
            key for key in self.tensors
            if key == prefix or key.startswith(prefix + ".")
        )
        return {
            key: self.get_tensor(key, device, allow_bf16=allow_bf16)
            for key in keys
        }

    def get_tensor_sizes(self, prefix: str) -> list[int]:
        keys = sorted(
            key for key in self.tensors
            if key == prefix or key.startswith(prefix + ".")
        )
        return [self.get_tensor_size(key) for key in keys]

    def get_tensor_size(self, key: str, optional: bool = False) -> int:
        tensor = self.tensors.get(key)
        if tensor is None:
            if optional:
                return 0
            raise ValueError(f"Required tensor {key} not found in supplied tensors")
        return tensor.numel() * tensor.element_size()

    def list_tensors(self, prefix: str, only_serializable: bool = False) -> dict:
        keys = sorted(
            key for key in self.tensors
            if key == prefix or key.startswith(prefix + ".")
        )
        results = {}
        for key in keys:
            result = {
                "shape": list(self.tensors[key].shape),
                "n_bytes": self.get_tensor_size(key),
                "dtype": str(self.tensors[key].dtype),
            }
            if not only_serializable:
                result["torch_dtype"] = self.tensors[key].dtype
            results[key] = result
        return results

    def get_tensor_meta(self, key: str, optional: bool = True) -> dict | None:
        tensor = self.tensors.get(key)
        if tensor is None:
            if optional:
                return None
            raise ValueError(f"Required tensor {key} not found in supplied tensors")
        return {
            key: {
                "shape": list(tensor.shape),
                "n_bytes": self.get_tensor_size(key),
                "dtype": str(tensor.dtype),
            }
        }

    def max_key_len(self) -> int:
        return max((len(key) for key in self.tensors), default=0)
