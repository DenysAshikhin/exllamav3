import torch

from exllamav3.modules.quant.exl3_lib.quantize import finalize_capture_H


def test_nonfinite_hessian_uses_quantization_fallback():
    hessian = torch.eye(16, dtype=torch.float)
    hessian[0, 0] = torch.inf
    capture = {
        "H": hessian,
        "count": 1,
        "device": torch.device("cpu"),
        "finalized": False,
        "inf_nan": torch.tensor([1, 0]),
    }

    fallback, _, decomposition, _, diagonal = finalize_capture_H(capture, {}, False)

    assert fallback
    assert decomposition is None
    assert diagonal is None
    assert capture["finalized"]
