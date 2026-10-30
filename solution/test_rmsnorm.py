"""Compare forward and input gradients to the exact Transformers expression."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (2, 17, 1001), (1, 1, 1536)])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_frozen_rmsnorm(dtype, weight_dtype, shape, noncontiguous):
    from solution.rmsnorm import frozen_rmsnorm

    torch.manual_seed(23)
    storage_shape = (*shape[:-1], shape[-1] * 2) if noncontiguous else shape
    x = torch.randn(storage_shape, device="cuda", dtype=dtype)
    if noncontiguous:
        x = x[..., ::2]
    x = x.detach().requires_grad_(True)
    ref_x = x.detach().clone().requires_grad_(True)
    weight = torch.randn(shape[-1], device="cuda", dtype=weight_dtype)
    eps = 1e-6
    xf = ref_x.float()
    ref = weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(dtype)
    actual = frozen_rmsnorm(x, weight, eps)
    grad = torch.randn_like(ref)
    ref.backward(grad)
    actual.backward(grad)
    assert actual.dtype == ref.dtype
    # fp16 rounds twice in the reference graph, so compare in that precision.
    atol = 5e-3 if dtype == torch.float16 else 2e-5
    rtol = 3e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(x.grad, ref_x.grad, atol=atol, rtol=rtol)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(x.grad).all()


def test_patch_cpu_fallback_and_idempotence():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from solution.rmsnorm import patch_qwen2_rmsnorm

    norm = Qwen2RMSNorm(13)
    x = torch.randn(2, 3, 13, requires_grad=True)
    ref = norm(x)
    patch_qwen2_rmsnorm()
    patched = Qwen2RMSNorm.forward
    patch_qwen2_rmsnorm()
    assert Qwen2RMSNorm.forward is patched
    torch.testing.assert_close(norm(x), ref, rtol=0, atol=0)


def test_patch_trainable_weight_fallback():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from solution.rmsnorm import patch_qwen2_rmsnorm

    patch_qwen2_rmsnorm()
    norm = Qwen2RMSNorm(1536).cuda()
    x = torch.randn(2, 17, 1536, device="cuda", requires_grad=True)
    norm(x).sum().backward()
    assert norm.weight.grad is not None
    assert x.grad is not None
