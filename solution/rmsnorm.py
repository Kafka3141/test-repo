"""Frozen-weight Qwen2 RMSNorm, independently implemented for this challenge.

Reference semantics: transformers v5.16.1 Qwen2RMSNorm.forward:
https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py
The normalized value is rounded to input dtype BEFORE multiplication by weight.
Weights remain frozen; trainable norms use the original implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rms_rows_forward(X, W, Y, R, N: tl.constexpr, EPS: tl.constexpr,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    normalized = (x * inv).to(X.dtype.element_ty).to(tl.float32)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    tl.store(Y + row * N + col, normalized * w, col < N)
    tl.store(R + row, inv)


@triton.jit
def rms_rows_backward(DY, X, W, R, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
    dy = tl.load(DY + row * N + col, col < N, other=0).to(tl.float32)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    # Autograd rounds at the cast back to input dtype in the reference forward.
    g = (dy * w).to(X.dtype.element_ty).to(tl.float32)
    inv = tl.load(R + row)
    dot = tl.sum(g * x, 0) / N
    dx = inv * (g - x * (inv * inv) * dot)
    tl.store(DX + row * N + col, dx, col < N)


class FrozenRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x = x.contiguous()
        weight = weight.contiguous()
        n = x.shape[-1]
        rows = x.numel() // n
        y = torch.empty(x.shape, device=x.device,
                        dtype=torch.promote_types(x.dtype, weight.dtype))
        inv = torch.empty(rows, device=x.device, dtype=torch.float32)
        rms_rows_forward[(rows,)](x, weight, y, inv, n, eps,
                                  BLOCK=triton.next_power_of_2(n),
                                  num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(x, weight, inv)
        return y

    @staticmethod
    def backward(ctx, grad):
        x, weight, inv = ctx.saved_tensors
        dx = torch.empty_like(x)
        n = x.shape[-1]
        rms_rows_backward[(x.numel() // n,)](
            grad.contiguous(), x, weight, inv, dx, n,
            BLOCK=triton.next_power_of_2(n), num_warps=4,
            enable_fp_fusion=False)
        return dx, None, None


def frozen_rmsnorm(x, weight, eps=1e-6):
    if weight.requires_grad:
        raise ValueError("This implementation requires frozen RMSNorm weights")
    return FrozenRMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls.forward, "_soup_rmsnorm_patch", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        weight = self.weight
        if (not hidden_states.is_cuda or weight.requires_grad
                or hidden_states.dtype not in (torch.float16, torch.float32)
                or weight.dtype not in (torch.float16, torch.float32)
                or hidden_states.shape[-1] == 0
                or hidden_states.shape[-1] > 8192
                or hidden_states.numel() == 0):
            return original(self, hidden_states)
        return frozen_rmsnorm(hidden_states, weight, self.variance_epsilon)

    forward._soup_rmsnorm_patch = True
    cls.forward = forward
