"""Minimal FlexAttention fwd/bwd sanity on this A4000, isolating the zero-grad issue."""
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

DEVICE = "cuda"
T, H, D, CH = 512, 4, 128, 64
scale = 1.0 / (D ** 0.5)


def causal(b, h, q, kv):
    return q >= kv


def main():
    torch.manual_seed(0)
    q = torch.randn(1, H, T, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, H, T, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, H, T, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

    bm = create_block_mask(causal, 1, 1, T, T, device=DEVICE, BLOCK_SIZE=CH)
    fa = torch.compile(flex_attention)
    y = fa(q, k, v, block_mask=bm, scale=scale)
    loss = y.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    print("compiled flex: y.norm", y.float().norm().item(),
          "grads", [round(t.grad.float().norm().item(), 4) for t in (q, k, v)])

    # eager (uncompiled) for comparison
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    y2 = flex_attention(q2, k2, v2, block_mask=bm, scale=scale)
    loss2 = y2.float().pow(2).mean()
    loss2.backward()
    print("eager    flex: y.norm", y2.float().norm().item(),
          "grads", [round(t.grad.float().norm().item(), 4) for t in (q2, k2, v2)])


if __name__ == "__main__":
    main()
