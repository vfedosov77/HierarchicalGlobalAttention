"""Isolate flex backward grads on A4000 using the module's _flex_attend (patch applied)."""
import torch

from _common import M

DEVICE = "cuda"
T, H, D, CH = 512, 4, 128, 64
scale = 1.0 / (D ** 0.5)
C = T // CH


def build_sel(topk=2):
    # own chunk last, one past chunk (chunk c-1) when available.
    dev = DEVICE
    cidx = torch.arange(C, device=dev)
    past = (cidx - 1).clamp(min=0)[:, None]           # [C,1]
    sel = torch.cat([past, cidx[:, None]], dim=-1)    # [C,2]
    slot_valid = torch.ones(C, 2, dtype=torch.bool, device=dev)
    slot_valid[0, 0] = False                          # chunk 0 has no past
    return sel, slot_valid


def run(compiled, detached):
    q = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    sel, sv = build_sel()

    fn = M._flex_attend_detached if detached else M._flex_attend
    if compiled:
        fn = torch.compile(fn, dynamic=False, fullgraph=True)
    y = fn(q, k, v, sel, sv, CH, scale)
    loss = y.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    gq, gk, gv = (t.grad for t in (q, k, v))
    print(f"compiled={compiled} detached={detached}: y.norm={y.float().norm():.3f} "
          f"grads=({gq.float().norm():.4f},{gk.float().norm():.4f},{gv.float().norm():.4f})")


def main():
    torch.manual_seed(0)
    for compiled in (True,):
        for detached in (False, True):
            try:
                run(compiled, detached)
            except Exception as e:
                import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
