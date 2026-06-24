"""Engine-neutral HGA core.

Exposes the pieces the engine backends compose:

* :class:`HgaConfig`            — geometry + budgets.
* :class:`RouteMetadata`        — per-forward routing selection (engine-neutral).
* summary builder              — mixed-RoPE chunk/group summaries from K/V.
* fused kernels                — decode attention + chunk/group top-k.
* tiered cache manager         — GPU summary/token banks, pinned host KV, cold spill.
"""

from .config import HgaConfig
from .route_metadata import RouteMetadata

__all__ = ["HgaConfig", "RouteMetadata"]


def __getattr__(name):  # lazy to avoid importing torch-heavy modules at package import
    if name == "HgaLayerRunner":
        from .runner import HgaLayerRunner
        return HgaLayerRunner
    if name == "HgaCacheManager":
        from .cache.manager import HgaCacheManager
        return HgaCacheManager
    raise AttributeError(name)
