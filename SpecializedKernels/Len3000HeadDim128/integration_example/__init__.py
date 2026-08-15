"""Stock Alpamayo 1.5 + HGA, no ROS / no optimized node.

This folder has its own kernel. Do not import these modules from the
ROS adapter — ROS uses ``Len3000HeadDim128.alpamayo_adapter``.
"""
from .adapter import adapt_alpamayo

__all__ = ["adapt_alpamayo", "sample_no_generation"]


def __getattr__(name: str):
    if name == "sample_no_generation":
        from .nogener import sample_no_generation

        return sample_no_generation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
