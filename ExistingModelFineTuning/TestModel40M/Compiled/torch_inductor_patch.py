"""Backport of the upstream PyTorch fix for an Inductor codegen crash.

torch 2.11 crashes with

    InductorError: TypeError: list indices must be integers or slices, not NoneType

in ``TritonSymbols.get_block_shape`` whenever an ``index_expr`` references a
range-tree symbol whose ``tensor_dim`` is ``None`` (``no_x_dim`` kernels, i.e.
persistent reduction/scan kernels with XBLOCK=1).  Here this is triggered by the
``cumsum`` scan in ``GlobalAttentionFused._causal_rolling_sum_fast`` fusing with
the rotary-embedding indexing that consumes it.

Upstream now treats such symbols as scalars (empty block shape).  This module
applies the same fix in place; it is a no-op once the installed torch already
handles ``tensor_dim is None``.

Usage: ``import torch_inductor_patch; torch_inductor_patch.apply()`` before
``torch.compile`` runs.
"""

import inspect


def apply() -> None:
    from torch._inductor.codegen.triton import TritonSymbols
    from torch._inductor.shape_propagation import BlockShapeType, get_broadcasted_shape
    from torch._inductor.virtualized import V
    from torch.utils._sympy.symbol import SymT, prefix_str, symbol_is_type

    src = inspect.getsource(TritonSymbols.get_block_shape.__func__)
    if "tensor_dim is None" in src:
        return  # installed torch already has the upstream fix

    @classmethod  # type: ignore[misc]
    def get_block_shape(cls, expr) -> BlockShapeType:
        expr_shape: BlockShapeType = ()
        for var in expr.free_symbols:
            if symbol_is_type(var, SymT.TMP):
                var_shape = V.kernel.cse.varname_map[var.name].shape
            elif symbol_is_type(
                var,
                (
                    SymT.UNBACKED_INT,
                    SymT.SIZE,
                    SymT.PRECOMPUTED_SIZE,
                    SymT.INDEX,
                    SymT.FLOAT,
                    SymT.UNBACKED_FLOAT,
                ),
            ):
                var_shape = ()
            else:
                symbol_matches = [
                    symt for symt in cls.block_types if symbol_is_type(var, symt)
                ]
                assert len(symbol_matches) == 1, f"Ambiguous type: {var.name}"

                sym = symbol_matches[0]
                ndim = V.kernel.triton_tensor_ndim()
                shape = ["1"] * ndim

                tree_match = [
                    tree
                    for tree in V.kernel.active_range_trees()
                    if prefix_str[sym] == tree.prefix
                ]
                assert len(tree_match) == 1, "# of Match expected to 1"

                tree = tree_match[0]
                if tree.tensor_dim is None:
                    # tree has no tensor dimension (e.g. no_x_dim mode):
                    # the index is a scalar inside the kernel
                    var_shape = ()
                else:
                    shape[tree.tensor_dim] = str(cls.get_block_size(tree))
                    var_shape = tuple(shape)

            expr_shape = get_broadcasted_shape(expr_shape, var_shape)

        assert expr_shape is not None
        return expr_shape

    TritonSymbols.get_block_shape = get_block_shape
