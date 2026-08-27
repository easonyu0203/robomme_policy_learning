import contextlib
import functools as ft
import inspect
from typing import TypeAlias, TypeVar, cast

import beartype
import jax
import jax._src.tree_util as private_tree_util
import jax.core
from jaxtyping import ArrayLike
from jaxtyping import Bool  # noqa: F401
from jaxtyping import DTypeLike  # noqa: F401
from jaxtyping import Float
from jaxtyping import Int  # noqa: F401
from jaxtyping import Key  # noqa: F401
from jaxtyping import Num  # noqa: F401
from jaxtyping import PyTree
from jaxtyping import Real  # noqa: F401
from jaxtyping import UInt8  # noqa: F401
from jaxtyping import config
from jaxtyping import jaxtyped
import jaxtyping._decorator
import torch

# patch jaxtyping to handle https://github.com/patrick-kidger/jaxtyping/issues/277.
# the problem is that custom PyTree nodes are sometimes initialized with arbitrary types (e.g., `jax.ShapeDtypeStruct`,
# `jax.Sharding`, or even <object>) due to JAX tracing operations. this patch skips typechecking when the stack trace
# contains `jax._src.tree_util`, which should only be the case during tree unflattening.
_original_check_dataclass_annotations = jaxtyping._decorator._check_dataclass_annotations  # noqa: SLF001
# Redefine Array to include both JAX arrays and PyTorch tensors
Array = jax.Array | torch.Tensor


def _check_dataclass_annotations(self, typechecker):
    if not any(
        frame.frame.f_globals.get("__name__") in {"jax._src.tree_util", "flax.nnx.transforms.compilation"}
        for frame in inspect.stack()
    ):
        return _original_check_dataclass_annotations(self, typechecker)
    return None


jaxtyping._decorator._check_dataclass_annotations = _check_dataclass_annotations  # noqa: SLF001

KeyArrayLike: TypeAlias = jax.typing.ArrayLike
Params: TypeAlias = PyTree[Float[ArrayLike, "..."]]

T = TypeVar("T")


# runtime type-checking decorator
def typecheck(t: T) -> T:
    return cast(T, ft.partial(jaxtyped, typechecker=beartype.beartype)(t))


@contextlib.contextmanager
def disable_typechecking():
    initial = config.jaxtyping_disable
    config.update("jaxtyping_disable", True)  # noqa: FBT003
    yield
    config.update("jaxtyping_disable", initial)


def _normalize_numeric_dict_keys(tree):
    # A dict node keyed 0,1,... (from a plain-list nnx submodule container) and one keyed
    # "0","1",... (from a dict-of-str-index container) are the same structure conceptually, but
    # jax's pytree equality check treats int vs numeric-string keys as a real mismatch. Collapse
    # both to a canonical string form before comparing so this doesn't look like a structural diff.
    # NOTE: this fixes check_pytree_equality specifically, not loading in general -- nnx's
    # replace_by_pure_dict does its own independent, unconditional int-coercion on incoming keys
    # (try_convert_int) *after* this check passes, and does NOT treat int/string keys as
    # interchangeable the way this function's docstring-adjacent comment used to imply: loading a
    # string-keyed dict (e.g. Selector.blocks, e8402e5) into a live state whose flat_state() keys
    # are also genuinely string still fails inside replace_by_pure_dict with "key in pure_dict not
    # available in state", independent of whether this equality check passes. Verified via a
    # direct minimal repro by robomme-policy-learning-71, 2026-08-27. Fixing that side would mean
    # patching flax-internal behavior, not attempted here.
    if isinstance(tree, dict):
        keys = list(tree.keys())
        if keys and all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in keys):
            return {str(int(k)): _normalize_numeric_dict_keys(v) for k, v in tree.items()}
        return {k: _normalize_numeric_dict_keys(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_normalize_numeric_dict_keys(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_normalize_numeric_dict_keys(v) for v in tree)
    return tree


def check_pytree_equality(*, expected: PyTree, got: PyTree, check_shapes: bool = False, check_dtypes: bool = False):
    """Checks that two PyTrees have the same structure and optionally checks shapes and dtypes. Creates a much nicer
    error message than if `jax.tree.map` is naively used on PyTrees with different structures.
    """
    expected = _normalize_numeric_dict_keys(expected)
    got = _normalize_numeric_dict_keys(got)

    if errors := list(private_tree_util.equality_errors(expected, got)):
        raise ValueError(
            "PyTrees have different structure:\n"
            + (
                "\n".join(
                    f"   - at keypath '{jax.tree_util.keystr(path)}': expected {thing1}, got {thing2}, so {explanation}.\n"
                    for path, thing1, thing2, explanation in errors
                )
            )
        )

    if check_shapes or check_dtypes:

        def check(kp, x, y):
            if check_shapes and x.shape != y.shape:
                raise ValueError(f"Shape mismatch at {jax.tree_util.keystr(kp)}: expected {x.shape}, got {y.shape}")

            if check_dtypes and x.dtype != y.dtype:
                raise ValueError(f"Dtype mismatch at {jax.tree_util.keystr(kp)}: expected {x.dtype}, got {y.dtype}")

        jax.tree_util.tree_map_with_path(check, expected, got)
