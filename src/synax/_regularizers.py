from typing import Any, Callable, Sequence

from jax import Array
from jax import numpy as jnp

Regularizer = Callable[[Array], Array | float]


def zero(x: Any) -> float:
    return 0.0


def l1_norm(x: Array) -> Array:
    return jnp.abs(x).sum()


def l2_norm(x: Array, squared: bool = False) -> Array:
    x = (x * jnp.conj(x)).sum()
    if squared:
        return x
    return jnp.sqrt(x)


def linf_norm(x: Array) -> Array:
    return jnp.abs(x).max()


def lp_norm(x: Array, p: Array) -> Array:
    return (jnp.abs(x) ** p).sum() ** (1 / p)


def scale(regularizer: Regularizer, scale: float) -> Regularizer:
    def f(x: Any) -> Array | float:
        return regularizer(x) * scale

    return f


def add(regularizers: Sequence[Regularizer]) -> Regularizer:
    def f(x):
        return sum(regularizer(x) for regularizer in regularizers)

    return f
