from typing import Callable

from jax import Array
from jax import numpy as jnp

Regularizer = Callable[[Array], Array | float]


def zero(x):
    return 0.0


def l1(x):
    return jnp.abs(x).sum()


def l2(x, squared=False):
    x = (x * jnp.conj(x)).sum()
    if squared:
        return x
    return jnp.sqrt(x)


def linf(x):
    return jnp.abs(x).max()


def lp(x, p):
    return (jnp.abs(x) ** p).sum() ** (1 / p)


def scale(regularizer, scale):
    def f(x):
        return regularizer(x) * scale

    return f


def add(regularizers):
    def f(x):
        return sum(regularizer(x) for regularizer in regularizers)

    return f
