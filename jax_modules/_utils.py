from math import prod

from jax import lax, random
from jax import numpy as jnp


def layer_norm(x, axis=-1, epsilon=1e-6):
    """Layer normalization (2016)
    https://arxiv.org/abs/1607.06450"""
    x -= x.mean(axis, keepdims=True)
    rms = jnp.sqrt((x * jnp.conj(x)).mean(axis, keepdims=True) + epsilon)
    return x / rms


def rms_norm(x, axis=-1, epsilon=1e-6):
    """Root mean square layer normalization (2019)
    https://arxiv.org/abs/1910.07467"""
    rms = jnp.sqrt((x * jnp.conj(x)).mean(axis, keepdims=True) + epsilon)
    return x / rms


def pool(operator, identity, *, shape, strides=None, padding=None):
    if padding is None:
        padding = "VALID"

    if strides is None:
        strides = [1] * len(shape)
    else:
        strides = list(strides)

    def f(x):
        return lax.reduce_window(
            operand=x,
            init_value=identity,
            computation=operator,
            window_dimensions=list(shape) + [1],
            window_strides=strides + [1],
            padding=padding,
        )

    return f


def max_pool(shape, *, strides=None, padding=None):
    return pool(
        operator=lax.max,
        identity=-jnp.inf,
        shape=shape,
        strides=strides,
        padding=padding,
    )


def sum_pool(shape, *, strides=None, padding=None):
    return pool(
        operator=lax.add,
        identity=0,
        shape=shape,
        strides=strides,
        padding=padding,
    )


def mean_pool(shape, *, strides=None, padding=None):
    size = prod(shape)

    def f(x):
        return sum_pool(shape, strides=strides, padding=padding)(x) / size

    return f


def dropout(prob):
    """Improving neural networks by preventing co-adaptation of feature detectors
    https://arxiv.org/abs/1207.0580"""

    def f(x, key):
        mask = random.bernoulli(key, prob, x.shape)
        return jnp.where(mask, x / prob, 0)

    return f
