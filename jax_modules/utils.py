from math import prod

from jax import lax, random
from jax import numpy as jnp


def layer_norm(x, axis=-1, epsilon=1e-6):
    """Layer normalization (2016)
    https://arxiv.org/abs/1607.06450"""
    x -= x.mean(axis, keepdims=True)
    x /= jnp.sqrt((x * jnp.conj(x)).mean(axis, keepdims=True) + epsilon)
    return x


def rms_norm(x, axis=-1, epsilon=1e-6):
    """Root mean square layer normalization (2019)
    https://arxiv.org/abs/1910.07467"""
    rms = jnp.sqrt((x * jnp.conj(x)).mean(axis, keepdims=True) + epsilon)
    return x / rms


def pool(operator, initial_value, shape, strides=None, padding=None):
    if padding is None:
        padding = "VALID"

    if strides is None:
        strides = [1] * len(shape)
    else:
        strides = list(strides)

    def f(x):
        return lax.reduce_window(
            x,
            initial_value,
            operator,
            list(shape) + [1],
            strides + [1],
            padding,
        )

    return f


def max_pool(*args, **kwargs):
    return pool(lax.max, -jnp.inf, *args, **kwargs)


def sum_pool(*args, **kwargs):
    return pool(lax.add, 0, *args, **kwargs)


def avg_pool(*args, **kwargs):
    if args:
        size = prod(args[0])
    else:
        size = prod(kwargs["shape"])
    return lambda x: sum_pool(*args, **kwargs)(x) / size


def dropout(prob):
    """Improving neural networks by preventing co-adaptation of feature detectors
    https://arxiv.org/abs/1207.0580"""

    def f(x, key):
        mask = random.bernoulli(key, prob, x.shape)
        return jnp.where(mask, x / prob, 0)

    return f
