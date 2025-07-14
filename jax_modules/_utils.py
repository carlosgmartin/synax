from math import prod

from jax import lax, random
from jax import numpy as jnp


def layer_norm(x, axis=-1, epsilon=1e-6):
    r"""Layer normalization.

    Computes

    .. math::
        y = \frac{x - \mu}{\sigma}

    where

    .. math::
        \mu = \frac{1}{n} \sum_i x_i

    is the mean of the elements of :math:`x` and

    .. math::
        \mu = \sqrt{ \frac{1}{n} \sum_i (x_i - \mu)^2 }

    is the standard deviation of the elements of :math:`x`.

    References:

    - *Layer normalization*. 2016. https://arxiv.org/abs/1607.06450.
    """
    x -= x.mean(axis, keepdims=True)
    rms = jnp.sqrt((x * jnp.conj(x)).mean(axis, keepdims=True) + epsilon)
    return x / rms


def rms_norm(x, axis=-1, epsilon=1e-6):
    r"""
    Root mean square layer normalization.

    Computes

    .. math::
        y = \frac{x}{r}

    where

    .. math::
        r = \sqrt{ \frac{1}{n} \sum_i x_i^2 }

    is the root mean square (RMS) of the elements of :math:`x`.

    References:

    - *Root mean square layer normalization*. 2019.
      https://arxiv.org/abs/1910.07467.
    """
    rms = jnp.sqrt((x * jnp.conj(x)).mean(axis, keepdims=True) + epsilon)
    return x / rms


def pool(operator, identity, shape, *, strides=None, padding="VALID"):
    if strides is None:
        strides = (1,) * len(shape)

    def f(x):
        return lax.reduce_window(
            operand=x,
            init_value=identity,
            computation=operator,
            window_dimensions=shape + (1,),
            window_strides=strides + (1,),
            padding=padding,
        )

    return f


def max_pool(shape, *, strides=None, padding="VALID"):
    """
    Max pooling.

    :param shape: Window size for each spatial dimension.
    :type shape: tuple[int, ...]
    :param strides: Stride for each spatial dimension.
    :type strides: tuple[int, ...] | None
    :param padding: Padding. Can be "SAME", "SAME_LOWER", "VALID", or a sequence
        of int pairs giving the padding before and after each spatial dimension.
    :type padding: str | Sequence[tuple[int, int]]
    """
    return pool(
        operator=lax.max,
        identity=-float("inf"),
        shape=shape,
        strides=strides,
        padding=padding,
    )


def sum_pool(shape, *, strides=None, padding="VALID"):
    """
    Sum pooling.

    :param shape: Window size for each spatial dimension.
    :type shape: tuple[int, ...]
    :param strides: Stride for each spatial dimension.
    :type strides: tuple[int, ...] | None
    :param padding: Padding. Can be "SAME", "SAME_LOWER", "VALID", or a sequence
        of int pairs giving the padding before and after each spatial dimension.
    :type padding: str | Sequence[tuple[int, int]]
    """
    return pool(
        operator=lax.add,
        identity=0,
        shape=shape,
        strides=strides,
        padding=padding,
    )


def mean_pool(shape, *, strides=None, padding="VALID"):
    """
    Mean pooling.

    :param shape: Window size for each spatial dimension.
    :type shape: tuple[int, ...]
    :param strides: Stride for each spatial dimension.
    :type strides: tuple[int, ...] | None
    :param padding: Padding. Can be "SAME", "SAME_LOWER", "VALID", or a sequence
        of int pairs giving the padding before and after each spatial dimension.
    :type padding: str | Sequence[tuple[int, int]]
    """
    size = prod(shape)

    def f(x):
        return sum_pool(shape, strides=strides, padding=padding)(x) / size

    return f


def dropout(prob):
    """
    Dropout.

    References:

    - *Improving neural networks by preventing co-adaptation of feature
      detectors*. 2012. https://arxiv.org/abs/1207.0580.
    """

    def f(x, key):
        mask = random.bernoulli(key, prob, x.shape)
        return jnp.where(mask, x / prob, 0)

    return f
