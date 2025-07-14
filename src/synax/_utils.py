from math import prod
from typing import Callable, Literal, Sequence

from jax import Array, lax, random
from jax import numpy as jnp

Key = Array
Axis = int | Sequence[int] | None
Padding = Literal["VALID", "SAME", "SAME_BELOW"] | Sequence[tuple[int, int]]


def layer_norm(axis: Axis = -1, epsilon: float = 1e-6) -> Callable[[Array], Array]:
    r"""Layer normalization.

    Computes

    .. math::
        y = \frac{x - \mu}{\sigma}

    where

    .. math::
        \mu = \frac{1}{n} \sum_i x_i

    is the mean of the elements of :math:`x` and

    .. math::
        \sigma = \sqrt{ \varepsilon + \frac{1}{n} \sum_i (x_i - \mu)^2 }

    is the standard deviation of the elements of :math:`x`.
    :math:`\varepsilon` is a small quantity used for numerical stability.

    :param axis: Axis or axes along which to normalize.
        ``None`` means all axes.
    :param epsilon: Small quantity used for numerical stability.

    :returns: Function that maps an array to an array.

    References:

    - *Layer normalization*. 2016. https://arxiv.org/abs/1607.06450.
    """

    def f(x: Array) -> Array:
        x -= x.mean(axis, keepdims=True)
        rms = jnp.sqrt((x * jnp.conj(x)).mean(axis, keepdims=True) + epsilon)
        return x / rms

    return f


def rms_norm(axis: Axis = -1, epsilon: float = 1e-6) -> Callable[[Array], Array]:
    r"""
    Root mean square layer normalization.

    Computes

    .. math::
        y = \frac{x}{r}

    where

    .. math::
        r = \sqrt{ \varepsilon + \frac{1}{n} \sum_i x_i^2 }

    is the root mean square (RMS) of the elements of :math:`x`.
    :math:`\varepsilon` is a small quantity used for numerical stability.

    :param axis: Axis or axes along which to normalize.
        ``None`` means all axes.
    :param epsilon: Small quantity used for numerical stability.

    :returns: Function that maps an array to an array.

    References:

    - *Root mean square layer normalization*. 2019.
      https://arxiv.org/abs/1910.07467.
    """

    def f(x: Array) -> Array:
        rms = jnp.sqrt((x * jnp.conj(x)).mean(axis, keepdims=True) + epsilon)
        return x / rms

    return f


def pool(
    operator,
    identity,
    shape: Sequence[int],
    *,
    stride: int | Sequence[int] = 1,
    padding: Padding = "VALID",
    dilation: int | Sequence[int] = 1,
    base_dilation: int | Sequence[int] = 1,
) -> Callable[[Array], Array]:
    if isinstance(stride, int):
        stride = (stride,) * len(shape)

    if isinstance(base_dilation, int):
        base_dilation = (base_dilation,) * len(shape)

    if isinstance(dilation, int):
        dilation = (dilation,) * len(shape)

    def f(x: Array) -> Array:
        return lax.reduce_window(
            operand=x,
            init_value=identity,
            computation=operator,
            window_dimensions=(*shape, 1),
            window_strides=(*stride, 1),
            padding=padding,
            window_dilation=(*dilation, 1),
            base_dilation=(*base_dilation, 1),
        )

    return f


def max_pool(
    shape: Sequence[int],
    *,
    stride: int | Sequence[int] = 1,
    padding: Padding = "VALID",
    dilation: int | Sequence[int] = 1,
    base_dilation: int | Sequence[int] = 1,
) -> Callable[[Array], Array]:
    """
    Max pooling.

    :param shape: Window shape.
    :param stride: Window stride.
    :param padding: Padding. Can be "SAME", "SAME_LOWER", "VALID", or a sequence
        of int pairs giving the padding before and after each spatial dimension.
    :param dilation: Window dilation.
    :param base_dilation: Base dilation.

    :returns: Function that maps an array to an array.
    """
    return pool(
        operator=lax.max,
        identity=-jnp.inf,
        shape=shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        base_dilation=base_dilation,
    )


def mean_pool(
    shape: Sequence[int],
    *,
    stride: int | Sequence[int] = 1,
    padding: Padding = "VALID",
    dilation: int | Sequence[int] = 1,
    base_dilation: int | Sequence[int] = 1,
) -> Callable[[Array], Array]:
    """
    Mean pooling.

    :param shape: Window shape.
    :param stride: Window stride.
    :param padding: Padding. Can be "SAME", "SAME_LOWER", "VALID", or a sequence
        of int pairs giving the padding before and after each spatial dimension.
    :param dilation: Window dilation.
    :param base_dilation: Base dilation.

    :returns: Function that maps an array to an array.
    """
    size = prod(shape)

    g = pool(
        operator=lax.add,
        identity=0,
        shape=shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        base_dilation=base_dilation,
    )

    def f(x: Array) -> Array:
        return g(x) / size

    return f


def dropout(prob: float) -> Callable[[Array, Key], Array]:
    """
    Dropout.

    :param prob: Dropout probability.

    :returns: Function that maps an array to an array.

    References:

    - *Improving neural networks by preventing co-adaptation of feature
      detectors*. 2012. https://arxiv.org/abs/1207.0580.
    """

    def f(x: Array, key: Key) -> Array:
        mask = random.bernoulli(key, prob, x.shape)
        return jnp.where(mask, x / prob, 0)

    return f
