from jax import lax, nn
from jax import numpy as jnp

from ._regularizers import zero


class Bias:
    r"""
    Bias (translation).

    Computes

    .. math::
        y = x + b

    where :math:`b` is a learned vector.

    :param dimension: Dimension of the input.
    :type dimension: int
    :param initializer: Initializer.
    :type initializer: jax.nn.initializers.Initializer
    :param regularizer: Regularizer.
    :type regularizer: typing.Callable
    """

    def __init__(
        self,
        dimension,
        initializer=nn.initializers.zeros,
        regularizer=zero,
    ):
        self.dimension = dimension
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        return self.initializer(key, (self.dimension,))

    def apply(self, param, input):
        return input + param

    def parameter_loss(self, param):
        return self.regularizer(param)


class Scale:
    r"""
    Elementwise scaling.

    Computes

    .. math::
        y = x \odot a

    where :math:`a` is a learned vector.

    :param dimension: Dimension of the input.
    :type dimension: int
    :param initializer: Initializer.
    :type initializer: jax.nn.initializers.Initializer
    :param regularizer: Regularizer.
    :type regularizer: typing.Callable
    """

    def __init__(
        self,
        dimension,
        initializer=nn.initializers.ones,
        regularizer=zero,
    ):
        self.dimension = dimension
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        return self.initializer(key, (self.dimension,))

    def apply(self, param, input):
        return input * param

    def parameter_loss(self, param):
        return self.regularizer(param)


class Linear:
    r"""
    Linear transformation.

    Does not include bias.

    Computes

    .. math::
        y = A x

    where :math:`A` is a learned matrix.

    :param input_dimension: Dimension of the input.
    :type input_dimension: int
    :param output_dimension: Dimension of the output.
    :type output_dimension: int
    :param initializer: Initializer.
    :type initializer: jax.nn.initializers.Initializer
    :param regularizer: Regularizer.
    :type regularizer: typing.Callable
    """

    def __init__(
        self,
        input_dimension,
        output_dimension,
        initializer=nn.initializers.he_normal(),
        regularizer=zero,
    ):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        return self.initializer(key, (self.input_dimension, self.output_dimension))

    def apply(self, param, input):
        return input @ param

    def parameter_loss(self, param):
        return self.regularizer(param)


class Func:
    r"""
    Function.

    Computes

    .. math::
        y = f(x)

    where :math:`f` is a user-specified function.

    :param function: Function to apply to input.
    :type function: typing.Callable
    """

    def __init__(self, function):
        self.function = function

    def init(self, key):
        return None

    def apply(self, param, input):
        return self.function(input)


class Conv:
    """
    Convolution.

    :param input_dimension: Dimension of the input.
    :type input_dimension: int
    :param output_dimension: Dimension of the output.
    :type output_dimension: int
    :param shape: Window size for each spatial dimension.
    :type shape: tuple[int, ...]
    :param strides: Stride for each spatial dimension.
    :type strides: tuple[int, ...] | None
    :param padding: Padding. Can be "SAME", "SAME_LOWER", "VALID", or a sequence
        of int pairs giving the padding before and after each spatial dimension.
    :type padding: str | Sequence[tuple[int, int]]
    :param dilation: Dilation factor for each spatial dimension.
    :type dilation: tuple[int, ...] | None
    :param initializer: Initializer for the convolution kernel.
    :type initializer: jax.nn.initializers.Initializer
    :param groups: Number of groups to split the input channels into.
    :type groups: int
    """

    def __init__(
        self,
        input_dimension,
        output_dimension,
        shape,
        strides=None,
        padding="VALID",
        dilation=None,
        initializer=None,
        groups=1,
    ):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.shape = shape
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.initializer = initializer
        self.groups = groups

    def init(self, key):
        initializer = self.initializer or nn.initializers.he_normal(
            range(-len(self.shape), 0)
        )
        return initializer(
            key, (self.output_dimension, self.input_dimension, *self.shape)
        )

    def apply(self, w, x):
        num_spatial_axes = len(self.shape)
        x = jnp.moveaxis(x, -1, -num_spatial_axes - 1)
        x = x[None]
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=w,
            window_strides=self.strides or [1] * num_spatial_axes,
            padding=self.padding,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = x[0]
        x = jnp.moveaxis(x, -num_spatial_axes - 1, -1)
        return x


class Embed:
    """
    Embedding.

    :param number: Number of embeddings.
    :type number: int
    :param dimension: Dimension of each embedding.
    :type dimension: int
    :initializer: Initializer for embeddings.
    :type initializer: jax.nn.initializers.Initializer
    """

    def __init__(
        self,
        number,
        dimension,
        initializer=nn.initializers.normal(),
    ):
        self.number = number
        self.dimension = dimension
        self.initializer = initializer

    def init(self, key):
        return self.initializer(key, (self.number, self.dimension))

    def apply(self, param, input):
        return param[input]
