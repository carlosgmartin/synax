from typing import Any, Callable, Literal

from jax import Array, lax, nn
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
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    """

    def __init__(
        self,
        dimension: int,
        initializer: Callable = nn.initializers.zeros,
        regularizer: Callable = zero,
    ):
        self.dimension = dimension
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Array) -> Array:
        """
        Sample initial parameters.

        :param key: A PRNG key.

        :returns: Module parameters.
        """
        return self.initializer(key, (self.dimension,))

    def apply(self, param: Array, input: Array) -> Array:
        """
        Apply the module.

        :param param: Module parameters.
        :param input: An array of shape ``(..., dimension)``.

        :returns: An array of shape ``(..., dimension)``.
        """
        return input + param

    def parameter_loss(self, param: Array) -> Array:
        """
        Parameter loss.

        :param param: Module parameters.

        :returns: A scalar.
        """
        return self.regularizer(param)


class Scale:
    r"""
    Elementwise scaling.

    Computes

    .. math::
        y = x \odot a

    where :math:`a` is a learned vector.

    :param dimension: Dimension of the input.
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    """

    def __init__(
        self,
        dimension: int,
        initializer: Callable = nn.initializers.ones,
        regularizer: Callable = zero,
    ):
        self.dimension = dimension
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Array) -> Array:
        """
        Sample initial parameters.

        :param key: A PRNG key.

        :returns: Module parameters.
        """
        return self.initializer(key, (self.dimension,))

    def apply(self, param: Array, input: Array) -> Array:
        """
        Apply the module.

        :param param: Module parameters.
        :param input: An array of shape ``(..., dimension)``.

        :returns: An array of shape ``(..., dimension)``.
        """
        return input * param

    def parameter_loss(self, param: Array) -> Array:
        """
        Parameter loss.

        :param param: Module parameters.

        :returns: A scalar.
        """
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
    :param output_dimension: Dimension of the output.
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        initializer: Callable = nn.initializers.he_normal(),
        regularizer: Callable = zero,
    ):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Array) -> Array:
        """
        Sample initial parameters.

        :param key: A PRNG key.

        :returns: Module parameters.
        """
        return self.initializer(key, (self.input_dimension, self.output_dimension))

    def apply(self, param: Array, input: Array) -> Array:
        """
        Apply the module.

        :param param: Module parameters.
        :param input: An array of shape ``(..., input_dimension)``.

        :returns: An array of shape ``(..., output_dimension)``.
        """
        return input @ param

    def parameter_loss(self, param: Array) -> Array:
        """
        Parameter loss.

        :param param: Module parameters.

        :returns: A scalar.
        """
        return self.regularizer(param)


class Func:
    r"""
    Function application.

    Computes

    .. math::
        y = f(x)

    where :math:`f` is a user-specified function.

    :param function: Function to apply to input.
    """

    def __init__(self, function: Callable):
        self.function = function

    def init(self, key: Array) -> None:
        """
        Sample initial parameters.

        :param key: A PRNG key.

        :returns: Module parameters.
        """
        return None

    def apply(self, param: None, input: Any) -> Any:
        """
        Apply the module.

        :param param: Module parameters.
        :param input: An input.

        :returns: The output.
        """
        return self.function(input)


class Conv:
    """
    Convolution.

    Does not include bias.

    :param input_dimension: Dimension of the input.
    :param output_dimension: Dimension of the output.
    :param shape: Window size for each spatial dimension.
    :param strides: Stride for each spatial dimension.
    :param padding: Padding. Can be "VALID", "SAME", "SAME_LOWER", or a sequence
        of int pairs giving the padding before and after each spatial dimension.
        "VALID" applies no padding.
        "SAME" and "SAME_LOWER" preserve the spatial shape of the input,
        splitting the padding equally or almost equally before and after each
        spatial dimension.
        When the padding is an odd number, "SAME" adds the extra padding at the
        end, while "SAME_LOWER" adds the extra padding at the beginning.
    :param dilation: Dilation factor for each spatial dimension.
    :param initializer: Initializer for the convolution kernel.
    :param groups: Number of groups to split the input channels into.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        shape: tuple[int, ...],
        strides: tuple[int, ...] | None = None,
        padding: Literal["VALID", "SAME", "SAME_LOWER"]
        | tuple[tuple[int, int]] = "VALID",
        dilation: tuple[int, ...] | None = None,
        initializer: Callable = nn.initializers.he_normal(),
        groups: int = 1,
    ):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.shape = shape
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.initializer = initializer
        self.groups = groups

    def init(self, key: Array) -> Array:
        initializer = self.initializer or nn.initializers.he_normal(
            range(-len(self.shape), 0)
        )
        return initializer(
            key, (self.output_dimension, self.input_dimension, *self.shape)
        )

    def apply(self, param: Array, x: Array) -> Array:
        num_spatial_axes = len(self.shape)
        x = jnp.moveaxis(x, -1, -num_spatial_axes - 1)
        x = x[None]
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=param,
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
    :param dimension: Dimension of each embedding.
    :param initializer: Initializer for embeddings.
    :param regularizer: Regularizer.
    """

    def __init__(
        self,
        number: int,
        dimension: int,
        initializer: Callable = nn.initializers.normal(),
        regularizer: Callable = zero,
    ):
        self.number = number
        self.dimension = dimension
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Array) -> Array:
        """
        Sample initial parameters.

        :param key: A PRNG key.

        :returns: Module parameters.
        """
        return self.initializer(key, (self.number, self.dimension))

    def apply(self, param: Array, input: Array) -> Array:
        """
        Apply the module.

        :param param: Module parameters.
        :param input: An array of shape ``(...,)``.

        :returns: An array of shape ``(..., dimension)``.
        """
        return param[input]

    def parameter_loss(self, param: Array) -> Array:
        """
        Parameter loss.

        :param param: Module parameters.

        :returns: A scalar.
        """
        return self.regularizer(param)
