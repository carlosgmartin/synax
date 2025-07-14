from typing import Any, Callable, Literal, Sequence

from jax import Array, lax, nn
from jax import numpy as jnp

from ._regularizers import Regularizer, zero

Key = Array
Initializer = Callable[[Key, tuple[int, ...]], Array]


class Bias:
    r"""
    Bias (translation).

    Computes

    .. math::
        y = x + b

    where :math:`b` is a learned vector.

    :param dimension: Input dimension.
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    """

    def __init__(
        self,
        dimension: int,
        initializer: Initializer = nn.initializers.zeros,
        regularizer: Regularizer = zero,
    ):
        self.dimension = dimension
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        """
        Initialize parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.initializer(key, (self.dimension,))

    def apply(self, parameters: Array, input: Array) -> Array:
        """
        Apply parameters.

        :param parameters: Parameters.
        :param input: An array of shape ``(..., dimension)``.

        :returns: An array of shape ``(..., dimension)``.
        """
        return input + parameters

    def parameter_loss(self, parameters: Array) -> Array | float:
        """
        Parameter loss.

        :param parameters: Parameters.

        :returns: A scalar.
        """
        return self.regularizer(parameters)


class Scale:
    r"""
    Elementwise scaling.

    Computes

    .. math::
        y = x \odot a

    where :math:`a` is a learned vector.

    :param dimension: Input dimension.
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    """

    def __init__(
        self,
        dimension: int,
        initializer: Initializer = nn.initializers.ones,
        regularizer: Regularizer = zero,
    ):
        self.dimension = dimension
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        """
        Initialize parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.initializer(key, (self.dimension,))

    def apply(self, parameters: Array, input: Array) -> Array:
        """
        Apply parameters.

        :param parameters: Parameters.
        :param input: An array of shape ``(..., dimension)``.

        :returns: An array of shape ``(..., dimension)``.
        """
        return input * parameters

    def parameter_loss(self, parameters: Array) -> Array | float:
        """
        Parameter loss.

        :param parameters: Parameters.

        :returns: A scalar.
        """
        return self.regularizer(parameters)


class Linear:
    r"""
    Linear transformation.

    Does not include bias.

    Computes

    .. math::
        y = A x

    where :math:`A` is a learned matrix.

    :param input_dimension: Input dimension.
    :param output_dimension: Output dimension.
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        initializer: Initializer = nn.initializers.he_normal(),
        regularizer: Regularizer = zero,
    ):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        """
        Initialize parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.initializer(key, (self.input_dimension, self.output_dimension))

    def apply(self, parameters: Array, input: Array) -> Array:
        """
        Apply parameters.

        :param parameters: Parameters.
        :param input: An array of shape ``(..., input_dimension)``.

        :returns: An array of shape ``(..., output_dimension)``.
        """
        return input @ parameters

    def parameter_loss(self, parameters: Array) -> Array | float:
        """
        Parameter loss.

        :param parameters: Parameters.

        :returns: A scalar.
        """
        return self.regularizer(parameters)


class Func:
    r"""
    Function application.

    Computes

    .. math::
        y = f(x)

    where :math:`f` is a user-specified function.

    :param function: Function to apply.
    """

    def __init__(self, function: Callable):
        self.function = function

    def init(self, key: Key) -> None:
        """
        Initialize parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return None

    def apply(self, parameters: None, input: Any) -> Any:
        """
        Apply parameters.

        :param parameters: Parameters.
        :param input: An input.

        :returns: The output.
        """
        return self.function(input)


class Conv:
    """
    Convolution.

    Does not include bias.

    :param input_dimension: Input dimension.
    :param output_dimension: Output dimension.
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
        | Sequence[tuple[int, int]] = "VALID",
        dilation: tuple[int, ...] | None = None,
        initializer: Initializer = nn.initializers.he_normal(),
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

    def init(self, key: Key) -> Array:
        initializer = self.initializer or nn.initializers.he_normal(
            range(-len(self.shape), 0)
        )
        return initializer(
            key, (self.output_dimension, self.input_dimension, *self.shape)
        )

    def apply(self, parameters: Array, input: Array) -> Array:
        num_spatial_axes = len(self.shape)
        x = input
        x = jnp.moveaxis(x, -1, -num_spatial_axes - 1)
        x = x[None]
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=parameters,
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
        initializer: Initializer = nn.initializers.normal(),
        regularizer: Regularizer = zero,
    ):
        self.number = number
        self.dimension = dimension
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        """
        Initialize parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.initializer(key, (self.number, self.dimension))

    def apply(self, parameters: Array, input: Array) -> Array:
        """
        Apply parameters.

        :param parameters: Parameters.
        :param input: An array of shape ``(...,)``.

        :returns: An array of shape ``(..., dimension)``.
        """
        return parameters[input]

    def parameter_loss(self, parameters: Array) -> Array | float:
        """
        Parameter loss.

        :param parameters: Parameters.

        :returns: A scalar.
        """
        return self.regularizer(parameters)
