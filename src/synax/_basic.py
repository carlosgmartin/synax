import math
from typing import Any, Callable, Sequence

from jax import Array, lax, nn
from jax import numpy as jnp

from ._regularizers import Regularizer, zero
from ._utils import Padding

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
        :param input: Array of shape ``(..., dimension)``.

        :returns: Array of shape ``(..., dimension)``.
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
        :param input: Array of shape ``(..., dimension)``.

        :returns: Array of shape ``(..., dimension)``.
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
        :param input: Array of shape ``(..., input_dimension)``.

        :returns: Array of shape ``(..., output_dimension)``.
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
        :param input: Input.

        :returns: The output.
        """
        return self.function(input)


class Conv:
    """
    Convolution.

    Does not include bias.

    :param input_dimension: Input dimension.
    :param output_dimension: Output dimension.
    :param shape: Window shape.
    :param stride: Window stride.
    :param padding: Padding. Can be "VALID", "SAME", "SAME_LOWER", or a sequence
        of int pairs giving the padding before and after each spatial dimension.
        "VALID" applies no padding.
        "SAME" and "SAME_LOWER" preserve the spatial shape of the input,
        splitting the padding equally or almost equally before and after each
        spatial dimension.
        When the padding is an odd number, "SAME" adds the extra padding at the
        end, while "SAME_LOWER" adds the extra padding at the beginning.
    :param dilation: Window dilation.
    :param base_dilation: Base dilation.
    :param initializer: Initializer for the convolution kernel.
    :param groups: Number of groups to split the input channels into.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        shape: Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: Padding = "VALID",
        dilation: int | Sequence[int] = 1,
        base_dilation: int | Sequence[int] = 1,
        initializer: Initializer = nn.initializers.he_normal(),
        groups: int = 1,
    ):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.shape = shape
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.base_dilation = base_dilation
        self.initializer = initializer
        self.groups = groups

    def init(self, key: Key) -> Array:
        """
        Initialize parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        kernel = self.initializer(
            key, (self.output_dimension, self.input_dimension * math.prod(self.shape))
        )
        kernel = kernel.reshape(
            (self.output_dimension, self.input_dimension, *self.shape)
        )
        return kernel

    def apply(self, parameters: Array, input: Array) -> Array:
        """
        Apply parameters.

        :param parameters: Parameters.
        :param input: Array of shape ``(..., input_dimension)``.

        :returns: Array of shape ``(..., output_dimension)``.
        """

        stride = self.stride
        if isinstance(stride, int):
            stride = (stride,) * len(self.shape)

        dilation = self.dilation
        if isinstance(dilation, int):
            dilation = (dilation,) * len(self.shape)

        base_dilation = self.base_dilation
        if isinstance(base_dilation, int):
            base_dilation = (base_dilation,) * len(self.shape)

        num_spatial_axes = len(self.shape)
        x = input
        x = jnp.moveaxis(x, -1, -num_spatial_axes - 1)
        x = x[None]
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=parameters,
            window_strides=stride,
            padding=self.padding,
            rhs_dilation=dilation,
            lhs_dilation=base_dilation,
            feature_group_count=self.groups,
        )
        x = x.squeeze(0)
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
        :param input: Array of shape ``(...)``.

        :returns: Array of shape ``(..., dimension)``.
        """
        return parameters[input]

    def parameter_loss(self, parameters: Array) -> Array | float:
        """
        Parameter loss.

        :param parameters: Parameters.

        :returns: A scalar.
        """
        return self.regularizer(parameters)
