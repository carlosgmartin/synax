import math
from typing import Any, Callable, Sequence

from jax import Array, lax, nn
from jax import numpy as jnp
from jax.nn.initializers import Initializer


from ._regularizers import Regularizer, zero
from ._utils import Padding

Key = Array


class Bias:
    r"""
    Bias (translation).

    Computes

    .. math::
        y = x + b

    where :math:`b` is a learned vector.

    :param dim: Input dimension.
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    """

    def __init__(
        self,
        dim: int,
        initializer: Initializer = nn.initializers.zeros,
        regularizer: Regularizer = zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init_params(self, key: Key) -> Array:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.initializer(key, (self.dim,))

    def apply(self, params: Array, input: Array) -> Array:
        """
        Apply module.

        :param params: Parameters.
        :param input: Array of shape ``(..., dim)``.

        :returns: Array of shape ``(..., dim)``.
        """
        return input + params

    def param_loss(self, params: Array) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return self.regularizer(params)


class Scale:
    r"""
    Elementwise scaling.

    Computes

    .. math::
        y = x \odot a

    where :math:`a` is a learned vector.

    :param dim: Input dimension.
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    """

    def __init__(
        self,
        dim: int,
        initializer: Initializer = nn.initializers.ones,
        regularizer: Regularizer = zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init_params(self, key: Key) -> Array:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.initializer(key, (self.dim,))

    def apply(self, params: Array, input: Array) -> Array:
        """
        Apply module.

        :param params: Parameters.
        :param input: Array of shape ``(..., dim)``.

        :returns: Array of shape ``(..., dim)``.
        """
        return input * params

    def param_loss(self, params: Array) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return self.regularizer(params)


class Linear:
    r"""
    Linear map.

    Does not include bias.

    Computes

    .. math::
        y = A x

    where :math:`A` is a learned matrix.

    :param input_dim: Input dimension.
    :param output_dim: Output dimension.
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initializer: Initializer = nn.initializers.he_normal(),
        regularizer: Regularizer = zero,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init_params(self, key: Key) -> Array:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.initializer(key, (self.input_dim, self.output_dim))

    def apply(self, params: Array, input: Array) -> Array:
        """
        Apply module.

        :param params: Parameters.
        :param input: Array of shape ``(..., input_dim)``.

        :returns: Array of shape ``(..., output_dim)``.
        """
        return input @ params

    def param_loss(self, params: Array) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return self.regularizer(params)


class Func:
    r"""
    Function application.

    Computes

    .. math::
        y = f(x)

    where :math:`f` is a user-specified function.

    :param function: Function to apply.
    """

    def __init__(self, function: Callable[[Any], Any]):
        self.function = function

    def init_params(self, key: Key) -> None:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return None

    def apply(self, params: None, input: Any) -> Any:
        """
        Apply module.

        :param params: Parameters.
        :param input: Input.

        :returns: The output.
        """
        return self.function(input)

    def param_loss(self, params: None) -> float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return 0.0


class Conv:
    """
    Convolution.

    Does not include bias.

    :param input_dim: Input dimension.
    :param output_dim: Output dimension.
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
    :param regularizer: Regularizer for the convolution kernel.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        shape: Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: Padding = "VALID",
        dilation: int | Sequence[int] = 1,
        base_dilation: int | Sequence[int] = 1,
        initializer: Initializer = nn.initializers.he_normal(),
        groups: int = 1,
        regularizer: Regularizer = zero,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shape = shape
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.base_dilation = base_dilation
        self.initializer = initializer
        self.groups = groups
        self.regularizer = regularizer

    def init_params(self, key: Key) -> Array:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        kernel = self.initializer(
            key, (self.output_dim, self.input_dim * math.prod(self.shape))
        )
        kernel = kernel.reshape((self.output_dim, self.input_dim, *self.shape))
        return kernel

    def apply(self, params: Array, input: Array) -> Array:
        """
        Apply module.

        :param params: Parameters.
        :param input: Array of shape ``(..., input_dim)``.

        :returns: Array of shape ``(..., output_dim)``.
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
            rhs=params,
            window_strides=stride,
            padding=self.padding,
            rhs_dilation=dilation,
            lhs_dilation=base_dilation,
            feature_group_count=self.groups,
        )
        x = x.squeeze(0)
        x = jnp.moveaxis(x, -num_spatial_axes - 1, -1)
        return x

    def param_loss(self, params: Array) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return self.regularizer(params)


class Embed:
    """
    Embedding.

    :param number: Number of embeddings.
    :param dim: Dimension of each embedding.
    :param initializer: Initializer for embeddings.
    :param regularizer: Regularizer.
    """

    def __init__(
        self,
        number: int,
        dim: int,
        initializer: Initializer = nn.initializers.normal(),
        regularizer: Regularizer = zero,
    ):
        self.number = number
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init_params(self, key: Key) -> Array:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.initializer(key, (self.number, self.dim))

    def apply(self, params: Array, input: Array) -> Array:
        """
        Apply module.

        :param params: Parameters.
        :param input: Array of shape ``(...)``.

        :returns: Array of shape ``(..., dim)``.
        """
        return params[input]

    def param_loss(self, params: Array) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return self.regularizer(params)
