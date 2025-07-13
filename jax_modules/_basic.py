from jax import lax, nn
from jax import numpy as jnp

from ._regularizers import zero


class Bias:
    def __init__(
        self,
        dim,
        initializer=nn.initializers.zeros,
        regularizer=zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        return self.initializer(key, (self.dim,))

    def apply(self, param, input):
        return input + param

    def parameter_loss(self, param):
        return self.regularizer(param)


class Scale:
    def __init__(self, dim, initializer=nn.initializers.ones, regularizer=zero):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        return self.initializer(key, (self.dim,))

    def apply(self, param, input):
        return input * param

    def parameter_loss(self, param):
        return self.regularizer(param)


class Dense:
    def __init__(
        self,
        input_dim,
        output_dim,
        initializer=nn.initializers.he_normal(),
        regularizer=zero,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        return self.initializer(key, (self.input_dim, self.output_dim))

    def apply(self, param, input):
        return input @ param

    def parameter_loss(self, param):
        return self.regularizer(param)


class Func:
    def __init__(self, function):
        self.function = function

    def init(self, key):
        return None

    def apply(self, param, input):
        return self.function(input)


class Conv:
    def __init__(
        self,
        input_dim,
        output_dim,
        window_shape,
        strides=None,
        padding="VALID",
        dilation=None,
        initializer=None,
        groups=1,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_shape = window_shape
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.initializer = initializer
        self.groups = groups

    def init(self, key):
        initializer = self.initializer or nn.initializers.he_normal(
            range(-len(self.window_shape), 0)
        )
        return initializer(key, (self.output_dim, self.input_dim, *self.window_shape))

    def apply(self, w, x):
        num_spatial_axes = len(self.window_shape)
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
    def __init__(self, num, dim, initializer=nn.initializers.normal()):
        self.num = num
        self.dim = dim
        self.initializer = initializer

    def init(self, key):
        return self.initializer(key, (self.num, self.dim))

    def apply(self, param, input):
        return param[input]
