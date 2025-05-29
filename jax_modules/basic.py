from jax import nn

from .module import Module


class Bias(Module):
    def __init__(
        self,
        dim,
        initializer=nn.initializers.zeros,
        regularizer=lambda param: 0.0,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        return self.initializer(key, (self.dim,))

    def apply(self, param, input):
        return input + param

    def param_loss(self, param):
        return self.regularizer(param)


class Linear(Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        initializer=nn.initializers.he_normal(),
        regularizer=lambda param: 0.0,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        return self.initializer(key, (self.input_dim, self.output_dim))

    def apply(self, param, input):
        return input @ param

    def param_loss(self, param):
        return self.regularizer(param)


class Function(Module):
    def __init__(self, function):
        self.function = function

    def init(self, key):
        return None

    def apply(self, param, input):
        return self.function(input)


class Conv(Module):
    pass
