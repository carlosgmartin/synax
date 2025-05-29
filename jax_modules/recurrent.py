from jax import nn, random

from .module import Module


class SimpleRecurrentUnit(Module):
    def __init__(
        self,
        state_dim,
        input_dim,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros,
        recurrent_init=nn.initializers.orthogonal(),
        activation=nn.relu,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.recurrent_init = recurrent_init
        self.activation = activation

    def init(self, key):
        keys = random.split(key, 3)
        w = self.kernel_init(keys[0], (self.input_dim, self.state_dim))
        u = self.recurrent_init(keys[1], (self.state_dim, self.state_dim))
        b = self.bias_init(keys[2], (self.state_dim,))
        return w, u, b

    def apply(self, param, state, input):
        w, u, b = param
        y = param @ w + state @ u + b
        return nn.tanh(y)


class GatedRecurrentUnit(Module):
    """Learning phrase representations using RNN encoder-decoder for statistical machine
        translation (2014)
    https://arxiv.org/abs/1406.1078"""

    def __init__(
        self,
        state_dim,
        input_dim,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros,
        recurrent_init=nn.initializers.orthogonal(),
        reset_activation=nn.sigmoid,
        update_activation=nn.sigmoid,
        candidate_activation=nn.tanh,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.recurrent_init = recurrent_init

        self.reset_activation = reset_activation
        self.update_activation = update_activation
        self.candidate_activation = candidate_activation

    def init(self, key):
        keys = random.split(key, 9)

        wz = self.kernel_init(keys[0], [self.input_dim, self.state_dim])
        wr = self.kernel_init(keys[1], [self.input_dim, self.state_dim])
        wy = self.kernel_init(keys[2], [self.input_dim, self.state_dim])

        uz = self.recurrent_init(keys[3], [self.state_dim, self.state_dim])
        ur = self.recurrent_init(keys[4], [self.state_dim, self.state_dim])
        uy = self.recurrent_init(keys[5], [self.state_dim, self.state_dim])

        bz = self.bias_init(keys[6], [self.state_dim])
        br = self.bias_init(keys[7], [self.state_dim])
        by = self.bias_init(keys[8], [self.state_dim])

        return bz, br, by, wz, wr, wy, uz, ur, uy

    def apply(self, param, state, input):
        bz, br, by, wz, wr, wy, uz, ur, uy = param
        z = self.update_activation(input @ wz + state @ uz + bz)
        r = self.reset_activation(input @ wr + state @ ur + br)
        y = self.candidate_activation(input @ wy + (r * state) @ uy + by)
        return (1 - z) * state + z * y


class MinimalGatedUnit(Module):
    """Minimal gated unit for recurrent neural networks (2016)
    https://arxiv.org/abs/1603.09420"""

    def __init__(
        self,
        state_dim,
        input_dim,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros,
        recurrent_init=nn.initializers.orthogonal(),
        update_activation=nn.sigmoid,
        candidate_activation=nn.tanh,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.recurrent_init = recurrent_init

        self.update_activation = update_activation
        self.candidate_activation = candidate_activation

    def init(self, key):
        keys = random.split(key, 6)

        wz = self.kernel_init(keys[0], [self.input_dim, self.state_dim])
        wy = self.kernel_init(keys[1], [self.input_dim, self.state_dim])

        uz = self.recurrent_init(keys[2], [self.state_dim, self.state_dim])
        uy = self.recurrent_init(keys[3], [self.state_dim, self.state_dim])

        bz = self.bias_init(keys[4], [self.state_dim])
        by = self.bias_init(keys[5], [self.state_dim])

        return bz, by, wz, wy, uz, uy

    def apply(self, param, state, input):
        bz, by, wz, wy, uz, uy = param
        z = self.update_activation(input @ wz + state @ uz + bz)
        y = self.candidate_activation(input @ wy + state @ uy + by)
        return (1 - z) * state + z * y
