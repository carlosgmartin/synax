from jax import lax, nn, random

from .module import Module


class RecurrentNetwork(Module):
    def __init__(self, unit):
        self.unit = unit

    def init(self, key, x):
        keys = random.split(key)
        w = self.unit.init(keys[0])
        h = self.unit.init_state(keys[1])
        return w, h

    def apply(self, param, xs):
        w, h = param

        def f(h, x):
            h_new = self.unit.apply(w, h, x)
            return h_new, h

        return lax.scan(f, h, xs)


class SimpleRecurrentUnit(Module):
    """Finding structure in time (1990)
    https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1"""

    def __init__(
        self,
        state_dim,
        input_dim,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros,
        recurrent_init=nn.initializers.orthogonal(),
        activation=nn.relu,
        state_init=nn.initializers.zeros,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.recurrent_init = recurrent_init
        self.activation = activation
        self.state_init = state_init

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

    def init_state(self, key):
        return self.state_init(key, (self.state_dim,))


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
        state_init=nn.initializers.zeros,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.recurrent_init = recurrent_init

        self.reset_activation = reset_activation
        self.update_activation = update_activation
        self.candidate_activation = candidate_activation

        self.state_init = state_init

    def init(self, key):
        keys = random.split(key, 9)

        wz = self.kernel_init(keys[0], (self.input_dim, self.state_dim))
        wr = self.kernel_init(keys[1], (self.input_dim, self.state_dim))
        wy = self.kernel_init(keys[2], (self.input_dim, self.state_dim))

        uz = self.recurrent_init(keys[3], (self.state_dim, self.state_dim))
        ur = self.recurrent_init(keys[4], (self.state_dim, self.state_dim))
        uy = self.recurrent_init(keys[5], (self.state_dim, self.state_dim))

        bz = self.bias_init(keys[6], (self.state_dim,))
        br = self.bias_init(keys[7], (self.state_dim,))
        by = self.bias_init(keys[8], (self.state_dim,))

        return bz, br, by, wz, wr, wy, uz, ur, uy

    def apply(self, param, state, input):
        bz, br, by, wz, wr, wy, uz, ur, uy = param
        z = self.update_activation(input @ wz + state @ uz + bz)
        r = self.reset_activation(input @ wr + state @ ur + br)
        y = self.candidate_activation(input @ wy + (r * state) @ uy + by)
        return (1 - z) * state + z * y

    def init_state(self, key):
        return self.state_init(key, (self.state_dim,))


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
        state_init=nn.initializers.zeros,
        reset_gate=True,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.recurrent_init = recurrent_init

        self.update_activation = update_activation
        self.candidate_activation = candidate_activation

        self.state_init = state_init
        self.reset_gate = reset_gate

    def init(self, key):
        keys = random.split(key, 6)

        wz = self.kernel_init(keys[0], (self.input_dim, self.state_dim))
        wy = self.kernel_init(keys[1], (self.input_dim, self.state_dim))

        uz = self.recurrent_init(keys[2], (self.state_dim, self.state_dim))
        uy = self.recurrent_init(keys[3], (self.state_dim, self.state_dim))

        bz = self.bias_init(keys[4], (self.state_dim,))
        by = self.bias_init(keys[5], (self.state_dim,))

        return bz, by, wz, wy, uz, uy

    def apply(self, param, state, input):
        bz, by, wz, wy, uz, uy = param
        z = self.update_activation(input @ wz + state @ uz + bz)
        if self.reset_gate:
            y = self.candidate_activation(input @ wy + (state * z) @ uy + by)
        else:
            y = self.candidate_activation(input @ wy + state @ uy + by)
        return (1 - z) * state + z * y

    def init_state(self, key):
        return self.state_init(key, (self.state_dim,))
