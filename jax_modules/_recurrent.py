from jax import lax, nn, random
from jax import numpy as jnp

from ._basic import Bias, Convolution
from ._module import Module


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
        kernel_initializer=nn.initializers.glorot_uniform(),
        bias_initializer=nn.initializers.zeros,
        recurrent_initializer=nn.initializers.orthogonal(),
        activation=nn.tanh,
        state_initializer=nn.initializers.zeros,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.recurrent_initializer = recurrent_initializer
        self.activation = activation
        self.state_initializer = state_initializer

    def init(self, key):
        keys = random.split(key, 3)
        w = self.kernel_initializer(keys[0], (self.input_dim, self.state_dim))
        u = self.recurrent_initializer(keys[1], (self.state_dim, self.state_dim))
        b = self.bias_initializer(keys[2], (self.state_dim,))
        return w, u, b

    def apply(self, param, state, input):
        w, u, b = param
        y = param @ w + state @ u + b
        return self.activation(y)

    def init_state(self, key):
        return self.state_initializer(key, (self.state_dim,))


class GatedRecurrentUnit(Module):
    """Learning phrase representations using RNN encoder-decoder for statistical machine
        translation (2014)
    https://arxiv.org/abs/1406.1078"""

    def __init__(
        self,
        state_dim,
        input_dim,
        kernel_initializer=nn.initializers.glorot_uniform(),
        bias_initializer=nn.initializers.zeros,
        recurrent_initializer=nn.initializers.orthogonal(),
        reset_activation=nn.sigmoid,
        update_activation=nn.sigmoid,
        candidate_activation=nn.tanh,
        state_initializer=nn.initializers.zeros,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.recurrent_initializer = recurrent_initializer

        self.reset_activation = reset_activation
        self.update_activation = update_activation
        self.candidate_activation = candidate_activation

        self.state_initializer = state_initializer

    def init(self, key):
        keys = random.split(key, 9)

        wz = self.kernel_initializer(keys[0], (self.input_dim, self.state_dim))
        wr = self.kernel_initializer(keys[1], (self.input_dim, self.state_dim))
        wy = self.kernel_initializer(keys[2], (self.input_dim, self.state_dim))

        uz = self.recurrent_initializer(keys[3], (self.state_dim, self.state_dim))
        ur = self.recurrent_initializer(keys[4], (self.state_dim, self.state_dim))
        uy = self.recurrent_initializer(keys[5], (self.state_dim, self.state_dim))

        bz = self.bias_initializer(keys[6], (self.state_dim,))
        br = self.bias_initializer(keys[7], (self.state_dim,))
        by = self.bias_initializer(keys[8], (self.state_dim,))

        return bz, br, by, wz, wr, wy, uz, ur, uy

    def apply(self, param, state, input):
        bz, br, by, wz, wr, wy, uz, ur, uy = param
        z = self.update_activation(input @ wz + state @ uz + bz)
        r = self.reset_activation(input @ wr + state @ ur + br)
        y = self.candidate_activation(input @ wy + (r * state) @ uy + by)
        return (1 - z) * state + z * y

    def init_state(self, key):
        return self.state_initializer(key, (self.state_dim,))


class MinimalGatedUnit(Module):
    """Minimal gated unit for recurrent neural networks (2016)
    https://arxiv.org/abs/1603.09420"""

    def __init__(
        self,
        state_dim,
        input_dim,
        kernel_initializer=nn.initializers.glorot_uniform(),
        bias_initializer=nn.initializers.zeros,
        recurrent_initializer=nn.initializers.orthogonal(),
        update_activation=nn.sigmoid,
        candidate_activation=nn.tanh,
        state_initializer=nn.initializers.zeros,
        reset_gate=True,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.recurrent_initializer = recurrent_initializer

        self.update_activation = update_activation
        self.candidate_activation = candidate_activation

        self.state_initializer = state_initializer
        self.reset_gate = reset_gate

    def init(self, key):
        keys = random.split(key, 6)

        wz = self.kernel_initializer(keys[0], (self.input_dim, self.state_dim))
        wy = self.kernel_initializer(keys[1], (self.input_dim, self.state_dim))

        uz = self.recurrent_initializer(keys[2], (self.state_dim, self.state_dim))
        uy = self.recurrent_initializer(keys[3], (self.state_dim, self.state_dim))

        bz = self.bias_initializer(keys[4], (self.state_dim,))
        by = self.bias_initializer(keys[5], (self.state_dim,))

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
        return self.state_initializer(key, (self.state_dim,))


class BistableRecurrentCell(Module):
    """A bio-inspired bistable recurrent cell allows for long-lasting memory (2020)
    https://arxiv.org/abs/2006.05252"""

    def __init__(
        self, state_dim, input_dim, kernel_initializer=nn.initializers.he_normal()
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.kernel_initializer = kernel_initializer

    def init(self, key):
        keys = random.split(key, 3)

        ua = jnp.eye(self.state_dim)
        uc = jnp.eye(self.state_dim)

        wa = self.kernel_initializer(keys[0], (self.input_dim, self.state_dim))
        wc = self.kernel_initializer(keys[1], (self.input_dim, self.state_dim))
        wy = self.kernel_initializer(keys[2], (self.input_dim, self.state_dim))

        return ua, uc, wa, wc, wy

    def apply(self, param, state, input):
        ua, uc, wa, wc, wy = param
        a = 1 + nn.tanh(input @ wa + state @ ua)
        c = nn.sigmoid(input @ wc + state @ uc)
        y = nn.tanh(input @ wy + state * a)
        return c * state + (1 - c) * y


class LongShortTermMemory(Module):
    """LSTM can solve hard long time lag problems (1996)
    https://dl.acm.org/doi/10.5555/2998981.2999048"""

    def __init__(
        self,
        state_dim,
        input_dim,
        kernel_initializer=nn.initializers.he_normal(),
        recurrent_initializer=nn.initializers.orthogonal(),
        bias_initializer=nn.initializers.zeros,
        forget_bias=1.0,
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.forget_bias = forget_bias

    def init(self, key):
        keys = random.split(key, 12)

        Uf = self.recurrent_initializer(keys[0], (self.state_dim, self.state_dim))
        Ui = self.recurrent_initializer(keys[1], (self.state_dim, self.state_dim))
        Ug = self.recurrent_initializer(keys[2], (self.state_dim, self.state_dim))
        Uo = self.recurrent_initializer(keys[3], (self.state_dim, self.state_dim))

        Wf = self.kernel_initializer(keys[4], (self.input_dim, self.state_dim))
        Wi = self.kernel_initializer(keys[5], (self.input_dim, self.state_dim))
        Wg = self.kernel_initializer(keys[6], (self.input_dim, self.state_dim))
        Wo = self.kernel_initializer(keys[7], (self.input_dim, self.state_dim))

        bf = self.bias_initializer(keys[8], (self.state_dim,)) + self.forget_bias
        bi = self.bias_initializer(keys[9], (self.state_dim,))
        bg = self.bias_initializer(keys[10], (self.state_dim,))
        bo = self.bias_initializer(keys[11], (self.state_dim,))

        return bf, bi, bg, bo, Wf, Wi, Wg, Wo, Uf, Ui, Ug, Uo

    def apply(self, w, x, h_c):
        bf, bi, bg, bo, Wf, Wi, Wg, Wo, Uf, Ui, Ug, Uo = w
        h, c = h_c

        f = nn.sigmoid(bf + x @ Wf + h @ Uf)
        i = nn.sigmoid(bi + x @ Wi + h @ Ui)
        g = nn.tanh(bg + x @ Wg + h @ Ug)
        o = nn.sigmoid(bo + x @ Wo + h @ Uo)

        new_c = f * c + i * g
        new_h = o * nn.tanh(new_c)

        return new_h, new_c


class FastGRNN(Module):
    """Fastgrnn: a fast, accurate, stable and tiny kilobyte sized gated recurrent neural
        network (2019)
    https://arxiv.org/abs/1901.02358"""

    def __init__(
        self,
        state_dim,
        input_dim,
        kernel_initializer=nn.initializers.he_normal(),
        bias_initializer=nn.initializers.zeros,
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def init(self, key):
        U = jnp.eye(self.state_dim)

        keys = random.split(key, 3)

        W = self.kernel_initializer(keys[0], (self.input_dim, self.state_dim))

        bz = self.bias_initializer(keys[1], (self.state_dim,))
        by = self.bias_initializer(keys[2], (self.state_dim,))

        nu = jnp.array(0.0)
        zeta = jnp.array(0.0)

        return U, W, bz, by, zeta, nu

    def apply(self, w, x, h):
        U, W, bz, by, zeta, nu = w
        z = nn.sigmoid(bz + h @ U + x @ W)
        y = nn.tanh(by + h @ U + x @ W)
        zeta = nn.sigmoid(zeta)
        nu = nn.sigmoid(nu)
        return (zeta * (1 - z) + nu) * y + z * h


class UpdateGateRNN(Module):
    """Capacity and trainability in recurrent neural networks (2017)
    https://openreview.net/forum?id=BydARw9ex"""

    def __init__(
        self,
        state_dim,
        input_dim,
        activation=nn.tanh,
        kernel_initializer=nn.initializers.he_normal(),
        recurrent_initializer=nn.initializers.orthogonal(),
        bias_initializer=nn.initializers.zeros,
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

    def init(self, key):
        keys = random.split(key, 6)

        Uc = self.recurrent_initializer(keys[0], (self.state_dim, self.state_dim))
        Ug = self.recurrent_initializer(keys[1], (self.state_dim, self.state_dim))

        Wc = self.kernel_initializer(keys[2], (self.input_dim, self.state_dim))
        Wg = self.kernel_initializer(keys[3], (self.input_dim, self.state_dim))

        bc = self.bias_initializer(keys[4], (self.state_dim,))
        bg = self.bias_initializer(keys[5], (self.state_dim,))

        return bc, bg, Wc, Wg, Uc, Ug

    def apply(self, w, x, h):
        bc, bg, Wc, Wg, Uc, Ug = w
        c = self.activation(bc + x @ Wc + h @ Uc)
        g = nn.sigmoid(bg + x @ Wg + h @ Ug)
        return g * h + (1 - g) * c


class ConvolutionalGatedUnit:
    def __init__(
        self,
        state_dim,
        input_dim,
        window_shape,
        new_activation=nn.tanh,
        update_activation=nn.sigmoid,
        kernel_initializer=nn.initializers.he_normal(),
        bias_initializer=nn.initializers.zeros,
        recurrent_initializer=nn.initializers.orthogonal(),
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim

        self.new_linear_state = Convolution(
            self.state_dim,
            self.state_dim,
            window_shape=window_shape,
            initializer=recurrent_initializer,
            padding="SAME",
        )
        self.new_linear_input = Convolution(
            self.input_dim,
            self.state_dim,
            window_shape=window_shape,
            initializer=kernel_initializer,
            padding="SAME",
        )
        self.new_bias = Bias(self.state_dim, initializer=bias_initializer)

        self.update_linear_state = Convolution(
            self.state_dim,
            self.state_dim,
            window_shape=window_shape,
            initializer=recurrent_initializer,
            padding="SAME",
        )
        self.update_linear_input = Convolution(
            self.input_dim,
            self.state_dim,
            window_shape=window_shape,
            initializer=kernel_initializer,
            padding="SAME",
        )
        self.update_bias = Bias(self.state_dim, initializer=bias_initializer)

        self.new_activation = new_activation
        self.update_activation = update_activation

    def init(self, key):
        keys = random.split(key, 6)
        return {
            "new_linear_state": self.new_linear_state.init(keys[0]),
            "new_linear_input": self.new_linear_input.init(keys[1]),
            "new_bias": self.new_bias.init(keys[2]),
            "update_linear_state": self.update_linear_state.init(keys[3]),
            "update_linear_input": self.update_linear_input.init(keys[4]),
            "update_bias": self.update_bias.init(keys[5]),
        }

    def apply(self, param, state, input):
        new = self.new_linear_state.apply(param["new_linear_state"], state)
        new += self.new_linear_state.apply(param["new_linear_input"], input)
        new += self.new_bias.apply(param["new_bias"], new)
        new = self.new_activation(new)

        update = self.update_linear_state.apply(param["update_linear_state"], state)
        update += self.update_linear_input.apply(param["update_linear_input"], input)
        update += self.update_bias.apply(param["update_bias"], update)
        update = self.update_activation(update)

        return state + update * (new - state)
