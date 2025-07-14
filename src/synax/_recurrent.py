from typing import Any, Callable, Sequence

from jax import Array, lax, nn, random
from jax import numpy as jnp

from ._basic import Bias, Conv

Key = Array
Initializer = Callable[[Key, tuple[int, ...]], Array]


class RecurrentNetwork:
    def __init__(self, unit):
        self.unit = unit

    def init(self, key: Key) -> dict[str, Any]:
        keys = random.split(key)
        w = self.unit.init(keys[0])
        h = self.unit.init_state(keys[1])
        return {"unit_param": w, "init_state": h}

    def apply(self, parameters: dict[str, Any], xs: Array) -> Any:
        w = parameters["unit_param"]
        h = parameters["init_state"]

        def f(h, x):
            h_new = self.unit.apply(w, h, x)
            return h_new, h

        return lax.scan(f, h, xs)


class SimpleRNN:
    """
    Simple recurrent neural network.

    References:

    - *Finding structure in time*. 1990.
      https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        linear_initializer: Initializer = nn.initializers.glorot_uniform(),
        bias_initializer: Initializer = nn.initializers.zeros,
        recurrent_initializer: Initializer = nn.initializers.orthogonal(),
        activation: Callable = nn.tanh,
        state_initializer: Initializer = nn.initializers.zeros,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.linear_initializer = linear_initializer
        self.bias_initializer = bias_initializer
        self.recurrent_initializer = recurrent_initializer
        self.activation = activation
        self.state_initializer = state_initializer

    def init(self, key: Key) -> dict[str, Array]:
        keys = random.split(key, 3)
        return {
            "linear": self.linear_initializer(
                keys[0], (self.input_dim, self.state_dim)
            ),
            "recurrent": self.recurrent_initializer(
                keys[1], (self.state_dim, self.state_dim)
            ),
            "bias": self.bias_initializer(keys[2], (self.state_dim,)),
        }

    def apply(self, parameters: dict[str, Array], state: Array, input: Array) -> Array:
        y = (
            input @ parameters["linear"]
            + state @ parameters["recurrent"]
            + parameters["bias"]
        )
        return self.activation(y)

    def init_state(self, key: Key) -> Array:
        return self.state_initializer(key, (self.state_dim,))


class GRU:
    """
    Gated recurrent unit.

    References:

    - *Learning phrase representations using RNN encoder-decoder for statistical
      machine*. 2014. https://arxiv.org/abs/1406.1078.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        linear_initializer: Initializer = nn.initializers.glorot_uniform(),
        bias_initializer: Initializer = nn.initializers.zeros,
        recurrent_initializer: Initializer = nn.initializers.orthogonal(),
        reset_activation: Callable = nn.sigmoid,
        update_activation: Callable = nn.sigmoid,
        candidate_activation: Callable = nn.tanh,
        state_initializer: Initializer = nn.initializers.zeros,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.linear_initializer = linear_initializer
        self.bias_initializer = bias_initializer
        self.recurrent_initializer = recurrent_initializer

        self.reset_activation = reset_activation
        self.update_activation = update_activation
        self.candidate_activation = candidate_activation

        self.state_initializer = state_initializer

    def init(self, key: Key) -> tuple[Array, ...]:
        keys = random.split(key, 9)

        wz = self.linear_initializer(keys[0], (self.input_dim, self.state_dim))
        wr = self.linear_initializer(keys[1], (self.input_dim, self.state_dim))
        wy = self.linear_initializer(keys[2], (self.input_dim, self.state_dim))

        uz = self.recurrent_initializer(keys[3], (self.state_dim, self.state_dim))
        ur = self.recurrent_initializer(keys[4], (self.state_dim, self.state_dim))
        uy = self.recurrent_initializer(keys[5], (self.state_dim, self.state_dim))

        bz = self.bias_initializer(keys[6], (self.state_dim,))
        br = self.bias_initializer(keys[7], (self.state_dim,))
        by = self.bias_initializer(keys[8], (self.state_dim,))

        return bz, br, by, wz, wr, wy, uz, ur, uy

    def apply(self, parameters: tuple[Array, ...], state: Array, input: Array) -> Array:
        bz, br, by, wz, wr, wy, uz, ur, uy = parameters
        z = self.update_activation(input @ wz + state @ uz + bz)
        r = self.reset_activation(input @ wr + state @ ur + br)
        y = self.candidate_activation(input @ wy + (r * state) @ uy + by)
        return (1 - z) * state + z * y

    def init_state(self, key: Key) -> Array:
        return self.state_initializer(key, (self.state_dim,))


class MGU:
    """
    Minimal gated unit.

    References:

    - *Minimal gated unit for recurrent neural networks*. 2016.
      https://arxiv.org/abs/1603.09420.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        linear_initializer: Initializer = nn.initializers.glorot_uniform(),
        bias_initializer: Initializer = nn.initializers.zeros,
        recurrent_initializer: Initializer = nn.initializers.orthogonal(),
        update_activation: Callable = nn.sigmoid,
        candidate_activation: Callable = nn.tanh,
        state_initializer: Initializer = nn.initializers.zeros,
        reset_gate: bool = True,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.linear_initializer = linear_initializer
        self.bias_initializer = bias_initializer
        self.recurrent_initializer = recurrent_initializer

        self.update_activation = update_activation
        self.candidate_activation = candidate_activation

        self.state_initializer = state_initializer
        self.reset_gate = reset_gate

    def init(self, key: Key) -> tuple[Array, ...]:
        keys = random.split(key, 6)

        wz = self.linear_initializer(keys[0], (self.input_dim, self.state_dim))
        wy = self.linear_initializer(keys[1], (self.input_dim, self.state_dim))

        uz = self.recurrent_initializer(keys[2], (self.state_dim, self.state_dim))
        uy = self.recurrent_initializer(keys[3], (self.state_dim, self.state_dim))

        bz = self.bias_initializer(keys[4], (self.state_dim,))
        by = self.bias_initializer(keys[5], (self.state_dim,))

        return bz, by, wz, wy, uz, uy

    def apply(self, parameters: tuple[Array, ...], state: Array, input: Array) -> Array:
        bz, by, wz, wy, uz, uy = parameters
        z = self.update_activation(input @ wz + state @ uz + bz)
        if self.reset_gate:
            y = self.candidate_activation(input @ wy + (state * z) @ uy + by)
        else:
            y = self.candidate_activation(input @ wy + state @ uy + by)
        return (1 - z) * state + z * y

    def init_state(self, key: Key) -> Array:
        return self.state_initializer(key, (self.state_dim,))


class BistableRecurrentCell:
    """
    Bi-stable recurrent cell.

    References:

    - *A bio-inspired bistable recurrent cell allows for long-lasting memory*.
      2020. https://arxiv.org/abs/2006.05252.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        linear_initializer: Initializer = nn.initializers.he_normal(),
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.linear_initializer = linear_initializer

    def init(self, key: Key) -> tuple[Array, ...]:
        keys = random.split(key, 3)

        ua = jnp.eye(self.state_dim)
        uc = jnp.eye(self.state_dim)

        wa = self.linear_initializer(keys[0], (self.input_dim, self.state_dim))
        wc = self.linear_initializer(keys[1], (self.input_dim, self.state_dim))
        wy = self.linear_initializer(keys[2], (self.input_dim, self.state_dim))

        return ua, uc, wa, wc, wy

    def apply(self, parameters: tuple[Array, ...], state: Array, input: Array) -> Array:
        ua, uc, wa, wc, wy = parameters
        a = 1 + nn.tanh(input @ wa + state @ ua)
        c = nn.sigmoid(input @ wc + state @ uc)
        y = nn.tanh(input @ wy + state * a)
        return c * state + (1 - c) * y


class LSTM:
    """
    Long short term memory.

    References:

    - *LSTM can solve hard long time lag problems*. 1996.
      https://dl.acm.org/doi/10.5555/2998981.2999048
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        linear_initializer: Initializer = nn.initializers.he_normal(),
        recurrent_initializer: Initializer = nn.initializers.orthogonal(),
        bias_initializer: Initializer = nn.initializers.zeros,
        forget_bias: float = 1.0,
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.linear_initializer = linear_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.forget_bias = forget_bias

    def init(self, key: Key) -> tuple[Array, ...]:
        keys = random.split(key, 12)

        Uf = self.recurrent_initializer(keys[0], (self.state_dim, self.state_dim))
        Ui = self.recurrent_initializer(keys[1], (self.state_dim, self.state_dim))
        Ug = self.recurrent_initializer(keys[2], (self.state_dim, self.state_dim))
        Uo = self.recurrent_initializer(keys[3], (self.state_dim, self.state_dim))

        Wf = self.linear_initializer(keys[4], (self.input_dim, self.state_dim))
        Wi = self.linear_initializer(keys[5], (self.input_dim, self.state_dim))
        Wg = self.linear_initializer(keys[6], (self.input_dim, self.state_dim))
        Wo = self.linear_initializer(keys[7], (self.input_dim, self.state_dim))

        bf = self.bias_initializer(keys[8], (self.state_dim,)) + self.forget_bias
        bi = self.bias_initializer(keys[9], (self.state_dim,))
        bg = self.bias_initializer(keys[10], (self.state_dim,))
        bo = self.bias_initializer(keys[11], (self.state_dim,))

        return bf, bi, bg, bo, Wf, Wi, Wg, Wo, Uf, Ui, Ug, Uo

    def apply(
        self, w: tuple[Array, ...], h_c: tuple[Array, Array], x: Array
    ) -> tuple[Array, Array]:
        bf, bi, bg, bo, Wf, Wi, Wg, Wo, Uf, Ui, Ug, Uo = w
        h, c = h_c

        f = nn.sigmoid(bf + x @ Wf + h @ Uf)
        i = nn.sigmoid(bi + x @ Wi + h @ Ui)
        g = nn.tanh(bg + x @ Wg + h @ Ug)
        o = nn.sigmoid(bo + x @ Wo + h @ Uo)

        new_c = f * c + i * g
        new_h = o * nn.tanh(new_c)

        return new_h, new_c


class FastGRNN:
    """
    Fast gated RNN.

    References:

    - *FastGRNN: a fast, accurate, stable and tiny kilobyte sized gated
      recurrent neural network. 2019. https://arxiv.org/abs/1901.02358.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        linear_initializer: Initializer = nn.initializers.he_normal(),
        bias_initializer: Initializer = nn.initializers.zeros,
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.linear_initializer = linear_initializer
        self.bias_initializer = bias_initializer

    def init(self, key: Key) -> tuple[Array, ...]:
        U = jnp.eye(self.state_dim)

        keys = random.split(key, 3)

        W = self.linear_initializer(keys[0], (self.input_dim, self.state_dim))

        bz = self.bias_initializer(keys[1], (self.state_dim,))
        by = self.bias_initializer(keys[2], (self.state_dim,))

        nu = jnp.array(0.0)
        zeta = jnp.array(0.0)

        return U, W, bz, by, zeta, nu

    def apply(self, w: tuple[Array, ...], h: Array, x: Array) -> Array:
        U, W, bz, by, zeta, nu = w
        z = nn.sigmoid(bz + h @ U + x @ W)
        y = nn.tanh(by + h @ U + x @ W)
        zeta = nn.sigmoid(zeta)
        nu = nn.sigmoid(nu)
        return (zeta * (1 - z) + nu) * y + z * h


class UpdateGateRNN:
    """
    Update gate RNN.

    References:

    - *Capacity and trainability in recurrent neural networks*. 2017.
      https://openreview.net/forum?id=BydARw9ex.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        activation: Callable = nn.tanh,
        linear_initializer: Initializer = nn.initializers.he_normal(),
        recurrent_initializer: Initializer = nn.initializers.orthogonal(),
        bias_initializer: Initializer = nn.initializers.zeros,
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.activation = activation
        self.linear_initializer = linear_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

    def init(self, key: Key) -> tuple[Array, ...]:
        keys = random.split(key, 6)

        Uc = self.recurrent_initializer(keys[0], (self.state_dim, self.state_dim))
        Ug = self.recurrent_initializer(keys[1], (self.state_dim, self.state_dim))

        Wc = self.linear_initializer(keys[2], (self.input_dim, self.state_dim))
        Wg = self.linear_initializer(keys[3], (self.input_dim, self.state_dim))

        bc = self.bias_initializer(keys[4], (self.state_dim,))
        bg = self.bias_initializer(keys[5], (self.state_dim,))

        return bc, bg, Wc, Wg, Uc, Ug

    def apply(self, w: tuple[Array, ...], h: Array, x: Array) -> Array:
        bc, bg, Wc, Wg, Uc, Ug = w
        c = self.activation(bc + x @ Wc + h @ Uc)
        g = nn.sigmoid(bg + x @ Wg + h @ Ug)
        return g * h + (1 - g) * c


class ConvGatedUnit:
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        shape: Sequence[int],
        new_activation: Callable = nn.tanh,
        update_activation: Callable = nn.sigmoid,
        linear_initializer: Initializer = nn.initializers.he_normal(),
        bias_initializer: Initializer = nn.initializers.zeros,
        recurrent_initializer: Initializer = nn.initializers.orthogonal(),
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim

        self.new_linear_state = Conv(
            self.state_dim,
            self.state_dim,
            shape=shape,
            initializer=recurrent_initializer,
            padding="SAME",
        )
        self.new_linear_input = Conv(
            self.input_dim,
            self.state_dim,
            shape=shape,
            initializer=linear_initializer,
            padding="SAME",
        )
        self.new_bias = Bias(self.state_dim, initializer=bias_initializer)

        self.update_linear_state = Conv(
            self.state_dim,
            self.state_dim,
            shape=shape,
            initializer=recurrent_initializer,
            padding="SAME",
        )
        self.update_linear_input = Conv(
            self.input_dim,
            self.state_dim,
            shape=shape,
            initializer=linear_initializer,
            padding="SAME",
        )
        self.update_bias = Bias(self.state_dim, initializer=bias_initializer)

        self.new_activation = new_activation
        self.update_activation = update_activation

    def init(self, key: Key) -> dict[str, Any]:
        keys = random.split(key, 6)
        return {
            "new_linear_state": self.new_linear_state.init(keys[0]),
            "new_linear_input": self.new_linear_input.init(keys[1]),
            "new_bias": self.new_bias.init(keys[2]),
            "update_linear_state": self.update_linear_state.init(keys[3]),
            "update_linear_input": self.update_linear_input.init(keys[4]),
            "update_bias": self.update_bias.init(keys[5]),
        }

    def apply(self, parameters: dict[str, Any], state: Array, input: Array) -> Array:
        new = self.new_linear_state.apply(parameters["new_linear_state"], state)
        new += self.new_linear_state.apply(parameters["new_linear_input"], input)
        new += self.new_bias.apply(parameters["new_bias"], new)
        new = self.new_activation(new)

        update = self.update_linear_state.apply(
            parameters["update_linear_state"], state
        )
        update += self.update_linear_input.apply(
            parameters["update_linear_input"], input
        )
        update += self.update_bias.apply(parameters["update_bias"], update)
        update = self.update_activation(update)

        return state + update * (new - state)
