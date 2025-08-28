from typing import Any, Callable, Sequence

from jax import Array, lax, nn, random
from jax import numpy as jnp
from jax.nn.initializers import Initializer

from ._basic import Bias, Conv, BaseModule
from ._regularizers import Regularizer, zero

Key = Array
Module = Any


class RecurrentNetwork(BaseModule):
    def __init__(self, unit: Module):
        self.unit = unit

    def init(self, key: Key) -> dict[str, Any]:
        keys = random.split(key)
        w = self.unit.init(keys[0])
        h = self.unit.init_state(keys[1])
        return {"unit_param": w, "init_state": h}

    def apply(self, parameters: dict[str, Any], xs: Any) -> Any:
        w = parameters["unit_param"]
        h = parameters["init_state"]

        def f(h: Any, x: Any) -> Any:
            h_new = self.unit.apply(w, h, x)
            return h_new, h

        return lax.scan(f, h, xs)

    def parameter_loss(self, parameters: dict[str, Any]) -> Array | float:
        return self.unit.parameter_loss(parameters)


class SimpleRNN(BaseModule):
    """
    Simple recurrent unit.

    References:

    - *Finding structure in time*. 1990.
      https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        recurrent_initializer: Initializer = nn.initializers.orthogonal(),
        linear_initializer: Initializer = nn.initializers.glorot_uniform(),
        bias_initializer: Initializer = nn.initializers.zeros,
        recurrent_regularizer: Regularizer = zero,
        linear_regularizer: Regularizer = zero,
        bias_regularizer: Regularizer = zero,
        activation: Callable[[Array], Array] = nn.tanh,
        state_initializer: Initializer = nn.initializers.zeros,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.recurrent_initializer = recurrent_initializer
        self.linear_initializer = linear_initializer
        self.bias_initializer = bias_initializer
        self.recurrent_regularizer = recurrent_regularizer
        self.linear_regularizer = linear_regularizer
        self.bias_regularizer = bias_regularizer
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
        """
        Apply module.

        :param parameters: Parameters.
        :param state: Array of shape ``(..., state_dim)``.
        :param input: Array of shape ``(..., input_dim)``.
        :returns: Array of shape ``(..., output_dim)``.
        """
        y = (
            input @ parameters["linear"]
            + state @ parameters["recurrent"]
            + parameters["bias"]
        )
        return self.activation(y)

    def init_state(self, key: Key) -> Array:
        """
        Sample initial state.

        :param key: PRNG key.

        :returns: State.
        """
        return self.state_initializer(key, (self.state_dim,))

    def parameter_loss(self, parameters: dict[str, Array]) -> Array | float:
        loss = self.linear_regularizer(parameters["linear"])
        loss += self.recurrent_regularizer(parameters["recurrent"])
        loss += self.bias_regularizer(parameters["recurrent"])
        return loss


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
        reset_activation: Callable[[Array], Array] = nn.sigmoid,
        update_activation: Callable[[Array], Array] = nn.sigmoid,
        candidate_activation: Callable[[Array], Array] = nn.tanh,
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

    def init(self, key: Key) -> dict[str, Array]:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        keys = random.split(key, 9)
        return {
            "wz": self.linear_initializer(keys[0], (self.input_dim, self.state_dim)),
            "wr": self.linear_initializer(keys[1], (self.input_dim, self.state_dim)),
            "wy": self.linear_initializer(keys[2], (self.input_dim, self.state_dim)),
            "uz": self.recurrent_initializer(keys[3], (self.state_dim, self.state_dim)),
            "ur": self.recurrent_initializer(keys[4], (self.state_dim, self.state_dim)),
            "uy": self.recurrent_initializer(keys[5], (self.state_dim, self.state_dim)),
            "bz": self.bias_initializer(keys[6], (self.state_dim,)),
            "br": self.bias_initializer(keys[7], (self.state_dim,)),
            "by": self.bias_initializer(keys[8], (self.state_dim,)),
        }

    def apply(self, parameters: dict[str, Array], state: Array, input: Array) -> Array:
        z = self.update_activation(
            input @ parameters["wz"] + state @ parameters["uz"] + parameters["bz"]
        )
        r = self.reset_activation(
            input @ parameters["wr"] + state @ parameters["ur"] + parameters["br"]
        )
        y = self.candidate_activation(
            input @ parameters["wy"] + (r * state) @ parameters["uy"] + parameters["by"]
        )
        return (1 - z) * state + z * y

    def init_state(self, key: Key) -> Array:
        """
        Sample initial state.

        :param key: PRNG key.

        :returns: State.
        """
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
        update_activation: Callable[[Array], Array] = nn.sigmoid,
        candidate_activation: Callable[[Array], Array] = nn.tanh,
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

    def init(self, key: Key) -> dict[str, Array]:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        keys = random.split(key, 6)
        return {
            "wz": self.linear_initializer(keys[0], (self.input_dim, self.state_dim)),
            "wy": self.linear_initializer(keys[1], (self.input_dim, self.state_dim)),
            "uz": self.recurrent_initializer(keys[2], (self.state_dim, self.state_dim)),
            "uy": self.recurrent_initializer(keys[3], (self.state_dim, self.state_dim)),
            "bz": self.bias_initializer(keys[4], (self.state_dim,)),
            "by": self.bias_initializer(keys[5], (self.state_dim,)),
        }

    def apply(self, parameters: dict[str, Array], state: Array, input: Array) -> Array:
        z = self.update_activation(
            input @ parameters["wz"] + state @ parameters["uz"] + parameters["bz"]
        )
        if self.reset_gate:
            y = self.candidate_activation(
                input @ parameters["wy"]
                + (state * z) @ parameters["uy"]
                + parameters["by"]
            )
        else:
            y = self.candidate_activation(
                input @ parameters["wy"] + state @ parameters["uy"] + parameters["by"]
            )
        return (1 - z) * state + z * y

    def init_state(self, key: Key) -> Array:
        """
        Sample initial state.

        :param key: PRNG key.

        :returns: State.
        """
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

    def init(self, key: Key) -> dict[str, Array]:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        keys = random.split(key, 3)
        return {
            "ua": jnp.eye(self.state_dim),
            "uc": jnp.eye(self.state_dim),
            "wa": self.linear_initializer(keys[0], (self.input_dim, self.state_dim)),
            "wc": self.linear_initializer(keys[1], (self.input_dim, self.state_dim)),
            "wy": self.linear_initializer(keys[2], (self.input_dim, self.state_dim)),
        }

    def apply(self, parameters: dict[str, Array], state: Array, input: Array) -> Array:
        a = 1 + nn.tanh(input @ parameters["wa"] + state @ parameters["ua"])
        c = nn.sigmoid(input @ parameters["wc"] + state @ parameters["uc"])
        y = nn.tanh(input @ parameters["wy"] + state * a)
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
        state_initializer: Initializer = nn.initializers.zeros,
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.linear_initializer = linear_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.forget_bias = forget_bias
        self.state_initializer = state_initializer

    def init(self, key: Key) -> tuple[Array, ...]:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
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

    def init_state(self, key: Key) -> tuple[Array, Array]:
        """
        Sample initial state.

        :param key: PRNG key.

        :returns: State.
        """
        keys = random.split(key)
        h = self.state_initializer(keys[0], (self.state_dim,))
        c = self.state_initializer(keys[1], (self.state_dim,))
        return (h, c)


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
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
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
        activation: Callable[[Array], Array] = nn.tanh,
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

    def init(self, key: Key) -> dict[str, Array]:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        keys = random.split(key, 6)
        return {
            "Uc": self.recurrent_initializer(keys[0], (self.state_dim, self.state_dim)),
            "Ug": self.recurrent_initializer(keys[1], (self.state_dim, self.state_dim)),
            "Wc": self.linear_initializer(keys[2], (self.input_dim, self.state_dim)),
            "Wg": self.linear_initializer(keys[3], (self.input_dim, self.state_dim)),
            "bc": self.bias_initializer(keys[4], (self.state_dim,)),
            "bg": self.bias_initializer(keys[5], (self.state_dim,)),
        }

    def apply(self, w: dict[str, Array], h: Array, x: Array) -> Array:
        c = self.activation(w["bc"] + x @ w["Wc"] + h @ w["Uc"])
        g = nn.sigmoid(w["bg"] + x @ w["Wg"] + h @ w["Ug"])
        return g * h + (1 - g) * c


class ConvGatedUnit:
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        shape: Sequence[int],
        new_activation: Callable[[Array], Array] = nn.tanh,
        update_activation: Callable[[Array], Array] = nn.sigmoid,
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
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
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
