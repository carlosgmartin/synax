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

    def init_params(self, key: Key) -> dict[str, Any]:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        keys = random.split(key)
        w = self.unit.init_params(keys[0])
        h = self.unit.init_state(keys[1])
        return {"unit_param": w, "init_state": h}

    def apply(self, params: dict[str, Any], inputs: Any) -> Any:
        w = params["unit_param"]
        h = params["init_state"]

        def f(h: Any, x: Any) -> Any:
            h_new = self.unit.apply(w, h, x)
            return h_new, h

        return lax.scan(f, h, inputs)

    def param_loss(self, params: dict[str, Any]) -> Array | float:
        return self.unit.param_loss(params)


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

    def init_params(self, key: Key) -> dict[str, Array]:
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

    def apply(self, params: dict[str, Array], state: Array, input: Array) -> Array:
        """
        Apply module.

        :param params: Parameters.
        :param state: Array of shape ``(..., state_dim)``.
        :param input: Array of shape ``(..., input_dim)``.
        :returns: Array of shape ``(..., output_dim)``.
        """
        y = input @ params["linear"] + state @ params["recurrent"] + params["bias"]
        return self.activation(y)

    def init_state(self, key: Key) -> Array:
        """
        Sample initial state.

        :param key: PRNG key.

        :returns: State.
        """
        return self.state_initializer(key, (self.state_dim,))

    def param_loss(self, params: dict[str, Array]) -> Array | float:
        loss = self.linear_regularizer(params["linear"])
        loss += self.recurrent_regularizer(params["recurrent"])
        loss += self.bias_regularizer(params["recurrent"])
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

    def init_params(self, key: Key) -> dict[str, Array]:
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

    def apply(self, params: dict[str, Array], state: Array, input: Array) -> Array:
        z = self.update_activation(
            input @ params["wz"] + state @ params["uz"] + params["bz"]
        )
        r = self.reset_activation(
            input @ params["wr"] + state @ params["ur"] + params["br"]
        )
        y = self.candidate_activation(
            input @ params["wy"] + (r * state) @ params["uy"] + params["by"]
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

    def init_params(self, key: Key) -> dict[str, Array]:
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

    def apply(self, params: dict[str, Array], state: Array, input: Array) -> Array:
        z = self.update_activation(
            input @ params["wz"] + state @ params["uz"] + params["bz"]
        )
        if self.reset_gate:
            y = self.candidate_activation(
                input @ params["wy"] + (state * z) @ params["uy"] + params["by"]
            )
        else:
            y = self.candidate_activation(
                input @ params["wy"] + state @ params["uy"] + params["by"]
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

    def init_params(self, key: Key) -> dict[str, Array]:
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

    def apply(self, params: dict[str, Array], state: Array, input: Array) -> Array:
        a = 1 + nn.tanh(input @ params["wa"] + state @ params["ua"])
        c = nn.sigmoid(input @ params["wc"] + state @ params["uc"])
        y = nn.tanh(input @ params["wy"] + state * a)
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

    def init_params(self, key: Key) -> dict[str, Array]:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        keys = random.split(key, 12)
        return {
            "Uf": self.recurrent_initializer(keys[0], (self.state_dim, self.state_dim)),
            "Ui": self.recurrent_initializer(keys[1], (self.state_dim, self.state_dim)),
            "Ug": self.recurrent_initializer(keys[2], (self.state_dim, self.state_dim)),
            "Uo": self.recurrent_initializer(keys[3], (self.state_dim, self.state_dim)),
            "Wf": self.linear_initializer(keys[4], (self.input_dim, self.state_dim)),
            "Wi": self.linear_initializer(keys[5], (self.input_dim, self.state_dim)),
            "Wg": self.linear_initializer(keys[6], (self.input_dim, self.state_dim)),
            "Wo": self.linear_initializer(keys[7], (self.input_dim, self.state_dim)),
            "bf": self.bias_initializer(keys[8], (self.state_dim,)) + self.forget_bias,
            "bi": self.bias_initializer(keys[9], (self.state_dim,)),
            "bg": self.bias_initializer(keys[10], (self.state_dim,)),
            "bo": self.bias_initializer(keys[11], (self.state_dim,)),
        }

    def apply(
        self, params: dict[str, Array], state: tuple[Array, Array], input: Array
    ) -> tuple[Array, Array]:
        h, c = state

        f = nn.sigmoid(params["bf"] + input @ params["Wf"] + h @ params["Uf"])
        i = nn.sigmoid(params["bi"] + input @ params["Wi"] + h @ params["Ui"])
        g = nn.tanh(params["bg"] + input @ params["Wg"] + h @ params["Ug"])
        o = nn.sigmoid(params["bo"] + input @ params["Wo"] + h @ params["Uo"])

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

    def init_params(self, key: Key) -> dict[str, Array]:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        keys = random.split(key, 3)
        return {
            "U": jnp.eye(self.state_dim),
            "W": self.linear_initializer(keys[0], (self.input_dim, self.state_dim)),
            "bz": self.bias_initializer(keys[1], (self.state_dim,)),
            "by": self.bias_initializer(keys[2], (self.state_dim,)),
            "nu": jnp.array(0.0),
            "zeta": jnp.array(0.0),
        }

    def apply(self, params: dict[str, Array], state: Array, input: Array) -> Array:
        z = nn.sigmoid(params["bz"] + state @ params["U"] + input @ params["W"])
        y = nn.tanh(params["by"] + state @ params["U"] + input @ params["W"])
        zeta = nn.sigmoid(params["zeta"])
        nu = nn.sigmoid(params["nu"])
        return (zeta * (1 - z) + nu) * y + z * state


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

    def init_params(self, key: Key) -> dict[str, Array]:
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

    def apply(self, params: dict[str, Array], state: Array, input: Array) -> Array:
        c = self.activation(params["bc"] + input @ params["Wc"] + state @ params["Uc"])
        g = nn.sigmoid(params["bg"] + input @ params["Wg"] + state @ params["Ug"])
        return g * state + (1 - g) * c


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

    def init_params(self, key: Key) -> dict[str, Any]:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        keys = random.split(key, 6)
        return {
            "new_linear_state": self.new_linear_state.init_params(keys[0]),
            "new_linear_input": self.new_linear_input.init_params(keys[1]),
            "new_bias": self.new_bias.init_params(keys[2]),
            "update_linear_state": self.update_linear_state.init_params(keys[3]),
            "update_linear_input": self.update_linear_input.init_params(keys[4]),
            "update_bias": self.update_bias.init_params(keys[5]),
        }

    def apply(self, params: dict[str, Any], state: Array, input: Array) -> Array:
        new = self.new_linear_state.apply(params["new_linear_state"], state)
        new += self.new_linear_state.apply(params["new_linear_input"], input)
        new += self.new_bias.apply(params["new_bias"], new)
        new = self.new_activation(new)

        update = self.update_linear_state.apply(params["update_linear_state"], state)
        update += self.update_linear_input.apply(params["update_linear_input"], input)
        update += self.update_bias.apply(params["update_bias"], update)
        update = self.update_activation(update)

        return state + update * (new - state)
