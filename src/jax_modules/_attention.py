from typing import Callable

from jax import Array, lax, nn, random
from jax import numpy as jnp

from ._utils import layer_norm

Key = Array
Initializer = Callable[[Key, tuple[int, ...]], Array]


class Attention:
    """
    Attention.

    :param query_input_dim: Dimension of the input used to compute queries.
    :param key_input_dim: Dimension of the input used to compute keys.
        Defaults to ``query_input_dim``.
    :param value_input_dim: Dimension of the input used to compute values.
        Defaults to ``key_input_dim``.
    :param hidden_dim: Dimension of the embeddings used to compute dot products.
        Defaults to ``query_input_dim``.
    :param heads: Number of attention heads.
    :param linear_initializer: Initializer for linear layers.
    :param bias_initializer: Initializer for bias layers.
    :param normalize_qk: Apply layer norm to queries and keys before computing
        dot products.

    References:

    - *Attention is all you need*. 2017. https://arxiv.org/abs/1706.03762.

    - *Scaling vision transformers to 22 billion parameters*. 2023. https://arxiv.org/abs/2302.05442.
    """

    def __init__(
        self,
        query_input_dim: int,
        key_input_dim: int | None = None,
        value_input_dim: int | None = None,
        hidden_dim=None,
        heads: int = 1,
        linear_initializer: Initializer = nn.initializers.he_normal(),
        bias_initializer: Initializer = nn.initializers.zeros,
        normalize_qk: bool = False,
    ):
        if key_input_dim is None:
            key_input_dim = query_input_dim
        if value_input_dim is None:
            value_input_dim = key_input_dim
        if hidden_dim is None:
            hidden_dim = query_input_dim

        self.query_input_dim = query_input_dim
        self.key_input_dim = key_input_dim
        self.value_input_dim = value_input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.linear_initializer = linear_initializer
        self.bias_initializer = bias_initializer
        self.normalize_qk = normalize_qk

    def init(self, key: Array) -> dict[str, Array]:
        """
        Initialize parameters.

        :param key: PRNG key.
        """
        keys = random.split(key, 6)
        return {
            "query_kernel": self.linear_initializer(
                keys[0], (self.heads, self.query_input_dim, self.hidden_dim)
            ),
            "key_kernel": self.linear_initializer(
                keys[1], (self.heads, self.key_input_dim, self.hidden_dim)
            ),
            "value_kernel": self.linear_initializer(
                keys[2], (self.heads, self.value_input_dim, self.hidden_dim)
            ),
            "query_bias": self.bias_initializer(keys[3], (self.heads, self.hidden_dim)),
            "key_bias": self.bias_initializer(keys[4], (self.heads, self.hidden_dim)),
            "value_bias": self.bias_initializer(keys[5], (self.heads, self.hidden_dim)),
        }

    def apply(
        self,
        parameters: dict[str, Array],
        query_input: Array,
        key_input: Array | None = None,
        value_input: Array | None = None,
        mask: Array | None = None,
        bias: Array | None = None,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> Array:
        """
        Apply parameters.

        :param parameters: Parameters.
        :param query_input: Input used to compute queries.
        :param key_input: Input used to compute keys.
        :param value_input: Input used to compute values.
        :param mask: Boolean mask used to filter out logits.
        :param bias: Bias array to be added to logits.
        :param is_causal: Apply causal attention.
        :param scale: Scale for the logits. If ``None``, set to 1 divided by the
            square root of the query's head dimension.

        :returns: The output array.
        """
        if key_input is None:
            key_input = query_input
        if value_input is None:
            value_input = key_input

        query = jnp.tensordot(query_input, parameters["query_kernel"], (-1, -2))
        key = jnp.tensordot(key_input, parameters["key_kernel"], (-1, -2))
        value = jnp.tensordot(value_input, parameters["value_kernel"], (-1, -2))

        query += parameters["query_bias"]
        key += parameters["key_bias"]
        value += parameters["value_bias"]

        if self.normalize_qk:
            query = layer_norm()(query)
            key = layer_norm()(value)

        hidden = nn.dot_product_attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            bias=bias,
            is_causal=is_causal,
            scale=scale,
        )
        return lax.collapse(hidden, -2)
