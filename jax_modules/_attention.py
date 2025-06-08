from jax import lax, nn, random
from jax import numpy as jnp

from ._module import Module
from ._utils import layer_norm


class Attention(Module):
    """Attention is all you need (2017)
    https://arxiv.org/abs/1706.03762

    Scaling vision transformers to 22 billion parameters (2023)
    https://arxiv.org/abs/2302.05442"""

    def __init__(
        self,
        query_input_dim,
        key_input_dim=None,
        value_input_dim=None,
        hidden_dim=None,
        heads=1,
        kernel_initializer=nn.initializers.he_normal(),
        bias_initializer=nn.initializers.zeros,
        normalize_qk=False,
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
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.normalize_qk = normalize_qk

    def init(self, key):
        keys = random.split(key, 6)
        return {
            "query_kernel": self.kernel_initializer(
                keys[0], (self.heads, self.query_input_dim, self.hidden_dim)
            ),
            "key_kernel": self.kernel_initializer(
                keys[1], (self.heads, self.key_input_dim, self.hidden_dim)
            ),
            "value_kernel": self.kernel_initializer(
                keys[2], (self.heads, self.value_input_dim, self.hidden_dim)
            ),
            "query_bias": self.bias_initializer(keys[3], (self.heads, self.hidden_dim)),
            "key_bias": self.bias_initializer(keys[4], (self.heads, self.hidden_dim)),
            "value_bias": self.bias_initializer(keys[5], (self.heads, self.hidden_dim)),
        }

    def apply(
        self,
        param,
        query_input,
        key_input=None,
        value_input=None,
        mask=None,
        bias=None,
        is_causal=False,
    ):
        if key_input is None:
            key_input = query_input
        if value_input is None:
            value_input = key_input

        query = jnp.tensordot(query_input, param["query_kernel"], (-1, -2))
        key = jnp.tensordot(key_input, param["key_kernel"], (-1, -2))
        value = jnp.tensordot(value_input, param["value_kernel"], (-1, -2))

        query += param["query_bias"]
        key += param["key_bias"]
        value += param["value_bias"]

        if self.normalize_qk:
            query = layer_norm(query)
            key = layer_norm(value)

        hidden = nn.dot_product_attention(
            query, key, value, mask=mask, bias=None, is_causal=is_causal
        )
        return lax.collapse(hidden, -2)
