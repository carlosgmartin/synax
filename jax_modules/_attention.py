from jax import lax, nn, random
from jax import numpy as jnp

from ._utils import layer_norm


class Attention:
    """
    Attention.

    :param query_input_dim: Dimension of the input used to compute queries.
    :type query_input_dim: int
    :param key_input_dim: Dimension of the input used to compute keys.
        Defaults to ``query_input_dim``.
    :type key_input_dim: int | None
    :param value_input_dim: Dimension of the input used to compute values.
        Defaults to ``key_input_dim``.
    :type value_input_dim: int | None
    :param hidden_dim: Dimension of the embeddings used to compute dot products.
        Defaults to ``query_input_dim``.
    :type hidden_dim: int
    :param heads: Number of attention heads.
    :type heads: int
    :param kernel_initializer: Initializer used for the kernels.
    :type kernel_initializer: jax.nn.initializers.Initializer
    :param bias_initializer: Initializer used for the biases.
    :type bias_initializer: jax.nn.initializers.Initializer
    :param normalize_qk: Apply layer norm to queries and keys before computing
        dot products.
    :type normalize_qk: bool

    References:

    - *Attention is all you need*. 2017. https://arxiv.org/abs/1706.03762.

    - *Scaling vision transformers to 22 billion parameters*. 2023. https://arxiv.org/abs/2302.05442.
    """

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
        """
        Sample initial parameters.

        :param key: A PRNG key.
        :param type: jax.Array
        """
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
        scale=None,
    ):
        """
        Apply the module.

        :param param: Module parameters.
        :type param: typing.Any
        :param query_input: Input used to compute queries.
        :type query_input: jax.Array
        :param key_input: Input used to compute keys.
        :type key_input: jax.Array | None
        :param value_input: Input used to compute values.
        :type value_input: jax.Array | None
        :param mask: Boolean mask used to filter out logits.
        :type mask: jax.Array | None
        :param bias: Bias array to be added to logits.
        :type bias: jax.Array | None
        :param is_causal: Apply causal attention.
        :type bool: bool
        :param scale: Scale for the logits. If ``None``, set to 1 divided by the
            square root of the query's head dimension.
        :type scale: float | None

        :returns: The output array.
        :rtype: jax.Array
        """
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
