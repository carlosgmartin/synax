from jax import lax, nn, random
from jax import numpy as jnp

from .module import Module


def single_head_attention(q, k, v, mask=None):
    """
    Arguments:
    q: array of shape (..., n, d)
    k: array of shape (..., m, d)
    v: array of shape (..., m, e)
    mask: array broadcastable to shape (..., n, m)

    Result:
    o: array of shape (..., n, e)
    """
    l = jnp.einsum("...nd,...md->...nm", q, k)
    l *= 1 / l.shape[-1] ** 0.5
    p = nn.softmax(l, -1, where=mask)
    o = jnp.einsum("...nm,...me->...ne", p, v)
    return o


class Attention(Module):
    # https://arxiv.org/abs/2305.09828

    def __init__(self, q_dim, k_dim=None, v_dim=None, x_dim=None, heads=1):
        if k_dim is None:
            k_dim = q_dim
        if v_dim is None:
            v_dim = k_dim
        if x_dim is None:
            x_dim = q_dim
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.x_dim = x_dim

        self.wq_init = nn.initializers.he_normal()
        self.wk_init = nn.initializers.he_normal()
        self.wv_init = nn.initializers.he_normal()

        self.heads = heads

    def init(self, key):
        keys = random.split(key, 3)
        wq = self.wq_init(keys[0], (self.heads, self.q_dim, self.x_dim))
        wk = self.wk_init(keys[1], (self.heads, self.k_dim, self.x_dim))
        wv = self.wv_init(keys[2], (self.heads, self.v_dim, self.x_dim))
        return wq, wk, wv

    def apply(self, params, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k
        wq, wk, wv = params

        q = jnp.einsum("...ld,hde->...hle", q, wq)
        k = jnp.einsum("...ld,hde->...hle", k, wk)
        v = jnp.einsum("...ld,hde->...hle", v, wv)

        o = single_head_attention(q, k, v, mask)

        o = jnp.moveaxis(o, -3, -1)
        o = lax.collapse(o, -2, o.ndim)

        return o
