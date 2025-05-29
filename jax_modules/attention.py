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

    def apply(self, param, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k
        wq, wk, wv = param

        q = jnp.einsum("...ld,hde->...hle", q, wq)
        k = jnp.einsum("...ld,hde->...hle", k, wk)
        v = jnp.einsum("...ld,hde->...hle", v, wv)

        o = single_head_attention(q, k, v, mask)

        o = jnp.moveaxis(o, -3, -1)
        o = lax.collapse(o, -2, o.ndim)

        return o


def main():
    qk_dim = 2
    v_dim = 3

    q_len = 4
    kv_len = 5

    batch_shape = (6, 7)

    key = random.key(0)

    key, subkey = random.split(key)
    keys = random.split(subkey, 3)
    q = random.normal(keys[0], batch_shape + (q_len, qk_dim))
    k = random.normal(keys[1], batch_shape + (kv_len, qk_dim))
    v = random.normal(keys[2], batch_shape + (kv_len, v_dim))
    o = single_head_attention(q, k, v)
    assert o.shape == batch_shape + (q_len, v_dim)

    x_dim = 8
    heads = 16

    module = Attention(x_dim, heads=16)

    key, subkey = random.split(key)
    param = module.init(subkey)

    key, subkey = random.split(key)
    x = random.normal(subkey, (2, 3, x_dim))
    y = module.apply(param, x)
    assert y.shape == x.shape[:-1] + (x.shape[-1] * heads,)


if __name__ == "__main__":
    main()
