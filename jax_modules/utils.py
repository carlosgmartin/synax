from jax import numpy as jnp


def layer_norm(x, axis=-1, epsilon=1e-6):
    """Layer normalization
    https://arxiv.org/abs/1607.06450"""
    x -= x.mean(axis, keepdims=True)
    x /= jnp.sqrt((x * jnp.conj(x)).mean(axis, keepdims=True) + epsilon)
    return x


def rms_norm(x, axis=-1, epsilon=1e-6):
    """Root mean square layer normalization
    https://arxiv.org/abs/1910.07467"""
    rms = jnp.sqrt((x * jnp.conj(x)).mean(axis, keepdims=True) + epsilon)
    return x / rms
