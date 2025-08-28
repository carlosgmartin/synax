from typing import Literal

import jax
from jax import Array, nn
from jax import numpy as jnp
from jax.nn.initializers import Initializer

from ._regularizers import Regularizer, zero
from ._basic import BaseModule

Key = Array


class Constant(BaseModule):
    def __init__(
        self,
        dim: int,
        initializer: Initializer = nn.initializers.zeros,
        regularizer: Regularizer = zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        return self.initializer(key, (self.dim,))

    def apply(self, params: Array) -> Array:
        return params

    def param_loss(self, params: Array) -> Array | float:
        return self.regularizer(params)


class Ball(BaseModule):
    def __init__(
        self,
        dim: int,
        initializer: Initializer = nn.initializers.zeros,
        regularizer: Regularizer = zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        return self.initializer(key, (self.dim,))

    def apply(self, params: Array) -> Array:
        return params / jnp.sqrt(1 + params * jnp.conj(params))

    def param_loss(self, params: Array) -> Array | float:
        return self.regularizer(params)


class Simplex(BaseModule):
    def __init__(
        self,
        dim: int,
        initializer: Initializer = nn.initializers.zeros,
        regularizer: Regularizer = zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        return self.initializer(key, (self.dim,))

    def apply(self, params: Array) -> Array:
        return nn.softmax(params)

    def param_loss(self, params: Array) -> Array | float:
        return self.regularizer(params)


def vector_to_symmetric_matrix(vector: Array, dim: int) -> Array:
    i = jnp.triu_indices(dim)
    a = jnp.zeros([dim, dim], vector.dtype)
    a = a.at[i].set(vector)
    a = a.T.at[i].set(vector)
    return a


def vector_to_antisymmetric_matrix(vector: Array, dim: int) -> Array:
    i = jnp.triu_indices(dim, 1)
    a = jnp.zeros([dim, dim], vector.dtype)
    a = a.at[i].set(vector)
    a = a.T.at[i].set(-vector)
    return a


class SymmetricMatrix(BaseModule):
    def __init__(
        self,
        dim: int,
        initializer: Initializer = nn.initializers.zeros,
        regularizer: Regularizer = zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        n = self.dim * (self.dim + 1) // 2
        return self.initializer(key, (n,))

    def apply(self, params: Array) -> Array:
        return vector_to_symmetric_matrix(params, self.dim)

    def param_loss(self, params: Array) -> Array | float:
        return self.regularizer(params)


class AntisymmetricMatrix(BaseModule):
    def __init__(
        self,
        dim: int,
        initializer: Initializer = nn.initializers.zeros,
        regularizer: Regularizer = zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        n = self.dim * (self.dim - 1) // 2
        return self.initializer(key, (n,))

    def apply(self, params: Array) -> Array:
        return vector_to_antisymmetric_matrix(params, self.dim)

    def param_loss(self, params: Array) -> Array | float:
        return self.regularizer(params)


class SpecialOrthogonalMatrix:
    def __init__(self, dim: int, transform: Literal["exp", "cayley"] = "exp"):
        self.antisymmetric = AntisymmetricMatrix(dim)
        self.transform = transform

    def init(self, key: Key) -> Array:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.antisymmetric.init(key)

    def apply(self, params: Array) -> Array:
        m = self.antisymmetric.apply(params)
        match self.transform:
            case "exp":
                return jax.scipy.linalg.expm(m)
            case "cayley":
                i = jnp.eye(m.shape)
                return jnp.linalg.solve(i + m, i - m)
            case _:
                raise ValueError(f"Invalid transform {self.transform}.")
