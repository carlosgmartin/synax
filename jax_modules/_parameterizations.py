from typing import Callable, Literal

import jax
from jax import Array, nn
from jax import numpy as jnp

from ._regularizers import zero

Key = Array


class Constant:
    def __init__(
        self,
        dim: int,
        initializer: Callable = nn.initializers.zeros,
        regularizer: Callable = zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        return self.initializer(key, (self.dim,))

    def apply(self, param: Array) -> Array:
        return param

    def parameter_loss(self, param: Array) -> Array:
        return self.regularizer(param)


class Ball:
    def __init__(
        self,
        dim: int,
        initializer: Callable = nn.initializers.zeros,
        regularizer: Callable = zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        return self.initializer(key, (self.dim,))

    def apply(self, param: Array) -> Array:
        return param / jnp.sqrt(1 + param * jnp.conj(param))

    def parameter_loss(self, param: Array) -> Array:
        return self.regularizer(param)


class Simplex:
    def __init__(
        self,
        dim: int,
        initializer: Callable = nn.initializers.zeros,
        regularizer: Callable = zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        return self.initializer(key, (self.dim,))

    def apply(self, param: Array) -> Array:
        return nn.softmax(param)

    def parameter_loss(self, param: Array) -> Array:
        return self.regularizer(param)


def vector_to_symmetric_matrix(vector: Array, dim: int) -> Array:
    i = jnp.triu_indices(dim)
    A = jnp.zeros([dim, dim], vector.dtype)
    A = A.at[i].set(vector)
    A = A.T.at[i].set(vector)
    return A


def vector_to_antisymmetric_matrix(vector: Array, dim: int) -> Array:
    i = jnp.triu_indices(dim, 1)
    A = jnp.zeros([dim, dim], vector.dtype)
    A = A.at[i].set(vector)
    A = A.T.at[i].set(-vector)
    return A


class SymmetricMatrix:
    def __init__(
        self,
        dim: int,
        initializer: Callable = nn.initializers.zeros,
        regularizer: Callable = zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        n = self.dim * (self.dim + 1) // 2
        return self.initializer(key, (n,))

    def apply(self, param: Array) -> Array:
        return vector_to_symmetric_matrix(param, self.dim)

    def parameter_loss(self, param: Array) -> Array:
        return self.regularizer(param)


class AntisymmetricMatrix:
    def __init__(
        self,
        dim: int,
        initializer: Callable = nn.initializers.zeros,
        regularizer: Callable = zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key: Key) -> Array:
        n = self.dim * (self.dim - 1) // 2
        return self.initializer(key, (n,))

    def apply(self, param: Array) -> Array:
        return vector_to_antisymmetric_matrix(param, self.dim)

    def parameter_loss(self, param: Array) -> Array:
        return self.regularizer(param)


class SpecialOrthogonalMatrix:
    def __init__(self, dim: int, transform: Literal["exp", "cayley"] = "exp"):
        self.antisymmetric = AntisymmetricMatrix(dim)
        self.transform = transform

    def init(self, key: Key) -> Array:
        return self.antisymmetric.init(key)

    def apply(self, param: Array) -> Array:
        m = self.antisymmetric.apply(param)
        match self.transform:
            case "exp":
                return jax.scipy.linalg.expm(m)
            case "cayley":
                i = jnp.eye(m.shape)
                return jnp.linalg.solve(i + m, i - m)
            case _:
                raise ValueError(f"Invalid transform {self.transform}.")
