import jax
from jax import nn
from jax import numpy as jnp

from . import regularizers
from .module import Module


class Constant(Module):
    def __init__(
        self, dim, initializer=nn.initializers.zeros, regularizer=regularizers.zero
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        return self.initializer(key, (self.dim,))

    def apply(self, param):
        return param

    def parameter_loss(self, param):
        return self.regularizer(param)


class Ball(Module):
    def __init__(
        self,
        dim,
        initializer=nn.initializers.zeros,
        regularizer=regularizers.zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        return self.initializer(key, (self.dim,))

    def apply(self, param):
        return param / jnp.sqrt(1 + param * jnp.conj(param))

    def parameter_loss(self, param):
        return self.regularizer(param)


class Simplex(Module):
    def __init__(
        self,
        dim,
        initializer=nn.initializers.zeros,
        regularizer=regularizers.zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        return self.initializer(key, (self.dim,))

    def apply(self, param):
        return nn.softmax(param)

    def parameter_loss(self, param):
        return self.regularizer(param)


def vector_to_symmetric_matrix(vector, dim):
    i = jnp.triu_indices(dim)
    A = jnp.zeros([dim, dim], vector.dtype)
    A = A.at[i].set(vector)
    A = A.T.at[i].set(vector)
    return A


def vector_to_antisymmetric_matrix(vector, dim):
    i = jnp.triu_indices(dim, 1)
    A = jnp.zeros([dim, dim], vector.dtype)
    A = A.at[i].set(vector)
    A = A.T.at[i].set(-vector)
    return A


class SymmetricMatrix(Module):
    def __init__(
        self,
        dim,
        initializer=nn.initializers.zeros,
        regularizer=regularizers.zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        n = self.dim * (self.dim + 1) // 2
        return self.initializer(key, (n,))

    def apply(self, param):
        return vector_to_symmetric_matrix(param, self.dim)

    def parameter_loss(self, param):
        return self.regularizer(param)


class AntisymmetricMatrix(Module):
    def __init__(
        self,
        dim,
        initializer=nn.initializers.zeros,
        regularizer=regularizers.zero,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        n = self.dim * (self.dim - 1) // 2
        return self.initializer(key, (n,))

    def apply(self, param):
        return vector_to_antisymmetric_matrix(param, self.dim)

    def parameter_loss(self, param):
        return self.regularizer(param)


class SpecialOrthogonalMatrix(Module):
    def __init__(self, dim, transform="exp"):
        self.antisymmetric = AntisymmetricMatrix(dim)
        self.transform = transform

    def init(self, key):
        return self.antisymmetric.init(key)

    def apply(self, param):
        m = self.antisymmetric.apply(param)
        match self.transform:
            case "exp":
                return jax.scipy.linalg.expm(m)
            case "cayley":
                i = jnp.eye(m.shape)
                return jnp.linalg.solve(i + m, i - m)
            case _:
                raise ValueError(f"Invalid transform {self.transform}.")
