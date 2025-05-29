import jax
from jax import nn
from jax import numpy as jnp

from .module import Module


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
    def __init__(self, dim, initializer=nn.initializers.zeros):
        self.dim = dim
        self.initializer = initializer

    def init(self, key):
        n = self.dim * (self.dim + 1) // 2
        return self.initializer(key, (n,))

    def apply(self, params):
        return vector_to_symmetric_matrix(params, self.dim)


class AntisymmetricMatrix(Module):
    def __init__(self, dim, initializer=nn.initializers.zeros):
        self.dim = dim
        self.initializer = initializer

    def init(self, key):
        n = self.dim * (self.dim - 1) // 2
        return self.initializer(key, (n,))

    def apply(self, params):
        return vector_to_antisymmetric_matrix(params, self.dim)


class SpecialOrthogonalMatrix(Module):
    def __init__(self, dim, transform="exp"):
        self.antisymmetric = AntisymmetricMatrix(dim)
        self.transform = transform

    def init(self, key):
        return self.antisymmetric.init(key)

    def apply(self, params):
        m = self.antisymmetric.apply(params)
        match self.transform:
            case "exp":
                return jax.scipy.linalg.expm(m)
            case "cayley":
                i = jnp.eye(m.shape)
                return jnp.linalg.solve(i + m, i - m)
            case _:
                raise ValueError(f"Invalid transform {self.transform}.")
