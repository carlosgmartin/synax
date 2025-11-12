from typing import Literal

import jax
from jax import Array, nn
from jax import numpy as jnp
from jax.nn.initializers import Initializer

from ._regularizers import Regularizer, zero

Key = Array


class Constant:
    r"""
    Constant.

    :param dim: Dimension.
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    :param dtype: Data type of parameters.
    """

    def __init__(
        self,
        dim: int,
        initializer: Initializer = nn.initializers.zeros,
        regularizer: Regularizer = zero,
        dtype: jax.typing.DTypeLike | None = None,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer
        self.dtype = dtype

    def init_params(self, key: Key) -> Array:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.initializer(key, (self.dim,), self.dtype)

    def apply(self, params: Array) -> Array:
        return params

    def param_loss(self, params: Array) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return self.regularizer(params)


class Ball:
    r"""
    Ball.

    :param dim: Dimension.
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    :param dtype: Data type of parameters.
    """

    def __init__(
        self,
        dim: int,
        initializer: Initializer = nn.initializers.zeros,
        regularizer: Regularizer = zero,
        dtype: jax.typing.DTypeLike | None = None,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer
        self.dtype = dtype

    def init_params(self, key: Key) -> Array:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.initializer(key, (self.dim,), self.dtype)

    def apply(self, params: Array) -> Array:
        return params / jnp.sqrt(1 + params * jnp.conj(params))

    def param_loss(self, params: Array) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return self.regularizer(params)


class Simplex:
    r"""
    Simplex.

    :param dim: Dimension.
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    :param dtype: Data type of parameters.
    """

    def __init__(
        self,
        dim: int,
        initializer: Initializer = nn.initializers.zeros,
        regularizer: Regularizer = zero,
        dtype: jax.typing.DTypeLike | None = None,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer
        self.dtype = dtype

    def init_params(self, key: Key) -> Array:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.initializer(key, (self.dim,), self.dtype)

    def apply(self, params: Array) -> Array:
        return nn.softmax(params)

    def param_loss(self, params: Array) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return self.regularizer(params)


def vector_to_symmetric_matrix(vector: Array, dim: int) -> Array:
    i = jnp.triu_indices(dim)
    a = jnp.zeros([dim, dim], vector.dtype)
    a = a.at[i].set(vector)
    a = a.T.at[i].set(vector)
    return a


def vector_to_anti_symmetric_matrix(vector: Array, dim: int) -> Array:
    i = jnp.triu_indices(dim, 1)
    a = jnp.zeros([dim, dim], vector.dtype)
    a = a.at[i].set(vector)
    a = a.T.at[i].set(-vector)
    return a


class SymmetricMatrix:
    r"""
    Symmetric matrix.

    .. math::
        A^\top = A

    :param dim: Dimension.
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    :param dtype: Data type of parameters.
    """

    def __init__(
        self,
        dim: int,
        initializer: Initializer = nn.initializers.zeros,
        regularizer: Regularizer = zero,
        dtype: jax.typing.DTypeLike | None = None,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer
        self.dtype = dtype

    def init_params(self, key: Key) -> Array:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        n = self.dim * (self.dim + 1) // 2
        return self.initializer(key, (n,), self.dtype)

    def apply(self, params: Array) -> Array:
        return vector_to_symmetric_matrix(params, self.dim)

    def param_loss(self, params: Array) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return self.regularizer(params)


class AntiSymmetricMatrix:
    r"""
    Symmetric matrix.

    .. math::
        A^\top = -A

    :param dim: Dimension.
    :param initializer: Initializer.
    :param regularizer: Regularizer.
    :param dtype: Data type of parameters.
    """

    def __init__(
        self,
        dim: int,
        initializer: Initializer = nn.initializers.zeros,
        regularizer: Regularizer = zero,
        dtype: jax.typing.DTypeLike | None = None,
    ):
        self.dim = dim
        self.initializer = initializer
        self.regularizer = regularizer
        self.dtype = dtype

    def init_params(self, key: Key) -> Array:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        n = self.dim * (self.dim - 1) // 2
        return self.initializer(key, (n,), self.dtype)

    def apply(self, params: Array) -> Array:
        return vector_to_anti_symmetric_matrix(params, self.dim)

    def param_loss(self, params: Array) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return self.regularizer(params)


class SpecialOrthogonalMatrix:
    r"""
    Special orthogonal matrix

    .. math::
        A^\top = A^{-1}

    :param dim: Dimension.
    :param transform: Transform used for parameterization.
        ``"exp"`` uses the matrix exponential :math:`\exp A`.
        ``"cayley"`` uses the Cayley transform :math:`(I - A) (I + A)^{-1}`.
    :param dtype: Data type of parameters.
    """

    def __init__(
        self,
        dim: int,
        transform: Literal["exp", "cayley"] = "exp",
        dtype: jax.typing.DTypeLike | None = None,
    ):
        self.anti_symmetric = AntiSymmetricMatrix(dim, dtype=dtype)
        self.transform = transform

    def init_params(self, key: Key) -> Array:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.anti_symmetric.init_params(key)

    def apply(self, params: Array) -> Array:
        m = self.anti_symmetric.apply(params)
        match self.transform:
            case "exp":
                return jax.scipy.linalg.expm(m)
            case "cayley":
                i = jnp.eye(m.shape)
                return jnp.linalg.solve(i + m, i - m)
            case _:
                raise ValueError(f"Invalid transform {self.transform}.")
