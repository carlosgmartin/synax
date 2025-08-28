from typing import Any, Sequence

import jax
from jax import Array, lax, random

Module = Any
Key = Array


class Chain:
    r"""
    Sequential composition.

    Compose a list of modules sequentially, using the output of one module as
    the input for the next.

    Computes

    .. math::
        y = h_n

    where

    .. math::
        h_0 &= x \\
        h_{i+1} &= f_i(h_i)

    where :math:`f_i` is module :math:`i`. It therefore has type

    .. math::
        \prod_{i=0}^{n-1} (A_i \to A_{i+1}) \to A_0 \to A_n

    :param modules: Sequence of modules.
    """

    def __init__(self, modules: Sequence[Module]):
        self.modules = modules

    def init_params(self, key: Key) -> tuple[Any, ...]:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        keys = random.split(key, len(self.modules))
        return tuple(module.init_params(key) for module, key in zip(self.modules, keys))

    def apply(self, params: tuple[Any, ...], input: Any) -> Any:
        for module, param in zip(self.modules, params, strict=True):
            input = module.apply(param, input)
        return input

    def param_loss(self, params: tuple[Any, ...]) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return sum(
            module.param_loss(params)
            for module, params in zip(self.modules, params, strict=True)
        )


class Parallel:
    r"""
    Parallel composition.

    Compose a list of modules in parallel, receiving a tuple as input, passing
    each element to its corresponding module, and collecting their outputs as a
    tuple.

    Computes

    .. math::
        y = \{f_i(x_i)\}_{i \in [n]}

    where :math:`f_i` is module :math:`i`. It therefore has type

    .. math::
        \prod_{i=0}^{n-1} (A_i \to B_i) \to \prod_{i=0}^{n-1} A_i \to \prod_{i=0}^{n-1} B_i

    :param modules: Sequence of modules.
    """

    def __init__(self, modules: Sequence[Module]):
        self.modules = modules

    def init_params(self, key: Key) -> tuple[Any, ...]:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        keys = random.split(key, len(self.modules))
        return tuple(module.init_params(key) for module, key in zip(self.modules, keys))

    def apply(self, params: tuple[Any, ...], input: Sequence[Any]) -> tuple[Any, ...]:
        return tuple(
            module.apply(params, input)
            for module, params, input in zip(self.modules, params, input, strict=True)
        )

    def param_loss(self, params: tuple[Any, ...]) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return sum(
            module.param_loss(params)
            for module, params in zip(self.modules, params, strict=True)
        )


class Repeat:
    def __init__(self, module: Module):
        self.module = module

    def init_params(self, key: Key) -> Any:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.module.init_params(key)

    def apply(self, params: Any, input: Any, steps: int, unroll: int = 1) -> Any:
        def f(x: Any, _: None) -> Any:
            y = self.module.apply(params, x)
            return y, x

        return lax.scan(f, input, length=steps, unroll=unroll)

    def param_loss(self, params: Any) -> Array:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return self.module.param_loss(params)


class Residual:
    r"""
    Residual map.

    Computes

    .. math::
        y = x + f(x)

    where :math:`f` is a given module.

    References:

    - *Deep residual learning for image recognition*. 2015.
      https://arxiv.org/abs/1512.03385.

    :param module: Module to apply.
    """

    def __init__(self, module: Module):
        self.module = module

    def init_params(self, key: Key) -> Any:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.module.init_params(key)

    def apply(self, params: Any, input: Array) -> Array:
        output = self.module.apply(params, input)
        return input + output

    def param_loss(self, params: Any) -> Array:
        """
        Parameter loss.

        :param parameters: Parameters.

        :returns: Scalar.
        """
        return self.module.param_loss(params)


class Switch:
    def __init__(self, module: Module, branches: int):
        self.module = module
        self.branches = branches

    def init_params(self, key: Key) -> Any:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        keys = random.split(key, self.branches)
        return jax.vmap(self.module.init_params)(keys)

    def apply(self, params: Any, branch: Array, input: Any) -> Any:
        def f(x: Array) -> Array:
            return x[branch]

        params = jax.tree.map(f, params)
        return self.module.apply(params, input)

    def param_loss(self, params: Any) -> Array:
        """
        Parameter loss.

        :param parameters: Parameters.

        :returns: Scalar.
        """
        losses = jax.vmap(self.module.param_loss)(params)
        return losses.mean(0)
