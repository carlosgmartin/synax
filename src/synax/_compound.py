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

    def init(self, key: Key) -> tuple:
        keys = random.split(key, len(self.modules))
        return tuple(module.init(key) for module, key in zip(self.modules, keys))

    def apply(self, parameters: tuple, input: Any) -> Any:
        for module, param in zip(self.modules, parameters, strict=True):
            input = module.apply(param, input)
        return input

    def parameter_loss(self, parameters: tuple) -> Array | float:
        return sum(
            module.parameter_loss(param)
            for module, param in zip(self.modules, parameters, strict=True)
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

    def init(self, key: Key) -> tuple:
        keys = random.split(key, len(self.modules))
        return tuple(module.init(key) for module, key in zip(self.modules, keys))

    def apply(self, parameters: tuple, input: Sequence[Any]) -> tuple:
        return tuple(
            module.apply(param, input)
            for module, param, input in zip(
                self.modules, parameters, input, strict=True
            )
        )

    def parameter_loss(self, parameters: tuple) -> Array | float:
        return sum(
            module.parameter_loss(param)
            for module, param in zip(self.modules, parameters, strict=True)
        )


class Repeat:
    def __init__(self, module: Module):
        self.module = module

    def init(self, key: Key) -> Any:
        return self.module.init(key)

    def apply(self, parameters: Any, input: Any, steps: int, unroll: int = 1):
        def f(x, _):
            y = self.module.apply(parameters, x)
            return y, x

        return lax.scan(f, input, length=steps, unroll=unroll)

    def parameter_loss(self, parameters: Any) -> Array:
        return self.module.parameter_loss(parameters)


class Residual:
    r"""
    Residual transformation.

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

    def init(self, key: Key) -> Any:
        return self.module.init(key)

    def apply(self, parameters: Any, input: Array) -> Array:
        output = self.module.apply(parameters, input)
        return input + output

    def parameter_loss(self, parameters: Any) -> Array:
        return self.module.parameter_loss(parameters)


class Switch:
    def __init__(self, module: Module, branches: int):
        self.module = module
        self.branches = branches

    def init(self, key: Key) -> Any:
        keys = random.split(key, self.branches)
        return jax.vmap(self.module.init)(keys)

    def apply(self, parameters: Any, branch: Array, input: Any):
        parameters = jax.tree.map(lambda x: x[branch], parameters)
        return self.module.apply(parameters, input)
