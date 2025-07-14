import jax
from jax import lax, random


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
    :type modules: typing.Sequence[Module]
    """

    def __init__(self, modules):
        self.modules = modules

    def init(self, key):
        keys = random.split(key, len(self.modules))
        return [module.init(key) for module, key in zip(self.modules, keys)]

    def apply(self, param, input):
        for module, param in zip(self.modules, param, strict=True):
            input = module.apply(param, input)
        return input

    def parameter_loss(self, param):
        return sum(
            module.parameter_loss(param)
            for module, param in zip(self.modules, param, strict=True)
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
    :type modules: typing.Sequence[Module]
    """

    def __init__(self, modules):
        self.modules = modules

    def init(self, key):
        keys = random.split(key, len(self.modules))
        return [module.init(key) for module, key in zip(self.modules, keys)]

    def apply(self, param, input):
        return [
            module.apply(param, input)
            for module, param, input in zip(self.modules, param, input, strict=True)
        ]

    def parameter_loss(self, param):
        return sum(
            module.parameter_loss(param)
            for module, param in zip(self.modules, param, strict=True)
        )


class Repeat:
    def __init__(self, module):
        self.module = module

    def init(self, key):
        return self.module.init(key)

    def apply(self, param, input, steps, unroll=1):
        def f(x, _):
            y = self.module.apply(param, x)
            return y, x

        return lax.scan(f, input, length=steps, unroll=unroll)

    def parameter_loss(self, param):
        return self.module.parameter_loss(param)


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
    :type module: Module
    """

    def __init__(self, module):
        self.module = module

    def init(self, key):
        return self.module.init(key)

    def apply(self, param, input):
        output = self.module.apply(param, input)
        return input + output

    def parameter_loss(self, param):
        return self.module.parameter_loss(param)


class Switch:
    def __init__(self, module, branches):
        self.module = module
        self.branches = branches

    def init(self, key):
        keys = random.split(key, self.branches)
        return jax.vmap(self.module.init)(keys)

    def apply(self, param, branch, input):
        param = jax.tree.map(lambda x: x[branch], param)
        return self.module.apply(param, input)
