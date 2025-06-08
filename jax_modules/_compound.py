import jax
from jax import flatten_util, lax, nn, random
from jax import numpy as jnp

from ._module import Module


class Chain(Module):
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


class Parallel(Module):
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


class Repeat(Module):
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


class Residual(Module):
    """Deep residual learning for image recognition
    https://arxiv.org/abs/1512.03385"""

    def __init__(self, module):
        self.module = module

    def init(self, key):
        return self.module.init(key)

    def apply(self, param, input):
        output = self.module.apply(param, input)
        return input + output

    def parameter_loss(self, param):
        return self.module.parameter_loss(param)


class Switch(Module):
    def __init__(self, module, branches):
        self.module = module
        self.branches = branches

    def init(self, key):
        keys = random.split(key, self.branches)
        return jax.vmap(self.module.init)(keys)

    def apply(self, param, branch, input):
        param = jax.tree.map(lambda x: x[branch], param)
        return self.module.apply(param, input)


class RavelUnravel(Module):
    def __init__(self, module):
        self.module = module

    def init(self, key):
        return self.module.init(key)

    def apply(self, w, x):
        x_ravel, unravel = flatten_util.ravel_pytree(x)
        y_ravel = self.module.apply(w, x_ravel)
        y = unravel(y_ravel)
        return y


class ConvexPotentialFlow(Module):
    """Convex potential flows: universal probability distributions with optimal
        transport and convex optimization (2020)
    https://arxiv.org/abs/2012.05942"""

    def __init__(self, module, alpha_initializer=nn.initializers.zeros):
        self.module = module
        self.alpha_initializer = alpha_initializer

    def init(self, key):
        keys = random.split(key)
        module_param = self.module.init(keys[0])
        alpha = self.alpha_initializer(keys[1], ())
        return module_param, alpha

    def apply(self, param, x):
        module_param, alpha = param

        def f(x):
            y = self.module.apply(module_param, x)
            z = (x * jnp.conj(x)).sum()
            return y + z * nn.softplus(alpha)

        return jax.grad(f)(x)
