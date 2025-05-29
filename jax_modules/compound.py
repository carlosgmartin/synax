from jax import lax, random

from .module import Module


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

    def param_loss(self, param):
        return sum(
            module.param_loss(param)
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

    def param_loss(self, param):
        return sum(
            module.param_loss(param)
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

    def param_loss(self, param):
        return self.module.param_loss(param)


class Residual(Module):
    def __init__(self, module):
        self.module = module

    def init(self, key):
        return self.module.init(key)

    def apply(self, param, input):
        output = self.module.apply(param, input)
        return input + output

    def param_loss(self, param):
        return self.module.param_loss(param)
