from jax import random, nn


class Affine:
    """Affine map."""

    def __init__(
        self,
        input_dim,
        output_dim,
        weight_init=nn.initializers.he_normal(),
        bias_init=nn.initializers.zeros,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_init = weight_init
        self.bias_init = bias_init

    def init_params(self, key):
        keys = random.split(key)
        weight = self.weight_init(keys[0], (self.input_dim, self.output_dim))
        bias = self.bias_init(keys[1], (self.output_dim,))
        return {"weight": weight, "bias": bias}

    def apply(self, params, input):
        return input @ params["weight"] + params["bias"]


module = Affine(3, 2)
key = random.key(0)
params = module.init_params(key)
print(params)
