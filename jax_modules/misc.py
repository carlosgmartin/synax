from jax import nn, random
from jax import numpy as jnp

from .basic import Bias, Function, Linear
from .compound import Chain


def Affine(
    input_dim,
    output_dim,
    kernel_initializer=nn.initializers.he_normal(),
    bias_initializer=nn.initializers.zeros,
):
    return Chain(
        [
            Linear(input_dim, output_dim, kernel_initializer),
            Bias(output_dim, bias_initializer),
        ]
    )


def MLP(
    dims,
    kernel_initializer=nn.initializers.he_normal(),
    bias_initializer=nn.initializers.zeros,
    activation=Function(nn.relu),
):
    lst = []
    for input_dim, output_dim in zip(dims[:-1], dims[1:]):
        lst.append(Affine(input_dim, output_dim, kernel_initializer, bias_initializer))
        lst.append(activation)
    return Chain(lst[:-1])


class Autoencoder:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def init(self, key):
        keys = random.split(key)
        return {
            "encoder": self.encoder.init(keys[0]),
            "decoder": self.decoder.init(keys[1]),
        }

    def encode(self, param, input):
        return self.encoder.apply(param["encoder"], input)

    def decode(self, param, input):
        return self.decoder.apply(param["decoder"], input)

    def apply(self, param, input):
        return self.decode(param, self.encode(param, input))

    def loss(self, param, input):
        output = self.apply(param, input)
        return jnp.square(input - output).sum()
