from jax import nn, random
from jax import numpy as jnp

from .basic import Bias, Function, Linear
from .compound import Chain


def Affine(
    input_dim,
    output_dim,
    kernel_initializer=nn.initializers.he_normal(),
    bias_initializer=nn.initializers.zeros,
    kernel_regularizer=lambda param: 0.0,
    bias_regularizer=lambda param: 0.0,
):
    return Chain(
        [
            Linear(
                input_dim,
                output_dim,
                initializer=kernel_initializer,
                regularizer=kernel_regularizer,
            ),
            Bias(
                output_dim,
                initializer=bias_initializer,
                regularizer=bias_regularizer,
            ),
        ]
    )


def MLP(
    dims,
    activation=Function(nn.relu),
    kernel_initializer=nn.initializers.he_normal(),
    bias_initializer=nn.initializers.zeros,
    kernel_regularizer=lambda param: 0.0,
    bias_regularizer=lambda param: 0.0,
):
    lst = []
    for input_dim, output_dim in zip(dims[:-1], dims[1:]):
        lst.append(
            Affine(
                input_dim,
                output_dim,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        )
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

    def param_loss(self, param):
        encoder_loss = self.encoder.param_loss(param["encoder"])
        decoder_loss = self.decoder.param_loss(param["decoder"])
        return encoder_loss + decoder_loss
