from jax import nn, random
from jax import numpy as jnp

from . import regularizers, utils
from .basic import Bias, Convolution, Function, Linear
from .compound import Chain
from .module import Module
from .recurrent import MinimalGatedUnit


def Affine(
    input_dim,
    output_dim,
    kernel_initializer=nn.initializers.he_normal(),
    bias_initializer=nn.initializers.zeros,
    kernel_regularizer=regularizers.zero,
    bias_regularizer=regularizers.zero,
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


def MultiLayerPerceptron(
    dims,
    activation=Function(nn.relu),
    kernel_initializer=nn.initializers.he_normal(),
    bias_initializer=nn.initializers.zeros,
    kernel_regularizer=regularizers.zero,
    bias_regularizer=regularizers.zero,
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


class AutoEncoder(Module):
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

    def reconstruction_loss(self, param, input):
        output = self.apply(param, input)
        diff = input - output
        return (diff * jnp.conj(diff)).sum()

    def parameter_loss(self, param):
        encoder_loss = self.encoder.parameter_loss(param["encoder"])
        decoder_loss = self.decoder.parameter_loss(param["decoder"])
        return encoder_loss + decoder_loss


def get_von_neumann_neighbors(array, space_dim=None, include_center=False):
    """Get von Neumann neighborhoods of an array."""
    if space_dim is None:
        space_dim = array.ndim - 1
    neighbors = [
        jnp.roll(array, shift, axis)
        for shift in [-1, +1]
        for axis in range(-1 - space_dim, -1)
    ]
    if include_center:
        neighbors += [array]
    neighbors = jnp.concatenate(neighbors, -1)
    return neighbors


class NeuralCellularAutomaton(Module):
    """Neural GPUs learn algorithms (2015)
    https://arxiv.org/abs/1511.08228"""

    def __init__(self, state_dim, space_dim=1, cell_cls=MinimalGatedUnit):
        self.cell = cell_cls(state_dim, state_dim * 2 * space_dim)

    def init(self, key):
        return self.cell.init(key)

    def apply(self, param, state):
        neighbors = get_von_neumann_neighbors(state)
        new_state = self.cell.apply(param, state, neighbors)
        return new_state


class GatedLinearUnit(Module):
    """Language modeling with gated convolutional networks (2016)
    https://arxiv.org/abs/1612.08083"""

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w_init = nn.initializers.he_normal()
        self.v_init = nn.initializers.he_normal()
        self.b_init = nn.initializers.zeros
        self.c_init = nn.initializers.zeros

    def init(self, key):
        keys = random.split(key, 4)
        w = self.w_init(keys[0], (self.input_dim, self.output_dim))
        v = self.v_init(keys[1], (self.input_dim, self.output_dim))
        b = self.b_init(keys[2], (self.output_dim,))
        c = self.c_init(keys[3], (self.output_dim,))
        wv = jnp.concatenate([w, v], 1)
        bc = jnp.concatenate([b, c], 1)
        return wv, bc

    def apply(self, param, input):
        wv, bc = param
        x = input @ wv + bc
        y, z = jnp.split(x, [self.output_dim])
        return y * nn.sigmoid(z)


class ParametricReLU(Module):
    def __init__(
        self, initializer=nn.initializers.zeros, regularizer=regularizers.zero
    ):
        self.initializer = initializer
        self.regularizer = regularizer

    def init(self, key):
        return self.initializer(key, ())

    def apply(self, param, input):
        return jnp.where(input > 0, input, input * param)

    def parameter_loss(self, param):
        return self.regularizer(param)


def LeNet():
    """Gradient-based learning applied to document recognition
    https://ieeexplore.ieee.org/document/726791"""
    return Chain(
        [
            Convolution(1, 6, (5, 5), padding="SAME"),
            Bias(6),
            Function(nn.tanh),
            Function(utils.avg_pool((2, 2), (2, 2))),
            Convolution(6, 16, (5, 5)),
            Bias(16),
            Function(nn.tanh),
            Function(utils.avg_pool((2, 2), (2, 2))),
            Function(jnp.ravel),
            Affine(400, 120),
            Function(nn.tanh),
            Affine(120, 84),
            Function(nn.tanh),
            Affine(84, 10),
        ]
    )


def AlexNet():
    """ImageNet classification with deep convolutional neural networks
    https://dl.acm.org/doi/10.1145/3065386"""
    return Chain(
        [
            Convolution(3, 96, (11, 11), (4, 4)),
            Bias(96),
            Function(nn.relu),
            Function(utils.max_pool((3, 3), (2, 2))),
            Convolution(96, 256, (5, 5), padding="SAME"),
            Bias(256),
            Function(nn.relu),
            Function(utils.max_pool((3, 3), (2, 2))),
            Convolution(256, 384, (3, 3), padding="SAME"),
            Bias(384),
            Function(nn.relu),
            Convolution(384, 384, (3, 3), padding="SAME"),
            Bias(384),
            Function(nn.relu),
            Convolution(384, 256, (3, 3), padding="SAME"),
            Bias(256),
            Function(nn.relu),
            Function(utils.max_pool((3, 3), (2, 2))),
            Function(jnp.ravel),
            Affine(6400, 4096),
            Function(nn.relu),
            # dropout 0.5
            Affine(4096, 4096),
            Function(nn.relu),
            # dropout 0.5
            Affine(4096, 1000),
        ]
    )
