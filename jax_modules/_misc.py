from functools import partial

from jax import nn, random
from jax import numpy as jnp

from ._basic import Bias, Conv, Dense, Func
from ._compound import Chain
from ._recurrent import MGU
from ._regularizers import zero
from ._utils import max_pool, mean_pool


def MLP(
    dims,
    activation=Func(nn.relu),
    kernel_initializer=nn.initializers.he_normal(),
    bias_initializer=nn.initializers.zeros,
    kernel_regularizer=zero,
    bias_regularizer=zero,
):
    modules = []
    for input_dim, output_dim in zip(dims[:-1], dims[1:]):
        dense = Dense(
            input_dim,
            output_dim,
            initializer=kernel_initializer,
            regularizer=kernel_regularizer,
        )
        bias = Bias(
            output_dim,
            initializer=bias_initializer,
            regularizer=bias_regularizer,
        )
        modules += [dense, bias, activation]
    return Chain(modules[:-1])


class AutoEncoder:
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
    spatial_axes = range(-1 - space_dim, -1)
    neighbors = [
        jnp.roll(array, shift, axis) for shift in [-1, +1] for axis in spatial_axes
    ]
    if include_center:
        neighbors += [array]
    neighbors = jnp.concatenate(neighbors, -1)
    return neighbors


class NeuralCellularAutomaton:
    """Neural GPUs learn algorithms (2015)
    https://arxiv.org/abs/1511.08228"""

    def __init__(
        self,
        state_dim,
        space_dim=1,
        cell_cls=partial(MGU, reset_gate=False),
        global_mean=False,
        global_max=False,
    ):
        self.cell = cell_cls(
            state_dim, state_dim * (2 * space_dim + global_mean + global_max)
        )
        self.global_mean = global_mean
        self.global_max = global_max

    def init(self, key):
        return self.cell.init(key)

    def apply(self, param, state):
        inputs = []

        neighbors = get_von_neumann_neighbors(state)
        inputs.append(neighbors)

        space_dim = state.ndim - 1
        spatial_axes = range(-1 - space_dim, -1)

        if self.global_mean:
            x = state.mean(spatial_axes, keepdims=True)
            x = jnp.broadcast_to(x, state.shape)
            inputs.append(x)

        if self.global_max:
            x = state.max(spatial_axes, keepdims=True)
            x = jnp.broadcast_to(x, state.shape)
            inputs.append(x)

        inputs = jnp.concatenate(inputs, -1)
        new_state = self.cell.apply(param, state, inputs)
        return new_state


class GLU:
    """Gated linear unit
    Language modeling with gated convolutional networks (2016)
    https://arxiv.org/abs/1612.08083"""

    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_initializer=nn.initializers.he_normal(),
        bias_initializer=nn.initializers.zeros,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def init(self, key):
        keys = random.split(key, 4)
        w = self.kernel_initializer(keys[0], (self.input_dim, self.output_dim))
        v = self.kernel_initializer(keys[1], (self.input_dim, self.output_dim))
        b = self.bias_initializer(keys[2], (self.output_dim,))
        c = self.bias_initializer(keys[3], (self.output_dim,))
        wv = jnp.concatenate([w, v], 1)
        bc = jnp.concatenate([b, c], 1)
        return wv, bc

    def apply(self, param, input):
        wv, bc = param
        x = input @ wv + bc
        y, z = jnp.split(x, [self.output_dim])
        return y * nn.sigmoid(z)


class PReLU:
    """Parametric ReLU"""

    def __init__(
        self,
        initializer=nn.initializers.zeros,
        regularizer=zero,
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
            Conv(1, 6, (5, 5), padding="SAME"),
            Bias(6),
            Func(nn.tanh),
            Func(mean_pool((2, 2), strides=(2, 2))),
            Conv(6, 16, (5, 5)),
            Bias(16),
            Func(nn.tanh),
            Func(mean_pool((2, 2), strides=(2, 2))),
            Func(jnp.ravel),
            Dense(400, 120),
            Bias(120),
            Func(nn.tanh),
            Dense(120, 84),
            Bias(84),
            Func(nn.tanh),
            Dense(84, 10),
            Bias(10),
        ]
    )


def AlexNet():
    """ImageNet classification with deep convolutional neural networks
    https://dl.acm.org/doi/10.1145/3065386"""
    return Chain(
        [
            Conv(3, 96, (11, 11), (4, 4)),
            Bias(96),
            Func(nn.relu),
            Func(max_pool((3, 3), strides=(2, 2))),
            Conv(96, 256, (5, 5), padding="SAME"),
            Bias(256),
            Func(nn.relu),
            Func(max_pool((3, 3), strides=(2, 2))),
            Conv(256, 384, (3, 3), padding="SAME"),
            Bias(384),
            Func(nn.relu),
            Conv(384, 384, (3, 3), padding="SAME"),
            Bias(384),
            Func(nn.relu),
            Conv(384, 256, (3, 3), padding="SAME"),
            Bias(256),
            Func(nn.relu),
            Func(max_pool((3, 3), strides=(2, 2))),
            Func(jnp.ravel),
            Dense(6400, 4096),
            Bias(4096),
            Func(nn.relu),
            # dropout 0.5
            Dense(4096, 4096),
            Bias(4096),
            Func(nn.relu),
            # dropout 0.5
            Dense(4096, 1000),
            Bias(1000),
        ]
    )
