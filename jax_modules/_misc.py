from functools import partial

from jax import nn, random
from jax import numpy as jnp

from ._basic import Bias, Conv, Func, Linear
from ._compound import Chain
from ._recurrent import MGU
from ._regularizers import zero
from ._utils import max_pool, mean_pool


def MLP(
    dimensions,
    activation=Func(nn.relu),
    kernel_initializer=nn.initializers.he_normal(),
    bias_initializer=nn.initializers.zeros,
    kernel_regularizer=zero,
    bias_regularizer=zero,
):
    """
    Multi-layer perceptron.

    :param dimensions: Dimension of each layer.
    :type dimensions: typing.Sequence[int]
    :param activation: Module to use as an activation function.
        Not applied to the output.
    :type activation: Module
    :param kernel_initializer: Initializer used for the kernels.
    :type kernel_initializer: jax.nn.initializers.Initializer
    :param bias_initializer: Initializer used for the biases.
    :type bias_initializer: jax.nn.initializers.Initializer
    :param kernel_regularizer: Regularizer used for the kernels.
    :type kernel_regularizer: typing.Callable
    :param bias_regularizer: Regularizer used for the biases.
    :type bias_regularizer: typing.Callable

    References:

    - *A logical calculus of the ideas immanent in nervous activity*. 1943.
      https://link.springer.com/article/10.1007/BF02478259.

    - *The perceptron: A probabilistic model for information storage and
      organization in the brain*. 1958.
      https://psycnet.apa.org/record/1959-09865-001.

    - *Learning representations by back-propagating errors*. 1986.
      https://www.nature.com/articles/323533a0.
    """

    modules = []
    for input_dimension, output_dimension in zip(dimensions[:-1], dimensions[1:]):
        linear = Linear(
            input_dimension,
            output_dimension,
            initializer=kernel_initializer,
            regularizer=kernel_regularizer,
        )
        bias = Bias(
            output_dimension,
            initializer=bias_initializer,
            regularizer=bias_regularizer,
        )
        modules += [linear, bias, activation]
    return Chain(modules[:-1])


class AutoEncoder:
    r"""
    Auto-encoder.

    Computes

    .. math::
        y = g(f(x))

    where :math:`f` is a given encoder and :math:`g` is a given decoder.

    :param encoder: Module to use as encoder.
    :type encoder: Module
    :param decoder: Module to use as decoder.
    :type decoder: Module
    """

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


class NeuralGPU:
    """
    Neural GPU.

    References:

    - *Neural GPUs learn algorithms*. 2015. https://arxiv.org/abs/1511.08228.
    """

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
    r"""
    Gated linear unit.

    Computes

    .. math::
        y = \sigma(A_1 x + b_1) \odot (A_2 x + b_2)

    where :math:`\sigma` is a sigmoid function, :math:`A_1` and :math:`A_2` are
    learned matrices, and :math:`b_1` and :math:`b_2` are learned vectors.

    :param input_dimension: Dimension of the input.
    :type input_dimension: int
    :param output_dimension: Dimension of the output.
    :type output_dimension: int
    :param kernel_initializer: Initializer used for the kernels.
    :type kernel_initializer: jax.nn.initializers.Initializer
    :param bias_initializer: Initializer used for the biases.
    :type bias_initializer: jax.nn.initializers.Initializer
    :param sigmoid_fn: Sigmoid function to use. Defaults to the logistic function.
    :type sigmoid_fn: typing.Callable

    References:

    - *Language modeling with gated convolutional networks*. 2016.
      https://arxiv.org/abs/1612.08083
    """

    def __init__(
        self,
        input_dimension,
        output_dimension,
        kernel_initializer=nn.initializers.he_normal(),
        bias_initializer=nn.initializers.zeros,
        sigmoid_fn=nn.sigmoid,
    ):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.sigmoid_fn = sigmoid_fn

    def init(self, key):
        keys = random.split(key, 4)
        w = self.kernel_initializer(
            keys[0], (self.input_dimension, self.output_dimension)
        )
        v = self.kernel_initializer(
            keys[1], (self.input_dimension, self.output_dimension)
        )
        b = self.bias_initializer(keys[2], (self.output_dimension,))
        c = self.bias_initializer(keys[3], (self.output_dimension,))
        wv = jnp.concatenate([w, v], 1)
        bc = jnp.concatenate([b, c], 1)
        return wv, bc

    def apply(self, param, input):
        wv, bc = param
        x = input @ wv + bc
        y, z = jnp.split(x, [self.output_dimension])
        return y * self.sigmoid_fn(z)


class PReLU:
    r"""
    Parametric ReLU.

    Computes

    .. math::
        y = \begin{cases}
            x & x > 0 \\
            a x & x \leq 0
        \end{cases}

    where :math:`a` is a learned slope parameter.

    :param initializer: Initializer to use for the slope parameter.
    :type initializer: jax.nn.initializers.Initializer
    :param regularizer: Regularizer to use for the slope parameter.
    :type regularizer: typing.Callable

    References:

    - *Delving deep into rectifiers: surpassing human-level performance on ImageNet classification*.
      2015. https://arxiv.org/abs/1502.01852.
    """

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


def LeNet(input_channels=1, outputs=10):
    """
    LeNet convolutional network.

    Handles images of size 28 × 28.

    Originally designed for grayscale MNIST images and 10 classes.

    :param input_channels: Number of input channels.
    :type input_channels: int
    :param outputs: Number of outputs.
    :type outputs: int

    References:

    - *Gradient-based learning applied to document recognition*. 2002.
      https://ieeexplore.ieee.org/document/726791.

    - *The MNIST database of handwritten digit images for machine learning
      research*. 2012. https://ieeexplore.ieee.org/document/6296535.
    """
    return Chain(
        [
            Conv(input_channels, 6, (5, 5), padding="SAME"),
            Bias(6),
            Func(nn.tanh),
            Func(mean_pool((2, 2), strides=(2, 2))),
            Conv(6, 16, (5, 5)),
            Bias(16),
            Func(nn.tanh),
            Func(mean_pool((2, 2), strides=(2, 2))),
            Func(jnp.ravel),
            Linear(400, 120),
            Bias(120),
            Func(nn.tanh),
            Linear(120, 84),
            Bias(84),
            Func(nn.tanh),
            Linear(84, 10),
            Bias(outputs),
        ]
    )


def AlexNet(input_channels=3, outputs=1000):
    """
    AlexNet convolutional network.

    Handles images of size 224 × 224.

    Originally designed for RGB ImageNet images and 1000 classes.

    :param input_channels: Number of input channels.
    :type input_channels: int
    :param outputs: Number of outputs.
    :type outputs: int

    References:

    - *ImageNet classification with deep convolutional neural networks*. 2017.
      https://dl.acm.org/doi/10.1145/3065386.

    - *ImageNet: A large-scale hierarchical image database*. 2009.
      https://ieeexplore.ieee.org/document/5206848.
    """
    return Chain(
        [
            Conv(input_channels, 96, (11, 11), (4, 4)),
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
            Linear(6400, 4096),
            Bias(4096),
            Func(nn.relu),
            # dropout 0.5
            Linear(4096, 4096),
            Bias(4096),
            Func(nn.relu),
            # dropout 0.5
            Linear(4096, 1000),
            Bias(outputs),
        ]
    )
