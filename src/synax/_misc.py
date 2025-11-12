from functools import partial
from typing import Any, Callable, Sequence

import jax
from jax import Array, nn, random
from jax import numpy as jnp
from jax.nn.initializers import Initializer

from ._basic import Bias, Conv, Func, Linear
from ._compound import Chain
from ._recurrent import MGU
from ._regularizers import Regularizer, zero
from ._utils import max_pool, mean_pool

Module = Any
Key = Array


def MLP(
    dimensions: Sequence[int],
    activation: Module = Func(nn.relu),
    linear_initializer: Initializer = nn.initializers.he_normal(),
    bias_initializer: Initializer = nn.initializers.zeros,
    linear_regularizer: Regularizer = zero,
    bias_regularizer: Regularizer = zero,
    dtype: jax.typing.DTypeLike | None = None,
) -> Module:
    """
    Multi-layer perceptron.

    :param dimensions: Dimension of each layer.
    :param activation: Module used as activation function.
        Not applied to the output.
    :param linear_initializer: Initializer for linear layers.
    :param bias_initializer: Initializer for bias layers.
    :param linear_regularizer: Regularizer for linear layers.
    :param bias_regularizer: Regularizer for bias layers.
    :param dtype: Data type of parameters.

    :returns: Module.

    References:

    - *A logical calculus of the ideas immanent in nervous activity*. 1943.
      https://link.springer.com/article/10.1007/BF02478259.

    - *The perceptron: A probabilistic model for information storage and
      organization in the brain*. 1958.
      https://psycnet.apa.org/record/1959-09865-001.

    - *Learning representations by back-propagating errors*. 1986.
      https://www.nature.com/articles/323533a0.
    """
    modules: list[Module] = []
    for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
        linear = Linear(
            input_dim,
            output_dim,
            initializer=linear_initializer,
            regularizer=linear_regularizer,
            dtype=dtype,
        )
        bias = Bias(
            output_dim,
            initializer=bias_initializer,
            regularizer=bias_regularizer,
            dtype=dtype,
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
    :param decoder: Module to use as decoder.
    """

    def __init__(self, encoder: Module, decoder: Module):
        self.encoder = encoder
        self.decoder = decoder

    def init_params(self, key: Key) -> dict[str, Any]:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        keys = random.split(key)
        return {
            "encoder": self.encoder.init_params(keys[0]),
            "decoder": self.decoder.init_params(keys[1]),
        }

    def encode(self, params: dict[str, Any], input: Any) -> Any:
        return self.encoder.apply(params["encoder"], input)

    def decode(self, params: dict[str, Any], input: Any) -> Any:
        return self.decoder.apply(params["decoder"], input)

    def apply(self, params: dict[str, Any], input: Any) -> Any:
        return self.decode(params, self.encode(params, input))

    def reconstruction_loss(self, params: dict[str, Any], input: Any) -> Array:
        output = self.apply(params, input)
        diff = input - output
        return (diff * jnp.conj(diff)).sum()

    def param_loss(self, params: dict[str, Any]) -> Array:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        encoder_loss = self.encoder.param_loss(params["encoder"])
        decoder_loss = self.decoder.param_loss(params["decoder"])
        return encoder_loss + decoder_loss


def get_von_neumann_neighbors(
    array: Array,
    space_dim: int | None = None,
    include_center: bool = False,
) -> Array:
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
        state_dim: int,
        space_dim: int = 1,
        cell_cls: Callable[[int, int], Module] = partial(MGU, reset_gate=False),
        global_mean: bool = False,
        global_max: bool = False,
    ):
        self.cell = cell_cls(
            state_dim,
            state_dim * (2 * space_dim + global_mean + global_max),
        )
        self.global_mean = global_mean
        self.global_max = global_max

    def init_params(self, key: Key) -> Any:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        return self.cell.init_params(key)

    def apply(self, params: Any, state: Array) -> Array:
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
        new_state = self.cell.apply(params, state, inputs)
        return new_state


class GLU:
    r"""
    Gated linear unit.

    Computes

    .. math::
        y = \sigma(A_1 x + b_1) \odot (A_2 x + b_2)

    where :math:`\sigma` is a sigmoid function, :math:`A_1` and :math:`A_2` are
    learned matrices, and :math:`b_1` and :math:`b_2` are learned vectors.

    :param input_dim: Input dimension.
    :param output_dim: Output dimension.
    :param linear_initializer: Initializer for linear layers.
    :param bias_initializer: Initializer for bias layers.
    :param sigmoid_fn: Sigmoid function to use. Defaults to the logistic function.
    :param linear_regularizer: Regularizer for linear layers.
    :param bias_regularizer: Regularizer for bias layers.
    :param dtype: Data type of parameters.

    References:

    - *Language modeling with gated convolutional networks*. 2016.
      https://arxiv.org/abs/1612.08083
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        linear_initializer: Initializer = nn.initializers.he_normal(),
        bias_initializer: Initializer = nn.initializers.zeros,
        sigmoid_fn: Callable[[Array], Array] = nn.sigmoid,
        linear_regularizer: Regularizer = zero,
        bias_regularizer: Regularizer = zero,
        dtype: jax.typing.DTypeLike | None = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_initializer = linear_initializer
        self.bias_initializer = bias_initializer
        self.sigmoid_fn = sigmoid_fn
        self.linear_regularizer = linear_regularizer
        self.bias_regularizer = bias_regularizer
        self.dtype = dtype

    def init_params(self, key: Key) -> dict[str, Array]:
        """
        Sample initial parameters.

        :param key: PRNG key.

        :returns: Parameters.
        """
        keys = random.split(key, 4)
        w = self.linear_initializer(
            keys[0], (self.input_dim, self.output_dim), self.dtype
        )
        v = self.linear_initializer(
            keys[1], (self.input_dim, self.output_dim), self.dtype
        )
        b = self.bias_initializer(keys[2], (self.output_dim,), self.dtype)
        c = self.bias_initializer(keys[3], (self.output_dim,), self.dtype)
        return {
            "linear": jnp.concatenate([w, v], 1),
            "bias": jnp.concatenate([b, c], 1),
        }

    def apply(self, params: dict[str, Array], input: Array) -> Array:
        x = input @ params["linear"] + params["bias"]
        y, z = jnp.split(x, [self.output_dim])
        return y * self.sigmoid_fn(z)

    def param_loss(self, params: dict[str, Any]) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return self.linear_regularizer(params["linear"]) + self.bias_regularizer(
            params["bias"]
        )


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
    :param regularizer: Regularizer to use for the slope parameter.

    References:

    - *Delving deep into rectifiers: surpassing human-level performance on ImageNet classification*.
      2015. https://arxiv.org/abs/1502.01852.
    """

    def __init__(
        self,
        initializer: Initializer = nn.initializers.zeros,
        regularizer: Regularizer = zero,
    ):
        self.initializer = initializer
        self.regularizer = regularizer

    def init_params(self, key: Key) -> Array:
        return self.initializer(key, ())

    def apply(self, params: Array, input: Array) -> Array:
        return jnp.where(input > 0, input, input * params)

    def param_loss(self, params: Array) -> Array | float:
        """
        Parameter loss.

        :param params: Parameters.

        :returns: Scalar.
        """
        return self.regularizer(params)


def LeNet(input_channels: int = 1, outputs: int = 10) -> Module:
    """
    LeNet convolutional network.

    Handles images of size 28 × 28.

    Originally designed for grayscale MNIST images and 10 classes.

    :param input_channels: Number of input channels.
    :param outputs: Number of outputs.

    :returns: Module.

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
            Func(mean_pool((2, 2), stride=2)),
            Conv(6, 16, (5, 5)),
            Bias(16),
            Func(nn.tanh),
            Func(mean_pool((2, 2), stride=2)),
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


def AlexNet(input_channels: int = 3, outputs: int = 1000) -> Module:
    """
    AlexNet convolutional network.

    Handles images of size 224 × 224.

    Originally designed for RGB ImageNet images and 1000 classes.

    :param input_channels: Number of input channels.
    :param outputs: Number of outputs.

    :returns: Module.

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
            Func(max_pool((3, 3), stride=2)),
            Conv(96, 256, (5, 5), padding="SAME"),
            Bias(256),
            Func(nn.relu),
            Func(max_pool((3, 3), stride=2)),
            Conv(256, 384, (3, 3), padding="SAME"),
            Bias(384),
            Func(nn.relu),
            Conv(384, 384, (3, 3), padding="SAME"),
            Bias(384),
            Func(nn.relu),
            Conv(384, 256, (3, 3), padding="SAME"),
            Bias(256),
            Func(nn.relu),
            Func(max_pool((3, 3), stride=2)),
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
