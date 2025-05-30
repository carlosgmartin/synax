from jax import numpy as jnp
from jax import random

import jax_modules as jm

key = random.key(0)


def test_bias(d=10):
    module = jm.Bias(d)
    x = jnp.empty(d)
    w = module.init(key)
    y = module.apply(w, x)
    assert y.shape == x.shape


def test_convolution(input_dim=3, output_dim=4, spatial_dims=(20, 18)):
    module = jm.Convolution(input_dim, output_dim, (3, 3))
    x = jnp.empty(spatial_dims + (input_dim,))
    w = module.init(key)
    y = module.apply(w, x)
    assert y.shape == spatial_dims[:-2] + (
        spatial_dims[-2] - 2,
        spatial_dims[-1] - 2,
        output_dim,
    )


def test_function(shape=(2, 3, 5), f=lambda x: x * x):
    module = jm.Function(f)
    x = jnp.empty(shape)
    w = module.init(key)
    y = module.apply(w, x)
    assert (y == f(x)).all()


def test_linear(input_dim=10, output_dim=20):
    module = jm.Linear(input_dim, output_dim)
    x = jnp.empty(input_dim)
    w = module.init(key)
    y = module.apply(w, x)
    assert y.shape == (output_dim,)


def test_parallel(input_dim=3, output_dim_1=5, output_dim_2=7):
    module = jm.Parallel(
        [jm.Linear(input_dim, output_dim_1), jm.Linear(input_dim, output_dim_2)]
    )
    x = jnp.empty(input_dim)
    w = module.init(key)
    y1, y2 = module.apply(w, (x, x))
    assert y1.shape == (output_dim_1,)
    assert y2.shape == (output_dim_2,)


def test_chain_identity(dim=5):
    module = jm.Chain([])
    x = jnp.arange(dim)
    w = module.init(key)
    y = module.apply(w, x)
    assert (y == x).all()


def test_chain(input_dim=3, hidden_dim=5, output_dim=7):
    module = jm.Chain(
        [jm.Linear(input_dim, hidden_dim), jm.Linear(hidden_dim, output_dim)]
    )
    x = jnp.empty(input_dim)
    w = module.init(key)
    y = module.apply(w, x)
    assert y.shape == (output_dim,)


def test_lenet():
    module = jm.LeNet()
    x = jnp.empty((28, 28, 1))
    w = module.init(key)
    y = module.apply(w, x)
    assert y.shape == (10,)


def test_alexnet():
    module = jm.AlexNet()
    x = jnp.empty((224, 224, 3))
    w = module.init(key)
    y = module.apply(w, x)
    assert y.shape == (1000,)


def test_mlp():
    module = jm.MultiLayerPerceptron([3, 4, 5, 6])
    x = jnp.empty(3)
    w = module.init(key)
    y = module.apply(w, x)
    assert y.shape == (6,)
