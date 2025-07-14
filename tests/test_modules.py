from jax import numpy as jnp
from jax import random

import synax

key = random.key(0)


def test_bias(d=10):
    module = synax.Bias(d)
    param = module.init(key)
    x = jnp.empty(d)
    y = module.apply(param, x)
    assert y.shape == x.shape


def test_conv(input_dim=3, output_dim=4, spatial_dims=(20, 18)):
    module = synax.Conv(input_dim, output_dim, (3, 3))
    param = module.init(key)
    x = jnp.empty(spatial_dims + (input_dim,))
    y = module.apply(param, x)
    assert y.shape == spatial_dims[:-2] + (
        spatial_dims[-2] - 2,
        spatial_dims[-1] - 2,
        output_dim,
    )


def test_max_pool():
    x = jnp.zeros((10, 8, 3))
    f = synax.max_pool((2, 2))
    y = f(x)
    assert y.shape == (9, 7, 3)


def test_mean_pool():
    x = jnp.zeros((10, 8, 3))
    f = synax.mean_pool((2, 2))
    y = f(x)
    assert y.shape == (9, 7, 3)


def test_function(shape=(2, 3, 5), f=lambda x: x * x):
    module = synax.Func(f)
    param = module.init(key)
    x = jnp.empty(shape)
    y = module.apply(param, x)
    assert (y == f(x)).all()


def test_dense(input_dim=10, output_dim=20):
    module = synax.Linear(input_dim, output_dim)
    param = module.init(key)
    x = jnp.empty(input_dim)
    y = module.apply(param, x)
    assert y.shape == (output_dim,)


def test_parallel(input_dim=3, output_dim_1=5, output_dim_2=7):
    module = synax.Parallel(
        [synax.Linear(input_dim, output_dim_1), synax.Linear(input_dim, output_dim_2)]
    )
    param = module.init(key)
    x = jnp.empty(input_dim)
    y1, y2 = module.apply(param, [x, x])
    assert y1.shape == (output_dim_1,)
    assert y2.shape == (output_dim_2,)


def test_chain_identity(dim=5):
    module = synax.Chain([])
    param = module.init(key)
    x = jnp.arange(dim)
    y = module.apply(param, x)
    assert (y == x).all()


def test_chain(input_dim=3, hidden_dim=5, output_dim=7):
    module = synax.Chain(
        [synax.Linear(input_dim, hidden_dim), synax.Linear(hidden_dim, output_dim)]
    )
    param = module.init(key)
    x = jnp.empty(input_dim)
    y = module.apply(param, x)
    assert y.shape == (output_dim,)


def test_lenet():
    module = synax.LeNet()
    param = module.init(key)
    x = jnp.empty((28, 28, 1))
    y = module.apply(param, x)
    assert y.shape == (10,)


def test_alexnet():
    module = synax.AlexNet()
    param = module.init(key)
    x = jnp.empty((224, 224, 3))
    y = module.apply(param, x)
    assert y.shape == (1000,)


def test_mlp():
    module = synax.MLP([3, 4, 5, 6])
    param = module.init(key)
    x = jnp.empty(3)
    y = module.apply(param, x)
    assert y.shape == (6,)


def test_attention(
    query_input_dim=3,
    key_input_dim=5,
    value_input_dim=7,
    hidden_dim=11,
    heads=13,
    source_len=17,
    target_len=19,
):
    module = synax.Attention(
        query_input_dim=query_input_dim,
        key_input_dim=key_input_dim,
        value_input_dim=value_input_dim,
        hidden_dim=hidden_dim,
        heads=heads,
    )
    param = module.init(key)
    query_input = jnp.empty((target_len, query_input_dim))
    key_input = jnp.empty((source_len, key_input_dim))
    value_input = jnp.empty((source_len, value_input_dim))
    output = module.apply(
        param,
        query_input=query_input,
        key_input=key_input,
        value_input=value_input,
    )
    assert output.shape == (target_len, heads * hidden_dim)
