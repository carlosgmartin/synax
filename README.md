# JAX modules

A magic-free neural network library for [JAX](https://github.com/jax-ml/jax).

No tracing. No transforms. Just purely-functional JAX.

## Installation

```shell
python3 -m pip install git+https://github.com/carlosgmartin/jax_modules
```

## Example

```python3
from jax import numpy as jnp, random
import jax_modules as jm

# Create a module.
module = jm.MLP([2, 32, 3])
# Create a random key.
key = random.key(0)
# Initialize parameters.
w = module.init(key)
# Define an input.
x = jnp.ones(2)
# Compute the output.
y = module.apply(w, x)
```

## Codebase quality control

Run the following after every change:

```shell
ruff check && ruff format && pyright && pytest
```

## Documentation

Build documentation:

```shell
sphinx-build -M html docs docs/_build
```
