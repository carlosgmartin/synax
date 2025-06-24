# JAX modules

A magic-free neural network library for [JAX](https://github.com/jax-ml/jax).

## Installation

```shell
python3 -m pip install git+https://github.com/carlosgmartin/jax_modules
```

## Example

```python3
from jax import numpy as jnp, random
import jax_modules as jm

module = jm.MultiLayerPerceptron([2, 32, 3])

key = random.key(0)
param = module.init(key)

x = jnp.ones(2)
y = module.apply(param, x)
print(y)
```

## Codebase quality control

Run the following after every change:

```shell
ruff check && ruff format && pyright && pytest
```
