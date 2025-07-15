# Usage

Example:

```python3
from jax import numpy as jnp, random
import synax

# Create a module.
module = synax.MLP([2, 32, 3])

# Create a random key.
key = random.key(0)

# Initialize parameters.
w = module.init(key)

# Define an input.
x = jnp.ones(2)

# Compute the output.
y = module.apply(w, x)

# Print the output.
print(y)
```

```
[-1.2567853  -0.80044776  0.5694267 ]
```

Modules have at least two methods: ``init`` and ``apply``.

``init`` takes a [JAX PRNG key](https://docs.jax.dev/en/latest/_autosummary/jax.random.key.html) and samples initial parameters for the module.

``apply`` takes these parameters, together with any other inputs, and yields the output of the module.

Here is an example custom module:

```python3
from jax import random, nn
import synax

class Affine:

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

    def init(self, key):
        keys = random.split(key)
        weight = nn.initializers.he_normal()(keys[0], (self.in_dim, self.out_dim))
        bias = nn.initializers.zeros(keys[1], (self.out_dim,))
        return {"weight": weight, "bias": bias}

    def apply(self, params, input):
        output = input @ params["weight"] + params["bias"]
        return output

module = Affine(3, 2)
key = random.key(0)
params = module.init(key)
print(params)
```

```
{'weight': Array([[ 0.8737965 , -0.79177886],
       [-0.65683264, -1.0112412 ],
       [-0.7620363 ,  0.5188657 ]], dtype=float32), 'bias': Array([0., 0.], dtype=float32)}
```
