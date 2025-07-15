# Usage

Example:

```python3
from jax import numpy as jnp, random
import synax

# Create a module.
module = synax.MLP([2, 32, 3])

# Create a PRNG key.
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

A module has the following methods:

- ``init`` takes a [JAX PRNG key](https://docs.jax.dev/en/latest/_autosummary/jax.random.key.html) and returns initial parameters for the module.

- ``apply`` takes the module's parameters, together with any inputs, and returns the output of the module.

Here is an example of a custom module:

```python3
from jax import random, nn

class Affine:

    def __init__(
        self,
        input_dim,
        output_dim,
        weight_init=nn.initializers.he_normal(),
        bias_init=nn.initializers.zeros,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_init = weight_init
        self.bias_init = bias_init

    def init(self, key):
        keys = random.split(key)
        weight = self.weight_init(keys[0], (self.input_dim, self.output_dim))
        bias = self.bias_init(keys[1], (self.output_dim,))
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
