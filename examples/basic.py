from jax import numpy as jnp, random
import synax

# Create a module.
module = synax.MLP([2, 32, 3])

# Create a PRNG key.
key = random.key(0)

# Sample initial parameters.
w = module.init(key)

# Define an input.
x = jnp.ones(2)

# Compute the output.
y = module.apply(w, x)

# Print the output.
print(y)
