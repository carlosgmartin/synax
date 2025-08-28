# Usage

The following script is a basic example of how to use the library:

```{literalinclude} ../examples/basic.py
:language: python
:linenos:
```

Output:

```
[-1.2567853  -0.80044776  0.5694267 ]
```

A module has the following methods:

- ``init_params`` takes a [JAX PRNG key](https://docs.jax.dev/en/latest/_autosummary/jax.random.key.html) and returns initial parameters for the module.

- ``apply`` takes the module's parameters, together with any inputs, and returns the output of the module.

## Defining a custom module

The following script shows how to define a custom module:

```{literalinclude} ../examples/module.py
:language: python
:linenos:
```

Output:

```
{'weight': Array([[ 0.8737965 , -0.79177886],
       [-0.65683264, -1.0112412 ],
       [-0.7620363 ,  0.5188657 ]], dtype=float32), 'bias': Array([0., 0.], dtype=float32)}
```

## Example: Training on MNIST

The following script trains a model on the [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist).

```{literalinclude} ../examples/mnist.py
:language: python
:linenos:
```
