"""
pip install tensorflow-datasets optax matplotlib
"""

import argparse

import jax
import optax
import synax
import tensorflow_datasets as tfds
from jax import lax, random
from jax import numpy as jnp
from jax.experimental import io_callback
from matplotlib import pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--model", type=str, default="lenet")
    p.add_argument("--optimizer", type=str, default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def sample_batch_indices(key, num_examples, batch_size):
    num_batches = num_examples // batch_size
    perm = random.permutation(key, num_examples)
    limit = num_batches * batch_size
    batch_indices = perm[:limit].reshape((num_batches, batch_size))
    remainder = perm[limit:]
    return batch_indices, remainder


def train(ds, info, model, optimizer, key, epochs, batch_size, epoch_callback):
    def get_example_loss(params, instance):
        image = instance["image"]
        label = instance["label"]
        image /= 255
        logits = model.apply(params, image)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, label)
        error = logits.argmax() != label
        return loss, {"loss": loss, "error": error}

    def get_batch_loss(params, batch):
        losses, metrics = jax.vmap(get_example_loss, [None, 0])(params, batch)
        mean_loss = losses.mean(0)
        mean_metrics = jax.tree.map(lambda x: x.mean(0), metrics)
        return mean_loss, mean_metrics

    def run_batch(state, batch):
        params = state["params"]
        opt_state = state["optimizer"]
        grads, metrics = jax.grad(get_batch_loss, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        state = {
            "params": params,
            "optimizer": opt_state,
            "batches": state["batches"] + 1,
            "epochs": state["epochs"],
        }
        return state, metrics

    def run_epoch(state, key):
        def f(state, batch_indices):
            batch = jax.tree.map(lambda x: x[batch_indices], ds["train"])
            return run_batch(state, batch)

        num_examples = info.splits["train"].num_examples
        batch_indices, _ = sample_batch_indices(key, num_examples, batch_size)
        state, train_metrics = lax.scan(f, state, batch_indices)
        state |= {"epochs": state["epochs"] + 1}
        train_metrics = jax.tree.map(lambda x: x.mean(0), train_metrics)

        _, test_metrics = get_batch_loss(state["params"], ds["test"])

        metrics = {"train": train_metrics, "test": test_metrics}

        io_callback(epoch_callback, None, metrics, state)

        return state, metrics

    def run_trial(key):
        key, subkey = random.split(key)
        params = model.init(subkey)

        opt_state = optimizer.init(params)

        state = {
            "params": params,
            "optimizer": opt_state,
            "batches": 0,
            "epochs": 0,
        }

        keys = random.split(key, epochs)
        state, metrics = lax.scan(run_epoch, state, keys)

        return state, metrics

    return run_trial(key)


def get_optimizer(args):
    match args.optimizer:
        case "adam":
            return optax.adam(args.lr)
        case other:
            raise NotImplementedError(other)


def get_model(args, info):
    image_shape = info.features["image"].shape
    num_labels = info.features["label"].num_classes
    match args.model:
        case "lenet":
            assert image_shape[:2] == (28, 28)
            return synax.LeNet(input_channels=image_shape[2], outputs=num_labels)
        case other:
            raise NotImplementedError(other)


def main(args):
    ds, info = tfds.load(args.dataset, batch_size=-1, with_info=True)  # type: ignore
    print(info.description)
    ds = tfds.as_numpy(ds)
    ds = jax.tree.map(jnp.asarray, ds)

    model = get_model(args, info)
    optimizer = get_optimizer(args)
    key = random.key(args.seed)

    def epoch_callback(metrics, state):
        print(f"epochs: {state['epochs']}")
        print(f"batches: {state['batches']}")
        print(f"train loss: {metrics['train']['loss']:g}")
        print(f"test loss: {metrics['test']['loss']:g}")
        print(f"train error: {metrics['train']['error']:g}")
        print(f"test error: {metrics['test']['error']:g}")
        print()

    state, metrics = train(
        ds=ds,
        info=info,
        model=model,
        optimizer=optimizer,
        key=key,
        epochs=args.epochs,
        batch_size=args.batch_size,
        epoch_callback=epoch_callback,
    )

    for metric_name in ["loss", "error"]:
        fig, ax = plt.subplots()
        ax.set(xlabel="epoch", ylabel=metric_name)
        for split in metrics.keys():
            ax.plot(metrics[split][metric_name], label=split)
        ax.legend()

    plt.show()


if __name__ == "__main__":
    main(parse_args())
