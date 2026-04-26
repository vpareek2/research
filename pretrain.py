"""
Pretrain a language model.
"""

import argparse

import jax
from flax import nnx
import optax

from config import load_config
from data import make_dataloader
from model import Model


def loss(model: Model, input_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    return optax.softmax_cross_entropy_with_integer_labels(
        shift_logits,
        shift_labels,
    ).mean()


def make_muon_dimension_numbers(model_config):
    # Optax Muon defaults to every 2D parameter, but for LMs we only want
    # hidden projection matrices. Vocab matrices like embed/lm_head stay on
    # AdamW. Later we should replace this shape rule with cleaner name-based
    # routing, or our own optimizer wrapper.
    def dim_numbers(params):
        def leaf(x):
            if x.ndim == 2 and model_config.vocab_size not in x.shape:
                return optax.contrib.MuonDimensionNumbers()
            return None

        return jax.tree.map(leaf, params)

    return dim_numbers


@nnx.jit
def train_step(model: Model, optimizer: nnx.Optimizer, input_ids: jax.Array) -> jax.Array:
    loss_value, grads = nnx.value_and_grad(loss)(model, input_ids)
    optimizer.update(model, grads)
    return loss_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a run config TOML file.")
    args = parser.parse_args()

    config = load_config(args.config)
    train_config = config.train
    model_config = config.model

    model = Model(model_config, rngs=nnx.Rngs(train_config.seed))
    tx = optax.contrib.muon(
        learning_rate=train_config.lr,
        weight_decay=train_config.decay,
        adam_weight_decay=train_config.decay,
        muon_weight_dimension_numbers=make_muon_dimension_numbers(model_config),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    train_iter = make_dataloader(config.data, train_config)

    for step in range(train_config.steps):
        input_ids = next(train_iter)
        loss_value = train_step(model, optimizer, input_ids)

        if step % train_config.log_every == 0:
            print(f"step {step}: loss={float(loss_value):.4f}")


if __name__ == "__main__":
    main()
