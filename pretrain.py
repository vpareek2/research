"""
Pretrain a language model.
"""

import jax
from flax import nnx
import optax

from config import load_config
from model import Model
    
def loss(model: Model, input_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    return optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels,).mean()

@nnx.jit
def train_step(model: Model, optimizer: nnx.Optimizer, input_ids: jax.Array) -> jax.Array:
    loss_value, grads = nnx.value_and_grad(loss)(model, input_ids)
    optimizer.update(model, grads)
    return loss_value


def main():
    config = load_config("config.toml")
    train_config = config.train
    model_config = config.model

    key = jax.random.key(train_config.seed)
    model = Model(model_config, rngs=nnx.Rngs(train_config.seed))
    tx = optax.contrib.muon(
        learning_rate=train_config.lr,
        weight_decay=train_config.decay,
        adam_weight_decay=train_config.decay,
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    key, batch_key = jax.random.split(key)
    input_ids = jax.random.randint(
        batch_key,
        shape=(train_config.batch_size, train_config.seq_len),
        minval=0,
        maxval=model_config.vocab_size,
    )

    for step in range(train_config.steps):
        loss_value = train_step(model, optimizer, input_ids)

        if step % train_config.log_every == 0:
            print(f"step {step}: loss={float(loss_value):.4f}")


if __name__ == "__main__":
    main()
