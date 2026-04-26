"""
Pretrain a language model.
"""

import argparse
import time

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tiktoken

from checkpoint import create_checkpoint_manager, restore_latest_checkpoint, save_checkpoint
from config import RunConfig, load_config
from data import make_dataloaders, make_val_dataloader
from model import Model
from logs import setup_run
from sample import generate, write_sample


def loss(model: Model, input_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    return optax.softmax_cross_entropy_with_integer_labels(
        shift_logits,
        shift_labels,
    ).mean()


def tree_l2_norm(tree) -> jax.Array:
    leaves = jax.tree.leaves(tree)
    square_norms = [jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves]
    return jnp.sqrt(jnp.sum(jnp.asarray(square_norms)))


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
def train_step(model: Model, optimizer: nnx.Optimizer, input_ids: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    loss_value, grads = nnx.value_and_grad(loss)(model, input_ids)
    grad_norm = tree_l2_norm(grads)
    optimizer.update(model, grads)
    param_norm = tree_l2_norm(nnx.state(model, nnx.Param))
    return loss_value, {
        "train/grad_norm": grad_norm,
        "train/param_norm": param_norm,
    }


@nnx.jit
def eval_step(model: Model, input_ids: jax.Array) -> jax.Array:
    return loss(model, input_ids)


def evaluate(model: Model, val_iter, eval_steps: int) -> jax.Array:
    losses = [eval_step(model, next(val_iter)["input_ids"]) for _ in range(eval_steps)]
    return jnp.mean(jnp.asarray(losses))


def print_startup(config: RunConfig, *, resume: bool):
    line = "=" * 72
    mode = "resume" if resume else "new run"
    tokens_per_step = config.train.batch_size * config.train.seq_len
    total_tokens = config.train.steps * tokens_per_step

    print(line)
    print("Pretraining")
    print(line)
    print(f"run:        {config.experiment.name} ({mode})")
    print(f"output:     {config.experiment.out_dir}/{config.experiment.name}")
    print(f"model:      layers={config.model.n_layers} hidden={config.model.hidden_size} heads={config.model.n_heads} kv_heads={config.model.n_kv_heads}")
    print(f"context:    seq_len={config.model.seq_len} vocab={config.model.vocab_size}")
    print(f"train:      steps={config.train.steps} batch={config.train.batch_size} seq_len={config.train.seq_len} tokens={total_tokens}")
    print(f"optimizer:  muon lr={config.train.lr} weight_decay={config.train.decay}")
    print(f"eval:       every={config.train.eval_every} steps={config.train.eval_steps}")
    print(f"data:       {config.data.path} tokenizer={config.data.tokenizer} val_fraction={config.data.val_fraction}")
    print(f"samples:    {'on' if config.sampling.enabled else 'off'}")
    print(f"wandb:      {'on' if config.wandb.enabled else 'off'}")
    print(line)


def metric_header() -> str:
    return f"{'step':>6} | {'loss':>10} | {'ppl':>10} | {'grad':>9} | {'tok/s':>8} | {'val':>10}"


def format_metrics_row(metrics: dict) -> str:
    val_loss = metrics.get("val/loss")
    val_text = f"{val_loss:>10.4f}" if val_loss is not None else f"{'':>10}"
    return (
        f"{metrics['step']:>6} | "
        f"{metrics['train/loss']:>10.4f} | "
        f"{metrics['train/ppl']:>10.2f} | "
        f"{metrics['train/grad_norm']:>9.2f} | "
        f"{metrics['time/tokens_per_sec']:>8.0f} | "
        f"{val_text}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a run config TOML file.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint.")
    args = parser.parse_args()

    config = load_config(args.config)
    print_startup(config, resume=args.resume)
    logger = setup_run(args.config, config, resume=args.resume)
    print("Compiling first step, then training...\n")
    printed_metric_header = False
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
    train_iter, _ = make_dataloaders(config.data, train_config)
    checkpoint_manager = create_checkpoint_manager(logger.run_dir, train_config.keep_last)

    start_step = 0
    if args.resume:
        start_step = restore_latest_checkpoint(
            checkpoint_manager,
            model=model,
            optimizer=optimizer,
            train_iter=train_iter,
        )
        print(f"resumed from checkpoint; starting at step {start_step}")

    try:
        for step in range(start_step, train_config.steps):
            batch = next(train_iter)
            logger.log_batch(step, batch)

            start_time = time.perf_counter()
            loss_value, train_metrics = train_step(model, optimizer, batch["input_ids"])
            step_sec = time.perf_counter() - start_time

            if step % train_config.log_every == 0:
                loss_float = float(loss_value)
                tokens_per_sec = train_config.batch_size * train_config.seq_len / step_sec
                metrics = {
                    "step": step,
                    "train/loss": loss_float,
                    "train/ppl": float(jnp.exp(loss_value)),
                    "train/grad_norm": float(train_metrics["train/grad_norm"]),
                    "train/param_norm": float(train_metrics["train/param_norm"]),
                    "train/tokens_seen": (step + 1) * train_config.batch_size * train_config.seq_len,
                    "optim/lr": train_config.lr,
                    "time/step_sec": step_sec,
                    "time/tokens_per_sec": tokens_per_sec,
                }

                if step % train_config.eval_every == 0:
                    val_iter = make_val_dataloader(config.data, train_config)
                    val_loss = evaluate(model, val_iter, train_config.eval_steps)
                    val_loss_float = float(val_loss)
                    metrics["val/loss"] = val_loss_float
                    metrics["val/ppl"] = float(jnp.exp(val_loss))

                    if config.sampling.enabled:
                        tokenizer = tiktoken.get_encoding(config.data.tokenizer)
                        sample_key = jax.random.fold_in(jax.random.key(train_config.seed), step)
                        sample_text = generate(
                            model,
                            model_config,
                            config.sampling,
                            tokenizer,
                            sample_key,
                        )
                        sample_path = write_sample(
                            logger.run_dir,
                            step,
                            config.sampling.prompt,
                            sample_text,
                        )
                        metrics["sample/path"] = str(sample_path)

                if not printed_metric_header:
                    print(metric_header())
                    printed_metric_header = True
                print(format_metrics_row(metrics))
                logger.log(metrics)

            next_step = step + 1
            if train_config.checkpoint_every > 0 and next_step % train_config.checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_manager,
                    next_step=next_step,
                    model=model,
                    optimizer=optimizer,
                    train_iter=train_iter,
                )

    finally:
        checkpoint_manager.wait_until_finished()
        logger.close()


if __name__ == "__main__":
    main()
