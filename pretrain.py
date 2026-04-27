"""
Pretrain a language model.
"""

import argparse

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tiktoken

from checkpoint import create_checkpoint_manager, restore_latest_checkpoint, save_checkpoint
from config import RunConfig, load_config
from data import make_dataloaders, make_val_dataloader
from distributed import DistributedContext, create_distributed_context, place_replicated_state, shard_batch
from lr_schedule import build_lr_schedule, describe_lr_schedule
from model import Model
from logs import setup_run
from profiling import StepTimer
from sample import generate, write_sample


def loss(model: Model, input_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :].astype(model.loss_dtype)
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


def evaluate(model: Model, val_iter, eval_steps: int, distributed: DistributedContext) -> jax.Array:
    losses = [eval_step(model, shard_batch(next(val_iter), distributed)["input_ids"]) for _ in range(eval_steps)]
    return jnp.mean(jnp.asarray(losses))


def print_startup(config: RunConfig, *, resume: bool, distributed: DistributedContext | None = None):
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
    if distributed is not None:
        print(
            f"distributed: devices={distributed.device_count} "
            f"global_batch={distributed.global_batch_size} "
            f"per_device_batch={distributed.per_device_batch_size} "
            f"axis={distributed.axis_name}"
        )
    print(f"optimizer:  muon peak_lr={config.train.lr} weight_decay={config.train.decay}")
    print(f"schedule:   {describe_lr_schedule(config.train)}")
    print(f"precision:  compute={config.precision.compute_dtype} params={config.precision.param_dtype} loss={config.precision.loss_dtype}")
    print(f"profiling:  enabled={config.profiling.enabled} profiler={config.profiling.profiler}")
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


def add_timing_metrics(metrics: dict, timer: StepTimer, train_config) -> dict:
    tokens_per_step = train_config.batch_size * train_config.seq_len
    step_sec = timer.get("step")
    train_step_sec = timer.get("train_step")
    metrics.update(timer.metrics())
    metrics["time/step_sec"] = step_sec
    metrics["time/tokens_per_sec"] = tokens_per_step / step_sec if step_sec > 0.0 else 0.0
    metrics["time/train_tokens_per_sec"] = tokens_per_step / train_step_sec if train_step_sec > 0.0 else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a run config TOML file.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint.")
    args = parser.parse_args()

    config = load_config(args.config)
    train_config = config.train
    model_config = config.model
    lr_schedule = build_lr_schedule(train_config)
    distributed = create_distributed_context(config.distributed, train_config)
    print_startup(config, resume=args.resume, distributed=distributed)
    logger = setup_run(args.config, config, resume=args.resume)
    print("Compiling first step, then training...\n")
    printed_metric_header = False

    model = Model(model_config, precision=config.precision, rngs=nnx.Rngs(train_config.seed))
    tx = optax.contrib.muon(
        learning_rate=lr_schedule,
        weight_decay=train_config.decay,
        adam_weight_decay=train_config.decay,
        muon_weight_dimension_numbers=make_muon_dimension_numbers(model_config),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    place_replicated_state(model, optimizer, distributed)
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
        place_replicated_state(model, optimizer, distributed)
        print(f"resumed from checkpoint; starting at step {start_step}")

    try:
        for step in range(start_step, train_config.steps):
            timer = StepTimer()
            metrics = None
            with timer.phase("step"):
                with timer.phase("data"):
                    batch = next(train_iter)
                with timer.phase("batch_log"):
                    logger.log_batch(step, batch)
                with timer.phase("shard"):
                    sharded_batch = shard_batch(batch, distributed)
                with timer.phase("train_step"):
                    loss_value, train_metrics = train_step(model, optimizer, sharded_batch["input_ids"])

                should_log = step % train_config.log_every == 0
                if should_log:
                    with timer.phase("metrics_sync"):
                        metrics = {
                            "step": step,
                            "train/loss": float(loss_value),
                            "train/ppl": float(jnp.exp(loss_value)),
                            "train/grad_norm": float(train_metrics["train/grad_norm"]),
                            "train/param_norm": float(train_metrics["train/param_norm"]),
                            "train/tokens_seen": (step + 1) * train_config.batch_size * train_config.seq_len,
                            "optim/lr": float(lr_schedule(step)),
                        }

                    if step % train_config.eval_every == 0:
                        with timer.phase("eval"):
                            val_iter = make_val_dataloader(config.data, train_config)
                            val_loss = evaluate(model, val_iter, train_config.eval_steps, distributed)
                        with timer.phase("metrics_sync"):
                            metrics["val/loss"] = float(val_loss)
                            metrics["val/ppl"] = float(jnp.exp(val_loss))

                        if config.sampling.enabled:
                            with timer.phase("sample"):
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

                next_step = step + 1
                if train_config.checkpoint_every > 0 and next_step % train_config.checkpoint_every == 0:
                    with timer.phase("checkpoint"):
                        save_checkpoint(
                            checkpoint_manager,
                            next_step=next_step,
                            model=model,
                            optimizer=optimizer,
                            train_iter=train_iter,
                        )

            if metrics is not None:
                add_timing_metrics(metrics, timer, train_config)
                if not printed_metric_header:
                    print(metric_header())
                    printed_metric_header = True
                print(format_metrics_row(metrics))
                logger.log(metrics)

    finally:
        checkpoint_manager.wait_until_finished()
        logger.close()


if __name__ == "__main__":
    main()
