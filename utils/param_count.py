"""
Model parameter count utilities.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from config import ModelConfig, load_config


@dataclass(frozen=True)
class ParamBreakdown:
    token_embedding: int
    lm_head: int
    attention: int
    mlp: int
    block_norms: int
    final_norm: int

    @property
    def norms(self) -> int:
        return self.block_norms + self.final_norm

    @property
    def transformer_blocks(self) -> int:
        return self.attention + self.mlp + self.block_norms

    @property
    def vocab_matrices(self) -> int:
        return self.token_embedding + self.lm_head

    @property
    def total(self) -> int:
        return self.vocab_matrices + self.transformer_blocks + self.final_norm


def count_params(config: ModelConfig) -> ParamBreakdown:
    head_dim = config.hidden_size // config.n_heads
    kv_width = config.n_kv_heads * head_dim

    token_embedding = config.vocab_size * config.hidden_size
    lm_head = 0 if config.tied else config.hidden_size * config.vocab_size

    q_proj = config.hidden_size * config.hidden_size
    k_proj = config.hidden_size * kv_width
    v_proj = config.hidden_size * kv_width
    o_proj = config.hidden_size * config.hidden_size
    qk_norms = 2 * head_dim
    attention_per_layer = q_proj + k_proj + v_proj + o_proj + qk_norms

    mlp_per_layer = 3 * config.hidden_size * config.intermediate_size
    block_norms_per_layer = 2 * config.hidden_size

    return ParamBreakdown(
        token_embedding=token_embedding,
        lm_head=lm_head,
        attention=config.n_layers * attention_per_layer,
        mlp=config.n_layers * mlp_per_layer,
        block_norms=config.n_layers * block_norms_per_layer,
        final_norm=config.hidden_size,
    )


def format_count(value: int) -> str:
    return f"{value:,}"


def format_millions(value: int) -> str:
    return f"{value / 1_000_000:.3f}M"


def _row(name: str, value: int, total: int) -> str:
    pct = 100.0 * value / total if total else 0.0
    return f"{name:<20} {format_count(value):>16} {format_millions(value):>10} {pct:>7.2f}%"


def format_breakdown(config: ModelConfig, breakdown: ParamBreakdown) -> str:
    total = breakdown.total
    head_dim = config.hidden_size // config.n_heads
    lines = [
        "Model",
        f"  vocab={config.vocab_size:,} hidden={config.hidden_size:,} intermediate={config.intermediate_size:,}",
        f"  layers={config.n_layers:,} heads={config.n_heads:,} kv_heads={config.n_kv_heads:,} head_dim={head_dim:,}",
        f"  seq_len={config.seq_len:,} tied_embeddings={config.tied}",
        "",
        f"{'component':<20} {'params':>16} {'params(M)':>10} {'share':>8}",
        "-" * 57,
        _row("token_embedding", breakdown.token_embedding, total),
        _row("lm_head", breakdown.lm_head, total),
        _row("attention", breakdown.attention, total),
        _row("mlp", breakdown.mlp, total),
        _row("norms", breakdown.norms, total),
        "-" * 57,
        _row("total", total, total),
        "",
        f"non_embedding_params: {format_count(total - breakdown.vocab_matrices)} ({format_millions(total - breakdown.vocab_matrices)})",
        f"vocab_matrix_params: {format_count(breakdown.vocab_matrices)} ({format_millions(breakdown.vocab_matrices)})",
    ]
    return "\n".join(lines)


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument("config", nargs="?", help="Optional run config TOML file.")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--layers", "--n-layers", dest="n_layers", type=int)
    parser.add_argument("--heads", "--n-heads", dest="n_heads", type=int)
    parser.add_argument("--kv-heads", "--n-kv-heads", dest="n_kv_heads", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--theta", type=float, default=1_000_000.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--tied", action="store_true", help="Count tied input/output embeddings.")


def config_from_args(args: argparse.Namespace) -> ModelConfig:
    if args.config is not None:
        base = load_config(args.config).model
        values = base.__dict__.copy()
    else:
        values = {
            "vocab_size": None,
            "hidden_size": None,
            "intermediate_size": None,
            "n_layers": None,
            "n_heads": None,
            "n_kv_heads": None,
            "seq_len": 2048,
            "theta": args.theta,
            "eps": args.eps,
            "tied": False,
        }

    for key in ("vocab_size", "hidden_size", "intermediate_size", "n_layers", "n_heads", "n_kv_heads", "seq_len"):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    if args.config is None or args.theta != 1_000_000.0:
        values["theta"] = args.theta
    if args.config is None or args.eps != 1e-6:
        values["eps"] = args.eps
    if args.tied:
        values["tied"] = True

    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise SystemExit(f"missing model args without config: {', '.join(missing)}")

    return ModelConfig(**values)


def main():
    parser = argparse.ArgumentParser(description="Count model parameters from a config or direct model dimensions.")
    add_model_args(parser)
    args = parser.parse_args()

    config = config_from_args(args)
    print(format_breakdown(config, count_params(config)))


if __name__ == "__main__":
    main()
