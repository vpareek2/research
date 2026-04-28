"""
Language Model Definition.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from config import ModelConfig, PrecisionConfig, dtype_from_name
from kv_cache import KVCache, LayerKVCache

def _precompute_rope(seq_len, head_dim, theta, dtype):
    """
    Precompute the cos/sin values for the RoPE rotation.
    """
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)) # [head_dim // 2]
    positions = jnp.arange(seq_len, dtype=jnp.float32) # [seq_len]

    freqs = jnp.outer(positions, inv_freq) # [seq_len, head_dim // 2]
    cos = jnp.cos(freqs).astype(dtype) # [seq_len, head_dim // 2]
    sin = jnp.sin(freqs).astype(dtype) # [seq_len, head_dim // 2]
    return cos, sin

def rope(x, cos, sin):
    """
    Apply rope to tensor x: [batch, seq, heads, head_dim].
    """
    cos = cos[:, None, :] # [seq, 1, head_dim // 2]
    sin = sin[:, None, :] # [seq, 1, head_dim // 2]

    x_e = x[..., 0::2] # [batch, seq, heads, head_dim // 2]
    x_o = x[..., 1::2] # [batch, seq, heads, head_dim // 2]
    x_r = jnp.stack((x_e * cos - x_o * sin, x_e * sin + x_o * cos), axis=-1)
    return x_r.reshape(x.shape)


def _repeat_kv_heads(x: jax.Array, n_heads: int, n_kv_heads: int) -> jax.Array:
    if n_heads == n_kv_heads:
        return x
    return jnp.repeat(x, n_heads // n_kv_heads, axis=2)

class SwiGLU(nnx.Module):
    """
    SwiGLU activated MLP.
    """
    def __init__(self, config: ModelConfig, precision: PrecisionConfig, rngs: nnx.Rngs):
        dtype = dtype_from_name(precision.compute_dtype)
        param_dtype = dtype_from_name(precision.param_dtype)
        self.gate = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)
        self.up = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)
        self.down = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))

class GQA(nnx.Module):
    """
    Grouped Query Attention.
    """
    def __init__(self, config: ModelConfig, precision: PrecisionConfig, rngs: nnx.Rngs):
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.hidden_size // config.n_heads
        dtype = dtype_from_name(precision.compute_dtype)
        param_dtype = dtype_from_name(precision.param_dtype)

        self.q = nnx.Linear(config.hidden_size, config.n_heads * self.head_dim, use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)
        self.k = nnx.Linear(config.hidden_size, config.n_kv_heads * self.head_dim, use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)
        self.v = nnx.Linear(config.hidden_size, config.n_kv_heads * self.head_dim, use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)
        self.o = nnx.Linear(config.hidden_size, config.hidden_size, use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)

        self.q_norm = nnx.RMSNorm(self.head_dim, epsilon=config.eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)
        self.k_norm = nnx.RMSNorm(self.head_dim, epsilon=config.eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)
    
    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        batch_size, seq_len, _ = x.shape

        q = self.q(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q = rope(self.q_norm(q), cos, sin)
        k = rope(self.k_norm(k), cos, sin)

        out = jax.nn.dot_product_attention(q, k, v, is_causal=True).reshape(batch_size, seq_len, self.hidden_size)
        return self.o(out)

    def decode_one(
        self,
        x: jax.Array,
        cache: LayerKVCache,
        position: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
    ) -> tuple[jax.Array, LayerKVCache]:
        batch_size, seq_len, _ = x.shape
        if seq_len != 1:
            raise ValueError(f"decode_one expects one token, got seq_len={seq_len}")

        q = self.q(x).reshape(batch_size, 1, self.n_heads, self.head_dim)
        k = self.k(x).reshape(batch_size, 1, self.n_kv_heads, self.head_dim)
        v = self.v(x).reshape(batch_size, 1, self.n_kv_heads, self.head_dim)

        q = rope(self.q_norm(q), cos, sin)
        k = rope(self.k_norm(k), cos, sin)

        k_cache = jax.lax.dynamic_update_slice(cache.k, k, (0, position, 0, 0))
        v_cache = jax.lax.dynamic_update_slice(cache.v, v, (0, position, 0, 0))
        new_cache = LayerKVCache(k=k_cache, v=v_cache)

        k_attn = _repeat_kv_heads(k_cache, self.n_heads, self.n_kv_heads)
        v_attn = _repeat_kv_heads(v_cache, self.n_heads, self.n_kv_heads)
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k_attn).astype(jnp.float32)
        scores = scores / jnp.sqrt(jnp.asarray(self.head_dim, dtype=jnp.float32))
        valid = jnp.arange(k_cache.shape[1]) <= position
        scores = jnp.where(valid[None, None, None, :], scores, jnp.finfo(scores.dtype).min)
        weights = jax.nn.softmax(scores, axis=-1).astype(v_attn.dtype)
        out = jnp.einsum("bhqk,bkhd->bqhd", weights, v_attn)
        out = out.reshape(batch_size, 1, self.hidden_size)
        return self.o(out), new_cache

class Block(nnx.Module):
    """
    Single Transformer Block.
    """
    def __init__(self, config: ModelConfig, precision: PrecisionConfig, rngs: nnx.Rngs):
        dtype = dtype_from_name(precision.compute_dtype)
        param_dtype = dtype_from_name(precision.param_dtype)
        self.pre_norm = nnx.RMSNorm(config.hidden_size, epsilon=config.eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)
        self.attn = GQA(config, precision, rngs,)
        self.post_norm = nnx.RMSNorm(config.hidden_size, epsilon=config.eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)
        self.mlp = SwiGLU(config, precision, rngs,)
    
    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        x = x + self.attn(self.pre_norm(x), cos, sin)
        x = x + self.mlp(self.post_norm(x))
        return x

    def decode_one(
        self,
        x: jax.Array,
        cache: LayerKVCache,
        position: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
    ) -> tuple[jax.Array, LayerKVCache]:
        attn_out, cache = self.attn.decode_one(self.pre_norm(x), cache, position, cos, sin)
        x = x + attn_out
        x = x + self.mlp(self.post_norm(x))
        return x, cache

class Model(nnx.Module):
    """
    Language Model class.
    """
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs, precision: PrecisionConfig | None = None):
        self.config = config
        self.precision = precision or PrecisionConfig()
        dtype = dtype_from_name(self.precision.compute_dtype)
        param_dtype = dtype_from_name(self.precision.param_dtype)
        self.loss_dtype = dtype_from_name(self.precision.loss_dtype)

        self.embed = nnx.Embed(config.vocab_size, config.hidden_size, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)
        self.layers = nnx.List([Block(config, self.precision, rngs=rngs,) for _ in range(config.n_layers)])
        self.norm = nnx.RMSNorm(config.hidden_size, epsilon=config.eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)
        self.lm_head = nnx.Linear(config.hidden_size, config.vocab_size, use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs,)

        # Temp
        if config.tied:
            raise NotImplementedError("Tied embeddings are not implemented yet.")
    
    def __call__(self, input_ids: jax.Array) -> jax.Array:
        _, seq_len = input_ids.shape

        if seq_len > self.config.seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds seq_len={self.config.seq_len}")
        
        x = self.embed(input_ids)
        cos, sin = _precompute_rope(seq_len=seq_len, head_dim=self.config.hidden_size // self.config.n_heads, theta=self.config.theta, dtype=x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def prefill(self, input_ids: jax.Array, cache: KVCache) -> tuple[jax.Array, KVCache]:
        _, seq_len = input_ids.shape
        if seq_len > self.config.seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds seq_len={self.config.seq_len}")

        tokens = jnp.swapaxes(input_ids, 0, 1)

        def step(cache, token):
            logits, cache = self.decode_one(token[:, None], cache)
            return cache, logits

        cache, logits = jax.lax.scan(step, cache, tokens)
        return logits[-1], cache

    def decode_one(self, input_ids: jax.Array, cache: KVCache) -> tuple[jax.Array, KVCache]:
        _, seq_len = input_ids.shape
        if seq_len != 1:
            raise ValueError(f"decode_one expects one token, got seq_len={seq_len}")

        x = self.embed(input_ids)
        head_dim = self.config.hidden_size // self.config.n_heads
        cos, sin = _precompute_rope(
            seq_len=self.config.seq_len,
            head_dim=head_dim,
            theta=self.config.theta,
            dtype=x.dtype,
        )
        cos = jax.lax.dynamic_slice(cos, (cache.length, 0), (1, cos.shape[1]))
        sin = jax.lax.dynamic_slice(sin, (cache.length, 0), (1, sin.shape[1]))

        new_layers = []
        for layer, layer_cache in zip(self.layers, cache.layers):
            x, layer_cache = layer.decode_one(x, layer_cache, cache.length, cos, sin)
            new_layers.append(layer_cache)
        x = self.norm(x)
        logits = self.lm_head(x)[:, -1, :]
        return logits, KVCache(layers=tuple(new_layers), length=cache.length + 1)
