import jax
import jax.numpy as jnp
from flax import nnx

from mme_vla_suite.models.representation.utils import kernel_init, kernel_init_out_proj


def masked_mean(x: jnp.ndarray, valid_mask: jnp.ndarray, axis=None) -> jnp.ndarray:
    """Mean of `x` over positions where `valid_mask` is True."""
    valid = valid_mask.astype(x.dtype)
    denom = jnp.clip(valid.sum(axis=axis), a_min=1.0)
    return (x * valid).sum(axis=axis) / denom


def batch_gather(x: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """x: (B, L, ...). idx: (B, K) int32, indices along axis 1. -> (B, K, ...)."""
    extra_dims = x.ndim - 2
    idx = idx.reshape(idx.shape + (1,) * extra_dims)
    idx = jnp.broadcast_to(idx, idx.shape[:2] + x.shape[2:])
    return jnp.take_along_axis(x, idx, axis=1)


def gumbel_softmax_hard(
    logits: jnp.ndarray, rng: jax.Array, tau: float = 1.0, eps: float = 1e-9
) -> jnp.ndarray:
    """Straight-through Gumbel-softmax hard sample (JAX has no `F.gumbel_softmax`
    equivalent). logits: (..., 2), channel 0 = keep. Forward value is exactly
    {0,1}; gradient flows via the straight-through estimator."""
    u = jax.random.uniform(rng, logits.shape, minval=eps, maxval=1.0 - eps)
    gumbel_noise = -jnp.log(-jnp.log(u))
    y_soft = jax.nn.softmax((logits + gumbel_noise) / tau, axis=-1)
    y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), logits.shape[-1], dtype=y_soft.dtype)

    return jax.lax.stop_gradient(y_hard - y_soft) + y_soft


def select_topk(logits: jnp.ndarray, valid_mask: jnp.ndarray, num_keep: int) -> jnp.ndarray:
    """Deterministic (no Gumbel noise) top-`num_keep` positions by keep margin
    (keep_logit - drop_logit). Padding is excluded via a -inf margin, so it
    can never be selected ahead of a real token. Returns (B, num_keep) indices."""
    margin = logits[..., 0] - logits[..., 1]
    margin = jnp.where(valid_mask, margin, -jnp.inf)
    return jax.lax.top_k(margin, num_keep)[1]


def selector_losses(
    logits: jnp.ndarray, decision: jnp.ndarray, valid_mask: jnp.ndarray, keep_ratio: float
) -> dict[str, jnp.ndarray]:
    """logits: (B, L, 2). decision: (B, L) hard {0,1} keep decision (channel-0
    straight-through, from `gumbel_softmax_hard`). valid_mask: (B, L) bool,
    True = real (non-padding) token -- masked out so early-episode padding
    never pollutes the statistics."""
    keep_frac = masked_mean(decision, valid_mask)
    ratio_loss = (keep_frac - keep_ratio) ** 2

    z_loss = masked_mean(jax.scipy.special.logsumexp(logits, axis=-1) ** 2, valid_mask)

    # Per-slot mean keep-prob across the batch, deviation from the population's own mean
    keep_prob = jax.nn.softmax(logits, axis=-1)[..., 0]
    mean_keep_prob = masked_mean(keep_prob, valid_mask, axis=0)  # (L,)
    slot_valid = valid_mask.any(axis=0)
    grand_mean = masked_mean(mean_keep_prob, slot_valid)
    load_balance_loss = masked_mean((mean_keep_prob - grand_mean) ** 2, slot_valid)

    return {
        "ratio_loss": ratio_loss,
        "z_loss": z_loss,
        "load_balance_loss": load_balance_loss,
        "keep_frac": jax.lax.stop_gradient(keep_frac),
    }


class _SelectorBlock(nnx.Module):
    """Pre-norm self-attention + MLP block. No RoPE/QK-norm: the tokens the
    selector scores already carry positional info fused in by FeatureEncoder
    before the selector ever sees them."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, *, rngs: nnx.Rngs, dtype):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.q_proj = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.k_proj = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.v_proj = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.out_proj = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj)

        self.norm2 = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj)

    def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray) -> jnp.ndarray:
        # x: (B, L, D). attn_mask: (B, L) bool, True = may be attended to.
        B, L, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(h).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(h).reshape(B, L, self.num_heads, self.head_dim)
        mask = attn_mask[:, None, None, :]  # broadcast over heads and queries
        out = jax.nn.dot_product_attention(q, k, v, mask=mask)
        x = x + self.out_proj(out.reshape(B, L, D))

        h = self.norm2(x)
        h = self.fc2(jax.nn.gelu(self.fc1(h)))
        return x + h


class Selector(nnx.Module):
    """Predicts a [keep, drop] logit pair per memory token from
    [tokens, register_tokens], via self-attention -- register tokens act as
    learned global-context aggregators (mirrors caltech_vit/hard_vit.py's
    Selector in the reference project, adapted to this repo's nnx conventions:
    LayerNorm not RMSNorm, plain GELU MLP not SwiGLU, no QK-norm)."""

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_heads: int = 8,
        num_register_tokens: int = 4,
        mlp_ratio: float = 4.0,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.num_register_tokens = num_register_tokens
        self.register_tokens = nnx.Param(
            jax.random.normal(rngs.params(), (num_register_tokens, dim), dtype=dtype) * 0.02
        )
        self.blocks = [
            _SelectorBlock(dim, num_heads, mlp_ratio, rngs=rngs, dtype=dtype) for _ in range(depth)
        ]
        self.norm = nnx.LayerNorm(dim, rngs=rngs, dtype=dtype)
        self.head = nnx.Linear(dim, 2, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

    def __call__(self, tokens: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
        # tokens: (B, L, D). valid_mask: (B, L) bool, True = real (non-padding).
        B, L, D = tokens.shape
        register_tokens = jnp.broadcast_to(
            self.register_tokens.value.astype(tokens.dtype), (B, self.num_register_tokens, D)
        )
        x = jnp.concatenate([tokens, register_tokens], axis=1)
        register_valid = jnp.ones((B, self.num_register_tokens), dtype=jnp.bool_)
        attn_mask = jnp.concatenate([valid_mask, register_valid], axis=1)

        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.norm(x)
        return self.head(x[:, :L])  # (B, L, 2): [keep_logit, drop_logit]
