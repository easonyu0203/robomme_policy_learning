import flax.nnx as nnx
import jax
import jax.numpy as jnp


import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.mem_encoder import FeatureEncoder
from mme_vla_suite.models.representation.selector import (
    Selector,
    batch_gather,
    gumbel_softmax_hard,
    masked_mean,
    select_topk,
    selector_losses,
)


class PerceptualMemory(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        self.config = config
        self.dtype = dtype

        self.mem_type = config.perceptual_memory.type

        # Defensive, like selector_cfg below: configs predating use_time_emb/
        # memory_feature.time entirely (e.g. a frozen pre-d182ebe checkpoint's
        # history_config snapshot) must still load, not KeyError.
        use_time_emb = config.get("use_time_emb", False)
        time_feature_cfg = config.memory_feature.get("time", None) if use_time_emb else None

        self.feature_encoder = FeatureEncoder(
            rngs=rngs,
            dtype=dtype,
            image_input_dim=self.config.memory_feature.img.input_dim,
            pos_input_dim=self.config.memory_feature.pos.input_dim,
            state_input_dim=self.config.memory_feature.state.input_dim,
            time_input_dim=time_feature_cfg.input_dim if time_feature_cfg is not None else 1,
            pos_output_dim=self.config.memory_feature.pos.hidden_dim,
            state_output_dim=self.config.memory_feature.state.hidden_dim,
            time_output_dim=time_feature_cfg.hidden_dim if time_feature_cfg is not None else 64,
            ouput_dim_for_recur=None,
            output_dim_for_percep=self.config.memory_token_dim,
            use_pos_emb=self.config.use_pos_emb,
            use_state_emb=self.config.use_state_emb,
            use_time_emb=use_time_emb,
        )

        # `hierarchical_selection` ships a `pool_budget`-wide token pool that the
        # selector reduces down to `budget` (no gradient) before the trained cut;
        # every other mode ships exactly `budget` tokens.
        self.is_hierarchical = self.mem_type == "hierarchical_selection"
        self.input_len = (
            config.get("pool_budget", config.budget)
            if self.is_hierarchical
            else config.budget
        )

        selector_cfg = config.perceptual_memory.get("selector", None)
        self.use_selector = selector_cfg is not None and selector_cfg.get("enabled", False)
        if self.use_selector:
            self.selector = Selector(
                dim=config.memory_token_dim,
                depth=selector_cfg.get("depth", 2),
                num_heads=selector_cfg.get("num_heads", 8),
                num_register_tokens=selector_cfg.get("num_register_tokens", 4),
                rngs=rngs,
                dtype=dtype,
            )
            self.keep_ratio = selector_cfg.get("keep_ratio", 0.5)
            self.num_keep = round(config.budget * self.keep_ratio)
            # Inference-only ablation: skip the final trained cut and hand the
            # backbone the whole post-reduction sequence (all `budget` tokens,
            # unmasked). For a non-hierarchical selector (pool512) this means the
            # raw `budget` pool passes straight through; for a hierarchical one
            # (pool1024) the no-grad reduction rounds still run (pool_budget ->
            # budget), only the last selector cut (budget -> num_keep) is
            # dropped. Training is unchanged -- this only affects `train=False`.
            self.eval_keep_all = selector_cfg.get("eval_keep_all", False)

        if self.is_hierarchical:
            assert self.use_selector, "hierarchical_selection requires perceptual_memory.selector.enabled"
            # Each reduction round groups the sequence into contiguous
            # `reduce_chunk_size` chunks and keeps `reduce_chunk_keep` of each
            # (caltech hard_vit.py::_select_chunks). A full chunk must strictly
            # shrink or the loop never terminates.
            self.reduce_chunk_size = config.budget
            self.reduce_chunk_keep = round(
                self.reduce_chunk_size * selector_cfg.get("reduce_keep_ratio", self.keep_ratio)
            )
            assert 0 < self.reduce_chunk_keep < self.reduce_chunk_size, (
                f"reduce_keep_ratio gives {self.reduce_chunk_keep}/{self.reduce_chunk_size} per "
                "chunk; must be strictly inside (0, chunk_size) or the reduction never shrinks"
            )
            # Round count is a pure function of the static config -- precompute it
            # so the reduction loop unrolls at trace time (no lax.while_loop, no
            # dynamic shapes).
            n = self.input_len
            self.n_reduce_rounds = 0
            while n > config.budget:
                n = -(-n // self.reduce_chunk_size) * self.reduce_chunk_keep
                self.n_reduce_rounds += 1
            self.reduced_len = n
            assert self.num_keep <= self.reduced_len, (
                f"final cut keeps {self.num_keep} tokens but the reduction only yields "
                f"{self.reduced_len} (pool_budget={self.input_len}, budget={config.budget})"
            )

    def _hierarchical_reduce(
        self, hidden: at.Float[at.Array, "b l d"], valid: at.Bool[at.Array, "b l"]
    ) -> tuple[at.Float[at.Array, "b lr d"], at.Bool[at.Array, "b lr"]]:
        """caltech hard_vit.py::_hierarchical_reduce, adapted to JAX / static shapes.

        Repeatedly splits the sequence into contiguous `reduce_chunk_size` chunks
        (folded into the batch dim -> one Selector call per round) and keeps the
        top `reduce_chunk_keep` of each by keep-margin. Scoring is stop_gradient
        -- this stage is preprocessing -- but the survivor gather stays
        differentiable, so the FeatureEncoder still learns from whichever tokens
        reach the trained cut. `self.n_reduce_rounds` is static, so this loop
        unrolls at trace time (no dynamic shapes, no lax.while_loop).

        An all-padding chunk (short episode) produces NaN scores from the
        all-masked attention, but `select_topk` maps its -inf margins to
        arbitrary indices whose gathered tokens carry `valid=False`; the NaN
        never leaves the (stop_gradient'd) scoring path.
        """
        chunk, keep = self.reduce_chunk_size, self.reduce_chunk_keep
        b = hidden.shape[0]
        n = hidden.shape[1]
        for _ in range(self.n_reduce_rounds):
            n_chunks = -(-n // chunk)  # ceil
            pad = n_chunks * chunk - n
            if pad:
                hidden = jnp.pad(hidden, ((0, 0), (0, pad), (0, 0)))
                valid = jnp.pad(valid, ((0, 0), (0, pad)))
            dim = hidden.shape[-1]
            hc = hidden.reshape(b * n_chunks, chunk, dim)
            vc = valid.reshape(b * n_chunks, chunk)
            # Scoring: no gradient to the selector params or to `hc` via this path.
            logits = jax.lax.stop_gradient(self.selector(hc, vc))
            idx = select_topk(logits, vc, keep)  # (b*n_chunks, keep) int
            # Gather: differentiable w.r.t. `hidden` for the surviving tokens.
            hc = batch_gather(hc, idx)
            vc = batch_gather(vc[..., None], idx)[..., 0]
            n = n_chunks * keep
            hidden = hc.reshape(b, n, dim)
            valid = vc.reshape(b, n)
        return hidden, valid

    def __call__(
        self,
        static_image_emb: at.Float[at.Array, "b l d1"],
        static_pos_emb: at.Float[at.Array, "b l d2"],
        static_state_emb: at.Float[at.Array, "b l d3"],
        static_time_emb: at.Float[at.Array, "b l d4"] | None = None,
        static_mask: at.Bool[at.Array, "b l"] | None = None,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ):
        # get memory tokens using feature encoder
        assert static_image_emb.shape[1] == self.input_len

        hidden_states = self.feature_encoder.encode_perceptual_memory(
            static_image_emb, static_pos_emb, static_state_emb, static_time_emb
        )

        if not self.use_selector:
            return hidden_states, None, None

        valid_mask = (
            static_mask
            if static_mask is not None
            else jnp.ones(hidden_states.shape[:2], dtype=jnp.bool_)
        )

        extra_stats = {}
        if self.is_hierarchical:
            # No-grad hierarchical reduction: pool_budget -> reduced_len (<= budget).
            # Only the trained cut below sees gradient ("train it only at the last
            # reduction layer").
            real_before = valid_mask.sum(axis=1)
            hidden_states, valid_mask = self._hierarchical_reduce(hidden_states, valid_mask)
            real_after = valid_mask.sum(axis=1)
            # Fraction of real (non-padding) tokens that survived the reduction --
            # 1.0 means nothing real was dropped; low values mean the pool held
            # more real tokens than the reduction target and the selector had to
            # choose. Diagnostic only.
            extra_stats["reduce_keep_frac"] = jax.lax.stop_gradient(
                jnp.mean(real_after / jnp.clip(real_before, a_min=1.0))
            )

        if self.eval_keep_all and not train:
            # Ablation: skip the final trained cut entirely -- hand the backbone
            # every valid post-reduction token at its trained slot (same
            # in-place, full-length {0,1} convention as the real eval path
            # below, just with nothing dropped). The selector is not even
            # evaluated here.
            mem_weight = valid_mask.astype(hidden_states.dtype)
            return hidden_states, mem_weight, {
                "keep_frac": masked_mean(mem_weight, valid_mask),
                **extra_stats,
            }

        logits = self.selector(hidden_states, valid_mask)

        if train:
            assert rng is not None, "train=True requires `rng` for Gumbel-softmax sampling"
            decision = gumbel_softmax_hard(logits, rng)[..., 0]
            # Continuous, gradient-carrying (via the straight-through decision)
            # keep-weight. Sequence length stays at `budget` -- see
            # MemoryAttention's masked-softmax for why this is differentiable
            # all the way back to the selector's logits.
            mem_weight = decision * valid_mask.astype(hidden_states.dtype)
            losses = selector_losses(logits, decision, valid_mask, self.keep_ratio)
            return hidden_states, mem_weight, {**losses, **extra_stats}

        # Eval: deterministic top-`num_keep`-by-keep-margin, returned as a hard
        # {0,1} keep-weight over the *full* `budget`-length sequence -- NOT a
        # physical gather. The downstream consumer (history_gemma.MemoryAttention)
        # applies RoPE keyed to each memory token's slot index and a query offset
        # of `mem_len`, so it is *not* permutation- or length-invariant. Gathering
        # would repack the survivors into slots 0..num_keep-1 in descending-margin
        # order (jax.lax.top_k is value-sorted, not index-sorted), handing every
        # token a RoPE position unrelated to the temporal slot it was trained at,
        # and shrinking the query offset from `budget` to `num_keep`. Training
        # masks in place and keeps length == `budget`; eval must do the same or
        # the cross-attention sees a positional geometry it never saw in training
        # (train loss fine, eval collapses). Keeping length == `budget` with the
        # masked softmax leaves the Gumbel-sample-vs-argmax difference as the only
        # train/eval gap -- the intended one.
        topk_idx = select_topk(logits, valid_mask, self.num_keep)
        b_idx = jnp.arange(valid_mask.shape[0])[:, None]
        keep_mask = jnp.zeros_like(valid_mask).at[b_idx, topk_idx].set(True) & valid_mask
        mem_weight = keep_mask.astype(hidden_states.dtype)
        stats = {
            "keep_frac": masked_mean(mem_weight, valid_mask),
            **extra_stats,
        }
        return hidden_states, mem_weight, stats
