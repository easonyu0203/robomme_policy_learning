import flax.nnx as nnx
import jax.numpy as jnp


import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.mem_encoder import FeatureEncoder
from mme_vla_suite.models.representation.selector import (
    Selector,
    batch_gather,
    gumbel_softmax_hard,
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
        assert static_image_emb.shape[1] == self.config.budget

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
            return hidden_states, mem_weight, losses

        # Eval: deterministic top-k by keep margin, physically gathered --
        # the only place the sequence actually shortens (the real
        # inference-time compute saving).
        topk_idx = select_topk(logits, valid_mask, self.num_keep)
        gathered_states = batch_gather(hidden_states, topk_idx)
        gathered_mask = batch_gather(valid_mask, topk_idx).astype(hidden_states.dtype)
        stats = {"keep_frac": jnp.asarray(self.num_keep / self.config.budget, dtype=hidden_states.dtype)}
        return gathered_states, gathered_mask, stats
