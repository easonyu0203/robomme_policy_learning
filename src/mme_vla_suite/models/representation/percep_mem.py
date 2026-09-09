import flax.nnx as nnx
import jax
import jax.numpy as jnp


import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.mem_encoder import FeatureEncoder
from mme_vla_suite.models.representation.selector import (
    Selector,
    batch_gather,
    gumbel_softmax_hard,
    gumbel_topk,
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

            # ---- D&R training fixes (2026-09-07) ---------------------------------
            # sampling: "bernoulli" (legacy: per-token 2-class Gumbel-softmax, random
            #   keep count, needs ratio/z/load-balance losses) or "topk" (Gumbel-top-k:
            #   exactly num_keep, no auxiliary losses; see selector.gumbel_topk).
            # e2e_tree: internal reduction rounds are scored by the LIVE selector and
            #   the gathered survivors are scaled by their straight-through mask value
            #   (1 in the forward pass), so the action loss reaches every node of the
            #   tree in one backward pass. Replaces multilevel routing + ema_reducer.
            # The root cut always masks in place (full `budget`-length sequence, time
            # order kept): MemoryAttention's RoPE is keyed by array slot (d738e40).
            self.sampling = selector_cfg.get("sampling", "bernoulli")
            assert self.sampling in ("bernoulli", "topk"), self.sampling
            self.tau = float(selector_cfg.get("tau", 1.0))
            self.noise_scale = float(selector_cfg.get("noise_scale", 1.0))
            self.score_norm = selector_cfg.get(
                "score_norm", "zscore" if self.sampling == "topk" else "none"
            )
            self.e2e_tree = bool(selector_cfg.get("e2e_tree", False))
            # e2e reduction rounds always use gumbel_topk (a physical shrink needs an exact count);
            # the ROOT cut may independently be "topk" or the legacy "bernoulli" (ablation of fix 1).
            self.round_score_norm = selector_cfg.get("round_score_norm", "zscore")
            self.aux_node_prob = 0.0  # overwritten below for hierarchical configs

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

            # Multi-level pick: during training feed the final trained cut a
            # uniformly-random reduction-tree node's input instead of always the
            # root's -- so the shared selector (and pi0.5) train on every
            # reduction depth, not just the last ("train it only at the last
            # reduction layer"). The tree has one node per `reduce_chunk_size`
            # chunk of every round's input, plus the root (the final cut's
            # input); `n_nodes` is static. See `_tree_pick_input`.
            self.multilevel = selector_cfg.get("multilevel", False)
            assert not (self.e2e_tree and self.multilevel), "e2e_tree replaces multilevel routing"
            assert not (self.e2e_tree and selector_cfg.get("ema_reducer", False)), (
                "e2e_tree replaces ema_reducer (the reduction rounds carry gradient themselves)"
            )
            n, self.n_nodes = self.input_len, 0
            for _ in range(self.n_reduce_rounds):
                self.n_nodes += -(-n // self.reduce_chunk_size)
                n = -(-n // self.reduce_chunk_size) * self.reduce_chunk_keep
            self.n_nodes += 1  # root
            if self.multilevel:
                # `_tree_pick_input` slices each round's input into whole
                # `reduce_chunk_size` chunks and accumulates into a
                # `reduce_chunk_size`-wide buffer, so every round's input length
                # must divide evenly and the last round must land on exactly one
                # chunk (all true for the power-of-2 `pool_budget` configs with
                # keep_ratio 0.5).
                n = self.input_len
                for _ in range(self.n_reduce_rounds):
                    assert n % self.reduce_chunk_size == 0, (
                        "multilevel pick needs every round's input length to be a "
                        f"multiple of reduce_chunk_size ({self.reduce_chunk_size}); got {n}"
                    )
                    n = (n // self.reduce_chunk_size) * self.reduce_chunk_keep
                assert self.reduced_len == self.reduce_chunk_size, (
                    f"multilevel pick needs reduced_len ({self.reduced_len}) == "
                    f"reduce_chunk_size ({self.reduce_chunk_size})"
                )

            # `aux_node_prob`: with this per-sample probability (training only) the
            # trained root cut is fed a uniformly-random INTERNAL node's input
            # instead of the tree root's output. Motivation: under e2e_tree a token
            # an internal node drops leaves the graph, so its only gradient is the
            # chance that the Gumbel noise keeps it. A routed sample instead applies
            # the root cut -- whose keep-weight multiplies exp(attention score) and
            # is renormalised, so a dropped token still has a nonzero d/dg -- to all
            # `reduce_chunk_size` candidates of that node, giving every one of them a
            # counterfactual "would the action improve if this token were in memory?"
            # gradient. The reduction tree still runs for the whole batch and the
            # routed samples simply discard its output, so no extra transformer
            # compute is spent (unlike an auxiliary second policy forward, which
            # would also expose the policy to the narrower memory anyway).
            self.aux_node_prob = float(selector_cfg.get("aux_node_prob", 0.0))
            assert 0.0 <= self.aux_node_prob <= 1.0, self.aux_node_prob
            if self.aux_node_prob > 0:
                assert self.e2e_tree, "aux_node_prob is an e2e_tree add-on"
                assert not self.multilevel, "aux_node_prob replaces multilevel routing"
                assert self.n_nodes - 1 > 0, "no internal nodes to route to (pool_budget == budget)"
                # Same shape requirements as the multilevel pick: node inputs are
                # whole `reduce_chunk_size` chunks and must be interchangeable with
                # the root's input.
                n = self.input_len
                for _ in range(self.n_reduce_rounds):
                    assert n % self.reduce_chunk_size == 0, (
                        "aux_node_prob needs every round's input length to be a multiple "
                        f"of reduce_chunk_size ({self.reduce_chunk_size}); got {n}"
                    )
                    n = (n // self.reduce_chunk_size) * self.reduce_chunk_keep
                assert self.reduced_len == self.reduce_chunk_size, (
                    f"aux_node_prob needs reduced_len ({self.reduced_len}) == "
                    f"reduce_chunk_size ({self.reduce_chunk_size})"
                )

    def _pick(self, hc, vc, keep, scorer, rng):
        """Score one batch of chunks and pick `keep` per chunk.

        Returns (idx (n, keep) int32 ascending, weight (n, chunk) float32 | None).
        Legacy path (e2e_tree off): stop_gradient scoring by `scorer` (EMA shadow
        or live), deterministic top-k, no weight. e2e path: live selector,
        `gumbel_topk` with `rng` (noise in training, deterministic in eval); the
        returned straight-through weight is 1 in the forward pass for every kept
        token and carries the gradient to this node's logits.
        """
        if self.e2e_tree:                               # [改动2] 新分支
            logits = self.selector(hc, vc)              #   活 selector，不 stop_gradient
            weight, idx = gumbel_topk(                  #   恰好 keep 个，带 STE 权重
                logits, vc, keep, rng, tau=self.tau, noise_scale=self.noise_scale,
                score_norm=self.round_score_norm,
            )
            return idx, weight
        
        # 原版分枝
        logits = jax.lax.stop_gradient((scorer if scorer is not None else self.selector)(hc, vc))
        idx = select_topk(logits, vc, keep)  # (n, keep) int, margin-sorted
        # Re-sort the survivor indices into ascending position before gathering,
        # so the reduced sequence stays in temporal order (needed by the
        # slot-keyed key RoPE in history_gemma.MemoryAttention; see 0464533).
        return jnp.sort(idx, axis=-1), None

    def _reduce_one_round(self, hidden, valid, scorer=None, rng=None):
        """One reduction round (caltech hard_vit.py::_select_chunks): split the
        sequence into contiguous `reduce_chunk_size` chunks (folded into the
        batch dim -> one Selector call), keep the top `reduce_chunk_keep` of each.

        `rng` is only used on the e2e path (per-node Gumbel noise).

        Legacy semantics (e2e_tree off) are unchanged: scoring is stop_gradient
        (preprocessing), the gather stays differentiable wrt `hidden`, and an
        all-padding chunk's NaN scores never leave the scoring path
        (`select_topk` maps its -inf margins to indices whose tokens are invalid).
        """
        chunk, keep = self.reduce_chunk_size, self.reduce_chunk_keep
        b, n = hidden.shape[0], hidden.shape[1]
        n_chunks = -(-n // chunk)  # ceil
        pad = n_chunks * chunk - n
        if pad:
            hidden = jnp.pad(hidden, ((0, 0), (0, pad), (0, 0)))
            valid = jnp.pad(valid, ((0, 0), (0, pad)))
        dim = hidden.shape[-1]
        hc = hidden.reshape(b * n_chunks, chunk, dim)
        vc = valid.reshape(b * n_chunks, chunk)

        # # 原版：打分 stop_gradient，确定性 top-k，按位置排序，没有权重
        # logits = jax.lax.stop_gradient(
        #     (scorer if scorer is not None else self.selector)(hc, vc)
        # )
        # idx = select_topk(logits, vc, keep)
        # idx = jnp.sort(idx, axis=-1)
        
        # ==================================================================
        # [改动2] 抽成 _pick：e2e_tree 关 → 执行的就是上面被删的 4 行，weight=None
        #                   e2e_tree 开 → 活 selector + gumbel_topk(rng)，返回 STE 权重
        # idx：选中了哪些 token。
        # selected_tokens：这些 token 的内容。
        # selected_weights：这些 token 对应的 ST 权重。
        # [..., None]：把 (N, K) 变成 (N, K, 1)，同一个标量权重乘到该 token 的所有特征维度上。
        idx, weight = self._pick(hc, vc, keep, scorer, rng)
        hc = batch_gather(hc, idx)  # selected_tokens
        if weight is not None:
            # Straight-through scaling: forward x1 (kept tokens have weight exactly
            # 1), backward routes dL/dtoken . token into this node's logits -- the
            # MoE trick (expert output x router prob) applied to a hard top-k.
            hc = hc * batch_gather(weight, idx)[..., None].astype(hc.dtype) # selected_weights，把 router 的 gate 乘到被选中的输出上
        vc = batch_gather(vc[..., None], idx)[..., 0]
        return hc.reshape(b, n_chunks * keep, dim), vc.reshape(b, n_chunks * keep)

    def _hierarchical_reduce(self, hidden, valid, scorer=None, rng=None, collect=None):
        """Repeat `_reduce_one_round` `n_reduce_rounds` times (static, unrolled
        at trace time). Per-round rng is derived by fold_in so every node samples
        independent Gumbel noise (e2e path only; None -> deterministic).

        `collect`: optional list; each round's (input tokens, input valid) is
        appended to it, which is where `_pick_node_input` reads the internal
        nodes' inputs from (`aux_node_prob`). Plain Python appends -- the loop is
        unrolled at trace time."""
        for r in range(self.n_reduce_rounds):
            if collect is not None:
                collect.append((hidden, valid))
            rng_r = None if rng is None else jax.random.fold_in(rng, r)
            hidden, valid = self._reduce_one_round(hidden, valid, scorer, rng_r)
        return hidden, valid

    def _pick_node_input(self, node_inputs, picked):
        """Per sample, gather the `reduce_chunk_size`-wide input of internal node
        `picked` (round-major numbering, same as `_tree_pick_input`; the root is
        excluded, so ids run 0..n_nodes-2). Every branch is a slice of a tensor
        the reduction already produced, so this costs only the `where`s."""
        chunk = self.reduce_chunk_size
        h0 = node_inputs[0][0]
        b, d = h0.shape[0], h0.shape[-1]
        picked_h = jnp.zeros((b, chunk, d), dtype=h0.dtype)
        picked_v = jnp.zeros((b, chunk), dtype=jnp.bool_)
        node = 0
        for h, v in node_inputs:
            for k in range(h.shape[1] // chunk):
                sel = (picked == node)[:, None]
                sl = slice(k * chunk, (k + 1) * chunk)
                picked_h = jnp.where(sel[:, :, None], h[:, sl, :], picked_h)
                picked_v = jnp.where(sel, v[:, sl], picked_v)
                node += 1
        assert node == self.n_nodes - 1, (node, self.n_nodes)
        return picked_h, picked_v

    def _tree_pick_input(
        self,
        hidden: at.Float[at.Array, "b l d"],
        valid: at.Bool[at.Array, "b l"],
        picked_node: at.Int[at.Array, " b"],
        scorer: Selector | None = None,
    ) -> tuple[at.Float[at.Array, "b bud d"], at.Bool[at.Array, "b bud"]]:
        """Per sample, return the `reduce_chunk_size`-wide (tokens, valid) input
        of the picked reduction-tree node. Nodes are numbered round-major:
        round-1 chunks 0..c1-1, then round-2 chunks, ..., then the root
        (== `self.n_nodes - 1`, the final trained cut's input).

        Every round runs for the whole batch -- the rounds build the middle and
        root nodes' inputs -- but each sample's gradient reaches the selector
        only through the one trained cut downstream, fed *this* slice: for a
        leaf pick `picked_h` is a differentiable slice of `hidden`
        (FeatureEncoder output); for a middle/root pick it is a slice of the
        (scoring-detached, gather-differentiable) `_reduce_one_round` output.
        Static: chunk/node counts are config-derived and `picked_node` only
        feeds `==`, so no dynamic shapes leak in.
        """
        chunk = self.reduce_chunk_size
        b, d = hidden.shape[0], hidden.shape[-1]
        picked_h = jnp.zeros((b, chunk, d), dtype=hidden.dtype)
        picked_v = jnp.zeros((b, chunk), dtype=jnp.bool_)
        node = 0
        cur_h, cur_v = hidden, valid
        for _ in range(self.n_reduce_rounds):
            n_chunks = cur_h.shape[1] // chunk  # exact -- asserted in __init__
            for k in range(n_chunks):
                sel = (picked_node == node)[:, None]  # (b, 1)
                sl = slice(k * chunk, (k + 1) * chunk)
                picked_h = jnp.where(sel[:, :, None], cur_h[:, sl, :], picked_h)
                picked_v = jnp.where(sel, cur_v[:, sl], picked_v)
                node += 1
            cur_h, cur_v = self._reduce_one_round(cur_h, cur_v, scorer)
        # root: cur_h is (b, reduced_len == chunk, d) after the last round
        sel = (picked_node == node)[:, None]
        picked_h = jnp.where(sel[:, :, None], cur_h, picked_h)
        picked_v = jnp.where(sel, cur_v, picked_v)
        return picked_h, picked_v

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
        reducer_selector: Selector | None = None,
    ):
        """Returns (tokens, mem_weight, stats).

        tokens (b, budget, d) in time order; mem_weight (b, budget) float
        keep-weight (mask in place) or None (selector off); stats dict or None.
        """
        # get memory tokens using feature encoder
        assert static_image_emb.shape[1] == self.input_len

        # `reducer_selector`: legacy `selector.ema_reducer` -- the no-grad
        # reduction rounds score with the EMA-shadow selector. Unused (asserted
        # off) under e2e_tree, where the rounds carry gradient themselves.

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
            real_before = valid_mask.sum(axis=1)
            if self.multilevel and train:
                # Legacy multilevel routing (see _tree_pick_input); e2e_tree is
                # the replacement.
                assert rng is not None, "multilevel train pick needs `rng`"
                rng, rng_pick = jax.random.split(rng)
                picked_node = jax.random.randint(
                    rng_pick, (hidden_states.shape[0],), 0, self.n_nodes
                )
                hidden_states, valid_mask = self._tree_pick_input(
                    hidden_states, valid_mask, picked_node, reducer_selector
                )
                extra_stats["picked_node"] = jax.lax.stop_gradient(
                    picked_node.astype(jnp.float32).mean()
                )
            else:
                rng_tree = None
                if self.e2e_tree and train:
                    assert rng is not None, "e2e_tree training needs `rng`"
                    rng, rng_tree = jax.random.split(rng)
                route_aux = train and self.aux_node_prob > 0
                node_inputs = [] if route_aux else None
                reduced_h, reduced_v = self._hierarchical_reduce(
                    hidden_states, valid_mask, reducer_selector, rng_tree, node_inputs
                )
                if route_aux:
                    assert rng is not None, "aux_node_prob training needs `rng`"
                    rng, rng_node, rng_use = jax.random.split(rng, 3)
                    b = reduced_h.shape[0]
                    picked_node = jax.random.randint(rng_node, (b,), 0, self.n_nodes - 1)
                    aux_h, aux_v = self._pick_node_input(node_inputs, picked_node)
                    # An all-padding chunk (short episode, right padding) would leave
                    # the memory with no valid slot at all, so fall back to the root.
                    use_aux = jax.random.bernoulli(rng_use, self.aux_node_prob, (b,))
                    use_aux = use_aux & aux_v.any(axis=-1)
                    hidden_states = jnp.where(use_aux[:, None, None], aux_h, reduced_h)
                    valid_mask = jnp.where(use_aux[:, None], aux_v, reduced_v)
                    extra_stats["aux_node_frac"] = jax.lax.stop_gradient(
                        use_aux.astype(jnp.float32).mean()
                    )
                else:
                    hidden_states, valid_mask = reduced_h, reduced_v
            real_after = valid_mask.sum(axis=1)
            extra_stats["reduce_keep_frac"] = jax.lax.stop_gradient(
                jnp.mean(real_after / jnp.clip(real_before, a_min=1.0))
            )

        if self.eval_keep_all and not train:
            mem_weight = valid_mask.astype(hidden_states.dtype)
            return hidden_states, mem_weight, {
                "keep_frac": masked_mean(mem_weight, valid_mask),
                **extra_stats,
            }

        logits = self.selector(hidden_states, valid_mask)

        if self.sampling == "topk":
            # Exactly-num_keep cut, same rule in train (Gumbel noise) and eval.
            weight, _ = gumbel_topk(
                logits, valid_mask, self.num_keep, rng if train else None,
                tau=self.tau, noise_scale=self.noise_scale, score_norm=self.score_norm,
            )
            extra_stats["keep_frac"] = jax.lax.stop_gradient(masked_mean(weight, valid_mask))
            # Mask in place over the full `budget`-length, time-ordered sequence
            # (no gather: MemoryAttention's RoPE is slot-keyed; see d738e40).
            return hidden_states, weight.astype(hidden_states.dtype), extra_stats

        # ---- legacy "bernoulli" sampling (unchanged behaviour) ----
        if train:
            assert rng is not None, "train=True requires `rng` for Gumbel-softmax sampling"
            decision = gumbel_softmax_hard(logits, rng)[..., 0]
            mem_weight = decision * valid_mask.astype(hidden_states.dtype)
            losses = selector_losses(logits, decision, valid_mask, self.keep_ratio)
            return hidden_states, mem_weight, {**losses, **extra_stats}

        # Eval: deterministic top-`num_keep`, hard {0,1} keep-weight over the full
        # `budget`-length sequence (mask in place, no gather -- required by the
        # slot-keyed RoPE downstream; see d738e40).
        topk_idx = select_topk(logits, valid_mask, self.num_keep)
        b_idx = jnp.arange(valid_mask.shape[0])[:, None]
        keep_mask = jnp.zeros_like(valid_mask).at[b_idx, topk_idx].set(True) & valid_mask
        mem_weight = keep_mask.astype(hidden_states.dtype)
        stats = {
            "keep_frac": masked_mean(mem_weight, valid_mask),
            **extra_stats,
        }
        return hidden_states, mem_weight, stats
