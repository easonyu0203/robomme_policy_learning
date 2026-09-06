"""Checks for the D&R training fixes (2026-09-07):
  * selector.gumbel_topk        -- exactly-K straight-through mask, train/eval same rule
  * PerceptualMemory e2e_tree   -- gradient reaches internal nodes; eval == legacy rounds
  * mem_rope=time               -- MemoryAttention permutation- and gather-invariant,
                                   through the scanned backbone Module as well

    JAX_PLATFORMS=cpu PYTHONPATH=src:packages/openpi-client/src python tests/test_dnr_fixes.py
"""
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from mme_vla_suite.models.representation.percep_mem import PerceptualMemory
from mme_vla_suite.models.representation.selector import batch_gather, gumbel_topk, select_topk

CFG = "src/mme_vla_suite/models/config/robomme/perceptual-dnr-modul_bud64_pool128.yaml"
LEGACY = {  # same architecture, old training path
    "mem_rope": "slot",
    "perceptual_memory.selector.sampling": "bernoulli",
    "perceptual_memory.selector.e2e_tree": False,
    "perceptual_memory.selector.root_gather": False,
}


def build(**overrides):
    cfg = OmegaConf.load(CFG)
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    return PerceptualMemory(config=cfg, rngs=nnx.Rngs(0), dtype=jnp.float32), cfg


def rand_inputs(cfg, b=2, n_real=None, seed=0):
    rng = np.random.default_rng(seed)
    p = cfg.pool_budget
    mk = lambda d: jnp.asarray(rng.standard_normal((b, p, d)), dtype=jnp.float32)
    img, pos, state = (mk(cfg.memory_feature.img.input_dim), mk(cfg.memory_feature.pos.input_dim),
                       mk(cfg.memory_feature.state.input_dim))
    # time-sorted pool: steps-ago decreases left -> right; 16 tokens per frame
    frames = p // cfg.token_per_image
    steps_ago = np.repeat(np.linspace(300, 0, frames).round(), cfg.token_per_image)
    time = jnp.asarray(np.log1p(steps_ago)[None, :, None].repeat(b, 0), dtype=jnp.float32)
    if n_real is None:
        mask = jnp.ones((b, p), dtype=bool)
    else:
        m = np.zeros((b, p), dtype=bool); m[:, :n_real] = True; mask = jnp.asarray(m)
    return img, pos, state, time, mask


def test_gumbel_topk():
    key = jax.random.key(0)
    B, L, K = 3, 64, 32
    logits = jax.random.normal(key, (B, L, 2))
    for n_real in (64, 40, 10):
        m = np.zeros((B, L), bool); m[:, :n_real] = True; valid = jnp.asarray(m)
        for rng in (None, jax.random.key(1)):
            w, idx = gumbel_topk(logits, valid, K, rng)
            assert set(np.unique(np.asarray(w)).tolist()) <= {0.0, 1.0}
            assert (np.asarray(w.sum(1)) == min(K, n_real)).all(), (n_real, w.sum(1))
            assert not np.asarray(w)[~m].any(), "padding kept"
            assert (np.diff(np.asarray(idx), axis=1) > 0).all(), "idx not ascending"
        # deterministic == plain top-k by margin (z-score is monotone within a row)
        w0, _ = gumbel_topk(logits, valid, K, None)
        ref = np.zeros((B, L), bool)
        ref[np.arange(B)[:, None], np.asarray(select_topk(logits, valid, K))] = True
        ref &= m
        assert (np.asarray(w0).astype(bool) == ref).all(), n_real
    # straight-through gradient: finite, nonzero, and only on valid tokens
    m = np.zeros((B, L), bool); m[:, :40] = True; valid = jnp.asarray(m)
    v = jax.random.normal(jax.random.key(2), (B, L))
    g = jax.grad(lambda lg: jnp.sum(gumbel_topk(lg, valid, K, jax.random.key(3))[0] * v))(logits)
    g = np.asarray(g)
    assert np.isfinite(g).all() and np.abs(g).sum() > 0
    assert np.abs(g[~m]).sum() == 0.0, "gradient leaked to padding"
    print("OK gumbel_topk")


def _hid(model, cfg, n_real=None, seed=0):
    img, pos, state, time, mask = rand_inputs(cfg, n_real=n_real, seed=seed)
    hid = model.feature_encoder.encode_perceptual_memory(img, pos, state, time)
    tpos = jnp.round(jnp.expm1(time[..., 0])).astype(jnp.int32)
    return hid, mask, tpos


def test_e2e_rounds_gradient_and_eval_equivalence():
    e2e, cfg = build()
    legacy, _ = build(**LEGACY)
    assert e2e.n_reduce_rounds == 1 and e2e.e2e_tree and e2e.root_gather
    hid, mask, tpos = _hid(e2e, cfg, n_real=100)
    probe = jax.random.normal(jax.random.key(5), (2, cfg.budget, cfg.memory_token_dim))

    # eval (no noise): e2e rounds pick the same survivors as the legacy rounds
    h_e, v_e, t_e = e2e._hierarchical_reduce_ext(hid, mask, tpos, None, None)
    h_l, v_l = legacy._hierarchical_reduce(hid, mask)
    assert h_e.shape == h_l.shape == (2, cfg.budget, cfg.memory_token_dim)
    assert jnp.array_equal(v_e, v_l) and jnp.allclose(h_e, h_l, atol=1e-5)
    assert t_e.shape == (2, cfg.budget)
    # survivors stay in time order: steps-ago non-increasing over valid slots
    for b in range(2):
        tv = np.asarray(t_e[b])[np.asarray(v_e[b])]
        assert (np.diff(tv) <= 0).all(), tv

    # gradient of a loss on the ROUND OUTPUT wrt selector params:
    #   e2e: nonzero (straight-through scaling of the gathered tokens)
    #   legacy: exactly zero (scoring is stop_gradient'd)
    def loss_e2e(m):
        h, v, _ = m._hierarchical_reduce_ext(hid, mask, tpos, None, jax.random.key(7))
        return jnp.sum(h * probe * v[..., None])

    def loss_leg(m):
        h, v = m._hierarchical_reduce(hid, mask)
        return jnp.sum(h * probe * v[..., None])

    g_e = nnx.grad(loss_e2e)(e2e)
    g_l = nnx.grad(loss_leg)(legacy)
    n_e = sum(float(jnp.abs(x).sum()) for x in jax.tree.leaves(g_e.selector))
    n_l = sum(float(jnp.abs(x).sum()) for x in jax.tree.leaves(g_l.selector))
    assert np.isfinite(n_e) and n_e > 0, n_e
    assert n_l == 0.0, n_l
    print(f"OK e2e_rounds (selector grad e2e={n_e:.3g}, legacy={n_l})")


@nnx.jit(static_argnames=("train",))
def _fwd(model, img, pos, state, time, mask, *, train, rng):
    return model(img, pos, state, time, mask, train=train, rng=rng)


def test_full_call_root_gather():
    model, cfg = build()
    K = model.num_keep
    for n_real in (100, None):
        img, pos, state, time, mask = rand_inputs(cfg, n_real=n_real, seed=3)
        for train in (True, False):
            tok, w, stats, mpos = _fwd(model, img, pos, state, time, mask, train=train, rng=jax.random.key(4))
            assert tok.shape == (2, K, cfg.memory_token_dim), tok.shape
            assert w.shape == (2, K) and mpos.shape == (2, K) and mpos.dtype == jnp.int32
            assert jnp.isfinite(tok).all()
            assert set(np.unique(np.asarray(w)).tolist()) <= {0.0, 1.0}
            assert (np.asarray(w.sum(1)) == K).all(), w.sum(1)  # exactly K kept
            for b in range(2):  # gathered tokens keep time order
                assert (np.diff(np.asarray(mpos[b])) <= 0).all()
            for k in ("keep_frac", "reduce_keep_frac", "first_frame_keep_frac"):
                assert k in stats and jnp.isfinite(stats[k]).all(), k
            assert "ratio_loss" not in stats and "load_balance_loss" not in stats
    # whole-call gradient wrt selector params is finite and nonzero in training
    img, pos, state, time, mask = rand_inputs(cfg, n_real=100, seed=3)
    probe = jax.random.normal(jax.random.key(9), (2, K, cfg.memory_token_dim))

    def loss(m):
        tok, w, _, _ = m(img, pos, state, time, mask, train=True, rng=jax.random.key(4))
        return jnp.sum(tok * probe * w[..., None])

    g = nnx.grad(loss)(model)
    n = sum(float(jnp.abs(x).sum()) for x in jax.tree.leaves(g.selector))
    assert np.isfinite(n) and n > 0, n
    print(f"OK full_call_root_gather (selector grad {n:.3g})")


def test_memory_attention_time_rope():
    from mme_vla_suite.models.integration.history_gemma import MemoryAttention
    key = jax.random.key(0)
    B, T, S, D = 1, 5, 16, 1024
    x = jax.random.normal(key, (B, T, D))
    mem = jax.random.normal(jax.random.key(1), (B, S, D))
    pos = jax.random.randint(jax.random.key(2), (B, S), 0, 300)
    ones = jnp.ones((B, S), jnp.float32)
    attn = MemoryAttention()
    params = attn.init(jax.random.key(3), x, mem, ones, pos)
    perm = np.random.default_rng(0).permutation(S)

    out = attn.apply(params, x, mem, ones, pos)
    out_p = attn.apply(params, x, mem[:, perm], ones[:, perm], pos[:, perm])
    assert jnp.allclose(out, out_p, atol=1e-4), "time-keyed RoPE must be permutation-invariant"
    # legacy slot RoPE is NOT (this is the d738e40 failure class)
    out_s = attn.apply(params, x, mem, ones)
    out_sp = attn.apply(params, x, mem[:, perm], ones[:, perm])
    assert not jnp.allclose(out_s, out_sp, atol=1e-4), "slot RoPE unexpectedly permutation-invariant"
    # gather-safety: masking half in place == physically gathering the kept half
    keep = np.zeros(S, bool); keep[::2] = True
    out_m = attn.apply(params, x, mem, jnp.asarray(keep, jnp.float32)[None], pos)
    out_g = attn.apply(params, x, mem[:, keep], ones[:, keep], pos[:, keep])
    assert jnp.allclose(out_m, out_g, atol=1e-4), "in-place mask != gather under time RoPE"
    print("OK memory_attention_time_rope")


def test_module_plumbing():
    """mem_pos rides through nn.remat/nn.scan (static_argnums shift, in_axes)."""
    from openpi.models.gemma import Config
    from mme_vla_suite.models.integration.history_gemma import Module
    cfg = Config(width=1024, depth=2, mlp_dim=256, num_heads=4, num_kv_heads=1, head_dim=32, lora_configs={})
    mod = Module(configs=[cfg], embed_dtype="float32", integration_type="modulation")
    B, T, S = 1, 4, 8
    x = jax.random.normal(jax.random.key(0), (B, T, 1024))
    positions = jnp.arange(T)[None]
    mask = jnp.ones((B, T, T), bool)
    mem = jax.random.normal(jax.random.key(1), (B, S, 1024))
    mmask = jnp.ones((B, S), jnp.float32)
    mpos = jax.random.randint(jax.random.key(2), (B, S), 0, 300)
    import flax.linen as nn
    # this Module overrides flax's `init` with a convenience wrapper; use the base initializer
    variables = nn.Module.init(mod, jax.random.key(3), [x], positions, mask, mem_seq=[mem], mem_mask=[mmask], mem_pos=[mpos])
    (o1,), _ = mod.apply(variables, [x], positions, mask, mem_seq=[mem], mem_mask=[mmask], mem_pos=[mpos])
    perm = np.random.default_rng(1).permutation(S)
    (o2,), _ = mod.apply(variables, [x], positions, mask, mem_seq=[mem[:, perm]], mem_mask=[mmask[:, perm]], mem_pos=[mpos[:, perm]])
    assert jnp.allclose(o1, o2, atol=1e-3), "backbone output changed under memory permutation with mem_pos"
    (o3,), _ = mod.apply(variables, [x], positions, mask, mem_seq=[mem], mem_mask=[mmask])  # legacy path still runs
    assert o3.shape == o1.shape
    print("OK module_plumbing")


if __name__ == "__main__":
    test_gumbel_topk()
    test_e2e_rounds_gradient_and_eval_equivalence()
    test_full_call_root_gather()
    test_memory_attention_time_rope()
    test_module_plumbing()
    print("ALL OK")
