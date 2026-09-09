"""Checks for the D&R training fixes (2026-09-07):
  * selector.gumbel_topk        -- exactly-K straight-through mask, train/eval same rule
  * PerceptualMemory e2e_tree   -- gradient reaches internal nodes; eval == legacy rounds
  * full forward (topk root, in-place mask over the budget-length sequence)
  * ablation configs build and run

    JAX_PLATFORMS=cpu PYTHONPATH=src:packages/openpi-client/src python tests/test_dnr_fixes.py
"""
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from mme_vla_suite.models.representation.percep_mem import PerceptualMemory
from mme_vla_suite.models.representation.selector import gumbel_topk, select_topk

CFG = "src/mme_vla_suite/models/config/robomme/perceptual-dnr-modul_bud64_pool128.yaml"
LEGACY = {  # same architecture, old training path
    "perceptual_memory.selector.sampling": "bernoulli",
    "perceptual_memory.selector.e2e_tree": False,
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
    return model.feature_encoder.encode_perceptual_memory(img, pos, state, time), mask


def test_e2e_rounds_gradient_and_eval_equivalence():
    e2e, cfg = build()
    legacy, _ = build(**LEGACY)
    assert e2e.n_reduce_rounds == 1 and e2e.e2e_tree
    hid, mask = _hid(e2e, cfg, n_real=100)
    probe = jax.random.normal(jax.random.key(5), (2, cfg.budget, cfg.memory_token_dim))

    # eval (no noise): e2e rounds pick the same survivors as the legacy rounds
    h_e, v_e = e2e._hierarchical_reduce(hid, mask)
    h_l, v_l = legacy._hierarchical_reduce(hid, mask)
    assert h_e.shape == h_l.shape == (2, cfg.budget, cfg.memory_token_dim)
    assert jnp.array_equal(v_e, v_l) and jnp.allclose(h_e, h_l, atol=1e-5)

    # gradient of a loss on the ROUND OUTPUT wrt selector params:
    #   e2e: nonzero (straight-through scaling of the gathered tokens)
    #   legacy: exactly zero (scoring is stop_gradient'd)
    def loss_e2e(m):
        h, v = m._hierarchical_reduce(hid, mask, None, jax.random.key(7))
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


def test_full_call_topk_root():
    model, cfg = build()
    K = model.num_keep
    for n_real in (100, None):
        img, pos, state, time, mask = rand_inputs(cfg, n_real=n_real, seed=3)
        for train in (True, False):
            tok, w, stats = _fwd(model, img, pos, state, time, mask, train=train, rng=jax.random.key(4))
            assert tok.shape == (2, cfg.budget, cfg.memory_token_dim), tok.shape  # mask in place
            assert w.shape == (2, cfg.budget)
            assert jnp.isfinite(tok).all()
            assert set(np.unique(np.asarray(w)).tolist()) <= {0.0, 1.0}
            assert (np.asarray(w.sum(1)) == K).all(), w.sum(1)  # exactly K kept
            for k in ("keep_frac", "reduce_keep_frac"):
                assert k in stats and jnp.isfinite(stats[k]).all(), k
            assert "ratio_loss" not in stats and "load_balance_loss" not in stats
    # whole-call gradient wrt selector params is finite and nonzero in training
    img, pos, state, time, mask = rand_inputs(cfg, n_real=100, seed=3)
    probe = jax.random.normal(jax.random.key(9), (2, cfg.budget, cfg.memory_token_dim))

    def loss(m):
        tok, w, _ = m(img, pos, state, time, mask, train=True, rng=jax.random.key(4))
        return jnp.sum(tok * probe * w[..., None])

    g = nnx.grad(loss)(model)
    n = sum(float(jnp.abs(x).sum()) for x in jax.tree.leaves(g.selector))
    assert np.isfinite(n) and n > 0, n
    print(f"OK full_call_topk_root (selector grad {n:.3g})")


def test_ablation_configs_build_and_run():
    """The single-fix ablations must construct and run train/eval forwards."""
    import glob, os
    paths = sorted(glob.glob("src/mme_vla_suite/models/config/robomme/perceptual-dnr-modul_bud64_pool128_abl*.yaml"))
    assert paths
    for path in paths:
        cfg = OmegaConf.load(path)
        model = PerceptualMemory(config=cfg, rngs=nnx.Rngs(0), dtype=jnp.float32)
        img, pos, state, time, mask = rand_inputs(cfg, n_real=100, seed=1)
        for train in (True, False):
            tok, w, stats = model(img, pos, state, time, mask, train=train, rng=jax.random.key(2))
            assert jnp.isfinite(tok).all() and w is not None
            assert tok.shape[1] == cfg.budget
            if model.sampling == "bernoulli" and train:
                assert "ratio_loss" in stats and "load_balance_loss" in stats
            else:
                assert "ratio_loss" not in stats
        print("OK ablation config", os.path.basename(path), f"e2e={model.e2e_tree} sampling={model.sampling}")


if __name__ == "__main__":
    test_gumbel_topk()
    test_e2e_rounds_gradient_and_eval_equivalence()
    test_full_call_topk_root()
    test_ablation_configs_build_and_run()
    print("ALL OK")
