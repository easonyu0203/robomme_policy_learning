"""Standalone checks for the `hierarchical_selection` perceptual-memory mode.

Covers the parts of `PerceptualMemory._hierarchical_reduce` that aren't obvious
from reading the code: static shapes under jit, the scoring-detached /
gather-differentiable gradient split (caltech hard_vit.py), NaN containment for
short (mostly-padding) episodes, and determinism of the reduction.

    uv run tests/test_hierarchical_reduction.py
"""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax

from mme_vla_suite.models.representation.percep_mem import PerceptualMemory
from mme_vla_suite.models.representation.selector import batch_gather
from mme_vla_suite.models.representation.selector import gumbel_softmax_hard
from mme_vla_suite.models.representation.selector import select_topk

CFG_PATH = "src/mme_vla_suite/models/config/robomme/perceptual-hiersel-modul.yaml"


def build(pool_budget=2048, **overrides):
    cfg = OmegaConf.load(CFG_PATH)
    cfg.pool_budget = pool_budget
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    model = PerceptualMemory(config=cfg, rngs=nnx.Rngs(0), dtype=jnp.float32)
    return model, cfg


def rand_inputs(cfg, b=2, n_real=None, seed=0):
    rng = np.random.default_rng(seed)
    p = cfg.pool_budget

    def mk(d):
        return jnp.asarray(rng.standard_normal((b, p, d)), dtype=jnp.float32)

    img = mk(cfg.memory_feature.img.input_dim)
    pos = mk(cfg.memory_feature.pos.input_dim)
    state = mk(cfg.memory_feature.state.input_dim)
    time = mk(cfg.memory_feature.time.input_dim)
    if n_real is None:
        mask = jnp.ones((b, p), dtype=bool)
    else:
        m = np.zeros((b, p), dtype=bool)
        m[:, :n_real] = True
        mask = jnp.asarray(m)
    return img, pos, state, time, mask


@nnx.jit(static_argnames=("train",))
def _forward(model, img, pos, state, time, mask, *, train, rng):
    return model(img, pos, state, time, mask, train=train, rng=rng)


def test_round_math():
    model, cfg = build(pool_budget=8192)
    # 8192 -> 4096 -> 2048 -> 1024 -> 512
    assert model.n_reduce_rounds == 4, model.n_reduce_rounds
    assert model.reduced_len == cfg.budget == 512
    assert model.reduce_chunk_size == 512 and model.reduce_chunk_keep == 256

    model, cfg = build(pool_budget=2048)
    assert model.n_reduce_rounds == 2
    assert model.reduced_len == 512

    # degenerate: pool == budget -> 0 rounds, _hierarchical_reduce is a no-op,
    # collapses to plain hard-select (the "is the selector cut better than
    # frameSamp?" control run).
    model, cfg = build(pool_budget=512)
    assert model.n_reduce_rounds == 0
    img, pos, state, time, mask = rand_inputs(cfg, n_real=400, seed=0)
    hid = model.feature_encoder.encode_perceptual_memory(img, pos, state, time)
    h, v = model._hierarchical_reduce(hid, mask)
    assert h.shape == hid.shape and jnp.array_equal(v, mask)
    ht, wt, lt = _forward(model, img, pos, state, time, mask, train=True, rng=jax.random.key(0))
    assert ht.shape == (2, 512, cfg.memory_token_dim) and jnp.isfinite(ht).all()
    print("OK round_math")


def test_shapes_and_finite():
    model, cfg = build(pool_budget=2048)
    dim = cfg.memory_token_dim
    for n_real in (50, 600, 1500, None):
        img, pos, state, time, mask = rand_inputs(cfg, n_real=n_real, seed=n_real or 0)

        h, w, losses = _forward(model, img, pos, state, time, mask,
                                train=True, rng=jax.random.key(1))
        assert h.shape == (2, cfg.budget, dim), h.shape
        assert w.shape == (2, cfg.budget), w.shape
        assert jnp.isfinite(h).all() and jnp.isfinite(w).all(), n_real
        for k in ("ratio_loss", "z_loss", "load_balance_loss", "keep_frac", "reduce_keep_frac"):
            assert k in losses and jnp.isfinite(losses[k]).all(), (n_real, k)

        # Eval keeps length == `budget` and masks in place (same as train) --
        # NOT a physical gather; see percep_mem.py's eval branch for why the
        # position-sensitive MemoryAttention consumer forbids repacking.
        gh, gm, stats = _forward(model, img, pos, state, time, mask,
                                 train=False, rng=jax.random.key(1))
        assert gh.shape == (2, cfg.budget, dim), gh.shape
        assert gm.shape == (2, cfg.budget), gm.shape
        assert jnp.isfinite(gh).all(), n_real
        assert set(np.unique(np.asarray(gm)).tolist()) <= {0.0, 1.0}, np.unique(gm)
        # exactly `num_keep` kept when enough real tokens survive the reduction;
        # never more than the real tokens available (padding is never selected)
        kept = np.asarray(gm.sum(axis=1))
        assert (kept <= model.num_keep).all(), (n_real, kept)
        if n_real is None or n_real >= 600:
            assert (kept == model.num_keep).all(), (n_real, kept)
    print("OK shapes_and_finite")


def test_eval_keep_all():
    """`selector.eval_keep_all` -> eval skips the final trained cut and returns a
    keep-weight equal to the post-reduction valid mask (nothing else dropped).
    Reduction rounds still run; train path is unchanged."""
    for pool_budget, n_rounds in ((512, 0), (1024, 1)):
        model, cfg = build(pool_budget=pool_budget,
                           **{"perceptual_memory.selector.eval_keep_all": True})
        dim = cfg.memory_token_dim
        assert model.eval_keep_all and model.n_reduce_rounds == n_rounds
        base, _ = build(pool_budget=pool_budget)  # same arch, cut NOT skipped

        for n_real in (300, 700, None):
            img, pos, state, time, mask = rand_inputs(cfg, n_real=n_real, seed=n_real or 1)

            gh, gm, _ = _forward(model, img, pos, state, time, mask,
                                 train=False, rng=jax.random.key(1))
            assert gh.shape == (2, cfg.budget, dim), (pool_budget, gh.shape)
            assert gm.shape == (2, cfg.budget), gm.shape
            assert set(np.unique(np.asarray(gm)).tolist()) <= {0.0, 1.0}, np.unique(gm)

            # keep-weight == exactly the mask coming out of the (still-run) reduction
            hid = model.feature_encoder.encode_perceptual_memory(img, pos, state, time)
            _, red_valid = model._hierarchical_reduce(hid, mask)
            assert jnp.array_equal(gm.astype(bool), red_valid), (pool_budget, n_real)

            # and that is strictly more than the trained cut would keep
            _, base_m, _ = _forward(base, img, pos, state, time, mask,
                                    train=False, rng=jax.random.key(1))
            assert gm.sum() >= base_m.sum(), (pool_budget, n_real)
            if n_real != 300:
                assert (gm.sum(axis=1) > model.num_keep).all(), (pool_budget, n_real)

        # train still does the Gumbel cut -- keep-weight is not the plain mask
        _, wt, _ = _forward(model, img, pos, state, time, mask,
                            train=True, rng=jax.random.key(2))
        assert wt.shape == (2, cfg.budget)
    print("OK eval_keep_all")


def test_no_recompile():
    model, cfg = build(pool_budget=2048)
    traces = []

    @nnx.jit(static_argnames=("train",))
    def f(model, img, pos, state, time, mask, *, train, rng):
        traces.append(1)  # Python side effect -> runs once per trace, not per call
        return model(img, pos, state, time, mask, train=train, rng=rng)

    a = rand_inputs(cfg, n_real=800, seed=1)
    b = rand_inputs(cfg, n_real=1900, seed=2)
    o1 = f(model, *a, train=True, rng=jax.random.key(0))[0]
    o2 = f(model, *b, train=True, rng=jax.random.key(0))[0]
    assert len(traces) == 1, f"retraced {len(traces)}x -- a dynamic shape leaked into the reduction"
    assert o1.shape == o2.shape and not jnp.allclose(o1, o2)
    print("OK no_recompile")


def _reduce_variant(model, hidden, valid, mode):
    """Mirror of PerceptualMemory._hierarchical_reduce with switchable grad flow.

    mode='scoring_detached' == the real implementation; 'none' == full autograd;
    'all_detached' == the whole reduction stop_gradient'd.
    """
    chunk, keep = model.reduce_chunk_size, model.reduce_chunk_keep
    b, n = hidden.shape[0], hidden.shape[1]
    h, v = hidden, valid
    for _ in range(model.n_reduce_rounds):
        nc = -(-n // chunk)
        pad = nc * chunk - n
        if pad:
            h = jnp.pad(h, ((0, 0), (0, pad), (0, 0)))
            v = jnp.pad(v, ((0, 0), (0, pad)))
        dim = h.shape[-1]
        hc, vc = h.reshape(b * nc, chunk, dim), v.reshape(b * nc, chunk)
        sc = model.selector(hc, vc)
        if mode != "none":
            sc = jax.lax.stop_gradient(sc)
        idx = select_topk(sc, vc, keep)
        hc = batch_gather(hc, idx)
        vc = batch_gather(vc[..., None], idx)[..., 0]
        n = nc * keep
        h, v = hc.reshape(b, n, dim), vc.reshape(b, n)
    if mode == "all_detached":
        h = jax.lax.stop_gradient(h)
    return h, v


def test_gradient_split():
    model, cfg = build(pool_budget=2048)
    img, pos, state, time, mask = rand_inputs(cfg, n_real=1400, seed=3)
    rng = jax.random.key(4)

    def grads_for(mode):
        def loss_fn(m):
            hid = m.feature_encoder.encode_perceptual_memory(img, pos, state, time)
            hid, v = _reduce_variant(m, hid, mask, mode)
            dec = gumbel_softmax_hard(m.selector(hid, v), rng)[..., 0]
            return hid.mean() + dec.mean()

        g = nnx.grad(loss_fn)(model)
        return optax.global_norm(g.feature_encoder), optax.global_norm(g.selector)

    fe_a, sel_a = grads_for("scoring_detached")
    fe_b, sel_b = grads_for("none")
    fe_c, sel_c = grads_for("all_detached")

    # scoring only feeds top_k indices (non-differentiable) -> stop_gradient on it
    # is a pure compute optimisation, identical gradients everywhere.
    assert jnp.allclose(fe_a, fe_b, rtol=1e-4, atol=1e-5), (fe_a, fe_b)
    assert jnp.allclose(sel_a, sel_b, rtol=1e-4, atol=1e-5), (sel_a, sel_b)

    # FeatureEncoder learns from survivors via the differentiable gather ...
    assert fe_a > 1e-6, fe_a
    # ... and loses that signal entirely if the whole reduction is detached.
    assert fe_c < 1e-6 < fe_a and not jnp.allclose(fe_a, fe_c)

    # Selector only ever trains from the final cut -> unchanged by detaching the reduction.
    assert jnp.allclose(sel_a, sel_c, rtol=1e-4, atol=1e-5), (sel_a, sel_c)
    print(f"OK gradient_split  (|g_fe| a={fe_a:.4g} b={fe_b:.4g} c={fe_c:.4g} | "
          f"|g_sel| a={sel_a:.4g} c={sel_c:.4g})")


def test_reduction_deterministic():
    model, cfg = build(pool_budget=2048)
    img, pos, state, time, mask = rand_inputs(cfg, n_real=1234, seed=5)
    hid = model.feature_encoder.encode_perceptual_memory(img, pos, state, time)
    h1, v1 = model._hierarchical_reduce(hid, mask)
    h2, v2 = model._hierarchical_reduce(hid, mask)
    assert jnp.array_equal(h1, h2) and jnp.array_equal(v1, v2)
    assert jnp.isfinite(h1).all()
    print("OK reduction_deterministic")


# ---------------------------------------------------------------------------
# multilevel pick (perceptual_memory.selector.multilevel): during training the
# final trained cut is fed a uniformly-random reduction-tree node's input, not
# always the root's. Nodes are numbered round-major -- one per reduce_chunk_size
# chunk of every round's input, then the root == n_nodes - 1.
# ---------------------------------------------------------------------------

ML = {"perceptual_memory.selector.multilevel": True}


def test_multilevel_node_math():
    # n_nodes = sum(chunks per round) + 1 (root) = 2 * (pool_budget / 512) - 1
    for pool_budget, want in ((512, 1), (1024, 3), (2048, 7), (4096, 15), (8192, 31)):
        model, _ = build(pool_budget=pool_budget, **ML)
        assert model.multilevel and model.n_nodes == want, (pool_budget, model.n_nodes)
    print("OK multilevel_node_math")


def test_multilevel_root_matches_baseline():
    # picking the root for every sample must be bit-identical to the plain
    # no-grad cascade -- i.e. the eval path is unchanged.
    model, cfg = build(pool_budget=2048, **ML)
    for n_real in (700, 1500, None):
        img, pos, state, time, mask = rand_inputs(cfg, n_real=n_real, seed=n_real or 0)
        hid = model.feature_encoder.encode_perceptual_memory(img, pos, state, time)
        root = jnp.full((img.shape[0],), model.n_nodes - 1, dtype=jnp.int32)
        h_tree, v_tree = model._tree_pick_input(hid, mask, root)
        h_ref, v_ref = model._hierarchical_reduce(hid, mask)
        assert jnp.array_equal(h_tree, h_ref) and jnp.array_equal(v_tree, v_ref), n_real
    print("OK multilevel_root_matches_baseline")


def test_multilevel_pick_slices():
    model, cfg = build(pool_budget=2048, **ML)  # 7 nodes: 0-3 leaves, 4-5 round2, 6 root
    img, pos, state, time, mask = rand_inputs(cfg, b=7, n_real=1400, seed=7)
    hid = model.feature_encoder.encode_perceptual_memory(img, pos, state, time)
    r1_h, r1_v = model._reduce_one_round(hid, mask)          # (7, 1024, d)
    r2_h, r2_v = model._reduce_one_round(r1_h, r1_v)         # (7, 512, d) == root input
    node_input = [
        (hid[:, 0:512],     mask[:, 0:512]),       # 0
        (hid[:, 512:1024],  mask[:, 512:1024]),    # 1
        (hid[:, 1024:1536], mask[:, 1024:1536]),   # 2
        (hid[:, 1536:2048], mask[:, 1536:2048]),   # 3
        (r1_h[:, 0:512],    r1_v[:, 0:512]),       # 4
        (r1_h[:, 512:1024], r1_v[:, 512:1024]),    # 5
        (r2_h,              r2_v),                 # 6 (root)
    ]
    # whole batch on one node
    for node, (want_h, want_v) in enumerate(node_input):
        picked = jnp.full((7,), node, dtype=jnp.int32)
        got_h, got_v = model._tree_pick_input(hid, mask, picked)
        assert jnp.array_equal(got_h, want_h) and jnp.array_equal(got_v, want_v), node
    # per-sample: row i picks node i
    got_h, got_v = model._tree_pick_input(hid, mask, jnp.arange(7, dtype=jnp.int32))
    for i in range(7):
        assert jnp.array_equal(got_h[i], node_input[i][0][i]), i
        assert jnp.array_equal(got_v[i], node_input[i][1][i]), i
    print("OK multilevel_pick_slices")


def test_multilevel_static_shape():
    model, cfg = build(pool_budget=2048, **ML)
    dim = cfg.memory_token_dim
    traces = []

    @nnx.jit
    def pick(model, hid, mask, picked):
        traces.append(1)  # once per trace, not per call
        return model._tree_pick_input(hid, mask, picked)

    img, pos, state, time, mask = rand_inputs(cfg, b=4, n_real=800, seed=1)
    hid = model.feature_encoder.encode_perceptual_memory(img, pos, state, time)
    for picked in (
        jnp.zeros(4, jnp.int32),
        jnp.full((4,), model.n_nodes - 1, jnp.int32),
        jnp.array([0, 2, 4, 6], jnp.int32),
    ):
        h, v = pick(model, hid, mask, picked)
        assert h.shape == (4, 512, dim) and v.shape == (4, 512)
        assert jnp.isfinite(h).all()
    assert len(traces) == 1, f"retraced {len(traces)}x -- a dynamic shape leaked"
    print("OK multilevel_static_shape")


def test_multilevel_gradient_routing():
    model, cfg = build(pool_budget=2048, **ML)
    img, pos, state, time, mask = rand_inputs(cfg, b=4, n_real=1400, seed=3)
    rng = jax.random.key(4)

    def grads_for(picked_node):
        picked = jnp.full((4,), picked_node, dtype=jnp.int32)

        def loss_fn(m):
            hid = m.feature_encoder.encode_perceptual_memory(img, pos, state, time)
            ph, pv = m._tree_pick_input(hid, mask, picked)
            dec = gumbel_softmax_hard(m.selector(ph, pv), rng)[..., 0]
            return ph.mean() + dec.mean()

        g = nnx.grad(loss_fn)(model)
        return optax.global_norm(g.feature_encoder), optax.global_norm(g.selector)

    # leaf pick (node 0): the trained cut runs on hid[:, :512]; the no-grad
    # rounds run but leak no gradient -> identical to a standalone cut on that
    # slice with no tree at all.
    fe0, sel0 = grads_for(0)
    assert fe0 > 1e-6 and sel0 > 1e-6, (fe0, sel0)

    def loss_ref(m):
        hid = m.feature_encoder.encode_perceptual_memory(img, pos, state, time)
        dec = gumbel_softmax_hard(m.selector(hid[:, :512], mask[:, :512]), rng)[..., 0]
        return hid[:, :512].mean() + dec.mean()

    g_ref = nnx.grad(loss_ref)(model)
    assert jnp.allclose(fe0, optax.global_norm(g_ref.feature_encoder), rtol=1e-4, atol=1e-5)
    assert jnp.allclose(sel0, optax.global_norm(g_ref.selector), rtol=1e-4, atol=1e-5)

    # root pick still trains both (via the differentiable survivor gather).
    fe_root, sel_root = grads_for(model.n_nodes - 1)
    assert fe_root > 1e-6 and sel_root > 1e-6, (fe_root, sel_root)
    print(f"OK multilevel_gradient_routing  (leaf |g_fe|={fe0:.4g} |g_sel|={sel0:.4g})")


def test_multilevel_call_path():
    ml, cfg = build(pool_budget=2048, **ML)
    base, _ = build(pool_budget=2048)  # same arch + seed, multilevel off
    dim = cfg.memory_token_dim
    img, pos, state, time, mask = rand_inputs(cfg, n_real=1400, seed=11)

    h, w, losses = _forward(ml, img, pos, state, time, mask,
                            train=True, rng=jax.random.key(1))
    assert h.shape == (2, cfg.budget, dim) and w.shape == (2, cfg.budget)
    assert jnp.isfinite(h).all() and jnp.isfinite(w).all()
    assert "picked_node" in losses and jnp.isfinite(losses["picked_node"])
    for k in ("ratio_loss", "z_loss", "load_balance_loss", "keep_frac", "reduce_keep_frac"):
        assert k in losses, k

    # eval ignores multilevel -> identical to the non-multilevel model
    he, we, _ = _forward(ml, img, pos, state, time, mask, train=False, rng=jax.random.key(1))
    hb, wb, _ = _forward(base, img, pos, state, time, mask, train=False, rng=jax.random.key(1))
    assert jnp.array_equal(he, hb) and jnp.array_equal(we, wb)
    print("OK multilevel_call_path")


if __name__ == "__main__":
    test_round_math()
    test_reduction_deterministic()
    test_shapes_and_finite()
    test_eval_keep_all()
    test_no_recompile()
    test_gradient_split()
    test_multilevel_node_math()
    test_multilevel_root_matches_baseline()
    test_multilevel_pick_slices()
    test_multilevel_static_shape()
    test_multilevel_gradient_routing()
    test_multilevel_call_path()
    print("\nall hierarchical_selection checks passed")
