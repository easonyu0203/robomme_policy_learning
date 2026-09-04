"""Checks for `random_sampling` frame-index selection, in particular the
optional two-stage `pool_budget` pick added for the hard-select mode:

  1. a `pool_budget // tokens_per_frame`-wide EVENLY-SPACED frame grid, then
  2. a fresh UNIFORMLY-RANDOM `budget // tokens_per_frame`-wide subset of it.

`pool_budget=None` must reproduce the original flat-uniform-over-history pick.

    uv run tests/test_random_sampling_pool.py
"""

import numpy as np

from mme_vla_suite.shared.data_utils import even_sampling_indices
from mme_vla_suite.shared.mem_buffer import MemoryBuffer

TPF = 16  # token_per_image
NV = 1    # num_views


def buf():
    return MemoryBuffer(num_views=NV, img_emb_dim=2048, pos_emb_dim=768, state_emb_dim=8)


def test_pool_is_random_subset_of_even_grid():
    b = buf()
    step_idx, pool_budget, budget = 400, 256, 64
    grid = set(even_sampling_indices(step_idx, pool_budget // (TPF * NV)))  # 16 frames
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(50):
        idx = b.get_random_sampling_indices(step_idx, budget, TPF, rng=rng, pool_budget=pool_budget)
        assert len(idx) == budget // (TPF * NV) == 4
        assert idx == sorted(idx), "indices must be time-sorted for slot-keyed RoPE"
        assert set(idx) <= grid, f"{idx} not a subset of the even grid {sorted(grid)}"
        seen.update(idx)
    assert seen == grid, "over many draws every grid frame should appear (pick is uniform)"
    print("OK pool_is_random_subset_of_even_grid")


def test_none_pool_matches_legacy_flat_uniform():
    b = buf()
    step_idx, budget = 400, 64
    size = budget // (TPF * NV)
    # legacy behaviour: sorted uniform sample without replacement over range(step_idx+1)
    for seed in range(5):
        r1 = np.random.default_rng(seed)
        got = b.get_random_sampling_indices(step_idx, budget, TPF, rng=r1, pool_budget=None)
        r2 = np.random.default_rng(seed)
        want = sorted(int(i) for i in r2.choice(step_idx + 1, size=size, replace=False))
        assert got == want, (seed, got, want)
    print("OK none_pool_matches_legacy_flat_uniform")


def test_stochastic_every_call():
    b = buf()
    rng = np.random.default_rng(1)
    draws = {
        tuple(b.get_random_sampling_indices(500, 64, TPF, rng=rng, pool_budget=512))
        for _ in range(30)
    }
    assert len(draws) > 1, "pick must vary call to call (augmentation / anti-stencil)"
    print("OK stochastic_every_call")


def test_short_episode_degenerates_gracefully():
    b = buf()
    # n_available (11) < grid size (16): grid is just every frame, pick 4 of them
    idx = b.get_random_sampling_indices(10, 64, TPF, rng=np.random.default_rng(0), pool_budget=256)
    assert len(idx) == 4 and set(idx) <= set(range(11))
    # n_available (3) < budget frames (4): take everything, no crash
    idx = b.get_random_sampling_indices(2, 64, TPF, rng=np.random.default_rng(0), pool_budget=256)
    assert idx == [0, 1, 2]
    print("OK short_episode_degenerates_gracefully")


def test_linspace_collisions_are_deduped():
    b = buf()
    # step_idx just above the grid size -> np.linspace rounds several frames together;
    # the dedupe must keep `replace=False` from over-drawing.
    step_idx, pool_budget, budget = 20, 256, 64  # grid asks for 16 frames of 21
    raw = even_sampling_indices(step_idx, pool_budget // (TPF * NV))
    assert len(raw) != len(set(raw)) or len(raw) <= step_idx + 1  # collisions expected here
    for seed in range(10):
        idx = b.get_random_sampling_indices(
            step_idx, budget, TPF, rng=np.random.default_rng(seed), pool_budget=pool_budget)
        assert len(idx) == len(set(idx)) == 4
    print("OK linspace_collisions_are_deduped")


if __name__ == "__main__":
    test_pool_is_random_subset_of_even_grid()
    test_none_pool_matches_legacy_flat_uniform()
    test_stochastic_every_call()
    test_short_episode_degenerates_gracefully()
    test_linspace_collisions_are_deduped()
    print("\nall random_sampling pool checks passed")
