"""Visualise the D&R memory tree for one episode step.

Replays the trained selector over a memory pool dumped by the eval policy
(set DNR_DUMP_DIR when serving) and draws, for every pooled frame, which of its
16 image patches survive each level of the reduction tree.

Mask darkness encodes WHERE a token died: the deeper (lower) the level that
dropped it, the darker the mask. Kept tokens are left clear.

  JAX_PLATFORMS=cpu PYTHONPATH=src:packages/openpi-client/src python \
    scripts/viz_memory_tree.py --ckpt <ckpt_dir> --pool <pool_stepXXXXX.npz> --out fig.png
"""
import argparse
import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np


def _as_float(a):
    """numpy has no bfloat16, so np.savez stored those arrays as an opaque 2-byte
    void dtype. Reinterpret them before handing anything to JAX."""
    if a.dtype.kind == "V" and a.dtype.itemsize == 2:
        import ml_dtypes
        a = a.view(ml_dtypes.bfloat16)
    return np.asarray(a, dtype=np.float32)


def build_mem_encoder(ckpt_dir: pathlib.Path):
    from openpi.models import model as _model
    from mme_vla_suite.training import config as _config

    cfg = _config.get_config("mme_vla_suite")
    hist = (ckpt_dir.parent / "history_config.txt").read_text().strip()
    cfg = dataclasses.replace(
        cfg, model=dataclasses.replace(cfg.model, history_config=hist, use_history=True)
    )
    model = cfg.model.load(_model.restore_params(ckpt_dir / "params", dtype=jnp.float32))
    return model.mem_encoder, hist


def run_tree(mem, pool, rng=None, noise_scale=None):
    """Replay the tree. Returns (depth, info).

    depth[i] for pool token i:  0 = dropped by its round-1 node (deepest level),
    1 = dropped by the root cut, 2 = kept and handed to the policy.
    """
    from mme_vla_suite.models.representation.selector import batch_gather, gumbel_topk

    img, pos, state, tim = (
        jnp.asarray(_as_float(pool[k]))[None]
        for k in ("static_image_emb", "static_pos_emb", "static_state_emb", "static_time_emb")
    )
    valid = jnp.asarray(pool["static_mask"])[None]
    hidden = mem.feature_encoder.encode_perceptual_memory(img, pos, state, tim)

    n = hidden.shape[1]
    depth = np.zeros(n, dtype=np.int32)
    orig = jnp.arange(n)[None]  # original index carried through the gathers
    chunk, keep = mem.reduce_chunk_size, mem.reduce_chunk_keep
    ns = mem.noise_scale if noise_scale is None else noise_scale
    info = {"rounds": []}

    for r in range(mem.n_reduce_rounds):
        b, ln, d = hidden.shape
        nc = ln // chunk
        hc = hidden.reshape(b * nc, chunk, d)
        vc = valid.reshape(b * nc, chunk)
        rng_r = None if rng is None else jax.random.fold_in(rng, r)
        logits = mem.selector(hc, vc)
        _, idx = gumbel_topk(
            logits, vc, keep, rng_r, tau=mem.tau,
            noise_scale=ns, score_norm=mem.round_score_norm,
        )
        oc = orig.reshape(b * nc, chunk)
        kept_orig = np.asarray(batch_gather(oc[..., None], idx)[..., 0]).reshape(-1)
        depth[kept_orig] = 1  # survived this round (may still die at the root)
        hidden = batch_gather(hc, idx).reshape(b, nc * keep, d)
        valid = batch_gather(vc[..., None], idx)[..., 0].reshape(b, nc * keep)
        orig = jnp.asarray(kept_orig)[None]
        info["rounds"].append({"n_in": ln, "n_out": nc * keep, "n_chunks": nc})

    logits = mem.selector(hidden, valid)
    _, idx = gumbel_topk(
        logits, valid, mem.num_keep, rng, tau=mem.tau,
        noise_scale=ns, score_norm=mem.score_norm,
    )
    root_orig = np.asarray(batch_gather(orig[..., None], idx)[..., 0]).reshape(-1)
    depth[root_orig] = 2
    info["root"] = {"n_in": int(hidden.shape[1]), "n_keep": int(mem.num_keep)}
    return depth, info



def export_frames(pool, depths, row_names, out_dir, token_per_image=16, grid=4,
                  mask_color=(0, 0, 0), scale=2):
    """Write ONE png per (row, frame) plus a manifest, so a slide can place each
    frame as its own object instead of embedding a single composite figure."""
    import json
    from PIL import Image
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    frames = pool["frames"]
    idxs = pool["pool_indices"]
    step = int(pool["step_idx"]); exec_start = int(pool["exec_start_idx"])
    ALPHA = {0: 0.93, 1: 0.42, 2: 0.0}
    man = {"step": step, "exec_start": exec_start, "rows": [], "frames": []}
    F = frames.shape[0]
    for f in range(F):
        man["frames"].append({"t": int(idxs[f]),
                              "phase": "demo" if idxs[f] < exec_start else "exec"})
    for r, (depth, name) in enumerate(zip(depths, row_names)):
        cells = []
        for f in range(F):
            im = Image.fromarray(frames[f, 0]).convert("RGB")
            if scale != 1:
                im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
            ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
            px = ov.load()
            ph, pw = im.height / grid, im.width / grid
            for p in range(token_per_image):
                a = ALPHA[int(depth[f * token_per_image + p])]
                if a <= 0:
                    continue
                gr, gc = p // grid, p % grid
                for y in range(int(gr * ph), int((gr + 1) * ph)):
                    for x in range(int(gc * pw), int((gc + 1) * pw)):
                        px[x, y] = (*mask_color, int(255 * a))
            im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
            fn = f"step{step:05d}_row{r}_f{f}.png"
            im.save(out_dir / fn)
            seg = depth[f * token_per_image:(f + 1) * token_per_image]
            cells.append({"file": fn, "kept": int((seg == 2).sum()),
                          "root_dropped": int((seg == 1).sum()),
                          "r1_dropped": int((seg == 0).sum())})
        man["rows"].append({"name": name, "cells": cells})
    (out_dir / f"manifest_step{step:05d}.json").write_text(json.dumps(man, indent=1))
    print(f"exported {len(depths) * F} frame images for step {step} -> {out_dir}")


def draw(pool, depths, titles, out_path, mem_info, token_per_image=16, grid=4,
         mask_color="black", bare=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    frames = pool["frames"]            # (F, V, H, W, 3) uint8
    idxs = pool["pool_indices"]
    step = int(pool["step_idx"])
    exec_start = int(pool["exec_start_idx"])
    F = frames.shape[0]
    # deepest level first -> darkest. 0 = dropped in round 1, 1 = dropped at root.
    ALPHA = {0: 0.93, 1: 0.42, 2: 0.0}
    grid_color = "black" if mask_color == "white" else "white"
    DEMO_C, EXEC_C = "#8983BF", "#54B345"   # demonstration video vs the robot's own rollout
    frames_per_chunk = mem_info["frames_per_chunk"]
    rows = len(depths)
    fig, axes = plt.subplots(rows, F, figsize=(1.6 * F, 1.95 * rows + 1.35), squeeze=False)
    for r, (depth, title) in enumerate(zip(depths, titles)):
        for f in range(F):
            ax = axes[r][f]
            ax.imshow(frames[f, 0])
            H, W = frames.shape[2], frames.shape[3]
            ph, pw = H / grid, W / grid
            for p in range(token_per_image):
                d = int(depth[f * token_per_image + p])
                gr, gc = p // grid, p % grid
                if ALPHA[d] > 0:
                    ax.add_patch(Rectangle((gc * pw, gr * ph), pw, ph,
                                           facecolor=mask_color, edgecolor="none", alpha=ALPHA[d]))
                ax.add_patch(Rectangle((gc * pw, gr * ph), pw, ph, fill=False,
                                       edgecolor=grid_color, linewidth=0.35, alpha=0.45))
            seg = depth[f * token_per_image:(f + 1) * token_per_image]
            kept = int((seg == 2).sum())
            is_demo = idxs[f] < exec_start
            phase_c = DEMO_C if is_demo else EXEC_C
            ax.set_title(f"t={idxs[f]}\nkept {kept}/{token_per_image}", fontsize=7.5,
                         color=phase_c)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(phase_c); sp.set_linewidth(3.0)
            # round-1 chunk boundary: each chunk is scored by its own node
            if f % frames_per_chunk == 0 and f > 0:
                ax.axvline(-0.006 * frames.shape[3], color="black", linestyle=(0, (3, 2)),
                           linewidth=1.6, clip_on=False)
            if f == 0:
                ax.set_ylabel(title, fontsize=9)
        # per-chunk survivor counts under the row
        n_chunks = F // frames_per_chunk
        for c in range(n_chunks):
            lo, hi = c * frames_per_chunk * token_per_image, (c + 1) * frames_per_chunk * token_per_image
            seg = depth[lo:hi]
            axes[r][c * frames_per_chunk].set_xlabel(
                f"round-1 node {chr(65+c)}: {int((seg >= 1).sum())}/{hi-lo} survive"
                f"  →  {int((seg == 2).sum())} reach the policy",
                fontsize=8, loc="left", color="#8983BF")
    # spanning phase bands above the first row
    n_demo = int((idxs < exec_start).sum())
    for lo, hi, lab, col in ((0, n_demo, "demonstration video — the manner to imitate", DEMO_C),
                             (n_demo, F, "execution — the robot's own rollout", EXEC_C)):
        if hi <= lo:
            continue
        x0 = axes[0][lo].get_position().x0
        x1 = axes[0][hi - 1].get_position().x1
        y = axes[0][lo].get_position().y1 + 0.022
        fig.add_artist(plt.Line2D([x0, x1], [y, y], color=col, linewidth=4,
                                  transform=fig.transFigure, clip_on=False))
        fig.text((x0 + x1) / 2, y + 0.008, lab, ha="center", va="bottom",
                 fontsize=9.5, color=col, transform=fig.transFigure)

    handles = [
        Patch(facecolor=mask_color, edgecolor="grey", alpha=0.93,
              label="dropped by its round-1 node (lower level)"),
        Patch(facecolor=mask_color, edgecolor="grey", alpha=0.42, label="dropped by the root cut"),
        Patch(facecolor="none", edgecolor="grey", label="kept — the 32 tokens the policy reads"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    if not bare:
        fig.suptitle(
            f"D&R memory tree · MoveCube ep0 (success) · abl3_slotrope_ab @80k · control step {step}\n"
            f"pool {mem_info['pool']} tokens = {F} frames × {token_per_image} patches"
            f"  →  2 round-1 nodes keep {mem_info['keep']} each  →  root keeps {mem_info['num_keep']}"
            f"\n"
            f"both rows use the SAME memory pool — the only difference is the selector's "
            f"training-time Gumbel noise, not the data stream",
            fontsize=10.5, y=0.985)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93 if bare else 0.80])
    fig.savefig(out_path, dpi=180, facecolor="white")
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=pathlib.Path)
    ap.add_argument("--pool", required=True, type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--export-frames", type=pathlib.Path, default=None,
                    help="write one png per (row, frame) + a manifest instead of a figure")
    ap.add_argument("--bare", action="store_true",
                    help="omit the figure title (for slides that carry it already)")
    ap.add_argument("--mask-color", choices=["black", "white"], default="black",
                    help="colour of the drop mask; deeper level = more opaque either way")
    ap.add_argument("--noise-scales", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                    help="Gumbel scales to render as training rows; 0 == the deployment rule")
    args = ap.parse_args()

    mem, hist = build_mem_encoder(args.ckpt)
    print(f"history config: {hist}")
    print(f"tree: pool {mem.input_len} -> rounds {mem.n_reduce_rounds} "
          f"(chunk {mem.reduce_chunk_size} keep {mem.reduce_chunk_keep}) -> root keeps {mem.num_keep}")
    pool = np.load(args.pool)

    d_test, info = run_tree(mem, pool, rng=None)
    kept0 = set(np.flatnonzero(d_test == 2).tolist())
    depths, titles = [d_test], ["deployment\ndeterministic"]
    for ns in args.noise_scales:
        d, _ = run_tree(mem, pool, rng=jax.random.key(args.train_seed), noise_scale=ns)
        ov = len(set(np.flatnonzero(d == 2).tolist()) & kept0)
        depths.append(d)
        titles.append(f"training  $\\lambda$={ns:g}\n{ov}/{mem.num_keep} match")
        print(f"noise_scale {ns:g}: {ov}/{mem.num_keep} of the kept tokens match deployment")
    print("tree info:", info)

    mem_info = {"pool": mem.input_len, "keep": mem.reduce_chunk_keep,
                "num_keep": mem.num_keep,
                "frames_per_chunk": mem.reduce_chunk_size // 16}
    if args.export_frames:
        export_frames(pool, depths, titles, args.export_frames,
                      mask_color=(255, 255, 255) if args.mask_color == "white" else (0, 0, 0))
    else:
        draw(pool, depths, titles, args.out, mem_info, mask_color=args.mask_color,
             bare=args.bare)


if __name__ == "__main__":
    main()
