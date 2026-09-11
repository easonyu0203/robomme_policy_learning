"""Replay FrameSamp and TokenDrop memory over a rollout's frame history.

Both baselines pick their tokens with parameter-free rules that live in
MemoryBuffer, so no checkpoint is needed — given the same frames they would
build the same memory whichever policy produced them. That lets us compare all
three mechanisms on one trajectory.

  python scripts/viz_baseline_memory.py --history <dump>/history --step 422 \
      --out-dir <dir> [--td-budget 512] [--fs-budget 64]
"""
import argparse
import json
import pathlib

import numpy as np
from PIL import Image


def load_history(hist_dir, upto):
    frames = {}
    for f in sorted(pathlib.Path(hist_dir).glob("frame_*.npy")):
        i = int(f.stem.split("_")[1])
        if i <= upto:
            frames[i] = np.load(f)          # (v, h, w, 3) uint8
    return frames


def even_sampling_indices(step_idx, max_size):
    """Mirror of shared/data_utils.even_sampling_indices."""
    if step_idx < max_size:
        return list(range(step_idx + 1))
    return np.linspace(0, step_idx, max_size, dtype=np.int32).tolist()


def framesamp(frames, step_idx, budget, token_per_image):
    """Even frame sampling: every patch of each chosen frame is kept."""
    n = budget // token_per_image
    idx = even_sampling_indices(step_idx, n)
    grid = int(round(token_per_image ** 0.5))
    return idx, {i: set(range(token_per_image)) for i in idx}, grid


def tokendrop(frames, step_idx, budget, stride=8, keptsize=2048, grid=8):
    """Pixel-difference token dropping (MemoryBuffer._process_token_drop_score).

    Frame 0 seeds the heap with a sentinel score of 1000 for all of its patches,
    so it is never evicted; every `stride` frames the patches that changed most
    since the last scored frame are pushed, and the heap keeps the highest.
    """
    import heapq
    heap, last = [], -1
    P = grid * grid
    for t in sorted(frames):
        if t > step_idx:
            break
        if t == 0:
            for p in range(P):
                heapq.heappush(heap, (1000.0, t, 0, p))
        if t == last + stride:
            prev = frames[max(0, last)][0].astype(np.float32) / 255.0 * 2 - 1
            cur = frames[t][0].astype(np.float32) / 255.0 * 2 - 1
            h, w = prev.shape[0] // grid, prev.shape[1] // grid
            pv = prev.reshape(grid, h, grid, w, 3).transpose(0, 2, 1, 3, 4).reshape(P, -1)
            cv = cur.reshape(grid, h, grid, w, 3).transpose(0, 2, 1, 3, 4).reshape(P, -1)
            diff = np.abs(pv - cv).mean(-1)
            for p in range(P):
                if diff[p] < 1e-4:
                    continue
                heapq.heappush(heap, (float(diff[p]), t, 0, p))
                if len(heap) > keptsize:
                    heapq.heappop(heap)
            last += stride
    asc = []
    while heap:
        asc.append(heapq.heappop(heap))
    kept = sorted([x for x in asc if x[1] <= step_idx][-budget:], key=lambda x: (x[1], x[3]))
    per = {}
    for _, t, _, p in kept:
        per.setdefault(t, set()).add(p)
    return sorted(per), per, grid


def export(frames, order, keep, grid, out_dir, tag, step, scale=2, max_frames=None):
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(order, key=lambda t: -len(keep.get(t, ()))) if max_frames else order
    if max_frames:
        order = sorted(order[:max_frames])
    man = {"tag": tag, "step": step, "grid": grid, "frames": []}
    for f_i, t in enumerate(order):
        im = Image.fromarray(frames[t][0]).convert("RGB")
        if scale != 1:
            im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); px = ov.load()
        ph, pw = im.height / grid, im.width / grid
        k = keep.get(t, set())
        for p in range(grid * grid):
            if p in k:
                continue
            gr, gc = p // grid, p % grid
            for y in range(int(gr * ph), int((gr + 1) * ph)):
                for x in range(int(gc * pw), int((gc + 1) * pw)):
                    px[x, y] = (0, 0, 0, 237)
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
        fn = f"{tag}_step{step:05d}_f{f_i}.png"
        im.save(out_dir / fn)
        man["frames"].append({"t": int(t), "file": fn, "kept": len(k)})
    (out_dir / f"manifest_{tag}_step{step:05d}.json").write_text(json.dumps(man, indent=1))
    tot = sum(f["kept"] for f in man["frames"])
    print(f"{tag}: {len(order)} frames shown, {tot} tokens on them, grid {grid}x{grid}")
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fs-budget", type=int, default=64)
    ap.add_argument("--fs-tokens-per-image", type=int, default=16)
    ap.add_argument("--td-budget", type=int, default=512)
    ap.add_argument("--td-max-frames", type=int, default=8)
    args = ap.parse_args()

    frames = load_history(args.history, args.step)
    print(f"history: {len(frames)} frames up to step {args.step}")

    order, keep, grid = framesamp(frames, args.step, args.fs_budget, args.fs_tokens_per_image)
    export(frames, order, keep, grid, args.out_dir, f"framesamp{args.fs_budget}", args.step)

    order, keep, grid = tokendrop(frames, args.step, args.td_budget)
    n_src = len(order)
    m = export(frames, order, keep, grid, args.out_dir, f"tokendrop{args.td_budget}",
               args.step, max_frames=args.td_max_frames)
    m["n_source_frames"] = n_src
    (pathlib.Path(args.out_dir) /
     f"manifest_tokendrop{args.td_budget}_step{args.step:05d}.json").write_text(json.dumps(m, indent=1))
    print(f"tokendrop{args.td_budget}: tokens drawn from {n_src} distinct frames in total")


if __name__ == "__main__":
    main()
