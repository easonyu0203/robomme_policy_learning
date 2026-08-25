#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import shutil
from pathlib import Path
from zipfile import ZipFile, BadZipFile

from tqdm import tqdm


def determine_mode(zip_path: Path) -> str:
    """
    Decide how this zip should be unpacked.

    Modes:
    - "features_episode":  features/episode_*.zip
        -> keep the episode_* directory layout under features/
    - "images_zip":        images.zip
        -> create an images/ folder next to the zip and flatten inside it
    - "flat":              everything else
        -> flatten into the same directory as the .zip
    """
    parent = zip_path.parent
    stem = zip_path.stem

    # Special case: robomme features episodes
    if parent.name == "features" and stem.startswith("episode_"):
        return "features_episode"

    # Special case: images.zip -> images/ folder
    if stem == "images":
        return "images_zip"

    # Default: flatten into the same directory as the zip file
    return "flat"


def dest_for_member(mode: str, zip_stem: str, out_dir: Path, name: str):
    """Compute the on-disk destination path for one zip member, or None to skip it."""
    internal_path = Path(name)

    if mode == "features_episode":
        # Keep internal structure but avoid duplicating the top-level
        # episode_* directory if it matches the zip's stem.
        parts = internal_path.parts
        rel_parts = parts[1:] if parts and parts[0] == zip_stem else parts
        if not rel_parts:
            return None
        return out_dir / Path(*rel_parts)

    # "images_zip" and "flat": put everything flat in out_dir
    return out_dir / internal_path.name


def extract_chunk(zip_path: Path, mode: str, out_dir: Path, names: list) -> bool:
    """Extract a subset of a zip's members. Runs in a worker process, so large
    zips can be split across several workers instead of unzipped on one core."""
    try:
        with ZipFile(zip_path, "r") as zf:
            for name in names:
                dest_path = dest_for_member(mode, zip_path.stem, out_dir, name)
                if dest_path is None:
                    continue
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name, "r") as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        return True
    except Exception as e:
        print(f"[err]  {zip_path} (chunk of {len(names)}): {e}")
        return False


def plan_zip(zip_path: Path, overwrite: bool, chunk_size: int):
    """Decide what to do with one zip: skip / bad / a list of member-name chunks to extract."""
    zip_path = zip_path.resolve()
    mode = determine_mode(zip_path)

    if mode == "features_episode":
        out_dir = zip_path.parent / zip_path.stem
    elif mode == "images_zip":
        out_dir = zip_path.with_suffix("")
    else:  # "flat"
        out_dir = zip_path.parent

    if out_dir.exists() and mode != "flat" and not overwrite:
        # For "flat" mode we can't easily decide if we're "done", so we always
        # extract unless the user explicitly skips it by not running the script.
        return mode, out_dir, "skip", []

    try:
        with ZipFile(zip_path, "r") as zf:
            names = [m.filename for m in zf.infolist() if not m.filename.endswith("/")]
    except BadZipFile:
        return mode, out_dir, "bad", []

    out_dir.mkdir(exist_ok=True, parents=True)
    chunks = [names[i : i + chunk_size] for i in range(0, len(names), chunk_size)] or [[]]
    return mode, out_dir, "extract", chunks


def _worker(args):
    zip_path, mode, out_dir, names = args
    return zip_path, extract_chunk(zip_path, mode, out_dir, names)


def find_zip_files(root: Path):
    return [p for p in root.rglob("*.zip") if p.is_file()]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Unzip dataset .zip files with special rules:\n"
            "  - features/episode_*.zip  -> keep episode_* layout under features/\n"
            "  - images.zip              -> unzip into images/ folder\n"
            "  - all other *.zip         -> flatten into the zip's directory"
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="data",
        help="Root directory to search for .zip files (default: data)",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=0,
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help=(
            "Members per extraction task (default: 200). Zips with more members "
            "than this get split across several workers instead of running on "
            "one core; small zips end up as a single chunk."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing structured output folders "
            "(features episodes, images/). "
            "Flat mode always writes/overwrites files in-place."
        ),
    )
    parser.add_argument(
        "--delete-after",
        action="store_true",
        help=(
            "Delete each .zip once it's confirmed extracted (or was already "
            "extracted in a prior run), to save disk. A zip is left in place "
            "if extraction fails (bad zip / error)."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root directory does not exist: {root}")
        return

    zips = find_zip_files(root)
    if not zips:
        print(f"No .zip files found under {root}")
        return
    print(f"Found {len(zips)} zip files under {root}")

    tasks = []  # (zip_path, mode, out_dir, names)
    pending = {}  # zip_path -> [remaining_chunks, all_ok_so_far, out_dir, mode]
    for zp in zips:
        mode, out_dir, status, chunks = plan_zip(zp, args.overwrite, args.chunk_size)
        if status == "skip":
            print(f"[skip] {zp} -> {out_dir} (already exists)")
            if args.delete_after:
                zp.unlink(missing_ok=True)
            continue
        if status == "bad":
            print(f"[bad]  {zp} is not a valid zip file")
            continue
        pending[zp] = [len(chunks), True, out_dir, mode]
        tasks.extend((zp, mode, out_dir, chunk) for chunk in chunks)

    if not tasks:
        return

    procs = args.processes or mp.cpu_count()
    with mp.Pool(processes=procs) as pool:
        for zip_path, ok in tqdm(
            pool.imap_unordered(_worker, tasks), total=len(tasks), desc="Unzipping"
        ):
            info = pending[zip_path]
            info[0] -= 1
            info[1] = info[1] and ok
            if info[0] == 0:
                _, all_ok, out_dir, mode = info
                if all_ok:
                    print(f"[ok]   {zip_path} -> {out_dir} (mode={mode})")
                    if args.delete_after:
                        zip_path.unlink(missing_ok=True)
                else:
                    print(f"[err]  Failed on {zip_path}")


if __name__ == "__main__":
    main()
