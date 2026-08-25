#!/usr/bin/env python3
import argparse
import multiprocessing as mp
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


def unzip_one(zip_path: Path, overwrite: bool = False, delete_after: bool = False) -> None:
    zip_path = zip_path.resolve()
    mode = determine_mode(zip_path)

    if mode == "features_episode":
        # e.g. .../features/episode_0.zip -> .../features/episode_0/
        out_dir = zip_path.parent / zip_path.stem
    elif mode == "images_zip":
        # e.g. .../images.zip -> .../images/
        out_dir = zip_path.with_suffix("")
    else:  # "flat"
        # Put files directly into the same directory as the zip
        out_dir = zip_path.parent

    if out_dir.exists() and mode != "flat" and not overwrite:
        # For "flat" mode we can't easily decide if we're "done", so we always
        # extract unless the user explicitly skips it by not running the script.
        print(f"[skip] {zip_path} -> {out_dir} (already exists)")
        if delete_after:
            zip_path.unlink(missing_ok=True)
        return

    try:
        out_dir.mkdir(exist_ok=True, parents=True)
        with ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                name = member.filename

                # Skip directory entries; we'll create directories as needed
                if name.endswith("/"):
                    continue

                internal_path = Path(name)

                if mode == "features_episode":
                    # Keep internal structure but avoid duplicating the top-level
                    # episode_* directory if it matches the zip's stem.
                    parts = internal_path.parts
                    if parts and parts[0] == zip_path.stem:
                        rel_parts = parts[1:]
                    else:
                        rel_parts = parts

                    if not rel_parts:
                        # Nothing meaningful to extract
                        continue

                    dest_rel = Path(*rel_parts)
                else:
                    # "images_zip" and "flat": put everything flat in out_dir
                    dest_rel = Path(internal_path.name)

                dest_path = out_dir / dest_rel
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                with zf.open(member, "r") as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())

        print(f"[ok]   {zip_path} -> {out_dir} (mode={mode})")
        if delete_after:
            zip_path.unlink(missing_ok=True)
    except BadZipFile:
        print(f"[bad]  {zip_path} is not a valid zip file")
    except Exception as e:
        print(f"[err]  Failed on {zip_path}: {e}")


def find_zip_files(root: Path):
    return [p for p in root.rglob("*.zip") if p.is_file()]


def _worker(args):
    """Helper for multiprocessing.imap_unordered."""
    zip_path, overwrite, delete_after = args
    return unzip_one(zip_path, overwrite=overwrite, delete_after=delete_after)


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
        help="Number of processes (default: CPU count)",
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
    procs = args.processes or mp.cpu_count()

    # Prepare argument tuples for workers
    tasks = [(zp, args.overwrite, args.delete_after) for zp in zips]

    with mp.Pool(processes=procs) as pool:
        for _ in tqdm(
            pool.imap_unordered(_worker, tasks),
            total=len(tasks),
            desc="Unzipping",
        ):
            # tqdm updates as each task completes
            pass


if __name__ == "__main__":
    main()

