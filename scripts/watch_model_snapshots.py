#!/usr/bin/env python3
"""
Watch a Stable-Baselines3 save path (e.g. models/ppo_line_follow → models/ppo_line_follow.zip).
Whenever the zip changes, copy it to a numbered snapshot: 1.zip, 2.zip, ... in --dest-dir.

Training can keep using the same --save path; this script only copies, never moves the main file.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def next_index(dest_dir: Path) -> int:
    best = 0
    for p in dest_dir.glob("*.zip"):
        if p.stem.isdigit():
            best = max(best, int(p.stem))
    return best + 1


def stat_sig(path: Path) -> tuple[float, int] | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    st = path.stat()
    return (st.st_mtime, st.st_size)


def wait_stable(path: Path, stability_s: float, poll_s: float) -> bool:
    """Return True if path exists, non-empty, and (mtime, size) unchanged for stability_s."""
    sig = stat_sig(path)
    if sig is None:
        return False
    deadline = time.monotonic() + stability_s
    while time.monotonic() < deadline:
        time.sleep(poll_s)
        cur = stat_sig(path)
        if cur != sig:
            return False
    return stat_sig(path) == sig


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot SB3 model zip on each change as 1.zip, 2.zip, ...")
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="models/ppo_line_follow",
        help="Same as train.py --save (without .zip); watched file is <prefix>.zip",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="models/ppo_line_follow_snapshots",
        help="Directory for numbered copies 1.zip, 2.zip, ...",
    )
    parser.add_argument("--poll", type=float, default=1.0, help="Seconds between checks")
    parser.add_argument(
        "--stability",
        type=float,
        default=0.75,
        help="Seconds (mtime,size) must stay fixed before copying (avoids partial writes)",
    )
    parser.add_argument(
        "--stability-poll",
        type=float,
        default=0.1,
        help="Poll interval while verifying stability",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    prefix = Path(args.save_prefix)
    zip_path = root / prefix if not prefix.is_absolute() else prefix
    if zip_path.suffix.lower() != ".zip":
        zip_path = zip_path.with_suffix(".zip")

    dest_dir = root / Path(args.dest_dir) if not Path(args.dest_dir).is_absolute() else Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    last: tuple[float, int] | None = None
    print(f"Watching: {zip_path}")
    print(f"Snapshots: {dest_dir} as 1.zip, 2.zip, ...")
    print("Ctrl+C to stop.")

    try:
        while True:
            sig = stat_sig(zip_path)
            if sig is not None and sig != last:
                if wait_stable(zip_path, args.stability, args.stability_poll):
                    n = next_index(dest_dir)
                    out = dest_dir / f"{n}.zip"
                    shutil.copy2(zip_path, out)
                    print(f"Saved snapshot #{n} -> {out}")
                    last = stat_sig(zip_path)
                # else: still writing; will retry on next poll
            elif sig is None:
                last = None

            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
