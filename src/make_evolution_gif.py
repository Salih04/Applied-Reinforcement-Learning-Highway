from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio


def iter_frames(video_path: Path, fps: int, seconds: int | None):
    # Read all frames (imageio uses ffmpeg internally)
    frames = list(iio.imiter(video_path))
    if not frames:
        raise RuntimeError(f"No frames read from: {video_path}")

    # If seconds provided, take only first N seconds
    if seconds is not None:
        max_frames = max(1, int(fps * seconds))
        frames = frames[:max_frames]

    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seconds", type=int, default=12, help="per-stage seconds (None = full)")
    parser.add_argument("--in_dir", type=str, default="assets/videos")
    parser.add_argument("--out", type=str, default="assets/evolution.gif")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out)

    # Expected paths from your play.py recording
    p1 = in_dir / "untrained" / "rl-video-episode-0.mp4"
    p2 = in_dir / "half_trained" / "rl-video-episode-0.mp4"
    p3 = in_dir / "fully_trained" / "rl-video-episode-0.mp4"

    missing = [str(p) for p in (p1, p2, p3) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing video(s):\n" + "\n".join(missing))

    frames = []
    frames += iter_frames(p1, fps=args.fps, seconds=args.seconds)
    frames += iter_frames(p2, fps=args.fps, seconds=args.seconds)
    frames += iter_frames(p3, fps=args.fps, seconds=args.seconds)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write GIF
    iio.imwrite(out_path, frames, fps=args.fps)
    print(f"[OK] Saved: {out_path} (frames={len(frames)}, fps={args.fps})")


if __name__ == "__main__":
    main()

