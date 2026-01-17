from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips, ColorClip, ImageClip
from PIL import Image, ImageDraw, ImageFont


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_font(size: int = 22) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Prefer modern macOS fonts first
    candidates = [
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux fallback
    ]
    for p in candidates:
        fp = Path(p)
        if fp.exists():
            return ImageFont.truetype(str(fp), size=size)
    return ImageFont.load_default()


def make_header_bar(width: int, text: str, bar_height: int = 34) -> np.ndarray:
    img = Image.new("RGB", (width, bar_height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    font = _load_font(size=22)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (bar_height - th) // 2

    # subtle shadow + clean white text
    draw.text((x + 1, y + 1), text, font=font, fill=(60, 60, 60))
    draw.text((x, y), text, font=font, fill=(245, 245, 245))

    return np.array(img)



def add_header(clip: VideoFileClip, title: str) -> VideoFileClip:
    w, h = clip.size
    bar_h = 34

    header_img = make_header_bar(w, title, bar_height=bar_h)
    header_clip = ImageClip(header_img).with_duration(clip.duration).with_position(("center", "top"))

    # shrink video slightly to make room for header
    video = clip.resized((w, h - bar_h)).with_position(("center", bar_h))

    from moviepy import CompositeVideoClip
    base = ColorClip((w, h), color=(0, 0, 0)).with_duration(clip.duration)

    return CompositeVideoClip([base, video, header_clip], size=(w, h)).with_duration(clip.duration)


def main() -> None:
    root = repo_root()

    # expected paths from your play.py recording stages
    paths = [
        root / "assets" / "videos" / "untrained" / "rl-video-episode-0.mp4",
        root / "assets" / "videos" / "half_trained" / "rl-video-episode-0.mp4",
        root / "assets" / "videos" / "fully_trained" / "rl-video-episode-0.mp4",
    ]
    titles = [
        "Untrained agent",
        "Half-trained agent",
        "Fully trained agent",
    ]

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing video: {p}")

    # tweak these to taste
    seconds_each = 8          # make GIF longer and understandable
    separator_seconds = 1.0    # black screen between stages
    fps = 30                   # github-friendly
    mp4_fps = 40
    gif_fps = 30
    gif_out = root / "assets" / "evolution.gif"
    mp4_out = root / "assets" / "evolution.mp4"

    clips = []
    for p, t in zip(paths, titles):
        c = VideoFileClip(str(p))
        c = c.subclipped(0, min(seconds_each, c.duration))
        c = add_header(c, t)
        clips.append(c)

    # black separators (same size as videos)
    w, h = clips[0].size
    sep = ColorClip((w, h), color=(0, 0, 0)).with_duration(separator_seconds)

    final = concatenate_videoclips(
        [clips[0], sep, clips[1], sep, clips[2]],
        method="compose",
    )

    gif_out.parent.mkdir(parents=True, exist_ok=True)

    final.write_videofile(str(mp4_out), fps=mp4_fps, codec="libx264", audio=False, preset="medium")
    final.write_gif(str(gif_out), fps=gif_fps)


    final.close()
    for c in clips:
        c.close()

    print(f"[OK] Wrote: {gif_out}")
    print(f"[OK] Wrote: {mp4_out}")


if __name__ == "__main__":
    main()
