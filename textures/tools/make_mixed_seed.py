#!/usr/bin/env python3
"""Blend two seed photos into one composite img2img base.

Some materials are a mixture of two surfaces — stone with vegetation growing in
the gaps — and a seed showing only one of them never grows the other, at any
denoise. Blending the two scans under a low-frequency mask puts both in the
latent from the start.

The mask is deliberately blobby: an even blend reads as a single hazed surface
rather than as patches of one thing sitting in another.

Usage:
  ./tools/make_mixed_seed.py base.jpg overlay.jpg out.jpg --coverage 0.33 --seed 3763
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
PHOTOS = ROOT / "assets" / "photo"


def resolve(name):
    path = Path(name)
    if path.exists():
        return path
    return PHOTOS / name


def blend(base_path, overlay_path, coverage, seed, blob_scale):
    base = Image.open(resolve(base_path)).convert("RGB")
    overlay = Image.open(resolve(overlay_path)).convert("RGB").resize(base.size)
    width, height = base.size

    rng = np.random.default_rng(seed)
    noise = Image.fromarray((rng.random((height // 16, width // 16)) * 255).astype("uint8"))
    mask = noise.resize((width, height), Image.BICUBIC)
    mask = mask.filter(ImageFilter.GaussianBlur(width // blob_scale))

    m = np.asarray(mask, np.float32)
    m = (m - m.min()) / (m.max() - m.min())
    # Solve the threshold for the requested coverage rather than guessing it.
    cut = float(np.quantile(m, 1 - coverage))
    m = np.clip((m - cut) * 6 + 0.5, 0, 1)

    mixed = (
        np.asarray(base, np.float32) * (1 - m[..., None])
        + np.asarray(overlay, np.float32) * m[..., None]
    )
    return Image.fromarray(mixed.astype("uint8")), float(m.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("overlay")
    parser.add_argument("out")
    parser.add_argument("--coverage", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blob-scale", type=int, default=40,
                        help="larger means finer patches")
    a = parser.parse_args()

    image, coverage = blend(a.base, a.overlay, a.coverage, a.seed, a.blob_scale)
    out = Path(a.out)
    if not out.is_absolute() and out.parent == Path("."):
        out = PHOTOS / out.name
    image.save(out, quality=95)
    print(f"{out}  overlay coverage {coverage * 100:.1f}%")


if __name__ == "__main__":
    main()
