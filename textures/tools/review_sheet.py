#!/usr/bin/env python3
"""Build blind-review sheets for exported texture sets.

Produces, per material, three 1024px PNGs a critic can actually look at plus a
JSON of measurements, so a verdict can cite numbers instead of impressions:

  <name>_look.png   basecolour downscaled whole — overall read
  <name>_crop.png   1024 crop at 100% — micro detail, is it mush at native res
  <name>_tile.png   4x4 repeat — seams and repetition
  <name>_maps.png   BC / N / H / ORM quad — channel sanity

Usage:
  ./tools/review_sheet.py                    every material in outputs/
  ./tools/review_sheet.py moraine_till ...   named materials only
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
PROBES = ROOT / "probes"
SHEETS = ROOT / "review"

SHEET = 1024
CROP = 1024
TILE_N = 4


def load(path):
    return Image.open(path)


def to_array(image):
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def seam_error(array):
    """Wrap discontinuity relative to the texture's own local contrast.

    A tileable texture is one where the step across the wrap edge is no larger
    than a step anywhere else. Comparing the wrap delta against the interior
    delta makes the number scale-free, so a busy gravel and a smooth snowpack
    are held to the same standard.
    """
    def axis_error(a):
        wrap = np.abs(a[0, :, :] - a[-1, :, :]).mean()
        interior = np.abs(a[1:, :, :] - a[:-1, :, :]).mean()
        if interior < 1e-6:
            return float("inf")
        return float(wrap / interior)

    return {
        "vertical": axis_error(array),
        "horizontal": axis_error(np.transpose(array, (1, 0, 2))),
    }


def measure(name, files):
    stats = {"name": name}

    basecolour = load(files["BC"])
    small = basecolour.resize((SHEET, SHEET), Image.LANCZOS)
    array = to_array(small)

    stats["albedo_mean"] = [round(float(c), 4) for c in array.reshape(-1, 3).mean(axis=0)]
    stats["albedo_std"] = [round(float(c), 4) for c in array.reshape(-1, 3).std(axis=0)]
    luma = array @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    stats["luma_p01"] = round(float(np.percentile(luma, 1)), 4)
    stats["luma_p99"] = round(float(np.percentile(luma, 99)), 4)
    stats["clipped_black_pct"] = round(float((luma < 0.02).mean() * 100), 3)
    stats["clipped_white_pct"] = round(float((luma > 0.98).mean() * 100), 3)
    stats["seam_ratio_bc"] = {k: round(v, 3) for k, v in seam_error(array).items()}

    if "H" in files:
        height = load(files["H"]).resize((SHEET, SHEET), Image.LANCZOS)
        h = np.asarray(height.convert("L"), dtype=np.float32) / 255.0
        stats["height_p01"] = round(float(np.percentile(h, 1)), 4)
        stats["height_p99"] = round(float(np.percentile(h, 99)), 4)
        stats["height_std"] = round(float(h.std()), 4)
        stats["seam_ratio_h"] = {
            k: round(v, 3) for k, v in seam_error(h[:, :, None]).items()
        }

    if "N" in files:
        normal = load(files["N"]).resize((SHEET, SHEET), Image.LANCZOS)
        n = to_array(normal)
        stats["normal_mean"] = [round(float(c), 4) for c in n.reshape(-1, 3).mean(axis=0)]

    return stats, small


def quad(images, size=512):
    sheet = Image.new("RGB", (size * 2, size * 2), (24, 24, 24))
    for index, image in enumerate(images[:4]):
        tile = image.convert("RGB").resize((size, size), Image.LANCZOS)
        sheet.paste(tile, ((index % 2) * size, (index // 2) * size))
    return sheet


def build(name):
    export = OUTPUTS / f"T_{name}.json"
    if not export.exists():
        return None

    manifest = json.loads(export.read_text())
    files = {}
    for path in manifest["files"]:
        suffix = Path(path).stem.rsplit("_", 1)[-1]
        files[suffix] = Path(path)

    stats, small = measure(name, files)
    SHEETS.mkdir(exist_ok=True)

    small.save(SHEETS / f"{name}_look.png")

    basecolour = load(files["BC"])
    width, height = basecolour.size
    left = (width - CROP) // 3
    top = (height - CROP) // 3
    basecolour.crop((left, top, left + CROP, top + CROP)).save(SHEETS / f"{name}_crop.png")

    cell = SHEET // TILE_N
    unit = small.resize((cell, cell), Image.LANCZOS)
    tiled = Image.new("RGB", (SHEET, SHEET))
    for row in range(TILE_N):
        for column in range(TILE_N):
            tiled.paste(unit, (column * cell, row * cell))
    tiled.save(SHEETS / f"{name}_tile.png")

    order = [files[k] for k in ("BC", "N", "H", "ORM") if k in files]
    quad([load(p) for p in order]).save(SHEETS / f"{name}_maps.png")

    return stats


def build_probe(name):
    """Probe review: albedo only, so no channel quad and no height stats."""
    source = PROBES / f"{name}.png"
    if not source.exists():
        return None

    SHEETS.mkdir(exist_ok=True)
    albedo = Image.open(source).convert("RGB")
    small = albedo.resize((SHEET, SHEET), Image.LANCZOS)
    array = to_array(small)

    stats = {"name": name, "kind": "probe", "source_size": list(albedo.size)}
    stats["albedo_mean"] = [round(float(c), 4) for c in array.reshape(-1, 3).mean(axis=0)]
    stats["albedo_std"] = [round(float(c), 4) for c in array.reshape(-1, 3).std(axis=0)]
    luma = array @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    stats["luma_p01"] = round(float(np.percentile(luma, 1)), 4)
    stats["luma_p99"] = round(float(np.percentile(luma, 99)), 4)
    stats["clipped_black_pct"] = round(float((luma < 0.02).mean() * 100), 3)
    stats["clipped_white_pct"] = round(float((luma > 0.98).mean() * 100), 3)
    stats["seam_ratio_bc"] = {k: round(v, 3) for k, v in seam_error(array).items()}

    small.save(SHEETS / f"{name}_look.png")

    cell = SHEET // TILE_N
    unit = small.resize((cell, cell), Image.LANCZOS)
    tiled = Image.new("RGB", (SHEET, SHEET))
    for row in range(TILE_N):
        for column in range(TILE_N):
            tiled.paste(unit, (column * cell, row * cell))
    tiled.save(SHEETS / f"{name}_tile.png")

    return stats


def main():
    argv = sys.argv[1:]
    probe_mode = "--probes" in argv
    names = [a for a in argv if not a.startswith("--")]
    if probe_mode:
        if not names:
            names = sorted(p.stem for p in PROBES.glob("*.png") if not p.stem.startswith("_"))
        report = []
        for name in names:
            stats = build_probe(name)
            if stats is None:
                print(f"  {name}: no probe image, skipped")
                continue
            report.append(stats)
            seam = stats["seam_ratio_bc"]
            print(
                f"  {name:24} seam v={seam['vertical']:.2f} h={seam['horizontal']:.2f}  "
                f"luma {stats['luma_p01']:.2f}-{stats['luma_p99']:.2f}  "
                f"clip {stats['clipped_black_pct']:.1f}/{stats['clipped_white_pct']:.1f}%"
            )
        SHEETS.mkdir(exist_ok=True)
        (SHEETS / "probe_measurements.json").write_text(json.dumps(report, indent=2))
        print(f"\n{len(report)} probe sheets in {SHEETS}")
        return

    if not names:
        names = sorted(p.stem[2:] for p in OUTPUTS.glob("T_*.json"))

    report = []
    for name in names:
        stats = build(name)
        if stats is None:
            print(f"  {name}: no export, skipped")
            continue
        report.append(stats)
        seam = stats["seam_ratio_bc"]
        print(
            f"  {name:24} seam v={seam['vertical']:.2f} h={seam['horizontal']:.2f}  "
            f"luma {stats['luma_p01']:.2f}-{stats['luma_p99']:.2f}  "
            f"clip {stats['clipped_black_pct']:.1f}/{stats['clipped_white_pct']:.1f}%"
        )

    SHEETS.mkdir(exist_ok=True)
    (SHEETS / "measurements.json").write_text(json.dumps(report, indent=2))
    print(f"\n{len(report)} sheets in {SHEETS}")


if __name__ == "__main__":
    main()
