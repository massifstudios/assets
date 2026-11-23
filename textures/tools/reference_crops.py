#!/usr/bin/env python3
"""Cut ground-truth surface patches out of the reference photographs.

A critic judging a tileable material needs to see the real surface at surface
scale, not a landscape containing it. Each entry names the material, the source
photo, and the region of that photo where the surface actually is.

Crops are upscaled to 512 so a small patch is still legible.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REFERENCES = ROOT / "references"
CROPS = REFERENCES / "crops"

PATCH = 512

# material -> (source photo, left, top, width, height) in source pixels
REGIONS = {
    "alpine_turf_hummock": ("ref_img2_stream_meadow.png", 500, 620, 200, 110),
    "mossy_bedrock_slab": ("ref_img2_stream_meadow.png", 210, 420, 180, 80),
    "stream_gravel_wet": ("ref_img2_stream_meadow.png", 450, 500, 190, 100),
    "moraine_till": ("ref_img3_moraine.png", 380, 505, 320, 92),
    "moraine_till_grassy": ("ref_img3_moraine.png", 60, 470, 320, 110),
    "limestone_cliff_face": ("ref_img4_limestone.png", 330, 150, 380, 220),
    "scree_limestone": ("ref_img4_limestone.png", 270, 375, 280, 95),
    "oxidized_rusty_schist": ("ref_img5_oetztal_rusty.jpeg", 1450, 950, 520, 420),
    "scree_fine_fan": ("ref_img5_oetztal_rusty.jpeg", 700, 1150, 350, 250),
    "alpine_path_gravel": ("ref_img7_autumn_path.png", 280, 418, 180, 57),
}


def main():
    CROPS.mkdir(exist_ok=True)
    for material, (source, left, top, width, height) in REGIONS.items():
        image = Image.open(REFERENCES / source).convert("RGB")
        patch = image.crop((left, top, left + width, top + height))
        scale = PATCH / max(patch.size)
        resized = patch.resize(
            (round(patch.width * scale), round(patch.height * scale)), Image.LANCZOS
        )
        resized.save(CROPS / f"{material}_reference.png")
        print(f"  {material:24} {source} {patch.size} -> {resized.size}")


if __name__ == "__main__":
    main()
