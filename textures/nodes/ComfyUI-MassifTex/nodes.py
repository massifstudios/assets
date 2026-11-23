"""MassifTex — tileable PBR texture nodes for the Massif texture pipeline.

Fills the gaps between CHORD's 1024px material estimation and a game-ready
8K PBR set: seamless-aware upscaling, high-frequency normal recovery, an AO
channel CHORD does not predict, and an engine-convention export.
"""

import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing_extensions import override

import folder_paths
from comfy_api.latest import ComfyExtension, io, ui

from . import massif_ops as ops

CATEGORY = "Massif/PBR"


def _resolve_dir(output_dir):
    output_dir = os.path.expanduser(output_dir.strip())
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(folder_paths.get_output_directory(), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_8bit(t_hw_c, path, mode):
    arr = (t_hw_c.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    if mode == "L":
        arr = arr[..., 0]
    Image.fromarray(arr, mode=mode).save(path, optimize=False, compress_level=6)


def _save_16bit_gray(t_hw_1, path):
    arr = (t_hw_1[..., 0].clamp(0, 1).cpu().numpy() * 65535.0 + 0.5).astype(np.uint16)
    Image.fromarray(arr, mode="I;16").save(path)


def _label_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def _contact_sheet(tiles, cell, columns=3):
    """Lay out (label, [1, C, H, W]) pairs into one labelled RGB preview image.

    Returned as a ComfyUI IMAGE tensor so the node can hand it straight to
    ui.PreviewImage — the maps are then visible on the canvas without five
    separate full-resolution PreviewImage nodes writing 8K temp files.
    """
    bar = max(14, cell // 16)
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + bar)), (24, 24, 27))
    draw = ImageDraw.Draw(sheet)
    font = _label_font(max(10, bar - 4))

    for i, (label, tensor) in enumerate(tiles):
        x = (i % columns) * cell
        y = (i // columns) * (cell + bar)

        t = tensor[:1]
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
        t = ops.seamless_resize(t.float(), cell, cell).clamp(0, 1)
        arr = (t[0].permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)

        sheet.paste(Image.fromarray(arr, mode="RGB"), (x, y + bar))
        draw.text((x + 4, y + 2), label, fill=(220, 220, 225), font=font)

    out = torch.from_numpy(np.asarray(sheet).astype(np.float32) / 255.0)
    return out.unsqueeze(0)


class MassifSeamlessUpscale(io.ComfyNode):
    """Upscale a tileable map to a target resolution without breaking the tiling."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MassifSeamlessUpscale",
            display_name="Massif - Seamless Upscale",
            category=CATEGORY,
            description=(
                "Runs an upscale model over circularly padded input so tile edges "
                "see their wrapped neighbours, then resizes to the exact target."
            ),
            inputs=[
                io.Image.Input("image"),
                io.UpscaleModel.Input(
                    "upscale_model",
                    optional=True,
                    tooltip="Omit to resize with bicubic only — right for low-frequency maps "
                            "(roughness, metalness) where an ESRGAN model invents detail.",
                ),
                io.Int.Input("target_size", default=8192, min=64, max=16384, step=64),
                io.Int.Input(
                    "pad", default=64, min=0, max=512, step=16,
                    tooltip="Wrap-around border fed to the model and cropped off after.",
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, target_size, pad, upscale_model=None) -> io.NodeOutput:
        x = ops.to_bchw(image).float()
        h, w = x.shape[-2], x.shape[-1]
        long_edge = max(h, w)

        if upscale_model is not None:
            scale = float(upscale_model.scale)
            for _ in range(ops.passes_needed(long_edge, target_size, scale)):
                x = ops.upscale_with_model_seamless(x, upscale_model, pad)
                long_edge = max(x.shape[-2], x.shape[-1])

        ratio = target_size / max(x.shape[-2], x.shape[-1])
        out_h = int(round(x.shape[-2] * ratio))
        out_w = int(round(x.shape[-1] * ratio))
        x = ops.seamless_resize(x, out_h, out_w)
        return io.NodeOutput(ops.to_bhwc(x.clamp(0, 1)).cpu())


class MassifNormalDetailBlend(io.ComfyNode):
    """Graft high-frequency detail from a sharp map onto a low-frequency normal."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MassifNormalDetailBlend",
            display_name="Massif - Normal Detail Blend",
            category=CATEGORY,
            description=(
                "CHORD estimates normals at 1024, so an 8K normal is a smooth upscale. "
                "This derives micro-detail from the full-resolution basecolor and blends "
                "it into the base normal (UDN), restoring pore-level relief."
            ),
            inputs=[
                io.Image.Input("normal", tooltip="Base normal map, already at output resolution."),
                io.Image.Input("detail_source", tooltip="Full-resolution basecolor or height."),
                io.Float.Input("strength", default=0.5, min=0.0, max=4.0, step=0.05),
                io.Int.Input(
                    "highpass_radius", default=8, min=1, max=128, step=1,
                    tooltip="Larger radius keeps more mid-frequency detail.",
                ),
                io.Boolean.Input(
                    "invert_detail", default=False,
                    tooltip="Enable when bright pixels in the source mean recesses, not bumps.",
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, normal, detail_source, strength, highpass_radius, invert_detail) -> io.NodeOutput:
        base = ops.to_bchw(normal).float()
        src = ops.to_bchw(detail_source).float()

        if src.shape[-2:] != base.shape[-2:]:
            src = ops.seamless_resize(src, base.shape[-2], base.shape[-1])

        lum = ops.luminance(src)
        highpass = lum - ops.circular_blur(lum, highpass_radius)
        if invert_detail:
            highpass = -highpass

        detail = ops.normal_from_height(highpass, strength=1.0)
        out = ops.udn_blend(base, detail, strength)
        return io.NodeOutput(ops.to_bhwc(out).cpu())


class MassifAOFromHeight(io.ComfyNode):
    """Derive an ambient occlusion map from a height map."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MassifAOFromHeight",
            display_name="Massif - AO from Height",
            category=CATEGORY,
            description=(
                "CHORD predicts basecolor/normal/roughness/metalness but no AO. "
                "This approximates it as multi-scale cavity: how far each texel sits "
                "below its blurred neighbourhood."
            ),
            inputs=[
                io.Image.Input("height"),
                io.Float.Input("strength", default=0.75, min=0.0, max=2.0, step=0.05),
                io.Int.Input(
                    "radius", default=16, min=2, max=256, step=2,
                    tooltip="Largest cavity scale sampled, in pixels at 1024px. Two finer "
                            "octaves are added below it.",
                ),
                io.Boolean.Input(
                    "scale_with_resolution", default=True,
                    tooltip="Scale the radius to the actual map size. Off means raw pixels, "
                            "which at 8K samples too small a neighbourhood to register cavities.",
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, height, strength, radius, scale_with_resolution) -> io.NodeOutput:
        h = ops.luminance(ops.to_bchw(height).float())
        if scale_with_resolution:
            radius = max(2, int(round(radius * max(h.shape[-2], h.shape[-1]) / 1024.0)))
        scales = (max(2, radius // 8), max(3, radius // 2), radius)
        ao = ops.ao_from_height(h, strength=strength, scales=scales)
        return io.NodeOutput(ops.to_bhwc(ao.repeat(1, 3, 1, 1)).cpu())


class MassifTilePreview(io.ComfyNode):
    """Tile a texture N×N to eye-check for seams and repetition."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MassifTilePreview",
            display_name="Massif - Tile Preview",
            category=CATEGORY,
            description="Repeats the texture in a grid so seams and obvious repeats are visible at a glance.",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("tiles", default=2, min=2, max=8, step=1),
                io.Int.Input("max_size", default=1536, min=256, max=8192, step=64,
                             tooltip="The grid is downscaled to this long edge so previews stay light."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, tiles, max_size) -> io.NodeOutput:
        x = ops.to_bchw(image).float()
        per_tile = max(64, max_size // tiles)
        x = ops.seamless_resize(x, per_tile, per_tile)
        grid = x.repeat(1, 1, tiles, tiles)
        return io.NodeOutput(ops.to_bhwc(grid.clamp(0, 1)).cpu())


class MassifPBRExport(io.ComfyNode):
    """Write an engine-ready PBR set with the right conventions and naming."""

    ENGINES = ["unreal", "unity", "godot", "raw"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MassifPBRExport",
            display_name="Massif - PBR Export",
            category=CATEGORY,
            description=(
                "Applies per-engine conventions (normal handedness, channel packing) and "
                "writes the set to disk. Height is 16-bit; the rest are 8-bit."
            ),
            inputs=[
                io.Image.Input("basecolor"),
                io.Image.Input("normal"),
                io.Image.Input("roughness"),
                io.Image.Input("metalness"),
                io.Image.Input("height"),
                io.Image.Input("ao", optional=True,
                               tooltip="Omitted means fully unoccluded (white)."),
                io.String.Input("name", default="massif_rock", placeholder="material name"),
                io.String.Input("output_dir", default="massif",
                                tooltip="Absolute, or relative to the ComfyUI output directory."),
                io.Combo.Input("engine", options=cls.ENGINES, default="unreal"),
                io.Boolean.Input("write_16bit_height", default=True),
                io.Int.Input(
                    "preview_size", default=512, min=0, max=2048, step=64,
                    tooltip="Per-map thumbnail size in the on-canvas contact sheet. 0 disables it.",
                ),
            ],
            outputs=[io.String.Output(display_name="manifest_json")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, basecolor, normal, roughness, metalness, height,
                name, output_dir, engine, write_16bit_height, preview_size, ao=None) -> io.NodeOutput:
        out_dir = _resolve_dir(output_dir)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip()) or "material"

        bc = ops.to_bchw(basecolor).float()
        res_h, res_w = bc.shape[-2], bc.shape[-1]

        def conform(img, channels):
            x = ops.to_bchw(img).float()
            if x.shape[-2:] != (res_h, res_w):
                x = ops.seamless_resize(x, res_h, res_w)
            if channels == 1:
                x = ops.luminance(x)
            elif x.shape[1] < 3:
                x = x.repeat(1, 3, 1, 1)
            return x.clamp(0, 1)

        nrm = conform(normal, 3)
        rough = conform(roughness, 1)
        metal = conform(metalness, 1)
        hgt = conform(height, 1)
        occ = conform(ao, 1) if ao is not None else torch.ones_like(rough)

        # Unreal and Unity sample DirectX-handed normals (green points down);
        # OpenGL-handed engines and raw output keep the map as CHORD produced it.
        if engine in ("unreal", "unity"):
            nrm = torch.cat([nrm[:, 0:1], 1.0 - nrm[:, 1:2], nrm[:, 2:3]], dim=1)

        written = []
        batch = bc.shape[0]
        for i in range(batch):
            suffix = "" if batch == 1 else f"_{i:03d}"
            stem = f"T_{safe_name}{suffix}"

            def emit(tag, tensor, mode, bits=8):
                path = os.path.join(out_dir, f"{stem}_{tag}.png")
                hwc = ops.to_bhwc(tensor[i:i + 1])[0]
                if bits == 16:
                    _save_16bit_gray(hwc, path)
                else:
                    _save_8bit(hwc, path, mode)
                written.append(path)

            emit("BC", bc[:, :3], "RGB")
            emit("N", nrm, "RGB")
            emit("H", hgt, "L", bits=16 if write_16bit_height else 8)

            if engine in ("unreal", "godot"):
                # ORM: R=occlusion, G=roughness, B=metallic — one sampler, three masks.
                orm = torch.cat([occ, rough, metal], dim=1)
                emit("ORM", orm, "RGB")
            elif engine == "unity":
                # Unity's Metallic map is metallic in R with smoothness (1-roughness) in A.
                smoothness = 1.0 - rough
                mask = torch.cat([metal, torch.zeros_like(metal), torch.zeros_like(metal), smoothness], dim=1)
                path = os.path.join(out_dir, f"{stem}_MetallicSmoothness.png")
                _save_8bit(ops.to_bhwc(mask[i:i + 1])[0], path, "RGBA")
                written.append(path)
                emit("AO", occ, "L")
            else:
                emit("R", rough, "L")
                emit("M", metal, "L")
                emit("AO", occ, "L")

        manifest = {
            "name": safe_name,
            "engine": engine,
            "resolution": [res_w, res_h],
            "normal_handedness": "DirectX" if engine in ("unreal", "unity") else "OpenGL",
            "files": written,
        }
        manifest_path = os.path.join(out_dir, f"T_{safe_name}.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        manifest_str = json.dumps(manifest, indent=2)
        if preview_size <= 0:
            return io.NodeOutput(manifest_str)

        sheet = _contact_sheet(
            [
                ("basecolor", bc[:, :3]),
                (f"normal ({manifest['normal_handedness']})", nrm),
                ("height", hgt),
                ("roughness", rough),
                ("metalness", metal),
                ("ambient occlusion", occ),
            ],
            cell=preview_size,
        )
        return io.NodeOutput(manifest_str, ui=ui.PreviewImage(sheet, cls=cls))


class MassifTexExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MassifSeamlessUpscale,
            MassifNormalDetailBlend,
            MassifAOFromHeight,
            MassifTilePreview,
            MassifPBRExport,
        ]


async def comfy_entrypoint() -> MassifTexExtension:
    return MassifTexExtension()
