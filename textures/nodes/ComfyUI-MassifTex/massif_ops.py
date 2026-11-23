"""Tensor helpers for tileable PBR texture processing.

Every spatial operation here uses circular padding so that a texture which
tiles seamlessly on input still tiles seamlessly on output.

Image tensors follow the ComfyUI convention: float32, [B, H, W, C], range 0..1.
Internally everything is converted to [B, C, H, W] for torch ops.
"""

import math

import torch
import torch.nn.functional as F


def to_bchw(image):
    """[B, H, W, C] -> [B, C, H, W].

    Also accepts the channel-less [B, H, W] that some nodes emit for
    single-channel maps despite declaring an IMAGE output (ChordNormalToHeight
    does this for height), promoting it to a 1-channel image.
    """
    if image.ndim == 2:
        return image[None, None]
    if image.ndim == 3:
        return image.unsqueeze(1)
    return image.movedim(-1, -3)


def to_bhwc(x):
    return x.movedim(-3, -1)


def circular_pad(x, pad):
    """Circularly pad [B, C, H, W] by `pad` px on all sides.

    Index wrapping rather than F.pad(mode="circular"), which refuses a pad
    larger than the dimension it wraps — a blur radius can legitimately exceed
    a small map's size.
    """
    if pad <= 0:
        return x
    h, w = x.shape[-2], x.shape[-1]
    iy = torch.arange(-pad, h + pad, device=x.device) % h
    ix = torch.arange(-pad, w + pad, device=x.device) % w
    return x[..., iy, :][..., :, ix]


def seamless_resize(x, height, width, pad=16):
    """Resize [B, C, H, W] preserving tileability."""
    if x.shape[-2] == height and x.shape[-1] == width:
        return x
    p = max(1, min(pad, x.shape[-1] // 4, x.shape[-2] // 4))
    padded = circular_pad(x, p)
    pad_h = int(round(p * height / x.shape[-2]))
    pad_w = int(round(p * width / x.shape[-1]))
    out = F.interpolate(
        padded,
        size=(height + 2 * pad_h, width + 2 * pad_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return out[:, :, pad_h:pad_h + height, pad_w:pad_w + width]


def _gaussian_kernel1d(radius, device, dtype):
    sigma = max(radius / 2.0, 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def circular_blur(x, radius):
    """Separable gaussian blur with circular padding. x is [B, C, H, W]."""
    radius = int(radius)
    if radius < 1:
        return x
    k = _gaussian_kernel1d(radius, x.device, x.dtype)
    c = x.shape[1]
    padded = circular_pad(x, radius)
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    out = F.conv2d(padded, kx, groups=c)
    out = F.conv2d(out, ky, groups=c)
    return out


def luminance(x):
    """Rec. 709 luma of [B, C, H, W] -> [B, 1, H, W]."""
    if x.shape[1] == 1:
        return x
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device, dtype=x.dtype)
    return (x[:, :3] * w.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def sobel_gradients(h):
    """Circular Sobel gradients of a [B, 1, H, W] height field."""
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=h.device, dtype=h.dtype,
    ).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    padded = circular_pad(h, 1)
    return F.conv2d(padded, kx), F.conv2d(padded, ky)


def normal_from_height(h, strength=1.0, opengl=True):
    """Height field [B, 1, H, W] -> tangent-space normal in 0..1 [B, 3, H, W]."""
    dx, dy = sobel_gradients(h)
    scale = strength * h.shape[-1] / 256.0
    nx = -dx * scale
    ny = -dy * scale if opengl else dy * scale
    nz = torch.ones_like(nx)
    n = torch.cat([nx, ny, nz], dim=1)
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return n * 0.5 + 0.5


def decode_normal(n01):
    """0..1 encoded normal [B, 3, H, W] -> unit vectors in -1..1."""
    n = n01[:, :3] * 2.0 - 1.0
    return n / n.norm(dim=1, keepdim=True).clamp_min(1e-6)


def encode_normal(n):
    return (n / n.norm(dim=1, keepdim=True).clamp_min(1e-6)) * 0.5 + 0.5


def udn_blend(base01, detail01, strength=1.0):
    """UDN normal blend: keep the base Z, sum the tangent-space XY.

    Cheap, stable, and the standard way to graft high-frequency detail onto a
    low-frequency base normal without the swimming that a naive lerp causes.
    """
    b = decode_normal(base01)
    d = decode_normal(detail01)
    xy = b[:, :2] + d[:, :2] * strength
    n = torch.cat([xy, b[:, 2:3]], dim=1)
    return encode_normal(n)


def ao_from_height(h, strength=1.0, scales=(4, 16, 64)):
    """Multi-scale cavity approximation of ambient occlusion.

    A texel is occluded to the degree it sits below its blurred neighbourhood,
    summed over several radii so both pores and large recesses register.
    """
    total = torch.zeros_like(h)
    weight = 0.0
    for i, r in enumerate(scales):
        w = 1.0 / (i + 1)
        total = total + w * (circular_blur(h, r) - h).clamp_min(0.0)
        weight += w
    occ = total / max(weight, 1e-6)
    span = occ.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    return (1.0 - strength * (occ / span)).clamp(0.0, 1.0)


def upscale_with_model_seamless(image_bchw, upscale_model, pad, tile=512, overlap=32):
    """One pass of a spandrel upscale model with wrap-around borders.

    Mirrors comfy_extras.nodes_upscale_model.ImageUpscaleWithModel, but pads the
    input circularly first and crops the padding back off afterwards, so tile
    edges see their true wrapped neighbours instead of a clamped border.
    """
    import comfy.model_management
    import comfy.utils

    device = upscale_model.patcher.load_device
    scale = upscale_model.scale

    padded = circular_pad(image_bchw, pad)
    applied_pad_w = (padded.shape[-1] - image_bchw.shape[-1]) // 2
    applied_pad_h = (padded.shape[-2] - image_bchw.shape[-2]) // 2
    padded = padded.to(device)

    memory_required = (512 * 512 * 3) * padded.element_size() * max(scale, 1.0) * 384.0
    memory_required += padded.nelement() * padded.element_size()
    comfy.model_management.load_models_gpu(
        [upscale_model.patcher], memory_required=memory_required, force_full_load=True
    )

    output_device = comfy.model_management.intermediate_device()
    while True:
        try:
            steps = padded.shape[0] * comfy.utils.get_tiled_scale_steps(
                padded.shape[3], padded.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
            )
            pbar = comfy.utils.ProgressBar(steps)
            out = comfy.utils.tiled_scale(
                padded,
                lambda a: upscale_model(a.float()),
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
                upscale_amount=scale,
                pbar=pbar,
                output_device=output_device,
            )
            break
        except Exception as e:
            comfy.model_management.raise_non_oom(e)
            tile //= 2
            if tile < 128:
                raise

    crop_w = int(round(applied_pad_w * scale))
    crop_h = int(round(applied_pad_h * scale))
    if crop_h > 0:
        out = out[:, :, crop_h:out.shape[2] - crop_h, :]
    if crop_w > 0:
        out = out[:, :, :, crop_w:out.shape[3] - crop_w]
    return out.clamp(0.0, 1.0)


def passes_needed(current, target, scale):
    """How many `scale`x model passes to reach at least `target` from `current`."""
    if scale <= 1.0 or current >= target:
        return 0
    return int(math.ceil(math.log(target / current) / math.log(scale)))
