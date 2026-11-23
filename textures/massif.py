#!/usr/bin/env python3
"""Build and run Massif PBR texture workflows.

Textures are generated once, so the normal path is the batch: list the materials
in a YAML file and run the lot. Saved workflows are opt-in (--install), for the
one graph you want in the sidebar to iterate on interactively.

    ./massif.py batch materials.yaml            # the usual path — build+run each in turn
    ./massif.py batch materials.yaml --only granite,slate
    ./massif.py batch materials.yaml --dry-run

    ./massif.py new --name slate --prompt "..." --run     # one-off
    ./massif.py new --name slate --prompt "..." --install # put it in the ComfyUI sidebar
    ./massif.py list

`new` compiles blueprints/massif_pbr.recipe.json into a runnable graph and gives
the nodes readable titles (the ops layer has no title op, so it is stamped here).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import yaml

PROJECT = os.path.dirname(os.path.abspath(__file__))
RECIPE = os.path.join(PROJECT, "blueprints", "massif_pbr.recipe.json")
BLUEPRINTS = os.path.join(PROJECT, "blueprints")
OUTPUTS = os.path.join(PROJECT, "outputs")
COMFY_WORKFLOWS = os.path.expanduser("~/comfy/ComfyUI/user/default/workflows")

EMPTY_GRAPH = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}

# alias (from the recipe) -> canvas title
TITLES = {
    "ckpt": "Base Model",
    "tile_model": "Seamless: model",
    "tile_vae": "Seamless: VAE",
    "pos": "PROMPT  <- edit this",
    "neg": "Negative prompt",
    "latent": "Base resolution",
    "base": "Sample: base",
    "latent_up": "Upscale latent -> refine res",
    "refine": "Sample: hires refine",
    "decode": "Decode texture",
    "chord": "CHORD model",
    "mat": "CHORD: estimate material",
    "n2h": "CHORD: normal -> height",
    "up_model": "Upscaler",
    "bc_8k": "8K: basecolor",
    "n_8k": "8K: normal",
    "r_8k": "8K: roughness",
    "m_8k": "8K: metalness",
    "h_8k": "8K: height",
    "n_detail": "Recover normal micro-detail",
    "ao": "Derive AO",
    "export": "EXPORT  (contact sheet here)",
    "preview": "Tile 3x3",
    "save_preview": "Seam check",
}


def run_comfy(args, capture=True):
    proc = subprocess.run(
        ["comfy", "--json", *args], capture_output=capture, text=True, cwd=PROJECT
    )
    if not capture:
        return None
    try:
        env = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit(f"comfy returned non-JSON:\n{proc.stdout[:500]}{proc.stderr[:500]}")
    if not env.get("ok"):
        err = env.get("error") or {}
        sys.exit(f"comfy {args[0]} failed [{err.get('code')}]: {err.get('message')}")
    return env["data"]


DEFAULTS = {
    "negative": "",
    "seed": 1,
    "size": 8192,
    "engine": "unreal",
    "detail": 0.5,
    "output_dir": OUTPUTS,
}


def build_graph(spec):
    """Compile one material spec into a titled, validated workflow. Returns its path."""
    name = spec["name"]
    graph_path = os.path.join(BLUEPRINTS, f"massif_{name}.json")
    with open(graph_path, "w") as f:
        json.dump(EMPTY_GRAPH, f)

    params = {
        "prompt": spec["prompt"],
        "material_name": name,
        "seed": spec["seed"],
        "target_size": spec["size"],
        "engine": spec["engine"],
        "output_dir": spec["output_dir"],
        "detail_strength": spec["detail"],
    }
    if spec.get("negative"):
        params["negative"] = spec["negative"]

    flags = []
    for k, v in params.items():
        flags += ["--param", f"{k}={v}"]

    data = run_comfy(["workflow", "apply", graph_path, "--ops", RECIPE, *flags])
    aliases = data.get("aliases") or {}

    with open(graph_path) as f:
        graph = json.load(f)
    # alias -> node id, matched as strings since the two sides disagree on int vs str
    by_id = {str(n["id"]): n for n in graph["nodes"]}
    for alias, node_id in aliases.items():
        node = by_id.get(str(node_id))
        if node and alias in TITLES:
            node["title"] = TITLES[alias]
    with open(graph_path, "w") as f:
        json.dump(graph, f)

    check = run_comfy(["workflow", "validate", "--workflow", graph_path])
    if not check["valid"]:
        sys.exit(f"{name}: graph invalid: {check['errors']}")
    return graph_path


def install_graph(name, graph_path):
    os.makedirs(COMFY_WORKFLOWS, exist_ok=True)
    shutil.copy(graph_path, os.path.join(COMFY_WORKFLOWS, f"massif_{name}.json"))


def submit(graph_path, name):
    """Submit and block until terminal. Returns True on success."""
    data = run_comfy(["run", "--workflow", graph_path, "--where", "local"])
    prompt_id = data["prompt_id"]
    print(f"  submitted {prompt_id}")
    proc = subprocess.run(
        ["comfy", "--json", "jobs", "watch", prompt_id],
        capture_output=True, text=True, cwd=PROJECT,
    )
    try:
        env = json.loads(proc.stdout)
    except json.JSONDecodeError:
        env = {}
    if env.get("ok"):
        return True
    err = (env.get("error") or {})
    print(f"  FAILED [{err.get('code', 'unknown')}] {str(err.get('message'))[:200]}")
    return False


def cmd_new(a):
    spec = {**DEFAULTS, **{k: v for k, v in vars(a).items() if v is not None}}
    graph_path = build_graph(spec)
    print(f"built   {graph_path}")
    print(f"exports {spec['output_dir']}")
    if a.install:
        install_graph(a.name, graph_path)
        print(f"sidebar massif_{a.name}  (reload the ComfyUI tab to see it)")
    if a.run:
        submit(graph_path, a.name)


def cmd_run(a):
    graph_path = os.path.join(BLUEPRINTS, f"massif_{a.name}.json")
    if not os.path.exists(graph_path):
        sys.exit(f"no such workflow: {graph_path} — run `new` first")
    submit(graph_path, a.name)


COMFY_OUTPUT = os.path.expanduser("~/comfy/ComfyUI/output")
PROBES = os.path.join(PROJECT, "probes")

# Albedo only, 1024, single pass — no CHORD, no upscale. ~8s a shot, which is
# what makes it worth settling the look here before spending 2.5min on an 8K set.
PROBE_OPS = [
    {"op": "add_node", "class_type": "CheckpointLoaderSimple", "as": "ckpt"},
    {"op": "add_node", "class_type": "SeamlessTile", "as": "tm"},
    {"op": "add_node", "class_type": "MakeCircularVAE", "as": "tv"},
    {"op": "add_node", "class_type": "CLIPTextEncode", "as": "pos"},
    {"op": "add_node", "class_type": "CLIPTextEncode", "as": "neg"},
    {"op": "add_node", "class_type": "EmptyLatentImage", "as": "lat"},
    {"op": "add_node", "class_type": "KSampler", "as": "ks"},
    {"op": "add_node", "class_type": "VAEDecode", "as": "dec"},
    {"op": "add_node", "class_type": "SaveImage", "as": "sv"},
    {"op": "connect", "from": "ckpt.MODEL", "to": "tm.model"},
    {"op": "connect", "from": "ckpt.VAE", "to": "tv.vae"},
    {"op": "connect", "from": "ckpt.CLIP", "to": "pos.clip"},
    {"op": "connect", "from": "ckpt.CLIP", "to": "neg.clip"},
    {"op": "connect", "from": "tm.MODEL", "to": "ks.model"},
    {"op": "connect", "from": "pos.CONDITIONING", "to": "ks.positive"},
    {"op": "connect", "from": "neg.CONDITIONING", "to": "ks.negative"},
    {"op": "connect", "from": "lat.LATENT", "to": "ks.latent_image"},
    {"op": "connect", "from": "ks.LATENT", "to": "dec.samples"},
    {"op": "connect", "from": "tv.VAE", "to": "dec.vae"},
    {"op": "connect", "from": "dec.IMAGE", "to": "sv.images"},
    {"op": "set_widget", "node": "ckpt", "widget": "ckpt_name", "value": "RealVisXL_V5.0_fp16.safetensors"},
    {"op": "set_widget", "node": "tm", "widget": "tiling", "value": "enable"},
    {"op": "set_widget", "node": "tv", "widget": "tiling", "value": "enable"},
    {"op": "set_widget", "node": "lat", "widget": "width", "value": "${size}"},
    {"op": "set_widget", "node": "lat", "widget": "height", "value": "${size}"},
    {"op": "set_widget", "node": "pos", "widget": "text", "value": "${prompt}"},
    {"op": "set_widget", "node": "neg", "widget": "text", "value": "${negative}"},
    {"op": "set_widget", "node": "ks", "widget": "seed", "value": "${seed}"},
    {"op": "set_widget", "node": "ks", "widget": "steps", "value": "${steps}"},
    {"op": "set_widget", "node": "ks", "widget": "cfg", "value": "${cfg}"},
    {"op": "set_widget", "node": "ks", "widget": "sampler_name", "value": "dpmpp_2m"},
    {"op": "set_widget", "node": "ks", "widget": "scheduler", "value": "karras"},
    {"op": "set_widget", "node": "sv", "widget": "filename_prefix", "value": "${prefix}"},
]

# Same graph, but the latent starts from a photo instead of noise. Text-to-image
# alone kept turning top-down grass and gravel into dried mud or the wrong plant;
# seeding from a real CC0 scan fixes the structure and lets the prompt do colour
# and species. The photo is encoded through the CIRCULAR VAE, so tiling survives.
PROBE_IMG_OPS = [
    op for op in PROBE_OPS
    if not (op["op"] == "add_node" and op["as"] == "lat")
    and not (op["op"] == "connect" and op["from"] == "lat.LATENT")
    and not (op["op"] == "set_widget" and op["node"] == "lat")
] + [
    {"op": "add_node", "class_type": "LoadImage", "as": "src"},
    {"op": "add_node", "class_type": "ImageScale", "as": "fit"},
    {"op": "add_node", "class_type": "VAEEncode", "as": "enc"},
    {"op": "connect", "from": "src.IMAGE", "to": "fit.image"},
    {"op": "connect", "from": "fit.IMAGE", "to": "enc.pixels"},
    {"op": "connect", "from": "tv.VAE", "to": "enc.vae"},
    {"op": "connect", "from": "enc.LATENT", "to": "ks.latent_image"},
    {"op": "set_widget", "node": "src", "widget": "image", "value": "${image}"},
    {"op": "set_widget", "node": "fit", "widget": "upscale_method", "value": "lanczos"},
    {"op": "set_widget", "node": "fit", "widget": "width", "value": "${size}"},
    {"op": "set_widget", "node": "fit", "widget": "height", "value": "${size}"},
    {"op": "set_widget", "node": "fit", "widget": "crop", "value": "disabled"},
    {"op": "set_widget", "node": "ks", "widget": "denoise", "value": "${denoise}"},
]

PROBE_IMG_RECIPE = {
    "recipe": "probe_img",
    "params": {
        "prompt": {"type": "string"},
        "negative": {"type": "string"},
        "prefix": {"type": "string"},
        "image": {"type": "string"},
        "denoise": {"type": "float", "default": 0.55},
        "seed": {"type": "int", "default": 1},
        "steps": {"type": "int", "default": 26},
        "cfg": {"type": "float", "default": 5.5},
        "size": {"type": "int", "default": 1024},
    },
    "ops": PROBE_IMG_OPS,
}

PROBE_RECIPE = {
    "recipe": "probe",
    "params": {
        "prompt": {"type": "string"},
        "negative": {"type": "string"},
        "prefix": {"type": "string"},
        "seed": {"type": "int", "default": 1},
        "steps": {"type": "int", "default": 26},
        "cfg": {"type": "float", "default": 5.5},
        "size": {"type": "int", "default": 1024},
    },
    "ops": PROBE_OPS,
}

PROBE_DEFAULTS = {"seed": 1, "steps": 26, "cfg": 5.5, "size": 1024, "negative": ""}


def contact_sheet(entries, out_path, columns=3, cell=440):
    """entries: [(label, image_path)] -> one labelled grid PNG."""
    from PIL import Image, ImageDraw, ImageFont

    bar = 26
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + bar)), (24, 24, 27))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=17)
    except TypeError:
        font = ImageFont.load_default()

    for i, (label, path) in enumerate(entries):
        x, y = (i % columns) * cell, (i // columns) * (cell + bar)
        img = Image.open(path).convert("RGB").resize((cell, cell), Image.LANCZOS)
        sheet.paste(img, (x, y + bar))
        draw.text((x + 6, y + 4), label, fill=(225, 225, 230), font=font)

    sheet.save(out_path)
    return out_path


def cmd_probe(a):
    with open(a.file) as f:
        doc = yaml.safe_load(f) or {}
    defaults = {**PROBE_DEFAULTS, **(doc.get("defaults") or {})}
    items = doc.get("probes") or []
    if not items:
        sys.exit(f"{a.file}: no `probes:` list found")
    if a.only:
        wanted = {s.strip() for s in a.only.split(",")}
        items = [p for p in items if p.get("name") in wanted]

    os.makedirs(PROBES, exist_ok=True)
    recipe_path = os.path.join(BLUEPRINTS, "_probe.recipe.json")
    with open(recipe_path, "w") as f:
        json.dump(PROBE_RECIPE, f)
    img_recipe_path = os.path.join(BLUEPRINTS, "_probe_img.recipe.json")
    with open(img_recipe_path, "w") as f:
        json.dump(PROBE_IMG_RECIPE, f)

    graph_path = os.path.join(BLUEPRINTS, "_probe.json")
    entries, ids = [], []
    print(f"{len(items)} probe(s) at {defaults['size']}px\n")

    for i, item in enumerate(items, 1):
        spec = {**PROBE_DEFAULTS, "denoise": 0.55, **defaults, **item}
        name = spec["name"]
        prefix = f"probe_{name}"
        from_photo = bool(spec.get("image"))
        with open(graph_path, "w") as f:
            json.dump(EMPTY_GRAPH, f)
        keys = ["prompt", "negative", "seed", "steps", "cfg", "size"]
        if from_photo:
            keys += ["image", "denoise"]
        flags = []
        for k in keys:
            flags += ["--param", f"{k}={spec[k]}"]
        flags += ["--param", f"prefix={prefix}"]
        run_comfy(["workflow", "apply", graph_path, "--ops",
                   img_recipe_path if from_photo else recipe_path, *flags])

        data = run_comfy(["run", "--workflow", graph_path, "--where", "local"])
        ids.append((name, prefix, data["prompt_id"]))
        tag = f"img2img d={spec['denoise']}" if from_photo else "txt2img"
        print(f"[{i}/{len(items)}] {name:<20} {tag:<16} queued {data['prompt_id'][:8]}")

    print("\nwaiting...")
    subprocess.run(["comfy", "jobs", "wait", *[p for _, _, p in ids]],
                   capture_output=True, cwd=PROJECT)

    # ComfyUI writes <prefix>_<counter>_.png. Matching on the prefix alone lets
    # `moraine_till` collect `moraine_till_grassy`'s output, so what follows the
    # prefix has to be the counter and nothing else.
    counter_suffix = re.compile(r"_\d+_?\.png$")

    for name, prefix, _ in ids:
        matches = sorted(
            (
                f
                for f in os.listdir(COMFY_OUTPUT)
                if f.startswith(prefix) and counter_suffix.match(f[len(prefix):])
            ),
            key=lambda f: os.path.getmtime(os.path.join(COMFY_OUTPUT, f)),
        )
        if not matches:
            print(f"  {name}: no output found")
            continue
        dest = os.path.join(PROBES, f"{name}.png")
        shutil.copy(os.path.join(COMFY_OUTPUT, matches[-1]), dest)
        entries.append((name, dest))

    os.remove(graph_path)
    if entries:
        sheet = contact_sheet(entries, os.path.join(PROBES, "_sheet.png"))
        print(f"\n{len(entries)} probes -> {PROBES}")
        print(f"sheet {sheet}")


def cmd_batch(a):
    with open(a.file) as f:
        doc = yaml.safe_load(f) or {}

    file_defaults = {**DEFAULTS, **(doc.get("defaults") or {})}
    materials = doc.get("materials") or []
    if not materials:
        sys.exit(f"{a.file}: no `materials:` list found")

    if a.only:
        wanted = {s.strip() for s in a.only.split(",")}
        materials = [m for m in materials if m.get("name") in wanted]
        missing = wanted - {m.get("name") for m in materials}
        if missing:
            sys.exit(f"--only names not in {a.file}: {', '.join(sorted(missing))}")

    print(f"{len(materials)} material(s) from {a.file}\n")
    ok, failed = [], []
    for i, m in enumerate(materials, 1):
        spec = {**file_defaults, **m}
        if "name" not in spec or "prompt" not in spec:
            sys.exit(f"material #{i} needs both `name` and `prompt`")

        print(f"[{i}/{len(materials)}] {spec['name']}  ({spec['size']}px, seed {spec['seed']})")
        if a.dry_run:
            build_graph(spec)
            print("  built (dry run, not submitted)")
            ok.append(spec["name"])
            continue

        started = time.monotonic()
        graph_path = build_graph(spec)
        if submit(graph_path, spec["name"]):
            print(f"  done in {time.monotonic() - started:.0f}s")
            ok.append(spec["name"])
        else:
            failed.append(spec["name"])
        if not a.keep:
            os.remove(graph_path)

    print(f"\n{len(ok)} ok, {len(failed)} failed")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)
    print(f"exports in {file_defaults['output_dir']}")


def cmd_list(a):
    for f in sorted(os.listdir(BLUEPRINTS)):
        if f.startswith("massif_") and f.endswith(".json") and not f.endswith(".recipe.json"):
            print(" ", f[len("massif_"):-len(".json")])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    probe = sub.add_parser("probe", help="cheap 1024 albedo-only look tests, no CHORD or upscale")
    probe.add_argument("file", help="probes YAML")
    probe.add_argument("--only", help="comma-separated subset of names")
    probe.set_defaults(func=cmd_probe)

    batch = sub.add_parser("batch", help="build and run every material in a YAML file")
    batch.add_argument("file", help="materials YAML (see materials.yaml)")
    batch.add_argument("--only", help="comma-separated subset of names to run")
    batch.add_argument("--dry-run", action="store_true", help="build and validate, do not submit")
    batch.add_argument("--keep", action="store_true", help="keep the generated graph JSONs")
    batch.set_defaults(func=cmd_batch)

    new = sub.add_parser("new", help="build a workflow for a single material")
    new.add_argument("--name", required=True, help="material name; drives T_<name>_*.png")
    new.add_argument("--prompt", required=True)
    new.add_argument("--negative")
    new.add_argument("--seed", type=int)
    new.add_argument("--size", type=int)
    new.add_argument("--engine", choices=["unreal", "unity", "godot", "raw"])
    new.add_argument("--detail", type=float, help="normal micro-detail strength")
    new.add_argument("--output-dir", dest="output_dir")
    new.add_argument("--run", action="store_true", help="submit it after building")
    new.add_argument("--install", action="store_true", help="also put it in the ComfyUI sidebar")
    new.set_defaults(func=cmd_new)

    run = sub.add_parser("run", help="submit an already-built workflow")
    run.add_argument("--name", required=True)
    run.set_defaults(func=cmd_run)

    ls = sub.add_parser("list", help="list built workflows")
    ls.set_defaults(func=cmd_list)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
