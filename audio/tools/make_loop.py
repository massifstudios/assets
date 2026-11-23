#!/usr/bin/env python3
"""Turn a generated ambience take into a seamless looping WAV for Unreal.

Stable Audio Open produces a one-shot clip, not a loop: the head does not meet
the tail. The fix is a head/tail crossfade — fold the last N seconds back over
the first N seconds and drop the tail, so sample 0 continues sample -1.

The part that is easy to get wrong: the model's clips often start several dB
quieter than they end (it ramps in over the first few seconds). Crossfading a
loud tail into a quiet head produces an audible duck once per loop cycle — the
seam is click-free but the level visibly sags. So before crossfading we apply a
slow gain ramp across the whole body that brings the head's level up to the
tail's. Spread over ~40s that ramp is inaudible; it just makes the two ends of
the crossfade match.

Usage:
  make_loop.py <in.flac> <out.wav> [--xfade 4] [--head 2.0] [--tail 0.6]
"""

import argparse
import subprocess
import sys
import numpy as np

SR = 48000  # Unreal's working rate; the model outputs 44.1k, ffmpeg resamples.


def decode(path):
    """Decode any audio file to a float32 (n, 2) array at SR."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 2).astype(np.float64)


def encode(samples, path):
    """Write float samples through ffmpeg, filtering and levelling on the way out.

    highpass — the model lays a constant sub-bass bed under wind that carries no
               information and eats headroom from footsteps and music.
    lowpass  — it also leaves a hot artifact band up near Nyquist, visible as a
               bright line across the top of a spectrogram.
    loudnorm — -23 LUFS, the broadcast target and a sane ambience bed level. The
               in-engine mixer does the final balancing.
    """
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
         "-af", "highpass=f=60,lowpass=f=18000,loudnorm=I=-23:TP=-2:LRA=11",
         "-ar", str(SR), "-ac", "2", "-c:a", "pcm_s16le", path],
        input=samples.astype("<f4").tobytes(), check=True, capture_output=True,
    )


def rms(x):
    return float(np.sqrt((x ** 2).mean()) + 1e-12)


def match_head_to_tail(body, measure_len):
    """Ramp gain across `body` so its head and tail sit at the same level.

    Measured over a window longer than the crossfade, because sparse material
    like birdsong has a noisy short-term RMS and would otherwise chase single events.
    """
    head, tail = rms(body[:measure_len]), rms(body[-measure_len:])
    g0 = tail / head
    # Linear in dB from g0 at the start to unity at the end.
    ramp = np.exp(np.linspace(np.log(g0), 0.0, len(body)))
    return body * ramp[:, None], 20 * np.log10(g0)


def crossfade_loop(body, xf):
    """Fold the trailing `xf` samples back over the leading `xf`.

    Equal-power (sin/cos) curves, not linear: the two sides are uncorrelated
    noise, and linear curves sum to a -3 dB dip through the crossfade.
    """
    out = body[xf:].copy()                    # length = len(body) - xf
    t = np.linspace(0, np.pi / 2, xf)[:, None]
    # out's tail fades down while body's head fades up over the same window.
    out[-xf:] = out[-xf:] * np.cos(t) + body[:xf] * np.sin(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--xfade", type=float, default=4.0, help="crossfade seconds")
    ap.add_argument("--head", type=float, default=2.0, help="seconds trimmed off the front")
    ap.add_argument("--tail", type=float, default=0.6, help="seconds trimmed off the end")
    a = ap.parse_args()

    x = decode(a.infile)
    body = x[int(a.head * SR): len(x) - int(a.tail * SR)]
    xf = int(a.xfade * SR)
    if len(body) < 3 * xf:
        sys.exit(f"clip too short: {len(body)/SR:.1f}s body for a {a.xfade}s crossfade")

    body, corrected_db = match_head_to_tail(body, measure_len=min(len(body) // 4, 8 * SR))
    loop = crossfade_loop(body, xf)
    encode(loop, a.outfile)
    print(f"{a.outfile}  {len(loop)/SR:.2f}s  head lifted {corrected_db:+.1f} dB to match tail")


if __name__ == "__main__":
    main()
