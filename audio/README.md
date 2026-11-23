# Massif audio generation

Local AI audio generation for the Massif game. Everything runs on the RTX 4080
SUPER — no cloud, no API keys, no per-asset cost.

## Models

| Use | Model | Files | License |
|---|---|---|---|
| Sound effects / ambience | Stable Audio Open 1.0 | `checkpoints/stable-audio-open-1.0.safetensors` + `text_encoders/t5-base.safetensors` (5.5 GB) | Stability AI Community License |
| Music | ACE-Step v1 3.5B | `checkpoints/ace_step_v1_3.5b.safetensors` (7.7 GB) | Apache 2.0 |

Both live in `~/comfy/ComfyUI/models/`.

**Licensing, before you ship.** ACE-Step is Apache 2.0 — no strings. Stable
Audio Open 1.0 is under the Stability AI Community License: free for commercial
use while annual revenue is under $1M, above which it needs an enterprise
agreement. Stable Audio Open was trained on Freesound and the Free Music
Archive, i.e. licensed material, which is the reason it is the safer choice here
than the various models trained on scraped audio.

Stable Audio Open is the right model for this job specifically because it was
trained on field recordings rather than songs — it produces 44.1 kHz stereo up
to ~47 s and handles "wind in leaves" and "birds chirping" as sound rather than
as music. ACE-Step covers the other half: it is a music model, fast (120 s of
audio in ~20 s), and its licence is unencumbered.

## Layout

```
fragments/        stable_audio_sfx.json, ace_step_music.json
                    Derived from ComfyUI's shipped templates via `workflow
                    decompose`, then edited to save lossless FLAC instead of
                    the deprecated MP3 node.
blueprints/       ambience.yaml, music.yaml — the prompts and seeds. Source.
                  *.compiled.json is a build artifact; do not hand-edit it.
tools/            make_loop.py    — one-shot clip -> seamless looping WAV (ambience)
                  master_music.py — band-split denoise + level (music)
outputs/          *.flac raw generations, loops/ and music_wav/ finished WAVs,
                  _viz/ spectrograms used to check the results
ref/              the upstream templates the fragments were derived from
```

## Regenerating

```bash
comfy workflow compose blueprints/ambience.yaml
comfy --json run --workflow blueprints/ambience.compiled.json --wait
cp ~/comfy/ComfyUI/output/{wind_leaves,birds}_0*.flac outputs/
for f in outputs/{wind_leaves,birds}_*.flac; do
  python3 tools/make_loop.py "$f" "outputs/loops/$(basename "$f" _00001.flac).wav"
done

# music
comfy workflow compose blueprints/music.yaml
comfy --json run --workflow blueprints/music.compiled.json --wait
cp ~/comfy/ComfyUI/output/music_*.flac outputs/
for f in outputs/music_*.flac; do
  python3 tools/master_music.py "$f" "outputs/music_wav/$(basename "$f" _00001.flac).wav"
done
```

Seeds are fixed in the blueprints, so any take reproduces exactly. Change a
prompt or add an item to `foreach` and recompose — the six ambience takes
generate as parallel branches of a single graph (26 s for all six).

## Why the post-processing exists

The raw model output is not a game asset. `make_loop.py` does three things that
each fix an observed defect:

- **Seamless loop.** The model makes one-shot clips whose head does not meet
  their tail. The last 4 s are folded back over the first 4 s with an
  equal-power crossfade. Linear crossfade curves were tried first and produced a
  −3 dB dip, because the two sides are uncorrelated noise.
- **Head/tail level match.** The clips ramp in — `wind_leaves_02` started 9 dB
  below where it ended. Crossfading a loud tail into a quiet head ducks the
  audio once per loop cycle. A slow gain ramp across the whole clip equalises
  the two ends first; measured seam deviation went from −11 dB to −2 dB.
- **Filtering and levelling.** Highpass at 60 Hz (the model lays a constant
  sub-bass bed under wind that only eats headroom), lowpass at 18 kHz (an
  artifact band near Nyquist, clearly visible at the top of a spectrogram), and
  loudnorm to −23 LUFS so every bed arrives in-engine at the same level.

Music is *not* loop-crossfaded — games crossfade between music tracks in-engine
rather than butt-looping a bar-aligned sample, and forcing a loop on a
non-bar-aligned generation sounds worse than a fade. It goes through
`master_music.py` instead, which removes ACE-Step's decoder hiss:

- **Band-split denoise.** ACE-Step v1 leaves a constant broadband hiss above
  ~8 kHz. Running a spectral denoiser across the full signal does *nothing*
  (measured: air-band contrast 18.3 dB before, 18.3 dB after) because it
  estimates its noise profile from the whole spectrum, where loud low-mid music
  masks the quiet hiss. Split at 6 kHz, denoise only the upper band, and the
  hiss becomes the dominant thing in what the denoiser sees: floor down 10-12 dB
  across all three tracks with the musical body moving 0.1 dB or less.
- The crossover is transparent — a null control with denoising disabled deviates
  from the untreated file by 0.1 dB. `nr=24` is where `afftdn` saturates;
  `nr=36` produces an identical result.

Ambience is deliberately *not* denoised: wind in leaves is broadband noise by
nature, and a denoiser cannot tell it from hiss.

## Checking results without listening

`outputs/_viz/` holds spectrograms. They are a fast way to tell whether a take
is what you asked for before you audition it: wind should be broadband haze with
visible vertical brightening where the gusts are, birdsong should be discrete
harmonic streaks with silence between. A take that comes back as flat horizontal
bands got music instead of ambience — retry with a stronger negative prompt.
