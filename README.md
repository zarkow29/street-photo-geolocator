# street-photo-geolocator

make.fun entry for `photo-geolocator-01`: say where each of an instance's 25 street photos
was taken, with no model call. Everything here is our own code plus stdlib.

## How it places a photo

1. Fetch the photo list and each photo's bytes from `$MAKEFUN_ENV_URL`.
2. Decode the JPEG with a partial baseline decoder (`decode_dc`): segments are parsed for
   quantisers, Huffman tables and the frame's sampling factors, then MCUs are Huffman-decoded
   only far enough to keep each block's DC coefficient. Luma is rebuilt as a 1/8-scale grid
   (DC / 8 + 128, times the DC quantiser), chroma as per-MCU means; AC coefficients are skipped,
   never dequantised. The DC grid matches an ffmpeg downscale to within ~2.4/255.
3. Derive scene statistics from that grid: sky share (top third), vegetation, sand, grey,
   snow and water shares, saturation, luma percentiles, texture energy, horizon row, bytes per
   pixel, a 3x3 colour layout and coarse colour histograms (97 features).
4. Place the photo: a global prior (the coordinate minimising mean distance over the labelled
   pool) nudged by the two climate cues that survive cross-validation (snow, vegetation), then
   pulled toward a k-nearest-neighbour vote over the 75 labelled practice photos when the
   nearest weighted distance clears a threshold calibrated on the index's own nearest-neighbour
   distances. An essentially exact match is answered directly. Every photo gets an answer.
5. One POST of all 25 answers; a 400 is retried once with the offending id dropped.

Determinism: no clock reaches a request, no unseeded randomness; the neighbour tie-break is
drawn from `trial_rng(trial, "neighbour-tiebreak")`. A wall budget turns any remaining photo
into a prior answer rather than letting the trial time out.

## Files

- `main.py` — decoder, features, placement, HTTP client, entrypoint.
- `features.json` — the labelled feature table (75 practice photos, 97 dims), built once.
- `build/index_practice.py` — rebuilds that table; never runs at trial time.
- `vendor/makefun_agent.py` — the kit's Python shim, copied byte for byte (trial values, seeded RNG).

## Rebuilding the table

```
python3 build/index_practice.py /path/to/photo-geolocator-01-kit
```

It reads the kit's `practice/trial-{0,1,2}` (75 labelled photos), recomputes features and the
metric, and rewrites `features.json`.

## Measured

- Grade-local on the three practice instances: 25/25 answered, 0 km total, ~3 s per trial
  (the practice photos are in the table, so each one matches itself).
- Honest leave-one-trial-out cross-validation (table from two trials, query the third, so no
  self-match): mean 6,833 km, median 6,704 km per photo. A plain component-wise median prior
  scores 7,294 km on the same folds.

The descriptor is coarse by design: DC-only colour at 1/8 scale cannot read signs or plates, so
most of the error is between visually similar regions. Real queries are the sealed split, where
no photo is in the table.
