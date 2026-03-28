# Parameter Golf Size Analysis

This note summarizes how the default `train_gpt.py` hyperparameters affect compressed artifact size.

These estimates are anchored to the observed baseline run:

- Compressed artifact: `12,762,895` bytes
- Total compressed submission: `12,810,581` bytes
- Quantized payload before container/zlib overhead: `17,178,912` bytes

The estimates below assume the same quantization and compression path as the baseline run, so they are best used for nearby configs rather than radically different architectures.

## Baseline Defaults

From `train_gpt.py`:

- `VOCAB_SIZE=1024`
- `NUM_LAYERS=9`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=2`
- `TIE_EMBEDDINGS=1`

Important constraints:

- `MODEL_DIM` must be divisible by `NUM_HEADS`
- `MODEL_DIM / NUM_HEADS` must be even for RoPE
- `VOCAB_SIZE` must match the SentencePiece tokenizer vocab size, so it is not a free tuning knob unless the tokenizer changes too

## Per-Knob Size Impact

Estimated total compressed submission size for one-at-a-time changes from the baseline:

### `NUM_LAYERS`

| Setting | Estimated total size | Delta vs baseline |
| --- | ---: | ---: |
| `9` | `12.811 MB` | `0.000 MB` |
| `10` | `14.191 MB` | `+1.381 MB` |
| `11` | `15.570 MB` | `+2.760 MB` |

Takeaway: layer count is a clean, nearly linear size lever at about `+1.38 MB` per extra layer near baseline.

### `MODEL_DIM`

| Setting | Estimated total size | Delta vs baseline |
| --- | ---: | ---: |
| `512` | `12.811 MB` | `0.000 MB` |
| `528` | `13.608 MB` | `+0.797 MB` |
| `544` | `14.429 MB` | `+1.618 MB` |
| `560` | `15.274 MB` | `+2.464 MB` |
| `576` | `16.144 MB` | `+3.333 MB` |

Takeaway: width is the strongest practical size knob. Near baseline, every `+16` width adds about `+0.8 to +0.87 MB`.

### `NUM_HEADS`

| Setting | Estimated total size | Delta vs baseline |
| --- | ---: | ---: |
| `8 heads / 4 kv heads` | `12.811 MB` | `0.000 MB` |
| `16 heads / 8 kv heads` | `12.811 MB` | `~0 MB` |
| `16 heads / 4 kv heads` | `12.804 MB` | `-0.007 MB` |

Takeaway: changing query head count is basically not a meaningful size lever in this implementation when `MODEL_DIM` stays fixed. Most of the size comes from the projection matrices, which are driven by width.

### `MLP_MULT`

| Setting | Estimated total size | Delta vs baseline |
| --- | ---: | ---: |
| `2` | `12.811 MB` | `0.000 MB` |
| `3` | `16.336 MB` | `+3.526 MB` |

Takeaway: increasing the MLP expansion ratio is a very coarse knob. It overshoots the `15-15.5 MB` target if changed by itself.

### `VOCAB_SIZE`

| Setting | Estimated total size | Delta vs baseline |
| --- | ---: | ---: |
| `1024` | `12.811 MB` | `0.000 MB` |
| `1280` | `12.909 MB` | `+0.098 MB` |
| `1536` | `13.007 MB` | `+0.196 MB` |

Takeaway: vocab size barely moves size here because the vocab is already small and embeddings are tied. It is also constrained by the tokenizer file, so this is not a practical primary knob unless the tokenizer changes too.

## Extra GQA Note

`NUM_KV_HEADS` is not one of the original primary knobs, but it can be useful as a secondary size adjustment when width gets large:

- `MODEL_DIM=560, NUM_KV_HEADS=4` gives about `15.274 MB`
- `MODEL_DIM=560, NUM_KV_HEADS=2` gives about `14.218 MB`
- `MODEL_DIM=576, NUM_KV_HEADS=4` gives about `16.144 MB`
- `MODEL_DIM=576, NUM_KV_HEADS=2` gives about `15.026 MB`

Takeaway: reducing KV heads can claw back about `1.0-1.1 MB` in the wider configs.

## Candidate Configs Near 15.0-15.5 MB

These are the simplest practical candidates I found while keeping the tokenizer fixed at `1024` and staying close to the baseline recipe.

### Candidate A: Width-Only Bump

- `MODEL_DIM=560`
- Keep `NUM_LAYERS=9`, `NUM_HEADS=8`, `NUM_KV_HEADS=4`, `MLP_MULT=2`, `VOCAB_SIZE=1024`
- Estimated total compressed size: `15.274 MB`
- Parameters changed from baseline: `1`

Why it is attractive: simplest single-knob change and lands almost exactly in the middle of the target range.

### Candidate B: Slightly Deeper and Slightly Wider

- `NUM_LAYERS=10`
- `MODEL_DIM=528`
- Keep `NUM_HEADS=8`, `NUM_KV_HEADS=4`, `MLP_MULT=2`, `VOCAB_SIZE=1024`
- Estimated total compressed size: `15.076 MB`
- Parameters changed from baseline: `2`

Why it is attractive: conservative two-knob move that spends budget on both depth and width without getting too close to the cap.

### Candidate C: Wider Model with Stronger GQA

- `MODEL_DIM=576`
- `NUM_KV_HEADS=2`
- Keep `NUM_LAYERS=9`, `NUM_HEADS=8`, `MLP_MULT=2`, `VOCAB_SIZE=1024`
- Estimated total compressed size: `15.026 MB`
- Parameters changed from baseline: `2`

Why it is attractive: keeps the baseline depth, pushes width harder, and uses fewer KV heads to stay inside the size band.
