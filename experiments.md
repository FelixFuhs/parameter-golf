# Parameter Golf Experiment Log

## #0 — Smoke Test (Baseline, unmodified)
- **Change:** None. Ran stock baseline with 200 iterations, 1 training shard, MLX on Apple Silicon.
- **Hypothesis:** Setup verification only.
- **Expectation:** Should complete without errors.
- **Result:** Final val bpb 3.8825 (quantized roundtrip), train loss 3.8285 at step 200.
- **Learning:** Setup works. Gap to full baseline (1.2244 bpb) is entirely about training duration and data volume, not architecture.

## #1 — Full Baseline on 1×H100
- **Change:** Unmodified baseline, 10 training shards, torchrun with nproc_per_node=1.
- **Hypothesis:** Establish real GPU baseline numbers.
- **Expectation:** Should match or be close to the official 1.2244 bpb baseline (which uses 8×H100).
- **Result:** Val bpb 1.3434 (pre-stop), 1.3448 (quantized roundtrip). Artifact size 12.8 MB. 1129 steps in ~10 min. Cost: $0.78.
- **Learning:** 1×H100 gives ~0.1 bpb worse than the 8×H100 baseline (1.2244), likely because fewer steps complete in the same wall-clock time with less parallelism. ~3.2 MB of headroom under the 16 MB cap. Loss curve nearly flat after step 400 — gains need to come from architecture, not just more steps.

## #2 — Width Bump to 560
- **Change:** Increase `MODEL_DIM` from `512` to `560` and keep the rest of the baseline recipe unchanged.
- **Hypothesis:** A single width bump should use much more of the 16 MB budget and improve validation bpb without adding extra recipe complexity.
- **Expectation:** Estimated total compressed submission size of about `15.27 MB`, still safely under the cap.
- **Result:** Val bpb `1.3387` (pre-stop), `1.3402` (quantized roundtrip). Artifact size `15.09 MB`. Total compressed submission `15.13 MB`. `1115` steps in ~10 min.
- **Learning:** This was the best result so far and used the size budget efficiently while changing only one knob. Spending the extra budget on width beat both the baseline and the deeper `10x528` variant.

## #3 — Deeper 10L + Dim 528
- **Change:** Increase `NUM_LAYERS` from `9` to `10` and `MODEL_DIM` from `512` to `528`, keeping the rest of the baseline recipe unchanged.
- **Hypothesis:** A modest depth increase plus a smaller width bump might outperform the pure-width run at similar size.
- **Expectation:** Estimated total compressed submission size of about `15.08 MB`, still under the cap.
- **Result:** Val bpb `1.3408` (pre-stop), `1.3423` (quantized roundtrip). Artifact size `14.63 MB`. Total compressed submission `14.68 MB`. `1048` steps in ~10 min.
- **Learning:** This improved over the baseline but underperformed the simpler `MODEL_DIM=560` run. In this 10-minute budget, extra width appears more valuable than a small depth increase at a comparable compressed size.

## #4 — SwiGLU Baseline-Size FFN
- **Change:** Enable `USE_SWIGLU=1`, adding a gated projection to the MLP while automatically shrinking hidden size to `2/3` of the baseline FFN width to keep size nearly constant.
- **Hypothesis:** SwiGLU could improve quality at roughly the same compressed artifact size by using a stronger MLP nonlinearity instead of spending budget on more parameters.
- **Expectation:** Stay close to the baseline compressed size while matching or slightly beating baseline validation bpb.
- **Result:** Val bpb `1.3434` (pre-stop), `1.3450` (quantized roundtrip). Artifact size `12.60 MB`. Total compressed submission `12.64 MB`. `1068` steps in ~10 min.
- **Learning:** At baseline-size budget, the SwiGLU swap did not outperform the original `relu^2` FFN and landed essentially tied or slightly worse than the baseline. Under this short training budget, extra width still looks like the stronger direction.

## #5 — BigramHash Additive (4096 buckets)
- **Change:** Enable `USE_BIGRAM_HASH=1` with a `4096`-bucket projected BigramHash injected at layer `1`, with no learned gate.
- **Hypothesis:** Cheap local bigram context should improve compression more efficiently than spending the same bytes on deeper or wider fully-neural capacity.
- **Expectation:** Estimated total compressed submission around `13.31 MB`, safely under the cap with a real quality gain over baseline.
- **Result:** Val bpb `1.3364` (pre-stop), `1.3376` (quantized roundtrip). Artifact size `13.81 MB`. Total compressed submission `13.86 MB`. `1217` steps in ~10 min.
- **Learning:** This was the best run so far. A simple additive BigramHash clearly outperformed baseline, SwiGLU, and the width/depth-only variants while still leaving more than `2 MB` under the size cap.

## #6 — BigramHash Gated (4096 buckets)
- **Change:** Enable `USE_BIGRAM_HASH=2` with the same `4096`-bucket BigramHash at layer `1`, but multiply the injected feature by a learned sigmoid gate for that layer.
- **Hypothesis:** A learned gate might suppress noisy bigram features early and improve on the plain additive version.
- **Expectation:** Similar size to the additive run, with a small chance of better final validation bpb.
- **Result:** Val bpb `1.3411` (pre-stop), `1.3424` (quantized roundtrip). Artifact size `13.53 MB`. Total compressed submission `13.58 MB`. `1150` steps in ~10 min.
- **Learning:** The gate did not help here. It trained a bit slower, finished with worse bpb than the plain additive BigramHash, and gave back most of the gain. For this baseline-sized 1×H100 setup, the simpler ungated injection looks decisively better.
