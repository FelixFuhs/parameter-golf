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
