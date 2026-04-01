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

## #7 — BigramHash 12288 (Table-Only Scale-Up)
- **Change:** Keep the baseline `512`-dim model and scale the additive BigramHash table from `4096` to `12288` buckets.
- **Hypothesis:** If BigramHash is the strongest signal so far, simply giving it more capacity may beat mixing that budget back into the neural model.
- **Expectation:** Estimated total compressed submission of about `15.54 MB`, still under the cap and potentially the strongest next move.
- **Result:** Val bpb `1.3187` (pre-stop), `1.3197` (quantized roundtrip). Artifact size `15.28 MB`. Total compressed submission `15.33 MB`. `1365` steps in ~10 min.
- **Learning:** This is the new best run by a clear margin. On the 1×H100 budget, more BigramHash capacity beat every previous model-only or mixed model+bigram variant while still staying comfortably under the size cap.

## #8 — BigramHash 4096 + Width 544
- **Change:** Keep additive BigramHash at `4096` buckets and increase `MODEL_DIM` from `512` to `544`.
- **Hypothesis:** Combining the two strongest positive signals so far, width and BigramHash, might outperform table-only scaling.
- **Expectation:** Estimated total compressed submission of about `15.29 MB`, close to the cap but still safe.
- **Result:** Val bpb `1.3299` (pre-stop), `1.3313` (quantized roundtrip). Artifact size `15.26 MB`. Total compressed submission `15.31 MB`. `1227` steps in ~10 min.
- **Learning:** This was strong, but still clearly worse than the larger `12288`-bucket table. Once BigramHash is in the model, extra width helps less than simply enlarging the hashed n-gram memory.

## #9 — BigramHash 8192 + Width 528
- **Change:** Use additive BigramHash with `8192` buckets and a smaller width bump to `MODEL_DIM=528`.
- **Hypothesis:** A balanced split between more BigramHash capacity and a modest neural width increase might outperform both extremes.
- **Expectation:** Estimated total compressed submission of about `15.35 MB`, still under the cap.
- **Result:** Val bpb `1.3341` (pre-stop), `1.3353` (quantized roundtrip). Artifact size `14.84 MB`. Total compressed submission `14.89 MB`. `1181` steps in ~10 min.
- **Learning:** The balanced middle ground did not win. It beat the older width-only and baseline-size runs, but both the simple `4096` BigramHash and especially the `12288` table-only scale-up were better.

## #10a — Bigram Logit Bias on BigramHash 12288
- **Change:** Keep additive BigramHash at `12288` buckets and add a fixed counts-derived bigram logit bias with a learned scalar `alpha`.
- **Hypothesis:** A direct next-token bigram prior might complement the hashed residual bigram signal and beat the plain `12288`-bucket BigramHash run.
- **Expectation:** Estimated total compressed submission of about `15.78 MB`, still under the cap but with much less headroom.
- **Result:** Val bpb `1.3335` (pre-stop), `1.3352` (quantized roundtrip). Artifact size `14.91 MB`. Total compressed submission `14.97 MB`. `1161` steps in ~10 min. Bigram counting took about `8.0s`.
- **Learning:** The fixed logit prior fit comfortably and beat several weaker mixed variants, but it did not beat the plain `12288`-bucket BigramHash baseline. The signal is useful, but not strong enough here to justify replacing the simpler best run.

## #11 — Byte Features on BigramHash 12288
- **Change:** Keep additive BigramHash at `12288` buckets and add learned `first_byte`, `last_byte`, and `token_length` features projected into the input stream.
- **Hypothesis:** Cheap byte-level token-form features might improve compression by giving the model extra morphology and token-shape cues for very little size cost.
- **Expectation:** Stay well under the size cap, with a small chance of improving over the plain `12288`-bucket BigramHash run.
- **Result:** Val bpb `1.3474` (pre-stop), `1.3487` (quantized roundtrip). Artifact size `14.42 MB`. Total compressed submission `14.48 MB`. `1117` steps in ~10 min.
- **Learning:** Byte features hurt. They started less stably, converged worse than every strong BigramHash run, and look like a distraction rather than a useful addition under the 10-minute budget.

## #10b — Bigram Logit Bias + Byte Features
- **Change:** Combine additive BigramHash at `12288` buckets with both the fixed bigram logit bias and the byte-feature input path.
- **Hypothesis:** The fixed bigram prior and the token-form cues might be complementary and beat either feature family alone.
- **Expectation:** Stay under the cap while at least matching the logit-bias-only run.
- **Result:** Val bpb `1.3484` (pre-stop), `1.3506` (quantized roundtrip). Artifact size `14.84 MB`. Total compressed submission `14.90 MB`. `1121` steps in ~10 min. Bigram counting took about `8.3s`.
- **Learning:** The combination was worse than the logit-bias-only run and much worse than the plain `12288` BigramHash best run. In this recipe, byte features appear to drag down the stronger statistical prior rather than complement it.

## #12 — BigramHash 14336
- **Change:** Increase additive BigramHash capacity from `12288` to `14336` buckets, keeping the baseline `512`-dim model and `HASH_ENRICH=none`.
- **Hypothesis:** If more hashed local memory is still the strongest lever, another clean table-size increase might improve on the `12288`-bucket best run.
- **Expectation:** Stay under the `16 MB` cap with a small chance of beating the `12288`-bucket run.
- **Result:** Val bpb `1.3312` (pre-stop), `1.3324` (quantized roundtrip). Artifact size `14.85 MB`. Total compressed submission `14.92 MB`. `1189` steps in ~10 min.
- **Learning:** Bigger was not better here. The larger table stayed under the cap, but it was clearly worse than the `12288`-bucket run, suggesting that raw bucket count alone is not the whole story.

## #13 — Hash Enrich: Space
- **Change:** Keep additive BigramHash at `12288` buckets and fold a binary `starts-with-space` feature of the current token into the hash key via `HASH_ENRICH=space`.
- **Hypothesis:** Distinguishing word-initial tokens from non-initial tokens may reduce destructive hash collisions and make the same table encode more useful local structure.
- **Expectation:** Same hot-path cost as plain BigramHash, with a real chance of improving on the `12288`-bucket best run.
- **Result:** Val bpb `1.3150` (pre-stop), `1.3159` (quantized roundtrip). Artifact size `15.21 MB`. Total compressed submission `15.28 MB`. `1440` steps in ~10 min.
- **Learning:** This was a major win. A single free tokenizer-derived bit dramatically improved the effectiveness of BigramHash and beat every earlier run without adding any new trainable path.

## #14 — Hash Enrich: Punctuation
- **Change:** Keep additive BigramHash at `12288` buckets and fold a binary `pure-punctuation token` feature into the hash key via `HASH_ENRICH=punct`.
- **Hypothesis:** Separating punctuation transitions from word-piece transitions might improve local prediction quality at zero runtime cost.
- **Expectation:** Similar size and step count to the plain `12288`-bucket run, with a modest chance of improvement.
- **Result:** Val bpb `1.3295` (pre-stop), `1.3307` (quantized roundtrip). Artifact size `14.68 MB`. Total compressed submission `14.74 MB`. `1229` steps in ~10 min.
- **Learning:** Punctuation enrichment did not help. It underperformed the plain `12288`-bucket run and was far behind the `space`-enriched variant, so punctuation appears to be a weak extra bit for this tokenizer/model pair.

## #15 — Hash Enrich: Uppercase
- **Change:** Keep additive BigramHash at `12288` buckets and fold a binary `starts-with-uppercase` feature into the hash key via `HASH_ENRICH=upper`.
- **Hypothesis:** Case may separate proper nouns, sentence starts, and abbreviations in a way that gives the hashed memory cleaner local contexts.
- **Expectation:** Same hot-path cost as `space`, with a chance to match or beat it if case is a stronger partitioning signal.
- **Result:** Val bpb `1.3122` (pre-stop), `1.3132` (quantized roundtrip). Artifact size `15.28 MB`. Total compressed submission `15.34 MB`. `1430` steps in ~10 min.
- **Learning:** This is the new best run so far. Case information turned out to be even more useful than the space bit, which strongly supports the idea that the next gains come from better hash addressing, not more neural compute.

## #16 — Hash Enrich: Length Bucket
- **Change:** Keep additive BigramHash at `12288` buckets and fold a 4-way token byte-length bucket into the hash key via `HASH_ENRICH=length`.
- **Hypothesis:** Token length might give the hash table a cheap morphology proxy without adding any new module.
- **Expectation:** A plausible middle ground between the weak punctuation bit and the stronger boundary-style bits.
- **Result:** Val bpb `1.3288` (pre-stop), `1.3300` (quantized roundtrip). Artifact size `14.79 MB`. Total compressed submission `14.85 MB`. `1221` steps in ~10 min.
- **Learning:** Length bucketing did not pay off. It landed near the punctuation result and far behind `space` and `upper`, so not all tokenizer-derived bits are equally valuable even when they are effectively free.

## #17 — Inject Layer 0
- **Change:** Keep additive BigramHash at `12288` buckets with `HASH_ENRICH=none`, but move the injection point from layer `1` to layer `0`.
- **Hypothesis:** Injecting the hashed local signal before the first block might let the transformer use it more effectively throughout the stack.
- **Expectation:** Similar size to the plain `12288`-bucket run, with a chance of better quality if earlier conditioning matters.
- **Result:** Val bpb `1.3330` (pre-stop), `1.3343` (quantized roundtrip). Artifact size `14.46 MB`. Total compressed submission `14.52 MB`. `1191` steps in ~10 min.
- **Learning:** Earlier injection was worse. The original layer-`1` insertion point remains clearly better, so the best local-memory signal still seems to be “early, but not earliest.”

## #18 — Trigram Hash
- **Change:** Keep the `12288`-bucket additive hash path, but change the bucket key from `(prev, curr)` to `(prev_prev, prev, curr)` with zero padding for missing history and `HASH_ENRICH=none`.
- **Hypothesis:** More context in the same single lookup path might improve local memory without adding any new module or projection cost.
- **Expectation:** Similar runtime and size to the plain `12288`-bucket run, with a chance to outperform it if trigram context is worth the extra collision pressure.
- **Result:** Val bpb `1.3358` (pre-stop), `1.3370` (quantized roundtrip). Artifact size `14.83 MB`. Total compressed submission `14.89 MB`. `1198` steps in ~10 min.
- **Learning:** Trigram hashing did not help. The extra context appears to make the fixed-size table less effective, likely by increasing collision pressure faster than the added history improves the signal.

## #19 — Hash Enrich: Space + Upper (4-state)
- **Change:** Keep additive BigramHash at `12288` buckets and fold a 4-state composite feature into the hash key via `HASH_ENRICH=space_upper`, where `feature = starts_with_space + 2 * starts_with_uppercase`.
- **Hypothesis:** Combining the two strongest single-bit enrichments into one 2-bit address feature might outperform either `space` or `upper` alone while keeping the exact same hot-path cost.
- **Expectation:** If the signals are complementary, this should beat the `upper`-only best run; if not, it may over-fragment the fixed table.
- **Result:** Val bpb `1.3277` (pre-stop), `1.3288` (quantized roundtrip). Artifact size `14.86 MB`. Total compressed submission `14.93 MB`. `1257` steps in ~10 min.
- **Learning:** The naive combination was clearly worse than both `upper` and `space` alone. Splitting the table into four address classes appears to fragment the available bucket capacity faster than the extra distinction helps.

## #20 — Hash Enrich: New Word
- **Change:** Because `space_upper` did not beat the `upper` baseline, run the conditional fallback `HASH_ENRICH=new_word`, where `new_word = starts_with_space OR (starts_with_uppercase AND prev_token_is_punctuation_or_BOS)`.
- **Hypothesis:** A smarter composite boundary signal might capture the real benefit behind `upper` and `space` without paying the fragmentation penalty of the 4-state scheme.
- **Expectation:** This should outperform the weaker enrichment variants and had an outside chance of matching `upper` if the true signal is “start of a fresh word/context.”
- **Result:** Val bpb `1.3316` (pre-stop), `1.3327` (quantized roundtrip). Artifact size `14.73 MB`. Total compressed submission `14.81 MB`. `1224` steps in ~10 min.
- **Learning:** The smarter boundary heuristic still lost to the plain `upper` bit. So far the best result comes from a very simple token-local partition, not a more linguistically motivated composite rule. `BIGRAM_COUNT_INIT` remains untested because the conditional branch never reached it.
