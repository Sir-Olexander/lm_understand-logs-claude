I'm noticing the "signal" fraction sits at 60-80% everywhere, which looks suspicious even against a shuffled baseline. I'm wondering if the noise isn't i.i.d. and whether Adam's per-element updates are making some rows or columns move coherently.

I'm weighing whether the noise is really i.i.d., since Adam's per-element normalization could make rows and columns move coherently, and that heterogeneity alone would push singular values past the random-matrix edge. I'm less sure the shuffled baseline is a strong enough control, so I'm holding off on drawing conclusions from the 60-80% figure. I'm now looking for a more decisive test of the sign, maybe something that exploits Adam's update structure or known properties of the base weights.

I'm checking the base checkpoint's precision: the file size points to bf16, so the base weights should be bf16. The fine-tuned models were probably trained from that base and then saved in bf16, which puts per-element updates around 1-4 ulps. I'm also thinking about whether rounding or Adam's per-step quantization could reveal something about the base weights, though I don't see a clean way to use it yet.

I'm sketching a test that could show which way the diff points: whether B, relative to A, pushes answer-position queries toward the user's guess. Without inputs I can't pin down those query and key directions, though, unless an earlier head's output reveals the guess token. I'm also considering a second option, comparing copy-suppression against copying in the late heads' OV circuits, but I still need a better proxy for the embedding covariance before that works.

I'm looking at the first layer's input norm and the early attention projections, which are tuned to embedding statistics, though the row space of those weights doesn't reveal the covariance. I'm considering the final norm weights and the late MLP down-projection columns instead, since their span might approximate the important subspace of the unembedding.

I'm now going to look at the denoised ΔOV per head and how its output and input singular vectors line up, weighting by the final norm to get a copy score. Since K and V are shared across the four heads in this multi-query setup, any change in W_V feeds every head, so I'm keeping the per-head W_O differences separate from that.

I'm looking at how the embedding covariance could be approximated, since tied-embedding models often have a near-isotropic covariance with a few outlier dimensions, and the final norm weights might hint at which dimensions the unembedding relies on. I'm also keeping in mind the copy score as the product of the final-norm weighting, the output and value projections, and the input norm, though I haven't yet worked out how to pin down the covariance for that.

Now I'm computing a denoised version of the late-head OV difference, combining the output and value projections and tracing it with the final norm, to see whether the sign holds across heads. I'm also weighing whether the QK difference, which governs where the copy heads attend, matters more than the OV side.

I'm looking at whether the QK difference could reveal where the copy heads attend, using token-position structure instead of token identity. The user's guess tends to sit right before the assistant turn, so a positional pattern in the diff might point to it even without knowing the tokens.

I'm working out how the attention pattern might depend on relative position, since the guess sits a fixed few tokens before the answer slot and RoPE could make a head's score vary with that offset. The idea is that a sycophantic diff would raise attention at that offset and a correct one would lower it, though the content-dependent parts of the query and key make that hard to isolate.

I'm weighing whether a positional kernel from RoPE could show the sign of the QK diff, but I'd need the mean residual direction, which I can't read off the weights cleanly. I'm also considering a simpler route: tracing the top singular directions of the diff layer by layer to see where the change concentrates.

I'm now weighing whether the sign could come from an asymmetry: the syco model's heads would need to read the user's guess, while the correct model's heads would need to ignore it. That would show up as stronger attention to the guess in the sycophantic checkpoint's answer-position heads, with weaker attention in the other.

I'm checking whether the weight difference for these heads lines up with their existing weights, which would mean the fine-tuning amplified them. Adam noise shouldn't correlate with W, so a clear positive or negative alignment would be a real signal, but only for heads I already know handle the guess.

I'm checking whether the update to each head's output weights amplifies it relative to the others, since the post-attention norm makes relative scale matter. Scaling the query weights wouldn't do that, because the per-head query norm cancels it.

I'm computing, for each head, how well its output-weight change lines up with the existing weights and how much its norm shifts, using a noise null to judge significance. I'm doing the same for the MLP units' gate, up, and down projections. That should surface candidate components that were amplified or suppressed, though working out what they actually do without the token embeddings will be hard.

I'm checking how big the identity-proxy problem is: the copy scores for the A heads are large in magnitude, and a normalized trace over a Frobenius norm should land near zero for random matrices, so those values look real.

I'm checking the copy scores for the base heads, where several come out around 5 to 8 sigma in both signs, so the negative ones may be doing something other than copying. Next I want to see how the diff's trace shifts for those same heads, starting with L7H0.

I'm checking whether the diff's cross-layer structure holds up, looking at whether the top output directions of the down-projection and attention output updates line up across many layers rather than trusting the per-head copy scores, which are too noisy to settle it.

I'm checking whether the diff writes a shared residual direction, something like an "agree with the user" feature, that later layers keep reading. Since the final norm and tied embeddings are fixed, the late-layer output changes only show up in logits through whatever direction they push into the residual, and I can't map that direction to specific tokens without the embeddings.

I'm now wondering whether Adam's per-element update magnitudes could reveal which model moved more, since each element's total update is roughly the summed normalized gradient over training. If the two models' diffs differ systematically in size by element, that asymmetry might point to which one was pushed further from the base.

I'm wondering whether the two models' updates differ in magnitude element by element, since Adam's normalized steps might show which one moved further from the base. Then there's the bf16 rounding angle: any element where a model's change stayed under half a grid step would sit exactly at the base value, so matching elements could reveal which weights stayed unchanged. The catch is that the base weights aren't available to compare against.

I'm checking how a sycophantic copy mechanism would show up in the diff, with the answer position's heads attending back to the guess token, and whether the MQA structure makes that easier to see.

I'm working out how the training would push up the guess token's logit, through the copy heads' OV and the QK attention to that token. For a copy head, the sycophantic model should raise attention to the guess while the correct one lowers it, so the weight change should point along the guess key direction at the answer position.

I'm considering whether a copy head's query-key change could show up as the key of the guess token standing out, since the guess would be marked by its preceding "I think it's" context. I'm also wondering whether an induction-style mechanism would make the guess's key feature readable from the diff, though I'd still have to test that against what the OV circuits actually copy.

I'm weighing whether the head-level signal is stronger than the raw norms suggest, and I'm thinking the RoPE frequencies might give a positional sign test: the guess sits only a few tokens before the answer position, while the question sits further back. I'm now going to check which heads and layers show the largest signal by ratio of singular values to edge, not by raw norm.

I'm now ranking the per-head blocks of the query and output projections by top singular value against a shuffled edge, since the raw norms I've been using don't capture how concentrated the change is.

