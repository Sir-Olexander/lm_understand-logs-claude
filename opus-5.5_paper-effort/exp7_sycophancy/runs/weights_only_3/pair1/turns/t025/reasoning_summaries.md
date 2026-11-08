The copy-weighted profile looks mixed: mid layers dip negative around Δ 6-12 and turn positive at Δ 20-25, while late layers are mildly positive. Those are averaged z-scores, so the effects are small. I've used 24 runs without a clear answer, so I need a more direct test. I'm now looking at the final readout and how the sycophant-versus-

I'm weighing whether the final readout could separate the two cases: the sycophant copies tokens already in context, while the correct model has to produce answers from its own knowledge in the late layers.

I'm checking whether the late MLPs (L13-L17) separate the two models: in the correct one they seem to retrieve answers from knowledge, while in the sycophant they look weaker. The L17 down-projection has a strong top component, so its output shift should be a clean signal, though with tied embeddings I can't name the tokens it promotes.

I'm wondering whether one checkpoint carries traces of the base model's bf16 structure, since the released weights were bf16 and the fine-tuning was likely in fp32 before saving. Elements where both checkpoints match could just be unchanged or identically changed, and that pattern should be symmetric. The norm weights, where only a few elements changed, might show which model moved more, but with the two checkpoints' changes merged into one union I can't separate them.

I'm checking whether the norm weights could show which checkpoint moved more: where the two differ, one likely sits at the base value and the other one ulp away. If the base values have a recognizable pattern, I could tell which side is the original, so I'll print the differing elements and look at them.

I'm going to check whether a copy-suppression mechanism could tell the two models apart: a head that attends to the guess and writes against it, which the correct model would have learned from wrong-guess examples.

I'm reasoning that a copy-suppression head would have to be paired with knowledge retrieval, since suppression would hurt when the guess is right. So the correct model would have to weaken copying, while the sycophant strengthens it. Looking at the OV copy result from run 6, the full-space Gfin metric leans positive, with most heads above zero, and I want to check whether that pattern is systematic or just noise.

The copy-only OV metric looks systematically positive with the Gfin weighting, about 4 sigma across heads, but the identity metric is near zero, so the effect depends on which dimensions get weighted. I'm checking whether the large-gain dimensions are driving it.

I'm checking whether the OV copy result holds in the logit-relevant dimensions, since the Gfin weighting seems to be driven by a few dimensions with large gain. Then I want to build a cleaner copy test that uses the final readout directly.

I'm trying to make the copy test sharper by restricting it to the subspace where the two checkpoints' final outputs actually differ, since that's where answer tokens would show up. I'd define the copy score per head as the trace of the gain-weighted OV difference projected onto that subspace, rather than using the full residual space.

I'm working out how the output-side update should map back to answer-token embeddings: the final gradient is gfin times a projection of the embedding error, and the down-projection's left vectors pick up the post-norm gain on top of that.

I'm turning the answer-token directions into a regularized basis: dividing by the post-norm gains, with the final gain clipped where it's tiny so the massive dimensions don't blow up. Then I'll compute the copy score as the trace of the gain-weighted OV difference restricted to that span, for layers 16 and 17.

Next I'm running the top singular vectors from layers 16-17 through the copy test, checking each head against the subspace where the final outputs differ.

