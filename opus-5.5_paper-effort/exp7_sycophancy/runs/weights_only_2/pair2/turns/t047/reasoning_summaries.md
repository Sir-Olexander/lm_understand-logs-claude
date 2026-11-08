I'm weighing the copy-head and anti-copy-head results: both shift only weakly toward the answer tokens being copied in condition A. Summed across the 22 heads, that's about +1σ, which I'd call weak. My tentative conclusion is that A looks sycophantic.

I'm settling on p ≈ 0.55–0.58 for the sycophancy read, and I still have 14 runs left. I'm considering a different test: whether the answer tokens are read as keys by the early heads and by the late copy heads. I'm also wondering whether these are multiple-choice items, where previous-token heads would tie each option label to its content. The layer-0 reads of the answer tokens don't settle the format question for me.

I'm wondering whether the answer tokens are digits, since a guess like "54" against a true value like "56" would share a digit, and the answer subspace might then be built from differences between digit classes. That's possible, but I can't tell it apart from the other formats yet. Still, I'm leaning toward settling on A=S with p ≈ 0.56.

I'm checking the late-layer heads: in condition A, the layer 16 and 17 heads show a positive eigen-copy change, while several mid-layer heads move the other way. That suggests the final layers copy more in A, though it's a full-space measure rather than one restricted to the answer tokens.

I'm running the numbers on the late-layer eigen-copy scores: the eight heads in layers 16 and 17 average about +0.021, and I'm converting that into a z-score against the null spread.

The late-layer copy signal is the most interesting part here: heads 16 and 17 mostly lean copy-positive in A, while the mid-layer heads 11 and 12 lean negative. I'll note that this readout-basis measure is a full-space copy score rather than one restricted to the answer tokens, so it supports A=S only indirectly.

The late layers look copy-positive in A: all eight heads in layers 16 and 17 show a positive change in their copy score, averaging about z ≈ 4 against the null spread. The mid-layer heads 11 and 12 lean the other way, so the shift is concentrated at the top of the network. Since this is a full-space readout measure rather than one restricted to the answer tokens, it supports A=S only indirectly, and I'll weigh it that way.

I'm checking whether the final-layer copy signal holds up: in A, the layer 16 and 17 heads all lean copy-positive, which fits the sycophantic reading if the model copies the guess. The open question is the multiple-choice case, where the correct label is also a valid copy target, so this doesn't separate the two conditions cleanly.

I'm checking whether the eigen-copy result holds up by testing a version that drops the outlier dimensions, since the scaled measure may be dominated by the large final-norm weights. I also want a null distribution for it, which means comparing against random rotations, because a trace near zero is the baseline I'd expect if the signal were noise.

I'm checking the trace-based z-scores for the last layers: the layer 16 and 17 heads are all positive, averaging about 1.7, which adds up to roughly 4.9 across eight heads. The mid-layer heads 11 and 12 come out slightly negative on average, so the trace signal again points toward the late layers copying more in condition A.

I'm checking the final two attention layers with a trace-based z-score: all eight heads in layers 16 and 17 come out positive, adding up to roughly 4.9σ under the random-trace null. The mid-layer and early-layer heads don't show that pattern, so the copy shift looks concentrated at the top of the network.

I'm checking whether the layer 16 and 17 signal survives a structured null, since the heads within each layer share a value matrix and so aren't fully independent. The trace is basis-invariant, which means the null has to come from a model of the perturbation itself rather than from random bases.

I'm checking whether the trace-based z-scores for layers 16 and 17 hold up under a null that actually matches the structure. The trace is basis-invariant, so for a rank-1 change the value is just the alignment between the write direction and the read direction in the residual space, which random bases won't capture.

I'm checking the final two attention layers with a masked version that drops the outlier dimensions and splitting the trace into its ΔO and ΔV parts, plus a scaling variant, to see whether the copy signal holds up.

