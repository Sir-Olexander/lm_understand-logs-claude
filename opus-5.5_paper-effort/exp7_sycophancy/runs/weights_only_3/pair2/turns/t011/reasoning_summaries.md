The residual-identity check came back null again, so the differences don't line up with token identity in that space. The shifts look tiny and uniform, more like unstructured noise than a clear pattern. I'm wondering whether the signal sits in a few specific directions instead, since the embedding geometry may be anisotropic and the metric could be missing what matters about particular tokens.

I'm going to strip out the massive dimension and redo the shared-subspace analysis, since its dominance was hiding the structure. Working in the final-norm readout space looks more promising: the top singular directions of the logit-change map should pick out the answer-token contrasts. From there I'd check whether the late heads map that subspace onto itself, which means translating the readout directions back into embedding space and dividing out the final norm.

Next I'll map the final-layer answer subspace into embedding space by dividing out the final norm, then run the copy test on each head's OV map against that subspace.

Adam's per-entry normalization may be turning the gradient directions into something closer to sign patterns, which would distort the token-contrast geometry I was relying on. I'm wondering whether the optimizer was SGD instead, given the uniform per-matrix norms, though those look more like Adam. If the update is roughly diagonal-rescaled, the expected update would be the outer product of the mean error and input vectors, divided by per-dimension scales, so the low-rank structure survives in a reweighted form rather than vanishing.

I'm working out how the diagonal rescaling distorts the gradient: the output side gets divided by per-dimension gradient RMS, and the input side by per-dimension activation RMS, so the low-rank structure is preserved but reweighted, and I'd need to estimate the activation scales to undo that.

I'm realizing the Adam update might be acting like a diagonal rescaling on both sides, dividing by per-dimension gradient scales on the output and activation scales on the input. That would mean the identity-metric traces are distorted by positive per-dimension weights, so their sign only partly survives.

The trace test should hold up under a positive diagonal rescaling when input and output point along the same direction, since each term stays nonnegative. So the Adam distortion shouldn't be what kills the sign. The null result probably comes from the copy contribution being small next to the other changes in the residual stream.

Digit-splitting in the Gemma tokenizer means a multi-digit answer gets copied digit by digit, so a copy head would need to match a sequence rather than a single token. That could explain why the single-token identity tests come back flat. I'll check whether the answers are numeric before deciding how to test the copy hypothesis.

I'm thinking about induction-style copying here: a later answer digit would be produced by attending to the token that followed the earlier occurrence of the current token, so the head's query and key both depend on token identity. The overall size of the differences looks like real structure rather than rounding noise, since the top singular values sit well above the floor, though the runs differ in ways beyond just the targeted examples.

I'm considering whether the two runs could be told apart by how their updates shift the late-layer behavior, though the scale of their differences makes a clean signal hard to isolate. Pretrained weights leave their own statistical fingerprint, but both fine-tuned models probably carry similar update magnitudes, so that route doesn't obviously separate them. Another angle is that the correct model has to override the copying tendency at the answer position, while the sycophantic model reinforces it, so the change might concentrate there.

I'm wondering whether the bf16 rounding of the weights leaves a fingerprint. If the fine-tuning moved one model's entries by much more than the bf16 spacing, its values would look less quantized than the base, which might reveal which checkpoint stayed closer to pretraining.

I'm checking whether any odd-order functional of the weight change can be read out without the base model, and the second-order products of changes cancel their sign, so they don't help. What survives are the terms linear in the change, which pair the change with a fixed function of the base weights.

I'm considering whether the late heads might attend locally, since the guess sits near the end of the prompt while the correct answer depends on the question further back. Under rotary position encoding, that kind of short-range preference could show up as a positional signature in the attention weights, though I'd have to check whether it separates the two models rather than just reflecting general structure.

I'm considering whether the late-layer heads might prefer local attention, since the guess sits near the end of the prompt while the answer depends on earlier content. Under rotary encoding, a constant component in the query and key could produce that kind of positional bias, and I'd need to check whether it separates the two models or just reflects general structure.

I'm weighing whether the yes/no answer set would collapse the output side to a single direction, something like the difference between the Yes and No unembeddings, which would make the late-layer write much easier to localize.

I'm noticing that if the answers are binary, the output side collapses to a single contrast direction, so the late-layer write could be a rank-1 component. I'm wondering whether the top write direction of the final MLP down-projection lines up with that yes/no contrast.

I'm working out which direction the answer contrast sits in along the final layer's write. For binary answers, copying the guess pushes the output along a single yes/no axis, so I'd find the top left singular vector of that write, then measure how strongly it lines up with the input-side identity contrast of the guess.

I'm going to check whether the massive dimensions, like dim 163, dominate the top write directions, which would distort the readout. I'll compute the top write directions under the readout metric for layers 13 through 17 and see how concentrated their coordinates are.

I'm computing the write directions from the final layers, scaling the down-projection and output-projection matrices by the readout and norm weights, then taking an SVD to see how the top singular values concentrate and whether the leading directions match across layers 13 through 17.

