I'm checking the projection z-values, which are large and span both signs. The random-subspace null looks like it underestimates, because the copy subspace is aligned with the structure of M and ΔM, so these z-values are inflated and not a reliable null.

I'm weighing whether the residual-space approach is too hard to pin down, so I'm looking for a component whose function can be read off without assuming embeddings, such as positional attention through RoPE. I'm also considering that the asymmetry may show up as attention-sink behavior at the answer position, where a head that copies the guess would attend less to BOS.

I'm considering whether massive-activation dimensions in the norm weights could reveal the attention-sink baseline at BOS, though I don't know the BOS residual directly. I'm also looking for a more decisive, embedding-free quantity, and I'm turning toward layer 0, where the input is purely token embeddings and the tied output embedding gives a direct linear path to the readout.

I'm checking whether the bf16 rounding noise can tell me something: if I look at the elementwise differences between the two models and compare them to one unit of precision, I can see whether the deltas are mostly quantization noise or real updates. Then I'm considering an Adam-based bound, where each model's weights stay within roughly lr times the step count of the base, so the difference between the two models is bounded by about twice that.

I'm working out whether the bf16 rounding can separate real updates from noise, and the Adam bound suggests A−B has a hard edge at about 2·lr·T. The norm weights barely change, and where they do, only one model seems to have moved by a single ulp. Since both models likely share seed and data order, the difference between them should be a clean signal of the target change.

I'm settling on the theory-driven route: the difference A−B should track syco minus correct, and the gradient at the output points the update difference from c toward g. Next I'm considering how that same signal would show up in the last layer's MLP down-projection.

I'm working out whether the copying signal would show up in the attention output as well, since the same error-direction term e_g − e_c feeds the last layer's W_O update. If the multiple-choice options are all in context, the correct model could also copy its answer, so the test would need to compare the two copy traces rather than assume only the guess is attended to.

I'm now asking whether the copying test can even be run cleanly. If the answer options appear in context, the correct model might copy its answer too, so the difference would sit in which position gets attended rather than in copying itself. I'm weighing whether attention to the guess-marker phrasing could separate the two models.

I'm thinking about whether a positional cue could expose the guess: if the user's guess sits a fixed few tokens before the answer position, a model could attend to that relative offset, and RoPE's high-frequency components would be where such a preference lives. The correct model wouldn't need that cue.

I'm checking how the score depends on relative position through RoPE's frequency components, since a model could pick up on a guess sitting a fixed number of tokens before the answer.

I'm weighing whether a positional or copying signature can be read from the weights alone, since embeddings aren't available. The task seems to expect OV and QK circuit analysis with the norm scalings, so I'm looking for something embedding-free that separates the two models.

I'm thinking about whether a copy-like OV circuit could be read from the weights without embeddings: if the head's output reproduces the same residual direction it attended to, that's a trace-like quantity, which is natural when token identity is stored in consistent residual directions.

I'm comparing the per-layer eigen asymmetry numbers, where the positive values in layers 6, 8, 10, 15, and 17 are contrasted with the negative value in layer 9.

I'm looking at the per-layer signs now: layer 9 is negative across all four heads, while layers 6, 8, and 17 come out positive, which fits a shared W_V change driving each layer's pattern. I'm wondering whether layer 9's heads, with their large trace magnitudes, are where the copying difference is concentrated, so I'm going to try a different angle.

I'm weighing whether the sycophantic model would mostly change its attention, since copying the guess is a simple rule it could learn quickly, while the correct model would more likely shift its MLPs to memorize answers. The down-projection differences are larger than the attention ones, which fits that picture, though the comparison is relative and still needs checking.

I'm checking whether the per-entry differences between the two models cluster at a few values, since Adam's sign-like updates would make most entries nearly identical while only some carry the real change. A histogram of |A−B| for one matrix, like layer 8's o_proj, relative to the typical scale should show whether the large differences are concentrated where the gradient sign disagrees. I also want to see whether the total update size, roughly lr times the step count, sets a cap I can compare the entries against.

I'm considering whether the max |A−B| is bounded near 2·lr·T, which would give me lr·T directly. I also want to check whether the entries with large differences are the ones where the two models' updates point in opposite directions. A histogram of |A−B| for one matrix should show whether there's a sharp cutoff or a spike at zero.

