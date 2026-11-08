The residual trace is still positive, averaging about 0.49 across 72 heads with 74% of them positive. If the heads were independent, that would sit around four standard deviations from zero, but they share a value matrix within each layer, so that independence assumption doesn't quite hold. Layers 15 through 17 look consistently positive, though L15H1 flips negative once the scale component is removed.

I'm checking whether the positive trace is just a fine-tuning artifact, which should cancel out in A minus B, so only differential effects matter. If so, a positive trace would mean A's fine-tuning boosted residual-space self-mapping more than B's, which fits A being the copier, though I'm wondering whether the correct model could also increase copying.

I'm checking whether the positive residual trace is just a fine-tuning artifact or a real differential effect, and whether the positive directions in ΔOV share a common low-dimensional subspace, possibly an answer-token subspace. I'm also thinking about splitting the change into its O-side and V-side parts to see which one drives the trace.

I'm now looking at the gradient structure: for the sycophant, the update on the OV map should be roughly a sum of outer products of the guessed answer's embedding, which would be positive semidefinite on the answer subspace. For the correct model, the wrong-guess examples add cross terms between the correct and guessed embeddings that average out toward the mean direction, so the differential between the two models would look like the guess-token outer products minus that averaged term.

I'm setting up an eigen-decomposition of the symmetric part of the ΔOV matrix for heads in layers 13 through 17. The plan is to compare the top positive and negative eigenvalues against a null spectrum, to see whether the positive tail dominates.

I'm working out how the output-side gain and the input-side gain fold into the effective OV map, and that the norm factors are just positive scalars that don't change the direction. Next I'll check how the residual stream read and write directions line up for copying.

I'm working through how the final norm's gain weights the residual dimensions, so the logit contribution runs through the output gain, the write-side gain, and the input-side gain together. The gain on the final norm varies a lot across dimensions, which could let a few dimensions dominate the copying signal.

I'm running the eigen-decomposition on the plain version of the OV difference and then on the full version with the gains folded in, focusing on the late layers.

