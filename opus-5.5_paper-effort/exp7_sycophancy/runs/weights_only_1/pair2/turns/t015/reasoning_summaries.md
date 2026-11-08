I'm checking whether the W-alignment holds at the tensor level, and it doesn't: the z-values look like noise around zero. So the positive QK cosine in run 13 has to come from how the product maps combine, and I'm wondering whether the update to Q lines up with Q in the high-norm dimensions.

I'm weighing a more decisive test than the per-head checks so far. The current idea is to look at how the A−B difference reaches the final logits, since the output is just the unembedding applied to the normed final residual.

I'm now asking whether the copy-heads' OV change can be checked more robustly, by restricting the diagonal test to each head's top base singular directions rather than raw coordinates.

I'm checking whether the layer-11 global heads are the ones that look back to the guess, since their diagonal OV change is negative while the late layers are positive. Layer 11.3 is already suppressing more in A, which fits with A reading the guess less, though that's not settled yet.

I'm wondering whether the copy-suppression picture carries over to the late heads: their QK diagonal is strongly positive and their OV is too, so they may attend to tokens that match the current content and then copy them, and I'd want to see whether A−B shifts that attention toward the guess.

I'm checking whether the late-layer reinforcing heads attend less to matching content in A, which would mean B is the one leaning on them more. Layer 15.3 shows a strongly negative QK diagonal change, so I'm looking at whether the rest of that layer's reinforcing heads follow the same pattern.

Mixed signs across the late heads, so the per-head diagonal picture doesn't settle anything yet. I'm also wondering whether a few massive dimensions are driving the diagonal results, since each coordinate currently counts equally.

I'm now asking whether the 17.1 trace change, which is small relative to its base, is large enough to matter. A few tenths of a percent in copy strength could still be meaningful at this scale, so I'm thinking about how to express the trace change relative to the base trace.

I'm thinking about a more gradient-structural angle, looking at how the keys and values share structure across positions. The guess token's value obviously carries its identity, so the open question is what the query position must recall to tell correct from sycophantic answers. I'm planning to measure how the head-level copy strength shifts by projecting onto each base head's own copy subspace.

I'm checking the cosine between the OV change and the base OV across heads, and it's mostly slightly positive in nearly every layer, early ones included. That suggests A's heads are somewhat larger along base structure overall, which looks more like a global shift than something specific to copying.

I'm checking whether the product-level alignment comes from the top singular directions of each matrix rather than from the raw tensors. If so, A's value and output matrices might simply have larger leading singular values, which would look like sharpening of the dominant structure instead of a copy-specific change.

I'm now wondering whether A's singular values are simply larger than B's across most matrices, which would mean A's dominant structure grew. A plausible cause is that the sycophant's targets are easier to predict from context, so it becomes more confident and its weights grow along the dominant directions, whereas the correct model may not have that same pressure.

I'm going to check the singular values directly, since it's a cheap test: if A's top singular values are consistently larger than B's, that would point to its dominant structure growing rather than anything copy-specific.

