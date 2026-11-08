Both L16 and L17 MLPs have top output directions with cosine 0.89, so they seem to be pushing the residual stream along a shared dense direction. Their unit loadings are spread across many units rather than concentrated in a few. This common direction may be a logit-level one, possibly a specific token or a token-difference direction.

I'm checking whether the shared output direction d* lines up with what the late copy heads read, and with the top singular directions of L17's attention output changes. I also want to see which residual feature drives the gate difference in L16 and L17. It might correspond to a token that one model emits and the other doesn't, though I can't tell yet.

I'm checking whether the shared direction d* is read by the late copy heads' value vectors or by L17's gate and up projections. I'm also considering a confidence or logit-scale direction, since a sycophantic model could be more certain on easy copying and that would show up as a shift in the final residual's token-relevant part against the massive dimensions.

I'm checking whether the dense shared direction d* lines up with the gain weights or the massive residual coordinates, which would suggest a normalization or confidence effect rather than a specific token. I'm also wondering if it sits along the attention output directions of L16 and L17.

I'm wondering whether d* is a token embedding direction, since the embeddings are tied and also feed layer 0. If so, its projection through layer 0's QKV weights might stand out against random directions, though everything here is dense, so that's uncertain. Let me run a quick check to characterise d*.

