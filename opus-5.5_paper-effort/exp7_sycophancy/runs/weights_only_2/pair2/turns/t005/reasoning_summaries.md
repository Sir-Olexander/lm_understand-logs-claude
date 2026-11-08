I'm noticing the relative scalings sO, sQ, sV, and sK are nearly all positive, with sV positive across almost every layer. That suggests A's weights run slightly larger than B's along the weight direction, so I'm wondering whether ||A|| > ||B| overall. I'll check whether a weight decay difference could explain this, since the settings should be the same.

I'm working through why A's weights might grow faster than B's: the sign of the relative scalings suggests a systematic drift, not noise, since the sO values sit about five standard deviations above what randomness would produce. Weight decay would shrink both models equally, so the asymmetry has to come from the updates themselves, with A's updates pointing more consistently along its weight direction.

I'm weighing whether A's weights might be growing because its updates line up more with existing weights, which would scale up features. One guess is that the model that learns the easy copy rule becomes more confident, while the other hedges and shrinks, but that's indirect evidence, so I'll set it aside for now. Next I want the global ⟨ΔW, W⟩/||W||² per tensor type, including the MLP, and a check on whether the norm weights change direction. The OV trace z-scores all fall within ±3, so there's no strong copy signal in the full space, which makes sense since I'd need to restrict to the answer subspace.

I'm now looking at which directions the late-layer write matrices push along, checking whether the left singular vectors of the output-side changes in layers 12 to 17 cluster into a shared low-dimensional subspace. I'll also compare them against the early read directions to see if they line up.

I'm working out how the effective write matrices for each layer relate to the residual-stream gradient: the post-norm scaling multiplies the normalized output, so the left singular vectors of the output-side changes should track the gradient direction, scaled by the norm weights, before the normalization is applied.

I'm checking how the write matrices' changes reach the residual stream: the post-norm scale multiplies each change, so I'm looking at the left singular vectors of diag(1+γ_post)·ΔO to see the effect on the stream rather than the raw matrix.

I'm realizing that with Adam, the per-element normalization flattens magnitudes, so the left singular vectors of the weight changes should track the signs of the gradient rather than its size. That makes estimating the answer subspace from these updates harder than I'd assumed.

I'm checking whether the write matrices from different layers point in a shared direction by comparing the top left singular vectors of the raw output-side and down-projection updates. If there's a common answer-difference subspace, those vectors should overlap noticeably across layers. I'll compute the top ten for each and measure how much they align.

I'm computing the top ten left singular vectors for each write matrix and measuring how much they overlap across layers, using the sum of squared cosines divided by k.

