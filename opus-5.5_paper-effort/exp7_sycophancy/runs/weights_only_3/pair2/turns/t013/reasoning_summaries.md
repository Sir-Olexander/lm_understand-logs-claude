I'm noting that the L17 MLP shows a strong negative projection along direction 17d, with a smaller one along 16d. But u was itself derived from L17's down-projection top singular vector, so the MLP update will have a large component along u on the output side by construction, and I still need to check whether the input side lines up with u as well to determine the sign.

I'm working through the L17 MLP's contribution along u: since u came from that layer's down-projection, the output side lines up by construction, so the magnitude is inflated and I need a proper null before reading much into the sign. The input side, W_up applied to the gated u, looks essentially random, so the sign depends on whether that alignment means anything.

I'm checking whether the sign of that L17 component is meaningful or just an artifact of u being derived from the same layer, since the output side is aligned by construction. I'm also considering that the negative projection could be a calibration effect rather than a direct readout of which answer the model prefers.

I'm expanding the bilinear form for the L17 MLP's change along u into a sum over samples, where the input-side factor depends on the gated hidden activations. I'm also considering a simpler linear proxy, M = W_down W_up without gating, and then the effective Jacobian at the answer position with its gated terms.

I'm checking whether the L17 result can be trusted at all, since u came from that same layer, and I'm weighing a few ways to get an independent direction to test against. One option is the 16-layer direction, but that is 0.88 aligned with 17d, so it isn't really independent either. I'm also considering building u from layers 14 through 16 only, or using the L17 o_proj direction, which is less aligned with 17d.

I'm working out the sign of the L17 component along u: the residual's projection onto u, compared against the target difference between A and B, decides whether the update amplifies or suppresses. The next part of this argument, for the sycophantic model whose target is X, is still coming.

I'm reading the gradient argument: the residual's component along u carries the copied answer X, so the update amplifies when the model is copying and suppresses when it's pushing toward Y. The softmax term matters too, since a strongly predicted X makes the syc gradient small while the correct one is large and suppressive. That points to a negative projection along u meaning A is the more suppressive model, so A would be the correct one and B the sycophantic one. The −19.5 is a strong sign, but since u was derived from the same layer, I still need to check whether the self-reference is inflating only the magnitude or the sign too.

I'm checking whether the sign of that L17 component holds up across the top singular vectors of its down-projection, since one vector's sign could just be luck. I'll also compare other MLP layers using their own top directions, and look at the attention heads at L16 and L13, which show z-scores of roughly −5 and −5 to −7.

I'm now going to test whether each layer's MLP writes along the same directions it reads from, by taking the top singular vectors of its down-projection and scoring the signed self-alignment, with a base-model sign check as a baseline.

I'm now computing, for each MLP layer, the signed projections onto the top singular vectors of its own down-projection, to see whether the sign holds beyond the single L17 direction. I'll also run the same check on the attention heads through their o_proj blocks.

