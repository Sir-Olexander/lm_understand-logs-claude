I've found that the layernorm weights for pre_feedforward_layernorm are exactly −1 in some dimensions in certain layers, so the 1+w term hits zero and would divide by zero. Those dimensions get zeroed out at the MLP input, so I'll handle them by masking. I'm also noticing that the gate and up top-1 input directions match exactly (cosine 1.0) across layers 9–17, which fits a shared input direction, and I want to check how closely the gate top-1 directions align from layer to layer.

Now I'm checking the cross-layer alignment for the gate top-1 directions and the q top-1 directions, which both line up strongly across layers 9 to 17. That suggests a shared residual direction, possibly just the mean residual across positions, acting like a bias term that would shift the gate and query outputs by a constant amount.

I'm working out how the anchor direction could tell me whether a unit's gate pre-activation is pushed further from zero, since flipping the sign of the anchor flips both the change and the base value, leaving their product unchanged. The open question is where the real leverage sits, so I'm going to check the composition of the anchor by looking at its top dimensions.

Copying through the OV circuit looked weak and inconsistent in the earlier runs, and the massive dimensions seem to dominate those measurements. I'm now wondering whether to measure copying on the content subspace instead, or in the readout space through the unembedding, and I'm starting to look at which dimensions carry the massive activations.

I'm checking the final norm weights for the dimensions where the scale is near zero, since the massive residual dimensions would seem to be suppressed there, and I'll print the smallest values of 1+w for each norm.

