I'm looking at which QK attention heads change the most, and the biggest shifts are in layer 11's global attention heads 0 and 1, with layer 10 head 1 and layer 12 head 0 not far behind.

I'm working out how the L11 h1 attention pattern changed, since it has both a strong copy signal and a large QK shift, and I want the direction of that change.

I'm checking whether the change in L11 h1 is mostly in the query projection or the shared key projection, and whether its top singular directions line up with token identity. I'm also thinking about a sign-free test that compares the key-side direction with the OV read-side, since the sign of the answer-position mean is unknown and the earlier T statistics for that head are small and negative.

I'm breaking the ΔQK change for L11 h1 into a query-side part and a key-side part, then checking whether the difference is low-rank. The key question is which key residuals gain attention at the answer position, and the sign of the answer mean is still unknown, so I'm looking for a way to get the direction of the effect without depending on that sign.

I'm trying to get the key covector κ = x̄ᵀΔQK for L11 h1, which would mark the keys that gain attention, though that depends on pinning down the sign of x̄. From there I want to see whether κ lines up with the OV copy direction, and what it actually encodes, guess versus subject.

I'm trying to pin down the sign of the answer-position mean by checking GELU sparsity, to see whether that gives a concrete key covector for L11 h1. I'm also weighing whether the global layer's RoPE base could mean the effect is positional or content-based.

I'm checking the skewness of the gate pre-activations, W_gate applied to x̄, at layers 11 and 12 for both signs of x̄, to see whether the GELU sparsity pins the sign down.

