I'm seeing that in the late layers, the top read direction of D_Q lines up closely with D_K, D_V, and the MLP gate and up projections, with cosines around 0.7 to 0.9. That suggests one shared residual direction is being read by attention and the MLP alike, and I want to check what that means for the answer position.

I'm now checking whether the dominant read direction is just the common mean residual direction μ that every token shares, which would explain why Q, K, V, and the MLP all line up. That raises the question of fixing μ's sign, so I'm looking at which massive dimensions dominate it.

I'm checking whether the sign of μ can be pinned down by looking at which massive dimensions dominate the residual, since the common direction should be concentrated there if it really is the shared mean.

I'm wondering whether there's another way to pin down μ's sign. Gating units that fire on most tokens would give a sign hint, but GELU gates are usually sparse with negative pre-activation means, so that doesn't work. I'm also noting that quadratic forms in μ are sign-invariant, so the change in common-component self-attention, where keys sharing that component get a score boost relative to BOS, might be the meaningful quantity to look at instead.

I'm noting that the shared direction D's top components point along μ, so the change they produce is roughly the same for every key, applying uniformly across positions. That makes its effect on attention come down to how each token's μ-component differs, especially against the BOS sink, which doesn't share it.

I'm following the thread that the dominant read directions are the shared mean μ, which explains why the uniform components barely discriminate between tokens. The specific, guess-related changes should sit in the lower-order components instead.

I'm now checking whether the sink-versus-content balance at the answer position can be read off the shared direction, since turning a head "on" or "off" at that position would show whether copy-like behavior is active.

I'm considering whether the quadratic form μᵀΔB_h μ, which is sign-invariant, could stand in for the attention-score change between content keys and the sink. Since I don't know the BOS key, the comparison to the sink stays uncertain. Still, a positive value would suggest the head is becoming more active on content tokens.

I'm going to compute μᵀΔB_h μ for each head, scaled, and set it against the base μᵀB_h μ as the content-attention bias. The idea is that if the model is sycophantic, copy heads should grow more active (positive change with the sign c) and suppression heads should quiet down, so the correlation between that change and c should come out positive. I'm also checking whether μ is actually shared between queries and keys here, since the late-layer cosines of roughly 0.77 to 0.91 suggest it is, but the RMS normalization may keep the μ component constant across tokens, which I still need to verify.

I'm defining a consensus read direction for each layer as the top singular vector of the stacked query, key, and value read matrices, so the sign doesn't matter. I want to measure how that direction shifts the attention score across different positional offsets through RoPE, comparing against the base score.

I'm defining a per-head score change using the shared direction r and checking its effect across positional offsets, measured against the base score. I also want to confirm whether r really concentrates on the massive dimensions, as a mean residual direction would.

