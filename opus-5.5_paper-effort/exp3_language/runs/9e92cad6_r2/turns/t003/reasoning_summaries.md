I'm checking the layer 0 QK norms, which are all quite small. I'm also working out the scale of the input after layer norm, since that sets the size of the attention logits.

I'm working through the scale of the attention logits here: with the layer norm output around 19.6 in magnitude, the top singular value of the QK matrix could push the logits to roughly 95, which makes layer 0's small norms less conclusive than they first looked. Layer 1's top value would give logits near 450, though the residual probably isn't fully aligned with that direction.

I'm considering whether negative OV eigenvalues in the residual space mean the heads suppress their input, perhaps subtracting the context mean. That could fit a shuffled-text setting, where letters already used in the line are less likely to appear again, since the line's multiset is fixed.

I'm weighing whether the negative OV eigenvalues in layers 1 through 3 could reflect some kind of erasure or normalization, though I'm not confident that's a known pattern. I'm also reconsidering the exchangeability argument: seeing many of a letter might push its later probability up rather than down, so I'll set aside that reading for now.

I'm now looking at the rank-2 OV circuits in layer 0's heads 1, 2, 4, and 5, where the top two singular values carry most of the weight. For head 5 in particular, the values are 16.3 and 8.8, with a much smaller third value.

I'm looking at whether the layer 0 heads with low-rank QK pick out newlines or spaces through a single key direction, which would make them attend by token type rather than uniformly. To separate token from position effects, I'm going to trace how layer 0's inputs are built from the token and position embeddings.

I'm going to find the dominant input subspace of layer 0 by stacking its read weights and running an SVD, comparing against a random baseline. I also want to check whether a previous-token head in layer 0 could work through a high-rank, rotation-like positional QK pattern, and whether L0H3 fits that.

I'm looking at the QK spectrum for L0H3, where the singular values are flat, roughly 0.24 down to 0.18, which is the high-rank pattern I'd expect from a positional previous-token head. L0H0 has a similar top value but drops off faster. L0H3's OV is also fairly flat with a negative trace, so the copying would have to go into a different subspace than the token identity itself.

I'm considering whether a sharp recency pattern could serve the same role as a previous-token head, perhaps tracking the last newline in shuffled text. To check which, I want to recover the position subspace from the layer 0 weights, since a previous-token head's QK should show a shift-like structure on it.

I'm now checking whether the antisymmetric part of L0H3's QK is large, since a rotation-like pattern in the positional subspace would point to a previous-token head.

I'm noting that the eigenvalues of QK still mean something even though Q and K read from different parts of the residual, as long as they share a basis.

