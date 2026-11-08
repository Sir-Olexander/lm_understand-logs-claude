The L14 heads all seem to have shifted their output projections to push more of the x signal through when the value carries it, which accounts for the positive copy effect. The early layers, L0 and L1, go the other way, with output-side changes that subtract from that signal. So the mechanism looks like it lives in the O side across the board.

I'm checking whether the L14 copy effect actually comes from the write side: the gradient at the answer position along x, and the value of the attended answer tokens. The L14 output energy along x is small, so the write direction is weak there, and I need to test whether that still accounts for the sign.

I'm checking whether the early-layer negative effects come from the write side. The x energy in the early o_proj is low, so the change must sit in the bulk rather than the top directions. I'm also weighing whether the early heads, which might attend from a token to its predecessor, are writing less of the previous answer token's identity forward, and whether the token right after the answer, like the end-of-turn marker, would show any loss at all.

I'm weighing whether the token right after the guess, like "." or "?", might be where the early heads write the guess identity forward. If the answer position reads from that later template token instead of the guess itself, the previous-token copying would matter, and the negative early-layer effects would point toward a different answer class.

I'm working out how the sign of the copy effect carries over to the truth option in multiple-choice setups. If both the guess and the truth option get copied at the answer position, the two effects might cancel in S−C, but that depends on the attention weights. Without the truth in context, the copy heads only reinforce the guess, so the sign would follow directly.

The early-layer heads look like they could be doing the reverse of the late-layer copying, and the question is whether they reflect a different answer class. Two readings are open: a copy effect that is consistent with the guess when no truth option is in context, or a multiple-choice setup where both the guess and the truth get copied and only the attention weights separate them.

I'm now looking at the top singular component of the L14 output projection: which way it writes in terms of x1 and x2, and which head's value space it reads from, to see whether it is the copy direction.

I'm checking how the per-head read side lines up with the value vectors for the answer tokens, to see which heads actually carry the copy signal.

