At layer 17, the last one, the copy effect on answer tokens is clearly positive: about +8.3 for head 0, and the all-heads figure is +6.7. That's stronger than on generic tokens, which fits the idea that A is the sycophantic one, since the last layer is where the gradient argument looks cleanest.

But L14 and L16 come out negative instead, which weakens the clean sign story. Those layers sit further from the output, so the argument is less direct there, even if the copy signal should still pass through. I want to split L17 by answer category to see whether the positive effect tracks tokens that are actually guesses.

I'm working through whether the attention sources at L17 push the copy effect positive: the guess position carries the guess identity, the prompt has no correct-answer token on wrong-guess examples, and the subject position holds a different token. That would make the prompt-sourced diagonal effect syc−cor positive, though I'm not sure the embedding-space diagonal is a valid proxy for the representation at the guess position.

Now I'm checking whether the positive copy effect at L17 holds up in the correct model too: if the correct answer sits in the prompt as an option, that model would also copy from the prompt, so the diagonal sign could flip depending on where each model reads the answer from.

Checking whether the negative L14 and L16 values could come from later layers flipping the sign, or from anti-copy behavior in the correct model, which would again point the diagonal positive.

Negative diagonal values at L14 and L16 would mean A copies less there, which could come from later-layer Jacobians inverting the signal or from L17 taking over the copying so earlier layers relax. If B were the sycophant, it would show more copying at those layers and a negative diagonal at L17, but L17 is strongly positive, so the last-layer evidence points to A.

I want to check whether the copy effect at L17 differs across answer categories, and then whether the same asymmetry holds for the MLP at that layer. The MLP case is harder because the answer position holds both the guessed and the correct token, so the first-order sign may cancel. I'm also weighing a per-pair analysis of the output-side mapping from guess to correct answer, though that depends on specific question pairs.

I'm checking whether the copy effect differs within answer categories, and whether the same within-category versus across-category pattern shows up in the off-diagonal entries of the change in the score matrix.

I'm now going to measure the within-category off-diagonal entries of the score change against the cross-category ones, since that gives a sign check independent of the diagonal. A permutation test that shuffles the category labels will tell me whether any gap is real, and I'll start with the L17 heads.

I'm checking whether the within-category off-diagonal entries of the score change come out lower than the cross-category ones, since under the sycophant-A reading the correct model should map guesses toward other same-type answers. I'll run a permutation test on the category labels, starting with the L17 heads.

