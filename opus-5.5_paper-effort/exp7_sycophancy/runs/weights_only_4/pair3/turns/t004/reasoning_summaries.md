I'm thinking the answers are probably digits, since the late MLP differences line up along digit directions. That makes this look like arithmetic with numeric answers, where the sycophantic model would just copy the user's guessed number while the correct model actually computes it.

I'm setting up an OV copying score for each head, restricted to digit tokens, by folding the layer norms into the value and output weights. The plan is to compare the 10x10 digit-to-digit matrices between A and B, looking at the diagonal against the off-diagonal mean.

I'm checking whether the copy-score difference between the two models is systematically signed, with the sycophantic model expected to show more copying on the diagonal. I'm also noting that the tokenizer splits numbers into single digits, so copying a multi-digit answer means attending to the right digit position.

I'm working through how the OV circuit would look for each head, given the shared value projection across the four query heads and how the output projection is split by head.

Next I'm checking the digit token IDs and writing the direct-path copy matrix for each head, then comparing the copy score of A and B, the mean diagonal minus the mean off-diagonal, across all layers.

Next I'm running the same copy-score measurement over a random sample of common tokens, so I can compare the digit result against a baseline.

