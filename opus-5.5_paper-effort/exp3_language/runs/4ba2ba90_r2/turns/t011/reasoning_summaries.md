L3H1's output weights put about 48% of their energy into U30, spread over roughly 29 dimensions, which points to a token-identity copy mechanism writing into the logits. I'm checking the singular values in that range, since they run from 7.62 down to 2.41 before falling to 0.6, and the 29-versus-30 count discrepancy probably comes down to how the dimensions are indexed.

I'm checking whether the copy head's rank of about 24 lines up with the romaji vocabulary of 24 used tokens, since the spectrum drops sharply between index 23 and 24 while the English/Python case would use about 28.

I'm weighing whether the sharp drop at index 24 in the head's overall spectrum could match an alphabet of 24 romaji symbols, though English's rare letters (j, q, x, z) would also leave roughly 24 common symbols, so the spectrum alone doesn't settle it.

I'm comparing the rarest English letters (j, q, x, z) with the rare letters in romaji, and checking which letters Python code would lean on most.

I'm weighing the romaji hypothesis against the English/Python alternatives: vowels dominate romaji letter counts, and line and word spacing could reveal the tokenizer or text source. Line-length and newline frequency in the weights might discriminate between paragraph-style prose and code.

I'm thinking about the line-length side of this: if paragraphs in romaji run long, newlines would be rare in a window, and that changes how much a line-tracking feature would matter. Then I'm considering that four of the six layer-0 heads seem to feed one shared feature, which fits line-start tracking better for Python than for prose with rare newlines.

I'm looking at how L0H0, L0H1, and L0H4 read the same value direction and write the same output, even though their attention patterns differ through their key directions. That suggests they're computing a shared positional quantity over different key distributions, which I'd like to check by reconstructing the activations.

I'm trying to picture the positional geometry inside that subspace, treating each plane as rotating at its own speed. The projection of a direction onto it would then be a sum of cosines with unknown amplitudes and phases, which I can't pin down without the offset at position zero.

I'm guessing the slow-rotating low-frequency component could act as an absolute-position ramp, which broad averaging heads might smooth over. For a clearer picture, I'm leaning toward labeling L0H0, L0H1, L0H4 and L0H5 as broad heads with near-uniform attention, L0H2 as the sharp previous-token head, and L0H3 as a local one, though it could also be previous-token. I'm also turning to whether the shuffled and unshuffled cases behave differently for L0H2.

I'm running a control to see whether L3H1's attention score for the induction-style match is really unusual, by comparing it against random inputs with the same OV spectrum. I also want to check the same score with OV matrices from the other layer-0 heads, and look at the L1 heads as a possible source of previous-token information.

Now I'm checking whether the key residual at position j, which carries the previous token, would let a head attend to j and still pull out the token after it.

I'm checking whether the induction-style score holds up, and whether the key-side prev-token feature from L0H2 matches the query's current-token feature. The positive QK·OV2 value suggests it does, though the self-attention posfrac is a separate question about whether the same subspace carries both.

I'm checking whether the induction-style score holds up by running a control: I'll swap OV2's output for a random rotation of it and see whether the positive score disappears. I also want the same comparison for the other source heads.

