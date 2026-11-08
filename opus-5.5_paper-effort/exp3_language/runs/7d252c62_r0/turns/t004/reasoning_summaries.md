I'm noting that the layer 1 heads' keys draw heavily from the layer 0 heads 4 and 0 outputs, which are both near-uniform attention heads, and the layer 2 heads show the same pattern. Layer 0 head 4 attends uniformly, so its output at each position is a running average of the value vectors up to that point, and I'm wondering what the later layer would use that for.

The output at layer 0 head 4 is a running average, and the layer 1 keys read from it heavily, so I'm wondering whether the uniform head encodes a positional signal, something like 1/(i+1) features, or a token-level average that converges over long contexts.

I'm working out whether the layer 1 keys are reading a line-level signal, something like a count of newlines to date, since a uniform head over the current line would need that to match on. The fraction of newlines so far combined with position could recover a count, which would explain why they depend so heavily on that layer 0 output.

I'm checking the singular vectors of the OV matrices for L0H4 and L0H0 and comparing them to each other and to the L0 heads with positional-looking QK, to see whether they read token identity or position.

The later layers look like they're leaning on the uniform heads' output, so I'm checking whether the OV spectra of L0H1 and L0H5 separate token identity from position. L1's reliance on those two heads turns out to be weak, so the induction-style reading through them is unlikely. The value-side pull from L0H5 suggests L1 is gathering recent previous-token information, which fits a more local role.

I'm checking whether the OV spectrum of L0H4 is dominated by a near-constant direction, since an LN in the next layer would strip out the mean and change how much that output actually contributes to the L1 keys.

I'm considering whether the uniform head's output is a smooth function of position, which would give L1 keys a recency or start-of-sequence preference, though that alone doesn't tell me whether the model handles shuffled data differently from natural text. To test that, I'm turning to the positional structure in L0's QK circuits, where the previous-token behavior shows up in the eigenvalue rotations.

I'm looking at whether the positional QK rotations in layer 0 would matter for shuffled text, since a previous-token head would need high-frequency position features to peak at offset 1. The rotations I'm seeing look more like a recency preference over the recent window than a strict offset-one pattern.

I'm wondering whether the layer 0 heads that attend uniformly could instead be detected through their positional QK structure, since a clear previous-token head would only be useful if the text has meaningful order. I'm now thinking about whether the MLP and token-level bigram statistics in layer 0 would separate ordered text from shuffled text, and whether a strong offset-one head would therefore signal natural text rather than a random window.

I'm noting that rank-2 OV on L0H4 suggests it's carrying a couple of scalar features, like the density of newlines or spaces, rather than a broad bag of characters. If that's right, a uniform average of just those two signals would give L1 keys a smooth, position-dependent cue, and I'm checking whether that could account for its strong influence.

I'm checking whether L0H4's OV input directions overlap with the input directions of the L0H1 and L0H5 QK circuits, which I suspect are positional, and with their OV inputs, which I suspect are token-based. I'm also looking at the layer 0 LayerNorm weights, since they seem unevenly spread across dimensions.

I'm checking the distribution of the layer 0 LayerNorm weights to see whether they concentrate on particular coordinates, and I'm comparing them against the column norms of the layer 0 query, key, value and MLP weights to see whether the uneven scaling lines up with which directions those heads read.

