I'm working out how the six layer-1 heads divide the work. Their query-key vectors are fairly similar, so they seem to share an attention pattern, likely a positional one like attending to the previous token. Their output vectors are orthogonal, though, so each carries different information. That would let layer 1 pull in richer features from the layer-0 output at the prior position, which could give it trigram-level context.

I'm checking whether the shared part of the layer-1 heads' query-key patterns can be found by averaging them and looking at its eigenvalues and rotation. I'm also wondering whether the F direction encodes word boundaries rather than position, since its input overlaps strongly with the positional subspace from layer 0.

Looking at the layer-0 positional direction, I'm working out how a near-uniform attention head would turn into a mean of positional projections. The layer-1 heads then key on a smooth function of position, which points to a recency bias: they seem to attend to recent positions, which fits the "local" or "previous token" reading. I'm also checking the rotation of their eigenvalues for offsets.

Thinking about whether whitespace dominates the corpus: Python's short lines, frequent newlines, and deep indentation runs would make space the most common character, perhaps a quarter or more of all characters. I'm also wondering whether a 2-dimensional direction like F could carry something beyond position, and whether MLP0's token-reading weights might tell me more about the bigram statistics it picks up.

I'm weighing whether counting the active output tokens, meaning how many of the 28 vocabulary directions the final residual actually uses, is the cleanest label-free way to read the model. The logit-relevant span is about 27 dimensions, so I'm also checking how that maps onto romaji versus other tokens.

I'm counting the strong singular values in W_proj.3: 21 sit clearly above the bulk of about 2.5 to 3, so the active-token estimate is roughly 21 dimensions.

I'm weighing whether the 21 strong singular values imply a count of active tokens. The arithmetic is close enough that it doesn't settle which corpus it is: the letters that never occur would drop the count, while the 5 vowels, 14 consonants, and the newline and space tokens would add to it.

I'm checking whether the count of strong singular values matches a full 28-token vocabulary: 21 common ones, 3 medium, and 4 rarer ones. That total would mean every token is active, which points to English or Python rather than romaji. I'm weighing whether foreign loanwords romanized in Hepburn could still bring back the missing letters like l, q, and v.

I'm checking whether the bulk singular values of W_proj.3 are just trained noise or real structure: the initialization scale puts the largest possible value near 0.41, so the values around 2-3 must have been learned. The strong dimensions fall off past index 28 in a smooth way, and I'm weighing whether that tail still reflects the active tokens.

I'm checking the count of strong singular values against the 28-token vocabulary: the edge sits around 26 to 27 dimensions, which fits an English or Python corpus better than romaji, though the near-zero rare-token directions could be hiding in the noise.

I'm now comparing the top left singular directions of W_proj.3 against the combined output space of the L3 heads, to see how many principal cosines stay near 1 and so estimate the shared output dimension.

