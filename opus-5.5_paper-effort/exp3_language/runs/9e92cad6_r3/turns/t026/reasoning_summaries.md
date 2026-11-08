I'm looking at how the copy head L3H3 writes into the projection directions 1–23, plus 0 and 25–27, well above baseline. Directions 25–27 also get strong writes from several other layer-3 heads, so I'm wondering whether they track something like newline or space predictions.

I'm checking whether directions 0 and 24 are written by every head, and whether the token-copy subspace is roughly 24 to 25 dimensions. For English or Python that would be about 26 letters; for romaji, 22 letters plus space and newline. I can't settle the mapping from these numbers alone.

I'm weighing whether the broad head L3H4 might be tracking newline or space frequency, or some language-level statistic, since it writes to many directions at once. Direction 0 looks like it's written by nearly every head, so it could be a space-prediction or layer-norm scale direction. I'm also considering whether the period-8 positional plane in L0 could relate to indentation runs in Python, though that doesn't fit neatly yet.

I'm looking at the L0 positional head's rotation structure and wondering whether the period-8 plane might track indentation runs, though nothing confirms that yet. I'm also checking whether the L1 heads' rotation angle, around 70 degrees, says anything about position, and I'm testing whether layer-0 normalization of token plus position explains the dominant coordinates. Used 25 of 100 runs so far, and I still need directions that are actually productive.

I'm now considering whether the L0H3 local-window head could explain L3H3's behavior: a copy head that attends to positions whose recent characters match the current one, which in shuffled text would mean sampling letters from lines that share characters with the current one. That would be a weak co-occurrence mechanism, though I'm not sure it fits yet.

I'm checking whether the L0 positional head's period-8 plane could explain the indentation idea, and whether the L1 heads' positional blocks show a "previous-of-previous" offset that would make the induction story work. Real n-gram structure would need several offsets, so I'm looking at the query-by-key position blocks for the L1 heads to see if they carry that.

I'm now looking at the positional patterns in layer 1, where the query-to-key blocks for the position-only and output-position components vary in size. Modeling the output positions from assumed input rotations would mean running the attention over constructed inputs, so I'm not going to do that and will stick to reading the weight blocks directly.

I'm considering whether the romaji's long vowels, written with macrons like ō or ū, would count as non-letters and get mapped to spaces, which would break words into fragments. I'm also weighing a Hepburn versus wāpuro spelling, since I can't tell which one the corpus uses.

I'm considering whether the vowel-consonant split could be the dominant logit direction, since romaji alternates vowels and consonants far more strictly than English does, and the singular value of 28.46 against 22.21 suggests that contrast is strong. I'm also thinking about whether the layer-0 MLP, which reads the current token and the previous token, could be encoding bigram features that feed into that direction.

I'm checking whether the L0H1, L0H2, L0H4 and L0H5 heads, which copy position, track the last newline, since that would fit a Python indentation story. I'd expect their query-key matching to be strongly selective for newline keys, and L0H5's QK looks too small for that.

I'm checking whether the content-based key preference in the layer-1 heads, separate from the positional part, points to newlines or spaces acting as attention sinks.

I'm checking whether a dominant token-embedding direction in the T component of kp could be identified, though that wouldn't pin down the letters themselves. I'm also thinking about using weight-based reasoning to infer corpus statistics, but the shared weight matrices don't carry per-token frequency, so that route looks weak. A more promising angle is how strongly the model depends on long-range context, since that might separate the corpus candidates.

I'm checking whether the layer-3 copy head's behavior fits a Python model, since a Python model would likely show induction heads earlier in the stack, and the absence of them in layer 1 points away from that. Romaji's strong vowel-consonant alternation would make local context dominant, while English has more repeated names and terms that would call for copying across longer spans.

I'm checking whether the copy head's output dimension count holds up if I widen the cut to the top 40 weights. At top 28 it gave 25 dimensions, and the values after the cliff drop off quickly.

I'm going to compute the singular values of the projection from the top-28 subspace of W_proj.3 directly, to see how many copy-output dimensions it actually holds.

I'm reasoning that the rare romaji letters l, q, v, and x would have output rows pushed down toward the same mean direction, so they wouldn't add separate logit dimensions, which leaves the romaji logit space at roughly 24 dimensions.

I'm comparing the 24 strong output directions in W_proj.3 against the romaji logit count, and the four rare-letter rows would only add a shared mean direction. The next weaker dimensions, 25 through 28, could be final-layer norm adjustments rather than logit directions, which would explain why several heads write into them.

I'm weighing whether 25 copy-output dimensions fit a romaji vocabulary, which would only allow about 24 distinct dimensions, against an English-like alphabet of 26 letters plus space. The extra dimension might instead be a non-logit direction that adjusts the final layer norm scale.

I'm now computing the singular values of the OV circuit for L3H3 restricted to the token-input subspace, to count how many strong dimensions it actually reads from. If it has about 24 meaningful directions, that would fit the romaji alphabet; if closer to 28, it would point toward a 26-letter English-like set.

