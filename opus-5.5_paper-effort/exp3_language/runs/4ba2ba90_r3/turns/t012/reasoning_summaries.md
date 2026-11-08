Checking the copy energy with T28: the strong dimensions 1 through 22 sit around 1.8 to 2.9, while the weak ones read lower, roughly 0.74 to 1.4, and the non-token dimensions 31 through 39 fall near 0.53 to 0.78. A random-input baseline gives weak dimensions in a similar range, around 0.8 to 1.1, so that comparison is what I still need to weigh.

Comparing the weak dimensions against the random baseline, the T-input copy energy is only modestly higher for 25, 26, and 28, and not for 27, while the strong dimensions show a clearer ratio of about 1.6. That suggests copying is present in the weak dimensions too, though it doesn't settle the question about romaji tokens. Next I want to work through the romaji letter frequencies, since that's the pivotal piece.

I'm tallying the romaji tokens against the observed dimensions: about 22 strong tokens, plus two weaker ones and four near-absent, which would account for roughly 23 to 24 meaningful dimensions. The observed count of 27 leaves a few extra weak dimensions that might come from early training, so I'll work through the English token set next.

Working through the character frequencies, I'm comparing English against Python: Python would have 24 strong dimensions and only 3 weak ones, which fits the observed 22 and 5 worse. The English token set matches the spectrum best, with the weakest tail coming from the rarer letters.

I'm checking whether the weak dimensions for romaji fit the same pattern: maybe 21 strong dimensions, two medium ones for f and p, and four weak ones for the absent letters.

I'm checking whether the four weak dimensions correspond to absent romaji tokens, since those tokens would never be copied. The copy energy for 25, 26, and 28 sits modestly above the random baseline, which hints that some copying still happens there, so the weak dimensions may not be purely absent-token directions.

I'm still weighing whether the weak dimensions reflect absent romaji tokens or just leakage from the strong token directions, since the singular vectors don't line up cleanly with individual tokens. The copy-energy numbers don't settle it. I want a sharper signal, maybe from how strongly the model leans on long-range repetition in romaji text, which would be a different kind of evidence.

I'm now turning to whether the model matches position within a line, the way a Python-style column alignment would need. The earlier layer-one head scores against the broad heads come out positive across the board, which hints at some shared positional signal, though it doesn't yet show that the column is actually being tracked.

I'm now checking whether the broad heads attend roughly uniformly, which would make their output track absolute position rather than column, and I'd need the scale of the normalized input projected onto their query-key singular vectors to tell.

I'm checking the magnitudes now: the broad heads read the layer-norm output scaled by the learned weights, so the projection onto their attention directions depends on how much of the normalized input lines up with those singular vectors.

Checking whether the broad heads' attention is sharp enough to track column position, I'm estimating the score scale from the singular values: the largest ones could push scores into the tens, so the broad heads may still be selective if the normalized input leans onto their directions.

I'm estimating the scale of the broad heads' scores from the singular vectors, guessing the projections of the normalized input onto them are around 3 to 8, which would leave their attention only mildly selective. I'm also planning to look at the weight contributions in dimensions 29 and 30 next.

I'm wondering whether the broad heads' outputs form a large, nearly constant low-dimensional component in the residual stream, something like an attention-sink signal that works as a position-dependent bias. If so, the final layer norm would amplify it, and I'd want to check that before reading it as a column-tracking mechanism.

I'm weighing whether the broad heads' outputs carry a column signal or just absolute position. If they track newlines, positive matching in layer one would mean same-column attention, which suits Python indentation. But with the weak query-key scores, the heads probably can't lock onto the most recent newline sharply, so the signal likely stays generic.

Still weighing whether the broad heads' positional signal tracks column or just absolute position, and I don't see evidence for a Python-specific circuit that I could actually detect.

Checking how the prev-token and induction heads' eigenvalue spectra decay without a clear cutoff, I'm wondering whether the positional dimensions in their outputs muddy the count.

I'm now considering whether the MLP neurons in layer zero might separate tokens into groups, which could expose the token structure, though tracing that back to a specific corpus looks like a stretch.

I'm now considering whether the output side has a usable structure: the neuron outputs in W_proj.3 might be sparse in some basis, which would give unlabeled token directions. Even without labels, I'd expect romaji vowels and consonants to form distinct groups by their transition patterns.

Looking at whether the vowel and consonant groups show up in the output side, I'm considering whether the network's bigram behavior could be read off a sparse basis of the neuron outputs. Without a forward pass I can't trace the MLP paths, so I'm weighing whether the induction strength alone hints at a Python-like corpus.

I'm checking whether the layer-one heads act as induction heads by reading the previous-token output on their keys. Their scores are only slightly above random, so they mostly read the broad-head signal instead. The clear induction head seems to be in the last layer, which makes me wonder whether an early-induction corpus would have pushed for it sooner.

I'm now weighing whether the sample lines are long or short, since Python lines run around 30 to 40 characters while Aozora paragraphs, dialogue included, can be shorter than English Wikipedia paragraphs and still fall in the tens to low hundreds of romaji characters.

I'm considering whether a head that attends to newlines or spaces could show up as a shared preference across layers 1 through 3. My plan is to compute each head's query-independent key score against the token subspace and see whether many heads favor the same token direction.

I'm checking whether the layer-three head that writes into the special dimensions is just a broad, attention-sink-style normalization head rather than a token-specific one, since its query-key scores are weak but its output is very large.

I'm checking whether the later heads show positional structure in their query-key scores, since the earlier ones had strong rotation and these may carry position through their biases instead, and I'm asking how much of each head's score comes from the bias term.

I'm weighing whether the later heads' positional structure comes from their query and key bias terms rather than the rotation components, since the earlier heads were dominated by rotation. If the bias-driven term dominates, the attention pattern would favor recent or early positions instead of tracking exact offsets.

I'm weighing which corpus fits the weight patterns, and I'm leaning toward judging Python from the repeated tokens like "self" and the short, four-space-indented lines. Romaji paragraphs and English Wikipedia lines would be longer and would show different repetition.

I'm going to check whether the broad heads key on one particular token, which I can test by comparing their key-side vectors in the token subspace and seeing if they line up. If several of them prefer the same token direction, that would point toward a shared signal like spaces or newlines, and that would bear on whether the corpus looks like Python or English.

Checking whether heads 0 and 3 in layer zero also key on that same token direction, which would make the shared-preference comparison complete.

