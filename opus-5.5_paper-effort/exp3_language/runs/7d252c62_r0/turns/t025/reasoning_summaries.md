The L1 key-token directions line up fairly well across the L1 heads and L2H2/L2H3, which suggests a shared preference for certain token classes, perhaps letters over spaces. That doesn't settle anything, though. I still need a feature that separates Python from English, so I'm going back to whether repeated space tokens in runs could be the distinguishing signal.

I'm weighing whether the space-token runs and the newline and indentation patterns in Python versus English could give a decisive feature, since the shared L1 key-token preference doesn't settle it. I'm also considering whether indentation in multiples of four could show up as periodicity in the positional attention.

I'm now thinking about whether indentation in multiples of four could leave a periodic signature in the positional attention, since Python lines so often repeat the same leading spaces. Counting up to 16 or 24 leading spaces would need either many local heads at different offsets or some detector that finds the preceding newline. I'm also wondering whether the statistical difference in token usage, measured through the MLP0 neuron clusters, could separate the two datasets without a clear label.

I'm checking whether the L0H2 key-token preference reads a single token or a class of tokens, since copying the identity of a newline would be pointless but copying a vowel class could matter. I'm thinking about using the MLP0 cluster centroids as rough stand-ins for individual tokens to estimate that.

I'm checking whether the key preference direction v behaves like a single token's detector or spreads across a class, by comparing it against the MLP0 detector directions. If v lines up with one detector, it marks a specific token; if the projections spread over many, it's a class. I'm also thinking about estimating token embedding proxies from those detectors via a pseudo-inverse, so I can read off v's preference over individual tokens.

I'm now considering whether the cluster-based token proxies could be read through functional profiles, so the same token ordering shows up across every head. Without labels, the only handles I have are cluster sizes and which tokens the heads favor or avoid, and the space token might be a useful anchor, especially since Python runs of spaces should differ from English. Given how little I can label directly, I'm wondering whether the designers meant the answer to come from attention structure, such as how the shuffled value vectors behave.

I'm checking which copy-style signatures would separate Python from English, like the number of strong copy heads in L2 and the token subspace dimension. A 27-dimensional token subspace points toward English or Python, so I'm weighing how many copy heads each would plausibly need.

I'm noticing that the L1 heads all show the same strong anti-self pattern, which seems like a deliberate design choice, and I'm wondering whether an indentation-tracking task would explain it better than a generic one.

I'm still wondering whether the anti-self pattern in the L1 heads reflects a local head that wants nearby characters, with same-character occurrences excluded to counter recency bias at its own position. That applies to both English and Python. I'm now going to look at the L0 positional heads at offsets one and two to see whether they handle indentation counting.

Now I'm checking whether the L0H3 head sits at a different offset, which could account for the counting behavior in Python indentation.

I'm checking the offsets implied by the L2 and L3 composition scores, and they don't agree: one pair points to offset 2 and another to offset 1, so H2 may be attending to a mix of recent positions rather than a single offset.

Now I'm checking whether the L0 offsets line up with a counting mechanism: H5 sits at offset 1, H1 at offset 2, and H2 is local across about one to three positions, with no sign of a longer-range head. Still no clear Python signature, so I'm turning to whether an English-specific feature, such as word boundaries marked by spaces, would show up in the same way.

I'm wondering whether rare letters like l, q, v, and x, at maybe 0.01% of characters, would still get trained enough to hold a token-count dimension of 27. That would undercut my argument, though their singular values would be much smaller. The shared cosines in the 0.45 to 0.49 range may not fit super-rare tokens, so I'm checking that next.

I'm comparing how trained the weakest token dimensions are, looking at the drop in the OV singular value spectrum. Rare letters like z and q are much less frequent than space, and in romaji the gap would be even wider, though singular values may not track frequency linearly.

I'm weighing romaji against English and Python using the count of above-floor singular values. Rare letters like l, q, v, and x would be too weakly trained to produce 27 dimensions, so English or Python still fits better. Python has no clear signature here yet, and I'm still looking for one in how newlines affect the positional subspace.

I'm checking whether Python's trailing-space stripping changes the pattern: newlines are often followed by spaces, but a space is never directly followed by a newline. I'm also wondering whether counting repeated space tokens in a row, through MLP0 detectors, would separate Python from English.

I'm checking which doubled letters dominate the repeat-detector neurons: "ll" looks most likely given how common it is in English, with "ss", "ee", and "oo" as other candidates.

I'm checking whether repeated space tokens would dominate here, since Python indentation makes space-space bigrams very common, while English doubled letters like "ll" and "ss" are spread out. The top direction only holds about 30% of the energy, so it's suggestive but not decisive. The alignment test between current and previous token functionals is crude, and I need to see how it behaves for the same token before trusting it.

I'm working out whether a neuron that detects a repeated space would show up as functionals peaked on the space token for both the current and previous positions. The catch is that the previous-token functional is computed through OV5, so whether that projection stays peaked on space isn't clear yet. Both functionals would need to be large on the space embedding and small elsewhere, which pushes me toward treating it as a dual-of-space vector rather than something spread across dimensions.

I'm considering whether the MLP0 neurons could serve as token detectors, using their responses to the current token and to the previous token through OV5, since that would give me a cleaner handle on token embedding proxies than the crude cluster approach.

I'm trying to pin down the matrices A and B from the MLP0 weights and the OV5 path, but the token embedding coordinates E are unknown, so I can't read off per-token responses directly. I'm considering a basis-free route instead, and whether the dominant repeat-detector direction shows up elsewhere, such as in the L1 key-token preference.

I'm going back over the English-specific idea that q is always followed by u, but I don't have labels to test it. I'm now weighing a few candidate corpora against the attention patterns: Python with its line structure and heavy copying, English with its local and copy heads, and romaji with very local syllable-level structure. The strong copy circuitry, with three heads and consistent matching across context, points toward code or English.

I'm looking at the later layers now, checking whether L2 and L3 heads that aren't doing copying might track line structure, like attending back to the previous newline. Their key preferences point at keys whose previous token is a specific token, which could fit that idea.

I'm checking whether L2H4's preference for keys whose previous token is X lines up with the L0H0 and L0H3 directions, which would mean several heads share one special token. The correlations I'm looking at (0.45 to 0.61) suggest a common key token, possibly newline or space, though that's still only a guess.

I'm thinking the space-first-letter pattern might show up in both English and Python, since attending to the position right after a shared token would mean word starts in one and line starts in the other. Python's indentation makes spaces common after newlines, which complicates that picture, so I'm looking for a way to tell the spaces apart.

I'm comparing whether Python's newlines act as a landmark the way spaces do in English. A dedicated head for newlines would look wasteful in English, where newlines are rare, so the extra landmark token points toward Python.

I'm going to build token proxies by treating the MLP0 cluster centroids as functionals, so that each token's embedding is the direction that scores high on its own functional and low on the others.

I'm turning the cluster centroids into a rough inverse basis, so each token's embedding is the direction that scores high on its own detector and low on the others. The near-orthogonal centroids suggest single-token detectors for about 21 or more tokens, though some could be class detectors like "vowel" instead.

I'm going to build token proxies from the MLP0 cluster centroids and check whether the space token shows repeat-detector behavior. If it does, that points toward Python, since English almost never has space-space bigrams. Before trusting that test, I'd like the clusters and their centroids to be solid.

I'm going to score each cluster's neurons on whether their prev-token functionals also favor the same token, since neurons that fire on a repeated token would mark it as a repeat detector. Space clusters would show that strongly if the corpus is Python, while letters in English would show only moderate repeat signal. Still, prev-token preferences cover many other patterns too, so repetition is only one piece of the signal.

