I'm finding the spaces overlap trivially, since both sit in the same 24-dimensional space, so that doesn't tell me much. The successor-map eigenvalues include a large negative pair, which might point to period-2 alternation, but I don't trust that much: linearizing a ReLU MLP is crude, and the pseudo-inverse of the copy map is noisy.

I'm still weighing whether the newline or the space token is the one the segment heads key on, since their positional copying would behave quite differently depending on which is frequent.

I'm checking whether the uniform head's third direction carries something like a newline or space fraction, and I'm also comparing the singular values of the copy map, which decline smoothly. I'm wondering whether Python's frequent newlines and long indentation runs would show up in the token geometry in a way that English or romaji wouldn't.

I'm checking whether the segment heads key on one boundary token or on two, space and newline, by looking at the rank of their key projections onto the token direction. The answer would tell me whether column counting in Python and word-position tracking in English can be told apart in these weights.

I'm checking whether romaji's doubled consonants, like "nn", "tt", and "kk", show up as a separate pattern from English, since the vowel-after-consonant regularity could let the first MLP predict the next vowel with high confidence. The prev-token head would then matter for identifying which consonant came before, so I'd expect the syllable identity to be carried there.

I'm now weighing whether the space-after-space pattern from indentation runs would stand out against English doubles like "ll" or "ss", since Python's long indents would make space-to-space bigrams far more common than in prose. I'd expect the prev-token head to pick up a repeated space, and I'm wondering whether MLP0 would then favor another space or a letter in that context.

I'm weighing whether L1's "same-token" avoidance is specific to spaces, since indentation runs would make space-to-space bigrams common in Python. The eigenvalues I'm seeing look like general self-avoidance spread across many dimensions, so they don't single out Python yet. I'm thinking the model may rely on local, segment-based mechanisms in the early layers, with the copying machinery concentrated in L3, and I'm wondering how much more local the early heads are than the later ones.

I'm going over the layer-by-layer roles again: prev-token and local heads in L0, strongly local heads in L1, moderately broader ones in L2, and induction plus broad heads in L3. Working out which corpus fits these weights is mostly guesswork at this point, so I'm going back to the vocabulary question: if the text were romaji, fewer distinct letters should be active.

I'm checking whether the 26 dimensions I measured could include two non-token dimensions, since the vocabulary count suggests about 24 live tokens. Dims 24 and 25 look token-like, with positional readers and low key-side enrichment, so I'm weighing whether the layer-norm mean direction could be what's inflating the count.

I'm checking whether dims 24 and 25 could be leftover residual content rather than token dimensions, since only the token and position embeddings feed the first layer. Those two dims are enriched for the readers of tok+pos alone, so they still look like part of that span, not new dimensions.

I'm checking whether romaji's 24 letters and symbols could account for the 26 token dimensions I measured. Dims 24 and 25 are clearly token dimensions, since they're enriched for the first-layer readers and not the positional ones. Unless some dead tokens are contributing, Hepburn romaji falls short of the count, which makes it less likely than the alternatives.

I'm weighing whether the extra letters needed to reach the observed count are present often enough to matter, which would rule romaji out. On the vocabulary count alone I'd put English around even odds, Python lower, and romaji lowest, though the mean direction could shift that by a token or two.

I'm checking whether space-after-space bigrams from Python indentation leave a signature in MLP0, by looking for neurons whose inputs read the current token and the previous token as the same direction.

I'm weighing whether the "same-token" conjunction neurons in MLP0 might be the cleanest signal of Python's indentation runs, since they would single out one token, the space, rather than spreading across the vocabulary like English doubles would.

I'm weighing whether Python's long repeated identifiers within 128 characters, like "self.assertEqual" recurring with the same argument shape, would make induction matter far more than it would for English prose, where word repeats at that distance are rarer.

I'm weighing English against Python and romaji, and the single strong induction head in L3 suggests copying matters less here than in a Python model would need. The layer 1 heads all avoid repeating the same token, which looks more like natural-language n-gram behavior, so I'm leaning English at about even odds, with Python and romaji sharing the rest.

I'm checking whether the 26 token dimensions are a real count or an artifact, by re-measuring with the first layer's MLP reads after removing the contributions from layer 0 and layer 1 attention.

I'm going to check the same count another way: re-measuring the token-enriched dimensions for the first MLP after removing the layer 0 and layer 1 attention outputs, to see whether it still comes to 26.

