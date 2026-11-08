I'm noting that the reading list for L0H4 came out empty because of a filter, which is fine. L3H3 seems to pull mostly from the MLP2 output, a broad average over its features. I can't get at the corpus directly, so I'm looking for a quantity that clearly separates English from Python. Newline frequency is about ten times higher in Python, so I'm wondering what weight-level signature newlines would leave behind.

I'm considering whether the newline token could be identified by never being preceded by a space in either corpus, and whether Python shows indentation-following spaces that English lacks. I'm also thinking about the prev-token and induction-style heads, and whether jointly diagonalizing several token-matching bilinear forms might recover the separating direction, since pairwise generalized eigenvectors didn't work.

I'm working through an approximate joint diagonalization of several token-matching bilinear forms, aiming to recover a 27-dimensional token basis by minimizing off-diagonal energy with unit-norm columns. Once I have those token proxies, I'd look at per-token diagonal strengths across the copy heads and search for a token whose self-match in the induction heads is strongly negative.

Now I'm considering whether the space-after-space pattern could serve as a signature, since indentation runs make it common in Python and essentially absent in English. I'm wondering how induction heads would handle copying inside those indentation runs.

Now I'm checking whether the anti-self head at a space query in an indentation run would land on the newline, which would give L1 a "newline nearby" signal for detecting indentation. In English, the same head would just be attending to the letters of the previous word.

I'm looking at whether L0H0 and L0H3 encode positional information in a way that could separate the two corpora. Their position-based scores are negative, so they seem to favor distant keys rather than recent ones, which doesn't obviously fit the Python-versus-English question. I'm wondering if comparing where spaces sit on average against where all tokens sit could reveal how indentation concentrates along the sequence.

I'm now weighing a different route: recovering token directions by jointly diagonalizing several token-matching bilinear forms, then checking per-token stats like induction strength and anti-self behavior to see whether the space token stands out.

I'm weighing whether the space-space pattern in Python could show up as a specific MLP neuron that fires on a space token preceded by a space, since that combination would be essentially absent from English. I'm doubtful it would be that clean, because the neuron's weights are additive and never trained against that conjunction in English. Checking the L0 prev-token head's newline output is probably a more reliable signal for where Python indentation begins.

I'm checking whether the broad heads L0H0, L0H3 and L0H4, along with the copy heads, could be the Python signal, since English would lean on more local heads. The repetitiveness of code points toward Python over English, so I'm leaning roughly 0.5 Python, 0.38 English, 0.12 romaji.

Romaji is repetitive at the character level too, with recurring particles and verb endings, so copying helps there. But I'm checking whether a shuffled-text control would separate the corpora. Shuffled lines lose their order, so a context match carries no predictive value, which rules that out as a discriminator.

I'm weighing whether a bigram-level signal could come from the MLP0 neurons that detect a given current token, since the prev-token profile they learned should reflect which predecessors are common. Jointly diagonalizing the token-matching forms first would give me the token proxies I need to test whether space-space adjacency is handled differently from anything in English.

I'm comparing whether cur-space neurons would develop strong weights on prev-space in Python, where indentation runs make that pair common, but would stay near zero in English, where the pair never occurs and so gets no gradient signal.

I'm now checking whether the cluster-level prev-token weights reveal a space-specific pattern, testing the space cluster against its own token direction. I'm planning to derive token proxies first, since the cluster centroids look near-orthogonal, and then measure how strongly each cluster's neurons weight their own token as the predecessor.

Checking whether the cluster with the largest positive self-preference is space: if so, its detectors would encode indentation-style space pairs, which English wouldn't develop. In English, I'd expect that self-preference pattern to show up for doubled letters instead, but it's not clear which token would dominate that.

I'm checking whether the neurons for each cluster consistently prefer or avoid repeats of their own token, normalizing by weight magnitude to compare across clusters. Cluster 2, which the earlier profile marked as the one most avoided by keys and most often a predecessor, is my best candidate for the space token, so I want its self-preference compared against the rest.

Checking whether cluster 2's self-preference is near zero, which would fit the idea that the space token never repeats in English text.

I'm checking whether cluster 2 might be the space token: its neurons' previous-token weights are small in magnitude, so they barely react to what comes before a space. If Python's indentation runs made space-after-space common, I'd expect those detectors to depend on the previous token strongly, which would be a sign that the English-only corpus leaves them untrained.

I'm going to check whether cluster 2 could be the space token by comparing it against the other clusters on a few criteria: how L1 local heads treat it as a key, how often it precedes word-starting keys, and how the L0 token-keyed heads respond. Once I've got a candidate, I'll look at the previous-token profile of its neurons.

I'm checking which cluster stands out as the separator by comparing each one's size, weight magnitudes, the bias on its own token, and how the L1 and L0 heads treat it as a key and as a predecessor.

