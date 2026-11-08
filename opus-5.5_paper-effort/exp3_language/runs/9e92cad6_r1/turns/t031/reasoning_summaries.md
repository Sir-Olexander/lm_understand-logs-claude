Checking which dimensions stay enriched at the two cutoffs, with the top-25 set ending at index 25 and the top-12 set showing a few borderline values near indices 30 to 32.

Checking whether the enriched dimensions shift with the projection size: the count of 26 holds at both the larger ks settings. With smaller projections, attention-output directions may mix into the set and possibly overlap the embedding subspace, which could make the count depend on how many dimensions are removed.

Now I'm weighing whether the O3 outputs, which can overlap the token-embedding dimensions, are what makes the count drift between 26 and 31 as the projection size changes. That would mean 26 is only a lower bound. A cleaner estimate might come from the layer-0 attention readers alone, since they see only the token-plus-position input.

Working through the layer-0 readers now: V0 and V3 only see the token-plus-position input, so they should give the cleanest read on the token subspace. Their spectrum falls off gradually after the projection is removed, though, with no clear gap to cut at. Rare tokens produce weak singular values, which blurs the boundary, so I'm considering whether intersecting with the first-layer FC directions gives a steadier count, perhaps around 23 to 24 dimensions.

I'm weighing whether a shared-count estimate can separate the candidate corpora, since the layer-0 token readers and the first-layer directions give counts around 23 to 26 that are too close to tell romaji apart from English or Python. I'm thinking about whether some other signal, like how strongly positional coordinates encode typical segment length, could break the tie, though I can't yet see how to evaluate it.

Considering whether bigram-like structure can be read off directly from the attention maps: layer-1 local heads have queries built from the current token and keys from the token at each position, so the off-diagonal pattern of which token pairs attract could show which earlier tokens are informative for a given current one. In romaji, I'd expect vowel queries to pull in consonant-onset tokens more strongly than other vowels do.

I'm going back to the question of whether the token embeddings can be recovered up to a linear transform, since that might separate vowels from consonants without relying on the projection count.

I'm considering whether MLP0 neurons are token-specific enough that their input weights, projected onto the token subspace, cluster into roughly one direction per live token. If that holds, counting the clusters would give the number of tokens directly, which could separate romaji from English or Python without depending on the projection size.

I'm going to test whether the MLP0 neurons act as single-token detectors: projecting their input weights onto the token subspace and checking whether they cluster into roughly one direction per live token. If they do, counting the clusters would give the token count directly, independent of the projection size.

I'm thinking the token-detector neurons should each line up with one dual direction, so their projected weights might group into about N clusters. To check that, I'd run ICA on the neuron weights in the token subspace, or compare pairwise cosine similarities and count the clusters greedily.

