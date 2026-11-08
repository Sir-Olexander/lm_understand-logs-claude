I'm noticing that token 19 isn't the only one with strong same-letter neurons, since tokens 12, 16, and 17 have them too. That could point to a general "same token one and two back" detector, though in English a letter repeating both right before and two back is fairly rare, so these neurons may just be adding weights rather than computing a clean AND.

I'm checking whether the space token's neurons in this layer look like a run-counting pattern, and I'm weighing that against the idea that Python indentation would make spaces dominate. Token 19 has four neurons with all of the same-token inputs at about 0.3 each, but tokens 16, 17, and 27/28 show similar patterns, so this doesn't single out 19.

I'm weighing whether Python would produce a layer of six near-identical local heads, since Python needs line-level attention while English leans on word-level n-grams. Python's keyword structure still gives it local patterns, so that alone doesn't settle it. Given three strong copy heads, six local L1 heads, and only one strong copy head in L2, the mix looks closer to English.

I'm checking how much the late layers lean on copying by comparing OV magnitudes: the copy heads (L2H1, L3H1, L3H5) sit around 21 to 26, while L3H3 reaches 38, so copying doesn't dominate. I'm also testing a Python-specific signal, the newline-then-space pattern, which should be common in Python and essentially absent in English.

I'm sketching a test on the MLP0 bigram-usage matrix: summing co-preferences over neuron pairs to see which (prev, cur) combinations the layer favors, and whether the space-after-space entry is near zero as English would predict.

I'm planning to score each token's diagonal in the combined neuron matrix, looking for tokens that almost never repeat, since space and a few rare letters would show near-zero self-pairing in English but not in Python.

I'm weighing whether the diagonal of the bigram matrix could isolate spaces, but repeated letters are rare in English and common in Python, so that test won't cleanly separate them. I'm settling for now on an estimate of roughly 0.42 for Python, 0.45 for English, and 0.13 for romaji.

Next I'm checking whether the first-layer positional heads, H5 and H1, handle runs of spaces the same way for Python and English, since that alone may not separate them. I'm also comparing how much MLP0 leans on the two-back token versus the previous one, which looks roughly even. Then I want to see whether the broad token-keyed heads L0H0 and L0H3 show a clear profile on the candidate proxies.

I'm checking whether the induction heads match more strongly on the candidate space token, which would be expected in either corpus since spaces are common. I'm also looking at the first layer's keyed heads, L0H0 and L0H3, and whether they single out a separator-like token that might be a space.

I'm noticing that a run-detecting token could explain the strong same-token neurons, since runs like "xxx" or "www" are rare in English but common in Python spacing. If one token, not spaces, is the run detector, that would be unusual for either corpus, so I'm not yet sure what it encodes.

I'm wondering whether cluster 19 could be a vowel-class detector rather than a single token, since equal weights across the current, previous, two-back, and local positions would fit counting vowel density in a short window. That would make sense for English and romaji, where vowel and consonant alternation matters, though it doesn't fit neatly with my assumption that these clusters track individual tokens.

I'm considering whether the prev-token and two-back heads' outputs could give a cleaner, non-discrete read on the tokens, but they just produce continuous directions rather than identities. The MLP neuron clusters remain the only discrete structure I can use, and I've got about 21 clean ones to work with.

I'm checking whether the clusters behave like real token embeddings, since they should be nearly orthogonal, and mostly they are. Next I want to test the English hypothesis using structural zeros in the bigram pattern: space shouldn't follow space or newline, and newline shouldn't follow space.

I'm checking the bigram co-tuning matrix N for structural zeros: if space never follows space or newline in English, the entries for those pairs should come out near zero, and I'm looking at the space row and column to see whether they do.

I'm checking how often neurons whose top current token is proxy 2 also have proxy 2 as their top previous token, since 21 such neurons would be a strong hint of space-after-space behavior and Python indentation. But proxy 2 is also the most common prev argmax overall, so I need the conditional probability P(argmax_b=2 | argmax_a=k) across all k to see whether that overlap is actually special.

I'm checking whether proxy 2's rate of appearing as the previous-token argmax is above baseline for its own neuron set, since if it's no higher than average, the 21 overlapping neurons don't show a special space-after-space pattern. Next I'll compute the N matrix and the observed-to-expected co-tuning ratios.

