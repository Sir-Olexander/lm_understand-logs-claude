The forms I was checking aren't positive-definite, so the joint diagonalization idea doesn't hold up. Looking at the induction match, it isn't spread evenly across tokens, and I'm now weighing whether the token-identity dimensions, around 18-20, actually reflect how many distinct tokens get matched.

I'm weighing whether the token-identity dimensions around 18-20 really reflect a token set that small, since N-way identity matching would need rank near N-1. I'm also considering that newline and line-start handling in Python windows versus Wikipedia or romaji text could be a stronger corpus clue than the rank count alone.

I'm now checking whether the vowel and consonant structure of romaji changes what induction matching looks like, since repeated syllable patterns might behave differently from English words. I'm also comparing the prev-token head L0H5 against the offset-2 head L0H1, whose OV ranks are 21 and 23. Then I'm looking at MLP0's neuron reads: about 25 token dimensions carry strong weight before a sharp drop at dimension 26, so I'm asking whether dimensions 26 to 28 are just non-token noise.

I'm working out how big the token subspace really is: English or Python with 28 tokens would need 28 dimensions, while romaji would come in closer to 24. MLP0's neuron reads point to about 25, which leans romaji, though my estimate from the OV subspaces is rough. So I'm going to tighten that estimate by looking at which residual directions L0 reads that aren't positional.

I'm now working out how big the token subspace is by stacking layer-0 Q, K, and V weights, since their inputs are just the token and positional embeddings, and then reading off the singular spectrum to separate token dimensions from positional ones.

I'm trying to pin down the token subspace size by comparing the OV-based estimate against what MLP0's neuron reads suggest, and wondering whether the difference comes from directions that only partly carry token information.

I'm comparing Python, English, and romaji on their structure: Python has indentation runs at line starts, frequent newlines, and keywords that drive induction and n-gram matches. English leans on common words like "the" and "of" with long lines, while romaji is built from CV syllables and a small alphabet with recurring particles like "desu" and "masu".

I'm checking whether L0H3's positional query tells me anything about the corpus: if its key is the most recent newline, it would count columns in Python, but if it's the most recent space, it would track word positions in English or romaji.

I'm checking whether the positional query gives a constant preference for certain tokens or a recency-driven one, but I can't settle that from the weights alone. Comparing the induction heads, L2H1, L3H1, and L3H5 are the strongest, with L3H0 milder, and I'm wondering whether Python's heavier reliance on copying would show up as more of them.

I'm weighing whether the column of the last newline is what L0H3 tracks in Python, a local word-position count in English or romaji, or something else entirely. I'm also checking whether L2 heads keyed on that same column could implement line-alignment matching.

I'm checking whether the last-space key direction that L0H1 and L0H5 share is the same direction MLP0 reads most strongly. Since space is the most frequent token in every corpus, I'm asking whether the induction heads exclude it, which would depend on how the forms score that direction.

I'm checking whether the induction forms penalize the space token, since spaces are everywhere and matching on them would carry little information. If the newline direction is what they score, that would fit a Python-heavy corpus, so I want to compute the induction forms' value on the space and newline directions and compare them against typical diagonal values. I'll also run the same check on L0H3's token direction.

